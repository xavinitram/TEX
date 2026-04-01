"""
TEX Optimizer — AST transformation passes for compile-time optimization.

Passes:
  1. Constant folding: evaluate expressions with all-literal operands at compile time
  2. Algebraic simplification: x*0 -> 0, x*1 -> x, pow(x,2) -> x*x, etc.
  3. Dead code elimination
  4. Common Subexpression Elimination (CSE)
  5. Loop-Invariant Code Motion (LICM): hoist pure expressions out of loops

All passes preserve semantic equivalence. Applied after type checking, before
interpretation. Operates on the AST in-place (mutates nodes).
"""
from __future__ import annotations
import math

from .ast_nodes import (
    ASTNode, Program, VarDecl, Assignment, IfElse, ForLoop, WhileLoop,
    ExprStatement, BreakStmt, ContinueStmt, ArrayDecl,
    FunctionDef, ReturnStmt, BindingIndexAccess, BindingSampleAccess,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor,
    MatConstructor, CastExpr, ArrayIndexAccess, ArrayLiteral,
    try_extract_static_range,
)

# Pure math functions safe for constant folding (no side effects, deterministic)
_PURE_FUNCTIONS: dict[str, callable] = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2,
    "sqrt": lambda x: math.sqrt(max(0.0, x)),
    "abs": abs, "sign": lambda x: (1.0 if x > 0 else -1.0 if x < 0 else 0.0),
    "floor": math.floor, "ceil": math.ceil, "round": round,
    "fract": lambda x: x - math.floor(x),
    "exp": math.exp,
    "log": lambda x: math.log(max(1e-8, x)),
    "log2": lambda x: math.log2(max(1e-8, x)),
    "log10": lambda x: math.log10(max(1e-8, x)),
    "pow": math.pow, "pow2": lambda x: 2.0 ** x, "pow10": lambda x: 10.0 ** x,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "min": min, "max": max,
    "clamp": lambda x, lo, hi: max(lo, min(x, hi)),
    "lerp": lambda a, b, t: a + (b - a) * t,
    "mix": lambda a, b, t: a + (b - a) * t,  # Alias for lerp
    "smoothstep": lambda e0, e1, x: (
        0.0 if x <= e0 else 1.0 if x >= e1 else
        (lambda t: t * t * (3 - 2 * t))((x - e0) / max(e1 - e0, 1e-8))
    ),
    "step": lambda edge, x: 0.0 if x < edge else 1.0,
    "mod": lambda a, b: math.fmod(a, b) if b != 0 else 0.0,
    "radians": math.radians, "degrees": math.degrees,
}


def optimize(program: Program) -> Program:
    """Run all optimization passes on a program AST (in-place).

    Passes (in order):
      1. Constant folding + algebraic simplification (per-expression)
      2. Dead code elimination (remove unused variable assignments)
      3. Common Subexpression Elimination (CSE) within basic blocks
    """
    # Pass 1: constant folding + algebraic simplification
    for i, stmt in enumerate(program.statements):
        program.statements[i] = _opt_stmt(stmt)

    # Pass 2: dead code elimination
    program.statements = _eliminate_dead_code(program.statements)

    # Pass 3: common subexpression elimination
    program.statements = _eliminate_common_subexpressions(program.statements)

    # Pass 4: loop-invariant code motion
    program.statements = _hoist_loop_invariants(program.statements)

    # Pass 5: small loop unrolling (after LICM so hoisted vars are already out)
    program.statements = _unroll_small_loops(program.statements)

    return program


# ── Statement optimization ────────────────────────────────────────────

def _opt_stmt(stmt: ASTNode) -> ASTNode:
    """Optimize a single statement (recursive)."""
    if isinstance(stmt, VarDecl):
        if stmt.initializer:
            stmt.initializer = _opt_expr(stmt.initializer)
        return stmt

    if isinstance(stmt, Assignment):
        stmt.target = _opt_expr(stmt.target)
        stmt.value = _opt_expr(stmt.value)
        return stmt

    if isinstance(stmt, IfElse):
        stmt.condition = _opt_expr(stmt.condition)
        stmt.then_body = [_opt_stmt(s) for s in stmt.then_body]
        stmt.else_body = [_opt_stmt(s) for s in stmt.else_body]
        return stmt

    if isinstance(stmt, ForLoop):
        stmt.init = _opt_stmt(stmt.init)
        stmt.condition = _opt_expr(stmt.condition)
        stmt.update = _opt_stmt(stmt.update)
        stmt.body = [_opt_stmt(s) for s in stmt.body]
        return stmt

    if isinstance(stmt, WhileLoop):
        stmt.condition = _opt_expr(stmt.condition)
        stmt.body = [_opt_stmt(s) for s in stmt.body]
        return stmt

    if isinstance(stmt, ExprStatement):
        stmt.expr = _opt_expr(stmt.expr)
        return stmt

    if isinstance(stmt, ArrayDecl):
        if stmt.initializer:
            if isinstance(stmt.initializer, ArrayLiteral):
                stmt.initializer.elements = [_opt_expr(e) for e in stmt.initializer.elements]
            else:
                stmt.initializer = _opt_expr(stmt.initializer)
        return stmt

    if isinstance(stmt, FunctionDef):
        stmt.body = [_opt_stmt(s) for s in stmt.body]
        return stmt

    if isinstance(stmt, ReturnStmt):
        if stmt.value:
            stmt.value = _opt_expr(stmt.value)
        return stmt

    # ParamDecl, BreakStmt, ContinueStmt — no change
    return stmt


# ── Expression optimization ──────────────────────────────────────────

def _opt_expr(expr: ASTNode) -> ASTNode:
    """Optimize an expression: constant fold + algebraic simplify."""
    if isinstance(expr, BinOp):
        expr.left = _opt_expr(expr.left)
        expr.right = _opt_expr(expr.right)
        return _fold_binop(expr)

    if isinstance(expr, UnaryOp):
        expr.operand = _opt_expr(expr.operand)
        return _fold_unary(expr)

    if isinstance(expr, FunctionCall):
        expr.args = [_opt_expr(a) for a in expr.args]
        return _fold_function(expr)

    if isinstance(expr, TernaryOp):
        expr.condition = _opt_expr(expr.condition)
        expr.true_expr = _opt_expr(expr.true_expr)
        expr.false_expr = _opt_expr(expr.false_expr)
        # Fold: literal_cond ? a : b -> a or b
        if isinstance(expr.condition, NumberLiteral):
            return expr.true_expr if expr.condition.value > 0.5 else expr.false_expr
        return expr

    if isinstance(expr, VecConstructor):
        expr.args = [_opt_expr(a) for a in expr.args]
        return expr

    if isinstance(expr, MatConstructor):
        expr.args = [_opt_expr(a) for a in expr.args]
        return expr

    if isinstance(expr, CastExpr):
        expr.expr = _opt_expr(expr.expr)
        return expr

    if isinstance(expr, ChannelAccess):
        expr.object = _opt_expr(expr.object)
        return expr

    if isinstance(expr, ArrayIndexAccess):
        expr.array = _opt_expr(expr.array)
        expr.index = _opt_expr(expr.index)
        return expr

    if isinstance(expr, BindingIndexAccess):
        expr.args = [_opt_expr(a) for a in expr.args]
        return expr

    if isinstance(expr, BindingSampleAccess):
        expr.args = [_opt_expr(a) for a in expr.args]
        return expr

    # NumberLiteral, StringLiteral, Identifier, BindingRef — leaf nodes
    return expr


# ── Constant folding ─────────────────────────────────────────────────

def _is_num_lit(node: ASTNode) -> bool:
    return isinstance(node, NumberLiteral)


def _num_val(node: NumberLiteral) -> float:
    return node.value


def _make_num(value: float, loc=None) -> NumberLiteral:
    """Create a NumberLiteral from a computed value."""
    return NumberLiteral(loc=loc or NumberLiteral().loc, value=float(value))


def _fold_binop(node: BinOp) -> ASTNode:
    """Constant fold binary operations + algebraic simplification."""
    left, right = node.left, node.right

    # Full constant fold: both operands are literals
    if _is_num_lit(left) and _is_num_lit(right):
        a, b = _num_val(left), _num_val(right)
        result = _eval_binop_const(node.op, a, b)
        if result is not None:
            return _make_num(result, node.loc)

    # Algebraic simplification
    op = node.op

    # x + 0 -> x, 0 + x -> x
    if op == "+":
        if _is_num_lit(right) and _num_val(right) == 0.0:
            return left
        if _is_num_lit(left) and _num_val(left) == 0.0:
            return right

    # x - 0 -> x, 0 - x -> -x
    if op == "-":
        if _is_num_lit(right) and _num_val(right) == 0.0:
            return left
        if _is_num_lit(left) and _num_val(left) == 0.0:
            return UnaryOp(loc=node.loc, op="-", operand=right)

    # x * 1 -> x, 1 * x -> x
    # NOTE: x * 0 -> 0 is UNSAFE because x may be a spatial tensor and
    # folding to scalar 0 loses the [B,H,W,C] shape. The full constant
    # fold (both operands literal) above handles the safe case already.
    if op == "*":
        if _is_num_lit(right) and _num_val(right) == 1.0:
            return left
        if _is_num_lit(left) and _num_val(left) == 1.0:
            return right

    # x / 1 -> x, x / const -> x * (1/const) (strength reduction)
    if op == "/":
        if _is_num_lit(right):
            rv = _num_val(right)
            if rv == 1.0:
                return left
            # Division by constant → multiplication by reciprocal
            # (multiplication is faster than division on most hardware)
            if rv != 0.0:
                recip = 1.0 / rv
                return BinOp(loc=node.loc, op="*", left=left,
                             right=_make_num(recip, right.loc))

    return node


def _eval_binop_const(op: str, a: float, b: float) -> float | None:
    """Evaluate a binary operation on two constant values. Returns None on error."""
    try:
        if op == "+": return a + b
        if op == "-": return a - b
        if op == "*": return a * b
        if op == "/": return a / b if b != 0 else None
        if op == "%": return math.fmod(a, b) if b != 0 else None
        if op == "==": return 1.0 if a == b else 0.0
        if op == "!=": return 1.0 if a != b else 0.0
        if op == "<": return 1.0 if a < b else 0.0
        if op == ">": return 1.0 if a > b else 0.0
        if op == "<=": return 1.0 if a <= b else 0.0
        if op == ">=": return 1.0 if a >= b else 0.0
        if op == "&&": return 1.0 if (a > 0.5 and b > 0.5) else 0.0
        if op == "||": return 1.0 if (a > 0.5 or b > 0.5) else 0.0
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return None


def _fold_unary(node: UnaryOp) -> ASTNode:
    """Constant fold unary operations."""
    if _is_num_lit(node.operand):
        val = _num_val(node.operand)
        if node.op == "-":
            return _make_num(-val, node.loc)
        if node.op == "!":
            return _make_num(0.0 if val > 0.5 else 1.0, node.loc)
    return node


def _fold_function(node: FunctionCall) -> ASTNode:
    """Constant fold pure function calls + algebraic simplification."""
    # Algebraic simplifications for specific functions
    name = node.name
    args = node.args

    # pow(x, 0) -> 1, pow(x, 1) -> x
    if name == "pow" and len(args) == 2 and _is_num_lit(args[1]):
        exp = _num_val(args[1])
        if exp == 0.0:
            return _make_num(1.0, node.loc)
        if exp == 1.0:
            return args[0]
        # pow(x, 2) -> x * x (strength reduction)
        if exp == 2.0:
            return BinOp(loc=node.loc, op="*", left=args[0], right=args[0])
        # pow(x, 3) -> x * x * x
        if exp == 3.0:
            x_sq = BinOp(loc=node.loc, op="*", left=args[0], right=args[0])
            return BinOp(loc=node.loc, op="*", left=x_sq, right=args[0])
        # pow(x, 0.5) -> sqrt(x)
        if exp == 0.5:
            return FunctionCall(loc=node.loc, name="sqrt", args=[args[0]])
        # pow(x, -1) -> 1 / x
        if exp == -1.0:
            return BinOp(loc=node.loc, op="/",
                         left=_make_num(1.0, node.loc), right=args[0])

    # lerp(a, b, 0) -> a, lerp(a, b, 1) -> b
    if name in ("lerp", "mix") and len(args) == 3 and _is_num_lit(args[2]):
        t = _num_val(args[2])
        if t == 0.0:
            return args[0]
        if t == 1.0:
            return args[1]

    # clamp(x, 0, 1) -> saturate-style (still clamp, but note for future vectorization)
    # abs(x) where x is already abs(y) -> abs(y) (idempotent, handled by constant fold)

    # sqrt(x*x) -> abs(x) — only when the argument is exactly x*x
    if name == "sqrt" and len(args) == 1:
        arg = args[0]
        if isinstance(arg, BinOp) and arg.op == "*":
            # Check if both sides are the same identifier
            if (isinstance(arg.left, Identifier) and isinstance(arg.right, Identifier)
                    and arg.left.name == arg.right.name):
                return FunctionCall(loc=node.loc, name="abs", args=[arg.left])

    # Full constant fold: all args are literals, function is pure
    if all(_is_num_lit(a) for a in args):
        fn = _PURE_FUNCTIONS.get(name)
        if fn is not None:
            try:
                float_args = [_num_val(a) for a in args]
                result = fn(*float_args)
                if isinstance(result, (int, float)) and math.isfinite(result):
                    return _make_num(float(result), node.loc)
            except (ValueError, ZeroDivisionError, OverflowError, TypeError):
                pass

    return node


# ── Dead Code Elimination ─────────────────────────────────────────────

def _eliminate_dead_code(stmts: list[ASTNode],
                         outer_used: set[str] | None = None) -> list[ASTNode]:
    """Remove dead variable declarations/assignments.

    A local variable is "dead" if it is assigned but never read by any
    subsequent statement, binding assignment, or nested scope. Binding
    references (@name) and params ($name) are never eliminated.

    outer_used: names referenced in enclosing scopes. Variables assigned
    inside a branch but read in the outer scope must not be eliminated.
    """
    # Step 1: collect all referenced variable names across the entire block
    used = _collect_used_names(stmts)
    if outer_used:
        used |= outer_used

    # Step 2: filter out dead VarDecls (local vars that are never read)
    result = []
    for stmt in stmts:
        if isinstance(stmt, VarDecl):
            if stmt.name not in used:
                continue  # Dead variable — skip it
        elif isinstance(stmt, Assignment):
            target = stmt.target
            if isinstance(target, Identifier) and target.name not in used:
                # Dead assignment to a local var — skip it
                # But preserve if the RHS has side effects (function calls)
                if not _has_side_effects(stmt.value):
                    continue

        # Recurse into compound statements, propagating outer liveness
        if isinstance(stmt, IfElse):
            stmt.then_body = _eliminate_dead_code(stmt.then_body, used)
            stmt.else_body = _eliminate_dead_code(stmt.else_body, used)
        elif isinstance(stmt, ForLoop):
            stmt.body = _eliminate_dead_code(stmt.body, used)
        elif isinstance(stmt, WhileLoop):
            stmt.body = _eliminate_dead_code(stmt.body, used)
        elif isinstance(stmt, FunctionDef):
            stmt.body = _eliminate_dead_code(stmt.body, used)

        result.append(stmt)

    return result


def _collect_used_names(stmts: list[ASTNode]) -> set[str]:
    """Collect all variable names that are READ (not just assigned) in a block."""
    used: set[str] = set()
    for stmt in stmts:
        _collect_used_in_stmt(stmt, used)
    return used


def _collect_used_in_stmt(stmt: ASTNode, used: set[str]):
    """Recursively collect variable names read in a statement."""
    if isinstance(stmt, VarDecl):
        if stmt.initializer:
            _collect_used_in_expr(stmt.initializer, used)

    elif isinstance(stmt, Assignment):
        # The target Identifier is a WRITE, not a read
        # But ChannelAccess reads the base, ArrayIndexAccess reads the index,
        # and BindingIndexAccess reads its coordinate arguments
        target = stmt.target
        if isinstance(target, ChannelAccess):
            _collect_used_in_expr(target.object, used)
        elif isinstance(target, ArrayIndexAccess):
            _collect_used_in_expr(target.array, used)
            _collect_used_in_expr(target.index, used)
        elif isinstance(target, BindingIndexAccess):
            for arg in target.args:
                _collect_used_in_expr(arg, used)
        _collect_used_in_expr(stmt.value, used)

    elif isinstance(stmt, IfElse):
        _collect_used_in_expr(stmt.condition, used)
        for s in stmt.then_body:
            _collect_used_in_stmt(s, used)
        for s in stmt.else_body:
            _collect_used_in_stmt(s, used)

    elif isinstance(stmt, ForLoop):
        _collect_used_in_stmt(stmt.init, used)
        _collect_used_in_expr(stmt.condition, used)
        _collect_used_in_stmt(stmt.update, used)
        for s in stmt.body:
            _collect_used_in_stmt(s, used)

    elif isinstance(stmt, WhileLoop):
        _collect_used_in_expr(stmt.condition, used)
        for s in stmt.body:
            _collect_used_in_stmt(s, used)

    elif isinstance(stmt, ExprStatement):
        _collect_used_in_expr(stmt.expr, used)

    elif isinstance(stmt, ArrayDecl):
        if stmt.initializer:
            if isinstance(stmt.initializer, ArrayLiteral):
                for elem in stmt.initializer.elements:
                    _collect_used_in_expr(elem, used)
            else:
                _collect_used_in_expr(stmt.initializer, used)

    elif isinstance(stmt, FunctionDef):
        for s in stmt.body:
            _collect_used_in_stmt(s, used)

    elif isinstance(stmt, ReturnStmt):
        if stmt.value:
            _collect_used_in_expr(stmt.value, used)


def _collect_used_in_expr(expr: ASTNode, used: set[str]):
    """Recursively collect variable names read in an expression."""
    if isinstance(expr, Identifier):
        used.add(expr.name)

    elif isinstance(expr, BinOp):
        _collect_used_in_expr(expr.left, used)
        _collect_used_in_expr(expr.right, used)

    elif isinstance(expr, UnaryOp):
        _collect_used_in_expr(expr.operand, used)

    elif isinstance(expr, TernaryOp):
        _collect_used_in_expr(expr.condition, used)
        _collect_used_in_expr(expr.true_expr, used)
        _collect_used_in_expr(expr.false_expr, used)

    elif isinstance(expr, FunctionCall):
        for arg in expr.args:
            _collect_used_in_expr(arg, used)

    elif isinstance(expr, VecConstructor):
        for arg in expr.args:
            _collect_used_in_expr(arg, used)

    elif isinstance(expr, MatConstructor):
        for arg in expr.args:
            _collect_used_in_expr(arg, used)

    elif isinstance(expr, CastExpr):
        _collect_used_in_expr(expr.expr, used)

    elif isinstance(expr, ChannelAccess):
        _collect_used_in_expr(expr.object, used)

    elif isinstance(expr, ArrayIndexAccess):
        _collect_used_in_expr(expr.array, used)
        _collect_used_in_expr(expr.index, used)

    elif isinstance(expr, ArrayLiteral):
        for elem in expr.elements:
            _collect_used_in_expr(elem, used)

    elif isinstance(expr, BindingIndexAccess):
        _collect_used_in_expr(expr.binding, used)
        for arg in expr.args:
            _collect_used_in_expr(arg, used)

    elif isinstance(expr, BindingSampleAccess):
        _collect_used_in_expr(expr.binding, used)
        for arg in expr.args:
            _collect_used_in_expr(arg, used)

    # NumberLiteral, StringLiteral, BindingRef — no local vars to collect


def _has_side_effects(expr: ASTNode) -> bool:
    """Check if an expression might have side effects (conservative)."""
    if isinstance(expr, FunctionCall):
        return True  # Functions could have side effects
    if isinstance(expr, BinOp):
        return _has_side_effects(expr.left) or _has_side_effects(expr.right)
    if isinstance(expr, UnaryOp):
        return _has_side_effects(expr.operand)
    if isinstance(expr, TernaryOp):
        return (_has_side_effects(expr.condition) or
                _has_side_effects(expr.true_expr) or
                _has_side_effects(expr.false_expr))
    if isinstance(expr, VecConstructor):
        return any(_has_side_effects(a) for a in expr.args)
    if isinstance(expr, (BindingIndexAccess, BindingSampleAccess)):
        return True  # Reads from bindings — preserve
    return False


# ── Common Subexpression Elimination (CSE) ───────────────────────────

# Minimum expression complexity to consider for CSE (avoid hoisting trivial nodes)
_CSE_MIN_DEPTH = 2

# Pure functions safe for CSE (deterministic, no side effects, no sampling)
_CSE_PURE_FUNCTIONS = frozenset(_PURE_FUNCTIONS.keys()) | frozenset({
    "vec2", "vec3", "vec4", "mat3", "mat4",
    "normalize", "length", "distance", "dot", "cross",
    "rgb_to_hsv", "hsv_to_rgb", "rgb_to_hsl", "hsl_to_rgb",
    "luminance", "contrast", "saturate",
})


def _expr_hash(expr: ASTNode) -> str | None:
    """Compute a canonical string hash for an expression.

    Returns None for expressions that should not be CSE candidates
    (leaves, side-effectful calls, binding refs that may be reassigned).
    """
    if isinstance(expr, NumberLiteral):
        return f"N:{expr.value}"
    if isinstance(expr, StringLiteral):
        return None  # Not worth CSE
    if isinstance(expr, Identifier):
        return f"I:{expr.name}"
    if isinstance(expr, BindingRef):
        return f"B:{expr.name}"
    if isinstance(expr, BinOp):
        lh = _expr_hash(expr.left)
        rh = _expr_hash(expr.right)
        if lh is None or rh is None:
            return None
        return f"({lh}{expr.op}{rh})"
    if isinstance(expr, UnaryOp):
        oh = _expr_hash(expr.operand)
        if oh is None:
            return None
        return f"(u{expr.op}{oh})"
    if isinstance(expr, FunctionCall):
        if expr.name not in _CSE_PURE_FUNCTIONS:
            return None  # Side-effectful or nondeterministic
        arg_hashes = []
        for a in expr.args:
            ah = _expr_hash(a)
            if ah is None:
                return None
            arg_hashes.append(ah)
        return f"F:{expr.name}({','.join(arg_hashes)})"
    if isinstance(expr, ChannelAccess):
        oh = _expr_hash(expr.object)
        if oh is None:
            return None
        return f"C:{oh}.{expr.channels}"
    if isinstance(expr, VecConstructor):
        arg_hashes = []
        for a in expr.args:
            ah = _expr_hash(a)
            if ah is None:
                return None
            arg_hashes.append(ah)
        return f"V{expr.size}({','.join(arg_hashes)})"
    if isinstance(expr, CastExpr):
        eh = _expr_hash(expr.expr)
        if eh is None:
            return None
        return f"T:{expr.target_type}({eh})"
    if isinstance(expr, TernaryOp):
        ch = _expr_hash(expr.condition)
        th = _expr_hash(expr.true_expr)
        fh = _expr_hash(expr.false_expr)
        if ch is None or th is None or fh is None:
            return None
        return f"?:{ch}?{th}:{fh}"
    return None


def _expr_depth(expr: ASTNode) -> int:
    """Compute the nesting depth of an expression."""
    if isinstance(expr, (NumberLiteral, StringLiteral, Identifier, BindingRef)):
        return 0
    if isinstance(expr, BinOp):
        return 1 + max(_expr_depth(expr.left), _expr_depth(expr.right))
    if isinstance(expr, UnaryOp):
        return 1 + _expr_depth(expr.operand)
    if isinstance(expr, FunctionCall):
        return 1 + max((_expr_depth(a) for a in expr.args), default=0)
    if isinstance(expr, ChannelAccess):
        return 1 + _expr_depth(expr.object)
    if isinstance(expr, VecConstructor):
        return 1 + max((_expr_depth(a) for a in expr.args), default=0)
    if isinstance(expr, CastExpr):
        return 1 + _expr_depth(expr.expr)
    if isinstance(expr, TernaryOp):
        return 1 + max(_expr_depth(expr.condition),
                       _expr_depth(expr.true_expr),
                       _expr_depth(expr.false_expr))
    return 0


def _collect_subexprs(expr: ASTNode, seen: dict[str, int], depth_threshold: int = _CSE_MIN_DEPTH):
    """Walk an expression and count sub-expressions by hash.

    seen: maps hash -> occurrence count.
    Only records expressions with depth >= depth_threshold.
    """
    h = _expr_hash(expr)
    if h is not None and _expr_depth(expr) >= depth_threshold:
        if h not in seen:
            seen[h] = 1
        else:
            seen[h] += 1

    # Recurse into children
    if isinstance(expr, BinOp):
        _collect_subexprs(expr.left, seen, depth_threshold)
        _collect_subexprs(expr.right, seen, depth_threshold)
    elif isinstance(expr, UnaryOp):
        _collect_subexprs(expr.operand, seen, depth_threshold)
    elif isinstance(expr, FunctionCall):
        for a in expr.args:
            _collect_subexprs(a, seen, depth_threshold)
    elif isinstance(expr, VecConstructor):
        for a in expr.args:
            _collect_subexprs(a, seen, depth_threshold)
    elif isinstance(expr, ChannelAccess):
        _collect_subexprs(expr.object, seen, depth_threshold)
    elif isinstance(expr, CastExpr):
        _collect_subexprs(expr.expr, seen, depth_threshold)
    elif isinstance(expr, TernaryOp):
        _collect_subexprs(expr.condition, seen, depth_threshold)
        _collect_subexprs(expr.true_expr, seen, depth_threshold)
        _collect_subexprs(expr.false_expr, seen, depth_threshold)
    elif isinstance(expr, (BindingIndexAccess, BindingSampleAccess)):
        for a in expr.args:
            _collect_subexprs(a, seen, depth_threshold)


def _collect_subexprs_in_stmt(stmt: ASTNode, seen: dict[str, int]):
    """Collect sub-expression hashes from a statement."""
    if isinstance(stmt, VarDecl):
        if stmt.initializer:
            _collect_subexprs(stmt.initializer, seen)
    elif isinstance(stmt, Assignment):
        _collect_subexprs(stmt.value, seen)
    elif isinstance(stmt, ExprStatement):
        _collect_subexprs(stmt.expr, seen)


def _replace_expr(expr: ASTNode, replacements: dict[str, str]) -> ASTNode:
    """Replace sub-expressions whose hash matches a CSE temp variable."""
    h = _expr_hash(expr)
    if h is not None and h in replacements:
        return Identifier(loc=expr.loc, name=replacements[h])

    if isinstance(expr, BinOp):
        expr.left = _replace_expr(expr.left, replacements)
        expr.right = _replace_expr(expr.right, replacements)
    elif isinstance(expr, UnaryOp):
        expr.operand = _replace_expr(expr.operand, replacements)
    elif isinstance(expr, FunctionCall):
        expr.args = [_replace_expr(a, replacements) for a in expr.args]
    elif isinstance(expr, VecConstructor):
        expr.args = [_replace_expr(a, replacements) for a in expr.args]
    elif isinstance(expr, ChannelAccess):
        expr.object = _replace_expr(expr.object, replacements)
    elif isinstance(expr, CastExpr):
        expr.expr = _replace_expr(expr.expr, replacements)
    elif isinstance(expr, TernaryOp):
        expr.condition = _replace_expr(expr.condition, replacements)
        expr.true_expr = _replace_expr(expr.true_expr, replacements)
        expr.false_expr = _replace_expr(expr.false_expr, replacements)
    elif isinstance(expr, (BindingIndexAccess, BindingSampleAccess)):
        expr.args = [_replace_expr(a, replacements) for a in expr.args]
    return expr


def _replace_in_stmt(stmt: ASTNode, replacements: dict[str, str]):
    """Replace CSE sub-expressions within a statement."""
    if isinstance(stmt, VarDecl):
        if stmt.initializer:
            stmt.initializer = _replace_expr(stmt.initializer, replacements)
    elif isinstance(stmt, Assignment):
        stmt.value = _replace_expr(stmt.value, replacements)
    elif isinstance(stmt, ExprStatement):
        stmt.expr = _replace_expr(stmt.expr, replacements)


def _eliminate_common_subexpressions(stmts: list[ASTNode]) -> list[ASTNode]:
    """CSE within a basic block (list of sequential statements).

    Scans for sub-expressions that appear 2+ times, hoists the first
    occurrence into a temp variable, and replaces all occurrences with
    the temp. Only operates within straight-line code blocks — compound
    statements (if/for/while) get their bodies processed recursively
    but form block boundaries for the outer scope.

    Conservative: does not CSE across control flow boundaries where
    variable reassignment could invalidate the cached value.
    """
    # First recurse into compound statement bodies
    for stmt in stmts:
        if isinstance(stmt, IfElse):
            stmt.then_body = _eliminate_common_subexpressions(stmt.then_body)
            stmt.else_body = _eliminate_common_subexpressions(stmt.else_body)
        elif isinstance(stmt, ForLoop):
            stmt.body = _eliminate_common_subexpressions(stmt.body)
        elif isinstance(stmt, WhileLoop):
            stmt.body = _eliminate_common_subexpressions(stmt.body)
        elif isinstance(stmt, FunctionDef):
            stmt.body = _eliminate_common_subexpressions(stmt.body)

    # Collect all sub-expression hashes across the block
    seen: dict[str, int] = {}  # hash -> occurrence count
    for stmt in stmts:
        _collect_subexprs_in_stmt(stmt, seen)

    # Find expressions that appear 2+ times
    duplicates = {h for h, count in seen.items() if count >= 2}
    if not duplicates:
        return stmts

    # Build replacement map and generate temp VarDecls
    # We need to find the actual expression node for each duplicate hash
    # Walk the statements again to find first occurrence of each duplicate,
    # tracking which statement index it was found in so we can insert the
    # temp decl just before that statement (not at the top of the block).
    first_occurrence: dict[str, tuple[ASTNode, int]] = {}  # hash -> (expr, stmt_index)

    def _find_first(expr: ASTNode, stmt_idx: int):
        h = _expr_hash(expr)
        if h is not None and h in duplicates and h not in first_occurrence:
            first_occurrence[h] = (expr, stmt_idx)
            # Don't recurse into children of a found duplicate —
            # we want the outermost match
            return
        if isinstance(expr, BinOp):
            _find_first(expr.left, stmt_idx)
            _find_first(expr.right, stmt_idx)
        elif isinstance(expr, UnaryOp):
            _find_first(expr.operand, stmt_idx)
        elif isinstance(expr, FunctionCall):
            for a in expr.args:
                _find_first(a, stmt_idx)
        elif isinstance(expr, VecConstructor):
            for a in expr.args:
                _find_first(a, stmt_idx)
        elif isinstance(expr, ChannelAccess):
            _find_first(expr.object, stmt_idx)
        elif isinstance(expr, CastExpr):
            _find_first(expr.expr, stmt_idx)
        elif isinstance(expr, TernaryOp):
            _find_first(expr.condition, stmt_idx)
            _find_first(expr.true_expr, stmt_idx)
            _find_first(expr.false_expr, stmt_idx)

    for idx, stmt in enumerate(stmts):
        if isinstance(stmt, VarDecl) and stmt.initializer:
            _find_first(stmt.initializer, idx)
        elif isinstance(stmt, Assignment):
            _find_first(stmt.value, idx)
        elif isinstance(stmt, ExprStatement):
            _find_first(stmt.expr, idx)

    if not first_occurrence:
        return stmts

    # Assign temp variable names and build replacements
    replacements: dict[str, str] = {}  # hash -> temp var name
    # Group CSE decls by the statement index they should be inserted before
    insert_before: dict[int, list[VarDecl]] = {}  # stmt_index -> [VarDecl, ...]
    for i, (h, (expr_node, stmt_idx)) in enumerate(first_occurrence.items()):
        temp_name = f"_cse{i}"
        replacements[h] = temp_name
        decl = VarDecl(
            loc=expr_node.loc,
            type_name="float",  # Type doesn't matter at this stage — type checker already ran
            name=temp_name,
            initializer=expr_node,
        )
        insert_before.setdefault(stmt_idx, []).append(decl)

    # Replace all occurrences in all statements
    for stmt in stmts:
        _replace_in_stmt(stmt, replacements)

    # Insert temp decls just before the statement that first uses them
    result: list[ASTNode] = []
    for idx, stmt in enumerate(stmts):
        if idx in insert_before:
            result.extend(insert_before[idx])
        result.append(stmt)
    return result


# ── Loop-Invariant Code Motion (LICM) ────────────────────────────────

# Minimum expression depth to consider hoisting (avoid trivial hoists)
_LICM_MIN_DEPTH = 2


def _collect_written_vars(stmts: list[ASTNode]) -> set[str]:
    """Collect all variable names written (assigned/declared) in a statement list."""
    written: set[str] = set()
    for stmt in stmts:
        _collect_written_in_stmt(stmt, written)
    return written


def _collect_written_in_stmt(stmt: ASTNode, written: set[str]):
    """Recursively collect written variable names."""
    if isinstance(stmt, VarDecl):
        written.add(stmt.name)
    elif isinstance(stmt, Assignment):
        target = stmt.target
        if isinstance(target, Identifier):
            written.add(target.name)
        elif isinstance(target, BindingRef):
            written.add(target.name)
        elif isinstance(target, ChannelAccess):
            obj = target.object
            if isinstance(obj, Identifier):
                written.add(obj.name)
            elif isinstance(obj, BindingRef):
                written.add(obj.name)
        elif isinstance(target, ArrayIndexAccess) and isinstance(target.array, Identifier):
            written.add(target.array.name)
        elif isinstance(target, BindingIndexAccess) and isinstance(target.binding, BindingRef):
            written.add(target.binding.name)
    elif isinstance(stmt, ArrayDecl):
        written.add(stmt.name)
    elif isinstance(stmt, IfElse):
        for s in stmt.then_body:
            _collect_written_in_stmt(s, written)
        for s in stmt.else_body:
            _collect_written_in_stmt(s, written)
    elif isinstance(stmt, ForLoop):
        _collect_written_in_stmt(stmt.init, written)
        _collect_written_in_stmt(stmt.update, written)
        for s in stmt.body:
            _collect_written_in_stmt(s, written)
    elif isinstance(stmt, WhileLoop):
        for s in stmt.body:
            _collect_written_in_stmt(s, written)
    elif isinstance(stmt, FunctionDef):
        for s in stmt.body:
            _collect_written_in_stmt(s, written)


def _expr_reads_vars(expr: ASTNode) -> set[str]:
    """Collect all variable names read by an expression."""
    names: set[str] = set()
    _collect_used_in_expr(expr, names)
    return names


def _is_pure_for_licm(expr: ASTNode) -> bool:
    """Check if an expression is pure enough for LICM hoisting.

    More permissive than _has_side_effects: allows known-pure stdlib functions
    (sin, cos, sqrt, etc.) but rejects sampling, binding access, and user functions.
    """
    if isinstance(expr, FunctionCall):
        if expr.name not in _CSE_PURE_FUNCTIONS:
            return False
        return all(_is_pure_for_licm(a) for a in expr.args)
    if isinstance(expr, BinOp):
        return _is_pure_for_licm(expr.left) and _is_pure_for_licm(expr.right)
    if isinstance(expr, UnaryOp):
        return _is_pure_for_licm(expr.operand)
    if isinstance(expr, TernaryOp):
        return (_is_pure_for_licm(expr.condition) and
                _is_pure_for_licm(expr.true_expr) and
                _is_pure_for_licm(expr.false_expr))
    if isinstance(expr, VecConstructor):
        return all(_is_pure_for_licm(a) for a in expr.args)
    if isinstance(expr, CastExpr):
        return _is_pure_for_licm(expr.expr)
    if isinstance(expr, ChannelAccess):
        return _is_pure_for_licm(expr.object)
    if isinstance(expr, (BindingIndexAccess, BindingSampleAccess)):
        return False  # Sampling is not pure for LICM
    if isinstance(expr, (NumberLiteral, StringLiteral, Identifier, BindingRef)):
        return True
    return False


def _is_loop_invariant(expr: ASTNode, modified: set[str]) -> bool:
    """Check if an expression is loop-invariant (doesn't depend on modified vars).

    An expression is loop-invariant if:
    - It is pure (no side effects, no sampling)
    - None of its referenced variables are in the modified set
    - It has sufficient depth to be worth hoisting
    """
    if not _is_pure_for_licm(expr):
        return False
    if _expr_depth(expr) < _LICM_MIN_DEPTH:
        return False
    reads = _expr_reads_vars(expr)
    return reads.isdisjoint(modified)


def _extract_invariant_subexpr(expr: ASTNode, modified: set[str],
                                hoisted: list[tuple[str, ASTNode]],
                                counter: list[int]) -> ASTNode:
    """Walk an expression tree and replace loop-invariant subtrees with temp vars.

    Replaces the largest (outermost) invariant subtrees first. If the entire
    expression is invariant, replaces the whole thing. Otherwise recurses into
    children to find smaller invariant sub-expressions.
    """
    # Check if the entire expression is invariant
    if _is_loop_invariant(expr, modified):
        temp_name = f"_licm{counter[0]}"
        counter[0] += 1
        hoisted.append((temp_name, expr))
        return Identifier(loc=expr.loc, name=temp_name)

    # Otherwise recurse into children to find invariant sub-expressions
    if isinstance(expr, BinOp):
        expr.left = _extract_invariant_subexpr(expr.left, modified, hoisted, counter)
        expr.right = _extract_invariant_subexpr(expr.right, modified, hoisted, counter)
    elif isinstance(expr, UnaryOp):
        expr.operand = _extract_invariant_subexpr(expr.operand, modified, hoisted, counter)
    elif isinstance(expr, FunctionCall):
        expr.args = [_extract_invariant_subexpr(a, modified, hoisted, counter) for a in expr.args]
    elif isinstance(expr, VecConstructor):
        expr.args = [_extract_invariant_subexpr(a, modified, hoisted, counter) for a in expr.args]
    elif isinstance(expr, TernaryOp):
        expr.condition = _extract_invariant_subexpr(expr.condition, modified, hoisted, counter)
        expr.true_expr = _extract_invariant_subexpr(expr.true_expr, modified, hoisted, counter)
        expr.false_expr = _extract_invariant_subexpr(expr.false_expr, modified, hoisted, counter)
    elif isinstance(expr, CastExpr):
        expr.expr = _extract_invariant_subexpr(expr.expr, modified, hoisted, counter)
    elif isinstance(expr, ChannelAccess):
        expr.object = _extract_invariant_subexpr(expr.object, modified, hoisted, counter)

    return expr


def _licm_stmt(stmt: ASTNode, modified: set[str],
               hoisted: list[tuple[str, ASTNode]], counter: list[int]):
    """Extract loop-invariant expressions from a single statement in a loop body."""
    if isinstance(stmt, VarDecl):
        if stmt.initializer:
            stmt.initializer = _extract_invariant_subexpr(
                stmt.initializer, modified, hoisted, counter)
    elif isinstance(stmt, Assignment):
        stmt.value = _extract_invariant_subexpr(stmt.value, modified, hoisted, counter)
    elif isinstance(stmt, ExprStatement):
        stmt.expr = _extract_invariant_subexpr(stmt.expr, modified, hoisted, counter)
    elif isinstance(stmt, ReturnStmt):
        if stmt.value:
            stmt.value = _extract_invariant_subexpr(stmt.value, modified, hoisted, counter)
    elif isinstance(stmt, IfElse):
        stmt.condition = _extract_invariant_subexpr(stmt.condition, modified, hoisted, counter)
        for s in stmt.then_body:
            _licm_stmt(s, modified, hoisted, counter)
        for s in stmt.else_body:
            _licm_stmt(s, modified, hoisted, counter)


def _licm_loop(loop: ForLoop | WhileLoop, counter: list[int]) -> list[ASTNode]:
    """Apply LICM to a single loop. Returns [hoisted_decls..., loop]."""
    # Collect all variables written inside the loop body
    modified = _collect_written_vars(loop.body)

    # For for-loops, the loop variable and update target are also modified
    if isinstance(loop, ForLoop):
        if isinstance(loop.init, VarDecl):
            modified.add(loop.init.name)
        if isinstance(loop.update, Assignment) and isinstance(loop.update.target, Identifier):
            modified.add(loop.update.target.name)

    # Extract invariant sub-expressions from body statements
    hoisted: list[tuple[str, ASTNode]] = []
    for stmt in loop.body:
        _licm_stmt(stmt, modified, hoisted, counter)

    if not hoisted:
        return [loop]

    # Create VarDecl nodes for hoisted expressions
    pre_loop: list[ASTNode] = []
    for temp_name, expr_node in hoisted:
        pre_loop.append(VarDecl(
            loc=expr_node.loc,
            type_name="float",  # Type doesn't matter — type checker already ran
            name=temp_name,
            initializer=expr_node,
        ))

    return pre_loop + [loop]


def _hoist_loop_invariants(stmts: list[ASTNode], counter: list[int] | None = None) -> list[ASTNode]:
    """LICM pass: hoist loop-invariant expressions out of for/while loops.

    Scans statement lists for loops, identifies pure expressions that don't
    depend on variables modified in the loop, and hoists them to temp variables
    before the loop.
    """
    if counter is None:
        counter = [0]  # top-level call creates the counter
    result: list[ASTNode] = []

    for stmt in stmts:
        # Recurse into compound statement bodies first
        if isinstance(stmt, IfElse):
            stmt.then_body = _hoist_loop_invariants(stmt.then_body, counter)
            stmt.else_body = _hoist_loop_invariants(stmt.else_body, counter)
            result.append(stmt)
        elif isinstance(stmt, FunctionDef):
            stmt.body = _hoist_loop_invariants(stmt.body, counter)
            result.append(stmt)
        elif isinstance(stmt, ForLoop):
            # Recurse into nested loops first
            stmt.body = _hoist_loop_invariants(stmt.body, counter)
            # Then hoist invariants from this loop
            result.extend(_licm_loop(stmt, counter))
        elif isinstance(stmt, WhileLoop):
            stmt.body = _hoist_loop_invariants(stmt.body, counter)
            result.extend(_licm_loop(stmt, counter))
        else:
            result.append(stmt)

    return result


# ── Small Loop Unrolling ──────────────────────────────────────────────

_UNROLL_MAX_ITERS = 8
_UNROLL_MAX_BODY_STMTS = 6  # Don't unroll large loop bodies


def _contains_break_continue(stmts: list[ASTNode]) -> bool:
    """Check if any statement contains break or continue."""
    for stmt in stmts:
        if isinstance(stmt, (BreakStmt, ContinueStmt)):
            return True
        if isinstance(stmt, IfElse):
            if _contains_break_continue(stmt.then_body):
                return True
            if _contains_break_continue(stmt.else_body):
                return True
        # break/continue in nested loops don't affect the outer loop
    return False


def _subst_expr(expr: ASTNode, var_name: str, value: float) -> ASTNode:
    """Substitute all occurrences of var_name with a NumberLiteral in an expression.

    Returns a new expression (does not mutate the original).
    """
    if isinstance(expr, Identifier) and expr.name == var_name:
        return NumberLiteral(loc=expr.loc, value=value)
    if isinstance(expr, BinOp):
        return BinOp(loc=expr.loc, op=expr.op,
                     left=_subst_expr(expr.left, var_name, value),
                     right=_subst_expr(expr.right, var_name, value))
    if isinstance(expr, UnaryOp):
        return UnaryOp(loc=expr.loc, op=expr.op,
                       operand=_subst_expr(expr.operand, var_name, value))
    if isinstance(expr, FunctionCall):
        return FunctionCall(loc=expr.loc, name=expr.name,
                            args=[_subst_expr(a, var_name, value) for a in expr.args])
    if isinstance(expr, TernaryOp):
        return TernaryOp(loc=expr.loc,
                         condition=_subst_expr(expr.condition, var_name, value),
                         true_expr=_subst_expr(expr.true_expr, var_name, value),
                         false_expr=_subst_expr(expr.false_expr, var_name, value))
    if isinstance(expr, VecConstructor):
        return VecConstructor(loc=expr.loc, size=expr.size,
                              args=[_subst_expr(a, var_name, value) for a in expr.args])
    if isinstance(expr, MatConstructor):
        return MatConstructor(loc=expr.loc, size=expr.size,
                              args=[_subst_expr(a, var_name, value) for a in expr.args])
    if isinstance(expr, ArrayLiteral):
        return ArrayLiteral(loc=expr.loc,
                            elements=[_subst_expr(e, var_name, value) for e in expr.elements])
    if isinstance(expr, CastExpr):
        return CastExpr(loc=expr.loc, target_type=expr.target_type,
                        expr=_subst_expr(expr.expr, var_name, value))
    if isinstance(expr, ChannelAccess):
        return ChannelAccess(loc=expr.loc, channels=expr.channels,
                             object=_subst_expr(expr.object, var_name, value))
    if isinstance(expr, ArrayIndexAccess):
        return ArrayIndexAccess(loc=expr.loc,
                                array=_subst_expr(expr.array, var_name, value),
                                index=_subst_expr(expr.index, var_name, value))
    if isinstance(expr, BindingIndexAccess):
        return BindingIndexAccess(loc=expr.loc,
                                  binding=_subst_expr(expr.binding, var_name, value),
                                  args=[_subst_expr(a, var_name, value) for a in expr.args])
    if isinstance(expr, BindingSampleAccess):
        return BindingSampleAccess(loc=expr.loc,
                                   binding=_subst_expr(expr.binding, var_name, value),
                                   args=[_subst_expr(a, var_name, value) for a in expr.args])
    # NumberLiteral, StringLiteral, BindingRef — no substitution needed
    return expr


def _subst_stmt(stmt: ASTNode, var_name: str, value: float) -> ASTNode:
    """Substitute loop variable in a statement. Returns new statement."""
    if isinstance(stmt, VarDecl):
        init = _subst_expr(stmt.initializer, var_name, value) if stmt.initializer else None
        return VarDecl(loc=stmt.loc, type_name=stmt.type_name, name=stmt.name,
                       initializer=init, is_const=stmt.is_const)
    if isinstance(stmt, Assignment):
        new_target = _subst_expr(stmt.target, var_name, value)
        new_value = _subst_expr(stmt.value, var_name, value)
        return Assignment(loc=stmt.loc, target=new_target, value=new_value, op=stmt.op)
    if isinstance(stmt, ExprStatement):
        return ExprStatement(loc=stmt.loc, expr=_subst_expr(stmt.expr, var_name, value))
    if isinstance(stmt, IfElse):
        return IfElse(loc=stmt.loc,
                      condition=_subst_expr(stmt.condition, var_name, value),
                      then_body=[_subst_stmt(s, var_name, value) for s in stmt.then_body],
                      else_body=[_subst_stmt(s, var_name, value) for s in stmt.else_body])
    if isinstance(stmt, ReturnStmt):
        val = _subst_expr(stmt.value, var_name, value) if stmt.value else None
        return ReturnStmt(loc=stmt.loc, value=val)
    if isinstance(stmt, ForLoop):
        new_init = _subst_stmt(stmt.init, var_name, value)
        new_cond = _subst_expr(stmt.condition, var_name, value)
        new_update = _subst_stmt(stmt.update, var_name, value)
        new_body = [_subst_stmt(s, var_name, value) for s in stmt.body]
        return ForLoop(loc=stmt.loc, init=new_init, condition=new_cond,
                       update=new_update, body=new_body)
    if isinstance(stmt, WhileLoop):
        new_cond = _subst_expr(stmt.condition, var_name, value)
        new_body = [_subst_stmt(s, var_name, value) for s in stmt.body]
        return WhileLoop(loc=stmt.loc, condition=new_cond, body=new_body)
    if isinstance(stmt, ArrayDecl):
        init = _subst_expr(stmt.initializer, var_name, value) if stmt.initializer else None
        return ArrayDecl(loc=stmt.loc, element_type_name=stmt.element_type_name,
                         name=stmt.name, size=stmt.size, initializer=init)
    return stmt


def _unroll_small_loops(stmts: list[ASTNode]) -> list[ASTNode]:
    """Unroll for-loops with a small static iteration count.

    Replaces `for (int i = 0; i < N; i++) { body }` with N copies of `body`
    where the loop variable is substituted with the iteration value.
    Only applies when N <= _UNROLL_MAX_ITERS and body has no break/continue.
    """
    result: list[ASTNode] = []
    for stmt in stmts:
        # Recurse into compound statements
        if isinstance(stmt, IfElse):
            stmt.then_body = _unroll_small_loops(stmt.then_body)
            stmt.else_body = _unroll_small_loops(stmt.else_body)
            result.append(stmt)
        elif isinstance(stmt, FunctionDef):
            stmt.body = _unroll_small_loops(stmt.body)
            result.append(stmt)
        elif isinstance(stmt, ForLoop):
            # Skip unrolling if this loop contains a nested ForLoop — these
            # are stencil candidates (box blur, min/max pool, median) that the
            # codegen specialises into bulk tensor ops (avg_pool2d, max_pool2d,
            # conv2d). Unrolling would destroy the nested structure.
            has_nested_for = any(isinstance(s, ForLoop) for s in stmt.body)
            if has_nested_for:
                # Still recurse into the INNER loop's body (but not the outer)
                for s in stmt.body:
                    if isinstance(s, ForLoop):
                        s.body = _unroll_small_loops(s.body)
                result.append(stmt)
                continue
            # Recurse into nested loops first
            stmt.body = _unroll_small_loops(stmt.body)
            # Try to unroll this loop
            static = try_extract_static_range(stmt)
            if (static is not None
                    and len(stmt.body) <= _UNROLL_MAX_BODY_STMTS
                    and not _contains_break_continue(stmt.body)):
                var_name, start, stop, step = static
                n_iters = len(range(start, stop, step))
                if 0 < n_iters <= _UNROLL_MAX_ITERS:
                    # Unroll: emit body N times with loop var substituted
                    for i in range(start, stop, step):
                        for body_stmt in stmt.body:
                            unrolled = _subst_stmt(body_stmt, var_name, float(i))
                            # Run constant folding on the substituted statement
                            unrolled = _opt_stmt(unrolled)
                            result.append(unrolled)
                    continue
            result.append(stmt)
        elif isinstance(stmt, WhileLoop):
            stmt.body = _unroll_small_loops(stmt.body)
            result.append(stmt)
        else:
            result.append(stmt)
    return result
