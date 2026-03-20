"""
TEX Optimizer — AST transformation passes for compile-time optimization.

Passes:
  1. Constant folding: evaluate expressions with all-literal operands at compile time
  2. Algebraic simplification: x*0 -> 0, x*1 -> x, pow(x,2) -> x*x, etc.

All passes preserve semantic equivalence. Applied after type checking, before
interpretation. Operates on the AST in-place (mutates nodes).
"""
from __future__ import annotations
import math

from .ast_nodes import (
    ASTNode, Program, VarDecl, Assignment, IfElse, ForLoop, WhileLoop,
    ExprStatement, BreakStmt, ContinueStmt, ParamDecl, ArrayDecl,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor,
    MatConstructor, CastExpr, ArrayIndexAccess, ArrayLiteral,
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

    return program


# ── Statement optimization ────────────────────────────────────────────

def _opt_stmt(stmt: ASTNode) -> ASTNode:
    """Optimize a single statement (recursive)."""
    if isinstance(stmt, VarDecl):
        if stmt.initializer:
            stmt.initializer = _opt_expr(stmt.initializer)
        return stmt

    if isinstance(stmt, Assignment):
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
        if stmt.initializer and isinstance(stmt.initializer, ArrayLiteral):
            stmt.initializer.elements = [_opt_expr(e) for e in stmt.initializer.elements]
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

    if isinstance(expr, CastExpr):
        expr.expr = _opt_expr(expr.expr)
        return expr

    if isinstance(expr, ArrayIndexAccess):
        expr.index = _opt_expr(expr.index)
        return expr

    # NumberLiteral, StringLiteral, Identifier, BindingRef, ChannelAccess — leaf nodes
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

    # lerp(a, b, 0) -> a, lerp(a, b, 1) -> b
    if name in ("lerp", "mix") and len(args) == 3 and _is_num_lit(args[2]):
        t = _num_val(args[2])
        if t == 0.0:
            return args[0]
        if t == 1.0:
            return args[1]

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

def _eliminate_dead_code(stmts: list[ASTNode]) -> list[ASTNode]:
    """Remove dead variable declarations/assignments.

    A local variable is "dead" if it is assigned but never read by any
    subsequent statement, binding assignment, or nested scope. Binding
    references (@name) and params ($name) are never eliminated.

    This is a conservative single-pass analysis — it only removes
    obviously dead assignments, not assignments made dead by later
    reassignments (which would require a more complex analysis).
    """
    # Step 1: collect all referenced variable names across the entire block
    used = _collect_used_names(stmts)

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

        # Recurse into compound statements
        if isinstance(stmt, IfElse):
            stmt.then_body = _eliminate_dead_code(stmt.then_body)
            stmt.else_body = _eliminate_dead_code(stmt.else_body)
        elif isinstance(stmt, ForLoop):
            stmt.body = _eliminate_dead_code(stmt.body)
        elif isinstance(stmt, WhileLoop):
            stmt.body = _eliminate_dead_code(stmt.body)

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
        # But ChannelAccess reads the base, and ArrayIndexAccess reads the index
        target = stmt.target
        if isinstance(target, ChannelAccess):
            _collect_used_in_expr(target.object, used)
        elif isinstance(target, ArrayIndexAccess):
            _collect_used_in_expr(target.array, used)
            _collect_used_in_expr(target.index, used)
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


def _collect_subexprs(expr: ASTNode, seen: dict[str, list], depth_threshold: int = _CSE_MIN_DEPTH):
    """Walk an expression and record all sub-expressions by hash.

    seen: maps hash -> [(expr_ref, parent_setter)] for occurrence tracking.
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
    # Walk the statements again to find first occurrence of each duplicate
    first_occurrence: dict[str, ASTNode] = {}  # hash -> first expr node

    def _find_first(expr: ASTNode):
        h = _expr_hash(expr)
        if h is not None and h in duplicates and h not in first_occurrence:
            first_occurrence[h] = expr
            # Don't recurse into children of a found duplicate —
            # we want the outermost match
            return
        if isinstance(expr, BinOp):
            _find_first(expr.left)
            _find_first(expr.right)
        elif isinstance(expr, UnaryOp):
            _find_first(expr.operand)
        elif isinstance(expr, FunctionCall):
            for a in expr.args:
                _find_first(a)
        elif isinstance(expr, VecConstructor):
            for a in expr.args:
                _find_first(a)
        elif isinstance(expr, ChannelAccess):
            _find_first(expr.object)
        elif isinstance(expr, CastExpr):
            _find_first(expr.expr)
        elif isinstance(expr, TernaryOp):
            _find_first(expr.condition)
            _find_first(expr.true_expr)
            _find_first(expr.false_expr)

    for stmt in stmts:
        if isinstance(stmt, VarDecl) and stmt.initializer:
            _find_first(stmt.initializer)
        elif isinstance(stmt, Assignment):
            _find_first(stmt.value)
        elif isinstance(stmt, ExprStatement):
            _find_first(stmt.expr)

    if not first_occurrence:
        return stmts

    # Assign temp variable names and build replacements
    replacements: dict[str, str] = {}  # hash -> temp var name
    temp_decls: list[VarDecl] = []
    for i, (h, expr_node) in enumerate(first_occurrence.items()):
        temp_name = f"_cse{i}"
        replacements[h] = temp_name
        temp_decls.append(VarDecl(
            loc=expr_node.loc,
            type_name="float",  # Type doesn't matter at this stage — type checker already ran
            name=temp_name,
            initializer=expr_node,
        ))

    # Replace all occurrences in all statements
    for stmt in stmts:
        _replace_in_stmt(stmt, replacements)

    # Insert temp decls at the beginning of the block
    return temp_decls + stmts
