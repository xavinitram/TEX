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
import copy
import math
from collections.abc import Callable

from .ast_nodes import (
    ASTNode, Program, VarDecl, Assignment, IfElse, ForLoop, WhileLoop,
    ExprStatement, BreakStmt, ContinueStmt, ArrayDecl,
    FunctionDef, ReturnStmt, BindingIndexAccess, BindingSampleAccess,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor,
    MatConstructor, CastExpr, ArrayIndexAccess, ArrayLiteral, SourceLoc,
    try_extract_static_range,
    iter_child_nodes as _iter_children,
)
from .type_checker import TEXType

# Reverse of TYPE_NAME_MAP: TEXType -> declaration keyword. Used so temp
# VarDecls synthesized by CSE/LICM declare the type they actually hold (vec/mat,
# not a hardcoded "float") and the optimized AST stays type-consistent.
_TYPE_TO_NAME = {
    TEXType.FLOAT: "float", TEXType.INT: "int",
    TEXType.VEC2: "vec2", TEXType.VEC3: "vec3", TEXType.VEC4: "vec4",
    TEXType.MAT3: "mat3", TEXType.MAT4: "mat4",
}


def _clone_expr(expr: ASTNode) -> ASTNode:
    """Deep-copy an expression subtree so it is not aliased into multiple tree
    positions. Passes that mutate the AST in place assume no shared subtrees."""
    return copy.deepcopy(expr)

# Pure math functions safe for constant folding (no side effects, deterministic)
_PURE_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    # asin/acos clamp their arg to [-1,1] to mirror the runtime domain guard
    # (fn_asin/fn_acos), matching how sqrt/log are clamped below. Without this
    # a constant out-of-domain fold (e.g. asin(2.0)) would raise ValueError.
    "asin": lambda x: math.asin(max(-1.0, min(1.0, x))),
    "acos": lambda x: math.acos(max(-1.0, min(1.0, x))),
    "atan": math.atan,
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
    # Mirror fn_smoothstep exactly: clamp((x-e0)/(e1-e0+SAFE_EPSILON), 0, 1)
    # then t*t*(3-2*t), with no separate early-out branches. The previous
    # early-out + max(e1-e0, 1e-8) form diverged from the runtime when e0 > e1.
    "smoothstep": lambda e0, e1, x: (
        (lambda t: t * t * (3 - 2 * t))(
            min(1.0, max(0.0, (x - e0) / (e1 - e0 + 1e-8)))
        )
    ),
    "step": lambda edge, x: 0.0 if x < edge else 1.0,
    "mod": lambda a, b: math.fmod(a, b) if b != 0 else 0.0,
    "radians": math.radians, "degrees": math.degrees,
}


def optimize(program: Program, type_map: dict | None = None) -> Program:
    """Run all optimization passes on a program AST (in-place).

    Passes (in order):
      1. Constant folding + algebraic simplification (per-expression)
      2. Dead code elimination (remove unused variable assignments)
      3. Common Subexpression Elimination (CSE) within basic blocks
      4. Loop-Invariant Code Motion (LICM) — hoist invariant expressions
      5. Small loop unrolling (preserving nested loops for stencil detection)

    If type_map (id(node) -> TEXType, from the type checker) is supplied, nodes
    synthesized by CSE/LICM are registered in it so the post-optimization AST and
    type_map stay consistent — otherwise id()-keyed lookups miss those nodes.
    """
    # Pass 0: constant-propagate literal locals (UC-4) so the folds below fire on
    # TEX's named-tuning-constant style (`float gamma = 1.0; ... pow(g, 1.0/gamma)`).
    program.statements = _propagate_literal_locals(program.statements)

    # Pass 1: constant folding + algebraic simplification
    for i, stmt in enumerate(program.statements):
        program.statements[i] = _opt_stmt(stmt)

    # Pass 2: dead code elimination
    program.statements = _eliminate_dead_code(program.statements)

    # Pass 3: common subexpression elimination
    program.statements = _eliminate_common_subexpressions(program.statements, type_map)

    # Pass 3.5: re-run DCE (Q-2). CSE creates a merged temp but leaves the
    # superseded inner temps as dead-but-executed VarDecls; and the now
    # purity-aware _has_side_effects can delete unused pure-call decls that the
    # first DCE pass (before the whitelist) had to keep. The post-optimize
    # re-typecheck (in compile_tex / compile_fused) covers the removed nodes.
    program.statements = _eliminate_dead_code(program.statements)

    # Pass 4: loop-invariant code motion
    program.statements = _hoist_loop_invariants(program.statements, type_map=type_map)

    # Pass 5: small loop unrolling (after LICM so hoisted vars are already out)
    program.statements = _unroll_small_loops(program.statements)

    return program


# ── UC-4: constant propagation of literal locals ──────────────────────

def _subst_all_literals(expr, subs: dict) -> ASTNode:
    if expr is None:
        return None
    for name, (val, is_int) in subs.items():
        expr = _subst_expr(expr, name, val, is_int)
    return expr


def _subst_stmt_literals(stmt: ASTNode, subs: dict) -> ASTNode:
    """Substitute the eligible literal locals throughout a statement, recursing
    into control-flow bodies but NOT function bodies (a separate scope)."""
    cls = stmt.__class__
    if cls is FunctionDef:
        return stmt
    if cls is VarDecl:
        stmt.initializer = _subst_all_literals(stmt.initializer, subs)
    elif cls is ArrayDecl:
        stmt.initializer = _subst_all_literals(stmt.initializer, subs)
    elif cls is Assignment:
        stmt.value = _subst_all_literals(stmt.value, subs)
        stmt.target = _subst_all_literals(stmt.target, subs)
    elif cls is ExprStatement:
        stmt.expr = _subst_all_literals(stmt.expr, subs)
    elif cls is ReturnStmt:
        stmt.value = _subst_all_literals(stmt.value, subs)
    elif cls is IfElse:
        stmt.condition = _subst_all_literals(stmt.condition, subs)
        stmt.then_body = [_subst_stmt_literals(s, subs) for s in stmt.then_body]
        stmt.else_body = [_subst_stmt_literals(s, subs) for s in stmt.else_body]
    elif cls is ForLoop:
        stmt.init = _subst_stmt_literals(stmt.init, subs)
        stmt.condition = _subst_all_literals(stmt.condition, subs)
        stmt.update = _subst_stmt_literals(stmt.update, subs)
        stmt.body = [_subst_stmt_literals(s, subs) for s in stmt.body]
    elif cls is WhileLoop:
        stmt.condition = _subst_all_literals(stmt.condition, subs)
        stmt.body = [_subst_stmt_literals(s, subs) for s in stmt.body]
    return stmt


def _propagate_literal_locals(statements: list[ASTNode]) -> list[ASTNode]:
    """Substitute top-level float/int locals that are (a) initialized to a
    NumberLiteral, (b) never reassigned, (c) declared exactly once and never a
    function parameter or loop variable — with that literal, so the existing
    folds fire. Scope-aware: a name shadowed anywhere (param / redeclare) is
    excluded, and function bodies are never substituted into (verified: a
    function param may legally shadow a top-level literal local)."""
    reassigned: set[str] = set()
    params: set[str] = set()
    loopvars: set[str] = set()
    decl_count: dict[str, int] = {}
    stack = list(statements)
    while stack:
        n = stack.pop()
        cls = n.__class__
        # UC-4: an ArrayDecl shadowing a top-level literal local must count as a
        # declaration too, or const-prop substitutes the scalar into the array's
        # index sites (`g[i]` on a NumberLiteral) and the mandatory re-typecheck
        # rejects a program that was legal in v0.14.1.
        if cls is VarDecl or cls is ArrayDecl:
            decl_count[n.name] = decl_count.get(n.name, 0) + 1
        elif cls is Assignment:
            t = n.target
            if isinstance(t, Identifier):
                reassigned.add(t.name)
            elif isinstance(t, ChannelAccess) and isinstance(t.object, Identifier):
                reassigned.add(t.object.name)
            elif isinstance(t, ArrayIndexAccess) and isinstance(t.array, Identifier):
                reassigned.add(t.array.name)
        elif cls is FunctionDef:
            for (_pt, pn) in n.params:
                params.add(pn)
        elif cls is ForLoop and isinstance(n.init, VarDecl):
            loopvars.add(n.init.name)
        stack.extend(_iter_children(n))

    subs: dict[str, tuple] = {}
    for stmt in statements:  # top level only
        if (stmt.__class__ is VarDecl and stmt.initializer.__class__ is NumberLiteral
                and stmt.name not in reassigned and stmt.name not in params
                and stmt.name not in loopvars and decl_count.get(stmt.name, 0) == 1):
            subs[stmt.name] = (stmt.initializer.value, stmt.initializer.is_int)
    if not subs:
        return statements
    return [_subst_stmt_literals(s, subs) for s in statements]


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


def _make_num(value: float, loc=None, is_int: bool = False) -> NumberLiteral:
    """Create a NumberLiteral from a computed value.

    is_int propagates integer-ness so a folded integer subexpression stays
    INT-typed after the post-optimization re-type-check (the type checker maps
    is_int -> INT/FLOAT, which affects array indexing, %, and INT dispatch).
    """
    return NumberLiteral(loc=loc or SourceLoc(0, 0), value=float(value),
                         is_int=is_int)


def _fold_binop(node: BinOp) -> ASTNode:
    """Constant fold binary operations + algebraic simplification."""
    left, right = node.left, node.right

    # Full constant fold: both operands are literals
    if _is_num_lit(left) and _is_num_lit(right):
        a, b = _num_val(left), _num_val(right)
        result = _eval_binop_const(node.op, a, b)
        if result is not None:
            # Preserve integer-ness: folding two INT operands under an
            # integer-closed op (+,-,*,%) yields an INT literal, so the
            # re-type-check keeps it INT instead of demoting to FLOAT.
            fold_int = (left.is_int and right.is_int
                        and node.op in ("+", "-", "*", "%"))
            return _make_num(result, node.loc, is_int=fold_int)

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
            # (multiplication is faster than division on most hardware).
            # Restrict to power-of-two divisors, where 1/c is exactly
            # representable in IEEE-754 so x*(1/c) is bit-identical to x/c.
            # For other divisors (e.g. 3.0 -> 0.3333333432) the reciprocal is
            # inexact and the rewrite would shift per-pixel results by 1 ULP.
            if rv != 0.0 and math.log2(abs(rv)).is_integer():
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
            # Negation is type-preserving, so keep the operand's integer-ness
            # (folding `-1` to a FLOAT literal would fail the re-type-check of
            # `int x = -1;` — _is_assignable rejects float -> int).
            return _make_num(-val, node.loc, is_int=node.operand.is_int)
        if node.op == "!":
            # `!` always yields FLOAT in the type checker — never propagate is_int.
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
        # pow(x, 2) -> x * x (strength reduction). Deep-copy the repeated uses so
        # the same subtree is not aliased into multiple positions (later passes
        # mutate in place and assume no shared subtrees).
        if exp == 2.0:
            return BinOp(loc=node.loc, op="*", left=args[0], right=_clone_expr(args[0]))
        # pow(x, 3) -> x * x * x
        if exp == 3.0:
            x_sq = BinOp(loc=node.loc, op="*", left=args[0], right=_clone_expr(args[0]))
            return BinOp(loc=node.loc, op="*", left=x_sq, right=_clone_expr(args[0]))
        # pow(x, 0.5) -> sqrt(x) is NOT applied: fn_sqrt clamps its arg to min 0
        # while fn_pow preserves NaN for negative bases with fractional exponents
        # (pow(-2, 0.5) is NaN). Rewriting to sqrt would silently turn NaN into 0
        # for signed inputs — not semantics-preserving — so leave pow(x, 0.5) as is.
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
                # Dead variable — skip it, unless the initializer has side
                # effects (same guard as dead assignments below)
                if stmt.initializer is None or not _has_side_effects(stmt.initializer):
                    continue
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

    elif isinstance(expr, BindingRef):
        # Record binding reads (@name/$name) in the same namespace as
        # _collect_reassigned_in_stmt's binding write targets, so the CSE
        # reassignment guard correctly blocks hoisting an expression across a
        # binding reassignment (bindings are reassignable at runtime).
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

    # NumberLiteral, StringLiteral — nothing to collect


def _has_side_effects(expr: ASTNode) -> bool:
    """Check if an expression might have side effects (conservative).

    Function calls count because user functions can persist scatter writes to
    bindings; indexed/sampled binding reads count because they can raise at
    runtime (e.g. fetching from a non-image binding). Wrapper nodes (casts,
    swizzles, indexing, constructors) recurse so a wrapped call or binding
    read is still preserved.
    """
    if isinstance(expr, FunctionCall):
        # Pure builtins (the CSE/LICM whitelist) have no side effects — but a
        # wrapped binding read or impure arg still does, so recurse into args
        # (Q-2: type_checker forbids redefining builtin names, so name-keyed
        # purity can't be spoofed). Non-whitelisted / user calls stay impure.
        if expr.name in _CSE_PURE_FUNCTIONS:
            return any(_has_side_effects(a) for a in expr.args)
        return True
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
    if isinstance(expr, MatConstructor):
        return any(_has_side_effects(a) for a in expr.args)
    if isinstance(expr, ArrayLiteral):
        return any(_has_side_effects(e) for e in expr.elements)
    if isinstance(expr, CastExpr):
        return _has_side_effects(expr.expr)
    if isinstance(expr, ChannelAccess):
        return _has_side_effects(expr.object)
    if isinstance(expr, ArrayIndexAccess):
        return _has_side_effects(expr.array) or _has_side_effects(expr.index)
    if isinstance(expr, (BindingIndexAccess, BindingSampleAccess)):
        return True  # Reads from bindings — preserve
    return False


# ── Common Subexpression Elimination (CSE) ───────────────────────────

# Minimum expression complexity to consider for CSE (avoid hoisting trivial nodes)
_CSE_MIN_DEPTH = 2

# Pure functions safe for CSE and LICM (deterministic, no side effects, no
# sampling — sample/fetch/img_* are binding reads and noise is excluded
# pending a determinism audit). Deliberately absent: inverse (raises on
# singular matrices, so hoisting it ahead of a zero-trip loop could turn a
# working program into a crashing one) and all string/array functions
# (STRING/ARRAY are missing from _TYPE_TO_NAME, so a synthesized temp VarDecl
# would be mistyped and fail the cache's re-type-check).
_CSE_PURE_FUNCTIONS = frozenset(_PURE_FUNCTIONS.keys()) | frozenset({
    "normalize", "length", "distance", "dot", "cross", "reflect",
    "rgb2hsv", "hsv2rgb", "luma", "hypot", "trunc",
    "spow", "sdiv", "sincos", "isnan", "isinf",
})


def _intern_key(intern: dict[tuple, int], sig: tuple) -> int:
    """Hash-cons a structural signature: equal signatures -> the same small int."""
    return intern.setdefault(sig, len(intern))


_NO_READS: frozenset[str] = frozenset()


def _expr_info(expr: ASTNode, memo: dict[int, tuple],
               intern: dict[tuple, int]) -> tuple:
    """Memoized per-node facts for CSE/LICM, computed in one bottom-up walk.

    Returns (hash_key, depth, reads, pure):
      hash_key: interned structural id (int) — equal iff the subtrees are
                structurally identical — or None for non-CSE candidates
                (leaves are hashable but string literals, impure calls, and
                array/matrix/binding accesses are not).
      depth:    expression nesting depth (leaves are 0).
      reads:    frozenset of variable/binding names the expression reads.
      pure:     safe to hoist for LICM — allows known-pure stdlib functions
                but rejects sampling, binding access, and user functions.

    memo is keyed by id(node) and must stay local to one CSE/LICM invocation:
    within an invocation the collect/find/replace phases each visit parents
    before children and mutation only happens at or below the visit point, so
    entries never go stale mid-pass — but a fixpoint loop would need a fresh
    memo per round. A shared long-lived memo would also risk id() reuse
    across unrelated ASTs.
    """
    info = memo.get(id(expr))
    if info is not None:
        return info

    if isinstance(expr, NumberLiteral):
        # repr(value) canonicalizes the float; is_int keeps INT-typed literals
        # from sharing a CSE temp with FLOAT-typed ones (dtype promotion).
        info = (_intern_key(intern, ("N", repr(expr.value), expr.is_int)),
                0, _NO_READS, True)
    elif isinstance(expr, StringLiteral):
        info = (None, 0, _NO_READS, True)  # Not worth CSE
    elif isinstance(expr, Identifier):
        info = (_intern_key(intern, ("I", expr.name)),
                0, frozenset((expr.name,)), True)
    elif isinstance(expr, BindingRef):
        info = (_intern_key(intern, ("B", expr.name)),
                0, frozenset((expr.name,)), True)
    elif isinstance(expr, BinOp):
        lh, ld, lr, lp = _expr_info(expr.left, memo, intern)
        rh, rd, rr, rp = _expr_info(expr.right, memo, intern)
        h = (None if lh is None or rh is None
             else _intern_key(intern, ("O", expr.op, lh, rh)))
        info = (h, 1 + max(ld, rd), lr | rr, lp and rp)
    elif isinstance(expr, UnaryOp):
        oh, od, orr, opure = _expr_info(expr.operand, memo, intern)
        h = None if oh is None else _intern_key(intern, ("U", expr.op, oh))
        info = (h, 1 + od, orr, opure)
    elif isinstance(expr, FunctionCall):
        infos = [_expr_info(a, memo, intern) for a in expr.args]
        is_pure_fn = expr.name in _CSE_PURE_FUNCTIONS
        if not is_pure_fn or any(i[0] is None for i in infos):
            h = None  # Side-effectful or nondeterministic
        else:
            h = _intern_key(intern, ("F", expr.name, *(i[0] for i in infos)))
        info = (h,
                1 + max((i[1] for i in infos), default=0),
                _NO_READS.union(*(i[2] for i in infos)),
                is_pure_fn and all(i[3] for i in infos))
    elif isinstance(expr, ChannelAccess):
        oh, od, orr, opure = _expr_info(expr.object, memo, intern)
        h = None if oh is None else _intern_key(intern, ("C", expr.channels, oh))
        info = (h, 1 + od, orr, opure)
    elif isinstance(expr, VecConstructor):
        infos = [_expr_info(a, memo, intern) for a in expr.args]
        h = (None if any(i[0] is None for i in infos)
             else _intern_key(intern, ("V", expr.size, *(i[0] for i in infos))))
        info = (h,
                1 + max((i[1] for i in infos), default=0),
                _NO_READS.union(*(i[2] for i in infos)),
                all(i[3] for i in infos))
    elif isinstance(expr, CastExpr):
        eh, ed, er, ep = _expr_info(expr.expr, memo, intern)
        h = None if eh is None else _intern_key(intern, ("T", expr.target_type, eh))
        info = (h, 1 + ed, er, ep)
    elif isinstance(expr, TernaryOp):
        ch, cd, cr, cp = _expr_info(expr.condition, memo, intern)
        th, td, tr, tp = _expr_info(expr.true_expr, memo, intern)
        fh, fd, fr, fp = _expr_info(expr.false_expr, memo, intern)
        h = (None if ch is None or th is None or fh is None
             else _intern_key(intern, ("?", ch, th, fh)))
        info = (h, 1 + max(cd, td, fd), cr | tr | fr, cp and tp and fp)
    elif isinstance(expr, MatConstructor):
        info = (None, 0,
                _NO_READS.union(*(_expr_info(a, memo, intern)[2] for a in expr.args)),
                False)
    elif isinstance(expr, ArrayLiteral):
        info = (None, 0,
                _NO_READS.union(*(_expr_info(e, memo, intern)[2] for e in expr.elements)),
                False)
    elif isinstance(expr, ArrayIndexAccess):
        info = (None, 0,
                _expr_info(expr.array, memo, intern)[2]
                | _expr_info(expr.index, memo, intern)[2],
                False)
    elif isinstance(expr, (BindingIndexAccess, BindingSampleAccess)):
        # Sampling is not pure for LICM, and binding refs may be reassigned
        info = (None, 0,
                _expr_info(expr.binding, memo, intern)[2]
                | _NO_READS.union(*(_expr_info(a, memo, intern)[2] for a in expr.args)),
                False)
    else:
        info = (None, 0, _NO_READS, False)

    memo[id(expr)] = info
    return info


def _collect_subexprs(expr: ASTNode, seen: dict[int, int],
                      memo: dict[int, tuple], intern: dict[tuple, int],
                      depth_threshold: int = _CSE_MIN_DEPTH):
    """Walk an expression and count sub-expressions by hash.

    seen: maps hash -> occurrence count.
    Only records expressions with depth >= depth_threshold.
    """
    h, depth, _reads, _pure = _expr_info(expr, memo, intern)
    if h is not None and depth >= depth_threshold:
        if h not in seen:
            seen[h] = 1
        else:
            seen[h] += 1

    # Recurse into children
    if isinstance(expr, BinOp):
        _collect_subexprs(expr.left, seen, memo, intern, depth_threshold)
        _collect_subexprs(expr.right, seen, memo, intern, depth_threshold)
    elif isinstance(expr, UnaryOp):
        _collect_subexprs(expr.operand, seen, memo, intern, depth_threshold)
    elif isinstance(expr, FunctionCall):
        for a in expr.args:
            _collect_subexprs(a, seen, memo, intern, depth_threshold)
    elif isinstance(expr, VecConstructor):
        for a in expr.args:
            _collect_subexprs(a, seen, memo, intern, depth_threshold)
    elif isinstance(expr, ChannelAccess):
        _collect_subexprs(expr.object, seen, memo, intern, depth_threshold)
    elif isinstance(expr, CastExpr):
        _collect_subexprs(expr.expr, seen, memo, intern, depth_threshold)
    elif isinstance(expr, TernaryOp):
        _collect_subexprs(expr.condition, seen, memo, intern, depth_threshold)
        _collect_subexprs(expr.true_expr, seen, memo, intern, depth_threshold)
        _collect_subexprs(expr.false_expr, seen, memo, intern, depth_threshold)
    elif isinstance(expr, (BindingIndexAccess, BindingSampleAccess)):
        for a in expr.args:
            _collect_subexprs(a, seen, memo, intern, depth_threshold)


def _collect_subexprs_in_stmt(stmt: ASTNode, seen: dict[int, int],
                              memo: dict[int, tuple], intern: dict[tuple, int]):
    """Collect sub-expression hashes from a statement."""
    if isinstance(stmt, VarDecl):
        if stmt.initializer:
            _collect_subexprs(stmt.initializer, seen, memo, intern)
    elif isinstance(stmt, Assignment):
        _collect_subexprs(stmt.value, seen, memo, intern)
    elif isinstance(stmt, ExprStatement):
        _collect_subexprs(stmt.expr, seen, memo, intern)


def _replace_expr(expr: ASTNode, replacements: dict[int, str],
                  memo: dict[int, tuple], intern: dict[tuple, int],
                  type_map: dict | None = None,
                  hash_to_type: dict | None = None) -> ASTNode:
    """Replace sub-expressions whose hash matches a CSE temp variable.

    When type_map/hash_to_type are supplied, each synthesized Identifier is
    registered in type_map with the hoisted expression's type, keeping id()-keyed
    type lookups valid for the optimizer-created references.
    """
    h = _expr_info(expr, memo, intern)[0]
    if h is not None and h in replacements:
        ident = Identifier(loc=expr.loc, name=replacements[h])
        # hash_to_type is only populated when type_map is not None, so a truthy
        # hash_to_type already implies type_map is available.
        if hash_to_type and h in hash_to_type:
            type_map[id(ident)] = hash_to_type[h]
        return ident

    if isinstance(expr, BinOp):
        expr.left = _replace_expr(expr.left, replacements, memo, intern, type_map, hash_to_type)
        expr.right = _replace_expr(expr.right, replacements, memo, intern, type_map, hash_to_type)
    elif isinstance(expr, UnaryOp):
        expr.operand = _replace_expr(expr.operand, replacements, memo, intern, type_map, hash_to_type)
    elif isinstance(expr, FunctionCall):
        expr.args = [_replace_expr(a, replacements, memo, intern, type_map, hash_to_type) for a in expr.args]
    elif isinstance(expr, VecConstructor):
        expr.args = [_replace_expr(a, replacements, memo, intern, type_map, hash_to_type) for a in expr.args]
    elif isinstance(expr, ChannelAccess):
        expr.object = _replace_expr(expr.object, replacements, memo, intern, type_map, hash_to_type)
    elif isinstance(expr, CastExpr):
        expr.expr = _replace_expr(expr.expr, replacements, memo, intern, type_map, hash_to_type)
    elif isinstance(expr, TernaryOp):
        expr.condition = _replace_expr(expr.condition, replacements, memo, intern, type_map, hash_to_type)
        expr.true_expr = _replace_expr(expr.true_expr, replacements, memo, intern, type_map, hash_to_type)
        expr.false_expr = _replace_expr(expr.false_expr, replacements, memo, intern, type_map, hash_to_type)
    elif isinstance(expr, (BindingIndexAccess, BindingSampleAccess)):
        expr.args = [_replace_expr(a, replacements, memo, intern, type_map, hash_to_type) for a in expr.args]
    return expr


def _replace_in_stmt(stmt: ASTNode, replacements: dict[int, str],
                     memo: dict[int, tuple], intern: dict[tuple, int],
                     type_map: dict | None = None,
                     hash_to_type: dict | None = None):
    """Replace CSE sub-expressions within a statement."""
    if isinstance(stmt, VarDecl):
        if stmt.initializer:
            stmt.initializer = _replace_expr(stmt.initializer, replacements, memo, intern, type_map, hash_to_type)
    elif isinstance(stmt, Assignment):
        stmt.value = _replace_expr(stmt.value, replacements, memo, intern, type_map, hash_to_type)
    elif isinstance(stmt, ExprStatement):
        stmt.expr = _replace_expr(stmt.expr, replacements, memo, intern, type_map, hash_to_type)


def _eliminate_common_subexpressions(stmts: list[ASTNode],
                                     type_map: dict | None = None) -> list[ASTNode]:
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
            stmt.then_body = _eliminate_common_subexpressions(stmt.then_body, type_map)
            stmt.else_body = _eliminate_common_subexpressions(stmt.else_body, type_map)
        elif isinstance(stmt, ForLoop):
            stmt.body = _eliminate_common_subexpressions(stmt.body, type_map)
        elif isinstance(stmt, WhileLoop):
            stmt.body = _eliminate_common_subexpressions(stmt.body, type_map)
        elif isinstance(stmt, FunctionDef):
            stmt.body = _eliminate_common_subexpressions(stmt.body, type_map)

    # Per-invocation scratch tables for _expr_info: valid across the whole
    # collect -> find -> replace sequence because each phase visits parents
    # before children and mutation only happens in replace, at or below the
    # visit point. If a fixpoint loop is ever added, rebuild them per round.
    memo: dict[int, tuple] = {}
    intern: dict[tuple, int] = {}

    # Collect all sub-expression hashes across the block
    seen: dict[int, int] = {}  # hash -> occurrence count
    for stmt in stmts:
        _collect_subexprs_in_stmt(stmt, seen, memo, intern)

    # Find expressions that appear 2+ times
    duplicates = {h for h, count in seen.items() if count >= 2}
    if not duplicates:
        return stmts

    # Variables REASSIGNED anywhere in this block. A subexpression that reads one
    # of them is NOT safe to hoist — its value can change between occurrences
    # (e.g. `a = s*s; s = v; b = s*s;` — the two `s*s` differ). The structural
    # hash keys identifiers by name only, so without this guard CSE would share
    # one temp across the write and silently corrupt the result.
    reassigned = _collect_reassigned_vars(stmts)

    # Build replacement map and generate temp VarDecls
    # We need to find the actual expression node for each duplicate hash
    # Walk the statements again to find first occurrence of each duplicate,
    # tracking which statement index it was found in so we can insert the
    # temp decl just before that statement (not at the top of the block).
    first_occurrence: dict[int, tuple[ASTNode, int]] = {}  # hash -> (expr, stmt_index)

    def _find_first(expr: ASTNode, stmt_idx: int):
        h, _depth, reads, _pure = _expr_info(expr, memo, intern)
        if (h is not None and h in duplicates and h not in first_occurrence
                and not (reads & reassigned)):
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
    replacements: dict[int, str] = {}  # hash -> temp var name
    hash_to_type: dict[int, TEXType] = {}  # hash -> hoisted expr's type (if known)
    # Group CSE decls by the statement index they should be inserted before
    insert_before: dict[int, list[VarDecl]] = {}  # stmt_index -> [VarDecl, ...]
    for i, (h, (expr_node, stmt_idx)) in enumerate(first_occurrence.items()):
        # The hoisted node is usually an original AST node, so its type is in
        # type_map. Carry it onto the temp (declared type + register the
        # synthesized refs) so the post-optimization AST stays type-consistent —
        # id()-keyed lookups would otherwise miss these new nodes (vec-component
        # miscounts, etc.).
        t = type_map.get(id(expr_node)) if type_map is not None else None
        # If a type_map was supplied but this node's type is unknown, it was
        # synthesized by an earlier pass (e.g. const-folded vec*vec) and is not
        # in type_map. Declaring a temp of a guessed type would mistype it, so
        # skip hoisting it — left inline, the post-optimization re-type-check
        # assigns it the correct type. (When no type_map is supplied at all the
        # caller has opted out of type tracking; preserve the old float default.)
        if type_map is not None and t is None:
            continue
        temp_name = f"_cse{i}"
        replacements[h] = temp_name
        decl = VarDecl(
            loc=expr_node.loc,
            type_name=_TYPE_TO_NAME.get(t, "float"),
            name=temp_name,
            initializer=expr_node,
        )
        if type_map is not None and t is not None:
            type_map[id(decl)] = t
            hash_to_type[h] = t
        insert_before.setdefault(stmt_idx, []).append(decl)

    # Replace all occurrences in all statements
    for stmt in stmts:
        _replace_in_stmt(stmt, replacements, memo, intern, type_map, hash_to_type)

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


def _collect_reassigned_vars(stmts: list[ASTNode]) -> set[str]:
    """Names REASSIGNED (Assignment target) in a block — not merely declared once.
    A subexpression reading a reassigned variable cannot be CSE-hoisted, since its
    value can change between occurrences (declarations alone are stable)."""
    out: set[str] = set()
    for stmt in stmts:
        _collect_reassigned_in_stmt(stmt, out)
    return out


def _collect_reassigned_in_stmt(stmt: ASTNode, out: set[str]):
    if isinstance(stmt, Assignment):
        t = stmt.target
        if isinstance(t, (Identifier, BindingRef)):
            out.add(t.name)
        elif isinstance(t, ChannelAccess) and isinstance(t.object, (Identifier, BindingRef)):
            out.add(t.object.name)
        elif isinstance(t, ArrayIndexAccess) and isinstance(t.array, Identifier):
            out.add(t.array.name)
        elif isinstance(t, BindingIndexAccess) and isinstance(t.binding, BindingRef):
            out.add(t.binding.name)
    elif isinstance(stmt, IfElse):
        for s in stmt.then_body:
            _collect_reassigned_in_stmt(s, out)
        for s in stmt.else_body:
            _collect_reassigned_in_stmt(s, out)
    elif isinstance(stmt, (ForLoop, WhileLoop)):
        for s in stmt.body:
            _collect_reassigned_in_stmt(s, out)
    elif isinstance(stmt, FunctionDef):
        for s in stmt.body:
            _collect_reassigned_in_stmt(s, out)


def _is_loop_invariant(expr: ASTNode, modified: set[str],
                       memo: dict[int, tuple], intern: dict[tuple, int]) -> bool:
    """Check if an expression is loop-invariant (doesn't depend on modified vars).

    An expression is loop-invariant if:
    - It is pure (no side effects, no sampling; see _expr_info)
    - None of its referenced variables are in the modified set
    - It has sufficient depth to be worth hoisting
    """
    _h, depth, reads, pure = _expr_info(expr, memo, intern)
    if not pure:
        return False
    if depth < _LICM_MIN_DEPTH:
        return False
    return reads.isdisjoint(modified)


def _extract_invariant_subexpr(expr: ASTNode, modified: set[str],
                                hoisted: list[tuple[str, ASTNode, TEXType | None]],
                                counter: list[int],
                                type_map: dict | None,
                                memo: dict[int, tuple],
                                intern: dict[tuple, int]) -> ASTNode:
    """Walk an expression tree and replace loop-invariant subtrees with temp vars.

    Replaces the largest (outermost) invariant subtrees first. If the entire
    expression is invariant, replaces the whole thing. Otherwise recurses into
    children to find smaller invariant sub-expressions.

    When type_map is supplied, the hoisted subtree's type is looked up so the
    synthesized temp declares the type it actually holds (vec/mat, not a
    hardcoded float) and the new Identifier reference is registered — keeping
    id()-keyed type lookups valid for the optimizer-created nodes.
    """
    # Check if the entire expression is invariant
    if _is_loop_invariant(expr, modified, memo, intern):
        t = type_map.get(id(expr)) if type_map is not None else None
        # If a type_map is supplied but this subtree's type is unknown, it was
        # synthesized by an earlier pass (e.g. const-folded vec*vec) and isn't
        # tracked. Hoisting it into a temp of a guessed type would mistype it,
        # so don't hoist this node — recurse into its children instead (a
        # smaller, typed invariant subtree may still be hoistable). The
        # post-optimization re-type-check assigns the correct type to whatever
        # stays inline.
        if type_map is None or t is not None:
            temp_name = f"_licm{counter[0]}"
            counter[0] += 1
            ident = Identifier(loc=expr.loc, name=temp_name)
            if t is not None:
                type_map[id(ident)] = t
            hoisted.append((temp_name, expr, t))
            return ident

    # Otherwise recurse into children to find invariant sub-expressions
    if isinstance(expr, BinOp):
        expr.left = _extract_invariant_subexpr(expr.left, modified, hoisted, counter, type_map, memo, intern)
        expr.right = _extract_invariant_subexpr(expr.right, modified, hoisted, counter, type_map, memo, intern)
    elif isinstance(expr, UnaryOp):
        expr.operand = _extract_invariant_subexpr(expr.operand, modified, hoisted, counter, type_map, memo, intern)
    elif isinstance(expr, FunctionCall):
        expr.args = [_extract_invariant_subexpr(a, modified, hoisted, counter, type_map, memo, intern) for a in expr.args]
    elif isinstance(expr, VecConstructor):
        expr.args = [_extract_invariant_subexpr(a, modified, hoisted, counter, type_map, memo, intern) for a in expr.args]
    elif isinstance(expr, TernaryOp):
        expr.condition = _extract_invariant_subexpr(expr.condition, modified, hoisted, counter, type_map, memo, intern)
        expr.true_expr = _extract_invariant_subexpr(expr.true_expr, modified, hoisted, counter, type_map, memo, intern)
        expr.false_expr = _extract_invariant_subexpr(expr.false_expr, modified, hoisted, counter, type_map, memo, intern)
    elif isinstance(expr, CastExpr):
        expr.expr = _extract_invariant_subexpr(expr.expr, modified, hoisted, counter, type_map, memo, intern)
    elif isinstance(expr, ChannelAccess):
        expr.object = _extract_invariant_subexpr(expr.object, modified, hoisted, counter, type_map, memo, intern)

    return expr


def _licm_stmt(stmt: ASTNode, modified: set[str],
               hoisted: list[tuple[str, ASTNode, TEXType | None]], counter: list[int],
               type_map: dict | None,
               memo: dict[int, tuple], intern: dict[tuple, int]):
    """Extract loop-invariant expressions from a single statement in a loop body."""
    if isinstance(stmt, VarDecl):
        if stmt.initializer:
            stmt.initializer = _extract_invariant_subexpr(
                stmt.initializer, modified, hoisted, counter, type_map, memo, intern)
    elif isinstance(stmt, Assignment):
        stmt.value = _extract_invariant_subexpr(stmt.value, modified, hoisted, counter, type_map, memo, intern)
    elif isinstance(stmt, ExprStatement):
        stmt.expr = _extract_invariant_subexpr(stmt.expr, modified, hoisted, counter, type_map, memo, intern)
    elif isinstance(stmt, ReturnStmt):
        if stmt.value:
            stmt.value = _extract_invariant_subexpr(stmt.value, modified, hoisted, counter, type_map, memo, intern)
    elif isinstance(stmt, IfElse):
        stmt.condition = _extract_invariant_subexpr(stmt.condition, modified, hoisted, counter, type_map, memo, intern)
        for s in stmt.then_body:
            _licm_stmt(s, modified, hoisted, counter, type_map, memo, intern)
        for s in stmt.else_body:
            _licm_stmt(s, modified, hoisted, counter, type_map, memo, intern)


def _licm_loop(loop: ForLoop | WhileLoop, counter: list[int],
               type_map: dict | None = None) -> list[ASTNode]:
    """Apply LICM to a single loop. Returns [hoisted_decls..., loop]."""
    # Collect all variables written inside the loop body
    modified = _collect_written_vars(loop.body)

    # For for-loops, the loop variable and update target are also modified
    if isinstance(loop, ForLoop):
        if isinstance(loop.init, VarDecl):
            modified.add(loop.init.name)
        if isinstance(loop.update, Assignment) and isinstance(loop.update.target, Identifier):
            modified.add(loop.update.target.name)

    # Extract invariant sub-expressions from body statements. The scratch
    # tables are per-loop: inner-loop LICM has already mutated this body
    # (hoisted decls, new _licm identifiers), so entries from another loop's
    # walk must not be reused.
    memo: dict[int, tuple] = {}
    intern: dict[tuple, int] = {}
    hoisted: list[tuple[str, ASTNode, TEXType | None]] = []
    for stmt in loop.body:
        _licm_stmt(stmt, modified, hoisted, counter, type_map, memo, intern)

    if not hoisted:
        return [loop]

    # Create VarDecl nodes for hoisted expressions. Declare each with the type
    # it actually holds (from the pre-optimization type_map) so the optimized
    # AST stays type-consistent and re-type-checks cleanly — a hardcoded "float"
    # would misdeclare hoisted vec/mat values and corrupt id()-keyed lookups.
    pre_loop: list[ASTNode] = []
    for temp_name, expr_node, t in hoisted:
        decl = VarDecl(
            loc=expr_node.loc,
            type_name=_TYPE_TO_NAME.get(t, "float"),
            name=temp_name,
            initializer=expr_node,
        )
        if type_map is not None and t is not None:
            type_map[id(decl)] = t
        pre_loop.append(decl)

    return pre_loop + [loop]


def _hoist_loop_invariants(stmts: list[ASTNode], counter: list[int] | None = None,
                           type_map: dict | None = None) -> list[ASTNode]:
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
            stmt.then_body = _hoist_loop_invariants(stmt.then_body, counter, type_map)
            stmt.else_body = _hoist_loop_invariants(stmt.else_body, counter, type_map)
            result.append(stmt)
        elif isinstance(stmt, FunctionDef):
            stmt.body = _hoist_loop_invariants(stmt.body, counter, type_map)
            result.append(stmt)
        elif isinstance(stmt, ForLoop):
            # Recurse into nested loops first
            stmt.body = _hoist_loop_invariants(stmt.body, counter, type_map)
            # Then hoist invariants from this loop
            result.extend(_licm_loop(stmt, counter, type_map))
        elif isinstance(stmt, WhileLoop):
            stmt.body = _hoist_loop_invariants(stmt.body, counter, type_map)
            result.extend(_licm_loop(stmt, counter, type_map))
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


def _subst_expr(expr: ASTNode, var_name: str, value: float,
                is_int: bool = False) -> ASTNode:
    """Substitute all occurrences of var_name with a NumberLiteral in an expression.

    Returns a new expression (does not mutate the original).

    is_int marks the substituted literal as INT (set when the loop variable is
    INT-typed) so unrolled bodies keep integer-context behavior — array indexing,
    %, and INT-typed dispatch — instead of silently becoming FLOAT.
    """
    if isinstance(expr, Identifier) and expr.name == var_name:
        return NumberLiteral(loc=expr.loc, value=value, is_int=is_int)
    if isinstance(expr, BinOp):
        return BinOp(loc=expr.loc, op=expr.op,
                     left=_subst_expr(expr.left, var_name, value, is_int),
                     right=_subst_expr(expr.right, var_name, value, is_int))
    if isinstance(expr, UnaryOp):
        return UnaryOp(loc=expr.loc, op=expr.op,
                       operand=_subst_expr(expr.operand, var_name, value, is_int))
    if isinstance(expr, FunctionCall):
        return FunctionCall(loc=expr.loc, name=expr.name,
                            args=[_subst_expr(a, var_name, value, is_int) for a in expr.args])
    if isinstance(expr, TernaryOp):
        return TernaryOp(loc=expr.loc,
                         condition=_subst_expr(expr.condition, var_name, value, is_int),
                         true_expr=_subst_expr(expr.true_expr, var_name, value, is_int),
                         false_expr=_subst_expr(expr.false_expr, var_name, value, is_int))
    if isinstance(expr, VecConstructor):
        return VecConstructor(loc=expr.loc, size=expr.size,
                              args=[_subst_expr(a, var_name, value, is_int) for a in expr.args])
    if isinstance(expr, MatConstructor):
        return MatConstructor(loc=expr.loc, size=expr.size,
                              args=[_subst_expr(a, var_name, value, is_int) for a in expr.args])
    if isinstance(expr, ArrayLiteral):
        return ArrayLiteral(loc=expr.loc,
                            elements=[_subst_expr(e, var_name, value, is_int) for e in expr.elements])
    if isinstance(expr, CastExpr):
        return CastExpr(loc=expr.loc, target_type=expr.target_type,
                        expr=_subst_expr(expr.expr, var_name, value, is_int))
    if isinstance(expr, ChannelAccess):
        return ChannelAccess(loc=expr.loc, channels=expr.channels,
                             object=_subst_expr(expr.object, var_name, value, is_int))
    if isinstance(expr, ArrayIndexAccess):
        return ArrayIndexAccess(loc=expr.loc,
                                array=_subst_expr(expr.array, var_name, value, is_int),
                                index=_subst_expr(expr.index, var_name, value, is_int))
    if isinstance(expr, BindingIndexAccess):
        return BindingIndexAccess(loc=expr.loc,
                                  binding=_subst_expr(expr.binding, var_name, value, is_int),
                                  args=[_subst_expr(a, var_name, value, is_int) for a in expr.args])
    if isinstance(expr, BindingSampleAccess):
        return BindingSampleAccess(loc=expr.loc,
                                   binding=_subst_expr(expr.binding, var_name, value, is_int),
                                   args=[_subst_expr(a, var_name, value, is_int) for a in expr.args])
    # NumberLiteral, StringLiteral, BindingRef — no substitution needed
    return expr


def _subst_stmt(stmt: ASTNode, var_name: str, value: float,
                is_int: bool = False) -> ASTNode:
    """Substitute loop variable in a statement. Returns new statement."""
    if isinstance(stmt, VarDecl):
        init = _subst_expr(stmt.initializer, var_name, value, is_int) if stmt.initializer else None
        return VarDecl(loc=stmt.loc, type_name=stmt.type_name, name=stmt.name,
                       initializer=init, is_const=stmt.is_const)
    if isinstance(stmt, Assignment):
        new_target = _subst_expr(stmt.target, var_name, value, is_int)
        new_value = _subst_expr(stmt.value, var_name, value, is_int)
        return Assignment(loc=stmt.loc, target=new_target, value=new_value, op=stmt.op)
    if isinstance(stmt, ExprStatement):
        return ExprStatement(loc=stmt.loc, expr=_subst_expr(stmt.expr, var_name, value, is_int))
    if isinstance(stmt, IfElse):
        return IfElse(loc=stmt.loc,
                      condition=_subst_expr(stmt.condition, var_name, value, is_int),
                      then_body=[_subst_stmt(s, var_name, value, is_int) for s in stmt.then_body],
                      else_body=[_subst_stmt(s, var_name, value, is_int) for s in stmt.else_body])
    if isinstance(stmt, ReturnStmt):
        val = _subst_expr(stmt.value, var_name, value, is_int) if stmt.value else None
        return ReturnStmt(loc=stmt.loc, value=val)
    if isinstance(stmt, ForLoop):
        new_init = _subst_stmt(stmt.init, var_name, value, is_int)
        new_cond = _subst_expr(stmt.condition, var_name, value, is_int)
        new_update = _subst_stmt(stmt.update, var_name, value, is_int)
        new_body = [_subst_stmt(s, var_name, value, is_int) for s in stmt.body]
        return ForLoop(loc=stmt.loc, init=new_init, condition=new_cond,
                       update=new_update, body=new_body)
    if isinstance(stmt, WhileLoop):
        new_cond = _subst_expr(stmt.condition, var_name, value, is_int)
        new_body = [_subst_stmt(s, var_name, value, is_int) for s in stmt.body]
        return WhileLoop(loc=stmt.loc, condition=new_cond, body=new_body)
    if isinstance(stmt, ArrayDecl):
        init = _subst_expr(stmt.initializer, var_name, value, is_int) if stmt.initializer else None
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
                    # Unroll: emit body N times with loop var substituted.
                    # The static range yields an integer loop variable, so mark
                    # the substituted literal INT to preserve integer-context
                    # behavior (indexing, %, INT dispatch) after re-type-check.
                    for i in range(start, stop, step):
                        for body_stmt in stmt.body:
                            unrolled = _subst_stmt(body_stmt, var_name, float(i), is_int=True)
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
