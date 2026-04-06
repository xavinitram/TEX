"""
TEX Codegen — compile TEX AST to Python functions for zero-overhead execution.

Instead of tree-walking the AST on every frame, this module generates a Python
function string, compiles it via exec(), and caches the callable. Subsequent
executions call the function directly, eliminating:
  - Per-node dispatch table lookups
  - Per-node Python function call overhead
  - Redundant dict lookups for env/bindings

Falls back to None (caller uses tree-walking interpreter) for unsupported
patterns. Includes stencil specialization (avg_pool2d, max_pool2d, conv2d,
unfold), sample/fetch inlining with hoisted BCHW + grid buffers, and
function specializations (pow, luma, clamp, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tex_compiler.ast_nodes import (
    ASTNode, Program, VarDecl, Assignment, IfElse, ForLoop, WhileLoop,
    ExprStatement, BreakStmt, ContinueStmt, ParamDecl, ArrayDecl,
    FunctionDef, ReturnStmt,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor,
    MatConstructor, CastExpr, ArrayIndexAccess, ArrayLiteral,
    BindingIndexAccess, BindingSampleAccess,
    try_extract_static_range,
    collect_assigned_vars,
)
from ..tex_compiler.type_checker import TEXType, CHANNEL_MAP, TYPE_NAME_MAP
from .interpreter import MAX_CALL_DEPTH, _BUILTIN_NAMES


class _Unsupported(Exception):
    """Raised when codegen encounters a pattern it cannot handle."""
    pass


class _CgBreak(Exception):
    """Signal for break in codegen-generated code."""
    pass


class _CgContinue(Exception):
    """Signal for continue in codegen-generated code."""
    pass


# Operators that map directly to Python infix ops
_INFIX_OPS = {"+", "-", "*"}

# Comparison operators
_CMP_OPS = {
    "==": "==", "!=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">=",
}

_IMG_REDUCE_OPS = {
    "img_sum": "sum", "img_mean": "mean",
    "img_min": "amin", "img_max": "amax",
}


def try_compile(program: Program, type_map: dict[int, TEXType]) -> Any | None:
    """Try to compile a TEX program AST to a Python function.

    Returns a callable with signature:
        fn(env, bindings, functions, device, spatial_shape) -> None

    The function mutates env and bindings dicts in-place.
    Returns None if the program uses unsupported features.
    """
    try:
        gen = _CodeGen(type_map)
        gen.emit_program(program)
        fn = gen.build()
        # Attach metadata: whether the generated code has stdlib function calls.
        # Programs with fn calls have graph breaks that make torch.compile slower.
        fn._has_fn_calls = bool(gen._fn_locals)
        return fn
    except _Unsupported:
        return None
    except Exception:
        import logging
        logging.getLogger("TEX.codegen").debug(
            "Codegen internal error (falling back to interpreter)", exc_info=True
        )
        return None


# Stdlib functions that are simple torch.XXX(arg) wrappers.
# In codegen, arguments are always tensors so _to_tensor is a no-op.
# We emit _torch.XXX(arg) directly, avoiding dict lookup + function call + _to_tensor.
_INLINE_TORCH_1ARG: dict[str, str] = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "exp": "exp", "abs": "abs",
    "floor": "floor", "ceil": "ceil", "round": "round", "trunc": "trunc",
    "sign": "sign",
    "degrees": "rad2deg", "radians": "deg2rad",
}

# 2-arg torch functions: fn(a, b) -> torch.XXX(a, b)
_INLINE_TORCH_2ARG: dict[str, str] = {
    "max": "maximum", "min": "minimum",
    "atan2": "atan2", "hypot": "hypot",
}

# math module equivalents for scalar-mode emission (Python float path).
_SCALAR_MATH_1ARG: dict[str, str] = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "exp": "exp", "abs": "fabs",
    "floor": "floor", "ceil": "ceil", "trunc": "trunc",
    "sign": "copysign",  # special-cased below
}
_SCALAR_MATH_2ARG: dict[str, str] = {
    "max": "max", "min": "min",
    "atan2": "atan2", "hypot": "hypot",
}

# Builtins that are spatially-varying tensors at runtime (shape [B,H,W] or
# [1,1,W] etc.) despite being typed as FLOAT by the type checker.
# Used to exclude loops from scalar mode.
# NOTE: iw/ih/fi/fn are scalar (0-dim) so NOT included here.
_SPATIAL_BUILTINS: frozenset[str] = frozenset(("u", "v", "ix", "iy"))

# Stdlib functions that read/write spatial image data — incompatible with
# the scalar loop fast path.
_SPATIAL_STDLIB: frozenset[str] = frozenset((
    "sample", "fetch", "sample_cubic", "sample_mip", "sample_grad",
    "sample_frame", "fetch_frame", "gauss_blur", "sample_lanczos",
    "sample_mip_gauss", "bilateral_filter",
))




def _collect_sample_bindings(stmts: list[ASTNode]) -> set[str]:
    """Collect binding names used in sample()/fetch() calls within statements."""
    names: set[str] = set()
    stack = list(stmts)
    while stack:
        node = stack.pop()
        if isinstance(node, (BindingSampleAccess, BindingIndexAccess)):
            if isinstance(node.binding, BindingRef):
                names.add(node.binding.name)
        if isinstance(node, FunctionCall) and node.name in (
            "sample", "fetch", "sample_cubic", "sample_lanczos",
            "sample_mip", "sample_mip_gauss",
        ):
            if node.args and isinstance(node.args[0], BindingRef):
                names.add(node.args[0].name)
        # Recurse into children
        for attr in ("body", "then_body", "else_body"):
            children = getattr(node, attr, None)
            if children:
                stack.extend(children)
        for attr in ("init", "condition", "update", "value", "initializer",
                     "left", "right", "operand", "true_expr", "false_expr",
                     "object", "expr", "target", "binding", "array", "index"):
            child = getattr(node, attr, None)
            if child is not None and isinstance(child, ASTNode):
                stack.append(child)
        if isinstance(node, (FunctionCall, VecConstructor, MatConstructor)):
            stack.extend(node.args)
        if isinstance(node, ArrayLiteral):
            stack.extend(node.elements)
        if isinstance(node, (BindingIndexAccess, BindingSampleAccess)):
            stack.extend(node.args)
    return names


def _body_has_break_continue(stmts: list[ASTNode]) -> bool:
    """Check if a list of statements contains break or continue (not in nested loops)."""
    for stmt in stmts:
        if isinstance(stmt, (BreakStmt, ContinueStmt)):
            return True
        if isinstance(stmt, IfElse):
            if _body_has_break_continue(stmt.then_body):
                return True
            if stmt.else_body and _body_has_break_continue(stmt.else_body):
                return True
        # Don't recurse into nested loops — their break/continue is local to them
    return False


# ── Stencil pattern detection ──────────────────────────────────────────

@dataclass(slots=True)
class _StencilInfo:
    """Detected stencil pattern from nested for-loops or inline fetch sequences."""
    kind: str                # "box", "minmax", "median", "conv2d"
    binding_name: str        # The @binding being fetched/sampled
    is_fetch: bool           # True for fetch() pattern, False for sample()
    channels: str | None     # Channel swizzle ("rgb", "r", etc.) or None for all
    # Radius expression: int for static, or ASTNode for parameterized
    y_radius: int | ASTNode | None = None
    x_radius: int | ASTNode | None = None
    is_symmetric: bool = False
    # For asymmetric static ranges only:
    dy_start: int | None = None
    dy_stop: int | None = None   # exclusive
    dx_start: int | None = None
    dx_stop: int | None = None   # exclusive
    # Box blur
    accum_var: str | None = None
    count_var: str | None = None
    # Min/max pool
    minmax_op: str | None = None  # "max" or "min"
    result_var: str | None = None
    # Conv2d (inline stencil with fixed kernel weights)
    kernel_weights: dict | None = None  # {(dy, dx): float_weight}
    consumed_stmts: set | None = None   # indices of statements consumed by the stencil
    # Median (array collect pattern)
    array_vars: list | None = None      # [(arr_name, ch_idx|None)] per array


def _is_ident(node: ASTNode, name: str) -> bool:
    return isinstance(node, Identifier) and node.name == name


def _is_affine_fetch_coord(expr: ASTNode, base_var: str, loop_var: str) -> bool:
    """Check if expr is `base_var + loop_var`."""
    if not isinstance(expr, BinOp) or expr.op != "+":
        return False
    if _is_ident(expr.left, base_var) and _is_ident(expr.right, loop_var):
        return True
    if _is_ident(expr.left, loop_var) and _is_ident(expr.right, base_var):
        return True
    return False


def _ast_equal(a: ASTNode, b: ASTNode) -> bool:
    """Shallow structural equality for simple expression trees."""
    if type(a) is not type(b):
        return False
    if isinstance(a, Identifier):
        return a.name == b.name
    if isinstance(a, NumberLiteral):
        return a.value == b.value
    if isinstance(a, BindingRef):
        return a.name == b.name and a.kind == b.kind
    if isinstance(a, BinOp):
        return a.op == b.op and _ast_equal(a.left, b.left) and _ast_equal(a.right, b.right)
    if isinstance(a, UnaryOp):
        return a.op == b.op and _ast_equal(a.operand, b.operand)
    return False


def _try_extract_symmetric_range(loop: ForLoop) -> tuple[str, ASTNode | int] | None:
    """Detect for(int var = -R; var <= R; var++) where R is a constant or parameter.

    Returns (var_name, radius) where radius is int (static) or ASTNode (parameterized).
    """
    init = loop.init
    if not isinstance(init, VarDecl) or init.initializer is None:
        return None
    loop_var = init.name

    # Check update is var = var + 1
    upd = loop.update
    if not isinstance(upd, Assignment):
        return None
    if not _is_ident(upd.target, loop_var):
        return None
    if not isinstance(upd.value, BinOp) or upd.value.op != "+":
        return None
    if not _is_ident(upd.value.left, loop_var):
        return None
    if not (isinstance(upd.value.right, NumberLiteral) and upd.value.right.value == 1.0):
        return None

    # Check condition is var <= R
    cond = loop.condition
    if not isinstance(cond, BinOp) or cond.op != "<=":
        return None
    if not _is_ident(cond.left, loop_var):
        return None
    radius_expr = cond.right

    # Check init is var = -R
    init_val = init.initializer
    if isinstance(init_val, UnaryOp) and init_val.op == "-":
        if not _ast_equal(init_val.operand, radius_expr):
            return None
    elif isinstance(init_val, NumberLiteral) and isinstance(radius_expr, NumberLiteral):
        if init_val.value != -radius_expr.value:
            return None
    else:
        return None

    # Return radius as int or ASTNode
    if isinstance(radius_expr, NumberLiteral):
        return (loop_var, int(radius_expr.value))
    return (loop_var, radius_expr)


def _find_fetch_call(expr: ASTNode) -> FunctionCall | BindingIndexAccess | None:
    """Find a fetch() or sample() call inside an expression tree.

    Handles both FunctionCall("fetch"/"sample", ...) and BindingIndexAccess/BindingSampleAccess.
    """
    if isinstance(expr, FunctionCall) and expr.name in ("fetch", "sample"):
        return expr
    if isinstance(expr, (BindingIndexAccess, BindingSampleAccess)):
        return expr
    if isinstance(expr, ChannelAccess):
        return _find_fetch_call(expr.object)
    if isinstance(expr, BinOp):
        found = _find_fetch_call(expr.left)
        if found:
            return found
        return _find_fetch_call(expr.right)
    return None


def _is_affine_sample_coord(expr: ASTNode, base_var: str, loop_var: str) -> bool:
    """Check if expr is `base_var + float(loop_var) * px` or similar pixel-step patterns.

    Matches: u + float(dx) * px,  u + dx * px,  u + float(dx) / iw, etc.
    """
    if not isinstance(expr, BinOp) or expr.op != "+":
        return False
    if not _is_ident(expr.left, base_var):
        return False
    rhs = expr.right
    if not isinstance(rhs, BinOp):
        return False
    px_var = "px" if base_var == "u" else "py"
    dim_var = "iw" if base_var == "u" else "ih"

    def _is_loop_var(n: ASTNode) -> bool:
        return (_is_ident(n, loop_var) or
                (isinstance(n, CastExpr) and n.target_type == "float"
                 and _is_ident(n.expr, loop_var)))

    if rhs.op == "*":
        if _is_loop_var(rhs.left) and _is_ident(rhs.right, px_var):
            return True
        if _is_ident(rhs.left, px_var) and _is_loop_var(rhs.right):
            return True
    elif rhs.op == "/":
        if _is_loop_var(rhs.left) and _is_ident(rhs.right, dim_var):
            return True
    return False


def _resolve_coord(expr: ASTNode, local_defs: dict[str, ASTNode]) -> ASTNode:
    """Resolve identifiers through local variable definitions (transitively)."""
    seen: set[str] = set()
    while isinstance(expr, Identifier) and expr.name in local_defs:
        if expr.name in seen:
            break  # cycle guard
        seen.add(expr.name)
        expr = local_defs[expr.name]
    return expr


def _check_fetch_coords(fetch_node: ASTNode, inner_var: str, outer_var: str,
                        local_defs: dict[str, ASTNode] | None = None
                        ) -> tuple[str, bool] | None:
    """Validate fetch/sample coordinates are affine in loop vars.

    Returns (binding_name, is_fetch) or None.
    local_defs: mapping from variable name to its definition expression
                (for tracing through intermediate vars like su, sv).
    """
    if local_defs is None:
        local_defs = {}

    if isinstance(fetch_node, FunctionCall):
        if len(fetch_node.args) != 3:
            return None
        binding_node, coord_a, coord_b = fetch_node.args
        if not isinstance(binding_node, BindingRef):
            return None
        is_fetch = fetch_node.name == "fetch"
        if is_fetch:
            if not _is_affine_fetch_coord(coord_a, "ix", inner_var):
                return None
            if not _is_affine_fetch_coord(coord_b, "iy", outer_var):
                return None
        else:
            # sample() — resolve coords through local definitions
            resolved_a = _resolve_coord(coord_a, local_defs)
            resolved_b = _resolve_coord(coord_b, local_defs)
            if not _is_affine_sample_coord(resolved_a, "u", inner_var):
                return None
            if not _is_affine_sample_coord(resolved_b, "v", outer_var):
                return None
        return (binding_node.name, is_fetch)
    if isinstance(fetch_node, BindingIndexAccess):
        if len(fetch_node.args) != 2:
            return None
        if not _is_affine_fetch_coord(fetch_node.args[0], "ix", inner_var):
            return None
        if not _is_affine_fetch_coord(fetch_node.args[1], "iy", outer_var):
            return None
        name = fetch_node.binding.name if isinstance(fetch_node.binding, BindingRef) else None
        return (name, True) if name else None
    return None


def _is_sum_accum(stmt: Assignment, inner_var: str, outer_var: str,
                  local_defs: dict[str, ASTNode] | None = None
                  ) -> tuple[str, str | None, str, bool] | None:
    """Check if stmt is `acc = acc + fetch(...)` or with .channels.

    Returns (accum_var_name, channels_or_none, binding_name, is_fetch) or None.
    """
    if not isinstance(stmt, Assignment):
        return None
    target = stmt.target
    if not isinstance(target, Identifier):
        return None
    value = stmt.value
    if not isinstance(value, BinOp) or value.op != "+":
        return None
    if _is_ident(value.left, target.name):
        rhs = value.right
    elif _is_ident(value.right, target.name):
        rhs = value.left
    else:
        return None

    # Unwrap channel access
    channels = None
    fetch_expr = rhs
    if isinstance(rhs, ChannelAccess):
        channels = rhs.channels
        fetch_expr = rhs.object

    fetch_node = _find_fetch_call(fetch_expr)
    if fetch_node is None:
        return None

    coord_info = _check_fetch_coords(fetch_node, inner_var, outer_var, local_defs)
    if coord_info is None:
        return None
    binding_name, is_fetch = coord_info

    return (target.name, channels, binding_name, is_fetch)


def _is_count_increment(stmt: ASTNode) -> str | None:
    """Check if stmt is `count = count + 1`. Return var name."""
    if not isinstance(stmt, Assignment):
        return None
    target = stmt.target
    if not isinstance(target, Identifier):
        return None
    value = stmt.value
    if not isinstance(value, BinOp) or value.op != "+":
        return None
    if not _is_ident(value.left, target.name):
        return None
    if isinstance(value.right, NumberLiteral) and value.right.value == 1.0:
        return target.name
    return None


def _is_array_collect_assign(stmt: Assignment, inner_var: str, outer_var: str,
                             local_defs: dict[str, ASTNode]
                             ) -> tuple[str, str | None, str, bool] | None:
    """Check if stmt is `arr[idx] = fetch(...)` or `arr[idx] = s.channel`.

    Returns (array_name, channel_or_none, binding_name, is_fetch) or None.
    """
    if not isinstance(stmt.target, ArrayIndexAccess):
        return None
    arr_node = stmt.target.array
    if not isinstance(arr_node, Identifier):
        return None

    value = stmt.value
    channel = None

    if isinstance(value, ChannelAccess):
        channel = value.channels
        fetch_expr = _resolve_coord(value.object, local_defs)
    elif isinstance(value, Identifier):
        fetch_expr = _resolve_coord(value, local_defs)
    else:
        fetch_expr = value

    fc = _find_fetch_call(fetch_expr)
    if fc is None:
        return None

    coord_info = _check_fetch_coords(fc, inner_var, outer_var, local_defs)
    if coord_info is None:
        return None

    binding_name, is_fetch = coord_info
    return (arr_node.name, channel, binding_name, is_fetch)


def _is_minmax_accum(stmt: Assignment, inner_var: str, outer_var: str,
                     local_defs: dict[str, ASTNode] | None = None
                     ) -> tuple[str, str, str | None, str, bool] | None:
    """Check if stmt is `result = max(result, fetch(...))` or min variant.

    Also handles indirect patterns like:
        vec3 s = fetch(@image, ix + dx, iy + dy);
        result = max(result, s);

    Returns (result_var, "max"|"min", channels, binding_name, is_fetch) or None.
    """
    if not isinstance(stmt, Assignment):
        return None
    target = stmt.target
    if not isinstance(target, Identifier):
        return None
    value = stmt.value
    if not isinstance(value, FunctionCall) or value.name not in ("max", "min"):
        return None
    if len(value.args) != 2:
        return None

    if local_defs is None:
        local_defs = {}

    op = value.name  # "max" or "min"

    # One arg must be the result var, the other must contain a fetch/sample
    for i in range(2):
        other = value.args[1 - i]
        if _is_ident(value.args[i], target.name):
            # Unwrap channel access
            channels = None
            fetch_expr = other
            if isinstance(other, ChannelAccess):
                channels = other.channels
                fetch_expr = other.object

            # Resolve through local variable definitions (e.g. s = fetch(...))
            resolved = _resolve_coord(fetch_expr, local_defs)

            # Unwrap channel access on the resolved expression too
            if isinstance(resolved, ChannelAccess):
                if channels is None:
                    channels = resolved.channels
                resolved = resolved.object

            fetch_node = _find_fetch_call(resolved)
            if fetch_node is None:
                continue
            coord_info = _check_fetch_coords(fetch_node, inner_var, outer_var, local_defs)
            if coord_info is None:
                continue
            binding_name, is_fetch = coord_info
            return (target.name, op, channels, binding_name, is_fetch)
    return None


def _collect_local_defs(stmts: list[ASTNode], local_defs: dict[str, ASTNode]) -> None:
    """Collect variable definitions from a statement list into local_defs."""
    for stmt in stmts:
        if isinstance(stmt, VarDecl) and stmt.initializer is not None:
            local_defs[stmt.name] = stmt.initializer
        elif isinstance(stmt, Assignment) and isinstance(stmt.target, Identifier):
            if not (isinstance(stmt.value, BinOp) and stmt.value.op == "+"
                    and _is_ident(stmt.value.left, stmt.target.name)):
                local_defs[stmt.target.name] = stmt.value


def _try_detect_stencil(outer_loop: ForLoop) -> _StencilInfo | None:
    """Detect a box-blur stencil in nested for-loops.

    Handles both static ranges (for dy = -2; dy <= 2) and parameterized
    ranges (for dy = -$radius; dy <= $radius).
    """
    # Try symmetric range first (handles both static and parameterized)
    outer_sym = _try_extract_symmetric_range(outer_loop)
    if outer_sym is None:
        # Also try static non-symmetric via try_extract_static_range
        static = try_extract_static_range(outer_loop)
        if static is None:
            return None
        outer_var, dy_start, dy_stop, step = static
        if step != 1:
            return None
        y_radius = None  # non-symmetric, use dy_start/dy_stop
    else:
        outer_var, y_radius = outer_sym
        dy_start = dy_stop = None

    # Find inner ForLoop in outer body
    inner_loop = None
    outer_count_var = None
    for stmt in outer_loop.body:
        if isinstance(stmt, ForLoop):
            if inner_loop is not None:
                return None
            inner_loop = stmt
        elif isinstance(stmt, Assignment):
            cv = _is_count_increment(stmt)
            if cv is not None:
                outer_count_var = cv
        elif isinstance(stmt, (VarDecl, ExprStatement)):
            pass
        else:
            return None

    if inner_loop is None:
        return None

    # Check inner loop range
    inner_sym = _try_extract_symmetric_range(inner_loop)
    if inner_sym is None:
        static = try_extract_static_range(inner_loop)
        if static is None:
            return None
        inner_var, dx_start, dx_stop, step = static
        if step != 1:
            return None
        x_radius = None
    else:
        inner_var, x_radius = inner_sym
        dx_start = dx_stop = None

    # Determine is_symmetric: both loops use symmetric range
    is_symmetric = y_radius is not None and x_radius is not None

    # Collect local variable definitions for sample() tracing.
    # Include outer loop body (LICM-hoisted vars like _licm0) and inner loop body.
    local_defs: dict[str, ASTNode] = {}
    _collect_local_defs(outer_loop.body, local_defs)
    _collect_local_defs(inner_loop.body, local_defs)

    # Inner body: classify the stencil pattern.
    # Try box blur (sum accumulation), min/max, median (array collect).
    accum_info = None
    minmax_info = None
    array_collects: list[tuple[str, str | None, str, bool]] = []
    inner_count_var = None
    has_unknown = False

    for stmt in inner_loop.body:
        if isinstance(stmt, Assignment):
            info = _is_sum_accum(stmt, inner_var, outer_var, local_defs)
            if info is not None:
                if accum_info is not None:
                    has_unknown = True
                else:
                    accum_info = info
                continue

            mm = _is_minmax_accum(stmt, inner_var, outer_var, local_defs)
            if mm is not None:
                if minmax_info is not None:
                    has_unknown = True
                else:
                    minmax_info = mm
                continue

            ac = _is_array_collect_assign(stmt, inner_var, outer_var, local_defs)
            if ac is not None:
                array_collects.append(ac)
                continue

            cv = _is_count_increment(stmt)
            if cv is not None:
                inner_count_var = cv
                continue
            # Allow intermediate variable definitions (su, sv for sample pattern)
            if isinstance(stmt.target, Identifier) and stmt.target.name in local_defs:
                continue
        elif isinstance(stmt, (VarDecl, ExprStatement)):
            continue
        # Unknown statement type or unrecognized assignment — not necessarily fatal
        has_unknown = True

    # Return the detected stencil kind (prefer box, then minmax, then median)
    if accum_info is not None and not has_unknown:
        accum_var, channels, binding_name, is_fetch = accum_info
        return _StencilInfo(
            kind="box",
            binding_name=binding_name,
            is_fetch=is_fetch,
            channels=channels,
            y_radius=y_radius,
            x_radius=x_radius,
            is_symmetric=is_symmetric,
            dy_start=dy_start,
            dy_stop=dy_stop,
            dx_start=dx_start,
            dx_stop=dx_stop,
            accum_var=accum_var,
            count_var=inner_count_var or outer_count_var,
        )

    if minmax_info is not None and not has_unknown:
        result_var, op, channels, binding_name, is_fetch = minmax_info
        return _StencilInfo(
            kind="minmax",
            binding_name=binding_name,
            is_fetch=is_fetch,
            channels=channels,
            y_radius=y_radius,
            x_radius=x_radius,
            is_symmetric=is_symmetric,
            dy_start=dy_start,
            dy_stop=dy_stop,
            dx_start=dx_start,
            dx_stop=dx_stop,
            minmax_op=op,
            result_var=result_var,
        )

    # Try median: array collect pattern (arr[idx] = fetch(...))
    if array_collects and not accum_info and not minmax_info:
        # Validate: all from same binding
        bindings = set(ac[2] for ac in array_collects)
        if len(bindings) == 1:
            binding_name = array_collects[0][2]
            is_fetch = array_collects[0][3]
            array_names = list(dict.fromkeys(ac[0] for ac in array_collects))
            # Build channel mapping: [(array_name, bchw_channel_index)]
            # For per-channel arrays (r[idx]=s.r): map channel name to index
            # For direct vec arrays (samples[idx]=fetch(...)): channel is None
            array_chan_pairs: list[tuple[str, int | None]] = []
            valid = True
            for arr_name in array_names:
                # Find first collect entry for this array
                for ac in array_collects:
                    if ac[0] == arr_name:
                        ch = ac[1]
                        if ch is not None:
                            ch_idx = CHANNEL_MAP.get(ch)
                            if ch_idx is None:
                                valid = False
                            else:
                                array_chan_pairs.append((arr_name, ch_idx))
                        else:
                            array_chan_pairs.append((arr_name, None))
                        break
            if valid:
                return _StencilInfo(
                    kind="median",
                    binding_name=binding_name,
                    is_fetch=is_fetch,
                    channels=None,
                    y_radius=y_radius,
                    x_radius=x_radius,
                    is_symmetric=is_symmetric,
                    dy_start=dy_start,
                    dy_stop=dy_stop,
                    dx_start=dx_start,
                    dx_stop=dx_stop,
                    array_vars=array_chan_pairs,
                    count_var=inner_count_var or outer_count_var,
                )

    return None


# ── Inline stencil detection (non-loop conv2d patterns) ───────────────


def _extract_fetch_offset(node: ASTNode) -> tuple[str, int, int, str | None] | None:
    """Extract (binding_name, dx, dy, channels) from a fetch/sample at constant offset.

    Matches patterns:
      fetch(@img, ix + CONST, iy + CONST)       → (img, CONST, CONST, None)
      fetch(@img, ix - CONST, iy)                → (img, -CONST, 0, None)
      fetch(@img, ix, iy).rgb                    → (img, 0, 0, "rgb")
      sample(@img, u + CONST*px, v - CONST*py)   → (img, CONST, -CONST, None)
      sample(@img, u, v)                          → (img, 0, 0, None)
      @img[ix + CONST, iy + CONST]               → (img, CONST, CONST, None)

    Returns None if the pattern doesn't match.
    """
    channels = None
    expr = node
    if isinstance(expr, ChannelAccess):
        channels = expr.channels
        expr = expr.object

    if isinstance(expr, BindingIndexAccess):
        # @img[x_expr, y_expr]
        if not isinstance(expr.binding, BindingRef) or len(expr.args) != 2:
            return None
        binding = expr.binding.name
        dx = _extract_pixel_offset(expr.args[0], "ix")
        dy = _extract_pixel_offset(expr.args[1], "iy")
        if dx is None or dy is None:
            return None
        return (binding, dx, dy, channels)

    if isinstance(expr, FunctionCall):
        if expr.name == "fetch" and len(expr.args) == 3:
            if not isinstance(expr.args[0], BindingRef):
                return None
            binding = expr.args[0].name
            dx = _extract_pixel_offset(expr.args[1], "ix")
            dy = _extract_pixel_offset(expr.args[2], "iy")
            if dx is None or dy is None:
                return None
            return (binding, dx, dy, channels)

        if expr.name == "sample" and len(expr.args) == 3:
            if not isinstance(expr.args[0], BindingRef):
                return None
            binding = expr.args[0].name
            dx = _extract_uv_offset(expr.args[1], "u")
            dy = _extract_uv_offset(expr.args[2], "v")
            if dx is None or dy is None:
                return None
            return (binding, dx, dy, channels)

    if isinstance(expr, BindingSampleAccess):
        if not isinstance(expr.binding, BindingRef) or len(expr.args) != 2:
            return None
        binding = expr.binding.name
        dx = _extract_uv_offset(expr.args[0], "u")
        dy = _extract_uv_offset(expr.args[1], "v")
        if dx is None or dy is None:
            return None
        return (binding, dx, dy, channels)

    return None


def _extract_pixel_offset(expr: ASTNode, base: str) -> int | None:
    """Extract integer pixel offset from `base + CONST` or `base - CONST` or `base`."""
    if _is_ident(expr, base):
        return 0
    if not isinstance(expr, BinOp):
        return None
    if expr.op == "+" and _is_ident(expr.left, base) and isinstance(expr.right, NumberLiteral):
        return int(expr.right.value)
    if expr.op == "+" and isinstance(expr.left, NumberLiteral) and _is_ident(expr.right, base):
        return int(expr.left.value)
    if expr.op == "-" and _is_ident(expr.left, base) and isinstance(expr.right, NumberLiteral):
        return -int(expr.right.value)
    return None


def _extract_uv_offset(expr: ASTNode, base: str) -> int | None:
    """Extract integer pixel offset from `base + CONST*px` or `base - px` etc."""
    if _is_ident(expr, base):
        return 0
    if not isinstance(expr, BinOp):
        return None
    px_var = "px" if base == "u" else "py"

    if expr.op in ("+", "-"):
        if not _is_ident(expr.left, base):
            return None
        rhs = expr.right
        sign = 1 if expr.op == "+" else -1

        # u + px  or  u - px  (offset = ±1)
        if _is_ident(rhs, px_var):
            return sign

        if isinstance(rhs, BinOp):
            # u + CONST * px  or  u + px * CONST
            if rhs.op == "*":
                if isinstance(rhs.left, NumberLiteral) and _is_ident(rhs.right, px_var):
                    return sign * int(rhs.left.value)
                if _is_ident(rhs.left, px_var) and isinstance(rhs.right, NumberLiteral):
                    return sign * int(rhs.right.value)
                # float(CONST) * px — CastExpr wrapping
                if isinstance(rhs.left, CastExpr) and _is_ident(rhs.right, px_var):
                    if isinstance(rhs.left.expr, NumberLiteral):
                        return sign * int(rhs.left.expr.value)
                if _is_ident(rhs.left, px_var) and isinstance(rhs.right, CastExpr):
                    if isinstance(rhs.right.expr, NumberLiteral):
                        return sign * int(rhs.right.expr.value)
            # u + float(CONST) / iw
            dim_var = "iw" if base == "u" else "ih"
            if rhs.op == "/":
                if isinstance(rhs.left, NumberLiteral) and _is_ident(rhs.right, dim_var):
                    return sign * int(rhs.left.value)
                if isinstance(rhs.left, CastExpr) and isinstance(rhs.left.expr, NumberLiteral):
                    if _is_ident(rhs.right, dim_var):
                        return sign * int(rhs.left.expr.value)
    return None


def _resolve_through_locals(expr: ASTNode, local_defs: dict[str, ASTNode]) -> ASTNode:
    """Resolve an expression by substituting known local variable definitions.

    For `u + off_u` where off_u is defined as `float(px2) * px`,
    returns `u + float(px2) * px`.
    """
    if isinstance(expr, Identifier) and expr.name in local_defs:
        return local_defs[expr.name]
    if isinstance(expr, BinOp):
        left = _resolve_through_locals(expr.left, local_defs)
        right = _resolve_through_locals(expr.right, local_defs)
        if left is expr.left and right is expr.right:
            return expr
        return BinOp(op=expr.op, left=left, right=right, loc=expr.loc)
    return expr


def _extract_uv_offset_expr(expr: ASTNode, base: str) -> ASTNode | bool | None:
    """Extract the pixel-offset expression from `base + expr * px` or `base + expr / iw`.

    Returns:
      True if expr is just `base` (zero offset)
      ASTNode for the offset expression (to be emitted as code)
      None if pattern doesn't match
    """
    if _is_ident(expr, base):
        return True  # zero offset
    if not isinstance(expr, BinOp) or expr.op not in ("+", "-"):
        return None
    if not _is_ident(expr.left, base):
        return None
    rhs = expr.right
    px_var = "px" if base == "u" else "py"
    dim_var = "iw" if base == "u" else "ih"

    # u + expr * px  or  u + px * expr
    if isinstance(rhs, BinOp) and rhs.op == "*":
        if _is_ident(rhs.right, px_var):
            offset = rhs.left
        elif _is_ident(rhs.left, px_var):
            offset = rhs.right
        else:
            return None
        # Handle float(expr) cast
        if isinstance(offset, CastExpr) and offset.target_type == "float":
            offset = offset.expr
        if expr.op == "-":
            return UnaryOp(op="-", operand=offset, loc=expr.loc)
        return offset

    # u + expr / iw
    if isinstance(rhs, BinOp) and rhs.op == "/":
        if _is_ident(rhs.right, dim_var):
            offset = rhs.left
            if isinstance(offset, CastExpr) and offset.target_type == "float":
                offset = offset.expr
            if expr.op == "-":
                return UnaryOp(op="-", operand=offset, loc=expr.loc)
            return offset

    # u + px (offset = 1) or u - px (offset = -1)
    if _is_ident(rhs, px_var):
        if expr.op == "-":
            return NumberLiteral(value=-1.0, loc=expr.loc)
        return NumberLiteral(value=1.0, loc=expr.loc)

    return None


def _extract_linear_weights(expr: ASTNode, tap_vars: dict[str, tuple[int, int]]
                            ) -> dict[tuple[int, int], float] | None:
    """Extract linear combination weights from an expression.

    tap_vars maps variable names to (dx, dy) offsets.
    Returns {(dy, dx): weight} or None if not a linear combination.
    """
    weights: dict[tuple[int, int], float] = {}

    def _extract(node: ASTNode, scale: float) -> bool:
        """Recursively extract weights. Returns True on success."""
        if isinstance(node, Identifier) and node.name in tap_vars:
            dy, dx = tap_vars[node.name]
            weights[(dy, dx)] = weights.get((dy, dx), 0.0) + scale
            return True

        if isinstance(node, BinOp):
            if node.op == "+":
                return _extract(node.left, scale) and _extract(node.right, scale)
            if node.op == "-":
                return _extract(node.left, scale) and _extract(node.right, -scale)
            if node.op == "*":
                # scalar * tap_expr or tap_expr * scalar
                if isinstance(node.left, NumberLiteral):
                    return _extract(node.right, scale * node.left.value)
                if isinstance(node.right, NumberLiteral):
                    return _extract(node.left, scale * node.right.value)
                return False
            return False

        if isinstance(node, UnaryOp) and node.op == "-":
            return _extract(node.operand, -scale)

        # Parenthesized expression or other — not a simple linear combination
        return False

    if _extract(expr, 1.0):
        return weights
    return None


def _collect_ident_refs(node: ASTNode, refs: set[str]) -> None:
    """Recursively collect all Identifier names referenced in an AST subtree."""
    if isinstance(node, Identifier):
        refs.add(node.name)
    elif isinstance(node, (BinOp,)):
        _collect_ident_refs(node.left, refs)
        _collect_ident_refs(node.right, refs)
    elif isinstance(node, UnaryOp):
        _collect_ident_refs(node.operand, refs)
    elif isinstance(node, TernaryOp):
        _collect_ident_refs(node.condition, refs)
        _collect_ident_refs(node.true_expr, refs)
        _collect_ident_refs(node.false_expr, refs)
    elif isinstance(node, FunctionCall):
        for arg in node.args:
            _collect_ident_refs(arg, refs)
    elif isinstance(node, VecConstructor):
        for arg in node.args:
            _collect_ident_refs(arg, refs)
    elif isinstance(node, ChannelAccess):
        _collect_ident_refs(node.object, refs)
    elif isinstance(node, CastExpr):
        _collect_ident_refs(node.expr, refs)
    elif isinstance(node, VarDecl):
        if node.initializer:
            _collect_ident_refs(node.initializer, refs)
    elif isinstance(node, Assignment):
        if isinstance(node.target, Identifier):
            pass  # target is a write, not a read
        else:
            _collect_ident_refs(node.target, refs)
        _collect_ident_refs(node.value, refs)
    elif isinstance(node, IfElse):
        _collect_ident_refs(node.condition, refs)
        for s in node.then_body:
            _collect_ident_refs(s, refs)
        for s in (node.else_body or []):
            _collect_ident_refs(s, refs)
    elif isinstance(node, ForLoop):
        _collect_ident_refs(node.init, refs)
        _collect_ident_refs(node.condition, refs)
        _collect_ident_refs(node.update, refs)
        for s in node.body:
            _collect_ident_refs(s, refs)
    elif isinstance(node, WhileLoop):
        _collect_ident_refs(node.condition, refs)
        for s in node.body:
            _collect_ident_refs(s, refs)
    elif isinstance(node, ExprStatement):
        _collect_ident_refs(node.expr, refs)
    elif isinstance(node, ArrayIndexAccess):
        _collect_ident_refs(node.array, refs)
        _collect_ident_refs(node.index, refs)


def _try_detect_inline_stencil(stmts: list[ASTNode], start: int
                               ) -> _StencilInfo | None:
    """Detect an inline stencil pattern starting at position `start` in stmts.

    Looks for a cluster of VarDecl/Assignment statements that fetch/sample from
    the same binding at constant offsets, followed by a VarDecl/Assignment that
    combines them linearly. Emits a conv2d stencil.

    Returns a _StencilInfo with kind="conv2d" or None.
    """
    if start >= len(stmts):
        return None

    # Phase 1: Collect consecutive fetch/sample taps at constant offsets
    # Each tap is: (var_name, binding, dx, dy, channels)
    taps: list[tuple[str, str, int, int, str | None]] = []
    tap_indices: list[int] = []
    binding_name: str | None = None
    all_channels: str | None = None
    is_fetch = True

    i = start
    while i < len(stmts):
        stmt = stmts[i]
        var_name = None
        init_expr = None

        if isinstance(stmt, VarDecl) and stmt.initializer is not None:
            var_name = stmt.name
            init_expr = stmt.initializer
        elif isinstance(stmt, Assignment) and isinstance(stmt.target, Identifier):
            # Also handle `gx += luma(fetch(...)) * weight` patterns
            # For now, only simple VarDecl fetch assignments
            break
        else:
            break

        # Check if initializer is a fetch/sample at constant offset
        info = _extract_fetch_offset(init_expr)
        if info is None:
            # Could be a binding ref like `@image` (center pixel at 0,0)
            if isinstance(init_expr, BindingRef):
                info = (init_expr.name, 0, 0, None)
            elif isinstance(init_expr, ChannelAccess) and isinstance(init_expr.object, BindingRef):
                info = (init_expr.object.name, 0, 0, init_expr.channels)
            else:
                break

        b_name, dx, dy, ch = info
        if binding_name is None:
            binding_name = b_name
            all_channels = ch
            # Determine if fetch or sample based on first tap.
            # Unwrap ChannelAccess to find the underlying fetch/sample call.
            inner = init_expr
            if isinstance(inner, ChannelAccess):
                inner = inner.object
            if isinstance(inner, FunctionCall):
                is_fetch = inner.name != "sample"
            elif isinstance(inner, BindingSampleAccess):
                is_fetch = False
            else:
                is_fetch = True  # BindingRef, BindingIndexAccess → fetch-like
        elif b_name != binding_name:
            break  # Different binding — stop

        taps.append((var_name, b_name, dx, dy, ch))
        tap_indices.append(i)
        i += 1

    if len(taps) < 3:
        return None  # Need at least 3 taps for a meaningful stencil

    # Phase 2: Look for a linear combination of the tap variables
    # in the next statement(s)
    tap_var_map = {name: (dy, dx) for name, _, dx, dy, _ in taps}
    combo_idx = None
    combo_var = None
    kernel_weights = None

    for j in range(i, min(i + 3, len(stmts))):
        stmt = stmts[j]
        expr = None
        vname = None

        if isinstance(stmt, VarDecl) and stmt.initializer is not None:
            vname = stmt.name
            expr = stmt.initializer
        elif isinstance(stmt, Assignment) and isinstance(stmt.target, Identifier):
            vname = stmt.target.name
            expr = stmt.value

        if expr is None:
            continue

        weights = _extract_linear_weights(expr, tap_var_map)
        if weights is not None and len(weights) >= 2:
            kernel_weights = weights
            combo_var = vname
            combo_idx = j
            break

    if kernel_weights is None or len(kernel_weights) > 49:  # max 7x7 kernel
        return None

    all_offsets = list(kernel_weights.keys())
    dy_min = min(o[0] for o in all_offsets)
    dy_max = max(o[0] for o in all_offsets)
    dx_min = min(o[1] for o in all_offsets)
    dx_max = max(o[1] for o in all_offsets)

    consumed = set(tap_indices)
    consumed.add(combo_idx)

    # Don't consume tap vars that are referenced in later statements
    # (e.g., sharpen.tex uses `center` both in the conv kernel and the output).
    later_refs: set[str] = set()
    for k in range(combo_idx + 1, len(stmts)):
        _collect_ident_refs(stmts[k], later_refs)
    for idx in list(tap_indices):
        stmt = stmts[idx]
        vname = stmt.name if isinstance(stmt, VarDecl) else None
        if vname and vname in later_refs:
            consumed.discard(idx)

    return _StencilInfo(
        kind="conv2d",
        binding_name=binding_name,
        is_fetch=is_fetch,
        channels=all_channels,
        dy_start=dy_min,
        dy_stop=dy_max + 1,
        dx_start=dx_min,
        dx_stop=dx_max + 1,
        kernel_weights=kernel_weights,
        result_var=combo_var,
        consumed_stmts=consumed,
    )


class _CodeGen:
    """Generates Python source code from a TEX AST."""

    _codegen_counter: int = 0  # class-level counter for unique filenames
    _LINECACHE_KEEP: int = 64  # max linecache entries to prevent unbounded growth

    def __init__(self, type_map: dict[int, TEXType]):
        self.type_map = type_map
        self._tmp_counter = 0
        self._lines: list[str] = []
        self._preamble: list[str] = []  # hoisted constant assignments
        self._indent = 1  # Start at 1 (inside function body)
        self._const_cache: dict[float, str] = {}  # value → variable name
        self._vec_const_cache: dict[tuple, str] = {}  # (v1, v2, ...) → variable name
        self._range_cache: dict[tuple, str] = {}  # (start, stop, step) → variable name
        # Local variable mapping: TEX var name → Python local var name.
        # When set, Identifier emission uses locals instead of _env[name].
        self._local_vars: dict[str, str] = {}
        # User-defined function names (populated during FunctionDef emission)
        self._user_functions: set[str] = set()
        # Track if we're inside a user function body (for return statements)
        self._in_user_function: bool = False
        # Variables eligible for in-place mutation in the current loop.
        # Maps TEX var name → operator (e.g., "acc" → "+").
        # Set by _emit_for_loop, consumed by _emit_assignment.
        # Declared vector types per variable name (for channel coercion)
        self._var_vec_type: dict[str, TEXType] = {}
        # Hoisted BCHW images for inline sample(): binding_name → (bchw_var, grid_var).
        # Cross-loop reuse is safe: bindings are immutable within a TEX program
        # (read from _bind dict, never reassigned), so the BCHW permute stays valid.
        self._hoisted_bchw: dict[str, tuple[str, str]] = {}
        # VarDecl initializer AST nodes for sample→fetch pattern resolution
        self._var_initializers: dict[str, ASTNode] = {}
        # Pre-resolved stdlib function locals: name → preamble variable.
        # Avoids _fns['name'] dict lookup on every call.
        self._fn_locals: dict[str, str] = {}
        # When True, BreakStmt/ContinueStmt emit native Python break/continue
        # instead of raising _CgBreak/_CgContinue. Set by static range for-loops.
        self._use_native_flow_control: bool = False
        # When True, emit Python float math instead of tensor ops.
        # Set by _emit_for_loop when the entire loop body is scalar-typed.
        self._scalar_loop: bool = False
        # Dispatch table: function name → handler method for tensor-path
        # specializations.  Each handler has signature
        #   (node: FunctionCall, args: list[str], tmp: str) -> str | None
        # and returns the result variable name, or None to fall through.
        self._fn_dispatch: dict[str, object] = {
            # Complex specializations (own methods)
            "pow": self._emit_fn_pow,
            "max": self._emit_fn_minmax, "min": self._emit_fn_minmax,
            "lerp": self._emit_fn_lerp,
            "luma": self._emit_fn_luma,
            # Math 1-arg
            "sqrt": self._emit_fn_math_1arg, "log": self._emit_fn_math_1arg,
            "log2": self._emit_fn_math_1arg, "log10": self._emit_fn_math_1arg,
            "fract": self._emit_fn_math_1arg, "isnan": self._emit_fn_math_1arg,
            "isinf": self._emit_fn_math_1arg, "pow2": self._emit_fn_math_1arg,
            "pow10": self._emit_fn_math_1arg, "sincos": self._emit_fn_math_1arg,
            # Vector geometry
            "dot": self._emit_fn_vector, "distance": self._emit_fn_vector,
            "normalize": self._emit_fn_vector, "length": self._emit_fn_vector,
            "cross": self._emit_fn_vector, "reflect": self._emit_fn_vector,
            # Shaping functions
            "smoothstep": self._emit_fn_shaping, "step": self._emit_fn_shaping,
            "clamp": self._emit_fn_shaping, "fit": self._emit_fn_shaping,
            # Safe arithmetic
            "spow": self._emit_fn_safe, "sdiv": self._emit_fn_safe,
            "smin": self._emit_fn_safe, "smax": self._emit_fn_safe,
            "mod": self._emit_fn_safe,
            # SDF primitives
            "sdf_circle": self._emit_fn_sdf, "sdf_box": self._emit_fn_sdf,
            "sdf_line": self._emit_fn_sdf,
            # Image reductions
            "img_sum": self._emit_fn_reduce, "img_mean": self._emit_fn_reduce,
            "img_min": self._emit_fn_reduce, "img_max": self._emit_fn_reduce,
            # Matrix operations
            "transpose": self._emit_fn_matrix, "determinant": self._emit_fn_matrix,
            "inverse": self._emit_fn_matrix,
            # Sampling — inline when binding is hoisted, else fallthrough to _fns[]
            "sample": self._emit_fn_sample, "fetch": self._emit_fn_fetch,
        }

    def _tmp(self) -> str:
        """Generate a unique temporary variable name."""
        self._tmp_counter += 1
        return f"_t{self._tmp_counter}"

    def _get_fn_local(self, name: str) -> str:
        """Get a preamble-hoisted local for a stdlib function (avoids dict lookup)."""
        local = self._fn_locals.get(name)
        if local is not None:
            return local
        local = f"_fn_{name}"
        self._preamble.append(f"    {local} = _fns[{name!r}]")
        self._fn_locals[name] = local
        return local

    def _emit_direct_fetch(self, binding_name: str,
                           dx_expr: ASTNode | bool, dy_expr: ASTNode | bool) -> str:
        """Emit direct tensor indexing for sample-with-integer-offset patterns.

        Shared by BindingSampleAccess and FunctionCall("sample") paths.
        """
        img_var = f"_bind[{binding_name!r}]"
        dx_code = self._emit_expr(dx_expr) if dx_expr is not True else "0"
        dy_code = self._emit_expr(dy_expr) if dy_expr is not True else "0"
        px_tmp = self._tmp()
        py_tmp = self._tmp()
        if dx_code == "0":
            self._emit(f"{px_tmp} = _lv_ix.clamp(0, {img_var}.shape[2] - 1).long()")
        else:
            self._emit(f"{px_tmp} = (_lv_ix + {dx_code}).clamp(0, {img_var}.shape[2] - 1).long()")
        if dy_code == "0":
            self._emit(f"{py_tmp} = _lv_iy.clamp(0, {img_var}.shape[1] - 1).long()")
        else:
            self._emit(f"{py_tmp} = (_lv_iy + {dy_code}).clamp(0, {img_var}.shape[1] - 1).long()")
        tmp = self._tmp()
        self._emit(f"{tmp} = {img_var}[:, {py_tmp}, {px_tmp}, :]"
                   f" if {px_tmp}.dim() < 3"
                   f" else {img_var}[_torch.arange({img_var}.shape[0]).view(-1,1,1), {py_tmp}, {px_tmp}, :]")
        return tmp

    def _emit_inline_grid_sample(self, binding_name: str,
                                  u_expr: str, v_expr: str) -> str:
        """Emit inline grid_sample using hoisted BCHW + pre-allocated grid buffer."""
        bchw_var, grid_var = self._hoisted_bchw[binding_name]
        gx = self._tmp()
        gy = self._tmp()
        self._emit(f"{gx} = {u_expr} * 2.0 - 1.0")
        self._emit(f"{gy} = {v_expr} * 2.0 - 1.0")
        self._emit(f"if {gx}.dim() < 3: {gx} = {gx}.expand({bchw_var}.shape[0], {bchw_var}.shape[2], {bchw_var}.shape[3])")
        self._emit(f"if {gy}.dim() < 3: {gy} = {gy}.expand({bchw_var}.shape[0], {bchw_var}.shape[2], {bchw_var}.shape[3])")
        self._emit(f"{grid_var}[..., 0] = {gx}")
        self._emit(f"{grid_var}[..., 1] = {gy}")
        result_bchw = self._tmp()
        self._emit(f"{result_bchw} = _torch.nn.functional.grid_sample("
                   f"{bchw_var}, {grid_var}, mode='bilinear', "
                   f"padding_mode='border', align_corners=True)")
        tmp = self._tmp()
        self._emit(f"{tmp} = {result_bchw}.permute(0, 2, 3, 1)")
        return tmp

    def _hoist_sample_setup(self, stmts: list[ASTNode]):
        """Emit BCHW conversion + grid buffer for bindings sampled in a loop.

        Hoisting outside the loop avoids per-call overhead from the stdlib
        fn_sample() wrapper (type checks, shape extraction, permute) and
        saves a torch.stack allocation per sample call (~0.1ms each).
        """
        bindings = _collect_sample_bindings(stmts)
        for bname in sorted(bindings):
            if bname in self._hoisted_bchw:
                continue  # Already hoisted by an outer loop
            bchw_var = self._tmp()
            grid_var = self._tmp()
            self._emit(f"{bchw_var} = _bind[{bname!r}].permute(0, 3, 1, 2)")
            self._emit(f"{grid_var} = _torch.empty("
                       f"{bchw_var}.shape[0], {bchw_var}.shape[2], {bchw_var}.shape[3], 2, "
                       f"dtype=_torch.float32, device=_dev)")
            self._hoisted_bchw[bname] = (bchw_var, grid_var)

    def _emit(self, line: str):
        """Emit a line of code at the current indentation level."""
        self._lines.append("    " * self._indent + line)

    def _emit_spatial_branch(self, spatial_code: str, non_spatial_code: str):
        """Emit an if _sp: / else: block with the given code for each branch."""
        self._emit("if _sp:")
        self._indent += 1
        self._emit(spatial_code)
        self._indent -= 1
        self._emit("else:")
        self._indent += 1
        self._emit(non_spatial_code)
        self._indent -= 1

    def _get_const(self, value: float) -> str:
        """Return a variable name for a numeric constant tensor.

        Constants are hoisted to the function preamble (before any loops),
        so each unique value is created once per execution regardless of
        how many times or where it appears in the source.

        In scalar loop mode, returns a bare Python float literal instead.
        """
        if self._scalar_loop:
            return repr(float(value))
        cached = self._const_cache.get(value)
        if cached is not None:
            return cached
        var = self._tmp()
        self._preamble.append(f"    {var} = _torch.scalar_tensor({value!r}, dtype=_torch.float32, device=_dev)")
        self._const_cache[value] = var
        return var

    def emit_program(self, program: Program):
        """Emit code for the entire program.

        All env variables are pre-registered as Python locals to avoid
        _env dict lookups on every read/write. This produces cleaner code
        for TorchInductor (no dict guard overhead) and is faster in eager
        mode too (local variable access is faster than dict access).
        """
        # Register all program-level env vars as locals
        all_env_vars, _ = self._collect_modified_vars(program.statements)
        for vname in sorted(all_env_vars):
            self._local_vars[vname] = f"_lv_{vname}"

        # Initialize locals from _env only for builtins pre-populated by
        # the caller (u, v, ix, iy, etc.).  User-defined vars are always
        # written before read and can start as None (avoids dict lookups).
        for vname in sorted(all_env_vars):
            local = self._local_vars[vname]
            if vname in _BUILTIN_NAMES:
                self._preamble.append(f"    {local} = _env.get({vname!r})")
            else:
                self._preamble.append(f"    {local} = None")

        # Pre-scan for inline stencil patterns (non-loop conv2d)
        inline_skip: set[int] = set()
        stmts = program.statements
        idx = 0
        while idx < len(stmts):
            # Fast path: inline stencils start with VarDecl fetch taps
            if idx not in inline_skip and isinstance(stmts[idx], VarDecl):
                inline = _try_detect_inline_stencil(stmts, idx)
                # Only apply inline conv2d for fetch()-based patterns (pixel-exact).
                # sample()-based patterns have sub-pixel coordinate errors because
                # px = 1/W but the UV pixel step is 1/(W-1) (align_corners=True).
                if inline is not None and inline.consumed_stmts and inline.is_fetch:
                    self._emit_conv2d_stencil(inline)
                    inline_skip.update(inline.consumed_stmts)
                    # Also emit the tap variables that are used elsewhere
                    # (outside the consumed set) as normal statements
                    idx = max(inline.consumed_stmts) + 1
                    continue
            idx += 1

        for i, stmt in enumerate(stmts):
            if i in inline_skip:
                continue
            self._emit_stmt(stmt)

        # No need to write locals back to _env: the caller only reads
        # bindings (outputs), not the env dict.  Loop-scoped writebacks
        # (inside _emit_for) still happen so that subsequent statements
        # can see loop-modified vars via _env when they aren't locals.

    def build(self) -> Any:
        """Compile the generated source to a callable function."""
        preamble = "\n".join(self._preamble)
        body = "\n".join(self._lines)
        # Hoisted constants go before the main body so they're available
        # inside loops without per-iteration tensor creation.
        func_body = f"{preamble}\n{body}" if preamble else body
        func_src = (
            "def _tex_fn(_env, _bind, _fns, _dev, _sp, "
            "_torch, _bp, _es, _tw, _math, _SAFE_EPS, _CMAP, _MAX_ITER, "
            "_CgBreak, _CgContinue):\n"
            f"{func_body}\n"
        )
        # Use a unique pseudo-filename so TorchInductor/linecache can find the source
        _CodeGen._codegen_counter += 1
        filename = f"<tex_codegen_{_CodeGen._codegen_counter}>"
        namespace: dict[str, Any] = {}
        code_obj = compile(func_src, filename, "exec")
        # Register source with linecache so torch.compile can read it for tracing.
        # Prune old entries to prevent unbounded linecache growth in long sessions.
        import linecache
        stale = _CodeGen._codegen_counter - _CodeGen._LINECACHE_KEEP
        if stale > 0:
            linecache.cache.pop(f"<tex_codegen_{stale}>", None)
        linecache.cache[filename] = (
            len(func_src), None, func_src.splitlines(True), filename,
        )
        exec(code_obj, namespace)
        return namespace["_tex_fn"]

    # ── Statement emission ──────────────────────────────────────────────

    def _emit_stmt(self, stmt: ASTNode):
        if isinstance(stmt, VarDecl):
            self._emit_var_decl(stmt)
        elif isinstance(stmt, Assignment):
            self._emit_assignment(stmt)
        elif isinstance(stmt, ExprStatement):
            expr = self._emit_expr(stmt.expr)
            self._emit(f"{expr}")
        elif isinstance(stmt, ParamDecl):
            pass  # no-op at runtime
        elif isinstance(stmt, IfElse):
            self._emit_if_else(stmt)
        elif isinstance(stmt, ForLoop):
            self._emit_for_loop(stmt)
        elif isinstance(stmt, WhileLoop):
            self._emit_while_loop(stmt)
        elif isinstance(stmt, (BreakStmt, ContinueStmt)):
            # Static range for-loops use native Python break/continue (no update
            # to skip). Non-static for-loops and while-loops use exception-based
            # flow because continue must not skip the update statement.
            if self._use_native_flow_control:
                self._emit("break" if isinstance(stmt, BreakStmt) else "continue")
            elif isinstance(stmt, BreakStmt):
                self._emit("raise _CgBreak()")
            else:
                self._emit("raise _CgContinue()")
        elif isinstance(stmt, ArrayDecl):
            self._emit_array_decl(stmt)
        elif isinstance(stmt, FunctionDef):
            self._emit_function_def(stmt)
        elif isinstance(stmt, ReturnStmt):
            self._emit_return_stmt(stmt)
        else:
            raise _Unsupported(f"Unknown statement: {type(stmt).__name__}")

    def _var_target(self, name: str) -> str:
        """Return the Python expression for writing to a TEX variable."""
        local = self._local_vars.get(name)
        return local if local is not None else f"_env[{name!r}]"

    def _emit_vec_coerce(self, expr: str, declared_type: TEXType) -> str:
        """Wrap expr with channel truncation/padding if declared type is a vector.

        Handles the case where sample()/fetch() returns 4 channels but the
        variable is declared as vec3 (or vec2). At runtime, the last dim is
        sliced to the declared channel count.
        """
        if not declared_type.is_vector:
            return expr
        ch = declared_type.channels
        if ch >= 4:
            return expr  # vec4 — no truncation needed
        tmp = self._tmp()
        self._emit(f"{tmp} = {expr}")
        self._emit(f"if _torch.is_tensor({tmp}) and {tmp}.dim() >= 1 and {tmp}.shape[-1] > {ch}: {tmp} = {tmp}[..., :{ch}]")
        return tmp

    def _emit_var_decl(self, stmt: VarDecl):
        if stmt.initializer:
            # Track local definitions for sample→fetch pattern resolution
            self._var_initializers[stmt.name] = stmt.initializer
            expr = self._emit_expr(stmt.initializer)
            # Coerce channel count when declared type is a smaller vector
            declared = TYPE_NAME_MAP.get(stmt.type_name)
            if declared and declared.is_vector:
                self._var_vec_type[stmt.name] = declared
                if declared.channels < 4:
                    expr = self._emit_vec_coerce(expr, declared)
            self._emit(f"{self._var_target(stmt.name)} = {expr}")
        else:
            # Default initialization based on type
            tgt = self._var_target(stmt.name)
            declared = self.type_map.get(id(stmt), TEXType.FLOAT)
            if declared == TEXType.STRING:
                self._emit(f"{tgt} = ''")
            elif declared.is_vector:
                ch = declared.channels
                self._emit_spatial_branch(
                    f"{tgt} = _torch.zeros(*_sp, {ch}, dtype=_torch.float32, device=_dev)",
                    f"{tgt} = _torch.zeros({ch}, dtype=_torch.float32, device=_dev)",
                )
            elif declared.is_matrix:
                n = declared.mat_size
                self._emit(f"{tgt} = _torch.zeros({n}, {n}, dtype=_torch.float32, device=_dev)")
            elif self._scalar_loop:
                self._emit(f"{tgt} = 0.0")
            else:
                self._emit(f"{tgt} = _torch.scalar_tensor(0.0, dtype=_torch.float32, device=_dev)")

    def _emit_array_decl(self, stmt: ArrayDecl):
        """Emit array declaration matching interpreter's tensor layout."""
        tgt = self._var_target(stmt.name)
        elem_type = stmt.element_type_name
        tex_type = TYPE_NAME_MAP.get(elem_type)
        is_vec = tex_type is not None and tex_type.is_vector
        is_string = elem_type == "string"

        if is_string:
            if stmt.initializer and isinstance(stmt.initializer, ArrayLiteral):
                elems = [self._emit_expr(e) for e in stmt.initializer.elements]
                self._emit(f"{tgt} = [{', '.join(elems)}]")
            elif stmt.initializer:
                src = self._emit_expr(stmt.initializer)
                self._emit(f"{tgt} = list({src}) if isinstance({src}, list) else [str({src})]")
            else:
                self._emit(f"{tgt} = [''] * {stmt.size}")
            return

        if stmt.initializer and isinstance(stmt.initializer, ArrayLiteral):
            elems = [self._emit_expr(e) for e in stmt.initializer.elements]
            es_list = ", ".join(f"_es({e}, _sp)" for e in elems)
            ns_list = ", ".join(elems)
            suffix = ".transpose(-2, -1)" if is_vec else ""
            self._emit_spatial_branch(
                f"{tgt} = _torch.stack([{es_list}], dim=-1){suffix}",
                f"{tgt} = _torch.stack([{ns_list}], dim=-1){suffix}",
            )
        elif stmt.initializer:
            src = self._emit_expr(stmt.initializer)
            self._emit(f"{tgt} = {src}.clone()")
        else:
            size = stmt.size
            if is_vec:
                channels = TYPE_NAME_MAP[elem_type].channels
                self._emit_spatial_branch(
                    f"{tgt} = _torch.zeros(*_sp, {size}, {channels}, dtype=_torch.float32, device=_dev)",
                    f"{tgt} = _torch.zeros({size}, {channels}, dtype=_torch.float32, device=_dev)",
                )
            else:
                self._emit_spatial_branch(
                    f"{tgt} = _torch.zeros(*_sp, {size}, dtype=_torch.float32, device=_dev)",
                    f"{tgt} = _torch.zeros({size}, dtype=_torch.float32, device=_dev)",
                )

    def _emit_array_index_read(self, node: ArrayIndexAccess) -> str:
        """Emit array[index] read with clamped bounds."""
        arr = self._emit_expr(node.array)
        idx = self._emit_expr(node.index)
        tmp = self._tmp()

        # String arrays are Python lists
        self._emit(f"if isinstance({arr}, list):")
        self._indent += 1
        self._emit(f"{tmp} = {arr}[max(0, min(int(round({idx}.item() if _torch.is_tensor({idx}) else float({idx}))), len({arr}) - 1))]")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        # Tensor arrays — check dim for vec (dim 2 or 5) vs scalar (dim 1 or 4)
        self._emit(f"if {arr}.dim() in (2, 5):")
        self._indent += 1
        self._emit(f"_ai = _torch.clamp(_torch.floor({idx}).long(), 0, {arr}.shape[-2] - 1)")
        self._emit(f"if _ai.dim() == 0:")
        self._indent += 1
        self._emit(f"{tmp} = {arr}[..., _ai.item(), :]")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_C = {arr}.shape[-1]")
        self._emit(f"_idx_exp = _ai.unsqueeze(-1).unsqueeze(-1).expand(*_ai.shape, 1, _C)")
        self._emit(f"if _idx_exp.shape[:3] != {arr}.shape[:3]:")
        self._indent += 1
        self._emit(f"_idx_exp = _idx_exp.expand({arr}.shape[:3] + (1, _C))")
        self._indent -= 1
        self._emit(f"{tmp} = _torch.gather({arr}, dim=-2, index=_idx_exp).squeeze(-2)")
        self._indent -= 1
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_ai = _torch.clamp(_torch.floor({idx}).long(), 0, {arr}.shape[-1] - 1)")
        self._emit(f"if _ai.dim() == 0:")
        self._indent += 1
        self._emit(f"{tmp} = {arr}[..., _ai.item()]")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_idx_exp = _ai.unsqueeze(-1)")
        self._emit(f"if _idx_exp.shape[:3] != {arr}.shape[:3]:")
        self._indent += 1
        self._emit(f"_idx_exp = _idx_exp.expand({arr}.shape[:3] + (1,))")
        self._indent -= 1
        self._emit(f"{tmp} = _torch.gather({arr}, dim=-1, index=_idx_exp).squeeze(-1)")
        self._indent -= 1
        self._indent -= 1
        self._indent -= 1

        return tmp

    def _emit_array_index_assign(self, target: ArrayIndexAccess, value_expr: str):
        """Emit array[index] = value with clamped bounds."""
        arr = self._emit_expr(target.array)
        idx = self._emit_expr(target.index)

        # String arrays
        self._emit(f"if isinstance({arr}, list):")
        self._indent += 1
        self._emit(f"_al = list({arr})")
        self._emit(f"_al[max(0, min(int(round({idx}.item() if _torch.is_tensor({idx}) else float({idx}))), len(_al) - 1))] = {value_expr} if isinstance({value_expr}, str) else str({value_expr})")
        if isinstance(target.array, Identifier):
            self._emit(f"{self._var_target(target.array.name)} = _al")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        # Clone for copy-on-write semantics
        self._emit(f"_arr_c = {arr}.clone()")
        self._emit(f"if _arr_c.dim() in (2, 5):")
        self._indent += 1
        self._emit(f"_ai = _torch.clamp(_torch.floor({idx}).long(), 0, _arr_c.shape[-2] - 1)")
        self._emit(f"if _ai.dim() == 0:")
        self._indent += 1
        self._emit(f"_arr_c[..., _ai.item(), :] = _es({value_expr}, _arr_c.shape[:-2]) if _arr_c.dim() > 2 else {value_expr}")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_C = _arr_c.shape[-1]")
        self._emit(f"_idx_exp = _ai.unsqueeze(-1).unsqueeze(-1).expand(*_ai.shape, 1, _C)")
        self._emit(f"if _idx_exp.shape[:3] != _arr_c.shape[:3]:")
        self._indent += 1
        self._emit(f"_idx_exp = _idx_exp.expand(_arr_c.shape[:3] + (1, _C))")
        self._indent -= 1
        self._emit(f"_val_exp = _es({value_expr}, _arr_c.shape[:-2]).unsqueeze(-2)")
        self._emit(f"_arr_c.scatter_(-2, _idx_exp, _val_exp)")
        self._indent -= 1
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_ai = _torch.clamp(_torch.floor({idx}).long(), 0, _arr_c.shape[-1] - 1)")
        self._emit(f"if _ai.dim() == 0:")
        self._indent += 1
        self._emit(f"_arr_c[..., _ai.item()] = _es({value_expr}, _arr_c.shape[:-1]) if _arr_c.dim() > 1 else {value_expr}")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_idx_exp = _ai.unsqueeze(-1)")
        self._emit(f"if _idx_exp.shape[:3] != _arr_c.shape[:3]:")
        self._indent += 1
        self._emit(f"_idx_exp = _idx_exp.expand(_arr_c.shape[:3] + (1,))")
        self._indent -= 1
        self._emit(f"_val_exp = _es({value_expr}, _arr_c.shape[:-1]).unsqueeze(-1)")
        self._emit(f"_arr_c.scatter_(-1, _idx_exp, _val_exp)")
        self._indent -= 1
        self._indent -= 1
        if isinstance(target.array, Identifier):
            self._emit(f"{self._var_target(target.array.name)} = _arr_c")
        self._indent -= 1

    def _emit_array_literal(self, node: ArrayLiteral) -> str:
        """Emit array literal {a, b, c} as a tensor stack."""
        elems = [self._emit_expr(e) for e in node.elements]
        tmp = self._tmp()
        es_list = ", ".join(f"_es({e}, _sp) if _sp else {e}" for e in elems)
        self._emit(f"{tmp} = _torch.stack([{es_list}], dim=-1)")
        return tmp

    def _emit_scatter_write(self, target: BindingIndexAccess, value_expr: str, op=None):
        """Emit @OUT[px, py] = value scatter write."""
        name = target.binding.name
        arg_exprs = [self._emit_expr(a) for a in target.args]
        px, py = arg_exprs[0], arg_exprs[1]
        has_frame = len(arg_exprs) == 3

        # Get or create output buffer
        self._emit(f"if {name!r} not in _bind or not _torch.is_tensor(_bind[{name!r}]):")
        self._indent += 1
        self._emit(f"_sv = {value_expr}")
        self._emit(f"_sc = _sv.shape[-1] if _torch.is_tensor(_sv) and _sv.dim() >= 1 and _sv.shape[-1] in (2,3,4) else 1")
        self._emit(f"if _sp:")
        self._indent += 1
        self._emit(f"_bind[{name!r}] = _torch.zeros(*_sp, _sc, dtype=_torch.float32, device=_dev) if _sc > 1 else _torch.zeros(*_sp, dtype=_torch.float32, device=_dev)")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_bind[{name!r}] = _torch.zeros(1, 1, 1, _sc, dtype=_torch.float32, device=_dev) if _sc > 1 else _torch.zeros(1, 1, 1, dtype=_torch.float32, device=_dev)")
        self._indent -= 1
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_sv = {value_expr}")
        self._indent -= 1

        self._emit(f"_sb = _bind[{name!r}]")
        self._emit(f"_sB, _sH, _sW = _sb.shape[0], _sb.shape[1], _sb.shape[2]")

        # Clamp coordinates
        self._emit(f"_six = _torch.clamp(_torch.floor(_es({px}, (_sB,_sH,_sW)) if _sp else {px}).long(), 0, _sW - 1)")
        self._emit(f"_siy = _torch.clamp(_torch.floor(_es({py}, (_sB,_sH,_sW)) if _sp else {py}).long(), 0, _sH - 1)")
        self._emit(f"_sval = _es(_sv, (_sB,_sH,_sW)) if _sp else _sv")

        if has_frame:
            self._emit(f"_sbi = _torch.clamp((_es({arg_exprs[2]}, (_sB,_sH,_sW)) if _sp else {arg_exprs[2]}).long(), 0, _sB - 1)")
        else:
            self._emit(f"_sbi = _torch.arange(_sB, device=_dev).view(_sB, 1, 1).expand(_sB, _sH, _sW)")

        self._emit(f"_fb = _sbi.reshape(-1)")
        self._emit(f"_fy = _siy.reshape(-1)")
        self._emit(f"_fx = _six.reshape(-1)")

        # Reshape value to match buffer layout
        self._emit(f"if _sb.dim() == 4:")
        self._indent += 1
        self._emit(f"_fv = _sval.reshape(-1, _sb.shape[-1])")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_fv = _sval.reshape(-1) if _torch.is_tensor(_sval) and _sval.dim() > 0 else _sval")
        self._indent -= 1

        # Apply scatter operation
        if op is None:
            self._emit(f"_sb[_fb, _fy, _fx] = _fv")
        elif op == "+":
            self._emit(f"_sb.index_put_((_fb, _fy, _fx), _fv, accumulate=True)")
        elif op == "-":
            self._emit(f"_sb.index_put_((_fb, _fy, _fx), -_fv, accumulate=True)")
        elif op == "*":
            self._emit(f"_sb[_fb, _fy, _fx] = _sb[_fb, _fy, _fx] * _fv")
        elif op == "/":
            self._emit(f"_sb[_fb, _fy, _fx] = _sb[_fb, _fy, _fx] / _tw(_fv == 0, _SAFE_EPS, _fv)")

    def _emit_function_def(self, stmt: FunctionDef):
        """Emit a user-defined function as a nested Python def."""
        self._user_functions.add(stmt.name)
        params = [f"_p_{pname}" for _, pname in stmt.params]
        params_str = ", ".join(params + ["_depth=0"])
        self._emit(f"def _uf_{stmt.name}({params_str}):")
        self._indent += 1
        self._emit(f"if _depth > {MAX_CALL_DEPTH}: raise RuntimeError('Maximum function call depth exceeded in {stmt.name}()')")

        # Save and swap local vars context for function body
        saved_locals = self._local_vars
        self._local_vars = {}
        # Register params as locals
        for _, pname in stmt.params:
            self._local_vars[pname] = f"_p_{pname}"

        # Collect all vars declared anywhere in function body (including nested blocks)
        body_vars, _ = self._collect_modified_vars(stmt.body)
        for vname in body_vars:
            if vname not in self._local_vars:  # don't overwrite params
                self._local_vars[vname] = f"_uf_lv_{vname}"

        saved_in_fn = self._in_user_function
        self._in_user_function = True

        for s in stmt.body:
            self._emit_stmt(s)

        # Default return if no explicit return
        self._emit(f"return _torch.scalar_tensor(0.0, dtype=_torch.float32, device=_dev)")
        self._in_user_function = saved_in_fn
        self._local_vars = saved_locals
        self._indent -= 1

    def _emit_return_stmt(self, stmt: ReturnStmt):
        """Emit a return statement inside a user function."""
        value = self._emit_expr(stmt.value)
        self._emit(f"return {value}")

    def _emit_assignment(self, stmt: Assignment):
        target = stmt.target
        value_expr = self._emit_expr(stmt.value)
        if isinstance(target, Identifier):
            # Coerce channel count for vec2/vec3 variables
            vec_type = self._var_vec_type.get(target.name)
            if vec_type and vec_type.channels < 4:
                value_expr = self._emit_vec_coerce(value_expr, vec_type)
            self._emit(f"{self._var_target(target.name)} = {value_expr}")
        elif isinstance(target, BindingRef):
            self._emit(f"_bind[{target.name!r}] = {value_expr}")
        elif isinstance(target, ChannelAccess):
            self._emit_channel_assign(target, value_expr)
        elif isinstance(target, ArrayIndexAccess):
            self._emit_array_index_assign(target, value_expr)
        elif isinstance(target, BindingIndexAccess):
            self._emit_scatter_write(target, value_expr, op=stmt.op)
        else:
            raise _Unsupported(f"Unsupported assignment target: {type(target).__name__}")

    def _emit_channel_assign(self, target: ChannelAccess, value_expr: str):
        channels = target.channels
        base_expr = self._emit_expr(target.object)
        tmp = self._tmp()
        self._emit(f"{tmp} = {base_expr}.clone()")

        if len(channels) == 1:
            idx = CHANNEL_MAP.get(channels)
            if idx is None:
                raise _Unsupported(f"Invalid channel: {channels}")
            self._emit(f"{tmp}[..., {idx}] = _es({value_expr}, {tmp}.shape[:-1])")
        else:
            # Multi-channel assignment: .rgb, .xy, .rgba, etc.
            indices = [CHANNEL_MAP.get(ch) for ch in channels]
            if any(i is None for i in indices):
                raise _Unsupported(f"Invalid swizzle: {channels}")
            val_tmp = self._tmp()
            self._emit(f"{val_tmp} = _es({value_expr}, {tmp}.shape[:-1])")
            multi_flag = self._tmp()
            self._emit(f"{multi_flag} = {val_tmp}.dim() >= 1 and {val_tmp}.shape[-1] > 1")
            for i, idx in enumerate(indices):
                self._emit(f"{tmp}[..., {idx}] = {val_tmp}[..., {i}] if {multi_flag} else {val_tmp}")

        if isinstance(target.object, Identifier):
            self._emit(f"{self._var_target(target.object.name)} = {tmp}")
        elif isinstance(target.object, BindingRef):
            self._emit(f"_bind[{target.object.name!r}] = {tmp}")
        else:
            raise _Unsupported("Channel assign on non-variable target")

    def _emit_if_else(self, stmt: IfElse):
        """Emit if/else.

        Scalar conditions use Python if/else (supports break/continue).
        Spatial conditions use vectorized both-branch with torch.where merging,
        same logic as the interpreter's selective cloning.
        """
        cond_expr = self._emit_expr(stmt.condition)
        cond_tmp = self._tmp()
        self._emit(f"{cond_tmp} = {cond_expr}")

        # Check if this could be a scalar condition
        # We emit both paths: scalar short-circuit and spatial vectorized
        self._emit(f"if not _torch.is_tensor({cond_tmp}) or {cond_tmp}.dim() == 0:")
        self._indent += 1
        self._emit(f"if float({cond_tmp}) > 0.5:")
        self._indent += 1
        if stmt.then_body:
            for s in stmt.then_body:
                self._emit_stmt(s)
        else:
            self._emit("pass")
        self._indent -= 1
        if stmt.else_body:
            self._emit(f"else:")
            self._indent += 1
            for s in stmt.else_body:
                self._emit_stmt(s)
            self._indent -= 1
        self._indent -= 1

        # Spatial path: this is complex (selective cloning + torch.where merge)
        # Fall back to unsupported to let the interpreter handle it
        # UNLESS there's no else body and the then body is simple
        self._emit(f"else:")
        self._indent += 1

        # For spatial if/else, we need the full clone/merge logic.
        # Emit it inline.
        lines_before = len(self._lines)
        self._emit_spatial_if_else(stmt, cond_tmp)
        if len(self._lines) == lines_before:
            self._emit("pass")  # guard against empty else block

        self._indent -= 1

    def _emit_spatial_if_else(self, stmt: IfElse, cond_var: str):
        """Emit spatial if/else with selective cloning and torch.where merge.

        Uses local variables for env vars when available (program-level locals),
        falls back to _env dict for any vars not in _local_vars.
        """
        # Collect modified variables (same analysis as interpreter)
        then_mods = self._collect_modified_vars(stmt.then_body)
        else_mods = self._collect_modified_vars(stmt.else_body) if stmt.else_body else (set(), set())
        all_env_mods = then_mods[0] | else_mods[0]
        all_bind_mods = then_mods[1] | else_mods[1]

        if not all_env_mods and not all_bind_mods:
            # Nothing modified — just execute both branches for side effects
            for s in stmt.then_body:
                self._emit_stmt(s)
            if stmt.else_body:
                for s in stmt.else_body:
                    self._emit_stmt(s)
            return

        # Snapshot modified env vars (using locals where available)
        # Some vars may be None if only declared inside one branch (not yet assigned)
        snap_vars: dict[str, str] = {}  # env_var_name -> snapshot_tmp_name
        for k in all_env_mods:
            snap = self._tmp()
            snap_vars[k] = snap
            src = self._var_target(k)
            self._emit(f"{snap} = {src}.clone() if ({src} is not None and _torch.is_tensor({src})) else {src}")

        snap_bind = self._tmp()
        bind_mod_repr = repr(all_bind_mods)
        self._emit(f"{snap_bind} = {{k: _bind[k].clone() if _torch.is_tensor(_bind[k]) else _bind[k] for k in {bind_mod_repr} if k in _bind}}")

        # Execute then-branch
        for s in stmt.then_body:
            self._emit_stmt(s)

        # Capture then-state
        then_vars: dict[str, str] = {}
        for k in all_env_mods:
            then_tmp = self._tmp()
            then_vars[k] = then_tmp
            src = self._var_target(k)
            self._emit(f"{then_tmp} = {src}")

        then_bind = self._tmp()
        self._emit(f"{then_bind} = {{k: _bind.get(k) for k in {bind_mod_repr}}}")

        # Restore snapshot
        for k in all_env_mods:
            tgt = self._var_target(k)
            self._emit(f"{tgt} = {snap_vars[k]}")
        self._emit(f"_bind.update({snap_bind})")

        if stmt.else_body:
            for s in stmt.else_body:
                self._emit_stmt(s)

        # Merge with torch.where
        cond_bool = self._tmp()
        self._emit(f"{cond_bool} = ({cond_var} > 0.5)")

        for k in all_env_mods:
            tv = then_vars[k]
            tgt = self._var_target(k)
            ev = self._tmp()
            self._emit(f"{ev} = {tgt}")
            self._emit(f"if {tv} is not None and {ev} is not None and _torch.is_tensor({tv}) and _torch.is_tensor({ev}):")
            self._indent += 1
            self._emit_tw_broadcast(cond_bool, tv, ev, tgt)
            self._indent -= 1
            self._emit(f"elif {tv} is not None:")
            self._indent += 1
            self._emit(f"{tgt} = {tv}")
            self._indent -= 1

        for k in all_bind_mods:
            tv = self._tmp()
            ev = self._tmp()
            self._emit(f"{tv} = {then_bind}.get({k!r})")
            self._emit(f"{ev} = _bind.get({k!r})")
            self._emit(f"if {tv} is not None and {ev} is not None and _torch.is_tensor({tv}) and _torch.is_tensor({ev}):")
            self._indent += 1
            self._emit_tw_broadcast(cond_bool, tv, ev, f"_bind[{k!r}]")
            self._indent -= 1
            self._emit(f"elif {tv} is not None:")
            self._indent += 1
            self._emit(f"_bind[{k!r}] = {tv}")
            self._indent -= 1

    def _is_scalar_body(self, stmts: list[ASTNode], loop_var: str) -> bool:
        """Check if a loop body operates only on scalar types (no spatial tensors).

        Returns True when every variable declaration and assignment target in the
        body has a scalar type in the type map, no binding refs, vec constructors,
        or spatial stdlib calls (sample, fetch, etc.) appear, and no read variables
        have spatial types (e.g., builtins like u, v, ix, iy).
        """
        for stmt in stmts:
            if not self._is_scalar_node(stmt, loop_var):
                return False
        return True

    def _is_scalar_node(self, node: ASTNode, loop_var: str) -> bool:
        """Recursively check if a node is scalar-only."""
        if isinstance(node, VarDecl):
            t = self.type_map.get(id(node))
            if t is None or not t.is_scalar:
                return False
            if node.initializer:
                return self._is_scalar_node(node.initializer, loop_var)
            return True
        if isinstance(node, Assignment):
            t = node.target
            if isinstance(t, (BindingRef, ChannelAccess, BindingIndexAccess, ArrayIndexAccess)):
                return False
            if isinstance(t, Identifier):
                tt = self.type_map.get(id(t))
                if tt is not None and not tt.is_scalar:
                    return False
            return self._is_scalar_node(node.value, loop_var)
        if isinstance(node, ExprStatement):
            return self._is_scalar_node(node.expr, loop_var)
        if isinstance(node, IfElse):
            if not self._is_scalar_node(node.condition, loop_var):
                return False
            for s in node.then_body:
                if not self._is_scalar_node(s, loop_var):
                    return False
            if node.else_body:
                for s in node.else_body:
                    if not self._is_scalar_node(s, loop_var):
                        return False
            return True
        if isinstance(node, (ForLoop, WhileLoop)):
            # Nested loops — check their bodies too
            for s in node.body:
                if not self._is_scalar_node(s, loop_var):
                    return False
            return True
        if isinstance(node, (BreakStmt, ContinueStmt)):
            return True
        # Expression nodes
        if isinstance(node, (NumberLiteral, StringLiteral)):
            return True
        if isinstance(node, Identifier):
            # Spatial builtins are per-pixel tensors at runtime even though
            # the type checker marks them as FLOAT.
            if node.name in _SPATIAL_BUILTINS:
                return False
            return True
        if isinstance(node, BinOp):
            return (self._is_scalar_node(node.left, loop_var)
                    and self._is_scalar_node(node.right, loop_var))
        if isinstance(node, UnaryOp):
            return self._is_scalar_node(node.operand, loop_var)
        if isinstance(node, TernaryOp):
            return (self._is_scalar_node(node.condition, loop_var)
                    and self._is_scalar_node(node.true_expr, loop_var)
                    and self._is_scalar_node(node.false_expr, loop_var))
        if isinstance(node, FunctionCall):
            # Spatial functions are not scalar
            if node.name in _SPATIAL_STDLIB:
                return False
            t = self.type_map.get(id(node))
            if t is not None and not t.is_scalar:
                return False
            return all(self._is_scalar_node(a, loop_var) for a in node.args)
        if isinstance(node, CastExpr):
            return self._is_scalar_node(node.expr, loop_var)
        if isinstance(node, (VecConstructor, MatConstructor, BindingRef,
                             BindingSampleAccess, BindingIndexAccess,
                             ArrayLiteral, ArrayIndexAccess)):
            return False
        # Unknown node type — conservative
        return False

    def _collect_modified_vars(self, stmts: list[ASTNode]) -> tuple[set[str], set[str]]:
        """Collect env var names and binding names assigned in a statement list."""
        return collect_assigned_vars(stmts)

    def _collect_read_vars(self, stmts: list[ASTNode]) -> set[str]:
        """Collect all Identifier names read in a list of statements."""
        names: set[str] = set()
        for stmt in stmts:
            self._collect_reads_node(stmt, names)
        return names

    def _collect_reads_node(self, node: ASTNode, names: set[str]):
        """Recursively collect Identifier names read in an AST node."""
        if isinstance(node, Identifier):
            names.add(node.name)
        elif isinstance(node, BinOp):
            self._collect_reads_node(node.left, names)
            self._collect_reads_node(node.right, names)
        elif isinstance(node, UnaryOp):
            self._collect_reads_node(node.operand, names)
        elif isinstance(node, TernaryOp):
            self._collect_reads_node(node.condition, names)
            self._collect_reads_node(node.true_expr, names)
            self._collect_reads_node(node.false_expr, names)
        elif isinstance(node, FunctionCall):
            for a in node.args:
                self._collect_reads_node(a, names)
        elif isinstance(node, VecConstructor):
            for a in node.args:
                self._collect_reads_node(a, names)
        elif isinstance(node, CastExpr):
            self._collect_reads_node(node.expr, names)
        elif isinstance(node, ChannelAccess):
            self._collect_reads_node(node.object, names)
        elif isinstance(node, Assignment):
            self._collect_reads_node(node.value, names)
            # Also collect reads from target (e.g., channel access on identifier)
            if isinstance(node.target, ChannelAccess):
                self._collect_reads_node(node.target.object, names)
        elif isinstance(node, VarDecl):
            if node.initializer:
                self._collect_reads_node(node.initializer, names)
        elif isinstance(node, IfElse):
            self._collect_reads_node(node.condition, names)
            for s in node.then_body:
                self._collect_reads_node(s, names)
            if node.else_body:
                for s in node.else_body:
                    self._collect_reads_node(s, names)
        elif isinstance(node, ExprStatement):
            self._collect_reads_node(node.expr, names)
        elif isinstance(node, ForLoop):
            # Recurse into nested loops so outer scope can localize all vars
            for s in node.body:
                self._collect_reads_node(s, names)
        elif isinstance(node, WhileLoop):
            self._collect_reads_node(node.condition, names)
            for s in node.body:
                self._collect_reads_node(s, names)

    def _emit_body_with_flow(self, body: list[ASTNode]):
        """Emit loop body wrapped in try/except for break/continue signals."""
        self._emit("try:")
        self._indent += 1
        if body:
            for s in body:
                self._emit_stmt(s)
        else:
            self._emit("pass")
        self._indent -= 1
        self._emit("except _CgBreak:")
        self._indent += 1
        self._emit("break")
        self._indent -= 1
        self._emit("except _CgContinue:")
        self._indent += 1
        self._emit("pass")
        self._indent -= 1

    def _emit_safe_divisor(self, divisor_expr: str) -> str:
        """Emit a zero-safe divisor guard. Returns the temp var name."""
        sd = self._tmp()
        self._emit(f"{sd} = (_SAFE_EPS if {divisor_expr} == 0 else {divisor_expr}) if not _torch.is_tensor({divisor_expr}) else _tw({divisor_expr} == 0, _SAFE_EPS, {divisor_expr})")
        return sd

    def _emit_bp(self, expr_a: str, expr_b: str) -> tuple[str, str]:
        """Emit _bp(a, b) and return the two temp var names."""
        ta = self._tmp()
        tb = self._tmp()
        self._emit(f"{ta}, {tb} = _bp({expr_a}, {expr_b})")
        return ta, tb

    def _emit_tw_broadcast(self, cond: str, tv: str, ev: str, target: str):
        """Emit torch.where with broadcast: _bp values, expand cond to match dims."""
        btv = self._tmp()
        bev = self._tmp()
        self._emit(f"{btv}, {bev} = _bp({tv}, {ev})")
        cb = self._tmp()
        self._emit(f"{cb} = {cond}")
        self._emit(f"if {cb}.dim() < {btv}.dim(): {cb} = {cb}.view(*{cb}.shape, *((1,) * ({btv}.dim() - {cb}.dim()))).expand_as({btv})")
        self._emit(f"{target} = _tw({cb}, {btv}, {bev})")

    def _emit_cond_break(self, cond_node: ASTNode):
        """Emit scalar/spatial condition check that breaks if false."""
        cond_expr = self._emit_expr(cond_node)
        cc = self._tmp()
        self._emit(f"{cc} = {cond_expr}")
        self._emit(f"if not _torch.is_tensor({cc}) or {cc}.dim() == 0:")
        self._indent += 1
        self._emit(f"if float({cc}) <= 0.5: break")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"if ({cc} > 0.5).sum().item() == 0: break")
        self._indent -= 1

    def _emit_iter_limit(self, iter_var: str, label: str):
        """Emit post-loop iteration limit check."""
        self._emit(f"if {iter_var} >= _MAX_ITER:")
        self._indent += 1
        self._emit(f"raise RuntimeError('{label} loop exceeded maximum iteration limit (' + str(_MAX_ITER) + '). Check your loop condition.')")
        self._indent -= 1

    def _stencil_to_bchw(self, stencil: _StencilInfo) -> tuple[str, str]:
        """Emit image binding → BCHW conversion with optional channel selection.

        Returns (selected_bchw_tmp, img_tmp).
        """
        img_tmp = self._tmp()
        bchw_tmp = self._tmp()
        self._emit(f"{img_tmp} = _bind[{stencil.binding_name!r}]")
        self._emit(f"{bchw_tmp} = {img_tmp}.permute(0, 3, 1, 2)")

        if stencil.channels:
            sel_tmp = self._tmp()
            ch_indices = [CHANNEL_MAP.get(ch) for ch in stencil.channels]
            if len(ch_indices) == 1:
                self._emit(f"{sel_tmp} = {bchw_tmp}[:, {ch_indices[0]}:{ch_indices[0]+1}]")
            elif ch_indices == list(range(ch_indices[0], ch_indices[0] + len(ch_indices))):
                self._emit(f"{sel_tmp} = {bchw_tmp}[:, {ch_indices[0]}:{ch_indices[-1]+1}]")
            else:
                idx_list = ", ".join(str(i) for i in ch_indices)
                self._emit(f"{sel_tmp} = {bchw_tmp}[:, [{idx_list}]]")
            return sel_tmp, img_tmp
        return bchw_tmp, img_tmp

    def _stencil_pad_and_kernel_size(self, stencil: _StencilInfo, sel_tmp: str
                                     ) -> tuple[str, str, str, str | None]:
        """Emit padding + compute kernel size for a stencil.

        Returns (padded_tmp, kh_expr, kw_expr, n_expr_or_none).
        """
        pad_tmp = self._tmp()

        if stencil.is_symmetric:
            rad_y = self._tmp()
            kh_tmp = self._tmp()

            if isinstance(stencil.y_radius, int):
                self._emit(f"{rad_y} = {stencil.y_radius}")
            else:
                r_expr = self._emit_expr(stencil.y_radius)
                self._emit(f"{rad_y} = int({r_expr}.item() if _torch.is_tensor({r_expr}) else {r_expr})")

            same_radius = (stencil.y_radius == stencil.x_radius if isinstance(stencil.y_radius, int)
                           else _ast_equal(stencil.y_radius, stencil.x_radius))
            if same_radius:
                rad_x = rad_y
                kw_tmp = kh_tmp
            else:
                rad_x = self._tmp()
                kw_tmp = self._tmp()
                if isinstance(stencil.x_radius, int):
                    self._emit(f"{rad_x} = {stencil.x_radius}")
                else:
                    r_expr = self._emit_expr(stencil.x_radius)
                    self._emit(f"{rad_x} = int({r_expr}.item() if _torch.is_tensor({r_expr}) else {r_expr})")

            self._emit(f"{kh_tmp} = 2 * {rad_y} + 1")
            if not same_radius:
                self._emit(f"{kw_tmp} = 2 * {rad_x} + 1")
            n_tmp = self._tmp()
            self._emit(f"{n_tmp} = {kh_tmp} * {kw_tmp}")
            self._emit(f"{pad_tmp} = _torch.nn.functional.pad({sel_tmp}, ({rad_x}, {rad_x}, {rad_y}, {rad_y}), mode='replicate')")
            return pad_tmp, kh_tmp, kw_tmp, n_tmp
        else:
            dy_start = stencil.dy_start
            dy_stop = stencil.dy_stop
            dx_start = stencil.dx_start
            dx_stop = stencil.dx_stop
            kH = dy_stop - dy_start
            kW = dx_stop - dx_start
            pad_t = -dy_start
            pad_b = dy_stop - 1
            pad_l = -dx_start
            pad_r = dx_stop - 1
            self._emit(f"{pad_tmp} = _torch.nn.functional.pad({sel_tmp}, ({pad_l}, {pad_r}, {pad_t}, {pad_b}), mode='replicate')")
            return pad_tmp, str(kH), str(kW), str(kH * kW)

    def _emit_box_stencil(self, stencil: _StencilInfo) -> bool:
        """Emit avg_pool2d for a box blur stencil."""
        accum_local = self._local_vars.get(stencil.accum_var, f"_env[{stencil.accum_var!r}]")
        count_local = None
        if stencil.count_var:
            count_local = self._local_vars.get(stencil.count_var, f"_env[{stencil.count_var!r}]")

        sel_tmp, _ = self._stencil_to_bchw(stencil)
        pad_tmp, kh, kw, n_expr = self._stencil_pad_and_kernel_size(stencil, sel_tmp)

        pool_tmp = self._tmp()
        result_tmp = self._tmp()
        # avg_pool2d with divisor_override=1 computes raw sum directly
        self._emit(f"{pool_tmp} = _torch.nn.functional.avg_pool2d({pad_tmp}, kernel_size=({kh}, {kw}), stride=1, padding=0, divisor_override=1)")
        self._emit(f"{result_tmp} = {pool_tmp}.permute(0, 2, 3, 1)")
        self._emit(f"{accum_local} = {result_tmp}")
        if count_local and n_expr:
            self._emit(f"{count_local} = _torch.scalar_tensor(float({n_expr}), dtype=_torch.float32, device=_dev)")
        return True

    def _emit_minmax_stencil(self, stencil: _StencilInfo) -> bool:
        """Emit max_pool2d or -max_pool2d(-x) for min/max stencils."""
        result_local = self._local_vars.get(stencil.result_var, f"_env[{stencil.result_var!r}]")

        sel_tmp, _ = self._stencil_to_bchw(stencil)
        pad_tmp, kh, kw, _ = self._stencil_pad_and_kernel_size(stencil, sel_tmp)

        pool_tmp = self._tmp()
        result_tmp = self._tmp()
        if stencil.minmax_op == "max":
            self._emit(f"{pool_tmp} = _torch.nn.functional.max_pool2d({pad_tmp}, kernel_size=({kh}, {kw}), stride=1, padding=0)")
        else:
            # min via negated max: min(x) = -max(-x)
            neg_tmp = self._tmp()
            self._emit(f"{neg_tmp} = -{pad_tmp}")
            self._emit(f"{pool_tmp} = -_torch.nn.functional.max_pool2d({neg_tmp}, kernel_size=({kh}, {kw}), stride=1, padding=0)")
        self._emit(f"{result_tmp} = {pool_tmp}.permute(0, 2, 3, 1)")
        self._emit(f"{result_local} = {result_tmp}")
        return True

    def _emit_conv2d_stencil(self, stencil: _StencilInfo) -> bool:
        """Emit depthwise conv2d for a weighted stencil with fixed kernel."""
        kw = stencil.kernel_weights
        dy_start, dy_stop = stencil.dy_start, stencil.dy_stop
        dx_start, dx_stop = stencil.dx_start, stencil.dx_stop
        kH = dy_stop - dy_start
        kW = dx_stop - dx_start

        kernel_vals = []
        for dy in range(dy_start, dy_stop):
            for dx in range(dx_start, dx_stop):
                kernel_vals.append(kw.get((dy, dx), 0.0))

        sel_tmp, _ = self._stencil_to_bchw(stencil)
        pad_tmp, _, _, _ = self._stencil_pad_and_kernel_size(stencil, sel_tmp)

        # Depthwise conv kernel: [C, 1, kH, kW] — same kernel per channel
        kern_tmp = self._tmp()
        kern_list = ", ".join(f"{v:.10g}" for v in kernel_vals)
        chan_tmp = self._tmp()
        self._emit(f"{chan_tmp} = {pad_tmp}.shape[1]")
        self._emit(f"{kern_tmp} = _torch.tensor([{kern_list}], dtype=_torch.float32, device=_dev).view(1, 1, {kH}, {kW}).expand({chan_tmp}, 1, {kH}, {kW})")

        conv_tmp = self._tmp()
        self._emit(f"{conv_tmp} = _torch.nn.functional.conv2d({pad_tmp}, {kern_tmp}.contiguous(), padding=0, groups={chan_tmp})")

        result_tmp = self._tmp()
        self._emit(f"{result_tmp} = {conv_tmp}.permute(0, 2, 3, 1)")

        target = self._local_vars.get(stencil.result_var, f"_env[{stencil.result_var!r}]")
        self._emit(f"{target} = {result_tmp}")
        return True

    def _emit_median_stencil(self, stencil: _StencilInfo) -> bool:
        """Emit unfold-based neighborhood collection for median/array-collect patterns.

        Replaces the nested for-loop that collects fetch() into arrays with a
        single Tensor.unfold operation. Post-loop code (sort, median, index)
        runs unchanged on the pre-populated arrays.
        """
        if not stencil.array_vars:
            return False

        sel_tmp, _ = self._stencil_to_bchw(stencil)
        pad_tmp, kh, kw, n_expr = self._stencil_pad_and_kernel_size(stencil, sel_tmp)

        # Unfold: [B, C, H, W] padded → [B, C, H, W, kH, kW]
        patches_tmp = self._tmp()
        self._emit(f"{patches_tmp} = {pad_tmp}.unfold(2, {kh}, 1).unfold(3, {kw}, 1)")

        # Reshape to [B, C, H, W, kH*kW]
        flat_tmp = self._tmp()
        self._emit(f"{flat_tmp} = {patches_tmp}.contiguous().reshape(*{patches_tmp}.shape[:4], -1)")

        # Write to array variables
        array_chan_pairs = stencil.array_vars  # list of (arr_name, ch_idx|None)
        has_channels = array_chan_pairs[0][1] is not None

        if has_channels:
            # Per-channel arrays (e.g. r[idx]=s.r, g[idx]=s.g, b[idx]=s.b)
            # flat_tmp is [B, C, H, W, N] — extract one channel per array
            for arr_name, ch_idx in array_chan_pairs:
                arr_tmp = self._tmp()
                self._emit(f"{arr_tmp} = {flat_tmp}[:, {ch_idx}, :, :, :]")
                tgt = self._local_vars.get(arr_name, f"_env[{arr_name!r}]")
                self._emit(f"{tgt} = {arr_tmp}")
        else:
            # Single vec array (e.g. samples[idx] = fetch(...))
            # flat_tmp is [B, C, H, W, N] → [B, H, W, N, C] for vec array
            arr_name = array_chan_pairs[0][0]
            result_tmp = self._tmp()
            self._emit(f"{result_tmp} = {flat_tmp}.permute(0, 2, 3, 4, 1)")
            tgt = self._local_vars.get(arr_name, f"_env[{arr_name!r}]")
            self._emit(f"{tgt} = {result_tmp}")

        # Update counter variable to total number of elements
        if stencil.count_var:
            ct = self._local_vars.get(stencil.count_var, f"_env[{stencil.count_var!r}]")
            self._emit(f"{ct} = _torch.scalar_tensor(float({n_expr}), dtype=_torch.float32, device=_dev)")

        return True

    def _try_emit_stencil(self, stmt: ForLoop) -> bool:
        """Try to detect and emit a stencil pattern. Returns True if handled."""
        stencil = _try_detect_stencil(stmt)
        if stencil is None:
            return False

        if stencil.kind == "box":
            return self._emit_box_stencil(stencil)
        elif stencil.kind == "minmax":
            return self._emit_minmax_stencil(stencil)
        elif stencil.kind == "median":
            return self._emit_median_stencil(stencil)
        return False

    def _emit_for_loop(self, stmt: ForLoop):
        """Emit a for loop, optimized for static ranges."""
        # Try stencil specialization first (box blur, etc.)
        if self._try_emit_stencil(stmt):
            return

        # Try fully static loop: pre-compute Python range()
        static_range = try_extract_static_range(stmt)
        if static_range is not None:
            self._emit_static_for_loop(stmt, static_range)
            return

        # General case: emit initializer + while loop
        self._emit_general_for_loop(stmt)

    def _emit_static_for_loop(
        self, stmt: ForLoop, static_range: tuple,
    ):
        """Emit a for loop with a fully static range (zero per-iteration overhead)."""
        loop_var, start, stop, step = static_range
        n = abs(stop - start) // max(abs(step), 1)
        if n > 1024:
            self._emit(f"raise RuntimeError('For loop would exceed {1024} iterations')")
            return

        has_flow_control = _body_has_break_continue(stmt.body)

        # Collect vars modified AND read in the loop body so we can use
        # local Python variables instead of _env[name] dict lookups.
        modified_vars, _ = self._collect_modified_vars(stmt.body)
        read_vars = self._collect_read_vars(stmt.body)
        all_vars = modified_vars | read_vars
        all_vars.add(loop_var)
        writeback_vars = modified_vars | {loop_var}

        # Register local variables for all vars used in this loop.
        # If a var is already a local (from an outer loop), reuse its name.
        saved_locals = {}
        for vname in all_vars:
            prev = self._local_vars.get(vname)
            saved_locals[vname] = prev
            if prev is None:
                self._local_vars[vname] = f"_lv_{vname}"

        # Initialize local vars from _env ONLY for vars not already locals
        loop_var_local = self._local_vars[loop_var]
        for vname in all_vars:
            if vname != loop_var and saved_locals[vname] is None:
                local = self._local_vars[vname]
                self._emit(f"{local} = _env.get({vname!r})")

        # Detect scalar-only loop body: all vars are float/int typed,
        # no bindings, vec constructors, or spatial stdlib calls.
        use_scalar = (
            self._scalar_loop  # already in scalar mode from outer loop
            or self._is_scalar_body(stmt.body, loop_var)
        )

        if use_scalar:
            self._setup_scalar_loop(all_vars, loop_var)
        else:
            self._setup_tensor_loop(start, stop, step)

        saved_scalar = self._scalar_loop
        self._scalar_loop = use_scalar

        self._emit(f"for _i_idx in range({n}):")
        self._indent += 1
        if use_scalar:
            if step == 1.0 or step == 1:
                self._emit(f"{loop_var_local} = {start!r} + _i_idx")
            else:
                self._emit(f"{loop_var_local} = {start!r} + _i_idx * {step!r}")
        else:
            vals_tmp = self._range_cache[(start, stop, step)]
            self._emit(f"{loop_var_local} = {vals_tmp}[_i_idx]")

        if has_flow_control:
            saved_flow = self._use_native_flow_control
            self._use_native_flow_control = True
            for s in stmt.body:
                self._emit_stmt(s)
            self._use_native_flow_control = saved_flow
        else:
            for s in stmt.body:
                self._emit_stmt(s)

        self._indent -= 1
        self._scalar_loop = saved_scalar

        if use_scalar:
            # Convert scalar results back to tensors for downstream code
            for vname in modified_vars:
                local = self._local_vars.get(vname)
                if local is not None:
                    self._emit(f"if not _torch.is_tensor({local}): {local} = _torch.scalar_tensor(float({local}), dtype=_torch.float32, device=_dev)")

        # Write back modified vars to _env
        for vname in writeback_vars:
            if saved_locals[vname] is None:
                local = self._local_vars[vname]
                self._emit(f"_env[{vname!r}] = {local}")

        # Restore previous local var mappings
        for vname, prev in saved_locals.items():
            if prev is None:
                if vname in self._local_vars:
                    del self._local_vars[vname]
            else:
                self._local_vars[vname] = prev

    def _setup_scalar_loop(
        self, all_vars: set[str], loop_var: str,
    ):
        """Prepare variables for a scalar-mode for loop.

        Converts tensor locals to Python floats before the loop body.
        """
        for vname in all_vars:
            if vname != loop_var:
                local = self._local_vars[vname]
                if local is not None:
                    self._emit(f"if _torch.is_tensor({local}): {local} = {local}.item()")

    def _setup_tensor_loop(self, start: int, stop: int, step: int):
        """Prepare variables for a tensor-mode for loop.

        Hoists arange/unbind to preamble for zero-overhead iteration.
        """
        range_key = (start, stop, step)
        vals_tmp = self._range_cache.get(range_key)
        if vals_tmp is None:
            vals_tmp = self._tmp()
            self._preamble.append(
                f"    {vals_tmp} = _torch.arange({start}, {stop}, {step}, dtype=_torch.float32, device=_dev).unbind(0)"
            )
            self._range_cache[range_key] = vals_tmp

    def _emit_general_for_loop(self, stmt: ForLoop):
        """Emit a general for loop as init + while (dynamic bounds)."""
        # Hoist sample/fetch BCHW setup outside loop to avoid per-call overhead
        self._hoist_sample_setup(stmt.body)
        self._emit_stmt(stmt.init)

        static = self._try_static_bound(stmt)
        iter_var = self._tmp()

        self._emit(f"{iter_var} = 0")
        self._emit(f"while {iter_var} < _MAX_ITER:")
        self._indent += 1

        if static is not None:
            loop_var, bound, is_le = static
            cv = self._tmp()
            self._emit(f"{cv} = {self._var_target(loop_var)}")
            self._emit(f"if not _torch.is_tensor({cv}) or {cv}.dim() == 0:")
            self._indent += 1
            ci = self._tmp()
            self._emit(f"{ci} = int({cv})")
            if is_le:
                self._emit(f"if {ci} > {bound}: break")
            else:
                self._emit(f"if {ci} >= {bound}: break")
            self._indent -= 1
            self._emit(f"else:")
            self._indent += 1
            cond_expr = self._emit_expr(stmt.condition)
            self._emit(f"if ({cond_expr} > 0.5).sum().item() == 0: break")
            self._indent -= 1
        else:
            self._emit_cond_break(stmt.condition)

        self._emit_body_with_flow(stmt.body)

        # Update always executes (even after continue)
        self._emit_stmt(stmt.update)
        self._emit(f"{iter_var} += 1")
        self._indent -= 1

        self._emit_iter_limit(iter_var, "For")

    def _try_static_bound(self, stmt: ForLoop) -> tuple[str, int, bool] | None:
        """Try to extract static loop bounds."""
        cond = stmt.condition
        if not isinstance(cond, BinOp):
            return None
        if cond.op not in ("<", "<="):
            return None
        if not isinstance(cond.left, Identifier):
            return None
        if not isinstance(cond.right, NumberLiteral):
            return None
        return (cond.left.name, int(cond.right.value), cond.op == "<=")

    def _emit_while_loop(self, stmt: WhileLoop):
        """Emit a while loop with native break/continue.

        While loops have no user-defined update statement, so native
        continue is safe. Note: continue skips the safety counter
        increment, but this is acceptable since _MAX_ITER is generous
        and the condition check still runs.
        """
        # Hoist sample/fetch BCHW setup outside loop
        self._hoist_sample_setup(stmt.body)
        iter_var = self._tmp()
        self._emit(f"{iter_var} = 0")
        self._emit(f"while {iter_var} < _MAX_ITER:")
        self._indent += 1

        self._emit_cond_break(stmt.condition)

        # Use native break/continue — no try/except needed
        saved_flow = self._use_native_flow_control
        self._use_native_flow_control = True
        for s in stmt.body:
            self._emit_stmt(s)
        self._use_native_flow_control = saved_flow

        self._emit(f"{iter_var} += 1")
        self._indent -= 1

        self._emit_iter_limit(iter_var, "While")

    # ── Expression emission ─────────────────────────────────────────────

    def _emit_expr(self, node: ASTNode) -> str:
        """Emit an expression, returning a Python expression string.

        For complex expressions, emits assignment to a temp var and
        returns the temp var name. For simple leaves, returns inline.
        """
        if isinstance(node, NumberLiteral):
            return self._get_const(node.value)

        if isinstance(node, StringLiteral):
            return repr(node.value)

        if isinstance(node, Identifier):
            local = self._local_vars.get(node.name)
            if local is not None:
                return local
            return f"_env[{node.name!r}]"

        if isinstance(node, BindingRef):
            if node.kind == "param":
                # $params may be Python float/int from defaults — ensure tensor
                tmp = self._tmp()
                self._emit(f"{tmp} = _bind[{node.name!r}]")
                param_type = self.type_map.get(id(node))
                if param_type is None or param_type == TEXType.STRING:
                    # Unknown or string type — guard needed
                    self._emit(f"if not isinstance({tmp}, (str, list)): {tmp} = _torch.as_tensor({tmp})")
                else:
                    # Known numeric type — convert directly (no isinstance guard)
                    self._emit(f"{tmp} = _torch.as_tensor({tmp})")
                return tmp
            return f"_bind[{node.name!r}]"

        if isinstance(node, ChannelAccess):
            return self._emit_channel_access(node)

        if isinstance(node, BinOp):
            return self._emit_binop(node)

        if isinstance(node, UnaryOp):
            return self._emit_unary(node)

        if isinstance(node, TernaryOp):
            return self._emit_ternary(node)

        if isinstance(node, FunctionCall):
            return self._emit_function_call(node)

        if isinstance(node, VecConstructor):
            return self._emit_vec_constructor(node)

        if isinstance(node, MatConstructor):
            return self._emit_mat_constructor(node)

        if isinstance(node, CastExpr):
            return self._emit_cast(node)

        if isinstance(node, ArrayIndexAccess):
            return self._emit_array_index_read(node)

        if isinstance(node, ArrayLiteral):
            return self._emit_array_literal(node)

        if isinstance(node, BindingIndexAccess):
            binding = self._emit_expr(node.binding)
            args = [self._emit_expr(a) for a in node.args]
            tmp = self._tmp()
            if len(args) == 3:
                fn = self._get_fn_local('fetch_frame')
                self._emit(f"{tmp} = {fn}({binding}, {args[2]}, {args[0]}, {args[1]})")
            else:
                fn = self._get_fn_local('fetch')
                self._emit(f"{tmp} = {fn}({binding}, {args[0]}, {args[1]})")
            return tmp

        if isinstance(node, BindingSampleAccess):
            binding_name = node.binding.name if isinstance(node.binding, BindingRef) else None

            # Fast path: sample with integer pixel offsets → direct tensor indexing
            if (binding_name and len(node.args) == 2
                    and binding_name in self._hoisted_bchw):
                dx_expr = _extract_uv_offset_expr(node.args[0], "u")
                dy_expr = _extract_uv_offset_expr(node.args[1], "v")
                if dx_expr is not None and dy_expr is not None:
                    return self._emit_direct_fetch(binding_name, dx_expr, dy_expr)

            args = [self._emit_expr(a) for a in node.args]

            # Medium path: inline grid_sample with hoisted BCHW + pre-allocated grid
            if (binding_name and binding_name in self._hoisted_bchw
                    and len(args) == 2):
                return self._emit_inline_grid_sample(binding_name, args[0], args[1])

            # Fallback: standard _fn_sample call
            binding = self._emit_expr(node.binding)
            tmp = self._tmp()
            if len(args) == 3:
                fn = self._get_fn_local('sample_frame')
                self._emit(f"{tmp} = {fn}({binding}, {args[2]}, {args[0]}, {args[1]})")
            else:
                fn = self._get_fn_local('sample')
                self._emit(f"{tmp} = {fn}({binding}, {args[0]}, {args[1]})")
            return tmp

        raise _Unsupported(f"Unknown expression: {type(node).__name__}")

    def _emit_channel_access(self, node: ChannelAccess) -> str:
        base = self._emit_expr(node.object)
        channels = node.channels

        if len(channels) == 1:
            idx = CHANNEL_MAP.get(channels)
            if idx is None:
                raise _Unsupported(f"Invalid channel: {channels}")
            return f"{base}[..., {idx}]"

        # Multi-channel swizzle
        indices = [CHANNEL_MAP.get(ch) for ch in channels]
        if any(i is None for i in indices):
            raise _Unsupported(f"Invalid swizzle: {channels}")

        # Contiguous slice optimization: .rgb (0,1,2) or .rg (0,1) → [..., :N]
        # This is a zero-copy view instead of 3 index ops + torch.stack.
        if indices == list(range(indices[0], indices[0] + len(indices))):
            start = indices[0]
            end = start + len(indices)
            if start == 0:
                return f"{base}[..., :{end}]"
            else:
                return f"{base}[..., {start}:{end}]"

        parts = ", ".join(f"{base}[..., {i}]" for i in indices)
        tmp = self._tmp()
        self._emit(f"{tmp} = _torch.stack([{parts}], dim=-1)")
        return tmp

    def _emit_binop(self, node: BinOp) -> str:
        left = self._emit_expr(node.left)
        right = self._emit_expr(node.right)

        # Scalar loop mode: pure Python arithmetic, no tensor dispatch
        if self._scalar_loop:
            op = node.op
            tmp = self._tmp()
            if op in _INFIX_OPS:
                self._emit(f"{tmp} = ({left} {op} {right})")
            elif op == "/":
                self._emit(f"{tmp} = ({left} / ({right} if {right} != 0 else 1e-10))")
            elif op == "%":
                self._emit(f"{tmp} = _math.fmod(float({left}), float({right}) if {right} != 0 else 1.0)")
            elif op in _CMP_OPS:
                py_op = _CMP_OPS[op]
                self._emit(f"{tmp} = (1.0 if {left} {py_op} {right} else 0.0)")
            elif op == "&&":
                self._emit(f"{tmp} = (1.0 if float({left}) > 0.5 and float({right}) > 0.5 else 0.0)")
            elif op == "||":
                self._emit(f"{tmp} = (1.0 if float({left}) > 0.5 or float({right}) > 0.5 else 0.0)")
            else:
                raise _Unsupported(f"Unknown operator: {op}")
            return tmp

        # String concatenation check — fall back
        # (can't easily detect at codegen time without type info on values)

        # Matrix operations
        lt = self.type_map.get(id(node.left))
        rt = self.type_map.get(id(node.right))
        if node.op == "*" and lt is not None and rt is not None:
            if lt.is_matrix and rt.is_matrix:
                tmp = self._tmp()
                self._emit(f"{tmp} = _torch.matmul({left}, {right})")
                return tmp
            if lt.is_matrix and rt.is_vector:
                tmp = self._tmp()
                self._emit(f"{tmp} = _torch.matmul({left}, {right}.unsqueeze(-1)).squeeze(-1)")
                return tmp

        # Check if broadcasting is needed (scalar vs vector mixed)
        needs_broadcast = False
        needs_channel_pad = False
        needs_runtime_bp = False
        if lt is not None and rt is not None:
            l_vec = lt.is_vector
            r_vec = rt.is_vector
            l_scalar = not l_vec and not lt.is_matrix
            r_scalar = not r_vec and not rt.is_matrix
            if (l_vec and r_scalar) or (l_scalar and r_vec):
                needs_broadcast = True
            elif l_vec and r_vec and lt.channels != rt.channels:
                needs_channel_pad = True
        elif lt is None or rt is None:
            # Type info missing (e.g., optimizer created new nodes) — use runtime broadcast
            needs_runtime_bp = True

        if needs_broadcast:
            tl = self._tmp()
            tr = self._tmp()
            self._emit(f"{tl}, {tr} = _bp({left}, {right})")
            left, right = tl, tr
        elif needs_runtime_bp:
            # Runtime fallback — only call _bp when shapes actually differ.
            # Cache .dim() into locals (_ld/_rd) to avoid calling it twice per side.
            tl = self._tmp()
            tr = self._tmp()
            ld = self._tmp()
            rd = self._tmp()
            self._emit(f"if _torch.is_tensor({left}) and _torch.is_tensor({right}):")
            self._indent += 1
            self._emit(f"{ld} = {left}.dim(); {rd} = {right}.dim()")
            self._emit(f"if {ld} != {rd} or ({ld} >= 1 and {rd} >= 1 and {left}.shape[-1] != {right}.shape[-1]):")
            self._indent += 1
            self._emit(f"{tl}, {tr} = _bp({left}, {right})")
            self._indent -= 1
            self._emit(f"else:")
            self._indent += 1
            self._emit(f"{tl}, {tr} = {left}, {right}")
            self._indent -= 1
            self._indent -= 1
            self._emit(f"else:")
            self._indent += 1
            self._emit(f"{tl}, {tr} = {left}, {right}")
            self._indent -= 1
            left, right = tl, tr
        elif needs_channel_pad:
            lc, rc = lt.channels, rt.channels
            if lc < rc:
                tl = self._tmp()
                self._emit(f"{tl} = _torch.nn.functional.pad({left}, (0, {rc - lc}))")
                left = tl
            else:
                tr = self._tmp()
                self._emit(f"{tr} = _torch.nn.functional.pad({right}, (0, {lc - rc}))")
                right = tr

        op = node.op
        tmp = self._tmp()

        if op in _INFIX_OPS:
            self._emit(f"{tmp} = ({left} {op} {right})")
        elif op == "/":
            # Skip safe-divisor guard when divisor is a non-zero compile-time constant
            if isinstance(node.right, NumberLiteral) and node.right.value != 0:
                self._emit(f"{tmp} = ({left} / {right})")
            else:
                sd = self._emit_safe_divisor(right)
                self._emit(f"{tmp} = ({left} / {sd})")
        elif op == "%":
            if isinstance(node.right, NumberLiteral) and node.right.value != 0:
                self._emit(f"{tmp} = _torch.fmod({left}, {right})")
            else:
                sd = self._emit_safe_divisor(right)
                self._emit(f"{tmp} = _torch.fmod({left}, {sd})")
        elif op in _CMP_OPS:
            py_op = _CMP_OPS[op]
            self._emit(f"{tmp} = ({left} {py_op} {right}).float()")
        elif op == "&&":
            self._emit(f"{tmp} = (({left} > 0.5) & ({right} > 0.5)).float()")
        elif op == "||":
            self._emit(f"{tmp} = (({left} > 0.5) | ({right} > 0.5)).float()")
        else:
            raise _Unsupported(f"Unknown operator: {op}")

        return tmp

    def _emit_unary(self, node: UnaryOp) -> str:
        operand = self._emit_expr(node.operand)
        tmp = self._tmp()
        if node.op == "-":
            self._emit(f"{tmp} = (-{operand})")
        elif node.op == "!":
            if self._scalar_loop:
                self._emit(f"{tmp} = (1.0 if float({operand}) <= 0.5 else 0.0)")
            else:
                self._emit(f"{tmp} = ({operand} <= 0.5).float()")
        else:
            raise _Unsupported(f"Unknown unary op: {node.op}")
        return tmp

    def _emit_ternary(self, node: TernaryOp) -> str:
        cond = self._emit_expr(node.condition)
        true_val = self._emit_expr(node.true_expr)
        false_val = self._emit_expr(node.false_expr)
        tmp = self._tmp()

        # Scalar loop mode: simple Python ternary
        if self._scalar_loop:
            self._emit(f"{tmp} = {true_val} if float({cond}) > 0.5 else {false_val}")
            return tmp

        # Check if string handling is needed via type_map
        true_type = self.type_map.get(id(node.true_expr))
        false_type = self.type_map.get(id(node.false_expr))
        both_numeric = (true_type is not None and true_type != TEXType.STRING
                        and false_type is not None and false_type != TEXType.STRING)

        if not both_numeric:
            # String ternary guard (causes graph break)
            self._emit(f"if isinstance({true_val}, str) or isinstance({false_val}, str):")
            self._indent += 1
            self._emit(f"_cs = {cond}.float().mean().item() if _torch.is_tensor({cond}) and {cond}.dim() > 0 else (float({cond}.item()) if _torch.is_tensor({cond}) else float({cond}))")
            self._emit(f"{tmp} = {true_val} if _cs > 0.5 else {false_val}")
            self._indent -= 1
            self._emit(f"elif not _torch.is_tensor({cond}) or {cond}.dim() == 0:")
        else:
            # Both arms are numeric — skip string guard (no graph break)
            self._emit(f"if not _torch.is_tensor({cond}) or {cond}.dim() == 0:")
        self._indent += 1
        self._emit(f"{tmp} = {true_val} if float({cond}) > 0.5 else {false_val}")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        atv = self._tmp()
        afv = self._tmp()
        self._emit(f"{atv} = _torch.as_tensor({true_val})")
        self._emit(f"{afv} = _torch.as_tensor({false_val})")
        cb = self._tmp()
        self._emit(f"{cb} = {cond} > 0.5")
        self._emit_tw_broadcast(cb, atv, afv, tmp)
        self._indent -= 1

        return tmp

    def _emit_function_call(self, node: FunctionCall) -> str:
        name = node.name

        # Fast path: convert sample(@img, u+expr*px, v+expr*py) → direct fetch
        # Resolves through local variable definitions (e.g. off_u = float(px2) * px)
        if (name == "sample" and len(node.args) == 3
                and isinstance(node.args[0], BindingRef)):
            bname = node.args[0].name
            if bname in self._hoisted_bchw:
                u_arg = _resolve_through_locals(node.args[1], self._var_initializers)
                v_arg = _resolve_through_locals(node.args[2], self._var_initializers)
                dx_expr = _extract_uv_offset_expr(u_arg, "u")
                dy_expr = _extract_uv_offset_expr(v_arg, "v")
                if dx_expr is not None and dy_expr is not None:
                    return self._emit_direct_fetch(bname, dx_expr, dy_expr)

        args = [self._emit_expr(a) for a in node.args]
        tmp = self._tmp()

        # User-defined function call
        if name in self._user_functions:
            args_str = ", ".join(args)
            depth_arg = ", _depth=_depth+1" if self._in_user_function else ""
            self._emit(f"{tmp} = _uf_{name}({args_str}{depth_arg})")
            return tmp

        # Scalar loop mode: use Python math module instead of torch
        if self._scalar_loop:
            result = self._emit_scalar_fn_call(node, name, args, tmp)
            if result is not None:
                return result

        # Dispatch table for specialized tensor-path handlers
        handler = self._fn_dispatch.get(name)
        if handler is not None:
            result = handler(node, args, tmp)
            if result is not None:
                return result

        # Table-driven inline torch functions (simple 1:1 mappings)
        if len(args) == 1:
            torch_fn = _INLINE_TORCH_1ARG.get(name)
            if torch_fn is not None:
                self._emit(f"{tmp} = _torch.{torch_fn}({args[0]})")
                return tmp
        elif len(args) == 2:
            torch_fn = _INLINE_TORCH_2ARG.get(name)
            if torch_fn is not None:
                self._emit(f"{tmp} = _torch.{torch_fn}({args[0]}, {args[1]})")
                return tmp

        # General case: pre-resolved local for stdlib function (avoids dict lookup per call)
        fn_local = self._get_fn_local(name)
        args_str = ", ".join(args)
        self._emit(f"{tmp} = {fn_local}({args_str})")
        # Skip isinstance guard when the return type is known to be numeric
        ret_type = self.type_map.get(id(node))
        if ret_type is None or ret_type == TEXType.STRING:
            self._emit(f"if not isinstance({tmp}, (str, list, _torch.Tensor)): {tmp} = _torch.scalar_tensor(float({tmp}), dtype=_torch.float32, device=_dev)")
        return tmp

    # ------------------------------------------------------------------ #
    #  Function-call dispatch handlers                                    #
    # ------------------------------------------------------------------ #

    def _emit_scalar_fn_call(
        self, node: FunctionCall, name: str, args: list[str], tmp: str,
    ) -> str | None:
        """Emit scalar (Python float/math) versions of stdlib functions.

        Returns the result variable name if handled, None to fall through
        to the tensor path.
        """
        if len(args) == 1:
            math_fn = _SCALAR_MATH_1ARG.get(name)
            if math_fn is not None:
                if math_fn == "copysign":
                    # sign(x) → math.copysign(1.0, x) (0 returns 0 in torch but 1.0 in math)
                    self._emit(f"{tmp} = (0.0 if {args[0]} == 0.0 else _math.copysign(1.0, {args[0]}))")
                else:
                    self._emit(f"{tmp} = _math.{math_fn}(float({args[0]}))")
                return tmp
            if name == "sqrt":
                self._emit(f"{tmp} = _math.sqrt(max(float({args[0]}), 0.0))")
                return tmp
            if name == "fract":
                self._emit(f"{tmp} = float({args[0]}) % 1.0")
                return tmp
            if name == "round":
                self._emit(f"{tmp} = round(float({args[0]}))")
                return tmp
        elif len(args) == 2:
            if name == "pow":
                exp_node = node.args[1]
                if isinstance(exp_node, NumberLiteral):
                    v = exp_node.value
                    if v == 0.0:
                        self._emit(f"{tmp} = 1.0")
                        return tmp
                    if v == 1.0:
                        return args[0]
                    if v == 2.0:
                        self._emit(f"{tmp} = float({args[0]}) * float({args[0]})")
                        return tmp
                self._emit(f"{tmp} = _math.pow(max(float({args[0]}), 1e-10), float({args[1]}))")
                return tmp
            math_fn = _SCALAR_MATH_2ARG.get(name)
            if math_fn is not None:
                # max/min are builtins, others are in the math module
                prefix = "" if math_fn in ("max", "min") else "_math."
                self._emit(f"{tmp} = {prefix}{math_fn}(float({args[0]}), float({args[1]}))")
                return tmp
            if name == "step":
                self._emit(f"{tmp} = 1.0 if float({args[1]}) >= float({args[0]}) else 0.0")
                return tmp
            if name == "mod":
                self._emit(f"{tmp} = _math.fmod(float({args[0]}), float({args[1]}))")
                return tmp
        elif len(args) == 3:
            if name == "lerp":
                self._emit(f"{tmp} = float({args[0]}) + (float({args[1]}) - float({args[0]})) * float({args[2]})")
                return tmp
            if name == "clamp":
                self._emit(f"{tmp} = max(float({args[1]}), min(float({args[2]}), float({args[0]})))")
                return tmp
            if name == "smoothstep":
                edge0, edge1, x = args
                t = self._tmp()
                self._emit(f"{t} = max(0.0, min(1.0, (float({x}) - float({edge0})) / (float({edge1}) - float({edge0}) + 1e-10)))")
                self._emit(f"{tmp} = {t} * {t} * (3.0 - 2.0 * {t})")
                return tmp
        # Fall through to tensor path for unhandled scalar functions
        return None

    def _emit_fn_pow(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit pow(base, exp) with constant-exponent specializations."""
        if len(args) != 2:
            return None
        exp_node = node.args[1]
        if isinstance(exp_node, NumberLiteral):
            v = exp_node.value
            if v == 0.0:
                self._emit(f"{tmp} = _torch.ones_like({args[0]})")
                return tmp
            if v == 1.0:
                return args[0]
            if v == 2.0:
                self._emit(f"{tmp} = {args[0]} * {args[0]}")
                return tmp
            if v == 3.0:
                sq = self._tmp()
                self._emit(f"{sq} = {args[0]} * {args[0]}")
                self._emit(f"{tmp} = {sq} * {args[0]}")
                return tmp
            if v == 0.5:
                self._emit(f"{tmp} = _torch.sqrt(_torch.clamp({args[0]}, min=0.0))")
                return tmp
            if v == -1.0:
                self._emit(f"{tmp} = _torch.reciprocal({args[0]} + _SAFE_EPS)")
                return tmp
            if v == -0.5:
                self._emit(f"{tmp} = _torch.rsqrt(_torch.clamp({args[0]}, min=_SAFE_EPS))")
                return tmp
            if v == 4.0:
                sq = self._tmp()
                self._emit(f"{sq} = {args[0]} * {args[0]}")
                self._emit(f"{tmp} = {sq} * {sq}")
                return tmp
            if v == -2.0:
                sq = self._tmp()
                self._emit(f"{sq} = {args[0]} * {args[0]}")
                self._emit(f"{tmp} = _torch.reciprocal({sq} + _SAFE_EPS)")
                return tmp
        # General case: exp-log trick (safe for non-negative base)
        self._emit(f"{tmp} = _torch.exp(_torch.log(_torch.clamp({args[0]}, min=_SAFE_EPS)) * {args[1]})")
        return tmp

    def _emit_fn_minmax(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit max/min with clamp specializations and nested-clamp detection."""
        if len(args) != 2:
            return None
        name = node.name
        # Detect min(max(x, lo), hi) or max(min(x, hi), lo) → torch.clamp
        inner_name = "max" if name == "min" else "min"
        for outer_const_idx in (0, 1):
            if not isinstance(node.args[outer_const_idx], NumberLiteral):
                continue
            inner_idx = 1 - outer_const_idx
            inner_node = node.args[inner_idx]
            if (isinstance(inner_node, FunctionCall)
                    and inner_node.name == inner_name
                    and len(inner_node.args) == 2):
                for inner_const_idx in (0, 1):
                    if isinstance(inner_node.args[inner_const_idx], NumberLiteral):
                        inner_val_idx = 1 - inner_const_idx
                        inner_arg = self._emit_expr(inner_node.args[inner_val_idx])
                        if name == "min":
                            lo = inner_node.args[inner_const_idx].value
                            hi = node.args[outer_const_idx].value
                        else:
                            hi = inner_node.args[inner_const_idx].value
                            lo = node.args[outer_const_idx].value
                        self._emit(f"{tmp} = _torch.clamp({inner_arg}, {lo}, {hi})")
                        return tmp
        # Single constant arg → clamp_min/clamp_max
        clamp_fn = "clamp_min" if name == "max" else "clamp_max"
        if isinstance(node.args[1], NumberLiteral):
            self._emit(f"{tmp} = _torch.{clamp_fn}({args[0]}, {node.args[1].value})")
            return tmp
        if isinstance(node.args[0], NumberLiteral):
            self._emit(f"{tmp} = _torch.{clamp_fn}({args[1]}, {node.args[0].value})")
            return tmp
        # No constant → standard torch.maximum/minimum
        torch_fn = "maximum" if name == "max" else "minimum"
        self._emit(f"{tmp} = _torch.{torch_fn}({args[0]}, {args[1]})")
        return tmp

    def _emit_fn_lerp(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit lerp(a, b, t) with type-aware broadcast."""
        if len(args) != 3:
            return None
        # a + (b - a) * t — avoids torch.lerp which requires tensor first arg
        at = self.type_map.get(id(node.args[0]))
        tt = self.type_map.get(id(node.args[2]))
        diff = self._tmp()
        self._emit(f"{diff} = {args[1]} - {args[0]}")
        if at is not None and tt is not None:
            if at.is_vector and tt.is_scalar:
                # Known vec/scalar: inline unsqueeze, skip _bp call entirely
                self._emit(f"{tmp} = {args[0]} + {diff} * {args[2]}.unsqueeze(-1)")
            else:
                # Same rank (vec+vec or scalar+scalar): no broadcast needed
                self._emit(f"{tmp} = {args[0]} + {diff} * {args[2]}")
        else:
            # Type info missing — conservative runtime fallback via _bp
            bd, bt = self._emit_bp(diff, args[2])
            self._emit(f"{tmp} = {args[0]} + {bd} * {bt}")
        return tmp

    def _emit_fn_luma(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit luma(color) as weighted channel sum (Rec.709)."""
        if len(args) != 1:
            return None
        ret_type = self.type_map.get(id(node.args[0]))
        if ret_type is None or not ret_type.is_vector or ret_type.channels < 3:
            return None
        c = args[0]
        self._emit(f"{tmp} = {c}[..., 0] * 0.2126 + {c}[..., 1] * 0.7152 + {c}[..., 2] * 0.0722")
        return tmp

    def _emit_fn_math_1arg(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit specialized 1-arg math: sqrt, log variants, fract, trig, etc."""
        if len(args) != 1:
            return None
        name = node.name
        if name == "sqrt":
            self._emit(f"{tmp} = _torch.sqrt(_torch.clamp({args[0]}, min=0.0))")
        elif name == "log":
            self._emit(f"{tmp} = _torch.log(_torch.clamp({args[0]}, min=_SAFE_EPS))")
        elif name == "log2":
            self._emit(f"{tmp} = _torch.log2(_torch.clamp({args[0]}, min=_SAFE_EPS))")
        elif name == "log10":
            self._emit(f"{tmp} = _torch.log10(_torch.clamp({args[0]}, min=_SAFE_EPS))")
        elif name == "fract":
            self._emit(f"{tmp} = {args[0]} - _torch.floor({args[0]})")
        elif name == "isnan":
            self._emit(f"{tmp} = _torch.isnan({args[0]}).float()")
        elif name == "isinf":
            self._emit(f"{tmp} = _torch.isinf({args[0]}).float()")
        elif name == "pow2":
            self._emit(f"{tmp} = _torch.pow(2.0, {args[0]})")
        elif name == "pow10":
            self._emit(f"{tmp} = _torch.pow(10.0, {args[0]})")
        elif name == "sincos":
            self._emit(f"{tmp} = _torch.stack([_torch.sin({args[0]}), _torch.cos({args[0]})], dim=-1)")
        else:
            return None
        return tmp

    def _emit_fn_vector(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit vector geometry: dot, distance, normalize, length, cross, reflect."""
        name = node.name
        if name == "dot" and len(args) == 2:
            self._emit(f"{tmp} = _torch.einsum('...c,...c->...', {args[0]}, {args[1]})")
        elif name == "distance" and len(args) == 2:
            self._emit(f"{tmp} = _torch.linalg.vector_norm({args[0]} - {args[1]}, dim=-1)")
        elif name == "normalize" and len(args) == 1:
            self._emit(f"{tmp} = {args[0]} / (_torch.linalg.vector_norm({args[0]}, dim=-1, keepdim=True) + _SAFE_EPS)")
        elif name == "length" and len(args) == 1:
            self._emit(f"{tmp} = _torch.linalg.vector_norm({args[0]}, dim=-1)")
        elif name == "cross" and len(args) == 2:
            self._emit(f"{tmp} = _torch.cross({args[0]}[..., :3], {args[1]}[..., :3], dim=-1)")
        elif name == "reflect" and len(args) == 2:
            dt = self._tmp()
            self._emit(f"{dt} = ({args[0]} * {args[1]}).sum(dim=-1, keepdim=True)")
            self._emit(f"{tmp} = {args[0]} - 2.0 * {dt} * {args[1]}")
        else:
            return None
        return tmp

    def _emit_fn_shaping(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit shaping functions: smoothstep, step, clamp, fit."""
        name = node.name
        if name == "clamp" and len(args) == 3:
            # Pass Python floats directly when lo/hi are constants
            # (avoids scalar tensor creation overhead)
            lo = node.args[1].value if isinstance(node.args[1], NumberLiteral) else args[1]
            hi = node.args[2].value if isinstance(node.args[2], NumberLiteral) else args[2]
            self._emit(f"{tmp} = _torch.clamp({args[0]}, {lo}, {hi})")
        elif name == "step" and len(args) == 2:
            threshold = node.args[0].value if isinstance(node.args[0], NumberLiteral) else args[0]
            self._emit(f"{tmp} = ({args[1]} >= {threshold}).float()")
        elif name == "smoothstep" and len(args) == 3:
            num, den = self._emit_bp(f"{args[2]} - {args[0]}", f"{args[1]} - {args[0]} + _SAFE_EPS")
            tt = self._tmp()
            self._emit(f"{tt} = _torch.clamp({num} / {den}, 0.0, 1.0)")
            self._emit(f"{tmp} = {tt} * {tt} * (3.0 - 2.0 * {tt})")
        elif name == "fit" and len(args) == 5:
            tt = self._tmp()
            self._emit(f"{tt} = ({args[0]} - {args[1]}) / ({args[2]} - {args[1]} + _SAFE_EPS)")
            rng = self._tmp()
            self._emit(f"{rng} = {args[4]} - {args[3]}")
            br, btt = self._emit_bp(rng, tt)
            self._emit(f"{tmp} = {args[3]} + {br} * {btt}")
        else:
            return None
        return tmp

    def _emit_fn_safe(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit safe arithmetic: spow, sdiv, smin, smax, mod."""
        name = node.name
        if name == "mod" and len(args) == 2:
            self._emit(f"{tmp} = _torch.fmod({args[0]}, _tw({args[1]} == 0, _SAFE_EPS, {args[1]}))")
        elif name == "spow" and len(args) == 2:
            at = self._tmp()
            self._emit(f"{at} = _torch.abs({args[0]})")
            mask = self._tmp()
            self._emit(f"{mask} = {at} < _SAFE_EPS")
            self._emit(f"{tmp} = _tw({mask}, _torch.zeros_like({args[0]}), _torch.sign({args[0]}) * _torch.pow(_torch.clamp({at}, min=_SAFE_EPS), {args[1]}))")
        elif name == "sdiv" and len(args) == 2:
            mask = self._tmp()
            self._emit(f"{mask} = _torch.abs({args[1]}) < _SAFE_EPS")
            self._emit(f"{tmp} = _tw({mask}, _torch.zeros_like({args[0]}), {args[0]} / _tw({mask}, _torch.ones_like({args[1]}), {args[1]}))")
        elif name == "smin" and len(args) == 3:
            h = self._tmp()
            self._emit(f"{h} = _torch.clamp(0.5 + 0.5 * ({args[1]} - {args[0]}) / ({args[2]} + _SAFE_EPS), 0.0, 1.0)")
            diff, bh = self._emit_bp(f"{args[0]} - {args[1]}", h)
            self._emit(f"{tmp} = {args[1]} + {diff} * {bh} - {args[2]} * {bh} * (1.0 - {bh})")
        elif name == "smax" and len(args) == 3:
            h = self._tmp()
            self._emit(f"{h} = _torch.clamp(0.5 - 0.5 * ({args[1]} - {args[0]}) / ({args[2]} + _SAFE_EPS), 0.0, 1.0)")
            diff, bh = self._emit_bp(f"{args[0]} - {args[1]}", h)
            self._emit(f"{tmp} = {args[1]} + {diff} * {bh} + {args[2]} * {bh} * (1.0 - {bh})")
        else:
            return None
        return tmp

    def _emit_fn_sdf(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit SDF primitives: sdf_circle, sdf_box, sdf_line."""
        name = node.name
        if name == "sdf_circle" and len(args) == 3:
            self._emit(f"{tmp} = _torch.hypot({args[0]}, {args[1]}) - {args[2]}")
        elif name == "sdf_box" and len(args) == 4:
            dx = self._tmp()
            dy = self._tmp()
            self._emit(f"{dx} = _torch.abs({args[0]}) - {args[2]}")
            self._emit(f"{dy} = _torch.abs({args[1]}) - {args[3]}")
            dxc = self._tmp()
            dyc = self._tmp()
            self._emit(f"{dxc} = _torch.clamp({dx}, min=0.0)")
            self._emit(f"{dyc} = _torch.clamp({dy}, min=0.0)")
            self._emit(f"{tmp} = _torch.sqrt({dxc} * {dxc} + {dyc} * {dyc}) + _torch.clamp(_torch.maximum({dx}, {dy}), max=0.0)")
        elif name == "sdf_line" and len(args) == 6:
            pax = self._tmp()
            pay = self._tmp()
            bax = self._tmp()
            bay = self._tmp()
            self._emit(f"{pax} = {args[0]} - {args[2]}")
            self._emit(f"{pay} = {args[1]} - {args[3]}")
            self._emit(f"{bax} = {args[4]} - {args[2]}")
            self._emit(f"{bay} = {args[5]} - {args[3]}")
            h = self._tmp()
            self._emit(f"{h} = _torch.clamp(({pax} * {bax} + {pay} * {bay}) / ({bax} * {bax} + {bay} * {bay} + _SAFE_EPS), 0.0, 1.0)")
            self._emit(f"{tmp} = _torch.hypot({pax} - {bax} * {h}, {pay} - {bay} * {h})")
        else:
            return None
        return tmp

    def _emit_fn_reduce(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit image reductions: img_sum, img_mean, img_min, img_max."""
        if len(args) != 1:
            return None
        op = _IMG_REDUCE_OPS.get(node.name)
        if op is None:
            return None
        self._emit(f"{tmp} = {args[0]}.{op}(dim=(1, 2), keepdim=True)")
        return tmp

    def _emit_fn_matrix(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit matrix operations: transpose, determinant, inverse."""
        if len(args) != 1:
            return None
        name = node.name
        if name == "transpose":
            self._emit(f"{tmp} = {args[0]}.transpose(-2, -1)")
        elif name == "determinant":
            self._emit(f"{tmp} = _torch.linalg.det({args[0]})")
        elif name == "inverse":
            self._emit(f"{tmp} = _torch.linalg.inv({args[0]})")
        else:
            return None
        return tmp

    def _emit_fn_sample(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit sample(@img, u, v) with inline grid_sample when binding is hoisted."""
        if len(args) != 3:
            return None  # sample_frame has 4 args — fall through
        # args[0] is the binding (already emitted), args[1]=u, args[2]=v
        binding_node = node.args[0]
        bname = binding_node.name if isinstance(binding_node, BindingRef) else None
        if bname and bname in self._hoisted_bchw:
            return self._emit_inline_grid_sample(bname, args[1], args[2])
        return None  # fall through to _fns[] path

    def _emit_fn_fetch(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit fetch(@img, px, py) with direct tensor indexing when binding is hoisted."""
        if len(args) != 3:
            return None
        binding_node = node.args[0]
        bname = binding_node.name if isinstance(binding_node, BindingRef) else None
        if bname and bname in self._hoisted_bchw:
            img_var = f"_bind[{bname!r}]"
            px = self._tmp()
            py = self._tmp()
            self._emit(f"{px} = {args[1]}.clamp(0, {img_var}.shape[2] - 1).long()")
            self._emit(f"{py} = {args[2]}.clamp(0, {img_var}.shape[1] - 1).long()")
            # B=1 fast path: direct indexing without batch dim
            self._emit(f"if {img_var}.shape[0] == 1:")
            self._indent += 1
            self._emit(f"{tmp} = {img_var}[0, {py}, {px}]"
                       f" if {px}.dim() < 3"
                       f" else {img_var}[0, {py}[0], {px}[0]]")
            self._indent -= 1
            self._emit(f"else:")
            self._indent += 1
            self._emit(f"{tmp} = {img_var}[_torch.arange({img_var}.shape[0]).view(-1,1,1), {py}, {px}]")
            self._indent -= 1
            return tmp
        return None

    def _emit_vec_constructor(self, node: VecConstructor) -> str:
        n = node.size

        # Optimization: hoist all-literal vec constructors to preamble.
        # vec3(0.2126, 0.7152, 0.0722) → single pre-computed tensor reused on every call.
        if len(node.args) > 1 and all(isinstance(a, NumberLiteral) for a in node.args) and len(node.args) == n:
            values = tuple(a.value for a in node.args)
            cached = self._vec_const_cache.get(values)
            if cached is not None:
                return cached
            var = self._tmp()
            vals_repr = ", ".join(repr(v) for v in values)
            self._preamble.append(
                f"    {var} = _torch.tensor([{vals_repr}], dtype=_torch.float32, device=_dev)"
                f".reshape(" + ", ".join(["1"] * 3) + f", {n}).expand(*_sp, {n})"
                f" if _sp else _torch.tensor([{vals_repr}], dtype=_torch.float32, device=_dev)"
            )
            self._vec_const_cache[values] = var
            return var

        args = [self._emit_expr(a) for a in node.args]
        tmp = self._tmp()

        if len(args) == 1:
            # Check compile-time type for identity case
            arg_type = self.type_map.get(id(node.args[0]))
            if arg_type is not None and arg_type.is_vector and arg_type.channels == n:
                # vec4(vec4_val) — identity / type cast
                self._emit(f"{tmp} = {args[0]}")
            else:
                # Broadcast: vec3(scalar) → call _es once, then expand
                es_tmp = self._tmp()
                self._emit(f"{es_tmp} = _es({args[0]}, _sp)")
                components = ", ".join([es_tmp] * n)
                self._emit(f"{tmp} = _torch.stack([{components}], dim=-1)")
        else:
            # Flatten composite args using compile-time types:
            # e.g. vec4(vec3_val, float_val) → [v[...,0], v[...,1], v[...,2], f]
            component_exprs: list[str] = []
            for i, a in enumerate(args):
                arg_type = self.type_map.get(id(node.args[i]))
                if arg_type is not None and arg_type.is_vector:
                    # Vector channel slices are already spatial — skip _es()
                    for ch in range(arg_type.channels):
                        component_exprs.append(f"{a}[..., {ch}]")
                else:
                    component_exprs.append(f"_es({a}, _sp)")

            components = ", ".join(component_exprs)
            self._emit(f"{tmp} = _torch.stack([{components}], dim=-1)")

        return tmp

    def _emit_mat_constructor(self, node: MatConstructor) -> str:
        n = node.size
        args = [self._emit_expr(a) for a in node.args]
        tmp = self._tmp()

        if len(args) == 1:
            self._emit(f"{tmp} = _torch.eye({n}, dtype=_torch.float32, device=_dev) * {args[0]}")
        elif len(args) == n * n:
            components = ", ".join(f"_es({a}, _sp)" for a in args)
            flat = self._tmp()
            self._emit(f"{flat} = _torch.stack([{components}], dim=-1)")
            self._emit(f"{tmp} = {flat}.reshape({flat}.shape[:-1] + ({n}, {n}))")
        else:
            raise _Unsupported(f"mat{n} wrong arg count")

        return tmp

    def _emit_cast(self, node: CastExpr) -> str:
        value = self._emit_expr(node.expr)

        # Fast path: float cast on already-float/int source is a no-op
        if node.target_type == "float":
            src_type = self.type_map.get(id(node.expr))
            if src_type is not None and src_type in (TEXType.FLOAT, TEXType.INT):
                return value

        # Scalar loop mode: Python float casts
        if self._scalar_loop:
            tmp = self._tmp()
            if node.target_type == "int":
                self._emit(f"{tmp} = float(_math.floor(float({value})))")
            elif node.target_type == "float":
                self._emit(f"{tmp} = float({value})")
            elif node.target_type == "string":
                self._emit(f"{tmp} = str({value})")
            else:
                self._emit(f"{tmp} = {value}")
            return tmp

        tmp = self._tmp()

        if node.target_type == "string":
            self._emit(f"if _torch.is_tensor({value}):")
            self._indent += 1
            self._emit(f"_cv = {value}.item() if {value}.dim() == 0 else {value}.float().mean().item()")
            self._emit(f"{tmp} = str(int(_cv)) if _cv == int(_cv) else str(_cv)")
            self._indent -= 1
            self._emit(f"else:")
            self._indent += 1
            self._emit(f"{tmp} = str({value})")
            self._indent -= 1
        elif node.target_type == "int":
            # floor() matches interpreter semantics (round toward -inf, not truncate)
            self._emit(f"{tmp} = _torch.floor({value}) if _torch.is_tensor({value}) else _torch.scalar_tensor(_math.floor({value}), dtype=_torch.float32, device=_dev)")
        elif node.target_type == "float":
            self._emit(f"{tmp} = {value}.float() if _torch.is_tensor({value}) else _torch.scalar_tensor(float({value}), dtype=_torch.float32, device=_dev)")
        else:
            self._emit(f"{tmp} = {value}")

        return tmp
