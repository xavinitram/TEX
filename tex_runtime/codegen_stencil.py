"""
STR-7 (cluster 2) — codegen stencil analysis (pure AST, zero emission).

The stencil-pattern detection cluster, extracted verbatim from `codegen.py`: it
inspects the AST for fetch/convolution/reduction loop patterns and returns routing
decisions + `_StencilInfo`, emitting NOTHING. Depends only on the IR (`ast_nodes`)
and `types` — a strict leaf below the emission core (`codegen.py` imports back the
handful of entry points it and its emission-island call). No `_CodeGen`/torch/stdlib
reference, so no import cycle; relocation cannot change emitted numeric output.
"""
from dataclasses import dataclass
from ..tex_compiler.ast_nodes import (
    ASTNode, Program, VarDecl, Assignment, ForLoop, ExprStatement, FunctionDef,
    BinOp, UnaryOp, FunctionCall, Identifier, BindingRef, ChannelAccess, NumberLiteral,
    CastExpr, ArrayIndexAccess, BindingIndexAccess, BindingSampleAccess,
    try_extract_static_range, collect_assigned_vars,
    iter_child_nodes as _iter_child_nodes,
)
from ..tex_compiler.types import CHANNEL_MAP


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


def _match_tap(expr: ASTNode) -> FunctionCall | BindingIndexAccess | BindingSampleAccess | None:
    """Match expr EXACTLY as a stencil tap: a bare fetch()/sample() call, or
    @binding[...] / @binding(...) access.

    This is a MATCH, not a search. It deliberately does NOT look inside BinOp:
    the lowerings this feeds (avg_pool2d / max_pool2d / unfold) can only express
    an *unweighted* neighbourhood over the tap itself, so the tap must BE the
    whole accumulated term. A composed term like `@A[ix+dx, iy+dy] * 0.5` has no
    such form — returning the inner fetch would emit a kernel with no trace of
    the `* 0.5`, silently computing 2x the right answer. Returning None instead
    makes the caller decline the stencil route, and the program falls back to the
    interpreter, which evaluates the weight correctly (just not accelerated).

    It likewise does not unwrap ChannelAccess: every caller peels the swizzle off
    itself so it can record `channels`. A ChannelAccess still present here means
    the swizzle sits somewhere this matcher cannot attribute (e.g. under a BinOp),
    and diving past it would silently drop the channel selection.
    """
    if isinstance(expr, FunctionCall) and expr.name in ("fetch", "sample"):
        return expr
    if isinstance(expr, (BindingIndexAccess, BindingSampleAccess)):
        return expr
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

    fetch_node = _match_tap(fetch_expr)
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

    fc = _match_tap(fetch_expr)
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

            fetch_node = _match_tap(resolved)
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


def _collect_local_decls(stmts: list[ASTNode], out: set) -> None:
    """Names DECLARED (`VarDecl`) directly in a loop body.

    Distinct from `_collect_local_defs`, and the distinction is the whole point:
    `_collect_local_defs` records every identifier ASSIGNMENT so taps can be resolved
    through temporaries (`su = u + i*px; ... sample(su, sv)`). Using that same map to
    decide which statements may be DISCARDED is circular — it contains every assignment
    by construction, so the check passes for all of them, and the lowering then deletes
    statements nobody accounted for.

    A name declared inside the body cannot be read after the loop, so dropping its
    assignment is safe. A name assigned but declared OUTSIDE escapes, and the lowering
    would silently stop computing it."""
    for stmt in stmts:
        if isinstance(stmt, VarDecl):
            out.add(stmt.name)


def _iter_for_loops(stmts: list[ASTNode]):
    """Yield every ForLoop node in a statement tree (nested included)."""
    stack = list(stmts)
    while stack:
        n = stack.pop()
        if isinstance(n, ForLoop):
            yield n
        stack.extend(_iter_child_nodes(n))


def detect_stencil_route(program: Program) -> bool:
    """UC-2: True iff this program should be default-routed through the codegen
    tier for stencil lowering. Routes only when it contains at least one EXACT
    (fetch-based / conv2d) stencil and NO sample-based stencil — the sample-based
    avg_pool2d lowering diverges from the interpreter's sub-pixel grid_sample
    (~25·r/W), so those stay opt-in. Uses codegen's own detectors, so the query
    sees exactly what emission would lower."""
    has_exact = False
    stmts = program.statements
    for idx in range(len(stmts)):
        inline = _try_detect_inline_stencil(stmts, idx)
        if inline is not None and inline.is_fetch:
            has_exact = True
    for loop in _iter_for_loops(stmts):
        st = _try_detect_stencil(loop)
        if st is not None:
            if st.is_fetch:
                has_exact = True
            else:
                return False  # sample-based stencil → lowering would diverge
    return has_exact


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
    for stmt in outer_loop.body:
        if isinstance(stmt, ForLoop):
            if inner_loop is not None:
                return None
            inner_loop = stmt
        elif isinstance(stmt, Assignment):
            # DECLINE every outer-body assignment. Emission replaces the ENTIRE nest, so an
            # outer-body statement's effect is simply dropped: an arbitrary assignment would be
            # deleted silently (and it defeats the one-lowering-per-loop check, which only
            # inspects the inner body), and a COUNT increment is never correctly materialized
            # either — the box/median emitters always emit kH*kW, the INNER count, so an outer
            # counter (incremented kH times) diverges from the interpreter on any downstream
            # use, with or WITHOUT a co-resident inner counter. The interpreter computes both
            # correctly; let it. (The parity fuzzer generates neither shape.)
            return None
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
    local_decls: set = set()
    _collect_local_decls(outer_loop.body, local_decls)
    _collect_local_decls(inner_loop.body, local_decls)

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
            # Allow intermediate definitions (su/sv for the sample pattern) — but only
            # for names DECLARED inside the nest. `local_defs` would accept every
            # assignment here, since it recorded them all itself (see
            # _collect_local_decls); a name declared outside outlives the loop, and the
            # lowering replaces the whole nest, so its update would just stop happening.
            if isinstance(stmt.target, Identifier) and stmt.target.name in local_decls:
                continue
        elif isinstance(stmt, (VarDecl, ExprStatement)):
            continue
        # Unknown statement type or unrecognized assignment — not necessarily fatal
        has_unknown = True

    # Exactly ONE lowering may claim the loop. Emission replaces the entire nest
    # with a single pool/unfold, so a second accumulator of a different kind would
    # simply stop being computed: `acc = acc + tap; m = max(m, tap);` in one body
    # used to lower as box and leave `m` at its init, silently. (Two accumulators
    # of the SAME kind already set has_unknown above; this covers the cross-kind
    # case, which the box-then-minmax-then-median preference order otherwise hides.)
    if sum((accum_info is not None, minmax_info is not None, bool(array_collects))) > 1:
        has_unknown = True

    # (count_var: the outer-counter route is declined in the outer-body scan above, where the
    # decision is made — an outer counter never survives the nest replacement.)

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
            count_var=inner_count_var,   # outer-counter case is declined above (has_unknown)
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
    # `not has_unknown` for the same reason box and minmax check it: lowering
    # REPLACES the whole loop nest with the unfold, so any statement in the body
    # this pass could not account for would simply stop running. Without it, a
    # body that collects taps AND does anything else — accumulate, branch, call —
    # silently loses that other work.
    if array_collects and not accum_info and not minmax_info and not has_unknown:
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
                    count_var=inner_count_var,   # outer-counter case is declined above (has_unknown)
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
    """Recursively collect all Identifier names READ in an AST subtree.

    Uses the generic child iterator so every node type (ArrayDecl/ArrayLiteral
    elements, MatConstructor, ReturnStmt, FunctionDef bodies, ...) is covered.
    Plain-Identifier assignment targets are writes, not reads, and stay
    excluded — stencil tap consumption must not become needlessly conservative.
    """
    if isinstance(node, Identifier):
        refs.add(node.name)
        return
    if isinstance(node, Assignment):
        if not isinstance(node.target, Identifier):
            _collect_ident_refs(node.target, refs)
        _collect_ident_refs(node.value, refs)
        return
    for child in _iter_child_nodes(node):
        _collect_ident_refs(child, refs)


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
        elif b_name != binding_name or ch != all_channels:
            # Different binding — stop. And likewise a different SWIZZLE: `all_channels` is
            # recorded from the FIRST tap and then applied to the whole lowered kernel, so
            # a `@A.r` tap followed by a `@A.g` one used to lower as if both read `.r` —
            # every later tap silently reading the wrong channel. The conv2d emitter has
            # one channel selection to give; a nest that needs two is not a conv2d, so stop
            # collecting and let the interpreter read each tap as written.
            break

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

    # Gap statements (skipped by the combo search between the last tap and the
    # combo) execute BETWEEN the taps and the combo in the interpreter, while
    # the emitted conv2d collapses taps+combo into one op at the combo's
    # position. That is only sound when the gap cannot observe or perturb
    # stencil state: a gap write to a tap var would be baked over by the kernel
    # weights, a gap read of a consumed tap would hit an undefined local, a gap
    # write to the combo target must not be overwritten out of order, and a gap
    # binding assignment would be seen by the deferred _bind read. Reject all
    # of those patterns — normal per-statement emission handles them correctly.
    if combo_idx > i:
        gap_stmts = stmts[i:combo_idx]
        gap_writes, gap_bind_writes = collect_assigned_vars(gap_stmts)
        if gap_bind_writes:
            return None
        gap_refs: set[str] = set()
        for g in gap_stmts:
            if isinstance(g, FunctionDef):
                return None  # function scoping — bail out conservatively
            _collect_ident_refs(g, gap_refs)
        touched = gap_writes | gap_refs
        if combo_var in touched or any(name in touched for name in tap_var_map):
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
