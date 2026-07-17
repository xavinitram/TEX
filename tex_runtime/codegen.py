"""
TEX Codegen — compile TEX AST to Python functions for zero-overhead execution.

Instead of tree-walking the AST on every frame, this module generates a Python
function string and compiles it via exec(); the caller (tex_runtime/compiled.py)
caches the callable per fingerprint. Subsequent executions call the function
directly, eliminating:
  - Per-node dispatch table lookups
  - Per-node Python function call overhead
  - Redundant dict lookups for env/bindings

Falls back to None (caller uses tree-walking interpreter) for unsupported
patterns. Includes stencil specialization (avg_pool2d, max_pool2d, conv2d,
unfold), sample/fetch inlining with hoisted BCHW + grid buffers, and
function specializations (pow, luma, clamp, etc.).
"""
from __future__ import annotations

import math
import os

from dataclasses import dataclass
from typing import Any

import torch

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
    iter_child_nodes as _ast_iter_child_nodes,
)
from ..tex_compiler.types import TEXType, CHANNEL_MAP, TYPE_NAME_MAP
from .codegen_stdfns import _EMIT_DISPATCH, _EmitStdFnsMixin
from .codegen_stencil import (
    _StencilInfo, _ast_equal, _is_ident, _try_detect_stencil, _try_detect_inline_stencil, detect_stencil_route,
)
from .interpreter import (MAX_CALL_DEPTH, MAX_LOOP_ITERATIONS, _BUILTIN_NAMES,
                          _broadcast_pair, _ensure_spatial)
from .stdlib import SAFE_EPSILON, _lerp_f32, _to_tensor


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

# M-5: infix op → in-place torch function for `out=` temp reuse.
_INFIX_OUT_FN = {"+": "add", "-": "sub", "*": "mul"}

# M-5: reuse a dead, freshly-allocated arithmetic temp as the `out=` target of
# the next elementwise op, saving one allocator call per fused arithmetic node.
# Legality (verified): the reused buffer is a codegen `_tN` produced by this same
# method (never a binding/builtin/literal/user-var/view), same-shape (no
# broadcast), single-use (the AST is a tree; CSE-shared exprs become Identifiers,
# which are excluded), and per-call (not a cross-execute ring). Kill switch:
# TEX_CODEGEN_NO_OUT_REUSE=1.
_OUT_REUSE_ENABLED = os.environ.get("TEX_CODEGEN_NO_OUT_REUSE") != "1"

# Comparison operators
_CMP_OPS = {
    "==": "==", "!=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">=",
}




from .codegen_persist import (
    _cg_filename, _register_codegen_linecache, materialize_codegen,
)


def _reads_time_builtin(program: Program) -> bool:
    """ENG-7: does this program read a host-time builtin? (see try_compile).

    Derived from `_collect_identifiers`, the compiler's existing builtin collector —
    ENG-7 put the time names into `_BUILTIN_NAMES`, which is what that function filters
    against, so it already knows the answer. A private AST walk here would be a second
    implementation of the same question, free to drift from the declaration it reads.
    Compile-path only (fingerprint-cached, never per-cook), so building the full set
    rather than early-exiting costs nothing that matters."""
    from .interpreter import _collect_identifiers, _TIME_BUILTIN_NAMES
    return not _TIME_BUILTIN_NAMES.isdisjoint(_collect_identifiers(program))


def try_compile(program: Program, type_map: dict[int, TEXType],
                fingerprint: str | None = None) -> Any | None:
    """Try to compile a TEX program AST to a Python function.

    Returns a callable with signature:
        fn(env, bindings, functions, device, spatial_shape) -> None

    The function mutates env and bindings dicts in-place.
    Returns None if the program uses unsupported features.

    *fingerprint* (when given) makes the compiled code object reproducible so
    it can be marshalled and persisted (PC-3). The returned fn carries
    `_tex_code` / `_tex_src` for that persistence.

    ENG-7 (v0.22): a program that reads the host-time builtins (`frame`/`fps`/`time`)
    is DECLINED here, so every compile tier self-falls-back to the interpreter — the
    one backend that reads the playhead fresh per cook.

    The reason is caching, not codegen capability: the emitter would handle these fine
    as ordinary `_env.get(...)` reads. But everything downstream of this function caches
    keyed by the program FINGERPRINT, which by design does not move when a playhead does
    — `_env_cached` would hand back frame 1's tensor forever, the codegen-only executor
    is a closure built once per fingerprint, and a captured CUDA graph replays the value
    it captured. Each of those fails the same way: the cook SUCCEEDS and the animation is
    frozen. Declining is the only variant that can't be silently wrong, and it costs
    nothing today (ComfyUI has no timeline, so these read 0 there anyway).

    Lifting this means feeding the playhead as a per-replay static input buffer — the
    same mechanism the graph tier needs — and belongs with the first host that has a
    real playhead (roadmap PORT-5 / GRAPH-1), not ahead of it.
    """
    try:
        if _reads_time_builtin(program):
            return None
        gen = _CodeGen(type_map)
        gen.emit_program(program)
        fn = gen.build(fingerprint)
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


def _invoke_cg(cg_fn: Any, env: dict, bindings: dict, stdlib_fns: dict,
               device: Any, spatial_shape: tuple | None) -> None:
    """Invoke a codegen-generated function with the constant argument tail.

    Single owner of the positional calling convention — it must match the
    _tex_fn signature emitted by _CodeGen.build().
    """
    cg_fn(env, bindings, stdlib_fns, device, spatial_shape,
          torch, _broadcast_pair, _ensure_spatial, torch.where,
          math, SAFE_EPSILON, CHANNEL_MAP, MAX_LOOP_ITERATIONS,
          _CgBreak, _CgContinue, _cg_lerp, _cg_lerpw)


def _cg_lerp(a, b, t):
    """Fused lerp, bit-exact with the interpreter's _lerp_f32 (torch.lerp / FMA).

    The unfused ``a + (b - a) * t`` codegen emitted before diverged from the
    interpreter by ~2e-8 (one rounding vs the FMA's one), which a singular
    smoothstep edge amplified into a full 0↔1 flip on 22% of pixels (found by
    the nightly differential fuzzer, seed 20260712). Coerces python-scalar
    operands like _to_tensor does, so torch.lerp's tensor-``start`` requirement
    holds for constant edges. Weights are pre-shaped by the caller (smin/smax/
    fit), exactly as the interpreter's own call sites do."""
    return _lerp_f32(_to_tensor(a), _to_tensor(b), _to_tensor(t))


def _cg_lerpw(a, b, t):
    """Fused lerp for lerp()/mix() — mirrors interpreter fn_lerp exactly, including
    its channel-broadcast dim-guard (a [B,H,W] weight against [B,H,W,C] values)."""
    at, bt, tt = _to_tensor(a), _to_tensor(b), _to_tensor(t)
    if tt.dim() + 1 == at.dim():
        tt = tt.unsqueeze(-1)
    return _lerp_f32(at, bt, tt)


def _matvec_expr(m: str, v: str) -> str:
    """P3: emit `m @ v` as the SAME device-gated expression `interpreter._matvec` computes
    -- CUDA elementwise broadcast-sum (3.4-3.9x faster for mat3), CPU matmul -- so codegen
    stays bit-exact with the interpreter on each device. The `A if cond else B` ternary
    short-circuits, so `v` (a slice/cat expression) is evaluated exactly once."""
    return (f"(({m} * {v}.unsqueeze(-2)).sum(-1) if {m}.is_cuda "
            f"else _torch.matmul({m}, {v}.unsqueeze(-1)).squeeze(-1))")


# Stdlib functions that are simple torch.XXX(arg) wrappers.
# In codegen, arguments are always tensors so _to_tensor is a no-op.
# We emit _torch.XXX(arg) directly, avoiding dict lookup + function call + _to_tensor.
_INLINE_TORCH_1ARG: dict[str, str] = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "atan": "atan",
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
    "atan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "exp": "exp", "abs": "fabs",
    "floor": "floor", "ceil": "ceil", "trunc": "trunc",
    "sign": "copysign",  # special-cased below
}
_SCALAR_MATH_2ARG: dict[str, str] = {
    "max": "max", "min": "min",
    "atan2": "atan2", "hypot": "hypot",
}

# The stdlib fns that `_emit_scalar_fn_call` can lower to pure Python/`_math` (so they are
# safe inside a scalar-mode loop). A stdlib call NOT in this set falls through to the TENSOR
# emission, which crashes on Python-scalar args (`spow(3.0, 0.5)` -> `_torch.abs(3.0)`) — the
# F5 codegen-crash class the differential fuzzer was blind to. `_is_scalar_node` gates on
# this so such calls (and all user-fn calls) stay in tensor mode. KEEP IN SYNC with
# `_emit_scalar_fn_call` (test_f5_codegen_scalar_loop_no_crash guards the behaviour).
_SCALAR_EMITTABLE_FNS: frozenset[str] = (
    frozenset(_SCALAR_MATH_1ARG) | frozenset(_SCALAR_MATH_2ARG)
    | frozenset({"asin", "acos", "sqrt", "fract", "round",   # 1-arg specials
                 "pow", "step", "mod",                         # 2-arg specials
                 "lerp", "clamp", "smoothstep"})               # 3-arg specials
)

# Builtins that are spatially-varying tensors at runtime (shape [B,H,W] or
# [1,1,W] etc.) despite being typed as FLOAT by the type checker.
# Used to exclude loops from scalar mode.
# NOTE: iw/ih/fi/fn are scalar (0-dim) so NOT included here.
_SPATIAL_BUILTINS: frozenset[str] = frozenset(("u", "v", "ix", "iy"))

# Stdlib functions that read/write spatial image data — incompatible with
# the scalar loop fast path.
# STR-7: single-sourced from the REG-1 registry's `spatial=` tags (TST-3 proves the
# derivation equals the old hand-maintained literal exactly). Dissolving this shared
# constant before the cluster-2 split removes the stencil/core coupling on it.
from .stdlib_registry import spatial_names as _spatial_names
_SPATIAL_STDLIB: frozenset[str] = _spatial_names()


# Shared memoized AST child-iterator lives in ast_nodes (CG-1); alias kept for
# the many call sites in this module.
_iter_child_nodes = _ast_iter_child_nodes


def _collect_reassigned_bindings(program: Program) -> set[str]:
    """Binding names REBOUND via `@name = ...` or `@name.ch = ...` anywhere in
    the program (loop/if/function bodies included). Rebinding replaces the
    _bind entry with a new tensor, so a hoisted BCHW permute view of it goes
    stale. In-place scatter writes (`@name[x, y] = ...`) mutate through shared
    storage and stay coherent with the view, so they are not collected.
    """
    names: set[str] = set()
    stack: list[ASTNode] = list(program.statements)
    while stack:
        node = stack.pop()
        if isinstance(node, Assignment):
            t = node.target
            if isinstance(t, BindingRef):
                names.add(t.name)
            elif isinstance(t, ChannelAccess) and isinstance(t.object, BindingRef):
                names.add(t.object.name)
        stack.extend(_iter_child_nodes(node))
    return names


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





class _CodeGen(_EmitStdFnsMixin):
    """Generates Python source code from a TEX AST."""

    _codegen_counter: int = 0  # class-level counter for unique (no-fingerprint) filenames

    def __init__(self, type_map: dict[int, TEXType]):
        self.type_map = type_map
        self._tmp_counter = 0
        self._lines: list[str] = []
        self._preamble: list[str] = []  # hoisted constant assignments
        self._indent = 1  # Start at 1 (inside function body)
        self._const_cache: dict[float, str] = {}  # value → variable name
        self._vec_const_cache: dict[tuple, str] = {}  # (v1, v2, ...) → variable name
        self._range_cache: dict[tuple, str] = {}  # (start, stop, step) → variable name
        self._kernel_const_cache: dict[tuple, str] = {}  # (kvals, kH, kW) → base kernel var
        # TEX local vars whose current tensor is exclusively owned (no live alias)
        # at this straight-line emission point — lets index/channel writes skip the
        # copy-on-write clone. Conservatively invalidated on ANY read (a read may
        # produce a view that shares storage) and never carried across control flow.
        self._owned: set[str] = set()
        # M-5: codegen `_tN` temps currently holding a freshly-allocated,
        # contiguous, exclusively-owned, single-use tensor — eligible to be the
        # `out=` target of the next elementwise op. Populated ONLY by
        # _emit_binop's own arithmetic results (the analysis stays local; every
        # stdlib/channel/identifier result is treated as possibly a view and is
        # never added). Never carried across control-flow boundaries.
        self._fresh_temps: set[str] = set()
        # STR-9: statements dispatch on type(node) (mirrors interpreter._stmt_dispatch).
        # The 12 statement types are mutually-exclusive flat ASTNode subclasses, so
        # this is behaviour-exact for the old isinstance cascade. NOTE: _emit_expr is
        # deliberately NOT table-driven (§5) — its branch order + inter-branch temp
        # state are load-bearing.
        self._stmt_dispatch = {
            VarDecl: self._emit_var_decl,
            ArrayDecl: self._emit_array_decl,
            Assignment: self._emit_assignment,
            ReturnStmt: self._emit_return_stmt,
            ExprStatement: self._stmt_expr,
            ParamDecl: self._stmt_noop,
            IfElse: self._stmt_if_else,
            ForLoop: self._stmt_for_loop,
            WhileLoop: self._stmt_while_loop,
            FunctionDef: self._stmt_function_def,
            BreakStmt: self._stmt_break,
            ContinueStmt: self._stmt_continue,
        }
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
        # Cross-loop reuse is only safe for bindings that are never REBOUND:
        # `@A = ...` / `@A.ch = ...` emit `_bind[name] = ...`, which stales the
        # hoisted permute view. _hoist_sample_setup skips _reassigned_bindings;
        # in-place scatter writes share storage with the view and stay coherent.
        self._hoisted_bchw: dict[str, tuple[str, str]] = {}
        # Bindings rebound anywhere in the program (set by emit_program).
        self._reassigned_bindings: set[str] = set()
        # VarDecl initializer AST nodes for sample→fetch pattern resolution
        self._var_initializers: dict[str, ASTNode] = {}
        # Vars currently holding a per-pixel/spatial value. Unlike _var_initializers
        # (only the VarDecl initializer, popped on reassignment) this tracks spatial
        # -ness across reassignments, so the scalar-loop fast path never misclassifies
        # a reassigned-spatial var and emits .item() on a tensor (a runtime crash).
        self._spatial_vars: set[str] = set()
        # Pre-resolved stdlib function locals: name → preamble variable.
        # Avoids _fns['name'] dict lookup on every call.
        self._fn_locals: dict[str, str] = {}
        # Pre-resolved $param locals: binding name → preamble variable.
        # Hoists the _bind[...] read + as_tensor conversion to one per
        # execution instead of one per use site (and per loop iteration —
        # bare $param reads are below the optimizer's LICM/CSE depth cutoff).
        self._param_locals: dict[str, str] = {}
        # When True, BreakStmt/ContinueStmt emit native Python break/continue
        # instead of raising _CgBreak/_CgContinue. Set by static range for-loops.
        self._use_native_flow_control: bool = False
        # When True, emit Python float math instead of tensor ops.
        # Set by _emit_for_loop when the entire loop body is scalar-typed.
        self._scalar_loop: bool = False
        # Dispatch table: function name → handler method for tensor-path
        # specializations.  Each handler has signature
        #   (node: FunctionCall, args: list[str], tmp: str) -> str | None
        # STR-6: dispatch table built from the co-located @_emits decorators
        # (see _EMIT_DISPATCH). Handler signature:
        #   (node: FunctionCall, args: list[str], tmp: str) -> str | None
        self._fn_dispatch: dict[str, object] = {
            name: getattr(self, attr) for name, attr in _EMIT_DISPATCH.items()
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

    def _get_param_local(self, name: str) -> str:
        """Preamble-hoisted local for a $param binding read.

        Params are read-only (the type checker rejects `$x = ...` and
        @wire/$param name collisions), so one _bind read + tensor conversion
        per execution is safe — widget changes between runs are still picked
        up. The str guard keeps string params as Python str; float-list params
        (vec/color widgets) convert via as_tensor exactly like the per-site
        known-numeric emission did. The guard is applied unconditionally
        because different use-site nodes of the same param can disagree in the
        id()-keyed type_map (optimizer-created nodes may be absent from it).
        """
        local = self._param_locals.get(name)
        if local is not None:
            return local
        local = self._tmp()
        self._preamble.append(f"    {local} = _bind[{name!r}]")
        self._preamble.append(
            f"    if not isinstance({local}, str): {local} = _torch.as_tensor({local})"
        )
        self._param_locals[name] = local
        return local

    def _direct_fetch_coords_available(self) -> bool:
        """True only when integer pixel coords ix/iy are materialized as locals.

        The direct-fetch fast path references the loop integer coordinates.
        ix/iy are read-only builtins that only exist as Python locals
        (_lv_ix/_lv_iy, initialized in the preamble from _env) when they were
        registered by _collect_modified_vars / builtin usage. If a program
        samples with u/v and never mentions ix/iy, those locals don't exist and
        the fast path would emit an undefined-name reference (NameError). Callers
        must gate on this and fall through to inline grid_sample otherwise.
        """
        return "ix" in self._local_vars and "iy" in self._local_vars

    def _emit_direct_fetch(self, binding_name: str,
                           dx_expr: ASTNode | bool, dy_expr: ASTNode | bool) -> str:
        """Emit direct tensor indexing for sample-with-integer-offset patterns.

        Shared by BindingSampleAccess and FunctionCall("sample") paths.
        Callers MUST first check _direct_fetch_coords_available(); this emitter
        references ix/iy via _var_target and assumes they are materialized.
        """
        img_var = f"_bind[{binding_name!r}]"
        ix_ref = self._var_target("ix")
        iy_ref = self._var_target("iy")
        dx_code = self._emit_expr(dx_expr) if dx_expr is not True else "0"
        dy_code = self._emit_expr(dy_expr) if dy_expr is not True else "0"
        px_tmp = self._tmp()
        py_tmp = self._tmp()
        if dx_code == "0":
            self._emit(f"{px_tmp} = {ix_ref}.clamp(0, {img_var}.shape[2] - 1).long()")
        else:
            self._emit(f"{px_tmp} = ({ix_ref} + {dx_code}).clamp(0, {img_var}.shape[2] - 1).long()")
        if dy_code == "0":
            self._emit(f"{py_tmp} = {iy_ref}.clamp(0, {img_var}.shape[1] - 1).long()")
        else:
            self._emit(f"{py_tmp} = ({iy_ref} + {dy_code}).clamp(0, {img_var}.shape[1] - 1).long()")
        tmp = self._tmp()
        self._emit(f"{tmp} = {img_var}[:, {py_tmp}, {px_tmp}, :]"
                   f" if {px_tmp}.dim() < 3"
                   f" else {img_var}[_torch.arange({img_var}.shape[0], device=_dev).view(-1,1,1), {py_tmp}, {px_tmp}, :]")
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
            if bname in self._reassigned_bindings:
                # Rebinding (`@A = ...`) anywhere in the program would stale
                # the hoisted permute view; the _fns['sample'] fallback reads
                # _bind at call time instead.
                continue
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
        # Bindings rebound anywhere in the program can't have their BCHW
        # permute view hoisted (the view would go stale on rebind).
        self._reassigned_bindings = _collect_reassigned_bindings(program)

        # Scatter copy-on-write (v0.20, mirrors interpreter._scatter_owned):
        # a scatter target the program didn't freshly allocate may alias
        # caller-owned storage — an input binding, or (worse, since the env
        # tensor cache) a cross-cook cached builtin grid (`@OUT = ix;
        # @OUT[x,y] = v;` would corrupt the cached arange for every later
        # cook). Emit a runtime owned-set + clone-before-first-write exactly
        # like the interpreter. Pre-scan so BindingRef reassignments can
        # disown even when they appear before the first scatter in the loop.
        self._has_scatter = False
        _scan = list(program.statements)
        while _scan:
            n = _scan.pop()
            if isinstance(n, Assignment) and isinstance(n.target, BindingIndexAccess):
                self._has_scatter = True
                break
            _scan.extend(_iter_child_nodes(n))
        if self._has_scatter:
            self._preamble.append("    _scat_owned = set()")

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

        # Pre-scan for inline stencil patterns (non-loop conv2d). Detection
        # runs up front, but EMISSION is deferred to the combo statement's
        # position in the main loop below: the conv reads _bind[...] when its
        # code runs, so emitting it at the top of the function would read the
        # binding BEFORE a preceding statement (e.g. `@A = @A * 0.5;`) mutates
        # it — silently diverging from the interpreter's textual order.
        inline_skip: set[int] = set()
        pending_stencils: dict[int, _StencilInfo] = {}
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
                    # max(consumed) is the combo statement — where the
                    # interpreter materializes the result. Consumed sets are
                    # strictly increasing, so pending keys never collide.
                    pending_stencils[max(inline.consumed_stmts)] = inline
                    inline_skip.update(inline.consumed_stmts)
                    # Tap variables used elsewhere (outside the consumed set)
                    # still emit as normal statements
                    idx = max(inline.consumed_stmts) + 1
                    continue
            idx += 1

        for i, stmt in enumerate(stmts):
            # Emit deferred stencils BEFORE the skip check: the combo index is
            # always in inline_skip.
            if i in pending_stencils:
                self._emit_conv2d_stencil(pending_stencils[i])
            if i in inline_skip:
                continue
            self._emit_stmt(stmt)

        # No need to write locals back to _env: the caller only reads
        # bindings (outputs), not the env dict.  Loop-scoped writebacks
        # (inside _emit_for) still happen so that subsequent statements
        # can see loop-modified vars via _env when they aren't locals.

    def build(self, fingerprint: str | None = None) -> Any:
        """Compile the generated source to a callable function.

        When *fingerprint* is given, the pseudo-filename is derived from it so
        the compiled module code object (and its marshalled blob) is
        reproducible across processes — the basis for PC-3 disk persistence.
        """
        preamble = "\n".join(self._preamble)
        body = "\n".join(self._lines)
        # Hoisted constants go before the main body so they're available
        # inside loops without per-iteration tensor creation.
        func_body = f"{preamble}\n{body}" if preamble else body
        func_src = (
            "def _tex_fn(_env, _bind, _fns, _dev, _sp, "
            "_torch, _bp, _es, _tw, _math, _SAFE_EPS, _CMAP, _MAX_ITER, "
            "_CgBreak, _CgContinue, _lerp, _lerpw):\n"
            f"{func_body}\n"
        )
        if fingerprint is not None:
            filename = _cg_filename(fingerprint)
        else:
            _CodeGen._codegen_counter += 1
            filename = f"<tex_codegen_{_CodeGen._codegen_counter}>"
        namespace: dict[str, Any] = {}
        code_obj = compile(func_src, filename, "exec")
        _register_codegen_linecache(filename, func_src)
        exec(code_obj, namespace)
        fn = namespace["_tex_fn"]
        # Stash the module code object + source for PC-3 marshal persistence.
        fn._tex_code = code_obj
        fn._tex_src = func_src
        return fn

    # ── Statement emission ──────────────────────────────────────────────

    def _emit_stmt(self, stmt: ASTNode):
        # M-5: fresh-temp reuse is scoped to a single statement's expression
        # tree. Clearing here guarantees a temp is never offered as an out=
        # target after it may have escaped into a var target on the previous
        # statement, nor carried across a control-flow boundary.
        self._fresh_temps.clear()
        handler = self._stmt_dispatch.get(type(stmt))
        if handler is None:
            raise _Unsupported(f"Unknown statement: {type(stmt).__name__}")
        handler(stmt)

    # ── STR-9 statement handlers (extracted verbatim from the old cascade) ──
    def _stmt_expr(self, stmt):
        # The expression's side effects live in the lines _emit_expr emits; the
        # returned name needs no bare-echo line. If nothing was emitted (bare
        # Identifier/NumberLiteral), keep the enclosing block non-empty — a
        # then-body may contain only this statement.
        lines_before = len(self._lines)
        self._emit_expr(stmt.expr)
        if len(self._lines) == lines_before:
            self._emit("pass")

    def _stmt_noop(self, stmt):
        pass  # ParamDecl — no runtime emission

    def _stmt_if_else(self, stmt):
        # Ownership established on one path may not hold on another, and pre-branch
        # ownership may not survive the merge — clear on both sides so in-place
        # writes are only elided within a straight-line run.
        self._owned.clear()
        self._emit_if_else(stmt)
        self._owned.clear()

    def _stmt_for_loop(self, stmt):
        self._owned.clear()
        self._emit_for_loop(stmt)
        self._owned.clear()

    def _stmt_while_loop(self, stmt):
        self._owned.clear()
        self._emit_while_loop(stmt)
        self._owned.clear()

    def _stmt_function_def(self, stmt):
        self._owned.clear()
        self._emit_function_def(stmt)
        self._owned.clear()

    def _stmt_break(self, stmt):
        # Static range for-loops use native Python break/continue (no update to
        # skip). Non-static for-loops and while-loops use exception-based flow
        # because continue must not skip the update statement.
        if self._use_native_flow_control:
            self._emit("break")
        else:
            self._emit("raise _CgBreak()")

    def _stmt_continue(self, stmt):
        if self._use_native_flow_control:
            self._emit("continue")
        else:
            self._emit("raise _CgContinue()")

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
        # Conservative: a vec/scalar initializer may be a view/alias (e.g. a
        # channel slice), so don't treat the new var as owned. The first
        # index/channel write to it will clone once; subsequent writes elide.
        self._owned.discard(stmt.name)
        if stmt.initializer:
            # Track local definitions for sample→fetch pattern resolution
            self._var_initializers[stmt.name] = stmt.initializer
            # Record whether this var now holds a spatial value (transitively).
            if self._init_is_spatial(stmt.initializer):
                self._spatial_vars.add(stmt.name)
            else:
                self._spatial_vars.discard(stmt.name)
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
            self._owned.discard(stmt.name)  # string arrays are Python lists, not COW tensors
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
        # zeros()/stack()/clone() all produce a brand-new, unaliased tensor.
        self._owned.add(stmt.name)

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
        """Emit array[index] = value with clamped bounds.

        Copy-on-write normally clones the array before writing. When the target
        is an Identifier whose local is exclusively owned here (freshly built /
        no live alias — self._owned), the clone is skipped and the element is
        written in place, turning an O(N) fill from O(N^2) tensor copies into
        O(N). Ownership is captured BEFORE the base read (which disowns it).
        """
        owned = isinstance(target.array, Identifier) and target.array.name in self._owned
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
        # Clone for copy-on-write semantics — unless the local is exclusively owned.
        if owned:
            c = arr
        else:
            c = "_arr_c"
            self._emit(f"{c} = {arr}.clone()")
        self._emit(f"if {c}.dim() in (2, 5):")
        self._indent += 1
        self._emit(f"_ai = _torch.clamp(_torch.floor({idx}).long(), 0, {c}.shape[-2] - 1)")
        self._emit(f"if _ai.dim() == 0:")
        self._indent += 1
        self._emit(f"{c}[..., _ai.item(), :] = _es({value_expr}, {c}.shape[:-2]) if {c}.dim() > 2 else {value_expr}")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_C = {c}.shape[-1]")
        self._emit(f"_idx_exp = _ai.unsqueeze(-1).unsqueeze(-1).expand(*_ai.shape, 1, _C)")
        self._emit(f"if _idx_exp.shape[:3] != {c}.shape[:3]:")
        self._indent += 1
        self._emit(f"_idx_exp = _idx_exp.expand({c}.shape[:3] + (1, _C))")
        self._indent -= 1
        self._emit(f"_val_exp = _es({value_expr}, {c}.shape[:-2]).unsqueeze(-2)")
        self._emit(f"{c}.scatter_(-2, _idx_exp, _val_exp)")
        self._indent -= 1
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_ai = _torch.clamp(_torch.floor({idx}).long(), 0, {c}.shape[-1] - 1)")
        self._emit(f"if _ai.dim() == 0:")
        self._indent += 1
        self._emit(f"{c}[..., _ai.item()] = _es({value_expr}, {c}.shape[:-1]) if {c}.dim() > 1 else {value_expr}")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_idx_exp = _ai.unsqueeze(-1)")
        self._emit(f"if _idx_exp.shape[:3] != {c}.shape[:3]:")
        self._indent += 1
        self._emit(f"_idx_exp = _idx_exp.expand({c}.shape[:3] + (1,))")
        self._indent -= 1
        self._emit(f"_val_exp = _es({value_expr}, {c}.shape[:-1]).unsqueeze(-1)")
        self._emit(f"{c}.scatter_(-1, _idx_exp, _val_exp)")
        self._indent -= 1
        self._indent -= 1
        if not owned and isinstance(target.array, Identifier):
            self._emit(f"{self._var_target(target.array.name)} = {c}")
        if isinstance(target.array, Identifier):
            # The local now holds a freshly-cloned (or already-owned) buffer.
            self._owned.add(target.array.name)
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
        self._emit(f"_scat_owned.add({name!r})")  # freshly allocated → owned
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_sv = {value_expr}")
        # COW (mirrors interpreter.py _scatter_owned): first scatter into a
        # buffer this cook didn't allocate clones before the in-place write —
        # the stored tensor may alias an input binding or a cached builtin
        # grid. Cloning also materializes expanded views so index writes work.
        self._emit(f"if {name!r} not in _scat_owned:")
        self._indent += 1
        self._emit(f"_bind[{name!r}] = _bind[{name!r}].clone()")
        self._emit(f"_scat_owned.add({name!r})")
        self._indent -= 1
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
        for vname in sorted(body_vars):  # sorted → deterministic local naming order
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
            # Rebound to an arbitrary (possibly aliasing) value — no longer owned.
            self._owned.discard(target.name)
            # Invalidate the stale VarDecl initializer: _resolve_through_locals must
            # NOT keep resolving this name to its original initializer after it has
            # been reassigned (otherwise sample()->direct-fetch misclassifies a
            # sub-pixel-shifted UV as a zero-offset nearest fetch).
            self._var_initializers.pop(target.name, None)
            # Track spatial-ness across the reassignment (transitively, via
            # _spatial_vars) so a later scalar-loop analysis isn't fooled into
            # emitting .item() on a now-per-pixel tensor.
            if self._init_is_spatial(stmt.value):
                self._spatial_vars.add(target.name)
            else:
                self._spatial_vars.discard(target.name)
        elif isinstance(target, BindingRef):
            self._emit(f"_bind[{target.name!r}] = {value_expr}")
            if self._has_scatter:
                # Rebound to an arbitrary (possibly aliasing) value — a later
                # scatter must clone again (mirrors interpreter disown-on-rebind).
                self._emit(f"_scat_owned.discard({target.name!r})")
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
        # Clone for copy-on-write — unless the base is an exclusively-owned local,
        # in which case write the channel slice in place. Capture ownership before
        # the base read (which disowns it).
        owned = isinstance(target.object, Identifier) and target.object.name in self._owned
        base_expr = self._emit_expr(target.object)
        if owned:
            tmp = base_expr
        else:
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

        if owned:
            # tmp IS the owned local; written in place, so no rebind needed.
            self._owned.add(target.object.name)
        elif isinstance(target.object, Identifier):
            self._emit(f"{self._var_target(target.object.name)} = {tmp}")
            self._owned.add(target.object.name)
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

        # A sample-hoist (BCHW permute view + grid buffer) is emitted by
        # _hoist_sample_setup at the CURRENT indent — not the preamble — so it is
        # only bound on the path it was emitted on. Every if/else re-emits its
        # bodies per branch, and the scalar-vs-spatial duplication below emits them
        # twice more; the _hoisted_bchw memo would otherwise let the second
        # emission skip the allocation yet still reference the temp
        # (UnboundLocalError). Snapshot the dominating hoists and reset to this
        # snapshot before each branch so every branch re-hoists what it samples —
        # its allocation then dominates its own uses. Preamble-hoisted caches
        # (const/range/vec/kernel/fn/param) already dominate the whole function
        # and need no such scoping.
        hoist_snap = dict(self._hoisted_bchw)

        if self._scalar_loop:
            # Scalar-mode loops (_is_scalar_body) guarantee every value in the
            # body — this condition included — stays a Python float (or 0-dim
            # tensor) on every iteration, so the spatial merge path is provably
            # dead. Emitting only the scalar branch avoids the 2^depth body
            # duplication of nested if/else (worst inside unrolled loops).
            self._emit(f"if float({cond_tmp}) > 0.5:")
            self._indent += 1
            self._hoisted_bchw = dict(hoist_snap)
            if stmt.then_body:
                for s in stmt.then_body:
                    self._emit_stmt(s)
            else:
                self._emit("pass")
            self._indent -= 1
            if stmt.else_body:
                self._emit(f"else:")
                self._indent += 1
                self._hoisted_bchw = dict(hoist_snap)
                for s in stmt.else_body:
                    self._emit_stmt(s)
                self._indent -= 1
            self._hoisted_bchw = hoist_snap
            return

        # Capture the condition's spatial-ness NOW, before either branch is
        # emitted: branch emission pops _var_initializers for any reassigned var,
        # which would make a later _init_is_spatial(condition) under-report. A
        # spatial condition means the vectorized merge below broadcasts every
        # modified var to the condition's [B,H,W] shape, so they must all be
        # treated as spatial afterwards (see the fixup in _emit_spatial_if_else).
        cond_spatial = self._init_is_spatial(stmt.condition)

        # Check if this could be a scalar condition
        # We emit both paths: scalar short-circuit and spatial vectorized
        self._emit(f"if not _torch.is_tensor({cond_tmp}) or {cond_tmp}.dim() == 0:")
        self._indent += 1
        self._emit(f"if float({cond_tmp}) > 0.5:")
        self._indent += 1
        self._hoisted_bchw = dict(hoist_snap)
        if stmt.then_body:
            for s in stmt.then_body:
                self._emit_stmt(s)
        else:
            self._emit("pass")
        self._indent -= 1
        if stmt.else_body:
            self._emit(f"else:")
            self._indent += 1
            self._hoisted_bchw = dict(hoist_snap)
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
        # Emit it inline. It re-emits the SAME bodies as the scalar path above, so
        # it must start from the same pre-if hoist snapshot (the scalar path's
        # branch-local hoists are unbound on this path).
        lines_before = len(self._lines)
        self._emit_spatial_if_else(stmt, cond_tmp, cond_spatial, hoist_snap)
        if len(self._lines) == lines_before:
            self._emit("pass")  # guard against empty else block

        self._indent -= 1

        # Every hoist emitted inside a branch above is branch-local; restore the
        # pre-if dominating snapshot for the straight-line code that follows.
        self._hoisted_bchw = hoist_snap

    def _emit_spatial_if_else(self, stmt: IfElse, cond_var: str, cond_spatial: bool,
                              hoist_snap: dict[str, tuple[str, str]]):
        """Emit spatial if/else with selective cloning and torch.where merge.

        Uses local variables for env vars when available (program-level locals),
        falls back to _env dict for any vars not in _local_vars.

        ``cond_spatial`` is the pre-computed spatial-ness of the condition (see
        _emit_if_else); it drives the post-merge _spatial_vars fixup at the end.

        ``hoist_snap`` is the sample-hoist (_hoisted_bchw) state that dominates the
        whole if/else. This method (and the scalar path in _emit_if_else) re-emit
        the same branch bodies, so a hoist from a sibling emission is unbound here;
        reset to this snapshot before each branch so every branch re-hoists what it
        samples (its allocation then dominates its own uses).
        """
        self._hoisted_bchw = dict(hoist_snap)
        # Collect modified variables (same analysis as interpreter)
        then_mods = self._collect_modified_vars(stmt.then_body)
        else_mods = self._collect_modified_vars(stmt.else_body) if stmt.else_body else (set(), set())
        # Sort to lists so emission order (and the repr() baked into source below)
        # is deterministic across processes — set iteration is PYTHONHASHSEED-
        # dependent, which breaks cross-process byte-identity of cached codegen.
        all_env_mods = sorted(then_mods[0] | else_mods[0])
        all_bind_mods = sorted(then_mods[1] | else_mods[1])

        if not all_env_mods and not all_bind_mods:
            # Nothing modified — just execute both branches for side effects
            self._hoisted_bchw = dict(hoist_snap)
            for s in stmt.then_body:
                self._emit_stmt(s)
            if stmt.else_body:
                self._hoisted_bchw = dict(hoist_snap)
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
        self._hoisted_bchw = dict(hoist_snap)
        for s in stmt.then_body:
            self._emit_stmt(s)
        # Which modified vars the then-branch left holding a spatial value. Read
        # here, before the else-branch below can reassign (and un-spatial) them.
        then_spatial = {k for k in all_env_mods if k in self._spatial_vars}

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
            self._hoisted_bchw = dict(hoist_snap)
            for s in stmt.else_body:
                self._emit_stmt(s)
        # Which modified vars are spatial on the else path (with no else body this
        # is the unchanged post-then/restore state, i.e. their pre-if value).
        else_spatial = {k for k in all_env_mods if k in self._spatial_vars}

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

        # Post-merge, a modified var can hold a per-pixel tensor even when its
        # declared type is scalar: the torch.where merge broadcasts to the
        # condition's [B,H,W] shape whenever the condition is spatial, and the
        # scalar-cond runtime path can assign a spatial value inside a branch.
        # The per-branch emission above leaves _spatial_vars reflecting only the
        # last branch (and pops _var_initializers on any reassignment), so a
        # later scalar-mode for-loop would misclassify the var as scalar and emit
        # `.item()` on the tensor -> RuntimeError -> silent interpreter fallback.
        # Conservatively mark every possibly-spatial merged var so the scalar-loop
        # analysis keeps it in tensor mode (over-marking only forgoes the scalar
        # fast path; under-marking crashes).
        if cond_spatial:
            self._spatial_vars.update(all_env_mods)
        else:
            self._spatial_vars.update(then_spatial)
            self._spatial_vars.update(else_spatial)

    def _init_is_spatial(self, expr: ASTNode, _seen: set | None = None) -> bool:
        """True if an expression draws from a per-pixel/spatial source — a binding,
        spatial builtin (u/v/ix/iy), or spatial stdlib (sample/fetch/...). A
        scalar-TYPED variable bound to such an expression is actually a per-pixel
        tensor at runtime, so a loop reading it is NOT scalar-eligible.
        """
        if _seen is None:
            _seen = set()
        if expr is None or id(expr) in _seen:
            return False
        _seen.add(id(expr))
        if isinstance(expr, (BindingRef, BindingSampleAccess, BindingIndexAccess)):
            return True
        if isinstance(expr, Identifier):
            if expr.name in _SPATIAL_BUILTINS or expr.name in self._spatial_vars:
                return True
            nxt = self._var_initializers.get(expr.name)
            return self._init_is_spatial(nxt, _seen) if nxt is not None else False
        if isinstance(expr, FunctionCall):
            if expr.name in _SPATIAL_STDLIB:
                return True
            return any(self._init_is_spatial(a, _seen) for a in expr.args)
        if isinstance(expr, BinOp):
            return (self._init_is_spatial(expr.left, _seen)
                    or self._init_is_spatial(expr.right, _seen))
        if isinstance(expr, UnaryOp):
            return self._init_is_spatial(expr.operand, _seen)
        if isinstance(expr, TernaryOp):
            return (self._init_is_spatial(expr.condition, _seen)
                    or self._init_is_spatial(expr.true_expr, _seen)
                    or self._init_is_spatial(expr.false_expr, _seen))
        if isinstance(expr, ChannelAccess):
            return self._init_is_spatial(expr.object, _seen)
        if isinstance(expr, CastExpr):
            return self._init_is_spatial(expr.expr, _seen)
        if isinstance(expr, VecConstructor):
            return any(self._init_is_spatial(a, _seen) for a in expr.args)
        return False

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
            # A scalar-typed var holding a spatial value (e.g. `float s = @A.r;`, or
            # made spatial by a pre-loop reassignment `s = @A.r*3.0;`) is a per-pixel
            # tensor at runtime. The scalar fast path emits `.item()` on it -> crash,
            # so the loop must not be treated as scalar. _spatial_vars tracks this
            # across reassignments (which pop _var_initializers).
            if node.name in self._spatial_vars:
                return False
            init = self._var_initializers.get(node.name)
            if init is not None and self._init_is_spatial(init):
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
            # F5 fix: a call is scalar-safe ONLY if it has a scalar (math/Python) codegen
            # lowering. A stdlib fn without one (spow/degrees/smin/mix) — or ANY user fn —
            # falls through to the tensor emission, which crashes on Python-scalar args; keep
            # such calls in tensor mode. (Previously any all-scalar-arg call was "scalar".)
            if node.name not in _SCALAR_EMITTABLE_FNS:
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
        """Recursively collect Identifier names read in an AST node.

        Generic child walk (covers array/matrix constructs and nested loop
        headers); plain-Identifier assignment targets are write-only and stay
        excluded, while channel/index write targets read their base.
        """
        if isinstance(node, Identifier):
            names.add(node.name)
            return
        if isinstance(node, Assignment):
            self._collect_reads_node(node.value, names)
            if not isinstance(node.target, Identifier):
                self._collect_reads_node(node.target, names)
            return
        for child in _iter_child_nodes(node):
            self._collect_reads_node(child, names)

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
        """Emit a zero-safe divisor guard. Returns the temp var name.

        In tensor-codegen mode every operand is a torch tensor, so the
        `not _torch.is_tensor(...)` branch was dead code. Bind the divisor to a
        temp once (avoids re-evaluating/re-slicing the expression 4x) and emit a
        single _tw guard.
        """
        d = self._tmp()
        sd = self._tmp()
        self._emit(f"{d} = {divisor_expr}")
        self._emit(f"{sd} = _tw({d} == 0, _SAFE_EPS, {d})")
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

    def _stencil_drop_channel_axis(self, stencil: _StencilInfo, tmp: str) -> str:
        """Undo the size-1 channel axis a single-channel selection carries.

        `_stencil_to_bchw` slices one channel as `bchw[:, i:i+1]` — the pool ops
        need that axis to stay 4-D — so permuting back yields [B, H, W, 1]. But a
        single channel is a SCALAR per pixel, which the interpreter holds as
        [B, H, W]. Assigning the un-squeezed form to a `float` accumulator leaves
        codegen one rank above the oracle, which surfaces as a rank-5 output or a
        stack() size error downstream. Multi-channel swizzles are already right
        (.rg -> 2 == vec2, .rgb -> 3 == vec3), so only len == 1 is squeezed.
        """
        if stencil.channels and len(stencil.channels) == 1:
            sq_tmp = self._tmp()
            self._emit(f"{sq_tmp} = {tmp}.squeeze(-1)")
            return sq_tmp
        return tmp

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

    @staticmethod
    def _seed_is_provably_nonzero(node: ASTNode | None) -> bool:
        """True iff `node` is a compile-time-constant seed with a non-zero component.

        Gates the box-sum pool lowering. `avg_pool2d` sums the taps in the interpreter's
        OWN left-to-right order, so a zero seed is bit-exact even at large magnitudes.
        But a non-identity additive seed folded onto the pool sum sits at the OPPOSITE
        end of the accumulation from the interpreter's seed-first left-fold
        (`((seed+t0)+t1)+...`), and FP reassociation error there scales with |seed|
        relative to the tap sum — enough to break invariant #2's 1e-5 for a large seed
        (measured: seed 100 over [0,1) data -> 2.3e-5; seed ~8 over latent-magnitude
        taps -> ~1.5e-5). Only a PROVABLE non-zero constant is declined (so the bit-exact
        static unroll runs instead); a zero/default/runtime seed keeps the pool — zero is
        exact, and a runtime seed is undecidable here yet still strictly improved by the
        fold below versus the old drop-the-seed overwrite. min/max are exempt: their fold
        is a selection (`torch.maximum`/`minimum`), which is order-independent and stays
        bit-exact for any seed."""
        if isinstance(node, NumberLiteral):
            return node.value != 0.0
        if isinstance(node, VecConstructor):
            # Provable only when every component is a literal; a runtime arg is undecidable.
            if node.args and all(isinstance(a, NumberLiteral) for a in node.args):
                return any(a.value != 0.0 for a in node.args)
        return False

    def _emit_box_stencil(self, stencil: _StencilInfo) -> bool:
        """Emit avg_pool2d for a box blur stencil."""
        accum_local = self._local_vars.get(stencil.accum_var, f"_env[{stencil.accum_var!r}]")
        count_local = None
        if stencil.count_var:
            count_local = self._local_vars.get(stencil.count_var, f"_env[{stencil.count_var!r}]")

        # A provably non-identity constant seed cannot be folded onto the pool sum within
        # invariant #2 (see _seed_is_provably_nonzero); decline BEFORE emitting anything so
        # _emit_for_loop falls through to the bit-exact static unroll (which accumulates in
        # the interpreter's order). Nothing has been emitted yet, so the fall-through is clean.
        if self._seed_is_provably_nonzero(self._var_initializers.get(stencil.accum_var)):
            return False

        sel_tmp, _ = self._stencil_to_bchw(stencil)
        pad_tmp, kh, kw, n_expr = self._stencil_pad_and_kernel_size(stencil, sel_tmp)

        pool_tmp = self._tmp()
        result_tmp = self._tmp()
        # avg_pool2d with divisor_override=1 computes raw sum directly
        self._emit(f"{pool_tmp} = _torch.nn.functional.avg_pool2d({pad_tmp}, kernel_size=({kh}, {kw}), stride=1, padding=0, divisor_override=1)")
        self._emit(f"{result_tmp} = {pool_tmp}.permute(0, 2, 3, 1)")
        result_tmp = self._stencil_drop_channel_axis(stencil, result_tmp)
        # Fold the accumulator's pre-loop SEED. The interpreter runs `acc = acc + tap`
        # from acc's entry value, so the neighbourhood sum must be ADDED to it, not
        # overwritten. `accum_local` already holds that seed — its VarDecl was emitted
        # before this loop (the nest replaced here does not touch it). We only reach here
        # for a seed that is provably zero (the identity — fold is a no-op, so bare-box
        # parity is byte-preserved) or one whose value is unknown at emit time (a runtime
        # expression, or a default-init/reassigned var not tracked in _var_initializers):
        # a provably non-zero CONSTANT was already declined above, because for it the fold
        # would reassociate past invariant #2. For the unknown case the fold is still the
        # right answer up to the pool's magnitude-dependent reassociation, and strictly
        # better than the old overwrite, which dropped the seed outright.
        self._emit(f"{accum_local} = {accum_local} + {result_tmp}")
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
        result_tmp = self._stencil_drop_channel_axis(stencil, result_tmp)
        # Fold the accumulator's pre-loop SEED. The interpreter runs `m = max/min(m, tap)`
        # starting from m's entry value, so the pooled neighbourhood must be COMBINED
        # with it, not overwritten. Unlike box-sum, max/min have NO finite identity: a
        # seed of `vec3(0.0)` over SIGNED data (latents — the core domain) legitimately
        # clamps the neighbourhood, and dropping it diverged by up to ~0.6 on the default
        # path. `result_local` holds the seed (its VarDecl was emitted before this loop);
        # `torch.maximum`/`torch.minimum` mirror the interpreter's `max()`/`min()` (both
        # go through `torch.maximum`/`minimum` in stdlib), so this is bit-exact. A seed
        # that can never win (max seed 0.0 over non-negative data) folds to a no-op, so
        # the existing bare-pool parity tests are unaffected. Invariant #2 fold.
        fold = "maximum" if stencil.minmax_op == "max" else "minimum"
        self._emit(f"{result_local} = _torch.{fold}({result_local}, {result_tmp})")
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

        # Depthwise conv kernel: [C, 1, kH, kW] — same kernel per channel.
        # The constant base kernel [1, 1, kH, kW] is a compile-time constant, so
        # hoist its host->device build to the preamble (cached by kernel shape +
        # values). Only the channel-count expand+contiguous is runtime-dependent.
        kern_key = (tuple(kernel_vals), kH, kW)
        base_kern = self._kernel_const_cache.get(kern_key)
        if base_kern is None:
            base_kern = self._tmp()
            kern_list = ", ".join(f"{v:.10g}" for v in kernel_vals)
            self._preamble.append(
                f"    {base_kern} = _torch.tensor([{kern_list}], dtype=_torch.float32, device=_dev).view(1, 1, {kH}, {kW})"
            )
            self._kernel_const_cache[kern_key] = base_kern
        chan_tmp = self._tmp()
        self._emit(f"{chan_tmp} = {pad_tmp}.shape[1]")
        kern_tmp = self._tmp()
        self._emit(f"{kern_tmp} = {base_kern}.expand({chan_tmp}, 1, {kH}, {kW}).contiguous()")

        conv_tmp = self._tmp()
        self._emit(f"{conv_tmp} = _torch.nn.functional.conv2d({pad_tmp}, {kern_tmp}, padding=0, groups={chan_tmp})")

        result_tmp = self._tmp()
        self._emit(f"{result_tmp} = {conv_tmp}.permute(0, 2, 3, 1)")
        result_tmp = self._stencil_drop_channel_axis(stencil, result_tmp)

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

        # UC-2: only lower EXACT (fetch-based) stencils. The sample-based
        # avg_pool/max_pool lowering diverges from the interpreter's sub-pixel
        # grid_sample (~25·r/W); `detect_stencil_route` already keeps them off the
        # DEFAULT path, but a direct codegen caller (compile_mode="auto"/
        # "torch_compile") bypasses that gate, so refuse the lowering here too and
        # fall through to the bit-exact per-sample codegen.
        if not stencil.is_fetch:
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
        # Sorted lists so per-var emission order is deterministic across processes
        # (set iteration is PYTHONHASHSEED-dependent — breaks cached-codegen
        # byte-identity). modified_vars stays a set for the O(1) membership tests.
        all_vars = sorted(modified_vars | read_vars | {loop_var})
        writeback_vars = sorted(modified_vars | {loop_var})

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
            for vname in sorted(modified_vars):  # deterministic emission order
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
        continue is safe. The safety counter is incremented at the TOP of
        the loop body so a native `continue` (emitted by a TEX continue)
        cannot skip it — otherwise a body that unconditionally hits continue
        would loop forever, whereas the interpreter caps at MAX_LOOP_ITERATIONS.
        """
        # Hoist sample/fetch BCHW setup outside loop
        self._hoist_sample_setup(stmt.body)
        iter_var = self._tmp()
        self._emit(f"{iter_var} = 0")
        self._emit(f"while {iter_var} < _MAX_ITER:")
        self._indent += 1
        # Increment first: immune to a native `continue` skipping it. With the
        # bound at `< _MAX_ITER` this runs exactly _MAX_ITER bodies before the
        # post-loop limit check fires — matching the interpreter's cap.
        self._emit(f"{iter_var} += 1")

        self._emit_cond_break(stmt.condition)

        # Use native break/continue — no try/except needed
        saved_flow = self._use_native_flow_control
        self._use_native_flow_control = True
        for s in stmt.body:
            self._emit_stmt(s)
        self._use_native_flow_control = saved_flow

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
            # Reading a var may alias/view its buffer (tensor slices share
            # storage), so it is no longer provably exclusively-owned. Callers
            # that write in place (index/channel assign) capture ownership
            # BEFORE emitting their base read, then re-assert it after.
            self._owned.discard(node.name)
            local = self._local_vars.get(node.name)
            if local is not None:
                return local
            return f"_env[{node.name!r}]"

        if isinstance(node, BindingRef):
            if node.kind == "param":
                # $params may be Python float/int/list from widget defaults —
                # hoisted to one _bind read + tensor conversion per execution.
                return self._get_param_local(node.name)
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
                # Only direct-fetch the zero-offset case. A nonzero integer offset
                # (dx_expr/dy_expr an ASTNode) lands sub-pixel under the interpreter's
                # bilinear grid_sample (UV step is 1/(W-1), align_corners=True), so a
                # nearest-neighbour pixel fetch would diverge — fall through to the
                # inline grid_sample path to preserve bilinear semantics.
                # Also require ix/iy to be materialized as locals; without them the
                # direct-fetch emitter would reference undefined names (NameError).
                if (dx_expr is True and dy_expr is True
                        and self._direct_fetch_coords_available()):
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
                self._emit(f"{tmp} = ({left} / ({right} if {right} != 0 else _SAFE_EPS))")
            elif op == "%":
                # Guard a zero divisor with a tiny epsilon (matching the '/' case
                # and the interpreter), NOT 1.0 — which gave 0.5 % 0 -> 0.5.
                self._emit(f"{tmp} = _math.fmod(float({left}), float({right}) if {right} != 0 else _SAFE_EPS)")
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
                m = lt.mat_size      # matrix dim: 3 or 4
                vc = rt.channels     # vector channels: 2, 3, or 4
                if vc == m:
                    self._emit(f"{tmp} = {_matvec_expr(left, right)}")
                elif m == 3 and vc == 4:
                    # mat3 * vec4: transform xyz, preserve w/alpha
                    self._emit(f"{tmp} = _torch.cat([{_matvec_expr(left, f'{right}[..., :3]')}, {right}[..., 3:4]], dim=-1)")
                elif m == 4 and vc == 3:
                    # mat4 * vec3: promote vec3 to a point (w = 1)
                    v4 = self._tmp()
                    self._emit(f"{v4} = _torch.cat([{right}, _torch.ones_like({right}[..., :1])], dim=-1)")
                    self._emit(f"{tmp} = {_matvec_expr(left, v4)}")
                else:
                    self._emit(f"{tmp} = {_matvec_expr(left, right)}")
                return tmp

        # M-5: reuse a dead fresh arithmetic temp as the out= target when the
        # OTHER operand is a compile-time scalar constant (NumberLiteral). A
        # scalar broadcasts INTO the fresh temp natively, so result.shape ==
        # fresh-temp.shape regardless of the temp's spatial-ness — provably
        # shape-safe. Handled BEFORE _bp so the dominant color-grade `vec*k ± j`
        # pattern (whose operands _bp would otherwise reassign to view temps)
        # actually reuses; native torch broadcasting matches _bp bit-for-bit for
        # scalar⊗tensor. Single-use is guaranteed (fresh temps come only from
        # this method's own results; CSE-shared exprs are Identifiers, excluded).
        if _OUT_REUSE_ENABLED and node.op in _INFIX_OUT_FN:
            target = None
            if left in self._fresh_temps and isinstance(node.right, NumberLiteral):
                target = left
            elif right in self._fresh_temps and isinstance(node.left, NumberLiteral):
                target = right
            if target is not None:
                self._fresh_temps.discard(left)
                self._fresh_temps.discard(right)
                self._emit(f"_torch.{_INFIX_OUT_FN[node.op]}({left}, {right}, out={target})")
                self._fresh_temps.add(target)  # result buffer stays owned+fresh
                return target

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
            elif (lt.is_matrix and r_scalar) or (l_scalar and rt.is_matrix):
                # matrix ⊗ scalar-field: route through _bp so the scalar gets
                # trailing singleton dims ([B,H,W]→[B,H,W,1,1]) and right-aligns
                # against the matrix's [...,N,N] axes, mirroring the interpreter.
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
            self._fresh_temps.difference_update((left, right))
            self._fresh_temps.add(tmp)  # fresh contiguous elementwise result
        elif op == "/":
            # Skip safe-divisor guard when divisor is a non-zero compile-time constant
            if isinstance(node.right, NumberLiteral) and node.right.value != 0:
                self._emit(f"{tmp} = ({left} / {right})")
            else:
                sd = self._emit_safe_divisor(right)
                self._emit(f"{tmp} = ({left} / {sd})")
            self._fresh_temps.difference_update((left, right))
            self._fresh_temps.add(tmp)
        elif op == "%":
            if isinstance(node.right, NumberLiteral) and node.right.value != 0:
                self._emit(f"{tmp} = _torch.fmod({left}, {right})")
            else:
                sd = self._emit_safe_divisor(right)
                self._emit(f"{tmp} = _torch.fmod({left}, {sd})")
        elif op in _CMP_OPS:
            # String operands compare as Python str (e.g. "a"=="b" -> bool), which
            # has no .float(). The interpreter returns a scalar 1.0/0.0; rather than
            # emit a crashing function, fall back cleanly to the interpreter.
            if lt == TEXType.STRING or rt == TEXType.STRING:
                raise _Unsupported("String comparison not supported in codegen")
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

        # LX-5: debug_print is an interpreter-only value probe (a thread-local
        # side-effect). Refuse to codegen it — _Unsupported forces the whole program
        # to the interpreter so the probe ALWAYS records, never silently no-ops.
        if name == "debug_print":
            raise _Unsupported("debug_print is interpreter-only (LX-5)")

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
                # Only the zero-offset case is bit-equivalent to a direct pixel fetch.
                # A nonzero integer offset lands sub-pixel under the interpreter's
                # bilinear sample (align_corners=True), so fall through to the
                # _fns['sample'] call to preserve bilinear semantics.
                # Also require ix/iy to be materialized as locals; otherwise the
                # direct-fetch emitter references undefined names (NameError) and we
                # fall through to the inline grid_sample path (verified equivalent).
                if (dx_expr is True and dy_expr is True
                        and self._direct_fetch_coords_available()):
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
            # asin/acos: mirror interpreter's [-1,1] domain clamp (avoids NaN)
            if name in ("asin", "acos"):
                self._emit(f"{tmp} = _torch.{name}(_torch.clamp({args[0]}, -1.0, 1.0))")
                return tmp
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
            # asin/acos: mirror interpreter's [-1,1] domain clamp (avoids ValueError)
            if name in ("asin", "acos"):
                self._emit(f"{tmp} = _math.{name}(max(-1.0, min(1.0, float({args[0]}))))")
                return tmp
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
                    if v == 3.0:
                        self._emit(f"{tmp} = float({args[0]}) * float({args[0]}) * float({args[0]})")
                        return tmp
                # No clamp: clamping the base to 1e-10 destroyed negative bases
                # (pow(-2,3) -> ~0, diverging from the interpreter/tensor path).
                # math.pow is sign-correct for integer exponents; a negative base
                # with a fractional exponent raises, which execute_compiled catches
                # and falls back to the interpreter (NaN) — matching its semantics.
                self._emit(f"{tmp} = _math.pow(float({args[0]}), float({args[1]}))")
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
                self._emit(f"{tmp} = _math.fmod(float({args[0]}), float({args[1]}) if float({args[1]}) != 0 else _SAFE_EPS)")
                return tmp
        elif len(args) == 3:
            if name == "lerp":
                self._emit(f"{tmp} = float({args[0]}) + (float({args[1]}) - float({args[0]})) * float({args[2]})")
                return tmp
            if name == "clamp":
                # Match torch.clamp's "max wins" behaviour when lo>hi:
                # min(hi, max(lo, x)) returns hi, like torch.clamp, whereas
                # max(lo, min(hi, x)) would return lo and diverge.
                self._emit(f"{tmp} = min(float({args[2]}), max(float({args[1]}), float({args[0]})))")
                return tmp
            if name == "smoothstep":
                edge0, edge1, x = args
                t = self._tmp()
                self._emit(f"{t} = max(0.0, min(1.0, (float({x}) - float({edge0})) / (float({edge1}) - float({edge0}) + _SAFE_EPS)))")
                self._emit(f"{tmp} = {t} * {t} * (3.0 - 2.0 * {t})")
                return tmp
        # Fall through to tensor path for unhandled scalar functions
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
            # Scaled identity. A per-pixel scalar field [B,H,W] must become a
            # per-pixel diagonal [B,H,W,n,n]; append two singleton axes at runtime
            # so it broadcasts against eye's trailing [n,n] (matches interpreter).
            v = self._tmp()
            self._emit(f"{v} = {args[0]}")
            self._emit(f"if _torch.is_tensor({v}) and {v}.dim() >= 3: {v} = {v}.reshape(*{v}.shape, 1, 1)")
            self._emit(f"{tmp} = _torch.eye({n}, dtype=_torch.float32, device=_dev) * {v}")
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
