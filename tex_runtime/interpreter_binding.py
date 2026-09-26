"""Interpreter binding/variable-write execution — SPLIT-I (v0.44 Phase A1).

Split mechanically out of `interpreter.py`, following the STR-7 codegen split's mixin
pattern: `Interpreter` still inherits this class, so `self._exec_assignment(...)` etc.
resolve exactly as before — no new call layer, no cross-module lookup added to any
per-pixel path. This module owns variable/array declaration, plain and channel/array-index
assignment (including the in-place-reuse fast path), and the scatter write
(`@OUT[px, py] = value`, including the LANG-L4 masked-write compaction).

`InterpreterError` and the spatial/index helpers `_ensure_spatial` / `_safe_array_index` /
`_const_index` / `_host_index` are `interpreter.py` module-level names defined after the
`Interpreter` class, so they are imported back lazily (inside the methods that need them)
rather than at module load time — the same deferred-import shape `masked_flow.py` already
uses for its own back-references into `interpreter.py`.

No behaviour changed by this move: every body below is byte-identical to the code it replaced
in `interpreter.py`.
"""
from __future__ import annotations

import torch

from ..tex_compiler.ast_nodes import (
    ArrayDecl, ArrayIndexAccess, ArrayLiteral, Assignment, BinOp, BindingIndexAccess,
    BindingRef, ChannelAccess, Identifier, VarDecl,
)
from ..tex_compiler.types import CHANNEL_MAP, TEXType, TYPE_NAME_MAP, base_is_vector
from .masked_flow import scatter_keep as _masked_flow_scatter_keep
# PHASEC-OBSROUTE follow-up: `InterpreterError` used to sit in a top-level
# `from .interpreter import InterpreterError` here, contradicting this docstring's own
# claim (above) that it is deferred — and, like R1's tex_engine_tiers fix, importing THIS
# module first in a fresh process ImportErrored the same way. Deferred into each of the
# four methods that raise it, below, alongside the spatial/index helpers that were
# already correctly deferred.
from .stdlib import SAFE_EPSILON, VEC_CHANNELS, ZERO_GUARD_EPS, _get_flat_batch_index


class _BindingExecMixin:
    """Mixin supplying `Interpreter`'s variable/binding write methods (SPLIT-I).

    Composed onto `Interpreter` alongside `MaskedFlowMixin` / `_SpatialContextMixin` /
    `_ControlFlowMixin` — every attribute referenced below (`self.env`, `self.bindings`,
    `self._inplace_ready`, `self._var_widths`, ...) is set on the instance by
    `Interpreter.__init__`.
    """

    def _exec_var_decl(self, node: VarDecl):
        if node.initializer:
            value = self._eval(node.initializer)
        else:
            # Default initialize based on the DECLARED type name first. type_map
            # is keyed by id(node), so it misses for VarDecls cloned by the
            # optimizer's loop-unroller — which would fall back to FLOAT and turn
            # `vec4 tmp;` into a scalar 0.0 (then crash on a later swizzle write).
            # node.type_name survives cloning, so resolve from it first.
            declared = TYPE_NAME_MAP.get(node.type_name) or self.type_map.get(id(node), TEXType.FLOAT)
            value = self._default_value(declared)
        # Record the declared vec2/vec3 width and coerce the value to it, so later
        # reassignment with a wider value is truncated to the declared channels.
        declared_t = TYPE_NAME_MAP.get(node.type_name)
        if declared_t is not None and declared_t.is_vector and declared_t.channels < 4:
            self._var_widths[node.name] = declared_t.channels
            value = self._coerce_vec_width(value, declared_t.channels)
        else:
            self._var_widths.pop(node.name, None)
        self.env[node.name] = value
        # New declaration invalidates in-place readiness (value may be aliased)
        self._inplace_ready.discard(node.name)
        # Any variable the initializer may alias by VIEW (bare `y=x`, swizzle/
        # channel read `y=x.rgb`/`y=x.r`, ternary passthrough, array-index) must
        # lose in-place readiness, so a later in-place op on it clones first
        # instead of mutating this declaration's shared buffer.
        if node.initializer is not None:
            for _name in self._aliased_vars(node.initializer):
                self._inplace_ready.discard(_name)

    def _exec_array_decl(self, node: ArrayDecl):
        """Execute an array declaration."""
        from .interpreter import _ensure_spatial
        is_vec = TYPE_NAME_MAP.get(node.element_type_name, TEXType.FLOAT).is_vector
        is_string = node.element_type_name == "string"

        # -- String arrays: Python list, not tensor -----------------------
        if is_string:
            if node.initializer:
                if isinstance(node.initializer, ArrayLiteral):
                    value = [self._eval(elem) for elem in node.initializer.elements]
                else:
                    src = self._eval(node.initializer)
                    value = list(src) if isinstance(src, list) else [str(src)]
                size = len(value)
            else:
                size = node.size
                value = [""] * size
            self.env[node.name] = value
            self._array_meta[node.name] = size
            return

        # -- Tensor arrays (float, int, vec3, vec4) -----------------------
        if node.initializer:
            if isinstance(node.initializer, ArrayLiteral):
                elements = [self._eval(elem) for elem in node.initializer.elements]
                size = len(elements)
                spatial = self.spatial_shape
                if spatial:
                    expanded = [_ensure_spatial(e, spatial) for e in elements]
                else:
                    expanded = [e if isinstance(e, torch.Tensor) else torch.scalar_tensor(float(e), dtype=self._dtype, device=self.device) for e in elements]
                # For vec3/vec4 arrays, promote elements to consistent channel count
                if is_vec:
                    channels = TYPE_NAME_MAP[node.element_type_name].channels
                    expanded = [self._promote_to_channels(e, channels) for e in expanded]
                value = torch.stack(expanded, dim=-1)
                # For vec arrays, stacking puts channels before N: [..., C, N]
                # Transpose to [..., N, C]
                if is_vec:
                    value = value.transpose(-2, -1)
            else:
                # Array copy from another variable
                value = self._eval(node.initializer).clone()
                size = value.shape[-2] if is_vec else value.shape[-1]
        else:
            # Zero-initialized array
            size = node.size
            if is_vec:
                channels = TYPE_NAME_MAP[node.element_type_name].channels
                if self.spatial_shape:
                    B, H, W = self.spatial_shape
                    value = torch.zeros(B, H, W, size, channels, dtype=self._dtype, device=self.device)
                else:
                    value = torch.zeros(size, channels, dtype=self._dtype, device=self.device)
            else:
                if self.spatial_shape:
                    B, H, W = self.spatial_shape
                    value = torch.zeros(B, H, W, size, dtype=self._dtype, device=self.device)
                else:
                    value = torch.zeros(size, dtype=self._dtype, device=self.device)

        self.env[node.name] = value
        self._array_meta[node.name] = size

    def _promote_to_channels(self, tensor: torch.Tensor, channels: int) -> torch.Tensor:
        """Ensure a tensor has the correct number of channels for a vec array element."""
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.scalar_tensor(float(tensor), dtype=self._dtype, device=self.device)
        if tensor.dim() >= 1 and tensor.shape[-1] in VEC_CHANNELS:
            c = tensor.shape[-1]
            if c == channels:
                return tensor
            if c < channels:
                # Pad with 1.0 (alpha) to reach target channels
                pad = torch.ones(*tensor.shape[:-1], channels - c, dtype=tensor.dtype, device=tensor.device)
                return torch.cat([tensor, pad], dim=-1)
            # Truncate extra channels
            return tensor[..., :channels]
        # Scalar — broadcast to vec
        return tensor.unsqueeze(-1).expand(*tensor.shape, channels)

    # In-place operation mapping: op -> (unbound tensor method, is_commutative)
    _INPLACE_OPS = {
        "+": (torch.Tensor.add_, True),
        "-": (torch.Tensor.sub_, False),
        "*": (torch.Tensor.mul_, True),
        "/": (torch.Tensor.div_, False),  # kept for the "/" key; division uses its own guarded branch
    }

    def _ensure_inplace_ready(self, name: str) -> torch.Tensor:
        """Ensure a variable is safe for in-place mutation (clone-on-first-write)."""
        current = self.env[name]
        if name not in self._inplace_ready:
            current = current.clone()
            self.env[name] = current
            self._inplace_ready.add(name)
        return current

    def _exec_assignment(self, node: Assignment):
        from .interpreter import InterpreterError
        target = node.target

        # In-place optimization: x = x OP expr or x = expr OP x
        # Reuses x's memory instead of allocating a new tensor
        if isinstance(target, Identifier) and isinstance(node.value, BinOp):
            rhs = node.value
            op_info = self._INPLACE_OPS.get(rhs.op)
            if op_info is not None:
                method, commutative = op_info
                name = target.name
                is_div = rhs.op == "/"
                # Pattern 1: x = x OP expr
                if isinstance(rhs.left, Identifier) and rhs.left.name == name:
                    current = self.env.get(name)
                    if current is not None and current.__class__ is torch.Tensor:
                        other_val = self._eval(rhs.right)
                        if other_val.__class__ is torch.Tensor and (
                            other_val.dim() == 0 or other_val.shape == current.shape
                        ):
                            current = self._ensure_inplace_ready(name)
                            if is_div:
                                # Safe in-place division: x /= where(rhs==0, eps, rhs)
                                eps = ZERO_GUARD_EPS.get(other_val.dtype, SAFE_EPSILON)
                                current.div_(torch.where(other_val == 0, eps, other_val))
                            else:
                                method(current, other_val)
                            return
                # Pattern 2: x = expr OP x (only for commutative ops)
                elif commutative and isinstance(rhs.right, Identifier) and rhs.right.name == name:
                    current = self.env.get(name)
                    if current is not None and current.__class__ is torch.Tensor:
                        other_val = self._eval(rhs.left)
                        if other_val.__class__ is torch.Tensor and (
                            other_val.dim() == 0 or other_val.shape == current.shape
                        ):
                            current = self._ensure_inplace_ready(name)
                            method(current, other_val)
                            return

        value = self._eval(node.value)

        if isinstance(target, Identifier):
            # Enforce the declared vec2/vec3 width on reassignment so the variable
            # doesn't silently widen (e.g. `vec3 sum; sum += sample()`), matching
            # codegen's _emit_vec_coerce.
            _w = self._var_widths.get(target.name)
            if _w is not None:
                value = self._coerce_vec_width(value, _w)
            self.env[target.name] = value
            # Invalidate in-place readiness for the target (rebound to a possibly
            # aliasing value) and for every variable the value may alias by view
            # (bare `y=x`, `y=x.rgb`/`y=x.r`, ternary passthrough, array-index),
            # so a later in-place op on the source clones instead of corrupting
            # this alias.
            self._inplace_ready.discard(target.name)
            for _name in self._aliased_vars(node.value):
                self._inplace_ready.discard(_name)

        elif isinstance(target, BindingRef):
            self.bindings[target.name] = value
            # The binding now shares storage with anything the value aliases:
            # this run no longer owns the buffer for scatter writes, and any
            # aliased variable must lose in-place readiness so a later write to
            # it can't mutate the stored output.
            self._scatter_owned.discard(target.name)
            for _name in self._aliased_vars(node.value):
                self._inplace_ready.discard(_name)

        elif isinstance(target, ChannelAccess):
            self._exec_channel_assign(target, value, node.value)

        elif isinstance(target, ArrayIndexAccess):
            self._exec_array_index_assign(target, value, node.value)

        elif isinstance(target, BindingIndexAccess):
            self._exec_scatter_write(target, value, op=node.op)

        else:
            raise InterpreterError(
                "This assignment target is not supported",
                node.loc, source=self._source, code="E6003",
                hint="Assignments work with variables, @bindings, channels (.r), array indices ([i]), and scatter writes (@OUT[x,y]).",
            )

    def _can_write_inplace(self, name: str | None, base, value, rhs_node) -> bool:
        """True when an indexed write may mutate `base` directly instead of
        cloning first: the buffer was cloned by this run (in-place ready), the
        RHS can't alias the target by AST shape, and the evaluated value doesn't
        share the target's storage (catches views the AST guard can't see, e.g.
        a user function returning its argument)."""
        return (
            name is not None
            and name in self._inplace_ready
            and (rhs_node is None or name not in self._aliased_vars(rhs_node))
            and not (value.__class__ is torch.Tensor
                     and base.__class__ is torch.Tensor
                     and value.untyped_storage().data_ptr() == base.untyped_storage().data_ptr())
        )

    def _exec_channel_assign(self, target: ChannelAccess, value: torch.Tensor, rhs_node=None):
        """Handle assignment to a channel: `@A.r = expr;` or `color.rgb = expr;`"""
        from .interpreter import _ensure_spatial, InterpreterError
        base = self._eval(target.object)
        channels = target.channels
        obj = target.object
        # Copy-on-first-write: only bare variables participate (bindings hold
        # caller-owned tensors and share the flat name space with variables).
        name = obj.name if obj.__class__ is Identifier else None
        inplace = self._can_write_inplace(name, base, value, rhs_node)

        if len(channels) == 1:
            idx = CHANNEL_MAP.get(channels)
            if idx is None:
                raise InterpreterError(
                    f"Expected a known channel name, but found '.{channels}'",
                    target.loc, source=self._source, code="E6004",
                    hint="Use one of: .r, .g, .b, .a (or .x, .y, .z, .w).",
                )
            # Spatial-scalar base guard (mirror of the read side): gate on the STATIC TYPE so
            # interp and codegen agree — a channel-less scalar/mask makes `m.r = v` mean `m = v`
            # (replace it), but a VECTOR-typed base (even one channel-less at runtime, e.g. a
            # `vec3 cc = @mask` local) writes a channel exactly as codegen does. `.g/.b/.a` on a
            # scalar are compile-rejected (E3301), so idx is 0; the >0 branch is defensive.
            # Rank test FIRST (cheap), then the shared tier-agreement predicate — mirrors the
            # read side exactly, including falling through to ONE shared raise.
            sp = self.spatial_shape
            result = None
            if (sp is not None and base.dim() == len(sp)
                    and not base_is_vector(self.type_map, target.object)):
                if idx == 0:
                    # `m.r = v` on a channel-less scalar means `m = v` — but it MUST own its
                    # buffer, exactly like the clone in the sibling branch below. `_ensure_spatial`
                    # hands back the RHS tensor ITSELF when its dims already match sp (and a
                    # stride-0 `.expand()` view for a 0-dim value), while the write-back below
                    # claims ownership via `_inplace_ready.add(name)`. Storing an alias there lets
                    # a later in-place op scribble on another variable — or on a CALLER-OWNED
                    # input tensor, which under ComfyUI is the wire shared with every consumer —
                    # and makes an expanded view raise "more than one element ... single memory
                    # location". Clone: correctness first, and it costs what every other
                    # channel-assign already pays.
                    result = _ensure_spatial(value, sp).clone()
                nchan = 1
            elif base.dim() >= 1 and base.shape[-1] > idx:
                result = base if inplace else base.clone()
                result[..., idx] = _ensure_spatial(value, result.shape[:-1])
            else:
                nchan = base.shape[-1] if base.dim() >= 1 else 1
            if result is None:
                raise InterpreterError(
                    f"This value has {nchan} channel{'s' if nchan != 1 else ''}, so it has no "
                    f"channel #{idx + 1} to write to.",
                    target.loc, source=self._source, code="E6004",
                    hint="A vec3 has 3 channels (r, g, b); build a vec4 (e.g. vec4(color, 1.0)) if you need a 4th (alpha).",
                )
        else:
            # Multi-channel assignment: .rgb, .xy, .rgba, etc.
            indices = [CHANNEL_MAP[ch] for ch in channels]
            result = base if inplace else base.clone()
            val = _ensure_spatial(value, result.shape[:-1]) if self.spatial_shape else value
            val_is_multi = isinstance(val, torch.Tensor) and val.dim() >= 1 and val.shape[-1] > 1
            for i, idx in enumerate(indices):
                if result.dim() >= 1 and result.shape[-1] > idx:
                    result[..., idx] = val[..., i] if val_is_multi else val

        # Write back to the correct location
        if name is not None:
            self.env[name] = result
            # result is either the already-owned buffer or a fresh clone —
            # subsequent writes to this variable may skip the clone.
            self._inplace_ready.add(name)
        elif isinstance(obj, BindingRef):
            self.bindings[obj.name] = result
            self._scatter_owned.discard(obj.name)
        else:
            raise InterpreterError(
                "This channel assignment needs a variable or @binding as its target",
                target.loc, source=self._source, code="E6004",
                hint="Assign to a named variable's channel, e.g. 'color.r = 1.0;' or '@OUT.r = 1.0;'.",
            )

    def _exec_array_index_assign(self, target: ArrayIndexAccess, value, rhs_node=None):
        """Handle: arr[i] = expr;"""
        from .interpreter import (_ensure_spatial, _safe_array_index, _const_index,
                                  _host_index, InterpreterError)
        array = self._eval(target.array)
        index = self._eval(target.index)

        # String array (Python list)
        if isinstance(array, list):
            idx_int = max(0, min(int(round(index.item() if isinstance(index, torch.Tensor) else float(index))), len(array) - 1))
            result = list(array)
            result[idx_int] = value if isinstance(value, str) else str(value)
            if isinstance(target.array, Identifier):
                self.env[target.array.name] = result
            return

        # Copy-on-first-write: clone only the first write to a variable-held
        # array; later writes (e.g. a fill loop) mutate the owned buffer.
        tgt = target.array
        name = tgt.name if tgt.__class__ is Identifier else None
        inplace = self._can_write_inplace(name, array, value, rhs_node)

        # Vector array: dim 5 (spatial) or 2 (non-spatial) → [..., N, C]
        if array.dim() in (2, 5):
            arr_size = array.shape[-2]
            idx = _safe_array_index(index, arr_size)
            result = array if inplace else array.clone()

            if result.dim() == 2:
                # Non-spatial vector array [N, C]
                val_t = value if isinstance(value, torch.Tensor) else torch.scalar_tensor(float(value), dtype=self._dtype, device=self.device)
                result[idx] = val_t
            elif idx.dim() == 0:
                # Constant index: [..., N, C] → assign vec to [..., C]. Literal
                # index resolves without .item() (UC-5); runtime scalar syncs.
                spatial_shape = result.shape[:-2]  # [B, H, W]
                ci = _const_index(target.index, arr_size)
                if ci is None:
                    ci = _host_index(index, arr_size)
                if ci is None:
                    ci = int(idx.item())
                result[..., ci, :] = _ensure_spatial(value, spatial_shape)
            else:
                # Per-pixel assignment via scatter on dim=-2
                C = result.shape[-1]
                idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, C)
                if idx_exp.shape[:3] != result.shape[:3]:
                    idx_exp = idx_exp.expand(result.shape[:3] + (1, C))
                val_spatial = _ensure_spatial(value, result.shape[:-2])
                val_exp = val_spatial.unsqueeze(-2)
                result.scatter_(-2, idx_exp, val_exp)

        else:
            # Scalar array: dim 4 (spatial) or 1 (non-spatial) → [..., N]
            arr_size = array.shape[-1]
            idx = _safe_array_index(index, arr_size)
            result = array if inplace else array.clone()

            if result.dim() == 1:
                result[idx] = value if isinstance(value, torch.Tensor) else torch.scalar_tensor(float(value), dtype=self._dtype, device=self.device)
            elif idx.dim() == 0:
                spatial_shape = result.shape[:-1]
                ci = _const_index(target.index, arr_size)
                if ci is None:
                    ci = _host_index(index, arr_size)
                if ci is None:
                    ci = int(idx.item())
                result[..., ci] = _ensure_spatial(value, spatial_shape)
            else:
                idx_expanded = idx.unsqueeze(-1)
                if idx_expanded.shape[:3] != result.shape[:3]:
                    idx_expanded = idx_expanded.expand(result.shape[:3] + (1,))
                val_spatial = _ensure_spatial(value, result.shape[:-1])
                val_expanded = val_spatial.unsqueeze(-1)
                result.scatter_(-1, idx_expanded, val_expanded)

        # Write back to the correct location
        if name is not None:
            self.env[name] = result
            # result is either the already-owned buffer or a fresh clone —
            # subsequent writes to this array may skip the clone.
            self._inplace_ready.add(name)
        elif isinstance(tgt, BindingRef):
            self.bindings[tgt.name] = result
            self._scatter_owned.discard(tgt.name)
        else:
            raise InterpreterError(
                "This array index assignment needs a variable or @binding as its target",
                target.loc, source=self._source, code="E6005",
                hint="Assign to a named array element, e.g. 'arr[i] = value;'.",
            )

    def _exec_scatter_write(self, target: BindingIndexAccess, value, op=None, live=None):
        """Handle @OUT[px, py] = value or @OUT[px, py] += value (scatter write).

        LANG-L4 (M5): `live` is the masked path's per-pixel live mask, and it gates the
        write **by SOURCE** — a source pixel contributes iff it is live on the path to
        this statement. `0.23` gates by destination, which has no per-pixel meaning once a
        transfer can leave a branch. `None` (every caller below `0.25`) is the unmasked
        write, byte-identical to before."""
        from .interpreter import _ensure_spatial, InterpreterError
        name = target.binding.name
        args = [self._eval(a) for a in target.args]
        px, py = args[0], args[1]
        frame = args[2] if len(args) == 3 else None

        # Determine channel count from value
        if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[-1] in VEC_CHANNELS:
            C = value.shape[-1]
        else:
            C = 1

        # Get or create output buffer — must be at least [B, H, W] for scatter indexing
        buf = self.bindings.get(name)
        if self.spatial_shape:
            B, H, W = self.spatial_shape
        else:
            B, H, W = 1, 1, 1

        needs_new_buf = (buf is None or not isinstance(buf, torch.Tensor) or buf.dim() < 3)
        if needs_new_buf:
            # Existing buffer too small — create spatial buffer, preserving old value if possible
            if C > 1:
                new_buf = torch.zeros(B, H, W, C, dtype=self._dtype, device=self.device)
            else:
                new_buf = torch.zeros(B, H, W, dtype=self._dtype, device=self.device)
            if isinstance(buf, torch.Tensor):
                new_buf[...] = buf
            buf = new_buf
            self.bindings[name] = buf
        elif name not in self._scatter_owned:
            # First scatter into a buffer this run didn't allocate: clone before
            # writing. The stored tensor may be (a view of) caller-owned storage —
            # an input binding, a cached literal, or a builtin grid (`@OUT = u;`) —
            # and an in-place index write would corrupt it for every other reader.
            # Cloning also materializes expanded views so index_put_ works.
            buf = buf.clone()
            self.bindings[name] = buf
        self._scatter_owned.add(name)

        # Clamp coordinates
        ix_t = torch.clamp(torch.floor(_ensure_spatial(px, (B, H, W)) if self.spatial_shape else px).long(), 0, W - 1)
        iy_t = torch.clamp(torch.floor(_ensure_spatial(py, (B, H, W)) if self.spatial_shape else py).long(), 0, H - 1)
        val = _ensure_spatial(value, (B, H, W)) if self.spatial_shape else value

        # Build flat indices — the batch index is data-independent, so its
        # flattened form comes from the shared sampler cache.
        if frame is not None:
            batch_idx = torch.clamp(
                (_ensure_spatial(frame, (B, H, W)) if self.spatial_shape else frame).long(),
                0, B - 1
            )
            flat_b = batch_idx.contiguous().reshape(-1)
        else:
            flat_b = _get_flat_batch_index(B, H, W, self.device)

        flat_y = iy_t.contiguous().reshape(-1)
        flat_x = ix_t.contiguous().reshape(-1)

        if C > 1 and buf.dim() == 4:
            flat_v = val.contiguous().reshape(-1, C)
        else:
            flat_v = val.reshape(-1) if isinstance(val, torch.Tensor) and val.dim() > 0 else val

        # Channel-count check: a clear message beats a raw torch shape error.
        buf_c = buf.shape[-1] if buf.dim() == 4 else 1
        if C != buf_c:
            if buf_c == 1:
                raise InterpreterError(
                    f"You're writing a {C}-channel color into '@{name}', but '@{name}' is a "
                    f"mask — it holds one value per pixel, not a color.",
                    target.loc, source=self._source, code="E6006",
                    hint="Write a single number into a mask (e.g. @M[x,y] = 0.5), or send "
                         "colors to an image output instead.")
            raise InterpreterError(
                f"You're writing a {C}-channel value into '@{name}', but '@{name}' holds "
                f"{buf_c} channels per pixel.",
                target.loc, source=self._source, code="E6006",
                hint=f"Match the channel count — use a vec{buf_c} value, or .rgb / .r to convert.")

        # M5: compact the SOURCES down to the live ones, in row-major order (the order
        # `docs/masked-control-flow.md` §5's divergence site 4 names, so an unspecified
        # collision resolves the same way on both tiers). Boolean indexing over the
        # already-flattened row-major arrays is that order by construction.
        if live is not None and live is not True:
            # LANG-L5: the compaction itself lives in `masked_flow.scatter_keep`, so the
            # codegen tier's emitted scatter selects the same sources in the same order
            # rather than in an equivalent one.
            keep = _masked_flow_scatter_keep(live, B, H, W)
            if keep is False:
                return
            flat_b = flat_b[keep]
            flat_y = flat_y[keep]
            flat_x = flat_x[keep]
            if isinstance(flat_v, torch.Tensor) and flat_v.dim() > 0:
                flat_v = flat_v[keep]

        idx = (flat_b, flat_y, flat_x)
        if op is None:
            buf[idx] = flat_v
        elif op == "+":
            buf.index_put_(idx, flat_v, accumulate=True)
        elif op == "-":
            buf.index_put_(idx, -flat_v, accumulate=True)
        elif op == "*":
            buf[idx] *= flat_v
        elif op == "/":
            eps = ZERO_GUARD_EPS.get(flat_v.dtype, SAFE_EPSILON) if isinstance(flat_v, torch.Tensor) else SAFE_EPSILON
            buf[idx] /= torch.where(flat_v == 0, eps, flat_v)

