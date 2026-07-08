"""
STR-7 (cluster 1) — codegen std-function emit handlers.

The `_CodeGen._emit_fn_*` tensor-path specialization handlers, extracted verbatim
from `codegen.py` as a mixin. Every handler uses only `self.*` state, so mixing
`_EmitStdFnsMixin` into `_CodeGen` is behaviour-identical to defining them inline.
The `@_emits` decorator + `_EMIT_DISPATCH` registry live here (co-located with the
handlers they register); `codegen.py` imports the registry to build `_fn_dispatch`.
"""
from ..tex_compiler.ast_nodes import BindingRef, FunctionCall, NumberLiteral


_IMG_REDUCE_OPS = {
    "img_sum": "sum", "img_mean": "mean",
    "img_min": "amin", "img_max": "amax",
}


_EMIT_DISPATCH: dict[str, str] = {}  # STR-6: stdlib name -> _CodeGen handler attr


def _emits(*names):
    """STR-6: register a `_CodeGen._emit_fn_*` handler for one or more stdlib names,
    co-located with the handler. `_fn_dispatch` is built from this registry in
    `__init__`, so the name->handler mapping lives next to the emit code instead of
    in a separate ~40-line dict that could drift from the handlers."""
    def deco(fn):
        for n in names:
            _EMIT_DISPATCH[n] = fn.__name__
        return fn
    return deco


class _EmitStdFnsMixin:
    """The `_emit_fn_*` handlers, mixed into `_CodeGen` (STR-7 cluster 1)."""

    @_emits("pow")
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
            # NOTE: pow(x, 0.5) is deliberately NOT specialized to sqrt(clamp):
            # sqrt(clamp(x,min=0)) returns 0 for negative bases whereas the
            # interpreter's fn_pow returns NaN. Fall through to _torch.pow to
            # preserve codegen<->interpreter equivalence (matches optimizer.py:364).
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
        # General case: mirror the interpreter (torch.pow) so codegen and the
        # interpreter agree on negative bases. The exp-log trick clamps negative
        # bases to ~0 and silently diverges from fn_pow (see stdlib.fn_pow).
        self._emit(f"{tmp} = _torch.pow({args[0]}, {args[1]})")
        return tmp

    @_emits("max", "min")
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
                        if name == "min":
                            lo = inner_node.args[inner_const_idx].value
                            hi = node.args[outer_const_idx].value
                        else:
                            hi = inner_node.args[inner_const_idx].value
                            lo = node.args[outer_const_idx].value
                        # torch.clamp(x, lo, hi) only equals the nested
                        # max(min(x,hi),lo) when lo<=hi. With inverted bounds
                        # clamp returns hi while the nested form returns lo, so
                        # fall through to the plain maximum/minimum composition.
                        if lo > hi:
                            continue
                        inner_arg = self._emit_expr(inner_node.args[inner_val_idx])
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

    @_emits("lerp")
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

    @_emits("luma")
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

    @_emits("sqrt", "log", "log2", "log10", "fract", "isnan", "isinf", "pow2", "pow10", "sincos")
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

    @_emits("dot", "distance", "normalize", "length", "cross", "reflect")
    def _emit_fn_vector(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit vector geometry: dot, distance, normalize, length, cross, reflect."""
        name = node.name
        # length/distance/normalize only reduce over the channel axis for true
        # vectors. The interpreter guards with _has_channel_axis: a scalar field
        # whose last dim is a spatial/width axis is NOT reduced (returns abs/sign).
        # When the static arg type is not a known vector, fall through to the
        # _fns[] stdlib path so the runtime _has_channel_axis guard decides,
        # preserving codegen↔interpreter equivalence.
        if name in ("length", "normalize") and len(args) == 1:
            at = self.type_map.get(id(node.args[0]))
            if at is None or not at.is_vector:
                return None
        if name == "distance" and len(args) == 2:
            at0 = self.type_map.get(id(node.args[0]))
            at1 = self.type_map.get(id(node.args[1]))
            if (at0 is None or not at0.is_vector) and (at1 is None or not at1.is_vector):
                return None
        if name == "dot" and len(args) == 2:
            # Mirror stdlib fn_dot's device split: (a*b).sum(-1) is ~6-10x faster
            # than einsum on CUDA; einsum is kept for CPU. Numerically equivalent.
            self._emit(f"{tmp} = ({args[0]} * {args[1]}).sum(dim=-1) if {args[0]}.is_cuda else _torch.einsum('...c,...c->...', {args[0]}, {args[1]})")
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

    @_emits("smoothstep", "step", "clamp", "fit")
    def _emit_fn_shaping(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit shaping functions: smoothstep, step, clamp, fit."""
        name = node.name
        if name == "clamp" and len(args) == 3:
            lo_const = isinstance(node.args[1], NumberLiteral)
            hi_const = isinstance(node.args[2], NumberLiteral)
            if lo_const and hi_const:
                # Both bounds constant: the scalar torch.clamp overload (one kernel),
                # matching fn_clamp's Python-number fast path.
                self._emit(f"{tmp} = _torch.clamp({args[0]}, "
                           f"{node.args[1].value}, {node.args[2].value})")
            else:
                # Mixed/spatial bounds: torch.clamp rejects a (Tensor, scalar, Tensor)
                # combo, so a program with one tensor bound used to fall back to the
                # interpreter (correct but no codegen). clamp_min().clamp_max() accepts
                # scalar OR tensor bounds and is BIT-IDENTICAL to fn_clamp's spatial
                # torch.minimum(torch.maximum(x, lo), hi) (clamp == min(max(...)), exact).
                lo = node.args[1].value if lo_const else args[1]
                hi = node.args[2].value if hi_const else args[2]
                self._emit(f"{tmp} = {args[0]}.clamp_min({lo}).clamp_max({hi})")
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

    @_emits("spow", "sdiv", "smin", "smax", "mod")
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

    @_emits("sdf_circle", "sdf_box", "sdf_line")
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

    @_emits("img_sum", "img_mean", "img_min", "img_max")
    def _emit_fn_reduce(
        self, node: FunctionCall, args: list[str], tmp: str,
    ) -> str | None:
        """Emit image reductions: img_sum, img_mean, img_min, img_max."""
        if len(args) != 1:
            return None
        op = _IMG_REDUCE_OPS.get(node.name)
        if op is None:
            return None
        # PR-LP4: mirror fn_img_* — .float() before the reduce so an fp16 sum can't
        # overflow to inf; a no-op on fp32 so this stays bit-identical to the interp.
        self._emit(f"{tmp} = {args[0]}.float().{op}(dim=(1, 2), keepdim=True)")
        return tmp

    @_emits("transpose", "determinant", "inverse")
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

    @_emits("sample")
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

    @_emits("fetch")
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
            self._emit(f"{tmp} = {img_var}[_torch.arange({img_var}.shape[0], device=_dev).view(-1,1,1), {py}, {px}]")
            self._indent -= 1
            return tmp
        return None
