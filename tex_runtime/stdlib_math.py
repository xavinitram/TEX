"""
TEX Standard Library — scalar, matrix, clamping/interpolation and vector builtins (one domain leaf of `stdlib.py`).

The `TEXStdlib` class-body section(s) Math functions, Matrix operations, Clamping / interpolation, Vector operations moved here verbatim, onto the `_StdlibMath`
mixin. `stdlib.py` composes the leaves' mixins into `TEXStdlib` in the class body's original
section order, which is the REG-1 registration order (`help_entries()`, the generated
reference and the help panel all read it) — so import this leaf THROUGH `stdlib`, not
directly, unless registering only this domain is what you want.
"""
from __future__ import annotations
import torch
from . import guard_trace  # C4-ux: guarded-division near-singularity trace (leaf, no cycle)
from .stdlib_registry import stdlib
from .stdlib_core import (
    SAFE_EPSILON,
    _POW_NAN_STATE,
    _has_channel_axis,
    _is_scalar,
    _lerp_f32,
    _to_float,
    _to_tensor,
)
# ZERO_GUARD_EPS / _texlog are bound by attribute lookup, not folded into the `from` import
# above: a name bound by `from X import name` compiles a later `name.method(...)` call site
# WITHOUT CPython's LOAD_ATTR+PUSH_NULL fusion, while a name bound by a plain assignment
# (even one whose RHS is an attribute lookup) keeps it — a compile-time instruction-selection
# quirk this split's G1 bytecode-identity gate caught (`fn_pow`'s `_texlog.warning(...)`,
# `fn_mod`'s `ZERO_GUARD_EPS.get(...)`), not a runtime difference; both bind the SAME object
# either way. See docs/worklog/lib-1.
from . import stdlib_core as _stdlib_core
ZERO_GUARD_EPS = _stdlib_core.ZERO_GUARD_EPS
_texlog = _stdlib_core._texlog


class _StdlibMath:
    """scalar, matrix, clamping/interpolation and vector builtins: the `fn_*` methods `stdlib.py` mixes into `TEXStdlib`."""

    # -- Math functions -------------------------------------------------

    @stdlib("sin", sig='sin(x) \\u2192 float', category='Math', doc='Sine (radians).', ex='float s = sin(u * PI * 2.0);')
    @staticmethod
    def fn_sin(x) -> torch.Tensor:
        return torch.sin(_to_tensor(x))

    @stdlib("cos", sig='cos(x) \\u2192 float', category='Math', doc='Cosine (radians).', ex='float c = cos(v * PI);')
    @staticmethod
    def fn_cos(x) -> torch.Tensor:
        return torch.cos(_to_tensor(x))

    @stdlib("tan", sig='tan(x) \\u2192 float', category='Math', doc='Tangent (radians).', ex='float t = tan(u);')
    @staticmethod
    def fn_tan(x) -> torch.Tensor:
        return torch.tan(_to_tensor(x))

    @stdlib("asin", sig='asin(x) \\u2192 float', category='Math', doc='Arcsine. Returns radians.', ex='float angle = asin(0.5);')
    @staticmethod
    def fn_asin(x) -> torch.Tensor:
        return torch.asin(torch.clamp(_to_tensor(x), -1.0, 1.0))

    @stdlib("acos", sig='acos(x) \\u2192 float', category='Math', doc='Arccosine. Returns radians.', ex='float angle = acos(0.5);')
    @staticmethod
    def fn_acos(x) -> torch.Tensor:
        return torch.acos(torch.clamp(_to_tensor(x), -1.0, 1.0))

    @stdlib("atan", sig='atan(x) \\u2192 float', category='Math', doc='Arctangent. Returns radians.', ex='float angle = atan(1.0);')
    @staticmethod
    def fn_atan(x) -> torch.Tensor:
        return torch.atan(_to_tensor(x))

    @stdlib("atan2", sig='atan2(y, x) \\u2192 float', category='Math', doc='Two-argument arctangent. Returns radians.', ex='float angle = atan2(v - 0.5, u - 0.5);')
    @staticmethod
    def fn_atan2(y, x) -> torch.Tensor:
        return torch.atan2(_to_tensor(y), _to_tensor(x))

    @stdlib("sincos", sig='sincos(x) \\u2192 vec2', category='Math', doc='Returns vec2(sin(x), cos(x)). More efficient than separate sin/cos calls.', ex='vec2 sc = sincos(angle);\nfloat s = sc.x;\nfloat c = sc.y;')
    @staticmethod
    def fn_sincos(x) -> torch.Tensor:
        """Returns vec2(sin(x), cos(x)) — computes both in a single pass."""
        t = _to_tensor(x)
        return torch.stack([torch.sin(t), torch.cos(t)], dim=-1)

    @stdlib("sqrt", sig='sqrt(x) \\u2192 float', category='Math', doc='Square root.', ex='float s = sqrt(u * u + v * v);')
    @staticmethod
    def fn_sqrt(x) -> torch.Tensor:
        return torch.sqrt(torch.clamp(_to_tensor(x), min=0.0))

    @stdlib("pow", sig='pow(x, y) \\u2192 float', category='Math', doc='Raise x to the power y.', ex='float p = pow(u, 2.2);')
    @staticmethod
    def fn_pow(base, exp) -> torch.Tensor:
        b = _to_tensor(base)
        e = _to_tensor(exp)
        # An exp-log fast path (exp(log(b)*e)) is faster for spatial tensors but
        # silently destroys the sign of negative bases: pow(x, 2) on a signed /
        # centered coordinate (vignettes, radial gradients, SDFs) would return
        # ~0 instead of x*x. torch.pow is correct for negative bases with whole
        # exponents and matches the scalar path, so results are path-independent.
        out = torch.pow(b, e)
        # Make silent NaN visible: pow(-2, 0.5) and similar have no real answer.
        # Bounded so this never adds a per-call sync to steady-state execution.
        # CUDA-only exception: the .any() bool sync is a verified CUDA-graph
        # CAPTURE BLOCKER for the first 32 pow calls process-wide (capture failed
        # with 9 warm pow calls, succeeded with 36). The diagnostic is a
        # best-effort nicety, so skip it on CUDA and keep it CPU-only.
        _st = _POW_NAN_STATE
        if not _st["warned"] and _st["checked"] < 32 and not out.is_cuda:
            _st["checked"] += 1
            if torch.isnan(out).any() and not torch.isnan(b).any():
                _st["warned"] = True
                _texlog.warning(
                    "[TEX] pow() produced NaN: a negative base with a fractional exponent "
                    "(e.g. pow(-2, 0.5)) has no real answer, so those pixels are NaN. Use "
                    "spow(x, y) for a sign-preserving power, or pow(abs(x), y) for magnitude."
                )
        return out

    @stdlib("exp", sig='exp(x) \\u2192 float', category='Math', doc='e raised to the power x.', ex='float e = exp(-u * 5.0);')
    @staticmethod
    def fn_exp(x) -> torch.Tensor:
        return torch.exp(_to_tensor(x))

    @stdlib("log", sig='log(x) \\u2192 float', category='Math', doc='Natural logarithm (base e).', ex='float l = log(u + 1.0);')
    @staticmethod
    def fn_log(x) -> torch.Tensor:
        return torch.log(torch.clamp(_to_tensor(x), min=SAFE_EPSILON))

    @stdlib("abs", sig='abs(x) \\u2192 float', category='Math', doc='Absolute value.', ex='float a = abs(u - 0.5);')
    @staticmethod
    def fn_abs(x) -> torch.Tensor:
        return torch.abs(_to_tensor(x))

    @stdlib("sign", sig='sign(x) \\u2192 float', category='Math', doc='Returns -1, 0, or 1.', ex='float s = sign(u - 0.5);')
    @staticmethod
    def fn_sign(x) -> torch.Tensor:
        return torch.sign(_to_tensor(x))

    @stdlib("floor", sig='floor(x) \\u2192 float', category='Math', doc='Round down to nearest integer.', ex='float f = floor(u * 10.0);')
    @staticmethod
    def fn_floor(x) -> torch.Tensor:
        return torch.floor(_to_tensor(x))

    @stdlib("ceil", sig='ceil(x) \\u2192 float', category='Math', doc='Round up to nearest integer.', ex='float c = ceil(u * 10.0);')
    @staticmethod
    def fn_ceil(x) -> torch.Tensor:
        return torch.ceil(_to_tensor(x))

    @stdlib("round", sig='round(x) \\u2192 float', category='Math', doc='Round to nearest integer.', ex='float r = round(u * 10.0) / 10.0;')
    @staticmethod
    def fn_round(x) -> torch.Tensor:
        return torch.round(_to_tensor(x))

    @stdlib("trunc", sig='trunc(x) \\u2192 float', category='Math', doc='Truncate toward zero (drop fractional part).', ex='float t = trunc(u * 10.0);')
    @staticmethod
    def fn_trunc(x) -> torch.Tensor:
        return torch.trunc(_to_tensor(x))

    @stdlib("fract", sig='fract(x) \\u2192 float', category='Math', doc='Fractional part: x - floor(x).', ex='float f = fract(u * 5.0);')
    @staticmethod
    def fn_fract(x) -> torch.Tensor:
        t = _to_tensor(x)
        return t - torch.floor(t)

    @stdlib("mod", sig='mod(x, y) \\u2192 float', category='Math', doc='Modulo (remainder).', ex='float m = mod(u * 10.0, 1.0);')
    @staticmethod
    def fn_mod(a, b) -> torch.Tensor:
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        zero = b_t == 0
        guard_trace.note(zero)  # C4-ux (no-op unless armed)
        safe_b = torch.where(zero, ZERO_GUARD_EPS.get(b_t.dtype, SAFE_EPSILON), b_t)
        return torch.fmod(a_t, safe_b)

    @stdlib("log2", sig='log2(x) \\u2192 float', category='Math', doc='Logarithm base 2.', ex='float l = log2(256.0);')
    @staticmethod
    def fn_log2(x) -> torch.Tensor:
        return torch.log2(torch.clamp(_to_tensor(x), min=SAFE_EPSILON))

    @stdlib("log10", sig='log10(x) \\u2192 float', category='Math', doc='Logarithm base 10.', ex='float l = log10(1000.0);')
    @staticmethod
    def fn_log10(x) -> torch.Tensor:
        return torch.log10(torch.clamp(_to_tensor(x), min=SAFE_EPSILON))

    @stdlib("pow2", sig='pow2(x) \\u2192 float', category='Math', doc='2 raised to the power x.', ex='float p = pow2(8.0);')
    @staticmethod
    def fn_pow2(x) -> torch.Tensor:
        return torch.pow(2.0, _to_tensor(x))

    @stdlib("pow10", sig='pow10(x) \\u2192 float', category='Math', doc='10 raised to the power x.', ex='float p = pow10(3.0);')
    @staticmethod
    def fn_pow10(x) -> torch.Tensor:
        return torch.pow(10.0, _to_tensor(x))

    @stdlib("sinh", sig='sinh(x) \\u2192 float', category='Math', doc='Hyperbolic sine.', ex='float s = sinh(u);')
    @staticmethod
    def fn_sinh(x) -> torch.Tensor:
        return torch.sinh(_to_tensor(x))

    @stdlib("cosh", sig='cosh(x) \\u2192 float', category='Math', doc='Hyperbolic cosine.', ex='float c = cosh(u);')
    @staticmethod
    def fn_cosh(x) -> torch.Tensor:
        return torch.cosh(_to_tensor(x))

    @stdlib("tanh", sig='tanh(x) \\u2192 float', category='Math', doc='Hyperbolic tangent.', ex='float t = tanh(u * 2.0);')
    @staticmethod
    def fn_tanh(x) -> torch.Tensor:
        return torch.tanh(_to_tensor(x))

    @stdlib("hypot", sig='hypot(x, y) \\u2192 float', category='Math', doc='Hypotenuse: sqrt(x*x + y*y).', ex='float d = hypot(u - 0.5, v - 0.5);')
    @staticmethod
    def fn_hypot(x, y) -> torch.Tensor:
        return torch.hypot(_to_tensor(x), _to_tensor(y))

    @stdlib("isnan", sig='isnan(x) \\u2192 float', category='Math', doc='Returns 1.0 if x is NaN, 0.0 otherwise.', ex='float check = isnan(x);')
    @staticmethod
    def fn_isnan(x) -> torch.Tensor:
        return torch.isnan(_to_tensor(x)).float()

    @stdlib("isinf", sig='isinf(x) \\u2192 float', category='Math', doc='Returns 1.0 if x is infinite, 0.0 otherwise.', ex='float check = isinf(x);')
    @staticmethod
    def fn_isinf(x) -> torch.Tensor:
        return torch.isinf(_to_tensor(x)).float()

    @stdlib("degrees", sig='degrees(x) \\u2192 float', category='Math', doc='Convert radians to degrees.', ex='float d = degrees(PI);')
    @staticmethod
    def fn_degrees(x) -> torch.Tensor:
        return torch.rad2deg(_to_tensor(x))

    @stdlib("radians", sig='radians(x) \\u2192 float', category='Math', doc='Convert degrees to radians.', ex='float r = radians(180.0);')
    @staticmethod
    def fn_radians(x) -> torch.Tensor:
        return torch.deg2rad(_to_tensor(x))

    @stdlib("spow", sig='spow(x, y) \\u2192 float', category='Math', doc='Sign-preserving power. Safe for negative x.', ex='float s = spow(u - 0.5, 2.0);')
    @staticmethod
    def fn_spow(x, y) -> torch.Tensor:
        """Safe power — sign(x) * pow(abs(x), y). Avoids NaN on negative bases."""
        t = _to_tensor(x)
        yt = _to_tensor(y)
        abs_t = torch.abs(t)
        mask = abs_t < SAFE_EPSILON
        safe_abs = torch.clamp(abs_t, min=SAFE_EPSILON)
        return torch.where(mask, torch.zeros_like(t), torch.sign(t) * torch.pow(safe_abs, yt))

    @stdlib("sdiv", sig='sdiv(a, b) \\u2192 float', category='Math', doc='Safe divide. Returns 0 when b is zero.', ex='float d = sdiv(1.0, u);')
    @staticmethod
    def fn_sdiv(a, b) -> torch.Tensor:
        """Safe division — returns 0.0 where abs(b) < SAFE_EPSILON."""
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        mask = torch.abs(b_t) < SAFE_EPSILON
        guard_trace.note(mask)  # C4-ux (no-op unless armed)
        safe_b = torch.where(mask, torch.ones_like(b_t), b_t)
        return torch.where(mask, torch.zeros_like(a_t), a_t / safe_b)

    # -- Matrix operations ----------------------------------------------

    @stdlib("transpose", sig='transpose(m) \\u2192 mat', category='Matrix', doc='Transpose a matrix.', ex='mat3 mt = transpose(m);')
    @staticmethod
    def fn_transpose(m) -> torch.Tensor:
        return m.transpose(-1, -2)

    @stdlib("determinant", sig='determinant(m) \\u2192 float', category='Matrix', doc='Compute the determinant.', ex='float det = determinant(m);')
    @staticmethod
    def fn_determinant(m) -> torch.Tensor:
        return torch.linalg.det(m)

    @stdlib("inverse", sig='inverse(m) \\u2192 mat', category='Matrix', doc='Compute the matrix inverse.', ex='mat3 inv = inverse(m);')
    @staticmethod
    def fn_inverse(m) -> torch.Tensor:
        try:
            return torch.linalg.inv(m)
        except torch.linalg.LinAlgError as e:
            # Only a genuinely singular matrix gets the friendly explanation;
            # other failures (e.g. a CUDA OOM) keep their real cause.
            if "singular" in str(e).lower():
                raise ValueError(
                    "inverse() can't invert this matrix because it's singular "
                    "(its determinant is zero) — usually two rows/columns are identical, "
                    "or one is all zeros. Check determinant(m) first, or rebuild the matrix."
                ) from e
            raise

    # -- Clamping / interpolation ---------------------------------------

    @stdlib("min", sig='min(a, b) \\u2192 float', category='Interpolation', doc='Returns the smaller value.', ex='float m = min(u, 0.5);')
    @staticmethod
    def fn_min(a, b) -> torch.Tensor:
        return torch.minimum(_to_tensor(a), _to_tensor(b))

    @stdlib("max", sig='max(a, b) \\u2192 float', category='Interpolation', doc='Returns the larger value.', ex='float m = max(u, 0.0);')
    @staticmethod
    def fn_max(a, b) -> torch.Tensor:
        return torch.maximum(_to_tensor(a), _to_tensor(b))

    @stdlib("clamp", sig='clamp(x, lo, hi) \\u2192 float', category='Interpolation', doc='Clamp x to [lo, hi] range.', ex='float c = clamp(u * 2.0, 0.0, 1.0);')
    @staticmethod
    def fn_clamp(x, lo, hi) -> torch.Tensor:
        # Python-number bounds: the scalar torch.clamp overload (one kernel).
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return torch.clamp(_to_tensor(x), min=lo, max=hi)
        # 0-dim tensor bounds (interpreter literals live on-device), split by
        # device: on CUDA pass them through as tensor bounds — sync-free (each
        # .item() would flush the launch-bound pipeline); on CPU .item() is
        # nearly free and the scalar overload's kernel is measurably faster
        # than the broadcasting tensor overload.
        if _is_scalar(lo) and _is_scalar(hi):
            xt = _to_tensor(x)
            if xt.is_cuda:
                return torch.clamp(xt, min=_to_tensor(lo), max=_to_tensor(hi))
            return torch.clamp(xt, min=_to_float(lo), max=_to_float(hi))
        # Spatially-varying bounds
        return torch.minimum(torch.maximum(_to_tensor(x), _to_tensor(lo)), _to_tensor(hi))

    @stdlib("lerp", sig='lerp(a, b, t) \\u2192 float', category='Interpolation', aliases=("mix",), doc='Linear interpolation from a to b by t.', ex='@OUT = lerp(@A, @B, 0.5);')
    @staticmethod
    def fn_lerp(a, b, t) -> torch.Tensor:
        a_t, b_t, t_t = _to_tensor(a), _to_tensor(b), _to_tensor(t)
        # Auto-unsqueeze weight for channel broadcast: [B,H,W] weight with [B,H,W,C] values
        if t_t.dim() + 1 == a_t.dim():
            t_t = t_t.unsqueeze(-1)
        return _lerp_f32(a_t, b_t, t_t)

    @stdlib("select", sig='select(cond, a, b) \\u2192 vec', category='Interpolation',
            doc='Pick a or b by cond, without an if. Both a and b are always computed — '
                'nothing is skipped — but this never syncs, so it stays capturable under '
                'CUDA graphs where an equivalent if on a per-pixel or uniform cond may not.',
            ex='@OUT = vec4(select(luma(@A.rgb) > 0.5, @A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_select(cond, a, b) -> torch.Tensor:
        # A non-syncing selector: torch.where under _tensor_where's broadcast (the
        # same merge the interpreter's per-pixel if/else and ?: use for their spatial
        # path), never a Python branch or a `.item()` sync — so a program calling
        # select stays CUDA-graph capturable where a uniform `if`/`?:` (which takes
        # the scalar-shortcut branch via float(cond), a host sync) is not. Local
        # import: interpreter.py imports stdlib.py, not the reverse, so this can only
        # be resolved at call time, after both modules have finished loading.
        from .interpreter import _tensor_where
        cond_t = _to_tensor(cond)
        cond_bool = (cond_t > 0.5) if cond_t.is_floating_point() else cond_t.bool()
        return _tensor_where(cond_bool, _to_tensor(a), _to_tensor(b))

    @stdlib("fit", sig='fit(x, inLo, inHi, outLo, outHi) \\u2192 float', category='Interpolation', doc='Remap x from [inLo, inHi] to [outLo, outHi].', ex='float y = fit(u, 0.2, 0.8, 0.0, 1.0);')
    @staticmethod
    def fn_fit(val, old_min, old_max, new_min, new_max) -> torch.Tensor:
        """Remap val from [old_min, old_max] to [new_min, new_max]."""
        v = _to_tensor(val)
        o_min, o_max = _to_tensor(old_min), _to_tensor(old_max)
        n_min, n_max = _to_tensor(new_min), _to_tensor(new_max)
        t = (v - o_min) / (o_max - o_min + SAFE_EPSILON)
        return _lerp_f32(n_min, n_max, t)

    @stdlib("smoothstep", sig='smoothstep(lo, hi, x) \\u2192 float', category='Interpolation', doc='Smooth Hermite interpolation between lo and hi.', ex='float s = smoothstep(0.3, 0.7, u);')
    @staticmethod
    def fn_smoothstep(edge0, edge1, x) -> torch.Tensor:
        e0, e1, xv = _to_tensor(edge0), _to_tensor(edge1), _to_tensor(x)
        t = torch.clamp((xv - e0) / (e1 - e0 + SAFE_EPSILON), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    @stdlib("step", sig='step(edge, x) \\u2192 float', category='Interpolation', doc='Returns 0 if x < edge, 1 otherwise.', ex='float s = step(0.5, u);')
    @staticmethod
    def fn_step(edge, x) -> torch.Tensor:
        return ((_to_tensor(x)) >= _to_tensor(edge)).float()

    # -- Vector operations ----------------------------------------------

    @stdlib("dot", sig='dot(a, b) \\u2192 float', category='Vector', doc='Dot product of two vectors.', ex='float d = dot(normal, lightDir);')
    @staticmethod
    def fn_dot(a, b) -> torch.Tensor:
        """Dot product over the channel (last) dim.

        einsum is ~4x faster than mul+sum on CPU but ~6-10x SLOWER on CUDA
        (measured), so pick per device. Numerically equivalent (~1e-7)."""
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        if a_t.is_cuda:
            return (a_t * b_t).sum(dim=-1)
        return torch.einsum('...c,...c->...', a_t, b_t)

    @stdlib("length", sig='length(v) \\u2192 float', category='Vector', doc='Length (magnitude) of a vector.', ex='float len = length(vec3(u, v, 0.0));')
    @staticmethod
    def fn_length(v) -> torch.Tensor:
        """Length (magnitude) of a vector, reduced over the channel (last) dim.

        A standard vec (last dim in {2,3,4}) reduces; additionally any 4D
        [B,H,W,C] tensor carries channels last, so C>1 reduces too (covers
        exotic channel counts). Lower-rank tensors (scalar fields / masks) have
        no channel dim and return abs — never reducing a mask's width axis.
        """
        t = _to_tensor(v)
        if _has_channel_axis(t):
            return torch.linalg.vector_norm(t, dim=-1)
        return torch.abs(t)

    @stdlib("distance", sig='distance(a, b) \\u2192 float', category='Vector', doc='Distance between two points.', ex='float d = distance(vec3(u,v,0), vec3(0.5,0.5,0));')
    @staticmethod
    def fn_distance(a, b) -> torch.Tensor:
        diff = _to_tensor(a) - _to_tensor(b)
        if _has_channel_axis(diff):
            return torch.linalg.vector_norm(diff, dim=-1)
        return torch.abs(diff)

    @stdlib("normalize", sig='normalize(v) \\u2192 vec', category='Vector', doc='Unit vector in the same direction.', ex='vec3 dir = normalize(vec3(u-0.5, v-0.5, 1.0));')
    @staticmethod
    def fn_normalize(v) -> torch.Tensor:
        # Pinned convention (the single oracle both backends use): a vector
        # (channel axis present) normalizes to unit length; a scalar / 1-channel
        # field normalizes to sign() — so sign(0)=0, and NOT x/abs(x) which is
        # NaN at 0. The type checker rejects normalize() on a non-vector
        # (E5003), and codegen routes scalar args here too, so interpreter and
        # codegen never diverge.
        t = _to_tensor(v)
        if _has_channel_axis(t):
            norm = torch.linalg.vector_norm(t, dim=-1, keepdim=True)
            return t / (norm + SAFE_EPSILON)
        return torch.sign(t)

    @stdlib("cross", sig='cross(a, b) \\u2192 vec3', category='Vector', doc='Cross product of two vec3 vectors.', ex='vec3 n = cross(tangent, bitangent);')
    @staticmethod
    def fn_cross(a, b) -> torch.Tensor:
        """Cross product. Only works on vec3 (last dim = 3)."""
        a_t, b_t = _to_tensor(a), _to_tensor(b)
        # Take first 3 channels if vec4
        if a_t.shape[-1] == 4:
            a_t = a_t[..., :3]
        if b_t.shape[-1] == 4:
            b_t = b_t[..., :3]
        return torch.cross(a_t, b_t, dim=-1)

    @stdlib("reflect", sig='reflect(v, n) \\u2192 vec', category='Vector', doc='Reflect vector v around normal n.', ex='vec3 r = reflect(incoming, normal);')
    @staticmethod
    def fn_reflect(incident, normal) -> torch.Tensor:
        i, n = _to_tensor(incident), _to_tensor(normal)
        d = (i * n).sum(dim=-1, keepdim=True)
        return i - 2.0 * d * n
