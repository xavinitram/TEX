"""
TEX Standard Library — colour, colour-management, compositing and blend-mode builtins (one domain leaf of `stdlib.py`).

The `TEXStdlib` class-body section(s) Color operations, Color management (SL-3), Compositing (SL-1), Blend modes (SL-2) moved here verbatim, onto the `_StdlibColor`
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
    LUMA_R,
    LUMA_G,
    LUMA_B,
    _grid_sample_f32,
    _to_tensor,
    _cook_device,
    _viewer_value,
    _uniform_dtype,
)
# ZERO_GUARD_EPS is bound by attribute lookup, not folded into the `from` import above: a
# name bound by `from X import name` compiles a later `name.method(...)` call site WITHOUT
# CPython's LOAD_ATTR+PUSH_NULL fusion, while a name bound by a plain assignment (even one
# whose RHS is an attribute lookup) keeps it — a compile-time instruction-selection quirk
# this split's G1 bytecode-identity gate caught (`_safe_div`'s `ZERO_GUARD_EPS.get(...)`),
# not a runtime difference; both bind the SAME object either way. See docs/worklog/lib-1.
from . import stdlib_core as _stdlib_core
ZERO_GUARD_EPS = _stdlib_core.ZERO_GUARD_EPS

# `TEXStdlib` is the class `stdlib.py` composes from every leaf. A leaf cannot import it at
# load time (the facade imports the leaves), so the facade BINDS it into this namespace the
# moment the class exists; the `TEXStdlib.fn_*(...)` delegations below then resolve at call
# time exactly as they did inside the one-file class. The spelling is load-bearing:
# `stdlib_registry._impl_looks_fragile` reads the literal `TEXStdlib.fn_*(` from the source
# to follow one level of delegation, so it must not be rewritten to the mixin's name.
TEXStdlib = None


class _StdlibColor:
    """colour, colour-management, compositing and blend-mode builtins: the `fn_*` methods `stdlib.py` mixes into `TEXStdlib`."""

    # -- Color operations -----------------------------------------------

    @stdlib("luma", sig='luma(rgb) \\u2192 float', category='Color', doc='Perceptual luminance of an RGB color.', ex='float gray = luma(@image);')
    @staticmethod
    def fn_luma(color) -> torch.Tensor:
        """Compute luminance from RGB(A). Returns scalar per pixel."""
        c = _to_tensor(color)
        if c.dim() >= 1 and c.shape[-1] >= 3:
            return LUMA_R * c[..., 0] + LUMA_G * c[..., 1] + LUMA_B * c[..., 2]
        return c

    @stdlib("hsv2rgb", sig='hsv2rgb(hsv) \\u2192 vec3', category='Color', doc='Convert HSV color to RGB.', ex='vec3 rgb = hsv2rgb(vec3(u, 1.0, 1.0));')
    @staticmethod
    def fn_hsv2rgb(hsv) -> torch.Tensor:
        """Convert HSV to RGB. Expects vec3 [H, S, V] with H in [0, 1]."""
        c = _to_tensor(hsv)
        h = c[..., 0:1] * 6.0  # scale to [0, 6]
        s = c[..., 1:2]
        v = c[..., 2:3]

        i = torch.floor(h)
        f = h - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))

        i_mod = torch.fmod(i, 6.0)

        # Compute masks once (shared across all 3 channels)
        # instead of 5-deep nested torch.where (which repeats comparisons 3×)
        m0 = (i_mod == 0.0)
        m1 = (i_mod == 1.0)
        m2 = (i_mod == 2.0)
        m3 = (i_mod == 3.0)
        m4 = (i_mod == 4.0)

        # r: 0->v, 1->q, 2->p, 3->p, 4->t, 5->v  (default v)
        r = torch.where(m1, q, torch.where(m2 | m3, p, torch.where(m4, t, v)))
        # g: 0->t, 1->v, 2->v, 3->q, 4->p, 5->p  (default p)
        g = torch.where(m1 | m2, v, torch.where(m3, q, torch.where(m4, p,
            torch.where(m0, t, p))))
        # b: 0->p, 1->p, 2->t, 3->v, 4->v, 5->q  (default q)
        b = torch.where(m1, p, torch.where(m2, t, torch.where(m3 | m4, v,
            torch.where(m0, p, q))))

        result = torch.cat([r, g, b], dim=-1)
        # If input was vec4, preserve alpha
        if c.shape[-1] == 4:
            result = torch.cat([result, c[..., 3:4]], dim=-1)
        return result

    @stdlib("rgb2hsv", sig='rgb2hsv(rgb) \\u2192 vec3', category='Color', doc='Convert RGB color to HSV.', ex='vec3 hsv = rgb2hsv(@image);')
    @staticmethod
    def fn_rgb2hsv(rgb) -> torch.Tensor:
        """Convert RGB to HSV. Returns vec3 [H, S, V] with H in [0, 1]."""
        c = _to_tensor(rgb)
        r, g, b = c[..., 0:1], c[..., 1:2], c[..., 2:3]

        cmax = torch.maximum(torch.maximum(r, g), b)
        cmin = torch.minimum(torch.minimum(r, g), b)
        diff = cmax - cmin + SAFE_EPSILON

        # Hue
        h = torch.where(cmax == r, torch.fmod((g - b) / diff, 6.0),
            torch.where(cmax == g, (b - r) / diff + 2.0,
                                   (r - g) / diff + 4.0))
        h = h / 6.0  # normalize to [0, 1]
        h = torch.fmod(h + 1.0, 1.0)  # ensure positive

        # Saturation
        s = torch.where(cmax > SAFE_EPSILON, diff / cmax, cmax.new_zeros(()))

        # Value
        v = cmax

        result = torch.cat([h, s, v], dim=-1)
        if c.shape[-1] == 4:
            result = torch.cat([result, c[..., 3:4]], dim=-1)
        return result

    # -- Color management (SL-3): sRGB<->linear + OKLab -----------------
    # Blurring/blending in gamma space produces wrong halos; convert to
    # linear-light first. OKLab gives perceptually-uniform gradients/mixes.
    # Each is elementwise and preserves a vec4 alpha unchanged.

    @staticmethod
    def _split_alpha(c):
        """(rgb, alpha) for a vec3/vec4 colour tensor: `alpha` is `c[..., 3:4]` when
        `c` is vec4, else `None`. Every colour function's "vec4 alpha passes through
        unchanged" is exactly this split plus `_join_alpha`'s re-attach at the end —
        factored out here because both halves are the SAME slice/cat ops the
        hand-written form already used, so converting a caller changes no tensor op,
        only where it's spelled (bit-identical, not merely equivalent)."""
        if c.dim() >= 1 and c.shape[-1] == 4:
            return c[..., 0:3], c[..., 3:4]
        return c, None

    @staticmethod
    def _join_alpha(rgb, alpha):
        """Inverse of `_split_alpha`: re-attach `alpha` (`torch.cat`) if present,
        else return `rgb` unchanged."""
        return torch.cat([rgb, alpha], dim=-1) if alpha is not None else rgb

    @stdlib("srgb_to_linear", sig='srgb_to_linear(c) \\u2192 vec', category='Color', doc='Gamma-encoded sRGB → linear-light. Blur/blend in linear to avoid halos.', ex='vec3 lin = srgb_to_linear(@image.rgb);')
    @staticmethod
    def fn_srgb_to_linear(color) -> torch.Tensor:
        """sRGB EOTF: gamma-encoded sRGB -> linear-light (piecewise). vec4 alpha
        passes through. Compose before blur/blend, then linear_to_srgb after."""
        rgb, alpha = TEXStdlib._split_alpha(_to_tensor(color))
        lin = torch.where(rgb <= 0.04045, rgb / 12.92,
                          ((rgb + 0.055) / 1.055).clamp(min=0.0) ** 2.4)
        return TEXStdlib._join_alpha(lin, alpha)

    @stdlib("linear_to_srgb", sig='linear_to_srgb(c) \\u2192 vec', category='Color', doc='Linear-light → gamma-encoded sRGB (inverse of srgb_to_linear).', ex='@OUT = vec4(linear_to_srgb(lin), 1.0);')
    @staticmethod
    def fn_linear_to_srgb(color) -> torch.Tensor:
        """sRGB OETF: linear-light -> gamma-encoded sRGB (inverse of
        srgb_to_linear). vec4 alpha passes through."""
        rgb, alpha = TEXStdlib._split_alpha(_to_tensor(color))
        srgb = torch.where(rgb <= 0.0031308, rgb * 12.92,
                           1.055 * rgb.clamp(min=0.0) ** (1.0 / 2.4) - 0.055)
        return TEXStdlib._join_alpha(srgb, alpha)

    @stdlib("oklab_from_rgb", sig='oklab_from_rgb(c) \\u2192 vec3', category='Color', doc='Linear RGB → OKLab. Mix/interpolate in OKLab for perceptually-even gradients.', ex='vec3 lab = oklab_from_rgb(srgb_to_linear(@image.rgb));')
    @staticmethod
    def fn_oklab_from_rgb(color) -> torch.Tensor:
        """Linear-light RGB -> OKLab (Ottosson). Mix/interpolate in OKLab then
        convert back for perceptually-even gradients. Expects LINEAR RGB — compose
        with srgb_to_linear for gamma-encoded images. vec4 alpha passes through."""
        c, alpha = TEXStdlib._split_alpha(_to_tensor(color))
        r, g, b = c[..., 0:1], c[..., 1:2], c[..., 2:3]
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
        l_ = torch.sign(l) * torch.abs(l).pow(1.0 / 3.0)
        m_ = torch.sign(m) * torch.abs(m).pow(1.0 / 3.0)
        s_ = torch.sign(s) * torch.abs(s).pow(1.0 / 3.0)
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        A = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        B = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        lab = torch.cat([L, A, B], dim=-1)
        return TEXStdlib._join_alpha(lab, alpha)

    @stdlib("oklab_to_rgb", sig='oklab_to_rgb(lab) \\u2192 vec3', category='Color', doc='OKLab → linear RGB (inverse of oklab_from_rgb).', ex='vec3 rgb = oklab_to_rgb(lab);')
    @staticmethod
    def fn_oklab_to_rgb(color) -> torch.Tensor:
        """OKLab -> linear-light RGB (inverse Ottosson). Compose with
        linear_to_srgb for a gamma-encoded result. vec4 alpha passes through."""
        c, alpha = TEXStdlib._split_alpha(_to_tensor(color))
        L, A, B = c[..., 0:1], c[..., 1:2], c[..., 2:3]
        l_ = L + 0.3963377774 * A + 0.2158037573 * B
        m_ = L - 0.1055613458 * A - 0.0638541728 * B
        s_ = L - 0.0894841775 * A - 1.2914855480 * B
        l, m, s = l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_
        r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
        g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
        b = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
        rgb = torch.cat([r, g, b], dim=-1)
        return TEXStdlib._join_alpha(rgb, alpha)

    # -- Rec.709 transfer + ACEScg<->linear matrix (COLOR-1, v0.40) -----
    # Rec.709 (BT.709) has its own OETF/EOTF — numerically distinct from sRGB's (a
    # different linear-segment slope and a different power), so it is its own pair, not
    # an alias. ACEScg (AP1 primaries, linear) <-> linear Rec.709/sRGB (D65) is a fixed
    # 3x3 change of primaries, inlined as scalar-coefficient sums — the SAME code shape
    # as fn_oklab_from_rgb above (constant-matrix x vec, elementwise), not torch.matmul:
    # a compile-time-known constant needs no device branch to already be the fast form
    # on both CPU and CUDA (unlike a RUNTIME matrix, which is what `_matvec`'s CPU/CUDA
    # gate in interpreter.py is for).

    @stdlib("rec709_to_linear", sig='rec709_to_linear(c) \\u2192 vec', category='Color', doc='Gamma-encoded Rec.709 → linear-light (BT.709 EOTF; distinct curve from sRGB).', ex='vec3 lin = rec709_to_linear(@image.rgb);')
    @staticmethod
    def fn_rec709_to_linear(color) -> torch.Tensor:
        """BT.709 EOTF: gamma-encoded Rec.709 -> linear-light (piecewise; distinct
        constants from sRGB's). vec4 alpha passes through unchanged."""
        rgb, alpha = TEXStdlib._split_alpha(_to_tensor(color))
        lin = torch.where(rgb < 0.081, rgb / 4.5,
                          ((rgb + 0.099) / 1.099).clamp(min=0.0) ** (1.0 / 0.45))
        return TEXStdlib._join_alpha(lin, alpha)

    @stdlib("linear_to_rec709", sig='linear_to_rec709(c) \\u2192 vec', category='Color', doc='Linear-light → gamma-encoded Rec.709 (inverse of rec709_to_linear).', ex='@OUT = vec4(linear_to_rec709(lin), 1.0);')
    @staticmethod
    def fn_linear_to_rec709(color) -> torch.Tensor:
        """BT.709 OETF: linear-light -> gamma-encoded Rec.709 (inverse of
        rec709_to_linear). vec4 alpha passes through unchanged."""
        rgb, alpha = TEXStdlib._split_alpha(_to_tensor(color))
        gam = torch.where(rgb < 0.018, rgb * 4.5,
                          1.099 * rgb.clamp(min=0.0) ** 0.45 - 0.099)
        return TEXStdlib._join_alpha(gam, alpha)

    @stdlib("acescg_to_linear", sig='acescg_to_linear(c) \\u2192 vec3', category='Color', doc='ACEScg (AP1, linear) → linear Rec.709/sRGB (D65) via a fixed 3×3 primary change.', ex='vec3 lin709 = acescg_to_linear(@aces_plate.rgb);')
    @staticmethod
    def fn_acescg_to_linear(color) -> torch.Tensor:
        """ACEScg (AP1 primaries, linear) -> linear Rec.709/sRGB (D65). Fixed 3x3
        matrix (ACES 1.0.3-class AP1->Rec.709 D65), inlined as scalar-coefficient
        sums (P3 shape). vec4 alpha passes through unchanged."""
        c, alpha = TEXStdlib._split_alpha(_to_tensor(color))
        r, g, b = c[..., 0:1], c[..., 1:2], c[..., 2:3]
        r2 = 1.70505 * r - 0.62179 * g - 0.08316 * b
        g2 = -0.13026 * r + 1.14080 * g - 0.01055 * b
        b2 = -0.02400 * r - 0.12897 * g + 1.15297 * b
        lin = torch.cat([r2, g2, b2], dim=-1)
        return TEXStdlib._join_alpha(lin, alpha)

    @stdlib("linear_to_acescg", sig='linear_to_acescg(c) \\u2192 vec3', category='Color', doc='Linear Rec.709/sRGB (D65) → ACEScg (AP1, linear) (inverse of acescg_to_linear).', ex='vec3 acescg = linear_to_acescg(srgb_to_linear(@image.rgb));')
    @staticmethod
    def fn_linear_to_acescg(color) -> torch.Tensor:
        """Linear Rec.709/sRGB (D65) -> ACEScg (AP1 primaries, linear) — the EXACT
        matrix inverse of acescg_to_linear's, so the round trip is float-precision
        clean. vec4 alpha passes through unchanged."""
        c, alpha = TEXStdlib._split_alpha(_to_tensor(color))
        r, g, b = c[..., 0:1], c[..., 1:2], c[..., 2:3]
        r2 = 0.61309721 * r + 0.33951747 * g + 0.04732740 * b
        g2 = 0.07019593 * r + 0.91635827 * g + 0.01344794 * b
        b2 = 0.02061416 * r + 0.10957019 * g + 0.86981469 * b
        acescg = torch.cat([r2, g2, b2], dim=-1)
        return TEXStdlib._join_alpha(acescg, alpha)

    # -- 3D LUT (COLOR-1, v0.40) -----------------------------------------
    # `lut` is a plain bound tensor (ruling 5 — no new TEXType), the shape
    # `tex_io.lut.read_cube` produces: [N,N,N,3] indexed [b_idx,g_idx,r_idx]. footprint
    # stays 'point' (the design's call): the LUT is a small fixed-size resource bound
    # once per cook, not a neighbourhood read of the cook's own tiled image, so it needs
    # no ROI halo. No new codegen: this falls through to the generic `_fns[name]`
    # dispatch (codegen.py's pre-resolved-local fallback) exactly like every other
    # Color-domain function above, so interp<->codegen stays bit-exact by construction.

    @stdlib("apply_lut3d", sig='apply_lut3d(rgb, lut) \\u2192 vec3', category='Color', footprint='point', non_spatial_args=(1,), doc='Trilinear 3D LUT lookup. `lut` is a bound [N,N,N,3] tensor (tex_io.lut.read_cube).', ex='@OUT = vec4(apply_lut3d(@image.rgb, @lut), 1.0);')
    @staticmethod
    def fn_apply_lut3d(rgb, lut) -> torch.Tensor:
        """Trilinear 3D LUT lookup via `grid_sample`'s volumetric (5D) form. `rgb` is
        the [0,1]-domain colour to transform (its own dtype is preserved in the
        output — the grid_sample itself always runs in fp32, mirroring invariant #4's
        treatment of a value used as a SAMPLING COORDINATE regardless of its origin);
        `lut` is a plain bound [N,N,N,3] tensor, axis order [b_idx,g_idx,r_idx] (see
        tex_io/lut.py). vec4 alpha passes through unchanged."""
        L = _to_tensor(lut)
        rgb3, alpha = TEXStdlib._split_alpha(_to_tensor(rgb))
        in_dtype = rgb3.dtype
        # [N(b),N(g),N(r),3] -> [1,3,N(b),N(g),N(r)] (BCDHW). grid_sample's grid axes
        # (x,y,z) address (W,H,D) respectively, so W<-r, H<-g, D<-b — which is exactly
        # rgb3's own (r,g,b) channel order, so the grid below needs no channel reorder.
        vol = L.permute(3, 0, 1, 2).unsqueeze(0)
        orig_shape = rgb3.shape
        grid = rgb3.reshape(1, 1, 1, -1, 3).to(torch.float32) * 2.0 - 1.0
        # M-3 (_grid_sample_f32): the grid is always fp32 (a coordinate use, forced fp32
        # above regardless of rgb3's own dtype, mirroring invariant #4); if `vol` (the LUT)
        # is ever not fp32, this samples in fp32 and casts back to vol's own dtype -- the
        # SAME dtype-reconciliation every other sampling builtin uses, not a hand-rolled
        # cast. `vol` is fp32 today (tex_io.lut's loader), so this is presently a no-op
        # fast path; the final in_dtype cast below is COLOR-1's own choice (match rgb's
        # dtype, not the LUT's), a separate concern M-3 doesn't own.
        out = _grid_sample_f32(
            vol, grid, mode='bilinear', padding_mode='border', align_corners=True,
        )                                              # [1, 3, 1, 1, P]
        out = out.reshape(3, -1).permute(1, 0).reshape(orig_shape)
        if out.dtype != in_dtype:
            out = out.to(in_dtype)
        return TEXStdlib._join_alpha(out, alpha)

    # -- PM-11: the fused viewer transform's two reserved builtins --
    #
    # Zero-arg, host-fed VALUES (never `$param`s — a `$param` is baked into the compile
    # fingerprint and would recompile/re-cache-miss on every drag of a viewer slider,
    # ENG-7's own reason for rejecting a `$time` param). `tex_engine.cook`/`prepare` take a
    # new `viewer_context=` kwarg mirroring `time_context=`; the ENGINE plumbs it to
    # `Interpreter.execute`/codegen's `_invoke_cg`, both of which publish it on the SAME
    # `stdlib_core._cook_ctx` thread-local `set_cook_grid` already uses (P0-D) — the one
    # seam every tier (interpreter, default codegen, torch_compile, auto) already calls at
    # the top of a cook, so no new subsystem is needed. `_cook_device()` is why: neither
    # builtin has a tensor ARGUMENT to size a device from (the first stdlib pair that
    # doesn't), so the device published at that same seam is the only way to build a
    # correctly-placed tensor. Returning a tensor (not a bare float) matches every other
    # FLOAT-returning builtin (`fn_img_width`) and skips codegen's generic float-wrap path,
    # which only fires for an UNTYPED return.
    #
    # No new fusion logic (design doc §3): `@OUT = @A.rgb * viewer_exposure();` is ordinary
    # trailing TEX reached by the existing `tex_fusion.py` splice, using the SAME two
    # builtins any program uses. Never DECLINED by codegen (contrast ENG-7's `frame`/`fps`/
    # `time`, which not `env`-cached because they ANIMATE): reads it fresh every call
    # through this ordinary `_fns[name]` dispatch, so the emitted `_tex_src` never embeds a
    # value and a viewer tweak alone cannot move the compile fingerprint or reopen a
    # `_compiled_cache`/dynamo entry. CUDA-graph capture is the one tier that DOES bake a
    # value into a replay buffer (the class ENG-7's own comment names), so `graphed._capturable`
    # bars it — same bar, same reason, `_VIEWER_BUILTIN_NAMES` beside `_TIME_BUILTIN_NAMES`.
    # `viewer_gamma()`'s own value is a POW exponent once composed downstream and the
    # exposure a multiplicative gain — both host-supplied and bounded by nothing (frame/
    # time's own reasoning), so both are registered in `stdlib_registry.FP16_FRAGILE`.

    @stdlib("viewer_exposure", sig='viewer_exposure() \\u2192 float', category='Color', footprint='point',
            doc="The host viewer's exposure gain for THIS cook (default 1.0 = no-op). Fed by "
                "tex_engine.cook(viewer_context={\"viewer_exposure\": ...}); never baked into "
                "the compile fingerprint (PM-11).",
            ex='@OUT = vec4(@A.rgb * viewer_exposure(), 1.0);')
    @staticmethod
    def fn_viewer_exposure() -> torch.Tensor:
        """PM-11: a 0-dim tensor on the cook's device, in the cook's working dtype (an
        ordinary VALUE builtin, unlike the fp32-forced coordinate/shape builtins — this
        multiplies image lineage directly, so it belongs in the same dtype as the pixels
        it scales). 1.0 (identity) when no host supplied a viewer_context."""
        dt = _uniform_dtype() or torch.float32
        return torch.scalar_tensor(_viewer_value("viewer_exposure", 1.0),
                                   dtype=dt, device=_cook_device() or "cpu")

    @stdlib("viewer_gamma", sig='viewer_gamma() \\u2192 float', category='Color', footprint='point',
            doc="The host viewer's gamma for THIS cook (default 1.0 = no-op). Fed by "
                "tex_engine.cook(viewer_context={\"viewer_gamma\": ...}); never baked into "
                "the compile fingerprint (PM-11).",
            ex='@OUT = vec4(pow(@A.rgb, vec3(1.0 / viewer_gamma())), 1.0);')
    @staticmethod
    def fn_viewer_gamma() -> torch.Tensor:
        """PM-11: see fn_viewer_exposure — the same seam, the same no-op default."""
        dt = _uniform_dtype() or torch.float32
        return torch.scalar_tensor(_viewer_value("viewer_gamma", 1.0),
                                   dtype=dt, device=_cook_device() or "cpu")

    # -- Compositing (SL-1): Porter-Duff on straight (un-premultiplied) vec4 --
    # ComfyUI IMAGE/MASK are un-premultiplied; over/under/atop take & return
    # straight-alpha vec4. premultiply/unpremultiply convert between conventions.

    @stdlib("premultiply", sig='premultiply(rgba) \\u2192 vec4', category='Color', doc='Straight → premultiplied alpha (rgb *= a).', ex='vec4 p = premultiply(@image);')
    @staticmethod
    def fn_premultiply(color) -> torch.Tensor:
        """Straight -> premultiplied alpha: rgb *= a (vec4)."""
        c = _to_tensor(color)
        a = c[..., 3:4]
        return torch.cat([c[..., 0:3] * a, a], dim=-1)

    @stdlib("unpremultiply", sig='unpremultiply(rgba) \\u2192 vec4', category='Color', doc='Premultiplied → straight alpha (rgb /= a).', ex='vec4 s = unpremultiply(p);')
    @staticmethod
    def fn_unpremultiply(color) -> torch.Tensor:
        """Premultiplied -> straight alpha: rgb /= a (vec4; safe at a=0, incl. fp16)."""
        c = _to_tensor(color)
        a = c[..., 3:4]
        return torch.cat([TEXStdlib._safe_div(c[..., 0:3], a), a], dim=-1)

    @stdlib("over", sig='over(fg, bg) \\u2192 vec4', category='Color', doc="Porter-Duff 'over': composite fg atop bg (straight-alpha RGBA).", ex='@OUT = over(@A, @B);')
    @staticmethod
    def fn_over(fg, bg) -> torch.Tensor:
        """Porter-Duff 'over': fg composited over bg (straight-alpha vec4)."""
        f = _to_tensor(fg)
        b = _to_tensor(bg)
        fa, ba = f[..., 3:4], b[..., 3:4]
        oa = fa + ba * (1.0 - fa)
        orgb = TEXStdlib._safe_div(f[..., 0:3] * fa + b[..., 0:3] * ba * (1.0 - fa), oa)
        return torch.cat([orgb, oa], dim=-1)

    @stdlib("under", sig='under(fg, bg) \\u2192 vec4', category='Color', doc='Composite fg under bg (= over(bg, fg)).', ex='@OUT = under(@A, @B);')
    @staticmethod
    def fn_under(fg, bg) -> torch.Tensor:
        """'under': fg under bg == over(bg, fg)."""
        return TEXStdlib.fn_over(bg, fg)

    @stdlib("atop", sig='atop(fg, bg) \\u2192 vec4', category='Color', doc="'atop': fg confined to bg's coverage.", ex='@OUT = atop(@A, @B);')
    @staticmethod
    def fn_atop(fg, bg) -> torch.Tensor:
        """'atop': fg atop bg — output confined to bg's coverage (out_a = bg.a)."""
        f = _to_tensor(fg)
        b = _to_tensor(bg)
        fa = f[..., 3:4]
        orgb = f[..., 0:3] * fa + b[..., 0:3] * (1.0 - fa)
        return torch.cat([orgb, b[..., 3:4]], dim=-1)

    # -- Blend modes (SL-2): per-channel, curated ~8 --------------------
    # Each op(base, blend) works on RGB channels; a vec4 base keeps its alpha.

    @staticmethod
    def _blend_rgb(base, blend, op):
        b = _to_tensor(base)
        s = _to_tensor(blend)
        n = min(b.shape[-1], 3)
        rgb = op(b[..., :n], s[..., :n])
        return torch.cat([rgb, b[..., 3:4]], dim=-1) if b.shape[-1] == 4 else rgb

    @staticmethod
    def _safe_div(num, denom):
        """num / denom with a DTYPE-AWARE, SIGN-PRESERVING zero floor on denom.

        The epsilon is dtype-aware because SAFE_EPSILON (1e-8) underflows to 0 in
        fp16 (ZERO_GUARD_EPS uses fp16's smallest normal there; fp32 keeps 1e-8).

        It floors the MAGNITUDE, not the signed value: `denom.clamp(min=eps)` would
        raise a small NEGATIVE denominator up to +eps — flipping the sign and
        blowing up the quotient (wrong for over/unpremultiply when an alpha goes
        out of [0,1], e.g. a mask subtraction dipping below zero). Here a
        below-threshold denominator is replaced by ±eps carrying denom's own sign."""
        eps = ZERO_GUARD_EPS.get(denom.dtype, SAFE_EPSILON)
        eps_t = torch.as_tensor(eps, dtype=denom.dtype, device=denom.device)
        below = denom.abs() < eps
        guard_trace.note(below)  # C4-ux (no-op unless armed)
        safe = torch.where(below, torch.copysign(eps_t, denom), denom)
        return num / safe

    @stdlib("screen", sig='screen(a, b) \\u2192 vec', category='Color', doc='Screen blend: 1 - (1-a)(1-b). Brightens.', ex='@OUT = vec4(screen(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_screen(base, blend) -> torch.Tensor:
        """1 - (1-a)(1-b)."""
        return TEXStdlib._blend_rgb(base, blend, lambda a, b: 1.0 - (1.0 - a) * (1.0 - b))

    @stdlib("overlay", sig='overlay(a, b) \\u2192 vec', category='Color', doc='Overlay blend (multiply/screen by base).', ex='@OUT = vec4(overlay(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_overlay(base, blend) -> torch.Tensor:
        """a<0.5 ? 2ab : 1-2(1-a)(1-b)."""
        return TEXStdlib._blend_rgb(base, blend, lambda a, b: torch.where(
            a < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b)))

    @stdlib("hard_light", sig='hard_light(a, b) \\u2192 vec', category='Color', doc='Hard-light blend (overlay with operands swapped).', ex='@OUT = vec4(hard_light(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_hard_light(base, blend) -> torch.Tensor:
        """overlay with the operands swapped."""
        return TEXStdlib._blend_rgb(base, blend, lambda a, b: torch.where(
            b < 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b)))

    @stdlib("soft_light", sig='soft_light(a, b) \\u2192 vec', category='Color', doc='Soft-light blend (Pegtop, smooth).', ex='@OUT = vec4(soft_light(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_soft_light(base, blend) -> torch.Tensor:
        """Pegtop soft-light: (1-2b)a^2 + 2ab (smooth, no branch)."""
        return TEXStdlib._blend_rgb(base, blend,
                                    lambda a, b: (1.0 - 2.0 * b) * a * a + 2.0 * a * b)

    @stdlib("color_dodge", sig='color_dodge(a, b) \\u2192 vec', category='Color', doc='Color-dodge: brightens base by blend.', ex='@OUT = vec4(color_dodge(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_color_dodge(base, blend) -> torch.Tensor:
        """min(1, a / (1-b)); b>=1 -> 1."""
        return TEXStdlib._blend_rgb(base, blend, lambda a, b: torch.clamp(
            TEXStdlib._safe_div(a, 1.0 - b), max=1.0))

    @stdlib("color_burn", sig='color_burn(a, b) \\u2192 vec', category='Color', doc='Color-burn: darkens base by blend.', ex='@OUT = vec4(color_burn(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_color_burn(base, blend) -> torch.Tensor:
        """1 - min(1, (1-a)/b); b<=0 -> 0."""
        return TEXStdlib._blend_rgb(base, blend, lambda a, b: 1.0 - torch.clamp(
            TEXStdlib._safe_div(1.0 - a, b), max=1.0))

    @stdlib("linear_light", sig='linear_light(a, b) \\u2192 vec', category='Color', doc='Linear-light blend: clamp(a + 2b - 1).', ex='@OUT = vec4(linear_light(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_linear_light(base, blend) -> torch.Tensor:
        """clamp(a + 2b - 1, 0, 1)."""
        return TEXStdlib._blend_rgb(base, blend,
                                    lambda a, b: torch.clamp(a + 2.0 * b - 1.0, 0.0, 1.0))

    @stdlib("vivid_light", sig='vivid_light(a, b) \\u2192 vec', category='Color', doc='Vivid-light blend (burn/dodge by blend).', ex='@OUT = vec4(vivid_light(@A.rgb, @B.rgb), 1.0);')
    @staticmethod
    def fn_vivid_light(base, blend) -> torch.Tensor:
        """b<0.5 -> color_burn(a,2b); else color_dodge(a,2(b-0.5))."""
        def _op(a, b):
            burn = 1.0 - torch.clamp(TEXStdlib._safe_div(1.0 - a, 2.0 * b), max=1.0)
            dodge = torch.clamp(TEXStdlib._safe_div(a, 1.0 - 2.0 * (b - 0.5)), max=1.0)
            return torch.where(b < 0.5, burn, dodge)
        return TEXStdlib._blend_rgb(base, blend, _op)
