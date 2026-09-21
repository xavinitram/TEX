"""
TEX Standard Library — runtime implementations of built-in functions.

All functions operate on PyTorch tensors. Scalars are represented as
0-dim tensors or Python floats and get broadcast automatically by PyTorch.

This module is the FACADE of a per-domain split. It re-exports the shared substrate
(`stdlib_core`) so `tex_runtime.stdlib.<name>` resolves for every reader exactly as it did
when everything lived in this one file; it imports the domain leaves in the old class body's
section order — which is the REG-1 registration order that `help_entries()`, the generated
reference and the help panel read — and composes their mixins into `TEXStdlib`. Importing
this module populates the whole registry. The leaves:

    stdlib_core.py     constants, host-scalar helpers, cook context, caches + builders
    stdlib_math.py     math / matrix / clamping-interpolation / vector
    stdlib_color.py    colour / colour management / compositing / blend modes
    stdlib_sample.py   morphology / sampling / blur / convolve / patch_dist
    stdlib_noise.py    the noise wrappers over noise.py
    stdlib_sdf.py      SDF primitives / smooth min-max / gradient sampling
    stdlib_string.py   strings
    stdlib_array.py    arrays / whole-image reductions / debug_print

Adding a builtin: put the `@stdlib(...)`/`@staticmethod`/`def fn_NAME` in the leaf whose
domain it belongs to (a new domain is a new leaf, composed below AND added to
`tex_cache._CODEGEN_FILES`); nothing here changes.
"""
from __future__ import annotations
# The re-export surface. Every name the one-file module defined at module level is bound
# here too, to the SAME object, so `stdlib._grid_buf` is the dict the leaves fill and
# `stdlib._scalar_from_tensor` is the function the interpreter imports. (`_scalar_avg_warned`,
# a one-shot flag read only by `_scalar_from_tensor` itself, is the deliberate exception: a
# re-exported bool would be a stale snapshot, not the live flag.)
from .stdlib_core import (  # noqa: F401
    SAFE_EPSILON,
    ZERO_GUARD_EPS,
    LUMA_R,
    LUMA_G,
    LUMA_B,
    VEC_CHANNELS,
    _HOST_SCALAR_ATTR,
    _POW_NAN_STATE,
    _texlog,
    _SAMPLER_CACHE_MAX,
    _GRID_BUF_MAX,
    _GAUSS_KERNEL_MAX_ENTRIES,
    _MIP_MAX_ENTRIES,
    _MIP_MAX_LEVELS,
    _sampler_cache,
    _grid_buf,
    _mip_cache,
    _gauss_mip_cache,
    _gauss_kernel_cache,
    _cook_ctx,
    _has_channel_axis,
    _dtype_rounded,
    _tag_host_scalar,
    _host_scalar,
    _scalar_from_tensor,
    _get_bchw,
    _grid_sample_f32,
    _lerp_f32,
    _expand_to_bhw,
    _provider_read,
    set_cook_grid,
    restore_cook_ctx,
    _uniform_grid,
    _uniform_dtype,
    _get_grid_buf,
    _get_batch_index,
    _get_flat_batch_index,
    _get_lanczos_taps,
    _build_sample_grid,
    _reduce_channels,
    _lanczos3,
    _get_gauss_kernels,
    _pad_replicate_chunked,
    _gauss_blur_bchw,
    _build_mip_pyramid,
    _safe_version,
    _get_mip_pyramid,
    _get_mip_pyramid_gauss,
    _sample_mip_level,
    _sample_mip_trilinear,
    _to_tensor,
    _to_float,
    _is_scalar,
)
# The leaves, in the order the sections sat in the one-file class body — this IS the
# registration order, so it is not free to change.
from . import stdlib_math, stdlib_color, stdlib_sample, stdlib_noise, stdlib_sdf, stdlib_string, stdlib_array


class TEXStdlib(stdlib_math._StdlibMath, stdlib_color._StdlibColor, stdlib_sample._StdlibSample,
                stdlib_noise._StdlibNoise, stdlib_sdf._StdlibSdf, stdlib_string._StdlibString,
                stdlib_array._StdlibArray):
    """Registry of built-in TEX functions."""

    @staticmethod
    def get_functions() -> dict[str, callable]:
        # REG-1: the name->impl map is the registry view — one line, no drift.
        # Each fn_* carries a co-located @stdlib("name") decorator (see
        # stdlib_registry); adding a function no longer edits this method.
        from .stdlib_registry import functions
        return functions()


# Bind the composed class into the leaves whose builtins delegate through `TEXStdlib.fn_*`
# (see the note beside each leaf's `TEXStdlib = None` placeholder).
for _leaf in (stdlib_color, stdlib_sample, stdlib_noise, stdlib_sdf):
    _leaf.TEXStdlib = TEXStdlib
del _leaf


# -- Noise functions (extracted to noise.py) --------------------------------
from .noise import (
    _perlin2d_fast, _perlin3d_fast, _simplex2d,
    _fbm2d, _fbm3d,
    _worley2d, _worley3d,
    _worley2d_id, _worley3d_id,
    _curl2d, _curl3d,
    _ridged2d, _ridged3d,
    _billow2d, _billow3d,
    _turbulence2d, _turbulence3d,
    _flow2d, _flow3d,
    _alligator2d, _alligator3d,
)
