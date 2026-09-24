"""
TEX Standard Library — array and whole-image reduction builtins (one domain leaf of `stdlib.py`).

The `TEXStdlib` class-body section(s) Array functions, Image reduction functions moved here verbatim, onto the `_StdlibArray`
mixin. `stdlib.py` composes the leaves' mixins into `TEXStdlib` in the class body's original
section order, which is the REG-1 registration order (`help_entries()`, the generated
reference and the help panel all read it) — so import this leaf THROUGH `stdlib`, not
directly, unless registering only this domain is what you want.
"""
from __future__ import annotations
import math
import torch
from .stdlib_registry import stdlib
from .stdlib_core import (
    _host_int,
    _reduce_channels,
    _to_tensor,
)


class _StdlibArray:
    """array and whole-image reduction builtins: the `fn_*` methods `stdlib.py` mixes into `TEXStdlib`."""

    # -- Array functions ------------------------------------------------

    @stdlib("sort", sig='sort(arr) \\u2192 array', category='Arrays', doc='Sort array elements in ascending order.', ex='sort(arr);')
    @staticmethod
    def fn_sort(arr):
        """Sort array elements in ascending order. Returns sorted copy."""
        if isinstance(arr, list):
            return sorted(arr)
        t = _to_tensor(arr)
        if t.dim() in (2, 5):  # vec array: sort along element dim per channel
            return torch.sort(t, dim=-2).values
        return torch.sort(t, dim=-1).values

    @stdlib("reverse", sig='reverse(arr) \\u2192 array', category='Arrays', doc='Reverse array element order.', ex='reverse(arr);')
    @staticmethod
    def fn_reverse(arr):
        """Reverse array elements. Returns reversed copy."""
        if isinstance(arr, list):
            return list(reversed(arr))
        t = _to_tensor(arr)
        if t.dim() in (2, 5):  # vec array
            return torch.flip(t, dims=[-2])
        return torch.flip(t, dims=[-1])

    @stdlib("arr_sum", sig='arr_sum(arr) \\u2192 float', category='Arrays', doc='Sum of all array elements.', ex='float total = arr_sum(arr);')
    @staticmethod
    def fn_arr_sum(arr) -> torch.Tensor:
        """Sum all elements of an array. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr).float(), lambda t, d: t.sum(dim=d))

    @stdlib("arr_min", sig='arr_min(arr) \\u2192 float', category='Arrays', doc='Minimum value in array.', ex='float lo = arr_min(arr);')
    @staticmethod
    def fn_arr_min(arr) -> torch.Tensor:
        """Minimum element of an array per channel. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr).float(), lambda t, d: t.min(dim=d).values)

    @stdlib("arr_max", sig='arr_max(arr) \\u2192 float', category='Arrays', doc='Maximum value in array.', ex='float hi = arr_max(arr);')
    @staticmethod
    def fn_arr_max(arr) -> torch.Tensor:
        """Maximum element of an array per channel. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr).float(), lambda t, d: t.max(dim=d).values)

    @stdlib("median", sig='median(arr) \\u2192 float', category='Arrays', doc='Median value of array.', ex='float mid = median(arr);')
    @staticmethod
    def fn_median(arr) -> torch.Tensor:
        """Median element of an array per channel. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr).float(), lambda t, d: torch.median(t, dim=d).values)

    @stdlib("arr_avg", sig='arr_avg(arr) \\u2192 float', category='Arrays', doc='Average of all array elements.', ex='float avg = arr_avg(arr);')
    @staticmethod
    def fn_arr_avg(arr) -> torch.Tensor:
        """Average of array elements per channel. Returns scalar (or vec) per pixel."""
        return _reduce_channels(_to_tensor(arr).float(), lambda t, d: t.mean(dim=d))

    @stdlib("join", sig='join(arr, sep) \\u2192 string', category='Arrays', doc='Concatenate string array with separator.', ex='string csv = join(names, ", ");')
    @staticmethod
    def fn_join(arr, sep) -> str:
        """Concatenate string array elements with separator."""
        if not isinstance(arr, list):
            raise ValueError("join() expects a string array")
        if not isinstance(sep, str):
            raise ValueError("join() separator must be a string")
        return sep.join(str(s) for s in arr)

    # -- Image reduction functions ---------------------------------------

    @stdlib("img_sum", sig='img_sum(img) \\u2192 vec', category='Image Stats', footprint='image', doc='Per-channel sum of all pixel values.', ex='vec3 total = img_sum(@A);')
    @staticmethod
    def fn_img_sum(image) -> torch.Tensor:
        """Sum of all pixels per channel per frame. Returns broadcast-friendly shape."""
        img = _to_tensor(image)
        if img.dim() >= 3:
            # PR-LP4: accumulate + return fp32 — an fp16 sum overflows to inf at
            # >=1024^2 (5.2e5 > 65504). .float() is a no-op on fp32 (bit-identical,
            # and identically mirrored in codegen_stdfns), so never cast the result back.
            return img.float().sum(dim=(1, 2), keepdim=True)
        return img

    @stdlib("img_mean", sig='img_mean(img) \\u2192 vec', category='Image Stats', footprint='image', doc='Per-channel mean (average) of the image.', ex='vec3 avg = img_mean(@A);')
    @staticmethod
    def fn_img_mean(image) -> torch.Tensor:
        """Mean of all pixels per channel per frame."""
        img = _to_tensor(image)
        if img.dim() >= 3:
            return img.float().mean(dim=(1, 2), keepdim=True)  # PR-LP4: fp32 accumulate
        return img

    @stdlib("img_min", sig='img_min(img) \\u2192 vec', category='Image Stats', footprint='image', doc='Per-channel minimum across the entire image.', ex='vec3 lo = img_min(@A);')
    @staticmethod
    def fn_img_min(image) -> torch.Tensor:
        """Min pixel value per channel per frame."""
        img = _to_tensor(image)
        if img.dim() >= 3:
            return img.float().amin(dim=(1, 2), keepdim=True)  # PR-LP4: fp32 return
        return img

    @stdlib("img_max", sig='img_max(img) \\u2192 vec', category='Image Stats', footprint='image', doc='Per-channel maximum across the entire image.', ex='vec3 hi = img_max(@A);')
    @staticmethod
    def fn_img_max(image) -> torch.Tensor:
        """Max pixel value per channel per frame."""
        img = _to_tensor(image)
        if img.dim() >= 3:
            return img.float().amax(dim=(1, 2), keepdim=True)  # PR-LP4: fp32 return
        return img

    @stdlib("img_median", sig='img_median(img) \\u2192 vec', category='Image Stats', footprint='image', doc='Per-channel median of the image.', ex='vec3 mid = img_median(@A);')
    @staticmethod
    def fn_img_median(image) -> torch.Tensor:
        """Median pixel value per channel per frame."""
        img = _to_tensor(image).float()  # PR-LP4: reduce + return fp32 (fp16-safe)
        if img.dim() == 4:
            B, H, W, C = img.shape
            flat = img.reshape(B, H * W, C)
            return torch.median(flat, dim=1).values.unsqueeze(1).unsqueeze(1)
        if img.dim() == 3:
            B, H, W = img.shape
            flat = img.reshape(B, H * W)
            return torch.median(flat, dim=1).values.unsqueeze(1).unsqueeze(1)
        return img

    @stdlib("img_width", sig='img_width(img) \\u2192 float', category='Image Stats', footprint='image',
            doc='Width in pixels of an image or mask (a uniform reads 1; iw is the cook grid).', ex='float kw = img_width(@kernel);')
    @staticmethod
    def fn_img_width(image) -> torch.Tensor:
        """Width (shape[2]) of a binding's own tensor, as a 0-dim fp32 tensor on its
        device — built exactly as `iw` is (invariant #4: forced fp32, never the cook
        dtype). Rank < 3 (a uniform) reads 1.0, not an error and not `iw`: a
        non-terminal fusion stage's @OUT becomes a local that can be compact-shaped
        (tex_fusion.py), so 'rank < 3 is 1' is the reading that agrees on every path —
        raising would error a fused chain the unfused cook serves."""
        img = _to_tensor(image)
        w = float(img.shape[2]) if img.dim() >= 3 else 1.0
        return torch.scalar_tensor(w, dtype=torch.float32, device=img.device)

    @stdlib("img_height", sig='img_height(img) \\u2192 float', category='Image Stats', footprint='image',
            doc='Height in pixels of an image or mask (a uniform reads 1; ih is the cook grid).', ex='float kh = img_height(@kernel);')
    @staticmethod
    def fn_img_height(image) -> torch.Tensor:
        """Height (shape[1]) of a binding's own tensor — see fn_img_width."""
        img = _to_tensor(image)
        h = float(img.shape[1]) if img.dim() >= 3 else 1.0
        return torch.scalar_tensor(h, dtype=torch.float32, device=img.device)

    @stdlib("debug_print", sig='debug_print(label, value[, x, y]) \\u2192 value', category='Debugging', sync=True,
            doc="Probe a value at a pixel — records it for the node's "
            "HUD and returns the value unchanged (a print-style debug tap). Interpreter"
            "-only; a compiled tier falls back so the probe always fires.",
            ex='float g = debug_print("luma", luma(@A.rgb), 0, 0);')
    @staticmethod
    def fn_debug_print(label, value, x=0.0, y=0.0):
        """LX-5: value-at-pixel probe. Records value at (x,y) into the tier_trace probe
        list (folded into the ui= HUD payload) and returns `value` UNCHANGED so @OUT is
        bit-identical with or without the probe. torch-native readout, no numpy."""
        from . import tier_trace
        import math

        def _json_safe(v):
            # audit: a NaN/Inf probe would serialize as a bare NaN/Infinity token — invalid
            # JSON that breaks the ui= websocket frame. Map non-finite floats to None (null).
            if isinstance(v, list):
                return [_json_safe(x) for x in v]
            return None if isinstance(v, float) and not math.isfinite(v) else v

        try:
            # TRK-67: `_host_int` takes a literal/`$param` x/y from the host reading it
            # was minted with, same as the string family; a device-computed x/y (rare
            # for a debug-probe coordinate) still reads back.
            xi = _host_int(x)
            yi = _host_int(y)
            if isinstance(value, torch.Tensor) and value.dim() >= 3:
                H, W = value.shape[1], value.shape[2]
                pv = value[0, min(max(yi, 0), H - 1), min(max(xi, 0), W - 1)]
                recorded = pv.detach().float().reshape(-1)[:4].tolist()
            elif isinstance(value, torch.Tensor):
                recorded = value.detach().float().reshape(-1)[:4].tolist()
            else:
                recorded = float(value)
            tier_trace.record_probe(label, _json_safe(recorded), xi, yi)
        except Exception:
            pass
        return value
