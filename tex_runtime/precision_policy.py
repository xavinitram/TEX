"""
PR-LP2 — `precision="auto"` classifier (doc 28 §2.3).

fp16 is the only measured low-precision lever, and it wins only in a specific region:
CUDA + resolution >= 1024^2 + a **pointwise, accuracy-robust** program. This module
resolves `auto` to `fp16` or `fp32` with a human reason (recorded in tier_trace).

Eligibility = `is_tile_safe(program)` (pointwise: no sampling/fetch/blur/reduction —
those are the fp16 losers: lens 1.9e-2, and the img_* reductions; and no scatter, which
is coordinate-sensitive 0.23) **AND** no image-derived threshold (a comparison / step /
smoothstep / branch whose operand traces to an input image — fp16 rounding flips the
threshold, e.g. `halftone`'s `smoothstep(luma(@image)+/-e, dot)`).

Over-declining is always safe (fp32 is exact and the 1.0x baseline); the gate only ever
declines a possible win, never accepts an inaccurate one.
"""
import torch

from ..tex_memory import is_tile_safe  # single source for pointwise-ness (M-4)
from ..tex_compiler.ast_nodes import iter_child_nodes as _children

_MIN_FP16_PX = 1024 * 1024          # measured: fp16 win starts at 1024^2 (0.96x at 512^2)
_COMPARE_OPS = frozenset({">", "<", ">=", "<=", "==", "!="})

# fp16-fragile stdlib functions — a half-ULP wrecks them (measured in the 114-sweep:
# posterize floor -> 0.33, film_vignette acos/sqrt/smoothstep -> 0.63, hue_shift
# fract -> NaN). Declined on ANY lineage, because fp16 *intermediates* lose precision
# even for coordinate-derived values (u/v stay fp32, but `float d = ...` does not).
#   - discontinuous: a fp16 rounding flips the branch/level
#   - domain-restricted: a fp16 overshoot leaves the domain -> NaN (acos/asin |x|>1, sqrt/log<0)
# NOTE: pow/spow/sdiv are NOT here — the grade-class headline needs pow([0,1], 0.45),
# which is exact enough; the rare pow/division NaN is caught by execute()'s runtime
# finiteness fallback instead (can't be told apart statically from grade's safe pow).
_FP16_FRAGILE_FNS = frozenset({
    "floor", "round", "ceil", "fract", "trunc", "mod", "sign",
    "step", "smoothstep",
    "acos", "asin", "sqrt", "inversesqrt", "log", "log2",
    # magnitude-amplifying / near-singular — a fp16 mantissa is too coarse above ~1
    # (measured in the fp16 fuzzer: exp-towers 0.02-0.08, smin/smax blends 0.4)
    "exp", "exp2", "smin", "smax",
})


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(_children(n))


def _output_names(program) -> set:
    """Names of @-bindings written by the program (so they aren't treated as image
    *inputs* for taint)."""
    outs = set()
    for n in _walk(program):
        if n.__class__.__name__ == "Assignment":
            t = n.target
            tc = t.__class__.__name__
            if tc == "BindingRef":
                outs.add(t.name)
            elif tc == "ChannelAccess" and t.object.__class__.__name__ == "BindingRef":
                outs.add(t.object.name)
            elif tc == "BindingIndexAccess" and t.binding.__class__.__name__ == "BindingRef":
                outs.add(t.binding.name)
    return outs


def _reads_image(node, tainted, out_names) -> bool:
    """True if this expression subtree reads an input image wire binding (@A, not an
    output, not a $param) or an image-tainted variable."""
    for n in _walk(node):
        cls = n.__class__.__name__
        if cls == "BindingRef" and n.kind == "wire" and n.name not in out_names:
            return True
        if cls == "Identifier" and n.name in tainted:
            return True
    return False


def _image_tainted_vars(program, out_names) -> set:
    """Variables whose value derives from an input image, via a fixed-point forward
    scan (handles chains like `col=@A; lum=luma(col); thr=lum;`). Fixed-point (not a
    single pass) so out-of-order/looped assignments can't under-taint (which would be
    the unsafe direction — accepting an inaccurate program)."""
    tainted: set = set()
    for _ in range(8):  # converges in <=depth passes; 8 is a generous cap
        grew = False
        for n in _walk(program):
            cls = n.__class__.__name__
            if cls == "VarDecl" and n.initializer is not None:
                if n.name not in tainted and _reads_image(n.initializer, tainted, out_names):
                    tainted.add(n.name); grew = True
            elif cls == "Assignment" and n.target.__class__.__name__ == "Identifier":
                if n.target.name not in tainted and _reads_image(n.value, tainted, out_names):
                    tainted.add(n.target.name); grew = True
        if not grew:
            break
    return tainted


def _pow_is_safe(node) -> bool:
    """`pow(base, k)` is fp16-safe only when k is a small POSITIVE CONSTANT (grade's
    `pow(c, 0.4545)`): the output then stays bounded for a bounded base. A variable /
    negative / large exponent can blow a near-zero base up (measured: 1.7e3) or NaN a
    negative one — indistinguishable from grade's safe pow except by the exponent."""
    if len(node.args) < 2:
        return False
    exp = node.args[1]
    ec = exp.__class__.__name__
    if ec == "NumberLiteral":
        vals = [exp.value]
    elif ec == "VecConstructor" and all(a.__class__.__name__ == "NumberLiteral" for a in exp.args):
        vals = [a.value for a in exp.args]
    else:
        return False
    return bool(vals) and all(0.0 < v <= 4.0 for v in vals)


def _has_fp16_hazard(program, out_names) -> bool:
    """Decline fp16 if the program has a fp16-fragile function anywhere, an unsafe
    `pow`, a data-dependent branch (if/ternary/while — a fp16 value decides control
    flow), or an image-lineage comparison (an image thresholded directly, `@A.r>0.5`)."""
    tainted = _image_tainted_vars(program, out_names)
    for n in _walk(program):
        cls = n.__class__.__name__
        if cls == "FunctionCall":
            if n.name in _FP16_FRAGILE_FNS:
                return True
            if n.name in ("pow", "spow") and not _pow_is_safe(n):
                return True
        elif cls in ("IfElse", "TernaryOp", "WhileLoop"):
            return True  # a fp16 value steering control flow -> unstable output
        elif cls == "BinOp" and n.op in _COMPARE_OPS:
            # image-lineage comparison (integer for-loop counters `i < N` are not
            # image-tainted, so bounded loops stay eligible)
            if _reads_image(n.left, tainted, out_names) or _reads_image(n.right, tainted, out_names):
                return True
    return False


def apply_tf32_profile(enable: bool = True):
    """PR-LP6 — opt-in TF32 for fp32 matmul/conv on Ampere+ (sm_80+). Returns a restore
    callable. Default OFF. **No-op on Turing (sm_75)** — no TF32 hardware, so it's inert
    here (verified: results bit-identical). The 1.5–2.5x Ampere claim is NOT shipped as a
    planning input until measured on sm_80+; this ships the Turing-safe half only — a
    profile a host/user can flip, with an honest restore. It's a *precision* trade (TF32
    keeps run-to-run determinism), never a determinism trade."""
    prev_mm = torch.get_float32_matmul_precision()
    prev_cudnn = bool(getattr(torch.backends.cudnn, "allow_tf32", False))

    def restore():
        torch.set_float32_matmul_precision(prev_mm)
        try:
            torch.backends.cudnn.allow_tf32 = prev_cudnn
        except Exception:
            pass

    if enable:
        torch.set_float32_matmul_precision("high")
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    return restore


def resolve_auto_precision(program, spatial_px: int, device_type: str):
    """Return (precision, reason) for `precision="auto"`. precision is "fp16" only in
    the measured win region; "fp32" (with the declining reason) otherwise."""
    if device_type != "cuda":
        return "fp32", "auto->fp32: CPU (fp16 is slower on CPU)"
    if spatial_px < _MIN_FP16_PX:
        return "fp32", "auto->fp32: <1024^2 (fp16 gain only above)"
    if not is_tile_safe(program):
        return "fp32", "auto->fp32: sampling/fetch/reduction/scatter (fp16-unsafe)"
    if _has_fp16_hazard(program, _output_names(program)):
        return "fp32", "auto->fp32: discontinuous/domain fn or data branch (fp16-fragile)"
    return "fp16", "auto->fp16: smooth pointwise, no branch, >=1024^2 CUDA"
