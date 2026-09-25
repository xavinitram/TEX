"""FUSEDDEV-46 — a fused chain forced through the `torch_compile` tier crashed on CUDA:
"Expected all tensors to be on the same device, but found at least two devices, cuda:0
and cpu!" TEX's own fallback net caught it, blacklisted the fused program, and silently
re-ran the interpreter — so the bug was invisible except as a lost speed-up (found by the
COMPILE-M3 measurement lane, `benchmarks/compile_modes_bench.py --fused`).

ROOT CAUSE: codegen's `_get_param_local` stages a `$param` binding with a bare
`_torch.as_tensor(value)` — deliberately CPU, by design (a sync-reading builtin's arg,
e.g. `gauss_blur`'s sigma, wants exactly that CPU scalar, never a device readback;
`compiled.py::_params_on_device`'s own docstring records "always placing measured
slower"). ATen lets a 0-dim CPU tensor mix with a CUDA operand ONLY while it stays 0-dim.
Two call sites turn such a value into a REAL tensor at the image's shape without ever
looking at its device: `_broadcast_pair`'s rank-mismatch expand (`interpreter.py`, shared
by the interpreter and every codegen `_bp` call) and `_ensure_spatial`'s `dim() == 0`
branch (the same file, codegen's `_es`) — the exact instant a genuinely-0-dim CPU tensor
stops being exempt. Fixed at both, at the point of expansion (`_co_locate` /
`_ensure_spatial`'s new `device=` kwarg), so nothing is moved unless it is about to be
used incorrectly, and nothing on the already-correct path (same device, or a param that
never reaches either function — `gauss_blur`'s sigma) pays anything beyond one `is`/`!=`
check. NOT fusion-specific: the identical crash reproduces on a plain single-node cook
whose codegen reaches either branch on CUDA (`test_fuseddev46_single_node_...` below) —
today, a real ComfyUI user's `compile_mode="auto"` or `"torch_compile"` on such a program
silently demotes to the interpreter, with no diagnostic reaching the node.

ComfyUI-invisible because: the DEFAULT path (`compile_mode="none"`) never reaches codegen
here — the interpreter binds every value on the cook device already (PERF-2) — and every
opt-in compiled path that DID reach this crash already had a "never hard-fail the node"
fallback net; this fix only removes the silent demotion, it does not change what a cook
that never hit the bug computes or how fast it runs.
"""
from helpers import *

from TEX_Wrangle import tex_engine, tex_fusion
from TEX_Wrangle.tex_runtime import compiled as tex_compiled

_CUDA = torch.cuda.is_available()

# The COMPILE-M3 shape, trimmed to the minimum that still crosses
# `compiled._COMPILE_OP_THRESHOLD` (8) so the fused program actually reaches codegen
# instead of `_plain_execute` — three upstream `$param` grade stages (which is where the
# `_bp`/`_es` un-deviced-constant expansion happens) feeding a `gauss_blur` terminal
# (a stdlib call, so this routes through `compiled.py`'s eager codegen adapter
# (`_codegen_exec_eager`) and never touches Inductor/Triton at all — the one skip guard
# below (`_CUDA`) is therefore also the "no Triton" guard: this row needs a real CUDA
# device to have a second device to mismatch against, and nothing else).
_STAGE0 = "@OUT = vec4(@IN.rgb * $exposure, 1.0);"
_STAGE1 = "@OUT = vec4(max(@IN.rgb - vec3($black), vec3(0.0)), 1.0);"
_STAGE2 = "@OUT = vec4(spow(@IN.rgb, vec3($gamma)), 1.0);"
_TERMINAL = "@OUT = gauss_blur(@IN, $sigma);"


def _fused_spec():
    return {"stages": [
        {"code": _STAGE0, "image_input": "IN", "params": {"exposure": 1.05}},
        {"code": _STAGE1, "image_input": "IN", "params": {"black": 0.02}},
        {"code": _STAGE2, "image_input": "IN", "params": {"gamma": 0.95}},
    ], "terminal_image_input": "IN"}


def test_fuseddev46_fused_torch_compile_cuda_stays_on_device(r: SubTestResult):
    print("\n--- FUSEDDEV-46: fused chain + torch_compile on CUDA stays on one device ---")
    if not _CUDA:
        r.skip("FUSEDDEV-46 fused",
              "no CUDA: the cuda:0/cpu mismatch needs a real second device to mismatch "
              "against (a CPU cook has only one), and this chain's terminal is a stdlib "
              "call (gauss_blur) that routes through the eager codegen adapter without "
              "ever reaching Inductor/Triton, so no separate no-Triton guard applies")
        return
    try:
        spec = _fused_spec()
        src = make_img(1, 32, 32, 3, seed=46).to("cuda")
        bindings = {"IN": src, "sigma": 1.5}
        fused_fp = tex_fusion.fused_fingerprint(spec, _TERMINAL, dict(bindings),
                                                _infer_binding_type)
        clear_compiled_cache()

        out_none = tex_engine.cook(_TERMINAL, dict(bindings), chain_payload=spec,
                                   device_mode="cuda", precision="fp32",
                                   compile_mode="none", cancel=None)
        out_tc = tex_engine.cook(_TERMINAL, dict(bindings), chain_payload=spec,
                                 device_mode="cuda", precision="fp32",
                                 compile_mode="torch_compile", cancel=None)
        a, b = out_none.outputs["OUT"], out_tc.outputs["OUT"]

        if fused_fp is not None and fused_fp in tex_compiled._compile_blacklist:
            r.fail("FUSEDDEV-46 blacklisted",
                  f"the fused program (fp={fused_fp}) was blacklisted from compiling — "
                  f"it crashed instead of running under torch_compile")
            return
        if b.device != a.device:
            r.fail("FUSEDDEV-46 device", f"torch_compile output landed on {b.device}, "
                  f"expected {a.device}")
            return
        if not torch.equal(a, b):
            maxdiff = (a.double() - b.double()).abs().max().item()
            r.fail("FUSEDDEV-46 bit-exact",
                  f"torch_compile diverged from the interpreter, maxdiff={maxdiff:.3e}")
            return
        r.ok("fused chain runs on CUDA under torch_compile, bit-identical to the "
             "interpreter, and the fingerprint is not blacklisted")
    except Exception as e:
        r.fail("FUSEDDEV-46 fused crashed", f"{type(e).__name__}: {e}")


def test_fuseddev46_single_node_torch_compile_cuda_stays_on_device(r: SubTestResult):
    """Point 4 of the ask: the SAME crash on a plain, unfused single-node cook — proving
    the bug (and the fix) live in shared codegen machinery, not in fusion's splice."""
    print("\n--- FUSEDDEV-46: a single (unfused) node + torch_compile on CUDA, too ---")
    if not _CUDA:
        r.skip("FUSEDDEV-46 single-node",
              "no CUDA: same reason as the fused row above — no second device to "
              "mismatch against, and this program's gauss_blur call keeps it off "
              "Inductor/Triton entirely")
        return
    try:
        code = ("@OUT = vec4(clamp(@IN.rgb + ($amount) * (@IN.rgb - gauss_blur(@IN, 2.0).rgb), "
               "vec3(0.0), vec3(1.0)), 1.0);")
        bindings = {"IN": make_img(1, 32, 32, 3, seed=461).to("cuda"), "amount": 0.6}
        clear_compiled_cache()
        out_none = tex_engine.cook(code, dict(bindings), device_mode="cuda",
                                   precision="fp32", compile_mode="none", cancel=None)
        out_tc = tex_engine.cook(code, dict(bindings), device_mode="cuda",
                                 precision="fp32", compile_mode="torch_compile", cancel=None)
        a, b = out_none.outputs["OUT"], out_tc.outputs["OUT"]
        if not torch.equal(a, b):
            maxdiff = (a.double() - b.double()).abs().max().item()
            r.fail("FUSEDDEV-46 single-node bit-exact",
                  f"torch_compile diverged from the interpreter, maxdiff={maxdiff:.3e}")
            return
        r.ok("a plain single-node cook runs on CUDA under torch_compile, bit-identical "
             "to the interpreter")
    except Exception as e:
        r.fail("FUSEDDEV-46 single-node crashed", f"{type(e).__name__}: {e}")


def test_fuseddev46_broadcast_pair_and_ensure_spatial_co_locate(r: SubTestResult):
    """Non-skip twin (helpers.devices()'s "a loop, not a skip" idiom): asserts, WITHOUT
    compiling anything, that every constant `_broadcast_pair`/`_ensure_spatial` expand
    lands on the SAME device as the tensor it's paired with. On a CPU-only box this is
    the (still real, still worth pinning) same-device case; wherever CUDA is present it
    also exercises the exact cross-device pairing that crashed (a bare, un-deviced 0-dim
    tensor — precisely what `_get_param_local`'s `as_tensor(value)` produces — against a
    device-resident image), never skipped."""
    print("\n--- FUSEDDEV-46 twin: _broadcast_pair/_ensure_spatial always co-locate ---")
    try:
        for dev in devices():
            img = make_img(1, 4, 4, 3, seed=462).to(dev)
            # A "$param"-shaped captured constant: exactly what _get_param_local emits —
            # torch.as_tensor(value) with NO device, always CPU regardless of `dev`.
            const = torch.as_tensor(0.6)

            a, b = _broadcast_pair(img, const)
            if a.device != img.device or b.device != img.device:
                r.fail(f"FUSEDDEV-46 twin _broadcast_pair ({dev})",
                      f"expanded pair landed on ({a.device}, {b.device}), expected "
                      f"both on {img.device}")
                continue
            (a * b)   # must not raise "Expected all tensors to be on the same device"

            spatial = tuple(img.shape[:-1])
            es = _ensure_spatial(const, spatial, device=img.device)
            if es.device != img.device:
                r.fail(f"FUSEDDEV-46 twin _ensure_spatial ({dev})",
                      f"expanded to {es.device}, expected {img.device}")
                continue
            (img[..., 0] * es)   # same check, the _es (channel-fill) shape

            # Back-compat: device=None (every caller before this ask) must still return
            # the value UNMOVED — additive, not a default-path behaviour change.
            es_default = _ensure_spatial(torch.as_tensor(0.6), spatial)
            if es_default.device != torch.device("cpu"):
                r.fail(f"FUSEDDEV-46 twin _ensure_spatial default ({dev})",
                      f"device=None moved a CPU tensor to {es_default.device} — "
                      f"the pre-existing interpreter call sites must be untouched")
                continue
        r.ok(f"_broadcast_pair and _ensure_spatial co-locate their expanded operands "
             f"on every available device ({devices()})")
    except Exception as e:
        r.fail("FUSEDDEV-46 twin crashed", f"{type(e).__name__}: {e}")
