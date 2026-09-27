"""TRK-220 — the FIX-ROI O1 fp32-fold fix (v0.48.0 Phase C) reaches `tex_tiling.
_halo_tile_plan`'s memory-pressure tiling path, confirmed on CUDA, not just by reading the CPU-
only proof `test_fixroi_o1_fp32_branch_fold.py::test_o1_tiling_halo_plan_consumes_the_same_fix`
already gives (the ROI bug hunt finding 1's own "not confirmed by running" note for this exact path;
the tracker's TRK-220 row).

THE MECHANISM UNDER TEST. `_halo_tile_plan` (`tex_tiling.py`) is the planner that decides
whether a CUDA cook of a bounded-halo op (blur/erode/dilate — the class `is_tile_safe` refuses)
runs in horizontal strips under memory pressure or the TDR time cap; its halo radius comes
straight from `tex_roi.roi_plan`, the exact call the ROI bug hunt finding 1 showed could silently
under-report `halo=0` for a uniform (`$param`-only) `IfElse` condition built from an arithmetic
combination of two or more params, whenever the true value sits within half an fp32 ulp of the
comparison boundary — the double-precision fold and the runtime's own fp32 tensor evaluation
then disagree about which arm executes. `_halo_tile_plan` is NOT gated behind `TEX_ROI_EXEC`
(unlike the ROI-3 window feature): it runs on every default CUDA cook, so an under-sized halo
there under-sizes the strip overlap and stitches back a WRONG picture on a production path, not
a flagged-off one.

FORCING THE PATH DETERMINISTICALLY (not by exhausting VRAM). Mirrors PERF-6's own
`_Stubbed` test harness (`test_perf6_free_memory_once.py`): a fake `HostServices.
get_free_memory` answer, a patched `tex_memory.device_total_mem`, and a patched
`tex_tiling._torch_allocated` together make `_halo_tile_plan`'s pressure gate (`est > 0.25 *
free`, `est >= total // 8`) fire for a tiny, real cook — the SAME mechanism the OOM-ladder
tests already use, none of it touches real VRAM. Calibrated at a 256x256 canvas
(`estimate_peak_bytes` reads 1 MiB for this exact program at this size — verified by running):
fake `total=4 MiB`, `free=512 KiB` forces `n_strips=4`, `halo=12` deterministically every time.

WHAT EACH TEST PROVES:
  - `test_trk220_cpu_planner_twin` (CPU-only, no lease needed): the planner call itself, under
    forced pressure, with the EXACT B1 fp32-boundary repro program -- confirms `roi_plan`'s
    fixed halo (12, not 0) is what `_halo_tile_plan` computes its strip plan from, and that the
    forced-pressure harness reliably drives `n_strips >= 2` (the CPU twin of the CUDA row below;
    `_halo_tile_plan` itself early-returns off any non-"cuda" device string, so this calls it
    with a "cuda:0" device STRING over CPU tensors -- the same shape PERF-6's own golden sweep
    and TRK-166/TRK-83 already use to exercise this planner with no GPU).
  - `test_trk220_cuda_end_to_end_tiled_equals_whole_frame` (CUDA-gated, skips without one): runs
    the ACTUAL production cook path (`tex_engine.cook`, real CUDA tensors) twice -- once under
    the forced-pressure stub (drives `_run_default` into `run_tiled_halo`, confirmed by a call
    spy) and once with no stub at all (real free VRAM, whole-frame) -- and asserts
    `torch.equal` between the two. A wrong (under-sized) halo would show up here as a real pixel
    divergence at every strip seam, exactly the ROI bug hunt's own measured signature (max diff
    ~0.15) had the fix not been in place.

GPU discipline: this file buys no timing and asserts no wall-clock bound (PACE's law: only
timing needs the bench lease); it still stays off the GPU test unless CUDA is actually free of
another lane's use per the standing per-cook laptop-lease protocol, checked by the lane running
it, not by this file.
"""
from helpers import *

from TEX_Wrangle import tex_engine, tex_roi as _R, tex_tiling as _T, tex_memory
from TEX_Wrangle.tex_cache import get_cache
from TEX_Wrangle.tex_marshalling import infer_binding_type
from TEX_Wrangle.tex_runtime import host as host_mod

# FIX-ROI49 Q6 (R1#3): the EXACT B1/O1 fp32-boundary repro -- the same load-bearing literal
# floats and program text `test_fixroi_o1_fp32_branch_fold.py` defines, imported rather than
# copied so the two files' repros are PINNED to stay byte-identical. Copying the literals
# would let this file keep "confirming" a boundary case that no longer demonstrates the bug
# if the O1 file's own repro ever moved to a different boundary-case pair. The Python-double
# sum of _A + _B is 0.5000000111758709 (> 0.5 -- the WRONG, double-precision branch decision),
# while torch.float32(_A) + torch.float32(_B) is exactly 0.5 (not > 0.5) -- the runtime
# actually executes the `else` arm (the halo op), not the `then` arm (a plain, halo-0 copy).
from test_fixroi_o1_fp32_branch_fold import _A, _B, _PARAMS, _BLUR_CODE

_CANVAS = 256   # calibrated size: estimate_peak_bytes reads exactly 1 MiB here (verified by
                # running against this exact program) -- see module docstring.

# Calibrated fake device numbers (bytes): total=4 MiB, free=512 KiB. total // 8 = 512 KiB <
# est (1 MiB) -> the cheap gate does not skip; est (1 MiB) > 0.25 * free (128 KiB) -> pressure
# fires; n_mem = ceil(est / (0.25*free)) = ceil(1048576 / 131072) = 8, capped by
# max_strips = H // max(64, 4*halo) = 256 // 64 = 4 -> n_strips == 4, deterministically.
_FAKE_TOTAL = 4 * 1024 * 1024
_FAKE_FREE = 512 * 1024


class _FakeHost:
    """Just enough of the HostServices protocol for the planner -- PERF-6's own `_FakeHost`
    shape (test_perf6_free_memory_once.py), reused here rather than imported (that file's
    class is test-local, not a shared fixture)."""

    def __init__(self, free):
        self.free = float(free)

    def get_free_memory(self, device):
        return self.free

    def free_memory(self, amount, device):
        pass

    def is_oom(self, exc):
        return False

    def soft_empty_cache(self):
        pass

    def get_user_dir(self):
        return None

    def cancel_token(self):
        return None

    def raise_if_interrupted(self):
        pass


class _ForcedPressure:
    """Deterministically force `_halo_tile_plan`'s memory-pressure gate to fire, without
    touching real VRAM -- the OOM-ladder tests' own hook shape (PERF-6's `_Stubbed`): a fake
    host (fixed free-VRAM answer), a patched `tex_memory.device_total_mem` (fixed total), and
    a patched `tex_tiling._torch_allocated` (fixed 0, so a real CUDA run's own small
    allocations don't perturb the forced numbers). Restores all three on exit, and drops
    `tex_tiling`'s free-VRAM memo on both ends so neither a stale real reading nor the fake one
    leaks across the boundary."""

    def __enter__(self):
        self._saved_total = tex_memory.device_total_mem
        self._saved_alloc = _T._torch_allocated
        tex_memory.device_total_mem = lambda device: _FAKE_TOTAL
        _T._torch_allocated = lambda idx: 0
        host_mod.set_host_services(_FakeHost(_FAKE_FREE))
        _T.forget_free_memory()
        return self

    def __exit__(self, *exc):
        tex_memory.device_total_mem = self._saved_total
        _T._torch_allocated = self._saved_alloc
        host_mod.reset_host_services()
        _T.forget_free_memory()
        return False


def _compile_b1():
    """Compile the B1/O1 program against a `_CANVAS`-sized image binding, exactly as the
    engine would -- returns (program, fingerprint, binding_types)."""
    image = torch.zeros(1, _CANVAS, _CANVAS, 3)
    bindings = {"image": image, **_PARAMS}
    binding_types = {n: infer_binding_type(v) for n, v in bindings.items()}
    fp = get_cache().fingerprint(_BLUR_CODE, binding_types)
    program, tm, referenced, assigned, param_info, used = \
        get_cache().compile_tex(_BLUR_CODE, binding_types)
    return program, fp, binding_types


def test_trk220_cpu_planner_twin(r: SubTestResult):
    """CPU twin: `_halo_tile_plan` compares its `device` argument as a STRING (it never
    touches the device otherwise before returning a plan), so calling it with a "cuda:0"
    string over CPU tensors exercises the exact same planner logic `_halo_tile_plan` runs on
    real CUDA with -- the same no-GPU-needed shape PERF-6's own golden sweep and
    TRK-166/TRK-83 already rely on for this function."""
    print("\n--- TRK-220 CPU twin: _halo_tile_plan under forced pressure, B1 fp32 boundary ---")
    try:
        program, fp, binding_types = _compile_b1()
        image = torch.zeros(1, _CANVAS, _CANVAS, 3)
        bindings = {"image": image, **_PARAMS}

        _R.clear_roi_memo()
        plan = _R.roi_plan(_BLUR_CODE, _T._scalar_params(_PARAMS), binding_types)
        if not plan.executable or plan.halo != 12:
            r.fail("TRK-220 roi_plan premise", f"expected executable halo=12, got {plan}")
            return
        r.ok(f"roi_plan halo={plan.halo} (the O1 fix: NOT the double-precision-folded 0)")

        with _ForcedPressure():
            halo_plan = _T._halo_tile_plan(program, _BLUR_CODE, bindings, "cuda:0", 0, 4, fp,
                                           None, "fp32", binding_types)
        if halo_plan is None:
            r.fail("TRK-220 CPU planner twin",
                   "_halo_tile_plan returned None -- the forced-pressure harness failed to "
                   "drive the pressure gate (or the O1 fix regressed and halo<=0 refused it)")
            return
        n, narrow, halo = halo_plan
        if n < 2 or halo != 12 or "image" not in narrow:
            r.fail("TRK-220 CPU planner twin",
                   f"got n_strips={n}, halo={halo}, narrow={sorted(narrow)} -- expected "
                   f"n_strips>=2, halo=12, 'image' in narrow")
            return
        r.ok(f"forced pressure -> n_strips={n}, halo={halo}, narrow={sorted(narrow)}")
    except Exception as e:
        r.fail("TRK-220 CPU planner twin", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


def test_trk220_cuda_end_to_end_tiled_equals_whole_frame(r: SubTestResult):
    """CUDA confirmation: a real halo-tiled cook (forced by the SAME pressure stub, driving
    `_run_default` all the way into `run_tiled_halo`) equals a real whole-frame cook of the
    identical program, bit-exactly."""
    print("\n--- TRK-220 CUDA: a forced-pressure halo-tiled cook == whole-frame, torch.equal ---")
    if not torch.cuda.is_available():
        r.skip("TRK-220 CUDA halo pressure", "no CUDA on this box")
        return
    try:
        torch.manual_seed(220)
        image = torch.rand(1, _CANVAS, _CANVAS, 3, device="cuda")

        # Baseline: no stub at all -- real free VRAM, whole-frame (this canvas is tiny; no
        # genuine pressure at this size on any CUDA box this suite targets).
        _R.clear_roi_memo()
        base = tex_engine.cook(_BLUR_CODE, dict(_PARAMS, image=image.clone()),
                               device_mode="cuda").outputs["OUT"]

        # Forced pressure: same program, same image, spy on run_tiled_halo to confirm the
        # tiled route actually fired (not just that it was ELIGIBLE to).
        orig_rth = tex_memory.run_tiled_halo
        calls = []

        def _spy(*a, **kw):
            calls.append(1)
            return orig_rth(*a, **kw)

        tex_memory.run_tiled_halo = _spy
        try:
            _R.clear_roi_memo()
            with _ForcedPressure():
                tiled_res = tex_engine.cook(_BLUR_CODE, dict(_PARAMS, image=image.clone()),
                                            device_mode="cuda")
        finally:
            tex_memory.run_tiled_halo = orig_rth

        if not calls:
            r.fail("TRK-220 CUDA halo pressure",
                   "run_tiled_halo was never called -- the forced-pressure stub did not "
                   "drive the tiling route on this run; the comparison below would be "
                   "whole-frame-vs-whole-frame and prove nothing")
            return
        tiled = tiled_res.outputs["OUT"]
        if tuple(tiled.shape) != tuple(base.shape):
            r.fail("TRK-220 CUDA halo pressure",
                   f"shape mismatch: tiled {tuple(tiled.shape)} vs whole {tuple(base.shape)}")
            return
        if torch.equal(tiled, base):
            r.ok(f"CUDA: {len(calls)} halo-tiled call(s), torch.equal vs whole-frame "
                 f"(the fp32-fold fix holds under real memory-pressure tiling on CUDA)")
        else:
            md = (tiled.float() - base.float()).abs().max().item()
            mn = (tiled.float() - base.float()).abs().mean().item()
            r.fail("TRK-220 CUDA halo pressure",
                   f"tiled != whole-frame: max diff {md:.4e}, mean diff {mn:.4e} -- this is "
                   f"exactly the ROI bug hunt's own measured under-halo signature (~0.15 max diff) "
                   f"if the fp32-fold fix has regressed on this path")
    except Exception as e:
        r.fail("TRK-220 CUDA halo pressure", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()
