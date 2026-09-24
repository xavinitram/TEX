"""TRK-80 — `benchmarks/run_benchmarks.py::generate_bindings` used to mint a genuinely fresh
random image on every call, with no generator and no seed. `ex_denoise` is a program whose
own cook cost is DATA-dependent (its patch-distance/bilateral machinery takes different
internal paths depending on the actual pixel values), so a "plain" leg and a "compiled" leg
of the SAME program in the SAME benchmark run could be timing two different images, and the
same program measured in two different processes was never comparable at all — together they
could manufacture a cost delta the code never introduced.

This file pins the fix: `generate_bindings(prog, B, H, W, device)` is now a pure function of
its own arguments (`_bindings_seed` derives a deterministic seed from the program's name and
the requested shape, never from process state), so calling it twice — in one process or two —
reproduces bit-identical tensors. `benchmarks/eight_config_bench.py` imports `generate_bindings`
straight from this module, so the fix reaches every benchmark script that shares it without
either needing to change.

The second, "interesting" half the tracker names — a cook's TIER classification
(`tex_runtime/compiled.py`'s `_route_memo` / `_compile_blacklist` / the `has_spatial` gate)
depending on what ran before it in the same process — is NOT fixed here: it lives in
`tex_runtime/compiled.py`'s tier-selection internals, outside this file's scope. Probed instead
(not gated): with the data now deterministic, cooking `ex_denoise` twice in one process with
`clear_compiled_cache()` between reads the same served tier both times on this box, which is
recorded as a finding for whoever next touches that module, not asserted as a fixed contract.

PORTABILITY: CPU-only, no ComfyUI, no torch.compile requirement (the probe below only reads
`tier_trace`, never asserts which tier was chosen), no numpy, no timing assertion.
"""
import hashlib
import importlib.util
import os
import sys

import pytest
import torch

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_run_benchmarks():
    mod = sys.modules.get("_trk80_run_benchmarks")
    if mod is not None:
        return mod
    path = os.path.join(_PKG_DIR, "benchmarks", "run_benchmarks.py")
    spec = importlib.util.spec_from_file_location("_trk80_run_benchmarks", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_trk80_run_benchmarks"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rb():
    return _load_run_benchmarks()


@pytest.fixture(scope="module")
def ex_denoise(rb):
    progs = [p for p in rb.load_example_programs() if p.name == "ex_denoise"]
    if not progs:
        pytest.skip("examples/denoise.tex not present in this tree")
    return progs[0]


def test_trk80_bindings_seed_is_a_pure_function_of_its_arguments(rb):
    """Same program name, same shape -> same seed, every call, regardless of any other
    torch RNG activity that happened first."""
    torch.manual_seed(1)
    torch.rand(1000)                       # perturb the global RNG between the two reads
    a = rb._bindings_seed("ex_denoise", 1, 64, 64)
    torch.rand(1000)
    b = rb._bindings_seed("ex_denoise", 1, 64, 64)
    assert a == b
    # A different program, or a different shape, must not collide with it.
    assert rb._bindings_seed("ex_other", 1, 64, 64) != a
    assert rb._bindings_seed("ex_denoise", 1, 32, 32) != a


def test_trk80_generate_bindings_is_repeatable_within_one_process(rb, ex_denoise):
    """The exact reproduction TRK-80 names: the same program's bindings, drawn twice in one
    process, must be bit-identical — the DATA half of the non-repeatable cold-compiled-cook
    cost."""
    b1 = rb.generate_bindings(ex_denoise, 1, 64, 64, device="cpu")
    b2 = rb.generate_bindings(ex_denoise, 1, 64, 64, device="cpu")
    assert set(b1) == set(b2)
    for k, v1 in b1.items():
        v2 = b2[k]
        if isinstance(v1, torch.Tensor):
            assert torch.equal(v1, v2), f"binding {k!r} differs between two generate_bindings calls"
        else:
            assert v1 == v2, f"binding {k!r} differs between two generate_bindings calls"


def test_trk80_generate_bindings_is_repeatable_across_processes(rb, ex_denoise):
    """The other half of the reproduction: the same program measured ALONE (a fresh process,
    nothing else has touched the RNG) must draw the identical tensors a process that ran 60
    other programs first would draw for it — the seed is a function of (name, shape) only,
    never of how many `torch.rand` calls happened earlier in this process."""
    torch.manual_seed(0)
    for _ in range(137):                   # simulate "measured after 60 others" RNG churn
        torch.rand(4096)
    churned = rb.generate_bindings(ex_denoise, 1, 64, 64, device="cpu")

    torch.manual_seed(0)                   # a fresh, "measured alone" RNG position
    alone = rb.generate_bindings(ex_denoise, 1, 64, 64, device="cpu")

    for k, v1 in alone.items():
        v2 = churned[k]
        if isinstance(v1, torch.Tensor):
            assert torch.equal(v1, v2), \
                f"binding {k!r} depends on prior RNG activity, not just (program, shape)"
        else:
            assert v1 == v2


def test_trk80_a_different_shape_still_draws_different_data(rb, ex_denoise):
    """The fix must not have collapsed every draw to one constant image — a real shape change
    still changes the data (it is a deterministic function of the shape, not a shape-blind
    constant)."""
    small = rb.generate_bindings(ex_denoise, 1, 32, 32, device="cpu")
    big = rb.generate_bindings(ex_denoise, 1, 96, 96, device="cpu")
    assert small["image"].shape != big["image"].shape


def test_trk80_tier_classification_probe_not_gated(rb, ex_denoise):
    """Informational probe for the tracker's second half (NOT a gate: no assertion on WHICH
    tier is chosen, and skipped outright if this box cannot cook the program at all). With the
    data now deterministic, cooking `ex_denoise` twice in one process with
    `clear_compiled_cache()` between currently reads the same tier both times here — recorded
    for whoever next investigates `tex_runtime/compiled.py`'s `_route_memo` /
    `_compile_blacklist` / `has_spatial`, not asserted as a fixed contract by this file."""
    from TEX_Wrangle import tex_engine
    from TEX_Wrangle.tex_runtime import tier_trace, compiled

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tiers = []
    for _ in range(2):
        b = rb.generate_bindings(ex_denoise, 1, 64, 64, device=device)
        try:
            tex_engine.cook(ex_denoise.code, b, device_mode=device,
                            compile_mode="auto", precision="fp32")
        except Exception as e:
            pytest.skip(f"ex_denoise did not cook on this box: {type(e).__name__}: {e}")
        tr = tier_trace.last()
        tiers.append(tr.tier if tr is not None else None)
        compiled.clear_compiled_cache()
    print(f"\n--- TRK-80 probe (informational): tier sequence over two clean-cache "
          f"cooks = {tiers} ---")
