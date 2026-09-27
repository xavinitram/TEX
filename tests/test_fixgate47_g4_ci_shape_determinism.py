"""FIX-GATE G4(b) (v0.47.0 Phase C, B4#2) -- the gate's CI-shape leg must not depend on
whatever toolchain happens to be installed on the `--ci-python` interpreter's box.

B4 confirmed by running: a `--ci-python` interpreter whose venv has a real MSVC install
(unlike the embedding host's usual embedded interpreter, which never finds one) makes
`tex_runtime.noise._can_inductor_compile("cpu")` return True there, so
`test_v031_noise_tiers.py::test_v031_noise_cold_frame_parity` reaches a REAL torch.compile
path this project has not characterized for stability under a coverage-tracing pytest run --
three otherwise-identical runs on that interpreter produced three different outcomes (a
clean pass, a stuck-eager tier, and a child-process crash). The gate's own VERDICT should
not depend on which box happens to have a compiler on PATH.

The fix: `TEX_GATE_NO_INDUCTOR=1` forces `_can_inductor_compile()` to answer False for every
device, unconditionally, before any probe -- `tools/gate.py::run_ci_shape` now sets it in
the subprocess environment, so THAT leg's own toolchain-dependent instability is removed at
the source, deterministically, regardless of `--ci-python`'s box. Never set by ComfyUI or
any production code path; it is a gate-only escape hatch.

Deliberately NOT set by `run_ci_exact`: that leg's entire, separately-tested purpose is to
run the CI workflow's OWN command shape VERBATIM (B4#9: its argv is already confirmed
byte-for-byte identical to `.github/workflows/tests.yml`'s own step). Real CI's Linux
runners always have gcc, so the live-compile path is plausibly reachable on CI too (B4's own
observation) -- forcing it off there would make `run_ci_exact` a WORSE mirror of CI, not a
better one. `run_ci_shape` carries no such "verbatim" promise; it exists to catch a host/CUDA
assumption, and its own verdict should not hinge on an incidental compiler installation.
"""
import importlib
import os

from helpers import SubTestResult

from TEX_Wrangle.tex_runtime import noise


def test_fixgate_g4_env_knob_forces_no_inductor_regardless_of_toolchain(r: SubTestResult):
    print("\n--- G4(b): TEX_GATE_NO_INDUCTOR=1 forces _can_inductor_compile() False ---")
    orig_env = os.environ.get("TEX_GATE_NO_INDUCTOR")
    orig_cache = dict(noise._inductor_available)
    try:
        # Prove the override wins even when the real probe WOULD say True -- fake a
        # findable compiler/Triton so the untouched code path is definitely reachable.
        import shutil as _shutil
        orig_which = _shutil.which
        _shutil.which = lambda name: "C:/fake/cl.exe" if name == "cl" else orig_which(name)
        try:
            noise._inductor_available.clear()
            os.environ["TEX_GATE_NO_INDUCTOR"] = "1"
            forced = noise._can_inductor_compile("cpu")
            assert forced is False, (
                f"TEX_GATE_NO_INDUCTOR=1 must force False even with a findable compiler, "
                f"got {forced}")

            noise._inductor_available.clear()
            del os.environ["TEX_GATE_NO_INDUCTOR"]
            unforced = noise._can_inductor_compile("cpu")
            assert unforced is True, (
                f"sanity: without the knob, a findable compiler must still read True, "
                f"got {unforced} -- otherwise this test proves nothing")
        finally:
            _shutil.which = orig_which
        r.ok("the env knob forces a deterministic False regardless of a findable toolchain")
    except Exception as e:
        r.fail("G4 env knob forces no-inductor", str(e))
    finally:
        noise._inductor_available.clear()
        noise._inductor_available.update(orig_cache)
        if orig_env is None:
            os.environ.pop("TEX_GATE_NO_INDUCTOR", None)
        else:
            os.environ["TEX_GATE_NO_INDUCTOR"] = orig_env


def test_fixgate_g4_knob_is_off_by_default(r: SubTestResult, monkeypatch):
    """ComfyUI-invisible: with the env var absent (the default, every real ComfyUI process),
    behaviour is byte-identical to before this ask -- the knob only ever activates when
    something explicitly sets it. This test controls its own env (rather than asserting on
    the ambient process environment) because a suite run under `run_ci_shape` itself sets
    the knob for its whole subprocess -- this row proves the ABSENT-knob behaviour, not
    which process it happens to run inside."""
    print("\n--- G4(b): with no env var set, behaviour is unchanged ---")
    try:
        monkeypatch.delenv("TEX_GATE_NO_INDUCTOR", raising=False)
        orig_cache = dict(noise._inductor_available)
        try:
            noise._inductor_available.clear()
            # Whatever this box's real answer is, it must come from the real probe, not a
            # hard-coded False -- just prove no exception and a bool comes back.
            result = noise._can_inductor_compile("cpu")
            assert isinstance(result, bool), result
        finally:
            noise._inductor_available.clear()
            noise._inductor_available.update(orig_cache)
        r.ok(f"absent the knob, _can_inductor_compile('cpu') reads its real probe: {result}")
    except Exception as e:
        r.fail("G4 knob off by default", str(e))


def test_fixgate_g4_run_ci_shape_sets_the_env_knob(r: SubTestResult):
    print("\n--- G4(b): run_ci_shape sets TEX_GATE_NO_INDUCTOR in the subprocess env ---")
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "gate.py")
        spec = importlib.util.spec_from_file_location("_fixgate47_g4_gate", path)
        gate = importlib.util.module_from_spec(spec)
        import sys
        sys.modules["_fixgate47_g4_gate"] = gate
        spec.loader.exec_module(gate)

        captured = []

        def spy_run(leg, argv, cwd, env_extra, scratch, verbose, expect_collect=True):
            captured.append(dict(env_extra))
            leg.rc, leg.summary = 0, "spy: not actually run"
            leg.failures, leg.collected, leg.failure_text = [], 1, {}
            return leg

        orig_run = gate._run
        orig_comfy_check = gate._ci_interpreter_can_import_comfy_api
        gate._run = spy_run
        gate._ci_interpreter_can_import_comfy_api = lambda ci_python: False
        try:
            fake_python = sys.executable   # any existing file; the real launch is spied out
            gate.run_ci_shape(fake_python, scratch=".", verbose=False)
        finally:
            gate._run = orig_run
            gate._ci_interpreter_can_import_comfy_api = orig_comfy_check

        assert len(captured) == 1, f"expected run_ci_shape to reach _run once, got {len(captured)}"
        assert captured[0].get("TEX_GATE_NO_INDUCTOR") == "1", (
            f"run_ci_shape's subprocess env must set TEX_GATE_NO_INDUCTOR=1, got {captured[0]}")
        r.ok("run_ci_shape sets TEX_GATE_NO_INDUCTOR=1 in its subprocess environment")
    except Exception as e:
        r.fail("G4 run_ci_shape sets the env knob", str(e))


def test_fixgate_g4_run_ci_exact_does_not_set_the_env_knob(r: SubTestResult):
    """`run_ci_exact`'s whole purpose is a VERBATIM CI command shape (B4#9); forcing the
    knob there would make it a worse mirror of real CI (whose Linux runners have gcc and
    plausibly reach the same live-compile path), not a better one -- see this file's own
    module docstring."""
    print("\n--- G4(b): run_ci_exact does NOT set the knob (deliberately) ---")
    try:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "tools", "gate.py")
        spec = importlib.util.spec_from_file_location("_fixgate47_g4_gate_b", path)
        gate = importlib.util.module_from_spec(spec)
        import sys
        sys.modules["_fixgate47_g4_gate_b"] = gate
        spec.loader.exec_module(gate)

        captured = []

        def spy_run(leg, argv, cwd, env_extra, scratch, verbose, expect_collect=True):
            captured.append(dict(env_extra))
            leg.rc, leg.summary = 0, "spy: not actually run"
            leg.failures, leg.collected, leg.failure_text = [], 1, {}
            return leg

        orig_run = gate._run
        orig_comfy_check = gate._ci_interpreter_can_import_comfy_api
        orig_missing = gate._ci_exact_missing_deps
        gate._run = spy_run
        gate._ci_interpreter_can_import_comfy_api = lambda ci_python: False
        gate._ci_exact_missing_deps = lambda ci_python: []
        try:
            fake_python = sys.executable
            gate.run_ci_exact(fake_python, scratch=".", verbose=False)
        finally:
            gate._run = orig_run
            gate._ci_interpreter_can_import_comfy_api = orig_comfy_check
            gate._ci_exact_missing_deps = orig_missing

        assert len(captured) == 1, f"expected run_ci_exact to reach _run once, got {len(captured)}"
        assert "TEX_GATE_NO_INDUCTOR" not in captured[0], (
            f"run_ci_exact must NOT set TEX_GATE_NO_INDUCTOR (it would break the verbatim "
            f"CI-command-shape contract B4#9 already confirmed), got {captured[0]}")
        r.ok("run_ci_exact stays a verbatim CI command shape, no knob added")
    except Exception as e:
        r.fail("G4 run_ci_exact keeps the verbatim shape", str(e))
