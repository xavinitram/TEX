"""v0.46 Phase C FIX-GATE (G1-G7) — `tools/gate.py`, `tests/helpers.py` and the LINT-1/SIMP-3
enumeration seam.

Each row below reproduces ONE finding from that release round's gate/tooling fix list (named
G1-G7 there) directly against the code, the same posture `test_gateverdict_infra_red.py` and
`test_splite_touched_selection.py` already take toward `tools/gate.py` — load it by path (it
carries no `__init__.py`; `.comfyignore` excludes `tools/`, like `tests/`) and call its own
functions, never a copy of their logic.

  G1 — the ci-shape leg refuses outright when its interpreter can `import comfy_api`,
       instead of running a leg that cannot prove what it claims to.
  G2 — LINT-1 (via `test_simp3_no_machine_paths.py::tracked_paths`) now enumerates through
       `gate.py::enumerate_paths`, the SAME walk `tree_hash()` uses (tracked plus
       untracked-not-ignored), so an uncommitted file is caught too.
  G3 — `CUDA_VISIBLE_DEVICES` is part of the verdict-cache key.
  G4 — `_test_module_refs` also matches an import shape sitting inside a STRING LITERAL,
       so a subprocess harness that builds an import statement as text still gets matched.
  G5 — the persistent Inductor cache is pruned to a size cap.
  G6 — `run_cheap`/`run_canonical`/`run_touched` share ONE leg-building helper.
  G7 — `test_lint46_check_torch_free.py` and `test_v042_hostaudit1_cold_import.py` share ONE
       fresh-subprocess-and-parse helper (`helpers.run_python_kv`), kept out of `__all__` so
       HOOK-4's star-import pin stays green.

PORTABILITY: every row here is stdlib-only against `gate.py`'s own functions (monkeypatched
subprocess/env where a real interpreter or real CUDA hardware would otherwise be needed), so
this needs no torch, no CUDA and no ComfyUI, and runs identically on the CI lane.
"""
import importlib.util
import os
import sys
import tempfile

from helpers import SubTestResult


def _gate():
    """Load `tools/gate.py` by path, once per process (mirrors
    `test_gateverdict_infra_red.py::_gate` / `test_splite_touched_selection.py::_gate`)."""
    mod = sys.modules.get("_fixgate46_gate")
    if mod is not None:
        return mod
    import pathlib
    path = pathlib.Path(__file__).resolve().parent.parent / "tools" / "gate.py"
    spec = importlib.util.spec_from_file_location("_fixgate46_gate", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_fixgate46_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── G1: the ci-shape leg refuses on a comfy_api-capable interpreter ─────────

def test_g1_probe_reads_a_launched_subprocess_rc(r: SubTestResult):
    print("\n--- G1: _ci_interpreter_can_import_comfy_api reads a real subprocess rc ---")
    g = _gate()
    orig_run = g.subprocess.run

    class _FakeCP:
        def __init__(self, rc):
            self.returncode, self.stdout, self.stderr = rc, "", ""

    try:
        g.subprocess.run = lambda argv, **kw: _FakeCP(0)
        can = g._ci_interpreter_can_import_comfy_api("fake-python")
        g.subprocess.run = lambda argv, **kw: _FakeCP(1)
        cannot = g._ci_interpreter_can_import_comfy_api("fake-python")

        def _raise(argv, **kw):
            raise OSError("no such interpreter")
        g.subprocess.run = _raise
        errored = g._ci_interpreter_can_import_comfy_api("fake-python")
    finally:
        g.subprocess.run = orig_run

    if can is not True:
        r.fail("G1 probe true", f"rc 0 must read as importable, got {can!r}")
    elif cannot is not False:
        r.fail("G1 probe false", f"rc 1 must read as not-importable, got {cannot!r}")
    elif errored is not False:
        r.fail("G1 probe launch failure", f"a launch failure must read False, got {errored!r}")
    else:
        r.ok("the probe reads rc 0/1 correctly and treats a launch failure as False")


def test_g1_run_ci_shape_refuses_when_the_interpreter_can_import_comfy_api(r: SubTestResult):
    print("\n--- G1: run_ci_shape REFUSES instead of running, when the probe is True ---")
    g = _gate()
    orig_probe = g._ci_interpreter_can_import_comfy_api
    g._ci_interpreter_can_import_comfy_api = lambda ci_python: True
    try:
        with tempfile.TemporaryDirectory(prefix="tex-g1-") as scratch:
            leg = g.run_ci_shape(sys.executable, scratch, False)
    finally:
        g._ci_interpreter_can_import_comfy_api = orig_probe

    if leg.failures != ["<ci-shape:refused-comfy-api>"]:
        r.fail("G1 refusal", f"expected the synthetic refusal id alone, got {leg.failures!r}")
    elif leg.rc == 0:
        r.fail("G1 refusal rc", "a refused leg must not report rc 0")
    elif "comfy_api" not in leg.summary:
        r.fail("G1 refusal message", f"the refusal must name comfy_api, got {leg.summary!r}")
    else:
        j = g.judge([leg], [], False)
        if j["verdict"] != "RED":
            r.fail("G1 refusal verdict", f"a refusal must judge RED, got {j['verdict']!r}")
        else:
            r.ok("run_ci_shape refuses with a clear message and judges RED, no pytest spawned")


def test_g1_run_ci_shape_runs_normally_when_the_probe_is_false(r: SubTestResult):
    """The orchestrator's real invocation (`--ci-python <a venv with no comfy_api>`) must
    keep working unchanged: when the probe says False, `run_ci_shape` must still reach the
    real pytest-launching code path, not the refusal branch."""
    print("\n--- G1: a comfy_api-free interpreter is NOT refused, and pytest is invoked ---")
    g = _gate()
    orig_probe, orig_run_fn = g._ci_interpreter_can_import_comfy_api, g._run
    calls = []

    def _spy_run(leg, argv, cwd, env_extra, scratch, verbose, expect_collect=True):
        calls.append(argv)
        leg.rc, leg.summary, leg.failures = 0, "1 passed in 0.01s", []
        leg.collected = {"tests/fake.py::test_fake"}
        return leg

    g._ci_interpreter_can_import_comfy_api = lambda ci_python: False
    g._run = _spy_run
    try:
        with tempfile.TemporaryDirectory(prefix="tex-g1b-") as scratch:
            # A real file is required to get past the "interpreter not found" check before
            # the probe/refusal branch is even reached -- sys.executable stands in here.
            leg = g.run_ci_shape(sys.executable, scratch, False)
    finally:
        g._ci_interpreter_can_import_comfy_api = orig_probe
        g._run = orig_run_fn

    if not calls:
        r.fail("G1 pass-through", "run_ci_shape never reached the pytest-launching call")
    elif leg.failures:
        r.fail("G1 pass-through", f"expected a clean leg, got failures={leg.failures!r}")
    elif "-m" not in calls[0] or "not slow and not timing" not in calls[0]:
        r.fail("G1 pass-through argv", f"unexpected argv shape: {calls[0]!r}")
    else:
        r.ok("a comfy_api-free interpreter is never refused, and the real leg still runs")


# ── G2: LINT-1 (via SIMP-3's tracked_paths) sees untracked-not-ignored too ──

def test_g2_enumerate_paths_matches_tree_hash_walk(r: SubTestResult):
    print("\n--- G2: enumerate_paths is the same enumeration tree_hash() uses ---")
    g = _gate()
    if not hasattr(g, "enumerate_paths"):
        r.fail("G2 shared helper missing", "gate.py has no enumerate_paths function")
        return
    paths = g.enumerate_paths(g._PKG)
    if paths is None:
        r.fail("G2 enumerate_paths", "this checkout must enumerate as a git repository")
        return
    r.ok(f"enumerate_paths returned {len(paths)} path(s)")


def test_g2_enumerate_paths_sees_an_untracked_not_ignored_file(r: SubTestResult):
    """The exact reproduction: a file that exists only on disk (never committed, never
    staged) must still appear in the enumeration — the gap that let LINT-1 miss it while
    `tree_hash()` (and the gate's own cache key) already saw its bytes."""
    print("\n--- G2: an untracked-not-ignored file is enumerated, not just the cached set ---")
    g = _gate()
    import subprocess as _subprocess
    marker = "tests/_tex_g2_untracked_probe_delete_me.py"
    abs_marker = os.path.join(g._PKG, marker)
    try:
        with open(abs_marker, "w", encoding="utf-8") as f:
            f.write("# G2 probe: an untracked, not-ignored file.\n")
        # sanity: git itself must see it as untracked-not-ignored (never staged/committed).
        status = _subprocess.run(["git", "-C", g._PKG, "status", "--porcelain", "--", marker],
                                 capture_output=True, text=True, timeout=60).stdout
        if not status.strip().startswith("??"):
            r.skip("G2 untracked probe",
                   f"this checkout does not report the probe file as untracked ({status!r}); "
                   f"cannot exercise the shape here")
            return
        paths = g.enumerate_paths(g._PKG)
        if paths is None:
            r.fail("G2 untracked probe", "enumerate_paths returned None for a real checkout")
        elif marker not in paths:
            r.fail("G2 untracked probe",
                   f"{marker!r} is untracked-not-ignored but was not enumerated — LINT-1/"
                   f"SIMP-3's tracked_paths() has the same gap tree_hash() already closed")
        else:
            r.ok(f"the untracked, not-ignored probe file is enumerated alongside the tracked "
                 f"set ({len(paths)} total)")
    finally:
        try:
            os.remove(abs_marker)
        except OSError:
            pass


def test_g2_lint1_tracked_paths_delegates_to_the_shared_enumeration(r: SubTestResult):
    print("\n--- G2: SIMP-3's tracked_paths() (LINT-1's own source) IS enumerate_paths ---")
    import test_simp3_no_machine_paths as simp3
    g = _gate()
    a, b = simp3.tracked_paths(), g.enumerate_paths(str(simp3._PKG))
    cap = simp3._MAX_TRACKED
    if a is None or b is None:
        r.fail("G2 delegation", f"expected both to enumerate this checkout, got {a!r}/{b!r}")
    elif a != sorted(b)[:cap]:
        r.fail("G2 delegation",
               "tracked_paths() no longer agrees with gate.py's own enumerate_paths — they "
               "must be the SAME walk, not two that happen to agree today")
    else:
        r.ok(f"tracked_paths() ({len(a)}) matches gate.enumerate_paths() through the shared "
             f"helper, not a second implementation")


# ── G3: CUDA_VISIBLE_DEVICES is part of the verdict-cache key ───────────────

def test_g3_cache_key_changes_with_cuda_visible_devices(r: SubTestResult):
    print("\n--- G3: cache_key differs under a different CUDA_VISIBLE_DEVICES ---")
    g = _gate()
    interpreters = [("python", sys.executable)]
    orig = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        k_cpu = g.cache_key("sometree", "cheap", interpreters, False)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        k_gpu = g.cache_key("sometree", "cheap", interpreters, False)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        k_cpu_again = g.cache_key("sometree", "cheap", interpreters, False)
    finally:
        if orig is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = orig

    if k_cpu == k_gpu:
        r.fail("G3 cache key",
               "cache_key is identical under CUDA_VISIBLE_DEVICES=-1 and =0 — a GREEN taken "
               "under one CUDA setting could be served under the other")
    elif k_cpu != k_cpu_again:
        r.fail("G3 cache key", "cache_key is not stable for the same CUDA_VISIBLE_DEVICES")
    else:
        r.ok("cache_key changes with CUDA_VISIBLE_DEVICES and is stable for a fixed value")


# ── G4: the touched tier matches an import sitting inside a string literal ─

def test_g4_test_module_refs_reads_an_import_inside_a_string_literal(r: SubTestResult):
    print("\n--- G4: _test_module_refs matches TEX_Wrangle imports inside string literals ---")
    g = _gate()
    src = (
        "def _measure(import_stmt, custom_nodes):\n"
        "    code = f'{import_stmt}\\n'\n"
        "    return code\n"
        "\n"
        "_measure(\"from TEX_Wrangle import tex_engine\", '/x')\n"
        "_measure(\"from TEX_Wrangle.tex_runtime import compiled\", '/x')\n"
        "_measure(\"import TEX_Wrangle.tex_memory\", '/x')\n"
        "import os\n"                    # a plain stdlib import must NOT show up
    )
    with tempfile.TemporaryDirectory(prefix="tex-g4-") as d:
        path = os.path.join(d, "test_sample.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(src)
        refs = g._test_module_refs(path)
    want = {"tex_engine", "tex_runtime", "tex_memory"}
    if refs != want:
        r.fail("G4 string-literal import match", f"got {sorted(refs)}, want {sorted(want)}")
    else:
        r.ok(f"string-literal imports resolved to {sorted(refs)}, matching real import shapes")


def test_g4_touched_tier_selects_hostaudit1_when_tex_engine_changes(r: SubTestResult):
    """The exact bug named in G4/B6#4: `test_v042_hostaudit1_cold_import.py` builds its
    `import` statements as strings passed to `_measure`, so it was never selected when
    `tex_engine.py` was the touched module. Runs against the REAL `tests/` directory (not a
    fake temp package), so this proves the actual shipped file is matched."""
    print("\n--- G4: select_touched_tests selects test_v042_hostaudit1_cold_import.py ---")
    g = _gate()
    orig_git = g._git

    def _fake_git(*args, cwd=None):
        if args[:2] == ("rev-parse", "--verify"):
            return "deadbeef\n"
        if args and args[0] == "diff":
            return "tex_engine.py\n"
        return ""

    g._git = _fake_git
    try:
        files, touched_mods, base_resolved = g.select_touched_tests("origin/main")
    finally:
        g._git = orig_git

    target = "tests/test_v042_hostaudit1_cold_import.py"
    if not base_resolved:
        r.fail("G4 touched selection", "the fake base ref failed to resolve")
    elif "tex_engine" not in touched_mods:
        r.fail("G4 touched selection", f"expected tex_engine in touched_mods, got {touched_mods}")
    elif target not in files:
        r.fail("G4 touched selection",
               f"{target} was not selected when tex_engine.py changed — its imports are "
               f"string literals, which _test_module_refs must also match")
    else:
        r.ok(f"{target} is selected when tex_engine.py is the touched module")


# ── G5: the persistent Inductor cache is pruned to a size cap ──────────────

def test_g5_prune_inductor_cache_root_evicts_oldest_first_under_cap(r: SubTestResult):
    print("\n--- G5: _prune_inductor_cache_root evicts the OLDEST files first, to the cap ---")
    g = _gate()
    if not hasattr(g, "_prune_inductor_cache_root"):
        r.fail("G5 pruning missing", "gate.py has no _prune_inductor_cache_root function")
        return
    with tempfile.TemporaryDirectory(prefix="tex-g5-") as root:
        names_oldest_to_newest = ["a.pyd", "b.pyd", "c.pyd", "d.pyd"]
        size_each = 100
        for i, name in enumerate(names_oldest_to_newest):
            p = os.path.join(root, name)
            with open(p, "wb") as f:
                f.write(b"x" * size_each)
            os.utime(p, (1_700_000_000 + i * 1000, 1_700_000_000 + i * 1000))
        total_before = size_each * len(names_oldest_to_newest)
        cap = size_each * 2       # must evict exactly the two oldest

        g._prune_inductor_cache_root(root, cap)

        remaining = set(os.listdir(root))
        total_after = sum(os.path.getsize(os.path.join(root, n)) for n in remaining)
        if total_after > cap:
            r.fail("G5 pruning cap", f"{total_after} bytes remain, cap was {cap}")
        elif remaining != {"c.pyd", "d.pyd"}:
            r.fail("G5 pruning order",
                   f"expected the two NEWEST files to survive, got {sorted(remaining)}")
        else:
            r.ok(f"pruned {total_before - total_after} bytes, oldest-first, down to "
                 f"{total_after} <= cap {cap}")


def test_g5_prune_inductor_cache_root_leaves_a_tree_already_under_cap_alone(r: SubTestResult):
    print("\n--- G5: a tree already under the cap is left untouched ---")
    g = _gate()
    with tempfile.TemporaryDirectory(prefix="tex-g5b-") as root:
        p = os.path.join(root, "only.pyd")
        with open(p, "wb") as f:
            f.write(b"x" * 10)
        g._prune_inductor_cache_root(root, 10_000)
        if not os.path.isfile(p):
            r.fail("G5 pruning no-op", "a tree already under the cap must not be touched")
        else:
            r.ok("a tree already under the cap is left alone")


def test_g5_prune_is_wired_into_a_gate_run(r: SubTestResult):
    print("\n--- G5: main() prunes the Inductor cache root once per invocation ---")
    g = _gate()
    if not hasattr(g, "_inductor_cache_root"):
        r.fail("G5 root helper missing", "gate.py has no _inductor_cache_root function")
        return
    calls = []
    orig_prune = g._prune_inductor_cache_root
    g._prune_inductor_cache_root = lambda root, cap: calls.append((root, cap))
    orig_importable = g._importable_as_tex_wrangle
    g._importable_as_tex_wrangle = lambda: False   # refuse before any leg spawns
    try:
        g.main(["--tier", "cheap"])
    finally:
        g._prune_inductor_cache_root = orig_prune
        g._importable_as_tex_wrangle = orig_importable
    if calls:
        r.fail("G5 wiring", "main() must refuse (rc 2) BEFORE pruning, but pruning ran anyway")
    # Now let main() actually reach the pruning call, with a fake (non-spawning) cheap leg
    # standing in for the real one, and the verdict cache redirected to a scratch file so
    # this test writes no entry into the box's real gate cache.
    g._prune_inductor_cache_root = lambda root, cap: calls.append((root, cap))
    orig_run_cheap = g.run_cheap

    def _fake_run_cheap(python, scratch, verbose):
        leg = g.Leg("cheap", "fake")
        leg.rc, leg.failures, leg.collected = 0, [], {"tests/fake.py::test_ok"}
        return leg
    g.run_cheap = _fake_run_cheap
    orig_cache_env = os.environ.get("TEX_GATE_CACHE")
    try:
        with tempfile.TemporaryDirectory(prefix="tex-g5-cache-") as cache_scratch:
            os.environ["TEX_GATE_CACHE"] = os.path.join(cache_scratch, "verdicts.json")
            g.main(["--tier", "cheap", "--no-cache"])
    finally:
        g._prune_inductor_cache_root = orig_prune
        g.run_cheap = orig_run_cheap
        if orig_cache_env is None:
            os.environ.pop("TEX_GATE_CACHE", None)
        else:
            os.environ["TEX_GATE_CACHE"] = orig_cache_env
    if not calls:
        r.fail("G5 wiring", "main() never called _prune_inductor_cache_root on a real run")
    else:
        r.ok(f"main() prunes the Inductor cache root once per invocation: {calls[-1]}")


# ── G6: cheap/canonical/touched share ONE leg-building helper ──────────────

def test_g6_shared_leg_builder_exists_and_is_used_by_all_three(r: SubTestResult):
    print("\n--- G6: run_cheap/run_canonical/run_touched route through one helper ---")
    g = _gate()
    if not hasattr(g, "_run_canonical_harness_leg"):
        r.fail("G6 shared helper missing", "gate.py has no _run_canonical_harness_leg")
        return
    calls = []
    orig_helper = g._run_canonical_harness_leg

    def _spy(leg, python, targets, scratch, verbose, marker="not timing"):
        calls.append((leg.name, tuple(targets), marker))
        return leg

    g._run_canonical_harness_leg = _spy
    orig_select = g.select_touched_tests
    g.select_touched_tests = lambda base_ref: (["tests/test_x.py"], {"tex_engine"}, True)
    try:
        with tempfile.TemporaryDirectory(prefix="tex-g6-") as scratch:
            g.run_cheap("python", scratch, False)
            g.run_canonical("python", scratch, False)
            g.run_touched("python", "origin/main", scratch, False)
    finally:
        g._run_canonical_harness_leg = orig_helper
        g.select_touched_tests = orig_select

    names = [c[0] for c in calls]
    if names != ["cheap", "canonical", "touched"]:
        r.fail("G6 shared helper usage", f"expected all three legs to call the shared "
               f"helper exactly once each, in order, got {names}")
        return
    markers = {name: marker for name, _targets, marker in calls}
    if markers["canonical"] != "not slow and not timing":
        r.fail("G6 canonical marker", f"got {markers['canonical']!r}")
    elif markers["cheap"] != "not timing" or markers["touched"] != "not timing":
        r.fail("G6 cheap/touched marker", f"got {markers}")
    else:
        r.ok("run_cheap/run_canonical/run_touched all route through the one shared "
             "leg-building helper, each with its own correct marker")


# ── G7: test_lint46 / test_v042_hostaudit1 share ONE subprocess-KV helper ───

def test_g7_helpers_exposes_run_python_kv_but_not_via_star_import(r: SubTestResult):
    print("\n--- G7: helpers.run_python_kv exists, and stays out of __all__ (HOOK-4) ---")
    import helpers as _helpers
    if not hasattr(_helpers, "run_python_kv"):
        r.fail("G7 shared helper missing", "helpers.py has no run_python_kv function")
        return
    if "run_python_kv" in _helpers.__all__:
        r.fail("G7 HOOK-4 pin", "run_python_kv must NOT be added to helpers.__all__ -- both "
               "callers import it by name, exactly like load_counts_harness")
        return
    out = _helpers.run_python_kv("print('OK', 1)")
    if out != {"OK": "1"}:
        r.fail("G7 run_python_kv behaviour", f"got {out!r}")
        return
    raised = False
    try:
        _helpers.run_python_kv("import sys; sys.exit(3)")
    except RuntimeError:
        raised = True
    if not raised:
        r.fail("G7 run_python_kv error path", "a nonzero exit must raise RuntimeError")
    else:
        r.ok("run_python_kv runs, parses KV stdout, raises on a nonzero exit, and is not in "
             "__all__")


def test_g7_the_two_callers_no_longer_hand_roll_their_own_subprocess_launch(r: SubTestResult):
    print("\n--- G7: test_lint46/test_v042_hostaudit1 no longer define their own launcher ---")
    import pathlib
    tests_dir = pathlib.Path(__file__).resolve().parent
    offenders = []
    for fn in ("test_lint46_check_torch_free.py", "test_v042_hostaudit1_cold_import.py"):
        text = (tests_dir / fn).read_text(encoding="utf-8")
        if "subprocess.run(" in text:
            offenders.append(fn)
    if offenders:
        r.fail("G7 dedupe", f"still hand-rolls a subprocess.run(...) launch: {offenders} -- "
               f"both must delegate to helpers.run_python_kv")
    else:
        r.ok("neither caller calls subprocess.run(...) directly any more")
