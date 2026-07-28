"""Mutation check — do the release's tests actually KILL the bugs they claim to pin?

NOT part of `run_all.py`: it copies the tree once per mutation and runs the v0.32 + v0.33 rows
against each, which is minutes, not seconds. Run it by hand when a fix lands:

    python tests/mutation_check.py

Why it exists. The v0.32 release audit found 11 of 41 mutations surviving — including the
entire `ResultCache` lock — i.e. code that could be deleted with every test still green. Two of
this release's own fixes were then pinned by tests that did not test them: the `patch_region`
atomicity row passed with the lock removed (the threads never interleaved), and CACHE-9's
`valid` guard was pinned by a row that checked window arithmetic while the pixels were still
wrong by 2.17e-01. Both are fixed; this is how that was established rather than assumed.

Each entry re-introduces a REAL bug this release fixed. A `*** SURVIVED ***` verdict means the
corresponding test is decorative.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

# Derived, not hardcoded: this file lives in <pkg>/tests/, and the interpreter running it is
# the one the mutants must run under (a wrong interpreter has already cost this project a
# baseline and five phantom fp16 failures).
SRC = pathlib.Path(__file__).resolve().parent.parent
VENV = sys.executable

MUTATIONS = [
    ("chain_windows: compose FORWARD instead of backward", "tex_roi.py",
     "    for i in range(n - 2, start - 1, -1):",
     "    for i in range(start, n - 1):"),
    ("chain_windows: grow by the stage's OWN halo, not its consumer's", "tex_roi.py",
     "        pad = int(halos[i + 1])",
     "        pad = int(halos[i])"),
    ("chain_windows: drop the `valid` guard", "tex_roi.py",
     "            if not covers(upstream_valid, grown):",
     "            if False:"),
    ("ResultCache: remove the lock from patch_region", "tex_results.py",
     "        with self._lock:\n"
     "            out = self._patch_region_locked(key, patch, window, base, base_key, canvas,\n"
     "                                            quality, storage)",
     "        if True:\n"
     "            out = self._patch_region_locked(key, patch, window, base, base_key, canvas,\n"
     "                                            quality, storage)"),
    ("ResultCache: _remove forgets the per-device total", "tex_results.py",
     "            self._bytes_by_dev[_dev_bucket(entry.device)] -= entry.nbytes",
     "            pass"),
    ("CACHE-7: drop the linear gate again", "tex_checkpoint.py",
     "    if not is_linear_stage_list(stages):",
     "    if False:"),
    ("stage_snapshot: resolve WITHOUT need_stages", "tex_runtime/profile.py",
     "        best, scale = _resolve_bucket(key, spatial, need_stages=True)\n"
     "        if best is None:\n"
     "            return {}, False",
     "        best, scale = _resolve_bucket(key, spatial, need_stages=False)\n"
     "        if best is None or not best.stages:\n"
     "            return {}, False"),
    ("GOV-1: balanced stops restoring the shipped budget", "tex_memory.py",
     "            default = defaults.get(knob)",
     "            default = None"),
    # ── v0.33 ──────────────────────────────────────────────────────────────────────────
    ("PREC-1: the quality tag stops gating (a storage hint alone reduces)", "tex_packing.py",
     "    if quality != PREVIEW:\n"
     "        return None                           # the default path, byte-identical to pre-v0.33",
     "    if quality != PREVIEW and storage is None:\n"
     "        return None"),
    ("PREC-1: get stops unpacking (storage precision leaks to the consumer)", "tex_results.py",
     "        if orig_dtype is not None:\n"
     "            from . import tex_packing",
     "        if False:\n"
     "            from . import tex_packing"),
    ("PREC-1: the fp16 range gate accepts anything (HDR -> inf)", "tex_packing.py",
     "    return max(abs(lo), abs(hi)) <= FP16_MAX",
     "    return True"),
    ("PREC-1: the spill forgets the stored representation", "tex_results.py",
     '"orig": _dtype_tables()[0].get(entry.orig_dtype), "viewed": viewed}',
     '"orig": None, "viewed": viewed}'),
    ("CACHE-8: the spill writes where the frame IS, not where it belongs", "tex_results.py",
     '"device": entry.home, "canvas": entry.canvas',
     '"device": entry.device, "canvas": entry.canvas'),
    # The guard is REMOVED, not neutered. A first attempt kept the `return` and only changed
    # the arithmetic below it, which is a no-op — the mutation "survived" because it was not a
    # mutation. A row that cannot change behaviour tests nothing.
    ("CACHE-8: residency runs even when disarmed", "tex_results.py",
     "        if self._vram_budget is None:\n"
     "            return\n"
     "        over = self._bytes_by_dev[\"cuda\"] - self._vram_budget",
     "        over = self._bytes_by_dev[\"cuda\"] - (self._vram_budget or 0)"),
    ("CACHE-8: a demoted frame is never promoted home", "tex_results.py",
     "        if demoted is not None:\n"
     "            frame = self._promote(key, demoted)",
     "        if False:\n"
     "            frame = self._promote(key, demoted)"),
    ("CACHE-8: the governor stops preferring demotion over eviction", "tex_results.py",
     "            if dev_type == \"cuda\" and self._vram_budget is not None:",
     "            if False:"),
    ("CACHE-8: uint16 stops refusing out-of-range frames (silent clipping)", "tex_packing.py",
     "        return lo >= 0.0 and hi <= 1.0",
     "        return True"),
    ("XPU-2: tensor() stops fencing", "tex_runtime/streams.py",
     "        return self.wait()._host",
     "        return self._host"),
    ("XPU-2: a RETAINED destination gets pinned anyway (retained= ignored)",
     "tex_runtime/streams.py",
     "    if (not isinstance(src, torch.Tensor) or src.device.type != \"cuda\" or retained",
     "    if (not isinstance(src, torch.Tensor) or src.device.type != \"cuda\""),
    # ── v0.33.1 (the release-audit findings) ───────────────────────────────────────────
    # RETIRED, with the reason — not silently deleted. This row SURVIVED, and the survival is
    # a fact about the CODE, not about the tests: the commit-block device re-check is
    # unreachable. The pop block already rejects on device, and `_demoting` prevents the only
    # route to a second pop of a live victim, so nothing can reach the commit with a device
    # that has already moved. The check stays (it mirrors `_promote` and costs one comparison,
    # and it is the guard a future queueing route would need), but a row that cannot be killed
    # asserts nothing, and leaving it as permanent SURVIVED noise trains the reader to ignore
    # the word. Re-arm this if a second producer of `_pending_demotes` ever appears.
    #   ("A1: the demote commit stops re-checking the DEVICE", ...)
    ("A1: an in-flight demotion is invisible to the victim walk again", "tex_results.py",
     "        queued = {k for k, _e in self._pending_demotes} | self._demoting\n"
     "        got = 0",
     "        queued = {k for k, _e in self._pending_demotes}\n"
     "        got = 0"),
    ("A2: get() re-looks-up orig_dtype after _restore (the two-acquisition read)",
     "tex_results.py",
     "            frame, orig_dtype = self._restore(key)",
     "            frame, _discard = self._restore(key)\n"
     "            with self._lock:\n"
     "                _e = self._ram.get(key)\n"
     "                orig_dtype = _e.orig_dtype if _e is not None else None"),
    # RETIRED, same reasoning, and it earned its keep on the way out: `raced` is redundant for
    # both cases a test can construct (a fresh cache is caught by `unknown_at_entry`; a learned
    # set is caught by the merge, because `_spill` records into it). Asking why the mutation
    # survived surfaced the case it is NOT redundant for — `_enforce_disk_budget` dropping
    # `_spilled` to None mid-scan, which was crashing the merge with a TypeError. That guard is
    # now explicit and the crash is fixed; the mutation still cannot be killed by a test.
    #   ("A3: reindex rebinds membership over a racing spill", ...)
    ("A7: the victim walk reaches the MRU frame again", "tex_results.py",
     "            if key == mru:\n"
     "                continue",
     "            if False:\n"
     "                continue"),
    ("B2/P0-5: remap_suffix_taps becomes the identity", "tex_fusion.py",
     "    if not k:\n"
     "        return outputs\n"
     "    out = {}",
     "    if True:\n"
     "        return outputs\n"
     "    out = {}"),
]

RUNNER = """
import sys
sys.argv = ["x"]
sys.path.insert(0, r"{tests}")
sys.path.insert(0, r"{parent}")
from helpers import SubTestResult
import test_v032_checkpoint as A, test_v032_region as B, test_v032_governor as C
import test_v033_precision as D, test_v033_cache8 as E, test_v033_xpu2 as F
import test_v033_phase0 as G, test_v0331_audit as H
r = SubTestResult()
for m in (A, B, C, D, E, F, G, H):
    for n in sorted(x for x in dir(m) if x.startswith("test_")):
        try:
            getattr(m, n)(r)
        except Exception:
            r.fail(n, "raised")
print("FAILCOUNT", r.failed)
"""


def run_tree(root):
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TEX_CACHE_DIR"] = tempfile.mkdtemp(prefix="mutcache_")
    try:
        out = subprocess.run(
            [VENV, "-c", RUNNER.format(tests=str(root / "tests"), parent=str(root.parent))],
            capture_output=True, text=True, cwd=str(root / "tests"), env=env, timeout=3600)
    except subprocess.TimeoutExpired:
        return -2
    for line in reversed(out.stdout.splitlines()):
        if line.startswith("FAILCOUNT"):
            return int(line.split()[1])
    return -1


base_root = pathlib.Path(tempfile.mkdtemp(prefix="mutbase_"))
shutil.copytree(SRC, base_root / "TEX_Wrangle",
                ignore=shutil.ignore_patterns("__pycache__", ".tex_cache", ".git", "results"))

print(f"{'mutation':58s} {'verdict'}")
print("-" * 84)
for label, rel, old, new in MUTATIONS:
    work_root = pathlib.Path(tempfile.mkdtemp(prefix="mut_"))
    shutil.copytree(base_root / "TEX_Wrangle", work_root / "TEX_Wrangle")
    p = work_root / "TEX_Wrangle" / rel
    s = p.read_text(encoding="utf-8")
    if s.count(old) != 1:
        print(f"{label:58s} SKIP (anchor matched {s.count(old)}x)")
        continue
    p.write_text(s.replace(old, new, 1), encoding="utf-8", newline="")
    failed = run_tree(work_root / "TEX_Wrangle")
    verdict = {"-2": "KILLED (hung)", "-1": "KILLED (crashed)"}.get(
        str(failed), "*** SURVIVED ***" if failed == 0 else "KILLED")
    print(f"{label:58s} {verdict:20s} ({failed} failing rows)")
    shutil.rmtree(work_root, ignore_errors=True)
shutil.rmtree(base_root, ignore_errors=True)
