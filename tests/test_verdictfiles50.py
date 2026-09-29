"""VERDICTFILES-50 — `tex_runtime/fncalls_compile.py` must sit on
`tex_cache._VERDICT_FILES`.

THE GAP: `fncalls_compile._memo` is exactly the kind of measured win/lose verdict CACHE-4's
`_VERDICT_FILES` docstring describes ("a change moves a measured win/lose verdict"),
persisted via `warm_state.json` under `verdict_epoch()` alongside
`compiled.py`/`autotier.py`/`graphed.py` — but the module itself was never added to the
watch-list. A future edit to its gating logic (e.g. what counts as `ok=True`) would not move
`verdict_epoch()`, so a stale on-disk verdict minted under the OLD gating logic could be
adopted after the edit, silently.

The first two tests are the red-first pair: RED against the base shape (fncalls_compile.py
absent from `_VERDICT_FILES`) and GREEN once it is added. The last two are regression pins
on the tag guard (a persisted failure verdict is dropped once the torch tag or the verdict
epoch moves); they do not look at watch-list membership and would pass without the fix.

PORTABILITY: filesystem + `tex_cache`/`warm_state`/`fncalls_compile` module state only. No
ComfyUI, no CUDA, no numpy, no timing. The shipped source is never written and the
process's real warm-state file is never touched.
"""
import json
import shutil

from helpers import *  # noqa: F401,F403
from helpers import isolated_warm_state as _isolated_warm_state, scratch_dir
from TEX_Wrangle import tex_cache as C
from TEX_Wrangle.tex_runtime import fncalls_compile as FC
from TEX_Wrangle.tex_runtime import warm_state as WS


def _fncalls_compile_path():
    return C._R_DIR / "fncalls_compile.py"


def test_verdictfiles50_fncalls_compile_is_on_verdict_files(r: SubTestResult):
    print("\n--- VERDICTFILES-50: fncalls_compile.py sits on tex_cache._VERDICT_FILES ---")
    try:
        p = _fncalls_compile_path()
        assert p.exists(), f"expected file missing on disk: {p}"
        names = {f.name for f in C._VERDICT_FILES}
        assert "fncalls_compile.py" in names, (
            "fncalls_compile.py is not on tex_cache._VERDICT_FILES — a change to its "
            "gating logic (what counts as a real compile win/lose) would not move "
            "verdict_epoch(), so a stale on-disk verdict from before the change could be "
            "adopted after it")
        # epoch_partitions() is the public seam every other watch-list census reads
        # through (TRK-189) — confirm the same file is visible there too, not only on
        # the private list.
        parts = C.epoch_partitions()
        v_names = {f.name for f in parts["verdict"]}
        assert "fncalls_compile.py" in v_names, \
            "epoch_partitions()['verdict'] disagrees with _VERDICT_FILES"
        r.ok("fncalls_compile.py is watched by _VERDICT_FILES / epoch_partitions()")
    except Exception as e:
        r.fail("VERDICTFILES-50 watch-list membership", f"{type(e).__name__}: {e}")


def test_verdictfiles50_fncalls_compile_edit_moves_verdict_epoch(r: SubTestResult):
    """Mechanism proof on a scratch copy (the shipped file is never written): with
    fncalls_compile.py on the watch-list, a change to its bytes moves the hash
    `verdict_epoch()` is built from."""
    print("\n--- VERDICTFILES-50: an edit to fncalls_compile.py moves the verdict "
          "epoch's hash ---")
    try:
        tmp = scratch_dir("verdictfiles50_") / "fncalls_compile.py"
        shutil.copyfile(_fncalls_compile_path(), tmp)
        seed = b"cg:" + C._CODEGEN_EPOCH.encode()
        others = [f for f in C._VERDICT_FILES if f.name != "fncalls_compile.py"]
        assert len(others) == len(C._VERDICT_FILES) - 1, \
            "fncalls_compile.py must be on _VERDICT_FILES for this to be a real proof"
        h_before = C._hash_files(others + [tmp], seed)
        assert h_before == C._hash_files(C._VERDICT_FILES, seed), \
            "the scratch copy does not stand in for the shipped file"
        tmp.write_bytes(tmp.read_bytes() + b"\n# VERDICTFILES-50 probe: a gating-logic edit\n")
        h_after = C._hash_files(others + [tmp], seed)
        assert h_before != h_after, (
            "editing fncalls_compile.py did not move the hash _VERDICT_EPOCH is built "
            "from — it is not really participating in the epoch")
        r.ok("editing fncalls_compile.py's bytes moves the verdict-epoch hash")
    except Exception as e:
        r.fail("VERDICTFILES-50 edit moves epoch", f"{type(e).__name__}: {e}")


def _seed_persisted_failure_verdict(path, tag: str):
    """Write a warm_state.json whose ONLY content is one fncalls_compile verdict, `False`
    (the failure case), tagged `tag`. Mirrors `test_cache3_version_tag_guard`'s own
    hand-written-JSON technique."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": tag, "capturable": {},
                   "fncalls_compile": {"probe_fp|cpu|fp32": False}}, f)


def _reload_verdict():
    """Forget the load latch and the fncalls memo, then load the seeded file."""
    WS._reset_for_test()
    FC.reset_for_test()
    WS.ensure_loaded()
    return FC.verdict("probe_fp", "cpu", "fp32")


def test_verdictfiles50_stale_failure_verdict_is_reattempted_after_torch_change(r: SubTestResult):
    """A persisted FAILURE verdict (ok=False) written under one torch-version tag must NOT
    be adopted once the box's own tag has moved (a torch upgrade/downgrade) — it must come
    back as unresolved (`None`), so `fncalls_compile.begin_attempt` can grant a fresh
    real attempt instead of the fingerprint being stuck `False` forever."""
    print("\n--- VERDICTFILES-50: a persisted failure verdict is re-attempted after a "
          "torch-version change, not stuck forever ---")
    try:
        with _isolated_warm_state() as snap:
            WS._reset_for_test()
            old_tag = WS._tag()
            _seed_persisted_failure_verdict(snap, old_tag)

            # Positive control: with the tag unchanged the same file IS adopted, so the
            # None below cannot come from a load path that adopts nothing.
            assert _reload_verdict() is False, \
                "the seeded verdict was not adopted under a matching tag"

            # Simulate "after a torch version change": the box's live tag no longer
            # matches the one the failure verdict was persisted under.
            WS._tag = lambda: old_tag + "_DIFFERENT_TORCH"
            v = _reload_verdict()
            assert v is None, (
                f"a failure verdict persisted under the OLD torch tag was adopted after the "
                f"tag moved (v={v}) — it would be stuck False forever instead of retried")
            assert FC.begin_attempt("probe_fp", "cpu", "fp32") is True, \
                "the key was not re-attemptable after the stale verdict was correctly ignored"
        r.ok("a torch-version change orphans the old-tagged failure verdict; the "
             "fingerprint is re-attempted, not stuck")
    except Exception as e:
        r.fail("VERDICTFILES-50 re-attempt after torch change", f"{type(e).__name__}: {e}")


def test_verdictfiles50_stale_failure_verdict_is_reattempted_after_codegen_epoch_change(r: SubTestResult):
    """Same proof, for the other half of `verdict_epoch()`'s nesting: a codegen-epoch
    change (any `_CODEGEN_FILES` edit — an interpreter/codegen/stdlib change) also moves
    `verdict_epoch()` (`_VERDICT_EPOCH = hash(_VERDICT_FILES, cg:+_CODEGEN_EPOCH)`), which
    feeds `warm_state._tag()`. Stood in for directly (mirrors
    `test_cache4_codegen_edit_spares_pkl`'s own technique of assigning the module
    constant) rather than editing a real codegen file, since `verdict_epoch()` reads the
    precomputed `_VERDICT_EPOCH` constant, not a live re-hash."""
    print("\n--- VERDICTFILES-50: a persisted failure verdict is re-attempted after a "
          "codegen-epoch change, not stuck forever ---")
    real_verdict_epoch = C._VERDICT_EPOCH
    try:
        with _isolated_warm_state() as snap:
            WS._reset_for_test()
            old_tag = WS._tag()
            _seed_persisted_failure_verdict(snap, old_tag)
            assert _reload_verdict() is False, \
                "the seeded verdict was not adopted under a matching tag"

            # Simulate "after a codegen-epoch change": _VERDICT_EPOCH moves (it nests the
            # codegen epoch), so the live tag no longer matches the persisted one.
            moved = "0123456789abcdef"
            assert moved != real_verdict_epoch
            C._VERDICT_EPOCH = moved
            v = _reload_verdict()
            assert v is None, (
                f"a failure verdict persisted under the OLD codegen/verdict epoch was adopted "
                f"after the epoch moved (v={v}) — it would be stuck False forever instead of "
                f"retried")
            assert FC.begin_attempt("probe_fp", "cpu", "fp32") is True, \
                "the key was not re-attemptable after the stale verdict was correctly ignored"
        r.ok("a codegen-epoch change orphans the old-tagged failure verdict; the "
             "fingerprint is re-attempted, not stuck")
    except Exception as e:
        r.fail("VERDICTFILES-50 re-attempt after codegen-epoch change", f"{type(e).__name__}: {e}")
    finally:
        C._VERDICT_EPOCH = real_verdict_epoch
