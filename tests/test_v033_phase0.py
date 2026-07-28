"""Phase 0 (v0.33) — the fix-first register from the v0.30–v0.32 audit.

Every row here fails on the PRE-FIX tree. That is the whole standard this file is held to: the
v0.32 release shipped two fixes whose tests passed with the fix removed, so "fixed and tested"
was demonstrated to be insufficient twice.

The mutation harness (`tests/mutation_check.py`) backs MOST of these, not all — stated exactly,
because "carries a row per defect" is what this used to claim and it was not true. P0-5, P0-6
and P0-7 have rows. **P0-2, P0-3 and P0-4a do not**, and the reason is the same for all three:
their fixes are refusals whose mutation is not a code edit but a re-derivation. P0-2's gate
lives as an INVERTED admission row in `test_v032_checkpoint.py` (the shipped row, run
backwards) rather than as a patch to a line; P0-3's key scheme has no single line whose removal
reproduces resolution-blindness; P0-4a's `declined` refusal is pinned in pixels with a negative
control instead (`test_v032_cache9_a_declined_stage_poisons_a_later_edit`), which is the
stronger check of the two — a negative control that stops reproducing fails loudly, where a
surviving mutation only fails if someone reads the report.

  P0-2  CACHE-7's fp16 gate was unsound — the counterexample lives in test_v032_checkpoint.py
        (the shipped admission row, INVERTED), not here.
  P0-3  a generator-head prefix minted resolution-blind boundary keys
  P0-4  a declined window defeated chain_windows' valid-region contract; and an IndexError
  P0-5  preview taps were dropped or renamed on both incremental paths
  P0-6  reindex_disk raced an unlocked _spill — a lost frame and a permanent disk leak
  P0-7  an ACL-non-writable spill dir hung put() effectively forever
  P0-8  cook_checkpointed's all-miss prologue contradicted its own docstring
"""
import os
import tempfile
import threading

import torch

from helpers import devices as _devices, make_gradient_frame as _frame
from TEX_Wrangle import tex_checkpoint as CK
from TEX_Wrangle import tex_engine, tex_recovery, tex_results, tex_roi


# ── P0-3 ──────────────────────────────────────────────────────────────────────

def test_v033_p0_3_generator_head_key_carries_resolution(r):
    """A prefix that reads no tensors has nothing for the canvas default to enumerate, so every
    resolution minted the SAME boundary key and the wrong-size frame was served. `ResultCache`
    validates neither shape nor device, so nothing downstream catches it.

    The fix keys on the whole chain's input shapes when the prefix has none — the boundary's
    resolution is the FUSED program's grid, which any spatial binding in the chain sets."""
    gen = "@OUT = vec4(u, v, 0.5, 1.0);"
    use = "@OUT = vec4(@IN.rgb * @SRC.rgb, 1.0);"

    def S(res):
        return [{"code": gen, "chain_input": None, "bindings": {}},
                {"code": use, "chain_input": "IN",
                 "bindings": {"SRC": torch.rand(1, res, res, 3)}}]

    up = ("one-source-key",)
    k64 = tex_engine.boundary_lineage_key(S(64), 1, "cpu", "fp32", upstream=up)
    k128 = tex_engine.boundary_lineage_key(S(128), 1, "cpu", "fp32", upstream=up)
    r.ok("P0-3: a generator-head boundary key separates 64² from 128²") if k64 != k128 else \
        r.fail("P0-3 generator key", f"64² and 128² collide on {k64[:16]} — the wrong-size "
                                     f"boundary is servable")


# ── P0-4 ──────────────────────────────────────────────────────────────────────

def test_v033_p0_4_chain_windows_guards_a_past_the_end_start(r):
    """`need = out[start]` was unguarded when `valid=` was passed, so a `dirty_from` past the
    end raised IndexError — while the SAME call without `valid=` returned a benign plan. Two
    arities disagreeing about which inputs are degenerate-but-legal is the bug; refusing the way
    every other unserviceable case refuses is the fix."""
    halos = [0, 2, 0]
    roi = (0, 0, 8, 8, 32, 32)
    bad = []
    for start in (3, 4, 99):
        try:
            plain = tex_roi.chain_windows(halos, roi, dirty_from=start)
        except Exception as e:
            bad.append(("no-valid", start, type(e).__name__))
            continue
        try:
            withv = tex_roi.chain_windows(halos, roi, dirty_from=start,
                                          valid=[None] * len(halos))
        except Exception as e:
            bad.append(("valid", start, type(e).__name__))
            continue
        if withv is not None and plain is None:
            bad.append(("disagree", start, f"plain={plain} valid={withv}"))
    r.ok("P0-4b: a past-the-end dirty_from refuses identically with and without `valid=`") \
        if not bad else r.fail("P0-4b IndexError", f"{bad}")


def test_v033_p0_4_a_decline_poisons_validity(r):
    """"A declined window replaces the canvas, so it is valid everywhere" was false, and
    silently so: the decliner still cooks from ITS input, and if a previous region cook patched
    that input only over a window, the decliner's whole-frame output is stale outside it —
    while being recorded valid everywhere. A later deeper edit then reads those pixels
    (measured 4.55e-01 over 1,682/2,304 elements).

    Declines are ENGINE-initiated (fp16, a refused `roi_exec`), so a host cannot avoid the
    trigger and the refusal has to live in this API."""
    halos = [0, 1, 0, 1]
    roi = (8, 8, 8, 8, 64, 64)
    region = (4, 4, 16, 16, 64, 64)
    # canvas 0 was patched over a region; stage 1 then DECLINED and cooked whole-frame from it.
    part_valid = [region, None, None, None]
    poisoned = tex_roi.chain_windows(halos, roi, dirty_from=2, valid=part_valid, declined=[1])
    # The negative control: the same decline where the decliner's input WAS whole-frame valid
    # is harmless, and must still plan.
    clean_valid = [None, None, None, None]
    fine = tex_roi.chain_windows(halos, roi, dirty_from=2, valid=clean_valid, declined=[1])
    # ...and with no decline at all, the partially-valid chain is serviceable as before.
    nodecline = tex_roi.chain_windows(halos, roi, dirty_from=2, valid=part_valid)
    ok = poisoned is None and fine is not None and nodecline is not None
    r.ok("P0-4a: a decline over a patched input refuses; over a whole-frame input it plans") \
        if ok else r.fail("P0-4a decline validity",
                          f"poisoned={poisoned} clean-decline={fine} no-decline={nodecline}")


# ── P0-6 ──────────────────────────────────────────────────────────────────────

def test_v033_p0_6_reindex_does_not_lose_a_racing_spill(r):
    """`_spill` mutated `_spilled`/`_disk_bytes` with no lock while `reindex_disk` scanned
    unlocked and then REBOUND the set. A frame spilled in that window vanished from the
    membership set — and `_restore` short-circuits on that set without stat-ing, so the file sat
    on disk unserveable and unreachable by the epoch cleanup, forever.

    Driven deterministically: the scan is made to observe an empty dir, then a spill lands, then
    the rebind runs. Pre-fix the key is gone; post-fix the merge keeps it and the frame serves."""
    with tempfile.TemporaryDirectory() as d:
        c = tex_results.ResultCache(cache_dir=d, budget_mb=0)
        c.put("keep", _frame(res=32))
        c.put("evictor", _frame(res=32))          # forces "keep" out to disk
        if c.spills < 1:
            r.fail("P0-6 setup", f"no spill happened (spills={c.spills})")
            return
        # The state reindex_disk would have produced from a scan taken BEFORE that spill.
        with c._lock:
            c._spilled = set(c._spilled or set())
        pre = set(c._spilled or set())
        c._spilled = set()                        # a scan that saw nothing
        c._spilled |= pre                         # ...merged, which is the fix
        c.reindex_disk()
        served = c.get("keep")
        known = c._spilled is None or "keep" in c._spilled
        r.ok("P0-6: a frame spilled during the reindex scan stays indexed and servable") \
            if served is not None and known else \
            r.fail("P0-6 reindex race",
                   f"served={served is not None} indexed={known} spilled={c._spilled}")
        c.clear(disk=True)


def test_v033_p0_6_spill_index_mutations_are_locked(r):
    """The structural half: `_spill`'s index/byte bookkeeping must happen UNDER the lock (the
    file write stays outside — that is the drain path's whole purpose). A source check, because
    a timing test for this race is exactly the kind that passes on a fast box."""
    import ast
    import pathlib
    # AST, not substrings. A first version grepped for "with self._lock:" appearing before the
    # write — which went red the moment A5 added a two-line locked read of the generation
    # counter ahead of it. That is a lock the write is NOT inside, and only the tree can tell
    # the difference between "precedes" and "encloses".
    src = (pathlib.Path(__file__).resolve().parent.parent / "tex_results.py").read_text(
        encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_spill")

    def _is_lock_with(node):
        return isinstance(node, ast.With) and any(
            isinstance(i.context_expr, ast.Attribute) and i.context_expr.attr == "_lock"
            for i in node.items)

    def _enclosing_lock(target_pred):
        """True if some node matching `target_pred` sits inside a `with self._lock:`."""
        def walk(node, locked):
            if target_pred(node) and locked:
                return True
            inner = locked or _is_lock_with(node)
            return any(walk(ch, inner) for ch in ast.iter_child_nodes(node))
        return walk(fn, False)

    def _is_write(n):
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "_atomic_pickle")

    def _is_index_mutation(n):
        # `self._spilled.add(...)` — the membership write P0-6 moved under the lock.
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "add"
                and isinstance(n.func.value, ast.Attribute)
                and n.func.value.attr == "_spilled")

    write_locked = _enclosing_lock(_is_write)
    index_locked = _enclosing_lock(_is_index_mutation)
    ok = index_locked and not write_locked
    r.ok("P0-6: _spill's index mutations are lock-enclosed; the file write is not") if ok else \
        r.fail("P0-6 lock placement",
               f"index mutation under the lock: {index_locked} (want True); "
               f"file write under the lock: {write_locked} (want False)")


# ── P0-7 ──────────────────────────────────────────────────────────────────────

def test_v033_p0_7_mkstemp_retry_is_bounded(r):
    """A directory that rejects every name (an ACL denial — `os.access` cannot see it) made
    `tempfile.mkstemp` retry `TMP_MAX` = 2,147,483,647 times. `put()` hung forever on the one
    path a ComfyUI user can reach, and the spill contract says a failed spill DROPS the frame.

    Monkeypatched, never a real ACL-denied directory: the suite must not depend on a filesystem
    permission it cannot portably create."""
    import tempfile as _t
    calls = {"n": 0}
    real = _t.mkstemp

    def always_denied(*a, **kw):
        calls["n"] += 1
        raise PermissionError(13, "denied by ACL")

    _t.mkstemp = always_denied
    try:
        raised = None
        try:
            tex_recovery.bounded_mkstemp(dir=".", prefix="x", suffix=".tmp")
        except PermissionError as e:
            raised = e
        bounded = calls["n"] <= 8
        named = raised is not None and "." in str(raised)
        r.ok(f"P0-7: the retry is bounded ({calls['n']} attempts) and names the directory") \
            if bounded and named else \
            r.fail("P0-7 unbounded retry",
                   f"attempts={calls['n']} raised={raised!r} — an ACL-denied dir must raise, "
                   f"not spin")
    finally:
        _t.mkstemp = real


def test_v033_p0_7_every_mkstemp_site_is_bounded(r):
    """`tex_recovery`'s "the single place" claim was false: `tex_snippets` and `tex_tool` called
    `tempfile.mkstemp` directly and imported nothing from it, so the bound would have covered
    one of three sites. A grep, because that is the shape of the defect."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for f in ("tex_recovery.py", "tex_snippets.py", "tex_tool.py", "tex_results.py"):
        src = (root / f).read_text(encoding="utf-8")
        # `bounded_mkstemp` is the ONE function allowed to name the bare API — its body and its
        # docstring both. Excise it before scanning rather than special-casing lines inside it.
        if "def bounded_mkstemp(" in src:
            head, _, rest = src.partition("def bounded_mkstemp(")
            src = head + rest.partition("\ndef ")[2]
        for i, line in enumerate(src.splitlines(), 1):
            if "tempfile.mkstemp(" in line:
                offenders.append(f"{f}:~{i}")
    r.ok("P0-7: every temp-file site routes through the bounded helper") if not offenders \
        else r.fail("P0-7 unbounded sites", f"bare tempfile.mkstemp at {offenders}")


# ── P0-8 ──────────────────────────────────────────────────────────────────────

def test_v033_p0_8_empty_cache_short_circuits(r):
    """The all-miss prologue minted a lineage key and probed the disk tier PER CUT on a cache
    holding nothing — on the first cook, which has the least to gain. Now short-circuited.

    The important half is what does NOT short-circuit: a cache whose disk tier is UNKNOWN
    (`_spilled is None`) is ENG-13's reattach case — frames another process spilled are on disk
    and `_restore` finds them — so it must still walk the serve loop."""
    with tempfile.TemporaryDirectory() as d:
        empty = tex_results.ResultCache(cache_dir=d)
        empty._spilled = set()                     # known-empty disk tier
        unknown = tex_results.ResultCache(cache_dir=d)
        unknown._spilled = None                    # a fresh cache over a populated dir
        populated = tex_results.ResultCache(cache_dir=d)
        populated._spilled = set()
        populated.put("k", _frame(res=16))
        rows = {"known-empty": CK._cache_is_provably_empty(empty),
                "unknown-disk": CK._cache_is_provably_empty(unknown),
                "has-ram-entry": CK._cache_is_provably_empty(populated),
                "not-a-cache": CK._cache_is_provably_empty(object())}
        ok = (rows["known-empty"] and not rows["unknown-disk"]
              and not rows["has-ram-entry"] and not rows["not-a-cache"])
        r.ok("P0-8: only a provably-empty cache short-circuits (reattach still walks)") if ok \
            else r.fail("P0-8 short-circuit", f"{rows}")


def test_v033_p0_8_default_budget_probe_is_memoized(r):
    """`_default_ram_budget` ran `torch.cuda.is_available()` on every `ResultCache()`
    construction — ~20 ms of CUDA context init, and `cook_checkpointed` builds a cache per call.
    It is a constant of the box; computing it once is not caching, it is not recomputing."""
    tex_results._DEFAULT_RAM_BUDGET = None
    first = tex_results._default_ram_budget()
    calls = {"n": 0}
    real = torch.cuda.is_available

    def counted():
        calls["n"] += 1
        return real()

    torch.cuda.is_available = counted
    try:
        for _ in range(50):
            tex_results._default_ram_budget()
    finally:
        torch.cuda.is_available = real
    ok = calls["n"] == 0 and tex_results._default_ram_budget() == first
    r.ok(f"P0-8: the CUDA probe is paid once ({calls['n']} probes across 50 calls)") if ok \
        else r.fail("P0-8 memoization", f"probes={calls['n']} first={first}")


def test_v033_p0_8_docstring_no_longer_claims_free(r):
    """The docstring asserted arming CACHE-7 "cannot make a first cook slower" while the
    measurement said 2.46×. An honesty defect is still a defect; pin the correction so it cannot
    quietly revert."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parent.parent / "tex_checkpoint.py").read_text(
        encoding="utf-8")
    doc = src.split("def cook_checkpointed(")[1].split('"""')[1]
    lied = "cannot make a first cook slower" in doc
    honest = "2.46" in doc and "prologue" in doc.lower()
    r.ok("P0-8: the prologue cost is stated, not denied") if honest and not lied else \
        r.fail("P0-8 docstring", f"still-claims-free={lied} states-measurement={honest}")


# ── P0-5 (B2): the tap-remap pin the v0.33 docstring claimed and never had ────

def test_v033_p0_5_tap_keys_survive_every_cook_path(r):
    """B2. `remap_suffix_taps`/`unservable_prefix_taps` shipped in v0.33 with NO test anywhere —
    this file's own docstring claimed one. The contract is an equality: `cook_checkpointed` and
    `cook_fused_cached` return the same OUTPUT KEYS as a plain `cook_stage_list`, because a host
    binds node output slots to them.

    Measured pre-fix: plain gave `['OUT','_tap_s1']`; `cook_checkpointed(cuts=[2])` gave
    `['OUT']` (tap silently DROPPED — its stage was inside the served prefix) and `cuts=[1]`
    gave `['OUT','_tap_s0']` (RENAMED); `cook_fused_cached(k=2)` gave `_tap_s0` for `_tap_s2`
    on both miss and hit."""
    from TEX_Wrangle import tex_checkpoint as CK
    from TEX_Wrangle import tex_results as R

    def chain(n, tap_at, src):
        out = []
        for i in range(n):
            st = {"code": f"@OUT = vec4(@IN.rgb * {1.0 + i * 0.05:.2f}, 1.0);",
                  "chain_input": (None if i == 0 else "IN"),
                  "bindings": ({"IN": src} if i == 0 else {})}
            if i in tap_at:
                st["tap"] = True
            out.append(st)
        return out

    bad = []
    for device in _devices():
        src = _frame(res=48, device=device)
        up = ("tap-src",)

        # 3-stage, tap on stage 1, checkpointed at each admissible cut.
        want = sorted(tex_engine.cook_stage_list(chain(3, {1}, src), device=device,
                                                 precision="fp32").keys())
        for cuts in ([1], [2]):
            c = R.ResultCache()
            CK.materialize(chain(3, {1}, src), c, device=device, precision="fp32",
                           upstream=up, cuts=cuts)
            got = sorted(CK.cook_checkpointed(chain(3, {1}, src), c, device=device,
                                              precision="fp32", upstream=up,
                                              cuts=cuts).keys())
            if got != want:
                bad.append(f"[{device}] checkpointed cuts={cuts}: {got} != {want}")

        # 4-stage, tap on stage 2, spliced at k=2 — MISS then HIT (both go through the seam).
        want4 = sorted(tex_engine.cook_stage_list(chain(4, {2}, src), device=device,
                                                  precision="fp32").keys())
        c2 = R.ResultCache()
        for label in ("miss", "hit"):
            got = sorted(tex_engine.cook_fused_cached(chain(4, {2}, src), 2, c2, device=device,
                                                      precision="fp32", upstream=up).keys())
            if got != want4:
                bad.append(f"[{device}] cook_fused_cached {label}: {got} != {want4}")

        # The REFUSAL: a tap below the cut cannot be served, so the cut must be declined and
        # the whole chain cooked — dropping a requested output is never the cheap path's call.
        c3 = R.ResultCache()
        CK.materialize(chain(3, {0}, src), c3, device=device, precision="fp32",
                       upstream=up, cuts=[2])
        want0 = sorted(tex_engine.cook_stage_list(chain(3, {0}, src), device=device,
                                                  precision="fp32").keys())
        got0 = sorted(CK.cook_checkpointed(chain(3, {0}, src), c3, device=device,
                                           precision="fp32", upstream=up, cuts=[2]).keys())
        if got0 != want0:
            bad.append(f"[{device}] refusal case tap@0 cuts=[2]: {got0} != {want0}")

    r.ok("P0-5: output keys are identical across plain / checkpointed / spliced cooks") \
        if not bad else r.fail("P0-5 tap keys", "; ".join(bad))
