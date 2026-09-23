"""Mutation check — do the release's tests actually KILL the bugs they claim to pin?

NOT part of `run_all.py`: it copies the tree once per mutation and runs the v0.32-v0.36 rows
(71 of them, 28 in `tex_results.py` alone) against each, which is minutes, not seconds. Run it
by hand when a fix lands:

    python tests/mutation_check.py
    python tests/mutation_check.py --rows TRK-25     # one row, or a substring's worth

Why it exists. The v0.32 release audit found 11 of 41 mutations surviving — including the
entire `ResultCache` lock — i.e. code that could be deleted with every test still green. Two of
this release's own fixes were then pinned by tests that did not test them: the `patch_region`
atomicity row passed with the lock removed (the threads never interleaved), and CACHE-9's
`valid` guard was pinned by a row that checked window arithmetic while the pixels were still
wrong by 2.17e-01. Both are fixed; this is how that was established rather than assumed.

Each entry re-introduces a REAL bug this release fixed. A `*** SURVIVED ***` verdict means the
corresponding test is decorative.

THE SUITE COLUMN, and why it is not optional (MUT-1). Every row carries the test module(s)
whose tests are supposed to kill it, and the subprocess's import list is the UNION of that
column over the rows actually being swept — derived, never hand-kept. This is the same
single-source discipline invariant 5 uses for `tex_memory._NON_LOCAL_FNS` (derived from the
stdlib registry's `footprint`, not a parallel literal), applied here for the same reason: the
hand-kept list had already drifted TWICE. It ended at v0.33 when v0.34 shipped, and it ended at
v0.35 when v0.36 shipped — so all three TRK-25 rows reported `SURVIVED (0 failing rows)` from
v0.36.0 to v0.36.2 while `test_v036_region_dependence`, the file holding every test that could
kill them, was never loaded. A row that cannot be killed is worse than no row: it reports a
guarantee the tree does not have. With the column, adding a row without its killing suite is
not something a reader has to remember — the row IS the wiring.

The list is curated rather than "import every test module" on purpose: the subprocess runs once
per row, so import cost is multiplied by the row count, and a sweep nobody can afford to run is
a sweep nobody runs. The column keeps the set minimal AND correct at the same time.

The column is also checked against reality, not trusted: the runner reports per-module failure
counts, so a row that is killed only by suites it did NOT declare is reported as MISATTRIBUTED
— the row passes for the wrong reason, which is how a guard can be deleted with the sweep still
green.
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
TESTS_DIR = pathlib.Path(__file__).resolve().parent
VENV = sys.executable

# Rows are (label, file, old, new, suites). `suites` is a tuple of test MODULE names (no
# `.py`, importable from tests/) — the suite(s) that claim to pin this bug. See the module
# docstring: this column is the runner's import list, so it is load-bearing, not annotation.
MUTATIONS = [
    ("chain_windows: compose FORWARD instead of backward", "tex_roi.py",
     "    for i in range(n - 2, start - 1, -1):",
     "    for i in range(start, n - 1):",
     ("test_v032_region",)),
    ("chain_windows: grow by the stage's OWN halo, not its consumer's", "tex_roi.py",
     "        pad = int(halos[i + 1])",
     "        pad = int(halos[i])",
     ("test_v032_region",)),
    ("chain_windows: drop the `valid` guard", "tex_roi.py",
     "            if not covers(upstream_valid, grown):",
     "            if False:",
     ("test_v032_region",)),
    ("ResultCache: remove the lock from patch_region", "tex_results.py",
     "        with self._lock:\n"
     "            out = self._patch_region_locked(key, patch, window, base, base_key, canvas,\n"
     "                                            quality, storage)",
     "        if True:\n"
     "            out = self._patch_region_locked(key, patch, window, base, base_key, canvas,\n"
     "                                            quality, storage)",
     ("test_v032_region",)),
    ("ResultCache: _remove forgets the per-device total", "tex_results.py",
     "            self._bytes_by_dev[_dev_bucket(entry.device)] -= entry.nbytes",
     "            pass",
     ("test_v032_governor", "test_v033_cache8")),
    ("CACHE-7: drop the linear gate again", "tex_checkpoint.py",
     "    if not is_linear_stage_list(stages):",
     "    if False:",
     ("test_v032_checkpoint",)),
    ("stage_snapshot: resolve WITHOUT need_stages", "tex_runtime/profile.py",
     "        best, scale = _resolve_bucket(key, spatial, need_stages=True)\n"
     "        if best is None:\n"
     "            return {}, False",
     "        best, scale = _resolve_bucket(key, spatial, need_stages=False)\n"
     "        if best is None or not best.stages:\n"
     "            return {}, False",
     ("test_v032_checkpoint",)),
    ("GOV-1: balanced stops restoring the shipped budget", "tex_memory.py",
     "            default = defaults.get(knob)",
     "            default = None",
     ("test_v032_governor",)),
    # ── v0.33 ──────────────────────────────────────────────────────────────────────────
    # MEASURED, not assumed: the suite named after this guard does NOT kill it. The killer is
    # `test_v033_cache8` (1 row); `test_v033_precision` contributes 0. PREC-1's own rows check
    # what a PREVIEW put stores, and a storage hint alone reducing is a case they never
    # construct. Filed as a finding — the column records who kills it, not who should.
    ("PREC-1: the quality tag stops gating (a storage hint alone reduces)", "tex_packing.py",
     "    if quality != PREVIEW:\n"
     "        return None                           # the default path, byte-identical to pre-v0.33",
     "    if quality != PREVIEW and storage is None:\n"
     "        return None",
     ("test_v033_cache8",)),
    ("PREC-1: get stops unpacking (storage precision leaks to the consumer)", "tex_results.py",
     "        if orig_dtype is not None:\n"
     "            from . import tex_packing",
     "        if False:\n"
     "            from . import tex_packing",
     ("test_v033_precision",)),
    ("PREC-1: the fp16 range gate accepts anything (HDR -> inf)", "tex_packing.py",
     "    return max(abs(lo), abs(hi)) <= FP16_MAX",
     "    return True",
     ("test_v033_precision",)),
    ("PREC-1: the spill forgets the stored representation", "tex_results.py",
     '"orig": _dtype_tables()[0].get(entry.orig_dtype), "viewed": viewed,',
     '"orig": None, "viewed": viewed,',
     ("test_v033_precision",)),
    ("CACHE-8: the spill writes where the frame IS, not where it belongs", "tex_results.py",
     '"device": entry.home, "canvas": entry.canvas',
     '"device": entry.device, "canvas": entry.canvas',
     ("test_v033_cache8",)),
    # The guard is REMOVED, not neutered. A first attempt kept the `return` and only changed
    # the arithmetic below it, which is a no-op — the mutation "survived" because it was not a
    # mutation. A row that cannot change behaviour tests nothing.
    # MEASURED: CACHE-8's own suite contributes 0 here. The only killer is `test_v0332_audit`
    # (1 row) — the A5 disarm row, written a release later. `test_v033_cache8` arms residency
    # before it looks, so the disarmed path is one it never exercises.
    ("CACHE-8: residency runs even when disarmed", "tex_results.py",
     "        if self._vram_budget is None:\n"
     "            return\n"
     "        over = self._bytes_by_dev[\"cuda\"] - self._vram_budget",
     "        over = self._bytes_by_dev[\"cuda\"] - (self._vram_budget or 0)",
     ("test_v0332_audit",)),
    ("CACHE-8: a demoted frame is never promoted home", "tex_results.py",
     "        if demoted is not None:\n"
     "            frame = self._promote(key, demoted)",
     "        if False:\n"
     "            frame = self._promote(key, demoted)",
     ("test_v033_cache8",)),
    ("CACHE-8: the governor stops preferring demotion over eviction", "tex_results.py",
     "            if dev_type == \"cuda\" and self._vram_budget is not None:",
     "            if False:",
     ("test_v033_cache8",)),
    ("CACHE-8: uint16 stops refusing out-of-range frames (silent clipping)", "tex_packing.py",
     "        return lo >= 0.0 and hi <= 1.0",
     "        return True",
     ("test_v033_cache8", "test_v033_precision")),
    ("XPU-2: tensor() stops fencing", "tex_runtime/streams.py",
     "        return self.wait()._host",
     "        return self._host",
     ("test_v033_xpu2",)),
    ("XPU-2: a RETAINED destination gets pinned anyway (retained= ignored)",
     "tex_runtime/streams.py",
     "    if (not isinstance(src, torch.Tensor) or src.device.type != \"cuda\" or retained",
     "    if (not isinstance(src, torch.Tensor) or src.device.type != \"cuda\"",
     ("test_v033_xpu2",)),
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
     "        got = 0",
     ("test_v0331_audit",)),
    ("A2: get() re-looks-up orig_dtype after _restore (the two-acquisition read)",
     "tex_results.py",
     "            frame, orig_dtype = self._restore(key)",
     "            frame, _discard = self._restore(key)\n"
     "            with self._lock:\n"
     "                _e = self._ram.get(key)\n"
     "                orig_dtype = _e.orig_dtype if _e is not None else None",
     ("test_v0331_audit",)),
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
     "                continue",
     ("test_v0331_audit",)),
    # -- v0.33.2 (the v0.33.1 release-audit findings) ---------------------------------
    ("A1: the spill drops its per-key ordering ticket", "tex_results.py",
     "                    if self._spill_seq.get(key, 0) != seq or self._generation != gen:",
     "                    if self._generation != gen:",
     ("test_v0332_audit",)),
    # The A1 fix went through two wrong shapes before this one; both are mutations here,
    # because "checked, then wrote" and "checked AND wrote" are indistinguishable to any
    # interleaving a test can force from outside -- which is why one of these rows is killed
    # by a SOURCE-shape assertion rather than by pixels.
    ("A1: the ticket check moves back outside the write", "tex_results.py",
     "            with wlock:\n"
     "                with self._lock:\n"
     "                    if self._spill_seq.get(key, 0) != seq or self._generation != gen:\n"
     "                        return            # a newer spill of this key won; touch nothing\n"
     "                if not _atomic_pickle(path, rec):",
     "            with self._lock:\n"
     "                if self._spill_seq.get(key, 0) != seq or self._generation != gen:\n"
     "                    return\n"
     "            if not _atomic_pickle(path, rec):",
     ("test_v0332_audit",)),
    ('A2: the re-admit stops checking the generation', 'tex_results.py',
     '            if gen is not None and gen != self._generation:\n                return None',
     '            if False:\n                return None',
     ("test_v0332_audit",)),
    # The first attempt at this row set `seq = None` after the popleft, which made every
    # spill bail and was killed by a dozen unrelated rows — a mutant that broad says nothing
    # about the claim SITE. This restores the pre-hunt code exactly: claim inside `_spill`.
    ("H1: the spill ticket is claimed at write time again, not at eviction time",
     "tex_results.py",
     "            with self._lock:\n"
     "                gen = self._generation          # A5: the generation this write belongs to",
     "            with self._lock:\n"
     "                gen = self._generation\n"
     "                seq = self._spill_seq.get(key, 0) + 1\n"
     "                self._spill_seq[key] = seq",
     ("test_v0332_audit",)),
    ("H7: the purge marker is read at the re-admit again, not at the read",
     "tex_results.py",
     "                if self._purge_depth:",
     "                if False:",
     ("test_v0332_audit",)),
    # Re-anchored in v0.35. The old anchor was the bare `        finally:`, which went ambiguous
    # the moment a second one appeared at that indent and was reported as a STALE ANCHOR. It is
    # unique again today, but "unique today" is what made it fragile — so it now carries the
    # first line of the body it guards, which no unrelated `finally:` can collide with.
    ("H7: the purge depth is dropped outside the finally again", "tex_results.py",
     "        finally:\n"
     "            # The depth MUST come back down on EVERY exit,",
     "        except BaseException:\n"
     "            raise\n"
     "        if True:\n"
     "            # The depth MUST come back down on EVERY exit,",
     ("test_v0332_audit",)),
    ("H4: _learn_spilled rebinds membership over a racing spill again", "tex_results.py",
     "                self._spilled = None if self.spills != spills_at_entry else found",
     "                self._spilled = found",
     ("test_v0332_audit",)),
    ("H5: a failed spill write counts as a success again", "tex_results.py",
     "                if not _atomic_pickle(path, rec):",
     "                _atomic_pickle(path, rec)\n"
     "                if False:",
     ("test_v0332_audit",)),
    ("A3: clear's tail asserts a definite empty index again", "tex_results.py",
     "            if self.spills != spills_at_entry:\n"
     "                self._disk_bytes = None\n"
     "                self._spilled = None        # unknown beats a confident wrong answer",
     "            if False:\n"
     "                self._disk_bytes = None\n"
     "                self._spilled = None",
     ("test_v0332_audit",)),
    ("A5: the disarm commit-check goes, so an in-flight demote still lands",
     "tex_results.py",
     "                if self._vram_budget is None:",
     "                if False:",
     ("test_v0332_audit",)),
    ("B2/P0-5: remap_suffix_taps becomes the identity", "tex_fusion.py",
     "    if not k:\n"
     "        return outputs\n"
     "    out = {}",
     "    if True:\n"
     "        return outputs\n"
     "    out = {}",
     ("test_v033_phase0",)),

    # ── v0.34.1: the v0.34.0 audit register, and the re-audit's holes ──
    #
    # RETIRED WITH REASON (3 rows), rather than left surviving:
    #  * 'the generation is read AFTER the provider' — the fix stopped depending on
    #    read order at all (both reads now happen under the registration lock), so
    #    there is no order left to mutate. The 'guard at put() goes' row below still
    #    covers the mechanism.
    #  * 'the wake path stops checking shed_requested' — every setter of that flag
    #    also removes the job from its deque under the same lock, so the branch is
    #    unreachable today. It is kept in the source as documented belt-and-braces
    #    (see its comment); an unkillable row asserting it would be decorative.
    #  * 'the generation guard at put() goes' — RETIRED BY ITS OWN FIX. The /simplify pass
    #    moved the generation into the pool KEY, so a stale insert now lands under a key
    #    nobody can look up and removing the guard changes no observable outcome. The row
    #    was killable only while the guard was the single door; keeping it would assert
    #    that the weaker design is still load-bearing.
    #  * 'an unlanded shapeless promise is keyed anyway' — the gate term it mutates
    #    is unobservable through `cook_fused_cached`: an unlanded promise is refused
    #    by `_full()`'s E7007 before the term could change any outcome, and the
    #    raise that DOES matter (a direct `boundary_lineage_key` call getting a
    #    clear refusal instead of a key over an unknown resolution) lives in
    #    `_shapes` and is pinned by the H-hole row. The term stays as the
    #    contract-shaped refusal for a serve path that must fall back, not raise.
    ('v0.34.1 B: cancel() stops treating WAITING like PENDING', 'tex_cookqueue.py',
     '            if job.state in (PENDING, WAITING):',
     '            if job.state == PENDING:',
     ("test_v0341_audit",)),
    ('v0.34.1 C: an integer provider frame is accepted again', 'tex_provider.py',
     '    if not frame.is_floating_point():',
     '    if False:',
     ("test_v0341_audit",)),
    ('v0.34.1 C: f64 is no longer narrowed at the pool boundary', 'tex_provider.py',
     '    if frame.dtype == torch.float64:\n        frame = frame.to(torch.float32)',
     '    if False:\n        frame = frame.to(torch.float32)',
     ("test_v0341_audit",)),
    ('v0.34.1 D: the const-coord grid falls back to (1,1,1) again', 'tex_runtime/stdlib_core.py',
     'def _uniform_grid():\n    return getattr(_cook_ctx, "grid", None)',
     'def _uniform_grid():\n    return None',
     ("test_v0341_audit",)),
    ('v0.34.1 D: the codegen tier stops publishing the grid', 'tex_runtime/codegen.py',
     '    _grid_token = _stdlib_set_cook_grid(spatial_shape, dtype)',
     '    _grid_token = _stdlib_set_cook_grid(None, None)',
     ("test_v0341_audit",)),
    ('v0.34.1 E: the pool stops copying at its boundary', 'tex_provider.py',
     '    if not getattr(prov, "frames_are_owned", False):',
     '    if False:',
     ("test_v0341_audit",)),
    ('v0.34.1 F: a speculative wake failure alarms again', 'tex_cookqueue.py',
     '                if job.klass == SPECULATIVE and str(getattr(err, "_code", "")).startswith("E7"):',
     '                if False:',
     ("test_v0341_audit",)),
    ('v0.34.1 G: land(None) is accepted again', 'tex_marshalling.py',
     '        if value is None:\n            # E7006, not a bare ValueError',
     '        if False:\n            # E7006, not a bare ValueError',
     ("test_v0341_audit",)),
    ('v0.34.1 G: fail(None) is accepted again', 'tex_marshalling.py',
     '        if exc is None:\n            from .tex_runtime.interpreter import InterpreterError',
     '        if False:\n            from .tex_runtime.interpreter import InterpreterError',
     ("test_v0341_audit",)),
    # NEG-2 moved _is_tensor_binding out of tex_engine.py. The anchor text also had to be
    # rewritten: DATA-6 added the PlanesValue arm and re-flowed the return into two lines, so
    # this row had matched 0x and asserted NOTHING since. It went unseen because the harness
    # is not part of run_all.py - the same class the suite column (MUT-1) was added for.
    ('v0.34.1 H: a Promise is not a tensor binding again', 'tex_chain.py',
     '    return (isinstance(v, torch.Tensor) or v.__class__ is _Promise\n'
     '            or v.__class__ is _PlanesValue)',
     '    return (isinstance(v, torch.Tensor)\n'
     '            or v.__class__ is _PlanesValue)',
     ("test_v0341_audit",)),
    ('v0.34.1 I: a >=5-D tensor types FLOAT again', 'tex_marshalling.py',
     '        elif value.dim() >= 5:',
     '        elif False:',
     ("test_v0341_audit",)),
    # ── v0.35 phase 0 ──
    # CF-6's two halves, each mutated back to the state an audit caught it in. The first is the
    # tier split: codegen deriving the grid ITSELF is how invariant #2 broke, so the row puts
    # the private first-wins loop back and the pin's `auto` leg must notice.
    ('CF-6: codegen derives the grid itself again (first-wins, invariant #2)',
     'tex_runtime/compiled.py',
     '    sp = _consensus_extent(bindings, program, roi=roi)',
     '    sp = None\n'
     '    for _v in bindings.values():\n'
     '        if isinstance(_v, torch.Tensor) and _v.dim() >= 3:\n'
     '            sp = (_v.shape[0], _v.shape[1], _v.shape[2])\n'
     '            break\n'
     '    if roi is not None and sp is not None:\n'
     '        sp = (sp[0], roi[3], roi[2])',
     ("test_v035_hygiene",)),
    ('CF-6: an unread binding is a consensus participant again',
     'tex_runtime/interpreter.py',
     '            if (name == "OUT" or name not in read\n'
     '                    or not isinstance(v, torch.Tensor) or v.dim() < 3):',
     '            if (name == "OUT"\n'
     '                    or not isinstance(v, torch.Tensor) or v.dim() < 3):',
     ("test_v035_hygiene",)),
    ('CF-6: the interpreter goes back to first-wins', 'tex_runtime/interpreter.py',
     '    if b_split or (hw_split and roi is None):',
     '    if False:',
     ("test_v035_hygiene",)),
    # The ROI branch as a BYPASS rather than an axis selector: participation decided before
    # `roi` is applied, so an unread binding could raise the batch under a window and not on
    # the whole frame. Both axis-blind gates downstream let that through.
    ('CF-6: the ROI branch bypasses the participation rule again',
     'tex_runtime/interpreter.py',
     '    if b_split or (hw_split and roi is None):',
     '    if roi is not None:\n'
     '        return (b, roi[3], roi[2])\n'
     '    if b_split or hw_split:',
     ("test_v035_hygiene",)),
    # ENG-14 moved _preflight_memory out of tex_engine.py; the anchor text is unchanged.
    ('CF-6: the peak-bytes preflight sizes itself first-wins again', 'tex_tiling.py',
     '        spatial = _consensus_extent(bindings, program)',
     '        spatial = next(((v.shape[0], v.shape[1], v.shape[2]) for v in bindings.values()\n'
     '                        if isinstance(v, torch.Tensor) and v.dim() >= 3), None)',
     ("test_v035_hygiene",)),
    # The mirror that reaches pixels: `auto` gates on cook_px, so a first-wins cook_px makes
    # the resolved precision depend on binding order. CUDA-only, like the pin that kills it.
    ('CF-6: the auto-precision gate sizes itself first-wins again', 'tex_engine.py',
     '    _grid = _consensus_extent(bindings, program)\n'
     '    cook_px = (_grid[1] * _grid[2]) if _grid is not None else 0',
     '    cook_px = next((v.shape[1] * v.shape[2] for v in bindings.values()\n'
     '                    if isinstance(v, torch.Tensor) and v.dim() >= 3), 0)',
     ("test_v035_hygiene",)),
    ('CF-2: a whole-frame partial recook skips the prefix validity check again',
     'examples/host_demo.py',
     '        if roi is None and dirty_from > 0 and any(\n'
     '                self._valid[j] is not None or j in self._declined '
     'for j in range(dirty_from)):\n'
     '            dirty_from = 0',
     '        if False:\n'
     '            dirty_from = 0',
     ("test_v035_hygiene",)),
    # R1's mirror of the grid rule: if this stops asking `_consensus_extent`, an ROI cook can
    # be served for a window the whole-frame cook would not agree with.
    ('CF-6: run_roi stops mirroring the whole-frame grid rule', 'tex_memory.py',
     '    if record_trace and (_grid[2], _grid[1]) != (W, H):',
     '    if False:',
     ("test_v030_phase1",)),
    ('CF-4: requalify evicts whatever is under the key, not the entry it replaced',
     'tex_results.py',
     '            if self._ram.get(preview_key) is prev:\n'
     '                self._remove(preview_key)',
     '            if True:\n'
     '                self._remove(preview_key)',
     ("test_v035_hygiene",)),
    ('CF-1: the patch stops inheriting the base HOME (the one-way trip to the CPU)',
     'tex_results.py',
     '                home = src.home',
     '                pass',
     ("test_v035_hygiene",)),
    # BRIEF-9 T3: a uniform (once-per-cook, 0-dim) output must stay 0-dim through every
    # run_tiled strip, not get broadcast across the tile height like a spatial one.
    ('BRIEF-9 T3: run_tiled broadcasts a uniform output across strips instead of leaving '
     'it 0-dim', 'tex_memory.py',
     '                outputs[name] = strip_out  # scalar/string: any strip suffices',
     "                outputs[name] = (strip_out.expand(H_total) if hasattr(strip_out, "
     "'dim') and strip_out.dim() == 0 else strip_out)",
     ("test_v035_hygiene",)),
    # ── residency hints (`touch` / `in`) ──
    # Each puts back one way a hint turns into a read, outranks demand, or stops being atomic.
    ('touch: a hint counts as a hit (speculation folded into the read counters)',
     'tex_results.py',
     '            self.touches += 1\n'
     '            return True',
     '            self.hits += 1\n'
     '            return True',
     ("test_v033_cache8",)),
    ('touch: the hint takes the top slot from the most recent demand', 'tex_results.py',
     '            if key not in self._ram:\n'
     '                return False\n'
     '            mru = next(reversed(self._ram))\n'
     '            if key != mru:\n'
     '                self._ram.move_to_end(key)\n'
     '                self._ram.move_to_end(mru)',
     '            if key not in self._ram:\n'
     '                return False\n'
     '            mru = next(reversed(self._ram))\n'
     '            if key != mru:\n'
     '                self._ram.move_to_end(key)\n'
     '                pass',
     ("test_v033_cache8",)),
    ('touch: the hint is spelled as a read again (delegates to get)', 'tex_results.py',
     '        with self._lock:\n'
     '            if key not in self._ram:\n'
     '                return False',
     '        return self.get(key, copy=False) is not None\n'
     '        with self._lock:\n'
     '            if key not in self._ram:\n'
     '                return False',
     ("test_v033_cache8",)),
    ('touch: reading the top entry and the moves stop being one critical section',
     'tex_results.py',
     '        with self._lock:\n'
     '            if key not in self._ram:\n'
     '                return False',
     '        if True:\n'
     '            if key not in self._ram:\n'
     '                return False',
     ("test_v033_cache8",)),
    ('in: membership is answered by a read (counts, restores, promotes)', 'tex_results.py',
     '            return key in self._ram',
     '            return self.get(key, copy=False) is not None',
     ("test_v033_cache8",)),
    # Codegen flow-mode scoping. The first row is the defect verbatim: the general for-loop
    # emitter stops pinning the mode for its own body, so a static/while loop's native flow
    # control leaks in and a nested `continue` skips the update and the counter — the rows
    # time out rather than fail, which run_tree scores as KILLED (hung). The second is the
    # over-reach in the other direction (pin the native form instead), and the third drops the
    # RESTORE, which strands the enclosing loop's own transfers in the wrong mode.
    ('flow scope: the general for-loop stops pinning its body flow mode',
     'tex_runtime/codegen.py',
     '        if _body_has_break_continue(stmt.body, (ContinueStmt,)):\n'
     '            self._use_native_flow_control = False',
     '        if False:\n'
     '            self._use_native_flow_control = False',
     ("test_codegen_flow_scope",)),
    ('flow scope: the general for-loop pins NATIVE flow for its body',
     'tex_runtime/codegen.py',
     '        if _body_has_break_continue(stmt.body, (ContinueStmt,)):\n'
     '            self._use_native_flow_control = False',
     '        if True:\n'
     '            self._use_native_flow_control = True',
     ("test_codegen_flow_scope",)),
    ('flow scope: the general for-loop never restores the enclosing mode',
     'tex_runtime/codegen.py',
     '        self._emit_body_with_flow(stmt.body)\n'
     '        self._use_native_flow_control = saved_flow',
     '        self._emit_body_with_flow(stmt.body)',
     ("test_codegen_flow_scope",)),
    # ── codegen value parity (the two wrong-rank reads) ────────────────────────────────
    # Both directions of the _SPATIAL_BUILTINS derivation are mutated: `fi` dropped from the
    # set (the defect as it shipped) and a genuinely 0-dim builtin added to it.
    ('_SPATIAL_BUILTINS forgets fi again (a loop reading it compiles scalar)',
     'tex_runtime/codegen.py',
     '_SPATIAL_BUILTINS: frozenset[str] = frozenset(("u", "v", "ix", "iy", "fi"))',
     '_SPATIAL_BUILTINS: frozenset[str] = frozenset(("u", "v", "ix", "iy"))',
     ("test_codegen_value_parity",)),
    ('_SPATIAL_BUILTINS gains a 0-dim builtin (the fast path is lost)',
     'tex_runtime/codegen.py',
     '_SPATIAL_BUILTINS: frozenset[str] = frozenset(("u", "v", "ix", "iy", "fi"))',
     '_SPATIAL_BUILTINS: frozenset[str] = frozenset(("u", "v", "ix", "iy", "fi", "iw"))',
     ("test_codegen_value_parity",)),
    ('the invocation seam stops staging vec params (rank-1 $tint.r again)',
     'tex_runtime/codegen.py',
     '    _stage_vec_params(bindings, device, dtype)\n'
     '    if program is not None:\n'
     '        _stage_wire_scalars(bindings, device, dtype, cg_fn, program)\n'
     '    _grid_token = _stdlib_set_cook_grid(spatial_shape, dtype)',
     '    if program is not None:\n'
     '        _stage_wire_scalars(bindings, device, dtype, cg_fn, program)\n'
     '    _grid_token = _stdlib_set_cook_grid(spatial_shape, dtype)',
     ("test_codegen_value_parity",)),
    ('the vec-param staging drops the [1,1,1,C] reshape', 'tex_runtime/interpreter.py',
     '    if t.dim() == 1 and t.shape[0] in (2, 3, 4):\n'
     '        t = t.view(1, 1, 1, -1)',
     '    if False:\n'
     '        t = t.view(1, 1, 1, -1)',
     ("test_codegen_value_parity",)),
    ('_params_on_device overwrites a staged vec param with a rank-1 one',
     'tex_runtime/compiled.py',
     '        if value is None or isinstance(value, (torch.Tensor, str)) or is_vec_param_list(value):',
     '        if value is None or isinstance(value, (torch.Tensor, str)):',
     ("test_codegen_value_parity",)),
    # ── TRK-25 (region dependence) ────────────────────────────────────────────────────
    # Each row closes a different route, so each gets its own: a mutant only one of them
    # kills would otherwise hide behind the others.
    #
    # These three are the reason the suite column exists (MUT-1). They shipped in v0.36.0
    # against a hand-kept import list that stopped at v0.35, so `test_v036_region_dependence`
    # — the only file with a test that can kill any of them — was never loaded, and all three
    # printed `SURVIVED (0 failing rows)` for three releases.
    ("TRK-25: roi_plan stops refusing a region-dependent program", "tex_roi.py",
     "    if blocked or region_dep:",
     "    if blocked:",
     ("test_v036_region_dependence",)),
    # ENG-14 moved the planners out of tex_engine.py; the anchor text is unchanged.
    ("TRK-25: the strip planner stops consulting the predicate", "tex_tiling.py",
     "        if tex_roi.region_dependent_cached(program, fingerprint, binding_types, code):\n"
     "            return None\n"
     "        return n",
     "        if False:\n"
     "            return None\n"
     "        return n",
     ("test_v036_region_dependence",)),
    ("TRK-25: the predicate fails OPEN instead of closed", "tex_roi.py",
     "        return bool(loops) and _language_tuple(program, code) < MASKED_FLOW_SINCE\n"
     "    except Exception:\n"
     "        return True",
     "        return bool(loops) and _language_tuple(program, code) < MASKED_FLOW_SINCE\n"
     "    except Exception:\n"
     "        return False",
     ("test_v036_region_dependence",)),
    # TRK-32: a per-pixel value cast straight to a STRING. Its own row, same reasoning as the
    # three above — clause (d) is a distinct id-set from clause (c), so a mutant that drops
    # only `casts` from the never-sunset check must be caught by a row that fails without it.
    ("TRK-32: clause (d) stops being treated as never-sunsetting", "tex_roi.py",
     "        if strings or casts:\n"
     "            return True                       # clauses (c) and (d) never sunset",
     "        if strings:\n"
     "            return True                       # clauses (c) and (d) never sunset",
     ("test_v036_region_dependence",)),
    # ── LANG-L2 (E3015: break/continue inside a function defined inside a loop) ────────
    # TRK-28's underlying scope defect: `_check_function_def` must reset `_loop_depth` to 0
    # for the body, or a function defined inside a loop inherits that loop's depth and the
    # E3002/E3015 guard in `_check_break_continue` never fires (the exact bug that let the
    # two tiers disagree — see docs/masked-control-flow.md §3). Removing just the reset
    # (leaving `_in_function_body` alone) reproduces it precisely.
    ("LANG-L2: function-def stops resetting _loop_depth for its body", "tex_compiler/type_checker.py",
     "        saved_loop_depth = self._loop_depth\n"
     "        self._loop_depth = 0\n",
     "        saved_loop_depth = self._loop_depth\n",
     ("test_lang_l2_e3015",)),
    # ── LANG-L6 (the satellite tiers under language 0.25) ──────────────────────────────
    # Both directions for each gate: the check REMOVED (a flagged program with a per-pass
    # live test would be handed to CUDA-graph capture / fp16 again) and the check WIDENED to
    # every flagged program (a transfer-free 0.25 program would lose the tier it has today).
    ("LANG-L6: the capture gate stops asking whether the masked path syncs",
     "tex_runtime/graphed.py",
     "    if _masked_flow_syncs(program, _masked_flow):\n        return (False, 0)\n",
     "    if False:\n        return (False, 0)\n",
     ("test_lang_l6_satellites",)),
    ("LANG-L6: the capture gate declines EVERY flagged program", "tex_runtime/graphed.py",
     "    return (not plan.complete) or bool(plan.sync_points or plan.scatter_sites)\n",
     "    return True\n",
     ("test_lang_l6_satellites",)),
    ("LANG-L6: auto stops declining a 0.25 per-pixel for", "tex_runtime/precision_policy.py",
     "    if _masked_per_pixel_for(program, _masked_flow):\n",
     "    if False:\n",
     ("test_lang_l6_satellites",)),
    ("LANG-L6: auto declines EVERY flagged program", "tex_runtime/precision_policy.py",
     "    if not plan.per_pixel_loops:\n        return False\n",
     "    if not plan.per_pixel_loops:\n        return True\n",
     ("test_lang_l6_satellites",)),
    # ── CG-1 (the emitter's `id()`-keyed type map is only truthful while its keys live) ──
    # Four rows, one per half of the fix and one per guard of the narrowing. The first is the
    # bug as it shipped: the emitter reading the map it was handed, dead entries and all.
    ("CG-1: the emitter reads the unnarrowed map again", "tex_runtime/codegen.py",
     "        gen = _CodeGen(_live_type_map(program, type_map))",
     "        gen = _CodeGen(type_map)",
     ("test_cg1_typemap_liveness",)),
    ("CG-1: narrowing keeps an entry the map did not record for THIS node", "tex_runtime/codegen.py",
     "        if t is not None and (is_own is None or is_own(node)):",
     "        if t is not None:",
     ("test_cg1_typemap_liveness",)),
    ("CG-1: the checker stops pinning the nodes it types", "tex_compiler/type_checker.py",
     "        self._types.record(node, t)",
     "        self._types[id(node)] = t",
     ("test_cg1_typemap_liveness",)),
    ("CG-1: the optimizer registers a temp without its pin", "tex_compiler/optimizer.py",
     "    record = getattr(type_map, \"record\", None)\n"
     "    if record is not None:\n"
     "        record(node, t)\n"
     "    else:\n"
     "        type_map[id(node)] = t",
     "    type_map[id(node)] = t",
     ("test_cg1_typemap_liveness",)),
    # ── CG-2 (L5-F1: a scatter into a rank-<3 `@` buffer; L4-F1: the fuzz seed's width) ──
    # The codegen scatter emission widens a buffer of rank < 3 on the interpreter's own
    # `needs_new_buf` condition and preserves the old value into it. Each half of that is
    # a separate mutant, because a row that only checks "no IndexError" would pass with the
    # seed silently zeroed.
    ("CG-2: scatter emission stops widening a rank-<3 buffer", "tex_runtime/codegen.py",
     '        need_buf = (f"{name!r} not in _bind or not _torch.is_tensor(_bind[{name!r}]) "\n'
     '                    f"or _bind[{name!r}].dim() < 3")',
     '        need_buf = (f"{name!r} not in _bind or not _torch.is_tensor(_bind[{name!r}]) "\n'
     '                    f"or False")',
     ("test_codegen_optimizer",)),
    ("CG-2: scatter emission forgets the old value when it widens", "tex_runtime/codegen.py",
     '        self._emit(f"if _torch.is_tensor(_sold): _bind[{name!r}][...] = _sold")',
     '        pass',
     ("test_codegen_optimizer",)),
    # The generator half: a stencil accumulator seeded `vec3` whatever the wire carries is
    # exactly L4-F1, and the row that pins the rate at zero must red on it.
    ("CG-2: the fuzz generator seeds vec3 whatever the wire carries", "tests/test_v017_phase1.py",
     '    vt = f"vec{channels}"',
     '    vt = "vec3"',
     ("test_v017_phase1",)),
]


def suite_modules(rows):
    """The runner's import list: the UNION of the killing suites of the rows being swept.

    Derived from the rows, never hand-kept (MUT-1) — see the module docstring. First-appearance
    order, so the list stays chronological and a row appended at the end cannot reorder the
    modules an earlier row loads.
    """
    seen = []
    for row in rows:
        for name in row[4]:
            if name not in seen:
                seen.append(name)
    return tuple(seen)


def validate_rows(rows, tests_dir=TESTS_DIR):
    """Every row must name at least one killing suite, and every named suite must exist.

    Returns a list of human-readable problems, each NAMING THE ROW. A row whose suite is
    missing or misspelled is the drift this file exists to make impossible, so it stops the
    sweep rather than printing a verdict beside it.
    """
    problems = []
    for row in rows:
        if len(row) != 5:
            problems.append(f"{row[0]!r}: a row is (label, file, old, new, suites); "
                            f"this one has {len(row)} fields and names no killing suite")
            continue
        label, _rel, _old, _new, suites = row
        if not isinstance(suites, tuple) or not suites:
            problems.append(f"{label!r}: names no killing suite (suites must be a non-empty "
                            f"tuple of test module names)")
            continue
        for name in suites:
            if not isinstance(name, str) or not (tests_dir / f"{name}.py").is_file():
                problems.append(f"{label!r}: names a killing suite that does not exist in "
                                f"{tests_dir}: {name!r}")
    return problems


# The child process. `{modules}` is filled from `suite_modules(rows)` — there is deliberately
# no literal `import test_...` anywhere in this template, because a literal is exactly what
# drifted twice. Per-module failure counts come back as FAILMOD lines so a KILLED verdict can
# be checked against the suite the row claims kills it.
RUNNER_TEMPLATE = """
import sys
sys.argv = ["x"]
sys.path.insert(0, r"{tests}")
sys.path.insert(0, r"{parent}")
from helpers import SubTestResult
_names = {modules}
_mods = [(_n, __import__(_n)) for _n in _names]
r = SubTestResult()
for _name, m in _mods:
    _before = r.failed
    for n in sorted(x for x in dir(m) if x.startswith("test_")):
        try:
            getattr(m, n)(r)
        except Exception:
            r.fail(n, "raised")
    if r.failed > _before:
        print("FAILMOD", _name, r.failed - _before)
print("FAILCOUNT", r.failed)
"""


def runner_source(rows, tests, parent):
    """The subprocess source for `rows`. The import list is derived; nothing else varies."""
    return RUNNER_TEMPLATE.format(tests=tests, parent=parent,
                                  modules=repr(list(suite_modules(rows))))


def run_tree(root, rows):
    """Run the derived suite against `root`. Returns (failcount, {module: failures}).

    failcount is -2 for a hang and -1 for a crash (no FAILCOUNT line), exactly as before.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TEX_CACHE_DIR"] = tempfile.mkdtemp(prefix="mutcache_")
    # The flow-scope rows cook 2x2 scalar-loop programs in child processes, so their verdict is
    # a timeout when a mutant reintroduces a non-terminating loop. Shorten the per-program
    # budget: at the 30 s default a single hang mutant would sit here for twenty minutes.
    env["TEX_FLOW_TIMEOUT"] = "8"
    src = runner_source(rows, str(root / "tests"), str(root.parent))
    try:
        out = subprocess.run(
            [VENV, "-c", src],
            capture_output=True, text=True, cwd=str(root / "tests"), env=env, timeout=3600)
    except subprocess.TimeoutExpired:
        return -2, {}
    by_mod = {}
    failed = -1
    for line in out.stdout.splitlines():
        if line.startswith("FAILMOD"):
            parts = line.split()
            if len(parts) == 3:
                by_mod[parts[1]] = int(parts[2])
        elif line.startswith("FAILCOUNT"):
            failed = int(line.split()[1])
    return failed, by_mod


def main(argv):
    rows = MUTATIONS
    wanted = [a for a in argv if not a.startswith("-")]
    if "--list" in argv:
        for label, rel, _old, _new, suites in rows:
            print(f"{label:58s} {rel:28s} {','.join(suites)}")
        return 0
    if "--rows" in argv:
        pat = argv[argv.index("--rows") + 1]
        rows = [row for row in rows if pat.lower() in row[0].lower()]
        if not rows:
            print(f"no mutation row matches {pat!r}")
            return 2
    elif wanted:
        print(f"unrecognised argument(s): {wanted}")
        return 2

    problems = validate_rows(rows)
    if problems:
        print("the suite column is broken - the sweep asserts nothing until it is fixed:")
        for p in problems:
            print(f"  - {p}")
        return 2

    modules = suite_modules(rows)
    print(f"{len(rows)} mutation row(s); derived suite ({len(modules)} modules): "
          f"{', '.join(modules)}", flush=True)

    base_root = pathlib.Path(tempfile.mkdtemp(prefix="mutbase_"))
    shutil.copytree(SRC, base_root / "TEX_Wrangle",
                    ignore=shutil.ignore_patterns("__pycache__", ".tex_cache", ".git",
                                                  "results"))

    # The baseline. Without it every verdict is a guess: a suite that fails (or fails to
    # import) on the UNMUTATED tree scores every row as KILLED, which reads as a clean sweep
    # and is the opposite of one.
    base_failed, base_by_mod = run_tree(base_root / "TEX_Wrangle", rows)
    if base_failed != 0:
        print(f"BASELINE IS NOT CLEAN ({base_failed} failing rows, {base_by_mod}) - "
              f"every verdict below would be meaningless; fix the tree first.")
        shutil.rmtree(base_root, ignore_errors=True)
        return 2
    print("baseline: 0 failing rows on the unmutated tree", flush=True)

    stale: list = []
    misattributed: list = []
    survived: list = []
    killed = 0
    print()
    print(f"{'mutation':58s} {'verdict'}")
    print("-" * 84, flush=True)
    for label, rel, old, new, suites in rows:
        work_root = pathlib.Path(tempfile.mkdtemp(prefix="mut_"))
        shutil.copytree(base_root / "TEX_Wrangle", work_root / "TEX_Wrangle")
        p = work_root / "TEX_Wrangle" / rel
        s = p.read_text(encoding="utf-8")
        if s.count(old) != 1:
            # NOT a skip. A row whose anchor no longer matches proves NOTHING while printing
            # something that reads like a benign outcome — and it goes stale exactly when the
            # source it guards is edited, i.e. when it is most needed. Two rows drifted this way
            # in v0.34.1 (a /simplify pass changed the very lines they anchored on) and were
            # visible only because someone read the log. Loud, and counted.
            stale.append(label)
            print(f"{label:58s} *** STALE ANCHOR *** (matched {s.count(old)}x) "
                  f"- re-anchor or retire it; this row asserts nothing", flush=True)
            shutil.rmtree(work_root, ignore_errors=True)
            continue
        p.write_text(s.replace(old, new, 1), encoding="utf-8", newline="")
        failed, by_mod = run_tree(work_root / "TEX_Wrangle", rows)
        verdict = {"-2": "KILLED (hung)", "-1": "KILLED (crashed)"}.get(
            str(failed), "*** SURVIVED ***" if failed == 0 else "KILLED")
        note = ""
        if failed == 0:
            survived.append(label)
        else:
            killed += 1
            if by_mod:
                note = "  by " + ",".join(f"{k}:{v}" for k, v in by_mod.items())
                if not any(name in by_mod for name in suites):
                    # Killed, but not by the suite that claims to pin it: the declared guard
                    # is decorative and the row is passing on somebody else's coverage.
                    misattributed.append((label, suites, tuple(by_mod)))
                    note += "  *** NOT by its declared suite ***"
        print(f"{label:58s} {verdict:20s} ({failed} failing rows){note}", flush=True)
        shutil.rmtree(work_root, ignore_errors=True)
    shutil.rmtree(base_root, ignore_errors=True)

    print()
    print(f"{killed} killed, {len(survived)} survived, {len(stale)} stale anchor(s), "
          f"{len(misattributed)} misattributed")
    if survived:
        print()
        print(f"{len(survived)} SURVIVED - the test that claims to pin this is decorative:")
        for label in survived:
            print(f"  - {label}")
    if stale:
        print()
        print(f"{len(stale)} STALE ANCHOR(S) - these rows asserted nothing:")
        for label in stale:
            print(f"  - {label}")
    if misattributed:
        print()
        print(f"{len(misattributed)} MISATTRIBUTED - killed, but not by the declared suite:")
        for label, suites, actual in misattributed:
            print(f"  - {label}: declares {list(suites)}, killed by {list(actual)}")
    return 1 if (stale or misattributed) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
