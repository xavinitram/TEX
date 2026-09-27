"""FIX-ROI49 Q4 — `tex_roi._fold_program` called `tex_lazy._prune_static_flow` unconditionally
on EVERY call, even the common program with no dead branch to strip at all: `_prune_static_flow`
is a full recursive rebuild (a fresh `out` list at every level, recursing into every `IfElse`/
`WhileLoop`/`ForLoop`/`FunctionDef` body) with no fast path for "there is nothing here a fold
could ever have made literal" (R3#2 of the v0.49 Phase C review, measured 9-20% of
`_fold_program`'s own per-call cost on an 8-statement, no-dead-branch program). `frame_window`
and `batch_sliceable` call `_fold_program` with NO memo of their own (per TRK-219's own note), so
this cost is paid on every uncached call, dead branch or not.

FIX: `tex_lazy._has_prunable_flow(stmts)` is a cheap (no-allocation) over-approximate pre-check
— an `IfElse` whose condition is already a `NumberLiteral` (either arm), or a `WhileLoop` whose
condition is a literal-false `NumberLiteral` — walking the exact same nodes
`_prune_static_flow` would recurse into. `_fold_program` now calls `_prune_static_flow` only
when this pre-check says there is something to prune; it can never say "nothing to prune" when
pruning would actually change something (invariant #11's own direction), so skipping the
rebuild in that case changes NO answer `_fold_program` returns.

COUNTS-PROVEN, not timed: this file counts calls to `tex_roi._prune_static_flow` (a spy
wrapping the real function) rather than asserting a wall-clock number, per the standing
avoidance of unbounded timing assertions CI (which runs coverage) must not deselect.

RED AT BASE (`32f6917`, confirmed by running): `_prune_static_flow` is called exactly once for
EVERY `_fold_program` call, including a program with no `IfElse`/`WhileLoop` at all and a
program whose `IfElse` condition never becomes a literal (a non-`$param` comparison) — the
walk runs and finds nothing to do, every time.

ComfyUI-invisible because: `_fold_program` is a ROI/lazy-analysis helper off the default cook
path; its return value (the folded, possibly-pruned `Program`) is byte-for-byte the same
either way — only whether the (now-skippable) rebuild runs changes.
"""
from helpers import *

from TEX_Wrangle import tex_roi
from TEX_Wrangle.tex_lazy import _has_prunable_flow


def _count_prune_calls(code: str, params: dict):
    """Call `tex_roi._fold_program(code, params)` with `_prune_static_flow` spied on (call
    COUNT, never timing) and return `(call_count, folded_program)`."""
    orig = tex_roi._prune_static_flow
    calls = [0]

    def _spy(stmts):
        calls[0] += 1
        return orig(stmts)

    tex_roi._prune_static_flow = _spy
    try:
        tex_roi.clear_roi_memo()
        folded = tex_roi._fold_program(code, params)
    finally:
        tex_roi._prune_static_flow = orig
    return calls[0], folded


# ── Part 1: `_has_prunable_flow` agrees with what `_prune_static_flow` would actually do ─────

def test_q4_has_prunable_flow_agrees_with_the_real_prune(r: SubTestResult):
    """For a battery of pre-prune trees (the exact tree `_fold_program` hands to the prune
    decision — parse + substitute + fold + revert already applied), `_has_prunable_flow` must
    say True if and only if `_prune_static_flow` actually changes that tree. This is the
    contract the fast path depends on to never under-approximate (invariant #11)."""
    print("\n--- FIX-ROI49 Q4: _has_prunable_flow agrees with the real prune, case by case ---")
    cases = [
        ("no control flow at all", "@OUT = @A * $k;", {"k": 2.0}),
        ("IfElse folds to a verified literal (true arm taken)",
         "if ($k > 0.5) { @OUT = @A; } else { @OUT = @A * 2.0; }", {"k": 1.0}),
        ("IfElse folds to a verified literal (false arm taken)",
         "if ($k > 0.5) { @OUT = @A; } else { @OUT = @A * 2.0; }", {"k": 0.0}),
        ("WhileLoop folds to literal-false (dropped)",
         "float s = 0.0;\nwhile ($k > 0.5) { s = s + 1.0; }\n@OUT = @A + vec4(s);", {"k": 0.0}),
        ("nested: literal IfElse inside a ForLoop body",
         "for (int i = 0; i < 3; i++) { if ($k > 0.5) { @OUT = @A; } else { @OUT = @A; } }",
         {"k": 1.0}),
        ("symbolic condition, never folds (no $param involved)",
         "if (@A.r > 0.5) { @OUT = @A; } else { @OUT = @A * 2.0; }", {}),
    ]
    bad = []
    for label, code, params in cases:
        pre_holder = {}
        orig = tex_roi._prune_static_flow

        def _spy(stmts, _pre=pre_holder, _orig=orig):
            _pre["pre"] = stmts
            return _orig(stmts)

        tex_roi._prune_static_flow = _spy
        try:
            tex_roi.clear_roi_memo()
            tex_roi._fold_program(code, params)
        finally:
            tex_roi._prune_static_flow = orig
        pre = pre_holder.get("pre")
        if pre is None:
            # The fast path skipped the call entirely: re-derive the same pre-prune shape via
            # the lower-level pieces `_fold_program` itself uses, and check the pre-check's
            # verdict directly against the real prune on that tree.
            tex_roi.clear_roi_memo()
            program = tex_roi.clone_tree(tex_roi._pristine_program(code))
            subs = {name: tex_roi.NumberLiteral(value=tex_roi._fp32(v),
                                                is_int=isinstance(v, (bool, int)))
                    for name, v in params.items() if isinstance(v, (bool, int, float))}
            stmts = program.statements
            if subs:
                for stmt in stmts:
                    tex_roi._substitute_params(stmt, subs)
                pre_fold_1 = tex_roi._capture_pre_fold_conditions(stmts)
                stmts = tex_roi._fold_all(stmts)
                stmts = tex_roi._propagate_literal_locals(stmts)
                pre_fold_2 = tex_roi._capture_pre_fold_conditions(stmts)
                stmts = tex_roi._fold_all(stmts)
                tex_roi._revert_unverified_folds(stmts, pre_fold_1, pre_fold_2)
            pre = stmts
        predicted = _has_prunable_flow(pre)
        actually_pruned = repr(orig(pre)) != repr(pre)
        if predicted != actually_pruned:
            bad.append(f"{label}: _has_prunable_flow said {predicted}, but "
                       f"_prune_static_flow actually changed the tree: {actually_pruned}")
    if bad:
        r.fail("Q4 has_prunable_flow agreement", "\n  ".join(bad))
        return
    r.ok(f"{len(cases)} cases: _has_prunable_flow agrees with the real prune")


# ── Part 2: the fast path actually skips the call — COUNTS, not timing ───────────────────────

def test_q4_no_prunable_condition_skips_the_prune_walk(r: SubTestResult):
    """RED AT BASE: a program with no control flow at all never calls `_prune_static_flow`
    after the fix (it did, unconditionally, at base)."""
    print("\n--- FIX-ROI49 Q4: no control flow at all -> zero prune-walk calls ---")
    n, folded = _count_prune_calls("@OUT = @A * $k;", {"k": 2.0})
    if n != 0:
        r.fail("Q4 skip (no control flow)", f"_prune_static_flow called {n} time(s), expected 0")
        return
    if "OUT" not in repr(folded.statements):
        r.fail("Q4 skip (no control flow)", "folded program lost its @OUT assignment")
        return
    r.ok("0 calls, @OUT still present")


def test_q4_a_symbolic_condition_that_never_folds_skips_the_prune_walk(r: SubTestResult):
    """RED AT BASE: an `IfElse` whose condition depends on IMAGE data (`@A.r > 0.5`), never a
    `$param`, can never fold to a `NumberLiteral` — `_has_prunable_flow` must see that and
    skip the walk too, not just the zero-control-flow case."""
    print("\n--- FIX-ROI49 Q4: a never-literal IfElse condition -> zero prune-walk calls ---")
    code = "if (@A.r > 0.5) { @OUT = @A; } else { @OUT = @A * 2.0; }"
    n, folded = _count_prune_calls(code, {})
    if n != 0:
        r.fail("Q4 skip (symbolic condition)",
               f"_prune_static_flow called {n} time(s), expected 0")
        return
    r.ok("0 calls")


def test_q4_a_verified_dead_branch_still_calls_the_prune_walk(r: SubTestResult):
    """Control: a program whose `$param`-fed condition DOES fold to an fp32-verified literal
    (a genuinely prunable dead branch — the TRK-219 class) must still reach
    `_prune_static_flow` at least once; the fast path must never suppress a real prune."""
    print("\n--- FIX-ROI49 Q4 control: a verified dead branch still calls the prune walk ---")
    code = "if ($k > 0.5) { @OUT = @A; } else { @OUT = gauss_blur(@A, 4.0); }"
    n, folded = _count_prune_calls(code, {"k": 1.0})
    if n < 1:
        r.fail("Q4 control (dead branch)",
               f"_prune_static_flow called {n} time(s), expected >= 1 for a verified dead branch")
        return
    stmts = folded.statements
    if len(stmts) != 1 or "gauss_blur" in repr(stmts):
        r.fail("Q4 control (dead branch)",
               f"expected the dead (else) arm spliced away, got {stmts!r}")
        return
    r.ok(f"{n} call(s), dead branch correctly spliced away")
