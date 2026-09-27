"""FIX-ROI O3 (v0.48.0 Phase C, B4#1/R2#7) -- the PERF-4 halo-identity oracle's ROI-48A/O2
exception must be tied to its CAUSE (pruning is the one and only thing that changed), not to
the mere structural fact "the folded tree contains a resolved `IfElse` somewhere".

THE BUG (confirmed by reading, `tests/test_perf4_front_end_rescans.py`'s pre-O3
`_has_resolved_ifelse`): the guard walked the whole folded tree and returned True the moment
ANY `IfElse` had a `NumberLiteral` condition -- it never checked that the SPECIFIC resolved
condition it found is the one whose pruning explains the observed divergence, nor even that
pruning explains the divergence AT ALL. `gated_block`
(`if ($k > 0.5) { @OUT = @A; } else { @OUT = @B * @C; }`) folds a resolved `IfElse` at every
`$k` in this file's own `_valuations_for` sweep, yet NEITHER arm contains a halo op -- pruning
it changes nothing about the halo answer. Under the old guard, an unrelated, genuine
regression in `_has_ungrounded_halo` that happened to land on `gated_block` (or ANY other row
sharing the "some `if` folds to a literal" tree shape) would have been silently swallowed as
an "expected ROI-48A exception" instead of failing the oracle.

THE FIX: `test_perf4_front_end_rescans._pruning_fully_explains_divergence(a, got, want)`
proves causation directly -- `tex_roi._has_ungrounded_halo` on the UNPRUNED tree `a` walks
every `IfElse`'s condition AND both bodies, exactly like the frozen base oracle always has;
if THAT already agrees with `want`, pruning (the only variable the new call changes) is what
accounts for the whole gap. If it does not, some other mechanism moved the answer and the
divergence must not be excused.

RED AT BASE (`5ae6288`, verified by hand against a snapshot of that commit before this fix
landed, and separately confirmed to `ImportError` on
`_pruning_fully_explains_divergence` not existing yet at base): `test_o3_gated_block_carries_a_resolved_ifelse_but_no_halo_gap`
establishes the premise (a resolved `IfElse` with no halo op in either arm — the exact shape
the old guard could not tell apart from the real ROI-48A case), and
`test_o3_synthetic_regression_on_gated_block_fails_the_new_check` proves the new,
causally-tied check correctly refuses to excuse a SYNTHETIC wrong answer on that row, while
the OLD tree-presence check (reproduced verbatim here for the comparison) would have."""
from helpers import *

import test_perf4_front_end_rescans as _p4
from TEX_Wrangle import tex_roi
from TEX_Wrangle.tex_compiler.ast_nodes import IfElse, NumberLiteral, iter_child_nodes

_GATED_BLOCK = _p4._LAZY_SENSITIVE["gated_block"]  # "if ($k > 0.5) {@OUT=@A;} else {@OUT=@B*@C;}"
_PARAMS = {"k": 0.0}  # $k > 0.5 folds False -> the `else` (@B * @C) arm is the one taken


def _old_tree_presence_guard(node) -> bool:
    """The pre-O3 `_has_resolved_ifelse`, reproduced verbatim for the comparison this test
    makes: True the moment ANY `IfElse` anywhere in the tree has a `NumberLiteral` condition,
    with no check that pruning IT is what caused anything."""
    stack = [node]
    while stack:
        n = stack.pop()
        if n.__class__ is IfElse and n.condition.__class__ is NumberLiteral:
            return True
        stack.extend(iter_child_nodes(n))
    return False


def test_o3_gated_block_carries_a_resolved_ifelse_but_no_halo_gap(r: SubTestResult):
    print("\n--- FIX-ROI O3: gated_block resolves an IfElse but pruning changes nothing ---")
    try:
        a = _p4._folded(_GATED_BLOCK, _PARAMS)
        b = _p4._folded(_GATED_BLOCK, _PARAMS)
        if a is None or b is None:
            r.fail("premise", "gated_block failed to fold")
            return
        if not _old_tree_presence_guard(a):
            r.fail("premise", "gated_block must fold a resolved IfElse for this test to be "
                   "the shape the old guard could not tell apart from a real ROI-48A case")
            return
        want = _p4._base_has_ungrounded_halo(b)
        unpruned_got = tex_roi._has_ungrounded_halo(a)
        if unpruned_got != want:
            r.fail("premise", f"unpruned answer ({unpruned_got}) already disagrees with the "
                   f"base ({want}) with no corruption involved — not a clean premise")
            return
        r.ok("gated_block resolves an IfElse (old guard would fire) but carries NO halo op "
             "in either arm — pruning it is causally irrelevant to the halo answer")
    except Exception as e:
        r.fail("O3 premise", f"{type(e).__name__}: {e}")


def test_o3_synthetic_regression_on_gated_block_fails_the_new_check(r: SubTestResult):
    print("\n--- FIX-ROI O3: a synthetic regression on gated_block must fail the oracle ---")
    try:
        a = _p4._folded(_GATED_BLOCK, _PARAMS)
        if a is None:
            r.fail("premise", "gated_block failed to fold")
            return
        # SYNTHETIC regression: a real bug in `_has_ungrounded_halo` (or `_accumulate`) that
        # has NOTHING to do with ROI-48A/O2 pruning, but happens to land on a row that also
        # folds a resolved IfElse — gated_block's REAL base/unpruned halo answer is False (no
        # halo op in either arm, pinned by the previous test), so `want=True` here is
        # deliberately synthetic: a wrong base answer this row could never actually produce,
        # standing in for "some other real bug flipped the true answer to True". `got=False`
        # is the one direction pruning is allowed to move an answer, so the test isolates
        # exactly what each guard decides, not what real code currently computes.
        want = True
        corrupted_got = False

        old_would_forgive = _old_tree_presence_guard(a)
        new_would_forgive = _p4._pruning_fully_explains_divergence(a, corrupted_got, want)

        if not old_would_forgive:
            r.fail("premise", "the old tree-presence guard must forgive this row (that's the "
                   "defect being demonstrated) — it did not, so the premise is broken")
            return
        if new_would_forgive:
            r.fail("O3 causal tie",
                   "_pruning_fully_explains_divergence still excuses a synthetic regression "
                   "unrelated to pruning on gated_block — the oracle would silently swallow "
                   "a real bug")
            return
        r.ok("the old tree-presence guard would have forgiven this synthetic regression; "
             "the new causally-tied check correctly refuses to")
    except Exception as e:
        r.fail("O3 synthetic regression", f"{type(e).__name__}: {e}")
