"""
tex_roi_dag — the DAG generalisation of `tex_roi.chain_windows` (JOIN-49), plus the
per-stage reach resolver a real caller needs (JOINWIRE-50).

Split out of `tex_roi.py` by SPLIT-47's own pattern (a pure move, re-exported at
`tex_roi`'s own top level so `from .tex_roi import StageSpec`/`chain_windows_dag` and
every existing test keep resolving unchanged) — `tex_roi.py` was AT its 2000-line hard
budget (REG-2) before this lane's JOINWIRE-50 addition (`stage_dag_arg_halos`) would
have pushed it over. `chain_windows`'s own body (in `tex_roi.py`) still calls
`_dag_grow` — the ONE shared grow-and-clamp implementation Q3 gave it — via a
function-local import (the AGENTS.md "Trades to REFUSE" idiom for a load-bearing
cross-module cycle: this module imports FROM `tex_roi` at its own top level, so
`tex_roi` must never import THIS module at ITS top level, only lazily, inside the one
function that needs it).

Nothing here changes behaviour: `_dag_grow`/`_dag_union`/`StageSpec`/`chain_windows_dag`
are byte-for-byte what JOIN-49 wrote (moved, not edited — `tests/test_join49_dag_windows.py`
proves it by continuing to pass against the re-exported names). `stage_dag_arg_halos` is
new (JOINWIRE-50): the per-stage `(halo, arg_halo)` resolver `StageSpec.arg_halo`'s own
docstring named as owed to "whoever wires REACH-48's per-argument registry through this
dataclass".
"""
from __future__ import annotations

from dataclasses import dataclass

from .tex_roi import (
    roi_plan, binding_footprints, POINT, WHOLE_FRAME, canonical_roi, covers, _scale_halo,
)


def _dag_grow(window, pad: float):
    """`window ⊕ pad`, clamped to the frame — grow-and-clamp, the one piece of correctness-
    sensitive arithmetic this module needs. As of Q3, the ONE shared implementation for FOUR
    call sites: `chain_windows`'s own backward step and its P0-4(b) check, and
    `chain_windows_dag`'s backward step and its own divergent-validity check — not two
    independent copies plus this as a third (the class of bug FIX-ROI's O2 finding was
    about: two per-node copies of one rule drifting apart)."""
    x0, y0, w, h, W, H = window
    pad = int(pad)
    nx0, ny0 = max(0, x0 - pad), max(0, y0 - pad)
    nx1, ny1 = min(W, x0 + w + pad), min(H, y0 + h + pad)
    return (nx0, ny0, nx1 - nx0, ny1 - ny0, W, H)


def _dag_union(a, b):
    """Bounding-box union of two windows over the SAME frame — composition rule 1
    (TIERS-48-design.md SS B.2): a canvas read by two consumers (a diamond, not a join —
    one input, two consumers) needs the union of what each separately demands. Same
    min/max-per-axis arithmetic `canonical_roi`'s own frame clamp already performs, lifted
    from "a window against the frame boundary" to "two windows against each other". `None`
    is the identity (no demand yet)."""
    if a is None:
        return b
    if b is None:
        return a
    ax0, ay0, aw, ah, W, H = a
    bx0, by0, bw, bh, _, _ = b
    x0 = min(ax0, bx0)
    y0 = min(ay0, by0)
    x1 = max(ax0 + aw, bx0 + bw)
    y1 = max(ay0 + ah, by0 + bh)
    return (x0, y0, x1 - x0, y1 - y0, W, H)


@dataclass(frozen=True)
class StageSpec:
    """One stage of a `chain_windows_dag` walk. `halo` is this stage's OWN reach into its
    input(s) — the same per-stage number `chain_windows`'s flat `halos` list already
    carries, applied UNIFORMLY to every input this stage reads UNLESS `arg_halo` overrides
    a SPECIFIC upstream index with its own reach. `inputs` names the upstream stage indices
    this stage reads (`()` for a source stage). The linear chain is the degenerate case
    `inputs=(i - 1,)` for every `i > 0`, `()` for `i == 0` — exactly `chain_windows`'s own
    model, so a stage with one input and no `arg_halo` behaves identically to a `halos[i]`
    entry.

    `arg_halo`, when given, is composition rule 2 (TIERS-48-design.md SS B.2 point 2): a
    join stage's demand on EACH of its inputs is independent — a composite's background
    argument may need no margin at all while its mask argument needs a blur-sized halo for
    feathering. The intended source of these per-argument numbers is REACH-48's registry
    (`tex_runtime.stdlib_registry.arg_footprint_by_name` / `tex_roi._call_arg_reach`) for
    whichever builtin the join stage's own code runs — resolving that from source is the
    CALLER's job (a `roi_plan`/`binding_footprints`-shaped one), exactly as `stage_halo`
    already resolves the single-input case; this dataclass only carries the resolved
    number, the same division of labour `halos[i]` already had. `stage_dag_arg_halos`
    (below) is JOINWIRE-50's answer to that CALLER's job.

    KNOWN LIMIT (Q5): `arg_halo` is keyed by UPSTREAM STAGE INDEX only, one number
    per index. A stage reading the SAME upstream index through TWO argument roles needing
    DIFFERENT margins (e.g. the same plate as both a zero-halo `bg` and a blurred `fg` needing
    one, with no intervening stage) would silently get the SMALLER one, under-serving whichever
    role needed more, unless the caller pre-maxes the roles first. `stage_dag_arg_halos` DOES
    pre-max (see its own docstring) whenever it builds this map itself, so a caller going
    through it is unaffected; a caller building `arg_halo` by hand must still pre-max."""
    halo: float
    inputs: tuple = ()
    arg_halo: "dict | None" = None

    def halo_for(self, upstream: int) -> float:
        """This stage's own reach into ONE SPECIFIC upstream input: `arg_halo[upstream]`
        if declared, else the uniform `halo` — matches the linear model exactly whenever no
        per-argument override exists (`arg_halo=None`, or `upstream` absent from it)."""
        if self.arg_halo is not None and upstream in self.arg_halo:
            return self.arg_halo[upstream]
        return self.halo


def chain_windows_dag(stages, roi, dirty_from: int = 0, valid=None, declined=(),
                      scale: float = 1.0) -> "list | None":
    """`chain_windows`, generalised from a linear chain to a DAG (JOIN-49). `stages[i]` is
    a `StageSpec` (a plain `(halo, inputs)` pair is coerced). Constraints inherited from
    the model this generalises, not new: stage indices are already topologically ordered
    — `stages[j].inputs` names only indices `< j` (raises `ValueError` otherwise, a defect
    in the CALLER's graph construction, not a case this function can serve any answer for)
    — and the SINK is always the last stage, `n - 1`; `roi` is the window wanted out of it.
    A DAG with more than one true sink has no representation here, same as `chain_windows`
    never had one for a linear chain with a branch off the end.

    **Same-input, multiple-consumer union** (rule 1): when two consumers `j1`, `j2` both
    read stage `i`, `i`'s required window is `_dag_union` of what each separately demands.

    **Multi-input join, single consumer** (rule 2): a join stage `k` reading BOTH `i` and
    `i'` projects its OWN outgoing window backward through EACH input independently, using
    THAT input's own `halo_for` — see `StageSpec.arg_halo`.

    **Refusal — divergent-validity join** (SS B.3's new case beyond the three the linear
    walk already has): for every edge from a stage `i` that is NOT being recomputed
    (`i < dirty_from`, so `valid[i]` is its recorded truth) into a dirty consumer, the
    consumer's demand on `i` must be covered by `valid[i]` — checked for EVERY such edge,
    not only the single boundary edge the linear walk has, because a join can cross the
    dirty/clean boundary on more than one input at once, and each is an independent
    correctness question ("do I still have what I need from EACH clean input"). Failing
    ANY one edge refuses the WHOLE plan (`None`) — the same fail-closed posture the linear
    walk's own boundary check already takes, generalised from one edge to all of them.

    **Declined-stage poisoning** (P0-4a, generalised): a stage that declined its window
    cooked whole-frame from ITS OWN inputs, which is only "valid everywhere" if every ONE
    of those inputs was ALSO whole-frame valid — checked over `stages[i].inputs`, not only
    `i - 1`.

    Returns one window per stage (`None` for a stage neither dirty nor demanded by anyone
    dirty — mirrors the linear walk's "clean prefix" `None`s), or `None` when the plan
    cannot be served incrementally at all (cook the whole graph from the source; widening
    the returned windows never repairs a stale upstream, per `chain_windows`'s own central
    argument, unchanged here).

    Pure arithmetic — this module stays torch-free, same as `chain_windows`."""
    if scale <= 0:
        raise ValueError(f"chain_windows_dag: scale must be > 0, got {scale!r}")
    specs = [s if isinstance(s, StageSpec) else StageSpec(s[0], tuple(s[1]))
             for s in stages]
    n = len(specs)
    for j, s in enumerate(specs):
        for i in s.inputs:
            if not (0 <= i < j):
                raise ValueError(
                    f"chain_windows_dag: stage {j} names input {i}, which is not an "
                    f"earlier stage index (< {j}) — inputs must be topologically ordered")

    # P0-4a, generalised: a decliner's whole-frame output is only as good as ALL of its
    # inputs being whole-frame valid, not just one. No bounds guard on `valid[m]` — matches
    # `chain_windows`'s own unguarded `valid[i - 1]` in this same check exactly; both assume
    # a `valid` list sized to match `stages`/`halos`, per the shared contract.
    if declined and valid is not None:
        for i in sorted(set(declined)):
            if 0 <= i < n:
                for m in specs[i].inputs:
                    if valid[m] is not None:
                        return None

    out = [None] * n
    if n == 0:
        return out
    out[n - 1] = canonical_roi(roi)
    start = max(0, dirty_from)

    if valid is not None and start >= n:
        # P0-4b's own past-the-end guard, lifted unchanged: `dirty_from` past the end of the
        # stage list is unconditionally not serviceable whenever validity is being tracked
        # at all — mirrors `chain_windows`'s exact `if start >= n: return None` inside its
        # own `valid is not None` branch.
        return None

    # Reverse adjacency: consumers[i] = [j, ...] such that i in specs[j].inputs.
    consumers: dict = {}
    for j, s in enumerate(specs):
        for i in s.inputs:
            consumers.setdefault(i, []).append(j)

    for i in range(n - 2, start - 1, -1):
        demand = None
        for j in consumers.get(i, ()):
            if j < start or out[j] is None:
                continue          # a clean (non-recomputed) consumer demands nothing here
            demand = _dag_union(demand, _dag_grow(out[j], specs[j].halo_for(i)))
        out[i] = demand

    if valid is not None:
        for i in range(0, start):
            for j in consumers.get(i, ()):
                if j < start or out[j] is None:
                    continue
                demand = _dag_grow(out[j], specs[j].halo_for(i))
                upstream_valid = valid[i] if i < len(valid) else None
                if not covers(upstream_valid, demand):
                    return None   # divergent-validity join — not serviceable
    return out


# ── JOINWIRE-50: per-stage DAG reach, resolved from source ────────────────────
#
# `StageSpec.arg_halo`'s own docstring names its resolution as the CALLER's job — "a
# roi_plan/binding_footprints-shaped one" — and says no such resolver exists yet (JOIN-49:
# "no caller does this today"). This is that resolver: the same two functions
# `tex_roi.stage_halo` already calls (`roi_plan` for the executable gate, plus
# `binding_footprints` for the PER-NAME numbers `stage_halo` never needed because a linear
# stage has only one upstream) answer a join stage's per-input reach without any new AST
# walk — `_accumulate` (the walker behind both) already tallies each wire binding's reach
# separately; a linear caller just never asked for more than the one number
# `roi_plan.halo` unions across all of them.

def stage_dag_arg_halos(code: str, name_to_upstream: dict, param_values: dict | None = None,
                        binding_types: dict | None = None, scale: float = 1.0):
    """Resolve one DAG stage's own uniform `halo` AND its per-upstream `arg_halo` map from
    its source — everything `StageSpec(halo, inputs, arg_halo)` needs for one stage of a
    `chain_windows_dag` walk. `name_to_upstream` is `{binding_name: upstream_stage_index}`
    for every one of this stage's chain-fed bindings (a plain literal/param binding is not
    in this map and needs no reach). Returns `(halo, arg_halo)` where `arg_halo` is
    `{upstream_index: reach}`, ready for
    `StageSpec(halo, inputs=tuple(sorted(set(name_to_upstream.values()))), arg_halo=arg_halo)`.

    **Whitelist posture, inherited from `stage_halo`/`roi_plan` (never widened here):** when
    the WHOLE program is not ROI-3-executable (a scatter, a symbolic halo radius outside a
    gather, a whole-image gather ANYWHERE — `roi_plan(...).executable` is False), every named
    upstream gets `WHOLE_FRAME` — the same inversion `stage_halo`'s own docstring explains:
    a program that cannot be narrowed AT ALL must not report any binding's LOCAL read as
    bounded, because a stage that is going to cook whole-frame anyway is going to READ every
    binding whole-frame too, whatever `binding_footprints` says about one name in isolation
    (it answers "where is this name read", not "will the engine actually narrow this cook").
    Only when the program IS executable does this trust `binding_footprints`'s per-name
    answer for the join-specific numbers `roi_plan.halo` alone can't give (it unions every
    binding's reach into ONE margin; a join needs them apart).

    A name mapped to a call argument carrying its OWN `arg_footprint` declaration (REACH-48;
    `convolve`'s `kernel`) reports `image` there exactly as `binding_footprints` already
    does — resolved to `WHOLE_FRAME` here, same as any other whole-image footprint. `Footprint`
    over-approximation (ROI-5's local-alias caveat) carries through unchanged: this adds no
    new analysis, it is the SAME walk, read a second way. Note that `convolve` ALSO declares
    its own arg-0 `footprint='image'` (its main image argument is unbounded too), so a stage
    that calls it is not ROI-3-executable AT ALL and this function returns the WHOLE-program
    `WHOLE_FRAME` answer for every one of its named inputs via the gate above, never a
    per-argument narrow one — there is no live case where `convolve`'s OWN per-argument
    declaration is what supplies a bounded answer through this function; it exists so
    `binding_footprints` reports `@kernel` correctly (image, not a false 'point') for ANY
    future per-binding consumer, this one included.

    A name appearing more than once at DIFFERENT reaches (the same upstream fed through two
    argument roles in one stage) takes the max, mirroring `binding_footprints`' own union of
    every read site of one name (`_lub`) — `StageSpec.arg_halo`'s Q5 known limit (one number
    per upstream STAGE index) is unaffected: if two DIFFERENT NAMES both map to the SAME
    upstream index, this also takes the max across them (pre-maxing the roles, per that
    docstring's own advice to whoever wires a caller).

    Never raises: mirrors `stage_halo`'s "a reach question must always have a conservative
    answer" contract."""
    upstreams = set(name_to_upstream.values())
    plan = roi_plan(code, param_values or {}, binding_types, scale=scale)
    if not plan.executable:
        return WHOLE_FRAME, {u: WHOLE_FRAME for u in upstreams}
    fps = binding_footprints(code, param_values or {}) or {}
    arg_halo: dict = {}
    for name, upstream in name_to_upstream.items():
        fp = fps.get(name, POINT)
        reach = WHOLE_FRAME if fp.kind == "image" else _scale_halo(fp.reach, scale)
        if upstream in arg_halo:
            arg_halo[upstream] = max(arg_halo[upstream], reach)
        else:
            arg_halo[upstream] = reach
    return int(plan.halo), arg_halo
