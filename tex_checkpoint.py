"""tex_checkpoint.py — CACHE-7: effort-based checkpoint placement.

Design note: `docs/effort-based-checkpoints.md`. The one-line version: CACHE-6 cuts a fused
chain ONCE, at a cut point the host guesses; CACHE-7 cuts it at every point where the
CUMULATIVE MEASURED cost since the last cut crosses a threshold, so a mid-graph edit recooks
at most (threshold + suffix) instead of the whole chain.

Three things live here and nothing else:

  * `plan_checkpoints`  — placement. Reads PROF-1 per-stage costs, returns cut indices.
  * `cook_checkpointed` — the serve path. Splices the suffix from the DEEPEST cached
                          boundary, or cooks whole when nothing is cached.
  * `materialize`       — phase 2. ONE re-cook with `tap: True` on the planned stages, which
                          harvests every boundary at once (`compile_fused` already exports
                          them as `@_tap_s{i}`), then `put`s them.

Everything is OFF the default path: placement needs PROF-1 (disarmed by default → no costs →
no checkpoints → today's cook), and the whole module needs a host-supplied `ResultCache`,
which the ComfyUI node never constructs. Invariant #7 is a consequence of the wiring, not a
promise.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# The report's default: "threshold a GOV-1 profile knob, default ~100 ms". GOV-1 OWNS it —
# `default_threshold_ms()` reads the active memory profile, because how often to checkpoint is
# a memory/latency trade and that is exactly what a profile expresses. This constant is the
# value the `balanced` preset carries and the answer when the governor is unreachable.
DEFAULT_THRESHOLD_MS = 100.0


def default_threshold_ms() -> float:
    """The placement threshold for the ACTIVE GOV-1 profile (performance checkpoints more
    eagerly, efficient less). Falls back to `DEFAULT_THRESHOLD_MS`."""
    try:
        from .tex_memory import checkpoint_threshold_ms
        return float(checkpoint_threshold_ms())
    except Exception:
        return DEFAULT_THRESHOLD_MS

# PROF-1 blends a running mean while a key is young and its first sample includes the COLD
# cook. Placing on that is worse than not placing: it spends materialization time and memory
# on the wrong boundaries and then reports success. How young is too young was MEASURED on a
# 3-stage chain whose stage 0 is a multiply and stage 1 a blur:
#
#     3 samples (warmup):  stage0=3.096  stage1=3.411  stage2=0.665
#    15 samples (settled): stage0=0.318  stage1=2.073  stage2=0.633
#    standalone truth:              0.301         1.928
#
# At 3 samples stage 0 is over-attributed ~10x and the RANKING IS INVERTED — a placement made
# there would checkpoint the cheap stage and skip the expensive one, which is the exact
# failure "effort-based" is named against. By 12 it is settled and the ranking is right.
#
# Handed to `profile.stage_snapshot()` rather than compared here: the number is a function of PROF-1's
# PRIVATE schedule (`_blend`'s max(_ALPHA, 1/n) rule, `_WARMUP_SAMPLES`, `_SAMPLE_EVERY`), so a
# consumer testing it against `samples()` goes stale the moment any of those change, with
# nothing turning red. The honest cost, stated once: at 3 warmup cooks plus 1-in-16 sampling,
# 12 samples is ~150 cooks, and until then a host gets NO checkpoints — today's cook, correct
# and not incremental. A host that knows its own chain skips the wait by passing `cuts=`.
MIN_SAMPLES = 12

# A checkpoint that costs more to materialize than it saves is not a checkpoint. `put` is a
# freeze + contiguous copy + key, and it is DEVICE-DEPENDENT by more than an order of
# magnitude (medians of 30, 4-channel fp32, torch 2.10 / RTX 2080 SUPER):
#
#              512²      1024²
#     cpu    1.117 ms   3.501 ms
#     cuda   0.062 ms   0.957 ms
#
# A single CPU-calibrated constant would over-state the CUDA floor ~18x at 512² and refuse
# checkpoints there that comfortably pay for themselves — on the device where interactive
# hosts actually live. Scaled from the 512² point by pixels, which is the right shape for
# what is essentially a memcpy.
_PUT_MS_AT_512_SQ = {"cpu": 1.117, "cuda": 0.062}
_PUT_REF_PX = 512 * 512

# ...and a checkpoint must CLEARLY pay, not merely break even. At 1024² a `put` is 4.5 ms, so
# a bare floor still admits a tap protecting 5 ms of recook — which buys ~0.5 ms per edit and
# spends a whole frame of RAM to do it. The saving is recovered on every later edit while the
# `put` is paid once on idle, so the factor does not need to be large; it needs to exclude the
# marginal case. 2x is "the protected work is worth at least twice what holding it costs".
_FLOOR_FACTOR = 2.0


def put_cost_ms(px: int | None, device="cpu") -> float:
    """Estimated cost of materializing one boundary at `px` pixels on `device`. Linear in
    pixels — a `put` is a copy, so unlike PROF-1's cook-time model there is no fixed-cost term
    worth modelling. Unknown devices key to the CPU (higher) constant, so an unrecognized
    accelerator refuses marginal checkpoints rather than placing ones it cannot pay for."""
    base = _PUT_MS_AT_512_SQ["cuda" if str(device).startswith("cuda") else "cpu"]
    if not px:
        return base
    return base * (float(px) / _PUT_REF_PX)


# ── placement ─────────────────────────────────────────────────────────────────

def _is_cut(stages: list[dict], k) -> bool:
    """A cut at `k` is the boundary AFTER stage `k-1`. `k == 0` is the source and
    `k == len(stages)` is the terminal: neither has both a prefix to cache and a suffix to
    splice. One spelling, because `stages[:k]` CLAMPS for `k >= len` — which is exactly how an
    out-of-range cut silently became the whole chain's fingerprint."""
    return isinstance(k, int) and not isinstance(k, bool) and 1 <= k < len(stages)


def _is_single_edge_cut(stages: list[dict], k) -> bool:
    """Is `k` a cut a v0.32 checkpoint may sit on? A cut crosses a SET of edges, and only a
    single-edge cut can be materialized as one tensor and rebound by `suffix_stage_list`.

    Enforced by the MECHANISM (`_resolve_cuts`), not only by the planner. Living in
    `plan_checkpoints` alone meant any caller supplying `cuts=` bypassed it — and that is not a
    hypothetical path: it is the one this module's docstring advertises, the one the benchmark
    drives, and the one most of the tests use."""
    return _is_cut(stages, k) and len(cut_set(stages, k)) == 1


def cut_set(stages: list[dict], k: int) -> list[tuple[int, str]]:
    """Every edge crossing a cut at `k`: `(producer_stage, output_name)` for each edge from a
    stage BELOW k to a stage at or above it.

    On a linear chain this is always exactly one edge — stage `k-1`'s `OUT`. On a DAG it is a
    SET, and a checkpoint there is a set of tensors, not one: the suffix rebind must inject one
    boundary per crossing edge and rewrite each consumer's positional `chain_inputs` entry.

    v0.32 ships this ANALYSIS and refuses to place a checkpoint on any cut whose set is not a
    single edge (see `plan_checkpoints`) — the DAG execution half is deferred with the gate
    recorded in the design note §11. Computing it here rather than asserting linearity is what
    lets the refusal be precise instead of a blanket `is_linear_stage_list`.
    """
    if not _is_cut(stages, k):
        return []
    from .tex_fusion import stage_edges
    return sorted({e for j in range(k, len(stages))
                   for e in stage_edges(stages, j) if e[0] < k})


def plan_checkpoints(stages: list[dict], *, costs: dict | None = None,
                     threshold_ms: float | None = None,
                     px: int | None = None, settled: bool = False,
                     device="cpu") -> list[int]:
    """The cut indices where a checkpoint pays. Ascending; may be empty.

    Walk from the source accumulating measured per-stage cost. When the sum SINCE THE LAST
    checkpoint crosses `threshold_ms`, cut at that boundary and reset the sum. `costs` is
    PROF-1's `{stage_index: ms}` (string or int keys — `profile.snapshot` stringifies).

    Returns `[]` — meaning "cook exactly as today" — whenever placement would be a guess
    rather than a measurement:

      * no costs at all (the profiler is disarmed, which is the default);
      * `settled=False` — PROF-1's estimate still carries the cold cook, and its stage RANKING
        is inverted while it does (see `MIN_SAMPLES`). The caller asks PROF-1 for this verdict
        rather than comparing a sample count, so it always describes the same bucket the costs
        came from;
      * a cut whose `cut_set` is not a single edge (a DAG region — analysis ships, execution
        does not);
      * a cut whose materialization would cost more than the work it protects.

    The last one is why the accumulated cost is compared against
    `max(threshold, put_cost * _FLOOR_FACTOR)` rather than the threshold alone: at 1024² a
    `put` is 4.5 ms, so a 0.1 ms threshold would otherwise place a tap at every stage and make
    the chain slower — while reporting success — and a bare (unfactored) floor would still
    admit taps that only break even.
    """
    n = len(stages)
    if n < 2 or not costs or not settled:
        return []
    # Normalize PROF-1's keys — snapshot() stringifies, the live table does not.
    by_index: dict[int, float] = {}
    for key, ms in costs.items():
        try:
            by_index[int(key)] = float(ms)
        except (TypeError, ValueError):
            continue          # the unfused `None` stage key — not a fused boundary
    if not by_index:
        return []

    if threshold_ms is None:
        threshold_ms = default_threshold_ms()      # the ACTIVE GOV-1 profile, not a def-time
        #                                            constant a later set_profile could not reach
    effective = max(float(threshold_ms), put_cost_ms(px, device) * _FLOOR_FACTOR)
    cuts: list[int] = []
    run = 0.0
    # A cut at k is a boundary AFTER stage k-1, so a cut at n is the terminal (no suffix to
    # splice) and a cut at 0 is the source: both are excluded by construction.
    for k in range(1, n):
        run += by_index.get(k - 1, 0.0)
        if run < effective:
            continue
        if not _is_single_edge_cut(stages, k):
            continue          # DAG cut set — analysis only in v0.32
        cuts.append(k)
        run = 0.0
    return cuts


def _resolve_cuts(stages, result_cache, cuts, *, latent_channel_count, upstream, precision,
                  threshold_ms, profile_key, spatial, device) -> list[int]:
    """The gate-then-plan prologue both entry points share. `[]` means "cook exactly as
    today", which is always correct.

    Supplied `cuts` go through the SAME single-edge and range rules placement applies, so the
    planned path and the host-supplied path cannot diverge on safety."""
    if not _gate_ok(stages, result_cache, latent_channel_count, upstream, precision):
        return []
    if cuts is None:
        cuts = _plan_from_profile(stages, threshold_ms=threshold_ms, profile_key=profile_key,
                                  spatial=spatial, device=device)
    return [k for k in sorted(set(cuts)) if _is_single_edge_cut(stages, k)]


# ── serve ─────────────────────────────────────────────────────────────────────

def cook_checkpointed(stages: list[dict], result_cache, *, device="cpu", precision="fp32",
                      upstream=(), cuts: list[int] | None = None,
                      threshold_ms: float | None = None,
                      profile_key: tuple | None = None, spatial=None,
                      latent_channel_count: int = 0, time_context=None,
                      cancel=None, on_progress=None) -> dict:
    """Cook a fused chain, splicing the suffix from the DEEPEST cached checkpoint.

    Deepest-first is the mechanism: it makes an edit's cost depend on the distance to the
    nearest checkpoint rather than on the length of the chain. A checkpoint above the edit is
    keyed by its own prefix's fingerprint AND param values, so a changed upstream param simply
    mints a different key and misses — there is no invalidation protocol, the key scheme is one.

    Nothing cached (the first cook of a chain, or a host that never ran phase 2) → the whole
    chain cooks exactly as today, plus a PROLOGUE. That prologue is not free, and this docstring
    used to assert that it was — measured at **1.09–1.33× warm and 2.46× on a true first cook**
    before P0-8. Where it went:

      * `ResultCache()` construction paid a ~20 ms `torch.cuda.is_available()` (the CUDA context
        initializes on first call). Now memoized in `_default_ram_budget` — it is a constant of
        the box, not a measurement worth repeating.
      * one lineage key minted and one disk-tier probe PER CUT, on a cache that has nothing.
        Now short-circuited by `_cache_is_provably_empty` — which requires a known-empty disk
        tier, so ENG-13's reattach case (frames on disk, `_spilled` unknown) still walks the
        loop and still finds them.

    What remains is placement (`_resolve_cuts`, one profile lookup) and, on a cache that is not
    provably empty, one key + one probe per cut. The honest claim: **arming CACHE-7 costs a
    bounded prologue on a cold cache, not zero** — see `docs/effort-based-checkpoints.md` §13.

    `cuts` may be supplied by a caller that already planned; otherwise placement runs here off
    PROF-1. Returns the interpreter's raw `{output: tensor}`, same as `cook_stage_list`.
    """
    from . import tex_engine

    def _full():
        return tex_engine.cook_stage_list(
            stages, device=device, precision=precision,
            latent_channel_count=latent_channel_count, time_context=time_context,
            cancel=cancel, on_progress=on_progress)

    cuts = _resolve_cuts(stages, result_cache, cuts,
                         latent_channel_count=latent_channel_count, upstream=upstream,
                         precision=precision, threshold_ms=threshold_ms,
                         profile_key=profile_key, spatial=spatial, device=device)
    if not cuts:
        return _full()

    # P0-8: an EMPTY cache cannot serve any cut, so minting a key per cut and probing the disk
    # tier for each is pure prologue on the exact cook that has the least to gain — the first
    # one. `stats()` is O(1) for this question. A cache with RAM entries or an unknown/populated
    # disk tier still walks the loop; only the provably-empty case short-circuits.
    if _cache_is_provably_empty(result_cache):
        return _full()

    from .tex_fusion import (FusionError, remap_suffix_taps, suffix_stage_list,
                             unservable_prefix_taps)
    for k in reversed(cuts):
        # P0-5: a tap on a stage strictly below `k-1` lives inside the served prefix and is
        # never cooked by the suffix. Serving the cut anyway DROPS a requested output — which
        # shifts the host's output slots and breaks this function's "same as `cook_stage_list`"
        # contract. Refuse the cut instead; a shallower one may still be servable, and the full
        # cook is always correct.
        if unservable_prefix_taps(stages, k):
            continue
        key = tex_engine.boundary_lineage_key(
            stages, k, device, precision, upstream=upstream, time_context=time_context,
            latent_channel_count=latent_channel_count)
        boundary = result_cache.get(key)
        if boundary is None:
            continue
        try:
            suffix = suffix_stage_list(stages, k, boundary)
        except FusionError:
            continue          # a malformed cut (a headless head stage) — try a shallower one
        out = remap_suffix_taps(tex_engine.cook_stage_list(
            suffix, device=device, precision=precision,
            latent_channel_count=latent_channel_count, time_context=time_context,
            cancel=cancel, on_progress=on_progress), k)
        # The boundary IS stage k-1's output, so a tap there is served for free rather than
        # costing a refusal.
        if k >= 1 and stages[k - 1].get("tap"):
            out.setdefault(f"_tap_s{k - 1}", boundary)
        return out
    return _full()


# ── phase 2: the idle harvest ────────────────────────────────────────────────

def materialize(stages: list[dict], result_cache, *, device="cpu", precision="fp32",
                upstream=(), cuts: list[int] | None = None,
                threshold_ms: float | None = None,
                profile_key: tuple | None = None, spatial=None,
                latent_channel_count: int = 0, time_context=None,
                cancel=None, on_progress=None) -> list[int]:
    """Phase 2. Re-cook the chain ONCE with the planned stages tapped, and cache every
    boundary that falls out. Returns the cuts actually materialized.

    ONE cook, not N segment cooks: `compile_fused` exports a tapped stage's handoff as
    `@_tap_s{i}` (tex_fusion.py:520,574), which is an assignment of a local the program already
    computed — so arming a tap costs no arithmetic and every checkpoint is harvested in a
    single pass. Verified bit-exact against the standalone prefix cook, CPU and CUDA.

    Preemption-safe by construction: the `put`s happen AFTER the cook returns, so a harvest
    that a SCHED-4 interactive arrival preempts publishes nothing and simply re-queues. A
    partially-harvested chain is never a partially-populated cache.
    """
    cuts = _resolve_cuts(stages, result_cache, cuts,
                         latent_channel_count=latent_channel_count, upstream=upstream,
                         precision=precision, threshold_ms=threshold_ms,
                         profile_key=profile_key, spatial=spatial, device=device)
    if not cuts:
        return []

    from . import tex_engine

    done: list[int] = []
    pending = sorted(cuts, reverse=True)          # DEEPEST first — see `_tap_budget`
    while pending:
        batch = pending[:_tap_budget()]
        tapped = [dict(st) for st in stages]
        for k in batch:
            tapped[k - 1] = {**tapped[k - 1], "tap": True}
        out = tex_engine.cook_stage_list(
            tapped, device=device, precision=precision,
            latent_channel_count=latent_channel_count, time_context=time_context,
            cancel=cancel, on_progress=on_progress)
        harvested = []
        for k in batch:
            b = out.get(f"_tap_s{k - 1}")
            if b is None:
                continue
            key = tex_engine.boundary_lineage_key(
                stages, k, device, precision, upstream=upstream, time_context=time_context,
                latent_channel_count=latent_channel_count)
            result_cache.put(key, b, canvas={"shape": list(b.shape)})
            harvested.append(k)
        done.extend(harvested)
        # Retry on GROUND TRUTH, not on a predicted budget. `_tap_budget()` assumes the chain
        # assigns one output; `compile_fused` computes the real budget as
        # `MAX_OUTPUTS - len(everything the stages assign)`, which only it can see — so a chain
        # carrying `exports` gets fewer taps than asked, and an earlier draft swallowed that
        # shortfall silently, losing exactly the checkpoints the batching existed to protect.
        # Dropping only what actually came back self-corrects for any budget.
        if not harvested:
            # A pass that harvests nothing can never make progress, so stop rather than spin.
            # Those cuts have no checkpoint, which is correct — just not incremental.
            logger.info("[TEX] CACHE-7: %d checkpoint(s) found no free output slot; "
                        "those cuts cook whole.", len(pending))
            break
        pending = [k for k in pending if k not in harvested]
    return sorted(done)


# ── internals ─────────────────────────────────────────────────────────────────

def _tap_budget() -> int:
    """How many taps one fused cook can carry beside its own `@OUT` — an ESTIMATE, and the
    reason `materialize` harvests DEEPEST FIRST.

    Two measured facts force the batching this sizes. (a) A fused program carries at most
    `MAX_OUTPUTS - 1` taps: asked for 8, the cook returns 7; asked for 12, still 7. (b)
    `compile_fused` drops the overflow in ASCENDING stage order, so it keeps the shallowest
    taps and discards the deepest — precisely inverting `cook_checkpointed`'s deepest-first
    serve, which is where the whole win comes from. Left alone, a chain long enough to need
    many checkpoints would silently keep only the ones nearest the source.

    An ESTIMATE because the real budget is `MAX_OUTPUTS - len(everything the stages assign)`
    and only `compile_fused` can see that. `materialize` therefore retries on what actually
    came back, so being wrong here costs an extra pass rather than a missing checkpoint."""
    from .tex_engine import MAX_OUTPUTS
    return max(1, int(MAX_OUTPUTS) - 1)


def _cache_is_provably_empty(result_cache) -> bool:
    """True only when the cache can be shown to hold nothing on EITHER tier (P0-8).

    "Provably" is doing real work here. A `ResultCache` whose spill tier is `unknown` (`_spilled
    is None` — a fresh cache over a directory a previous process populated) is NOT empty: that
    is precisely ENG-13's reattach case, where the frames exist and `_restore` finds them. So
    the fast path requires an empty RAM tier AND a KNOWN-empty disk tier, and anything it cannot
    establish falls through to the full serve loop.

    Best-effort against a duck-typed cache: a host may arm something that only implements the
    governor hooks, and an unknown object is treated as non-empty (walk the loop)."""
    try:
        if result_cache._ram:
            return False
        spilled = result_cache._spilled
        return spilled is not None and not spilled
    except Exception:
        return False


def _gate_ok(stages, result_cache, latent_channel_count: int, upstream,
             precision: str) -> bool:
    """The CACHE-6 gate: fp32 only, and a resolved-precision clause.

    FP16 IS REFUSED, and the reason is worth stating because this gate briefly did admit it.

    v0.32 lifted the fp32 clause on a 22-row measurement showing a prefix/suffix split at fp16
    bit-exact against the straight-through fp16 cook. That measurement was real and its
    conclusion was wrong, because the matrix only ever produced **fp16-representable
    boundaries**. M-3 keeps coordinate builtins fp32 (invariant #4), so a stage like
    `u * 1000.0 + 0.123` carries an interior local at fp32 precision through a straight-through
    fused cook — while the checkpointed path *materializes* that boundary and the suffix
    downcasts it at ingest. Measured on the counterexample: **maxdiff 6.58e-01 over
    49,152/65,536 elements, CPU and CUDA**, with the fp32 control bit-exact.

    So the precondition was never "fp16", it was "this boundary happens to hold values fp16 can
    represent" — a property of the VALUES, which a precision label cannot carry. Refusing
    restores parity with `cook_fused_cached`'s fp32-only clause (`tex_engine`), which never
    lifted it.

    The principled reopening — admitting a tap whose boundary tensor is *checked* to be
    fp16-representable, which is the same argument half-packing a checkpoint needs — belongs to
    PREC-1's storage-precision decision, not to this gate. It requires a representability check
    on the actual tensor, never a blanket precision label.

    `precision` must be RESOLVED — "fp32" or "fp16", never "auto". `prepare()` is where auto
    resolves, and this path never calls it: `cook_stage_list` passes the string straight to the
    interpreter, which cooks "auto" as fp32. That is not a wrong pixel, but it would key a
    boundary under the label "auto" while the tensor is fp32, so the same frame cached by a
    host that resolved first would never be found — and the label would mean fp32 on one box
    and, once auto resolution moves, something else on another. Refusing is one line; a
    mislabeled cache entry is not.

    Everything else stands: a LATENT narrows the wrong axis and forces fp32 (M-3); no cache
    means nowhere to put a boundary; and `upstream` must key EVERY tensor the prefix reads,
    because a program fingerprint is value-independent and params fold only NON-tensor
    bindings — without a content-sensitive source key a cached boundary can be served for a
    different image. The count (not just non-emptiness) is what makes a PARTIAL cover fail too.

    HONEST LIMIT, inherited and not fixed here: this is an ARITY check, not an identity check.
    TEX never inspects what an `upstream` string means, so a host that passes a stable but
    content-INSENSITIVE key (a node id, a file path it later overwrites) still gets a stale
    boundary — measured at maxdiff 0.91 on a swapped source. The contract is documented on
    `boundary_lineage_key` and is the host's to keep; an engine-side content check would mean
    hashing every source tensor on every cook, which is the cost caching exists to avoid.
    """
    if result_cache is None or latent_channel_count or len(stages) < 2:
        return False
    # fp32 ONLY. See the docstring: the fp16 lift was measured on a matrix that only produced
    # fp16-representable boundaries, and the counterexample is 6.58e-01 of wrong pixels.
    if precision != "fp32":
        return False
    # LINEAR ONLY — the same gate CACHE-6 has always had, restored after a per-cut
    # `cut_set(...) == 1` check was tried in its place and was WRONG.
    #
    # A single-edge cut is necessary but NOT sufficient. `suffix_stage_list` rebuilds the
    # suffix as `[rebased head] + stages[k+1:]`, which RENUMBERS every stage: a stage that
    # was index 7 becomes index 7-k. Its `chain_inputs` entries are ABSOLUTE stage indices
    # and are copied verbatim, so on any DAG they now point at the wrong producers. The cut
    # can be perfectly single-edge and the SUFFIX still be mis-wired.
    #
    # Measured through the weaker gate: 425 DAG lists admitted, 30 returning wrong pixels
    # (maxdiff 0.146, ndiff 3072) and 81 raising, against 0 and 0 through CACHE-6. Reachable
    # in production via `region_to_stages` on a FUS-1 fan-out region. `cut_set` stays as the
    # ANALYSIS the design note ships (§9); this is the execution gate it does not replace.
    from .tex_fusion import is_linear_stage_list
    if not is_linear_stage_list(stages):
        return False
    import torch
    # `isinstance(v, torch.Tensor)`, matching `cook_fused_cached`'s gate exactly. A duck-typed
    # `hasattr(v, "shape")` would also count a host object that merely exposes a shape, which
    # makes the two gates disagree about how many source keys a chain needs — the one number
    # that decides whether a boundary may be cached at all. Scope DOES differ, deliberately:
    # the single-tap gate counts `stages[:k]` because it has one cut, while multi-tap has cuts
    # at many k and every stage's tensors can feed some prefix.
    tensors = sum(1 for st in stages
                  for v in (st.get("bindings") or {}).values()
                  if isinstance(v, torch.Tensor))
    return len(upstream) >= tensors


def _plan_from_profile(stages, *, threshold_ms: float, profile_key, spatial,
                       device="cpu") -> list[int]:
    """Placement off the live PROF-1 table. No key or no measurements → no checkpoints, which
    is today's cook — the profiler is disarmed by default, so this is the ordinary answer."""
    if profile_key is None:
        return []
    from .tex_runtime import profile as _profile
    # ONE call, so the costs and the confidence in them describe the SAME bucket. Two calls
    # read the table twice and can resolve differently — only the costs query filters to
    # buckets carrying a per-stage breakdown — which is how a planner ends up placing on
    # numbers whose trustworthiness was measured somewhere else.
    costs, is_settled = _profile.stage_snapshot(profile_key, spatial, need=MIN_SAMPLES)
    if not costs:
        return []
    px = _profile.bucket_of(spatial)[1] if spatial is not None else None
    return plan_checkpoints(stages, costs=costs, threshold_ms=threshold_ms, px=px,
                            device=device, settled=is_settled)
