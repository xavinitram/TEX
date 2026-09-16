"""HOOK-3 — a linear collapse, and an audible refusal, on the checkpoint gate.

Two halves of one complaint: a host that wires a fused region the way the engine's own
`region_to_stages` wires it gets `cook_checkpointed` refusing every region, silently.

  * THE COLLAPSE (`tex_fusion.collapse_linear`) — `region_to_stages` emits `chain_inputs` on
    every chained stage, so `is_linear_stage_list` is False for a region that is a plain path
    and the gate refuses it. The collapse rewrites that list to the legacy `chain_input` shape
    the gate admits, and returns None when the region is NOT a path. Pinned in BOTH directions,
    because the cheap version of this function is the bug the gate already survived once: a
    weaker linearity rule admitted 425 DAG lists, 30 of them returning wrong pixels
    (`tex_checkpoint._gate_ok`). The mutation row below rebuilds that weaker rule and MEASURES
    the divergence it admits, so "the DAG half still refuses" is evidence, not an assertion.

  * THE REFUSAL (`tex_checkpoint.gate_refusal`) — the gate's own verdict as data: a stable
    reason code, the offending stage index, a human message. `_gate_ok` is derived from it, so
    every row here also asserts the two agree — the decision is ADDITIONAL data, never a
    changed decision.

Shapes: DIFFERENTIAL (a collapsed chain cooks bit-exactly what the DAG-spelled one cooks),
NEVER-SEVER (the non-linear lists that must keep being refused), MUTATION (the weaker rule,
and what it costs), and a CANARY that the boolean gate is unchanged.
"""
from helpers import *

import torch

from TEX_Wrangle import tex_checkpoint as CK
from TEX_Wrangle import tex_engine, tex_fusion as F, tex_results
from TEX_Wrangle.tex_marshalling import Promise

_CODE = {
    "A": "@OUT = vec4(@IN.rgb * 1.05, 1.0);",
    "B": "@OUT = vec4(max(@IN.rgb - vec3(0.02), vec3(0.0)), 1.0);",
    "C": "@OUT = vec4(@IN.rgb * vec3(0.9, 1.0, 1.1), 1.0);",
    "D": "@OUT = vec4((@P.rgb + @Q.rgb) * 0.5, 1.0);",
}


def _edge(src, dst, binding="IN"):
    return {"from": src, "from_slot": 0, "to": dst, "to_binding": binding}


def _region_stages(nodes, edges, source):
    """A REAL region, assembled the way a host's executor assembles one: detector ->
    region_to_stages -> the single external source injected at every injection point."""
    plan = {k: {"code_wired": False} for k in nodes}
    regions = F.detect_fusable_regions(plan, edges)
    assert len(regions) == 1, f"expected one region, got {len(regions)}"
    region = regions[0]
    stages = F.region_to_stages(region, _CODE, {k: {} for k in nodes})
    for inj in region["source"]["injections"]:
        stages[inj["stage"]]["bindings"][inj["binding"]] = source
    return stages


def _linear_stages(source):
    """src -> A -> B -> C. A path, and the shape the audit's evidence names."""
    return _region_stages(["A", "B", "C"],
                          [_edge("src", "A"), _edge("A", "B"), _edge("B", "C")], source)


def _legacy(source, n=3):
    """The same chain already in the legacy spelling, as `compile_fused` builds it."""
    keys = ["A", "B", "C"][:n]
    out = [{"code": _CODE[keys[0]], "chain_input": None, "bindings": {"IN": source}}]
    out += [{"code": _CODE[k], "chain_input": "IN", "bindings": {}} for k in keys[1:]]
    return out


def _skip_edge(source):
    """A DAG whose forced collapse still COMPILES — stage 2 reads stage 0, not stage 1. This
    is the shape that produces wrong pixels rather than an exception, which is why the
    mutation row uses it."""
    S = _legacy(source)
    S[1] = {"code": _CODE["B"], "bindings": {}, "chain_inputs": {"IN": [0, "OUT"]}}
    S[2] = {"code": _CODE["C"], "bindings": {}, "chain_inputs": {"IN": [0, "OUT"]}}
    return S


def _diamond(source):
    """stage 3 reads stage 0 AND stage 2 — the fan-in `_gate_ok`'s note was measured on."""
    S = _legacy(source) + [{"code": _CODE["D"], "bindings": {},
                            "chain_inputs": {"P": [0, "OUT"], "Q": [2, "OUT"]}}]
    S[1] = {"code": _CODE["B"], "bindings": {}, "chain_inputs": {"IN": [0, "OUT"]}}
    S[2] = {"code": _CODE["C"], "bindings": {}, "chain_inputs": {"IN": [1, "OUT"]}}
    return S


def _force_collapse(stages):
    """THE MUTATION: the collapse without the producer-index rule — "one in-edge is enough",
    which is the same weakening that admitted 425 DAG lists. Kept in the test rather than
    reachable from the tree."""
    out = []
    for j, st in enumerate(stages):
        rest = {k: v for k, v in st.items() if k != "chain_inputs"}
        ci = st.get("chain_inputs") or {}
        if len(ci) == 1:
            rest["chain_input"] = next(iter(ci))
        out.append(rest)
    return out


# ── the collapse, on a real region_to_stages output ──────────────────────────

def test_hook3_collapse_linear_on_a_real_region(r: SubTestResult):
    print("\n--- HOOK-3: collapse_linear on a real region_to_stages output ---")
    src = torch.rand(1, 32, 32, 3)
    stages = _linear_stages(src)

    # The premise the audit records: the engine's own region assembler produces a list its own
    # gate refuses. If this row ever goes green, HOOK-3's evidence has moved.
    if not F.is_linear_stage_list(stages):
        r.ok("premise: region_to_stages output is NOT admitted by is_linear_stage_list")
    else:
        r.fail("HOOK-3 premise", "region_to_stages no longer emits chain_inputs on a path")

    collapsed = F.collapse_linear(stages)
    if collapsed is None:
        r.fail("HOOK-3 collapse", "refused a linear region")
        return
    r.ok("collapse_linear returns a list for a linear region")

    if F.is_linear_stage_list(collapsed):
        r.ok("the collapsed list IS admitted by is_linear_stage_list (the post-condition)")
    else:
        r.fail("HOOK-3 post-condition", "collapsed list still fails is_linear_stage_list")

    want = [None, "IN", "IN"]
    got = [st.get("chain_input") for st in collapsed]
    if got == want and not any("chain_inputs" in st for st in collapsed):
        r.ok(f"each stage reads the one before it by name: {got}")
    else:
        r.fail("HOOK-3 legacy shape", f"chain_input {got}, wanted {want}")

    # Every other key survives, and the caller's list is not mutated.
    if collapsed[0]["bindings"].get("IN") is src and stages[1].get("chain_inputs"):
        r.ok("bindings carried through; the input list is left untouched")
    else:
        r.fail("HOOK-3 purity", "the source binding or the caller's stage list was disturbed")

    # DIFFERENTIAL: the collapse is a rewrite, not a re-plan — same pixels, bit for bit.
    a = tex_engine.cook_stage_list(stages, device="cpu", precision="fp32")["OUT"]
    b = tex_engine.cook_stage_list(collapsed, device="cpu", precision="fp32")["OUT"]
    if torch.equal(a, b):
        r.ok("collapsed chain cooks the DAG-spelled chain's pixels, bit-exact")
    else:
        d = (a.float() - b.float()).abs()
        r.fail("HOOK-3 differential",
               f"maxdiff {d.max().item():.3e} over {int((d > 0).sum())} elements")

    # A host may call it unconditionally: an already-legacy list passes through unchanged.
    again = F.collapse_linear(collapsed)
    legacy = F.collapse_linear(_legacy(src))
    if again == collapsed and legacy is not None and F.is_linear_stage_list(legacy):
        r.ok("idempotent, and an already-legacy linear list collapses to itself")
    else:
        r.fail("HOOK-3 idempotence", f"second pass -> {again!r}, legacy -> {legacy!r}")


# ── the never-sever half: a non-linear list is still refused ─────────────────

def test_hook3_collapse_refuses_non_linear(r: SubTestResult):
    """The direction that must NOT regress. `collapse_linear` is only useful because it is
    exact: a DAG collapsed by force is a mis-wired suffix, which is the bug the gate carries a
    measurement of."""
    print("\n--- HOOK-3: a non-linear stage list is still refused ---")
    src = torch.rand(1, 32, 32, 3)

    cases = {
        "skip edge (stage 2 reads stage 0)": _skip_edge(src),
        "diamond (stage 3 reads 0 and 2)": _diamond(src),
        "fan-out (two stages read stage 0)": _region_stages(
            ["A", "B", "C", "D"],
            [_edge("src", "A"), _edge("A", "B"), _edge("A", "C"),
             _edge("B", "D", "P"), _edge("C", "D", "Q")], src),
        "multi-injection (one producer fanning in)": _region_stages(
            ["A", "B", "C"],
            [_edge("src", "A"), _edge("src", "B"), _edge("A", "C", "P"),
             _edge("B", "C", "Q")], src),
        "stage 0 reads a chain": [{"code": _CODE["A"], "bindings": {"IN": src},
                                   "chain_inputs": {"IN": [0, "OUT"]}},
                                  {"code": _CODE["B"], "bindings": {},
                                   "chain_inputs": {"IN": [0, "OUT"]}}],
        "two bindings on one edge": [{"code": _CODE["A"], "bindings": {"IN": src}},
                                     {"code": _CODE["D"], "bindings": {},
                                      "chain_inputs": {"P": [0, "OUT"], "Q": [0, "OUT"]}}],
        "empty list": [],
    }
    for label, S in cases.items():
        if F.collapse_linear(S) is None:
            r.ok(f"refused: {label}")
        else:
            r.fail(f"HOOK-3 refusal ({label})", "collapsed a list that is not a path")

    # ...and the other direction, so the refusal is not just a function that always says None.
    for n in (2, 3):
        S = _linear_stages(src)[:n] if n == 3 else _legacy(src, n)
        if F.collapse_linear(S) is not None:
            r.ok(f"still collapses a genuine {n}-stage path")
        else:
            r.fail("HOOK-3 over-refusal", f"refused a linear {n}-stage list")

    # THE MUTATION, measured. "One in-edge is enough" is the weaker rule; it admits the
    # skip-edge list, and the list it admits cooks DIFFERENT PIXELS.
    S = _skip_edge(src)
    forced = _force_collapse(S)
    if F.collapse_linear(S) is None and F.is_linear_stage_list(forced):
        truth = tex_engine.cook_stage_list(S, device="cpu", precision="fp32")["OUT"]
        try:
            wrong = tex_engine.cook_stage_list(forced, device="cpu", precision="fp32")["OUT"]
            d = (truth.float() - wrong.float()).abs()
            n_diff = int((d > 0).sum())
        except Exception as exc:                       # the 81-raising half of the measurement
            d, n_diff, wrong = None, -1, exc
        if n_diff != 0:
            detail = (f"maxdiff {d.max().item():.3e} over {n_diff} elements"
                      if d is not None else f"{type(wrong).__name__}")
            r.ok(f"the weaker rule admits this list and it is wrong: {detail}")
        else:
            r.fail("HOOK-3 mutation", "the forced collapse cooked identical pixels — the row "
                                      "proves nothing; find a sharper counterexample")
    else:
        r.fail("HOOK-3 mutation setup",
               "the weaker rule no longer admits what the real one refuses")


# ── the refusal, structured ──────────────────────────────────────────────────

def test_hook3_gate_refusal_is_structured(r: SubTestResult):
    print("\n--- HOOK-3: the gate refuses audibly ---")
    src = torch.rand(1, 32, 32, 3)
    stages = _linear_stages(src)
    cache = tex_results.ResultCache()
    up = ("hook3-src",)

    def _ref(S, **kw):
        kw.setdefault("upstream", up)
        return CK.gate_refusal(S, kw.pop("cache", cache), **kw)

    # (1) the region a host actually hands over: linear, but not collapsed.
    ref = _ref(stages)
    if ref is not None and ref.code == CK.REFUSE_NOT_COLLAPSED and ref.stage == 1:
        r.ok(f"region_to_stages output -> {ref.code!r} at stage {ref.stage}")
    else:
        r.fail("HOOK-3 not-collapsed", f"got {ref!r}")
    if ref is not None and "collapse_linear" in ref.message:
        r.ok("the message names the way out")
    else:
        r.fail("HOOK-3 message", f"got {ref.message!r}" if ref else "no refusal")

    # (2) a real DAG: a different, equally stable code, naming the stage that breaks the path.
    for label, S, want_stage in (("skip edge", _skip_edge(src), 2),
                                 ("diamond", _diamond(src), 3)):
        ref = _ref(S)
        if ref is not None and ref.code == CK.REFUSE_NOT_LINEAR and ref.stage == want_stage:
            r.ok(f"{label} -> {ref.code!r} at stage {ref.stage}")
        else:
            r.fail(f"HOOK-3 not-linear ({label})", f"got {ref!r}, wanted stage {want_stage}")

    # (3) the collapsed list is ADMITTED — the refusal was the only thing standing between a
    # host region and the feature.
    collapsed = F.collapse_linear(stages)
    if _ref(collapsed) is None:
        r.ok("the collapsed region is admitted (no refusal)")
    else:
        r.fail("HOOK-3 admit", f"refused a collapsed region: {_ref(collapsed)!r}")

    # (4) every other clause keeps its own code, and `stage` is None when no stage is to blame.
    rows = [
        (CK.REFUSE_NO_CACHE, "no cache", dict(cache=None)),
        (CK.REFUSE_LATENT, "a LATENT", dict(latent_channel_count=4)),
        (CK.REFUSE_PRECISION, "fp16", dict(precision="fp16")),
        (CK.REFUSE_UPSTREAM_KEYS, "no upstream keys", dict(upstream=())),
    ]
    for code, label, kw in rows:
        ref = _ref(collapsed, **kw)
        if ref is not None and ref.code == code and ref.stage is None:
            r.ok(f"{label} -> {ref.code!r}")
        else:
            r.fail(f"HOOK-3 code ({label})", f"got {ref!r}, wanted {code!r}")
    ref = _ref(collapsed[:1])
    if ref is not None and ref.code == CK.REFUSE_TOO_FEW_STAGES:
        r.ok(f"one stage -> {ref.code!r}")
    else:
        r.fail("HOOK-3 code (one stage)", f"got {ref!r}")

    # An unlanded, shapeless Promise is the one per-STAGE refusal that is not about wiring.
    shapeless = [dict(collapsed[0]), dict(collapsed[1]), dict(collapsed[2])]
    shapeless[1] = {**shapeless[1], "bindings": {"P": Promise("late")}}
    ref = _ref(shapeless, upstream=("a", "b"))
    if ref is not None and ref.code == CK.REFUSE_BINDING_SHAPE and ref.stage == 1:
        r.ok(f"an unkeyable binding -> {ref.code!r} at stage {ref.stage}")
    else:
        r.fail("HOOK-3 code (shapeless binding)", f"got {ref!r}")


# ── the canary: the boolean gate decided nothing differently ─────────────────

def test_hook3_gate_decision_is_unchanged(r: SubTestResult):
    """INVARIANT #7's row for this change. `_gate_ok` is now derived from `gate_refusal`; if
    the derivation and the old spelling ever disagree, a host's cook changes shape."""
    print("\n--- HOOK-3: _gate_ok's decisions are unchanged ---")
    src = torch.rand(1, 32, 32, 3)
    cache = tex_results.ResultCache()
    stages = _linear_stages(src)
    collapsed = F.collapse_linear(stages)

    table = [
        # (stages, cache, latent, upstream, precision) -> the decision at the base sha
        ((collapsed, cache, 0, ("k",), "fp32"), True),
        ((stages, cache, 0, ("k",), "fp32"), False),
        ((_diamond(src), cache, 0, ("k",), "fp32"), False),
        ((_skip_edge(src), cache, 0, ("k",), "fp32"), False),
        ((collapsed, None, 0, ("k",), "fp32"), False),
        ((collapsed, cache, 4, ("k",), "fp32"), False),
        ((collapsed, cache, 0, ("k",), "fp16"), False),
        ((collapsed, cache, 0, ("k",), "auto"), False),
        ((collapsed, cache, 0, (), "fp32"), False),
        ((collapsed[:1], cache, 0, ("k",), "fp32"), False),
        (([], cache, 0, ("k",), "fp32"), False),
    ]
    for args, want in table:
        got = CK._gate_ok(*args)
        agrees = got == (CK.gate_refusal(args[0], args[1], latent_channel_count=args[2],
                                         upstream=args[3], precision=args[4]) is None)
        label = f"{len(args[0])} stages, cache={args[1] is not None}, {args[4]}, " \
                f"latent={args[2]}, upstream={len(args[3])}"
        if got == want and agrees:
            r.ok(f"_gate_ok -> {got} ({label})")
        else:
            r.fail("HOOK-3 decision drift",
                   f"{label}: _gate_ok -> {got}, wanted {want}, bool/refusal agree={agrees}")

    # And the whole point, end to end: the collapse is what lets the feature arm at all, and
    # the checkpointed cook is still bit-exact against the straight-through one.
    if CK.materialize(stages, cache, device="cpu", precision="fp32", upstream=("k",),
                      cuts=[1]) == []:
        r.ok("a DAG-spelled region still materializes nothing")
    else:
        r.fail("HOOK-3 arming", "materialized a checkpoint for an uncollapsed region")

    done = CK.materialize(collapsed, cache, device="cpu", precision="fp32", upstream=("k",),
                          cuts=[1])
    whole = tex_engine.cook_stage_list(collapsed, device="cpu", precision="fp32")["OUT"]
    got = CK.cook_checkpointed(collapsed, cache, device="cpu", precision="fp32",
                               upstream=("k",), cuts=[1])["OUT"]
    if done == [1] and torch.equal(got, whole):
        r.ok("the collapsed region arms a checkpoint and cooks bit-exact")
    else:
        d = (got.float() - whole.float()).abs()
        r.fail("HOOK-3 end to end",
               f"materialized {done}, maxdiff {d.max().item():.3e}")
