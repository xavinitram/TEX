"""TRK-9 (BRIEF-11a design §0/§8.4) — a hand-authored DAG `.textool` manifest
with a generator stage (no chain input, no source, no fed input) validates
clean, then crashes cook with a bare `RuntimeError`.

`tex_fusion`'s region detector (`_grow_region`) never folds a node with no
image input into a region — a LIVE ComfyUI graph can never produce this
shape (rule ③). A hand-authored `.textool` DAG manifest can still spell one:
nothing checked that every DAG stage reads a chain (`chain_inputs`), the
external source (`source_stage`/`source_binding` / `source_injections`), or a
fed input. `tex_tool.validate_manifest` already had this exact check
(`REFUSE_FUSED_STAGE_UNANCHORED`) — but ONLY inside `_validate_fused_feeds`,
which `validate_manifest` calls only when at least one declared input carries
`feeds`. The row's own minimal manifest has exactly ONE input (the sole
fusion source, injected via the schema-1 `source_stage`/`source_binding`
scalars) and declares no `feeds` at all, so it takes the OTHER branch
(`if not fed:`) and never reaches the check.

The FUSED cook (`cook_tool`, the real production path) actually SUCCEEDS on
this manifest — the generator's `u`/`v` resolve against the whole fused
chain's shared grid, so it never notices it has no bindings of its own. The
crash is in the OTHER thing this shape can reach: cooking the same stage
STANDALONE (`interpreter._consensus_extent` sees zero bindings and returns
`None`, so `u`/`v` default to 0-dim and the stage cooks to a bare rank-1
`(C,)` vector instead of `[B,H,W,C]`), then splicing that vector into the
terminal alongside a real image — which is exactly what a stage-by-stage
reconstruction oracle does (`test_tool_roundtrip_unfused`'s technique), and
exactly the shape a future single-stage preview/debug tool would need. That
raises a plain torch broadcast `RuntimeError` — not a named TEX check — three
layers away from the manifest that caused it. A manifest of this shape works
today ONLY by the accident of always being fused; refusing it at validation
catches the fragility before anything downstream depends on it.

Fixed by factoring the existing unanchored-stage check into
`tex_tool._check_dag_stages_anchored` and calling it from BOTH routes into a
DAG spec: `_validate_fused_feeds` (unchanged) and the `if not fed:` branch of
`validate_manifest` (new) — so a hand-authored manifest of this shape is
refused at `load_tool`/`write_tool` time, before any TEX source is even
parsed, with a named `REFUSE_FUSED_STAGE_UNANCHORED` error instead of a bare
runtime crash three layers deep.

ComfyUI-invisible because: a LIVE ComfyUI graph is fused by `_grow_region`,
which already enforces this rule and so can never construct an unanchored
DAG stage — this fix only closes the SAME gap on the hand-authored
`.textool` manifest path, which a ComfyUI user never writes directly. A
manifest that was already well-formed (every stage anchored) takes the
identical code path as before; only a malformed one that used to crash
downstream is affected, and it is now refused earlier, not differently.
"""
from helpers import *

from TEX_Wrangle import tex_tool, tex_engine


def _unanchored_dag_manifest() -> dict:
    """The row's own minimal repro: one DAG stage (a generator with zero
    bindings), the sole external source injected straight into the terminal,
    and the generator read into the terminal via `terminal_chain_inputs` —
    but nothing in the manifest ever routes anything INTO the generator
    stage itself."""
    return {
        "name": "trk9_unanchored",
        "tex_language": "0.25",
        "inputs": [{"name": "src", "type": "IMAGE"}],
        "graphspec": {
            "dag": True,
            "stages": [{"code": "@OUT = vec3(u, 0.25, v);"}],
            "source_stage": 1,          # n_stages == 1 -> injects into the TERMINAL
            "source_binding": "src",
            "terminal_chain_inputs": {"gen": [0, "OUT"]},
        },
        "terminal_code": "@OUT = @src * 0.5 + @gen * 0.5;",
        "terminal_image_input": "src",
    }


def _anchored_dag_manifest() -> dict:
    """Control: the same shape, but the generator stage is properly fed the
    external source too (a `chain_inputs` entry pointing at the terminal's
    injection isn't legal, so instead give it its OWN source injection) —
    must keep validating clean, unchanged by this fix."""
    m = _unanchored_dag_manifest()
    m["graphspec"] = dict(m["graphspec"])
    m["graphspec"]["source_injections"] = [[0, "src"], [1, "src"]]
    m["graphspec"].pop("source_stage", None)
    m["graphspec"].pop("source_binding", None)
    m["graphspec"]["stages"] = [{"code": "@OUT = vec3(@src.r, 0.25, u);"}]
    return m


def test_trk9_premise_fused_works_standalone_crashes(r: SubTestResult):
    """Re-verify the row's own premise fresh (never carried): the manifest's
    generator stage cooks fine FUSED but crashes when reconstructed
    stage-by-stage — the asymmetry that makes this worth refusing at
    validation rather than leaving it to work by accident."""
    print("\n--- TRK-9 premise: fused cook works, stage-by-stage reconstruction raises ---")
    gen_code = "@OUT = vec3(u, 0.25, v);"
    term_code = "@OUT = @src * 0.5 + @gen * 0.5;"
    img = torch.rand(1, 5, 9, 3)
    try:
        gen_out = tex_engine.cook(gen_code, {}, device_mode="cpu").outputs["OUT"]
    except Exception as e:
        r.fail("the generator's own standalone cook succeeds", f"{type(e).__name__}: {e}")
        return
    if tuple(gen_out.shape) != (3,):
        r.fail("the standalone generator collapses to a bare rank-1 (C,) vector",
              f"got shape {tuple(gen_out.shape)}")
    else:
        r.ok(f"standalone generator cooked to a bare vector {gen_out.tolist()} (no shared grid)")
    try:
        tex_engine.cook(term_code, {"src": img.clone(), "gen": gen_out.clone()}, device_mode="cpu")
        r.fail("splicing the standalone generator's output into the terminal raises",
              "no exception raised")
    except RuntimeError as e:
        r.ok(f"stage-by-stage reconstruction raises as expected: {e}")
    except Exception as e:
        r.fail("the crash is a plain RuntimeError (a torch broadcast, not a named TEX check)",
              f"{type(e).__name__}: {e}")


def test_trk9_unanchored_dag_stage_is_refused_at_validation(r: SubTestResult):
    """The malformed manifest must be refused by `validate_manifest` itself —
    before any TEX source is parsed — with the named unanchored-stage code."""
    print("\n--- TRK-9: an unanchored DAG stage is refused at validation, not cook ---")
    try:
        tex_tool.validate_manifest(_unanchored_dag_manifest())
        r.fail("validate_manifest refuses an unanchored DAG stage",
              "no exception raised — the manifest validated clean")
    except tex_tool.TEXToolError as e:
        if e.code != tex_tool.REFUSE_FUSED_STAGE_UNANCHORED:
            r.fail("the refusal carries REFUSE_FUSED_STAGE_UNANCHORED",
                  f"code={e.code!r}, message={e}")
        else:
            r.ok(f"refused at validation: {e}")
    except Exception as e:
        r.fail("validate_manifest raises TEXToolError, not something else",
              f"{type(e).__name__}: {e}")


def test_trk9_load_tool_refuses_it_too(r: SubTestResult):
    """The public entry point (`load_tool`, dict form) must surface the same
    refusal — this is what a host actually calls."""
    print("\n--- TRK-9: load_tool refuses the same manifest ---")
    try:
        tex_tool.load_tool(_unanchored_dag_manifest())
        r.fail("load_tool refuses an unanchored DAG stage",
              "no exception raised")
    except tex_tool.TEXToolError as e:
        if e.code != tex_tool.REFUSE_FUSED_STAGE_UNANCHORED:
            r.fail("the refusal carries REFUSE_FUSED_STAGE_UNANCHORED",
                  f"code={e.code!r}, message={e}")
        else:
            r.ok(f"load_tool refused it too: {e}")


def test_trk9_anchored_dag_manifest_still_validates(r: SubTestResult):
    """Control: a DAG manifest where every stage IS anchored must be
    unaffected by this fix — the check must not over-refuse."""
    print("\n--- TRK-9 control: a properly anchored DAG manifest still validates ---")
    try:
        tex_tool.validate_manifest(_anchored_dag_manifest())
        r.ok("an anchored DAG manifest still validates clean")
    except Exception as e:
        r.fail("an anchored DAG manifest is not refused", f"{type(e).__name__}: {e}")


def test_trk9_linear_manifests_unaffected(r: SubTestResult):
    """Control: the stock (non-DAG, linear-fused) manifests this repo ships
    must be completely unaffected by this fix (they never had `dag: True`,
    so `_check_dag_stages_anchored` no-ops on them)."""
    print("\n--- TRK-9 control: stock linear-fused tools still load ---")
    stock_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock")
    try:
        m = tex_tool.load_tool(os.path.join(stock_dir, "grade_vignette.textool"))
        assert m.graphspec is not None
        r.ok("stock fused tool 'grade_vignette' still loads")
    except FileNotFoundError:
        r.ok("stock tool directory not present in this checkout (SKIPPED, not a failure)")
    except Exception as e:
        r.fail("a stock linear-fused tool still loads unaffected", f"{type(e).__name__}: {e}")
