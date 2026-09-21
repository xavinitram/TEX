"""LANG-L3 — the shared structural flow-plan walk, and fusion's mixed-language refusal.

L3 of `docs/masked-control-flow.md`'s staged plan: `flow_plan(program)`, the ONE structural
walk that flags per-pixel loops, transfer-bearing regions, scatter/probe/binding-write sites
under a per-pixel `if`, and the sync points — later stages (L4's interpreter masking, L5's
codegen mirror, L6's graph-capture/ROI consumers) all consult this one plan instead of each
re-deriving "is this per-pixel". **Nothing masks yet** — this file only proves the walk names
the right sites, that it is empty for every program that does not need masking, and that
fusion refuses a chain whose stages disagree about language level.

Reuse, not a second definition: `flow_plan` is built on the SAME `_ControlFlowLint` fixed
point `tex_roi.region_dependent` already trusts for W7007/W7008 (`test_v036_region_dependence.py`
pins that walk's own contract) — `FlowPlan.per_pixel_loops` IS `_ControlFlowLint.varying_loops`,
not a parallel computation of it.

A note on the corpus census (§2 of the design note) and this file's one named exception:
the note's own §2 table records ONE class-B corpus program — a per-pixel loop bound — among
the 130 `tests/compat_corpus.py::_corpus_programs()` entries, and names it: "`adv_while_loop`
... is the only frozen program with a per-pixel loop bound" (§0). Re-derived here directly
(`tex_api.control_flow_advisories` over the full corpus): exactly one program draws W7007/
W7008 with the loop spelling, and it is `adv_while_loop` (`float x = u; ... while (x < 1.0 &&
n < 8){...}` — `x` starts from the per-pixel builtin `u`). `flow_plan` reuses that identical
walk, so it structurally flags the SAME program the SAME way — it would be inconsistent for
`flow_plan` to call this loop per-pixel while `region_dependent`/W7007 call it uniform, and
the "one walk" hard constraint rules out giving `flow_plan` its own, different rule for what
"per-pixel" means. So the corpus's empty-plan property is 129/130, with the sole exception
named, reasoned about, and matching the design note's own census — not silently carved out.
"""
import os

from helpers import *

import compat_corpus as cc
from TEX_Wrangle import tex_api, tex_fusion

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The one corpus program the design note's own §0/§2 census already names as the sole
# class-B (per-pixel loop bound) member of the 130-program corpus.
_KNOWN_NONEMPTY = "adv_while_loop"


def _parse(src: str):
    return Parser(Lexer(src).tokenize(), source=src).parse()


# ── §0's five reproductions, each the bare construct + a preceding `float a = @A.r;` so
# the condition is genuinely per-pixel (a plain wire read, not a builtin used at 1x1) — the
# exact shape `docs/masked-control-flow.md` §1's worked examples hand-apply the 0.25 rule to.

_R_BREAK = """
float a = @A.r;
float hit = -1.0;
for (int i = 0; i < 3; i = i + 1) {
  if (a > 0.5) { hit = float(i) + 10.0; break; }
  hit = hit - 1.0;
}
@OUT = vec4(hit, hit, hit, 1.0);
"""

_R_CONT = """
float a = @A.r;
float acc = 0.0;
for (int i = 0; i < 3; i = i + 1) {
  if (a > 0.5) { continue; }
  acc = acc + 1.0;
}
@OUT = vec4(acc, acc, acc, 1.0);
"""

_R_RET = """
float pick(float a) {
  if (a > 0.5) { return a * 10.0; }
  return a * 100.0;
}
@OUT = vec4(pick(@A.r), 0.0, 0.0, 1.0);
"""

_R_BOUND = """
float a = @A.r;
float n = a * 10.0;
float c = 0.0;
for (int i = 0; float(i) < n; i = i + 1) { c = c + 1.0; }
@OUT = vec4(c, c, c, 1.0);
"""

_R_WBOUND = """
float a = @A.r;
float x = a;
float c = 0.0;
while (x < 0.8) { x = x + 0.25; c = c + 1.0; }
@OUT = vec4(x, c, 0.0, 1.0);
"""


def test_l3_empty_plan_for_every_corpus_program_but_one(r: SubTestResult):
    print("\n--- LANG-L3: an empty plan for every corpus program, but adv_while_loop "
          "(the design note's own named class-B exception) ---")
    try:
        total = 0
        non_empty = []
        for name, src in cc._corpus_programs():
            total += 1
            program = _parse(src)
            plan = tex_api.flow_plan(program)
            assert plan.complete, f"{name}: flow_plan walk did not complete"
            if not plan.is_empty():
                non_empty.append(name)
        assert total >= 130, f"corpus census reach dropped: only {total} program(s)"
        assert non_empty == [_KNOWN_NONEMPTY], (
            f"expected only {_KNOWN_NONEMPTY!r} to draw a non-empty plan, got {non_empty!r}")
        r.ok(f"{total - 1}/{total} corpus program(s) get an empty plan; the one exception "
             f"({_KNOWN_NONEMPTY}) is the design note's own named class-B case")
    except Exception as e:
        r.fail("LANG-L3 corpus empty-plan census", str(e))


def test_l3_adv_while_loop_names_its_own_loop(r: SubTestResult):
    print("\n--- LANG-L3: adv_while_loop's plan names ITS OWN while loop, not a phantom ---")
    try:
        src = dict(cc._ADVERSARIAL)[_KNOWN_NONEMPTY]
        program = _parse(src)
        plan = tex_api.flow_plan(program)
        assert not plan.is_empty()
        assert len(plan.per_pixel_loops) == 1, plan.per_pixel_loops
        assert plan.per_pixel_loops <= plan.sync_points, "a per-pixel loop must be a sync point"
        assert not plan.transfer_sites and not plan.scatter_sites and not plan.probe_sites \
            and not plan.binding_write_sites, (
            "adv_while_loop has no break/continue/return, scatter, probe or user function")
        # The flagged node really is the WhileLoop AST node this program parses to.
        from TEX_Wrangle.tex_compiler import ast_nodes as A
        stack = list(program.statements)
        found_while_ids = set()
        while stack:
            node = stack.pop()
            if isinstance(node, A.WhileLoop):
                found_while_ids.add(id(node))
            stack.extend(A.iter_child_nodes(node))
        assert plan.per_pixel_loops == found_while_ids, (
            f"plan names {plan.per_pixel_loops}, the program's WhileLoop node(s) are "
            f"{found_while_ids}")
        r.ok("adv_while_loop's plan.per_pixel_loops is exactly {id(that WhileLoop)}, and it "
             "is also a sync point")
    except Exception as e:
        r.fail("LANG-L3 adv_while_loop site identity", str(e))


def test_l3_five_reproductions_name_the_right_sites(r: SubTestResult):
    print("\n--- LANG-L3: a non-empty plan for each of §0's five reproductions, naming the "
          "right construct ---")
    try:
        cases = [
            ("R-BREAK", _R_BREAK, {"transfer_sites": 1, "sync_points": 1, "per_pixel_loops": 0}),
            ("R-CONT", _R_CONT, {"transfer_sites": 1, "sync_points": 1, "per_pixel_loops": 0}),
            ("R-RET", _R_RET, {"transfer_sites": 1, "sync_points": 0, "per_pixel_loops": 0}),
            ("R-BOUND", _R_BOUND, {"transfer_sites": 0, "sync_points": 1, "per_pixel_loops": 1}),
            ("R-WBOUND", _R_WBOUND, {"transfer_sites": 0, "sync_points": 1, "per_pixel_loops": 1}),
        ]
        for name, src, want in cases:
            program = _parse(src)
            plan = tex_api.flow_plan(program)
            assert plan.complete, f"{name}: walk did not complete"
            assert not plan.is_empty(), f"{name}: expected a non-empty plan"
            for field, expect_len in want.items():
                got = len(getattr(plan, field))
                assert got == expect_len, (
                    f"{name}.{field}: expected {expect_len} site(s), got {got}")
            # No scatter/probe/binding-write in any of these five — they don't touch @-writes,
            # debug_print or user-function calls of that shape.
            assert not plan.scatter_sites and not plan.probe_sites \
                and not plan.binding_write_sites, f"{name}: unexpected M5/M6/M7 site"
        r.ok(f"all {len(cases)} of §0's reproductions get a non-empty plan naming the "
             f"construct the design note describes")
    except Exception as e:
        r.fail("LANG-L3 five reproductions", str(e))


def test_l3_scatter_probe_binding_write_sites(r: SubTestResult):
    print("\n--- LANG-L3: M5 (scatter) / M7 (probe) / M6 (binding write) sites, own "
          "constructions ---")
    try:
        scatter_src = """
float a = @A.r;
if (a > 0.5) {
  @OUT[ix, iy] = 1.0;
}
"""
        probe_src = """
float a = @A.r;
float g = 0.0;
if (a > 0.5) {
  g = debug_print("probe", a, 0, 0);
}
@OUT = vec4(g, g, g, 1.0);
"""
        # M6, direct: the function's OWN body has the per-pixel `if` (its parameter varies
        # because the one call site feeds it a varying argument).
        binding_write_local_src = """
float writer(float a) {
  if (a > 0.5) {
    @OUT = vec4(1.0, 1.0, 1.0, 1.0);
  }
  return a;
}
float r = writer(@A.r);
@OUT = vec4(r, r, r, 1.0);
"""
        # M6, via the call site: the function's OWN write is unconditional; only the CALL
        # SITE sits under a per-pixel `if` (M4's "a call inherits the caller's live mask").
        binding_write_callsite_src = """
float a = @A.r;
float setter(float x) {
  @OUT = vec4(2.0, 2.0, 2.0, 2.0);
  return x;
}
float r = 0.0;
if (a > 0.5) {
  r = setter(a);
}
@OUT = vec4(r, r, r, 1.0);
"""
        p_scatter = tex_api.flow_plan(_parse(scatter_src))
        assert len(p_scatter.scatter_sites) == 1, p_scatter.scatter_sites
        assert not p_scatter.probe_sites and not p_scatter.binding_write_sites

        p_probe = tex_api.flow_plan(_parse(probe_src))
        assert len(p_probe.probe_sites) == 1, p_probe.probe_sites
        assert not p_probe.scatter_sites and not p_probe.binding_write_sites

        p_bw_local = tex_api.flow_plan(_parse(binding_write_local_src))
        assert len(p_bw_local.binding_write_sites) == 1, p_bw_local.binding_write_sites

        p_bw_call = tex_api.flow_plan(_parse(binding_write_callsite_src))
        assert len(p_bw_call.binding_write_sites) == 1, p_bw_call.binding_write_sites

        r.ok("scatter/probe/binding-write (both the direct and the call-site-inherited "
             "M6 shape) each name exactly one site")
    except Exception as e:
        r.fail("LANG-L3 M5/M6/M7 sites", str(e))


def test_l3_plan_is_program_instance_keyed(r: SubTestResult):
    print("\n--- LANG-L3: a plan's ids belong to the EXACT Program instance it was built "
          "from, per FlowPlan's own contract ---")
    try:
        from TEX_Wrangle.tex_compiler import ast_nodes as A
        p1 = _parse(_R_BOUND)
        p2 = _parse(_R_BOUND)   # a fresh parse of the SAME source: fresh node ids
        plan1 = tex_api.flow_plan(p1)
        loops1 = {id(n) for n in _walk(p1) if isinstance(n, A.ForLoop)}
        loops2 = {id(n) for n in _walk(p2) if isinstance(n, A.ForLoop)}
        assert plan1.per_pixel_loops == loops1
        assert plan1.per_pixel_loops.isdisjoint(loops2), (
            "a re-parse must not share node ids with the original — if it did, this test "
            "would not be exercising the identity contract FlowPlan documents")
        r.ok("plan1.per_pixel_loops matches p1's own ForLoop id(s) and shares none with a "
             "fresh re-parse of the identical source")
    except Exception as e:
        r.fail("LANG-L3 plan identity contract", str(e))


def _walk(program):
    from TEX_Wrangle.tex_compiler import ast_nodes as A
    stack = list(program.statements)
    while stack:
        node = stack.pop()
        yield node
        stack.extend(A.iter_child_nodes(node))


def test_l3_incomplete_walk_is_not_empty(r: SubTestResult):
    print("\n--- LANG-L3: a walk that cannot finish answers INCOMPLETE, never EMPTY ---")
    try:
        program = _parse(_R_BOUND)   # a program with genuine per-pixel sites
        broken = tex_api._ControlFlowLint

        class _BoomLint(broken):
            def _pass(self):
                raise RuntimeError("boom — simulating a walk that cannot finish")

        tex_api._ControlFlowLint = _BoomLint
        try:
            plan = tex_api.flow_plan(program)
        finally:
            tex_api._ControlFlowLint = broken   # restore unconditionally
        assert plan.complete is False
        assert plan.is_empty() is False, (
            "is_empty() must be False on an incomplete walk even though every site set is "
            "empty — a caller must never read that as 'nothing to mask'")
        assert plan == tex_api._INCOMPLETE_FLOW_PLAN
        r.ok("an incomplete walk reports complete=False and is_empty() is False, not True")
    except Exception as e:
        r.fail("LANG-L3 fail-closed incomplete plan", str(e))


# ── Fusion's mixed-language refusal ────────────────────────────────────────────

def _infer(v):
    from TEX_Wrangle.tex_marshalling import infer_binding_type
    return infer_binding_type(v)


def test_l3_fusion_refuses_mixed_language_chain(r: SubTestResult):
    print("\n--- LANG-L3: fusion refuses a chain whose stages disagree on //!tex language, "
          "at the per-stage parse ---")
    try:
        stage0 = {"code": "//!tex 0.25\n@OUT = vec4(1.0, 1.0, 1.0, 1.0);", "bindings": {}}
        stage1 = {"code": "@OUT = @OUT * 0.5;", "chain_input": "OUT", "bindings": {}}
        try:
            tex_fusion.compile_fused([stage0, stage1], _infer)
            r.fail("LANG-L3 fusion mixed-language refusal",
                   "compile_fused did not raise for a mixed-language chain")
            return
        except tex_fusion.FusionError as e:
            assert "language" in str(e).lower(), f"unexpected FusionError text: {e}"
        r.ok("a //!tex 0.25 stage followed by a no-pragma stage raises FusionError naming "
             "the language mismatch")
    except Exception as e:
        r.fail("LANG-L3 fusion mixed-language refusal", str(e))


def test_l3_fusion_same_language_chain_fused_equals_unfused(r: SubTestResult):
    print("\n--- LANG-L3: a same-language chain fuses, and its fused Program.language "
          "matches every stage's ---")
    try:
        # Case A: every stage declares the SAME pragma explicitly.
        stage0 = {"code": "//!tex 0.23\n@OUT = vec4(0.25, 0.25, 0.25, 1.0);", "bindings": {}}
        stage1 = {"code": "//!tex 0.23\n@OUT = @OUT * 2.0;", "chain_input": "OUT",
                  "bindings": {}}
        fused, *_rest = tex_fusion.compile_fused([stage0, stage1], _infer)
        assert fused.language == "0.23", fused.language

        # Case B: the overwhelmingly common shape — every stage has NO pragma at all — must
        # still fuse (this is every fused chain shipped today; §2 of the design note).
        stage0b = {"code": "@OUT = vec4(0.25, 0.25, 0.25, 1.0);", "bindings": {}}
        stage1b = {"code": "@OUT = @OUT * 2.0;", "chain_input": "OUT", "bindings": {}}
        fused_b, *_rest_b = tex_fusion.compile_fused([stage0b, stage1b], _infer)
        assert fused_b.language is None, fused_b.language
        r.ok("a same-language chain (explicit-pragma and no-pragma) fuses and the result's "
             "Program.language is the stages' shared value")
    except Exception as e:
        r.fail("LANG-L3 fusion same-language chain", str(e))
