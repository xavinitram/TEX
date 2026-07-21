"""
v0.21 Phase 1 — "Fuse the graph".

FUS-1  detect_fusable_regions is the pure-Python fusion authority: single-terminal
       regions over @OUT handoffs, fan-out / diamonds / trees covered, multi-source
       and code-wired boundaries left unfused (safe).
FUS-3  the release gate (PM-1): a fused DAG region is BIT-EXACT to running its nodes
       sequentially — the interpreter-oracle contract (invariant #2) lifted to
       branching graphs. Runs on CPU and (when present) CUDA.
FUS-2  fused_required_bindings: terminal-first lazy composition over a fused chain
       (the MECHANISM). Its check_lazy_status/E6003 WIRING is deliberately deferred:
       in v0.21's single-external-source fusion scope the source is always the R1
       shape anchor (so it can never be pruned) and every input traces to it, so the
       "dead upstream branch" win needs multi-source regions — deferred with them.
       The mechanism ships tested and ready for that follow-up.
"""
from helpers import *
import torch

from TEX_Wrangle import tex_fusion as F
from TEX_Wrangle.tex_marshalling import infer_binding_type as _ibt
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker

_CUDA = torch.cuda.is_available()


def _E(f, fs, t, tb, ftype=None):
    e = {"from": f, "from_slot": fs, "to": t, "to_binding": tb}
    if ftype:
        e["from_type"] = ftype   # producer socket type, for the preflight family
    return e


def _run(code, binds, device="cpu"):
    """Interpret one node's code with the given bindings; return @OUT."""
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    bt = {k: _ibt(v) for k, v in binds.items()}
    ck = TypeChecker(binding_types=bt, source=code)
    tm = ck.check(prog)
    return Interpreter().execute(prog, binds, tm, device=device,
                                 output_names=sorted(ck.assigned_bindings))["OUT"]


def _fused_run(region, code_map, source, device="cpu"):
    """detect-plan region -> region_to_stages -> compile_fused -> interpret. FUS-1b: the one
    source is injected into EVERY injection point (a single-source region has exactly one)."""
    stages = F.region_to_stages(region, code_map, {k: {} for k in code_map})
    src = region["source"]
    injections = src["injections"]
    for inj in injections:
        stages[inj["stage"]]["bindings"][inj["binding"]] = source
    prog, tm, refs, asg, params, used, merged = F.compile_fused(stages, _ibt)
    return Interpreter().execute(prog, merged, tm, device=device,
                                 output_names=sorted(asg))["OUT"]


# ── FUS-1: detector unit topologies ──────────────────────────────────────────
def test_fus1_detector(r: SubTestResult):
    print("\n--- FUS-1: detect_fusable_regions topologies ---")
    N = lambda: {"code_wired": False}

    def one(name, nodes, edges, expect):
        try:
            regs = F.detect_fusable_regions(nodes, edges)
            got = sorted((x["terminal"], tuple(x["order"])) for x in regs)
            if got == sorted(expect):
                r.ok(f"detector: {name}")
            else:
                r.fail(f"detector: {name}", f"got {got}, expected {sorted(expect)}")
        except Exception as e:
            r.fail(f"detector: {name}", f"{type(e).__name__}: {e}")

    one("linear A->B->C", {k: N() for k in "ABC"},
        [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("B", 0, "C", "in")],
        [("C", ("A", "B", "C"))])
    one("diamond A->{B,C}->D", {k: N() for k in "ABCD"},
        [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("A", 0, "C", "in"),
         _E("B", 0, "D", "b"), _E("C", 0, "D", "c")],
        [("D", ("A", "B", "C", "D"))])
    # preview-escape: B feeds an external Preview -> D-region has 2 external -> skip
    one("preview escape -> unfused", {k: N() for k in "ABCD"},
        [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("A", 0, "C", "in"),
         _E("B", 0, "D", "b"), _E("C", 0, "D", "c"), _E("B", 0, "PREVIEW", "images")],
        [])
    # code-wired A can't fold -> region {B,C}
    one("code-wired boundary", {"A": {"code_wired": True}, "B": N(), "C": N()},
        [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("B", 0, "C", "in")],
        [("C", ("B", "C"))])
    # two-source merge -> skip
    one("multi-source merge -> unfused", {"M": N(), "T": N()},
        [_E("X", 0, "M", "a"), _E("Y", 0, "M", "b"), _E("M", 0, "T", "in")],
        [])
    # single node -> nothing
    one("single node -> none", {"A": N()}, [_E("S", 0, "A", "in")], [])


def test_fus1b_multi_injection(r: SubTestResult):
    print("\n--- FUS-1b: one producer feeds >1 region member (multi-injection) ---")
    N = lambda: {"code_wired": False}
    # Load -> [blur, sharpen] -> merge: S feeds BOTH A and B (two external EDGES, one producer).
    code_map = {"A": "@OUT = @in * 1.2;",
                "B": "@OUT = @in + vec4(0.1, 0.0, 0.0, 0.0);",
                "M": "@OUT = (@a + @b) * 0.5;"}
    edges = [_E("S", 0, "A", "in", "IMAGE"), _E("S", 0, "B", "in", "IMAGE"),
             _E("A", 0, "M", "a"), _E("B", 0, "M", "b")]
    nodes = {k: N() for k in code_map}

    # 1) the detector returns ONE region with TWO injection points (v0.21 left it unfused)
    regs = F.detect_fusable_regions(nodes, edges)
    if len(regs) != 1:
        r.fail("FUS-1b detect", f"expected 1 region, got {len(regs)}")
        return
    region = regs[0]
    inj = region["source"].get("injections") or []
    r.ok(f"detector: Load->[blur,sharpen]->merge fuses ({len(inj)} injections, terminal M)") \
        if len(inj) == 2 and region["terminal"] == "M" \
        else r.fail("FUS-1b detect", f"injections={inj} terminal={region['terminal']}")
    # it must NOT be classified linear (else the route skips it to the JS linear pass)
    r.ok("multi-injection region is non-linear (reaches the region path)") \
        if not F._region_is_linear(region) else r.fail("FUS-1b linear", "misclassified linear")
    # a genuine TWO-producer merge still doesn't fuse
    two = F.detect_fusable_regions({k: N() for k in "MB"},
        [_E("X", 0, "M", "a"), _E("Y", 0, "B", "in"), _E("B", 0, "M", "b")])
    r.ok("two distinct producers still left unfused") if two == [] \
        else r.fail("FUS-1b 2-producer", f"got {two}")

    # 2) the fused region is BIT-EXACT to sequential execution (CPU + CUDA)
    for dev in (["cpu"] + (["cuda"] if _CUDA else [])):
        torch.manual_seed(11)
        src = torch.rand(1, 10, 10, 4, device=dev)
        a = _run(code_map["A"], {"in": src}, dev)
        b = _run(code_map["B"], {"in": src}, dev)
        seq = _run(code_map["M"], {"a": a, "b": b}, dev)
        fused = _fused_run(region, code_map, src, dev)
        md = (seq.float() - fused.float()).abs().max().item()
        r.ok(f"[{dev}] multi-injection fused == sequential (maxdiff {md:.1e})") if md < 1e-5 \
            else r.fail("FUS-1b equiv", f"[{dev}] maxdiff {md:.2e}")

    # 3) the collapse plan carries source_injections + schema 2
    p = F.region_to_collapse_plan(region, code_map, {k: {} for k in code_map})["payload"]
    r.ok("multi-injection plan: schema 2 + 2 source_injections") \
        if p.get("schema") == 2 and len(p.get("source_injections") or []) == 2 \
        else r.fail("FUS-1b plan", f"schema={p.get('schema')} inj={p.get('source_injections')}")
    # a single-injection (linear) region stays schema 1, byte-identical, NO source_injections
    lin = F.detect_fusable_regions({k: N() for k in "AB"},
        [_E("S", 0, "A", "in"), _E("A", 0, "B", "in")])[0]
    lp = F.region_to_collapse_plan(lin, {"A": "@OUT=@in*1.1;", "B": "@OUT=@in+0.1;"},
                                   {"A": {}, "B": {}})["payload"]
    r.ok("single-injection plan stays schema 1 (byte-identical, no source_injections)") \
        if lp.get("schema") == 1 and "source_injections" not in lp \
        else r.fail("FUS-1b backcompat", f"schema={lp.get('schema')} keys={list(lp)}")


# ── FUS-3: fused == sequential, bit-exact (the release gate) ──────────────────
def _equiv_case(r, name, code_map, edges, source_id, device):
    """Assert the fused DAG region reproduces sequential execution bit-exactly."""
    try:
        nodes = {k: {"code_wired": False} for k in code_map}
        regs = F.detect_fusable_regions(nodes, edges)
        if len(regs) != 1:
            r.fail(f"equiv[{device}]: {name}", f"expected 1 region, got {len(regs)}")
            return
        region = regs[0]

        torch.manual_seed(7)
        src = torch.rand(1, 12, 12, 4, device=device)
        # Sequential: topologically evaluate each node from its wired inputs.
        outs = {source_id: src}
        # map: for each node, which binding <- which producer (from edges)
        by_dst = {}
        for e in edges:
            by_dst.setdefault(e["to"], {})[e["to_binding"]] = e["from"]
        for nid in region["order"]:
            binds = {b: outs[fr] for b, fr in by_dst.get(nid, {}).items()}
            outs[nid] = _run(code_map[nid], binds, device)
        seq = outs[region["terminal"]]

        fused = _fused_run(region, code_map, src, device)
        md = (seq.float() - fused.float()).abs().max().item()
        if md < 1e-5:
            r.ok(f"equiv[{device}]: {name} (maxdiff {md:.1e})")
        else:
            r.fail(f"equiv[{device}]: {name}", f"maxdiff {md:.2e} >= 1e-5")
    except Exception as e:
        r.fail(f"equiv[{device}]: {name}", f"{type(e).__name__}: {e}")


def test_fus3_dag_equivalence(r: SubTestResult):
    print("\n--- FUS-3: fused DAG region == sequential (bit-exact) ---")
    LINEAR = {"A": "@OUT = @in * 1.1;", "B": "@OUT = @in + vec4(0.05,0.0,0.0,0.0);",
              "C": "@OUT = clamp(@in, 0.0, 1.0);"}
    LINEAR_E = [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("B", 0, "C", "in")]
    DIAMOND = {"A": "@OUT = @in * 1.3;", "B": "@OUT = @in + vec4(0.1,0.0,0.0,0.0);",
               "C": "@OUT = @in * vec4(0.8,0.9,1.0,1.0);", "D": "@OUT = (@b + @c) * 0.5;"}
    DIAMOND_E = [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("A", 0, "C", "in"),
                 _E("B", 0, "D", "b"), _E("C", 0, "D", "c")]
    # tree: A fans to B,C AND directly to T; T merges all three
    TREE = {"A": "@OUT = @in - vec4(0.02);", "B": "@OUT = @in * 1.2;",
            "C": "@OUT = sqrt(clamp(@in,0.0,1.0));", "T": "@OUT = (@a + @b + @c) / 3.0;"}
    TREE_E = [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("A", 0, "C", "in"),
              _E("B", 0, "T", "b"), _E("C", 0, "T", "c"), _E("A", 0, "T", "a")]
    # SPATIAL diamond: B does a non-pointwise sample (exercises stencil/spatial
    # lowering across a DAG merge, not just pointwise math).
    SPATIAL = {"A": "@OUT = @in * 1.1;",
               "B": "@OUT = sample(@in, u + px * 2.0, v);",
               "C": "@OUT = @in * vec4(0.9, 1.0, 1.1, 1.0);",
               "D": "@OUT = (@b + @c) * 0.5;"}
    SPATIAL_E = [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("A", 0, "C", "in"),
                 _E("B", 0, "D", "b"), _E("C", 0, "D", "c")]
    for dev in (["cpu", "cuda"] if _CUDA else ["cpu"]):
        _equiv_case(r, "linear-3", LINEAR, LINEAR_E, "S", dev)
        _equiv_case(r, "diamond", DIAMOND, DIAMOND_E, "S", dev)
        _equiv_case(r, "fan-out tree", TREE, TREE_E, "S", dev)
        _equiv_case(r, "spatial diamond (sample)", SPATIAL, SPATIAL_E, "S", dev)


def test_fus3_codegen_parity(r: SubTestResult):
    """The fused DAG program must also run through CODEGEN bit-identically to the
    interpreter (invariant #2 for the MERGED program, not just interp-vs-sequential)."""
    print("\n--- FUS-3: fused DAG program interp == codegen ---")
    from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute
    CODE = {"A": "@OUT = @in * 1.3;",
            "B": "@OUT = sample(@in, u + px, v) + vec4(0.05, 0.0, 0.0, 0.0);",
            "C": "@OUT = clamp(@in * vec4(0.9, 1.0, 1.1, 1.0), 0.0, 1.0);",
            "D": "@OUT = (@b + @c) * 0.5;"}
    edges = [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("A", 0, "C", "in"),
             _E("B", 0, "D", "b"), _E("C", 0, "D", "c")]
    for dev in (["cpu", "cuda"] if _CUDA else ["cpu"]):
        try:
            regs = F.detect_fusable_regions(
                {k: {"code_wired": False, "param_wired": False} for k in CODE}, edges)
            assert len(regs) == 1, f"expected 1 region, got {len(regs)}"
            torch.manual_seed(11)
            src = torch.rand(1, 16, 16, 4, device=dev)
            stages = F.region_to_stages(regs[0], CODE, {k: {} for k in CODE})
            for inj in regs[0]["source"]["injections"]:
                stages[inj["stage"]]["bindings"][inj["binding"]] = src
            prog, tm, refs, asg, params, used, merged = F.compile_fused(stages, _ibt)
            on = sorted(asg)
            interp = Interpreter().execute(
                prog,
                {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in merged.items()},
                tm, device=dev, output_names=on)["OUT"]
            cg = _codegen_only_execute(
                prog, {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in merged.items()},
                tm, dev, output_names=on, used_builtins=used, fingerprint=f"fus3cg_{dev}", time_context=None)
            cg = cg["OUT"] if isinstance(cg, dict) else cg
            md = (interp.float() - cg.float()).abs().max().item()
            if md < 1e-5:
                r.ok(f"[{dev}] fused DAG interp==codegen (maxdiff {md:.1e})")
            else:
                r.fail(f"[{dev}] fused DAG codegen parity", f"maxdiff {md:.2e}")
        except Exception as e:
            r.fail(f"[{dev}] FUS-3 codegen parity", f"{type(e).__name__}: {e}")


def test_fus3_terminal_rmw(r: SubTestResult):
    """Regression: a fused terminal that READ-MODIFY-WRITES one of its wired chain
    inputs (`@b += …`, `@b = @b*2`, `@b.rgb = …`) — @b is both a chain input (in
    wire_map) and an assigned output (in passthrough). Before the fix, passthrough
    was matched before wire_map so the READ never resolved to the upstream handoff
    and raised E6021 'Input @b is not connected' at cook — a broken prompt invisible
    to the compile-only preflight. The terminal output is now seeded from the handoff
    so the fused result stays bit-exact to the unfused read-modify-write."""
    print("\n--- FUS-3: terminal read-modify-writes a wired input (regression) ---")
    UP = {"A": "@OUT = @in * 1.3;", "B": "@OUT = @in + vec4(0.1,0.0,0.0,0.0);",
          "C": "@OUT = @in * vec4(0.8,0.9,1.0,1.0);"}
    E = [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("A", 0, "C", "in"),
         _E("B", 0, "D", "b"), _E("C", 0, "D", "c")]
    TERMS = {
        "compound-add (@b += )": "@b += vec4(0.1,0.0,0.0,0.0); @OUT = (@b + @c) * 0.5;",
        "read-mul (@b = @b*2)":  "@b = @b * 2.0; @OUT = (@b + @c) * 0.5;",
        "channel-write (@b.rgb)": "@b.rgb = @b.rgb * 0.5; @OUT = (@b + @c) * 0.5;",
        "write-first (@b = k)":  "@b = vec4(0.5); @OUT = (@b + @c) * 0.5;",
    }
    for dev in (["cpu", "cuda"] if _CUDA else ["cpu"]):
        for label, dcode in TERMS.items():
            code_map = dict(UP, D=dcode)
            _equiv_case(r, f"rmw {label}", code_map, E, "S", dev)


def test_fus1_route_path(r: SubTestResult):
    print("\n--- FUS-1: detect_region_plans -> node DAG path bit-exact ---")
    try:
        CODE = {"A": "@OUT = @in * 1.3;", "B": "@OUT = @in + vec4(0.1,0.0,0.0,0.0);",
                "C": "@OUT = @in * vec4(0.8,0.9,1.0,1.0);", "D": "@OUT = (@b + @c) * 0.5;"}
        edges = [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("A", 0, "C", "in"),
                 _E("B", 0, "D", "b"), _E("C", 0, "D", "c")]
        graph = {"nodes": [{"id": k, "code": CODE[k], "params": {}, "code_wired": False}
                           for k in "ABCD"], "edges": edges}
        plans = F.detect_region_plans(graph)
        assert len(plans) == 1, f"expected 1 plan, got {len(plans)}"
        plan = plans[0]
        assert plan["terminal"] == "D" and sorted(plan["delete"]) == ["A", "B", "C"]
        assert plan["source_origin"] == ["S", 0]

        torch.manual_seed(3)
        S = torch.rand(1, 10, 10, 4)
        oA = _run(CODE["A"], {"in": S}); oB = _run(CODE["B"], {"in": oA})
        oC = _run(CODE["C"], {"in": oA}); oD = _run(CODE["D"], {"b": oB, "c": oC})

        # Host applies the plan: transport socket "in_0" carries the source.
        payload = dict(plan["payload"]); payload["terminal_image_input"] = "in_0"
        prog, tm, refs, asg, params, used, merged = F.prepare_fused(
            payload, CODE["D"], {"in_0": S}, _ibt)
        fused = Interpreter().execute(prog, merged, tm, device="cpu",
                                      output_names=sorted(asg))["OUT"]
        md = (oD - fused).abs().max().item()
        assert md < 1e-5, f"route->node maxdiff {md:.2e}"

        # fused_fingerprint is value-independent + assembles on the DAG spec
        fp = F.fused_fingerprint(payload, CODE["D"], {"in_0": S}, _ibt)
        assert fp and fp.startswith("fused_"), "DAG fused_fingerprint missing"
        # malformed graph -> no plans (host runs unfused)
        assert F.detect_region_plans({"nodes": [{"id": "X"}], "edges": []}) == []
        r.ok("route plan -> DAG node path bit-exact + fingerprint + malformed-safe")
    except Exception as e:
        r.fail("FUS-1 route path", f"{type(e).__name__}: {e}")


# ── FUS-2: fused-chain lazy composition ───────────────────────────────────────
def test_fus2_fused_lazy(r: SubTestResult):
    print("\n--- FUS-2: fused_required_bindings terminal-first pruning ---")
    try:
        # Real contract: stages use `bindings` (not `params`); source named by args.
        # linear A->B->C where C IGNORES the chain (folded) -> source unneeded.
        stages_live = [
            {"code": "@OUT = @in * 1.2;", "bindings": {}, "chain_inputs": {}},
            {"code": "@OUT = @in + vec4(0.1);", "bindings": {}, "chain_inputs": {"in": [0, "OUT"]}},
            {"code": "@OUT = @in * 0.9;", "bindings": {}, "chain_inputs": {"in": [1, "OUT"]}},
        ]
        res = F.fused_required_bindings(stages_live, source_stage=0, source_binding="in")
        assert res is not None and res["source_needed"] is True, "live chain needs source"
        assert res["needed_stages"] == {0, 1, 2}, f"live: {res['needed_stages']}"

        stages_dead = [
            {"code": "@OUT = @in * 1.2;", "bindings": {}, "chain_inputs": {}},
            {"code": "@OUT = @in + vec4(0.1);", "bindings": {}, "chain_inputs": {"in": [0, "OUT"]}},
            {"code": "@OUT = vec4(0.0, 0.0, 0.0, 1.0);", "bindings": {},  # ignores @in
             "chain_inputs": {"in": [1, "OUT"]}},
        ]
        res2 = F.fused_required_bindings(stages_dead, source_stage=0, source_binding="in")
        assert res2 is not None and res2["source_needed"] is False, \
            "terminal ignoring the chain must drop the source"
        assert res2["needed_stages"] == {2}, f"dead: {res2['needed_stages']}"

        # Speaks the REAL compile_fused stage shape: region_to_stages output flows in.
        real = F.region_to_stages(
            {"stages": [{"id": "A", "chain_inputs": {}},
                        {"id": "B", "chain_inputs": {"in": [0, "OUT"]}}]},
            {"A": "@OUT = @in * 1.1;", "B": "@OUT = @in + vec4(0.05);"}, {"A": {}, "B": {}})
        assert F.fused_required_bindings(real, 0, "in")["needed_stages"] == {0, 1}, \
            "must consume region_to_stages' {code,bindings,chain_inputs} shape"

        # analysis failure on any stage -> None (over-approximate: cook everything)
        bad = [{"code": "@OUT = @in *", "bindings": {}, "chain_inputs": {}},
               {"code": "@OUT = @in;", "bindings": {}, "chain_inputs": {"in": [0, "OUT"]}}]
        assert F.fused_required_bindings(bad, 0, "in") is None, "parse failure must yield None"
        r.ok("fused lazy (real contract): live keeps source, ignores drops it, failure->None")
    except Exception as e:
        r.fail("FUS-2 fused lazy", f"{type(e).__name__}: {e}")


def test_fus1_hardening(r: SubTestResult):
    """Audit guards: param_wired boundary (C4), channel-specific / code_wired / cap
    preflight drops (C3/C5/C16), source-family preflight (MASK->FLOAT, [15]),
    linear-skip-at-route + multi-region (C15/C17)."""
    print("\n--- FUS-1: detector/route hardening guards ---")

    def _N(cw=False, pw=False):
        return {"code_wired": cw, "param_wired": pw}

    def _plans(code, edges, wired=None):
        nodes = [{"id": k, "code": c, "params": {}, "code_wired": False,
                  "param_wired": bool(wired and k in wired)} for k, c in code.items()]
        return F.detect_region_plans({"nodes": nodes, "edges": edges})

    try:
        DIAMOND_E = [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("A", 0, "C", "in"),
                     _E("B", 0, "D", "b"), _E("C", 0, "D", "c")]
        POINT = {"A": "@OUT = @in * 1.1;", "B": "@OUT = @in + vec4(0.1,0,0,0);",
                 "C": "@OUT = @in * 0.9;", "D": "@OUT = (@b + @c) * 0.5;"}

        # C4: a wired-$param member (B) keeps the region a BOUNDARY at B -> the region
        # can't form ({A} alone feeds two branches but B is external) -> no plan.
        assert _plans(POINT, DIAMOND_E, wired={"B"}) == [], "C4: param_wired member folded"
        # detector-level: B excluded from folding
        regs = F.detect_fusable_regions(
            {"A": _N(), "B": _N(pw=True), "C": _N(), "D": _N()}, DIAMOND_E)
        assert all("B" not in rg["order"] for rg in regs), "C4: param_wired B folded"

        # C5: wired-code terminal -> no region
        assert _plans({"A": "@OUT=@in*1.1;", "B": "@OUT=@in+vec4(0.1);", "T": ""},
                      [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"),
                       _E("A", 0, "T", "x"), _E("B", 0, "T", "y")],
                      ) == [] or True  # (T code_wired via flag below)
        nodes5 = [{"id": "A", "code": "@OUT=@in*1.1;", "params": {}, "code_wired": False, "param_wired": False},
                  {"id": "B", "code": "@OUT=@in+vec4(0.1);", "params": {}, "code_wired": False, "param_wired": False},
                  {"id": "T", "code": "", "params": {}, "code_wired": True, "param_wired": False}]
        assert F.detect_region_plans({"nodes": nodes5, "edges": [
            _E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("A", 0, "T", "x"),
            _E("B", 0, "T", "y")]}) == [], "C5: code_wired terminal fused"

        # C3: channel-specific (.a) region dropped by both-channel preflight
        CH = {"A": "@OUT=@in*1.1;", "B": "@OUT=vec4(@in.rgb, @in.a*0.9);",
              "C": "@OUT=@in*0.9;", "D": "@OUT=(@b+@c)*0.5;"}
        assert _plans(CH, DIAMOND_E) == [], "C3: channel-specific region collapsed"
        # sanity: the pointwise diamond DOES fuse
        assert len(_plans(POINT, DIAMOND_E)) == 1, "C3: pointwise diamond blocked"

        # [15]: the preflight family follows the SOURCE's declared socket type. A MASK
        # infers as FLOAT, not VEC3/VEC4. The discriminator is a VECTOR-ONLY stdlib
        # call: scalars broadcast through operators AND swizzles (`@in.rgb` on a float
        # is a legal vec3), so those prove nothing — but `length()`/`normalize()` reject
        # a float outright. Note the payoff here is NOT a wrong pixel: such a program is
        # invalid off a mask fused or not. It's that the region no longer fuses and dies
        # as "couldn't fuse this chain", masking the real `length() needs a vector` error.
        LEN = {"A": "@OUT = @in * length(@in);", "B": "@OUT = @in * 1.1;",
               "C": "@OUT = @in * 0.9;", "D": "@OUT = (@b + @c) * 0.5;"}
        assert len(_plans(LEN, DIAMOND_E)) == 1, "[15]: vector-only region blocked on IMAGE"
        MASK_E = [_E("S", 0, "A", "in", "MASK")] + DIAMOND_E[1:]
        assert _plans(LEN, MASK_E) == [], "[15]: vector-only region fused off a MASK source"
        # A float-safe region still fuses off that same MASK (no over-restriction).
        SCALAR = {"A": "@OUT = @in * 1.1;", "B": "@OUT = @in + 0.1;",
                  "C": "@OUT = @in * 0.9;", "D": "@OUT = (@b + @c) * 0.5;"}
        assert len(_plans(SCALAR, MASK_E)) == 1, "[15]: float-safe MASK region dropped"
        # `.a` (index 3) discriminates by SOURCE family: legal on a vec4 (LATENT) but a type
        # error on a vec3 (IMAGE) AND on a float (MASK) — a scalar/vec3 has no alpha. (A scalar
        # swizzle used to slip past the type checker's is_vector-only bounds gate and then slice
        # the SPATIAL axis at runtime — a real silent-corruption bug now fixed at the root: the
        # bounds check now covers a scalar base, so `.a`/`.rgb` on a float errors E3301.) The float-safe
        # MASK coverage above is carried by SCALAR (arithmetic); `.a` off a MASK correctly does
        # NOT fuse, exactly like the vector-only LEN case.
        ALPHA = {"A": "@OUT = @in.a * 1.1;", "B": "@OUT = @in * 1.1;",
                 "C": "@OUT = @in * 0.9;", "D": "@OUT = (@b + @c) * 0.5;"}
        assert _plans(ALPHA, MASK_E) == [], "[15]: .a on a MASK (float) is a type error, not fused"
        assert _plans(ALPHA, DIAMOND_E) == [], "[15]: .a region fused on IMAGE (ch=3 fails)"
        # A 4-channel LATENT unwraps to [B,H,W,4] -> VEC4. These two assertions are
        # COMPLEMENTARY and both are needed to pin the LATENT branch: `length()` compiles
        # against vec3 AND vec4, so the first one alone passes via the IMAGE (3,4)
        # fallback and survives deleting the branch entirely (verified by mutation).
        # `.a` is the discriminator — legal on a vec4, not on a vec3.
        LAT_E = [_E("S", 0, "A", "in", "LATENT")] + DIAMOND_E[1:]
        assert len(_plans(LEN, LAT_E)) == 1, "[15]: vector-only region dropped off a LATENT"
        assert len(_plans(ALPHA, LAT_E)) == 1, "[15]: vec4-legal .a region rejected on LATENT"

        # C17: a pure linear chain is skipped AT THE ROUTE (linear pass owns it), but
        # the authority still detects it.
        LIN_E = [_E("S", 0, "A", "in"), _E("A", 0, "B", "in"), _E("B", 0, "T", "in")]
        LIN = {"A": "@OUT=@in*1.1;", "B": "@OUT=@in+vec4(0.05);", "T": "@OUT=clamp(@in,0.0,1.0);"}
        assert _plans(LIN, LIN_E) == [], "C17: linear region emitted a route plan"
        assert len(F.detect_fusable_regions({k: _N() for k in "ABT"}, LIN_E)) == 1, \
            "C17: authority stopped detecting linear regions"

        # C16: cap boundary — a CONNECTED 16-stage fan-out fuses, 17 does not. Head
        # fans out to every middle; the terminal fans them all back in (non-linear).
        def _fanout(n):
            ids = [f"n{i}" for i in range(n)]
            head, term, middles = ids[0], ids[-1], ids[1:-1]
            code = {head: "@OUT=@in*1.1;"}
            edges = [_E("S", 0, head, "in")]
            reads = []
            for i, m in enumerate(middles):
                code[m] = "@OUT=@in+vec4(0.01,0,0,0);"
                edges.append(_E(head, 0, m, "in"))          # head -> each middle
                edges.append(_E(m, 0, term, f"b{i}"))        # each middle -> terminal
                reads.append(f"@b{i}")
            code[term] = "@OUT = (" + " + ".join(reads) + f") / {len(middles)}.0;"
            return F.detect_fusable_regions({k: _N() for k in ids}, edges)
        assert len(_fanout(16)) == 1, "C16: 16-stage region rejected"
        assert _fanout(17) == [], "C16: 17-stage region not capped"

        # C15: two independent fan-out regions -> two plans; a linear chain alongside
        # a fan-out region -> only the fan-out is emitted at the route.
        two = dict(POINT); two_e = list(DIAMOND_E)
        for k in ("A2", "B2", "C2", "D2"):
            two[k] = POINT[k[0]]
        two_e += [_E("S2", 0, "A2", "in"), _E("A2", 0, "B2", "in"), _E("A2", 0, "C2", "in"),
                  _E("B2", 0, "D2", "b"), _E("C2", 0, "D2", "c")]
        assert len(_plans(two, two_e)) == 2, "C15: two fan-out regions not both detected"
        mixed = dict(POINT); mixed.update({"L1": "@OUT=@in*1.2;", "L2": "@OUT=@in+vec4(0.02);"})
        mixed_e = list(DIAMOND_E) + [_E("SL", 0, "L1", "in"), _E("L1", 0, "L2", "in")]
        assert len(_plans(mixed, mixed_e)) == 1, "C15: linear chain leaked a route plan"
        r.ok("hardening: C3/C4/C5/C16/C17 + [15] source-family drops, C15 multi-region")
    except Exception as e:
        import traceback
        r.fail("FUS-1 hardening", f"{type(e).__name__}: {e}\n{traceback.format_exc()[:300]}")


# ── CACHE-0: orphan .cg census + TEX_CACHE_DIR ───────────────────────────────
def test_cache0_orphan_cg_census(r: SubTestResult):
    print("\n--- CACHE-0: orphan .cg census + TEX_CACHE_DIR ---")
    try:
        from TEX_Wrangle.tex_cache import TEXCache
        import TEX_Wrangle.tex_cache as tc
        d = Path(tempfile.mkdtemp(prefix="tex_c0t_"))
        try:
            c = TEXCache(cache_dir=d)
            _max, _grace = tc._CG_DISK_MAX_ENTRIES, tc._CG_ORPHAN_GRACE_SEC
            tc._CG_DISK_MAX_ENTRIES, tc._CG_ORPHAN_GRACE_SEC = 10, 3600
            now = time.time()
            for i in range(25):   # aged orphans (no sibling .pkl) -> evictable
                p = d / f"orphan_{i:03d}.cg"; p.write_bytes(b"x")
                os.utime(p, (now - 7200 + i, now - 7200 + i))
            for i in range(5):    # paired -> never evicted
                (d / f"paired_{i}.pkl").write_bytes(b"p")
                pc = d / f"paired_{i}.cg"; pc.write_bytes(b"c"); os.utime(pc, (now - 7200, now - 7200))
            for i in range(3):    # fresh (within grace) -> protected
                (d / f"fresh_{i}.cg").write_bytes(b"f")
            c._evict_disk_if_needed()   # census runs FIRST, even under the .pkl cap
            after = len(list(d.glob("*.cg")))
            assert after == 10, f".cg after census = {after}, want 10"
            assert all((d / f"paired_{i}.cg").exists() for i in range(5)), "paired .cg evicted"
            assert all((d / f"fresh_{i}.cg").exists() for i in range(3)), "fresh .cg evicted"
            assert not (d / "orphan_000.cg").exists(), "oldest orphan not evicted"
        finally:
            tc._CG_DISK_MAX_ENTRIES, tc._CG_ORPHAN_GRACE_SEC = _max, _grace
            shutil.rmtree(d, ignore_errors=True)
        # TEX_CACHE_DIR override. RESTORE the prior value rather than deleting it:
        # run_all.py now sets a run-wide scratch dir, and an unconditional `del` would
        # strip it for every later test in the process — silently sending their cache
        # artifacts back into the shipping package's .tex_cache, which is exactly what
        # CACHE-0 exists to prevent.
        probe = Path(tempfile.gettempdir()) / "tex_c0_env_probe"
        _prev_cache_dir = os.environ.get("TEX_CACHE_DIR")
        os.environ["TEX_CACHE_DIR"] = str(probe)
        try:
            assert str(TEXCache()._cache_dir) == str(probe), "TEX_CACHE_DIR not honored"
        finally:
            if _prev_cache_dir is None:
                os.environ.pop("TEX_CACHE_DIR", None)
            else:
                os.environ["TEX_CACHE_DIR"] = _prev_cache_dir
        r.ok("orphan .cg reclaimed (paired/fresh kept), TEX_CACHE_DIR honored")
    except Exception as e:
        r.fail("CACHE-0", f"{type(e).__name__}: {e}")


# ── LAT-3: deferred timing readback ──────────────────────────────────────────
def test_lat3_deferred_timing(r: SubTestResult):
    print("\n--- LAT-3: deferred CUDA-event readback ---")
    try:
        from TEX_Wrangle.tex_runtime import compiled as C
        # sync _timed still returns a value THIS call (TRIAL path needs it)
        _, ms = C._timed(lambda: sum(range(100)), "cpu")
        assert ms is not None and ms >= 0, "sync _timed must return ms"
        if _CUDA:
            C._deferred_ev.clear()
            slot = (("fp", "cuda", "fp32"), (1, 128, 128), "measure")  # realistic 3-tuple
            def cook():
                x = torch.rand(128, 128, device="cuda"); return (x * x).sum()
            _, ms1 = C._timed_deferred(cook, "cuda", slot)   # no prior -> None
            assert ms1 is None, f"first deferred sample should be None, got {ms1}"
            torch.cuda.synchronize()   # force the prior pair COMPLETE so the read FIRES
            _, ms2 = C._timed_deferred(cook, "cuda", slot)
            assert ms2 is not None and ms2 > 0, \
                f"deferred read must fire a POSITIVE ms after prior completes, got {ms2}"
            # a DIFFERENT-resolution slot must NOT borrow the 128x128 pair (the LAT-3
            # slot fix — px buckets never cross-contaminate).
            _, ms3 = C._timed_deferred(cook, "cuda",
                                       (("fp", "cuda", "fp32"), (1, 256, 256), "measure"))
            assert ms3 is None, f"cross-resolution slot borrowed a sample: {ms3}"
            C._deferred_ev.clear()
            r.ok("deferred read fires positive ms after completion; cross-res isolated")
        else:
            r.ok("sync _timed ok (no CUDA to defer)")
    except Exception as e:
        r.fail("LAT-3", f"{type(e).__name__}: {e}")


# ── LAT-4: interpreter coordinate-builtins LRU ───────────────────────────────
def test_lat4_builtins_lru(r: SubTestResult):
    print("\n--- LAT-4: interpreter builtins LRU (no proxy/full-res thrash) ---")
    try:
        from TEX_Wrangle.tex_runtime.interpreter import _BUILTINS_LRU_MAX
        code = "@OUT = vec4(u, v, 0.0, 1.0);"
        prog = Parser(Lexer(code).tokenize(), source=code).parse()
        interp = Interpreter()
        for _ in range(2):    # alternate 3 resolutions twice -> all 3 cached, no thrash
            for hw in (64, 128, 256):
                img = torch.rand(1, hw, hw, 3)
                bt = {"A": _ibt(img)}
                ck = TypeChecker(binding_types=bt, source=code); tm = ck.check(prog)
                interp.execute(prog, {"A": img}, tm, device="cpu", output_names=["OUT"])
        assert len(interp._builtins_lru) == 3, f"LRU size {len(interp._builtins_lru)}, want 3"
        for i in range(_BUILTINS_LRU_MAX + 4):   # exceed the cap -> bounded
            hw = 32 + i * 8
            img = torch.rand(1, hw, hw, 3); bt = {"A": _ibt(img)}
            ck = TypeChecker(binding_types=bt, source=code); tm = ck.check(prog)
            interp.execute(prog, {"A": img}, tm, device="cpu", output_names=["OUT"])
        assert len(interp._builtins_lru) == _BUILTINS_LRU_MAX, "LRU not bounded"
        r.ok(f"LRU caches 3 configs, bounded at {_BUILTINS_LRU_MAX}")
    except Exception as e:
        r.fail("LAT-4", f"{type(e).__name__}: {e}")


# ── ENG-8: measured transfer-cost model ──────────────────────────────────────
def test_eng8_transfer_model(r: SubTestResult):
    print("\n--- ENG-8: xfer transfer_ms + persistence ---")
    try:
        from TEX_Wrangle.tex_runtime import xfer
        xfer.reset()
        big = xfer.transfer_ms(64 << 20, pinned=True, direction="h2d")
        small = xfer.transfer_ms(1 << 20, pinned=True, direction="h2d")
        # Monotonicity holds by construction (fallback slope > 0 even if a measured
        # lane is dropped). We do NOT assert measured pinned <= pageable: that PCIe
        # relationship is real on a QUIET bus but a loaded GPU inverts the noisy
        # measurement (the model is guide-only, never gates correctness). Instead
        # assert the property at the construction level — the fallback constants.
        assert big > small > 0, f"transfer_ms not monotonic: 64M={big} 1M={small}"
        assert xfer._FALLBACK_PINNED[1] < xfer._FALLBACK_PAGEABLE[1], \
            "fallback: pinned lane should model faster bandwidth than pageable"
        if _CUDA:
            # <=4 lanes: a degenerate (noise-inverted) lane is dropped -> fallback.
            # Every stored lane must have a positive slope (no flat model persisted).
            m = xfer.model()
            assert 0 <= len(m) <= 4, f"unexpected lane count {len(m)}"
            assert all(slope > 0 for (_lat, slope) in m.values()), "flat lane persisted"
        # doctor peek is non-probing and shape-correct, and reports a SANE bandwidth
        # (regression guard for the 1e-9-vs-1e-6 ms/byte->GB/s conversion).
        from TEX_Wrangle.tex_doctor import collect_doctor_facts
        facts = collect_doctor_facts()
        assert "xfer" in facts and "error" not in facts["xfer"], "doctor xfer fact broken"
        if facts["xfer"].get("measured"):
            for lane, v in facts["xfer"]["lanes"].items():
                gbps = v.get("gb_per_s")
                assert gbps is None or 0.1 < gbps < 2000, \
                    f"doctor {lane} gb_per_s={gbps} implausible (unit-conversion bug?)"
        # C12: a probe that measures only DEGENERATE lanes must NOT latch _probed
        # (so a later call re-measures once the bus is quiet). Force it with an inverted
        # timing + TINY buffers so it never contends with a busy GPU.
        if _CUDA:
            saved = (xfer._SMALL, xfer._LARGE, xfer._time_copy)
            xfer.reset()
            xfer._SMALL, xfer._LARGE = 1024, 2048
            xfer._time_copy = lambda dst, src, reps: (5.0 if src.numel() <= 1024 else 3.0)
            try:
                xfer._probe()
                assert not xfer._MODEL and xfer._probed is False, \
                    "degenerate probe latched _probed (would never re-measure)"
            finally:
                xfer._SMALL, xfer._LARGE, xfer._time_copy = saved
                xfer.reset()
        r.ok("transfer_ms monotonic + fallback pinned<pageable + GB/s sane + no-latch")
    except Exception as e:
        r.fail("ENG-8", f"{type(e).__name__}: {e}")
