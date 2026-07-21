"""
v0.29 Phase 1 — "Close the register" (the consolidation the v0.28 audit ordered).

No new mechanisms — this release closes the deferral register §9 accumulated:

ENG-4   the re-cut: `tex_engine._compile_or_raise` is the ONE site that catches the raw
        per-phase compiler exceptions and raises the public `TEXCompileError`; the node,
        the CLI, and `tex_api.compile` all catch that one type (no host imports the
        compiler's internals). The raw tuple collapsed from three modules to one raiser.
SCHED-3 the bridge: the ComfyUI adapter passes the host's own interrupt into the cook via
        `prepare(cancel=get_host_services().cancel_token())`. The token reads comfy's
        interrupt flag (READ-ONLY) and raises `CookCancelled` at a yield point; the node's
        `except CookCancelled` re-surfaces the host's clean InterruptProcessingException.
FUS-1b  multi-injection: one external producer may feed >1 region member (Load ->
        [blur, sharpen] -> merge) — the detector groups external edges by producer and the
        source spec carries a list of injection points (GraphSpec schema 1 -> 2).
small   the sweep: count_var's outer-counter route declined (falls back to the interpreter
        oracle, never a wrong divisor); the spatial-scalar `.r`/`.x` runtime mis-slice fixed
        with an interpreter-side base-rank check.

Everything here is a consolidation off the default cook path or a host-armed seam
(invariant #7). CPU-pinned: the tier-parity checks run through `run_both`, which is CPU-only.
"""
from helpers import *  # noqa: F401,F403  (SubTestResult, torch, make_img, run_both, assert_equiv,
                       # TEXType and the per-phase compiler error types)
from TEX_Wrangle import tex_engine, tex_api
from TEX_Wrangle.tex_runtime import host as _host
from TEX_Wrangle.tex_runtime.host import CookCancelled
from test_v027_phase1 import _TripToken   # the SCHED-3 trip token (defined with its own tests)


def _tiers_agree(i, c):
    """A codegen result of None means codegen declined and the interpreter is the sole executor —
    no divergence is possible. Otherwise the two tiers must be bit-identical (invariant #2)."""
    return c is None or (i["OUT"].shape == c["OUT"].shape and torch.equal(i["OUT"], c["OUT"]))


# ── ENG-4: the single raiser ───────────────────────────────────────────────────

def test_eng4_recut_single_raiser(r: SubTestResult):
    print("\n--- ENG-4 re-cut: engine is the one TEXCompileError raiser ---")
    # LexerError / ParseError / TypeCheckError / TEXMultiError already come from `helpers`.
    from TEX_Wrangle.tex_compiler.diagnostics import TEXCompileError

    # 1) tex_engine._compile_or_raise wraps every per-phase failure into the public type.
    cases = {
        "lexer": "@OUT = vec4(1.0, 1.0, 1.0, 1.0) ` ;",   # stray backtick
        "parser": "@OUT = @A +;",                          # dangling operator
        "typecheck": "@OUT = undefinedfn(@A);",            # unknown function
    }
    ok = True
    for label, src in cases.items():
        try:
            tex_engine._compile_or_raise(src, {"A": TEXType.VEC3})
            ok = False
            r.fail("ENG-4 raiser", f"{label}: no raise")
        except TEXCompileError as e:
            if not (e.diagnostics and hasattr(e.diagnostics[0], "to_dict")):
                ok = False
                r.fail("ENG-4 raiser", f"{label}: empty diagnostics")
        except (LexerError, ParseError, TypeCheckError, TEXMultiError) as e:
            ok = False
            r.fail("ENG-4 raiser", f"{label}: leaked raw {type(e).__name__}")
    if ok:
        r.ok("engine._compile_or_raise wraps lexer/parser/typecheck -> TEXCompileError")

    # 2) tex_api.compile delegates to the same raiser (still the public type).
    try:
        tex_api.compile("@OUT = undefinedfn(@A);", {"A": TEXType.VEC3})
        r.fail("ENG-4 api", "compile did not raise")
    except tex_api.TEXCompileError as e:
        r.ok(f"tex_api.compile -> TEXCompileError ({e.diagnostics[0].code})")

    # 3) a multi-error program yields a multi-diagnostic TEXCompileError (list, not a raw
    #    TEXMultiError) — the node/CLI render the whole batch from .diagnostics.
    try:
        tex_engine._compile_or_raise("@OUT = badfn(@A) + alsobad(@A);", {"A": TEXType.VEC3})
        r.fail("ENG-4 multi", "no raise")
    except TEXCompileError as e:
        r.ok(f"multi-error -> one TEXCompileError with {len(e.diagnostics)} diagnostics") \
            if len(e.diagnostics) >= 1 else r.fail("ENG-4 multi", "lost diagnostics")

    # 4) the node still surfaces a compile error as RuntimeError + the TEX_DIAG: JSON suffix
    #    (the frontend contract) — built from .diagnostics, not the raw per-phase types.
    from TEX_Wrangle.tex_node import TEXWrangleNode
    img = make_img(1, 8, 8, 3, seed=1)
    try:
        TEXWrangleNode.execute(code="@OUT = undefinedfn(@A);", device="cpu",
                               compile_mode="none", precision="fp32", A=img)
        r.fail("ENG-4 node", "node did not raise")
    except RuntimeError as e:
        msg = str(e)
        import json as _json
        if "\nTEX_DIAG:" in msg:
            payload = _json.loads(msg.split("\nTEX_DIAG:", 1)[1])
            r.ok(f"node RuntimeError carries TEX_DIAG JSON ({len(payload)} diag)") \
                if isinstance(payload, list) and payload and "code" in payload[0] \
                else r.fail("ENG-4 node", f"bad TEX_DIAG payload: {payload}")
        else:
            r.fail("ENG-4 node", f"no TEX_DIAG suffix: {msg[:80]}")

    # 5) valid code is unaffected by the re-cut (both api and engine).
    prog = tex_api.compile("@OUT = vec4(@A.rgb * 1.2, 1.0);", {"A": TEXType.VEC3})
    r.ok("valid code compiles clean through the re-cut") if "OUT" in prog.assigned \
        else r.fail("ENG-4 valid", "clean program failed")

    # 6) the FUSED path also raises the public type. compile_fused's per-stage compile never
    #    ran through _compile_or_raise, so a typo in a fused upstream node used to escape RAW to
    #    the node's bug-report catch-all (losing the TEX_DIAG). It now wraps to TEXCompileError.
    from TEX_Wrangle import tex_fusion as _F
    from TEX_Wrangle.tex_marshalling import infer_binding_type as _ibt
    img3 = make_img(1, 8, 8, 3, seed=3)
    for label, up in (("typecheck (.a on a vec3)", "@OUT = vec4(@A.a, 0.0, 0.0, 1.0);"),
                      ("parse", "@OUT = @A *;")):
        chain = [{"code": up, "bindings": {"A": img3}},
                 {"code": "@OUT = @X;", "chain_input": "X", "bindings": {}}]
        try:
            _F.compile_fused(chain, _ibt)
            r.fail("ENG-4 fused", f"{label}: per-stage error did not raise")
        except TEXCompileError as e:
            r.ok(f"fused per-stage {label} error -> TEXCompileError ({e.diagnostics[0].code})")
        except (LexerError, ParseError, TypeCheckError, TEXMultiError) as e:
            r.fail("ENG-4 fused", f"{label}: leaked raw {type(e).__name__} on the fused path")
    # a genuine FUSION structural problem (missing @OUT) stays a FusionError, not a compile error
    try:
        _F.compile_fused([{"code": "vec3 c = @A.rgb;", "bindings": {"A": img3}},
                          {"code": "@OUT = @X;", "chain_input": "X", "bindings": {}}], _ibt)
        r.fail("ENG-4 fused", "missing-@OUT structural error did not raise")
    except _F.FusionError:
        r.ok("a structural fusion error (missing @OUT) stays FusionError, not TEXCompileError")
    except TEXCompileError:
        r.fail("ENG-4 fused", "a structural fusion error was mis-wrapped as TEXCompileError")

    # audit regression: the SPLICED compile can raise TEXMultiError (>=2 cross-stage type errors),
    # a SIBLING of TypeCheckError — it must be caught too, else it escapes to the node's bug-report
    # catch-all (ENG-4 removed the node's old `except TEXMultiError` net). Force it and assert the
    # wrap holds.
    from TEX_Wrangle import tex_cache as _tc
    from TEX_Wrangle.tex_compiler.diagnostics import make_diagnostic as _mk
    _cache = _tc.get_cache()
    _orig = _cache.compile_ast
    _cache.compile_ast = lambda *a, **k: (_ for _ in ()).throw(
        TEXMultiError([_mk("E3301", "err one", None), _mk("E3301", "err two", None)]))
    try:
        _F.compile_fused([{"code": "@OUT = vec4(@A.rgb, 1.0);", "bindings": {"A": img3}},
                          {"code": "@OUT = vec4(@X.rgb + 0.1, 1.0);", "chain_input": "X", "bindings": {}}], _ibt)
        r.fail("ENG-4 fused multi", "spliced TEXMultiError did not raise")
    except _F.FusionError:
        r.ok("spliced TEXMultiError (>=2 errors) -> FusionError, not the bug-report catch-all")
    except TEXMultiError:
        r.fail("ENG-4 fused multi", "TEXMultiError escaped the fused compile raw")
    finally:
        _cache.compile_ast = _orig


# ── SCHED-3 bridge: the ComfyUI adapter passes the host interrupt as cancel= ─────

class _FakeInterrupt(BaseException):
    """Stands in for comfy.model_management.InterruptProcessingException (a BaseException
    the ComfyUI executor treats as a clean interrupt, bypassing every `except Exception`)."""


class _FakeMM:
    """A stand-in for comfy.model_management: a settable interrupt flag + the two APIs the
    SCHED-3 bridge reads (processing_interrupted / throw_exception_if_processing_interrupted)."""
    def __init__(self):
        self.flag = False
        self.threw = False

    def processing_interrupted(self):
        return self.flag

    def throw_exception_if_processing_interrupted(self):
        if self.flag:
            self.flag = False          # clears, exactly like the real API
            self.threw = True
            raise _FakeInterrupt()


def test_sched3_bridge_token(r: SubTestResult):
    print("\n--- SCHED-3 bridge: the interrupt token (unit) ---")

    # Null host: no interrupt to bridge -> None token, no-op re-surface (CLI/standalone).
    null = _host.NullHostServices()
    r.ok("Null.cancel_token() is None (default path byte-identical)") if null.cancel_token() is None \
        else r.fail("SCHED-3 null", "cancel_token not None")
    null.raise_if_interrupted()   # must not raise
    r.ok("Null.raise_if_interrupted() is a no-op")

    mm = _FakeMM()
    comfy = _host.ComfyHostServices(mm)
    tok = comfy.cancel_token()

    # flag clear -> check() is a no-op
    tok.check()
    r.ok("token.check() no-op while the host flag is clear")

    # flag set -> check() raises CookCancelled, and DOES NOT clear the flag (read-only)
    mm.flag = True
    try:
        tok.check()
        r.fail("SCHED-3 token", "check() did not raise on a set flag")
    except CookCancelled:
        r.ok("token.check() raises CookCancelled on a set flag") if mm.flag is True \
            else r.fail("SCHED-3 token", "check() cleared the flag (must be read-only)")

    # raise_if_interrupted() re-surfaces the host's OWN clean interrupt (BaseException) and clears
    try:
        comfy.raise_if_interrupted()
        r.fail("SCHED-3 resurface", "raise_if_interrupted did not throw the host interrupt")
    except _FakeInterrupt:
        r.ok("raise_if_interrupted() -> host InterruptProcessingException, flag cleared") \
            if mm.flag is False and mm.threw else r.fail("SCHED-3 resurface", "flag not cleared")

    # a broken host API never fails a cook
    class _BrokenMM:
        def processing_interrupted(self):
            raise RuntimeError("boom")
    _host.ComfyHostServices(_BrokenMM()).cancel_token().check()  # must not raise
    r.ok("a broken host interrupt API degrades to no-op (never fails a cook)")


def test_count_var_outer_decline(r: SubTestResult):
    print("\n--- small-item (a): count_var outer-counter declines to the oracle ---")
    img = make_img(1, 10, 10, 3, seed=5)

    # An OUTER-body counter increments only kH (=3) times, so the interpreter divides by 3. The
    # stencil fast-path emitted the count as kH*kW (=9) regardless of WHICH loop the counter lived
    # in, so codegen produced sum/9 — an invariant-#2 divergence (~0.5). The v0.29 decline refuses
    # that lowering, and codegen falls to a generic loop that matches the interpreter bit-for-bit.
    outer = ("vec3 sum = vec3(0.0);\nfloat count = 0.0;\n"
             "for (int dx = -1; dx <= 1; dx = dx + 1) {\n"
             "    count = count + 1.0;\n"
             "    for (int dy = -1; dy <= 1; dy = dy + 1) {\n"
             "        sum = sum + sample(@A, u + float(dx)/iw, v + float(dy)/ih);\n"
             "    }\n}\n@OUT = vec4(sum / count, 1.0);\n")
    assert_equiv(r, "count_var outer-counter (codegen == interpreter, no kW error)",
                 outer, {"A": img})

    # The INNER-counter box blur (the normal, parity-fuzzed pattern) is untouched — it still
    # lowers to the fast path and matches, so the decline is strictly a shrink of the route.
    inner = ("vec3 sum = vec3(0.0);\nfloat count = 0.0;\n"
             "for (int dx = -1; dx <= 1; dx = dx + 1) {\n"
             "    for (int dy = -1; dy <= 1; dy = dy + 1) {\n"
             "        sum = sum + sample(@A, u + float(dx)/iw, v + float(dy)/ih);\n"
             "        count = count + 1.0;\n"
             "    }\n}\n@OUT = vec4(sum / count, 1.0);\n")
    assert_equiv(r, "count_var inner-counter box blur (normal case untouched)",
                 inner, {"A": img})

    # audit regression: a nest with BOTH an outer AND an inner counter must ALSO decline — the
    # box lowers with count=inner_count_var but DELETES the whole nest, dropping the outer
    # counter's increment (codegen leaves it at init, the interpreter counted kH -> divergence
    # on any downstream use). The first-cut guard only handled the outer-ONLY case.
    both = ("vec3 sum = vec3(0.0);\nfloat ocount = 0.0;\nfloat icount = 0.0;\n"
            "for (int dx = -1; dx <= 1; dx = dx + 1) {\n"
            "    ocount = ocount + 1.0;\n"
            "    for (int dy = -1; dy <= 1; dy = dy + 1) {\n"
            "        sum = sum + sample(@A, u + float(dx)/iw, v + float(dy)/ih);\n"
            "        icount = icount + 1.0;\n"
            "    }\n}\n@OUT = vec4(sum / icount * (ocount / 3.0), 1.0);\n")
    assert_equiv(r, "count_var outer+inner counters (outer used downstream, declined)",
                 both, {"A": img})


def test_pm5_governor_soak(r: SubTestResult):
    print("\n--- BENCH-1 / PM-5: many-frame governor soak (stays under budget) ---")
    import gc
    import shutil
    import tempfile
    from TEX_Wrangle import tex_memory
    from TEX_Wrangle.tex_results import ResultCache, lineage_key
    try:
        import psutil
        proc = psutil.Process(os.getpid())
    except ImportError:
        proc = None

    reg = tex_memory.get_cache_registry()
    d = tempfile.mkdtemp(prefix="tex_pm5_")
    rc = ResultCache(budget_mb=100000, cache_dir=d)     # huge self-budget: only the governor evicts
    # The governor sees only numel x itemsize, and the frames' VALUES are never read back here
    # (the bit-exact restore is pinned by test_cache5_governor), so use `empty` and a modest HW:
    # every assertion is derived from `per`, so it is structurally identical at a fraction of the
    # RNG + spill-to-disk cost this test would otherwise pay on every suite run.
    HW = 256
    per = HW * HW * 3 * 4                                # ~0.8 MB per fp32 frame
    budget = 8 * per                                    # room for ~8 of the 50 frames

    def fill(base, n):
        for i in range(n):
            rc.put(lineage_key(program_fp=f"pm5_{base}_{i}", device="cpu", precision="fp32"),
                   torch.empty(1, HW, HW, 3), canvas={"shape": [1, HW, HW, 3]})

    # a true 4K frame proves the governor handles a full-res result without OOM (the arbitration
    # is frame-size-agnostic — the 50-node soak below scales the COUNT, not the frame size).
    rc.put(lineage_key(program_fp="pm5_4k", device="cpu", precision="fp32"),
           torch.empty(1, 4096, 4096, 3), canvas={"shape": [1, 4096, 4096, 3]})
    tex_memory.register_result_cache(rc, name="pm5")
    try:
        fill("a", 50)                                   # 50 frames -> most must spill under budget
        reg.arbitrate("cpu", budget=budget)
        gc.collect()
        rss0 = proc.memory_info().rss / (1 << 20) if proc else 0.0
        held = rc.governed_bytes("cpu") <= budget
        for cycle in range(5):                          # soak: churn more frames + re-arbitrate
            fill(f"c{cycle}", 20)
            reg.arbitrate("cpu", budget=budget)
            if rc.governed_bytes("cpu") > budget:
                held = False
        gc.collect()
        r.ok(f"governor holds 50+ frames (incl. a 4K) under {budget >> 20} MB across 6 arbitrations") \
            if held else r.fail("PM-5 governor", f"over budget: {rc.governed_bytes('cpu')} > {budget}")
        if proc:
            grow = proc.memory_info().rss / (1 << 20) - rss0
            r.ok(f"soak: RSS flat over the governor churn (+{grow:.1f} MB)") if grow < 120.0 \
                else r.fail("PM-5 soak-rss", f"RSS grew {grow:.1f} MB over the churn")
    finally:
        reg.unregister("pm5")
        shutil.rmtree(d, ignore_errors=True)


def test_spatial_scalar_channel_access(r: SubTestResult):
    print("\n--- small-item (b): spatial-scalar .r/.x base-rank check ---")
    # A [B,H,W] mask / per-pixel scalar has a SPATIAL trailing dim, not channels. `@mask.r`
    # used to slice a pixel column (base[..., 0] on [1,H,W] -> [1,H]); it is now identity. Both
    # tiers agree: the interpreter does the base-rank check, codegen bails to it (no divergence).
    mask = torch.rand(1, 8, 8)

    # read: @mask.r == the mask (identity), and the result is full-resolution (not a column)
    i, c = run_both("@OUT = vec4(@in.r, @in.r, @in.r, 1.0);", {"in": mask.clone()})
    got = i["OUT"][..., 0]
    r.ok("@mask.r is identity (full-res mask, not an x-column)") \
        if got.shape == mask.shape and torch.allclose(got, mask, atol=1e-6) \
        else r.fail("spatial-scalar read", f"shape {tuple(got.shape)} identity={torch.allclose(got, mask, atol=1e-6)}")
    r.ok("read: codegen agrees (bails to interp — no invariant-#2 divergence)") \
        if _tiers_agree(i, c) else r.fail("spatial-scalar read #2", "codegen diverged")

    # a genuine vector .r still fast-paths on codegen (the common case is untouched)
    img = make_img(1, 8, 8, 3, seed=1)
    iv, cv = run_both("@OUT = vec4(@A.r, @A.r, @A.r, 1.0);", {"A": img.clone()})
    r.ok("@image.r (vec3) keeps the codegen fast path, bit-exact") \
        if cv is not None and torch.equal(iv["OUT"], cv["OUT"]) else r.fail("spatial-scalar vec", "vec .r lost codegen or diverged")

    # write: @mask.r = v means the whole mask becomes v (not a column write)
    iw, cw = run_both("@in.r = 0.5; @OUT = vec4(@in, @in, @in, 1.0);", {"in": mask.clone()})
    gw = iw["OUT"][..., 0]
    r.ok("@mask.r = v writes the whole scalar (not a column)") \
        if torch.allclose(gw, torch.full_like(gw, 0.5), atol=1e-6) \
        else r.fail("spatial-scalar write", "column-write instead of scalar replace")
    r.ok("write: codegen agrees (bails to interp)") \
        if _tiers_agree(iw, cw) else r.fail("spatial-scalar write #2", "codegen diverged")

    # .g/.b/.a on a scalar is still a compile error (unchanged — the type layer rejects it)
    errs = [d for d in tex_api.check("@OUT = vec4(@in.g, 0.0, 0.0, 1.0);", {"in": TEXType.FLOAT})
            if d.severity == "error"]
    r.ok("scalar .g still compile-rejected (E3301)") if errs and errs[0].code == "E3301" \
        else r.fail("spatial-scalar typecheck", f"expected E3301, got {[e.code for e in errs]}")

    # bug-hunt regression (CRITICAL): the `m.r = v` identity write must OWN its buffer. The first
    # cut stored `_ensure_spatial(value, sp)`, which hands back the RHS tensor ITSELF when its dims
    # already match (or a stride-0 expand for a 0-dim), while the write-back claims ownership
    # (`_inplace_ready.add`) — so a later in-place op scribbled on another local, on an
    # already-stored @OUT, or on the CALLER's input tensor (the ComfyUI wire), and a 0-dim RHS
    # crashed with "more than one element ... single memory location".
    IMG = make_img(1, 4, 4, 3, seed=9)
    src_before = IMG.clone()
    out = tex_api.execute(tex_api.compile(
        "float a = @IN.r; float b2 = @IN.g; a.r = b2; a = a + 0.5; @OUT = vec4(a, b2, b2, 1.0);",
        {"IN": TEXType.VEC3}), {"IN": IMG}, device="cpu")
    r.ok("`m.r = v` owns its buffer: the caller's input is not mutated") \
        if torch.equal(IMG, src_before) else r.fail("spatial-scalar alias", "the INPUT tensor was mutated")
    r.ok("`m.r = v` then `m + k`: the aliased source keeps its value") \
        if torch.allclose(out["OUT"][..., 1], src_before[..., 1], atol=1e-6) \
        else r.fail("spatial-scalar alias", "the RHS local was corrupted by the later in-place add")
    try:   # a 0-dim RHS must not produce an expanded (stride-0) buffer that a later op mutates
        tex_api.execute(tex_api.compile(
            "float lum = @IN.r; lum.r = 0.5; lum = lum * 2.0; @OUT = vec4(lum, lum, lum, 1.0);",
            {"IN": TEXType.VEC3}), {"IN": IMG.clone()}, device="cpu")
        r.ok("`m.r = <0-dim>` then in-place op does not crash on a stride-0 view")
    except RuntimeError as e:
        r.fail("spatial-scalar alias", f"0-dim RHS produced a non-owned buffer: {str(e)[:60]}")
    # the @binding write-back arm was equally aliased pre-fix — a later op on the RHS local must
    # not leak into the binding it was assigned into.
    ob = tex_api.execute(tex_api.compile(
        "float y = @IN.r; y = y * 2.0; @M.r = y; y = y + 1.0; @OUT = vec4(@M, @M, @M, 1.0);",
        {"IN": TEXType.VEC3, "M": TEXType.FLOAT}),
        {"IN": IMG.clone(), "M": torch.zeros(1, 4, 4)}, device="cpu")
    r.ok("`@binding.r = v` owns its buffer (the later `v + 1` does not leak in)") \
        if torch.allclose(ob["OUT"][..., 0], src_before[..., 0] * 2.0, atol=1e-6) \
        else r.fail("spatial-scalar alias", "the BindingRef write-back aliased the RHS local")

    # invariant #2 (bug-hunt regression): a VECTOR-typed local that is channel-less at RUNTIME
    # (`vec3 cc = @mask`, truncate-coerced to [B,H,W]) must be INDEXED by both tiers, not go
    # identity in the interpreter — the guard keys on the STATIC type, matching codegen's
    # non-vector bail. Before the fix, interp did identity while codegen indexed (they diverged).
    m2 = torch.rand(1, 6, 6)
    agree = True
    for prog in ("vec3 cc = @M; @OUT = cc.r;",
                 "vec3 cc = @M; cc.r = 0.5; @OUT = cc;",
                 "vec3 cc = @M; @OUT = cc.g;"):
        iv, cv = run_both(prog, {"M": m2.clone()})
        if not _tiers_agree(iv, cv):
            agree = False
    r.ok("vector-typed channel-less base: interp == codegen (no invariant-#2 divergence)") \
        if agree else r.fail("spatial-scalar inv2", "a tier diverged on vec-typed channel-less .r/.g")


class _CancelHost(_host.NullHostServices):
    """A host whose cook is cancellable: cancel_token() trips at the Nth yield, and
    raise_if_interrupted() re-surfaces a (fake) clean host interrupt — the ComfyUI adapter's
    behaviour, headlessly. Inherits Null's memory/OOM answers so the engine cooks normally."""
    def __init__(self, trip_n):
        self._tok = _TripToken(trip_n)

    def cancel_token(self):
        return self._tok

    def raise_if_interrupted(self):
        raise _FakeInterrupt()


def test_sched3_bridge_node(r: SubTestResult):
    print("\n--- SCHED-3 bridge: the node aborts a cook on a host interrupt ---")
    from TEX_Wrangle.tex_node import TEXWrangleNode
    code = "@OUT = vec4(@A.rgb * 1.2, 1.0);"
    img = make_img(1, 32, 32, 3, seed=2)

    # default (real) host in this non-comfy env is Null -> cancel_token() is None -> the node
    # cooks normally (byte-identical). Prove the wiring doesn't perturb the happy path.
    out = TEXWrangleNode.execute(code=code, device="cpu", compile_mode="none",
                                 precision="fp32", A=img.clone())
    node_out = (out if isinstance(out, tuple) else getattr(out, "args", out))[0]
    r.ok("default host (None token): node cooks normally") if node_out.shape == (1, 32, 32, 3) \
        else r.fail("SCHED-3 node default", f"perturbed {tuple(node_out.shape)}")

    # arm a cancellable host: the cook aborts at yield A, the node re-surfaces the CLEAN host
    # interrupt (a BaseException) — NOT the red "TEX bug" RuntimeError from the catch-all.
    _host.set_host_services(_CancelHost(1))
    try:
        TEXWrangleNode.execute(code=code, device="cpu", compile_mode="none",
                               precision="fp32", A=img.clone())
        r.fail("SCHED-3 node", "node did not abort under a host interrupt")
    except _FakeInterrupt:
        r.ok("node re-surfaces the host's clean interrupt (not a bug-report RuntimeError)")
    except RuntimeError as e:
        r.fail("SCHED-3 node", f"CookCancelled leaked as a node error: {str(e)[:80]}")
    finally:
        _host.reset_host_services()
