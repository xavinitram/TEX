"""
v0.22 Phase 1 — "The engine seam".

ENG-1  the cook orchestrator moved out of the ComfyUI node into `tex_engine`. The move
       is MECHANICAL, so most of its evidence is the rest of the suite staying green;
       what's pinned here is the SEAM: that the engine cooks with no node in the call
       path, that `cook()` is reachable, and that the node is now a marshaller.
ENG-3  egress profiles. 'comfy' is canary-pinned byte-identical (clamp / alpha-drop /
       gray-expand); 'engine' preserves values (no clamp, alpha kept, channels kept).
ENG-4  structured diagnostics: tex_api raises TEXCompileError(diagnostics=[...]) instead
       of leaking raw compiler exception types; TEXDiagnostic.to_dict's key set is a
       de-facto frontend contract and is canary-pinned.
ENG-5  embedding contracts: the ui-payload keys, the GraphSpec stage shape, and the
       HostServices method set are pinned so a host can depend on them.
ENG-7  host time context: frame/fps/time enter as BUILTINS (never $params) and must not
       churn the interpreter's coordinate-builtin LRU (LAT-4) per frame.
SCHED-1 the _tex_chain spec is a schema-versioned GraphSpec.
"""
from helpers import *
import torch

_PKG = Path(__file__).resolve().parent.parent


def _resolve_auto(code, px=2048 * 2048, dev="cuda"):
    """The C1 auto-precision gate's verdict for `code` at the fp16 region (CUDA, >=2048²).
    Pure static analysis — no GPU needed, so this pins the gate on any box."""
    from TEX_Wrangle.tex_runtime.precision_policy import resolve_auto_precision
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    return resolve_auto_precision(prog, px, dev)


# ── ENG-3: egress profiles ────────────────────────────────────────────────

def test_eng3_comfy_profile_canary(r: SubTestResult):
    print("\n--- ENG-3: the 'comfy' egress profile is byte-identical (canary) ---")
    # This profile IS the ComfyUI IMAGE/MASK contract every workflow downstream of a TEX
    # node depends on. It is pinned value-by-value: if a refactor ever "tidies" the clamp
    # or the alpha-drop away, every existing user's output changes silently.
    from TEX_Wrangle.tex_marshalling import prepare_output, get_egress_profile
    from TEX_Wrangle.tex_runtime.stdlib import LUMA_R, LUMA_G, LUMA_B
    fails = []
    if get_egress_profile() != "comfy":
        fails.append(f"default profile is {get_egress_profile()!r}, must be 'comfy'")

    raw4 = torch.tensor([[[[1.6, -0.4, 0.5, 0.25]]]])          # [1,1,1,4], out of range
    img = prepare_output(raw4, "IMAGE")
    if img.shape[-1] != 3:
        fails.append(f"IMAGE kept {img.shape[-1]} channels — alpha must be dropped")
    if not torch.equal(img, torch.tensor([[[[1.0, 0.0, 0.5]]]])):
        fails.append(f"IMAGE not clamped to [0,1]: {img.flatten().tolist()}")

    gray = prepare_output(torch.tensor([[[0.5]]]), "IMAGE")     # [1,1,1] -> [1,1,1,3]
    if gray.shape != (1, 1, 1, 3):
        fails.append(f"gray IMAGE not expanded to RGB: {tuple(gray.shape)}")

    two = prepare_output(torch.tensor([[[[0.5, 0.25]]]]), "IMAGE")   # 2ch -> zero-padded
    if not torch.equal(two, torch.tensor([[[[0.5, 0.25, 0.0]]]])):
        fails.append(f"2ch IMAGE not zero-padded: {two.flatten().tolist()}")

    m = prepare_output(raw4, "MASK")
    want = min(max(LUMA_R * 1.6 + LUMA_G * -0.4 + LUMA_B * 0.5, 0.0), 1.0)
    if abs(m.item() - want) > 1e-6:
        fails.append(f"MASK luma/clamp drifted: {m.item()} != {want}")

    lat = prepare_output(torch.rand(1, 4, 5, 4) * 3 - 1, "LATENT")    # never clamped
    if lat.shape != (1, 4, 1, 4) and lat.dim() != 4:
        fails.append(f"LATENT shape unexpected: {tuple(lat.shape)}")
    if lat.max() <= 1.0 and lat.min() >= 0.0:
        fails.append("LATENT looks clamped — latents must keep out-of-range values")

    if fails:
        r.fail("ENG-3 comfy canary", "; ".join(fails))
    else:
        r.ok("'comfy' profile pinned: clamp + alpha-drop + gray-expand + 2ch-pad + "
             "MASK luma; LATENT unclamped")


def test_eng3_engine_profile_preserves_values(r: SubTestResult):
    print("\n--- ENG-3: the 'engine' profile preserves scene-linear values ---")
    from TEX_Wrangle.tex_marshalling import (
        prepare_output, set_egress_profile, get_egress_profile)
    fails = []
    raw4 = torch.tensor([[[[1.6, -0.4, 0.5, 0.25]]]])

    img = prepare_output(raw4, "IMAGE", profile="engine")
    if img.shape[-1] != 4:
        fails.append(f"engine IMAGE dropped alpha ({img.shape[-1]}ch) — must keep it")
    if not torch.equal(img, raw4):
        fails.append(f"engine IMAGE altered values: {img.flatten().tolist()}")
    if img.dtype != torch.float32:
        fails.append(f"engine IMAGE dtype {img.dtype}, expected fp32")

    gray = prepare_output(torch.tensor([[[0.5]]]), "IMAGE", profile="engine")
    if gray.shape != (1, 1, 1, 1):
        fails.append(f"engine gray IMAGE should stay 1ch BHWC: {tuple(gray.shape)}")

    m = prepare_output(torch.tensor([[[1.6]]]), "MASK", profile="engine")
    if abs(m.item() - 1.6) > 1e-6:
        fails.append(f"engine MASK clamped: {m.item()} != 1.6")

    # The two profiles must actually DIFFER (a canary that can't fail is worthless).
    if torch.equal(prepare_output(raw4, "IMAGE", profile="comfy").float(),
                   prepare_output(raw4, "IMAGE", profile="engine")[..., :3]):
        fails.append("comfy and engine IMAGE agree on an out-of-range input — "
                     "the clamp is not being exercised")

    # B5: the profile's docstring says "Never pinned" — LATENT used to delegate to the
    # comfy body and inherit its page-locking, contradicting it. Pinning needs CUDA and
    # >=1MB, so this only bites on a real box.
    if torch.cuda.is_available():
        lat = prepare_output(torch.randn(1, 512, 512, 4), "LATENT", profile="engine")
        if lat.is_pinned():
            fails.append("engine LATENT egress is page-locked, but the profile says "
                         "'Never pinned' — an engine host owns its own transfers (XPU-2)")
        if prepare_output(torch.randn(1, 512, 512, 4), "LATENT").is_pinned() is False:
            fails.append("comfy LATENT egress STOPPED being pinned — that is the XPU "
                         "v0.20 overlap the node-hop path depends on")

    # Host-set, process-wide, and restorable.
    try:
        set_egress_profile("engine")
        if get_egress_profile() != "engine":
            fails.append("set_egress_profile did not take effect")
        if not torch.equal(prepare_output(raw4, "IMAGE"), raw4):
            fails.append("process-wide profile not honored by the default call")
    finally:
        set_egress_profile("comfy")
    if get_egress_profile() != "comfy":
        fails.append("profile not restored")
    try:
        set_egress_profile("nonsense")
        fails.append("set_egress_profile accepted an unknown profile")
    except ValueError:
        pass

    if fails:
        r.fail("ENG-3 engine profile", "; ".join(fails))
    else:
        r.ok("'engine' profile: no clamp, alpha + channels kept, fp32 BHWC; host-set, "
             "validated, and demonstrably different from 'comfy'")


# ── ENG-4 / ENG-5: the embedding contracts ────────────────────────────────

def test_eng4_structured_compile_error(r: SubTestResult):
    print("\n--- ENG-4: tex_api raises TEXCompileError(diagnostics=[...]) ---")
    from TEX_Wrangle import tex_api
    from TEX_Wrangle.tex_compiler.diagnostics import TEXCompileError, TEXDiagnostic
    fails = []
    if tex_api.TEXCompileError is not TEXCompileError:
        fails.append("tex_api does not re-export the canonical TEXCompileError")

    # One public type for every compiler phase — the point of the item.
    cases = {
        "lexer": "@OUT = vec3(1.0) $$$ ;",
        "parser": "@OUT = vec3(1.0",
        "type_checker": "@OUT = vec3(1.0) * \"str\";",
        "undefined": "@OUT = vec3(nope);",
    }
    for phase, src in cases.items():
        try:
            tex_api.compile(src, {})
            fails.append(f"{phase}: invalid code compiled without raising")
        except TEXCompileError as e:
            if not e.diagnostics:
                fails.append(f"{phase}: TEXCompileError carried no diagnostics")
            elif not all(isinstance(d, TEXDiagnostic) for d in e.diagnostics):
                fails.append(f"{phase}: diagnostics are not TEXDiagnostic instances")
            elif not str(e):
                fails.append(f"{phase}: TEXCompileError renders to an empty message")
        except Exception as e:
            fails.append(f"{phase}: leaked a raw {type(e).__name__} instead of "
                         "TEXCompileError (the host would have to import the compiler)")

    # Valid code is unaffected.
    try:
        if tex_api.compile("@OUT = vec4(@A.rgb, 1.0);", {"A": TEXType.VEC3}) is None:
            fails.append("valid compile returned None")
    except Exception as e:
        fails.append(f"valid code raised {type(e).__name__}: {e}")

    # The NODE deliberately still uses the raw types for its TEX_DIAG: suffix.
    from TEX_Wrangle.tex_node import TEXWrangleNode
    try:
        TEXWrangleNode.execute(code=cases["type_checker"], device="cpu",
                               compile_mode="none", A=make_img(1, 4, 4, 3))
        fails.append("node did not raise on invalid code")
    except RuntimeError as e:
        if "TEX_DIAG:" not in str(e):
            fails.append("node's RuntimeError lost the TEX_DIAG: suffix the JS parses")
    except Exception as e:
        fails.append(f"node raised {type(e).__name__}, not the RuntimeError the JS expects")

    if fails:
        r.fail("ENG-4 structured diagnostics", "; ".join(fails))
    else:
        r.ok("tex_api raises TEXCompileError with TEXDiagnostics for lexer/parser/type "
             "errors; valid code unaffected; the node keeps its TEX_DIAG: protocol")


def test_eng5_embedding_canaries(r: SubTestResult):
    print("\n--- ENG-5: pinned embedding contracts (ui payload / GraphSpec / host) ---")
    # These key sets are consumed by code we do not control on the other side (the
    # shipped JS, an embedding host). Renaming one is a breaking change that no type
    # checker catches, so it is pinned here explicitly.
    from TEX_Wrangle.tex_compiler.diagnostics import TEXDiagnostic
    from TEX_Wrangle.tex_runtime import host as _host
    from TEX_Wrangle import tex_fusion
    from TEX_Wrangle.tex_compiler.ast_nodes import SourceLoc
    fails = []

    # (a) TEXDiagnostic.to_dict — a de-facto frontend contract since v0.15.
    d = TEXDiagnostic(code="E3001", severity="error", message="m",
                      loc=SourceLoc(1, 1), source_line="x", docs_url="u")
    want = {"code", "severity", "message", "line", "col", "end_col",
            "source_line", "suggestions", "hint", "phase", "docs_url"}
    got = set(d.to_dict().keys())
    if got != want:
        fails.append(f"TEXDiagnostic.to_dict keys changed: +{got - want} -{want - got}")
    # docs_url is conditional — absent, not null, when empty.
    if "docs_url" in TEXDiagnostic(code="E1", severity="error", message="m", loc=None,
                                   source_line="").to_dict():
        fails.append("to_dict emits docs_url when empty (was conditional)")

    # (b) the ui= HUD payload the JS reads.
    from TEX_Wrangle.tex_node import TEXWrangleNode
    payload = TEXWrangleNode._build_ui_payload(1.0, "cpu", "fp32", None)
    if set(payload.keys()) - {"tex_perf", "tex_probes"}:
        fails.append(f"ui payload grew a key: {set(payload.keys())}")
    # tex_probes must be pinned with a payload that actually HAS probes. Without one this
    # assertion is vacuous — review mutation-tested it: renaming the key left the test
    # green, because no debug_print had run so the key was never attached and
    # `keys() - {..., "tex_probes"}` was trivially empty. A canary that survives its own
    # mutant pins nothing.
    from TEX_Wrangle.tex_runtime import tier_trace
    tier_trace.reset()
    tier_trace.record_probe("dbg", 1.0, 0, 0)
    if not tier_trace.get_probes():
        fails.append("harness: record_probe did not register — the tex_probes canary "
                     "would be vacuous again")
    elif "tex_probes" not in TEXWrangleNode._build_ui_payload(1.0, "cpu", "fp32", None):
        fails.append("the ui payload dropped tex_probes while probes existed — the JS "
                     "reads that key by name")
    tier_trace.reset()
    perf_want = {"tier", "fallback_from", "reason", "elapsed_ms", "device",
                 "precision", "precision_reason"}
    perf_got = set(payload["tex_perf"][0].keys())
    if perf_got != perf_want:
        fails.append(f"tex_perf keys changed: +{perf_got - perf_want} -{perf_want - perf_got}")
    if "near_singularities" not in TEXWrangleNode._build_ui_payload(
            1.0, "cpu", "fp32", 3)["tex_perf"][0]:
        fails.append("near_singularities missing when the debug toggle supplies a count")

    # (c) HostServices — the method set a host must implement (PORT-1).
    for m in ("get_free_memory", "free_memory", "is_oom", "soft_empty_cache"):
        if not callable(getattr(_host.NullHostServices(), m, None)):
            fails.append(f"HostServices lost {m}()")

    # (d) SCHED-1: the GraphSpec is versioned, defaults to 1, and refuses the future.
    if tex_fusion.GRAPHSPEC_SCHEMA != 1:
        fails.append(f"GRAPHSPEC_SCHEMA is {tex_fusion.GRAPHSPEC_SCHEMA}, expected 1")
    spec = {"stages": [{"code": "@OUT = @A * 2.0;", "image_input": "A", "params": {}}],
            "terminal_image_input": "A"}
    img = make_img(1, 8, 8, 3)
    try:      # absent schema == pre-v0.22 emitter == schema 1; every saved workflow.
        tex_fusion.prepare_fused(spec, "@OUT = @A + 0.1;", {"A": img},
                                 _infer_binding_type)
    except Exception as e:
        fails.append(f"a schema-less (pre-v0.22) GraphSpec was rejected: {e}")
    try:      # explicit current schema
        tex_fusion.prepare_fused({**spec, "schema": 1}, "@OUT = @A + 0.1;", {"A": img},
                                 _infer_binding_type)
    except Exception as e:
        fails.append(f"schema=1 rejected: {e}")
    try:      # a NEWER emitter must fail legibly, never be mis-spliced
        tex_fusion.prepare_fused({**spec, "schema": 99}, "@OUT = @A + 0.1;", {"A": img},
                                 _infer_binding_type)
        fails.append("a schema-99 GraphSpec was accepted — a future shape would be "
                     "silently mis-read")
    except tex_fusion.FusionError as e:
        if "newer TEX" not in str(e):
            fails.append(f"schema-99 error is not actionable: {e}")

    if fails:
        r.fail("ENG-5 embedding canaries", "; ".join(fails))
    else:
        r.ok("pinned: TEXDiagnostic.to_dict (11 keys, conditional docs_url), the ui= "
             "payload (tex_perf/tex_probes), HostServices' 4 methods, GraphSpec schema=1 "
             "(absent==1, future rejected)")


# ── ENG-7: host time context ──────────────────────────────────────────────

def test_eng7_time_builtins_advance(r: SubTestResult):
    print("\n--- ENG-7: frame/fps/time are per-cook VALUES that actually advance ---")
    # The whole hazard of this feature is a FROZEN animation: every cache between a
    # playhead and a pixel is keyed on things that do not move when the playhead does.
    # This drives the same interpreter instance across cooks at ONE resolution — the
    # exact shape that hits the LAT-4 builtins LRU — and asserts the value moves.
    from TEX_Wrangle import tex_engine
    from TEX_Wrangle.tex_runtime.interpreter import (
        _TIME_BUILTIN_NAMES, _CACHEABLE_BUILTIN_NAMES, _BUILTIN_NAMES)
    fails = []
    # Structural, and deliberately so: mutation-testing showed the behavioural check below
    # does NOT catch a broken cacheable-set (the store runs before _set_time_builtins, so
    # the entry holds no playhead either way). This assertion is the only thing that pins
    # the contract, and the contract is what keeps the guarantee from resting on the order
    # of two statements. Pinning it structurally is the honest option, not a proxy for the
    # behaviour test — they cover different mutants.
    if _TIME_BUILTIN_NAMES & _CACHEABLE_BUILTIN_NAMES:
        fails.append("a time builtin is in the cacheable set — the LRU could hold a "
                     "playhead if the store is ever reordered below _set_time_builtins")
    if not _TIME_BUILTIN_NAMES <= _BUILTIN_NAMES:
        fails.append("time builtins missing from _BUILTIN_NAMES")

    img = make_img(1, 8, 8, 3, seed=7)
    code = "@OUT = vec4(vec3(frame), 1.0);"
    seen = []
    for f in (0.0, 1.0, 2.0, 1.0):     # includes a REVISIT: a stale LRU would also break it
        res = tex_engine.cook(code, {"A": img}, time_context={"frame": f})
        seen.append(res.outputs["OUT"][0, 0, 0, 0].item())
    if seen != [0.0, 1.0, 2.0, 1.0]:
        fails.append(f"frame did not track the playhead across cooks: {seen} != "
                     "[0.0, 1.0, 2.0, 1.0] (a cache is serving a stale value)")

    # The coordinate builtins must STILL be cached — the fix must not defeat LAT-4.
    interp = tex_engine._get_interpreter()
    interp._builtins_lru.clear()
    for f in (0.0, 1.0, 2.0):
        tex_engine.cook("@OUT = vec4(vec3(u * frame), 1.0);", {"A": img},
                        time_context={"frame": f})
    if len(interp._builtins_lru) != 1:
        fails.append(f"3 cooks at one resolution made {len(interp._builtins_lru)} LRU "
                     "entries — the playhead is churning the coordinate cache (LAT-4)")

    # The TILED path. Found by review, not by this test: _run_default can route a cook
    # through run_tiled (M-4 memory pressure, or ENG-2's OOM rung), which re-enters the
    # interpreter per strip — and the playhead lives on per-execute state, so a strip
    # cooked without it silently reads zero. That is the same freeze as the other four
    # caches, on the one route none of them cover, and it bites exactly the big cooks that
    # need tiling. Driven through run_tiled directly so it does not depend on provoking
    # real memory pressure.
    from TEX_Wrangle.tex_memory import run_tiled
    from TEX_Wrangle.tex_cache import get_cache
    tcode = "@OUT = vec4(vec3(frame), 1.0);"
    prog, tm, _rf, _asg, _pi, ub = get_cache().compile_tex(tcode, {"A": TEXType.VEC3})
    tiled = run_tiled(interp, prog, {"A": make_img(1, 64, 64, 3, seed=1)}, tm, "cpu",
                      0, ["OUT"], ub, "fp32", 4, {"frame": 7.0})
    got = tiled["OUT"][0, 0, 0, 0].item()
    if got != 7.0:
        fails.append(f"the TILED path lost the playhead: frame read {got}, expected 7.0 "
                     "(strips re-enter the interpreter; time_context must reach each one)")

    # fps/time carry independently; an absent key reads 0 (a host with no timeline).
    out = tex_engine.cook("@OUT = vec4(fps, time, 0.0, 1.0);", {"A": img},
                          time_context={"fps": 24.0, "time": 1.5})
    px = out.outputs["OUT"][0, 0, 0]
    if abs(px[0].item() - 24.0) > 1e-5 or abs(px[1].item() - 1.5) > 1e-5:
        fails.append(f"fps/time did not reach the cook: {px[:2].tolist()}")
    z = tex_engine.cook("@OUT = vec4(vec3(frame + fps + time), 1.0);", {"A": img})
    if z.outputs["OUT"][0, 0, 0, 0].item() != 0.0:
        fails.append("a host with no time context must read zeros")

    if fails:
        r.fail("ENG-7 time builtins", "; ".join(fails))
    else:
        r.ok("frame/fps/time track the playhead across cooks (incl. a revisit); absent "
             "context reads 0; the LAT-4 coordinate LRU still hits (1 entry / 3 cooks)")


def test_eng7_time_barred_from_frozen_tiers(r: SubTestResult):
    print("\n--- ENG-7: the caching tiers DECLINE a time-reading program ---")
    # Each of these tiers is keyed by the program fingerprint, which deliberately does not
    # move when a playhead does. They must decline rather than serve a frozen frame — the
    # failure would be a successful cook with a still image, which no error would reveal.
    from TEX_Wrangle.tex_runtime.codegen import try_compile, _reads_time_builtin
    from TEX_Wrangle.tex_runtime.graphed import _capturable
    from TEX_Wrangle import tex_engine
    fails = []
    timed = "@OUT = vec4(vec3(u * frame), 1.0);"
    plain = "@OUT = vec4(vec3(u * 0.5), 1.0);"
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

    def _prog(code):
        p = Parser(Lexer(code).tokenize(), source=code).parse()
        return p, TypeChecker(binding_types=bt, source=code).check(p)

    pt, tmt = _prog(timed)
    pp, tmp = _prog(plain)
    if not _reads_time_builtin(pt):
        fails.append("_reads_time_builtin missed a `frame` read")
    if _reads_time_builtin(pp):
        fails.append("_reads_time_builtin false-positived on a plain program")
    if try_compile(pt, tmt, fingerprint="eng7t") is not None:
        fails.append("codegen COMPILED a time-reading program — it would freeze the "
                     "playhead in _env_cached / the executor closure")
    if try_compile(pp, tmp, fingerprint="eng7p") is None:
        fails.append("codegen declined a PLAIN program — the bar is over-broad")
    if _capturable(pt)[0]:
        fails.append("cuda_graph would capture a time-reading program — a replay would "
                     "re-serve the captured frame")
    if not _capturable(pp)[0]:
        fails.append("cuda_graph declined a plain program — the bar is over-broad")

    # The CLASS fix. Every route below is one `time_context=` away from a frozen playhead,
    # and the interpreter cannot tell a caller that passed None from one that forgot — both
    # read 0.0 and cook a still image. So the two functions the declines land on take it
    # KEYWORD-ONLY with NO DEFAULT: an omission is a TypeError at the call, not a plausible
    # picture. This pins the signature, because the cheap "fix" for the resulting TypeErrors
    # is to hand the parameter a default back, which would re-arm the exact v0.22 bug.
    import inspect as _inspect
    from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute as _coe
    from TEX_Wrangle.tex_runtime.compiled import _plain_execute as _pe
    for _fn in (_coe, _pe):
        _sig = _inspect.signature(_fn)
        _tc = _sig.parameters.get("time_context")
        if _tc is None:
            fails.append(f"{_fn.__name__} lost its time_context parameter")
        elif _tc.kind is not _inspect.Parameter.KEYWORD_ONLY:
            fails.append(f"{_fn.__name__}: time_context must be KEYWORD-ONLY (a positional "
                         f"forward is one argument-order change from silently wrong)")
        elif _tc.default is not _inspect.Parameter.empty:
            fails.append(f"{_fn.__name__}: time_context grew a default ({_tc.default!r}) — a "
                         f"forgotten forward is now a frozen playhead again, not a TypeError")

    # END-TO-END, through the real node, on every tier. The unit assertions above are not
    # enough and this is not a hypothetical: they only proved try_compile/_capturable
    # return None, and a bug hunt found that the DECLINE ITSELF dropped the playhead —
    # the fall-back lands on compiled._plain_execute, INSIDE the tier, not on the engine's
    # _interp_fallback. That froze frame at 0 on the SHIPPED DEFAULT path (an exact
    # stencil routes through _codegen_only_execute at compile_mode="none"). A bar is only
    # worth anything if the path it forces you onto is correct, so cook and look.
    import json as _json
    from TEX_Wrangle.tex_node import TEXWrangleNode as _N
    stencil = ("vec3 s = vec3(0.0); for(int j=-1;j<=1;j++){ for(int i=-1;i<=1;i++){ "
               "s = s + @A[ix+i, iy+j].rgb; } } @OUT = vec4(s/9.0*frame, 1.0);")
    simple = "@OUT = vec4(@A.rgb * frame, 1.0);"
    # The EIGHTH route, and the one the six rows below could not see: execute_compiled has
    # a branch of its own for deeply nested loops that hands off to _codegen_only_execute
    # WITHOUT torch.compile, and v0.22 shipped it without the playhead. Reaching it needs
    # all three of: compile_mode="torch_compile" (to enter execute_compiled at all),
    # post-optimizer loop_depth > _COMPILE_MAX_LOOP_DEPTH (=2), and op_count >=
    # _COMPILE_OP_THRESHOLD (=8) — which is why every existing row missed it: `simple` has
    # no loops and `stencil`'s two are unrolled away (trip count 3 <= _UNROLL_MAX_ITERS=8).
    # Hence 10: the innermost loop must out-trip the unroller to survive as depth 3, and
    # 1000 iterations of frame*0.001 sum back to exactly `frame`, so this asserts the same
    # 0.9 as its siblings. Measured pre-fix: none -> 0.9000, torch_compile -> 0.0000.
    deep = ("float acc = 0.0; "
            "for (int a = 0; a < 10; a = a + 1) { for (int b = 0; b < 10; b = b + 1) { "
            "for (int c = 0; c < 10; c = c + 1) { acc = acc + frame * 0.001; } } } "
            "@OUT = vec4(vec3(acc * @A.r), 1.0);")

    # Pin the ROUTE, not just the value. The row below is only evidence while `deep` still
    # reaches the deep-loop branch, and what puts it there is the post-OPTIMIZER shape — an
    # unroller or DCE change could flatten it, sending the cook down a route that was never
    # broken while the assertion still passed. A green test that stopped testing anything
    # is how this bug survived a suite with seven ENG-7 routes already pinned.
    _dp = Parser(Lexer(deep).tokenize(), source=deep).parse()
    _dtm = TypeChecker(binding_types=bt, source=deep).check(_dp)
    _dopt = optimize(_dp, _dtm)
    from TEX_Wrangle.tex_runtime.compiled import (
        _count_tensor_ops, _max_loop_depth, _COMPILE_OP_THRESHOLD, _COMPILE_MAX_LOOP_DEPTH)
    if _max_loop_depth(_dopt) <= _COMPILE_MAX_LOOP_DEPTH:
        fails.append(f"the deep-nest probe no longer survives the optimizer as a deep nest "
                     f"(loop_depth {_max_loop_depth(_dopt)} <= {_COMPILE_MAX_LOOP_DEPTH}) — "
                     f"it stopped covering execute_compiled's deep-loop route")
    if _count_tensor_ops(_dopt) < _COMPILE_OP_THRESHOLD:
        fails.append(f"the deep-nest probe fell under the compile op threshold "
                     f"({_count_tensor_ops(_dopt)} < {_COMPILE_OP_THRESHOLD}) — it now "
                     f"routes to _plain_execute, not the deep-loop branch it was written for")

    flat = torch.full((1, 16, 16, 3), 1.0)
    for label, code, mode in (("stencil", stencil, "none"), ("simple", simple, "none"),
                              ("simple", simple, "torch_compile"),
                              ("stencil", stencil, "auto"), ("simple", simple, "auto"),
                              ("simple", simple, "cuda_graph"),
                              ("deep-nest", deep, "none"),
                              ("deep-nest", deep, "torch_compile"),
                              ("deep-nest", deep, "auto")):
        out = _N.execute(code=code, device="cpu", compile_mode=mode, precision="fp32",
                         A=flat, _tex_time=_json.dumps({"frame": 0.9}))
        vals = out if isinstance(out, tuple) else getattr(out, "args", out)
        got = vals[0][0, 8, 8, 0].item()
        if abs(got - 0.9) > 1e-3:
            fails.append(f"{label}/compile_mode={mode}: frame reached the pixels as "
                         f"{got:.3f}, expected 0.9 (a tier route dropped the playhead)")

    # The BUILTIN's own value is fp32 and exact (fp16 holds integers exactly only to 2048).
    # This pins the construction, and NOT more than that — see the honest limit below.
    for f in (2049.0, 70000.0):
        got = tex_engine.cook("@OUT = vec4(vec3(frame), 1.0);",
                              {"A": make_img(1, 8, 8, 3, seed=3)}, precision="fp16",
                              time_context={"frame": f}).outputs["OUT"][0, 0, 0, 0].item()
        if got != f:
            fails.append(f"the playhead builtin itself quantised: frame {f} read as {got}")

    # THE GATE is what protects users, not the fp32 construction — pin that, because the
    # obvious reading of the above is wrong. `frame` is a 0-dim tensor, and torch does not
    # let a 0-dim operand lift the result dtype, so `@A.rgb * frame` under fp16 is fp16 and
    # 2049 rounds back to 2048 at the multiply. (`fi` escapes only because it is [B,1,1] —
    # dimensioned — which is a shape these cannot borrow: it mis-aligns against [B,H,W,C].)
    # So precision="auto" must DECLINE any program mixing a playhead into image lineage.
    # Shipped without this, `sin(@A.r * frame)` was ACCEPTED for fp16 and at frame=500
    # measured maxdiff 0.2443 — 63x the 3.9e-3 budget, every pixel finite so the C2 net
    # never fired. Invariant #10: "Any accepted program exceeding 3.9e-3 is a gate bug."
    # Iterate the AUTHORITATIVE name set, not a literal tuple: a 4th host-time builtin added
    # to _TIME_BUILTIN_NAMES must be forced into _BUILTIN_MAG too, or the gate goes blind to
    # it — the exact invariant-#10 hole this whole block exists to close. Federating the
    # canary to the source set is what makes "add a time builtin, forget the magnitude"
    # a red test instead of a silent gate bug (mirrors type_checker's keep-in-step comment).
    from TEX_Wrangle.tex_runtime.precision_policy import _BUILTIN_MAG
    from TEX_Wrangle.tex_runtime.interpreter import _TIME_BUILTIN_NAMES
    for name in _TIME_BUILTIN_NAMES:
        if name not in _BUILTIN_MAG:
            fails.append(f"{name} is in _TIME_BUILTIN_NAMES but not _BUILTIN_MAG — the C1 "
                         "amplification gate cannot see it and will accept fp16 for it")
    # frame/time are UNBOUNDED (a playhead grows all day), so their magnitude must be inf —
    # not merely "large". A FINITE pin (65504) is a bug the bug hunt found: the analysis
    # const-folds a scale constant, so `frame*0.00001` folds under the hazard threshold and
    # evades. Only inf survives `inf * tiny_const`. fps is genuinely bounded, so it is exempt.
    import math as _math
    for nm in _TIME_BUILTIN_NAMES - {"fps"}:
        if not _math.isinf(_BUILTIN_MAG.get(nm, 0)):
            fails.append(f"{nm} magnitude {_BUILTIN_MAG.get(nm)} is finite — a scale-down "
                         "constant folds it under the hazard threshold and evades the gate; "
                         "an unbounded playhead needs inf")
    # ...and the evasion itself: a tiny scale constant on the playhead must STILL decline.
    for src in ("float sf = frame*0.00001; @OUT = vec4(vec3(sin(@A.r*sf)), 1.0);",
                "@OUT = vec4(vec3(sin(@A.r * (frame/60000.0))), 1.0);",
                "@OUT = vec4(@A.rgb + vec3(time*0.00001), 1.0);"):
        if _resolve_auto(src)[0] != "fp32":
            fails.append(f"auto gate ACCEPTED fp16 for a SCALED playhead: {src!r} — a finite "
                         "magnitude pin lets frame*tiny_const evade (ships wrong at large frame)")
    # The legit counterpart — a BOUNDED function of the playhead — must stay fp16, or the inf
    # pin is over-broad (sin caps magnitude regardless of its argument).
    if _resolve_auto("@OUT = vec4(@A.rgb + vec3(sin(frame) * 0.1), 1.0);")[0] != "fp16":
        fails.append("the gate declined @A.rgb + sin(frame)*0.1 — a bounded function of the "
                     "playhead is safe; the inf pin must not decline it")
    for src in ("@OUT = vec4(vec3(sin(@A.r * frame)), 1.0);",
                "@OUT = vec4(vec3(sin(@A.r * time)), 1.0);",
                "@OUT = vec4(@A.rgb * frame, 1.0);"):
        p, why = _resolve_auto(src)
        if p != "fp32":
            fails.append(f"auto gate ACCEPTED fp16 for {src!r} ({why}) — a playhead is "
                         "unbounded and amplifies image lineage without bound")
    # ...and the gate must not have become a blanket "decline anything" (a canary that
    # declines everything pins nothing).
    p, why = _resolve_auto("@OUT = vec4(1.0 - @A.rgb, 1.0);")
    if p != "fp16":
        fails.append(f"the gate declined a smooth pointwise program ({why}) — the "
                     "frame/fps/time entries are over-broad")

    if fails:
        r.fail("ENG-7 tier bars", "; ".join(fails))
    else:
        r.ok("codegen + cuda_graph decline a time-reading program (and only that one); "
             "all 9 tier routes deliver the playhead to the pixels end-to-end (incl. "
             "execute_compiled's deep-loop branch, on a nest pinned to survive the "
             "optimizer); time_context is keyword-only + required on the two functions "
             "the declines land on, so a forgotten forward is a TypeError; the "
             "builtin is exact fp32; the auto gate DECLINES fp16 for any program mixing "
             "a playhead into image lineage (that gate, not the fp32, is the protection)")


# ── ENG-2: standalone memory authority + the OOM ladder ───────────────────

def test_eng2_null_host_measures_vram(r: SubTestResult):
    print("\n--- ENG-2: a host-less cook can still see free VRAM ---")
    from TEX_Wrangle.tex_runtime.host import NullHostServices, _cuda_free_memory
    fails = []
    null = NullHostServices()
    if not torch.cuda.is_available():
        if null.get_free_memory("cuda") is not None or _cuda_free_memory("cuda") is not None:
            fails.append("no CUDA present but a VRAM figure was reported")
    else:
        free = null.get_free_memory("cuda")
        total = torch.cuda.get_device_properties(0).total_memory
        if not isinstance(free, float):
            fails.append(f"expected a float, got {free!r}")
        elif not (0 < free <= total):
            fails.append(f"free VRAM {free} outside (0, {total}]")
        # The allocator-slack correction is the point: memory torch has RESERVED but not
        # allocated is free to TEX, though the driver counts it used. Without it a
        # standalone cook under-reads its budget and tiles for no reason.
        blob = torch.empty(int(64e6 // 4), dtype=torch.float32, device="cuda")
        del blob                       # freed to torch's cache, NOT back to the driver
        driver_only, _ = torch.cuda.mem_get_info(0)
        with_slack = null.get_free_memory("cuda")
        if with_slack < driver_only:
            fails.append(f"slack correction went backwards: {with_slack} < {driver_only}")
    if null.get_free_memory("cpu") is not None:
        fails.append("get_free_memory('cpu') should be None (this probe is VRAM-only)")

    if fails:
        r.fail("ENG-2 null host VRAM", "; ".join(fails))
    else:
        r.ok("NullHostServices measures free VRAM (driver + allocator slack), None off "
             "CUDA — preflight/tiling/retry now work with no host")


def test_eng2_oom_ladder(r: SubTestResult):
    print("\n--- ENG-2: the engine's OOM ladder recovers, and stays additive ---")
    from TEX_Wrangle import tex_engine
    import TEX_Wrangle.tex_memory as M
    oom_t = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_t is None:
        r.skip("ENG-2 ladder", "no OutOfMemoryError type on this torch")
        return
    fails = []
    img = make_img(1, 8, 8, 3, seed=2)
    code = "@OUT = vec4(@A.rgb * 1.1, 1.0);"

    # (a) a NON-OOM error must propagate untouched — the ladder must not become a
    #     catch-all that turns real bugs into retries.
    orig_tier = tex_engine._run_tier
    tex_engine._run_tier = lambda ctx, t: (_ for _ in ()).throw(ValueError("not an oom"))
    try:
        tex_engine.cook(code, {"A": img})
        fails.append("a ValueError was swallowed by the OOM ladder")
    except ValueError:
        pass
    except Exception as e:
        fails.append(f"a ValueError became {type(e).__name__}")
    finally:
        tex_engine._run_tier = orig_tier

    # (b) an unrecoverable OOM must RE-RAISE THE ORIGINAL, so ComfyUI's own ladder
    #     (unload_all_models + prompt retry) still fires. This is the additive contract.
    the_oom = oom_t("CUDA out of memory")
    calls = {"n": 0, "freed": 0}
    orig_free = M.free_tensor_caches
    M.free_tensor_caches = lambda: calls.__setitem__("freed", calls["freed"] + 1)

    def _always_oom(ctx, t):
        calls["n"] += 1
        raise the_oom
    tex_engine._run_tier = _always_oom
    try:
        tex_engine.cook(code, {"A": img})       # cpu: no tiled rung -> straight re-raise
        fails.append("an unrecoverable OOM did not propagate")
    except BaseException as e:
        if e is not the_oom:
            fails.append(f"OOM was re-wrapped as {type(e).__name__} — ComfyUI's is_oom "
                         "check would miss it and its recovery would never run")
    finally:
        tex_engine._run_tier = orig_tier
        M.free_tensor_caches = orig_free
    if calls["freed"] < 1:
        fails.append("the ladder did not drop TEX's tensor caches before giving up")

    # (c) the RECOVERY rung, for real: a tile-safe CUDA cook OOMs on the whole-image
    #     attempt and the ladder re-cooks it in strips. Asserting only that "an OOM
    #     re-raises" would leave the ladder's actual claim — that it turns a failure into
    #     a picture — completely untested, and it would pass with the rung deleted.
    recovered = "skipped (no CUDA)"
    if torch.cuda.is_available():
        gimg = make_img(1, 256, 256, 3, seed=5).cuda()
        state = {"n": 0}

        def _oom_first(ctx, t):
            state["n"] += 1
            if state["n"] == 1:
                raise oom_t("CUDA out of memory")
            return orig_tier(ctx, t)
        reference = tex_engine.cook(code, {"A": gimg}, device_mode="cuda").outputs["OUT"]
        tex_engine._run_tier = _oom_first
        try:
            out = tex_engine.cook(code, {"A": gimg}, device_mode="cuda").outputs["OUT"]
            # Recovering is only half of it: M-4 strips are seam-exact, so a recovered
            # cook must be BIT-IDENTICAL to the whole-image one. A retry that quietly
            # returns different pixels would be worse than the OOM it replaced.
            if not torch.equal(out, reference):
                d = (out.float() - reference.float()).abs().max().item()
                fails.append(f"the tiled retry diverged from the untiled cook "
                             f"(maxdiff {d:.3e}) — strips are supposed to be seam-exact")
            recovered = f"recovered in strips after 1 OOM, bit-identical {tuple(out.shape)}"
        except BaseException as e:
            fails.append(f"the tiled rung did not recover a tile-safe CUDA cook: "
                         f"{type(e).__name__}: {e}")
            recovered = "FAILED"
        finally:
            tex_engine._run_tier = orig_tier

    if fails:
        r.fail("ENG-2 OOM ladder", "; ".join(fails))
    else:
        r.ok("non-OOM errors propagate untouched; an unrecoverable OOM drops TEX caches "
             f"then re-raises THE ORIGINAL (ComfyUI's ladder still fires); {recovered}")


def test_eng1_cook_outputs_do_not_alias_inputs(r: SubTestResult):
    print("\n--- ENG-1/ENG-3: a cooked output never aliases an input binding ---")
    # `@OUT = @A;` binds the output name straight to the input tensor, so the "result" IS
    # the caller's buffer. The ComfyUI node never noticed: its egress clamp materializes a
    # fresh tensor. ENG-3's `engine` profile removed the clamp — and with it that
    # accidental copy — so a host recycling frame buffers would have its input silently
    # rewritten by its own output. Asserted on STORAGE: a reshape hands back a new object
    # over the same storage, so `is` misses the real cases.
    from TEX_Wrangle import tex_engine
    from TEX_Wrangle.tex_marshalling import (prepare_output, set_egress_profile,
                                             get_egress_profile)
    fails = []

    def _storage(t):
        return t.untyped_storage().data_ptr()

    # Under the ENGINE profile nothing downstream will copy, so the engine must.
    # `.a` is the case that matters and the one a data_ptr()-keyed guard misses: it is a
    # view at offset 3, so its FIRST-ELEMENT address differs from the input's while the
    # BUFFER is the same. `.rgb` starts at offset 0 and is caught either way — which is
    # exactly what made the broken guard look like it worked.
    try:
        set_egress_profile("engine")
        for code, out_name in (("@OUT = @A;", "OUT"),            # literal identity
                               ("@OUT = @A * 1.0;", "OUT"),      # folds to identity
                               ("@X = @A.a;", "X"),              # VIEW at a NON-ZERO offset
                               ("@X = @A.gb;", "X")):            # ditto, 2ch
            img = make_img(1, 16, 16, 4, seed=11)
            before = img.clone()
            out = tex_engine.cook(code, {"A": img}).outputs[out_name]
            if _storage(out) == _storage(img):
                fails.append(f"{code!r}: the output shares the input's BUFFER")
                continue
            out.add_(5.0)          # prove it: a host writing its own output
            if not torch.equal(img, before):
                fails.append(f"{code!r}: writing the output corrupted the caller's input")
        # A genuinely computed output must NOT be cloned, or every cook pays a full-frame
        # copy it does not need.
        img = make_img(1, 16, 16, 3, seed=12)
        res = tex_engine.cook("@OUT = vec4(@A.rgb + 1.0, 1.0);", {"A": img})
        if _storage(res.outputs["OUT"]) == _storage(img):
            fails.append("a computed output aliased the input (unexpected)")

        # THE NODE, under this same profile. The node skips the engine's clone because its
        # own egress clamp materializes — but 'engine' is the profile that REMOVES the
        # clamp, and the node pins no profile, so under it that proof evaporates and the
        # skip must not. A node hardcoding disown=False passes every other assertion in
        # this test and hands ComfyUI a live view of the caller's buffer right here.
        from TEX_Wrangle.tex_node import TEXWrangleNode as _N2
        img = make_img(1, 16, 16, 3, seed=19)
        before = img.clone()
        out = _N2.execute(code="@OUT = @A;", device="cpu", compile_mode="none",
                          precision="fp32", A=img)
        got = (out if isinstance(out, tuple) else getattr(out, "args", out))[0]
        if _storage(got) == _storage(img):
            fails.append("under the 'engine' profile the NODE handed back the input's "
                         "BUFFER — its disown skip assumes a clamp this profile removes")
        got.add_(5.0)
        if not torch.equal(img, before):
            fails.append("under the 'engine' profile, writing the node's output corrupted "
                         "the caller's input")
    finally:
        set_egress_profile("comfy")

    # THE DEFAULT PATH — and the reason the arm above was not evidence. It set the profile
    # first, so it only ever tested a host that had opted in. tex_api.py:35 promises
    # "cook() guarantees its outputs do not alias your input bindings" with no precondition,
    # and a host doing literally what that says never calls set_egress_profile at all: it
    # gets 'comfy', which used to make the disown a no-op and hand back the caller's own
    # buffer. Zero in-tree callers were affected (the node clamps, the CLI pins 'comfy'),
    # which is exactly why nothing went red — a false guarantee on a Tier-1 public surface
    # is invisible from inside the tree. NOTHING is set here, deliberately: this is the
    # profile a fresh import has, and the guarantee is now a property of the CALL.
    if get_egress_profile() != "comfy":
        fails.append("the default egress profile moved; this arm no longer tests the "
                     "default path (which is the whole point of it)")
    for code, out_name in (("@OUT = @A;", "OUT"), ("@OUT = @A * 1.0;", "OUT"),
                           ("@X = @A.a;", "X"), ("@X = @A.gb;", "X")):
        img = make_img(1, 16, 16, 4, seed=17)
        before = img.clone()
        out = tex_engine.cook(code, {"A": img}).outputs[out_name]
        if _storage(out) == _storage(img):
            fails.append(f"{code!r}: cook() aliased the caller's buffer under the DEFAULT "
                         f"profile — the guarantee tex_api publishes is false")
            continue
        out.add_(5.0)              # a host recycling its frame buffers, per the docstring
        if not torch.equal(img, before):
            fails.append(f"{code!r}: writing cook()'s output corrupted the caller's input")

    # ...and the opt-out is real, or the node pays a full-frame copy it does not need.
    # `disown=False` is the ONE thing that may skip the clone, and only a caller that can
    # prove its own egress materializes may pass it (invariant #7: +0.236 ms at 2048²
    # passthrough, +118% of egress, on the default path, in a release budgeted at
    # +1.3 us/cook). Pinned as an ALIAS: this asserts the clone was genuinely skipped, not
    # that some other copy happened to be absent.
    img = make_img(1, 16, 16, 4, seed=18)
    if _storage(tex_engine.cook("@OUT = @A;", {"A": img}, disown=False).outputs["OUT"]) \
            != _storage(img):
        fails.append("disown=False still cloned — the node's egress-clamp opt-out is gone "
                     "and every ComfyUI passthrough cook now pays for it (invariant #7)")
    from TEX_Wrangle import tex_engine as _eng_mod
    import inspect as _insp
    if _insp.signature(_eng_mod.prepare).parameters["disown"].default is not True:
        fails.append("prepare(disown=) no longer defaults True — cook()'s published "
                     "guarantee is opt-IN again, which is how this shipped broken")

    # What the NODE hands ComfyUI must still be un-aliased — via the clamp, now that the
    # node is the caller passing disown=False. This is the assertion the skip rests on.
    from TEX_Wrangle.tex_node import TEXWrangleNode
    img = make_img(1, 16, 16, 3, seed=15)
    before = img.clone()
    out = TEXWrangleNode.execute(code="@OUT = @A;", device="cpu", compile_mode="none",
                                 precision="fp32", A=img)
    node_out = (out if isinstance(out, tuple) else getattr(out, "args", out))[0]
    if _storage(node_out) == _storage(img):
        fails.append("the NODE handed ComfyUI a tensor sharing the input's buffer — the "
                     "comfy clamp is what the engine's skip relies on")
    node_out.add_(5.0)
    if not torch.equal(img, before):
        fails.append("writing the node's output corrupted the caller's input")
    raw = make_img(1, 8, 8, 4, seed=13)
    if _storage(prepare_output(raw, "IMAGE", profile="comfy")) == _storage(raw):
        fails.append("the comfy profile stopped materializing — its clamp is the copy "
                     "every ComfyUI workflow has always relied on, and the engine now "
                     "skips its own clone because of it")

    if fails:
        r.fail("ENG-1 output ownership", "; ".join(fails))
    else:
        r.ok("no cooked output shares an input's BUFFER (identity, const-folded identity, "
             "offset views .a/.gb) under the 'engine' profile AND under the DEFAULT one, "
             "which is the guarantee tex_api actually publishes; computed outputs are not "
             "cloned; ownership rides the CALL (prepare(disown=) defaults True) so the one "
             "caller that provably materializes — the node, via its clamp — is the one that "
             "skips the clone (invariant #7)")


def test_eng1_fp16_compiled_tier_clamp(r: SubTestResult):
    print("\n--- ENG-1: fp16 is clamped to fp32 on the compiling tiers (engine policy) ---")
    # Hoisted out of tex_node in v0.22 so `cook()` obeys it too — the engine's public entry
    # previously fed fp16 into tiers documented as unvalidated for it. Shipped untested:
    # review deleted the clamp and ZERO of the suite's 1894 sub-tests died.
    from TEX_Wrangle import tex_engine
    fails = []
    img = make_img(1, 8, 8, 3, seed=14)
    code = "@OUT = vec4(@A.rgb, 1.0);"
    for mode in ("torch_compile", "auto", "cuda_graph"):
        got = tex_engine.prepare(code, {"A": img}, precision="fp16",
                                 compile_mode=mode).ctx.eff_precision
        if got != "fp32":
            fails.append(f"compile_mode={mode!r} + precision='fp16' ran at {got!r} — the "
                         "compiled tiers bake precision into their keys and are not "
                         "validated for fp16")
    kept = tex_engine.prepare(code, {"A": img}, precision="fp16",
                              compile_mode="none").ctx.eff_precision
    if kept != "fp16":
        fails.append(f"compile_mode='none' + precision='fp16' was clamped to {kept!r} — "
                     "the clamp is over-broad; expert fp16 on the interpreter is the "
                     "one mode where it IS validated")
    if fails:
        r.fail("ENG-1 fp16 clamp", "; ".join(fails))
    else:
        r.ok("fp16 clamps to fp32 on torch_compile/auto/cuda_graph and survives on the "
             "interpreter — enforced by the ENGINE, so cook() obeys it too")


# ── ENG-1: the seam ───────────────────────────────────────────────────────

def test_eng1_engine_cooks_without_the_node(r: SubTestResult):
    print("\n--- ENG-1: tex_engine cooks with no ComfyUI node in the call path ---")
    # The point of the module: before v0.22 the ONLY way to reach tier selection,
    # fallbacks, tiling and the auto-precision gate was a ComfyUI v3 node classmethod.
    from TEX_Wrangle import tex_engine
    fails = []
    img = make_img(1, 16, 16, 3, seed=4)
    try:
        res = tex_engine.cook("@OUT = vec4(@A.rgb * 1.5, 1.0);", {"A": img})
        # RAW: the engine does not apply a host egress profile, so 1.5x survives.
        if res.outputs["OUT"].shape[-1] != 4:
            fails.append("engine cook dropped alpha (it must return raw)")
        if res.outputs["OUT"].max().item() <= 1.0:
            fails.append("engine cook clamped — cook() must return RAW tensors")
        if res.output_names != ["OUT"] or res.precision != "fp32":
            fails.append(f"unexpected result meta: {res.output_names}/{res.precision}")
        if "A" not in res.binding_names:
            fails.append(f"binding_names missing the input: {res.binding_names}")
    except Exception as e:
        fails.append(f"cook() raised {type(e).__name__}: {e}")

    # prepare() is pure planning: it must pick a tier without executing anything.
    try:
        plan = tex_engine.prepare("@OUT = vec4(@A.rgb, 1.0);", {"A": img},
                                  compile_mode="none")
        if plan.tier_id != "default":
            fails.append(f"tier_id {plan.tier_id!r}, expected 'default'")
        if plan.fused_chain is not False:
            fails.append("a single program reported fused_chain=True")
        out = tex_engine.run(plan)
        if not torch.isfinite(out.outputs["OUT"]).all():
            fails.append("run(plan) produced non-finite output")
    except Exception as e:
        fails.append(f"prepare/run raised {type(e).__name__}: {e}")

    if fails:
        r.fail("ENG-1 engine seam", "; ".join(fails))
    else:
        r.ok("cook() runs the full engine host-agnostically and returns RAW outputs; "
             "prepare() plans, run() executes")


def test_eng1_node_is_a_marshaller(r: SubTestResult):
    print("\n--- ENG-1: tex_node holds no cook logic (S-1 boundary) ---")
    # A lint, not a behaviour test: the engine symbols must not have grown a second home
    # back in the adapter. This is the drift that made ENG-1 necessary in the first place.
    import TEX_Wrangle.tex_node as TN
    from TEX_Wrangle import tex_engine
    fails = []
    moved = ("select_tier", "_run_tier", "_interp_fallback", "_fp16_finiteness_net",
             "_tile_plan", "_preflight_memory", "_run_default", "_run_torch_compile",
             "_run_auto", "_run_cuda_graph", "_TIER_METHOD", "ExecContext",
             "_AUTO_DECISION", "_get_interpreter", "_apply_cpu_threads_env")
    for name in moved:
        if hasattr(TN, name) or hasattr(getattr(TN, "TEXWrangleNode", object), name):
            fails.append(f"tex_node still exposes {name} (should be engine-only)")
        if not hasattr(tex_engine, name):
            fails.append(f"tex_engine is missing {name}")
    # The CLI must not reach the engine THROUGH the node (roadmap v0.22 exit criterion).
    # Checked on the IMPORT GRAPH, not the raw text: a prose mention of the old call site
    # is exactly the institutional memory this repo keeps on purpose, and a substring
    # lint would forbid explaining the change it is policing.
    import ast
    cli_ast = ast.parse((_PKG / "tex_cli.py").read_text(encoding="utf-8"))
    cli_imports = set()
    for node in ast.walk(cli_ast):
        if isinstance(node, ast.ImportFrom):
            # `from .tex_node import X` -> module="tex_node"; but `from . import tex_engine`
            # has module=None and carries the module name in `names` — the form tex_cli
            # actually uses, so missing it would make this lint silently vacuous.
            if node.module:
                cli_imports.add(node.module.lstrip("."))
            else:
                cli_imports.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            cli_imports.update(a.name for a in node.names)
    if any(m.split(".")[0] == "tex_node" for m in cli_imports):
        fails.append(f"tex_cli imports the ComfyUI node ({sorted(cli_imports)}) — "
                     "it must cook via engine.cook")
    if not any(m.split(".")[0] == "tex_engine" for m in cli_imports):
        fails.append("tex_cli does not import tex_engine")

    if fails:
        r.fail("ENG-1 S-1 boundary", "; ".join(fails))
    else:
        r.ok(f"all {len(moved)} cook symbols live only in tex_engine; tex_cli cooks via "
             "the engine, not the node")
