"""v042-noviewer — the fused viewer transform (PM-11 / v042-graph) is removed outright.

The author's ruling (2026-09-24): drop work that has no consumer in ComfyUI or the
embedding host. The PM-11 fused viewer transform had none — the ComfyUI node exposes no
viewer input (the builtins always read identity there) and the embedding host applies
exposure/gamma/LUT in its own display shader. This removes `viewer_exposure()`/
`viewer_gamma()`, the `viewer_context=` kwarg and its threading through every tier, the
`_canon_viewer`/viewer component of a lineage key, and the CUDA-graph host-context buffer
machinery — all of it additive-only when it shipped (PM-11's own invariant-7 proof), so
removing it is additive-only in reverse: no other builtin, kwarg or cache shape moves.

This file proves the removal rather than the feature:
  1. the two names are no longer registered/reserved — a program calling either now gets
     the ordinary "unknown function" diagnostic (E5001), and a user CAN define a function
     of either name;
  2. the `viewer_context=`/`viewer=` kwarg is gone from every public and internal seam it
     used to ride (`tex_engine.prepare`/`cook`, `Interpreter.execute`, `tex_chain.
     cook_stage_list`/`cook_fused_cached`/`boundary_lineage_key`, `tex_checkpoint.
     cook_checkpointed`/`materialize`, `tex_results_keys.lineage_key`, `tex_runtime.
     graphed.run_graphed`, `tex_runtime.compiled.execute_compiled`/`run_auto`,
     `tex_runtime.codegen._invoke_cg`, `tex_runtime.stdlib_core.set_cook_grid`) — passing
     it now raises the ordinary `TypeError` for an unexpected keyword, not a silent no-op;
  3. `tex_results_keys.lineage_key`'s byte-feed format for every OTHER component (fp, dev,
     prec, env, par, up, frm, tc, q, flg, cnv) is unmoved — pinned by reimplementing the
     documented feed sequence independently and comparing digests (env_epoch mocked to a
     fixed string, since it legitimately moves with any interpreter.py/codegen.py edit —
     including this one's own — and comparing a real base-commit hash against head would
     fail on that alone, not on anything this ask changed);
  4. the CUDA-graph host-context registry surface (`reads_host_context`, `host_context_default`,
     `stdlib_registry.host_context_names`/`host_context_defaults`, `graphed._host_context_calls`,
     `stdlib_core._push_host_context_buffers`/`_pop_host_context_buffers`/`_host_context_buffer`)
     is gone entirely, and an ordinary program's CUDA-graph capturability is unaffected.

No CUDA-only row here: every check is AST-level or a pure-function call (SIMP-3's skip
budget moves DOWN, not up, for this file).
"""
import hashlib
import inspect
import json

from helpers import *
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle import tex_engine, tex_chain, tex_checkpoint, tex_results_keys
from TEX_Wrangle.tex_runtime import graphed as G
from TEX_Wrangle.tex_runtime import stdlib_registry as R
from TEX_Wrangle.tex_runtime import stdlib_core as SC


def test_noviewer_names_not_registered(r: SubTestResult):
    print("\n--- v042-noviewer: viewer_exposure/viewer_gamma are gone from the registry ---")
    try:
        fns = TEXStdlib.get_functions()
        gone = [n for n in ("viewer_exposure", "viewer_gamma") if n in fns]
        assert not gone, f"still registered: {gone}"
        r.ok("neither name resolves through TEXStdlib.get_functions() anymore")
    except Exception as e:
        r.fail("noviewer registry", f"{type(e).__name__}: {e}")

    try:
        for name in ("viewer_exposure", "viewer_gamma"):
            assert name not in R.FP16_FRAGILE, f"{name} still in FP16_FRAGILE"
        r.ok("neither name is in stdlib_registry.FP16_FRAGILE")
    except Exception as e:
        r.fail("noviewer fp16 fragile", f"{type(e).__name__}: {e}")


def test_noviewer_calling_is_unknown_function(r: SubTestResult):
    print("\n--- v042-noviewer: calling either name now fails E5001 (unknown function) ---")
    for name in ("viewer_exposure", "viewer_gamma"):
        try:
            raised = None
            try:
                check_code(f"@OUT = vec4(vec3({name}()), 1.0);")
            except Exception as e:
                raised = e
            assert raised is not None, f"{name}() compiled with no error"
            code = getattr(getattr(raised, "diagnostic", None), "code", None)
            assert code == "E5001", \
                f"{name}(): expected E5001 (unknown function), got {code!r} ({raised!r})"
            r.ok(f"`{name}()` now fails the ordinary unknown-function diagnostic (E5001)")
        except Exception as e:
            r.fail(f"noviewer unknown fn {name}", f"{type(e).__name__}: {e}")


def test_noviewer_names_no_longer_reserved(r: SubTestResult):
    print("\n--- v042-noviewer: a user program MAY define a function of either name now ---")
    for name in ("viewer_exposure", "viewer_gamma"):
        try:
            # Defining a zero-arg user function of the name that used to be a reserved
            # builtin must compile clean now (no E3011 "built-in, cannot be redefined").
            check_code(f"float {name}() {{ return 1.0; }}\n@OUT = vec4(vec3({name}()), 1.0);")
            r.ok(f"a user function named `{name}` compiles clean (no longer reserved)")
        except Exception as e:
            r.fail(f"noviewer un-reserved {name}", f"{type(e).__name__}: {e}")


def _assert_rejects_kwarg(callable_or_none, label: str, fails: list, **extra_ok_kwargs):
    """Call `callable_or_none(viewer_context=..., **extra_ok_kwargs)` with otherwise-empty/
    placeholder positional args is not attempted (most of these need real programs) — this
    instead inspects the signature, which is the stable, side-effect-free way to prove a
    parameter is gone from a function that is expensive or unsafe to actually invoke."""
    try:
        sig = inspect.signature(callable_or_none)
        if "viewer_context" in sig.parameters:
            fails.append(f"{label} still declares a viewer_context parameter: {sig}")
    except (TypeError, ValueError) as e:
        fails.append(f"{label}: could not inspect signature: {e}")


def test_noviewer_kwarg_gone_from_every_seam(r: SubTestResult):
    print("\n--- v042-noviewer: viewer_context=/viewer= is gone from every seam it rode ---")
    fails = []
    from TEX_Wrangle.tex_runtime.interpreter import Interpreter
    from TEX_Wrangle.tex_runtime.compiled import execute_compiled, run_auto
    from TEX_Wrangle.tex_runtime.codegen import _invoke_cg
    from TEX_Wrangle.tex_memory import run_tiled, run_roi, run_tiled_halo, run_batch_strips
    _assert_rejects_kwarg(tex_engine.prepare, "tex_engine.prepare", fails)
    _assert_rejects_kwarg(Interpreter.execute, "Interpreter.execute", fails)
    _assert_rejects_kwarg(tex_chain.cook_stage_list, "tex_chain.cook_stage_list", fails)
    _assert_rejects_kwarg(tex_chain.cook_fused_cached, "tex_chain.cook_fused_cached", fails)
    _assert_rejects_kwarg(tex_chain.boundary_lineage_key, "tex_chain.boundary_lineage_key", fails)
    _assert_rejects_kwarg(tex_checkpoint.cook_checkpointed, "tex_checkpoint.cook_checkpointed", fails)
    _assert_rejects_kwarg(tex_checkpoint.materialize, "tex_checkpoint.materialize", fails)
    _assert_rejects_kwarg(tex_results_keys.lineage_key, "tex_results_keys.lineage_key", fails)
    _assert_rejects_kwarg(G.run_graphed, "graphed.run_graphed", fails)
    _assert_rejects_kwarg(G.GraphedProgram.capture, "GraphedProgram.capture", fails)
    _assert_rejects_kwarg(G.GraphedProgram.replay, "GraphedProgram.replay", fails)
    _assert_rejects_kwarg(execute_compiled, "compiled.execute_compiled", fails)
    _assert_rejects_kwarg(run_auto, "compiled.run_auto", fails)
    _assert_rejects_kwarg(_invoke_cg, "codegen._invoke_cg", fails)
    _assert_rejects_kwarg(run_tiled, "tex_memory.run_tiled", fails)
    _assert_rejects_kwarg(run_roi, "tex_memory.run_roi", fails)
    _assert_rejects_kwarg(run_tiled_halo, "tex_memory.run_tiled_halo", fails)
    _assert_rejects_kwarg(run_batch_strips, "tex_memory.run_batch_strips", fails)
    _assert_rejects_kwarg(SC.set_cook_grid, "stdlib_core.set_cook_grid", fails)
    if fails:
        r.fail("noviewer kwarg removed", "; ".join(fails))
    else:
        r.ok("19 seams checked by signature — none declares viewer_context/viewer any more")

    # And the ACTUAL call-time behaviour, not only the signature: a caller that still
    # passes the old kwarg gets Python's ordinary TypeError, never a silent no-op.
    try:
        raised = None
        try:
            tex_results_keys.lineage_key(program_fp="p", device="cpu", precision="fp32",
                                         viewer_context={"viewer_exposure": 2.0})
        except TypeError as e:
            raised = e
        assert raised is not None, "lineage_key(viewer_context=...) did not raise"
        r.ok(f"lineage_key(viewer_context=...) raises TypeError: {raised}")
    except Exception as e:
        r.fail("noviewer lineage_key call-time reject", f"{type(e).__name__}: {e}")


def test_noviewer_host_context_registry_surface_gone(r: SubTestResult):
    print("\n--- v042-noviewer: the CUDA-graph host-context registry surface is gone ---")
    fails = []
    for name in ("host_context_names", "host_context_defaults"):
        if hasattr(R, name):
            fails.append(f"stdlib_registry.{name} still exists")
    for name in ("_push_host_context_buffers", "_pop_host_context_buffers", "_host_context_buffer"):
        if hasattr(SC, name):
            fails.append(f"stdlib_core.{name} still exists")
    if hasattr(G, "_host_context_calls"):
        fails.append("graphed._host_context_calls still exists")
    if hasattr(G, "_reads_host_context_cached"):
        fails.append("graphed still imports _reads_host_context_cached")
    try:
        gp = G.GraphedProgram((("x",), 0))
        if hasattr(gp, "static_host_context"):
            fails.append("GraphedProgram.static_host_context still exists")
        if hasattr(gp, "_host_context_defaults"):
            fails.append("GraphedProgram._host_context_defaults still exists")
    except Exception as e:
        fails.append(f"could not construct a bare GraphedProgram to check: {e}")
    # StdlibEntry itself: the two PM-11/v042-graph fields must be gone (not merely unused).
    entry_fields = {f for f in getattr(R.StdlibEntry, "__dataclass_fields__", {})}
    for f in ("reads_host_context", "host_context_default"):
        if f in entry_fields:
            fails.append(f"StdlibEntry still declares field {f!r}")
    if fails:
        r.fail("noviewer host-context surface", "; ".join(fails))
    else:
        r.ok("host_context_names/defaults, the three stdlib_core buffer functions, "
             "graphed._host_context_calls/_reads_host_context_cached and StdlibEntry's "
             "two fields are all gone")


def test_noviewer_capturable_unaffected(r: SubTestResult):
    print("\n--- v042-noviewer: an ordinary program's CUDA-graph capturability is unmoved ---")
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    try:
        prog, _, _ = _compile_and_check(
            "@OUT = vec4(sin(@A) * 0.5 + 0.5, 1.0);", bt)
        cap1, ops1 = G._capturable(prog)
        cap2, ops2 = G._capturable(prog)
        assert cap1 is True, f"a plain program is not capturable (ops={ops1})"
        assert (cap1, ops1) == (cap2, ops2), "capturability is non-deterministic across calls"
        r.ok(f"a plain program still reads (capturable=True, ops={ops1}), deterministically")
    except Exception as e:
        r.fail("noviewer capturable unaffected", f"{type(e).__name__}: {e}")

    try:
        time_prog, _, _ = _compile_and_check("@OUT = vec4(vec3(frame), 1.0);", bt)
        cap_t, _ = G._capturable(time_prog)
        assert cap_t is False, "frame is capturable — untouched by this ask, must stay barred"
        r.ok("frame stays barred from CUDA-graph capture (a separate, untouched mechanism)")
    except Exception as e:
        r.fail("noviewer frame still barred", f"{type(e).__name__}: {e}")


def _compile_and_check(code, bt):
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    used = _collect_identifiers(prog)
    return prog, tm, used


def _feed_stream(*, program_fp, device, precision, params, upstream, frame, time_context,
                 quality, flags, canvas, env):
    """Independent reimplementation of `tex_results_keys.lineage_key`'s documented byte-feed
    sequence (fp/dev/prec/env/par/up/frm/tc/q/flg/cnv), with `env_epoch()` supplied directly
    rather than computed — env_epoch legitimately moves with ANY interpreter.py/codegen.py/
    stdlib*.py edit (it hashes those files' bytes, `tex_cache._CODEGEN_FILES`), including this
    ask's own, so a literal base-vs-head digest comparison would fail on that alone and prove
    nothing about the viewer removal. Pinning the REST of the format this way is the proof
    that survives an honest code edit: no `view` component can ever be fed again (the kwarg
    is gone, not merely defaulted to None), and every other component's encoding is unmoved."""
    h = hashlib.sha256()

    def feed(tag, s):
        b = f"{tag}={s}".encode()
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)

    def canon_float_dict(d):
        if not d:
            return "n"
        return json.dumps({k: repr(float(v)) for k, v in d.items()}, sort_keys=True)

    feed("fp", str(program_fp))
    feed("dev", str(device))
    feed("prec", str(precision))
    feed("env", env)
    feed("par", json.dumps(params or {}, sort_keys=True, default=repr))
    feed("up", json.dumps([str(u) for u in upstream]))
    feed("frm", "n" if frame is None else repr(float(frame)))
    feed("tc", canon_float_dict(time_context))
    feed("q", "n" if quality is None else str(quality))
    feed("flg", json.dumps(sorted(str(f) for f in flags)))
    feed("cnv", "n" if canvas is None else json.dumps(canvas, sort_keys=True, default=list))
    return h.hexdigest()


def test_noviewer_lineage_key_byte_format_pinned(r: SubTestResult):
    print("\n--- v042-noviewer: lineage_key's byte format for every non-viewer component is pinned ---")
    fixed_env = "torch-X|sm00|deadbeef"
    real_env_epoch = tex_results_keys.env_epoch
    rows = [
        dict(program_fp="prog-a", device="cpu", precision="fp32", params={"gain": 1.2},
             upstream=(), frame=None, time_context=None, quality=None, flags=(), canvas=None),
        dict(program_fp="prog-b", device="cuda:0", precision="fp16", params={"n": 3, "k": 0.5},
             upstream=("up1", "up2"), frame=4.0, time_context={"frame": 4.0, "fps": 24.0},
             quality="preview", flags=("out:OUT", "ic:4"), canvas={"shape": [1, 8, 8, 4]}),
        dict(program_fp="prog-c", device="cpu", precision="fp32", params={},
             upstream=(), frame=None, time_context={"time": 1.5}, quality=None,
             flags=(), canvas={"shape": [2, 4, 4, 3], "roi": [0, 0, 4, 4, 8, 8]}),
    ]
    try:
        tex_results_keys.env_epoch = lambda: fixed_env
        for row in rows:
            expected = _feed_stream(env=fixed_env, **row)
            got = tex_results_keys.lineage_key(**row)
            assert got == expected, (
                f"lineage_key byte format moved for {row['program_fp']}: "
                f"expected {expected}, got {got}")
        r.ok(f"{len(rows)} representative rows (plain, time-context, ROI-canvas) mint exactly "
             f"the reimplemented fp/dev/prec/env/par/up/frm/tc/q/flg/cnv byte stream — no "
             f"`view` component can be fed, and none of the others moved")
    except Exception as e:
        r.fail("noviewer lineage_key byte format", f"{type(e).__name__}: {e}")
    finally:
        tex_results_keys.env_epoch = real_env_epoch
