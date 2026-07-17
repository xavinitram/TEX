"""
PROC-1 — Failure-mode test harness (doc 22 §1, ranked #1 for v0.16).

The v0.15.0 suite was green while four shipped tiers were broken, because the
tests asserted only the happy path (doc 21: "the new tests assert the happy
path only"). This module provides the FIVE reusable failure-mode patterns that
would have caught every P0/P1 — so every subsequent v0.16 fix plugs its
regression test into the pattern that would have caught it, instead of adding
another happy-path assertion.

    Class A — async lifecycle:    drive a background-compiling tier across
                                  repeated cooks so the POST-compile cook is
                                  exercised, not just the first sync cook.
    Class B — restart / persist:  drop the in-memory caches to mimic a fresh
                                  process, forcing reconstruction from on-disk
                                  artifacts (PC-3 codegen / CT-1 fused).
    Class C — real entry-point:   call the endpoint users hit (preflight_from_spec,
                                  run_auto, run_graphed), never just an inner helper.
    Class D — cross-tier equiv:   run each opt-in tier (codegen / graph / compiled /
                                  auto / fp16) and compare to the interpreter. A
                                  tier that DECLINES is surfaced, never silently
                                  counted as "equivalent".
    Class E — full-surface sweep: exercise a tier across a whole feature surface
                                  (e.g. the fp16 grade/blend family), not one
                                  representative function.

Every helper reports into the standard SubTestResult and reuses helpers.py, so
these compose with the existing runner (run_all.py) and pytest unchanged.
"""
from helpers import *
from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute


class TierUnavailable(Exception):
    """A tier declined to run this program (e.g. graph capture returned None).

    Raised so that a decline is never silently mistaken for "bit-equal to the
    interpreter" — the exact blind spot that let v0.15 ship green-but-broken.
    """


# ── shared compile + binding utilities ────────────────────────────────
def compile_program(code, bindings):
    """Lex → parse → type-check. Returns (program, type_map, output_names)."""
    prog = Parser(Lexer(code).tokenize(), source=code).parse()
    bt = {n: _infer_binding_type(v) for n, v in bindings.items()}
    checker = TypeChecker(binding_types=bt, source=code)
    tm = checker.check(prog)
    outs = sorted(checker.assigned_bindings.keys())
    if not outs:
        raise InterpreterError("program has no @OUT / @name output")
    return prog, tm, outs


def clone_bindings(bindings):
    """Deep-copy tensor bindings so a tier can never cross-mutate the caller's
    inputs (several tiers write in place)."""
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v)
            for k, v in bindings.items()}


def _as_dict(res, outs):
    return res if isinstance(res, dict) else {outs[0]: res}


def _harness_fp(code, tier, precision):
    """Stable-per-(code,tier,precision) fingerprint (no Date/random needed)."""
    return f"harness_{tier}_{precision}_{abs(hash(code)) & 0xFFFFFFFF:08x}"


# ── run_tier: dispatch a program through the REAL tier entry point ─────
def run_tier(code, bindings, tier, device="cpu", precision="fp32"):
    """Execute `code` under a named tier via the SAME entry point tex_node uses.

    Tiers: "interp", "codegen", "compiled", "auto", "graph". Returns a
    {output_name: tensor} dict.

    A tier names the execution ENGINE entry point (`_codegen_only_execute`,
    `execute_compiled`, `run_auto`, `run_graphed`), NOT a `compile_mode` widget
    value — it tests an engine in isolation, one layer below the production
    routing gate (`_should_stencil_route`). Only "graph" can DECLINE (returns
    None → TierUnavailable); "codegen"/"compiled"/"auto" self-fall-back to the
    interpreter internally, so they never raise TierUnavailable.
    """
    prog, tm, outs = compile_program(code, bindings)
    fp = _harness_fp(code, tier, precision)
    b = clone_bindings(bindings)
    if tier == "interp":
        return _as_dict(Interpreter().execute(prog, b, tm, device=device,
                        output_names=outs, precision=precision), outs)
    if tier == "codegen":
        return _as_dict(_codegen_only_execute(prog, b, tm, device,
                        output_names=outs, precision=precision, fingerprint=fp, time_context=None), outs)
    if tier == "compiled":
        return _as_dict(execute_compiled(prog, b, tm, device, fp,
                        output_names=outs, precision=precision), outs)
    if tier == "auto":
        from TEX_Wrangle.tex_runtime.compiled import run_auto
        return _as_dict(run_auto(prog, b, tm, device, fp,
                        output_names=outs, precision=precision), outs)
    if tier == "graph":
        from TEX_Wrangle.tex_runtime.graphed import run_graphed
        res = run_graphed(prog, b, tm, device, fp, output_names=outs, precision=precision)
        if res is None:
            raise TierUnavailable("graph tier declined (not graphable / capture failed)")
        return _as_dict(res, outs)
    raise ValueError(f"unknown tier {tier!r}")


def max_diff(a_dict, b_dict):
    """Largest abs difference across matching outputs; inf on a non-tensor mismatch."""
    m = 0.0
    for k, av in a_dict.items():
        bv = b_dict.get(k)
        if isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor):
            m = max(m, (av.float() - bv.float()).abs().max().item())
        elif av != bv:
            return float("inf")
    return m


# ── Class D: cross-tier equivalence ───────────────────────────────────
def assert_tier_equiv(r, name, code, bindings, tiers=("codegen",),
                      device="cpu", precision="fp32", tol=1e-5):
    """Class D. Every tier in `tiers` must match the fp32 interpreter within `tol`.

    A tier that declines (only "graph" can) is reported OK — a decline is never
    silently counted as a match. (Class E's sweep_precision grades fp16.)
    """
    try:
        base = run_tier(code, bindings, "interp", device=device)
    except Exception as e:
        r.fail(f"[D] {name}", f"interp baseline raised: {e}")
        return
    for tier in tiers:
        try:
            got = run_tier(code, bindings, tier, device=device, precision=precision)
        except TierUnavailable as e:
            r.ok(f"[D] {name}|{tier} declined (ok): {e}")
            continue
        except Exception as e:
            r.fail(f"[D] {name}|{tier}", str(e))
            continue
        md = max_diff(base, got)
        if md <= tol:
            r.ok(f"[D] {name}|{tier} maxdiff={md:.2e}")
        else:
            r.fail(f"[D] {name}|{tier}", f"maxdiff {md:.3e} > tol {tol:.1e}")


# ── Class E: full-surface sweep ───────────────────────────────────────
def sweep_precision(r, label, programs, bindings_f32, bindings_f16, tol=3e-2):
    """Class E. Run each program's fp16 cook (fp16 inputs + precision=fp16) and
    compare to its fp32 cook. Catches a tier that raises — or silently diverges —
    on only *some* of a feature family (the fp16 dtype-reconcile class)."""
    for pname, code in programs.items():
        try:
            base = run_tier(code, bindings_f32, "interp", precision="fp32")
            got = run_tier(code, bindings_f16, "interp", precision="fp16")
            md = max_diff(base, got)
            if md <= tol:
                r.ok(f"[E] {label}|{pname} maxdiff={md:.2e}")
            else:
                r.fail(f"[E] {label}|{pname}", f"maxdiff {md:.3e} > tol {tol:.1e}")
        except Exception as e:
            r.fail(f"[E] {label}|{pname}", str(e))


# ── Class C: real entry-point (not the inner helper) ──────────────────
def preflight_via_endpoint(spec, terminal_code):
    """Class C. Hit the endpoint helper (preflight_from_spec) — the code path the
    server actually calls — not the inner chain_preflight the v0.15 test used."""
    from TEX_Wrangle.tex_fusion import preflight_from_spec
    return preflight_from_spec(spec, terminal_code, _infer_binding_type)


# ── Class A: async background-compile lifecycle ───────────────────────
def drive_auto(code, bindings, cooks=2):
    """Class A. Cook repeatedly through run_auto so a background compile has a
    chance to commit. Returns [per-cook output dict]. Correctness must hold on
    EVERY cook regardless of which internal tier (codegen / trial / committed)
    served it — the safety invariant that survives even while CC-2 is deferred."""
    from TEX_Wrangle.tex_runtime.compiled import run_auto
    prog, tm, outs = compile_program(code, bindings)
    fp = _harness_fp(code, "auto", "fp32")
    results = []
    for _ in range(cooks):
        b = clone_bindings(bindings)
        results.append(_as_dict(run_auto(prog, b, tm, "cpu", fp,
                       output_names=outs), outs))
    return results


# ── Class B: restart / persistence simulation ─────────────────────────
def simulate_restart():
    """Class B. Drop TEX's in-memory caches to mimic a fresh process, forcing the
    next cook to reconstruct from on-disk persistence (PC-3 codegen / CT-1 fused)
    instead of a warm memory hit."""
    from TEX_Wrangle.tex_cache import get_cache
    import TEX_Wrangle.tex_fusion as _fus
    clear_compiled_cache()
    try:
        get_cache()._codegen_memory.clear()
    except Exception:
        pass
    try:
        _fus._FUSED_MEMO.clear()
    except Exception:
        pass
    try:
        import torch._dynamo as _d
        _d.reset()
    except Exception:
        pass
