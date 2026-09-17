"""ASK-6c — `select(cond, a, b)`, a non-syncing per-pixel selector.

This file carries ASK-6c's own edge cases (NaN isolation, the CUDA-graph capturability
contrast that is the whole point of the builtin, and the E3011 reserved-name row); the
other rows land as edits to the existing test files that own each property:
tests/test_codegen_optimizer.py (codegen-equivalence corpus: uniform cond, per-pixel
cond, int cond, scalar arms, vec3/vec4 arms — CPU, both tiers), tests/test_type_checker.py
(promoted return typing + E5003 on a non-scalar cond / string-or-matrix arm),
tests/test_v018_precision.py (`_C1_MUST_DECLINE` — precision="auto" declines select
like TernaryOp), tests/test_lazy_cooking.py (invariant 11 — select never severs, even
at a literal-cond value that would fold an equivalent `?:`), tests/test_v017_phase1.py
/ tests/stdlib_probe.py (TST-6 registry-parity, auto-covered from FUNCTION_SIGNATURES —
no edit needed), tests/test_v017_phase2.py (LANG-4 JS<->registry sig drift + DOC-4
Function-Reference.md/tex_help.json regen — no edit needed beyond the registrations
themselves).

Each CUDA-dependent row skips (not silently passes) without a GPU.
"""
from helpers import *
from failure_harness import run_tier, max_diff, TierUnavailable

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]


def test_ask6c_nan_untaken_arm_does_not_leak(r: SubTestResult):
    print("\n--- ASK-6c: an untaken arm's NaN/Inf does not leak (torch.where, not lerp) ---")
    # The design's own probe, restated as the contrast it is: torch.lerp(1, inf, 0) is
    # NaN (0*inf under IEEE), but torch.where(False, inf, 1) is 1 — select is built on
    # the latter, never the former, so an untaken arm's non-finite value is discarded,
    # not multiplied by a zero weight.
    from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib
    cond = torch.tensor([1.0, 0.0, 1.0, 0.0])
    a = torch.tensor([1.0, float("nan"), float("inf"), 3.0])   # taken at [0],[2]; untaken at [1],[3]
    b = torch.tensor([float("nan"), 2.0, 5.0, float("inf")])   # untaken at [0],[2]; taken at [1],[3]
    try:
        out = TEXStdlib.fn_select(cond, a, b)
        expected = torch.tensor([1.0, 2.0, float("inf"), float("inf")])
        assert torch.equal(out, expected), f"got {out.tolist()}, expected {expected.tolist()}"
        assert torch.isfinite(out[:2]).all(), "a finite pixel leaked non-finite from its untaken arm"
        r.ok(f"fn_select: untaken-arm NaN/Inf isolated, taken values pass through ({out.tolist()})")
    except Exception as e:
        r.fail("ASK-6c NaN isolation (unit)", f"{type(e).__name__}: {e}")

    # Same contrast, through the language: a per-pixel cond, both tiers, both devices.
    code = "@OUT = vec4(vec3(select(@A.r > 0.5, @A.g, @B.g)), 1.0);"
    img_a = torch.tensor([[[[1.0, 0.7, 0.0], [0.0, 0.3, 0.0]]]])   # r: [1] taken=a, [0] taken=b
    img_b = img_a.clone()
    img_b[..., 1] = float("nan")   # @B.g is NaN everywhere — only read where cond is false
    for dev in _DEVICES:
        for tier in ("interp", "codegen"):
            try:
                got = run_tier(code, {"A": img_a.to(dev), "B": img_b.to(dev)}, tier, device=dev)["OUT"]
                # cond = @A.r > 0.5 -> [True, False] on row 0. Pixel 0 takes @A.g=0.7 (finite);
                # pixel 1 takes @B.g=NaN (the untaken side there is @A.g=0.3, also finite, but
                # the TAKEN side is the NaN we deliberately planted).
                assert torch.isfinite(got[0, 0, 0, :3]).all(), f"[{dev}/{tier}] pixel0 (cond=True, reads @A.g) leaked NaN"
                assert torch.isnan(got[0, 0, 1, :3]).all(), f"[{dev}/{tier}] pixel1 (cond=False, reads @B.g=NaN) should surface it"
                r.ok(f"[{dev}/{tier}] select reads exactly the taken arm per pixel, no cross-leak")
            except Exception as e:
                r.fail(f"ASK-6c NaN isolation [{dev}/{tier}]", f"{type(e).__name__}: {e}")


def test_ask6c_reserved_name_e3011(r: SubTestResult):
    print("\n--- ASK-6c: select is a reserved builtin name (E3011) ---")
    try:
        raised = None
        try:
            check_code("float select(float c, float a, float b){ return a; }\n@OUT = vec4(0.0);")
        except Exception as e:
            raised = e
        assert raised is not None, "redefining select as a user function did not raise"
        code = getattr(getattr(raised, "diagnostic", None), "code", None)
        assert code == "E3011", f"wrong error code: {code!r} (raised={raised!r})"
        r.ok("`float select(...)` user function is refused as E3011 (reserved builtin)")
    except Exception as e:
        r.fail("ASK-6c E3011", f"{type(e).__name__}: {e}")


def test_ask6c_captures_where_uniform_if_would_not(r: SubTestResult):
    """The rationale for the whole builtin, pinned as measured behaviour: codegen's
    scalar-shortcut `if` (`float(cond) > 0.5`, codegen.py's `_emit_if_else`) syncs the
    host on a 0-dim condition, which CUDA-graph capture forbids outright, so a uniform
    `if`/`?:` is blacklisted on sight. select never takes that branch (no `float(cond)`,
    no `.item()`), so the identical decision captures, and a later param flip replays
    the same graph."""
    print("\n--- ASK-6c: select captures under CUDA graphs where an equivalent uniform "
          "`if` on the SAME condition does not ---")
    if not _CUDA:
        r.skip("ASK-6c capturability", "no CUDA on this box")
        return

    from TEX_Wrangle.tex_runtime import graphed as G
    img = make_img(1, 64, 64, 3, seed=17).cuda()

    code_if = ("vec3 c = @A.rgb; if ($m > 0.5) { c = @A.rgb * 2.0; } else "
               "{ c = @A.rgb * 0.5; } @OUT = vec4(c, 1.0);")
    code_sel = "@OUT = vec4(select($m > 0.5, @A.rgb * 2.0, @A.rgb * 0.5), 1.0);"

    G.clear_graph_cache()
    try:
        run_tier(code_if, {"A": img, "m": 1.0}, "graph", device="cuda")
        r.fail("ASK-6c capturability: uniform if", "expected the graph tier to decline "
               "(TierUnavailable), but it captured")
    except TierUnavailable:
        r.ok("uniform `if` on a uniform $param declines CUDA-graph capture (as designed)")
    except Exception as e:
        r.fail("ASK-6c capturability: uniform if", f"unexpected {type(e).__name__}: {e}")

    G.clear_graph_cache()
    try:
        got1 = run_tier(code_sel, {"A": img, "m": 1.0}, "graph", device="cuda")["OUT"]
    except TierUnavailable as e:
        r.fail("ASK-6c capturability: select captures", f"declined: {e}")
        return
    except Exception as e:
        r.fail("ASK-6c capturability: select captures", f"{type(e).__name__}: {e}")
        return
    r.ok("select(), the SAME condition/arms as the declined `if` above, captures")

    # Same fingerprint (same code/tier/precision) -> the SAME cached GraphedProgram is
    # replayed with the new $m value, not recompiled.
    try:
        got2 = run_tier(code_sel, {"A": img, "m": 0.0}, "graph", device="cuda")["OUT"]
        ref1 = run_tier(code_sel, {"A": img.cpu(), "m": 1.0}, "interp", device="cpu")["OUT"]
        ref2 = run_tier(code_sel, {"A": img.cpu(), "m": 0.0}, "interp", device="cpu")["OUT"]
        md1 = (got1.cpu() - ref1).abs().max().item()
        md2 = (got2.cpu() - ref2).abs().max().item()
        assert md1 < 1e-5, f"$m=1.0 maxdiff {md1} vs CPU interp"
        assert md2 < 1e-5, f"$m=0.0 maxdiff {md2} vs CPU interp"
        assert not torch.equal(got1.cpu(), got2.cpu()), \
            "the two $m values produced identical output — the flip did not take"
        r.ok(f"the captured graph follows a $param flip and matches CPU interp both "
             f"sides (maxdiff {max(md1, md2):.2e})")
    except AssertionError as e:
        r.fail("ASK-6c capturability: follows a param flip", str(e))
    except Exception as e:
        r.fail("ASK-6c capturability: follows a param flip", f"{type(e).__name__}: {e}")
