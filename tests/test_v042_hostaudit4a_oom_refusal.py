"""
v0.42 HOSTAUDIT-4a — an unrecoverable OOM carries a structured refusal.

`test_eng2_oom_ladder` (test_v022_phase1.py) already pins the additive contract that
matters most: an unrecoverable OOM re-raises the ORIGINAL exception object, unwrapped,
so `tex_node.py` and ComfyUI's own `is_oom` handling keep working exactly as before. This
file pins the ADDITION on top of that: `tex_engine.cook()` — ENG-1's host-agnostic entry
point, the one a second (non-ComfyUI) host calls directly — used to hand that caller a bare
torch exception with nothing TEX-specific on it. A host with no ComfyUI-shaped `is_oom`
of its own could only detect "this was OOM" by re-implementing TEX's own detection.

`getattr(exc, "tex_refusal", None)` now resolves to an `EngineRefusal(code, stage, message)`
— the same `(code, stage, message)` shape `tex_checkpoint.GateRefusal` already uses — with
`code == tex_engine.REFUSE_OUT_OF_MEMORY`. The exception's TYPE and IDENTITY are unchanged
(this is `is`-identical to the object `test_eng2_oom_ladder` already checks), so nothing
that already catches it seeing a difference is exactly the point.
"""
import torch


def test_hostaudit4a_unrecoverable_oom_carries_a_refusal(r):
    from TEX_Wrangle import tex_engine
    import TEX_Wrangle.tex_memory as M

    oom_t = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_t is None:
        r.skip("HOSTAUDIT-4a refusal", "this torch build has no cuda.OutOfMemoryError type")
        return

    from helpers import make_img  # same helper test_v022_phase1 uses
    img = make_img(1, 8, 8, 3, seed=2)
    code = "@OUT = vec4(@A.rgb * 1.1, 1.0);"

    the_oom = oom_t("CUDA out of memory")
    orig_tier = tex_engine._run_tier
    orig_free = M.free_tensor_caches
    M.free_tensor_caches = lambda: None

    def _always_oom(ctx, t):
        raise the_oom
    tex_engine._run_tier = _always_oom
    try:
        tex_engine.cook(code, {"A": img})   # cpu: no tiled rung -> straight re-raise
        r.fail("HOSTAUDIT-4a refusal", "an unrecoverable OOM did not propagate at all")
        return
    except BaseException as e:
        if e is not the_oom:
            r.fail("HOSTAUDIT-4a refusal identity",
                   f"OOM was re-wrapped as {type(e).__name__} — this must stay unchanged "
                   "(test_eng2_oom_ladder's own contract)")
            return
        refusal = getattr(e, "tex_refusal", None)
        if refusal is None:
            r.fail("HOSTAUDIT-4a refusal", "no .tex_refusal was attached to the raised OOM")
            return
        if (refusal.code == tex_engine.REFUSE_OUT_OF_MEMORY
                and isinstance(refusal.message, str) and refusal.message):
            r.ok(f"a bare tex_engine.cook() OOM carries EngineRefusal(code={refusal.code!r})")
        else:
            r.fail("HOSTAUDIT-4a refusal shape", f"got {refusal!r}")
    finally:
        tex_engine._run_tier = orig_tier
        M.free_tensor_caches = orig_free


def test_hostaudit4a_non_oom_error_carries_no_refusal(r):
    """The refusal is OOM-specific — an ordinary bug must not grow one, which would make
    a real defect look like a recoverable-memory condition to a host reading `tex_refusal`."""
    from TEX_Wrangle import tex_engine

    orig_tier = tex_engine._run_tier
    tex_engine._run_tier = lambda ctx, t: (_ for _ in ()).throw(ValueError("not an oom"))
    try:
        from helpers import make_img
        img = make_img(1, 8, 8, 3, seed=2)
        tex_engine.cook("@OUT = vec4(@A.rgb * 1.1, 1.0);", {"A": img})
        r.fail("HOSTAUDIT-4a non-OOM", "a ValueError was swallowed by the OOM ladder")
    except ValueError as e:
        if getattr(e, "tex_refusal", None) is None:
            r.ok("a non-OOM error carries no .tex_refusal")
        else:
            r.fail("HOSTAUDIT-4a non-OOM", f"a ValueError grew a refusal: {e.tex_refusal!r}")
    except Exception as e:
        r.fail("HOSTAUDIT-4a non-OOM", f"a ValueError became {type(e).__name__}")
    finally:
        tex_engine._run_tier = orig_tier
