"""v0.35 CF-7 — pins for three facts that were synchronised by comment.

Each of these is a set or a constant duplicated across files with no machinery holding the
copies together. None has drifted yet; all three are one edit away, and two of them drift
SILENTLY (a missing builtin types as an undefined variable; a stale publish manifest ships a
tool declaring the wrong language version).
"""
import pathlib
import re


def test_v035_cf7_builtin_name_sets_agree(r):
    """The three hand-synced builtin-name sets are the same set.

    `type_checker._BUILTIN_VAR_NAMES` seeds the top scope, `interpreter._BUILTIN_NAMES`
    decides what `_create_builtins` may bind, and `diagnostics._BUILTIN_VAR_HINTS` supplies
    the "did you mean" text. Adding a builtin means editing all three, and the only thing
    saying so is a comment.

    The failure is not loud: a name in the checker but not the interpreter type-checks and
    then dies at runtime as an undefined variable; a name in the interpreter but not the
    checker is rejected at compile time as unknown, which reads like a user typo. Neither
    points at the set that was missed."""
    from TEX_Wrangle.tex_compiler.type_checker import _BUILTIN_VAR_NAMES
    from TEX_Wrangle.tex_runtime.interpreter import _BUILTIN_NAMES, _TIME_BUILTIN_NAMES
    from TEX_Wrangle.tex_compiler.diagnostics import _BUILTIN_VAR_HINTS

    try:
        checker = set(_BUILTIN_VAR_NAMES)
        interp = set(_BUILTIN_NAMES)
        hints = set(_BUILTIN_VAR_HINTS)
        assert checker == interp, (f"checker-only={sorted(checker - interp)}, "
                                   f"interpreter-only={sorted(interp - checker)}")
        # The hint table is allowed to be a SUPERSET — it also carries entries for names that
        # are not builtins but are commonly mistyped as one. It may not be missing a real one.
        assert checker <= hints, f"builtins with no diagnostic hint: {sorted(checker - hints)}"
        # The time subset must be part of the whole, or `_CACHEABLE_BUILTIN_NAMES` (defined as
        # the difference) silently keeps a playhead builtin in the cross-cook LRU.
        assert _TIME_BUILTIN_NAMES <= interp, sorted(set(_TIME_BUILTIN_NAMES) - interp)
        r.ok(f"CF-7: the {len(checker)} builtin names agree across checker/interpreter/hints")
    except Exception as e:
        r.fail("CF-7 builtin-name parity", f"{type(e).__name__}: {e}")


def test_v035_cf7_js_publish_manifest_tracks_the_language_version(r):
    """`js/tex_extension.js`'s publish-manifest `tex_language` equals `LANGUAGE_VERSION`.

    The editor stamps this constant into every `.textool` it publishes. It is a hardcoded
    string in JavaScript, so no Python drift test has ever covered it — and §2.3's bump to
    0.24 would have stranded it at 0.23, shipping tools that declare an older language than
    they were authored against. That is exactly the compatibility claim LANG-3 exists to make
    true, broken by a constant nobody would think to grep."""
    from TEX_Wrangle.tex_api import LANGUAGE_VERSION

    js = (pathlib.Path(__file__).resolve().parent.parent / "js" / "tex_extension.js")
    try:
        src = js.read_text(encoding="utf-8")
        found = re.findall(r'tex_language:\s*"([0-9.]+)"', src)
        assert found, "no `tex_language: \"X.Y\"` constant found in js/tex_extension.js"
        bad = [v for v in found if v != LANGUAGE_VERSION]
        assert not bad, (f"js publish manifest declares tex_language {bad}, "
                         f"but tex_api.LANGUAGE_VERSION is {LANGUAGE_VERSION!r}")
        r.ok(f"CF-7: the JS publish manifest tracks LANGUAGE_VERSION ({LANGUAGE_VERSION})")
    except Exception as e:
        r.fail("CF-7 js language-version drift", f"{type(e).__name__}: {e}")


def test_v035_cf6_the_grid_is_a_consensus_not_first_wins(r):
    """CF-6: two orderings of the same bindings produce the same pixels.

    `_determine_spatial_shape` returned on the FIRST spatial binding, so a `[B,1,W,C]`
    broadcast strip declared before a `[B,H,W,C]` frame collapsed `v`/`iy`/`ih` to one row —
    the whole output grid depended on dict order, with nothing saying which order was meant.
    Measured 0.60 maxdiff between the two orderings (0.9992 on the v0.32 probe).

    The consensus is the BROADCAST extent, which is what every downstream op already does with
    a singleton, and `max` is commutative so the ordering dependence is gone by construction
    rather than by convention."""
    import torch
    from TEX_Wrangle import tex_engine
    try:
        strip = torch.rand(1, 1, 16, 4)          # singleton in H
        frame = torch.rand(1, 16, 16, 4)
        code = "@OUT = vec4(vec3(v), 1.0);"      # `v` is what collapses
        a = tex_engine.cook(code, {"S": strip, "A": frame}, device_mode="cpu").outputs["OUT"]
        b = tex_engine.cook(code, {"A": frame, "S": strip}, device_mode="cpu").outputs["OUT"]
        assert tuple(a.shape) == (1, 16, 16, 4), tuple(a.shape)
        assert tuple(a.shape) == tuple(b.shape), (tuple(a.shape), tuple(b.shape))
        assert torch.equal(a, b), f"binding order moved pixels: {float((a - b).abs().max())}"
        # ...and a lone singleton still sizes from itself — consensus of one is that one.
        only = tex_engine.cook(code, {"S": strip}, device_mode="cpu").outputs["OUT"]
        assert tuple(only.shape) == (1, 1, 16, 4), tuple(only.shape)
        # INVARIANT #2, and the leg that was missing: codegen derived the grid with its own
        # first-wins loop, so fixing the interpreter alone put the tiers 0.98-0.999 apart.
        #
        # This drives `_build_codegen_env` DIRECTLY rather than cooking with
        # `compile_mode="auto"`. That matters, and the first cut of this leg got it wrong: on a
        # small CPU program `auto` measures the tiers and routes to the INTERPRETER, so the
        # codegen derivation was never reached and mutating it back to first-wins left this
        # test green — a decorative pin on exactly the defect it was written for. Asking the
        # builder for its spatial shape is unambiguous and cannot be routed away from.
        from TEX_Wrangle.tex_compiler.lexer import Lexer
        from TEX_Wrangle.tex_compiler.parser import Parser
        from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
        from TEX_Wrangle.tex_compiler.types import TEXType
        from TEX_Wrangle.tex_runtime.compiled import _build_codegen_env
        mixed = "@OUT = vec4(vec3(v) * @A.rgb + @S.rgb, 1.0);"
        prog = Parser(Lexer(mixed).tokenize()).parse()
        TypeChecker(binding_types={"S": TEXType.VEC4, "A": TEXType.VEC4,
                                   "OUT": TEXType.VEC4}).check(prog)
        _env, sp, _used = _build_codegen_env(prog, {"S": strip, "A": frame},
                                             torch.device("cpu"), 0)
        assert sp == (1, 16, 16), f"codegen derived {sp}, not the consensus (1, 16, 16)"
        # ...and end-to-end through the engine, so the claim covers cooked pixels too.
        i = tex_engine.cook(mixed, {"S": strip, "A": frame}, device_mode="cpu",
                            compile_mode="none").outputs["OUT"]
        g = tex_engine.cook(mixed, {"S": strip, "A": frame}, device_mode="cpu",
                            compile_mode="auto").outputs["OUT"]
        assert tuple(i.shape) == tuple(g.shape), (tuple(i.shape), tuple(g.shape))
        assert torch.equal(i, g), f"interp<->codegen diverged: {float((i - g).abs().max())}"
        r.ok("CF-6: the output grid is the consensus extent, and both tiers derive it once")
    except Exception as e:
        r.fail("CF-6 consensus extent", f"{type(e).__name__}: {e}")


def test_v035_cf6_binding_order_does_not_move_the_auto_precision_gate(r):
    """CF-6, the mirror that reaches PIXELS: `precision="auto"` must not depend on dict order.

    `tex_engine`'s `cook_px` derived H*W from the FIRST spatial binding while the grid had
    become a consensus, and it is what `auto` gates on (`cook_px >= _MIN_FP16_PX`). So a
    `[1,1,W,C]` strip bound before a `[1,H,W,C]` frame showed the gate W pixels instead of
    H*W: on CUDA at 2048², strip-first resolved fp32 and frame-first resolved fp16 — the same
    graph, **maxdiff 7.32e-04** decided by binding order. Under the display quantum, and `auto`
    is opt-in, but it is exactly the property this release claims is gone by construction.

    CUDA-ONLY BY NECESSITY, which is why the CPU pin above could not see it: `auto` resolves
    fp32 on CPU whatever the resolution, so the gate never splits there. A pin that skips on
    this box would have been worse than none — it is the GPU leg that carries the claim."""
    import torch
    from TEX_Wrangle import tex_engine
    if not torch.cuda.is_available():
        r.skip("CF-6 auto-precision order independence", "no CUDA — the auto gate cannot split")
        return
    try:
        from TEX_Wrangle.tex_runtime.precision_policy import _MIN_FP16_PX
        side = int(_MIN_FP16_PX ** 0.5)            # the smallest frame that clears the gate
        frame = torch.rand(1, side, side, 4, device="cuda")
        strip = torch.rand(1, 1, side, 4, device="cuda")     # H*W = side, far under the gate
        code = "@OUT = vec4(@S.rgb * 0.25 + @F.rgb * 0.75, 1.0);"

        def cook(b):
            out = tex_engine.cook(code, b, device_mode="cuda", precision="auto",
                                  compile_mode="none").outputs["OUT"]
            torch.cuda.synchronize()
            return out

        a = cook({"S": strip, "F": frame})
        b = cook({"F": frame, "S": strip})
        assert tuple(a.shape) == tuple(b.shape) == (1, side, side, 4), \
            (tuple(a.shape), tuple(b.shape))
        md = float((a.float() - b.float()).abs().max())
        assert md == 0.0, \
            (f"binding order moved pixels under precision='auto' by {md:.3e} — the auto gate "
             f"is sized off a different grid than the cook")
        r.ok("CF-6: the auto-precision gate is order-independent (cuda, both orders agree)")
    except Exception as e:
        r.fail("CF-6 auto-precision order", f"{type(e).__name__}: {e}")


def test_v035_cf6_an_unread_binding_does_not_size_the_grid(r):
    """CF-6's second half: only the bindings the PROGRAM READS are consensus participants.

    A consensus over EVERY binding lets a wired-but-unread input raise an axis. That is worse
    than the ordering bug it replaced: the grid inflates past what the read bindings can
    broadcast to and the cook dies with a RuntimeError on a program that cooked fine before,
    and the output shape starts depending on whether lazy pruning happened to drop the unused
    wire. Both close by asking who actually reads.

    The companion clause is the generative program — an image wired and never read is the ONLY
    thing sizing the grid there, so narrowing to the read set must not narrow to nothing."""
    import torch
    from TEX_Wrangle import tex_engine
    try:
        small = torch.rand(1, 64, 64, 4)
        big = torch.rand(1, 256, 256, 4)
        code = "@OUT = vec4(@A.rgb, 1.0);"        # @B is wired and never mentioned
        out = tex_engine.cook(code, {"A": small, "B": big}, device_mode="cpu").outputs["OUT"]
        assert tuple(out.shape) == (1, 64, 64, 4), \
            f"an unread binding sized the grid: {tuple(out.shape)}"
        # Pruning parity: dropping the unread wire must not move a pixel or a dimension.
        pruned = tex_engine.cook(code, {"A": small}, device_mode="cpu").outputs["OUT"]
        assert tuple(out.shape) == tuple(pruned.shape) and torch.equal(out, pruned), \
            "the output depends on whether the unused binding was pruned"
        # ...and narrowing to the read set never narrows to nothing.
        gen = tex_engine.cook("@OUT = vec4(u, v, 0.0, 1.0);", {"A": torch.rand(1, 48, 24, 4)},
                              device_mode="cpu").outputs["OUT"]
        assert tuple(gen.shape)[:3] == (1, 48, 24), \
            f"a generative program lost its grid: {tuple(gen.shape)}"
        r.ok("CF-6: unread bindings are not consensus participants; generative cooks keep theirs")
    except Exception as e:
        r.fail("CF-6 read-set participants", f"{type(e).__name__}: {e}")


def test_v035_cf6_the_roi_grid_uses_the_same_participants(r):
    """CF-6: an ROI cook decides WHO participates the same way a whole-frame cook does.

    The first cut applied `roi` as an early return ABOVE the read-set narrowing, reasoning
    that only the batch comes from the bindings under an ROI so ordering could not bite.
    Ordering could not; participation could. `@A [1,H,W,4]` read with `@B [8,H,W,4]` wired and
    never mentioned gridded the whole-frame cook at B=1 and the ROI cook at B=8 — and BOTH
    gates that should catch it are axis-blind (`run_roi`'s R1 and the ROI postcondition loop
    compare H and W only), so the window is reported served and a host blits an 8-batch patch
    into a 1-batch canvas.

    `roi` is an axis selector now, not a bypass: H and W come from the cook region, so only B
    is still decided by the bindings — and it is decided by the same participants."""
    import torch
    from TEX_Wrangle.tex_runtime.interpreter import _consensus_extent
    from TEX_Wrangle.tex_compiler.lexer import Lexer
    from TEX_Wrangle.tex_compiler.parser import Parser
    from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
    from TEX_Wrangle.tex_compiler.types import TEXType
    try:
        code = "@OUT = vec4(@A.rgb, 1.0);"          # @B is wired and never read
        prog = Parser(Lexer(code).tokenize()).parse()
        TypeChecker(binding_types={"A": TEXType.VEC4, "B": TEXType.VEC4,
                                   "OUT": TEXType.VEC4}).check(prog)
        b = {"A": torch.rand(1, 32, 32, 4), "B": torch.rand(8, 32, 32, 4)}
        whole = _consensus_extent(b, prog)
        window = _consensus_extent(b, prog, roi=(0, 0, 8, 8, 32, 32))
        assert whole == (1, 32, 32), f"whole-frame grid {whole}, want (1, 32, 32)"
        assert window[0] == whole[0], \
            (f"the ROI cook grids batch {window[0]} where the whole-frame cook grids "
             f"{whole[0]} — the ROI branch is bypassing the participation rule")
        assert window == (1, 8, 8), f"ROI grid {window}, want (1, 8, 8)"
        r.ok("CF-6: an ROI cook narrows participants exactly as the whole-frame cook does")
    except Exception as e:
        r.fail("CF-6 roi participants", f"{type(e).__name__}: {e}")


def test_v035_cf6_the_peak_estimate_describes_the_grid_the_cook_uses(r):
    """CF-6: M-1's peak-bytes preflight sizes itself off the SAME grid the cook will use.

    Under first-wins this agreed by construction — the preflight and the cook both collapsed
    to the first spatial binding. The consensus moved the cook and would have left the mirror
    behind: with a `[B,1,W,C]` strip bound first, `estimate_peak_bytes` is handed a one-row
    grid for a cook that grids H rows, under-estimating peak VRAM by up to H×. The estimate's
    only job is to decide whether to unload models before a big cook, so an under-estimate
    means the preflight quietly no-ops on exactly the cook that needed it.

    Asserted by capturing what `estimate_peak_bytes` is CALLED with, which is the fact in
    question and happens before any host or driver query — so this needs no CUDA and no
    memory pressure."""
    import torch
    from TEX_Wrangle import tex_engine, tex_memory
    from TEX_Wrangle.tex_compiler.lexer import Lexer
    from TEX_Wrangle.tex_compiler.parser import Parser
    from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
    from TEX_Wrangle.tex_compiler.types import TEXType
    seen = []
    real = tex_memory.estimate_peak_bytes
    try:
        code = "@OUT = vec4(@S.rgb + @A.rgb, 1.0);"
        prog = Parser(Lexer(code).tokenize()).parse()
        TypeChecker(binding_types={"S": TEXType.VEC4, "A": TEXType.VEC4,
                                   "OUT": TEXType.VEC4}).check(prog)
        bindings = {"S": torch.rand(1, 1, 64, 4), "A": torch.rand(1, 64, 64, 4)}

        def spy(program, spatial, dtype_bytes, fingerprint=None, *a, **k):
            seen.append(spatial)
            return real(program, spatial, dtype_bytes, fingerprint, *a, **k)

        tex_memory.estimate_peak_bytes = spy
        tex_engine._preflight_memory(prog, bindings, "cuda", 4, None)
        assert seen, "the preflight never reached estimate_peak_bytes"
        assert seen[0] == (1, 64, 64), \
            (f"the peak estimate was handed {seen[0]} while the cook grids (1, 64, 64) — "
             f"M-1 would under-estimate by {64 // max(seen[0][1], 1)}x")
        r.ok("CF-6: the M-1 peak estimate is sized off the consensus grid, not first-wins")
    except Exception as e:
        r.fail("CF-6 preflight grid", f"{type(e).__name__}: {e}")
    finally:
        tex_memory.estimate_peak_bytes = real


def test_v035_cf2_a_whole_frame_partial_recook_checks_its_prefix(r):
    """CF-2's second half: `cook(None, dirty_from=k)` is a validity question too.

    `chain_windows` is only consulted when a roi is given, so a whole-frame PARTIAL recook
    walked straight past it. Stage k reads canvas k-1, which an earlier window cook may have
    left correct only inside its window — so stage k cooks whole from a partly-stale input and
    then records `_valid=None`, i.e. "correct everywhere". That is the same false claim a
    decline makes, and unlike a decline nothing downstream can detect it.

    The guard widens `dirty_from` to 0 when the clean prefix is not whole-frame valid. Pinned
    at the state level rather than by pixels: what makes the bug silent is the RECORD, so the
    record is what this asserts."""
    import torch
    import sys
    import pathlib
    ex = pathlib.Path(__file__).resolve().parent.parent / "examples"
    if str(ex) not in sys.path:
        sys.path.insert(0, str(ex))
    try:
        import host_demo
        comp = host_demo.RoiComp(res=64, device="cpu")
        comp.cook(None, 0, use_cache=False)                    # everything valid everywhere
        assert all(v is None for v in comp._valid), comp._valid
        # A window cook leaves the touched stages valid only over their windows.
        comp.cook((8, 8, 16, 16, 64, 64), 3, use_cache=False)
        windowed = [i for i, v in enumerate(comp._valid) if v is not None]
        assert windowed, "setup: the window cook left every stage whole-frame valid"
        # ...now a WHOLE-FRAME partial recook whose prefix contains one of those stages.
        comp.cook(None, max(windowed) + 1, use_cache=False)
        still = [i for i, v in enumerate(comp._valid) if v is not None]
        assert not still, f"a whole-frame partial recook left stages {still} window-valid"
        assert not comp._declined, f"declines survived a full recook: {comp._declined}"
        r.ok("CF-2: a whole-frame partial recook over a window-valid prefix widens to a full one")
    except Exception as e:
        r.fail("CF-2 whole-frame partial recook", f"{type(e).__name__}: {e}")


def test_v035_cf4_requalify_lands_the_final_and_evicts_the_preview(r):
    """CF-4: the requalify helper, and the eviction that is its point.

    Coexistence pays the governor twice — preview and final both resident, both counted — so
    requalifying a scrubbed sequence without evicting would grow the pool by exactly the bytes
    the preview tier exists to save."""
    import torch
    from TEX_Wrangle import tex_results, tex_packing
    try:
        c = tex_results.ResultCache(budget_mb=256)
        c.put("prev", torch.rand(1, 64, 64, 4), quality=tex_packing.PREVIEW, kind="IMAGE")
        assert c.preview_entries() == ["prev"], c.preview_entries()
        assert c.requalify("prev", "final", torch.rand(1, 64, 64, 4)) is True
        assert "prev" not in c._ram, "the preview entry survived requalification"
        assert "final" in c._ram
        assert c._ram["final"].orig_dtype is None, "the final frame was stored reduced"
        assert c.requalified == 1
        # A preview that vanished while the host was cooking stores nothing.
        assert c.requalify("prev", "final2", torch.rand(1, 64, 64, 4)) is False
        assert "final2" not in c._ram
        # UPGRADE IN PLACE. The obvious host spelling — requalify a key to ITSELF — ran the
        # eviction against the key `put` had just written, so the frame was destroyed and True
        # was returned anyway: the caller is told it succeeded and then misses.
        c.put("same", torch.rand(1, 32, 32, 4), quality=tex_packing.PREVIEW, kind="IMAGE")
        assert c.requalify("same", "same", torch.rand(1, 32, 32, 4)) is True
        assert "same" in c._ram, "requalifying a key to itself destroyed the frame it landed"
        assert c._ram["same"].orig_dtype is None, "the in-place upgrade stayed reduced"
        r.ok("CF-4: requalify lands the final frame, evicts the preview, upgrades in place")
    except Exception as e:
        r.fail("CF-4 requalify", f"{type(e).__name__}: {e}")


def test_v035_cf1_a_patch_over_a_demoted_base_keeps_its_home(r):
    """CF-1's residency half: a patch inherits the BASE's home, not the frame's location.

    `_admit` records `home` as wherever the tensor it was handed lives, which for a patch over
    a demoted base is the host — so the result recorded `home=cpu` and left the residency
    ladder for good, taking every downstream stage with it.

    THE RECIPE MATTERS, and the first cut of this test got it wrong. Addressing the base by
    `base_key=` ALONE makes `_patch_region_locked` reach it through `get`, and a `get` on a
    demoted entry promotes it — so the patch happens on CUDA, `_admit` sees a CUDA tensor, and
    `home=cuda` is recorded whether or not the propagation exists. The test passed on the
    unfixed tree.

    `base=` WITH `base_key=` is the recipe that discriminates: passing the buffer is precisely
    what skips the promoting `get`, so the frame being patched really is the host copy and
    `home` can only come from the base ENTRY. That is the line under test.

    And the assertion is the ladder, not the label: after the patch, a `get` must PROMOTE the
    result back to CUDA. `_promote` fires on `device != home`, so a frame mislabelled `home=cpu`
    is not merely mistagged — it is unreachable by the residency tier forever, which is the
    actual damage. `promotions` moving by one is that reachability, measured."""
    import torch
    from TEX_Wrangle import tex_results
    if not torch.cuda.is_available():
        r.skip("CF-1 patched frame keeps its home", "no CUDA on this box — nothing to demote")
        return
    try:
        c = tex_results.ResultCache(budget_mb=512)
        c.set_vram_budget(1)
        c.put("b", torch.rand(1, 512, 512, 4, device="cuda"))
        c.put("x", torch.rand(1, 512, 512, 4, device="cuda"))
        base = c._ram.get("b")
        assert base is not None and tex_results._dev_bucket(base.device) == "cpu", \
            f"setup: base not demoted (device={None if base is None else base.device})"
        assert tex_results._dev_bucket(base.home) == "cuda", \
            f"setup: base lost its home ({base.home})"
        # The host's own copy of the demoted base — a host that holds its stage buffers holds
        # them where they live, so this is a CPU tensor. No `get`, therefore no promotion.
        own = torch.rand(1, 512, 512, 4)
        c.patch_region("p", torch.rand(1, 64, 64, 4, device="cuda"),
                       (0, 0, 64, 64, 512, 512), base=own, base_key="b")
        e = c._ram.get("p")
        assert e is not None, "the patch stored nothing"
        assert tex_results._dev_bucket(e.home) == "cuda", \
            f"patched frame homed to {e.home} — it has left the residency ladder"
        assert tex_results._dev_bucket(e.device) == "cpu", \
            (f"the patch landed on {e.device}: this recipe is supposed to patch on the HOST, so "
             f"a CUDA result means the promoting `get` was not skipped and the pin is vacuous")
        # The ladder is reachable again: a hit on a demoted frame is the reuse signal, and this
        # frame is demoted only because `home` says so.
        before = c.promotions
        got = c.get("p")
        assert got is not None and c.promotions == before + 1, \
            (f"a hit did not promote the patched frame (promotions {before} -> {c.promotions}) "
             f"— `home` is what makes it eligible, so this is the one-way trip, still open")
        r.ok("CF-1: a patch over a demoted base keeps the base's home and can be promoted back")
    except Exception as e:
        r.fail("CF-1 patched home", f"{type(e).__name__}: {e}")


def test_v035_port6_engine_import_is_adapter_free(r):
    """PORT-6: a second host can import the engine without loading the ComfyUI adapter.

    `__init__.py` used to do `from .tex_node import TEXWrangleNode` at module scope, so
    `import TEX_Wrangle.tex_api` — the entry a non-ComfyUI host uses — executed the package
    root and pulled the adapter in behind it. The S-1 lint could not catch this: it proves
    `tex_core` never *imports comfy*, not that the package root leaves the adapter alone.
    The second, subtler half was the route block's `from server import PromptServer`:
    ComfyUI's `server` imports `nodes`, which imports `comfy`, so on any machine with
    ComfyUI merely on `sys.path` that one line dragged all of ComfyUI into a second host.

    Both halves are checked in a SUBPROCESS, because the assertion is about what a fresh
    interpreter loads and this suite has already imported everything. The check is the
    strict one: ComfyUI imports are RECORDED, not blocked, so a probe that would have
    succeeded still trips the assertion.

    The same subprocess then proves the ComfyUI contract is intact — reading
    `NODE_CLASS_MAPPINGS` resolves the lazy attribute, loads the adapter, and yields the
    registered node class. A version of this that only checked the first half would pass
    just as well with node registration completely broken."""
    import subprocess
    import sys as _sys
    import TEX_Wrangle

    custom_nodes = str(pathlib.Path(TEX_Wrangle.__file__).resolve().parent.parent)
    code = (
        "import sys\n"
        f"sys.path.insert(0, {custom_nodes!r})\n"
        "import builtins\n"
        "_real = builtins.__import__\n"
        "seen = []\n"
        "def _hook(name, *a, **k):\n"
        "    if name.split('.')[0] in ('comfy','comfy_api','comfy_execution','folder_paths','nodes','server'):\n"
        "        seen.append(name.split('.')[0])\n"
        "    return _real(name, *a, **k)\n"
        "builtins.__import__ = _hook\n"
        "import TEX_Wrangle.tex_api\n"
        "print('ADAPTER', 'TEX_Wrangle.tex_node' in sys.modules)\n"
        "print('COMFY', ','.join(sorted(set(seen))))\n"
        "import TEX_Wrangle as pkg\n"
        "print('HASATTR', hasattr(pkg, 'NODE_CLASS_MAPPINGS'))\n"
        "print('MAPPING', ','.join(sorted(pkg.NODE_CLASS_MAPPINGS)))\n"
        "print('NODECLS', type(pkg.NODE_CLASS_MAPPINGS['TEX_Wrangle']).__name__ != 'NoneType')\n"
        "print('LOADED', 'TEX_Wrangle.tex_node' in sys.modules)\n"
    )
    try:
        proc = subprocess.run([_sys.executable, "-X", "utf8", "-c", code],
                              capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            r.fail("PORT-6 adapter-free import",
                   f"subprocess exit {proc.returncode}: {(proc.stderr or '')[-400:]}")
            return
        out = dict(line.split(" ", 1) if " " in line else (line, "")
                   for line in proc.stdout.strip().splitlines() if line[:1].isupper())
        assert out.get("ADAPTER") == "False", \
            "importing TEX_Wrangle.tex_api still loads the ComfyUI adapter (tex_node)"
        assert out.get("COMFY", "") == "", \
            f"a ComfyUI import was attempted by the engine import path: {out.get('COMFY')}"
        # ...and the ComfyUI registration contract still holds in the same process.
        assert out.get("HASATTR") == "True", "hasattr(pkg, 'NODE_CLASS_MAPPINGS') is False"
        assert out.get("MAPPING") == "TEX_Wrangle", \
            f"NODE_CLASS_MAPPINGS keys changed: {out.get('MAPPING')!r}"
        assert out.get("NODECLS") == "True", "NODE_CLASS_MAPPINGS holds no node class"
        assert out.get("LOADED") == "True", \
            "reading NODE_CLASS_MAPPINGS did not load the adapter — registration is broken"
        r.ok("PORT-6: engine import is adapter-free and comfy-free; node registration intact")
    except Exception as e:
        r.fail("PORT-6 adapter-free import", f"{type(e).__name__}: {e}")


def test_v035_port6_routes_still_register_under_comfyui(r):
    """PORT-6 (the other side): the route block must still run when ComfyUI IS the host.

    The routes are now gated on `"server" in sys.modules` — a membership test that imports
    nothing. Under ComfyUI that is always true by the time custom nodes load (`server` is
    imported and `PromptServer.instance` exists long before `nodes.init_extra_nodes()`), so
    the routes register exactly as before. This drives a fresh interpreter with a STUB
    `server` in `sys.modules` and asserts the TEX routes were actually registered — the
    guard against 'made it lazy, silently lost the API surface'."""
    import subprocess
    import sys as _sys
    import TEX_Wrangle

    custom_nodes = str(pathlib.Path(TEX_Wrangle.__file__).resolve().parent.parent)
    code = (
        "import sys, types\n"
        f"sys.path.insert(0, {custom_nodes!r})\n"
        "try:\n"
        "    import aiohttp  # the route block needs it; skip cleanly if absent\n"
        "except Exception:\n"
        "    print('SKIP no-aiohttp'); raise SystemExit(0)\n"
        "reg = []\n"
        "class _R:\n"
        "    def get(self, p):\n"
        "        def d(f):\n"
        "            reg.append(('GET', p)); return f\n"
        "        return d\n"
        "    def post(self, p):\n"
        "        def d(f):\n"
        "            reg.append(('POST', p)); return f\n"
        "        return d\n"
        "stub = types.ModuleType('server')\n"
        "stub.PromptServer = type('PS', (), {'instance': types.SimpleNamespace(routes=_R())})\n"
        "sys.modules['server'] = stub\n"
        "import TEX_Wrangle\n"
        "print('ROUTES', len(reg))\n"
        "print('SNIPPETS', any(p == '/tex_wrangle/snippets' for _, p in reg))\n"
    )
    try:
        proc = subprocess.run([_sys.executable, "-X", "utf8", "-c", code],
                              capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            r.fail("PORT-6 routes under ComfyUI",
                   f"subprocess exit {proc.returncode}: {(proc.stderr or '')[-400:]}")
            return
        text = proc.stdout.strip()
        if "SKIP" in text:
            r.ok("PORT-6 routes: skipped (aiohttp absent in this environment)")
            return
        out = dict(line.split(" ", 1) for line in text.splitlines() if " " in line)
        assert int(out.get("ROUTES", "0")) > 0, \
            "no TEX routes registered with a ComfyUI-like host present (the gate is too tight)"
        assert out.get("SNIPPETS") == "True", "the /tex_wrangle/snippets route did not register"
        r.ok(f"PORT-6: {out['ROUTES']} routes still register when ComfyUI is the host")
    except Exception as e:
        r.fail("PORT-6 routes under ComfyUI", f"{type(e).__name__}: {e}")


# ── Uniform outputs (LANGUAGE.md §5.2) ────────────────────────────────────────────────
#
# A once-per-cook scalar `@` output — an `f@`/`i@` binding computed only from literals,
# scalar params and the 0-dim builtins — was already cooking identically on every tier and
# route before any of this landed; only the DOC and the PIN are new (no engine file changed).
# These tests convert that from "true by construction, untested" (the design probe's own
# words for six of the twelve covered builtins) into a machine-checked fact, and pin the two
# declined builtins' absence implicitly by never naming them.
#
# The compound program below is the exact one the design probe used: four scalar outputs
# beside a normal `@OUT`, mixing a param, `iw`/`ih`, arithmetic and both `f@`/`i@` prefixes.

_U_CODE = ("@OUT = @A * 0.5;\n"
           "f@dw_x = $cx - 1.0;\n"
           "f@dw_y = max($cx, 0.0);\n"
           "f@dw_w = iw * 0.5;\n"
           "i@dw_h = ih - 1;\n")
_U_NAMES = ("dw_x", "dw_y", "dw_w", "dw_h")


def test_brief9_t1_uniform_output_interp_codegen_equal(r):
    """T1: interpreter and codegen return the SAME 0-dim value for a uniform output.

    This is invariant #2 (interp/codegen bit-exactness) for the uniform-output shape
    specifically: the compound program above, plus one row per named builtin that has a real
    second tier to agree with. `frame`/`fps`/`time` are deliberately NOT rows here — a
    time-reading program never reaches codegen at all (ENG-7's caching decline), so there is
    no second tier for `run_both`'s codegen path to compare against; T2 below covers them by
    checking every `compile_mode` route returns the identical value instead.

    Each single-builtin row keeps a trivial `@A` read (`@OUT = @A * 0.0;`) so the program has
    a real spatial anchor — `iw`/`ih`/`px`/`py`/`fn` are 0-dim either way, but `run_both`'s
    codegen-side env builder (a test-only helper, not the engine) only populates them when a
    binding the program actually reads gives it a shape to derive them from; without one this
    would test an artifact of the test helper, not the engine."""
    import torch
    from helpers import run_both
    try:
        A = torch.rand(1, 6, 8, 4)
        interp_res, cg_res = run_both(_U_CODE, {"A": A, "cx": 10.0})
        assert cg_res is not None, "the compound program did not reach codegen at all"
        for name in _U_NAMES:
            i_t, c_t = interp_res[name], cg_res[name]
            assert tuple(i_t.shape) == (), f"interpreter {name} shape {tuple(i_t.shape)}"
            assert tuple(c_t.shape) == (), f"codegen {name} shape {tuple(c_t.shape)}"
            assert torch.equal(i_t, c_t), \
                f"{name}: interp {i_t.item()} != codegen {c_t.item()}"
        assert tuple(interp_res["OUT"].shape) == (1, 6, 8, 4), tuple(interp_res["OUT"].shape)

        for name in ("iw", "ih", "px", "py", "fn", "ic", "PI", "TAU", "E"):
            row_code = f"@OUT = @A * 0.0;\nf@w = {name};"
            i_res, c_res = run_both(row_code, {"A": A})
            assert c_res is not None, f"{name}: never reached codegen"
            i_t, c_t = i_res["w"], c_res["w"]
            assert tuple(i_t.shape) == (), f"{name}: interpreter shape {tuple(i_t.shape)}"
            assert tuple(c_t.shape) == (), f"{name}: codegen shape {tuple(c_t.shape)}"
            assert torch.equal(i_t, c_t), \
                f"{name}: interp {i_t.item()} != codegen {c_t.item()}"
        r.ok("BRIEF-9 T1: the compound program and 9 named builtins agree, "
             "interpreter vs codegen, 0-dim")
    except Exception as e:
        r.fail("BRIEF-9 T1 interp/codegen parity", f"{type(e).__name__}: {e}")


def test_brief9_t2_uniform_output_stable_across_compile_modes(r):
    """T2: every `compile_mode` route returns the same 0-dim value for a uniform output.

    Closes the gap T1 cannot: `frame`/`fps`/`time` never reach codegen (a time-reading
    program always cooks on the interpreter, ENG-7), so "every route agrees" for them means
    every `compile_mode` still funnels to that one tier and returns the SAME value — not a
    vacuous truth, since a future change to the decline gate could easily let one route
    through while leaving another declined, silently reading a different playhead spelling.
    `cuda_graph` only exists on CUDA and skips cleanly by device, matching house style."""
    import torch
    from TEX_Wrangle import tex_engine
    try:
        A = torch.rand(1, 6, 8, 4)
        modes = ("none", "torch_compile", "auto")
        results = {}
        for m in modes:
            res = tex_engine.cook(_U_CODE, {"A": A, "cx": 10.0}, device_mode="cpu",
                                  compile_mode=m)
            for n in _U_NAMES:
                shp = tuple(res.outputs[n].shape)
                assert shp == (), f"compile_mode={m}: {n} shape {shp}"
            results[m] = res
        for m in modes[1:]:
            for n in _U_NAMES:
                assert torch.equal(results["none"].outputs[n], results[m].outputs[n]), \
                    f"compile_mode={m} vs none diverged on {n}"

        tcode = "f@w_frame = frame;\nf@w_fps = fps;\nf@w_time = time;\n"
        tc = {"frame": 12.0, "fps": 24.0, "time": 0.5}
        tnames = ("w_frame", "w_fps", "w_time")
        tresults = {}
        for m in modes:
            res = tex_engine.cook(tcode, {}, device_mode="cpu", compile_mode=m,
                                  time_context=tc)
            for n in tnames:
                shp = tuple(res.outputs[n].shape)
                assert shp == (), f"compile_mode={m}: {n} shape {shp}"
            tresults[m] = res
        for m in modes[1:]:
            for n in tnames:
                assert torch.equal(tresults["none"].outputs[n], tresults[m].outputs[n]), \
                    f"compile_mode={m} vs none diverged on {n} (frame/fps/time)"

        if torch.cuda.is_available():
            Acuda = A.cuda()
            resg = tex_engine.cook(_U_CODE, {"A": Acuda, "cx": 10.0}, device_mode="cuda",
                                   compile_mode="cuda_graph")
            torch.cuda.synchronize()
            for n in _U_NAMES:
                t = resg.outputs[n]
                assert tuple(t.shape) == (), f"cuda_graph: {n} shape {tuple(t.shape)}"
                assert torch.equal(t.cpu(), results["none"].outputs[n]), \
                    f"cuda_graph vs none diverged on {n}"
        else:
            r.skip("BRIEF-9 T2 cuda_graph route", "no CUDA on this box")
        r.ok("BRIEF-9 T2: uniform outputs are 0-dim and equal across every compile_mode route")
    except Exception as e:
        r.fail("BRIEF-9 T2 route stability", f"{type(e).__name__}: {e}")


def test_brief9_t3_uniform_output_survives_tiled_batch_roi_assemblers(r):
    """T3: the tiled, batch-strip and ROI assemblers pass a uniform output through unchanged.

    Guards against the exact shape of bug this pin exists to catch: an assembler that treats
    EVERY output as spatial (a strip write, a per-batch-chunk concat, a crop) would silently
    broadcast or slice a 0-dim value into something strip- or window-shaped. The `mutation_check.py`
    row added alongside this test reintroduces exactly that at `tex_memory.py`'s `run_tiled`
    passthrough and must turn this row red.

    `shared_tile_height`/`shared_batch_size` are asserted non-None BEFORE calling the
    assembler — the same precondition `run_tiled`/`run_batch_strips` check themselves before
    deciding to tile at all — so a silent whole-frame fallback (which would make this test
    pass without ever exercising a strip) is caught here rather than trusted."""
    import torch
    from TEX_Wrangle.tex_compiler.lexer import Lexer
    from TEX_Wrangle.tex_compiler.parser import Parser
    from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
    from TEX_Wrangle.tex_compiler.types import TEXType
    from TEX_Wrangle.tex_runtime.interpreter import Interpreter
    from TEX_Wrangle import tex_memory
    try:
        def _build():
            prog = Parser(Lexer(_U_CODE).tokenize()).parse()
            checker = TypeChecker(binding_types={"A": TEXType.VEC4, "cx": TEXType.FLOAT,
                                                 "OUT": TEXType.VEC4})
            tm = checker.check(prog)
            outs = sorted(checker.assigned_bindings.keys())
            return prog, tm, outs

        prog, tm, outs = _build()
        interp = Interpreter()

        # run_tiled, 3 strips over a real H=6 image — not a silent whole-frame fallback.
        A = torch.rand(1, 6, 8, 4)
        assert tex_memory.shared_tile_height({"A": A, "cx": 10.0}) == 6, \
            "the binding does not anchor a tile height — this would not exercise run_tiled"
        whole = interp.execute(prog, {"A": A, "cx": 10.0}, tm, device="cpu",
                               output_names=outs)
        tiled = tex_memory.run_tiled(interp, prog, {"A": A, "cx": 10.0}, tm, "cpu", 0,
                                     outs, None, "fp32", 3)
        for n in _U_NAMES:
            assert tuple(tiled[n].shape) == (), f"run_tiled: {n} shape {tuple(tiled[n].shape)}"
            assert torch.equal(tiled[n], whole[n]), f"run_tiled moved {n}"

        # run_batch_strips, B=4, 2 strips.
        Ab = torch.rand(4, 6, 8, 4)
        assert tex_memory.shared_batch_size({"A": Ab, "cx": 10.0}) == 4, \
            "the binding does not anchor a batch size — this would not exercise run_batch_strips"
        wholeb = interp.execute(prog, {"A": Ab, "cx": 10.0}, tm, device="cpu",
                                output_names=outs)
        batched = tex_memory.run_batch_strips(interp, prog, {"A": Ab, "cx": 10.0}, tm, "cpu",
                                              0, outs, None, "fp32", 2)
        for n in _U_NAMES:
            assert tuple(batched[n].shape) == (), \
                f"run_batch_strips: {n} shape {tuple(batched[n].shape)}"
            assert torch.equal(batched[n], wholeb[n]), f"run_batch_strips moved {n}"

        # run_roi, a pointwise program narrowed to a real sub-window (not the whole frame).
        roi = (1, 1, 4, 3, 8, 6)
        roid = tex_memory.run_roi(interp, prog, {"A": A, "cx": 10.0}, tm, "cpu", 0, outs,
                                  None, "fp32", roi, frozenset({"A"}), 0)
        assert tuple(roid["OUT"].shape) == (1, 3, 4, 4), \
            f"run_roi: OUT came back {tuple(roid['OUT'].shape)}, not the (1,3,4,4) window " \
            f"— this would not exercise real ROI narrowing"
        for n in _U_NAMES:
            assert tuple(roid[n].shape) == (), f"run_roi: {n} shape {tuple(roid[n].shape)}"
            assert torch.equal(roid[n], whole[n]), f"run_roi moved {n}"

        r.ok("BRIEF-9 T3: run_tiled/run_batch_strips/run_roi all pass uniform outputs "
             "through unchanged")
    except Exception as e:
        r.fail("BRIEF-9 T3 assemblers", f"{type(e).__name__}: {e}")


def test_brief9_t4_uniform_output_fusion_terminal_and_midchain_refusal(r):
    """T4: a uniform output fuses as a chain's TERMINAL stage, byte-for-byte vs the unfused
    cook, and REFUSES to fuse mid-chain, naming the offending stage.

    The refusal is not a new rule for this shape — `tex_fusion` already refuses any upstream
    stage that writes something besides `@OUT` (only a single-@OUT handoff splices) — so a
    declaring program mid-chain ends fusion the same way any other extra-output stage would.
    This pins that the SAME compound program gets the SAME two outcomes depending only on
    chain position, so a future fusion change can't quietly start mis-splicing it instead of
    refusing it."""
    import torch
    from TEX_Wrangle import tex_engine, tex_fusion
    from TEX_Wrangle.tex_marshalling import infer_binding_type
    try:
        A = torch.rand(1, 6, 8, 4)

        # (a) terminal: fused via cook_stage_list == the same program cooked unfused.
        stage0 = {"code": "@OUT = @A * 0.5;", "bindings": {"A": A}}
        term_code = ("@OUT = @A * 1.0;\n"
                     "f@dw_x = $cx - 1.0;\n"
                     "f@dw_y = max($cx, 0.0);\n"
                     "f@dw_w = iw * 0.5;\n"
                     "i@dw_h = ih - 1;\n")
        stage1 = {"code": term_code, "chain_input": "A", "bindings": {"cx": 10.0}}
        fused = tex_engine.cook_stage_list([stage0, stage1], device="cpu")
        unfused = tex_engine.cook(term_code, {"A": A * 0.5, "cx": 10.0},
                                  device_mode="cpu").outputs
        for n in ("OUT", *_U_NAMES):
            assert tuple(fused[n].shape) == tuple(unfused[n].shape), \
                f"fused/unfused {n} shape: {tuple(fused[n].shape)} vs {tuple(unfused[n].shape)}"
            assert torch.equal(fused[n], unfused[n]), f"fused terminal moved {n}"

        # (b) mid-chain (stage 0 of 2): refused, naming stage 0.
        stage0b = {"code": _U_CODE, "bindings": {"A": A, "cx": 10.0}}
        stage1b = {"code": "@OUT = @A * 2.0;", "chain_input": "A", "bindings": {}}
        result = tex_fusion.chain_preflight([stage0b, stage1b], infer_binding_type)
        assert result["ok"] is False, f"a mid-chain declaration was not refused: {result}"
        assert result["stage_of_error"] == 0, \
            f"the refusal named stage {result['stage_of_error']!r}, not 0: {result['error']!r}"

        r.ok("BRIEF-9 T4: a uniform output fuses as a terminal stage exactly like the "
             "unfused cook, and is refused (naming stage 0) mid-chain")
    except Exception as e:
        r.fail("BRIEF-9 T4 fusion", f"{type(e).__name__}: {e}")


def test_brief9_t5_uniform_output_param_query_never_recompiles_moves_lineage(r):
    """T5: the QUERY half. A host that already knows an input's window passes it in as an
    ordinary `$param` — no new syntax. Sweeping it must never recompile (ANIM-1, applied to
    this specific shape) and must move the produced frame's lineage key (CACHE-1), since the
    value it carries moves pixels and a lineage key that didn't move would silently serve a
    stale window on the next cook.

    Spies patch the same three choke points `test_v031_anim_contract.py` does
    (`TEXCache.compile_ast`, `compiled._try_codegen`, `graphed.GraphedProgram.capture`) so a
    counter that could never move is not mistaken for a guarantee that holds."""
    import torch
    from TEX_Wrangle import tex_engine
    from TEX_Wrangle.tex_cache import TEXCache, get_cache
    try:
        from TEX_Wrangle.tex_runtime import compiled as _C, graphed as _G
        A = torch.rand(1, 6, 8, 4)
        code = "@OUT = @A * 0.5 + vec4($dw_x, $dw_y, $dw_w, $dw_h);"
        cache = get_cache()
        counts = {"compiles": 0, "emissions": 0, "captures": 0}
        orig = (TEXCache.compile_ast, _C._try_codegen, _G.GraphedProgram.capture)

        def compile_ast(self_, *a, **k):
            counts["compiles"] += 1
            return orig[0](self_, *a, **k)

        def try_codegen(*a, **k):
            counts["emissions"] += 1
            return orig[1](*a, **k)

        def capture(self_, *a, **k):
            counts["captures"] += 1
            return orig[2](self_, *a, **k)

        TEXCache.compile_ast, _C._try_codegen, _G.GraphedProgram.capture = \
            compile_ast, try_codegen, capture
        try:
            base_binds = {"A": A, "dw_x": 0.0, "dw_y": 0.0, "dw_w": 0.0, "dw_h": 0.0}
            first = tex_engine.cook(code, base_binds, device_mode="cpu",
                                    compile_mode="auto", want_lineage=True)
            after_cold = dict(counts)
            fps_before = set(cache._memory.keys())
            keys_seen = {first.lineage["OUT"]}
            for i in range(1, 6):
                res = tex_engine.cook(code, {**base_binds, "dw_x": float(i)},
                                      device_mode="cpu", compile_mode="auto",
                                      want_lineage=True)
                keys_seen.add(res.lineage["OUT"])
            new_fps = set(cache._memory.keys()) - fps_before
            moved = (counts["compiles"] - after_cold["compiles"],
                    counts["emissions"] - after_cold["emissions"],
                    counts["captures"] - after_cold["captures"])
        finally:
            TEXCache.compile_ast, _C._try_codegen, _G.GraphedProgram.capture = orig

        assert moved == (0, 0, 0), f"sweeping $dw_x recompiled: {moved}"
        assert not new_fps, f"sweeping $dw_x created a new program fingerprint: {new_fps}"
        assert len(keys_seen) == 6, \
            f"the OUT lineage key did not move with every $dw_x value: only " \
            f"{len(keys_seen)} distinct keys over 6 cooks"
        r.ok("BRIEF-9 T5: sweeping a host-bound window $param never recompiles and moves "
             "the output's lineage key")
    except Exception as e:
        r.fail("BRIEF-9 T5 query sweep", f"{type(e).__name__}: {e}")


def test_brief9_t6_uniform_output_fp32_exact_at_3841(r):
    """T6: a uniform output holds an integer value exactly at fp32, and CPU `precision="auto"`
    (which resolves fp32 on CPU regardless of resolution — the gate never splits there, CF-6)
    agrees. `3841` is chosen because it is exactly the boundary the design probe measured:
    fp16 is integer-exact only to 2048, so `precision="fp16"` would round it to `3840.0` —
    the promise this pin makes is deliberately narrower than "any tier", and says so."""
    import torch
    from TEX_Wrangle import tex_engine
    try:
        code = "f@dw_x = $cx;"
        for precision in ("fp32", "auto"):
            res = tex_engine.cook(code, {"cx": 3841.0}, device_mode="cpu",
                                  compile_mode="none", precision=precision)
            t = res.outputs["dw_x"]
            assert tuple(t.shape) == (), f"precision={precision}: shape {tuple(t.shape)}"
            assert float(t) == 3841.0, f"precision={precision}: {float(t)} != 3841.0"
        r.ok("BRIEF-9 T6: a uniform output holds $cx=3841.0 exactly at fp32 and CPU auto")
    except Exception as e:
        r.fail("BRIEF-9 T6 fp32 exactness", f"{type(e).__name__}: {e}")


def test_brief9_t7_language_md_documents_uniform_outputs(r):
    """T7: the guarantee T1-T6 pin is written down. Modeled on
    `test_v031_anim_contract_is_documented` — the doc and the behaviour cannot drift apart
    silently if a doc edit that drops a promise also drops this test's evidence for it."""
    import pathlib
    try:
        root = pathlib.Path(__file__).resolve().parent.parent
        txt = (root / "LANGUAGE.md").read_text(encoding="utf-8", errors="ignore")
        need = [
            "### 5.2 Uniform outputs",
            "iw ih px py fn ic PI TAU E frame fps time",   # the covered builtins, verbatim
            "v2@",                                          # exclusion 1: broadcast vec outputs
            "u v ix iy fi",                                  # exclusion 2: per-pixel/batch reads
            "rank disagrees across tiers",                   # exclusion 3: vec params
        ]
        missing = [n for n in need if n not in txt]
        r.ok("LANGUAGE.md documents uniform outputs (§5.2) and its three exclusions") \
            if not missing else \
            r.fail("BRIEF-9 T7 doc pin", f"LANGUAGE.md is missing: {missing}")
    except Exception as e:
        r.fail("BRIEF-9 T7 doc pin", f"{type(e).__name__}: {e}")
