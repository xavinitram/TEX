"""v0.30 Phase 1 — First viewer: the ROI flag flip (per-cook arm + window validation).

v0.30 turns `roi=` into a production path, so three things must be pinned:

  * the ARM is per cook (`cook(roi=..., roi_exec=True)`), not a new global default — a host
    wants an ROI viewport cook and a whole-frame final render in one process, and the
    ComfyUI node must keep cooking exactly as it did in v0.29 (invariant #7);
  * a MALFORMED window falls back to the whole frame instead of returning wrong pixels.
    Two cases measured this silently before v0.30: an overhanging window came back 4x4 for a
    requested 10x10, and a negative origin produced a wrong-sized, wrong-pixel crop;
  * the host can TELL which it got (`CookResult.cooked_roi`) — shapes alone are ambiguous.

Shapes (roadmap §10.4): CANARY for the arm precedence and the ComfyUI contract, NEVER-SEVER
ROWS for the malformed windows, DERIVATION for the nightly-gate wiring.
"""
import os as _os

from helpers import *

_TOL = 1e-5


def _cook(code, binds, **kw):
    from TEX_Wrangle import tex_engine
    return tex_engine.cook(code, dict(binds), device_mode="cpu", **kw)


def test_v030_roi_host_optin(r: SubTestResult):
    print("\n--- v0.30: per-cook ROI arm (roi_exec) + cooked_roi reporting ---")
    from TEX_Wrangle import tex_roi as _R
    W, H = 32, 24
    roi = (8, 6, 10, 9, W, H)
    x0, y0, w, h, _W, _H = roi
    prev_env = _os.environ.get("TEX_ROI_EXEC")
    code = "@OUT = gauss_blur(@A, 2.0);"
    try:
        torch.manual_seed(11)
        A = torch.rand(1, H, W, 4)
        _os.environ.pop("TEX_ROI_EXEC", None)          # the shipped default: no env, no arm
        _R.clear_roi_memo()
        full = _cook(code, {"A": A}).outputs["OUT"]

        # (1) DEFAULT — a window with no arm is ignored. This is the invariant-#7 guarantee.
        d = _cook(code, {"A": A}, roi=roi)
        ok = tuple(d.outputs["OUT"].shape) == (1, H, W, 4) and d.cooked_roi is None
        r.ok("default (no roi_exec, no env) → whole frame, cooked_roi=None") if ok else \
            r.fail("v0.30 default", f"shape {tuple(d.outputs['OUT'].shape)} cooked_roi={d.cooked_roi}")

        # (2) ARMED per cook, with NO env var set → real ROI cook, and the pixels must equal
        #     the whole-frame crop (this is the assertion that cannot pass vacuously).
        _R.clear_roi_memo()
        a = _cook(code, {"A": A}, roi=roi, roi_exec=True)
        got = a.outputs["OUT"]
        if tuple(got.shape) == (1, h, w, 4) and a.cooked_roi == roi:
            md = (full[:, y0:y0 + h, x0:x0 + w].float() - got.float()).abs().max().item()
            r.ok(f"roi_exec=True → ROI cook, cooked_roi set (maxdiff {md:.1e})") if md < _TOL else \
                r.fail("v0.30 arm pixels", f"maxdiff {md:.2e}")
        else:
            r.fail("v0.30 arm", f"shape {tuple(got.shape)} cooked_roi={a.cooked_roi}")

        # (3) The per-cook refusal BEATS the env — a host can hold v0.29 behaviour explicitly.
        _os.environ["TEX_ROI_EXEC"] = "1"
        _R.clear_roi_memo()
        f2 = _cook(code, {"A": A}, roi=roi, roi_exec=False)
        ok = tuple(f2.outputs["OUT"].shape) == (1, H, W, 4) and f2.cooked_roi is None
        r.ok("roi_exec=False overrides TEX_ROI_EXEC=1 → whole frame") if ok else \
            r.fail("v0.30 refusal", f"shape {tuple(f2.outputs['OUT'].shape)}")

        # (4) roi_exec=None reads the env (the CI / nightly-oracle / rollback channel).
        _R.clear_roi_memo()
        e = _cook(code, {"A": A}, roi=roi)
        r.ok("roi_exec=None → env fallback (TEX_ROI_EXEC=1 arms it)") \
            if e.cooked_roi == roi else r.fail("v0.30 env fallback", f"cooked_roi={e.cooked_roi}")

        # (5) Arming with no window is a documented no-op, not an error.
        _R.clear_roi_memo()
        n = _cook(code, {"A": A}, roi_exec=True)
        ok = tuple(n.outputs["OUT"].shape) == (1, H, W, 4) and n.cooked_roi is None
        r.ok("roi_exec=True with roi=None → no-op whole frame") if ok else \
            r.fail("v0.30 arm-without-window", f"cooked_roi={n.cooked_roi}")
    except Exception as e:
        r.fail("v0.30 ROI per-cook arm", f"{type(e).__name__}: {e}")
    finally:
        if prev_env is None:
            _os.environ.pop("TEX_ROI_EXEC", None)
        else:
            _os.environ["TEX_ROI_EXEC"] = prev_env
        _R.clear_roi_memo()


def test_v030_roi_malformed_windows(r: SubTestResult):
    print("\n--- v0.30: malformed roi= windows fall back whole-frame (never wrong pixels) ---")
    # NEVER-SEVER ROWS. The first two rows are the cases measured returning WRONG PIXELS
    # before v0.30: an overhang silently shrank the output (asked 10x10, got 4x4), and a
    # negative origin produced a wrong-sized crop through a negative slice offset.
    from TEX_Wrangle import tex_roi as _R
    W, H = 32, 24
    # POINTWISE on purpose. With a coordinate-reading program these rows passed for the wrong
    # reason: deleting the extent check made the narrow raise a shape error, the engine caught
    # it into a whole-frame fallback, and the assertions still held — so the check itself was
    # untested (mutation-verified: removing it left the suite green). A pointwise program has
    # no coordinate mismatch to raise, so a missing check yields a WRONG-SIZED result and the
    # rows fail, which is the point of having them.
    code = "@OUT = @A * 0.5;"
    rows = [
        ((28, 20, 10, 10, W, H), "overhangs right+bottom (measured: returned 4x4 for 10x10)"),
        ((-4, 0, 8, 8, W, H),    "negative origin (measured: wrong-sized, wrong-pixel crop)"),
        ((0, -2, 8, 8, W, H),    "negative y origin"),
        ((4, 4, 0, 8, W, H),     "zero width"),
        ((4, 4, 8, -1, W, H),    "negative height"),
        ((0, 0, 8, 8, 64, 64),   "window's (W,H) disagrees with the binding shape"),
        ((0.0, 0, 8, 8, W, H),   "non-int entry"),
        ((0, 0, 8, 8, W),        "wrong arity"),
    ]
    prev_env = _os.environ.get("TEX_ROI_EXEC")
    try:
        torch.manual_seed(3)
        A = torch.rand(1, H, W, 4)
        _os.environ.pop("TEX_ROI_EXEC", None)
        _R.clear_roi_memo()
        full = _cook(code, {"A": A}).outputs["OUT"]
        bad_rows = []
        for roi, why in rows:
            _R.clear_roi_memo()
            try:
                res = _cook(code, {"A": A}, roi=roi, roi_exec=True)
            except Exception as exc:            # a malformed window must never raise either
                bad_rows.append(f"{why}: raised {type(exc).__name__}")
                continue
            out = res.outputs["OUT"]
            if tuple(out.shape) != (1, H, W, 4):
                bad_rows.append(f"{why}: shape {tuple(out.shape)} != whole frame")
            elif res.cooked_roi is not None:
                bad_rows.append(f"{why}: claimed cooked_roi={res.cooked_roi}")
            elif (full.float() - out.float()).abs().max().item() > _TOL:
                bad_rows.append(f"{why}: pixels differ from the whole-frame cook")
        if bad_rows:
            r.fail("v0.30 malformed windows", "; ".join(bad_rows))
        else:
            r.ok(f"all {len(rows)} malformed windows → whole frame, cooked_roi=None, exact pixels")
    except Exception as e:
        r.fail("v0.30 malformed windows", f"{type(e).__name__}: {e}")
    finally:
        if prev_env is None:
            _os.environ.pop("TEX_ROI_EXEC", None)
        else:
            _os.environ["TEX_ROI_EXEC"] = prev_env
        _R.clear_roi_memo()


def test_v030_roi_extent_per_binding(r: SubTestResult):
    print("\n--- v0.30: the window's extent is checked per BINDING, not against a shared size ---")
    # This is the row that actually tests the extent check, and it needs a specific shape to do
    # it: the ANCHOR binding must be full-size (so the broadcast-anchor guard stays quiet) while
    # a LATER binding disagrees. Mutation-verified — deleting the check turns this red, whereas
    # the malformed-window table stayed green because its mismatch is on the first binding and
    # the anchor guard catches that one first.
    #
    # The bug it pins: a shared-size lookup returns None whenever an axis has several distinct
    # non-singleton sizes, and folding that into one all-or-nothing test dropped BOTH axes'
    # checks. A 4x4 window of a claimed 8x4 image then came back 32 wide, reporting success,
    # because the 32-wide binding was never narrowed (`shape[2] != W`) and passed through whole.
    from TEX_Wrangle import tex_engine, tex_roi as _R
    from TEX_Wrangle.tex_memory import shared_tile_width
    try:
        torch.manual_seed(2)
        A = torch.rand(1, 16, 32, 4)      # anchor: full-size, so the anchor guard is satisfied
        B = torch.rand(1, 16, 8, 4)       # a later binding the window does NOT describe
        code = "@OUT = @A * 0.5;"         # reads only @A, so the whole-frame cook is well-defined
        _R.clear_roi_memo()
        if shared_tile_width({"A": A, "B": B}) is not None:
            r.fail("v0.30 extent probe", "the probe needs an unresolvable shared width")
            return
        full = tex_engine.cook(code, {"A": A, "B": B}, device_mode="cpu").outputs["OUT"]
        res = tex_engine.cook(code, {"A": A, "B": B}, device_mode="cpu",
                              roi=(0, 0, 4, 4, 32, 16), roi_exec=True)
        out = res.outputs["OUT"]
        if res.cooked_roi is not None:
            r.fail("v0.30 extent per-binding", f"claimed cooked_roi={res.cooked_roi} for a window "
                                               f"a binding does not match")
        elif tuple(out.shape) != tuple(full.shape):
            r.fail("v0.30 extent per-binding", f"returned {tuple(out.shape)}, not the whole frame "
                                               f"{tuple(full.shape)}")
        elif not torch.equal(full, out):
            r.fail("v0.30 extent per-binding", "fallback pixels differ from the whole-frame cook")
        else:
            r.ok("a wrong-sized NON-anchor binding refuses the window (whole frame, cooked_roi=None)")
    except Exception as e:
        r.fail("v0.30 extent per-binding", f"{type(e).__name__}: {e}")


def test_v030_roi_broadcast_anchor(r: SubTestResult):
    print("\n--- v0.30: a BROADCAST-singleton anchor refuses the window (the two would disagree) ---")
    # The sibling of the extent row above, and the one the extent check cannot cover: here the
    # anchor binding is a legal broadcast (`[1,1,W,4]` — one row, stretched over y), so the
    # per-dim extent loop passes it (1 is always allowed) and only the anchor guard is left.
    # Mutation-verified: stubbing `if record_trace and _anchor != (W, H)` to `if False` turns
    # this red. Without it the guard had NO test at all.
    #
    # What it pins: `Interpreter._determine_spatial_shape` sizes the whole-frame grid from the
    # FIRST spatial binding INCLUDING singletons, so the whole-frame cook of this program
    # collapses `v`/`iy`/`ih` to a single row while the ROI cook grids the real window. The ROI
    # answer is the correct one, which is exactly the problem — the two disagree (measured
    # maxdiff 0.60) and `cooked_roi` would report success on a window that does not match its
    # own whole-frame cook. Refusing is the conservative half: the caller keeps v0.29's answer,
    # right or wrong, instead of getting two different ones from the same program. Fixing the
    # root cause is a default-path PIXEL change and ships on its own, not inside an ROI release.
    from TEX_Wrangle import tex_engine, tex_roi as _R
    try:
        torch.manual_seed(3)
        A = torch.rand(1, 1, 32, 4)       # anchor: a broadcast singleton in y
        code = "@OUT = @A * vec4(v, v, v, 1.0);"   # reads `v`, so the collapse is observable
        _R.clear_roi_memo()
        full = tex_engine.cook(code, {"A": A}, device_mode="cpu").outputs["OUT"]
        res = tex_engine.cook(code, {"A": A}, device_mode="cpu",
                              roi=(0, 0, 8, 8, 32, 16), roi_exec=True)
        out = res.outputs["OUT"]
        if res.cooked_roi is not None:
            r.fail("v0.30 broadcast anchor", f"claimed cooked_roi={res.cooked_roi} on a window "
                                             f"whose whole-frame cook is grid-collapsed")
        elif tuple(out.shape) != tuple(full.shape) or not torch.equal(full, out):
            r.fail("v0.30 broadcast anchor", f"fallback returned {tuple(out.shape)}, not the "
                                             f"whole-frame {tuple(full.shape)} pixel-for-pixel")
        else:
            r.ok("a broadcast-singleton anchor refuses the window (whole frame, cooked_roi=None)")
    except Exception as e:
        r.fail("v0.30 broadcast anchor", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


def test_v030_roi_refusals_stay_off_the_default_path(r: SubTestResult):
    print("\n--- v0.30: the ROI refusals never fire on the DEFAULT cook path (invariant #7) ---")
    # NEVER-SEVER. The two rows here are the two ways v0.30's new refusals leaked into cooks
    # that never asked for an ROI. Both were real: the first shipped broken and was caught by a
    # regression hunt, the second is the reason the whole-frame test lives in the engine gate.
    from TEX_Wrangle import tex_engine as _TE, tex_roi as _R
    from TEX_Wrangle.tex_memory import run_tiled_halo

    # (a) `run_tiled_halo` — the OOM ladder — drives `run_roi` per STRIP to assemble a whole
    # frame. The broadcast-anchor refusal must not fire there: a strip is not a host's window,
    # there is no second answer for a host to disagree with, and refusing sends every strip to
    # the whole-frame path, which is the grid-COLLAPSED cook. Measured with the refusal ungated:
    # `RuntimeError: expanded size (1) must match 96` on a cook with no roi and no roi_exec.
    #
    # The oracle has to be a program whose collapse is VISIBLE, or the refusal is invisible: a
    # pure direct-tensor chain cooks whole-frame just fine and only wastes work. So read `v`,
    # and compare against the same cook with the anchor EXPANDED to full size — broadcasting
    # makes the two inputs equivalent, so the results must be bit-identical, and the halo comes
    # from the plan (a hand-picked one is not seam-exact and would mask the signal at 4e-02).
    # Mutation-verified: removing `record_trace and` moves this from 0.0 to 6.7e-01.
    try:
        torch.manual_seed(5)
        A_s = torch.rand(1, 1, 80, 4)                     # anchor: a broadcast singleton in y
        A_f = A_s.expand(1, 96, 80, 4).contiguous()       # …the same pixels, stated in full
        B = torch.rand(1, 96, 80, 4)
        code = "@OUT = gauss_blur(@B, 2) * vec4(v, v, v, 1.0) + @A * 0.25;"
        _R.clear_roi_memo()
        plan = _R.roi_plan(code, {})
        ref = _TE.cook(code, {"A": A_f, "B": B}, device_mode="cpu").outputs["OUT"]
        _R.clear_roi_memo()
        ctx = _TE.prepare(code, {"A": A_s, "B": B}, device_mode="cpu").ctx
        out = run_tiled_halo(_TE._get_interpreter(), ctx.program, ctx.bindings, ctx.type_map,
                             ctx.device, ctx.latent_channel_count, ctx.output_names,
                             ctx.used_builtins, ctx.eff_precision, 2, list(plan.narrow),
                             plan.halo)["OUT"]
        if tuple(out.shape) != tuple(ref.shape):
            r.fail("v0.30 OOM-ladder strips", f"assembled {tuple(out.shape)}, not the whole "
                                              f"frame {tuple(ref.shape)}")
        elif not torch.equal(out, ref):
            r.fail("v0.30 OOM-ladder strips",
                   f"the tiled cook drifted {(out - ref).abs().max().item():.3e} from the "
                   f"untiled one — an ROI refusal leaked into the OOM ladder")
        else:
            r.ok("run_tiled_halo stays seam-exact with a broadcast-singleton anchor")
    except Exception as e:
        r.fail("v0.30 OOM-ladder strips",
               f"a DEFAULT-path tiled cook raised: {type(e).__name__}: {e}")

    # (b) A window covering the WHOLE frame narrows nothing, so arming it is all cost: the
    # plan, the narrow-cook-crop round trip, and the ROI fp32 clamp — which drops an
    # fp16-eligible cook out of fp16 for byte-identical output. A zoom-to-fit viewport sends
    # exactly this window every frame. It must be declined (`cooked_roi is None`), not served.
    try:
        torch.manual_seed(6)
        A = torch.rand(1, 16, 32, 4)
        _R.clear_roi_memo()
        res = _TE.cook("@OUT = @A * 0.5;", {"A": A}, device_mode="cpu",
                              roi=(0, 0, 32, 16, 32, 16), roi_exec=True)
        if res.cooked_roi is not None:
            r.fail("v0.30 whole-frame window", f"armed ROI for a whole-frame window "
                                               f"(cooked_roi={res.cooked_roi}) — pure overhead")
        elif not torch.equal(res.outputs["OUT"], A * 0.5):
            r.fail("v0.30 whole-frame window", "the declined cook returned wrong pixels")
        else:
            r.ok("a whole-frame window is declined, not armed (no plan, no fp32 clamp)")
    except Exception as e:
        r.fail("v0.30 whole-frame window", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


def test_v030_roi_folded_binding_still_narrows(r: SubTestResult):
    print("\n--- v0.30: a $param-FOLDED-away binding is still narrowed (the whole-frame lie) ---")
    # NEVER-SEVER, and the worst bug the v0.30 hunt found. `roi_plan` folds `$params` before it
    # tallies binding reads, so `mix(@A, @B, $k)` at k=0.0 folds to `@A` and @B never reached
    # the plan's `narrow` set. The ENGINE does not fold $params — they are runtime bindings and
    # the compile cache is keyed on their TYPES, not their values — so the program that ran
    # still read @B, at FULL extent. Measured: a (5,5,1,1) window of a 64x48 image returned the
    # whole (1,48,64,4) frame while `cooked_roi` reported the window as SERVED, and the shipped
    # host blitted it: "expanded size (1) must match 64".
    #
    # Two independent guards now stop it, and this pins both: the plan includes folded-away
    # bindings, and `run_roi` checks the OUTPUT extent before marking the window served — so
    # `cooked_roi` can never name a window the outputs do not have, whatever the cause.
    from TEX_Wrangle import tex_engine, tex_roi as _R
    W, H = 64, 48
    torch.manual_seed(11)
    A, B = torch.rand(1, H, W, 4), torch.rand(1, H, W, 4)
    cases = [("@OUT = mix(@A, @B, $k);", {"k": 0.0}),
             ("@OUT = mix(@A, @B, $k);", {"k": 1.0}),
             ("@OUT = lerp(@A, @B, $k);", {"k": 0.0}),
             ("@OUT = ($op > 0.5) ? @B : @A;", {"op": 0.0}),
             ("@OUT = mix(gauss_blur(@A, 2.0), @B, $k);", {"k": 0.0})]
    windows = [(5, 5, 1, 1, W, H), (0, 0, W, 1, W, H), (7, 0, 1, H, W, H), (8, 8, 16, 16, W, H)]
    bad = []
    try:
        for code, ps in cases:
            _R.clear_roi_memo()
            plan = _R.roi_plan(code, ps)
            if plan.executable and not {"A", "B"} <= set(plan.narrow):
                bad.append(f"{code} {ps}: narrow={sorted(plan.narrow)} drops a live binding")
            if "OUT" in plan.narrow:
                bad.append(f"{code}: narrow names the WRITE target OUT")
            full = tex_engine.cook(code, {"A": A, "B": B, **ps}, device_mode="cpu",
                                   precision="fp32").outputs["OUT"]
            for win in windows:
                res = tex_engine.cook(code, {"A": A, "B": B, **ps}, device_mode="cpu",
                                      precision="fp32", roi=win, roi_exec=True)
                out, (x0, y0, w, h, _W, _H) = res.outputs["OUT"], win
                if res.cooked_roi is not None:
                    want, ref = (1, h, w, 4), full[:, y0:y0 + h, x0:x0 + w]
                else:
                    want, ref = tuple(full.shape), full        # declined → the whole frame
                if tuple(out.shape) != want:
                    bad.append(f"{code} {ps} roi={win[:4]}: cooked_roi="
                               f"{'SERVED' if res.cooked_roi else 'declined'} but shape "
                               f"{tuple(out.shape)} != {want}")
                elif not torch.equal(out, ref):
                    bad.append(f"{code} {ps} roi={win[:4]}: pixels differ from the whole-frame "
                               f"cook by {(out - ref).abs().max().item():.3e}")
        if bad:
            r.fail("v0.30 folded-binding narrow", f"{len(bad)} case(s): " + "; ".join(bad[:3]))
        else:
            r.ok(f"{len(cases)}x{len(windows)} folded-binding windows: extent and pixels agree "
                 f"with the whole-frame cook")

        # The BACKSTOP, tested on its own. Fixing the plan makes the postcondition unreachable
        # through the engine — so drive `run_roi` with a deliberately incomplete `narrow_names`,
        # which is exactly the shape of the bug above and of the next analysis gap. The window
        # must be abandoned (whole frame, cooked_roi None), never served at the wrong extent.
        from TEX_Wrangle.tex_memory import run_roi
        from TEX_Wrangle.tex_runtime import tier_trace as _tt
        from TEX_Wrangle.tex_runtime.interpreter import Interpreter
        code = "@OUT = @A * 0.5 + @B * 0.25;"
        _R.clear_roi_memo()
        prep = tex_engine.prepare(code, {"A": A, "B": B}, device_mode="cpu", precision="fp32")
        ctx, it = prep.ctx, Interpreter()
        # A 1x1 window, which is what made the original bug SILENT rather than loud: the
        # narrowed binding becomes broadcastable, so the un-narrowed one propagates its full
        # extent into the output instead of raising a shape error the engine would have caught.
        win = (5, 5, 1, 1, W, H)
        _tt.reset()
        out = run_roi(it, ctx.program, dict(ctx.bindings), ctx.type_map, "cpu", 0,
                      ctx.output_names, ctx.used_builtins, "fp32", win,
                      frozenset({"A"}), 0)["OUT"]          # @B deliberately left un-narrowed
        served, _why = _tt.last_roi()
        if served is not None:
            r.fail("v0.30 extent postcondition",
                   f"marked {served} served while the output came back "
                   f"{tuple(out.shape)} — cooked_roi can name a window it does not have")
        elif tuple(out.shape) != (1, H, W, 4):
            r.fail("v0.30 extent postcondition",
                   f"abandoned the window but returned {tuple(out.shape)}, not the whole frame")
        else:
            r.ok("an incomplete narrow set abandons the window instead of claiming it served")
    except Exception as e:
        r.fail("v0.30 folded-binding narrow", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


def test_v030_roi_window_is_copied_and_coerced(r: SubTestResult):
    print("\n--- v0.30: the reported window is the ENGINE's own tuple, and roi_exec is coerced ---")
    from TEX_Wrangle import tex_engine, tex_roi as _R
    W, H = 32, 24
    torch.manual_seed(12)
    A = torch.rand(1, H, W, 4)
    code = "@OUT = @A * 0.5;"
    try:
        # A viewport that recycles ONE rect buffer per frame is the obvious way to write one.
        # `cooked_roi` used to BE that object, so mutating it retroactively changed the window
        # the engine had already reported and the next blit landed in the wrong place.
        _R.clear_roi_memo()
        rect = [4, 4, 8, 8, W, H]
        res = tex_engine.cook(code, {"A": A}, device_mode="cpu", roi=rect, roi_exec=True)
        snapshot = tuple(res.cooked_roi)
        rect[0] = 99
        if not isinstance(res.cooked_roi, tuple):
            r.fail("v0.30 cooked_roi type", f"a list roi= yielded {type(res.cooked_roi).__name__}, "
                                            f"not the documented 6-tuple")
        elif tuple(res.cooked_roi) != snapshot:
            r.fail("v0.30 cooked_roi aliasing",
                   f"the host mutating its own rect changed cooked_roi to {res.cooked_roi}")
        else:
            r.ok("cooked_roi is a plain-int tuple the host cannot mutate underneath the engine")

        # `roi_exec=False` is documented as an explicit kill switch, but a STRING is truthy —
        # so "0"/"false"/"off", the spellings a host reads from config, all ARMED the path.
        offs = [_R.roi_exec_enabled(v) for v in ("0", "false", "off", "FALSE", " off ")]
        ons = [_R.roi_exec_enabled(v) for v in ("1", "true", "on", "TRUE")]
        if any(offs) or not all(ons) or _R.roi_exec_enabled(False) or not _R.roi_exec_enabled(True):
            r.fail("v0.30 roi_exec coercion",
                   f"off-spellings -> {offs}, on-spellings -> {ons} (want all False / all True)")
        else:
            r.ok("roi_exec coerces '0'/'false'/'off' to a real kill switch")

        # A host computing its window with numpy hands over int64, which is not an `int`.
        # Refusing those silently dropped exactly those hosts back to whole-frame cooks.
        class _Idx:                       # stands in for numpy.int64 (numpy is absent in CI)
            def __init__(self, v): self.v = v
            def __index__(self): return self.v
        boxed = tuple(_Idx(v) for v in (2, 2, 8, 8, W, H))
        _R.clear_roi_memo()
        res2 = tex_engine.cook(code, {"A": A}, device_mode="cpu", roi=boxed, roi_exec=True)
        if _R.validate_roi(boxed) is not None or res2.cooked_roi != (2, 2, 8, 8, W, H):
            r.fail("v0.30 integer coercion", f"an __index__ window was refused "
                                             f"({_R.validate_roi(boxed)}, {res2.cooked_roi})")
        elif tuple(res2.outputs["OUT"].shape) != (1, 8, 8, 4):
            r.fail("v0.30 integer coercion", f"shape {tuple(res2.outputs['OUT'].shape)}")
        elif _R.validate_roi((True, 0, 8, 8, W, H)) is None or \
                _R.validate_roi((0.5, 0, 8, 8, W, H)) is None:
            r.fail("v0.30 integer coercion", "a bool or float coordinate was accepted")
        else:
            r.ok("any __index__ integer is accepted; bools and floats still refused")
    except Exception as e:
        r.fail("v0.30 window copy/coercion", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


def test_v030_roi_never_desyncs_from_its_canvas(r: SubTestResult):
    print("\n--- v0.30: an ROI patch always matches the canvas it is composited into ---")
    # DIFFERENTIAL ORACLE. v0.30's whole use case is compositing a WINDOW cook into a canvas
    # produced by a WHOLE-FRAME cook, so the invariant is not "the ROI cook is accurate" — it is
    # "the two agree". They did not: the ROI arm clamped the window to fp32 (ROI is only
    # oracle-validated there) while the canvas cook stayed fp16, and the pair disagreed by an
    # fp16 ulp — measured 1.05e-03 max, 47% of pixels past 1e-4, on CUDA at 1024^2 under
    # precision="auto". A clamp is conservative for a whole cook and NOT for half of a matched
    # pair. The window is now declined instead whenever the cook is not fp32.
    #
    # Asserting agreement rather than "declines under fp16" keeps the test honest if the ROI
    # oracle is ever extended to fp16 and the decline becomes a serve.
    from TEX_Wrangle import tex_engine, tex_roi as _R
    n = 256
    torch.manual_seed(13)
    code = "@OUT = vec4(@A.rgb * 1.7 + vec3(0.05), 1.0);"
    win = (64, 64, 64, 64, n, n)
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    bad = []
    try:
        for dev in devs:
            A = torch.rand(1, n, n, 4, device=dev)
            for prec in ("auto", "fp32", "fp16"):
                _R.clear_roi_memo()
                whole = tex_engine.cook(code, {"A": A}, device_mode=dev, precision=prec)
                _R.clear_roi_memo()
                res = tex_engine.cook(code, {"A": A}, device_mode=dev, precision=prec,
                                      roi=win, roi_exec=True)
                if res.cooked_roi is not None:
                    ref = whole.outputs["OUT"][:, 64:128, 64:128]
                else:
                    ref = whole.outputs["OUT"]               # declined → the whole frame
                out = res.outputs["OUT"]
                if res.precision != whole.precision:
                    bad.append(f"{dev}/{prec}: window ran {res.precision}, canvas ran "
                               f"{whole.precision}")
                elif tuple(out.shape) != tuple(ref.shape) or not torch.equal(out, ref):
                    bad.append(f"{dev}/{prec}: patch differs from its canvas by "
                               f"{(out.float() - ref.float()).abs().max().item():.3e}")
        if bad:
            r.fail("v0.30 roi/canvas precision", "; ".join(bad))
        else:
            r.ok(f"{len(devs)}x3 device/precision combinations: the window and its canvas "
                 f"ran the same precision and agree bit-for-bit")

        # …and the MECHANISM, separately. Agreement alone does not pin the decline: cooking the
        # window at fp16 also agrees (both halves are then fp16), so that assertion stays green
        # if the refusal is deleted. The refusal exists because ROI is oracle-validated at fp32
        # ONLY — an unvalidated fp16 window is the thing being refused, not a mismatch that has
        # already been measured away. Pin the contract itself, or it reverts silently.
        for dev in devs:
            A = torch.rand(1, n, n, 4, device=dev)
            _R.clear_roi_memo()
            res = tex_engine.cook(code, {"A": A}, device_mode=dev, precision="fp16",
                                  roi=win, roi_exec=True)
            if res.cooked_roi is not None:
                r.fail("v0.30 fp16 declines the window",
                       f"{dev}: served {res.cooked_roi} at {res.precision} — ROI is validated "
                       f"at fp32 only")
                break
        else:
            r.ok("an fp16 cook declines the window rather than cooking it unvalidated")
    except Exception as e:
        r.fail("v0.30 roi/canvas precision", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


def test_v030_roi_accuracy_envelope(r: SubTestResult):
    print("\n--- v0.30: the MEASURED narrow-cook-crop envelope, by program class ---")
    # "Bit-exact" was the shipped claim, and it is true for the class it was measured on and
    # false for one it was not. torch's CPU kernels are SHAPE-dependent at the last ulp (the
    # vectorized body and the scalar tail are different code paths), so a narrowed cook can
    # differ from the whole-frame crop even with no spatial reach at all. Measured here:
    #
    #   class                     CPU         CUDA
    #   pointwise, no noise       0.0         0.0      <- the class the claim was measured on
    #   perlin                    2.98e-08    0.0      <- 1 ulp, from the tail-lane split
    #   curl (a noise DERIVATIVE) 2.98e-05    0.0      <- that ulp x its own 1/(2*eps) = 500
    #
    # The differing pixels are always the tail of the narrowed tensor, and a window large
    # enough to leave no tail is exact. This test PINS the envelope rather than restating the
    # claim: a regression that widens it (or a fix that closes it) shows up here as a number,
    # not as a viewport seam a user has to notice. Sub-ulp classes stay strict — `pointwise`
    # asserts EXACT equality, so the common case cannot silently drift into the noise budget.
    from TEX_Wrangle import tex_engine, tex_roi as _R
    W, H = 72, 56
    torch.manual_seed(7)
    A = torch.rand(1, H, W, 4)
    #                label                 code                                    cpu tol
    cases = [("pointwise (no noise)", "@OUT = vec4(@A.rgb * 1.7 + vec3(0.05), 1.0);", 0.0),
             ("perlin",         "@OUT = vec4(vec3(perlin(u*3.0, v*3.0)), 1.0);",      1e-7),
             ("curl derivative", "vec2 c = curl(u*3.0, v*3.0);\n@OUT = vec4(c, 0.0, 1.0);", 1e-4)]
    wins = [(31, 17, 12, 38, W, H), (2, 3, 7, 9, W, H), (5, 5, 64, 40, W, H)]
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    try:
        for dev in devs:
            a = A.to(dev)
            for label, code, cpu_tol in cases:
                tol = cpu_tol if dev == "cpu" else 0.0     # CUDA is exact for every class
                worst, served = 0.0, 0
                for win in wins:
                    _R.clear_roi_memo()
                    full = tex_engine.cook(code, {"A": a}, device_mode=dev,
                                           precision="fp32").outputs["OUT"]
                    _R.clear_roi_memo()
                    res = tex_engine.cook(code, {"A": a}, device_mode=dev, precision="fp32",
                                          roi=win, roi_exec=True)
                    if res.cooked_roi is None:
                        continue                            # declined → nothing to compare
                    served += 1
                    x0, y0, w, h, _W, _H = win
                    worst = max(worst, (res.outputs["OUT"]
                                        - full[:, y0:y0 + h, x0:x0 + w]).abs().max().item())
                if not served:
                    r.fail(f"v0.30 envelope {dev}/{label}", "every window was declined — the "
                                                            "probe measures nothing")
                elif worst > tol:
                    r.fail(f"v0.30 envelope {dev}/{label}",
                           f"maxdiff {worst:.3e} exceeds the pinned {tol:.0e} over {served} "
                           f"window(s) — the narrow-cook-crop envelope widened")
                else:
                    r.ok(f"{dev}/{label}: maxdiff {worst:.3e} within the pinned {tol:.0e}")
    except Exception as e:
        r.fail("v0.30 accuracy envelope", f"{type(e).__name__}: {e}")
    finally:
        _R.clear_roi_memo()


def test_v030_comfy_never_arms_roi(r: SubTestResult):
    print("\n--- v0.30: the ComfyUI node never passes roi= / roi_exec= (invariant #7) ---")
    # CANARY, the ENG-4/5/6 contract-key-set shape: "a ComfyUI cook is unaffected" must be a
    # red test, not a promise. tex_node builds its prepare() kwargs explicitly; assert the two
    # ROI keys are absent from the source of that call, so adding one is a deliberate act.
    import re
    from pathlib import Path
    try:
        src = (Path(__file__).resolve().parent.parent / "tex_node.py").read_text(encoding="utf-8")
        # every prepare(...) call in the node, flattened
        calls = re.findall(r"prepare\((.*?)\)\s*$", src, re.S | re.M)
        joined = " ".join(calls)
        offenders = [k for k in ("roi=", "roi_exec=") if k in joined]
        if offenders:
            r.fail("v0.30 comfy-never-arms", f"tex_node passes {offenders} to prepare()")
        else:
            r.ok("tex_node.prepare() passes neither roi= nor roi_exec= (v0.29 behaviour held)")
    except Exception as e:
        r.fail("v0.30 comfy-never-arms", f"{type(e).__name__}: {e}")


def test_v030_roi_codegen_route_equivalence(r: SubTestResult):
    print("\n--- v0.30: an ROI cook routed through CODEGEN matches the interpreter exactly ---")
    # DIFFERENTIAL ORACLE (§10.4), the shape ROI-4 already uses, extended to the second tier.
    # v0.30 threads the window into codegen's coordinate-env builder (the v0.27 deferral's
    # reopen condition). Before that, pointing codegen at a narrowed cook gave maxdiff 4.5e-1 —
    # silent wrong pixels — because every coordinate builtin was derived from the narrowed
    # tensor's own shape. These rows fail loudly if that offset is ever dropped again.
    import os as _os
    from TEX_Wrangle import tex_engine, tex_roi as _R
    from TEX_Wrangle.tex_runtime import tier_trace

    class _Rec:                      # stand-in when nothing recorded, so `.tier` is always safe
        tier = None

    W, H = 48, 32
    progs = [
        "@OUT = @A * 0.5 + vec4(u, v, 0.0, 0.0);",        # coordinate-sensitive (the canary)
        "@OUT = gauss_blur(@A, 2.0);",                    # halo op (cook region = ROI + 6)
        "@OUT = vec4(vec3(ix / iw, iy / ih, 0.5), 1.0);", # ix/iy/iw/ih under an offset window
        "@OUT = @A * vec4(px * 100.0, py * 100.0, 1.0, 1.0);",  # px/py reference the FULL image
        # Codegen DECLINES this one (ENG-7: it reads a host playhead) while still reading
        # coordinates, so it exercises the interpreter-fallback path INSIDE the codegen route.
        # Without that fallback forwarding `roi=`, the fallback derives coordinates from the
        # narrowed tensor and returns wrong pixels — mutation-verified: dropping the forward
        # left the suite green before this row existed (maxdiff 0.51).
        "@OUT = @A * 0.5 + vec4(u, v, fract(time), 0.0);",
    ]
    # SAME SIZE, DIFFERENT ORIGIN is the pair that catches a cache key which forgets the
    # window's position: rows 1 and 4 are both 16x12, so a key keyed only on extent serves
    # the first window's coordinates for the second (mutation-verified: maxdiff 0.51 on a
    # panning viewport, and the suite stayed green without this pair).
    rois = [(8, 6, 16, 12, W, H), (0, 0, 12, 9, W, H), (W - 14, H - 11, 14, 11, W, H),
            (14, 10, 16, 12, W, H)]
    prev = _os.environ.get("TEX_ROI_CODEGEN")
    bad = []
    try:
        torch.manual_seed(5)
        A = torch.rand(1, H, W, 4)
        for code in progs:
            expect_decline = "time" in code      # ENG-7: a playhead read is never codegen-cached
            full = tex_engine.cook(code, {"A": A}, device_mode="cpu").outputs["OUT"]
            for roi in rois:
                x0, y0, w, h, _W, _H = roi
                ref = full[:, y0:y0 + h, x0:x0 + w]
                got = {}
                for route, flag in (("interp", "0"), ("codegen", "1")):
                    _os.environ["TEX_ROI_CODEGEN"] = flag
                    _R.clear_roi_memo()
                    tier_trace.reset()
                    res = tex_engine.cook(code, {"A": A}, device_mode="cpu",
                                          roi=roi, roi_exec=True)
                    if res.cooked_roi != roi:
                        bad.append(f"{route} did not take the ROI path for {roi}: {code[:28]}")
                        continue
                    # Assert the tier that ACTUALLY ran. Without this the whole oracle would
                    # quietly degrade to interp-vs-interp the day codegen starts declining
                    # these programs, and still pass with a perfect 0.0 diff.
                    rec = tier_trace.last() or _Rec()
                    served, fell = rec.tier, getattr(rec, "fallback_from", None)
                    if route == "codegen":
                        if expect_decline:
                            # This program must reach the interpreter FALLBACK inside the
                            # codegen route — not simply never attempt codegen.
                            if fell != "codegen":
                                bad.append(f"expected a codegen decline, got tier={served!r} "
                                           f"fallback={fell!r}: {code[:28]}")
                        elif served != "codegen":
                            bad.append(f"codegen route served tier={served!r}: {code[:28]}")
                    if route == "interp" and served == "codegen":
                        bad.append(f"interp route served codegen: {code[:28]}")
                    got[route] = res.outputs["OUT"]
                if len(got) != 2:
                    continue
                # vs the whole-frame crop (the ROI contract) and vs each other (tier parity)
                for route, out in got.items():
                    if tuple(out.shape) != (1, h, w, 4):
                        bad.append(f"{route} shape {tuple(out.shape)} != {(1, h, w, 4)}")
                    elif (ref - out).abs().max().item() > _TOL:
                        bad.append(f"{route} vs whole-frame crop: maxdiff "
                                   f"{(ref - out).abs().max().item():.2e} :: {code[:28]} {roi}")
                if got.get("interp") is not None and got.get("codegen") is not None:
                    md = (got["interp"].float() - got["codegen"].float()).abs().max().item()
                    if md != 0.0:      # the tiers must agree BIT-EXACTLY, not merely closely
                        bad.append(f"interp!=codegen maxdiff {md:.2e} :: {code[:28]} {roi}")

        # The KNOWN exception, pinned rather than hidden. `mix()`/`lerp()` against a hoisted
        # constant vector is 1 ulp apart on CPU: codegen hands `torch.lerp` a broadcast view of
        # the constant (stride 4,0,0,1) while the interpreter materializes it, and that kernel
        # is layout-sensitive. It is the v0.19.1 fused-lerp family with a different mechanism,
        # and it is NOT reachable from run_roi (making the narrowed binding contiguous does not
        # move it — measured). The route ships OFF, so this is recorded as a bounded envelope,
        # not a bit-exact claim; if the route is ever turned on, this is the row to fix first.
        _os.environ["TEX_ROI_CODEGEN"] = "0"
        _R.clear_roi_memo()
        mix_code = "@OUT = mix(@A, vec4(0.25, 0.5, 0.75, 1.0), 0.3);"
        m_i = tex_engine.cook(mix_code, {"A": A}, device_mode="cpu",
                              roi=rois[0], roi_exec=True).outputs["OUT"]
        _os.environ["TEX_ROI_CODEGEN"] = "1"
        _R.clear_roi_memo()
        m_c = tex_engine.cook(mix_code, {"A": A}, device_mode="cpu",
                              roi=rois[0], roi_exec=True).outputs["OUT"]
        m_d = (m_i.float() - m_c.float()).abs().max().item()
        if m_d > 1e-6:
            bad.append(f"mix() codegen-ROI drifted beyond the recorded 1-ulp envelope: {m_d:.2e}")
        if bad:
            r.fail("v0.30 ROI codegen route", "; ".join(bad[:6]))
        else:
            r.ok(f"{len(progs)}x{len(rois)} ROI cooks: codegen == interpreter == whole-frame crop "
                 f"(bit-exact; mix()/lerp() pinned to its recorded 1-ulp envelope)")
    except Exception as e:
        r.fail("v0.30 ROI codegen route", f"{type(e).__name__}: {e}")
    finally:
        if prev is None:
            _os.environ.pop("TEX_ROI_CODEGEN", None)
        else:
            _os.environ["TEX_ROI_CODEGEN"] = prev
        _R.clear_roi_memo()


def test_v030_pm6_roi_viewport(r: SubTestResult):
    print("\n--- v0.30 / PM-6: the 10-node ROI viewport comp (correctness + the scrub win) ---")
    # PM-6 is "scrub a 10-node comp at proxy resolution with ROI cooks + cache hits at
    # interactive rate". A wall-clock claim alone would be a one-off, so this pins the two
    # facts that make it MEAN something and cannot flake: (1) an ROI-updated window is
    # BIT-IDENTICAL to a whole-frame recook — the viewport never trades pixels for speed; and
    # (2) editing the terminal node's slider actually serves the nine upstream stages from
    # canvas/cache rather than recooking them (the thing that makes the scrub cheap). The
    # milliseconds live in the benchmark (`host_demo.bench_pm6`), reported per box, because
    # timings on a loaded CI runner would flake.
    import sys as _sys
    from pathlib import Path
    ex = str(Path(__file__).resolve().parent.parent / "examples")
    if ex not in _sys.path:
        _sys.path.insert(0, ex)
    try:
        import host_demo as _H
        if len(_H._COMP_STAGES) < 10:
            r.fail("PM-6 comp size", f"{len(_H._COMP_STAGES)} stages, PM-6 asks for 10")
            return
        res, side = 192, 96
        base = _H.RoiComp(res, device="cpu")
        base.cook(None, 0)
        base.params["vignette"]["strength"] = 0.83
        whole = base.cook(None, 0)[0]

        win = _H.RoiComp(res, device="cpu")
        win.cook(None, 0)
        win.params["vignette"]["strength"] = 0.83
        roi = (32, 24, side, side, res, res)
        got, _ms, hits = win.cook(roi, len(_H._COMP_STAGES) - 1)
        x0, y0, w, h, _W, _H_ = roi
        md = (whole[:, y0:y0 + h, x0:x0 + w] - got[:, y0:y0 + h, x0:x0 + w]).abs().max().item()
        if md != 0.0:
            r.fail("PM-6 window pixels", f"ROI-updated window != whole-frame recook (maxdiff {md:.2e})")
        else:
            r.ok(f"ROI-updated window is bit-identical to the whole-frame recook ({len(_H._COMP_STAGES)} stages)")
        # The scrub must actually reuse the clean prefix, or "interactive" is an accident.
        if hits < len(_H._COMP_STAGES) - 1:
            r.fail("PM-6 cache reuse", f"only {hits} upstream stages reused, expected "
                                       f"{len(_H._COMP_STAGES) - 1}")
        else:
            r.ok(f"a terminal-node edit reuses all {hits} upstream stages (no recook)")

        # An UPSTREAM edit must reach the window — through nine downstream stages, five of
        # which have a halo. This is the row that fails if the per-stage cache key forgets its
        # upstream chain (the frame is served stale and matches the PRE-edit image), and the
        # row that fails if a window patch leaves a stale halo ring for the next stage to read.
        up = _H.RoiComp(res, device="cpu")
        pre = up.cook(None, 0)[0].clone()
        up.params["exposure"]["exposure"] = 2.00
        win = up.cook(roi, 0)[0]
        truth = _H.RoiComp(res, device="cpu")
        truth.cook(None, 0)
        truth.params["exposure"]["exposure"] = 2.00
        ref = truth.cook(None, 0)[0]
        moved = (win[:, y0:y0 + h, x0:x0 + w] - pre[:, y0:y0 + h, x0:x0 + w]).abs().max().item()
        exact = (win[:, y0:y0 + h, x0:x0 + w] - ref[:, y0:y0 + h, x0:x0 + w]).abs().max().item()
        if moved == 0.0:
            r.fail("PM-6 upstream edit", "editing the FIRST stage changed nothing in the window "
                                         "— the per-stage cache key is blind to its upstream")
        elif exact != 0.0:
            r.fail("PM-6 halo ring", f"window differs from a full recook by {exact:.2e} after an "
                                     f"upstream edit — a downstream halo read a stale ring")
        else:
            r.ok(f"an upstream edit reaches the window exactly (moved {moved:.3f}, matches truth)")

        # "A revisited value is a cache HIT" needs a real observation, not the demo's own skip
        # counter (which counts clean-canvas reuse and would still pass with the ResultCache
        # deleted). Count actual engine cooks: scrub A -> B -> back to A, and the return trip
        # must cook strictly fewer stages than the first visit did.
        from TEX_Wrangle import tex_engine as _TE
        cooks = {"n": 0}
        _real = _TE.cook

        def _counting(*a, **k):
            cooks["n"] += 1
            return _real(*a, **k)
        sc = _H.RoiComp(res, device="cpu")
        sc.cook(None, 0)
        last = len(_H._COMP_STAGES) - 1
        try:
            _TE.cook = _counting
            sc.params["vignette"]["strength"] = 0.11
            sc.cook(roi, last); first = cooks["n"]          # cold value
            sc.params["vignette"]["strength"] = 0.22
            sc.cook(roi, last)                              # a different value
            cooks["n"] = 0
            sc.params["vignette"]["strength"] = 0.11
            sc.cook(roi, last); revisit = cooks["n"]        # back to a value already cooked
        finally:
            _TE.cook = _real
        if first == 0:
            r.fail("PM-6 cache-hit probe", "the cold visit cooked nothing — probe is vacuous")
        elif revisit >= first:
            r.fail("PM-6 cache hit", f"revisiting a scrubbed value cooked {revisit} time(s), "
                                     f"same as the cold visit ({first}) — CACHE-2 is not serving")
        else:
            r.ok(f"revisiting a scrubbed value is a CACHE-2 hit ({first} cook(s) cold → {revisit} warm)")
    except Exception as e:
        r.fail("PM-6 ROI viewport", f"{type(e).__name__}: {e}")


def test_v030_codegen_roi_defaults_off(r: SubTestResult):
    print("\n--- v0.30: the codegen ROI route defaults OFF (the measured decision) ---")
    # CANARY. v0.30 built the codegen ROI route, proved it bit-exact, measured it at
    # 0.94-0.96x (SLOWER on this class of box) and shipped it OFF. That default IS the
    # decision, so it gets a test: flipping it on by accident would silently make every ROI
    # cook slower, and nothing else in the suite would notice.
    import os as _os
    from TEX_Wrangle import tex_engine
    prev = _os.environ.pop("TEX_ROI_CODEGEN", None)
    try:
        if tex_engine._roi_codegen_enabled():
            r.fail("v0.30 codegen-ROI default", "TEX_ROI_CODEGEN is ON by default")
        else:
            r.ok("codegen ROI routing is OFF unless TEX_ROI_CODEGEN=1 (measured: slower)")
        _os.environ["TEX_ROI_CODEGEN"] = "1"
        r.ok("the switch still turns it on") if tex_engine._roi_codegen_enabled() else \
            r.fail("v0.30 codegen-ROI switch", "TEX_ROI_CODEGEN=1 did not enable it")
    except Exception as e:
        r.fail("v0.30 codegen-ROI default", f"{type(e).__name__}: {e}")
    finally:
        _os.environ.pop("TEX_ROI_CODEGEN", None)
        if prev is not None:
            _os.environ["TEX_ROI_CODEGEN"] = prev


def test_v030_nightly_wires_roi_oracle(r: SubTestResult):
    print("\n--- v0.30: the ROI-4 oracle is wired into the nightly (the flip's gate (a)) ---")
    # docs/roi-spatial-laziness.md gates the flag flip on "ROI-4's differential fuzz lane green
    # across a nightly run". Before v0.30 the nightly ran ONLY the TST-1 interp<->codegen fuzzer,
    # so that gate had no nightly to be green on. DERIVATION test: the workflow must name the
    # oracle, or the gate becomes unsatisfiable again.
    from pathlib import Path
    import re
    wf = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "nightly_fuzz.yml"
    try:
        text = wf.read_text(encoding="utf-8")
        if "test_v024_phase1" not in text:
            r.fail("v0.30 nightly ROI lane", "nightly_fuzz.yml does not run test_v024_phase1")
            return
        # …and it must select EVERY ROI-4 test, not merely mention one. Checking for a name was
        # nearly vacuous: it stayed green while the step's explicit or-list ran 4 of the 5
        # (`test_roi4_partial_broadcast_crop` matched none of its terms), so the lane the flag
        # flip is gated on was quietly narrower than the suite it claims to run. Derive the
        # answer instead — read the `-k` expression out of the workflow and confirm each
        # `test_roi4_*` defined in the oracle file matches one of its terms.
        expr = re.search(r'-k\s+"([^"]+)"', text)
        if expr is None:
            r.fail("v0.30 nightly ROI lane", "no -k selector found in the ROI oracle step")
            return
        terms = [t.strip() for t in re.split(r"\bor\b", expr.group(1)) if t.strip()]
        src = (Path(__file__).resolve().parent / "test_v024_phase1.py").read_text(encoding="utf-8")
        defined = re.findall(r"^def (test_roi4_\w+)", src, re.M)
        skipped = [n for n in defined if not any(t in n for t in terms)]
        if not defined:
            r.fail("v0.30 nightly ROI lane", "no test_roi4_* found — the probe is vacuous")
        elif skipped:
            r.fail("v0.30 nightly ROI lane",
                   f"the nightly -k {expr.group(1)!r} does not select {skipped}")
        else:
            r.ok(f"nightly_fuzz.yml runs all {len(defined)} ROI-4 tests (gate (a) satisfiable)")
    except Exception as e:
        r.fail("v0.30 nightly ROI lane", f"{type(e).__name__}: {e}")
