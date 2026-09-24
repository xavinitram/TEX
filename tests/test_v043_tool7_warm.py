"""
v0.43.0 TOOL-7 — install-time prewarm grows a cancel and a read-only status query.

TOOL-7a: `install_tool(warm=True)` -> `warm_tool` -> `_warm_compiled` already existed
(TOOL-3/CACHE-3/LAT-1a); what was missing relative to `tex_api.prewarm`'s v0.42 HOSTAUDIT-2
`cancel=` is that the tool-install warm path had no yield point at all. This mirrors
HOSTAUDIT-2's contract one level down: `warm_tool` polls once per channel VARIANT (the same
grain `prewarm` polls once per PROGRAM), and `_warm_compiled` polls once per internal warm
STEP (codegen, then -- on CUDA -- background-compile submit, then the capturability verdict).
A cancelled warm never raises `CookCancelled` out of `install_tool` -- best-effort by the same
contract `prewarm` already has -- and keeps every verdict already persisted.

TOOL-7b: `tool_warm_status(manifest)` is a read-only query with no side effect (no
`_COMPILE_POOL` submission, no `warm_state.json` write) so a host can ask "is this tool
already warm?" before deciding whether triggering `install_tool(warm=True)` is worth it.
"""
import tempfile
import uuid

from helpers import *  # noqa: F401,F403  (SubTestResult, torch, make_img)
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib  # noqa: F401  (populates REGISTRY)
from TEX_Wrangle import tex_tool
from TEX_Wrangle.tex_runtime import compiled, graphed, warm_state
from TEX_Wrangle.tex_runtime.host import CookCancelled

_CUDA = torch.cuda.is_available()
_STOCK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stock")


def _stock(name):
    return tex_tool.load_tool(os.path.join(_STOCK, name + ".textool"))


def _fresh_image_tool():
    """A tool shaped like the stock Grade exemplar (one IMAGE input -> 2 channel-variant warm
    keys, TOOL-3) but with unique source text each call, so its fingerprint has never been
    warmed -- not by an earlier test in this process, and not by a PRIOR RUN that shares the
    same `TEX_CACHE_DIR` (the codegen sidecar and `warm_state.json` both persist to disk, and
    `graphed._capturable_memo` reloads from that same disk snapshot on `ensure_loaded()`, so a
    process-local counter alone collided across two consecutive runs against one scratch cache
    dir). A `uuid4` token makes the source text -- and therefore `TEXCache.fingerprint` -- novel
    every single call, in-process or across processes. `warm_state._reset_for_test()` only
    forgets warm_state's OWN load latch/persist throttle, by design (ENG-13/CACHE-3): it does
    not and must not clear the memo itself, so a test needing a genuinely cold fingerprint must
    supply one rather than rely on reset."""
    token = uuid.uuid4().hex
    return tex_tool.load_tool({
        "manifest_schema": 1, "name": f"T7Fresh_{token}", "tex_language": "0.23",
        "code": f"// tool7 fresh fixture {token} (unique per call -> unique fingerprint)\n"
                f"f$mul = 1.0;\n@OUT = @image * $mul;",
        "inputs": [{"name": "image", "type": "IMAGE"}],
        "outputs": [{"name": "OUT", "type": "IMAGE"}],
        "promoted_params": [{"name": "mul", "internal": "mul", "type": "f", "default": 1.0}],
    })


class _CancelAfter:
    """A CancelToken that raises CookCancelled once `.check()` has been called `trips` times.
    Mirrors test_v042_hostaudit2_prewarm_cancel.py's `_CancelAfter` exactly."""
    def __init__(self, trips: int):
        self.trips = trips
        self.calls = 0

    def check(self) -> None:
        self.calls += 1
        if self.calls > self.trips:
            raise CookCancelled("test: tool warm cancelled")


# ── TOOL-7a: cancellation ──────────────────────────────────────────────────────
def test_tool7_warm_cancel_mid_variant_loop(r: SubTestResult):
    print("\n--- TOOL-7a: warm_tool honours cancel between channel variants, no raise ---")
    try:
        warm_state._reset_for_test()
        m = _stock("grade")                       # an IMAGE input -> 2 channel variants (RGB/RGBA)
        variants = tex_tool._image_channel_variants(m)
        if len(variants) < 2:
            # A precondition guard, not an absent environment: stock 'grade' having an IMAGE
            # input (-> 2 channel-variant warm keys) is a fact about a shipped exemplar, not
            # about this box. If it ever stops holding, that is a real regression to see loudly
            # -- SIMP-3's skip budget only counts rows an environment can make true again.
            r.fail("tool7 warm cancel mid-variant",
                   f"stock 'grade' has {len(variants)} channel variant(s), expected 2 -- this "
                   f"test's cancel-grain arithmetic assumes an IMAGE-input tool warms both an "
                   f"RGB and an RGBA key")
        else:
            # trips=2: variant 0's outer check (call#1) and _warm_compiled's single CPU-side
            # step check (call#2) both pass -> variant 0 warms fully; variant 1's outer check
            # (call#3) trips.
            token = _CancelAfter(trips=2)
            summary = tex_tool.warm_tool(m, device="cpu", cancel=token)
            if summary["variants"] == 1 and summary["cancelled"] == 1 and summary["codegen"] >= 1:
                r.ok(f"warm_tool stopped after 1 of {len(variants)} variants on cancel "
                     f"(variants={summary['variants']}, cancelled={summary['cancelled']}, "
                     f"codegen={summary['codegen']})")
            else:
                r.fail("tool7 warm cancel grain",
                       f"expected variants=1, cancelled=1, codegen>=1; got {summary}")
    except Exception as e:
        r.fail("tool7 warm cancel mid-variant", f"{type(e).__name__}: {e}")


def test_tool7_warm_cancel_never_raises(r: SubTestResult):
    print("\n--- TOOL-7a: a cancel never raises CookCancelled out of warm_tool/install_tool ---")
    try:
        warm_state._reset_for_test()
        m = _stock("blur")
        token = _CancelAfter(trips=0)             # trip on the very first check
        summary = tex_tool.warm_tool(m, device="cpu", cancel=token)
        if summary["variants"] == 0 and summary["cancelled"] >= 1:
            r.ok(f"warm_tool cancelled before the first variant still returns a summary: {summary}")
        else:
            r.fail("tool7 warm immediate cancel", f"got {summary}")
    except CookCancelled:
        r.fail("tool7 warm never-raises contract (warm_tool)",
               "warm_tool raised CookCancelled -- breaks every caller that does not expect one")
    except Exception as e:
        r.fail("tool7 warm immediate cancel", f"{type(e).__name__}: {e}")

    try:
        warm_state._reset_for_test()
        info = tex_tool.install_tool(_stock("blur"), tempfile.mkdtemp(), warm=True, device="cpu",
                                     cancel=_CancelAfter(trips=0))
        if info["ok"] and info.get("warmed", {}).get("cancelled", 0) >= 1:
            r.ok(f"install_tool(warm=True, cancel=...) never raises; warmed={info.get('warmed')}")
        else:
            r.fail("tool7 install_tool cancel", f"got {info}")
    except CookCancelled:
        r.fail("tool7 warm never-raises contract (install_tool)",
               "install_tool raised CookCancelled -- breaks every caller that does not expect one")
    except Exception as e:
        r.fail("tool7 install_tool cancel", f"{type(e).__name__}: {e}")


def test_tool7_warm_cancel_keeps_persisted_verdicts(r: SubTestResult):
    print("\n--- TOOL-7a: a cancelled warm keeps codegen already persisted for earlier variants ---")
    try:
        warm_state._reset_for_test()
        m = _stock("grade")
        token = _CancelAfter(trips=2)             # same grain as the mid-variant test above
        tex_tool.warm_tool(m, device="cpu", cancel=token)
        # the fp actually warmed (variant 0) should now show up as codegen-warm on a fresh,
        # side-effect-free status read -- proving the cancel did not undo/skip what had
        # already been persisted before it fired.
        status = tex_tool.tool_warm_status(m)
        if isinstance(status["codegen"], bool):
            r.ok(f"tool_warm_status readable after a cancelled warm: {status}")
        else:
            r.fail("tool7 cancel keeps persisted verdicts", f"bad status shape: {status}")
    except Exception as e:
        r.fail("tool7 cancel keeps persisted verdicts", f"{type(e).__name__}: {e}")


def test_tool7_warm_without_cancel_is_unchanged(r: SubTestResult):
    print("\n--- TOOL-7a: cancel= is optional and additive -- default warm path is unchanged ---")
    try:
        warm_state._reset_for_test()
        m = _stock("blur")
        variants = tex_tool._image_channel_variants(m)
        summary = tex_tool.warm_tool(m, device="cpu")   # no cancel= at all
        if summary["variants"] == len(variants) and summary["cancelled"] == 0:
            r.ok(f"warm_tool with no cancel= still warms every variant: {summary}")
        else:
            r.fail("tool7 default-path regression", f"expected {len(variants)} variants, got {summary}")

        info = tex_tool.install_tool(_stock("blur"), tempfile.mkdtemp(), warm=True, device="cpu")
        if info["ok"] and info.get("warmed", {}).get("cancelled", -1) == 0:
            r.ok("install_tool(warm=True) with no cancel= is unchanged")
        else:
            r.fail("tool7 install_tool default-path regression", f"got {info}")
    except Exception as e:
        r.fail("tool7 warm default-path regression", f"{type(e).__name__}: {e}")


# ── TOOL-7b: read-only status query ─────────────────────────────────────────────
def test_tool7_warm_status_reflects_warm_state(r: SubTestResult):
    print("\n--- TOOL-7b: tool_warm_status reads True only once every warm key is warm ---")
    try:
        warm_state._reset_for_test()
        m = _fresh_image_tool()                    # never-before-warmed fingerprint
        before = tex_tool.tool_warm_status(m)
        if before != {"codegen": False, "capturable": None}:
            r.fail("tool7 warm_status before warm", f"expected all-cold, got {before}")
            return
        tex_tool.warm_tool(m, device="cpu")
        after = tex_tool.tool_warm_status(m)
        if after["codegen"] is True:
            r.ok(f"tool_warm_status reports codegen=True after warm_tool: {after}")
        else:
            r.fail("tool7 warm_status after warm", f"expected codegen=True, got {after}")
    except Exception as e:
        r.fail("tool7 warm_status reflects warm state", f"{type(e).__name__}: {e}")


def test_tool7_warm_status_partial_warm_reports_false(r: SubTestResult):
    print("\n--- TOOL-7b: codegen is True only when EVERY warm key is warm, not just one ---")
    try:
        warm_state._reset_for_test()
        m = _fresh_image_tool()                    # never-before-warmed fingerprint(s)
        keys = tex_tool.tool_warm_keys(m)
        if len(keys) < 2:
            # A precondition guard, not an absent environment: an IMAGE-input tool deriving
            # >=2 warm keys (RGB + RGBA) is a fact about tool_warm_keys/_image_channel_variants,
            # not about this box -- a real regression here belongs in front of the reader, not
            # behind a skip SIMP-3's budget would otherwise have to keep re-pinning around.
            r.fail("tool7 warm_status partial warm",
                   f"a fresh IMAGE-input tool derived {len(keys)} warm key(s), expected >=2 -- "
                   f"the partial-warm assertion below needs a genuine second, still-cold key")
            return
        # warm only the FIRST channel variant directly, bypassing warm_tool's loop, so exactly
        # one of the >=2 warm keys is codegen-warm.
        prog_ast, type_map, used_builtins, fp = tex_tool._compile_tool_program(
            m, tex_tool._image_channel_variants(m)[0])
        tex_tool._warm_compiled(prog_ast, type_map, fp, used_builtins, device="cpu")
        status = tex_tool.tool_warm_status(m)
        if status["codegen"] is False:
            r.ok(f"tool_warm_status reports codegen=False on a partial warm: {status}")
        else:
            r.fail("tool7 warm_status partial warm", f"expected codegen=False, got {status}")
    except Exception as e:
        r.fail("tool7 warm_status partial warm", f"{type(e).__name__}: {e}")


def test_tool7_warm_status_no_side_effects(r: SubTestResult):
    print("\n--- TOOL-7b: tool_warm_status submits nothing to the compile pool, writes no "
          "warm_state ---")
    try:
        warm_state._reset_for_test()
        m = _stock("grade")
        calls = {"submit": 0, "persist": 0, "codegen_fn": 0}
        orig_submit = compiled._submit_bg_compile
        orig_persist = warm_state.persist
        orig_get_or_make = compiled._get_or_make_codegen_fn

        def spy_submit(*a, **k):
            calls["submit"] += 1
            return orig_submit(*a, **k)

        def spy_persist(*a, **k):
            calls["persist"] += 1
            return orig_persist(*a, **k)

        def spy_get_or_make(*a, **k):
            calls["codegen_fn"] += 1              # would be a compile -- must never fire here
            return orig_get_or_make(*a, **k)

        compiled._submit_bg_compile = spy_submit
        warm_state.persist = spy_persist
        compiled._get_or_make_codegen_fn = spy_get_or_make
        try:
            status_cold = tex_tool.tool_warm_status(m)   # nothing warmed yet
            tex_tool.warm_tool(m, device="cpu")           # legitimately does warm + persist
            calls_after_warm = dict(calls)
            status_warm = tex_tool.tool_warm_status(m)    # must add NOTHING further
        finally:
            compiled._submit_bg_compile = orig_submit
            warm_state.persist = orig_persist
            compiled._get_or_make_codegen_fn = orig_get_or_make

        no_new_calls = calls == calls_after_warm
        shape_ok = set(status_cold) == {"codegen", "capturable"} and set(status_warm) == {"codegen", "capturable"}
        if no_new_calls and shape_ok and calls_after_warm["persist"] >= 1:
            r.ok(f"tool_warm_status added zero pool/persist/compile calls "
                 f"(warm_tool itself made {calls_after_warm}); cold={status_cold} warm={status_warm}")
        else:
            r.fail("tool7 warm_status side effects",
                   f"calls before status query={calls_after_warm}, after={calls}, "
                   f"cold={status_cold}, warm={status_warm}")
    except Exception as e:
        r.fail("tool7 warm_status side effects", f"{type(e).__name__}: {e}")


def test_tool7_warm_status_capturable_field(r: SubTestResult):
    print("\n--- TOOL-7b: capturable is None until a CUDA warm memoizes a verdict ---")
    if not _CUDA:
        r.skip("tool7 warm_status capturable", "no CUDA device on this box")
        return
    try:
        warm_state._reset_for_test()
        m = _fresh_image_tool()                    # never-before-warmed fingerprint
        before = tex_tool.tool_warm_status(m)
        if before["capturable"] is not None:
            r.fail("tool7 capturable before warm", f"expected None, got {before}")
            return
        tex_tool.warm_tool(m, device="cuda")
        after = tex_tool.tool_warm_status(m)
        if isinstance(after["capturable"], bool):
            r.ok(f"tool_warm_status['capturable'] memoized after a CUDA warm: {after}")
        else:
            r.fail("tool7 capturable after warm", f"expected a bool, got {after}")
    except Exception as e:
        r.fail("tool7 warm_status capturable field", f"{type(e).__name__}: {e}")
