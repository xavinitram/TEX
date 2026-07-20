"""
v0.28 Phase 1 — "Second host" (the proof release).

DATA-1  buffer metadata sidecar: per-binding {colorspace, premult, frame, extra} tags riding the
        ExecContext value channel (the time_context model — NEVER keyed); merge on conflict ->
        `unknown`; the W7005 gamma-space halo advisory (tex_api.color_advisories).
DATA-2  storage-format descriptor (BufferDesc) + tex_io: a pure-torch scanline EXR reader/writer
        (NONE/ZIPS/ZIP, HALF/FLOAT) and a 16-bit PNG writer — half/uint16 are storage dtypes cast
        to fp32 at the seam, exactly like uint8.
DATA-3  ARRAY host wires under the engine profile: an `a@name` input and array outputs (curve /
        palette / histogram), rejected under ComfyUI (E3203 + the always-on egress guard).
DATA-4  EngineSession phase 1: one handle over the module-level cook singletons (views, byte-
        identical), plus the soak lane (many cooks + reset cycles, flat RSS/VRAM watermarks).
PORT-5  the standalone host demo (examples/host_demo.py): a fused grade->blur->vignette GraphSpec
        cooked with NO comfy import, arming CACHE-2 for a param scrub — PM-2's <50 ms/frame warm.

Everything here ships OFF the default ComfyUI cook path (invariant #7): the tags/session are
host-armed, array wires need the engine profile, and EXR/PNG are opt-in I/O. CPU-pinned; CUDA
looped where present. Any test that flips the egress profile restores 'comfy' in a finally.
"""
import os
import sys
import tempfile

from helpers import *  # noqa: F401,F403  (SubTestResult, torch, make_img)
from TEX_Wrangle import tex_engine, tex_api, tex_marshalling
from TEX_Wrangle.tex_marshalling import BufferMeta, merge_buffer_meta, COLORSPACES, PREMULT
from TEX_Wrangle.tex_io import BufferDesc, decode_to_fp32, encode_from_fp32
from TEX_Wrangle.tex_io import exr as tex_exr
from TEX_Wrangle.tex_io.png import write_png16
from TEX_Wrangle.tex_compiler.types import TEXType, array_wires_enabled

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu", "cuda"] if _CUDA else ["cpu"]


# ── DATA-1 ──────────────────────────────────────────────────────────────────

def test_data1_metadata(r: SubTestResult):
    print("\n--- DATA-1: buffer metadata sidecar ---")

    # canary: the tag vocabularies are a pinned contract
    if set(COLORSPACES) == {"srgb", "linear", "oklab", "unknown"} and \
       set(PREMULT) == {"premultiplied", "unassociated", "opaque", "unknown"}:
        r.ok("colorspace/premult vocabularies pinned")
    else:
        r.fail("DATA-1 vocab", f"{COLORSPACES} / {PREMULT}")
    try:
        BufferMeta(colorspace="bogus")
        r.fail("DATA-1 validate", "bad colorspace accepted")
    except ValueError:
        r.ok("BufferMeta validates its enums")

    # merge policy: agree -> keep, conflict -> unknown/None (never a silent pick)
    a = BufferMeta("srgb", "premultiplied", frame=5, extra={"src": "a"})
    b = BufferMeta("srgb", "unassociated", frame=5, extra={"src": "a"})
    m = merge_buffer_meta([a, b])
    if (m.colorspace == "srgb" and m.premult == "unknown" and m.frame == 5
            and m.extra == {"src": "a"}):
        r.ok("merge: agree kept, conflict -> unknown, extra intersected")
    else:
        r.fail("DATA-1 merge", f"{m}")
    if merge_buffer_meta([]).colorspace == "unknown" and merge_buffer_meta([a]) is a:
        r.ok("merge edge cases (empty -> unknown, single -> itself)")
    else:
        r.fail("DATA-1 merge-edge", "empty/single wrong")

    # value channel: binding_meta rides the cook to CookResult.out_meta, and NEVER a key
    for dev in _DEVICES:
        img = make_img(1, 16, 16, 3, seed=3).to(dev)
        code = "@OUT = vec4(@A.rgb * 1.2, 1.0);"
        base = tex_engine.cook(code, {"A": img.clone()}, device_mode=dev)
        if base.out_meta is not None:
            r.fail("DATA-1 default", f"[{dev}] default cook produced out_meta")
            continue
        tagged = tex_engine.cook(code, {"A": img.clone()}, device_mode=dev,
                                 binding_meta={"A": BufferMeta("linear", "opaque")},
                                 want_lineage=True)
        plain = tex_engine.cook(code, {"A": img.clone()}, device_mode=dev, want_lineage=True)
        ok_meta = tagged.out_meta["OUT"].colorspace == "linear"
        ok_key = tagged.lineage["OUT"] == plain.lineage["OUT"]   # meta must NOT move the key
        r.ok(f"[{dev}] binding_meta -> out_meta, absent from lineage key") if (ok_meta and ok_key) \
            else r.fail("DATA-1 channel", f"[{dev}] meta={ok_meta} key-clean={ok_key}")

    # B2 (doc 32): the tags must survive a FUSED cook — the flagship path. _prepare_fused renames
    # every external binding to `_s{i}_u_<name>`, so egress_meta (given the fused `ctx.bindings`)
    # must strip that prefix to match the host's binding_meta, keyed by the ORIGINAL name. Before
    # the fix out_meta was silently None on EVERY fused chain (the unfused cook above propagated it).
    for dev in _DEVICES:
        fimg = make_img(1, 8, 8, 3, seed=4).to(dev)
        spec = {"stages": [{"code": "@OUT = @A * 2.0;", "image_input": "A", "params": {}}],
                "terminal_image_input": "A"}
        bm = {"A": BufferMeta("linear", "opaque")}
        fused = tex_engine.cook("@OUT = @A + 0.1;", {"A": fimg.clone()}, chain_payload=spec,
                                binding_meta=bm, device_mode=dev)
        # sanity: it really WAS fused (binding was prefix-renamed), and the tags came through.
        was_fused = any(n.startswith("_s") for n in fused.binding_names)
        ok_fused = (fused.out_meta is not None and fused.out_meta["OUT"].colorspace == "linear"
                    and fused.out_meta["OUT"].premult == "opaque")
        r.ok(f"[{dev}] fused-chain cook propagates DATA-1 tags (B2)") if (was_fused and ok_fused) \
            else r.fail("DATA-1 fused", f"[{dev}] fused={was_fused} out_meta={fused.out_meta}")

    # W7005: the gamma-halo advisory fires on a non-linear tag + a spatial read, else silent
    d_srgb = tex_api.color_advisories("@OUT = gauss_blur(@A, 3.0);", {}, {"A": BufferMeta("srgb")})
    d_lin = tex_api.color_advisories("@OUT = gauss_blur(@A, 3.0);", {}, {"A": BufferMeta("linear")})
    d_point = tex_api.color_advisories("@OUT = vec4(@A.rgb, 1.0);", {}, {"A": BufferMeta("srgb")})
    d_none = tex_api.color_advisories("@OUT = gauss_blur(@A, 3.0);", {}, None)
    if (len(d_srgb) == 1 and d_srgb[0].code == "W7005" and d_srgb[0].severity == "warning"
            and d_lin == [] and d_point == [] and d_none == []):
        r.ok("W7005 fires on non-linear + spatial only (linear/pointwise/untagged silent)")
    else:
        r.fail("DATA-1 W7005", f"srgb={[d.code for d in d_srgb]} lin={d_lin} pt={d_point}")


# ── DATA-2 ──────────────────────────────────────────────────────────────────

def test_data2_storage_exr(r: SubTestResult):
    print("\n--- DATA-2: storage descriptor + EXR + 16-bit PNG ---")

    # BufferDesc round-trips per storage dtype (uint8/16 quantise, float exact/half-precise)
    ok = True
    for st, tol in [("uint8", 4e-3), ("uint16", 3e-5), ("float16", 1e-3), ("float32", 0.0)]:
        x = torch.rand(4, 5, 3)
        d = BufferDesc(st)
        err = (decode_to_fp32(encode_from_fp32(x, d), d) - x).abs().max().item()
        ok = ok and err <= tol + 1e-7
    r.ok("BufferDesc encode/decode round-trips (uint8/16/half/float)") if ok \
        else r.fail("DATA-2 bufferdesc", f"{st} err too high")

    td = tempfile.mkdtemp()
    # EXR: float exact, half ~1e-3, across NONE/ZIPS/ZIP and channel counts + a non-square frame
    ok = True
    cases = [(64, 64, 3, False, "zip"), (64, 64, 3, True, "zip"),
             (48, 48, 4, False, "zips"), (50, 50, 1, False, "none"),
             (37, 91, 3, False, "zip"), (17, 3, 3, False, "zip")]   # odd width -> ZIP interleave edge
    for (H, W, C, half, comp) in cases:
        x = torch.rand(H, W, C)
        p = os.path.join(td, f"e{H}_{W}_{C}_{half}_{comp}.exr")
        tex_exr.write_exr(p, x, half=half, compression=comp)
        img = tex_exr.read_exr(p)
        err = (img.pixels - x).abs().max().item()
        ok = ok and list(img.pixels.shape) == [H, W, C] and err <= (5e-2 if half else 1e-6)
    r.ok("EXR write/read round-trip: float exact, half ~1e-3 (NONE/ZIPS/ZIP, 1/3/4ch, non-square)") \
        if ok else r.fail("DATA-2 exr", "a round-trip diverged")

    # B1 (doc 32): a MASK / scalar program egresses [1,H,W] under the engine profile. write_exr
    # must read that as [H,W,1] — NOT crash ("need W channel names") on a normal width, nor
    # silently transpose a narrow one (widths 1-4). Cook a real single-channel program end-to-end.
    from TEX_Wrangle.tex_marshalling import prepare_output as _po, map_inferred_type as _mit
    ok_mask = True
    for (H, W) in [(8, 12), (8, 3), (13, 1)]:   # normal width (was a crash), narrow + 1 (was a transpose)
        src = make_img(1, H, W, 3, seed=2)
        res = tex_engine.cook("@OUT = (@A.r + @A.g + @A.b) / 3.0;", {"A": src.clone()},
                              device_mode="cpu")
        mask = _po(res.outputs["OUT"], _mit(res.assigned["OUT"], False), profile="engine")  # [1,H,W]
        pm = os.path.join(td, f"mask{H}_{W}.exr")
        tex_exr.write_exr(pm, mask)
        back = tex_exr.read_exr(pm).pixels                       # must be [H,W,1]
        ref = mask[0].unsqueeze(-1)                              # [1,H,W] -> [H,W,1] reference
        ok_mask = (ok_mask and list(back.shape) == [H, W, 1]
                   and (back - ref).abs().max().item() < 1e-6)
    r.ok("EXR: a MASK-output program round-trips ([1,H,W]->[H,W,1], no crash/transpose)") if ok_mask \
        else r.fail("DATA-2 exr-mask", "a mask EXR round-trip diverged")

    # EXR HDR: values outside [0,1] survive a FLOAT round-trip (the whole point vs PNG)
    hdr = torch.rand(20, 20, 3) * 12.0 - 3.0
    p = os.path.join(td, "hdr.exr")
    tex_exr.write_exr(p, hdr, half=False)
    back = tex_exr.read_exr(p).pixels
    r.ok("EXR preserves HDR (negatives + >1) bit-exact at FLOAT") \
        if (back - hdr).abs().max().item() < 1e-6 else r.fail("DATA-2 exr-hdr", "HDR clipped")

    # out-of-scope compression / tiled -> a clean error, never a mis-decode
    try:
        tex_exr.write_exr(os.path.join(td, "bad.exr"), torch.rand(4, 4, 3), compression="piz")
        r.fail("DATA-2 exr-reject", "accepted PIZ")
    except tex_exr.EXRError:
        r.ok("EXR refuses out-of-scope compression (clean EXRError)")

    # DECREASING_Y (lineOrder=1) reads UPRIGHT — blocks are placed by absolute y, no post-hoc flip
    import struct as _st
    x = torch.rand(24, 40, 3)
    pn = os.path.join(td, "up.exr")
    tex_exr.write_exr(pn, x)
    raw = bytearray(open(pn, "rb").read())
    mk = b"lineOrder\x00lineOrder\x00" + _st.pack("<i", 1)
    raw[raw.find(mk) + len(mk)] = 1                         # flip the lineOrder value byte 0 -> 1
    pd = os.path.join(td, "dec.exr"); open(pd, "wb").write(raw)
    r.ok("EXR DECREASING_Y reads upright (no double-flip)") \
        if (tex_exr.read_exr(pd).pixels - x).abs().max().item() < 1e-6 \
        else r.fail("DATA-2 exr-lineorder", "DECREASING_Y mis-oriented")

    # HALF stores an over-range HDR value as inf (matches torch), never an OverflowError crash
    ph = os.path.join(td, "hdrhalf.exr")
    try:
        tex_exr.write_exr(ph, torch.tensor([[[70000.0, 0.5, 0.5]]]), half=True)
        r.ok("EXR HALF saturates >65504 to inf (no OverflowError)") \
            if torch.isinf(tex_exr.read_exr(ph).pixels[0, 0, 0]) else r.fail("DATA-2 exr-half-hdr", "not inf")
    except Exception as e:
        r.fail("DATA-2 exr-half-hdr", f"{type(e).__name__}")

    # a malformed / truncated file raises a clean EXRError, never an opaque struct/index/zlib crash
    bad_ok = True
    for blob in (b"\x00\x00\x00\x00x", b"\x76\x2f\x31\x01\x02", b"", bytes(range(40))):
        try:
            tex_exr.read_exr(blob); bad_ok = False
        except tex_exr.EXRError:
            pass
        except Exception:
            bad_ok = False
    r.ok("EXR malformed input -> clean EXRError") if bad_ok else r.fail("DATA-2 exr-malformed", "opaque crash")

    # B3 (doc 32): a NONE/raw block whose declared dataSize is 0 hits torch.frombuffer(b'') ->
    # a RAW ValueError (not a struct/zlib/index error), which read_exr's net must also catch so it
    # keeps its docstring promise ("raises EXRError, never a raw error"). Build a valid 3ch FLOAT
    # NONE header with one block claiming dataSize=0.
    from TEX_Wrangle.tex_io.exr import _exr_header, _C_NONE
    _hdr = _exr_header(4, 1, ["R", "G", "B"], 2, _C_NONE)         # W=4,H=1,3ch FLOAT NONE
    _blob = bytes(_hdr + _st.pack("<Q", len(_hdr) + 8) + _st.pack("<ii", 0, 0))  # y=0, dataSize=0
    try:
        tex_exr.read_exr(_blob)
        r.fail("DATA-2 exr-empty-block", "empty NONE block did not raise")
    except tex_exr.EXRError:
        r.ok("EXR empty NONE block -> clean EXRError (B3, no raw ValueError leak)")
    except Exception as e:
        r.fail("DATA-2 exr-empty-block", f"raw {type(e).__name__}: {e}")

    # 16-bit PNG writes and decodes back bit-exact through torchvision
    try:
        from torchvision.io import decode_image, read_file
        u16 = encode_from_fp32(torch.rand(16, 24, 3), BufferDesc("uint16"))
        p = os.path.join(td, "a16.png")
        write_png16(p, u16)
        dec = decode_image(read_file(p))          # [C,H,W] uint16
        good = dec.dtype == torch.uint16 and bool((dec.permute(1, 2, 0) == u16).all())
        r.ok("16-bit PNG round-trips bit-exact (torchvision decode)") if good \
            else r.fail("DATA-2 png16", f"dtype={dec.dtype}")
    except ImportError:
        r.ok("16-bit PNG test skipped (no torchvision)")

    # the EXR reader is wired into the CLI ingest seam: `tex run --in a.exr` reads scene-linear
    # fp32 with HDR + all channels kept (no [0,1] normalize, no 3-channel force) — the DATA-2
    # round-trip the write path alone left half-open.
    from TEX_Wrangle import tex_cli
    src = torch.stack([torch.full((6, 8), 3.7), torch.rand(6, 8),
                       torch.rand(6, 8), torch.ones(6, 8)], dim=-1)   # [6,8,4] HDR + alpha
    pin = os.path.join(td, "ingest.exr"); tex_exr.write_exr(pin, src)
    loaded = tex_cli.load_image(pin)
    r.ok("tex_cli.load_image('.exr') -> [1,H,W,C] fp32, HDR + alpha kept") \
        if (tuple(loaded.shape) == (1, 6, 8, 4) and abs(loaded[0, 0, 0, 0].item() - 3.7) < 1e-5) \
        else r.fail("DATA-2 exr-ingest", f"{tuple(loaded.shape)}")

    # ingest is CONSTRAINED to RGB(3) / RGBA(4): a depth/AOV/multi-plane EXR (C=1/2/5+) that the
    # engine would silently reinterpret (crush to luma, or crash a mask egress) is refused with a
    # clean ValueError (caught by the CLI's F3 handler), never a traceback or silent channel loss.
    guard_ok = True
    for c, nm in ((1, ["Y"]), (5, ["R", "G", "B", "A", "Z"])):
        pc = os.path.join(td, f"aov{c}.exr"); tex_exr.write_exr(pc, torch.rand(4, 4, c), channels=nm)
        try:
            tex_cli.load_image(pc); guard_ok = False
        except ValueError:
            pass
    r.ok("EXR ingest refuses non-RGB(A) channel counts cleanly (no crash / silent luma-collapse)") \
        if guard_ok else r.fail("DATA-2 exr-ingest-guard", "accepted a 1/5-channel EXR")


# ── DATA-3 ──────────────────────────────────────────────────────────────────

def test_data3_array_wires(r: SubTestResult):
    print("\n--- DATA-3: ARRAY wires under the engine profile ---")
    if array_wires_enabled():
        r.fail("DATA-3 default", "array wires on by default (should be off)")
        return
    ARR = "float pal[3] = {0.1, 0.5, 0.9}; @OUT = pal;"

    # comfy (default): array output rejected at compile (E3203)
    errs = [d.code for d in tex_api.check(ARR, {}) if d.severity == "error"]
    r.ok("comfy rejects an array output (E3203)") if "E3203" in errs \
        else r.fail("DATA-3 comfy-reject", f"{errs}")

    try:
        tex_marshalling.set_egress_profile("engine")
        if not array_wires_enabled():
            r.fail("DATA-3 flip", "engine profile did not enable array wires")
        errs2 = [d.code for d in tex_api.check(ARR, {}) if d.severity == "error"]
        r.ok("engine profile lifts the array-output rejection") if "E3203" not in errs2 \
            else r.fail("DATA-3 engine-allow", f"{errs2}")

        # a [N,C] tensor infers ARRAY; an array input is consumable via a builtin
        palette = torch.tensor([[0.2, 0.4, 0.6], [0.8, 0.5, 0.2], [0.1, 0.9, 0.3]])
        r.ok("a low-rank tensor infers ARRAY (engine)") \
            if tex_marshalling.infer_binding_type(palette) == TEXType.ARRAY \
            else r.fail("DATA-3 infer", "not ARRAY")
        for dev in _DEVICES:
            img = make_img(1, 8, 8, 3, seed=1).to(dev)
            res = tex_engine.cook("@OUT = vec4(@A.rgb * arr_avg(a@palette).r, 1.0);",
                                  {"A": img.clone(), "palette": palette.clone().to(dev)}, device_mode=dev)
            r.ok(f"[{dev}] array INPUT consumed via arr_avg builtin") \
                if res.outputs["OUT"].shape == (1, 8, 8, 4) else r.fail("DATA-3 ingress", f"[{dev}]")
            # non-spatial array pass-through: [N,C] in -> [N,C] out, values preserved
            pt = tex_engine.cook("@OUT = a@palette;", {"palette": palette.clone().to(dev)}, device_mode=dev)
            r.ok(f"[{dev}] array wire pass-through [N,C] preserved") \
                if torch.allclose(pt.outputs["OUT"].cpu(), palette) else r.fail("DATA-3 passthrough", f"[{dev}]")

        # the always-on egress guard: comfy egress refuses an ARRAY output tensor
        try:
            tex_marshalling.prepare_output(torch.tensor([0.1, 0.5, 0.9]), "ARRAY", profile="comfy")
            r.fail("DATA-3 egress-guard", "comfy accepted ARRAY")
        except RuntimeError:
            r.ok("comfy egress refuses an ARRAY output (always-on guard)")
    finally:
        tex_marshalling.set_egress_profile("comfy")
        if array_wires_enabled():
            r.fail("DATA-3 restore", "array wires still on after restoring comfy")
        else:
            r.ok("restored comfy: array wires off (no leak into later tests)")


# ── DATA-4 ──────────────────────────────────────────────────────────────────

def test_data4_session_soak(r: SubTestResult):
    print("\n--- DATA-4: EngineSession + soak ---")
    import gc
    from TEX_Wrangle import tex_cache, tex_memory
    from TEX_Wrangle.tex_runtime.host import get_host_services

    s = tex_api.default_session()
    # phase-1: the session's views ARE the module singletons (byte-identical)
    if (s.cache is tex_cache.get_cache() and s.registry is tex_memory.get_cache_registry()
            and s.host is get_host_services() and s.isolated is False):
        r.ok("session views ARE the module singletons (phase 1, byte-identical)")
    else:
        r.fail("DATA-4 views", "a view diverged from its singleton")
    r.ok("default_session() is a process singleton") if tex_api.default_session() is s \
        else r.fail("DATA-4 singleton", "not identical")

    # cook -> reset -> cook is bit-identical (reset sheds tensor caches, not correctness)
    img = make_img(1, 32, 32, 3, seed=7)
    r1 = tex_engine.cook("@OUT = gauss_blur(@A, 1.5);", {"A": img.clone()}, device_mode="cpu")
    s.reset()
    r2 = tex_engine.cook("@OUT = gauss_blur(@A, 1.5);", {"A": img.clone()}, device_mode="cpu")
    r.ok("cook -> session.reset() -> cook is bit-identical") \
        if torch.allclose(r1.outputs["OUT"], r2.outputs["OUT"]) else r.fail("DATA-4 reset", "diverged")

    # soak: a bounded run of cooks across a few program shapes + reset cycles, flat memory watermark
    try:
        import psutil
        proc = psutil.Process(os.getpid())
    except ImportError:
        proc = None
    progs = ["@OUT = vec4(@A.rgb * 1.2, 1.0);", "@OUT = gauss_blur(@A, 2.0);",
             "float m=(@A.r+@A.g+@A.b)/3.0; @OUT = vec4(m,m,m,1.0);"]

    def batch(n, dev):
        for i in range(n):
            im = torch.rand(1, 48, 48, 3, device=dev)
            tex_engine.cook(progs[i % len(progs)], {"A": im}, device_mode=dev)

    batch(30, "cpu")                                  # warm
    gc.collect()
    rss0 = proc.memory_info().rss / (1 << 20) if proc else 0.0
    for _ in range(5):
        batch(40, "cpu")
        s.reset()
    gc.collect()
    rss_grow = (proc.memory_info().rss / (1 << 20) - rss0) if proc else 0.0
    r.ok(f"soak: RSS flat over ~230 cooks + 5 resets (+{rss_grow:.1f} MB)") if rss_grow < 60.0 \
        else r.fail("DATA-4 soak-rss", f"RSS grew {rss_grow:.1f} MB")
    if _CUDA:
        torch.cuda.empty_cache(); batch(20, "cuda"); torch.cuda.synchronize()
        v0 = torch.cuda.memory_allocated() / (1 << 20)
        for _ in range(4):
            batch(30, "cuda"); s.reset(); torch.cuda.empty_cache()
        torch.cuda.synchronize()
        vg = torch.cuda.memory_allocated() / (1 << 20) - v0
        r.ok(f"soak: VRAM flat over cuda cooks + resets ({vg:+.1f} MB)") if vg < 60.0 \
            else r.fail("DATA-4 soak-vram", f"VRAM grew {vg:.1f} MB")


# ── PORT-5 ──────────────────────────────────────────────────────────────────

def test_port5_second_host(r: SubTestResult):
    print("\n--- PORT-5 / PM-2: the standalone host demo ---")
    import re
    ex = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
    sys.path.insert(0, ex)
    saved_modules = set(sys.modules)

    # S-1: the demo imports no comfy. A STATIC source scan (the S-1 regex), not a sys.modules
    # check — the suite harness already loaded the package (and thus comfy on a box that has it),
    # so only the source is a trustworthy signal here. The runtime import-blocker proof lives in
    # the manual run (tests/test_v019_phase2 is the package-wide S-1 lane).
    demo_src = open(os.path.join(ex, "host_demo.py"), encoding="utf-8").read()
    if re.search(r"^\s*(import|from)\s+(comfy|comfy_api|comfy_execution|server|folder_paths)\b",
                 demo_src, re.M):
        r.fail("PORT-5 s1", "host_demo.py imports comfy")
    else:
        r.ok("host_demo.py imports no comfy (static S-1 scan)")

    try:
        import host_demo
        r.ok("examples/host_demo imports standalone")
        host = host_demo.Host(256)
        frame, ms, hit = host.cook(0.5)
        r.ok("demo cooks the fused grade->blur->vignette chain standalone") \
            if (frame.shape == (1, 256, 256, 4) and not hit) else r.fail("PORT-5 cook", f"{frame.shape}")

        # CACHE-2: scrubbing back to a visited strength is a cache HIT (no recook)
        _, ms2, hit2 = host.cook(0.5)
        r.ok("CACHE-2: revisiting a strength is a cache HIT (param scrub)") \
            if (hit2 and ms2 == 0.0) else r.fail("PORT-5 cache", f"hit={hit2} ms={ms2}")

        # SCHED-3: a superseded cook aborts
        tok = host.new_request()
        host.new_request()                            # supersede tok
        from TEX_Wrangle.tex_runtime.host import CookCancelled
        try:
            host.cook(0.999, cancel=tok)
            r.fail("PORT-5 cancel", "superseded cook did not abort")
        except CookCancelled:
            r.ok("SCHED-3: a superseded cook aborts (CookCancelled)")

        # PM-2: the warm 1024^2 cook budget. Assert the pass on CUDA; on CPU just require it runs.
        ok = host_demo.run_benchmark()
        if _CUDA:
            r.ok("PM-2: <50 ms/frame warm at 1024^2 (CUDA)") if ok \
                else r.fail("PORT-5 pm2", "over 50 ms warm on CUDA")
        else:
            r.ok("PM-2 benchmark runs (CPU; the <50 ms gate is the sm_120 box)")
    finally:
        # host_demo.Host.__init__ flips two process globals (the engine egress profile and the host
        # services to Null); restore both so a comfy box is unperturbed for anything after.
        tex_marshalling.set_egress_profile("comfy")
        from TEX_Wrangle.tex_runtime.host import reset_host_services
        reset_host_services()
        for m in list(sys.modules):
            if m not in saved_modules and (m == "host_demo" or m.startswith("host_demo.")):
                del sys.modules[m]


# ── ENG-5 canaries: the DATA-2 / DATA-4 public surfaces ───────────────────────

def test_data_canaries(r: SubTestResult):
    print("\n--- ENG-5 canaries: BufferDesc + EngineSession public surfaces ---")
    # These are consumed by an embedding host — the DATA-2 storage descriptor (a file on disk
    # names its dtype) and the DATA-4 session handle. A silent rename / field-drop is a breaking
    # change no type checker catches, exactly what ENG-5 canaries pin (cf. test_eng5 in v022).
    import dataclasses as _dc
    from TEX_Wrangle.tex_io import BufferDesc, STORAGE_DTYPES
    from TEX_Wrangle.tex_session import EngineSession, default_session
    fails = []

    # (a) BufferDesc — the storage vocabulary AND the frozen dataclass's fields+defaults (a host
    # may construct it positionally, so a reordered/renamed/dropped field is a break).
    if tuple(STORAGE_DTYPES) != ("uint8", "uint16", "float16", "float32"):
        fails.append(f"STORAGE_DTYPES changed: {STORAGE_DTYPES}")
    fields = {f.name: f.default for f in _dc.fields(BufferDesc)}
    if fields != {"storage": "float32", "transfer": "unknown"}:
        fails.append(f"BufferDesc fields/defaults changed: {fields}")
    if not (BufferDesc("float32").is_float and not BufferDesc("uint8").is_float):
        fails.append("BufferDesc.is_float property changed")

    # (b) EngineSession — the method/property surface a long-running host holds (DATA-4 phase 1).
    want = {"cache", "registry", "host", "interpreter", "set_host", "stats", "reset", "close",
            "isolated"}
    got = {n for n in dir(EngineSession) if not n.startswith("_")}
    if got != want:
        fails.append(f"EngineSession surface changed: +{got - want} -{want - got}")
    if default_session() is not default_session():
        fails.append("default_session() is not a process singleton")

    r.fail("ENG-5 DATA canaries", "; ".join(fails)) if fails else \
        r.ok("pinned: STORAGE_DTYPES, BufferDesc(storage/transfer + defaults + is_float), "
             "EngineSession's 9-member surface + default_session singleton")


# ── Root fixes: the C∉{3,4} 4-D-image family + scalar swizzle + $param wire count ─────────────

def test_root_channel_and_swizzle_fixes(r: SubTestResult):
    """Root-cause fixes for five latent defects the v0.28 EXR-ingest audit surfaced (doc 32);
    (5) is the ARRAY-base follow-up to (3) (chip task_c2df490f).
    Each was UNREACHABLE by any shipping path while ComfyUI's IMAGE wire stayed 3-channel and
    load_image's .exr ingest was pinned to C∈{3,4} — so none blocked v0.28 — but all are genuine
    root defects in how the engine handles a 4-D IMAGE `[B,H,W,C]` with C∉{3,4}, or a scalar base:

      (1) tex_marshalling._to_mask_shape: an unguarded LUMA*raw[...,2] IndexError'd on a 1-/2-
          channel 4-D image at a MASK egress. Now channel-count-guarded (missing channels read
          as 0, matching the IMAGE 2ch zero-pad) so a mask egress of any width is DEFINED.
      (2) tex_marshalling.infer_binding_type: C=1 and C≥5 both fell to `else → FLOAT` while the
          tensor stayed 4-D, so a >4-channel passthrough SILENTLY collapsed to 1-channel luma via
          the MASK egress. Now a deliberate, SHARED policy (`_spatial_channels_to_type`, one source
          for the dict AND tensor branches): C=1→FLOAT (lossless mask), C=0/C≥5 refuse. Reachable
          not only via EXR/AOV but via the mainstream LATENT wire — the node unwraps a >4-ch latent
          (SD3/Flux/Wan 16ch, LTX-2 128ch) to [B,H,W,C] before inference, so it now fails LOUD (a
          clean cook error) instead of the old silent luma-MASK that couldn't re-wire to a latent.
      (3) tex_compiler.type_checker: the swizzle bounds check was gated on is_vector, so `.rgb`/
          `.rgba`/`.g` on a FLOAT (scalar, C=1) base was neither errored nor expanded — the
          interpreter then sliced the spatial axis, not channels. The bounds check now covers a
          scalar base (E3301); `.r`/`.x` (index 0) stays valid — a count-preserving reduction no-op
          (e.g. `arr_avg(a@pal).r`), not the wrong-channel-count bug this closes.
      (4) tex_cli.run_program: `referenced − assigned` counted $param names as wire inputs
          (param_info is empty at that compile site), so a legit single-image program with a
          $param (examples/vignette.tex's f$strength) was FALSELY rejected as >1 input.
      (5) tex_compiler.type_checker (follow-up to (3), chip task_c2df490f): channel access on an
          ARRAY base (`a@name`, a DATA-3 wire) hit NONE of the guards — the single/multi bounds
          checks gate on (is_vector or is_scalar), so `a@pal.rgb`/`a@pal.g` fell through, claimed a
          VECn from the swizzle width, and the interpreter then sliced an [...,N,C]/[...,N] array by
          a channel index (undefined/silently-wrong). Now an early is_array guard rejects it (E3300),
          mirroring the string/matrix guards; swizzling a reduction's SCALAR result
          (`arr_avg(a@pal).r`) is a FLOAT base, never reaches the guard, so it stays valid — the
          same load-bearing pattern (3) preserves.

    Each sub-check is independent and mutation-pinned: reverting only its own fix reds it.
    """
    print("\n--- root fixes: low/high-channel egress + scalar swizzle + $param wire count ---")
    from TEX_Wrangle.tex_marshalling import (_to_mask_shape, infer_binding_type,
                                             prepare_output, unwrap_latent)
    from TEX_Wrangle.tex_runtime.stdlib import LUMA_R, LUMA_G, LUMA_B
    from TEX_Wrangle import tex_cli

    # (1) _to_mask_shape of a low-channel 4-D image is DEFINED (was raw[...,2] IndexError)
    try:
        t1 = make_img(1, 6, 5, 1, seed=11)
        t2 = make_img(1, 6, 5, 2, seed=12)
        t3 = make_img(1, 6, 5, 3, seed=13)
        ok1 = (list(_to_mask_shape(t1).shape) == [1, 6, 5]
               and torch.equal(_to_mask_shape(t1), t1[..., 0])                       # c=1: the channel
               and torch.allclose(_to_mask_shape(t2), LUMA_R * t2[..., 0] + LUMA_G * t2[..., 1])  # c=2: blue=0
               and torch.allclose(_to_mask_shape(t3),
                                  LUMA_R * t3[..., 0] + LUMA_G * t3[..., 1] + LUMA_B * t3[..., 2])  # c=3 unchanged
               # through the public MASK egress, both profiles — [1,H,W], no crash
               and list(prepare_output(t1.clone(), "MASK", profile="engine").shape) == [1, 6, 5]
               and list(prepare_output(t2.clone(), "MASK", profile="comfy").shape) == [1, 6, 5])
        r.ok("_to_mask_shape guards low channel counts (1/2-ch image -> [B,H,W], no IndexError)") \
            if ok1 else r.fail("root mask-shape", "a low-channel mask egress diverged")
    except Exception as e:
        r.fail("root mask-shape", f"{type(e).__name__}: {e}")

    # (2) infer_binding_type: deliberate policy for C∉{1,2,3,4}, SHARED by the tensor AND the
    # LATENT-dict branch (they must not drift — a wide latent must refuse either way).
    try:
        okp = (infer_binding_type(make_img(1, 4, 4, 1)) == TEXType.FLOAT
               and infer_binding_type(make_img(1, 4, 4, 2)) == TEXType.VEC2
               and infer_binding_type(make_img(1, 4, 4, 3)) == TEXType.VEC3
               and infer_binding_type(make_img(1, 4, 4, 4)) == TEXType.VEC4)
        refused = 0
        for c in (5, 6):
            try:
                infer_binding_type(make_img(1, 4, 4, c))
            except ValueError:
                refused += 1
        # dict-branch parity: a 4-ch latent dict -> VEC4; a 16-ch (SD3/Flux/Wan) dict REFUSES,
        # exactly like the post-unwrap tensor above (else the node and a raw-dict caller diverge).
        dict_ok = infer_binding_type({"samples": torch.randn(1, 4, 4, 4)}) == TEXType.VEC4
        dict_refused = False
        try:
            infer_binding_type({"samples": torch.randn(1, 16, 4, 4)})
        except ValueError:
            dict_refused = True
        r.ok("infer_binding_type: C=1->FLOAT, C>=5 refuses (tensor AND latent-dict, no luma collapse)") \
            if (okp and refused == 2 and dict_ok and dict_refused) \
            else r.fail("root infer-policy",
                        f"okp={okp} refused={refused} dict_ok={dict_ok} dict_refused={dict_refused}")
    except Exception as e:
        r.fail("root infer-policy", f"{type(e).__name__}: {e}")

    # (1)+(2) end-to-end: a 1-/2-channel passthrough cooks; a 5-channel is refused at ingress; and
    # the REACHABLE latent path — the node unwraps a >4-ch latent to [B,H,W,C] before inference,
    # so a wide-latent cook must refuse (was a silent luma-MASK that couldn't re-wire to a latent).
    try:
        okc = True
        for c, ttype in ((1, TEXType.FLOAT), (2, TEXType.VEC2)):
            res = tex_engine.cook("@OUT = @A;", {"A": make_img(1, 8, 8, c, seed=20 + c)},
                                  device_mode="cpu")
            okc = okc and res.outputs["OUT"].shape[-1] == c and res.assigned["OUT"] == ttype
        for wide in (make_img(1, 8, 8, 5),                                    # a 5-ch IMAGE/EXR
                     unwrap_latent({"samples": torch.randn(1, 16, 8, 8)})[0]):  # a 16-ch LATENT (unwrapped)
            try:
                tex_engine.cook("@OUT = @A;", {"A": wide}, device_mode="cpu")
                okc = False                                      # a >4-ch cook must refuse
            except ValueError:
                pass
        r.ok("passthrough cook: 1->FLOAT, 2->VEC2, 5-ch IMAGE + 16-ch LATENT refused at ingress") if okc \
            else r.fail("root passthrough", "a low/high-channel passthrough diverged")
    except Exception as e:
        r.fail("root passthrough", f"{type(e).__name__}: {e}")

    # (3) scalar `.g`/`.b`/`.a` and multi-swizzles now error E3301 (a scalar has no such channel —
    # the runtime slices the spatial axis, not channels); `.r`/`.x` (index 0) stays valid, a
    # count-preserving reduction no-op (arr_avg(...).r) that would break if rejected; a valid
    # vector swizzle still passes, and vec3(x) remains the correct scalar->vector broadcast.
    def _errs(code):
        return [d.code for d in tex_api.check(code, {}) if d.severity == "error"]
    s_rgb = _errs("float x = 0.5; @OUT = vec4(x.rgb, 1.0);")     # multi-swizzle on a scalar
    s_g   = _errs("float x = 0.5; @OUT = vec4(x.g, x.g, x.g, 1.0);")  # out-of-range single channel
    s_r   = _errs("float x = 0.5; @OUT = vec4(x.r, x.r, x.r, 1.0);")  # .r (index 0): count-preserving, stays valid
    v_rgb = _errs("@OUT = vec4(@A.rgb, 1.0);")                   # VEC4 base — must NOT regress
    v_bcast = _errs("float x = 0.5; @OUT = vec4(vec3(x), 1.0);") # the correct broadcast — must stay clean
    ok3 = ("E3301" in s_rgb and "E3301" in s_g and s_r == [] and v_rgb == [] and v_bcast == [])
    r.ok("scalar .rgb/.g -> E3301; scalar .r, vector .rgb, and vec3(x) broadcast still valid") \
        if ok3 else r.fail("root scalar-swizzle", f"rgb={s_rgb} g={s_g} r={s_r} vrgb={v_rgb} bcast={v_bcast}")

    # (4) run_program excludes $param names from its wire-input count (vignette.tex has f$strength)
    try:
        ex_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
        with open(os.path.join(ex_dir, "vignette.tex"), encoding="utf-8") as fh:
            vign = fh.read()
        out = tex_cli.run_program(vign, make_img(1, 16, 16, 3, seed=31))
        ok4 = tuple(out.shape) == (1, 16, 16, 3)
        # the >1-wire-input guard is PRESERVED — a genuine 2-input program still refuses
        try:
            tex_cli.run_program("@OUT = @A + @B;", make_img(1, 8, 8, 3))
            ok4 = False
        except ValueError:
            pass
        r.ok("run_program: a $param single-image program (vignette.tex) runs; 2-wire still refused") \
            if ok4 else r.fail("root param-wire-count", "vignette rejected or the >1-input guard broke")
    except Exception as e:
        r.fail("root param-wire-count", f"{type(e).__name__}: {e}")

    # (5) channel access directly on an ARRAY base (`a@name`) errors E3300 (follow-up to (3)): the
    # single/multi bounds checks below the guard gate on (is_vector or is_scalar), so an array
    # swizzle used to fall through and mis-slice at runtime. Direct assignment keeps each error pin
    # to a single code (a vec4 wrapper would add an E3601 arity error off the FLOAT return). The
    # clean case swizzles arr_avg's SCALAR result — a FLOAT base that never reaches the guard.
    a_rgb    = _errs("@OUT = a@pal.rgb;")                              # multi-swizzle on an array base
    a_g      = _errs("@OUT = a@pal.g;")                               # single channel on an array base
    a_reduce = _errs("@OUT = vec4(arr_avg(a@pal).r, 0.0, 0.0, 1.0);")  # scalar reduction .r — clean
    ok5 = ("E3300" in a_rgb and "E3300" in a_g and a_reduce == [])
    r.ok("array-base .rgb/.g -> E3300; arr_avg(a@pal).r (scalar result) stays clean") \
        if ok5 else r.fail("root array-channel-access", f"rgb={a_rgb} g={a_g} reduce={a_reduce}")
