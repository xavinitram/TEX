"""
v0.37 DATA-6 L-D — EXR layer grouping: `read_layers` / `write_layers` in `tex_io/exr.py`.

The FILE half of PM-10 ("a multi-layer EXR round-trips losslessly"). An EXR *layer* is a
naming convention inside one part's channel list (`beauty.R`, `specular.G`, a bare `Z`); the
DATA-2 reader already returns those channels verbatim and round-trips them bitwise, so the
lane is a grouping rule over `ExrImage` plus its inverse flattener — never a decoder change.
Multipart / deep / tiled stay refused, and this file pins that they still are.

Every fixture is written at runtime by the product's own writer (or byte-patched from what it
wrote): no committed binary. CPU only, no tier, no dispatcher, no numpy (invariant 1). The
bitwise assertions compare two tensors produced in this process by the same torch build
through a file the process itself wrote — the exact shape LNT-2's escape hatch exists for.
"""
import os
import struct
import tempfile

import torch

from helpers import *  # noqa: F401,F403  (SubTestResult)
from TEX_Wrangle.tex_io import BufferDesc
from TEX_Wrangle.tex_io import exr as tex_exr

_H, _W = 4, 5


def _distinct(n, seed=0):
    """[H,W,n] fp32 whose channel i lives around 10*i, so a mis-ordered plane is visible."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(_H, _W, n, generator=g) * 4.0 - 2.0
    return x + 10.0 * torch.arange(n, dtype=torch.float32)


def _raw(td, fname, names, **kw):
    """Write a raw layered file through the DATA-2 writer (a foreign writer's shape: verbatim
    channel names, one part) → (path, {name: [H,W] source channel})."""
    x = _distinct(len(names))
    p = os.path.join(td, fname)
    tex_exr.write_exr(p, x, channels=names, **kw)
    return p, {n: x[..., i] for i, n in enumerate(names)}


def _patch_pixel_type(blob: bytes, channel: str, ptype: int) -> bytes:
    """Rewrite one chlist entry's pixelType in place (UINT/HALF/FLOAT are all 4 or 2 bytes a
    sample — only the 4-byte ones are patched here, so the block byte-math is untouched)."""
    start = blob.index(b"channels\x00chlist\x00")
    at = blob.index(channel.encode("latin-1") + b"\x00", start) + len(channel) + 1
    return blob[:at] + struct.pack("<i", ptype) + blob[at + 4:]


def _patch_version_flags(blob: bytes, flags: int) -> bytes:
    (version,) = struct.unpack_from("<i", blob, 4)
    return blob[:4] + struct.pack("<i", version | (flags << 8)) + blob[8:]


def _expect_exr_error(fn, *needles):
    """Run `fn`; it must raise EXRError whose message contains every needle. Returns the
    message, or raises AssertionError naming what was wrong."""
    try:
        fn()
    except tex_exr.EXRError as e:
        msg = str(e)
        for n in needles:
            assert n in msg, f"EXRError did not name {n!r}: {msg}"
        return msg
    except Exception as e:  # pragma: no cover - a raw error is the defect being pinned
        raise AssertionError(f"raised {type(e).__name__} instead of EXRError: {e}")
    raise AssertionError("did not raise")


# ── the round-trip (F6's probe shape, promoted into a test) ──────────────────

def test_two_layers_and_a_bare_Z_round_trip_bitwise(r: SubTestResult):
    print("\n--- DATA-6 L-D: two layers + a bare Z round-trip BITWISE per plane ---")
    names = ["beauty.R", "beauty.G", "beauty.B",
             "specular.R", "specular.G", "specular.B", "Z"]
    with tempfile.TemporaryDirectory() as td:
        try:
            p1, src = _raw(td, "f6.exr", names)
            first = tex_exr.read_layers(p1)
            assert set(first) == {"beauty", "specular", "Z"}, sorted(first)
            for plane in ("beauty", "specular"):
                t, desc = first[plane]
                assert tuple(t.shape) == (_H, _W, 3), tuple(t.shape)
                expected = torch.stack([src[f"{plane}.{c}"] for c in "RGB"], dim=-1)
                assert torch.equal(t, expected), plane  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
                assert desc == BufferDesc(storage="float32", transfer="linear"), desc
            z, zdesc = first["Z"]
            assert tuple(z.shape) == (_H, _W, 1), tuple(z.shape)
            assert torch.equal(z[..., 0], src["Z"])  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            assert zdesc.storage == "float32" and zdesc.transfer == "linear"
            r.ok("read_layers: beauty [H,W,3] R,G,B + specular [H,W,3] + Z [H,W,1], "
                 "bitwise against the written channels, desc float32/linear")

            with open(p1, "rb") as f:
                blob = f.read()
            again = tex_exr.read_layers(blob)
            assert set(again) == set(first)
            assert all(torch.equal(again[k][0], first[k][0]) for k in first)  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            r.ok("read_layers accepts raw bytes as read_exr does")

            # The PM-10 file half: grouped read -> write_layers -> grouped read, bitwise.
            p2 = os.path.join(td, "rt.exr")
            tex_exr.write_layers(p2, {k: v[0] for k, v in first.items()})
            second = tex_exr.read_layers(p2)
            assert set(second) == set(first), sorted(second)
            for k in first:
                assert torch.equal(second[k][0], first[k][0]), k  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
                assert second[k][1] == first[k][1], k
            r.ok("read_layers -> write_layers -> read_layers is bitwise per plane, desc kept")

            # Spelling: the default `root='beauty'` writes the beauty plane as the bare root
            # group (a Nuke/Arnold root file keeps its own names); `root=None` writes every
            # plane as an explicit layer, which is exactly this file's own spelling.
            assert tex_exr.read_exr(p2).channels == \
                ["B", "G", "R", "Z", "specular.B", "specular.G", "specular.R"], \
                tex_exr.read_exr(p2).channels
            p3 = os.path.join(td, "rt_layered.exr")
            tex_exr.write_layers(p3, {k: v[0] for k, v in first.items()}, root=None)
            assert tex_exr.read_exr(p3).channels == sorted(names), tex_exr.read_exr(p3).channels
            third = tex_exr.read_layers(p3)
            assert all(torch.equal(third[k][0], first[k][0]) for k in first)  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            r.ok("write_layers spelling: root='beauty' -> bare R,G,B; root=None -> the "
                 "file's own layer.channel names, verbatim")

            # HALF storage: per-plane desc says float16 and the values are half-rounded —
            # the storage dtype's own precision, nothing lost beyond it (DATA-2's contract).
            p4 = os.path.join(td, "rt_half.exr")
            tex_exr.write_layers(p4, {k: v[0] for k, v in first.items()}, half=True)
            halves = tex_exr.read_layers(p4)
            for k in first:
                assert halves[k][1].storage == "float16", (k, halves[k][1])
                assert torch.equal(halves[k][0], first[k][0].half().float()), k  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            r.ok("write_layers(half=True): per-plane desc float16, values half-rounded")
        except Exception as e:
            r.fail("DATA-6 L-D round-trip", f"{type(e).__name__}: {e}")


# ── the grouping rule, each clause with the mutation that would break it ─────

def test_grouping_splits_on_the_last_dot_and_orders_rgba(r: SubTestResult):
    print("\n--- DATA-6 L-D: last-dot split, R,G,B,A order, sorted fallback ---")
    names = ["beauty.diffuse.B", "beauty.diffuse.G", "beauty.diffuse.R",   # two-dot layer
             "N.Z", "N.Y", "N.X",                                          # non-RGBA set
             "glow.R", "glow.G", "glow.B", "glow.A"]                       # full RGBA
    with tempfile.TemporaryDirectory() as td:
        try:
            p, src = _raw(td, "rule.exr", names)
            planes = tex_exr.read_layers(p)
            # A first-dot split would file these under `beauty` with channels `diffuse.R`.
            assert set(planes) == {"beauty.diffuse", "N", "glow"}, sorted(planes)
            r.ok("a channel splits on its LAST dot: `beauty.diffuse.R` -> plane `beauty.diffuse`")

            # The file stores B,G,R (sorted); the plane must come back R,G,B.
            d = planes["beauty.diffuse"][0]
            assert torch.equal(d[..., 0], src["beauty.diffuse.R"])  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            assert torch.equal(d[..., 1], src["beauty.diffuse.G"])  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            assert torch.equal(d[..., 2], src["beauty.diffuse.B"])  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            g = planes["glow"][0]
            assert tuple(g.shape) == (_H, _W, 4)
            assert torch.equal(g[..., 3], src["glow.A"]) and torch.equal(g[..., 0], src["glow.R"])  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            r.ok("an R,G,B(,A) subset comes back in R,G,B,A order, not the file's B,G,R")

            # A non-RGBA set keeps the file's own sorted order — never forced into R,G,B slots.
            n = planes["N"][0]
            assert torch.equal(n[..., 0], src["N.X"]) and torch.equal(n[..., 2], src["N.Z"])  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            r.ok("a non-RGBA layer (`N.X/Y/Z`) keeps the file's sorted order X,Y,Z")

            # Inverse: the grouped view round-trips even though the spellings normalise.
            p2 = os.path.join(td, "rule_rt.exr")
            tex_exr.write_layers(p2, {k: v[0] for k, v in planes.items()})
            back = tex_exr.read_layers(p2)
            assert set(back) == set(planes)
            assert all(torch.equal(back[k][0], planes[k][0]) for k in planes)  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            assert "N.R" in tex_exr.read_exr(p2).channels    # `N.X` re-written by position
            r.ok("write_layers normalises spellings (`N.X` -> `N.R`); the grouped view round-trips")
        except Exception as e:
            r.fail("DATA-6 L-D grouping rule", f"{type(e).__name__}: {e}")


def test_bare_names_are_own_planes_and_root_rgba_is_beauty(r: SubTestResult):
    print("\n--- DATA-6 L-D: root R/G/B/A -> `beauty`; any other bare name -> its own plane ---")
    names = ["R", "G", "B", "A", "Z", "N", "id"]
    with tempfile.TemporaryDirectory() as td:
        try:
            p, src = _raw(td, "root.exr", names)
            planes = tex_exr.read_layers(p)
            # Not grouping the root colour channels would leave planes `R`,`G`,`B`,`A`;
            # folding `Z` into the root group would lose the `Z` plane.
            assert set(planes) == {"beauty", "Z", "N", "id"}, sorted(planes)
            b = planes["beauty"][0]
            assert tuple(b.shape) == (_H, _W, 4)
            assert all(torch.equal(b[..., i], src[c]) for i, c in enumerate("RGBA"))  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            for bare in ("Z", "N", "id"):
                t = planes[bare][0]
                assert tuple(t.shape) == (_H, _W, 1), (bare, tuple(t.shape))
                assert torch.equal(t[..., 0], src[bare]), bare  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            r.ok("root R,G,B,A -> plane `beauty` [H,W,4]; Z / N / id -> [H,W,1] planes of their own name")

            # Flatten back: these came from bare names, so they go back to bare names.
            p2 = os.path.join(td, "root_rt.exr")
            tex_exr.write_layers(p2, {k: v[0] for k, v in planes.items()})
            assert tex_exr.read_exr(p2).channels == sorted(names), tex_exr.read_exr(p2).channels
            back = tex_exr.read_layers(p2)
            assert all(torch.equal(back[k][0], planes[k][0]) for k in planes)  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            r.ok("write_layers: root/beauty and bare single-channel planes flatten back to "
                 "the names they came from (channel set identical)")

            # A lone root colour channel still groups (a 1-channel beauty), not a plane `A`.
            p3, src3 = _raw(td, "lone.exr", ["A", "Z"])
            lone = tex_exr.read_layers(p3)
            assert set(lone) == {"beauty", "Z"} and tuple(lone["beauty"][0].shape) == (_H, _W, 1)
            r.ok("a lone root `A` is a 1-channel `beauty`, never a plane named `A`")
        except Exception as e:
            r.fail("DATA-6 L-D bare/root", f"{type(e).__name__}: {e}")


def test_beauty_layer_vs_root_precedence(r: SubTestResult):
    print("\n--- DATA-6 L-D: explicit `beauty` layer vs root R/G/B ---")
    with tempfile.TemporaryDirectory() as td:
        try:
            # Explicit layer alone: the plane IS the layer.
            p, src = _raw(td, "explicit.exr", ["beauty.R", "beauty.G", "beauty.B", "Z"])
            planes = tex_exr.read_layers(p)
            assert set(planes) == {"beauty", "Z"}
            assert torch.equal(planes["beauty"][0][..., 1], src["beauty.G"])  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            r.ok("an explicit `beauty.*` layer is the `beauty` plane")

            # Root alone: the root group takes the name.
            p, src = _raw(td, "root.exr", ["R", "G", "B", "Z"])
            planes = tex_exr.read_layers(p)
            assert set(planes) == {"beauty", "Z"}
            assert torch.equal(planes["beauty"][0][..., 2], src["B"])  # lnt2-ok: same-process torch round-trip, no dispatcher, no file-text hash
            r.ok("root R,G,B with no `beauty` layer is the `beauty` plane")

            # Both: refused, loudly, naming both groups. The container's root layer has no
            # name of its own (it is the empty string), so there is no second name to give
            # the root group — a silent `""` plane would be unaddressable and invisible.
            p, _ = _raw(td, "both.exr", ["R", "G", "B", "beauty.R", "beauty.G", "beauty.B"])
            msg = _expect_exr_error(lambda: tex_exr.read_layers(p), "root", "beauty")
            r.ok(f"root R,G,B beside an explicit `beauty` layer -> EXRError: {msg[:72]}...")
            # ...and the raw reader is untouched by the refusal.
            assert len(tex_exr.read_exr(p).channels) == 6
            r.ok("read_exr still reads the same file's 6 channels verbatim")

            # The same collision shape for any name: a bare `Z` beside a `Z.*` layer.
            p, _ = _raw(td, "zz.exr", ["Z", "Z.R", "Z.G"])
            _expect_exr_error(lambda: tex_exr.read_layers(p), "'Z'")
            r.ok("a bare `Z` beside a `Z.*` layer -> EXRError naming the plane")
        except Exception as e:
            r.fail("DATA-6 L-D precedence", f"{type(e).__name__}: {e}")


# ── the refusals ────────────────────────────────────────────────────────────

def test_more_than_four_channels_is_refused_naming_the_layer(r: SubTestResult):
    print("\n--- DATA-6 L-D: a >4-channel layer is an EXRError naming the layer ---")
    names = ["big.A", "big.B", "big.C", "big.D", "big.E", "Z"]
    with tempfile.TemporaryDirectory() as td:
        try:
            p, _ = _raw(td, "big.exr", names)
            msg = _expect_exr_error(lambda: tex_exr.read_layers(p), "'big'", "5")
            r.ok(f"read_layers: {msg[:80]}...")
            assert tuple(tex_exr.read_exr(p).pixels.shape) == (_H, _W, 6)
            r.ok("read_exr still reads the raw 6 channels (the refusal is the grouping's, not the decoder's)")

            _expect_exr_error(lambda: tex_exr.write_layers(os.path.join(td, "w.exr"),
                                                           {"wide": torch.rand(_H, _W, 5)}),
                              "'wide'", "5")
            r.ok("write_layers refuses a 5-channel plane, naming it")
        except Exception as e:
            r.fail("DATA-6 L-D >4 channels", f"{type(e).__name__}: {e}")


def test_uint_channel_is_refused_loudly(r: SubTestResult):
    print("\n--- DATA-6 L-D: a UINT channel raises, never a silent float plane ---")
    with tempfile.TemporaryDirectory() as td:
        try:
            # A bare UINT id plane (the cryptomatte shape), patched into a FLOAT file's chlist.
            p, _ = _raw(td, "id.exr", ["R", "G", "B", "id"], compression="none")
            with open(p, "rb") as f:
                blob = f.read()
            uint_blob = _patch_pixel_type(blob, "id", 0)
            img = tex_exr.read_exr(uint_blob)                 # DATA-2: UINT is read-only as fp32
            assert img.pixel_types == (2, 2, 2, 0), img.pixel_types
            assert tex_exr.read_exr(blob).pixel_types == (2, 2, 2, 2)
            r.ok("read_exr is unchanged: UINT still decodes, and pixel_types reports it per channel")
            msg = _expect_exr_error(lambda: tex_exr.read_layers(uint_blob), "UINT", "'id'")
            r.ok(f"read_layers: {msg[:80]}...")

            # Inside a layer: the message names the LAYER.
            p, _ = _raw(td, "crypto.exr", ["crypto.R", "crypto.G", "Z"], compression="none")
            with open(p, "rb") as f:
                blob = f.read()
            _expect_exr_error(lambda: tex_exr.read_layers(_patch_pixel_type(blob, "crypto.R", 0)),
                              "UINT", "'crypto'", "R")
            r.ok("a UINT channel inside a layer -> EXRError naming the layer and the channel")

            # The unpatched twin reads fine (the refusal is the pixel type, not the name).
            assert set(tex_exr.read_layers(blob)) == {"crypto", "Z"}
            r.ok("the same names at FLOAT group normally")
        except Exception as e:
            r.fail("DATA-6 L-D UINT", f"{type(e).__name__}: {e}")


def test_multipart_deep_tiled_still_refused(r: SubTestResult):
    print("\n--- DATA-6 L-D: multipart / deep / tiled refusals unchanged ---")
    with tempfile.TemporaryDirectory() as td:
        try:
            p, _ = _raw(td, "flags.exr", ["beauty.R", "beauty.G", "beauty.B", "Z"])
            with open(p, "rb") as f:
                blob = f.read()
            assert set(tex_exr.read_layers(blob)) == {"beauty", "Z"}    # the unflagged twin
            for flag, needle in ((0x10, "multipart"), (0x8, "deep"), (0x2, "tiled")):
                bad = _patch_version_flags(blob, flag)
                _expect_exr_error(lambda: tex_exr.read_exr(bad), needle)
                _expect_exr_error(lambda: tex_exr.read_layers(bad), needle)
            r.ok("version flags 0x10 / 0x8 / 0x2 -> EXRError from read_exr AND read_layers, same message")
        except Exception as e:
            r.fail("DATA-6 L-D flags", f"{type(e).__name__}: {e}")


def test_write_layers_input_contract(r: SubTestResult):
    print("\n--- DATA-6 L-D: write_layers shapes and refusals ---")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "w.exr")
        try:
            # It takes every shape write_exr takes, per plane.
            tex_exr.write_layers(p, {"beauty": torch.rand(1, _H, _W, 3),   # [1,H,W,C]
                                     "Z": torch.rand(1, _H, _W),           # [1,H,W] batched mask
                                     "N": torch.rand(_H, _W)})             # [H,W] mask
            back = tex_exr.read_layers(p)
            assert {k: tuple(v[0].shape) for k, v in back.items()} == \
                {"beauty": (_H, _W, 3), "Z": (_H, _W, 1), "N": (_H, _W, 1)}
            r.ok("write_layers accepts [1,H,W,C] / [1,H,W] / [H,W] per plane (write_exr's rule)")

            _expect_exr_error(lambda: tex_exr.write_layers(p, {}), "at least one")
            _expect_exr_error(lambda: tex_exr.write_layers(p, {"a": torch.rand(_H, _W, 3),
                                                               "b": torch.rand(_H + 1, _W, 3)}),
                              "'b'", "extent")
            _expect_exr_error(lambda: tex_exr.write_layers(p, {"beauty": torch.rand(_H, _W, 3),
                                                               "R": torch.rand(_H, _W)}),
                              "duplicate", "R")
            _expect_exr_error(lambda: tex_exr.write_layers(p, {"": torch.rand(_H, _W, 3)}),
                              "non-empty")
            _expect_exr_error(lambda: tex_exr.write_layers(p, {"a": torch.rand(2, _H, _W, 3)}),
                              "cannot write")
            r.ok("write_layers refuses: no planes / mismatched extents / duplicate flattened "
                 "names / an empty name / a B>1 batch — each an EXRError")

            # A dotted single-channel plane keeps its dotted name through the round-trip.
            tex_exr.write_layers(p, {"depth.Z": torch.rand(_H, _W, 1)})
            assert set(tex_exr.read_layers(p)) == {"depth.Z"}
            r.ok("a dotted 1-channel plane (`depth.Z`) is written `depth.Z.R` and reads back as `depth.Z`")
        except Exception as e:
            r.fail("DATA-6 L-D write contract", f"{type(e).__name__}: {e}")
