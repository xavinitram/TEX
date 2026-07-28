"""XPU-2 (v0.33) — engine-owned async D2H egress: frame handles that carry a CUDA event.

The rule this item lifts has been in the tree since v0.20, at `tex_marshalling`:

    D2H is never non_blocking (a CPU-side read could observe an in-flight buffer).

It is correct for a bare tensor and there is no way to make it wrong: a `torch.Tensor` has
nowhere to keep "not yet". XPU-2 changes the OBJECT, so the rule becomes "a handle may be
async, because every route to its bytes fences first."

The release exit gate names the test that has to exist:

    "egress fences proven by a stress test that consumes frames from the wrong side on purpose"

`test_v033_xpu2_the_fence_is_load_bearing` is that test, and it is DETERMINISTIC rather than
hopeful: a race that has to be caught by luck produces a test that passes on a fast box and
proves nothing. It enqueues heavy GPU work ahead of the copy, so the copy provably cannot have
started, then reads the staging buffer from the wrong side and shows it holds the sentinel —
and that the same read after `wait()` holds the frame. If the fence were removed, the second
half would return the sentinel too.
"""
import tempfile

import torch

from TEX_Wrangle import tex_results
from TEX_Wrangle.tex_runtime import streams


def _has_cuda():
    return torch.cuda.is_available()


def _big(device="cuda", mb=64, tag=0.0):
    """A frame in the pinned-staging band (1-256 MB) and large enough that a D2H is not
    instantaneous. Below the band `egress` correctly returns a synchronous handle, which
    would make the asynchrony rows vacuous.

    `tag` makes the CONTENT unique per caller, and that is load-bearing rather than tidy:
    torch's caching host allocator RECYCLES pinned blocks, so a second egress of the same
    bytes can be handed a buffer that already contains them. The fence test below compared an
    unfenced read against the truth and they matched — not because the copy had landed (the
    event said it had not) but because the block still held an identical frame from an earlier
    row. A test that can be satisfied by the allocator is not testing the fence."""
    n = (mb << 20) // 4
    side = int((n / 4) ** 0.5)
    return (torch.arange(side * side * 4, dtype=torch.float32, device=device).reshape(
        1, side, side, 4) * 1e-6) + tag


# ── the contract ──────────────────────────────────────────────────────────────

def test_v033_xpu2_handle_metadata_never_fences(r):
    """The asymmetry that makes a handle worth having: shape/dtype/nbytes are decided when the
    copy is ISSUED, so a consumer that only needs to know how big a frame is — the byte
    accounting a cache does on every eviction — never waits for one."""
    if not _has_cuda():
        r.ok("XPU-2: metadata SKIPPED (no CUDA)")
        return
    src = _big()
    torch.cuda.synchronize()
    h = streams.egress(src)
    meta = (tuple(h.shape), h.dtype, h.nbytes(), str(h.device))
    ok = (meta == (tuple(src.shape), torch.float32, src.numel() * 4, "cpu")
          and isinstance(h.is_ready(), bool))
    h.wait()
    r.ok("XPU-2: shape/dtype/nbytes/is_ready are answerable without fencing") if ok else \
        r.fail("XPU-2 metadata", f"{meta}")


def test_v033_xpu2_fenced_read_is_bit_exact(r):
    """The thing that must never be traded for speed. Every route through the handle returns
    the frame exactly, on every size and on both devices."""
    devs = ["cpu"] + (["cuda"] if _has_cuda() else [])
    bad = []
    for dev in devs:
        for mb in (0.25, 8, 64):
            src = _big(dev, mb=max(1, int(mb))) if mb >= 1 else torch.rand(1, 64, 64, 4,
                                                                          device=dev)
            got = streams.egress(src).tensor()
            if not torch.equal(got, src.cpu()):
                bad.append((dev, mb, float((got - src.cpu()).abs().max())))
    r.ok(f"XPU-2: a fenced read is bit-exact across {len(devs)} device(s) x 3 sizes") \
        if not bad else r.fail("XPU-2 exact", f"{bad}")


def test_v033_xpu2_the_fence_is_load_bearing(r):
    """THE EXIT-GATE ROW: consume a frame from the wrong side, on purpose, deterministically.

    Heavy GPU work is enqueued on the stream BEFORE the copy, so the copy cannot have begun
    when `egress` returns. Reading `unsafe_buffer()` at that moment must show the buffer's
    pre-copy contents; reading it after `wait()` must show the frame. A missing fence collapses
    the two, which is precisely the silent wrong-frame this handle exists to prevent."""
    if not _has_cuda():
        r.ok("XPU-2: fence stress SKIPPED (no CUDA — the copy is synchronous by definition)")
        return
    def _behind_ballast(tag):
        """Issue an egress that provably cannot have started: heavy GPU work is enqueued on the
        stream first, so the copy is still behind it when `egress` returns. The content `tag` is
        unique per call so a RECYCLED pinned block cannot coincidentally hold these bytes."""
        src = _big(mb=64, tag=tag)
        torch.cuda.synchronize()
        ballast = torch.rand(2048, 2048, device="cuda")
        for _ in range(60):
            ballast = ballast @ ballast.T * 1e-4
        return src, streams.egress(src)

    # HALF ONE — `tensor()` must FENCE. This is read FIRST, while the copy is provably still in
    # flight, and nothing else is touched in between. An earlier draft read `unsafe_buffer()`
    # first and only then called `tensor()`, by which time the copy had landed anyway — so the
    # mutation "make tensor() return self._host without waiting" SURVIVED. A fence test whose
    # fenced read happens late is not testing the fence.
    src_a, ha = _behind_ballast(7.25)
    ready_a = ha.is_ready()
    fenced = ha.tensor().clone()
    fenced_right = torch.equal(fenced, src_a.cpu())

    # HALF TWO — the counterfactual. Same setup, but read from the WRONG side on purpose.
    src_b, hb = _behind_ballast(3.5)
    ready_b = hb.is_ready()
    early = hb.unsafe_buffer().clone()
    early_wrong = not torch.equal(early, src_b.cpu())
    hb.wait()

    ok = (not ready_a) and fenced_right and (not ready_b) and early_wrong
    r.ok(f"XPU-2: tensor() fences (exact while in flight); the unfenced read is WRONG "
         f"(is_ready before either: {ready_a}/{ready_b})") if ok else \
        r.fail("XPU-2 fence", f"ready={ready_a}/{ready_b} fenced_exact={fenced_right} "
                              f"unfenced_differs={early_wrong} — if the unfenced read matched, "
                              f"the copy completed despite the ballast and this box cannot "
                              f"demonstrate the race")


def test_v033_xpu2_declines_asynchrony_rather_than_faking_it(r):
    """`egress` returns an already-complete handle whenever async is unavailable or wrong —
    a CPU source, a dtype conversion, a size outside the pinned band, or `retained=True`. The
    caller writes one code path either way, which is the point: an API whose fast path has a
    different SHAPE from its slow path grows a consumer that only fences on one of them."""
    devs = ["cpu"] + (["cuda"] if _has_cuda() else [])
    rows = {}
    small = torch.rand(1, 8, 8, 4, device=devs[-1])
    rows["below the pin band"] = streams.egress(small).is_ready()
    rows["cpu source"] = streams.egress(torch.rand(1, 512, 512, 4)).is_ready()
    if _has_cuda():
        big = _big(mb=64)
        torch.cuda.synchronize()
        rows["dtype conversion"] = streams.egress(big, dtype=torch.float16).is_ready()
        rows["retained=True"] = streams.egress(big, retained=True).is_ready()
    ok = all(rows.values())
    r.ok(f"XPU-2: {len(rows)} non-async cases return complete handles, same interface") if ok \
        else r.fail("XPU-2 decline", f"{rows}")


def test_v033_xpu2_wait_is_idempotent_and_releases_the_source(r):
    """A handle pins its SOURCE until the fence — dropping the last reference to a tensor a DMA
    is still reading returns its block to torch's caching allocator, which may hand it to
    another stream with no ordering relationship. After the fence that reason is gone, so the
    reference is dropped: a handle held for a while must not keep a VRAM frame alive."""
    if not _has_cuda():
        r.ok("XPU-2: source release SKIPPED (no CUDA)")
        return
    src = _big(mb=8)
    torch.cuda.synchronize()
    h = streams.egress(src)
    held_before = h._src is not None
    h.wait()
    h.wait()                                    # idempotent
    t1, t2 = h.tensor(), h.tensor()
    ok = held_before and h._src is None and t1 is t2 and h.is_ready()
    r.ok("XPU-2: wait() is idempotent and drops the source reference once fenced") if ok else \
        r.fail("XPU-2 release", f"held_before={held_before} src={h._src is None} "
                                f"same={t1 is t2}")


# ── the two engine-owned consumers ────────────────────────────────────────────

def test_v033_xpu2_spill_round_trips_through_the_handle(r):
    """The spill path is one of exactly two consumers, and both are in `tex_results.py`. This
    is the whole reason engine custody makes async egress admissible: the consumer list is
    finite, in-repo, and reviewable — under ComfyUI it would be an arbitrary third-party node."""
    devs = ["cpu"] + (["cuda"] if _has_cuda() else [])
    bad = []
    for dev in devs:
        with tempfile.TemporaryDirectory() as d:
            src = torch.rand(1, 96, 96, 4, device=dev)
            c = tex_results.ResultCache(cache_dir=d, budget_mb=0)
            c.put("a", src)
            c.put("b", torch.rand(1, 96, 96, 4, device=dev))    # forces "a" to disk
            got = c.get("a")
            if got is None or not torch.equal(got.to(dev), src):
                bad.append((dev, "None" if got is None else
                            float((got.to(dev) - src).abs().max())))
            c.clear(disk=True)
    r.ok("XPU-2: the spill path round-trips bit-exact through the fenced handle") if not bad \
        else r.fail("XPU-2 spill", f"{bad}")


def test_v033_xpu2_is_engine_only(r):
    """Engine custody is the entire safety argument, so it is asserted as a source fact rather
    than a convention. The default ComfyUI path must be unable to reach a handle — and the
    consumer list must stay short enough to review, which is why this counts them."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    node = (root / "tex_node.py").read_text(encoding="utf-8")
    # NOT a bare "egress": `tex_marshalling.egress_materializes` is a pre-existing and unrelated
    # concept the node legitimately imports, and matching it made this canary fire on shipped
    # code. Match the names that only XPU-2 owns.
    leaked = [n for n in ("tex_runtime.streams", "FrameHandle", "streams.egress",
                          "unsafe_buffer") if n in node]
    consumers = sorted(p.name for p in root.rglob("*.py")
                       if p.name not in ("streams.py",)
                       and "tests" not in p.parts and "benchmarks" not in p.parts
                       and "from .tex_runtime.streams import" in p.read_text(encoding="utf-8"))
    ok = not leaked and consumers == ["tex_results.py"]
    r.ok(f"XPU-2: engine-only — tex_node cannot reach it; consumers = {consumers}") if ok else \
        r.fail("XPU-2 custody", f"tex_node mentions {leaked}; consumers={consumers}")
