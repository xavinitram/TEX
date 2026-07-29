"""v0.34 DATA-7 — the host source protocol (`fetch_time` / `sample_time`) and its pool.

Design doc: `docs/frame-providers.md`. The rows follow its §10 definition of done.

INVARIANT #7 posture, stated once: with no provider registered nothing is constructed, the
governor's `media` pool reports 0 bytes, and no cook path reaches this module —
`test_v034_data7_unarmed_costs_nothing` is that assertion, not this sentence.
"""
import torch

from helpers import devices as _devices
from TEX_Wrangle import tex_memory, tex_engine, tex_provider


def _armed(**kw):
    """A synthetic provider on a cleared pool. Every row arms and disarms explicitly —
    the provider is process-global, so a row that leaked one would silently arm the next."""
    tex_provider.reset_provider()
    p = tex_provider.SyntheticFrameProvider(**kw)
    tex_provider.set_provider(p)
    return p


def _cook(code, device="cpu", res=16, **binds):
    b = {"A": torch.rand(1, res, res, 4, device=device)}
    b.update(binds)
    return tex_engine.cook(code, b, device_mode=device, precision="fp32")


# ── the protocol and its Null default ────────────────────────────────────────

def test_v034_data7_null_provider_refuses(r):
    """No provider registered -> E7001, naming the source and what to do.

    The Null default is a real object rather than `None` for exactly this: the refusal
    carries a message instead of an AttributeError saying a field was missing."""
    tex_provider.reset_provider()
    try:
        _cook('@OUT = fetch_time("plate", 0.0, ix, iy);')
        r.fail("DATA-7 null refusal", "an unarmed engine cooked fetch_time without error")
        return
    except Exception as e:
        code = getattr(e, "_code", "")
        if code != tex_provider.E_NO_PROVIDER:
            r.fail("DATA-7 null refusal", f"expected E7001, got {code}: {e}")
            return
        assert "plate" in str(e), str(e)
    r.ok("DATA-7: an unarmed engine refuses fetch_time as E7001, naming the source")


def test_v034_data7_reads_cook_on_both_devices(r):
    """`fetch_time`/`sample_time` produce a frame on the COOK's grid, on every device.

    The source is 32x32 and the cook is 16x16 on purpose: a source's resolution is its own,
    and reading it must not reshape the output."""
    for dev in _devices():
        p = _armed(res=32, device="cpu")
        try:
            a = _cook('@OUT = fetch_time("plate", 12.0, ix, iy);', device=dev)
            b = _cook('@OUT = sample_time("plate", 12.0, u, v);', device=dev)
            for name, res in (("fetch_time", a), ("sample_time", b)):
                out = res.outputs["OUT"]
                assert tuple(out.shape) == (1, 16, 16, 4), f"{name}: {tuple(out.shape)}"
                assert out.device.type == torch.device(dev).type, f"{name}: {out.device}"
                assert torch.isfinite(out).all(), f"{name}: non-finite"
            # The two modes are allowed to differ, and the synthetic provider makes them,
            # so this also proves they were not served from one another's cache entry.
            assert not torch.allclose(a.outputs["OUT"], b.outputs["OUT"])
            r.ok(f"DATA-7: fetch_time/sample_time cook on the cook's grid ({dev})")
        except Exception as e:
            r.fail(f"DATA-7 device cook ({dev})", f"{type(e).__name__}: {e}")
        finally:
            tex_provider.reset_provider()


def test_v034_data7_motion_blur_exemplar(r):
    """The DoD's first exemplar: N shutter samples averaged out of the batch.

    This is the workflow the item exists for — `fetch_frame` cannot express it, because the
    frames are not in the batch. The loop bound is static so the program stays codegen- and
    tier-legal; each iteration asks for a distinct source time."""
    for dev in _devices():
        p = _armed(res=32, rate=24.0)
        try:
            code = ("vec4 acc = vec4(0.0, 0.0, 0.0, 0.0);\n"
                    "for (int i = 0; i < 4; i = i + 1) {\n"
                    "  acc = acc + sample_time(\"plate\", 10.0 + float(i) * 0.25, u, v);\n"
                    "}\n"
                    "@OUT = acc * 0.25;")
            out = _cook(code, device=dev).outputs["OUT"]
            assert tuple(out.shape) == (1, 16, 16, 4), tuple(out.shape)
            assert torch.isfinite(out).all()
            # Four DISTINCT quantized times at rate 24 -> four provider fetches, and the
            # pool holds all four. A blur that fetched one frame four times would be a
            # quantization bug that the pixels alone would not show.
            assert p.fetches == 4, f"{p.fetches} fetches for a 4-sample shutter"
            r.ok(f"DATA-7: motion-blur exemplar cooks over 4 out-of-batch times ({dev})")
        except Exception as e:
            r.fail(f"DATA-7 motion blur ({dev})", f"{type(e).__name__}: {e}")
        finally:
            tex_provider.reset_provider()


def test_v034_data7_temporal_median_exemplar(r):
    """The DoD's second exemplar: a 3-frame temporal median, out of batch.

    median(a,b,c) = max(min(a,b), min(max(a,b), c)) — branch-free, so it vectorizes and
    stays bit-identical between the interpreter and codegen."""
    for dev in _devices():
        _armed(res=32, rate=24.0)
        try:
            code = ('vec4 a = sample_time("plate", 9.0, u, v);\n'
                    'vec4 b = sample_time("plate", 10.0, u, v);\n'
                    'vec4 c = sample_time("plate", 11.0, u, v);\n'
                    '@OUT = max(min(a, b), min(max(a, b), c));')
            out = _cook(code, device=dev).outputs["OUT"]
            assert tuple(out.shape) == (1, 16, 16, 4), tuple(out.shape)
            assert torch.isfinite(out).all()
            r.ok(f"DATA-7: temporal-median exemplar cooks over 3 source frames ({dev})")
        except Exception as e:
            r.fail(f"DATA-7 temporal median ({dev})", f"{type(e).__name__}: {e}")
        finally:
            tex_provider.reset_provider()


# ── the refusals ─────────────────────────────────────────────────────────────

def test_v034_data7_a_per_pixel_time_is_refused(r):
    """A `t` that varies across the grid is E7003, not a silent single-frame read.

    A per-pixel retime is unbounded I/O from one statement — as many source frames as there
    are distinct values. There is no honest cap, and the failure mode of guessing one is
    wrong pixels, so it refuses and says how many distinct times it saw."""
    _armed(res=32)
    try:
        _cook('@OUT = sample_time("plate", u * 10.0, u, v);')
        r.fail("DATA-7 non-uniform time", "a per-pixel t was accepted")
    except Exception as e:
        code = getattr(e, "_code", "")
        if code == tex_provider.E_NONUNIFORM_TIME and "distinct times" in str(e):
            r.ok("DATA-7: a per-pixel time is refused as E7003, with the count named")
        else:
            r.fail("DATA-7 non-uniform time", f"expected E7003, got {code}: {e}")
    finally:
        tex_provider.reset_provider()

    # ...and a UNIFORM tensor-valued time (the `time` builtin, a $param, an expression over
    # them) is the case that must keep working — the check is on variance, not on rank.
    _armed(res=32)
    try:
        out = _cook('@OUT = sample_time("plate", time + 0.5, u, v);').outputs["OUT"]
        assert torch.isfinite(out).all()
        r.ok("DATA-7: a uniform tensor-valued time still reads")
    except Exception as e:
        r.fail("DATA-7 uniform tensor time", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.reset_provider()


def test_v034_data7_a_bad_frame_is_refused(r):
    """A provider returning the wrong shape is E7004 naming the shape it returned.

    C == 4 exactly: `fetch_time` is typed VEC4 and there is no wire to take a channel count
    from, so the declared type is enforced rather than hoped for."""
    class Wrong:
        provider_id = "wrong"

        def __init__(self, t):
            self._t = t

        def fetch_time(self, k, t):
            return self._t

        sample_time = fetch_time

    cases = [("a 3-channel frame", torch.rand(1, 8, 8, 3)),
             ("a batch of frames", torch.rand(4, 8, 8, 4)),
             ("not a tensor", "some pixels")]
    for label, bad in cases:
        tex_provider.reset_provider()
        tex_provider.set_provider(Wrong(bad))
        try:
            _cook('@OUT = fetch_time("plate", 0.0, ix, iy);')
            r.fail("DATA-7 bad frame", f"{label} was accepted")
            return
        except Exception as e:
            if getattr(e, "_code", "") != tex_provider.E_BAD_FRAME:
                r.fail("DATA-7 bad frame",
                       f"{label}: expected E7004, got {getattr(e, '_code', '')}: {e}")
                return
        finally:
            tex_provider.reset_provider()
    r.ok("DATA-7: a wrong-shaped / non-tensor provider frame is refused as E7004")


def test_v034_data7_a_provider_failure_is_named(r):
    """A provider that raises becomes E7002 naming the source, the time, and the host's
    exception — and says whose fault it is, because it is not TEX's."""
    class Broken:
        provider_id = "broken"

        def fetch_time(self, k, t):
            raise OSError("disk fell over")

        sample_time = fetch_time

    tex_provider.reset_provider()
    tex_provider.set_provider(Broken())
    try:
        _cook('@OUT = fetch_time("plate", 7.5, ix, iy);')
        r.fail("DATA-7 provider failure", "a raising provider produced a frame")
    except Exception as e:
        msg, code = str(e), getattr(e, "_code", "")
        if code == tex_provider.E_FETCH_FAILED and "plate" in msg and "disk fell over" in msg:
            r.ok("DATA-7: a raising provider is E7002, naming source, time and cause")
        else:
            r.fail("DATA-7 provider failure", f"expected E7002, got {code}: {e}")
    finally:
        tex_provider.reset_provider()


# ── temporal identity and the pool ───────────────────────────────────────────

def test_v034_data7_quantization_collapses_neighbouring_times(r):
    """23.999999 and 24.0 are ONE frame at rate 1, and two at rate 0 (no quantization).

    The engine quantizes once, at the pool boundary, and passes the quantized value to the
    provider — so the key and the fetch argument are the same number by construction rather
    than by two implementations agreeing."""
    p = _armed(res=16, rate=1.0)
    try:
        _cook('@OUT = fetch_time("plate", 23.999999, ix, iy);')
        _cook('@OUT = fetch_time("plate", 24.0, ix, iy);')
        collapsed = p.fetches
        tex_provider.reset_provider()
        p2 = _armed(res=16, rate=0.0)
        _cook('@OUT = fetch_time("plate", 23.999999, ix, iy);')
        _cook('@OUT = fetch_time("plate", 24.0, ix, iy);')
        distinct = p2.fetches
        assert collapsed == 1, f"rate=1 should collapse to one fetch, got {collapsed}"
        assert distinct == 2, f"rate=0 should not quantize, got {distinct}"
        r.ok("DATA-7: quantize_time collapses 23.999999/24.0 at rate 1, keeps both at rate 0")
    except Exception as e:
        r.fail("DATA-7 time quantization", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.reset_provider()


def test_v034_data7_no_source_key_never_caches(r):
    """CACHE-6's precedent: an empty source key fetches every time.

    A host that cannot name a source cannot promise anything about it, so nothing about it
    may be remembered."""
    p = _armed(res=16)
    try:
        for _ in range(3):
            _cook('@OUT = fetch_time("", 5.0, ix, iy);')
        assert p.fetches == 3, f"{p.fetches} fetches for 3 unkeyed reads"
        assert tex_provider.get_media_cache().stats()["frames"] == 0
        r.ok("DATA-7: an unkeyed source is fetched every time and never pooled")
    except Exception as e:
        r.fail("DATA-7 unkeyed source", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.reset_provider()


def test_v034_data7_a_version_bump_invalidates(r):
    """`bump_source_version` re-fetches, drops the old bytes, and moves the lineage flags.

    The flags half is the honest half: `lineage_key` knows nothing about sources, so a host
    that does not stamp them serves v3's pixels for a v4 cook. That is a host obligation
    with a helper, and this row pins the helper."""
    from TEX_Wrangle.tex_results import lineage_key
    p = _armed(res=16)
    try:
        _cook('@OUT = fetch_time("plate", 5.0, ix, iy);')
        _cook('@OUT = fetch_time("plate", 5.0, ix, iy);')
        assert p.fetches == 1, f"a repeat read should hit the pool, got {p.fetches}"
        before_flags = tex_provider.source_flags("plate")
        assert before_flags == ("src=plate@0",), before_flags

        tex_provider.bump_source_version("plate")
        assert tex_provider.get_media_cache().stats()["frames"] == 0, \
            "the bumped source's bytes are still pooled"
        _cook('@OUT = fetch_time("plate", 5.0, ix, iy);')
        assert p.fetches == 2, f"a bumped source should re-fetch, got {p.fetches}"

        after_flags = tex_provider.source_flags("plate")
        assert after_flags == ("src=plate@1",), after_flags
        k0 = lineage_key(program_fp="fp", device="cpu", precision="fp32", flags=before_flags)
        k1 = lineage_key(program_fp="fp", device="cpu", precision="fp32", flags=after_flags)
        assert k0 != k1, "a source bump did not move the lineage key"
        r.ok("DATA-7: a version bump re-fetches, drops the bytes, and moves the lineage key")
    except Exception as e:
        r.fail("DATA-7 invalidation", f"{type(e).__name__}: {e}")
    finally:
        tex_provider._versions.pop("plate", None)
        tex_provider.reset_provider()


# ── the derivations the footprint buys ───────────────────────────────────────

def test_v034_data7_refuses_tiling_and_graph_capture(r):
    """Both derivations follow from the registry tags with no new machinery.

    `footprint='image'` => non-local => M-4 refuses to strip it. `sync=True` => the
    CUDA-graph tier refuses to capture it, because a host callback inside a capture is not a
    sync but a FOREIGN CALL: baked once and replayed forever, so every later frame would
    serve the first frame's pixels with no error anywhere."""
    from TEX_Wrangle.tex_runtime import graphed, stdlib_registry
    from TEX_Wrangle.tex_compiler.lexer import Lexer
    from TEX_Wrangle.tex_compiler.parser import Parser

    try:
        for name in ("fetch_time", "sample_time"):
            assert name in stdlib_registry.non_local_names(), f"{name} is not non-local"
            assert name in graphed._SYNC_STDLIB, f"{name} is not sync-gated"

        prog = Parser(Lexer('@OUT = fetch_time("p", 1.0, ix, iy);').tokenize()).parse()
        assert not tex_memory.is_tile_safe(prog), "a fetch_time program was called tile-safe"
        assert graphed._capturable(prog)[0] is False, "a fetch_time program was capturable"

        # The negative control: the same shape WITHOUT the call is both.
        plain = Parser(Lexer('@OUT = vec4(u, v, 0.0, 1.0);').tokenize()).parse()
        assert tex_memory.is_tile_safe(plain) and graphed._capturable(plain)[0]
        r.ok("DATA-7: fetch_time/sample_time refuse tiling and graph capture, by derivation")
    except Exception as e:
        r.fail("DATA-7 derivations", f"{type(e).__name__}: {e}")


# ── the governor ─────────────────────────────────────────────────────────────

def test_v034_data7_unarmed_costs_nothing(r):
    """Invariant #7: the `media` pool is registered always and reports 0 until armed.

    Registered at `CacheRegistry` construction like the graph pool — and the deferred import
    means an engine nobody wired a provider into never builds the pool at all."""
    try:
        tex_provider.reset_provider()
        tex_provider._media_cache = None
        reg = tex_memory.get_cache_registry()
        assert "media" in reg._pools, "the media pool is not registered"
        assert reg.stats("cpu").get("media") == 0, reg.stats("cpu")
        assert tex_provider._media_cache is None, \
            "reading the governor's byte count BUILT the pool"
        r.ok("DATA-7: an unarmed media pool reports 0 bytes and is never constructed")
    except Exception as e:
        r.fail("DATA-7 unarmed cost", f"{type(e).__name__}: {e}")


def test_v034_data7_pool_arbitrates_under_the_governor(r):
    """The governor can reclaim media bytes, and does it before result-cache bytes.

    evict_order 40 vs 50: a media frame rebuilds with one host fetch, a result frame with a
    cook that may itself have to fetch. Strictly cheaper to rebuild is drained first."""
    _armed(res=64)
    try:
        for t in range(6):
            _cook(f'@OUT = fetch_time("plate", {t}.0, ix, iy);')
        held = tex_memory._media_pool_bytes("cpu")
        assert held > 0, "the pool holds nothing after 6 distinct reads"

        freed = tex_memory._evict_media("cpu", held // 2)
        assert freed >= held // 2, f"freed {freed} of a requested {held // 2}"
        assert tex_memory._media_pool_bytes("cpu") < held

        reg = tex_memory.get_cache_registry()
        order = {n: p[2] for n, p in reg._pools.items()}
        assert order["stdlib"] < order["media"] < order.get("results", 50) <= order["graphs"], \
            order
        r.ok("DATA-7: the media pool arbitrates, and drains before the result cache")
    except Exception as e:
        r.fail("DATA-7 governor arbitration", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.reset_provider()


def test_v034_data7_eviction_is_playhead_aware(r):
    """The furthest frame from the playhead goes first — not the least recently used.

    This is the one pool in the tree whose entries carry a time, so the `playhead` hint the
    governor has always passed can finally mean something. A scrub backwards through a
    window must not evict the frames it is about to re-read."""
    _armed(res=64, rate=1.0)
    try:
        for t in (0.0, 1.0, 2.0, 20.0):
            _cook(f'@OUT = fetch_time("plate", {t}, ix, iy);')
        cache = tex_provider.get_media_cache()
        one = next(iter(cache._entries.values())).nbytes
        # Ask for exactly one frame's worth with the playhead at 1.0: frame 20 is furthest.
        cache.evict_bytes(one, dev_type="cpu", playhead=1.0)
        left = sorted(e.t for e in cache._entries.values())
        assert left == [0.0, 1.0, 2.0], f"evicted the wrong frame; left={left}"
        r.ok("DATA-7: eviction drops the frame furthest from the playhead first")
    except Exception as e:
        r.fail("DATA-7 playhead eviction", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.reset_provider()


def test_v034_data7_reattach_reports_the_media_pool(r):
    """`reattach()` grows media_frames/media_bytes, and reports ZERO honestly.

    Providers are host-side and process-global, so nothing is restored — the contract is
    that a host re-registers. Printing the zero is the point: a host that assumed its pool
    survived otherwise has no way to tell."""
    from TEX_Wrangle import tex_recovery
    try:
        tex_provider.reset_provider()
        tex_provider._media_cache = None
        rep = tex_recovery.reattach()
        assert rep["media_frames"] == 0 and rep["media_bytes"] == 0, rep
        assert not rep["errors"], rep["errors"]

        _armed(res=32)
        _cook('@OUT = fetch_time("plate", 1.0, ix, iy);')
        rep2 = tex_recovery.reattach()
        assert rep2["media_frames"] == 1 and rep2["media_bytes"] > 0, rep2
        r.ok("DATA-7: reattach reports the media pool, and reports an empty one as 0")
    except Exception as e:
        r.fail("DATA-7 reattach report", f"{type(e).__name__}: {e}")
    finally:
        tex_provider.reset_provider()


def test_v034_data7_speculative_io_failure_never_alarms(r):
    """A host-I/O failure is class-dependent: COMMITTED raises, SPECULATIVE ledgers.

    The provider cannot know its job's class, so the decision lives in the queue, where it
    is known. A prefetch that alarms about a source the user never asked for is worse than
    one that quietly did not happen."""
    from TEX_Wrangle import tex_cookqueue as Q

    class Broken:
        provider_id = "broken"

        def fetch_time(self, k, t):
            raise OSError("nope")

        sample_time = fetch_time

    tex_provider.reset_provider()
    tex_provider.set_provider(Broken())
    q = Q.CookQueue(name="tex-data7-test")
    q.install_policy(Q.SpeculativePolicy(min_confidence=0.0, min_value_ms=0.0,
                                         unknown_min_confidence=0.0))
    try:
        def boom(cancel):
            return tex_provider.materialize("plate", 1.0, "fetch")

        spec = q.submit(boom, klass=Q.SPECULATIVE, reason=Q.PREFETCH, confidence=0.9)
        spec.wait(10)
        comm = q.submit(boom, klass=Q.COMMITTED)
        comm.wait(10)

        assert spec.state == Q.CANCELLED, f"speculative job ended {spec.state}"
        assert comm.state == Q.FAILED, f"committed job ended {comm.state}"
        assert getattr(comm.error, "_code", "") == tex_provider.E_FETCH_FAILED, comm.error
        led = q._policy.refusals.get(Q.PREFETCH)
        assert led and "host I/O failed" in led[1], q._policy.refusals
        assert q.stats.failed == 1, f"failed={q.stats.failed}, want 1 (the COMMITTED one)"
        r.ok("DATA-7: a speculative host-I/O failure is ledgered; a committed one raises")
    except Exception as e:
        r.fail("DATA-7 speculative I/O failure", f"{type(e).__name__}: {e}")
    finally:
        q.close()
        tex_provider.reset_provider()
