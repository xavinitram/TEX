"""tex_runtime/streams.py — XPU-2: engine-owned async D2H egress (v0.33).

This module lifts a rule the codebase has carried since v0.20, stated at
`tex_marshalling.to_fp32_if_int_image`:

    D2H is never non_blocking (a CPU-side read could observe an in-flight buffer).

That rule is correct for a *bare tensor*, and it is the reason every device-to-host copy in
this engine blocks: the spill path, the demote path, the file writers. There is no way to hand
a plain `torch.Tensor` to a consumer and also tell it "not yet".

XPU-2 changes the object, not the rule. A `FrameHandle` is a frame **plus the CUDA event that
says when it is real**, and every route to its bytes goes through a fence. The rule becomes:

    D2H may be non_blocking when its result is carried by a handle, because a handle has
    nowhere to be read from that does not wait first.

WHY THIS IS SAFE HERE AND WAS NOT SAFE BEFORE. The register shelved "return-time transfer"
twice, on two objections — ComfyUI API fragility, and custody loss — and recorded that both are
properties of *that host*, not of the idea:

  * Under ComfyUI, a cook's output is handed to a node graph the engine does not own. A handle
    would have to survive an arbitrary third-party consumer that has never heard of one, and
    the first `.cpu()` or `.shape` on the wrong side is a silent wrong frame.
  * Under ENGINE custody every consumer is in this repository. There are exactly two today —
    `ResultCache`'s demote drain and its spill path — and both are in `tex_results.py`, three
    lines apart, fencing explicitly.

So this module is **engine-only, by construction and not by convention**: nothing in
`tex_node.py` can reach it, and `ResultCache` is already dormant under ComfyUI (the host caches
results itself). The exit gate for the release is a stress test that consumes a frame from the
wrong side ON PURPOSE and shows the fence is what makes the difference — see
`tests/test_v033_xpu2.py`.

THE PINNED PRECONDITION, which is not optional and is why `egress` can decline. CUDA can only
DMA asynchronously to page-locked host memory; torch silently downgrades `non_blocking=True` to
a synchronous copy from pageable memory. A handle over a pageable buffer would therefore carry
an event that is always already signalled — correct, but pure overhead, and worse, it would
read as asynchrony in a benchmark while delivering none. `egress` allocates pinned staging
through the shipped `_pin_worthwhile` band (1 MB – 256 MB; pinned pages are unswappable and
torch's caching host allocator keeps freed blocks for the process lifetime, so an uncapped
video-scale egress would permanently page-lock RAM the OS can never reclaim) and returns an
already-complete handle otherwise. A caller therefore never has to ask which it got.
"""
from __future__ import annotations

import torch


class FrameHandle:
    """A frame whose device-to-host copy may still be in flight.

    THE CONTRACT, in one line: `shape`/`dtype`/`nbytes` are safe at any time; **the bytes are
    not, until you fence**. `tensor()` fences for you and is what every consumer should call.

    Metadata is safe because it was decided when the copy was *issued*, not when it lands — a
    consumer that only wants to know how big a frame is (the byte accounting a cache does on
    every eviction) never has to wait for it. That asymmetry is the whole reason this is a
    handle and not just a `(tensor, event)` tuple: the useful part of async egress is exactly
    the work a consumer can do while the copy is in flight.

    A handle holds a reference to the SOURCE tensor as well as the destination. That is not
    bookkeeping: with the copy still in flight, dropping the last reference to the source
    returns its block to torch's caching allocator, which may hand it to another stream that
    has no ordering relationship with ours. Holding it until the fence is what makes the
    asynchrony free rather than a use-after-free with good manners.
    """

    __slots__ = ("_host", "_src", "_event")

    def __init__(self, host: torch.Tensor, src=None, event=None):
        self._host = host
        self._src = src
        # `_event is None` IS "already fenced" — there is no separate `_waited` flag, because a
        # flag would be a second place the same fact lives and a third place `is_ready`/`wait`/
        # `__repr__` have to agree. Clearing it in `wait()` also releases the Event object at
        # fence time rather than at handle death.
        self._event = event

    # ── metadata: always safe, never fences ──
    @property
    def shape(self):
        return self._host.shape

    @property
    def dtype(self):
        return self._host.dtype

    @property
    def device(self):
        return self._host.device

    def nbytes(self) -> int:
        return self._host.numel() * self._host.element_size()

    def is_ready(self) -> bool:
        """True if the copy has landed. Never blocks. A `False` here is what tells a scheduler
        to go and do something else; it is NOT permission to read the buffer when it turns
        True on a later poll without also having fenced, because polling is not ordering."""
        if self._event is None:
            return True
        try:
            if not self._event.query():
                return False
        except Exception:
            pass                        # a wedged/absent event reads as ready
        # Observing completion RELEASES, exactly as `wait` does. Without this a poll-based
        # consumer that never fences keeps both the source's VRAM block and the page-locked
        # host buffer alive indefinitely — measured: 64 MB still allocated at 2048² after the
        # source name was deleted. Safe because the copy is done: that is what we just asked.
        self._event = None
        self._src = None
        return True

    # ── the fence ──
    def wait(self) -> "FrameHandle":
        """Block until the copy has landed. Idempotent and cheap once satisfied."""
        if self._event is not None:
            try:
                self._event.synchronize()
            except Exception:
                torch.cuda.synchronize()        # last resort: fence the whole device
            self._event = None                  # fenced; also releases the Event
            self._src = None                    # the source is free the moment the copy lands
        return self

    def tensor(self) -> torch.Tensor:
        """The frame, fenced. **This is the only supported way to reach the bytes.**"""
        return self.wait()._host

    def unsafe_buffer(self) -> torch.Tensor:
        """The staging buffer WITHOUT fencing.

        It exists for exactly two callers: the stress test that reads from the wrong side on
        purpose to prove the fence is load-bearing, and a consumer that has already fenced
        another way (a full `torch.cuda.synchronize()`). The name is the documentation."""
        return self._host

    def __repr__(self) -> str:
        return (f"FrameHandle({tuple(self._host.shape)}, {self._host.dtype}, "
                f"{'ready' if self.is_ready() else 'in flight'})")


def _blocking(src: torch.Tensor, want) -> FrameHandle:
    """The plain synchronous copy, wrapped in an already-complete handle. Spelled once: it is
    the answer to four different questions in `egress` below, and three copies of it is three
    places to forget a `memory_format` the day one is needed."""
    host = torch.empty(src.shape, dtype=want, device="cpu")
    host.copy_(src)
    return FrameHandle(host)


def egress(src: torch.Tensor, *, dtype=None, retained: bool = False) -> FrameHandle:
    """Start a device-to-host copy of `src` and return a handle immediately.

    Returns an ALREADY-COMPLETE handle — same interface, no event — when asynchrony is not
    available or not wanted: a CPU source, no CUDA, a dtype conversion (torch keeps those
    synchronous), a size outside the pinned band, or `retained=True`. The caller writes one
    code path and gets whichever the situation allows, which is the point: an API whose fast
    path has a different shape from its slow path grows a consumer that only fences on one.

    `retained=True` states a fact the CALLER knows — **I am keeping this buffer** — rather than
    a decision about the implementation. Everything else follows from it: asynchrony needs a
    page-locked destination, pinned pages are unswappable and torch's caching host allocator
    holds freed blocks for the process lifetime, so a retained pinned frame is a slow leak of
    memory the OS can never reclaim. The alternative — copy into pinned, then clone to pageable
    to release the lock — costs a SECOND full host memcpy, which is more than the asynchrony
    saves on a copy with little to overlap. So a retained destination takes the blocking copy.
    (The parameter was once `staging=`, named for what egress allocates; a caller cannot answer
    that without already knowing why pinning enables asynchrony, which is not a fact a caller
    should need.)
    """
    want = dtype or src.dtype
    if (not isinstance(src, torch.Tensor) or src.device.type != "cuda" or retained
            or want != src.dtype or not torch.cuda.is_available()):
        return _blocking(src, want)
    # The shipped pinned-staging band, read from its owner rather than restated here — the cap
    # exists because pinned pages are unswappable and torch's caching host allocator keeps
    # freed blocks for the process lifetime. `_pin_worthwhile` itself asks about a CPU tensor;
    # here the source is on CUDA and it is the DESTINATION that gets pinned, so the band is
    # applied to the byte count directly.
    from ..tex_marshalling import _PIN_MAX_BYTES, _PIN_MIN_BYTES
    nbytes = src.numel() * src.element_size()
    if not (_PIN_MIN_BYTES <= nbytes <= _PIN_MAX_BYTES):
        return _blocking(src, want)
    try:
        host = torch.empty(src.shape, dtype=want, pin_memory=True)
        stream = torch.cuda.current_stream(src.device)
        host.copy_(src, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(stream)
        return FrameHandle(host, src, ev)
    except Exception:
        # Host memory pressure, a wedged driver: pinning is an optimisation, never a
        # requirement. Fall through to the blocking copy.
        pass
    # CF-5a: the fallback runs OUTSIDE the handler. Calling `_blocking` from inside the
    # `except` meant that when it failed too — and the case where both fail is a real one,
    # mid-CUDA-graph-capture, where `copy_` is capture-illegal by either route — the second
    # exception was raised *during handling of* the first. Python chains them, so the
    # traceback led with the pinning failure and buried the capture error underneath, and the
    # raise escaped `egress` either way. Out here the capture error surfaces as itself.
    return _blocking(src, want)
