"""
ENG-14 / ENG-6 / ENG-12 — `tex_buffers`: what a cooked frame is, who may write it,
and how it leaves.

Two standing contracts, moved here whole out of `tex_engine` (ENG-14) with no body
changed: ENG-6's zero-copy handoff (`to_dlpack` / `from_dlpack`, and the `copy=True`
ownership default that makes it safe) and ENG-12's buffer-ownership and immutability
rules (`freeze` / `frozen_copy` / `is_frozen` / `frame_version` / `verify_unmutated`,
plus the `_disown_inputs` input-alias net that upholds them at egress). The two banner
blocks below carry the standing arguments in full — "FROZEN IS A SIGNAL, NOT A FENCE"
and its torch-2.12 measurement, and the copy-on-read cache contract they imply.

This module is a LEAF: it imports `torch` and nothing else from the package, so
`tex_engine` imports it at load and re-exports every name. Every caller still reads
these names off `tex_engine` (`tex_engine.freeze`, `tex_engine.to_dlpack`, …) — the
move is invisible at every call site, and the moved bodies compile to byte-identical
bytecode, so the per-cook cost is zero by construction rather than by measurement.

Pinned by `tests/test_v023_phase1.py` (the ENG-6 canary) and
`tests/test_v025_phase1.py` (the ENG-12 block).
"""
from __future__ import annotations

import torch


# ── ENG-6: zero-copy AI handoff (DLPack) ─────────────────────────────────────
# A cook output ALREADY is what a vision model wants: a device-resident, fp32,
# channels-last [B,H,W,C] image (or [B,H,W] mask). These helpers hand one to another
# framework over the DLPack protocol with no host round-trip.
#
# CONTRACT (canary-pinned, test_v023_phase1): an engine output is (1) a torch.Tensor,
# (2) fp32, (3) on the cook's device, (4) BHWC. `to_dlpack` can transpose to BCHW (the
# NCHW most models expect) as a zero-copy view.
#
# OWNERSHIP — copy=True is the DEFAULT and the safe posture. Codegen may reuse an output
# buffer (M-5 `out=`), and in the engine era an output may be a CACHED frame (CACHE-2);
# a consumer writing in place through a zero-copy view would corrupt engine state. So by
# default we hand out an OWNED contiguous copy. Pass copy=False only for a buffer you own
# and will not let the model mutate.
#
# AUTOGRAD — every cook runs under torch.inference_mode(), so a raw output carries the
# inference flag. The DLPack round-trip drops it (from_dlpack hands back an ORDINARY tensor
# over the shared memory either way), so a consumer CAN attach the result to an autograd
# graph — but for copy=False that graph would be backed by an engine buffer the next cook
# overwrites. Use copy=True (default) whenever the consumer will train through or mutate the
# tensor; copy=False is for read-only, same-tick consumption. Differentiable cooking (grad
# flowing back INTO the cook) is out of scope until the engine era.

def _owned_copy(t):
    """An owned, contiguous copy of `t` that is NOT inference-flagged (so an ML consumer
    can attach it to an autograd graph). `empty_like`+`copy_` runs outside the cook's
    inference_mode, unlike `.clone()`, which would inherit the flag."""
    src = t.contiguous()
    out = torch.empty_like(src)
    out.copy_(src)
    return out


def to_dlpack(tensor, *, layout="bhwc", copy=True):
    """Export a cooked output tensor as a DLPack capsule (ENG-6). `layout='bchw'` returns
    an NCHW-shaped view (a zero-copy permute); `copy=True` (default) first re-materializes
    an owned, contiguous, grad-ready tensor — see the ownership/autograd notes above."""
    import torch.utils.dlpack as _dl
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"to_dlpack expects a torch.Tensor, got {type(tensor).__name__}")
    t = tensor
    if layout == "bchw":
        if t.dim() != 4:
            raise ValueError(f"layout='bchw' needs a 4-D [B,H,W,C] tensor, got {tuple(t.shape)}")
        t = t.permute(0, 3, 1, 2)
    elif layout != "bhwc":
        raise ValueError(f"layout must be 'bhwc' or 'bchw', got {layout!r}")
    if copy:
        t = _owned_copy(t)
    return _dl.to_dlpack(t)


def from_dlpack(capsule):
    """Import a DLPack capsule (from torch or another framework) as a torch tensor that
    shares its memory (ENG-6). The inverse of `to_dlpack`."""
    import torch.utils.dlpack as _dl
    return _dl.from_dlpack(capsule)


# ── ENG-12: buffer ownership & immutability contract ─────────────────────────
# WHO MAY WRITE A COOKED TENSOR AFTER IT IS PRODUCED. Undefined write discipline is the
# bug class that killed Natron's engine, so it lands BEFORE any frame is cached (CACHE-2)
# and every later cross-owner edge (GRAPH-2 threads, XPU-2 frame handles) inherits it.
#
# FROZEN IS A SIGNAL, NOT A FENCE. A cook output is usually born an *inference tensor* (the
# interpreter and compiled tiers run under torch.inference_mode()), and torch RAISES on an in-place
# op on one — BUT that raise is not a rollback: on torch 2.12 (verified, CPU + CUDA) the in-place op
# LANDS THE WRITE and *then* raises. So a frozen frame handed straight back to a consumer that does
# `frame.clamp_(...)` is silently corrupted (the write took) even though the op "failed", and a
# version stamp can't catch it (frame_version of an inference tensor is a constant 0). Freezing is
# therefore a loud tripwire for a cache-INTERNAL mistake, never a guarantee against a CONSUMER write.
# (This is also why the born-frozen floor is not universal anyway: the cuda_graph replay hands back
# a normal .clone(), the debug paint a normal tensor.)
#
# THE CACHE CONTRACT (CACHE-2) — the real guarantee is COPY-ON-READ, mirroring to_dlpack(copy=True):
#   * put() stores the frame frozen (the canonical master — freezing keeps the cache's own code from
#     scribbling it, and torch's raise surfaces such a bug loudly).
#   * get()/_restore return an OWNED COPY by default (frame.clone() — a normal, mutable, independent
#     buffer), so a consumer's in-place write can never reach the stored master. copy=False is the
#     opt-in fast path for a read-only consumer that promises not to mutate (same posture as
#     to_dlpack(copy=False) / a `.data` alias — the caller opting out, not a hole).
#   * verify_unmutated (stratum 2, torch's t._version) is a live defense only for a NORMAL
#     (host-supplied, non-frozen) entry — a bumped counter drops it; it is inert for a frozen master
#     (which is why copy-on-read, not the version stamp, is what protects that master).
# _disown_inputs (input-alias net) and to_dlpack's copy=True default uphold the same ownership at
# egress.
#
# M-5 out= reuse (codegen, a DO-NOT-TOUCH) can NEVER target a cached frame: its reuse set is
# codegen _tN arithmetic temps only — never a binding — and a cached frame can only re-enter
# a cook AS an input binding. The one residual write into a binding, a scatter, is already
# COW-guarded (interpreter _scatter_owned / codegen _scat_owned: clone-before-first-write).
# See docs/results-caching.md and the AGENTS.md DO-NOT-TOUCH register.

def is_frozen(t) -> bool:
    """True if `t` is an inference tensor — one torch RAISES on for an in-place op. Note that on
    torch 2.12 the raise does not roll back the write (the op lands, then raises), so "frozen" is a
    loud tripwire, not a write fence; CACHE-2's actual consumer-facing guarantee is copy-on-read."""
    return isinstance(t, torch.Tensor) and t.is_inference()


def frame_version(t) -> int:
    """ENG-12 version stamp: torch's in-place mutation counter `t._version` for a normal tensor,
    else a constant 0 for a frozen (inference) tensor — reading _version on one would itself raise.
    Mirrors stdlib._safe_version. Because it is constant for a frozen tensor, verify_unmutated is a
    live mutation-detector only for NORMAL entries (a frozen master is protected by copy-on-read)."""
    return 0 if (not isinstance(t, torch.Tensor) or t.is_inference()) else t._version


def verify_unmutated(t, stamp) -> bool:
    """ENG-12 re-entry check: True iff `t`'s in-place mutation counter is unchanged since `stamp`.
    A live detector for a NORMAL entry (a bumped counter -> drop it); always True for a frozen
    master (stamp is a constant 0), which the cache protects by copy-on-read instead."""
    return frame_version(t) == stamp


def frozen_copy(t):
    """An immutable (inference-flagged) copy of `t`: any in-place write to the RESULT raises.
    Made inside inference_mode so the clone carries the inference flag even when `t` is a
    normal tensor — the exact inverse of _owned_copy (which strips the flag for autograd).
    Use to store a tamper-PROOF frame when the source is a normal (mutable) buffer."""
    with torch.inference_mode():
        return t.clone()


def freeze(t):
    """Idempotent hard-freeze: return `t` unchanged if it is already frozen (immutable),
    else a frozen_copy. The one call a frame cache uses to guarantee an entry cannot be
    written through, whatever the provenance of the tensor handed to it."""
    return t if is_frozen(t) else frozen_copy(t)


def _disown_inputs(raw_output: dict, bindings: dict) -> dict:
    """Make sure no cooked output shares storage with an INPUT BINDING.

    `@OUT = @A;` binds the output name straight to the input tensor, so the "result" IS
    the caller's buffer — and const-folding widens that past literal identity (`@OUT =
    @A * 1.0;` folds to the same thing). The ComfyUI node never noticed because its egress
    clamp materializes a fresh tensor on the way out; ENG-3's `engine` profile removed the
    clamp and, with it, that accidental copy. A host recycling frame buffers would then
    have its input silently rewritten by its own output.

    This is the right layer for it: only the engine knows what the bindings were. The
    egress profile downstream is a format conversion and cannot tell an aliased input from
    a freshly computed tensor, so a clone there would have to be unconditional — measured
    at 39.7 ms on a 398 MB buffer, paid on every cook including the overwhelming majority
    that never alias.

    Keyed on the STORAGE POINTER, and that word is load-bearing twice over:

      * NOT object identity. A reshape (`unsqueeze`) hands back a new object over the same
        storage — different object, same pixels, same corruption.
      * NOT `.data_ptr()`, which is the address of the tensor's FIRST ELEMENT, not of its
        buffer. A view at a non-zero offset has a different one. `@X = @A.rgb;` starts at
        offset 0 and looks caught, which is exactly what makes that spelling dangerous —
        `@X = @A.a;` starts at offset 3, compares unequal, and sails through aliased.
        `untyped_storage().data_ptr()` is the buffer, and is the same for every view of it.

    Whether it runs at all is `plan.disown`, decided by the CALLER — see `prepare()`. It
    is not read off the process-global egress profile: a caller that pins its own profile
    per-call (`tex_cli`) or applies none at all (the documented `cook()` contract — raw
    tensors) is invisible to that global, so consulting it answered a question nobody
    asked. Ownership is a property of the call.
    """
    try:
        src = {v.untyped_storage().data_ptr()
               for v in bindings.values() if isinstance(v, torch.Tensor)}
        if not src:
            return raw_output
        return {name: (out.clone() if isinstance(out, torch.Tensor)
                       and out.untyped_storage().data_ptr() in src else out)
                for name, out in raw_output.items()}
    except Exception:
        return raw_output   # ownership is a safety net; it must never fail a cook
