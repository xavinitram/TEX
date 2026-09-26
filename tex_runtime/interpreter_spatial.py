"""Interpreter spatial-context setup — SPLIT-I (v0.44 Phase A1).

Split mechanically out of `interpreter.py` (STR-3/STR-4/STR-7's pattern: a mixin the
main `Interpreter` class still inherits, so every call site keeps calling `self._method(...)`
unchanged — no new call layer, no cross-module attribute lookup on any per-pixel path).
This module owns the FOUR methods that establish a cook's spatial context before the AST
walk starts: the CF-6 consensus extent lookup, the TRK-84 coordinate-ramp cache, the LAT-4
cached-builtins assembly, and the ENG-7 host-time builtins. None of it runs per pixel — it
runs once per cook (or once per LRU miss) — so a deferred import back into `interpreter.py`
for the few names this module does not own itself (`_consensus_extent`, `_collect_identifiers`,
the builtin-name sets) costs nothing measurable and avoids a load-time import cycle, exactly
the way `masked_flow.py` already defers its own back-references into `interpreter.py`.

No behaviour changed by this move: every body below is byte-identical to the code it replaced
in `interpreter.py` (AGENTS.md §"Trades to REFUSE" — mechanical moves only, never an
"improvement" mid-move).
"""
from __future__ import annotations

import math
import torch

from ..tex_compiler.ast_nodes import Program
# PHASEC-OBSROUTE follow-up: `_BUILTINS_LRU_MAX`/`_COORD_RAMP_LRU_MAX`/
# `_SCALAR_BUILTIN_DEFAULTS` used to sit in a top-level `from .interpreter import ...`
# here, contradicting this docstring's own claim (above) that the names this module does
# not own are deferred — and, like R1's tex_engine_tiers fix, importing THIS module first
# in a fresh process ImportErrored the same way. Deferred into the two methods that use
# them, below, alongside `_consensus_extent`/`_collect_identifiers`, which were already
# correctly deferred.


class _SpatialContextMixin:
    """Mixin supplying `Interpreter`'s spatial-context setup methods (SPLIT-I).

    Composed onto `Interpreter` alongside `MaskedFlowMixin` / `_ControlFlowMixin` /
    `_BindingExecMixin` — every attribute referenced below (`self.bindings`, `self.env`,
    `self._coord_ramp_lru`, ...) is set on the instance by `Interpreter.__init__`.
    """

    def _determine_spatial_shape(self, program, roi=None) -> tuple[int, int, int] | None:
        """Find the spatial dimensions from image inputs — the CF-6 consensus extent, derived
        by the shared `_consensus_extent` so this tier and codegen cannot disagree.

        Under an ROI cook (`roi` set) the grid is the cook-region (w, h) the executor sliced
        to, with only the batch taken from the bindings — a whole-passed gather input keeps
        the full W×H, so it must not size the grid.

        Corpus impact, scanned before landing (doc 40 §1 R2 asks for exactly this): of 129
        frozen programs, ZERO bind two spatial tensors whose extents disagree, so no golden
        moves and the fix needs no freeze-boundary argument. It lands before freeze #2 so the
        freeze snapshots the post-fix truth.
        """
        from .interpreter import _consensus_extent
        sp = _consensus_extent(self.bindings, program, roi=roi, source=self._source)
        if sp is None and roi is not None:
            # An ROI cook with NO spatial binding at all. `run_roi` refuses this case before it
            # can arrive ("no spatial binding to anchor the window"), so this is unreachable
            # through the engine; it is kept because it is v0.29's answer for a direct caller,
            # and a shape rule is a bad place to start returning None where something used to
            # come back.
            return (1, roi[3], roi[2])
        return sp

    def _coord_ramps(self, size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """TRK-84: the full-extent `[0, size)` fp32 ramp and its `/max(size-1,1)`
        normalization, cached per `(device, size)`. Coordinate dtype is always fp32
        (invariant #4), so only device needs keying — and it does: a
        `ThreadLocalInterpreterPool`-pooled instance's `self.device` is reassigned per
        `execute()` call, so a CPU cook followed by a CUDA cook on the SAME instance is
        ordinary, not an edge case. A window move (ROI/tile/pan) changes only the origin
        `x0`/`y0`, never `W_full`/`H_full` or the device, so this hits on every pan tick
        where `_builtins_lru`'s (…, roi, …) key cannot — the window slices this ramp as
        a VIEW instead of paying a fresh `torch.arange` + divide.

        BIT-EXACT for every origin, including a tiled cook's `y0`:
        `torch.arange(0, size, dtype=fp32)[i] == float32(i)` exactly (TEX never nears
        fp32's 2**24 exact-integer ceiling), so slicing at `[x0:x0+w]` reproduces
        `torch.arange(x0, x0+w, dtype=fp32)` bit-for-bit — two exactly-representable
        values whose exact sum is also exactly representable round to that sum, so there
        is no "different rounding order" for a view to pick up. `u`/`v` are the SAME
        `ramp / max(size-1,1)` division `_create_builtins` already applied to `ix`,
        computed once over the full extent instead of once per window."""
        from .interpreter import _COORD_RAMP_LRU_MAX
        key = (self._device_str, size)
        hit = self._coord_ramp_lru.get(key)
        if hit is not None:
            self._coord_ramp_lru.move_to_end(key)
            return hit
        ramp = torch.arange(0, max(size, 0), dtype=torch.float32, device=self.device)
        norm = ramp / max(size - 1, 1)
        self._coord_ramp_lru[key] = (ramp, norm)
        if len(self._coord_ramp_lru) > _COORD_RAMP_LRU_MAX:
            self._coord_ramp_lru.popitem(last=False)
        return ramp, norm

    def _create_builtins(self, program: Program,
                         used_builtins: frozenset[str] | None = None,
                         tile: tuple[int, int] | None = None,
                         roi: tuple[int, int, int, int, int, int] | None = None,
                         batch_slice: tuple[int, int] | None = None):
        """Create built-in variables lazily — only allocate what the program uses.

        Builtins use compact broadcast-friendly shapes instead of full
        [B, H, W] expansion. PyTorch broadcasts automatically in ops.

        ROI-3: `roi=(x0, y0, w, h, W, H)` cooks a sub-window — ix/iy start at x0/y0 and
        u/v/iw/ih/px/py reference the FULL W/H, so the window's coordinates match the
        untiled cook exactly (seam-exact in 2-D). The M-4 strip `tile=(y0, H_total)` is the
        1-D special case `roi=(0, y0, W, H, W, H_total)`; it is normalized to `roi` here so
        there is a SINGLE seam-exact coordinate path to reason about.
        """
        from .interpreter import (_collect_identifiers, _CACHEABLE_BUILTIN_NAMES,
                                  _SCALAR_BUILTIN_DEFAULTS, _BUILTINS_LRU_MAX)
        used = used_builtins if used_builtins is not None else _collect_identifiers(program)

        # Cache builtins: reuse tensors when spatial config hasn't changed (LAT-4: small LRU,
        # so proxy<->full-res alternation hits instead of rebuilding). Key on the RAW
        # (tile, roi, batch_slice) — on the default cook all three are None (like the old
        # `tile=None` slot), so the warm-hit path allocates nothing; the tile→roi normalization
        # runs only AFTER a miss (below), for the coordinate build (the sole `roi` consumer).
        cache_key = (self.spatial_shape, self._device_str, self._dtype, used,
                     self.latent_channel_count, tile, roi, batch_slice)
        hit = self._builtins_lru.get(cache_key)
        if hit is not None:
            self._builtins_lru.move_to_end(cache_key)
            self.env.update(hit)
            self._set_time_builtins(used)   # ENG-7: never cached — see _TIME_BUILTIN_NAMES
            return

        # MISS: normalize the strip form into the general ROI form (the one coordinate path).
        if self.spatial_shape and roi is None:
            _B, _H, _W = self.spatial_shape
            roi = (0, tile[0], _W, _H, _W, tile[1]) if tile is not None else (0, 0, _W, _H, _W, _H)

        # M-3: coordinate/spatial builtins are ALWAYS fp32 (never self._dtype).
        # fp16 `u` has only 4097 distinct values across 8192 pixels and
        # floor().long() mis-addresses 1024 of 4096 rows at H=4096 — coordinates
        # and sampler grids must stay fp32 regardless of the image-data precision.
        cdt = torch.float32
        if self.spatial_shape:
            B, H, W = self.spatial_shape
            # ROI-3: the grid is W×H (the cook-region); coordinates offset by (x0, y0)
            # reference the full W_full×H_full image.
            x0, y0, _w, _h, W_full, H_full = roi

            # ix: pixel x-coordinate (offset by the ROI's left column)
            # u/v use expand() which creates a view (no memory copy) at [B,H,W]
            # for compatibility with torch.stack in sampling functions.
            if "ix" in used or "u" in used:
                # TRK-84: slice the cached full-extent ramp (bit-identical view, see
                # `_coord_ramps`) instead of a fresh arange+divide. Fallback for an
                # out-of-bounds window (should never occur) is the old direct math.
                if 0 <= x0 and x0 + W <= W_full:
                    full_ix, full_u = self._coord_ramps(W_full)
                    ix = full_ix[x0:x0 + W].view(1, 1, W)
                    u_flat = full_u[x0:x0 + W]
                else:
                    ix = torch.arange(x0, x0 + W, dtype=cdt, device=self.device).view(1, 1, W)
                    u_flat = ix / max(W_full - 1, 1)
                if "ix" in used:
                    self.env["ix"] = ix
                if "u" in used:
                    self.env["u"] = u_flat.view(1, 1, W).expand(B, H, W)

            # iy: pixel y-coordinate (offset by the ROI's top row)
            if "iy" in used or "v" in used:
                if 0 <= y0 and y0 + H <= H_full:
                    full_iy, full_v = self._coord_ramps(H_full)
                    iy = full_iy[y0:y0 + H].view(1, H, 1)
                    v_flat = full_v[y0:y0 + H]
                else:
                    iy = torch.arange(y0, y0 + H, dtype=cdt, device=self.device).view(1, H, 1)
                    v_flat = iy / max(H_full - 1, 1)
                if "iy" in used:
                    self.env["iy"] = iy
                if "v" in used:
                    self.env["v"] = v_flat.view(1, H, 1).expand(B, H, W)

            # iw, ih: image dimensions (the FULL image under an ROI/strip)
            if "iw" in used:
                self.env["iw"] = torch.scalar_tensor(float(W_full), dtype=cdt, device=self.device)
            if "ih" in used:
                self.env["ih"] = torch.scalar_tensor(float(H_full), dtype=cdt, device=self.device)

            # px, py: pixel step in UV space (1/full-width, 1/full-height)
            if "px" in used:
                self.env["px"] = torch.scalar_tensor(1.0 / max(W_full, 1), dtype=cdt, device=self.device)
            if "py" in used:
                self.env["py"] = torch.scalar_tensor(1.0 / max(H_full, 1), dtype=cdt, device=self.device)

            # ROI-6: under a batch strip, fi references the FULL batch (fi starts at f0) and
            # fn reports B_total — the temporal twin of iy/ih under a tile, so a per-frame
            # program's frame builtins match the whole-batch cook exactly (seam-exact).
            f0, B_total = batch_slice if batch_slice is not None else (0, B)
            if "fi" in used:
                self.env["fi"] = torch.arange(f0, f0 + B, dtype=cdt, device=self.device).view(B, 1, 1)
            if "fn" in used:
                self.env["fn"] = torch.scalar_tensor(float(B_total), dtype=cdt, device=self.device)
        else:
            # No spatial context — pure scalar mode (only create what's used)
            for name, val in _SCALAR_BUILTIN_DEFAULTS.items():
                if name in used:
                    self.env[name] = torch.scalar_tensor(val, dtype=cdt, device=self.device)

        if "PI" in used:
            self.env["PI"] = torch.scalar_tensor(math.pi, dtype=self._dtype, device=self.device)
        if "TAU" in used:
            self.env["TAU"] = torch.scalar_tensor(math.tau, dtype=self._dtype, device=self.device)
        if "E" in used:
            self.env["E"] = torch.scalar_tensor(math.e, dtype=self._dtype, device=self.device)
        if "ic" in used:
            self.env["ic"] = torch.scalar_tensor(float(self.latent_channel_count), dtype=self._dtype, device=self.device)

        # Store only builtin tensors in cache (not user variables). cache_key is a
        # fresh key here (we returned above on a hit), so the insert already lands
        # MRU — no move_to_end needed (mirrors compiled._env_cached).
        # ENG-7: _CACHEABLE_BUILTIN_NAMES, not _BUILTIN_NAMES — the host-time builtins are
        # not a function of this key. ORDER IS LOAD-BEARING: the store must stay ABOVE
        # `_set_time_builtins`, so a playhead is never in `env` when the entry is taken.
        # (The filter is the belt to that braces — see _CACHEABLE_BUILTIN_NAMES.)
        self._builtins_lru[cache_key] = {k: v for k, v in self.env.items()
                                         if k in _CACHEABLE_BUILTIN_NAMES}
        while len(self._builtins_lru) > _BUILTINS_LRU_MAX:
            self._builtins_lru.popitem(last=False)
        self._set_time_builtins(used)

    def _set_time_builtins(self, used: frozenset[str]) -> None:
        """ENG-7: write the host-time builtins fresh for THIS cook.

        Deliberately outside the LRU (see _TIME_BUILTIN_NAMES): their value comes from the
        host's playhead, not from the cache key, so they must be rewritten on the hit path
        too. Cheap enough that not caching them costs nothing measurable — three scalar
        tensors, only for the names the program actually reads.

        Forced fp32: these are TIMELINE coordinates, not image data, and the builtin's own
        value should not be lossy (fp16 holds integers exactly only to 2048, so a raw fp16
        `frame` would read 2049 as 2048 and round every later frame to even — a 24fps
        two-hour timeline is ~172k frames).

        HONEST LIMIT — fp32 here does NOT make `@A.rgb * frame` exact under fp16. These are
        0-dim tensors, and torch's promotion treats a 0-dim operand as a scalar that does
        not lift the result dtype: `fp16_image * fp32_scalar -> fp16`, so the 2049 rounds
        back to 2048 at the multiply. `fi` genuinely does escape this, but not for the
        reason it looks: it is `[B,1,1]` — DIMENSIONED — and a dimensioned fp32 operand does
        promote. So the "same rule as fi" reading of invariant #4 is wrong here, and an
        earlier version of this comment claimed it.

        What actually protects users is the gate, not this line: `frame`/`time` are
        registered in precision_policy._BUILTIN_MAG at _FP16_MAX, so the C1 amplification
        gate always declines a program that mixes a playhead into image lineage, and
        `precision="auto"` never reaches fp16 for one. Expert `precision="fp16"` has no
        gate by definition ("no safety net" — AGENTS.md invariant #10), and there the
        multiply is fp16 like everything else. Making these dimensioned to force promotion
        was considered and rejected: `fi`'s `[B,1,1]` broadcasts correctly against a
        `[B,H,W]` mask but mis-aligns against a `[B,H,W,C]` image, so the shape that saves
        `fi` is not one these can borrow."""
        # A host with no timeline (ComfyUI's default) passes nothing; every name then
        # falls to the .get() default below. Spelling that as a {name: 0.0} constant
        # would be a second copy of _TIME_BUILTIN_NAMES, wrong-but-harmless the day a
        # fourth one is added — the worst kind of stale.
        from .interpreter import _TIME_BUILTIN_NAMES
        tc = self.time_context or {}
        for name in _TIME_BUILTIN_NAMES:
            if name in used:
                self.env[name] = torch.scalar_tensor(
                    float(tc.get(name, 0.0)), dtype=torch.float32, device=self.device)

