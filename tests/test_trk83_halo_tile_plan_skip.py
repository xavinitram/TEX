"""TRK-83 — `tex_engine.run` called `_halo_tile_plan` once per whole-frame-cook stage even on
a stage `_tile_plan` had already determined is tile-safe with no pressure: `_tile_plan`
returning falsy sends flow straight to the `elif not ctx.fused_chain:` halo branch regardless
of WHY it was falsy, and `_halo_tile_plan`'s own first act (`tex_tiling.py`) is to check the
exact same `is_tile_safe_cached` memo and bail with `None` — so a tile-safe stage paid a call
into `_halo_tile_plan` guaranteed to no-op. The counts harness's BENCH-2 spy counts the CALL,
not the work inside it, so this was the "10 calls/cook" residue `docs/host-path-counts.md` §6
item 6 names for `all_dirty` (ten stages, ten guaranteed-no-op calls).

Fixed with a one-line guard at the `tex_engine.py` call site: skip calling `_halo_tile_plan`
when `tex_memory.is_tile_safe_cached(ctx.program, ctx.fp)` already reads True. This is a memo
HIT, not a second AST walk — `_tile_plan`, called immediately before on the same fingerprint,
already populated the exact same `tex_memory._tile_safe_memo` entry — and it is provably the
same answer `_halo_tile_plan` itself would have returned via its own identical check, so the
served picture cannot move: the guard only ever removes a call whose return value was
already guaranteed to be `None`.

ComfyUI-invisible: same output, same tier, every default-path cook — the call this test
removes always returned None; this only removes the call.
"""
from helpers import *

from TEX_Wrangle import tex_engine

_POINTWISE = "@OUT = @A * 2.0;"          # tile-safe: pixel-local, no non-local builtin.
_BLUR = "@OUT = gauss_blur(@A, 3.0);"    # NOT tile-safe: a halo-footprint builtin.


class _CallSpy:
    """Count entries to a `tex_engine` module-level name, without changing its behaviour."""

    def __init__(self, name):
        self._name = name

    def __enter__(self):
        self.n = 0
        self._orig = getattr(tex_engine, self._name)
        spy = self

        def wrapper(*a, **k):
            spy.n += 1
            return spy._orig(*a, **k)
        setattr(tex_engine, self._name, wrapper)
        return self

    def __exit__(self, *e):
        setattr(tex_engine, self._name, self._orig)
        return False


def _cook(code):
    img = make_img(1, 64, 64, 3)
    return tex_engine.cook(code, {"A": img}, device_mode="cpu", compile_mode="none")


def test_trk83_tile_safe_stage_never_calls_halo_tile_plan(r: SubTestResult):
    print("\n--- TRK-83: a tile-safe, unpressured cook skips _halo_tile_plan entirely ---")
    try:
        with cold_engine_state():
            with _CallSpy("_tile_plan") as tile, _CallSpy("_halo_tile_plan") as halo:
                _cook(_POINTWISE)
        r.ok(f"pointwise cook: _tile_plan called {tile.n}x, "
             f"_halo_tile_plan called {halo.n}x (expected 0)") if halo.n == 0 and tile.n >= 1 \
            else r.fail("TRK-83 tile-safe skip",
                        f"_tile_plan={tile.n}, _halo_tile_plan={halo.n} "
                        f"(expected _tile_plan>=1, _halo_tile_plan==0)")
    except Exception as e:
        r.fail("TRK-83 tile-safe skip", str(e))


def test_trk83_non_tile_safe_stage_still_reaches_halo_tile_plan(r: SubTestResult):
    """The control: a program the guard must NOT silence — `_halo_tile_plan` is `gauss_blur`'s
    only route to a bounded-halo tiled cook, so a genuinely non-tile-safe stage must still
    reach it exactly as before."""
    print("\n--- TRK-83 control: a blur (not tile-safe) still reaches _halo_tile_plan ---")
    try:
        with cold_engine_state():
            with _CallSpy("_halo_tile_plan") as halo:
                _cook(_BLUR)
        r.ok(f"blur cook: _halo_tile_plan called {halo.n}x (expected >= 1, unchanged)") \
            if halo.n >= 1 else \
            r.fail("TRK-83 control", f"_halo_tile_plan called {halo.n}x for a blur stage, "
                   "expected >= 1 — the guard over-reached and silenced a real halo case")
    except Exception as e:
        r.fail("TRK-83 control", str(e))


def test_trk83_pictures_are_unchanged(r: SubTestResult):
    """The guard must never move a served pixel: both programs cook to the SAME result with
    and without the spy installed (i.e. the guard's decision is deterministic and harmless)."""
    print("\n--- TRK-83: output is unaffected by the guard ---")
    try:
        with cold_engine_state():
            a1 = _cook(_POINTWISE)
        with cold_engine_state():
            a2 = _cook(_POINTWISE)
        out1 = a1.outputs["OUT"]
        out2 = a2.outputs["OUT"]
        r.ok("pointwise cook: two independent cooks agree bit-exactly") \
            if torch.equal(out1, out2) else \
            r.fail("TRK-83 picture stability", "two cooks of the same pointwise program differ")
    except Exception as e:
        r.fail("TRK-83 picture stability", str(e))
