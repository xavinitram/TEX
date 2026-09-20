"""PERF-6 — the cook-fit planners ask the host for free VRAM once, and plan the same.

WHAT THE LANE CHANGED. `_tile_plan` bought its own `host.get_free_memory` on every pointwise
stage of every whole-frame CUDA cook — measured at 90-112 us a call, seven times per ten-stage
1024^2 frame, ~10 % of the tick, all of it spent concluding `est <= budget` and returning None.
The P1 `free_hint` hand-off that was supposed to prevent this is inert on that cook: LAT-2's
cheap path returns from `_preflight_memory` before it buys a reading, so the hint is None on
exactly the stages that reach the query. `tex_tiling` now records what the last LIVE reading
implied about the bytes on the device that torch's allocator does not own (`_free_foreign`),
and serves the planners from it whenever that already settles their question.

WHAT THIS FILE HAS TO PROVE, and why each test exists:

  1. THE PLANS DID NOT MOVE. A golden minted at the base sha records what `_tile_plan` and
     `_halo_tile_plan` answer for every shipped `examples/*.tex` plus the ten
     `examples/host_demo.py::_COMP_STAGES` programs, at 1024^2 and 2048^2, across three device
     totals and four fixed free-VRAM answers — 3048 cells, of which 558 are real strip counts
     and 10 are halo plans, so the corpus is not a wall of nulls. The check re-runs the sweep
     twice: COLD (the memo dropped before every call, so the planner is measured alone) and
     WARM (the memo dropped only when the fake's ANSWER changes, which is the one kind of
     change the decomposition cannot see). Both must equal the golden.
  2. THE COUNT ACTUALLY FELL. Ten cooks of the same unpressured program buy ONE reading, not
     ten. This is the red-first row: at the base sha it reads ten.
  3. A PRESSURED COOK STILL READS LIVE. When the estimate is anywhere near the budget the memo
     declines and the host is asked, every time — so no strip count is ever planned on an old
     number.
  4. THE MARGIN IS LOAD-BEARING. Weakening `_FREE_MEMO_MARGIN` to 1.0 changes a plan; at its
     shipped 8.0 it does not. A safety factor no test can move is decoration.
  5. THE BOUND TRACKS THE ALLOCATOR. Allocating N bytes on the device drops the served bound
     by N (CUDA only) — the identity `free == total - foreign - torch_allocated` at work.
  6. THE GOLDEN IS NOT VACUOUS. Doubling the peak estimate must move a large number of cells.
  7. A SWAPPED HOST IS ASKED AFRESH. Replacing the services object invalidates what the old
     one implied, without any caller having to remember to say so.

PORTABILITY. Everything here runs with no CUDA: the planners compare the device as a STRING,
the bindings are CPU tensors, and the host, the device total and the persisted TDR medians are
all faked. Test 5 is the one that needs a device and SKIPs without one.
"""
from helpers import *

import hashlib
import importlib.util
import json

from TEX_Wrangle import tex_api, tex_memory, tex_tiling
from TEX_Wrangle.tex_cache import TEXCache
from TEX_Wrangle.tex_runtime import autotier
from TEX_Wrangle.tex_runtime import host as host_mod

_HERE = Path(__file__).resolve().parent
_PKG = _HERE.parent
_GOLDEN = _HERE / "perf6_goldens" / "tile_plans.json"

#: The sweep the golden was minted over. Every one of these is part of the QUESTION: change
#: one and the golden is answering something else, so re-mint it at the base sha rather than
#: re-recording it at head. THE RECIPE IS `_sweep` + `_plan_cell` BELOW, run under `_Stubbed`
#: against a checkout of the sha being pinned: a row is `{key, sha256_source, binding_types,
#: tensor_names, compiles, plans}` with `plans` the COLD pass keyed `res/total/free`. Re-mint by
#: pointing this file's corpus walk at the old checkout — there is no separate generator to
#: fall out of step with what is checked here.
_RES = (1024, 2048)
_FREE = ((64 * 1024 ** 3, "64GiB"), (1024 ** 3, "1GiB"),
         (128 * 1024 ** 2, "128MiB"), (16 * 1024 ** 2, "16MiB"))
#: `_halo_tile_plan`'s cheap gate is `est < total // 8`, so a big device answers None before it
#: plans anything and a small one crosses into the real work. Both are needed, or the halo half
#: of the golden is nulls all the way down.
_TOTALS = ((12 * 1024 ** 3, "12GiB"), (256 * 1024 ** 2, "256MiB"), (64 * 1024 ** 2, "64MiB"))
_DEVICE = "cuda:0"          # a STRING the planners compare; no device is touched


class _FakeHost:
    """Just enough of the HostServices protocol for the planners, with a fixed answer and a
    call counter — the counter is what turns "ask once" into an assertion."""

    def __init__(self, free):
        self.free = float(free)
        self.calls = 0

    def get_free_memory(self, device):
        self.calls += 1
        return self.free

    def free_memory(self, amount, device):
        pass

    def is_oom(self, exc):
        return False

    def soft_empty_cache(self):
        pass

    def get_user_dir(self):
        return None

    def cancel_token(self):
        return None

    def raise_if_interrupted(self):
        pass


class _Stubbed:
    """Fake host + fixed device total + no persisted TDR medians + a fixed allocator reading,
    all restored on the way out.

    None of the four is a convenience. `device_total_mem` and `autotier.cook_ms` are where this
    box's own hardware and its persisted cook medians would otherwise leak into a golden that
    has to mean the same thing on a CI runner with no GPU. `_torch_allocated` is the third:
    without a device it has nothing to read and answers None, which makes the memo decline
    every time — so a CPU-only runner would pass every row below while exercising none of the
    mechanism at all. `alloc=None` opts back into the real reader, and
    `test_perf6_the_bound_tracks_the_allocator` is the row that does."""

    def __init__(self, free=_FREE[0][0], total=_TOTALS[0][0], alloc=0):
        self.host = _FakeHost(free)
        self.total = total
        self.alloc = alloc

    def __enter__(self):
        self._saved = (tex_memory.device_total_mem, autotier.cook_ms,
                       tex_tiling._torch_allocated)
        tex_memory.device_total_mem = lambda device: self.total
        autotier.cook_ms = lambda key: None
        if self.alloc is not None:
            tex_tiling._torch_allocated = lambda idx: self.alloc
        host_mod.set_host_services(self.host)
        tex_tiling.forget_free_memory()
        return self

    def __exit__(self, *exc):
        (tex_memory.device_total_mem, autotier.cook_ms,
         tex_tiling._torch_allocated) = self._saved
        host_mod.reset_host_services()
        tex_tiling.forget_free_memory()
        return False


def _load_golden():
    with open(_GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)["corpus"]


def _sources():
    """`{key: source}` for the corpus the golden names — the shipped examples plus the demo
    host's ten comp stages."""
    out = {}
    ex = _PKG / "examples"
    for f in sorted(ex.glob("*.tex")):
        with open(f, encoding="utf-8") as fh:
            out[f"examples/{f.name}"] = fh.read()
    spec = importlib.util.spec_from_file_location(
        "_perf6_host_demo", str(ex / "host_demo.py"))
    demo = importlib.util.module_from_spec(spec)
    sys.modules["_perf6_host_demo"] = demo
    spec.loader.exec_module(demo)
    for name, code, _d in demo._COMP_STAGES:
        out[f"_COMP_STAGES/{name}"] = code
    return out


def _plan_cell(prog, code, bt, fp, bindings):
    tile = tex_tiling._tile_plan(prog.ast, bindings, _DEVICE, 0, 4, fp,
                                 free_hint=None, code=code, binding_types=bt)
    halo = tex_tiling._halo_tile_plan(prog.ast, code, bindings, _DEVICE, 0, 4, fp,
                                      None, "fp32", bt)
    return {"tile": tile,
            "halo": None if halo is None else [halo[0], sorted(halo[1]), halo[2]]}


def _sweep(st, prog, code, bt, fp, tensor_names, *, warm):
    """Every cell of one program's plan table. `warm` lets PERF-6's memo answer (it is dropped
    only when the fake's free-VRAM ANSWER changes, which is a `foreign` change); the cold pass
    drops it before every single call, so it measures the planner with nothing memoized."""
    cells = {}
    for res in _RES:
        bindings = {n: torch.zeros(1, res, res, 4) for n in tensor_names}
        for total_v, total_label in _TOTALS:
            for free_v, free_label in _FREE:
                st.total = total_v
                st.host.free = float(free_v)
                tex_tiling.forget_free_memory()
                for _rep in range(2):     # the 2nd rep is the memo-served one when warm
                    if not warm:
                        tex_tiling.forget_free_memory()
                    cell = _plan_cell(prog, code, bt, fp, bindings)
                cells[f"{res}/{total_label}/{free_label}"] = cell
    return cells


def _corpus_rows():
    """`(row, prog, code, bt, fp, tensor_names)` for every golden row that compiles, after
    checking the source has not moved under the golden."""
    srcs = _sources()
    out, drift = [], []
    for row in _load_golden():
        key = row["key"]
        code = srcs.get(key)
        if code is None:
            drift.append(f"{key}: the golden names a program this tree does not have")
            continue
        if hashlib.sha256(code.encode()).hexdigest() != row["sha256_source"]:
            drift.append(f"{key}: the SOURCE moved, so the golden answers a different question")
            continue
        if not row.get("compiles"):
            out.append((row, None, code, None, None, None))
            continue
        bt = {n: TEXType(v) for n, v in row["binding_types"].items()}
        prog = tex_api.compile(code, bt)
        fp = TEXCache.fingerprint(code, bt)
        out.append((row, prog, code, bt, fp, row["tensor_names"]))
    return out, drift


# ── 1. the plans did not move ───────────────────────────────────────────────

def test_perf6_the_tile_plans_are_identical(r: SubTestResult):
    """The behaviour oracle: every cell of the golden, cold AND warm."""
    print("\n--- PERF-6: the cook-fit plans against the base-sha golden ---")
    with _Stubbed() as st:
        rows, drift = _corpus_rows()
        for d in drift:
            r.fail("PERF-6 corpus drift", d)
        cells = moved = 0
        strips = halos = 0
        for row, prog, code, bt, fp, names in rows:
            if prog is None:
                continue
            want = row["plans"]
            for label, warm in (("cold", False), ("warm", True)):
                got = _sweep(st, prog, code, bt, fp, names, warm=warm)
                for k, v in want.items():
                    cells += 1
                    if got.get(k) != v:
                        moved += 1
                        if moved <= 6:
                            r.fail("PERF-6 plan moved",
                                   f"{row['key']} [{label}] {k}: {v} -> {got.get(k)}")
            strips += sum(1 for v in want.values() if v["tile"] is not None)
            halos += sum(1 for v in want.values() if v["halo"] is not None)
        if moved:
            r.fail("PERF-6 plans", f"{moved} of {cells} plan cells moved")
        else:
            r.ok(f"{cells} plan cells identical to the base-sha golden "
                 f"({strips} strip counts, {halos} halo plans in it), cold and warm")


def test_perf6_the_golden_catches_a_planner_change(r: SubTestResult):
    """MUTATION. A golden that cannot fail protects nothing: double the peak estimate and a
    large share of the cells must move. `estimate_peak_bytes` is the input the whole
    `est <= 0.25 * free` question is built on, so nothing that reaches a plan is untouched."""
    print("\n--- PERF-6 mutation: a doubled peak estimate moves the golden ---")
    with _Stubbed() as st:
        rows, _ = _corpus_rows()
        real = tex_memory.estimate_peak_bytes
        tex_memory.estimate_peak_bytes = lambda *a, **k: 2 * real(*a, **k)
        try:
            moved = cells = 0
            for row, prog, code, bt, fp, names in rows:
                if prog is None:
                    continue
                got = _sweep(st, prog, code, bt, fp, names, warm=False)
                for k, v in row["plans"].items():
                    cells += 1
                    moved += got.get(k) != v
        finally:
            tex_memory.estimate_peak_bytes = real
    if moved > cells // 10:
        r.ok(f"a doubled peak estimate moves {moved} of {cells} cells — the golden has teeth")
    else:
        r.fail("PERF-6 mutation", f"only {moved} of {cells} cells moved when the peak estimate "
               f"was doubled — the golden is not measuring the planner")


# ── 2-4. the query itself ───────────────────────────────────────────────────

_SMALL = "@OUT = @A * 2.0;\n"


def _small_program():
    bt = {"A": TEXType.VEC4}
    return tex_api.compile(_SMALL, bt), TEXCache.fingerprint(_SMALL, bt), bt


def test_perf6_an_unpressured_frame_asks_the_host_once(r: SubTestResult):
    """RED-FIRST. Ten cooks of the same unpressured program used to buy ten free-VRAM
    readings, one per cook; the reading is the same number each time and none of the ten plans
    can depend on it. At the base sha this row reads 10."""
    print("\n--- PERF-6: one host free-VRAM reading per frame, not one per cook ---")
    prog, fp, bt = _small_program()
    bindings = {"A": torch.zeros(1, 1024, 1024, 4)}
    with _Stubbed() as st:
        plans = [tex_tiling._tile_plan(prog.ast, bindings, _DEVICE, 0, 4, fp,
                                       free_hint=None, code=_SMALL, binding_types=bt)
                 for _ in range(10)]
        calls = st.host.calls
    if calls == 1 and plans == [None] * 10:
        r.ok("ten unpressured cooks -> 1 host.get_free_memory call, 10 identical plans")
    else:
        r.fail("PERF-6 query count",
               f"ten unpressured cooks made {calls} host.get_free_memory call(s) "
               f"(expected 1) and planned {plans}")


def test_perf6_a_pressured_cook_always_reads_live(r: SubTestResult):
    """The other half, and the one that keeps the change honest: when the estimate is anywhere
    near the budget the memo must DECLINE, so every strip count TEX plans still comes from a
    reading bought inside that same call."""
    print("\n--- PERF-6: a cook near the budget buys its own reading, every time ---")
    prog, fp, bt = _small_program()
    bindings = {"A": torch.zeros(1, 1024, 1024, 4)}
    # 16 MiB free against a ~16 MiB frame: the quarter-of-free budget is nowhere near it.
    with _Stubbed(free=16 * 1024 ** 2) as st:
        plans = [tex_tiling._tile_plan(prog.ast, bindings, _DEVICE, 0, 4, fp,
                                       free_hint=None, code=_SMALL, binding_types=bt)
                 for _ in range(5)]
        calls = st.host.calls
    if calls == 5 and len(set(plans)) == 1 and plans[0] is not None:
        r.ok(f"five pressured cooks -> 5 live readings and the same {plans[0]}-strip plan")
    else:
        r.fail("PERF-6 pressured", f"five pressured cooks made {calls} host.get_free_memory "
               f"call(s) (expected 5) and planned {plans} (expected five equal strip counts)")


def test_perf6_the_memo_margin_is_load_bearing(r: SubTestResult):
    """MUTATION on the safety factor. `_FREE_MEMO_MARGIN` is what makes a memoized reading
    unable to change an answer: it is served only while the estimate is under HALF the budget,
    so `est <= budget` holds for the memoized number and the live one alike.

    The test drives the one situation the decomposition cannot see — the host's answer changes
    under it, without `forget_free_memory` — and picks an estimate between an eighth and a
    quarter of the seeded reading. At the shipped margin the memo declines and the live (now
    much smaller) reading tiles the cook; weakened to 1.0 the memo answers from the stale
    number and the cook is planned whole. One line of difference, one changed picture-path."""
    print("\n--- PERF-6 mutation: the memo margin ---")
    prog, fp, bt = _small_program()
    bindings = {"A": torch.zeros(1, 1024, 1024, 4)}
    seeded = 1024 ** 3                       # 1 GiB at the seeding call
    est = int(seeded / 6)                    # between seeded/8 and seeded/4

    def run(margin):
        real_margin = tex_tiling._FREE_MEMO_MARGIN
        real_est = tex_memory.estimate_peak_bytes
        tex_tiling._FREE_MEMO_MARGIN = margin
        tex_memory.estimate_peak_bytes = lambda *a, **k: est
        try:
            with _Stubbed(free=seeded) as st:
                tex_tiling._tile_plan(prog.ast, bindings, _DEVICE, 0, 4, fp,
                                      free_hint=None, code=_SMALL, binding_types=bt)
                st.host.free = seeded / 4.0     # a `foreign` change, deliberately unannounced
                return tex_tiling._tile_plan(prog.ast, bindings, _DEVICE, 0, 4, fp,
                                             free_hint=None, code=_SMALL, binding_types=bt)
        finally:
            tex_tiling._FREE_MEMO_MARGIN = real_margin
            tex_memory.estimate_peak_bytes = real_est

    shipped, weakened = run(tex_tiling._FREE_MEMO_MARGIN), run(1.0)
    if shipped is not None and weakened is None:
        r.ok(f"margin {tex_tiling._FREE_MEMO_MARGIN} -> {shipped} strips from a live reading; "
             f"margin 1.0 -> None from the stale one")
    else:
        r.fail("PERF-6 margin", f"the margin did not decide the plan: shipped={shipped!r}, "
               f"weakened={weakened!r} (expected a strip count and None)")


def test_perf6_the_bound_tracks_the_allocator(r: SubTestResult):
    """`free == total - foreign - torch_allocated`, checked against the allocator rather than
    asserted in a comment: with the reading memoized, allocating N bytes on the device must
    drop the served bound by N. This is the property that makes a cook unable to invalidate
    the memo — everything a cook takes or releases moves `torch_allocated`, which is re-read
    on every call, and nothing a cook does moves `foreign`."""
    print("\n--- PERF-6: the served bound follows torch's allocated bytes ---")
    if not torch.cuda.is_available():
        r.skip("PERF-6 allocator bound", "no CUDA device — `torch_allocated` is unreadable, "
               "so the memo declines and every planner call reads live (also correct)")
        return
    dev = torch.device("cuda", torch.cuda.current_device())
    with _Stubbed(free=8 * 1024 ** 3, total=12 * 1024 ** 3, alloc=None):
        tex_tiling._query_free_memory(dev)
        before = tex_tiling._free_memory_bound(dev, 1)
        n = 64 * 1024 * 1024
        hog = torch.empty(n // 4, dtype=torch.float32, device=dev)
        after = tex_tiling._free_memory_bound(dev, 1)
        del hog
    if before is None or after is None:
        r.fail("PERF-6 allocator bound", "the memo declined a 1-byte estimate — the "
               "decomposition did not resolve at all")
        return
    drop = before - after
    if abs(drop - n) <= n // 8:
        r.ok(f"a {n // (1024 * 1024)} MiB allocation dropped the bound by "
             f"{drop / (1024 * 1024):.1f} MiB")
    else:
        r.fail("PERF-6 allocator bound",
               f"a {n} byte allocation moved the bound by {drop} bytes; the identity "
               f"`free == total - foreign - torch_allocated` does not hold here")


def test_perf6_a_swapped_host_is_asked_afresh(r: SubTestResult):
    """A host REPLACED under a warm memo is the one `foreign` change TEX can actually be told
    about — the CLI resolving its services, an embedding host wiring its own in, a test
    installing a fake — and it is the shape this lane's own probe walked straight into. Seed
    the memo from a host that says 64 GiB, install one that says 16 MiB, and the planner must
    ask the NEW host rather than plan on the old one's number. `services_generation()` is the
    hook; no caller has to remember to invalidate anything."""
    print("\n--- PERF-6: swapping the host invalidates what the old one implied ---")
    prog, fp, bt = _small_program()
    bindings = {"A": torch.zeros(1, 1024, 1024, 4)}

    def plan():
        return tex_tiling._tile_plan(prog.ast, bindings, _DEVICE, 0, 4, fp,
                                     free_hint=None, code=_SMALL, binding_types=bt)

    with _Stubbed(free=64 * 1024 ** 3) as st:
        rich = st.host
        first = plan()
        plan()                               # a second cook, served from `rich`'s reading
        poor = _FakeHost(16 * 1024 ** 2)
        host_mod.set_host_services(poor)     # deliberately WITHOUT forget_free_memory()
        second = plan()
        calls = (rich.calls, poor.calls)
    if first is None and second is not None and calls == (1, 1):
        r.ok(f"the rich host was asked once for two cooks; the swapped-in poor host was asked "
             f"immediately and tiled the cook into {second} strips")
    else:
        r.fail("PERF-6 host swap", f"first={first!r} second={second!r} calls={calls} "
               f"(expected None, a strip count, and (1, 1))")
