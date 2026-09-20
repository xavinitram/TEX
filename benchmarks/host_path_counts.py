#!/usr/bin/env python3
"""
Host-path COUNTS (BENCH-2 — "count, don't time")
================================================
A STRUCTURAL benchmark for the interactive paths an embedding host drives. It reports how
many times each seam on the cook path is entered per tick — API calls, TEX Python frames,
CUDA kernel launches, memcpys and allocator allocations — and never how long any of it took.

Why counts and not times, on this class of box
----------------------------------------------
`docs/roadmap.md` §10 item 3 records the null controls: an `eight_config_bench` run of a
tree against ITSELF returns per-config geomeans from 0.949 to 1.105 with individual rows
spanning 0.70-2.32, and `cpu_off_warm` once tripped the 0.95 stop-ship threshold against
byte-identical code. Wall-clock on a laptop cannot gate an interactive-path regression; it
can only open an investigation at a release sitting on a quiet box.

Counts can. A structural trace of a real host driving a ten-stage comp through TEX's
ROI/results-cache pattern found the per-tick counts to be EXACT integers across steady ticks
(min == max over 29 ticks) while wall-clock on the same ticks varied 16-26 %. So this harness
measures the integers, `tests/test_bench2_counts.py` pins them, and CI gates on them.

What it drives
--------------
TEX's own `examples/host_demo.py::RoiComp` — the ten-stage `_COMP_STAGES` comp with a
persistent per-stage canvas, a CACHE-2 `ResultCache` armed by the host, and CACHE-1 lineage
keys that carry the upstream chain. That is the pattern an embedding host ports, so a count
that moves here is a count that moves in the host.

Seven scenarios
---------------
    prewarm          `tex_api.prewarm` over the comp's ten programs, each tick in its OWN
                     cold cache dir (the project-load path; the only cold scenario)
    source_edit      the first WHOLE-FRAME cook after a source edit of one stage
    terminal         terminal-knob scrub: viewport window, `dirty_from` = last stage
    midgraph         the same scrub on stage 5 (so the dirty suffix is five stages)
    pan              the window MOVES 16 px per tick, params constant
    all_dirty        a SOURCE-side knob each tick, whole frame, so the cache misses
    lint             `tex_api.check` with a one-character edit per tick (no cook at all)

Three measurement passes per scenario, each from a freshly built comp, so a counter never
perturbs another counter's reading:

    A. API spies      monkeypatched dotted targets (generalised from
                      `tests/test_v031_anim_contract.py::_Spies`), plus
                      `torch.cuda.memory_stats()` deltas
    B. frames         `sys.setprofile`, filtered to files under the package directory,
                      aggregated per `module:function`
    C. cuda           `torch.profiler` chrome trace, kernels and memcpys attributed to a
                      tick by their launch's CPU timestamp (CUDA only)

Each pass runs a WARM-UP tick, reported separately, then `--ticks` steady ticks reported as
min / median / max / total with a `stable` flag (min == max). Only stable rows are gateable.

Usage
-----
    python benchmarks/host_path_counts.py                       # both devices if present
    python benchmarks/host_path_counts.py --device cpu --res 96 --window 48 --ticks 5
    python benchmarks/host_path_counts.py --device cuda --res 1024 --window 512 --prof1 on
    python benchmarks/host_path_counts.py --save results/counts_head.json
    python benchmarks/host_path_counts.py --compare results/counts_head.json   # rc 1 on drift
    python benchmarks/host_path_counts.py --selftest            # the spies are not inert

Portability: runs with no CUDA (pass C is skipped and the cuda rows are absent), no ComfyUI
and no compiler. `--res 96 --window 48 --ticks 4 --device cpu --prof1 off` is the shape
`tests/test_bench2_counts.py` runs in-process.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                                # .../TEX_Wrangle
if os.path.dirname(_PKG) not in sys.path:
    sys.path.insert(0, os.path.dirname(_PKG))                # .../<package parent>

import torch                                                  # noqa: E402

from TEX_Wrangle import tex_api                               # noqa: E402
from TEX_Wrangle.tex_compiler.types import TEXType            # noqa: E402
from TEX_Wrangle.tex_testkit import cold_engine_state, armed_profiler   # noqa: E402  HOOK-4

_PKG_PREFIX = os.path.normcase(_PKG + os.sep)
#: `examples/` ships the demo HOST, not the engine. A sync called from there is the host's own
#: frame-completion barrier (`RoiComp.cook`'s trailing `torch.cuda.synchronize()`), which is a
#: host policy decision and not a stall TEX imposed — counting it as engine-side would make the
#: "zero engine syncs per interactive tick" row unpinnable and, worse, wrong.
_EXAMPLES_PREFIX = os.path.normcase(os.path.join(_PKG, "examples") + os.sep)
#: `benchmarks/` lives under the package too, so the frame filter would otherwise charge every
#: tick for this harness's own sampling closures — measurement counting itself.
_BENCH_PREFIX = os.path.normcase(_HERE + os.sep)


# ──────────────────────────────────────────────────────────────────────────────
# The host under measurement: TEX's own examples/host_demo.py::RoiComp
# ──────────────────────────────────────────────────────────────────────────────

def load_host_demo():
    """Import `examples/host_demo.py` as a module.

    `examples/` is not a package (it ships `.tex` exemplars, not Python packaging), so this
    loads the file by path under a private name rather than inventing an `__init__.py` the
    archive would then have to carry. The module's own `sys.path` insert is a no-op here —
    the package is already imported — so which tree gets measured is decided by where
    `TEX_Wrangle` resolved, exactly as `docs/bench1-v020-v028.md` warns."""
    mod = sys.modules.get("_tex_bench2_host_demo")
    if mod is not None:
        return mod
    path = os.path.join(_PKG, "examples", "host_demo.py")
    spec = importlib.util.spec_from_file_location("_tex_bench2_host_demo", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tex_bench2_host_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────────────────────────────────────────────────────────
# (a) API call spies
# ──────────────────────────────────────────────────────────────────────────────

#: row name -> the dotted target(s) that row counts. A row with several targets counts every
#: invocation through any of them: a name imported into a second module is a SECOND binding
#: site, and patching only the definition site silently reports zero (`tex_memory.run_roi` is
#: reached through a function-local `from .tex_memory import run_roi`, which reads the module
#: attribute at call time and so IS caught — but `tests` and hosts have been bitten by the
#: module-level form, so the list shape is the default).
SPY_TARGETS: "dict[str, tuple[str, ...]]" = {
    "tex_engine.cook":            ("TEX_Wrangle.tex_engine.cook",),
    "tex_engine.prepare":         ("TEX_Wrangle.tex_engine.prepare",),
    "tex_engine.run":             ("TEX_Wrangle.tex_engine.run",),
    "TEXCache.compile_ast":       ("TEX_Wrangle.tex_cache.TEXCache.compile_ast",),
    "TEXCache.compile_tex":       ("TEX_Wrangle.tex_cache.TEXCache.compile_tex",),
    "TEXCache.fingerprint":       ("TEX_Wrangle.tex_cache.TEXCache.fingerprint",),
    "Lexer.tokenize":             ("TEX_Wrangle.tex_compiler.lexer.Lexer.tokenize",),
    "Parser.parse":               ("TEX_Wrangle.tex_compiler.parser.Parser.parse",),
    "TypeChecker.check":          ("TEX_Wrangle.tex_compiler.type_checker.TypeChecker.check",),
    "TypeChecker.check_collect":  ("TEX_Wrangle.tex_compiler.type_checker.TypeChecker.check_collect",),
    "tex_roi._fold_program":      ("TEX_Wrangle.tex_roi._fold_program",),
    "tex_roi.roi_plan":           ("TEX_Wrangle.tex_roi.roi_plan",),
    "tex_roi.stage_halo":         ("TEX_Wrangle.tex_roi.stage_halo",),
    "tex_roi.chain_windows":      ("TEX_Wrangle.tex_roi.chain_windows",),
    "tex_results.lineage_key":    ("TEX_Wrangle.tex_results.lineage_key",),
    "ResultCache.get":            ("TEX_Wrangle.tex_results.ResultCache.get",),
    "ResultCache.put":            ("TEX_Wrangle.tex_results.ResultCache.put",),
    "tex_memory.run_roi":         ("TEX_Wrangle.tex_memory.run_roi",),
    "Interpreter._exec_stmt":     ("TEX_Wrangle.tex_runtime.interpreter.Interpreter._exec_stmt",),
    "profile.record":             ("TEX_Wrangle.tex_runtime.profile.record",),

    # ── the per-cook FIXED pipeline, and the memo keys that make a scrub re-parse ──────
    # Not part of the gate: these are the rows a follow-up would move, each named in
    # `docs/host-path-counts.md` §"Avoidable per tick" with the fix it would show. Several
    # have TWO binding sites because `tex_engine` imports the name at module level — patching
    # only the definition site reports a confident, wrong zero.
    "param_only_names":           ("TEX_Wrangle.tex_marshalling.param_only_names",),
    "_tile_plan":                 ("TEX_Wrangle.tex_tiling._tile_plan",
                                   "TEX_Wrangle.tex_engine._tile_plan"),
    "_halo_tile_plan":            ("TEX_Wrangle.tex_tiling._halo_tile_plan",
                                   "TEX_Wrangle.tex_engine._halo_tile_plan"),
    "_preflight_memory":          ("TEX_Wrangle.tex_tiling._preflight_memory",
                                   "TEX_Wrangle.tex_engine._preflight_memory"),
    "enforce_cache_budget":       ("TEX_Wrangle.tex_memory.enforce_cache_budget",),
    "trim_reserved_pool":         ("TEX_Wrangle.tex_memory.trim_reserved_pool",),
    "_disown_inputs":             ("TEX_Wrangle.tex_buffers._disown_inputs",
                                   "TEX_Wrangle.tex_engine._disown_inputs"),
    "torch.cuda.mem_get_info":    ("torch.cuda.mem_get_info",),
}

#: Counted with the CALLER's file, so a host-side sync and an engine-side sync are separate
#: rows. With PROF-1 disarmed the ENGINE row must be 0 on an interactive tick: a sync the host
#: did not ask for is a pipeline stall charged to somebody else's frame. The `[host-demo]` row
#: is the demo's own per-frame barrier and the `[out-of-pkg]` row is this harness's.
SYNC_TARGET = "torch.cuda.synchronize"
SYNC_ROWS = ("torch.cuda.synchronize[engine]", "torch.cuda.synchronize[host-demo]",
             "torch.cuda.synchronize[out-of-pkg]")


def _resolve(dotted: str):
    """`"a.b.C.d"` -> `(C, "d")`: the longest importable module prefix, then attribute walk."""
    parts = dotted.split(".")
    for i in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for p in parts[i:-1]:
            obj = getattr(obj, p)
        return obj, parts[-1]
    raise ImportError(f"no importable module prefix in {dotted!r}")


def _rewrap(owner, name, fn):
    """Re-apply the descriptor kind the original had.

    `TEXCache.fingerprint` is a `@staticmethod`; assigning a plain function over it turns
    every `cache.fingerprint(code, types)` call into a bound method and the cook dies on an
    arity error. `getattr` hands back the underlying function for both kinds, so the kind has
    to be read from the class `__dict__` (`getattr_static`) rather than from the value."""
    if inspect.isclass(owner):
        try:
            raw = inspect.getattr_static(owner, name)
        except AttributeError:
            raw = None
        if isinstance(raw, staticmethod):
            return staticmethod(fn)
        if isinstance(raw, classmethod):
            return classmethod(fn)
    return fn


class CallSpies:
    """Count entries into a set of dotted targets, installed and restored as a context manager.

    Generalised from `tests/test_v031_anim_contract.py::_Spies`, which counts three fixed
    mechanisms; the shape that mattered there and matters here is that the counter is a PATCH
    on the mechanism, not a sample of some downstream state. A cache that evicts and re-fills
    nets to zero growth while having recompiled every frame, and an allocator statistic nets
    to zero while having thrashed — only the patch sees the call.

        with CallSpies(SPY_TARGETS) as spies:
            ...
            spies.snapshot()      # a dict of row -> count so far
    """

    def __init__(self, targets=SPY_TARGETS, *, sync_caller_attribution: bool = True):
        self.targets = dict(targets)
        self.sync_caller_attribution = sync_caller_attribution
        self.counts: Counter = Counter()
        self._saved: list = []

    def rows(self) -> list:
        rows = list(self.targets)
        if self.sync_caller_attribution:
            rows += list(SYNC_ROWS)
        return rows

    def __enter__(self):
        counts = self.counts
        for row, dotted_list in self.targets.items():
            for dotted in dotted_list:
                owner, name = _resolve(dotted)
                orig = getattr(owner, name)
                self._saved.append((owner, name, inspect.getattr_static(owner, name)
                                    if inspect.isclass(owner) else orig))

                def make(row=row, orig=orig):
                    def spy(*a, **k):
                        counts[row] += 1
                        return orig(*a, **k)
                    spy.__name__ = getattr(orig, "__name__", "spy")
                    return spy

                setattr(owner, name, _rewrap(owner, name, make()))
        if self.sync_caller_attribution and hasattr(torch, "cuda"):
            owner, name = _resolve(SYNC_TARGET)
            orig = getattr(owner, name)
            self._saved.append((owner, name, orig))

            def sync_spy(*a, _orig=orig, **k):
                # sys._getframe over inspect.stack(): the latter materialises the whole stack
                # with source context, which on a hot path is the difference between a counter
                # and a second benchmark.
                try:
                    f = os.path.normcase(os.path.abspath(sys._getframe(1).f_code.co_filename))
                except Exception:
                    f = ""
                if f.startswith(_EXAMPLES_PREFIX):
                    counts[SYNC_ROWS[1]] += 1
                elif f.startswith(_PKG_PREFIX):
                    counts[SYNC_ROWS[0]] += 1
                else:
                    counts[SYNC_ROWS[2]] += 1
                return _orig(*a, **k)

            setattr(owner, name, sync_spy)
        for row in self.rows():
            counts.setdefault(row, 0)
        return self

    def __exit__(self, *exc):
        for owner, name, orig in reversed(self._saved):
            setattr(owner, name, orig)
        self._saved.clear()
        return False

    def snapshot(self) -> dict:
        return {row: int(self.counts.get(row, 0)) for row in self.rows()}


# ──────────────────────────────────────────────────────────────────────────────
# (b) TEX Python frames
# ──────────────────────────────────────────────────────────────────────────────

class FrameCounter:
    """Count Python calls into files under the package directory, per `module:function`.

    `sys.setprofile` rather than `sys.settrace`: the profile hook fires once per call instead
    of once per line, which is the difference between a benchmark and a hang. Frames outside
    the package (torch, the stdlib, this file, the spies' wrappers) are dropped on the first
    comparison, so what is reported is TEX's own work and nothing else."""

    def __init__(self):
        self.counts: Counter = Counter()
        self._on = False

    def _hook(self, frame, event, arg):
        if event != "call":
            return
        code = frame.f_code
        fn = os.path.normcase(code.co_filename)
        if not fn.startswith(_PKG_PREFIX) or fn.startswith(_BENCH_PREFIX):
            return
        self.counts[_frame_key(code.co_filename, code)] += 1

    def __enter__(self):
        self._on = True
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)
        self._on = False
        return False

    def snapshot(self) -> dict:
        return dict(self.counts)


_FRAME_KEY_MEMO: dict = {}


def _frame_key(fn: str, code) -> str:
    key = (fn, code.co_firstlineno, code.co_name)
    hit = _FRAME_KEY_MEMO.get(key)
    if hit is None:
        rel = os.path.relpath(fn, _PKG).replace(os.sep, "/")
        mod = rel[:-3] if rel.endswith(".py") else rel
        # co_qualname is 3.11+; CI runs 3.10 too, so fall back to the bare name.
        qual = getattr(code, "co_qualname", None) or code.co_name
        hit = _FRAME_KEY_MEMO[key] = f"{mod.replace('/', '.')}:{qual}"
    return hit


# ──────────────────────────────────────────────────────────────────────────────
# (c) CUDA: kernel launches, memcpys, allocator statistics
# ──────────────────────────────────────────────────────────────────────────────

_MEM_ROWS = ("alloc.allocated", "alloc.num_device_alloc", "alloc.num_alloc_retries")


def _cache_entries(comp) -> int:
    """How many entries the host's CACHE-2 results cache holds (0 when there is no comp)."""
    try:
        return int(comp.cache.stats()["ram_entries"])
    except Exception:
        return 0


def _mem_probe(device: str) -> tuple:
    if device != "cuda" or not torch.cuda.is_available():
        return (0, 0, 0)
    st = torch.cuda.memory_stats()
    return (int(st.get("allocation.all.allocated", 0)),
            int(st.get("num_device_alloc", 0)),
            int(st.get("num_alloc_retries", 0)))


_CUDA_ROWS = ("cuda.kernels", "cuda.memcpy_HtoD", "cuda.memcpy_DtoH", "cuda.memcpy_DtoD",
              "cuda.memcpy_DtoH_bytes", "cuda.memset")


def _parse_chrome_trace(path: str, tick_prefix: str) -> dict:
    """Per-tick CUDA event counts from a chrome trace.

    Kernels and memcpys land on the GPU timeline, whose clock does NOT share an origin with
    the `record_function` annotation that names the tick — so attributing by the device
    timestamp assigns work to whichever tick happens to overlap, which is not a fact about
    the program. Each device event carries the `correlation` id of the `cuda_runtime` launch
    that issued it, and THAT event is on the CPU timeline inside the annotation. So: build
    correlation -> launch-ts once, then bucket every device event by its launch."""
    with open(path, "r", encoding="utf-8") as fh:
        trace = json.load(fh)
    events = trace.get("traceEvents", trace if isinstance(trace, list) else [])
    spans: list = []          # (ts, ts+dur, tick_index)
    launch_ts: dict = {}      # correlation -> cpu ts
    devices: list = []        # (cat, name, correlation, bytes)
    for e in events:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat") or ""
        args = e.get("args") or {}
        name = e.get("name") or ""
        if cat in ("user_annotation", "cpu_op") and name.startswith(tick_prefix):
            try:
                spans.append((float(e["ts"]), float(e["ts"]) + float(e.get("dur", 0)),
                              int(name[len(tick_prefix):])))
            except (KeyError, ValueError):
                pass
        elif cat in ("cuda_runtime", "runtime", "cuda_driver"):
            corr = args.get("correlation")
            if corr is not None:
                launch_ts[int(corr)] = float(e.get("ts", 0.0))
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            devices.append((cat, name, args.get("correlation"),
                            int(args.get("bytes", 0) or 0)))
    spans.sort()
    per_tick: dict = {}

    def _bucket(ts):
        for lo, hi, idx in spans:
            if lo <= ts <= hi:
                return idx
        return None

    for cat, name, corr, nbytes in devices:
        if corr is None:
            continue
        ts = launch_ts.get(int(corr))
        if ts is None:
            continue
        idx = _bucket(ts)
        if idx is None:
            continue
        row = per_tick.setdefault(idx, Counter())
        if cat == "kernel":
            row["cuda.kernels"] += 1
        elif cat == "gpu_memset":
            row["cuda.memset"] += 1
        else:
            up = name.replace(" ", "")
            if "HtoD" in up:
                row["cuda.memcpy_HtoD"] += 1
            elif "DtoH" in up:
                row["cuda.memcpy_DtoH"] += 1
                row["cuda.memcpy_DtoH_bytes"] += nbytes
            elif "DtoD" in up:
                row["cuda.memcpy_DtoD"] += 1
    return {idx: {r: int(c.get(r, 0)) for r in _CUDA_ROWS} for idx, c in per_tick.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Scenarios
# ──────────────────────────────────────────────────────────────────────────────

_SCENARIO_SALT = [0]          # see Scenario._seq


class Scenario:
    """One interactive path: build a comp, warm it, then drive `ticks` steady ticks.

    `before`/`after` run OUTSIDE the counted region, which is what lets the cold scenarios
    exist at all: `cold_engine_state` lives in the package, so entering it inside the region
    would charge every prewarm tick for the fixture's own frames."""

    name = "?"
    needs_comp = True

    def __init__(self, res: int, window: int, device: str, ticks: int = 8,
                 pan_step: int = 16):
        self.res, self.window, self.device = res, window, device
        self.ticks, self.pan_step = ticks, pan_step
        span = max(1, res - window)
        self.roi = (span // 2, span // 2, window, window, res, res)
        self.demo = load_host_demo()
        self._saved_stages = None
        self.epoch = 0
        _SCENARIO_SALT[0] += 1
        self._salt = _SCENARIO_SALT[0]

    # -- lifecycle -------------------------------------------------------------
    def build(self):
        """A fresh comp per pass, primed, so pass B never inherits pass A's warm caches."""
        demo = self.demo
        prev_device = demo.DEVICE
        demo.DEVICE = self.device          # `_make_source` builds on the module default
        try:
            comp = demo.RoiComp(self.res, self.device)
        finally:
            demo.DEVICE = prev_device
        self.prime(comp)
        return comp

    def prime(self, comp):
        comp.cook(None, 0)

    def before(self, comp, i):
        pass

    def tick(self, comp, i):
        raise NotImplementedError

    def after(self, comp, i):
        pass

    def teardown(self):
        if self._saved_stages is not None:
            self.demo._COMP_STAGES[:] = self._saved_stages
            self._saved_stages = None

    # -- helpers ---------------------------------------------------------------
    def _seq(self, i: int) -> int:
        """A tick ordinal unique across ticks, across the three MEASUREMENT PASSES, and across
        every Scenario built in this process — and it is load-bearing, not hygiene.

        Each pass rebuilds the comp, so per-comp state (canvases, the results cache) is
        genuinely fresh. The memos that matter are NOT per-comp: `tex_roi`'s fold memo keys on
        the program text plus the param VALUES and is module-global, and so is the program
        cache. Two readings caught this, both by disagreeing with a reading that should have
        matched them:

          * replaying the same slider values in pass B served every fold from the memo pass A
            had just filled, and the frame pass reported 448 TEX frames for a terminal tick
            whose API pass had already counted a full re-lex + re-parse;
          * running `--device both` in one process, the CUDA leg reported `Lexer.tokenize = 0`
            per terminal tick where the CPU leg — same code, same host, same tick — reported
            1, because the CPU leg had already folded those exact param values.

        Offsetting the values by a per-instance salt makes every pass, and every device, see
        the same first-sight work. The salt is deliberately NOT used for the pan walk, whose
        positions must stay inside the canvas (`_pan_seq`)."""
        return (self._salt * 3 + self.epoch) * (self.ticks + 2) + i

    def _pan_seq(self, i: int) -> int:
        """The pan walk's ordinal: injective across this instance's three passes only.

        Unlike `_seq` it must stay BOUNDED — a window position has to fit in the canvas — and
        it does not need a global salt: the state a repeated window would warm (the LAT-4
        coordinate-builtin LRU) is keyed per device, and each device runs this scenario once."""
        return self.epoch * (self.ticks + 2) + i

    def _edit_source(self, comp, stage: int, i: int):
        """Replace stage `stage`'s SOURCE with a fresh variant, as a host's text editor does.

        The stage table is module state the comp reads through, so the edit is made there and
        restored in `teardown`. `_halo_memo` keys on (stage, params) and knows nothing about
        the source, so a host that edits code and keeps the memo answers the reach question
        from the OLD program — clearing it is part of what a source edit means."""
        demo = self.demo
        if self._saved_stages is None:
            self._saved_stages = list(demo._COMP_STAGES)
        name, code, defaults = self._saved_stages[stage]
        demo._COMP_STAGES[stage] = (name, code + f"\n// edit {i}\n", defaults)
        comp._halo_memo.clear()

    def _pan_roi(self, i: int):
        """The window for tick `i`, walking in x and NEVER revisiting a position.

        A repeat is not a pan — it is a CACHE HIT, because the window is in the CACHE-1
        lineage key. Wrapping the walk modulo the span (the obvious first shape) made the
        4th tick at 96^2 serve tick 0's patch from the results cache, and the scenario went
        from `cook = 1 every tick` to an unstable row that could not be pinned. So the step
        shrinks when the span cannot hold `ticks + 2` distinct positions at `pan_step`;
        the reported step is in the saved JSON."""
        span = max(1, self.res - self.window)
        x = min(span, (self._pan_seq(i) + 1) * self.pan_stride())
        return (x, x, self.window, self.window, self.res, self.res)

    def pan_stride(self) -> int:
        """A step small enough that all THREE passes' ticks get distinct windows.

        The window position feeds two caches that outlive the comp — the CACHE-1 lineage key
        (per-comp, rebuilt each pass) and the LAT-4 coordinate-builtin LRU (module-global,
        NOT rebuilt). Reusing pass A's positions in pass C served the coordinate tensors from
        that LRU and the pan tick reported 22 CUDA kernels instead of 26, i.e. the harness
        measured its own warm-up. The walk therefore has to be injective across passes, which
        is what `_seq` gives it and what this stride keeps inside the span."""
        span = max(1, self.res - self.window)
        slots = 3 * (self.ticks + 2)          # three passes x (warm-up + ticks), with slack
        step = self.pan_step
        if slots * step > span:
            step = max(1, span // slots)
        return step


class PrewarmScenario(Scenario):
    """CACHE-3 project load. Each tick is a COLD one: its own scratch cache dir, its own warm
    state, so every tick pays the real first-sight cost instead of the second tick replaying."""
    name = "prewarm"

    def prime(self, comp):
        pass                                   # nothing to prime: the point is the cold path

    def build(self):
        return None

    def before(self, comp, i):
        self._cold = cold_engine_state()
        self._cold.__enter__()

    def tick(self, comp, i):
        bt = {"IN": TEXType.VEC4}
        programs = [(code, bt) for _n, code, _d in self.demo._COMP_STAGES]
        tex_api.prewarm(programs, device=self.device, precision="fp32", compile_mode="auto")

    def after(self, comp, i):
        self._cold.__exit__(None, None, None)


class SourceEditScenario(Scenario):
    """The first WHOLE-FRAME cook after a source edit of one stage — the shape of a host's
    'the user typed in the code editor and hit apply'. Stages above the edit are served from
    the results cache; the edited stage and everything below it must actually cook."""
    name = "source_edit"
    _STAGE = 3

    def before(self, comp, i):
        self._edit_source(comp, self._STAGE, i)

    def tick(self, comp, i):
        comp.cook(None, 0)


class TerminalKnobScenario(Scenario):
    """The commonest interactive tick there is: drag the LAST node's slider with a viewport
    window open, so nine canvases stand and one stage cooks its window."""
    name = "terminal"

    def tick(self, comp, i):
        comp.params["vignette"]["strength"] = 0.30 + self._seq(i) * 0.001
        comp.cook(self.roi, len(self.demo._COMP_STAGES) - 1)


class MidGraphKnobScenario(Scenario):
    """The same drag five nodes up: the dirty suffix is five stages, so every per-cook fixed
    cost is paid five times and the difference from `terminal` is what one extra stage costs."""
    name = "midgraph"
    _STAGE = 5

    def tick(self, comp, i):
        name = self.demo._COMP_STAGES[self._STAGE][0]
        pname = next(iter(comp.params[name]))
        comp.params[name][pname] = 0.40 + self._seq(i) * 0.001
        comp.cook(self.roi, self._STAGE)


class PanScenario(Scenario):
    """The window MOVES and no parameter changes — a viewport pan. The memo keys that a
    scrub misses every tick are all hits here, so what is left is what moving the window
    itself costs (coordinate builtins, a fresh plan, a fresh lineage key)."""
    name = "pan"

    def tick(self, comp, i):
        comp.cook(self._pan_roi(i), len(self.demo._COMP_STAGES) - 1)


class AllDirtyScenario(Scenario):
    """A knob on the FIRST stage with no window: every stage's lineage key changes, so the
    results cache misses all ten and the whole frame recooks. The per-cook fixed pipeline is
    paid ten times here, which is what makes it visible."""
    name = "all_dirty"

    def tick(self, comp, i):
        name, _code, defaults = self.demo._COMP_STAGES[0]
        pname = next(iter(defaults))
        comp.params[name][pname] = 1.00 + self._seq(i) * 0.001
        comp.cook(None, 0)


class LintScenario(Scenario):
    """`tex_api.check` on every keystroke — the editor's live-lint path. No cook, no cache,
    no device: whatever this costs, it costs on the UI thread between keystrokes."""
    name = "lint"

    def build(self):
        return None

    def prime(self, comp):
        pass

    def tick(self, comp, i):
        code = self.demo._COMP_STAGES[-1][1]
        # ONE character, so the program is a different program and nothing can be served
        # from a memo, while the parse remains exactly as hard as the real one.
        edited = code.replace("0.15", f"0.1{self._seq(i) % 10}", 1)
        tex_api.check(edited, {"IN": TEXType.VEC4})


SCENARIOS = (PrewarmScenario, SourceEditScenario, TerminalKnobScenario,
             MidGraphKnobScenario, PanScenario, AllDirtyScenario, LintScenario)
SCENARIO_NAMES = tuple(s.name for s in SCENARIOS)


# ──────────────────────────────────────────────────────────────────────────────
# Passes
# ──────────────────────────────────────────────────────────────────────────────

def _stats(per_tick: list) -> dict:
    """min / median / max / total / stable over the steady ticks of one row."""
    if not per_tick:
        return {"min": 0, "median": 0, "max": 0, "total": 0, "stable": True}
    lo, hi = min(per_tick), max(per_tick)
    return {"min": lo, "median": int(statistics.median(per_tick)), "max": hi,
            "total": sum(per_tick), "stable": lo == hi}


def _drive(scn: Scenario, ticks: int, sample):
    """warm-up tick, then `ticks` steady ticks. `sample(i)` returns that tick's row dict."""
    comp = scn.build()
    scn.before(comp, -1)
    warm = sample(comp, -1)
    scn.after(comp, -1)
    steady = []
    for i in range(ticks):
        scn.before(comp, i)
        steady.append(sample(comp, i))
        scn.after(comp, i)
    return warm, steady


def _fold(warm: dict, steady: list) -> dict:
    rows = set(warm)
    for s in steady:
        rows |= set(s)
    return {r: dict(_stats([int(s.get(r, 0)) for s in steady]), warmup=int(warm.get(r, 0)))
            for r in sorted(rows)}


def pass_api(scn: Scenario, ticks: int) -> dict:
    """Pass A — spies + allocator statistics."""
    with CallSpies() as spies:
        def sample(comp, i):
            before = spies.snapshot()
            m0, c0 = _mem_probe(scn.device), _cache_entries(comp)
            scn.tick(comp, i)
            m1, c1 = _mem_probe(scn.device), _cache_entries(comp)
            after = spies.snapshot()
            out = {k: after[k] - before[k] for k in after}
            for row, a, b in zip(_MEM_ROWS, m0, m1):
                out[row] = b - a
            # Retention is HOST policy, so it is reported, never pinned: an interactive tick
            # that adds an entry and evicts nothing grows the results cache without bound, and
            # the row that would show a retention policy working is this one going to 0.
            out["results_cache.entries_added"] = c1 - c0
            return out
        warm, steady = _drive(scn, ticks, sample)
    return _fold(warm, steady)


def pass_frames(scn: Scenario, ticks: int, top: int) -> dict:
    """Pass B — TEX Python frames per `module:function`, top-N by total."""
    fc = FrameCounter()

    def sample(comp, i):
        before = dict(fc.counts)
        with fc:
            scn.tick(comp, i)
        after = fc.counts
        out = {k: after[k] - before.get(k, 0) for k in after}
        out = {k: v for k, v in out.items() if v}
        # Per-MODULE subtotals beside the per-function rows: a top-N list moves whenever two
        # functions swap places, which reads as drift and is not. A module subtotal only moves
        # when work moved between modules, which is the fact a design note wants.
        mods: Counter = Counter()
        for k, v in out.items():
            mods["frames.mod." + k.split(":", 1)[0]] += v
        out["frames.total"] = sum(out.values())
        out.update(mods)
        return out

    warm, steady = _drive(scn, ticks, sample)
    folded = _fold(warm, steady)
    mods = sorted((k for k in folded if k.startswith("frames.mod.")),
                  key=lambda k: -folded[k]["total"])
    keep = sorted((k for k in folded if k != "frames.total" and not k.startswith("frames.mod.")),
                  key=lambda k: -folded[k]["total"])[:top]
    return {k: folded[k] for k in ["frames.total"] + mods + keep}


def pass_cuda(scn: Scenario, ticks: int) -> dict:
    """Pass C — kernel launches and memcpys, per tick, from a chrome trace. CUDA only."""
    if scn.device != "cuda" or not torch.cuda.is_available():
        return {}
    from torch.profiler import profile, ProfilerActivity, record_function
    prefix = "bench2_tick_"
    comp = scn.build()
    torch.cuda.synchronize()
    out_dir = tempfile.mkdtemp(prefix="tex_bench2_")
    path = os.path.join(out_dir, "trace.json")
    scn.before(comp, -1)
    with record_function(prefix + "0"):        # a throwaway warm-up outside the profiler
        scn.tick(comp, -1)
    scn.after(comp, -1)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(ticks):
            scn.before(comp, i)
            with record_function(f"{prefix}{i}"):
                scn.tick(comp, i)
            torch.cuda.synchronize()
            scn.after(comp, i)
    prof.export_chrome_trace(path)
    per_tick = _parse_chrome_trace(path, prefix)
    try:
        os.remove(path)
        os.rmdir(out_dir)
    except OSError:
        pass
    steady = [per_tick.get(i, {r: 0 for r in _CUDA_ROWS}) for i in range(ticks)]
    return _fold({}, steady)


def run_scenario(cls, res: int, window: int, ticks: int, device: str, *,
                 top: int = 12, want_cuda: bool = True, pan_step: int = 16) -> dict:
    scn = cls(res, window, device, ticks=ticks, pan_step=pan_step)
    try:
        scn.epoch = 0
        api = pass_api(scn, ticks)
        scn.epoch = 1
        frames = pass_frames(scn, ticks, top)
        out = {"scenario": scn.name, "ticks": ticks, "pan_stride": scn.pan_stride(),
               "api": api, "frames": frames}
        scn.epoch = 2
        cuda = pass_cuda(scn, ticks) if want_cuda else {}
        if cuda:
            out["cuda"] = cuda
        return out
    finally:
        scn.teardown()


def run_all(res: int, window: int, ticks: int, device: str, *, prof1: bool = False,
            only=None, top: int = 12, pan_step: int = 16) -> dict:
    """Every scenario on one device. PROF-1 is armed around the WHOLE run when asked, because
    its cost is a per-cook decision (`should_sample`) whose phase a per-scenario arm would
    reset — and the four device syncs it adds per sampled cook are exactly what a host wants
    to see counted."""
    chosen = [c for c in SCENARIOS if only is None or c.name in only]
    out = {"device": device, "res": res, "window": window, "ticks": ticks,
           "prof1": bool(prof1), "scenarios": {}}
    ctx = armed_profiler() if prof1 else None
    if ctx is not None:
        ctx.__enter__()
    try:
        for cls in chosen:
            out["scenarios"][cls.name] = run_scenario(
                cls, res, window, ticks, device, top=top,
                want_cuda=(device == "cuda"), pan_step=pan_step)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Reporting, save / compare, selftest
# ──────────────────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", _PKG, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "?"
    except Exception:
        return "?"


def environment() -> dict:
    from TEX_Wrangle import __version__ as tex_version
    return {"tex_version": tex_version, "tex_sha": _git_sha(),
            "torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(),
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "package_dir": _PKG}


def _print_block(title: str, rows: dict, *, hide_zero: bool = True):
    print(f"\n  {title}")
    print(f"    {'row':44s} {'warm':>7s} {'min':>7s} {'med':>7s} {'max':>7s} {'total':>8s}  st")
    for row, s in rows.items():
        if hide_zero and s["total"] == 0 and s["warmup"] == 0:
            continue
        print(f"    {row:44s} {s['warmup']:7d} {s['min']:7d} {s['median']:7d} "
              f"{s['max']:7d} {s['total']:8d}  {'=' if s['stable'] else '~'}")


def report(result: dict):
    print(f"\n{'=' * 78}")
    print(f"host-path counts — device={result['device']} res={result['res']}^2 "
          f"window={result['window']}^2 ticks={result['ticks']} prof1={result['prof1']}")
    print(f"{'=' * 78}")
    for name, scn in result["scenarios"].items():
        print(f"\n--- {name} ---")
        _print_block("api + allocator", scn["api"], hide_zero=False)
        _print_block("TEX python frames (top)", scn["frames"])
        if scn.get("cuda"):
            _print_block("cuda", scn["cuda"], hide_zero=False)


def _flatten(result: dict) -> dict:
    """`(device, scenario, block, row)` -> the row's stats, for an exact comparison."""
    flat = {}
    dev = result["device"]
    for sname, scn in result["scenarios"].items():
        for block in ("api", "frames", "cuda"):
            for row, s in (scn.get(block) or {}).items():
                flat[f"{dev}/{sname}/{block}/{row}"] = s
    return flat


def compare(current: dict, baseline_path: str) -> int:
    """Row-by-row EXACT diff over the rows that are STABLE on both sides.

    An unstable row cannot gate: its own reading disagrees with itself, so a difference
    against a baseline says nothing. It is reported as `unstable` and excluded from the
    verdict rather than silently compared — the failure mode this harness exists to avoid is
    a gate that fires on noise."""
    with open(baseline_path, "r", encoding="utf-8") as fh:
        base = json.load(fh)
    base_runs = base.get("runs", [])
    cur_runs = current.get("runs", [current])
    bflat, cflat = {}, {}
    for r in base_runs:
        bflat.update(_flatten(r))
    for r in cur_runs:
        cflat.update(_flatten(r))
    changed, unstable, appeared, vanished = [], [], [], []
    for k in sorted(set(bflat) | set(cflat)):
        b, c = bflat.get(k), cflat.get(k)
        if b is None:
            if c["total"]:
                appeared.append((k, c))
            continue
        if c is None:
            if b["total"]:
                vanished.append((k, b))
            continue
        if not (b["stable"] and c["stable"]):
            if b["median"] != c["median"]:
                unstable.append((k, b["median"], c["median"]))
            continue
        if b["min"] != c["min"]:
            changed.append((k, b["min"], c["min"]))
    print(f"\n{'=' * 78}\ncompare vs {baseline_path}\n{'=' * 78}")
    print(f"  baseline: TEX {base.get('env', {}).get('tex_version')} "
          f"@ {str(base.get('env', {}).get('tex_sha'))[:12]}")
    print(f"  current : TEX {current.get('env', {}).get('tex_version')} "
          f"@ {str(current.get('env', {}).get('tex_sha'))[:12]}")
    for k, b, c in changed:
        print(f"  CHANGED   {k}: {b} -> {c}  (per tick, stable both sides)")
    for k, s in appeared:
        print(f"  NEW ROW   {k}: {s['min']}..{s['max']} per tick")
    for k, s in vanished:
        print(f"  GONE      {k}: was {s['min']}..{s['max']} per tick")
    for k, b, c in unstable:
        print(f"  unstable  {k}: median {b} -> {c} (not gated)")
    n = len(changed) + len(appeared) + len(vanished)
    print(f"\n  {n} stable row(s) moved; {len(unstable)} unstable row(s) differ (ignored).")
    return 1 if n else 0


def selftest(device: str) -> int:
    """The ANIM-1 mutation guard, in this harness's terms: show the counters can be NON-ZERO.

    Every gateable assertion downstream is of the form "this row is exactly N", and several
    of the interesting N are 0. A spy list that silently failed to install would satisfy all
    of them. So a COLD tick — one that must compile — is driven here and the compile counters
    are required to fire; a harness that cannot count is a harness that cannot fail."""
    print("--- BENCH-2 selftest: the spies are not inert ---")
    bad = []
    scn = PrewarmScenario(96, 48, device, ticks=1)
    try:
        res = pass_api(scn, 1)
    finally:
        scn.teardown()
    for row, need in (("TEXCache.compile_ast", 1), ("Lexer.tokenize", 1), ("Parser.parse", 1)):
        got = res.get(row, {}).get("min", 0)
        print(f"  cold prewarm tick: {row} = {got} (need >= {need})")
        if got < need:
            bad.append(f"{row}={got} < {need}")
    scn = TerminalKnobScenario(96, 48, device, ticks=1)
    scn.epoch = 1
    try:
        fr = pass_frames(scn, 1, 3)
    finally:
        scn.teardown()
    tot = fr.get("frames.total", {}).get("min", 0)
    print(f"  terminal tick: TEX python frames = {tot} (need > 0)")
    if tot <= 0:
        bad.append(f"frames.total={tot}")
    api = None
    scn = TerminalKnobScenario(96, 48, device, ticks=1)
    scn.epoch = 2
    try:
        api = pass_api(scn, 1)
    finally:
        scn.teardown()
    cooks = api.get("tex_engine.cook", {}).get("min", 0)
    print(f"  terminal tick: tex_engine.cook = {cooks} (need > 0)")
    if cooks <= 0:
        bad.append(f"tex_engine.cook={cooks}")
    if bad:
        print("\nSELFTEST FAILED: " + "; ".join(bad))
        return 1
    print("\nselftest OK — every counter demonstrated non-zero")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="BENCH-2 structural host-path counts")
    p.add_argument("--res", type=int, default=1024, help="canvas resolution (default 1024)")
    p.add_argument("--window", type=int, default=512, help="viewport window (default 512)")
    p.add_argument("--ticks", type=int, default=8, help="steady ticks per scenario")
    p.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    p.add_argument("--scenario", action="append", choices=SCENARIO_NAMES,
                   help="limit to these scenarios (repeatable)")
    p.add_argument("--top", type=int, default=12, help="frames rows to report per scenario")
    p.add_argument("--prof1", choices=("on", "off"), default="off",
                   help="arm PROF-1 (the cost profiler) for the run")
    p.add_argument("--save", metavar="PATH", help="write the counts as JSON")
    p.add_argument("--compare", metavar="PATH", help="exact row diff against a saved JSON")
    p.add_argument("--selftest", action="store_true",
                   help="prove the spies fire, then exit")
    a = p.parse_args(argv)

    have_cuda = torch.cuda.is_available()
    if a.selftest:
        return selftest("cuda" if have_cuda else "cpu")
    devices = ["cpu", "cuda"] if a.device == "both" else [a.device]
    if "cuda" in devices and not have_cuda:
        if a.device == "cuda":
            print("no CUDA device — nothing to measure"); return 2
        devices.remove("cuda")

    runs = []
    for dev in devices:
        r = run_all(a.res, a.window, a.ticks, dev, prof1=(a.prof1 == "on"),
                    only=set(a.scenario) if a.scenario else None, top=a.top)
        report(r)
        runs.append(r)
    payload = {"env": environment(), "runs": runs}
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)) or ".", exist_ok=True)
        with open(a.save, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nsaved -> {a.save}")
    if a.compare:
        return compare(payload, a.compare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
