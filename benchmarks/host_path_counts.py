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

Eight scenarios
---------------
Seven of them drive the comp above. The eighth drives the ComfyUI NODE, because the other
seven structurally cannot: they enter `tex_engine.prepare` with `forgive_dead_refs` off, and
the whole lazy tier hangs off that flag (BENCH-3, from PERF-4's finding F5).

    prewarm          `tex_api.prewarm` over the comp's ten programs, each tick in its OWN
                     cold cache dir (the project-load path; the only cold scenario)
    source_edit      the first WHOLE-FRAME cook after a source edit of one stage
    terminal         terminal-knob scrub: viewport window, `dirty_from` = last stage
    midgraph         the same scrub on stage 5 (so the dirty suffix is five stages)
    pan              the window MOVES 16 px per tick, params constant
    all_dirty        a SOURCE-side knob each tick, whole frame, so the cache misses
    lint             `tex_api.check` with a one-character edit per tick (no cook at all)
    node_scrub       a slider drag on a WIRED ComfyUI node: two `check_lazy_status` rounds
                     (the T4-lite protocol) and then the node's own `execute`

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
    python benchmarks/host_path_counts.py --counters-only --compare results/counts_head.json
    python benchmarks/host_path_counts.py --selftest            # the spies are not inert

`--compare`'s verdict counts the **api** and **cuda** rows only. The `frames.*` census moves
for every lawful change that adds a call or moves a module, so it is reported under its own
heading with the per-scenario sums and never enters the exit code (`--counters-only` names
that rule explicitly for a caller that depends on it).

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


def _cache_warmth(path: str | None) -> str:
    """How full the program/result cache directory was BEFORE anything here ran.

    Two separate lanes lost a comparison to a shared warm cache and read its cold/warm
    difference as a structural change (six `source_edit` rows that looked exactly like one).
    The directory and its warmth therefore travel in the saved provenance, so a comparison
    between two legs that did not both start cold is visible in the file rather than
    reconstructed from memory afterwards.
    """
    if not path:
        return "unset"
    if not os.path.isdir(path):
        return "cold (absent)"
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += len(files)
        if n > 9999:
            break
    return "cold (empty)" if n == 0 else f"warm ({n} file(s))"


#: Captured at IMPORT, before torch or any TEX module can create or fill the directory.
_CACHE_DIR_AT_START = os.environ.get("TEX_CACHE_DIR")
_CACHE_WARMTH_AT_START = _cache_warmth(_CACHE_DIR_AT_START)


if os.path.dirname(_PKG) not in sys.path:
    sys.path.insert(0, os.path.dirname(_PKG))                # .../<package parent>

import torch                                                  # noqa: E402

from TEX_Wrangle import tex_api                               # noqa: E402
from TEX_Wrangle.tex_compiler.types import TEXType            # noqa: E402
from TEX_Wrangle.tex_testkit import cold_engine_state, armed_profiler   # noqa: E402  HOOK-4


def path_prefixes(directory: str) -> tuple:
    """Every spelling a `co_filename` under `directory` can legitimately carry.

    THE BUG THIS EXISTS FOR. On Windows this tree is commonly reached through a junction —
    `custom_nodes\\TEX_Wrangle` -> `custom_nodes\\TEX` — because the package must be importable
    under its import name while the checkout keeps its own. `Path.resolve()` and
    `os.path.realpath()` FOLLOW that junction; an imported module keeps in `code.co_filename`
    the spelling it was imported under. So a profile hook that computes one spelling and
    compares it against the other matches nothing and reports ZERO for every row — silently,
    which is the worst way a counter can fail, because every "this must not run" assertion it
    feeds then passes vacuously. Measured: `tests/test_perf7_compiled_cold.py` read
    `4 passed, 2 errors` when the suite ran from `custom_nodes` (the canonical location) and
    green from any worktree. Accept both spellings and the question does not arise.

    Returned normcased and `os.sep`-terminated, so a prefix match cannot straddle a name
    (`.../tex` must not match `.../tex_wrangle`)."""
    out = []
    for p in (os.path.abspath(directory), os.path.realpath(directory)):
        pref = os.path.normcase(p + os.sep)
        if pref not in out:
            out.append(pref)
    return tuple(out)


def package_relpath(fn: str, prefixes, exclude=()) -> str | None:
    """`co_filename` -> its `/`-separated path under the package, or None if it is outside.

    `exclude` wins over `prefixes` (a directory nested inside the package that must not be
    counted). The slice is taken from the ORIGINAL string, not the normcased one, so the row
    name keeps the file's real spelling; `os.path.normcase` never changes a path's length."""
    norm = os.path.normcase(fn)
    for pref in exclude:
        if norm.startswith(pref):
            return None
    for pref in prefixes:
        if norm.startswith(pref):
            return fn[len(pref):].replace(os.sep, "/").replace("\\", "/")
    return None


_PKG_PREFIXES = path_prefixes(_PKG)
#: `examples/` ships the demo HOST, not the engine. A sync called from there is the host's own
#: frame-completion barrier (`RoiComp.cook`'s trailing `torch.cuda.synchronize()`), which is a
#: host policy decision and not a stall TEX imposed — counting it as engine-side would make the
#: "zero engine syncs per interactive tick" row unpinnable and, worse, wrong.
_EXAMPLES_PREFIXES = path_prefixes(os.path.join(_PKG, "examples"))
#: `benchmarks/` lives under the package too, so the frame filter would otherwise charge every
#: tick for this harness's own sampling closures — measurement counting itself.
_BENCH_PREFIXES = path_prefixes(_HERE)


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
    # BENCH-3: the lazy tier, which only the NODE path enters. `tex_engine.prepare` consults
    # the analysis solely when its caller passes `forgive_dead_refs`, and the only caller that
    # does is `tex_node` (the ComfyUI lazy input pool) — so this row reads 0 on all seven
    # engine-driven scenarios and non-zero only on `node_scrub`, which is exactly what makes
    # it that scenario's non-inert witness.
    "lazy_required_bindings":     ("TEX_Wrangle.tex_lazy.lazy_required_bindings",),
    "Interpreter._exec_stmt":     ("TEX_Wrangle.tex_runtime.interpreter.Interpreter._exec_stmt",),
    "profile.record":             ("TEX_Wrangle.tex_runtime.profile.record",),
    # BENCH-4: the checkpoint-serve tier, which only `checkpoint_serve` enters. Neither the
    # seven comp scenarios nor `node_scrub` ever reach `tex_checkpoint` — the comp has no
    # `ResultCache`-backed linear chain to checkpoint and the node's own chain is never
    # multi-tap — so both rows read 0 everywhere else in this file.
    "tex_checkpoint.cook_checkpointed": ("TEX_Wrangle.tex_checkpoint.cook_checkpointed",),
    "tex_engine.boundary_lineage_key":  ("TEX_Wrangle.tex_engine.boundary_lineage_key",),

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
    # PERF-6: the free-VRAM question, counted at the seam that COSTS the money. The driver
    # call below is only the inner 13-17 us of a 90-112 us host call (the host's own
    # `get_free_memory` also folds in allocator stats), so a `mem_get_info` reading alone
    # under-reports this per-cook cost ~7x. BOTH implementations are patched: which one a run
    # uses depends on whether ComfyUI is importable, and patching one reports a confident zero
    # in the other shape.
    "host.get_free_memory":       ("TEX_Wrangle.tex_runtime.host.NullHostServices.get_free_memory",
                                   "TEX_Wrangle.tex_runtime.host.ComfyHostServices.get_free_memory"),
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
                if f.startswith(_EXAMPLES_PREFIXES):
                    counts[SYNC_ROWS[1]] += 1
                elif f.startswith(_PKG_PREFIXES):
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
        rel = package_relpath(code.co_filename, _PKG_PREFIXES,
                              exclude=_BENCH_PREFIXES)
        if rel is None:
            return
        self.counts[_frame_key(rel, code, frame)] += 1

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


#: The attribute a code object carries its QUALIFIED name in, read through this name and
#: never spelled inline, so one test can force the pre-3.11 path on any interpreter. CI runs
#: Python 3.10, 3.11 and 3.12; neither development box has a 3.10, so the seam is the only
#: way the fallback below can be exercised before CI sees it.
_QUALNAME_ATTR = "co_qualname"

#: `(attr, id(code)) -> (code, qualname)`. The code object is held in the VALUE deliberately:
#: `id()` is unique only among LIVE objects, so a memo that stored the address alone would
#: hand a freed code object's answer to whatever was allocated at the same address next. The
#: reference makes that impossible, at the cost of pinning the code objects a measured run
#: touched — a harness-lifetime cost, paid once per function, which is what keeps the profile
#: hook from doing an attribute derivation per FRAME.
_QUALNAME_MEMO: dict = {}


def _derive_qualname(code, frame) -> str:
    """The 3.10 fallback: `Class.method` from the running frame, or the bare `co_name`.

    A method frame binds its instance as `self` (a classmethod its class as `cls`), and the
    class that actually DEFINES the running code is found through the MRO — `type(self)`
    alone would label a method inherited by a subclass with the subclass's name, which is a
    different row from the one the pins spell. The candidate is confirmed by identity against
    the code object (`__code__ is code`), so a closure that merely closes over an enclosing
    method's `self` — `f_locals` carries free variables too — is not mislabelled as a method
    of that class; it keeps its bare name, which is what it had before this existed."""
    name = code.co_name
    locs = getattr(frame, "f_locals", None) if frame is not None else None
    if not locs:
        return name
    for slot in ("self", "cls"):
        obj = locs.get(slot)
        if obj is None:
            continue
        owner = obj if isinstance(obj, type) else type(obj)
        for klass in getattr(owner, "__mro__", ()):
            fn = klass.__dict__.get(name)
            if fn is None:
                continue
            target = getattr(fn, "__func__", fn)     # classmethod / staticmethod wrapper
            if getattr(target, "__code__", None) is code:
                return klass.__qualname__ + "." + name
    return name


def frame_qualname(code, frame=None) -> str:
    """`code` -> the qualified name a row is keyed by, on every Python this project runs on.

    THE BUG THIS EXISTS FOR. `co_qualname` is 3.11+. On 3.10 — which CI still runs, and which
    neither box here has — a method's code object carries only `co_name`, so a hook keying
    `module:co_name` spells `tex_compiler/lexer:tokenize` where every pin in the suite reads
    `tex_compiler/lexer:Lexer.tokenize`. The pinned row then counts ZERO for ever: the
    "this must not run" assertions pass vacuously and only the mutation guard notices, which
    is precisely how it was found (the 3.10 leg red, 3.11 and 3.12 green). It is the same
    silent-zero failure this harness already has one scar from — see `path_prefixes`.

    On 3.11+ this returns `co_qualname` unchanged, so every key and every pin is
    byte-identical to what it was before the fallback existed. Below that it derives the name
    from the frame (`_derive_qualname`).

    NOT derived, deliberately: a nested function's `<outer>.<locals>.<inner>`, and the name
    of a method reached through a wrapper that does not expose `__func__`. Nothing in the
    suite pins either by name, so on 3.10 such a frame keys by its bare name — a difference
    in the `frames.*` CENSUS between interpreter versions, never in a pinned row. Build it
    the day a pin needs it, not before."""
    attr = _QUALNAME_ATTR
    memo_key = (attr, id(code))
    hit = _QUALNAME_MEMO.get(memo_key)
    if hit is not None and hit[0] is code:
        return hit[1]
    qual = getattr(code, attr, None) or _derive_qualname(code, frame)
    _QUALNAME_MEMO[memo_key] = (code, qual)
    return qual


_FRAME_KEY_MEMO: dict = {}


def _frame_key(rel: str, code, frame=None) -> str:
    """`rel` is already the package-relative, `/`-separated path (package_relpath)."""
    # `_QUALNAME_ATTR` is part of the key, not read past it: a test that forces the pre-3.11
    # path must not be served the native answer this memo warmed a moment earlier.
    key = (_QUALNAME_ATTR, rel, code.co_firstlineno, code.co_name)
    hit = _FRAME_KEY_MEMO.get(key)
    if hit is None:
        mod = rel[:-3] if rel.endswith(".py") else rel
        hit = _FRAME_KEY_MEMO[key] = f"{mod.replace('/', '.')}:{frame_qualname(code, frame)}"
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


class NodeScrubScenario(Scenario):
    """The ComfyUI NODE's own tick: a slider drag on a wired `TEX Wrangle` node.

    THE SEVEN SCENARIOS ABOVE CANNOT SEE THIS PATH, and that is structural rather than an
    oversight. They drive `tex_api` / `tex_engine` directly, where `forgive_dead_refs` defaults
    to False — and `tex_engine.prepare` consults the lazy analysis only when a caller passes
    it. The only caller that does is `tex_node.execute` (`forgive_dead_refs=bool(slot_entries)`,
    the lazy input pool), so `tex_lazy.lazy_required_bindings` reads 0 per tick on every one of
    the seven, including `all_dirty`, which enters `prepare` ten times. A regression on the
    FIRST-CLASS host's per-tick cost could therefore not move a single counts row. It moved
    24 lexes and 24 parses per 12 slider ticks before PERF-4 and 1 and 1 after, and nothing in
    this harness noticed either number.

    So this scenario drives what a user's slider drives, in order:

      round 1  `TEXWrangleNode.check_lazy_status` with the wired scalar still uncooked (it
               arrives None) — ComfyUI's lazy protocol, which names the `in_N` pool slots this
               cook needs;
      round 2  the same call re-invoked once that scalar HAS cooked (the T4-lite round, which
               is what lets a wired scalar fold like a widget value);
      execute  the node's own cook, which reaches the analysis a third time through
               `prepare`'s E6003 forgiveness gate.

    Off ComfyUI exactly as on it: `tex_node` falls back to a plain-`object` base when
    `comfy_api` is absent (`_V3_AVAILABLE`), `execute` then returns a tuple instead of a
    `NodeOutput`, and the slot map is the same JSON-shaped list of dicts the frontend sends.
    `tests/test_lazy_cooking.py` drives the node the same way.

    ONE `$param` MOVES PER TICK and nothing else, so what is counted is a scrub and not a
    re-wire. The value walks in 1e-6 steps from 0.25 and never repeats in a process (`_seq` is
    injective across ticks, passes and scenarios, and fp32 resolves 1e-6 at 0.25 thirty times
    over), so no tick is served a lazy answer another tick minted — the same first-sight
    discipline `_seq` exists for. It also stays strictly inside `(0, 1)`: a `mix` weight that
    reached an endpoint would fold an arm away and change the required set, which is a
    different tick shape and would make every row unstable."""
    name = "node_scrub"

    #: Two image wires, one wired FLOAT scalar (so round 2 exists), one widget `$param`.
    CODE = ("float g = $gain;\n"
            "@OUT = mix(@A, @B, $k) * g;\n")
    SLOTS = ({"name": "A", "slot": "in_0", "type": "IMAGE"},
             {"name": "B", "slot": "in_1", "type": "IMAGE"},
             {"name": "gain", "slot": "in_2", "type": "FLOAT"})

    def build(self):
        import TEX_Wrangle.tex_node as tex_node
        self._node = tex_node.TEXWrangleNode
        # The frontend sends the slot map as a JSON STRING constant in the queued prompt, so
        # the scenario sends one too: `_parse_slot_map` decodes it on every one of the three
        # calls, which is part of what a tick costs and would be missed by handing it a list.
        self._slots = json.dumps([dict(e) for e in self.SLOTS])
        dev = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        torch.manual_seed(7)
        self._a = torch.rand(1, self.res, self.res, 3, device=dev)
        self._b = torch.rand(1, self.res, self.res, 3, device=dev)
        return None

    def prime(self, comp):
        pass                                   # `build` primes; the warm-up tick does the rest

    def _kwargs(self, i: int) -> dict:
        # The same `$k` for all three calls of one tick: the node hands `check_lazy_status` the
        # widget values and the already-cooked wired scalars, and `execute` the same set once
        # they have all arrived. A tick where the two disagreed would be a different bug.
        return {"code": self.CODE, "_tex_slot_map": self._slots,
                "k": 0.25 + self._seq(i) * 1e-6, "device": self.device,
                "compile_mode": "none", "precision": "fp32"}

    def tick(self, comp, i):
        base = self._kwargs(i)
        node = self._node
        node.check_lazy_status(**dict(base, in_0=None, in_1=None, in_2=None))
        node.check_lazy_status(**dict(base, in_0=None, in_1=None, in_2=2.0))
        node.execute(**dict(base, in_0=self._a, in_1=self._b, in_2=2.0))


class CheckpointServeScenario(Scenario):
    """BENCH-4 — the ninth scenario, and the gap `docs/host-path-counts.md` §3 named and left
    unbuilt: a checkpoint-SERVE tick on `tex_checkpoint.cook_checkpointed`.

    A second embedding host reports (2026-09-21) that on its tree this is the INTERACTIVE
    path, not the render path: its router sends the commonest interactive edit — a linear
    fused chain with a SETTLED cost table and a NON-EMPTY cut plan, asking for no window —
    to `cook_checkpointed` for the whole frame. That route also takes one
    `boundary_lineage_key` probe per planned cut, and calls the lazy-binding analysis
    DIRECTLY from its own planner rather than through `prepare()` — a second, independent
    door onto the lazy tier, beside the `forgive_dead_refs` one `node_scrub` already covers.

    None of the eight scenarios above can reach `cook_checkpointed`: it needs a `ResultCache`,
    a LINEAR stage list and a SETTLED PROF-1 table, and neither the ROI/results-cache comp nor
    the ComfyUI node supplies any of the three. So this scenario builds its own tiny fixture —
    a two-stage chain, a triple-blur then the one stage a slider actually drags — and does the
    two things a host must do before a tick can ever reach the served path, both OUTSIDE the
    counted region (in `build()`, exactly where `PrewarmScenario`'s cold state and every other
    scenario's `prime()` already sit — see `Scenario._seq`'s note on why a build never pollutes
    a tick):

      1. SETTLE the cost table. `plan_checkpoints` returns `[]` — cook exactly as today — until
         PROF-1 has `MIN_SAMPLES` (12) samples of this key, and the sampling rule is every cook
         of an unseen key for the first `_WARMUP_SAMPLES` (3), then one in `_SAMPLE_EVERY` (16):
         reaching 12 costs `3 + (12 - 3) * 16` = **147 cooks**, measured here rather than
         assumed (`self.cooks_to_settle`, and `--selftest`-shaped: a scenario that settled by
         assumption would silently pin the UNSETTLED shape the moment the sampling rule moved).
      2. MATERIALIZE the plan once (phase 2), so every steady tick is a genuine cache HIT and
         not a first-cook fallback — `cook_checkpointed` degrades to a whole-chain
         `cook_stage_list` on a miss, and a scenario that measured the fallback would be pinning
         that shape under a checkpoint's name.

    PROF-1 is armed ONLY for the settling loop and disarmed again before the first counted
    tick (`self._profile.disable()` in `build()`), so a steady tick here costs no engine-side
    `torch.cuda.synchronize` — same contract every other interactive scenario in this file
    holds, and `tests/test_bench2_counts.py` pins it the same way.

    Each tick edits the terminal stage's `$knob` — the slider a user actually drags — calls
    the lazy-binding analysis DIRECTLY (never through `prepare()`), then calls
    `cook_checkpointed` with `cuts=None`, so the placement planner runs for real every tick,
    reading the now-frozen table rather than a hand-fed answer. A two-stage chain has exactly
    ONE possible cut (`k=1`, between the blur and the terminal stage), which is therefore also
    its DEEPEST — so `cook_checkpointed`'s deepest-first probe hits on the FIRST
    `boundary_lineage_key` call: one probe, one cache read, one cheap suffix cook of the single
    terminal stage, never the blur.

    THE THRESHOLD (`_THRESHOLD_MS`) is deliberately far below the blur's own cost, so which
    side of the boundary wins is decided by the MATERIALIZATION FLOOR
    (`tex_checkpoint.put_cost_ms(...) * _FLOOR_FACTOR`), never by a knife-edge comparison
    between two similar numbers. Measured at this file's gate shape (96^2, CPU, three
    independent settlings): stage 0 costs 0.37-0.46 ms against a 0.0785 ms floor — a >=4.6x
    margin, which is what makes `cuts == [1]` reproduce identically run after run. That margin
    is NOT claimed at every shape: a from-scratch CPU run at 1024^2 puts the same floor at
    ~8.9 ms, inside the blur's own run-to-run noise there (measured 7.6-11.5 ms over three
    settlings) — so this scenario is driven ONLY at the gate shape and the CUDA report shape
    (96^2 and 1024^2 both margin >=2.7x on the CUDA leg), never at a CPU 1024^2 shape, and nothing
    in this file pins one.
    """
    name = "checkpoint_serve"
    needs_comp = False

    #: Three chained blurs, not one: comfortably clear of the materialization floor at every
    #: shape this file measures (see the class docstring's margin numbers). A single blur's
    #: margin at 96^2 was only ~1.6-2.7x — closer to the noise floor than this file's other
    #: pins, all of which are pure call-count structure and pay no such margin at all.
    _BLUR_CODE = "@OUT = gauss_blur(gauss_blur(gauss_blur(@IN, 8.0), 8.0), 8.0);"
    _EDITED_CODE = "@OUT = vec4(@IN.rgb * $knob, 1.0);"
    #: Far below every measured stage-0 cost in the class docstring's margin table. Kept
    #: explicit — never the GOV-1 default (`tex_checkpoint.default_threshold_ms`) — so this
    #: scenario's plan cannot move because an earlier test in the same process changed the
    #: active memory profile; the MATERIALIZATION FLOOR is what actually gates the cut at
    #: every shape this file drives (see `put_cost_ms`).
    _THRESHOLD_MS = 0.05
    #: CACHE-1's content-sensitive source identity. A fixed string is fine: this scenario never
    #: swaps the source tensor underneath a served boundary.
    _UPSTREAM = ("bench4-checkpoint-src-v1",)
    #: `tex_checkpoint.MIN_SAMPLES` plus generous slack, so a sampling-rule change that pushes
    #: the settling point out (rather than removing it) fails LOUD in `build()` instead of
    #: silently pinning the unsettled shape.
    _MAX_WARMUP_COOKS = 400

    def _stages(self, src, knob):
        """The two-stage linear chain: `chain_input` spelling (CACHE-6's linear shape), never
        the `chain_inputs` DAG one `gate_refusal` requires `collapse_linear` for first."""
        return [{"code": self._BLUR_CODE, "chain_input": None, "bindings": {"IN": src}},
                {"code": self._EDITED_CODE, "chain_input": "IN", "bindings": {"knob": knob}}]

    def build(self):
        from TEX_Wrangle import tex_checkpoint, tex_engine, tex_results
        from TEX_Wrangle.tex_runtime import profile as _profile
        self._checkpoint, self._engine, self._profile = tex_checkpoint, tex_engine, _profile
        dev = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
        torch.manual_seed(11)
        src = torch.rand(1, self.res, self.res, 3, device=dev)
        cache = tex_results.ResultCache()
        # Salted so pass B (frames) and pass C (cuda) each settle their OWN bucket rather than
        # inheriting pass A's — the exact global-memo trap `Scenario._seq`'s docstring names.
        pkey = _profile.make_key(f"bench4-checkpoint-serve-{self._salt}-{self.epoch}",
                                 dev, "fp32")
        spatial = (1, self.res, self.res)
        _profile.reset()
        _profile.enable()
        warm = 0
        try:
            while warm < self._MAX_WARMUP_COOKS and not _profile.settled(
                    pkey, spatial, need=tex_checkpoint.MIN_SAMPLES):
                with _profile.measure(pkey, spatial, device=dev, stages=True):
                    tex_engine.cook_stage_list(
                        self._stages(src, 0.5 + (warm % 17) * 0.01),
                        device=dev, precision="fp32")
                warm += 1
        finally:
            _profile.disable()          # OFF before the first counted tick — see the docstring
        self.cooks_to_settle = warm
        costs, is_settled = _profile.stage_snapshot(pkey, spatial, need=tex_checkpoint.MIN_SAMPLES)
        cuts = tex_checkpoint.plan_checkpoints(
            self._stages(src, 0.5), costs=costs, threshold_ms=self._THRESHOLD_MS,
            px=self.res * self.res, settled=is_settled, device=dev)
        if not cuts:
            # The scenario's whole premise is a NON-EMPTY plan (§3's gap is specifically about
            # the checkpointed cook being reached). Failing loud here, rather than silently
            # falling through to a whole-chain cook, is what makes an unmet margin a build
            # error instead of a scenario that quietly stopped testing what it claims to.
            raise RuntimeError(
                f"CheckpointServeScenario: no checkpoint planned at res={self.res} device={dev} "
                f"(costs={costs}, settled={is_settled}) — the margin the class docstring "
                f"measures did not hold on this box/shape")
        self.cuts_planned = cuts
        tex_checkpoint.materialize(self._stages(src, 0.5), cache, device=dev, precision="fp32",
                                   upstream=self._UPSTREAM, cuts=cuts)
        return _CheckpointComp(cache=cache, pkey=pkey, spatial=spatial, device=dev, src=src)

    def prime(self, comp):
        pass                                   # `build` primes; there is no separate warm-up

    def tick(self, comp, i):
        knob = 0.5 + self._seq(i) * 1e-6        # never repeats — node_scrub's discipline
        # The host's own planner call — DIRECT, never through `prepare()`'s
        # `forgive_dead_refs` gate (that is node_scrub's door onto this tier; this is the
        # OTHER one, per the class docstring).
        from TEX_Wrangle import tex_lazy
        tex_lazy.lazy_required_bindings(self._EDITED_CODE, {"knob": knob})
        stages = self._stages(comp.src, knob)
        self._checkpoint.cook_checkpointed(
            stages, comp.cache, device=comp.device, precision="fp32",
            upstream=self._UPSTREAM, cuts=None, threshold_ms=self._THRESHOLD_MS,
            profile_key=comp.pkey, spatial=comp.spatial)


class _CheckpointComp:
    """The tiny fixture `CheckpointServeScenario.build()` hands to `tick()` — not a `RoiComp`,
    because the comp's ten-stage ROI/results-cache pattern has no checkpoint concept at all
    (BENCH-4's whole reason for existing is that none of the other scenarios can reach this
    tier). Deliberately not a dict: `comp.cache.stats()` is read by nothing here, but keeping
    the same attribute-access shape as `RoiComp` is what lets `_cache_entries` in this file
    stay one function instead of branching on scenario type."""

    __slots__ = ("cache", "pkey", "spatial", "device", "src")

    def __init__(self, cache, pkey, spatial, device, src):
        self.cache, self.pkey, self.spatial = cache, pkey, spatial
        self.device, self.src = device, src


class InterpChainScrubScenario(Scenario):
    """v0.42's ASK-shaped scenario (`docs/host-path-counts.md`'s "the interactive floor"):
    the path an embedding host's own interactive tick actually runs, end to end — the
    interpreter tier (`compile_mode="none"`, TEX's own default, never touched by this
    scenario), unfused (one `tex_engine.cook` per stage, exactly as every scenario in this
    file already drives it — none of them ever route through a fused chain), ROI-windowed,
    fp32, with EVERY stage treated as an uncached node between the nearest cache and the
    viewer. That last part is the one no scenario above combines with a moving window: an
    embedding host's own caching policy does not memoize a cheap node at all, so on a real
    interactive tick it re-cooks every uncached stage between the nearest cache and the
    viewer, and `RoiComp.cook(use_cache=False)` is the comp's own honest setting for exactly
    that claim (see its docstring — "forces every dirty stage to actually cook").

    Both axes a real scrub moves, together: the viewport window pans (`_pan_roi`, never
    revisiting a position — `PanScenario`'s own discipline) AND the terminal stage's
    `$param` changes (`TerminalKnobScenario`'s own discipline) — on the SAME tick. No
    existing scenario combines them: `pan` holds every param constant so the walk memo
    hits, and `terminal`/`midgraph` hold the window still so the LAT-4 coordinate-builtin
    LRU hits. Here neither memo can hit, `dirty_from=0` so the clean-prefix skip never
    fires, and `use_cache=False` so the results-cache probe never fires either — the whole
    ten-stage chain's fixed per-cook overhead is paid once per stage, every tick, which is
    the shape TRK-64/TRK-65/TRK-72/TRK-84 each proved themselves against."""
    name = "interp_chain_scrub"

    def tick(self, comp, i):
        comp.params["vignette"]["strength"] = 0.30 + self._seq(i) * 0.001
        comp.cook(self._pan_roi(i), 0, use_cache=False)


SCENARIOS = (PrewarmScenario, SourceEditScenario, TerminalKnobScenario,
             MidGraphKnobScenario, PanScenario, AllDirtyScenario, LintScenario,
             NodeScrubScenario, CheckpointServeScenario, InterpChainScrubScenario)
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
    """`HEAD`, with a `-dirty` suffix when the worktree carries uncommitted changes.

    Without the suffix an edited tree saves under the baseline's own sha, so a `--compare`
    header prints the same twelve characters on both sides and the reader has no way to see
    that the two legs are not the two commits they name.
    """
    try:
        sha = subprocess.run(["git", "-C", _PKG, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "?"
    if not sha:
        return "?"
    try:
        porcelain = subprocess.run(["git", "-C", _PKG, "status", "--porcelain"],
                                   capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return sha + "-dirty?"
    return sha + "-dirty" if porcelain.strip() else sha


def environment(res=None, window=None, ticks=None, device=None) -> dict:
    """`res`/`window`/`ticks`/`device` are the MEASUREMENT SHAPE (TRK-100): a saved
    baseline is coupled to the shape it was taken at, and until these fields existed
    nothing recorded that, so `--compare` between two saves taken at different
    `--res`/`--window`/`--ticks`/`--device` silently reported every row as `NEW ROW` /
    `GONE` rather than saying the shapes differ. `tools/gate.py` still hardcodes the
    gate shape for its own leg; this is what lets a manual `--compare` catch the
    mismatch on its own instead of relying on a caller to have remembered to match it."""
    from TEX_Wrangle import __version__ as tex_version
    return {"tex_version": tex_version, "tex_sha": _git_sha(),
            "torch": torch.__version__, "python": platform.python_version(),
            "platform": platform.platform(), "machine": platform.machine(),
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "package_dir": _PKG,
            "tex_cache_dir": _CACHE_DIR_AT_START,
            "tex_cache_warmth": _CACHE_WARMTH_AT_START,
            "res": res, "window": window, "ticks": ticks, "device": device}


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


#: Blocks whose rows carry the verdict. `frames` is deliberately absent — see `compare`.
_COUNTER_BLOCKS = ("api", "cuda")


def _block_of(key: str) -> str:
    """`(device, scenario, block, row)` is the flat key's shape; return the block."""
    parts = key.split("/")
    return parts[2] if len(parts) > 3 else ""


def _frames_totals(flat: dict) -> dict:
    """Per `(device, scenario)`: the census total and the sum of the per-module subtotals.

    A module split moves every `frames.mod.*` row it touches while conserving both of these,
    which is what makes the frame rows a census rather than a gate: the only honest reading
    of thirteen moved subtotals is "the same work, attributed differently", and that claim is
    only checkable if the sums travel beside the rows."""
    out = {}
    for key, s in flat.items():
        parts = key.split("/")
        if len(parts) < 4 or parts[2] != "frames":
            continue
        scope = f"{parts[0]}/{parts[1]}"
        tot, mod = out.setdefault(scope, [0, 0])
        row = "/".join(parts[3:])
        if row == "frames.total":
            tot += s["min"]
        elif row.startswith("frames.mod."):
            mod += s["min"]
        out[scope] = [tot, mod]
    return out


def compare(current: dict, baseline_path: str, scenario=None) -> int:
    """Row-by-row EXACT diff over the rows that are STABLE on both sides.

    `scenario`, when given (the CLI's `--scenario` names), restricts BOTH legs to
    those scenario names before diffing — not just the current run. Without this, a
    scenario ADDED since the baseline was saved arrives on the current side alone, its
    rows read as `NEW ROW` and count toward the verdict, so a bare `--compare` against
    a pre-change baseline returns rc 1 for a change that moved nothing on any EXISTING
    scenario. `--scenario` already narrowed the current run (`run_all`'s `only=`); this
    narrows the baseline the same way, so naming the scenarios the stored baseline
    actually has is what the comparison is scoped to, on both sides, not just one.

    An unstable row cannot gate: its own reading disagrees with itself, so a difference
    against a baseline says nothing. It is reported as `unstable` and excluded from the
    verdict rather than silently compared — the failure mode this harness exists to avoid is
    a gate that fires on noise.

    The **frame census is excluded from the verdict for the same reason, one level up.** The
    `frames.*` rows count Python frames per `module:function`, so every lawful change that
    adds a call, renames a helper or moves code between modules moves them by construction —
    a module split moved thirteen `frames.mod.*` rows with the sums conserved to the unit,
    and a scenario ADDITION moves them too. A verdict that counted those rows returned 1 for
    every such change, which made the exit code carry no information and forced the reader to
    reason past it by hand; that is how a gate becomes decoration. The API and CUDA rows are
    the ones that state a structural claim ("this seam is entered exactly N times per tick"),
    so they alone are counted. The frame rows are reported in full, with their sums, under
    their own heading."""
    with open(baseline_path, "r", encoding="utf-8") as fh:
        base = json.load(fh)
    benv, cenv = base.get("env", {}) or {}, current.get("env", {}) or {}
    shape_fields = ("res", "window", "ticks", "device")
    mismatched = [f for f in shape_fields
                  if benv.get(f) is not None and cenv.get(f) is not None
                  and benv.get(f) != cenv.get(f)]
    if mismatched:
        print(f"\n{'=' * 78}\ncompare vs {baseline_path}\n{'=' * 78}")
        print("  REFUSED: the measurement SHAPE differs between the two legs, so no row "
              "is comparable (TRK-100):")
        for f in mismatched:
            print(f"    {f}: baseline={benv.get(f)!r}  current={cenv.get(f)!r}")
        print(f"    re-save the baseline at THIS shape first: --res {cenv.get('res')} "
              f"--window {cenv.get('window')} --ticks {cenv.get('ticks')} "
              f"--device {cenv.get('device')}")
        return 1
    base_runs = base.get("runs", [])
    cur_runs = current.get("runs", [current])
    bflat, cflat = {}, {}
    for r in base_runs:
        bflat.update(_flatten(r))
    for r in cur_runs:
        cflat.update(_flatten(r))
    if scenario:
        scenario = set(scenario)
        bflat = {k: v for k, v in bflat.items() if k.split("/")[1] in scenario}
        cflat = {k: v for k, v in cflat.items() if k.split("/")[1] in scenario}
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
    print(f"  baseline: TEX {benv.get('tex_version')} @ {str(benv.get('tex_sha'))[:19]}"
          f"  cache {benv.get('tex_cache_dir') or '<unset>'} [{benv.get('tex_cache_warmth', '?')}]")
    print(f"  current : TEX {cenv.get('tex_version')} @ {str(cenv.get('tex_sha'))[:19]}"
          f"  cache {cenv.get('tex_cache_dir') or '<unset>'} [{cenv.get('tex_cache_warmth', '?')}]")
    if scenario:
        print(f"  --scenario filter applied to BOTH legs: {sorted(scenario)}")
    for note in _provenance_warnings(benv, cenv):
        print(f"  ! {note}")

    def _counters(rows):
        return [t for t in rows if _block_of(t[0]) in _COUNTER_BLOCKS]

    def _frames(rows):
        return [t for t in rows if _block_of(t[0]) == "frames"]

    for k, b, c in _counters(changed):
        print(f"  CHANGED   {k}: {b} -> {c}  (per tick, stable both sides)")
    for k, s in _counters(appeared):
        print(f"  NEW ROW   {k}: {s['min']}..{s['max']} per tick")
    for k, s in _counters(vanished):
        print(f"  GONE      {k}: was {s['min']}..{s['max']} per tick")
    for k, b, c in unstable:
        print(f"  unstable  {k}: median {b} -> {c} (not gated)")

    fch, fap, fvn = _frames(changed), _frames(appeared), _frames(vanished)
    nframes = len(fch) + len(fap) + len(fvn)
    print(f"\n  --- frames census: {nframes} row(s) moved (REPORTED, never gated) ---")
    if nframes:
        for k, b, c in fch:
            print(f"    frames    {k}: {b} -> {c}")
        for k, s in fap:
            print(f"    frames +  {k}: {s['min']}..{s['max']} per tick")
        for k, s in fvn:
            print(f"    frames -  {k}: was {s['min']}..{s['max']} per tick")
        btot, ctot = _frames_totals(bflat), _frames_totals(cflat)
        touched = sorted({"/".join(k.split("/")[:2]) for k, *_ in (fch + fap + fvn)})
        print(f"    {'sums (per tick)':44s} {'base':>10s} {'current':>10s}")
        for scope in touched:
            bt, bm = btot.get(scope, [0, 0])
            ct, cm = ctot.get(scope, [0, 0])
            mark = "=" if (bt, bm) == (ct, cm) else "~"
            print(f"    {scope + '  frames.total':44s} {bt:10d} {ct:10d}  {mark}")
            print(f"    {scope + '  sum(frames.mod.*)':44s} {bm:10d} {cm:10d}  {mark}")

    n = len(_counters(changed)) + len(_counters(appeared)) + len(_counters(vanished))
    print(f"\n  {n} stable counter row(s) moved; {nframes} frame row(s) moved (not gated); "
          f"{len(unstable)} unstable row(s) differ (ignored).")
    return 1 if n else 0


def _provenance_warnings(benv: dict, cenv: dict) -> list:
    """What the two saved environments say about whether the comparison is meaningful."""
    out = []
    bs, cs = str(benv.get("tex_sha")), str(cenv.get("tex_sha"))
    if bs.endswith("-dirty") or cs.endswith("-dirty"):
        out.append("one side was measured on a DIRTY worktree — the sha names a commit the "
                   "measured tree is not")
    if bs != "None" and bs == cs:
        out.append("both sides carry the SAME sha — this is a null control, not a change")
    bd, cd = benv.get("tex_cache_dir"), cenv.get("tex_cache_dir")
    if bd and cd and os.path.normcase(str(bd)) == os.path.normcase(str(cd)):
        out.append("both legs used the SAME TEX_CACHE_DIR — the second leg cannot have been "
                   "cold; give each leg its own directory")
    for side, env in (("baseline", benv), ("current", cenv)):
        w = str(env.get("tex_cache_warmth", ""))
        if w.startswith("warm"):
            out.append(f"the {side} leg started with a WARM cache [{w}] — cold/warm program "
                       "cache rows will read as a structural change")
        elif w == "unset":
            out.append(f"the {side} leg ran with TEX_CACHE_DIR unset — its warmth is unknown")
    return out


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
    p.add_argument("--counters-only", action="store_true",
                   help="state explicitly that --compare's verdict counts the API and CUDA "
                        "rows only and reports the frame census outside it; this is the "
                        "default and the flag is an assertion of it, so a caller (tools/"
                        "gate.py) can name the rule it relies on instead of inheriting it")
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
    payload = {"env": environment(a.res, a.window, a.ticks, a.device), "runs": runs}
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)) or ".", exist_ok=True)
        with open(a.save, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nsaved -> {a.save}")
    if a.compare:
        if a.counters_only:
            print("\n  --counters-only: the verdict counts api + cuda rows; the frame census "
                  "is reported below it and never in the exit code.")
        return compare(payload, a.compare, scenario=set(a.scenario) if a.scenario else None)
    if a.counters_only and not a.compare:
        print("\n  --counters-only has no effect without --compare (it names a verdict rule).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
