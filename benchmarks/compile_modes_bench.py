#!/usr/bin/env python3
"""
compile_modes_bench.py — COMPILE-M1: `compile_mode="none"` vs `"auto"`, measured
==============================================================================
A timing companion to `benchmarks/host_path_counts.py` (which counts calls and never times
anything). This script answers the v0.46 measurement step's own question: on the two shapes
an embedding host actually drives — one interactive edit tick, and steady playback — does
`compile_mode="auto"` (CC-2's measured background-compile-then-trial tier)
beat `"none"` (TEX's own default, and the shape `Bible/Perceived Performance.md`-class hosts
run today), and at what cost.

The CLI names no host. `--shape` takes the two host-neutral scenario names
`benchmarks/host_path_counts.py` already pins the CALL COUNTS of:

    host_tick_exact   one interactive edit tick on TEX's own 10-stage demo chain
                      (`examples/host_demo.py::_COMP_STAGES`) — one dirty stage (the terminal),
                      whole-frame, exactly `test_bench2_counts.py`'s `_HOST_TICK_EXACT_D1` shape.
    playback_frames   the SAME 10-stage chain, all ten stages cooked every tick, only
                      `time_context`'s frame moving — `_PLAYBACK_FRAMES`'s shape.

This file does not reuse those scenario CLASSES (they hardcode `compile_mode="none"`, which is
the fact the counts gate pins about real host behaviour) — it reuses the same fixture (the
same ten programs, the same chain shape) with `compile_mode` as a free variable instead.

WHAT IT REPORTS, per (shape, device, compile_mode):
    (a) first-cook wall time; for "auto", the tick at which every cooked program in this
        shape reaches autotier's COMMITTED state (or "never" inside the run).
    (b) p50/p95/p99 wall-clock ms per tick, over >= 200 steady ticks after warm-up.
    (c) INTERFERENCE: while "auto"'s background trial compile is in flight (a `_COMPILE_POOL`
        worker thread), an unrelated interactive workload's ticks, at a fixed cadence, against
        the same workload with no compile in flight — p95/p99 deltas, plus a coarse
        thread-blocked-time signal (`time.thread_time()` vs wall clock) that names possible
        GIL/resource contention when the ratio drops materially during the compile.
    (d) compiled-state accounting: the VRAM delta (`torch.cuda.memory_allocated`/`reserved`)
        and on-disk bytes of the compiled artefacts (TEXCache's `*.cg` sidecars + the Inductor
        cache dir) after warm-up.
    (e) whether "auto"'s output is bit-identical to "none"'s (TEX's invariant #2 says codegen
        and the interpreter must already agree; "none" is what stands in for that oracle here),
        or the max abs diff if not.

Robust to a kernel-load block (Windows Smart App Control killing a freshly-compiled Inductor
DLL, v0.45.1's own finding): every leg is wrapped, and a failure is reported as a `"error"`
finding in that leg's JSON, never a crashed run. This script never sets
`TORCHINDUCTOR_CACHE_DIR` itself — if your box needs a persistent, trusted Inductor cache
directory to avoid that block, point the environment variable at one BEFORE invoking this
script (`tex_runtime/compiled.py::_ensure_inductor_cache_dir` respects a pre-set value).

Usage
-----
    python benchmarks/compile_modes_bench.py --shape host_tick_exact --device cpu --res 96
    python benchmarks/compile_modes_bench.py --shape both --device both --res 512 --ticks 220
    python benchmarks/compile_modes_bench.py --smoke                    # a handful of ticks
    python benchmarks/compile_modes_bench.py --save results/compile_modes_<box>.json

`TEX_CACHE_DIR` is read from the environment, exactly as every other harness in this repo
reads it (set it BEFORE this script imports the package — see AGENTS.md's command reference). Left
unset, a fresh temp directory is minted per run so "first cook" and "time to adopt" are
measured cold rather than served from a previous invocation's persisted verdicts.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                                 # .../TEX_Wrangle
if os.path.dirname(_PKG) not in sys.path:
    sys.path.insert(0, os.path.dirname(_PKG))                 # .../<package parent>

#: Captured before any TEX import can create the directory — mirrors host_path_counts.py.
_CACHE_DIR_AT_START = os.environ.get("TEX_CACHE_DIR")
if not _CACHE_DIR_AT_START:
    _tmp_cache = tempfile.mkdtemp(prefix="tex_compile_modes_")
    os.environ["TEX_CACHE_DIR"] = _tmp_cache
    _CACHE_DIR_AT_START = _tmp_cache
    _CACHE_WAS_MINTED = True
else:
    _CACHE_WAS_MINTED = False

import torch                                                   # noqa: E402

from TEX_Wrangle import tex_engine                              # noqa: E402
from TEX_Wrangle import tex_fusion                               # noqa: E402  COMPILE-M3 --fused
from TEX_Wrangle.tex_cache import TEXCache, get_cache           # noqa: E402
from TEX_Wrangle.tex_marshalling import (                        # noqa: E402
    identity_binding_types, infer_binding_type)
from TEX_Wrangle.tex_runtime import autotier                    # noqa: E402
from TEX_Wrangle.tex_runtime import compiled as tex_compiled    # noqa: E402  observability only

SHAPES = ("host_tick_exact", "playback_frames")
MODES = ("none", "auto")
# COMPILE-M3: the hypothesis under test is that compiling a FUSED region (one Inductor
# program spanning the old node boundaries) wins where M2/M2b already showed per-node
# compiles losing on every trial (measured on the sm_75 reference box). "torch_compile" is
# the forced compiled tier (skips autotier's measure/trial loop and always compiles) — the
# lever the brief asks to pick alongside "auto" so a fused program's compile cost and
# adoption are visible even when the measured trial would reject it. Scoped to
# playback_frames only (the brief's own scope: "on the playback shape") — host_tick_exact's
# whole point is that only ONE stage is dirty per tick, so forcing all ten through one fused
# program every tick would defeat the shape it is meant to measure.
FUSED_MODES = ("none", "auto", "torch_compile")


def _load_host_demo():
    """`examples/host_demo.py`, loaded by path — same loader as `host_path_counts.py`'s (kept
    as its own copy rather than an import of that file, which is not a package and is
    `.comfyignore`d; see that file's own note on why a path load is what keeps this honest
    about measuring the tree the design note names)."""
    mod = sys.modules.get("_tex_compile_modes_host_demo")
    if mod is not None:
        return mod
    path = os.path.join(_PKG, "examples", "host_demo.py")
    spec = importlib.util.spec_from_file_location("_tex_compile_modes_host_demo", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tex_compile_modes_host_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


# ──────────────────────────────────────────────────────────────────────────────
# The fixture: the SAME ten-stage chain both shapes drive, `compile_mode` free.
# ──────────────────────────────────────────────────────────────────────────────

class Fixture:
    """One shape ("host_tick_exact" or "playback_frames") at one (res, device)."""

    def __init__(self, shape: str, res: int, device: str):
        if shape not in SHAPES:
            raise ValueError(f"unknown shape {shape!r}, want one of {SHAPES}")
        demo = _load_host_demo()
        self.shape, self.res, self.device = shape, res, device
        self.names = [n for n, _c, _d in demo._COMP_STAGES]
        self.codes = [c for _n, c, _d in demo._COMP_STAGES]
        self.defaults = [dict(d) for _n, _c, d in demo._COMP_STAGES]
        self.n = len(self.codes)
        # host_tick_exact's own D=1 (`_HOST_TICK_EXACT_D1`'s shape) — the terminal stage alone.
        self.dirty = 1 if shape == "host_tick_exact" else self.n
        torch.manual_seed(101)
        self.src = torch.rand(1, res, res, 3, device=device)
        self.clean_src = self.src
        if shape == "host_tick_exact":
            # Prime the clean prefix ONCE at compile_mode="none" — never timed, never counted
            # toward either mode's stats. Mirrors `WholeFrameChainScenario`'s standing canvas.
            clean = self.src
            for i in range(self.n - self.dirty):
                out = tex_engine.cook(self.codes[i], {"IN": clean, **self.defaults[i]},
                                      device_mode=device, precision="fp32",
                                      compile_mode="none", time_context=None)
                clean = out.outputs["OUT"]
            self.clean_src = clean

    def dirty_range(self):
        return range(self.n - self.dirty, self.n)

    def stage_bindings(self, j: int, src, tc: dict) -> dict:
        return {"IN": src, **self.defaults[j]}

    def program_key(self, j: int, device_type: str, precision: str = "fp32"):
        """The (fingerprint, autotier-key) pair for stage `j`, computed the SAME way the
        engine derives its own cache key — `identity_binding_types` + `TEXCache.fingerprint`
        — so polling `autotier.verdict()` from OUTSIDE the engine asks about the exact key
        `run_auto` commits under. Observability only: no product call is made here."""
        bindings = self.stage_bindings(j, self.clean_src if self.shape == "host_tick_exact"
                                       else self.src, {})
        types = identity_binding_types(self.codes[j], bindings)
        fp = TEXCache.fingerprint(self.codes[j], types)
        key = autotier.make_key(fp, device_type, precision, (1, self.res, self.res))
        return fp, key

    def tick(self, i: int, compile_mode: str):
        """Run ONE tick at `compile_mode`. Returns the final output tensor."""
        tc = {"frame": float(i), "fps": 24.0, "time": i / 24.0}
        if self.shape == "playback_frames":
            src = self.src
            for j in range(self.n):
                out = tex_engine.cook(self.codes[j], self.stage_bindings(j, src, tc),
                                      device_mode=self.device, precision="fp32",
                                      compile_mode=compile_mode, cancel=None,
                                      time_context=tc)
                src = out.outputs["OUT"]
            return src
        # host_tick_exact
        src = self.clean_src
        for j in self.dirty_range():
            out = tex_engine.cook(self.codes[j], self.stage_bindings(j, src, tc),
                                  device_mode=self.device, precision="fp32",
                                  compile_mode=compile_mode, cancel=None, time_context=tc)
            src = out.outputs["OUT"]
        return src

    # ── COMPILE-M3: the same ten stages spliced into ONE fused program ──────────
    # (`tex_fusion.compile_fused`, reached through `tex_engine.cook(chain_payload=...)` —
    # the same GraphSpec shape a real host's frontend hands the terminal node, per
    # `tests/test_v020_phase1.py::test_f1b_fused_node_path_reaches_compile_tier`). A fresh
    # spec is built every tick, matching `tex_fusion`'s own module docstring ("The frontend
    # rebuilds the _tex_chain payload on every queue") rather than caching it across ticks,
    # which would understate a real host's per-cook payload-assembly cost.

    def build_fused_spec(self):
        """(spec, terminal_code, terminal_bindings) for the WHOLE ten-stage chain, source-
        first, terminal last — the shape `tex_fusion.prepare_fused` / `_stages_from_spec`
        read (`image_input`/`terminal_image_input` both "IN", matching every stage's own
        `@IN` binding name in `examples/host_demo.py::_COMP_STAGES`)."""
        stages_payload = [
            {"code": self.codes[j], "image_input": "IN", "params": dict(self.defaults[j])}
            for j in range(self.n - 1)
        ]
        spec = {"stages": stages_payload, "terminal_image_input": "IN"}
        terminal_code = self.codes[-1]
        bindings = {"IN": self.src, **dict(self.defaults[-1])}
        return spec, terminal_code, bindings

    def fused_tick(self, i: int, compile_mode: str):
        """Run ONE tick of the WHOLE chain as a single fused cook. `playback_frames`
        only (see FUSED_MODES's comment) — every stage is dirty every tick there anyway,
        so fusing changes nothing about which pixels are produced, only how they're
        computed (invariant #2: the two must stay bit-identical)."""
        tc = {"frame": float(i), "fps": 24.0, "time": i / 24.0}
        spec, terminal_code, bindings = self.build_fused_spec()
        out = tex_engine.cook(terminal_code, bindings, chain_payload=spec,
                              device_mode=self.device, precision="fp32",
                              compile_mode=compile_mode, cancel=None, time_context=tc)
        return out.outputs["OUT"]


# ──────────────────────────────────────────────────────────────────────────────
# Timing helpers (invariant #6: sync only at the measurement boundary)
# ──────────────────────────────────────────────────────────────────────────────

def _sync(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_tick(fx: Fixture, i: int, compile_mode: str):
    _sync(fx.device)
    t0 = time.perf_counter()
    tt0 = time.thread_time()
    out = fx.tick(i, compile_mode)
    _sync(fx.device)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    thread_ms = (time.thread_time() - tt0) * 1000.0
    return out, wall_ms, thread_ms


def _pctl(xs: list, p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    idx = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[idx]


def _stats(xs: list) -> dict:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "p50": round(_pctl(xs, 0.50), 4), "p95": round(_pctl(xs, 0.95), 4),
            "p99": round(_pctl(xs, 0.99), 4), "min": round(min(xs), 4), "max": round(max(xs), 4),
            "mean": round(statistics.fmean(xs), 4)}


# ──────────────────────────────────────────────────────────────────────────────
# Compiled-state accounting (d)
# ──────────────────────────────────────────────────────────────────────────────

def _dir_bytes(path, pattern: str = "*") -> int:
    p = Path(path)
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob(pattern):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _cg_and_inductor_bytes() -> dict:
    cache = get_cache()
    cg_bytes = _dir_bytes(cache._cache_dir, "*.cg")
    # `TORCHINDUCTOR_CACHE_DIR`, when the CALLER pre-set it (the Windows Smart App Control
    # workaround this file's own docstring names), wins over `cache.torch_compile_cache_dir` —
    # `_ensure_inductor_cache_dir` is a no-op once that variable is already set, so Inductor's
    # real bytes land THERE, not under this run's `TEX_CACHE_DIR`. Measuring the wrong one
    # would silently read 0 on exactly the box this ask is written for.
    inductor_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR") or str(cache.torch_compile_cache_dir)
    inductor_bytes = _dir_bytes(inductor_dir, "*")
    return {"cg_bytes": cg_bytes, "inductor_bytes": inductor_bytes,
            "cg_dir": str(cache._cache_dir), "inductor_dir": inductor_dir}


def _vram_snapshot(device: str) -> dict:
    if device != "cuda" or not torch.cuda.is_available():
        return {"allocated": None, "reserved": None}
    return {"allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved())}


def _vram_delta(before: dict, after: dict) -> dict:
    if before["allocated"] is None or after["allocated"] is None:
        return {"allocated_delta": None, "reserved_delta": None}
    return {"allocated_delta": after["allocated"] - before["allocated"],
            "reserved_delta": after["reserved"] - before["reserved"]}


# ──────────────────────────────────────────────────────────────────────────────
# (e) output agreement between compile_mode legs
# ──────────────────────────────────────────────────────────────────────────────

def _compare_outputs(a: "torch.Tensor | None", b: "torch.Tensor | None") -> dict:
    if a is None or b is None:
        return {"comparable": False, "reason": "one leg produced no output (see its own error)"}
    if a.shape != b.shape:
        return {"comparable": False, "reason": f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"}
    a64, b64 = a.detach().to("cpu", torch.float64), b.detach().to("cpu", torch.float64)
    diff = (a64 - b64).abs()
    max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
    return {"comparable": True, "bit_identical": bool(torch.equal(a, b.to(a.device))),
            "max_abs_diff": max_abs_diff}


# ──────────────────────────────────────────────────────────────────────────────
# One leg: (shape, device, compile_mode) -> the full report dict
# ──────────────────────────────────────────────────────────────────────────────

def run_leg(shape: str, device: str, res: int, compile_mode: str, ticks: int,
            warmup: int) -> dict:
    out: dict = {"shape": shape, "device": device, "res": res, "compile_mode": compile_mode,
                "ticks": ticks, "warmup": warmup}
    try:
        if compile_mode == "auto":
            autotier.reset()          # a fresh verdict table — see the module docstring
        fx = Fixture(shape, res, device)
        vram_before = _vram_snapshot(device)

        # -- adoption tracking (auto only): the (fp, autotier key) for every stage this shape
        # cooks, and the first tick index at which EVERY one of them reads COMMITTED.
        keys = {}
        if compile_mode == "auto":
            device_type = torch.device(device).type
            idxs = fx.dirty_range() if shape == "host_tick_exact" else range(fx.n)
            for j in idxs:
                fp, key = fx.program_key(j, device_type)
                keys[j] = (fp, key)
        adopt_tick = {j: None for j in keys}

        # -- tick 0: first-cook, uncounted toward steady stats.
        _, first_ms, _ = _timed_tick(fx, 0, compile_mode)
        out["first_cook_ms"] = round(first_ms, 4)

        # -- warm-up (uncounted).
        for i in range(1, 1 + warmup):
            _timed_tick(fx, i, compile_mode)
            for j, (_fp, key) in keys.items():
                if adopt_tick[j] is None and autotier.verdict(key) == autotier.COMMITTED:
                    adopt_tick[j] = i

        # -- steady ticks (counted).
        wall_ms, thread_ms, last_out = [], [], None
        base_i = 1 + warmup
        for k in range(ticks):
            i = base_i + k
            last_out, w, t = _timed_tick(fx, i, compile_mode)
            wall_ms.append(w)
            thread_ms.append(t)
            for j, (_fp, key) in keys.items():
                if adopt_tick[j] is None and autotier.verdict(key) == autotier.COMMITTED:
                    adopt_tick[j] = i

        out["wall_ms"] = _stats(wall_ms)
        out["thread_ms"] = _stats(thread_ms)
        if keys:
            out["adopt_tick_per_stage"] = {fx.names[j]: adopt_tick[j] for j in keys}
            never = [fx.names[j] for j in keys if adopt_tick[j] is None]
            out["adopted_by_tick"] = (max(v for v in adopt_tick.values() if v is not None)
                                      if not never else None)
            out["never_adopted"] = never or None
            if never:
                # WHY it never adopted, for the stages that did not: still MEASURING/COMPILING
                # (the background compile has not finished within this run's wall-clock — a
                # cold Inductor compile commonly takes seconds while a tiny cook here can take
                # well under a millisecond, so a tick-bounded run can end long before the
                # compile does) vs REJECTED (it finished and lost to the interpreter/codegen
                # baseline) vs still MEASURING (never even submitted — `_MEASURE_COOKS` samples
                # not yet reached, which should not happen once `ticks + warmup > 3`).
                out["never_adopted_state"] = {
                    fx.names[j]: autotier.verdict(keys[j][1]) for j in keys if adopt_tick[j] is None}

        vram_after = _vram_snapshot(device)
        out["vram"] = _vram_delta(vram_before, vram_after)
        out["disk"] = _cg_and_inductor_bytes()
        out["_output"] = last_out
        out["_fixture"] = fx
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


# ──────────────────────────────────────────────────────────────────────────────
# COMPILE-M3: one fused-region leg — (playback_frames, device, res, fused compile_mode)
# ──────────────────────────────────────────────────────────────────────────────

_AUTOTIER_VERDICT_REASON = {
    autotier.COMMITTED: ("the fused program's compiled trial beat the codegen/interpreter "
                        "median by the commit ratio (autotier._COMMIT_RATIO)"),
    autotier.REJECTED: ("the fused program's compiled trial did NOT beat the codegen median "
                        "(or the trial/compile itself failed) — routed to codegen/interpreter"),
    autotier.MEASURING: ("still sampling the codegen baseline; not yet eligible to submit a "
                        "background compile within this run's tick budget"),
    autotier.COMPILING: ("a background compile was submitted but had not resolved (ready or "
                        "failed) within this run's wall time"),
    autotier.TRIAL: ("a compiled artifact became ready but was not re-cooked to a verdict "
                    "within this run"),
}


def run_fused_leg(device: str, res: int, compile_mode: str, ticks: int, warmup: int,
                  baseline_first_ms: float | None = None) -> dict:
    """One (playback_frames, device, res, fused compile_mode) leg — the fused-region twin
    of `run_leg`. `compile_mode="none"` is (ii) in the ask (fusion alone, no compile tier);
    `"auto"` and `"torch_compile"` are the two levers `select_tier` exposes for a fused
    chain (ENG-... `select_tier`'s `fused_fp_present` branch) — "auto" measures-then-trials
    exactly as the unfused path does, keyed by the ONE fused fingerprint instead of ten
    per-stage ones; "torch_compile" is the forced tier, skipping the measure/trial loop, so
    its own compile cost and adoption are visible even on a shape too small to win the
    measured trial. `baseline_first_ms` (the "none"/unfused leg's own first_cook_ms) lets
    "torch_compile"'s leg report a compile-time ESTIMATE (see below) — never asserted as
    exact, since the two first ticks pay different codegen-setup costs too."""
    shape = "playback_frames"
    out: dict = {"shape": shape, "device": device, "res": res,
                "compile_mode": f"fused:{compile_mode}", "ticks": ticks, "warmup": warmup,
                "fused": True}
    try:
        if compile_mode == "auto":
            autotier.reset()
        if compile_mode == "torch_compile":
            tex_compiled.clear_compiled_cache()
        fx = Fixture(shape, res, device)
        vram_before = _vram_snapshot(device)
        device_type = torch.device(device).type

        spec0, term_code0, bind0 = fx.build_fused_spec()
        fused_fp = tex_fusion.fused_fingerprint(spec0, term_code0, dict(bind0),
                                                infer_binding_type)
        out["fused_fp"] = fused_fp
        autotier_key = (autotier.make_key(fused_fp, device_type, "fp32", (1, res, res))
                        if fused_fp else None)
        compiled_cache_key = (fused_fp, device_type, "fp32") if fused_fp else None

        def _adopted() -> bool:
            if compile_mode == "auto":
                return autotier_key is not None and autotier.verdict(autotier_key) == \
                    autotier.COMMITTED
            if compile_mode == "torch_compile":
                return (compiled_cache_key is not None
                       and compiled_cache_key in tex_compiled._compiled_cache)
            return False   # "none" never adopts anything — there is no compile tier

        adopt_tick = None
        submitted_tick = None   # first tick autotier leaves MEASURING (auto only; a proxy
                                # for "compile submitted", per COMPILE-M2's own admission
                                # that a clean compile-wall-time number isn't resolvable
                                # from outside autotier's state machine)

        # -- tick 0: first-cook, uncounted toward steady stats.
        _sync(device)
        t0 = time.perf_counter()
        fx.fused_tick(0, compile_mode)  # first-cook, timed above; output unused (see steady loop)
        _sync(device)
        out["first_cook_ms"] = round((time.perf_counter() - t0) * 1000.0, 4)
        if compile_mode == "auto" and autotier_key is not None and \
                autotier.verdict(autotier_key) != autotier.MEASURING:
            submitted_tick = 0
        if _adopted():
            adopt_tick = 0

        # -- warm-up (uncounted).
        for i in range(1, 1 + warmup):
            fx.fused_tick(i, compile_mode)
            if submitted_tick is None and compile_mode == "auto" and autotier_key is not None \
                    and autotier.verdict(autotier_key) != autotier.MEASURING:
                submitted_tick = i
            if adopt_tick is None and _adopted():
                adopt_tick = i

        # -- steady ticks (counted), split into before/after adoption (the ask's own
        # before/after p50/p95/p99 request) — "after" stays empty for every leg that never
        # adopts, which every "none" leg and most "auto"/"torch_compile" legs at these
        # resolutions do (COMPILE-M2/M2b's own finding, unfused).
        wall_ms, before_ms, after_ms, last_out = [], [], [], None
        base_i = 1 + warmup
        for k in range(ticks):
            i = base_i + k
            _sync(device)
            t0 = time.perf_counter()
            last_out = fx.fused_tick(i, compile_mode)
            _sync(device)
            w = (time.perf_counter() - t0) * 1000.0
            wall_ms.append(w)
            if submitted_tick is None and compile_mode == "auto" and autotier_key is not None \
                    and autotier.verdict(autotier_key) != autotier.MEASURING:
                submitted_tick = i
            if adopt_tick is None and _adopted():
                adopt_tick = i
            (after_ms if adopt_tick is not None else before_ms).append(w)

        out["wall_ms"] = _stats(wall_ms)
        out["wall_ms_before_adoption"] = _stats(before_ms)
        out["wall_ms_after_adoption"] = _stats(after_ms)
        out["adopted_by_tick"] = adopt_tick
        out["submitted_by_tick"] = submitted_tick

        if compile_mode == "none":
            out["verdict"] = "n/a"
            out["verdict_reason"] = ("no compile tier is in play; this leg IS the fusion-"
                                     "only comparison point for (ii) vs (i)/(iii)")
        elif compile_mode == "auto":
            v = autotier.verdict(autotier_key) if autotier_key is not None else None
            out["verdict"] = v if v is not None else "unkeyable"
            out["verdict_reason"] = _AUTOTIER_VERDICT_REASON.get(
                v, "fused_fingerprint() could not assemble a key for this spec")
            # A rough, clearly-labelled ESTIMATE only: the wall-clock span between the tick
            # autotier left MEASURING and the tick it reached COMMITTED, at this run's own
            # measured mean tick cost. Real compile latency is not separately timestamped by
            # autotier/compiled.py (COMPILE-M2b's own gap, unfused) — a tighter number needs
            # a wrapper that hooks `_submit_bg_compile`/`_bg_status` directly, not done here.
            if adopt_tick is not None and submitted_tick is not None and wall_ms:
                out["compile_wall_s_estimate"] = round(
                    (adopt_tick - submitted_tick) * (statistics.fmean(wall_ms) / 1000.0), 4)
        else:  # torch_compile — forced, no measure/trial loop
            if compiled_cache_key is not None and compiled_cache_key in tex_compiled._compiled_cache:
                out["verdict"] = "compiled"
                out["verdict_reason"] = "the forced tier's compiled artifact is cached and serving"
                if baseline_first_ms is not None:
                    # First-tick delta vs the fused/"none" leg's own first tick — both pay the
                    # same splice+codegen setup, so the delta is dominated by the compile
                    # itself. Labelled an ESTIMATE, not a measured compile-only span.
                    out["compile_time_ms_estimate"] = round(
                        out["first_cook_ms"] - baseline_first_ms, 4)
            elif fused_fp in tex_compiled._compile_blacklist:
                out["verdict"] = "blacklisted"
                out["verdict_reason"] = ("torch.compile crashed on this program earlier in "
                                        "this process and the fingerprint is now blacklisted "
                                        "— every tick ran the interpreter/codegen fallback")
            else:
                out["verdict"] = "fallback"
                out["verdict_reason"] = ("never entered _compiled_cache — either below the "
                                        "op-count/spatial gates in execute_compiled(), or "
                                        "every attempt failed without being blacklisted")

        vram_after = _vram_snapshot(device)
        out["vram"] = _vram_delta(vram_before, vram_after)
        out["disk"] = _cg_and_inductor_bytes()
        out["_output"] = last_out
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def _print_fused_leg(leg: dict):
    if "error" in leg:
        print(f"  [fused {leg['shape']}/{leg['device']}/{leg['compile_mode']}] "
             f"ERROR: {leg['error']}")
        return
    w = leg["wall_ms"]
    print(f"  [fused {leg['shape']:16s} {leg['device']:4s} {leg['compile_mode']:18s}] "
         f"first={leg['first_cook_ms']:8.3f} ms  "
         f"p50={w.get('p50', float('nan')):7.3f}  p95={w.get('p95', float('nan')):7.3f}  "
         f"p99={w.get('p99', float('nan')):7.3f} ms  n={w.get('n', 0)}  "
         f"adopted_by_tick={leg.get('adopted_by_tick')}  "
         f"verdict={leg.get('verdict')} ({leg.get('verdict_reason')})")


# ──────────────────────────────────────────────────────────────────────────────
# (c) interference: an unrelated interactive workload while auto compiles in the background
# ──────────────────────────────────────────────────────────────────────────────

def run_interference(shape: str, device: str, res: int, cadence_ms: float,
                     n_ticks: int) -> dict:
    """Drive `shape` under `compile_mode="auto"` until a background compile is submitted for
    at least one of its stages, then race an UNRELATED tiny interactive workload (a fresh
    one-stage cook, `compile_mode="none"`, never itself compiled) at a fixed cadence while that
    compile is in flight, and again with nothing in flight — p95/p99 deltas between the two,
    plus a coarse thread-blocked-time signal (see the module docstring)."""
    result: dict = {"shape": shape, "device": device, "res": res, "cadence_ms": cadence_ms,
                    "n_ticks": n_ticks}
    try:
        autotier.reset()
        fx = Fixture(shape, res, device)
        device_type = torch.device(device).type
        idxs = list(fx.dirty_range() if shape == "host_tick_exact" else range(fx.n))
        cache_keys = []
        for j in idxs:
            fp, key = fx.program_key(j, device_type)
            cache_keys.append((fp, device_type, "fp32"))

        def _unrelated_workload(i: int):
            """A tiny, DIFFERENT one-stage cook (never `"auto"`, never compiled) — the
            interactive edit an unrelated node would be paying for on the same process while
            this program's background compile runs."""
            code = "@OUT = vec4(@IN.rgb * $k, 1.0);"
            img = fx.src
            return tex_engine.cook(code, {"IN": img, "k": 0.5 + (i % 97) * 1e-4},
                                   device_mode=device, precision="fp32",
                                   compile_mode="none", cancel=None,
                                   time_context={"frame": float(i), "fps": 24.0, "time": 0.0})

        def _cadence_run(start_i: int, stop_when=None):
            wall_ms, thread_ms = [], []
            i = start_i
            for _ in range(n_ticks):
                t_cycle0 = time.perf_counter()
                _sync(device)
                t0 = time.perf_counter()
                tt0 = time.thread_time()
                _unrelated_workload(i)
                _sync(device)
                wall_ms.append((time.perf_counter() - t0) * 1000.0)
                thread_ms.append((time.thread_time() - tt0) * 1000.0)
                i += 1
                if stop_when is not None and stop_when():
                    break
                remaining = (cadence_ms / 1000.0) - (time.perf_counter() - t_cycle0)
                if remaining > 0:
                    time.sleep(remaining)
            return wall_ms, thread_ms, i

        # 1) BASELINE — nothing in flight.
        base_wall, base_thread, next_i = _cadence_run(0)

        # 2) Drive the fixture under "auto" until a background compile is submitted.
        submitted = False
        i = next_i
        max_prime_ticks = max(50, 4 * autotier._MEASURE_COOKS)
        for _ in range(max_prime_ticks):
            fx.tick(i, "auto")
            i += 1
            if any(tex_compiled._bg_futures.get(ck) is not None and
                  not tex_compiled._bg_futures[ck].done() for ck in cache_keys):
                submitted = True
                break
            if all(autotier.verdict(k) in (autotier.COMMITTED, autotier.REJECTED)
                  for _fp, k in (fx.program_key(j, device_type) for j in idxs)):
                break            # settled without us ever observing an in-flight compile
        result["compile_observed_in_flight"] = submitted

        # 3) DURING — race the unrelated workload while that compile is (still) in flight.
        def _still_compiling():
            return not any(tex_compiled._bg_futures.get(ck) is not None and
                           not tex_compiled._bg_futures[ck].done() for ck in cache_keys)

        during_wall, during_thread, _ = _cadence_run(i, stop_when=_still_compiling if submitted
                                                      else None)

        result["baseline"] = {"wall_ms": _stats(base_wall), "thread_ms": _stats(base_thread)}
        result["during_compile"] = {"wall_ms": _stats(during_wall),
                                    "thread_ms": _stats(during_thread)}
        if base_wall and during_wall:
            b, d = _stats(base_wall), _stats(during_wall)
            result["p95_delta_ms"] = round(d["p95"] - b["p95"], 4)
            result["p99_delta_ms"] = round(d["p99"] - b["p99"], 4)
            # thread_time / wall_time: near 1.0 means the thread was actually running for
            # (almost) the whole wall-clock span; a materially LOWER ratio during the compile
            # than at baseline is consistent with time spent blocked (GIL or another lock)
            # rather than computing — named, never asserted, since a busy compile worker
            # legitimately steals wall-clock without ever blocking this thread on the GIL.
            def _ratio(st_wall, st_thread):
                return (st_thread["mean"] / st_wall["mean"]) if st_wall["mean"] else float("nan")
            base_ratio = _ratio(b, _stats(base_thread))
            during_ratio = _ratio(d, _stats(during_thread))
            result["thread_wall_ratio"] = {"baseline": round(base_ratio, 4),
                                           "during_compile": round(during_ratio, 4)}
            contention_suspected = (submitted and during_ratio < base_ratio * 0.7
                                    and result["p95_delta_ms"] > 0)
            result["gil_contention_suspected"] = bool(contention_suspected)
            if contention_suspected:
                result["finding"] = ("possible GIL/resource contention: the unrelated tick's "
                                     "thread-time/wall-time ratio dropped from "
                                     f"{base_ratio:.2f} to {during_ratio:.2f} while auto's "
                                     "background compile was in flight, and p95 rose by "
                                     f"{result['p95_delta_ms']:.3f} ms")
        return result
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def environment() -> dict:
    from TEX_Wrangle import __version__ as tex_version
    return {"tex_version": tex_version, "torch": torch.__version__,
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "tex_cache_dir": _CACHE_DIR_AT_START, "tex_cache_dir_minted": _CACHE_WAS_MINTED,
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR")}


def _print_leg(leg: dict):
    if "error" in leg:
        print(f"  [{leg['shape']}/{leg['device']}/{leg['compile_mode']}] ERROR: {leg['error']}")
        return
    w = leg["wall_ms"]
    print(f"  [{leg['shape']:16s} {leg['device']:4s} {leg['compile_mode']:5s}] "
          f"first={leg['first_cook_ms']:8.3f} ms  "
          f"p50={w.get('p50', float('nan')):7.3f}  p95={w.get('p95', float('nan')):7.3f}  "
          f"p99={w.get('p99', float('nan')):7.3f} ms  n={w.get('n', 0)}"
          + (f"  adopted_by_tick={leg.get('adopted_by_tick')}"
             if leg.get("compile_mode") == "auto" else ""))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="COMPILE-M1: compile_mode=\"none\" vs \"auto\"")
    p.add_argument("--shape", choices=SHAPES + ("both",), default="both")
    p.add_argument("--device", choices=("cpu", "cuda", "both"), default="both")
    p.add_argument("--res", type=int, default=512)
    p.add_argument("--ticks", type=int, default=220, help=">= 200 for a real p95/p99 (default)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--interference-ticks", type=int, default=60)
    p.add_argument("--interference-cadence-ms", type=float, default=16.0)
    p.add_argument("--no-interference", action="store_true")
    p.add_argument("--fused", action="store_true",
                   help="COMPILE-M3: also cook the whole 10-stage playback_frames chain as "
                        "ONE fused region, at each of none/auto/torch_compile, and compare "
                        "against the unfused/none baseline (i)/(ii)/(iii) of the ask. Forces "
                        "--shape playback_frames (fusing host_tick_exact's single dirty "
                        "stage every tick would defeat that shape's own point).")
    p.add_argument("--smoke", action="store_true",
                   help="a handful of ticks (proves the script works; not a timing claim)")
    p.add_argument("--save", metavar="PATH")
    a = p.parse_args(argv)
    if a.fused and a.shape not in ("playback_frames", "both"):
        print(f"--fused only runs on playback_frames; overriding --shape {a.shape!r}")
    if a.fused:
        a.shape = "playback_frames"

    if a.smoke:
        a.ticks, a.warmup = 6, 2
        a.interference_ticks = 6

    shapes = SHAPES if a.shape == "both" else (a.shape,)
    have_cuda = torch.cuda.is_available()
    devices = ["cpu", "cuda"] if a.device == "both" else [a.device]
    if "cuda" in devices and not have_cuda:
        if a.device == "cuda":
            print("no CUDA device — nothing to measure")
            return 2
        devices.remove("cuda")

    env = environment()
    print(f"{'=' * 78}\ncompile_modes_bench  res={a.res}  ticks={a.ticks}  warmup={a.warmup}"
          f"  smoke={a.smoke}\nTEX {env['tex_version']}  torch {env['torch']}  "
          f"device(s)={devices}\ncache_dir={env['tex_cache_dir']}"
          f" (minted fresh: {env['tex_cache_dir_minted']})\n{'=' * 78}")

    legs, comparisons, interference = [], [], []
    fused_legs, fused_comparisons = [], []
    for shape in shapes:
        for device in devices:
            outs = {}
            for mode in MODES:
                leg = run_leg(shape, device, a.res, mode, a.ticks, a.warmup)
                outs[mode] = leg.pop("_output", None)
                leg.pop("_fixture", None)
                _print_leg(leg)
                legs.append(leg)
            cmp = _compare_outputs(outs.get("none"), outs.get("auto"))
            cmp.update({"shape": shape, "device": device})
            comparisons.append(cmp)
            if not cmp.get("comparable"):
                print(f"  [{shape}/{device}] outputs NOT comparable: {cmp.get('reason')}")
            elif cmp["bit_identical"]:
                print(f"  [{shape}/{device}] auto == none: BIT-IDENTICAL")
            else:
                print(f"  [{shape}/{device}] auto vs none: max abs diff = "
                      f"{cmp['max_abs_diff']:.3e}")

            # COMPILE-M3: (i) is `outs["none"]` above (unfused, compile_mode="none");
            # (ii)/(iii) are the fused legs below, at "none"/"auto"/"torch_compile".
            if a.fused and shape == "playback_frames":
                print(f"  -- fused-region legs ({device}, res={a.res}) --")
                fused_outs, first_none_ms = {}, None
                for fmode in FUSED_MODES:
                    fleg = run_fused_leg(device, a.res, fmode, a.ticks, a.warmup,
                                         baseline_first_ms=first_none_ms)
                    fused_outs[fmode] = fleg.pop("_output", None)
                    if fmode == "none":
                        first_none_ms = fleg.get("first_cook_ms")
                    _print_fused_leg(fleg)
                    fused_legs.append(fleg)
                for fmode in FUSED_MODES:
                    fcmp = _compare_outputs(outs.get("none"), fused_outs.get(fmode))
                    fcmp.update({"shape": shape, "device": device,
                                "compare": f"unfused:none vs fused:{fmode}"})
                    fused_comparisons.append(fcmp)
                    if not fcmp.get("comparable"):
                        print(f"  [{fcmp['compare']}] NOT comparable: {fcmp.get('reason')}")
                    elif fcmp["bit_identical"]:
                        print(f"  [{fcmp['compare']}] BIT-IDENTICAL")
                    else:
                        print(f"  [{fcmp['compare']}] max abs diff = "
                              f"{fcmp['max_abs_diff']:.3e}")
                gc.collect()

            if not a.no_interference:
                inter = run_interference(shape, device, a.res, a.interference_cadence_ms,
                                         a.interference_ticks)
                interference.append(inter)
                if "error" in inter:
                    print(f"  [{shape}/{device}] interference ERROR: {inter['error']}")
                elif inter.get("finding"):
                    print(f"  [{shape}/{device}] INTERFERENCE FINDING: {inter['finding']}")
                else:
                    print(f"  [{shape}/{device}] interference: compile_in_flight_observed="
                          f"{inter.get('compile_observed_in_flight')}  "
                          f"p95_delta={inter.get('p95_delta_ms')} ms  "
                          f"p99_delta={inter.get('p99_delta_ms')} ms")
            gc.collect()

    payload = {"env": env, "legs": legs, "comparisons": comparisons,
              "interference": interference, "fused_legs": fused_legs,
              "fused_comparisons": fused_comparisons}
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)) or ".", exist_ok=True)
        with open(a.save, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"\nsaved -> {a.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
