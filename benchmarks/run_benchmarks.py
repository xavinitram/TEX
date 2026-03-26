#!/usr/bin/env python3
"""
TEX Benchmark Suite
===================
Reproducible performance benchmarks for the TEX Wrangle ComfyUI node.

Measures compilation and interpreter performance across multiple resolutions,
batch sizes, and program complexities. Results are saved as JSON for cross-
system and cross-version comparisons.

Quick start
-----------
    cd TEX_Wrangle
    python benchmarks/run_benchmarks.py                          # Standard run
    python benchmarks/run_benchmarks.py --full                   # Full matrix
    python benchmarks/run_benchmarks.py --compare results/v0.4.0.json

All flags
---------
    --full              All resolutions (256-4096) x batch (1,4) x cache (warm,cold)
    --quick             512x512 B=1 warm only (fast sanity check, ~2 min)
    --resolution N      Single resolution (default: 256,512,1024)
    --batch N           Single batch size   (default: 1)
    --cold              Include cold-cache (recompile) benchmarks
    --device cpu|cuda   Force device (default: cpu)
    --precision P       fp32 (default), fp16, bf16
    --save PATH         Save results JSON (default: auto-named in results/)
    --compare PATH      Compare against a baseline JSON
    --examples-only     Skip synthetic micro-benchmarks
    --synthetic-only    Skip real-world example programs
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path

# -- Path setup --------------------------------------------------------
_bench_dir = Path(__file__).resolve().parent
_pkg_dir = _bench_dir.parent
_custom_nodes_dir = _pkg_dir.parent
sys.path.insert(0, str(_custom_nodes_dir))

import torch
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker, TEXType

# Optional modules — not present in all TEX versions
try:
    from TEX_Wrangle.tex_compiler.optimizer import optimize
except ImportError:
    optimize = None  # type: ignore[assignment]

try:
    from TEX_Wrangle.tex_runtime.interpreter import Interpreter, _collect_identifiers
except ImportError:
    from TEX_Wrangle.tex_runtime.interpreter import Interpreter
    _collect_identifiers = None  # type: ignore[assignment]

# -- Constants ---------------------------------------------------------

RESOLUTIONS_STANDARD = [256, 512, 1024]
RESOLUTIONS_FULL = [256, 512, 1024, 2048, 4096]
BATCH_SIZES_STANDARD = [1]
BATCH_SIZES_FULL = [1, 4]

# Adaptive measurement: keep running until coefficient of variation is below
# target, between MIN and MAX runs.  WARMUP runs are discarded.
MIN_RUNS = 10
MAX_RUNS = 50
WARMUP_RUNS = 5
TARGET_CV = 0.05  # 5%

EXAMPLES_DIR = _pkg_dir / "examples"
RESULTS_DIR = _bench_dir / "results"


# -- Data structures ---------------------------------------------------

@dataclass
class BenchmarkProgram:
    """A TEX program to benchmark."""
    name: str
    code: str
    category: str  # "synthetic" or "example"
    needs_image: bool = False
    needs_ref: bool = False
    needs_frames: bool = False
    needs_latent_ab: bool = False
    needs_latent: bool = False
    needs_base_overlay_blend: bool = False
    needs_text: bool = False
    needs_mask: bool = False
    param_defaults: dict = field(default_factory=dict)
    string_only: bool = False
    multi_output: bool = False


@dataclass
class BenchResult:
    """Result of a single benchmark measurement."""
    program: str
    resolution: int
    batch: int
    cache_mode: str        # "cold" or "warm"
    compile_ms: float      # median compile time
    interp_ms: float       # median interpreter time
    total_ms: float        # median total E2E time
    cv_percent: float      # coefficient of variation %
    p5_ms: float           # 5th percentile total
    p95_ms: float          # 95th percentile total
    mem_mb: float          # peak memory delta (MB)
    num_runs: int          # measurement runs (excluding warmup)


# -- Synthetic micro-benchmarks ----------------------------------------
#
# These isolate specific engine capabilities: dispatch overhead, math
# throughput, branching, loop unrolling, stdlib calls, vector ops, etc.

SYNTHETIC_PROGRAMS = [
    # -- Dispatch / pass-through --
    BenchmarkProgram("passthrough",
        "@OUT = @A;",
        "synthetic", needs_image=True),

    # -- Arithmetic --
    BenchmarkProgram("math_chain",
        "@OUT = vec4(sin(u) * cos(v) + 0.5);",
        "synthetic", needs_ref=True),

    BenchmarkProgram("function_heavy", """\
float a = clamp(sin(u * 3.14159), 0.0, 1.0);
float b = smoothstep(0.0, 1.0, v);
float c = lerp(a, b, 0.5);
float d = pow(c, 2.2);
float e = sqrt(abs(d - 0.5));
float f = fract(atan2(v - 0.5, u - 0.5) / 6.28318);
float g = floor(f * 10.0) / 10.0;
float h = exp(-e * 3.0);
float k = sign(u - 0.5) * 0.5 + 0.5;
float m = mod(a + b + c, 1.0);
@OUT = vec4(h, g, k, 1.0);""",
        "synthetic", needs_ref=True),

    # -- Branching --
    BenchmarkProgram("branch_simple",
        "if (u > 0.5) { @OUT = @A; } else { @OUT = vec3(1.0) - @A; }",
        "synthetic", needs_image=True),

    BenchmarkProgram("branch_nested", """\
vec4 c = @A;
if (u > 0.5) {
    if (v > 0.5) { c = vec4(1.0, 0.0, 0.0, 1.0); }
    else          { c = vec4(0.0, 1.0, 0.0, 1.0); }
} else {
    if (v > 0.5) { c = vec4(0.0, 0.0, 1.0, 1.0); }
    else          { c = vec4(0.0); }
}
@OUT = c;""",
        "synthetic", needs_image=True),

    # -- Loops --
    BenchmarkProgram("for_10", """\
float s = 0.0;
for (int i = 0; i < 10; i++) { s = s + sin(u * float(i)); }
@OUT = vec4(s * 0.1);""",
        "synthetic", needs_ref=True),

    BenchmarkProgram("for_100", """\
float s = 0.0;
for (int i = 0; i < 100; i++) { s = s + sin(u * float(i) * 0.01); }
@OUT = vec4(s * 0.01);""",
        "synthetic", needs_ref=True),

    BenchmarkProgram("while_loop", """\
float x = 1.0; int n = 0;
while (x > 0.01) { x = x * 0.9; n++; }
@OUT = vec4(float(n) * 0.01);""",
        "synthetic", needs_ref=True),

    # -- Vectors --
    BenchmarkProgram("vector_ops", """\
vec4 a = vec4(u, v, u * v, 1.0);
vec4 b = vec4(1.0 - u, 1.0 - v, 0.5, 1.0);
vec4 c = a * 0.7 + b * 0.3;
vec3 d = c.rgb * 2.0 - vec3(0.5, 0.5, 0.5);
@OUT = vec4(clamp(d.r, 0.0, 1.0), clamp(d.g, 0.0, 1.0), clamp(d.b, 0.0, 1.0), 1.0);""",
        "synthetic", needs_ref=True),

    # -- Arrays --
    BenchmarkProgram("array_ops", """\
float arr[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
arr = sort(arr);
float s = arr_sum(arr);
float mn = arr_min(arr);
float mx = arr_max(arr);
@OUT = vec4(s / 15.0, mn / 5.0, mx / 5.0, 1.0);""",
        "synthetic", needs_ref=True),

    # -- String ops (scalar, no spatial) --
    BenchmarkProgram("string_ops", """\
string s = "hello world";
string u_str = upper(s);
float length = len(s);
@OUT = u_str;""",
        "synthetic", string_only=True),

    # -- Noise --
    BenchmarkProgram("noise_perlin",
        "float n = perlin(u * 10.0, v * 10.0); @OUT = vec4(n * 0.5 + 0.5);",
        "synthetic", needs_ref=True),

    BenchmarkProgram("noise_fbm", """\
vec4 _ref = @ref;
float n = fbm(u * 6.0, v * 6.0, 6);
@OUT = vec4(n * 0.5 + 0.5);""",
        "synthetic", needs_ref=True),

    # -- Sampling --
    BenchmarkProgram("sample_bilinear",
        "@OUT = sample(@A, u + sin(v * 6.28) * 0.01, v);",
        "synthetic", needs_image=True),

    BenchmarkProgram("sample_lanczos",
        "@OUT = sample_lanczos(@A, u + sin(v * 6.28) * 0.01, v);",
        "synthetic", needs_image=True),

    # -- User functions --
    BenchmarkProgram("user_func_simple", """\
float square(float x) { return x * x; }
float a = square(u);
float b = square(v);
float c = square(a + b);
@OUT = vec4(c);""",
        "synthetic", needs_ref=True),

    BenchmarkProgram("user_func_heavy", """\
float remap(float val, float lo, float hi) {
    return clamp((val - lo) / max(hi - lo, 0.001), 0.0, 1.0);
}
vec3 color_ramp(float t) {
    float r = smoothstep(0.0, 0.5, t);
    float g = smoothstep(0.25, 0.75, t);
    float b = smoothstep(0.5, 1.0, t);
    return vec3(r, g, b);
}
float pattern(float x, float y) {
    return sin(x * 6.28) * cos(y * 6.28) * 0.5 + 0.5;
}
float p = pattern(u * 3.0, v * 3.0);
float r = remap(p, 0.2, 0.8);
@OUT = vec4(color_ramp(r), 1.0);""",
        "synthetic", needs_ref=True),

    BenchmarkProgram("user_func_loop", """\
float gaussian_weight(int dx, int dy, float sigma) {
    float d2 = float(dx * dx + dy * dy);
    return exp(-d2 / (2.0 * sigma * sigma));
}
vec3 weighted_blur(float sigma) {
    vec3 total = vec3(0.0);
    float wsum = 0.0;
    for (int dy = -2; dy <= 2; dy++) {
        for (int dx = -2; dx <= 2; dx++) {
            float w = gaussian_weight(dx, dy, sigma);
            total += fetch(@A, ix + dx, iy + dy).rgb * w;
            wsum += w;
        }
    }
    return total / wsum;
}
@OUT = weighted_blur(1.5);""",
        "synthetic", needs_image=True),

    # -- Binding access --
    BenchmarkProgram("binding_fetch", """\
vec4 tl = @A[ix - 1, iy - 1];
vec4 tr = @A[ix + 1, iy - 1];
vec4 bl = @A[ix - 1, iy + 1];
vec4 br = @A[ix + 1, iy + 1];
@OUT = (tl + tr + bl + br) * 0.25;""",
        "synthetic", needs_image=True),

    BenchmarkProgram("binding_sample", """\
float ofs = 0.005;
vec4 c  = @A(u, v);
vec4 l  = @A(u - ofs, v);
vec4 r2 = @A(u + ofs, v);
vec4 t  = @A(u, v - ofs);
vec4 b  = @A(u, v + ofs);
@OUT = (c * 4.0 + l + r2 + t + b) / 8.0;""",
        "synthetic", needs_image=True),

    BenchmarkProgram("binding_frame_access", """\
vec4 prev = @frames[ix, iy, fi - 1];
vec4 curr = @frames[ix, iy, fi];
vec4 next = @frames[ix, iy, fi + 1];
@OUT = prev * 0.25 + curr * 0.5 + next * 0.25;""",
        "synthetic", needs_frames=True),

    # -- Combined: user functions + binding access --
    BenchmarkProgram("func_and_binding", """\
float edge_strength(float px, float py) {
    float c = luma(@A[int(px), int(py)]);
    float l = luma(@A[int(px) - 1, int(py)]);
    float r = luma(@A[int(px) + 1, int(py)]);
    float t = luma(@A[int(px), int(py) - 1]);
    float b = luma(@A[int(px), int(py) + 1]);
    return abs(l - r) + abs(t - b);
}
float e = edge_strength(ix, iy);
vec3 color = @A(u, v).rgb;
@OUT = lerp(color, vec3(e), clamp(e * 2.0, 0.0, 1.0));""",
        "synthetic", needs_image=True),

    # -- Matrix --
    BenchmarkProgram("matrix_heavy", """\
mat3 rot = mat3(
    cos(u * 3.14), -sin(u * 3.14), 0.0,
    sin(u * 3.14),  cos(u * 3.14), 0.0,
    0.0,            0.0,            1.0
);
vec3 p = vec3(u - 0.5, v - 0.5, 0.0);
vec3 rp = rot * p;
float d = length(rp);
@OUT = vec4(smoothstep(0.3, 0.0, d));""",
        "synthetic", needs_ref=True),

    # -- Ternary --
    BenchmarkProgram("ternary_chain", """\
float t = u * 4.0;
float r = t < 1.0 ? t : t < 2.0 ? 2.0 - t : t < 3.0 ? t - 2.0 : 4.0 - t;
float g = v * 4.0;
float gv = g < 1.0 ? g : g < 2.0 ? 2.0 - g : g < 3.0 ? g - 2.0 : 4.0 - g;
@OUT = vec4(r, gv, (r + gv) * 0.5, 1.0);""",
        "synthetic", needs_ref=True),

    # -- Multi-output --
    BenchmarkProgram("multi_output", """\
float lum = luma(@A);
vec3 color = @A.rgb;
@color_graded = pow(color, vec3(1.0 / 2.2));
m@luminance_mask = lum;
f@avg_brightness = lum;""",
        "synthetic", needs_image=True, multi_output=True),
]


# -- Example loader ----------------------------------------------------

def _detect_bindings(code: str) -> dict:
    """Detect what input bindings a TEX program references."""
    return {
        "needs_image": bool(re.search(r"@(?:image|A)\b", code)),
        "needs_ref": bool(re.search(r"@ref\b", code)),
        "needs_frames": bool(re.search(r"@frames\b", code)),
        "needs_latent_ab": bool(
            re.search(r"@latent_a\b", code) and re.search(r"@latent_b\b", code)
        ),
        "needs_latent": (
            bool(re.search(r"@latent\b", code))
            and not bool(re.search(r"@latent_[ab]\b", code))
        ),
        "needs_base_overlay_blend": bool(re.search(r"@base\b", code)),
        "needs_text": bool(re.search(r"@text\b", code)),
        "needs_mask": bool(re.search(r"@mask\b", code)),
        "string_only": (
            not any(re.search(p, code) for p in [
                r"@(?:image|A)\b", r"@ref\b", r"@frames\b",
                r"@latent", r"@base\b", r"@mask\b",
            ])
            and bool(re.search(r"string\b", code))
        ),
        "multi_output": (
            bool(re.search(r"@\w+\s*=", code))
            and not bool(re.search(r"@OUT\s*=", code))
        ),
    }


def _extract_param_defaults(code: str) -> dict:
    """Extract $param declarations and their default values."""
    defaults = {}
    for m in re.finditer(r"([fis])\$(\w+)\s*=\s*([^;]+);", code):
        hint, name, raw = m.groups()
        raw = raw.strip().strip("\"'")
        try:
            if hint == "f":
                defaults[name] = float(raw)
            elif hint == "i":
                defaults[name] = int(raw)
            else:
                defaults[name] = raw
        except ValueError:
            defaults[name] = 0.5 if hint == "f" else 1 if hint == "i" else raw
    return defaults


def load_example_programs() -> list[BenchmarkProgram]:
    """Load all .tex files from the examples/ directory."""
    programs = []
    if not EXAMPLES_DIR.exists():
        print(f"  Warning: {EXAMPLES_DIR} not found")
        return programs
    for f in sorted(EXAMPLES_DIR.glob("*.tex")):
        code = f.read_text(encoding="utf-8")
        bindings = _detect_bindings(code)
        params = _extract_param_defaults(code)
        programs.append(BenchmarkProgram(
            name=f"ex_{f.stem}", code=code, category="example",
            param_defaults=params, **bindings,
        ))
    return programs


# -- Input generation --------------------------------------------------

def _detect_channels(code: str) -> int:
    if re.search(r"vec4\s+\w+\[", code) and re.search(r"fetch\(", code):
        return 4
    if re.search(r"@(?:image|A)\.(?:a|w)\b", code):
        return 4
    return 3


def generate_bindings(prog: BenchmarkProgram, B: int, H: int, W: int,
                      device: str = "cpu") -> dict:
    """Create synthetic input tensors for a program."""
    b = {}
    C = _detect_channels(prog.code)
    if prog.needs_image:
        t = torch.rand(B, H, W, C, dtype=torch.float32, device=device)
        b["A"] = t; b["image"] = t
    if prog.needs_ref:
        b["ref"] = torch.rand(B, H, W, 4, dtype=torch.float32, device=device)
    if prog.needs_frames:
        b["frames"] = torch.rand(max(B, 4), H, W, 4, dtype=torch.float32, device=device)
    if prog.needs_latent_ab:
        lH, lW = max(1, H // 8), max(1, W // 8)
        b["latent_a"] = torch.rand(B, lH, lW, 4, dtype=torch.float32, device=device)
        b["latent_b"] = torch.rand(B, lH, lW, 4, dtype=torch.float32, device=device)
    if prog.needs_latent:
        lH, lW = max(1, H // 8), max(1, W // 8)
        b["latent"] = torch.rand(B, lH, lW, 4, dtype=torch.float32, device=device)
    if prog.needs_base_overlay_blend:
        b["base"] = torch.rand(B, H, W, 3, dtype=torch.float32, device=device)
        b["overlay"] = torch.rand(B, H, W, 3, dtype=torch.float32, device=device)
        b["blend"] = torch.rand(B, H, W, dtype=torch.float32, device=device) * 0.5 + 0.25
    if prog.needs_text:
        b["text"] = "  Hello World  "
    if prog.needs_mask:
        b["mask"] = torch.rand(B, H, W, dtype=torch.float32, device=device)
    for pname, pval in prog.param_defaults.items():
        b[pname] = pval
    return b


# -- Compilation / execution helpers -----------------------------------

def _infer_types(bindings: dict) -> dict[str, TEXType]:
    types = {}
    for name, val in bindings.items():
        if isinstance(val, str):
            types[name] = TEXType.STRING
        elif isinstance(val, torch.Tensor):
            if val.dim() == 4:
                c = val.shape[-1]
                types[name] = (TEXType.VEC4 if c == 4
                               else TEXType.VEC3 if c == 3
                               else TEXType.FLOAT)
            else:
                types[name] = TEXType.FLOAT
        elif isinstance(val, int):
            types[name] = TEXType.INT
        else:
            types[name] = TEXType.FLOAT
    return types


def compile_program(code: str, binding_types: dict):
    tokens = Lexer(code).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker(binding_types=binding_types)
    type_map = checker.check(program)
    if optimize is not None:
        program = optimize(program)
    used = _collect_identifiers(program) if _collect_identifiers else None
    return program, type_map, checker.assigned_bindings, used


_interp: Interpreter | None = None

def run_interpreter(program, bindings, type_map, device="cpu",
                    output_names=None, precision="fp32", used_builtins=None):
    global _interp
    if _interp is None:
        _interp = Interpreter()
    # Build kwargs compatible with both old and new interpreter versions
    import inspect
    sig = inspect.signature(_interp.execute)
    kwargs: dict = dict(device=device, output_names=output_names)
    if "precision" in sig.parameters:
        kwargs["precision"] = precision
    if "used_builtins" in sig.parameters:
        kwargs["used_builtins"] = used_builtins
    return _interp.execute(program, bindings, type_map, **kwargs)


# -- Adaptive measurement ---------------------------------------------

def _pct(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100.0
    f, c = math.floor(k), math.ceil(k)
    return data[int(k)] if f == c else data[f] * (c - k) + data[c] * (k - f)


def measure(prog: BenchmarkProgram, B: int, H: int, W: int,
            cache_mode: str = "warm", device: str = "cpu",
            precision: str = "fp32") -> BenchResult | None:
    """Measure a program with adaptive run count.  Returns None on error."""
    res = max(H, W)

    # Memory guard — skip if estimated input > 2 GB
    px = B * H * W
    est_mb = 0
    if prog.needs_image: est_mb += px * 4 * 4 / 1048576
    if prog.needs_ref:   est_mb += px * 4 * 4 / 1048576
    if prog.needs_frames: est_mb += max(B, 4) * H * W * 4 * 4 / 1048576
    if prog.needs_mask:  est_mb += px * 4 / 1048576
    if est_mb > 2048:
        return None

    try:
        bindings = generate_bindings(prog, B, H, W, device)
        btypes = _infer_types(bindings)

        program, type_map, assigned, used = compile_program(prog.code, btypes)
        output_names = (list(assigned.keys())
                        if assigned and "OUT" not in assigned else None)

        # Warmup
        for _ in range(WARMUP_RUNS):
            if cache_mode == "cold":
                program, type_map, assigned, used = compile_program(prog.code, btypes)
            run_interpreter(program, bindings, type_map, device,
                            output_names, precision, used)

        comp_t, interp_t, total_t = [], [], []
        n = 0
        while n < MAX_RUNS:
            gc.collect()
            tracemalloc.start()
            mem0 = tracemalloc.get_traced_memory()[1]

            t0 = time.perf_counter()
            if cache_mode == "cold":
                tc = time.perf_counter()
                program, type_map, assigned, used = compile_program(prog.code, btypes)
                if assigned and "OUT" not in assigned:
                    output_names = list(assigned.keys())
                comp_t.append((time.perf_counter() - tc) * 1000)
            else:
                comp_t.append(0.0)

            ti = time.perf_counter()
            run_interpreter(program, bindings, type_map, device,
                            output_names, precision, used)
            interp_t.append((time.perf_counter() - ti) * 1000)
            total_t.append((time.perf_counter() - t0) * 1000)

            mem_peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            n += 1

            if n >= MIN_RUNS:
                mu = statistics.mean(total_t)
                if mu > 0 and statistics.stdev(total_t) / mu < TARGET_CV:
                    break

        total_t.sort(); interp_t.sort(); comp_t.sort()
        mu = statistics.mean(total_t)
        cv = (statistics.stdev(total_t) / mu * 100) if mu > 0 and len(total_t) > 1 else 0.0

        return BenchResult(
            program=prog.name, resolution=res, batch=B, cache_mode=cache_mode,
            compile_ms=round(statistics.median(comp_t), 3),
            interp_ms=round(statistics.median(interp_t), 3),
            total_ms=round(statistics.median(total_t), 3),
            cv_percent=round(cv, 1),
            p5_ms=round(_pct(total_t, 5), 3),
            p95_ms=round(_pct(total_t, 95), 3),
            mem_mb=round(max(0, mem_peak - mem0) / 1048576, 2),
            num_runs=n,
        )
    except Exception as e:
        print(f"  SKIP  {prog.name}: {e}")
        return None


# -- System info -------------------------------------------------------

def system_info(device: str) -> dict:
    """Collect system metadata for the results file."""
    info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
    }
    try:
        info["cpu_name"] = platform.processor()
        import multiprocessing
        info["cpu_cores"] = multiprocessing.cpu_count()
    except Exception:
        pass
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        info["gpu_memory_gb"] = round(mem / 1073741824, 1)
    # TEX version
    try:
        ver_file = _pkg_dir / "__init__.py"
        for line in ver_file.read_text().splitlines():
            if "VERSION" in line and "=" in line:
                info["tex_version"] = line.split("=")[1].strip().strip("\"'")
                break
    except Exception:
        pass
    # Git commit
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_pkg_dir), stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(_pkg_dir), stderr=subprocess.DEVNULL,
        ).decode().strip())
    except Exception:
        pass
    return info


# -- Reporting ---------------------------------------------------------

def print_header(info: dict):
    print(f"\n{'='*70}")
    print(f"  TEX Benchmark Report — {info['timestamp']}")
    print(f"{'='*70}")
    print(f"  Platform : {info['platform']} {info.get('machine','')}")
    print(f"  CPU      : {info.get('processor','?')}  ({info.get('cpu_cores','?')} cores)")
    print(f"  Python   : {info['python']}   PyTorch: {info['torch']}")
    if "gpu_name" in info:
        print(f"  GPU      : {info['gpu_name']}  ({info.get('gpu_memory_gb','?')} GB)")
    print(f"  Device   : {info['device']}")
    if "tex_version" in info:
        print(f"  TEX      : {info['tex_version']}  (commit {info.get('git_commit','?')}"
              f"{'*' if info.get('git_dirty') else ''})")
    print()


def print_results(results: list[BenchResult], show_compile: bool = True):
    if not results:
        print("No results."); return

    nw = max(len(r.program) for r in results) + 2
    hdr = f"{'Program':<{nw}} {'Res':>9} {'B':>2} {'Cache':>5}"
    if show_compile:
        hdr += f" {'Compile':>10}"
    hdr += f" {'Interp':>10} {'Total':>10} {'CV%':>5} {'p5':>9} {'p95':>9} {'Mem':>6} {'N':>3}"
    sep = "=" * len(hdr)

    print(sep); print(hdr); print(sep)
    cat = None
    for r in results:
        c = "example" if r.program.startswith("ex_") else "synthetic"
        if c != cat:
            if cat is not None: print("-" * len(hdr))
            cat = c
        line = f"{r.program:<{nw}} {r.resolution:>4}x{r.resolution:<4} {r.batch:>2} {r.cache_mode:>5}"
        if show_compile:
            line += f" {r.compile_ms:>8.3f}ms"
        line += (f" {r.interp_ms:>8.3f}ms {r.total_ms:>8.3f}ms"
                 f" {r.cv_percent:>4.1f}%"
                 f" {r.p5_ms:>7.3f}ms {r.p95_ms:>7.3f}ms"
                 f" {r.mem_mb:>5.1f}M {r.num_runs:>3}")
        print(line)
    print(sep)

    ti = sum(r.interp_ms for r in results)
    avg_cv = statistics.mean(r.cv_percent for r in results)
    print(f"\n  Total interpreter time : {ti:,.1f} ms  ({len(results)} benchmarks)")
    print(f"  Average CV            : {avg_cv:.1f}%  (target <{TARGET_CV*100:.0f}%)")


def compare_results(current: list[BenchResult], baseline_path: str):
    with open(baseline_path) as f:
        base = json.load(f)
    bmap = {(e["program"], e["resolution"], e["batch"], e["cache_mode"]): e
            for e in base["results"]}

    print(f"\n  Comparison vs {Path(baseline_path).name}")
    base_info = base.get("system", {})
    if base_info:
        print(f"  Baseline: TEX {base_info.get('tex_version','?')}"
              f"  commit {base_info.get('git_commit','?')}"
              f"  {base_info.get('timestamp','')}")
    nw = max((len(r.program) for r in current), default=20) + 2
    hdr = f"{'Program':<{nw}} {'Res':>9} {'Baseline':>10} {'Current':>10} {'Speedup':>8} {'Verdict':>7}"
    print("=" * len(hdr)); print(hdr); print("=" * len(hdr))

    speedups = []
    for r in current:
        key = (r.program, r.resolution, r.batch, r.cache_mode)
        if key not in bmap:
            continue
        bms = bmap[key]["interp_ms"]
        if bms <= 0:
            continue
        sp = bms / r.interp_ms
        speedups.append(sp)
        tag = ("FASTER" if sp > 1.05 else "SLOWER" if sp < 0.95 else "~same")
        print(f"{r.program:<{nw}} {r.resolution:>4}x{r.resolution:<4}"
              f" {bms:>8.3f}ms {r.interp_ms:>8.3f}ms {sp:>6.2f}x  {tag}")

    if speedups:
        geo = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        faster = sum(1 for s in speedups if s > 1.05)
        slower = sum(1 for s in speedups if s < 0.95)
        print(f"\n  Geometric mean speedup : {geo:.3f}x")
        print(f"  Faster (>1.05x)        : {faster}/{len(speedups)}")
        print(f"  Slower (<0.95x)        : {slower}/{len(speedups)}")
        print(f"  Neutral                : {len(speedups)-faster-slower}/{len(speedups)}")


def save_results(results: list[BenchResult], path: str, info: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "system": info,
        "config": {
            "min_runs": MIN_RUNS, "max_runs": MAX_RUNS,
            "warmup": WARMUP_RUNS, "target_cv": TARGET_CV,
        },
        "results": [asdict(r) for r in results],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to {path}")


# -- Main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="TEX Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--full", action="store_true",
                    help="Full matrix: res 256-4096, batch 1+4, warm+cold")
    ap.add_argument("--quick", action="store_true",
                    help="Quick sanity check: 512x512 B=1 warm only")
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--cold", action="store_true",
                    help="Include cold-cache (recompile) measurements")
    ap.add_argument("--device", type=str, default="cpu",
                    choices=["cpu", "cuda"])
    ap.add_argument("--precision", type=str, default="fp32",
                    choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--save", type=str, default=None,
                    help="Save results JSON (default: auto-named)")
    ap.add_argument("--compare", type=str, default=None,
                    help="Compare against baseline JSON")
    ap.add_argument("--examples-only", action="store_true")
    ap.add_argument("--synthetic-only", action="store_true")
    args = ap.parse_args()

    # -- Resolve run matrix --
    if args.quick:
        resolutions = [512]
        batches = [1]
        cache_modes = ["warm"]
    elif args.full:
        resolutions = RESOLUTIONS_FULL
        batches = BATCH_SIZES_FULL
        cache_modes = ["warm", "cold"] if args.cold else ["warm"]
    else:
        resolutions = ([args.resolution] if args.resolution
                       else RESOLUTIONS_STANDARD)
        batches = [args.batch] if args.batch else BATCH_SIZES_STANDARD
        cache_modes = ["warm", "cold"] if args.cold else ["warm"]

    # -- Load programs --
    programs: list[BenchmarkProgram] = []
    if not args.examples_only:
        programs.extend(SYNTHETIC_PROGRAMS)
    if not args.synthetic_only:
        programs.extend(load_example_programs())
    if not programs:
        print("No programs."); return

    info = system_info(args.device)
    print_header(info)
    print(f"  Programs    : {len(programs)}")
    print(f"  Resolutions : {resolutions}")
    print(f"  Batches     : {batches}")
    print(f"  Cache modes : {cache_modes}")
    print(f"  Runs        : {MIN_RUNS}-{MAX_RUNS}  (target CV < {TARGET_CV*100:.0f}%)")
    print()

    total_jobs = len(programs) * len(resolutions) * len(batches) * len(cache_modes)
    results: list[BenchResult] = []
    done = 0

    for res in resolutions:
        for batch in batches:
            for cm in cache_modes:
                tag = f"{res}x{res} B={batch} {cm}"
                print(f"--- {tag} {'-'*(50 - len(tag))}")
                for prog in programs:
                    done += 1
                    if prog.string_only and res > 256:
                        continue
                    r = measure(prog, batch, res, res,
                                cache_mode=cm, device=args.device,
                                precision=args.precision)
                    if r:
                        results.append(r)
                        cv_flag = "*" if r.cv_percent > 5 else " "
                        print(f"  [{done:>{len(str(total_jobs))}}/{total_jobs}]"
                              f" {prog.name:<30}"
                              f" {r.interp_ms:>9.3f}ms"
                              f"  CV={r.cv_percent:.1f}%{cv_flag}"
                              f"  ({r.num_runs} runs)")

    # -- Report --
    print("\n")
    print_results(results, show_compile="cold" in cache_modes)

    # -- Save --
    save_path = args.save
    if save_path is None:
        label = info.get("git_commit", "unknown")
        if info.get("git_dirty"):
            label += "-dirty"
        save_path = str(RESULTS_DIR / f"{label}.json")
    save_results(results, save_path, info)

    # -- Compare --
    if args.compare:
        compare_results(results, args.compare)


if __name__ == "__main__":
    main()
