"""
TEX Benchmark Suite — comprehensive performance measurement for TEX programs.

Measures compile time, interpreter time, and total E2E time across multiple
resolutions, batch sizes, and cache states. Includes all 29 real-world
examples plus 14 synthetic micro-benchmarks.

Usage:
    python tests/bench_tex.py                       # Default: 512x512 B=1 warm
    python tests/bench_tex.py --quick               # Quick: 512x512 B=1 warm, fewer programs
    python tests/bench_tex.py --full                 # Full: all resolutions x batches x cache modes
    python tests/bench_tex.py --save baseline.json   # Save results
    python tests/bench_tex.py --compare baseline.json # Compare against baseline
    python tests/bench_tex.py --resolution 1024      # Specific resolution
    python tests/bench_tex.py --batch 4              # Specific batch size
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path

# Add custom_nodes dir to path
_pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_custom_nodes_dir = os.path.dirname(_pkg_dir)
sys.path.insert(0, _custom_nodes_dir)

import torch
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker, TEXType
from TEX_Wrangle.tex_compiler.optimizer import optimize
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, _collect_identifiers
from TEX_Wrangle.tex_cache import TEXCache

# ── Configuration ─────────────────────────────────────────────────────

RESOLUTIONS = [64, 256, 512, 1024, 2048, 4096, 8192]
BATCH_SIZES = [1, 4, 8]
DEFAULT_RESOLUTION = 512
DEFAULT_BATCH = 1

# Adaptive run parameters
MIN_RUNS = 10
MAX_RUNS = 50
WARMUP_RUNS = 5
TARGET_CV = 0.05  # 5% coefficient of variation

EXAMPLES_DIR = Path(_pkg_dir) / "examples"


# ── Data Structures ───────────────────────────────────────────────────

@dataclass
class BenchmarkProgram:
    """A TEX program to benchmark."""
    name: str
    code: str
    category: str  # "synthetic" or "example"
    # What bindings this program needs
    needs_image: bool = False
    needs_ref: bool = False
    needs_frames: bool = False
    needs_latent_ab: bool = False
    needs_latent: bool = False
    needs_base_overlay_blend: bool = False
    needs_text: bool = False
    needs_mask: bool = False
    # Param defaults (name -> value)
    param_defaults: dict = field(default_factory=dict)
    # Whether this is a string-only program (no spatial context needed)
    string_only: bool = False
    # Whether the program produces multi-output (no single @OUT)
    multi_output: bool = False


@dataclass
class BenchResult:
    """Result of a single benchmark measurement."""
    program: str
    resolution: int
    batch: int
    cache_mode: str  # "cold" or "warm"
    compile_ms: float  # median compile time
    interp_ms: float   # median interpreter time
    total_ms: float    # median total time
    cv_percent: float  # coefficient of variation of total_ms
    p5_ms: float       # 5th percentile total
    p95_ms: float      # 95th percentile total
    mem_mb: float      # peak memory delta in MB
    num_runs: int       # actual number of measurement runs


# ── Synthetic Benchmark Programs ──────────────────────────────────────

SYNTHETIC_PROGRAMS = [
    BenchmarkProgram(
        name="trivial",
        code="@OUT = @A;",
        category="synthetic",
        needs_image=True,
    ),
    BenchmarkProgram(
        name="math_chain",
        code="@OUT = vec4(sin(u) * cos(v) + 0.5);",
        category="synthetic",
        needs_ref=True,
    ),
    BenchmarkProgram(
        name="branching",
        code="""\
if (u > 0.5) {
    @OUT = @A;
} else {
    @OUT = vec3(1.0) - @A;
}""",
        category="synthetic",
        needs_image=True,
    ),
    BenchmarkProgram(
        name="nested_if",
        code="""\
vec4 c = @A;
if (u > 0.5) {
    if (v > 0.5) {
        c = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        c = vec4(0.0, 1.0, 0.0, 1.0);
    }
} else {
    if (v > 0.5) {
        c = vec4(0.0, 0.0, 1.0, 1.0);
    } else {
        c = vec4(0.0);
    }
}
@OUT = c;""",
        category="synthetic",
        needs_image=True,
    ),
    BenchmarkProgram(
        name="for_loop_10",
        code="""\
float s = 0.0;
for (int i = 0; i < 10; i++) {
    s = s + sin(u * float(i));
}
@OUT = vec4(s * 0.1);""",
        category="synthetic",
        needs_ref=True,
    ),
    BenchmarkProgram(
        name="for_loop_100",
        code="""\
float s = 0.0;
for (int i = 0; i < 100; i++) {
    s = s + sin(u * float(i) * 0.01);
}
@OUT = vec4(s * 0.01);""",
        category="synthetic",
        needs_ref=True,
    ),
    BenchmarkProgram(
        name="while_loop",
        code="""\
float x = 1.0;
int n = 0;
while (x > 0.01) {
    x = x * 0.9;
    n++;
}
@OUT = vec4(float(n) * 0.01);""",
        category="synthetic",
        needs_ref=True,
    ),
    BenchmarkProgram(
        name="function_heavy",
        code="""\
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
        category="synthetic",
        needs_ref=True,
    ),
    BenchmarkProgram(
        name="vector_ops",
        code="""\
vec4 a = vec4(u, v, u * v, 1.0);
vec4 b = vec4(1.0 - u, 1.0 - v, 0.5, 1.0);
vec4 c = a * 0.7 + b * 0.3;
vec3 d = c.rgb * 2.0 - vec3(0.5, 0.5, 0.5);
@OUT = vec4(clamp(d.r, 0.0, 1.0), clamp(d.g, 0.0, 1.0), clamp(d.b, 0.0, 1.0), 1.0);""",
        category="synthetic",
        needs_ref=True,
    ),
    BenchmarkProgram(
        name="array_ops",
        code="""\
float arr[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
arr = sort(arr);
float s = arr_sum(arr);
float avg = s / 5.0;
float mn = arr_min(arr);
float mx = arr_max(arr);
@OUT = vec4(avg / 5.0, mn / 5.0, mx / 5.0, 1.0);""",
        category="synthetic",
        needs_ref=True,
    ),
    BenchmarkProgram(
        name="string_ops",
        code="""\
string s = "hello world";
string u_str = upper(s);
float length = len(s);
@OUT = u_str;""",
        category="synthetic",
        string_only=True,
    ),
    BenchmarkProgram(
        name="noise_perlin",
        code="""\
float n = perlin(u * 10.0, v * 10.0);
@OUT = vec4(n * 0.5 + 0.5);""",
        category="synthetic",
        needs_ref=True,
    ),
    BenchmarkProgram(
        name="noise_fbm",
        code="""\
vec4 _ref = @ref;
float n = fbm(u * 6.0, v * 6.0, 6);
@OUT = vec4(n * 0.5 + 0.5);""",
        category="synthetic",
        needs_ref=True,
    ),
    BenchmarkProgram(
        name="lanczos_sample",
        code="""\
@OUT = sample_lanczos(@A, u + sin(v * 6.28) * 0.01, v);""",
        category="synthetic",
        needs_image=True,
    ),
]


# ── Example Loader ────────────────────────────────────────────────────

def _detect_bindings(code: str) -> dict:
    """Detect what bindings a TEX program needs from its source."""
    info = {
        "needs_image": bool(re.search(r"@image\b", code)),
        "needs_ref": bool(re.search(r"@ref\b", code)),
        "needs_frames": bool(re.search(r"@frames\b", code)),
        "needs_latent_ab": bool(re.search(r"@latent_a\b", code) and re.search(r"@latent_b\b", code)),
        "needs_latent": bool(re.search(r"@latent\b", code)) and not bool(re.search(r"@latent_[ab]\b", code)),
        "needs_base_overlay_blend": bool(re.search(r"@base\b", code)),
        "needs_text": bool(re.search(r"@text\b", code)),
        "needs_mask": bool(re.search(r"@mask\b", code)),
        "string_only": not any(re.search(pat, code) for pat in [r"@image\b", r"@ref\b", r"@frames\b", r"@latent", r"@base\b", r"@mask\b"]) and bool(re.search(r"@OUT\s*=\s*\w+\(", code) or re.search(r"string\b", code)),
        "multi_output": bool(re.search(r"@\w+\s*=", code)) and not bool(re.search(r"@OUT\s*=", code)),
    }
    return info


def _extract_param_defaults(code: str) -> dict:
    """Extract parameter declarations and their defaults from TEX source."""
    defaults = {}
    for m in re.finditer(r'([fis])\$(\w+)\s*=\s*([^;]+);', code):
        type_hint, name, value = m.groups()
        value = value.strip().strip('"').strip("'")
        if type_hint == 'f':
            try:
                defaults[name] = float(value)
            except ValueError:
                defaults[name] = 0.5
        elif type_hint == 'i':
            try:
                defaults[name] = int(value)
            except ValueError:
                defaults[name] = 1
        elif type_hint == 's':
            defaults[name] = value
    return defaults


def load_example_programs() -> list[BenchmarkProgram]:
    """Load all .tex example files as benchmark programs."""
    programs = []
    if not EXAMPLES_DIR.exists():
        print(f"  Warning: examples directory not found at {EXAMPLES_DIR}")
        return programs

    for tex_file in sorted(EXAMPLES_DIR.glob("*.tex")):
        code = tex_file.read_text(encoding="utf-8")
        name = tex_file.stem
        bindings = _detect_bindings(code)
        params = _extract_param_defaults(code)

        programs.append(BenchmarkProgram(
            name=f"ex_{name}",
            code=code,
            category="example",
            param_defaults=params,
            **bindings,
        ))

    return programs


# ── Synthetic Data Generation ─────────────────────────────────────────

def _detect_image_channels(code: str) -> int:
    """Detect whether @image/@A should be 3 or 4 channels based on code usage."""
    # If code uses vec4 arrays with fetch(), or accesses .a/.w, needs 4 channels
    if re.search(r"vec4\s+\w+\[", code) and re.search(r"fetch\(", code):
        return 4
    if re.search(r"@image\.a\b|@A\.a\b|@image\.w\b", code):
        return 4
    return 3


def generate_bindings(prog: BenchmarkProgram, B: int, H: int, W: int,
                      device: str = "cpu") -> dict:
    """Generate synthetic input bindings for a benchmark program."""
    bindings = {}
    img_c = _detect_image_channels(prog.code)

    if prog.needs_image:
        # @A or @image — random image
        bindings["A"] = torch.rand(B, H, W, img_c, dtype=torch.float32, device=device)
        bindings["image"] = bindings["A"]

    if prog.needs_ref:
        bindings["ref"] = torch.rand(B, H, W, 4, dtype=torch.float32, device=device)

    if prog.needs_frames:
        # Multi-frame: use a batch of 4+ frames
        frame_B = max(B, 4)
        bindings["frames"] = torch.rand(frame_B, H, W, 4, dtype=torch.float32, device=device)

    if prog.needs_latent_ab:
        lH, lW = max(1, H // 8), max(1, W // 8)
        bindings["latent_a"] = torch.rand(B, lH, lW, 4, dtype=torch.float32, device=device)
        bindings["latent_b"] = torch.rand(B, lH, lW, 4, dtype=torch.float32, device=device)

    if prog.needs_latent:
        lH, lW = max(1, H // 8), max(1, W // 8)
        bindings["latent"] = torch.rand(B, lH, lW, 4, dtype=torch.float32, device=device)

    if prog.needs_base_overlay_blend:
        bindings["base"] = torch.rand(B, H, W, 3, dtype=torch.float32, device=device)
        bindings["overlay"] = torch.rand(B, H, W, 3, dtype=torch.float32, device=device)
        # @blend is a scalar mask [B, H, W], not a vec3
        bindings["blend"] = torch.rand(B, H, W, dtype=torch.float32, device=device) * 0.5 + 0.25

    if prog.needs_text:
        bindings["text"] = "  Hello World  "

    if prog.needs_mask:
        bindings["mask"] = torch.rand(B, H, W, dtype=torch.float32, device=device)

    # Add param defaults as bindings
    for pname, pval in prog.param_defaults.items():
        bindings[pname] = pval

    return bindings


# ── Compilation Helpers ───────────────────────────────────────────────

def _infer_binding_types(bindings: dict) -> dict[str, TEXType]:
    """Infer TEXType for each binding value."""
    types = {}
    for name, val in bindings.items():
        if isinstance(val, str):
            types[name] = TEXType.STRING
        elif isinstance(val, torch.Tensor):
            if val.dim() == 4:
                c = val.shape[-1]
                types[name] = TEXType.VEC4 if c == 4 else TEXType.VEC3 if c == 3 else TEXType.FLOAT
            elif val.dim() == 3:
                types[name] = TEXType.FLOAT
            else:
                types[name] = TEXType.FLOAT
        elif isinstance(val, int):
            types[name] = TEXType.INT
        elif isinstance(val, float):
            types[name] = TEXType.FLOAT
        else:
            types[name] = TEXType.FLOAT
    return types


def compile_program(code: str, binding_types: dict[str, TEXType]):
    """Compile a TEX program (lex -> parse -> type check -> optimize). Returns (program, type_map, assigned_bindings, used_builtins)."""
    tokens = Lexer(code).tokenize()
    program = Parser(tokens).parse()
    checker = TypeChecker(binding_types=binding_types)
    type_map = checker.check(program)
    program = optimize(program)
    used_builtins = _collect_identifiers(program)
    return program, type_map, checker.assigned_bindings, used_builtins


# Reuse a single Interpreter instance across benchmark runs (mirrors
# the production pattern in tex_node.py where _get_interpreter() caches).
_bench_interp: Interpreter | None = None


def run_interpreter(program, bindings, type_map, device="cpu",
                    output_names=None, precision="fp32", used_builtins=None):
    """Run the interpreter on a compiled program."""
    global _bench_interp
    if _bench_interp is None:
        _bench_interp = Interpreter()
    return _bench_interp.execute(program, bindings, type_map, device=device,
                                 output_names=output_names, precision=precision,
                                 used_builtins=used_builtins)


# ── Adaptive Measurement ─────────────────────────────────────────────

def _percentile(data: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of a sorted list."""
    if not data:
        return 0.0
    k = (len(data) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[f] * (c - k) + data[c] * (k - f)


def measure_program(prog: BenchmarkProgram, B: int, H: int, W: int,
                    cache_mode: str = "warm", device: str = "cpu",
                    precision: str = "fp32") -> BenchResult | None:
    """
    Measure a single program with adaptive run count.

    Returns BenchResult or None if the program can't be run.
    """
    resolution = max(H, W)

    # Memory guard: estimate required memory and skip if too large
    # Rough estimate: each [B, H, W, C] tensor = B * H * W * C * 4 bytes
    pixel_count = B * H * W
    est_input_mb = 0
    if prog.needs_image:
        est_input_mb += pixel_count * 4 * 4 / (1024 * 1024)  # 4 channels
    if prog.needs_ref:
        est_input_mb += pixel_count * 4 * 4 / (1024 * 1024)
    if prog.needs_frames:
        est_input_mb += max(B, 4) * H * W * 4 * 4 / (1024 * 1024)
    if prog.needs_mask:
        est_input_mb += pixel_count * 4 / (1024 * 1024)
    # Skip if estimated input alone > 2048MB (leave room for intermediates + PyTorch overhead)
    if est_input_mb > 2048:
        print(f"  SKIP  {prog.name}: estimated {est_input_mb:.0f}MB input exceeds memory limit")
        return None

    try:
        bindings = generate_bindings(prog, B, H, W, device)
        binding_types = _infer_binding_types(bindings)

        # Determine output names for multi-output programs
        output_names = None

        # Pre-compile once to check validity and detect outputs
        program, type_map, assigned_bindings, used_builtins = compile_program(prog.code, binding_types)
        if assigned_bindings and "OUT" not in assigned_bindings:
            output_names = list(assigned_bindings.keys())

        # Warmup runs
        for _ in range(WARMUP_RUNS):
            if cache_mode == "cold":
                program, type_map, assigned_bindings, used_builtins = compile_program(prog.code, binding_types)
            run_interpreter(program, bindings, type_map, device, output_names, precision, used_builtins)

        # Measurement runs with adaptive stopping
        compile_times = []
        interp_times = []
        total_times = []

        num_runs = 0
        while num_runs < MAX_RUNS:
            gc.collect()

            # Track memory
            tracemalloc.start()
            mem_before = tracemalloc.get_traced_memory()[1]

            t_total_start = time.perf_counter()

            # Compile phase
            if cache_mode == "cold":
                t_compile_start = time.perf_counter()
                program, type_map, assigned_bindings, used_builtins = compile_program(prog.code, binding_types)
                if assigned_bindings and "OUT" not in assigned_bindings:
                    output_names = list(assigned_bindings.keys())
                t_compile_end = time.perf_counter()
                compile_times.append((t_compile_end - t_compile_start) * 1000)
            else:
                compile_times.append(0.0)

            # Interpreter phase
            t_interp_start = time.perf_counter()
            run_interpreter(program, bindings, type_map, device, output_names, precision, used_builtins)
            t_interp_end = time.perf_counter()

            t_total_end = time.perf_counter()

            interp_times.append((t_interp_end - t_interp_start) * 1000)
            total_times.append((t_total_end - t_total_start) * 1000)

            mem_peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

            num_runs += 1

            # Check if we have enough runs for stable results
            if num_runs >= MIN_RUNS:
                mean_total = statistics.mean(total_times)
                if mean_total > 0:
                    cv = statistics.stdev(total_times) / mean_total
                    if cv < TARGET_CV:
                        break

        # Compute statistics
        total_times.sort()
        interp_times.sort()
        compile_times.sort()

        median_total = statistics.median(total_times)
        median_interp = statistics.median(interp_times)
        median_compile = statistics.median(compile_times)

        mean_total = statistics.mean(total_times)
        cv_percent = (statistics.stdev(total_times) / mean_total * 100) if mean_total > 0 and len(total_times) > 1 else 0.0

        p5 = _percentile(total_times, 5)
        p95 = _percentile(total_times, 95)

        # Memory: rough estimate from last run's tracemalloc
        mem_delta_mb = max(0, (mem_peak - mem_before)) / (1024 * 1024)

        return BenchResult(
            program=prog.name,
            resolution=resolution,
            batch=B,
            cache_mode=cache_mode,
            compile_ms=round(median_compile, 3),
            interp_ms=round(median_interp, 3),
            total_ms=round(median_total, 3),
            cv_percent=round(cv_percent, 1),
            p5_ms=round(p5, 3),
            p95_ms=round(p95, 3),
            mem_mb=round(mem_delta_mb, 2),
            num_runs=num_runs,
        )

    except Exception as e:
        print(f"  SKIP  {prog.name}: {e}")
        return None


# ── Reporting ─────────────────────────────────────────────────────────

def print_header():
    """Print system info header."""
    print(f"\nTEX Benchmark Report -- {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print(f"  Python:   {platform.python_version()}")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CPU:      {platform.processor() or 'unknown'}")
    if torch.cuda.is_available():
        print(f"  GPU:      {torch.cuda.get_device_name(0)}")
    print()


def print_results(results: list[BenchResult], show_compile: bool = True):
    """Print results table."""
    if not results:
        print("No results to display.")
        return

    # Column widths
    name_w = max(len(r.program) for r in results) + 2
    header = (
        f"{'Program':<{name_w}} {'Res':>9} {'B':>3} {'Cache':>5}"
    )
    if show_compile:
        header += f" {'Compile':>10}"
    header += f" {'Interp':>10} {'Total':>10} {'CV%':>5} {'p5':>9} {'p95':>9} {'Mem MB':>7} {'Runs':>4}"

    sep = "=" * len(header)
    print(sep)
    print(header)
    print(sep)

    current_category = None
    for r in results:
        # Group separator
        cat = "example" if r.program.startswith("ex_") else "synthetic"
        if cat != current_category:
            if current_category is not None:
                print("-" * len(header))
            current_category = cat

        line = f"{r.program:<{name_w}} {r.resolution:>4}x{r.resolution:<4} {r.batch:>3} {r.cache_mode:>5}"
        if show_compile:
            line += f" {r.compile_ms:>9.3f}ms"
        line += f" {r.interp_ms:>9.3f}ms {r.total_ms:>9.3f}ms {r.cv_percent:>4.1f}% {r.p5_ms:>8.3f}ms {r.p95_ms:>8.3f}ms {r.mem_mb:>6.2f}M {r.num_runs:>4}"
        print(line)

    print(sep)

    # Summary statistics
    total_interp = sum(r.interp_ms for r in results)
    avg_cv = statistics.mean(r.cv_percent for r in results) if results else 0
    print(f"\nTotal interpreter time: {total_interp:.1f}ms across {len(results)} benchmarks")
    print(f"Average CV: {avg_cv:.1f}% (target <5%)")


def compare_results(current: list[BenchResult], baseline_path: str):
    """Compare current results against a saved baseline."""
    with open(baseline_path, "r") as f:
        baseline_data = json.load(f)

    baseline_map = {}
    for entry in baseline_data["results"]:
        key = (entry["program"], entry["resolution"], entry["batch"], entry["cache_mode"])
        baseline_map[key] = entry

    print(f"\nComparison against: {baseline_path}")
    print(f"{'Program':<30} {'Res':>9} {'Baseline':>10} {'Current':>10} {'Speedup':>8}")
    print("=" * 75)

    speedups = []
    for r in current:
        key = (r.program, r.resolution, r.batch, r.cache_mode)
        if key in baseline_map:
            base_ms = baseline_map[key]["total_ms"]
            if base_ms > 0:
                speedup = base_ms / r.total_ms
                speedups.append(speedup)
                marker = "**" if speedup > 1.1 else "  " if speedup > 0.95 else "!!"
                print(f"{r.program:<30} {r.resolution:>4}x{r.resolution:<4} {base_ms:>9.3f}ms {r.total_ms:>9.3f}ms {speedup:>6.2f}x {marker}")

    if speedups:
        geo_mean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        print(f"\n{'Geometric mean speedup:':<45} {geo_mean:.3f}x")
        print(f"{'Programs faster (>1.05x):':<45} {sum(1 for s in speedups if s > 1.05)}/{len(speedups)}")
        print(f"{'Programs slower (<0.95x):':<45} {sum(1 for s in speedups if s < 0.95)}/{len(speedups)}")


def save_results(results: list[BenchResult], path: str):
    """Save results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "results": [
            {
                "program": r.program,
                "resolution": r.resolution,
                "batch": r.batch,
                "cache_mode": r.cache_mode,
                "compile_ms": r.compile_ms,
                "interp_ms": r.interp_ms,
                "total_ms": r.total_ms,
                "cv_percent": r.cv_percent,
                "p5_ms": r.p5_ms,
                "p95_ms": r.p95_ms,
                "mem_mb": r.mem_mb,
                "num_runs": r.num_runs,
            }
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TEX Benchmark Suite")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 512x512 B=1 warm, synthetic only")
    parser.add_argument("--full", action="store_true",
                        help="Full mode: all resolutions x batches x cache modes")
    parser.add_argument("--resolution", type=int, default=None,
                        help="Specific resolution (e.g., 512, 1024)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Specific batch size")
    parser.add_argument("--save", type=str, default=None,
                        help="Save results to JSON file")
    parser.add_argument("--compare", type=str, default=None,
                        help="Compare against baseline JSON file")
    parser.add_argument("--cold", action="store_true",
                        help="Include cold-cache benchmarks")
    parser.add_argument("--examples-only", action="store_true",
                        help="Only run example programs")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Only run synthetic programs")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"],
                        help="Precision: fp32 (default), fp16, bf16")
    args = parser.parse_args()

    # Determine what to run
    if args.quick:
        resolutions = [512]
        batches = [1]
        cache_modes = ["warm"]
    elif args.full:
        resolutions = RESOLUTIONS
        batches = BATCH_SIZES
        cache_modes = ["warm", "cold"] if args.cold else ["warm"]
    else:
        resolutions = [args.resolution] if args.resolution else [DEFAULT_RESOLUTION]
        batches = [args.batch] if args.batch else [DEFAULT_BATCH]
        cache_modes = ["warm", "cold"] if args.cold else ["warm"]

    # Load programs
    programs = []
    if not args.examples_only:
        programs.extend(SYNTHETIC_PROGRAMS)
    if not args.synthetic_only:
        examples = load_example_programs()
        programs.extend(examples)

    if not programs:
        print("No programs to benchmark.")
        return

    print_header()
    print(f"Benchmarking {len(programs)} programs")
    print(f"  Resolutions: {resolutions}")
    print(f"  Batch sizes: {batches}")
    print(f"  Cache modes: {cache_modes}")
    print(f"  Adaptive runs: {MIN_RUNS}-{MAX_RUNS} (target CV < {TARGET_CV*100:.0f}%)")
    print()

    results: list[BenchResult] = []
    total_benchmarks = len(programs) * len(resolutions) * len(batches) * len(cache_modes)
    completed = 0

    for res in resolutions:
        for batch in batches:
            for cache_mode in cache_modes:
                print(f"--- {res}x{res} B={batch} {cache_mode} ---")
                for prog in programs:
                    completed += 1
                    progress = f"[{completed}/{total_benchmarks}]"

                    # Skip string-only programs for large resolutions
                    if prog.string_only and res > 256:
                        continue

                    result = measure_program(prog, batch, res, res,
                                             cache_mode=cache_mode,
                                             precision=args.precision)
                    if result:
                        results.append(result)
                        cv_marker = "*" if result.cv_percent > 5 else " "
                        print(f"  {progress} {prog.name:<30} {result.total_ms:>9.3f}ms  CV={result.cv_percent:.1f}%{cv_marker}  ({result.num_runs} runs)")

    # Print full results table
    print("\n")
    print_results(results, show_compile="cold" in cache_modes)

    # Save if requested
    if args.save:
        save_results(results, args.save)

    # Compare if requested
    if args.compare:
        compare_results(results, args.compare)


if __name__ == "__main__":
    main()
