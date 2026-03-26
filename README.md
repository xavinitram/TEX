# TEX Wrangle

### Tensor Expression Language for ComfyUI

<p align="center">
  <img src="TEX_node.webp" alt="TEX Wrangle node" width="500">
</p>

A compact per-pixel DSL inspired by **Houdini VEX**, **VDB AX**, and **Nuke BlinkScript**. Write image, mask, latent, and string processing logic directly in a node — with static typing, GPU acceleration, and 100+ stdlib functions.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-orange.svg)](https://github.com/comfyanonymous/ComfyUI)

---

## Quick Start

Add a **TEX Wrangle** node (category: TEX). Write code using `@name` to reference inputs — sockets are created automatically:

```c
// Grayscale — connect any image to the "image" socket
float gray = luma(@image);
@OUT = vec3(gray);
```

**Blend two images** with an adjustable parameter:

```c
f$blend = 0.5;
@OUT = lerp(@base, @overlay, $blend);
```

**Multiple outputs** — each `@name` written to becomes an output socket:

```c
f$strength = 1.0;
@result = @image * $strength;
@mask = luma(@image);
```

Click the **?** icon on the node for a built-in quick reference card.

## Installation

**Prerequisites:** ComfyUI with Python 3.10+ and PyTorch (both included with standard installs). No additional dependencies.

| Method | Instructions |
|--------|-------------|
| **ComfyUI Manager** *(recommended)* | Search "TEX Wrangle" → Install |
| **Git clone** | `cd ComfyUI/custom_nodes && git clone https://github.com/xavinitram/TEX.git TEX_Wrangle` |
| **Manual** | Download & extract into `ComfyUI/custom_nodes/TEX_Wrangle/` |

Restart ComfyUI after installation. The node appears under the **TEX** category.

## Features

| Feature | Details |
|---------|---------|
| **Per-pixel processing** | Code runs on every pixel, automatically vectorized via PyTorch |
| **Static typing** | `float`, `int`, `vec3`, `vec4`, `mat3`, `mat4`, `string`, `T[]` arrays |
| **`@` bindings** | Name inputs freely (`@image`, `@base`, `@overlay`) — sockets auto-created |
| **`$` parameters** | `f$strength = 0.5;` creates adjustable widgets on the node |
| **Multiple outputs** | Write to `@result`, `@mask`, etc. for multiple output sockets |
| **Control flow** | `if/else` (vectorized), `for` loops, `while` loops, `break`/`continue` |
| **GPU acceleration** | CPU or GPU with auto device detection |
| **`torch.compile`** | Optional JIT compilation for faster repeated execution |
| **Two-tier caching** | In-memory LRU + disk persistence for instant re-execution |
| **100+ stdlib functions** | Math, color, noise, sampling, strings, arrays, image reductions |
| **Latent support** | Process latent tensors directly (SD1.5, SDXL, SD3) |
| **Batch & temporal** | `fi`/`fn` for frame-aware effects, `fetch_frame`/`sample_frame` for cross-frame access |
| **Snippets** | Right-click → Snippets for 36 built-in examples; save your own with folder organization |
| **Nodes v3** | Built on ComfyUI's Nodes v3 API (`comfy_api.latest`) |

## Language Reference

### Types

| Type | Description | Example |
|------|-------------|---------|
| `float` | Scalar value | `float x = 0.5;` |
| `int` | Integer value | `int n = 42;` |
| `vec3` | 3-component vector (RGB) | `vec3 c = vec3(1.0, 0.0, 0.0);` |
| `vec4` | 4-component vector (RGBA) | `vec4 p = vec4(r, g, b, 1.0);` |
| `mat3` / `mat4` | Matrices (internal only) | `mat3 m = mat3(1.0);` |
| `string` | Text value (scalar-only) | `string s = "hello";` |
| `T[]` | Fixed-size array | `float arr[5];` `vec4 colors[9];` |

**Promotion:** `int` → `float` → `vec3` → `vec4` (automatic). Strings require explicit `str()` / `to_float()`.

### Bindings & Parameters

```c
// @ bindings — auto-create input/output sockets
@OUT = lerp(@base, @overlay, 0.5);

// $ parameters — create adjustable widgets
f$strength = 0.5;    // FLOAT widget
i$radius = 2;        // INT widget
s$label = "hello";   // STRING widget

@OUT = @image * $strength;
```

### Channel Access & Swizzling

```c
float red = @A.r;       // single channel
vec3 rgb = @A.rgb;      // 3-channel swizzle
vec4 bgra = @A.bgra;    // reorder channels
```

### Operators

| Category | Operators |
|----------|-----------|
| Arithmetic | `+` `-` `*` `/` `%` |
| Comparison | `==` `!=` `<` `>` `<=` `>=` |
| Logical | `&&` `\|\|` `!` |
| Ternary | `cond ? a : b` |
| Compound | `+=` `-=` `*=` `/=` `++` `--` |

### Control Flow

```c
// if / else if / else (vectorized via torch.where)
if (luma(@A) > 0.5) {
    @OUT = vec3(1.0, 0.0, 0.0);
} else {
    @OUT = vec3(0.0, 0.0, 1.0);
}

// for loops (bounded, vectorized body)
vec4 sum = vec4(0.0);
for (int i = -2; i <= 2; i++) {
    sum += fetch(@A, ix + i, iy);
}
@OUT = sum / 5.0;

// while loops (with break/continue, capped at 1024 iterations)
float val = 1.0;
while (val < 100.0) { val = val * 2.0; }
```

### Built-in Variables

| Variable | Description |
|----------|-------------|
| `ix`, `iy` | Pixel coordinates (integers) |
| `u`, `v` | Normalized coordinates (0.0 – 1.0) |
| `iw`, `ih` | Image dimensions |
| `fi`, `fn` | Frame index / frame count |
| `ic` | Latent channel count (0 for images) |
| `PI`, `E` | Math constants |

### Standard Library (100+ functions)

**Math:** `sin` `cos` `tan` `asin` `acos` `atan` `atan2` `sinh` `cosh` `tanh` `sqrt` `pow` `pow2` `pow10` `exp` `log` `log2` `log10` `abs` `sign` `floor` `ceil` `round` `fract` `mod` `hypot` `degrees` `radians` `spow` `sdiv` `isnan` `isinf`

**Interpolation:** `min` `max` `clamp` `lerp`/`mix` `fit` `smoothstep` `step`

**Vector & Matrix:** `dot` `length` `distance` `normalize` `cross` `reflect` `transpose` `determinant` `inverse`

**Color:** `luma` `hsv2rgb` `rgb2hsv`

**Noise:** `perlin` `simplex` `fbm`

**Sampling:**

| Function | Interpolation | Coordinates |
|----------|--------------|-------------|
| `sample(@A, u, v)` | Bilinear | UV [0,1] |
| `fetch(@A, px, py)` | Nearest | Pixel ints |
| `sample_cubic(@A, u, v)` | Bicubic | UV [0,1] |
| `sample_lanczos(@A, u, v)` | Lanczos-3 | UV [0,1] |
| `fetch_frame(@A, f, px, py)` | Nearest | Pixel ints |
| `sample_frame(@A, f, u, v)` | Bilinear | UV [0,1] |

**Image Reductions:** `img_sum` `img_mean` `img_min` `img_max` `img_median`

**String:** `str` `len` `replace` `strip` `lower` `upper` `contains` `startswith` `endswith` `find` `substr` `to_int` `to_float` `sanitize_filename` `split` `pad_left` `pad_right` `format` `repeat` `str_reverse` `count` `matches` `hash` `hash_float` `hash_int` `char_at`

**Array:** `sort` `reverse` `arr_sum` `arr_min` `arr_max` `median` `arr_avg` `len` `join`

## Examples

The `examples/` directory contains 36 ready-to-use snippets:

| Category | Examples |
|----------|---------|
| **Basics** | `grayscale` `invert` `brightness_contrast` `channel_swap` `color_mix` `threshold_mask` |
| **Effects** | `vignette` `blur` `sharpen` `unsharp_mask` `swirl` `pixelate` `chromatic_aberration` |
| **Color** | `hue_shift` `color_grade` `levels` `tone_map` `auto_levels` `conditional` |
| **Filters** | `median_filter` `vec4_median` `edge_detect` `bilateral_approx` `mask_from_color` `normal_map` |
| **Procedural** | `gradient` `radial_gradient` `perlin_clouds` `simplex_terrain` `lens_distortion` |
| **Latent** | `latent_scale` `latent_blend` |
| **String** | `string_build` `string_case` |
| **Temporal** | `frame_blend` `motion_detect` |

## Troubleshooting

<details>
<summary><strong>Common issues</strong></summary>

**1×1 output with no inputs:** Procedural code without image inputs produces 1×1 because output resolution comes from connected inputs. Connect an image to set the size.

**Variable `v` conflict:** The built-in `v` (normalized y-coordinate) is always defined. Use `val` or `value` instead. Same for `u`, `ix`, `iy`, `iw`, `ih`, `fi`, `fn`, `ic`, `PI`, `E`.

**torch.compile on Windows:** Install Visual Studio Build Tools with "Desktop development with C++" for full `inductor` support. The default `none` mode works everywhere.

| Error | Fix |
|-------|-----|
| `Undefined variable 'x'` | Declare before use: `float x = ...;` |
| `Type mismatch: cannot assign VEC4 to FLOAT` | Use `.r` channel access or declare as `vec4` |
| `Unknown function 'foo'` | Check spelling in the **?** help popup |
| `TEX program must assign to at least one output` | Add `@OUT = ...;` |

</details>

## Performance

v0.9.0 is **3.4× faster** overall than v0.6.0 across all benchmark scenarios. Typical CPU times at 512×512 (compile on, warm start):

| Program | v0.8.0 | v0.9.0 | Speedup |
|---------|--------|--------|---------|
| passthrough | 0.03 ms | 0.03 ms | 1.0x |
| color_grade | 13.4 ms | 8.5 ms | 1.6x |
| noise_fbm | 178 ms | 22 ms | **9.0x** |
| fetch_kernel | 42 ms | 25 ms | 1.7x |

Run `python benchmarks/four_scenario_bench.py` for results on your system.

## Development

See **[DEVELOPMENT.md](DEVELOPMENT.md)** for architecture, compilation pipeline internals, and guides for adding functions, types, and operators.

```bash
# Run tests
cd custom_nodes/TEX_Wrangle
python -m pytest tests/test_tex.py -v    # 61 tests expected
```

## License

MIT
