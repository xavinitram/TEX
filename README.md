# TEX Wrangle — Tensor Expression Language for ComfyUI

TEX Wrangle is a compact per-pixel DSL for ComfyUI, inspired by Houdini VEX, VDB AX and Nuke BlinkScript. Write image, mask, latent, scalar, and string-processing logic directly in a node, with static typing, automatic input bindings, PyTorch vectorization, GPU support, and clear errors.

## What TEX Is

TEX sits between simple math-expression nodes and full Python scripting nodes. It provides:

- **Per-pixel processing** — write code that runs on every pixel, automatically vectorized via PyTorch
- **Static typing** — `float`, `int`, `vec3`, `vec4`, `string`, arrays with compile-time checking
- **`@` bindings** — reference node inputs with any name (`@A`, `@base_image`, `@strength`); write output to `@OUT`
- **Control flow** — `if/else` (vectorized via `torch.where`) and bounded `for` loops
- **GPU acceleration** — execute on CPU or GPU with automatic device detection
- **`torch.compile` support** — optional JIT compilation for faster repeated execution
- **Two-tier caching** — in-memory LRU + disk persistence for instant re-execution
- **75+ stdlib functions** — math, interpolation, vector ops, color conversion, image sampling, cross-frame sampling, string manipulation, array operations, image reductions
- **Good error messages** — line/column-mapped errors from the compiler

## Quick Start

Add a **TEX Wrangle** node (category: TEX). Write code using `@name` to reference inputs — sockets are created automatically:

```c
// Grayscale conversion — connect any image to the "A" socket
float gray = luma(@A);
@OUT = vec3(gray, gray, gray);
```

Use descriptive names for clarity:

```c
// Blend two images — sockets "base" and "overlay" appear on the node
@OUT = lerp(@base, @overlay, 0.5);
```

Click the **?** icon on the node for a quick reference card.

## Language Reference

### Types

| Type | Description | Example |
|------|-------------|---------|
| `float` | Scalar value | `float x = 0.5;` |
| `int` | Integer value | `int n = 42;` |
| `vec3` | 3-component vector (RGB) | `vec3 color = vec3(1.0, 0.0, 0.0);` |
| `vec4` | 4-component vector (RGBA) | `vec4 pixel = vec4(r, g, b, 1.0);` |
| `string` | Text value (scalar-only) | `string name = "hello";` |
| `T[]` | Fixed-size array (T = float/int/vec3/vec4/string) | `float arr[5];` `vec4 colors[9];` |

### @ Bindings

Reference node inputs with `@name` — any valid identifier. Write output to `@OUT`. Input slots are created automatically when you reference them in code.

```c
// Classic single-letter bindings still work
float r = @A.r;           // red channel of input A
@OUT = lerp(@A, @B, @C);  // mix two images with parameter C

// Arbitrary descriptive names
@OUT = lerp(@base_image, @overlay, @strength);

// Any valid identifier: lowercase, uppercase, underscores, digits
vec4 result = @layer1 + @layer2;
float g = luma(@high_res_input);
string tag = @prefix + "_output";
```

Names can be any combination of letters, digits, and underscores (must start with a letter or underscore). Reserved names: `OUT`, `code`, `output_type`, `device`, `compile_mode`.

### Channel Access

Access individual channels with `.r/.g/.b/.a` (color) or `.x/.y/.z/.w` (position):

```c
float red = @A.r;     // single channel -> float
vec3 rgb = @A.rgb;    // 3-channel swizzle -> vec3
vec4 bgra = @A.bgra;  // reorder channels -> vec4
```

### Operators

**Arithmetic:** `+`, `-`, `*`, `/`, `%`
**Comparison:** `==`, `!=`, `<`, `>`, `<=`, `>=`
**Logical:** `&&`, `||`, `!`
**Ternary:** `cond ? a : b`

**Compound assignment:** `+=`, `-=`, `*=`, `/=`
**Increment/decrement:** `++`, `--`

### Control Flow

#### if/else

```c
if (luma(@A) > 0.5) {
    @OUT = vec3(1.0, 0.0, 0.0);  // bright pixels -> red
} else {
    @OUT = vec3(0.0, 0.0, 1.0);  // dark pixels -> blue
}
```

`if/else` is vectorized via `torch.where` — both branches execute and results are selected per-pixel by the condition.

#### for loops

```c
vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
for (int i = -2; i <= 2; i++) {
    sum += fetch(@A, ix + i, iy);
}
@OUT = sum / 5.0;
```

Loops must have integer loop variables with compile-time-deterministic bounds. Iteration is capped at 1000 to prevent hangs. Loop bodies run sequentially, with each iteration's operations vectorized across all pixels.

### Built-in Variables

| Variable | Description |
|----------|-------------|
| `ix`, `iy` | Pixel coordinates (0 to width/height - 1) |
| `u`, `v` | Normalized coordinates (0.0 to 1.0) |
| `iw`, `ih` | Image dimensions |
| `fi` | Frame/batch index (0 to B-1) — per-pixel, varies per frame |
| `fn` | Total frame/batch count (B) — scalar |
| `ic` | Latent channel count (0 for images, 4 for SD1.5/SDXL, 16 for SD3) |
| `PI`, `E` | Mathematical constants |

### Standard Library

**Math:** `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sqrt`, `pow`, `exp`, `log`, `abs`, `sign`, `floor`, `ceil`, `round`, `fract`, `mod`

**Interpolation:** `min`, `max`, `clamp`, `lerp`/`mix`, `fit`, `smoothstep`, `step`

**Vector:** `dot`, `length`, `distance`, `normalize`, `cross`, `reflect`

**Color:** `luma`, `hsv2rgb`, `rgb2hsv`

**Noise:** `perlin`, `simplex`, `fbm`

**Sampling:**

| Function | Interpolation | Coordinates | Description |
|----------|--------------|-------------|-------------|
| `sample(@A, u, v)` | Bilinear | UV [0, 1] | Standard texture sampling |
| `fetch(@A, px, py)` | Nearest neighbor | Pixel integers | Direct pixel access, ideal for `fetch(@A, ix+1, iy)` neighbor patterns |
| `sample_cubic(@A, u, v)` | Bicubic (Catmull-Rom) | UV [0, 1] | Smoother than bilinear, good for distortion effects |
| `sample_lanczos(@A, u, v)` | Lanczos-3 | UV [0, 1] | Sharpest reconstruction with minimal ringing |
| `fetch_frame(@A, frame, px, py)` | Nearest neighbor | Pixel integers | Read from a specific frame (cross-frame access) |
| `sample_frame(@A, frame, u, v)` | Bilinear | UV [0, 1] | Bilinear sample from a specific frame |

All sampling functions return `vec4` and clamp to image bounds. Frame indices are clamped to `[0, B-1]`.

**Noise:**

| Function | Description |
|----------|-------------|
| `perlin(x, y)` | 2D Perlin noise, returns float in [-1, 1]. Deterministic, C2 continuous. |
| `simplex(x, y)` | 2D Simplex noise, returns float in [-1, 1]. Fewer grid artifacts than Perlin. |
| `fbm(x, y, octaves)` | Fractional Brownian Motion (multi-octave Perlin). `octaves`: 1-10. |

Noise coordinates are in world space — multiply `u`/`v` by a frequency to control scale:
```c
float n = perlin(u * 8.0, v * 8.0);  // 8x frequency
float cloud = fbm(u * 6.0, v * 6.0, 6);  // 6 octaves of detail
```

**String:** `str`, `len`, `replace`, `strip`, `lower`, `upper`, `contains`, `startswith`, `endswith`, `find`, `substr`, `to_int`, `to_float`, `sanitize_filename`

**Array:** `sort`, `reverse`, `arr_sum`, `arr_min`, `arr_max`, `median`, `arr_avg`, `len`, `join`

**Image Reductions:** `img_sum`, `img_mean`, `img_min`, `img_max`, `img_median`

### Latent Support

TEX can process latent tensors directly. Connect latent data to any input slot — with `output_type=auto` (default), LATENT output is detected automatically when any input is a latent tensor.

Latents are automatically converted from channel-first `[B,C,H,W]` to channel-last for processing, then converted back on output. Values are **not clamped** — latent space typically ranges from -4 to 4. All LATENT metadata (noise_mask, batch_index) is preserved through execution.

The built-in variable `ic` provides the latent channel count (4 for SD1.5/SDXL, 16 for SD3, 0 for non-latent inputs).

```c
// Scale latent intensity
@OUT = @A * 1.1;
```

```c
// Blend two latents with a parameter
@OUT = lerp(@A, @B, @C);
```

```c
// Per-channel normalization
float ch = @A.r;
@OUT = @A;
```

### String Support

TEX supports a `string` type for text processing — dynamic filename construction, prompt manipulation, tag formatting, and metadata extraction. Strings are **scalar-only** (not per-pixel tensors); they execute once per node invocation.

**Literals:** Double-quoted with escape sequences: `\"`, `\\`, `\n`, `\t`, `\r`

```c
string greeting = "hello world";
string path = "C:\\output\\file.png";
```

**Operators:** `+` (concatenation), `==`/`!=` (comparison returns 0.0/1.0)

```c
string full = "hello" + " " + "world";
float eq = ("abc" == "abc");  // 1.0
```

**No implicit coercion** between strings and numbers — use `str()` to convert:

```c
string label = "frame_" + str(42);  // "frame_42"
```

**Casting:** `string(number)` converts a number to its string representation.

**String Functions:**

| Function | Args | Return | Description |
|----------|------|--------|-------------|
| `str(x)` | number | string | Number to string |
| `len(s)` | string | float | String length |
| `replace(s, old, new)` | 3 strings | string | Replace all occurrences |
| `strip(s)` | string | string | Trim whitespace |
| `lower(s)` | string | string | To lowercase |
| `upper(s)` | string | string | To uppercase |
| `contains(s, sub)` | 2 strings | float | 1.0 if sub found, else 0.0 |
| `startswith(s, prefix)` | 2 strings | float | 1.0 if match, else 0.0 |
| `endswith(s, suffix)` | 2 strings | float | 1.0 if match, else 0.0 |
| `find(s, sub)` | 2 strings | float | Index of sub, or -1.0 |
| `substr(s, start, len?)` | string, num(s) | string | Extract substring (0-based) |
| `to_int(s)` | string | int | Parse integer from string |
| `to_float(s)` | string | float | Parse float from string |
| `sanitize_filename(s)` | string | string | Remove illegal filename chars |

With `output_type=auto` (default), string output is detected automatically when `@OUT` is assigned a string:

```c
// Build a dynamic filename
string name = @A + "_processed_" + @B;
@OUT = sanitize_filename(name);
```

### Arrays

TEX supports fixed-size arrays of `float`, `int`, `vec3`, `vec4`, or `string` elements. Numeric arrays are backed by tensors for full GPU compatibility. Arrays enable workflows that require collecting, sorting, or aggregating multiple values — median filters, neighbor sampling, color palettes, tag manipulation, and more.

**Declaration:**

```c
// Scalar arrays (zero-initialized)
float arr[5];
int indices[3];

// Vec3/vec4 arrays
vec4 samples[9];
vec3 palette[] = {vec3(1,0,0), vec3(0,1,0), vec3(0,0,1)};

// String arrays
string tags[] = {"cherry", "apple", "banana"};

// Inferred size from initializer list
float values[] = {1.0, 2.0, 3.0};

// Copy from another array
float copy[5] = arr;
```

**Indexing:**

```c
float x = arr[2];       // read element
arr[0] = 1.0;           // write element
arr[i] = some_value;    // per-pixel indexing
```

Out-of-bounds indices are **clamped** to `[0, size-1]` — no runtime errors, matching `fetch()` safety behavior.

**Typical pattern — loop population:**

```c
float samples[5];
for (int i = 0; i < 5; i++) {
    samples[i] = fetch(@A, ix + i - 2, iy).r;
}
samples = sort(samples);
float med = samples[2];  // median of 5 samples
```

**Element types:** `float`, `int`, `vec3`, `vec4`, `string`. Vec3/vec4 arrays store `[B,H,W,N,C]` tensors; string arrays are Python lists.
**Max size:** 1024 elements.

**Array Functions:**

| Function | Return | Description |
|----------|--------|-------------|
| `sort(arr)` | array | Sort ascending (per-channel for vec arrays, alphabetical for strings) |
| `reverse(arr)` | array | Reverse order, returns new array |
| `arr_sum(arr)` | element type | Sum of elements (returns vec3/vec4 for vec arrays) |
| `arr_min(arr)` | element type | Minimum element (per-channel for vec arrays) |
| `arr_max(arr)` | element type | Maximum element (per-channel for vec arrays) |
| `median(arr)` | element type | Median element (per-channel for vec arrays) |
| `arr_avg(arr)` | element type | Mean of elements (per-channel for vec arrays) |
| `len(arr)` | float | Array length |
| `join(arr, sep)` | string | Concatenate string array with separator |

**Example — 3x3 vec4 median filter (clean, single-array approach):**

```c
vec4 samples[9];
int idx = 0;
for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
        samples[idx] = fetch(@A, ix + dx, iy + dy);
        idx++;
    }
}
@OUT = median(samples);
```

**Example — string array tag sorting:**

```c
string tags[] = {"cherry", "apple", "banana"};
tags = sort(tags);
@OUT = join(tags, ", ");  // "apple, banana, cherry"
```

### Batch & Temporal

TEX processes all frames in an image batch simultaneously — per-pixel operations are vectorized across the batch dimension. Use `fi` and `fn` to write frame-aware effects, and `fetch_frame`/`sample_frame` to read from specific frames.

**Built-in variables:**
- `fi` — frame/batch index (0 to B-1), [B,H,W] tensor — each pixel knows which frame it belongs to
- `fn` — total frame/batch count, scalar — use for normalization (`fi / max(fn-1, 1)`)

**Fade in over sequence:**
```c
@OUT = @A * (fi / max(fn - 1, 1));
```

**3-frame temporal average (noise reduction):**
```c
vec4 prev = fetch_frame(@A, fi - 1, ix, iy);
vec4 curr = fetch_frame(@A, fi, ix, iy);
vec4 next = fetch_frame(@A, fi + 1, ix, iy);
@OUT = (prev + curr + next) / 3.0;
```

**Cross-frame motion detection:**
```c
vec4 curr = fetch_frame(@A, fi, ix, iy);
vec4 prev = fetch_frame(@A, max(fi - 1, 0), ix, iy);
@OUT = abs(curr - prev);
```

Frame indices are clamped to `[0, B-1]` — reading beyond the first or last frame returns the boundary frame. Single images (B=1) work seamlessly: `fi = 0`, `fn = 1`.

### Image Reductions

TEX provides whole-image reduction functions that compute per-channel statistics across all pixels. Results broadcast naturally in per-pixel expressions via PyTorch broadcasting.

| Function | Return | Description |
|----------|--------|-------------|
| `img_sum(@A)` | same type | Sum of all pixels per channel |
| `img_mean(@A)` | same type | Mean of all pixels per channel |
| `img_min(@A)` | same type | Minimum per channel |
| `img_max(@A)` | same type | Maximum per channel |
| `img_median(@A)` | same type | Median per channel |

Reductions return broadcast-friendly shapes (`[B,1,1,C]` for images, `[B,1,1]` for masks) that participate directly in per-pixel expressions. Each frame in a batch is reduced independently.

**Auto-levels (normalize to full [0,1] range):**
```c
vec4 lo = img_min(@A);
vec4 hi = img_max(@A);
@OUT = (@A - lo) / max(hi - lo, 0.001);
```

**Threshold above average brightness:**
```c
float avg = luma(img_mean(@A));
@OUT = step(avg, luma(@A));
```

### Type Promotion

- `int` promotes to `float` automatically
- `float` broadcasts to `vec3`/`vec4` (e.g., `@A * 0.5`)
- `vec3` promotes to `vec4` (alpha = 1.0)
- `string` does **not** coerce to/from numeric types (use `str()`, `to_int()`, `to_float()` explicitly)

## Node Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `code` | STRING | grayscale example | TEX source code |
| `output_type` | COMBO | auto | Output format. `auto` infers from code (vec3/vec4→IMAGE, float→MASK, string→STRING, latent inputs→LATENT). Manual: IMAGE, MASK, LATENT, FLOAT, INT, STRING |
| `device` | COMBO | auto | Execution device. `auto` follows input tensors, `cpu`/`cuda` forces a device |
| `compile_mode` | COMBO | none | `none`: standard interpreter. `torch_compile`: JIT-compile for speed |
| *(dynamic)* | ANY | — | Input slots auto-created from `@name` references in code (e.g. `@A`, `@base_image`) |

## Examples

See the `examples/` directory for complete snippets:

- `grayscale.tex` — luminance-based grayscale
- `brightness_contrast.tex` — adjustable brightness/contrast
- `threshold_mask.tex` — binary mask from luminance
- `vignette.tex` — radial vignette with strength parameter
- `color_mix.tex` — blend two images
- `channel_swap.tex` — swap color channels
- `gradient.tex` — procedural gradient generation
- `invert.tex` — color inversion
- `hue_shift.tex` — HSV-based hue rotation
- `conditional.tex` — warm/cool split by brightness
- `blur.tex` — box blur using for loops and `fetch()`
- `swirl.tex` — swirl distortion using `sample_cubic()`
- `perlin_clouds.tex` — procedural cloud pattern using FBM noise
- `latent_scale.tex` — scale latent intensity
- `latent_blend.tex` — blend two latents with a parameter
- `string_build.tex` — build a filename from string inputs
- `string_case.tex` — normalize tag casing
- `median_filter.tex` — 3x3 median filter using arrays and `sort()`
- `frame_blend.tex` — 3-frame temporal average for noise reduction
- `auto_levels.tex` — normalize image to full [0,1] range using image reductions
- `vec4_median.tex` — clean 3x3 median filter using vec4 arrays

## Architecture

```
TEX_Wrangle/
  __init__.py              # ComfyUI entry point
  tex_node.py              # Node class with device/compile_mode params
  tex_cache.py             # Two-tier compilation cache (memory LRU + disk pickle)
  tex_compiler/
    lexer.py               # Tokenizer
    parser.py              # Recursive-descent parser -> AST
    ast_nodes.py           # AST node definitions
    type_checker.py        # Static type analysis
    stdlib_signatures.py   # Function signatures for type checking
  tex_runtime/
    interpreter.py         # Tree-walking tensor evaluator
    stdlib.py              # Built-in function implementations (75+)
    compiled.py            # torch.compile wrapper with backend cascade
  js/
    tex_extension.js       # Frontend: auto-socket, syntax highlighting, help popup
  tests/
    test_tex.py            # 292-test suite
  examples/                # Example TEX snippets (21 files)
  .tex_cache/              # Disk cache directory (auto-created, gitignored)
```

## Performance

On CPU at 512×512 (grayscale conversion):
- First execution: ~3ms (includes compilation)
- Cached execution: ~0.7ms

Compilation results are stored in a two-tier Mega-Cache:
- **Memory:** LRU cache (128 entries) for instant re-execution
- **Disk:** Pickle-based persistence (512 entries) in `.tex_cache/`, survives ComfyUI restarts

With `torch_compile` enabled, first execution has a one-time tracing overhead (~1-5s), but subsequent runs benefit from fused tensor operations.

## Limitations

- **No scatter writes** — cannot write to arbitrary coordinates (`@A(x,y) = val`)
- **No histogram operations** — no per-image histogram computation
- **No CodeMirror editor** — uses standard ComfyUI textarea with syntax highlighting overlay

## Notes

- **Variable name `v`:** The built-in variable `v` (normalized y-coordinate) is always defined. If you declare `float v = ...;` it will shadow the built-in. Use a different name if you need the y-coordinate later.
- **torch.compile on Windows:** For full `inductor` backend support, install Visual Studio Build Tools with the "Desktop development with C++" workload. Without it, GPU falls back to `cudagraphs` and CPU runs in eager mode.

## Roadmap

**v1.5:** CodeMirror 6 editor, Triton kernel ops, histogram operations
**v2.0:** Direct Triton codegen, atomic scatter writes

## Running Tests

```bash
cd custom_nodes/TEX_Wrangle
python tests/test_tex.py
```

Expected: 292/292 passed.

## License

MIT
