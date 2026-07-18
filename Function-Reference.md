# TEX Function Reference

> **Generated** by `tools/gen_function_reference.py` from `TEX_HELP_DATA` +
> the REG-1 registry + `FUNCTION_SIGNATURES`. Do not edit by hand — edit the
> source and regenerate. The drift test (`test_doc4_reference`) keeps it current.

## Math

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `sin` | `sin(x) → float` | Sine (radians). | — |
| `cos` | `cos(x) → float` | Cosine (radians). | — |
| `tan` | `tan(x) → float` | Tangent (radians). | — |
| `asin` | `asin(x) → float` | Arcsine. Returns radians. | — |
| `acos` | `acos(x) → float` | Arccosine. Returns radians. | — |
| `atan` | `atan(x) → float` | Arctangent. Returns radians. | — |
| `atan2` | `atan2(y, x) → float` | Two-argument arctangent. Returns radians. | — |
| `sincos` | `sincos(x) → vec2` | Returns vec2(sin(x), cos(x)). More efficient than separate sin/cos calls. | — |
| `sinh` | `sinh(x) → float` | Hyperbolic sine. | — |
| `cosh` | `cosh(x) → float` | Hyperbolic cosine. | — |
| `tanh` | `tanh(x) → float` | Hyperbolic tangent. | — |
| `pow` | `pow(x, y) → float` | Raise x to the power y. | — |
| `sqrt` | `sqrt(x) → float` | Square root. | — |
| `exp` | `exp(x) → float` | e raised to the power x. | — |
| `log` | `log(x) → float` | Natural logarithm (base e). | — |
| `log2` | `log2(x) → float` | Logarithm base 2. | — |
| `log10` | `log10(x) → float` | Logarithm base 10. | — |
| `abs` | `abs(x) → float` | Absolute value. | — |
| `sign` | `sign(x) → float` | Returns -1, 0, or 1. | — |
| `pow2` | `pow2(x) → float` | 2 raised to the power x. | — |
| `pow10` | `pow10(x) → float` | 10 raised to the power x. | — |
| `hypot` | `hypot(x, y) → float` | Hypotenuse: sqrt(x*x + y*y). | — |
| `floor` | `floor(x) → float` | Round down to nearest integer. | — |
| `ceil` | `ceil(x) → float` | Round up to nearest integer. | — |
| `round` | `round(x) → float` | Round to nearest integer. | — |
| `trunc` | `trunc(x) → float` | Truncate toward zero (drop fractional part). | — |
| `fract` | `fract(x) → float` | Fractional part: x - floor(x). | — |
| `mod` | `mod(x, y) → float` | Modulo (remainder). | — |
| `degrees` | `degrees(x) → float` | Convert radians to degrees. | — |
| `radians` | `radians(x) → float` | Convert degrees to radians. | — |
| `spow` | `spow(x, y) → float` | Sign-preserving power. Safe for negative x. | — |
| `sdiv` | `sdiv(a, b) → float` | Safe divide. Returns 0 when b is zero. | — |
| `isnan` | `isnan(x) → float` | Returns 1.0 if x is NaN, 0.0 otherwise. | — |
| `isinf` | `isinf(x) → float` | Returns 1.0 if x is infinite, 0.0 otherwise. | — |

## Interpolation

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `min` | `min(a, b) → float` | Returns the smaller value. | — |
| `max` | `max(a, b) → float` | Returns the larger value. | — |
| `clamp` | `clamp(x, lo, hi) → float` | Clamp x to [lo, hi] range. | — |
| `lerp` | `lerp(a, b, t) → float` | Linear interpolation from a to b by t. | — |
| `mix` | `mix(a, b, t) → float` | Linear interpolation from a to b by t. | — |
| `fit` | `fit(x, inLo, inHi, outLo, outHi) → float` | Remap x from [inLo, inHi] to [outLo, outHi]. | — |
| `step` | `step(edge, x) → float` | Returns 0 if x < edge, 1 otherwise. | — |
| `smoothstep` | `smoothstep(lo, hi, x) → float` | Smooth Hermite interpolation between lo and hi. | — |

## Vector

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `dot` | `dot(a, b) → float` | Dot product of two vectors. | — |
| `length` | `length(v) → float` | Length (magnitude) of a vector. | — |
| `distance` | `distance(a, b) → float` | Distance between two points. | — |
| `normalize` | `normalize(v) → vec` | Unit vector in the same direction. | — |
| `cross` | `cross(a, b) → vec3` | Cross product of two vec3 vectors. | — |
| `reflect` | `reflect(v, n) → vec` | Reflect vector v around normal n. | — |

## Color

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `luma` | `luma(rgb) → float` | Perceptual luminance of an RGB color. | — |
| `hsv2rgb` | `hsv2rgb(hsv) → vec3` | Convert HSV color to RGB. | — |
| `rgb2hsv` | `rgb2hsv(rgb) → vec3` | Convert RGB color to HSV. | — |
| `srgb_to_linear` | `srgb_to_linear(c) → vec` | Gamma-encoded sRGB → linear-light. Blur/blend in linear to avoid halos. | — |
| `linear_to_srgb` | `linear_to_srgb(c) → vec` | Linear-light → gamma-encoded sRGB (inverse of srgb_to_linear). | — |
| `oklab_from_rgb` | `oklab_from_rgb(c) → vec3` | Linear RGB → OKLab. Mix/interpolate in OKLab for perceptually-even gradients. | — |
| `oklab_to_rgb` | `oklab_to_rgb(lab) → vec3` | OKLab → linear RGB (inverse of oklab_from_rgb). | — |
| `premultiply` | `premultiply(rgba) → vec4` | Straight → premultiplied alpha (rgb *= a). | — |
| `unpremultiply` | `unpremultiply(rgba) → vec4` | Premultiplied → straight alpha (rgb /= a). | — |
| `over` | `over(fg, bg) → vec4` | Porter-Duff 'over': composite fg atop bg (straight-alpha RGBA). | — |
| `under` | `under(fg, bg) → vec4` | Composite fg under bg (= over(bg, fg)). | — |
| `atop` | `atop(fg, bg) → vec4` | 'atop': fg confined to bg's coverage. | — |
| `screen` | `screen(a, b) → vec` | Screen blend: 1 - (1-a)(1-b). Brightens. | — |
| `overlay` | `overlay(a, b) → vec` | Overlay blend (multiply/screen by base). | — |
| `hard_light` | `hard_light(a, b) → vec` | Hard-light blend (overlay with operands swapped). | — |
| `soft_light` | `soft_light(a, b) → vec` | Soft-light blend (Pegtop, smooth). | — |
| `color_dodge` | `color_dodge(a, b) → vec` | Color-dodge: brightens base by blend. | — |
| `color_burn` | `color_burn(a, b) → vec` | Color-burn: darkens base by blend. | — |
| `linear_light` | `linear_light(a, b) → vec` | Linear-light blend: clamp(a + 2b - 1). | — |
| `vivid_light` | `vivid_light(a, b) → vec` | Vivid-light blend (burn/dodge by blend). | — |

## Sampling

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `sample` | `sample(img, u, v) → vec` | Bilinear sample at normalized UV coordinates. | spatial, non-local |
| `fetch` | `fetch(img, px, py) → vec` | Nearest-neighbor fetch at pixel coordinates. | spatial, non-local |
| `sample_cubic` | `sample_cubic(img, u, v) → vec` | Bicubic (Catmull-Rom) sampling. | spatial, non-local |
| `sample_lanczos` | `sample_lanczos(img, u, v) → vec` | Lanczos-3 high-quality sampling. | spatial, non-local |
| `sample_mip` | `sample_mip(img, u, v, lod) → vec` | Mipmap sampling with LOD. 0 = full res, 1 = half, etc. Trilinear between levels. | spatial, sync, non-local |
| `sample_mip_gauss` | `sample_mip_gauss(img, u, v, lod) → vec` | Gaussian-prefiltered mipmap sampling. Smoother pyramid (sigma=1.13) gives ~5 dB better exponential blur accuracy vs sample_mip. | spatial, sync, non-local |
| `gauss_blur` | `gauss_blur(img, sigma) → vec` | Separable Gaussian blur. Kernel radius ≈ 3×sigma pixels. Replicate border padding. | spatial, sync, non-local |
| `bilateral_filter` | `bilateral_filter(img, spatial_sigma, range_sigma) → vec` | Edge-preserving smoothing: blurs within regions but keeps edges. Window capped at 7×7. | spatial, sync, non-local |
| `erode` | `erode(img, radius) → vec` | Morphological erosion (local min over a (2r+1)² square). Shrinks bright regions. | sync, non-local |
| `dilate` | `dilate(img, radius) → vec` | Morphological dilation (local max). Grows bright regions. | sync, non-local |
| `fetch_frame` | `fetch_frame(img, frame, px, py) → vec` | Nearest-neighbor fetch from a specific batch frame. | spatial, non-local |
| `sample_frame` | `sample_frame(img, frame, u, v) → vec` | Bilinear sample from a specific batch frame. | spatial, non-local |
| `sample_grad` | `sample_grad(img, u, v) → vec2` | Image gradient (Sobel) at UV. Returns vec2(dI/dx, dI/dy) of luminance. | spatial, non-local |

## Noise

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `perlin` | `perlin(x, y) → float` | 2D Perlin noise. Returns value in [-1, 1]. | — |
| `simplex` | `simplex(x, y) → float` | 2D Simplex noise. Returns value in [-1, 1]. | — |
| `fbm` | `fbm(x, y, octaves) → float` | Fractal Brownian Motion (multi-octave Perlin). | sync |
| `worley_f1` | `worley_f1(x, y) → float` | Worley (cellular) noise — distance to nearest cell center. | — |
| `worley_f2` | `worley_f2(x, y) → float` | Worley noise — distance to second-nearest cell center. | — |
| `voronoi` | `voronoi(x, y) → float` | Voronoi cell ID noise. Returns a unique value per cell. | — |
| `billow` | `billow(x, y, octaves) → float` | Billowy noise — abs(fbm). Puffy cloud shapes. | sync |
| `turbulence` | `turbulence(x, y, octaves) → float` | Turbulence — sum of abs(noise) per octave. Veiny patterns. | sync |
| `ridged` | `ridged(x, y, octaves) → float` | Ridged multifractal — sharp ridges, good for mountains. | sync |
| `flow` | `flow(x, y, angle) → float` | Flow noise — Perlin rotated by angle per octave. Avoids static patterns. | sync |
| `curl` | `curl(x, y) → vec2` | Curl of 2D noise field. Returns a divergence-free vector. | — |
| `alligator` | `alligator(x, y) → float` | Alligator noise — cellular crack patterns. | sync |

## SDF & Smooth

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `sdf_circle` | `sdf_circle(px, py, radius) → float` | Signed distance to a circle centered at the origin (offset px/py to move it). Negative inside, positive outside. | — |
| `sdf_box` | `sdf_box(px, py, half_w, half_h) → float` | Signed distance to an axis-aligned box centered at the origin (half-extents half_w/half_h). | — |
| `sdf_line` | `sdf_line(x, y, x1, y1, x2, y2) → float` | Distance to line segment. | — |
| `sdf_polygon` | `sdf_polygon(px, py, radius, sides) → float` | Signed distance to a regular polygon (sides>=3) centered at the origin. | — |
| `smin` | `smin(a, b, k) → float\|vec` | Smooth minimum. Polynomial blending with radius k. Works on scalars and vectors. | — |
| `smax` | `smax(a, b, k) → float\|vec` | Smooth maximum. Polynomial blending with radius k. Works on scalars and vectors. | — |

## Image Stats

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `img_min` | `img_min(img) → vec` | Per-channel minimum across the entire image. | non-local |
| `img_max` | `img_max(img) → vec` | Per-channel maximum across the entire image. | non-local |
| `img_mean` | `img_mean(img) → vec` | Per-channel mean (average) of the image. | non-local |
| `img_sum` | `img_sum(img) → vec` | Per-channel sum of all pixel values. | non-local |
| `img_median` | `img_median(img) → vec` | Per-channel median of the image. | non-local |

## Strings

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `str` | `str(x) → string` | Convert a number to a string. | — |
| `replace` | `replace(s, old, new) → string` | Replace all occurrences of old with new. | — |
| `strip` | `strip(s) → string` | Remove leading/trailing whitespace. | — |
| `lower` | `lower(s) → string` | Convert to lowercase. | — |
| `upper` | `upper(s) → string` | Convert to uppercase. | — |
| `contains` | `contains(s, sub) → float` | Returns 1.0 if s contains sub, 0.0 otherwise. | — |
| `startswith` | `startswith(s, prefix) → float` | Returns 1.0 if s starts with prefix. | — |
| `endswith` | `endswith(s, suffix) → float` | Returns 1.0 if s ends with suffix. | — |
| `find` | `find(s, sub) → float` | Index of first occurrence, or -1.0 if not found. | — |
| `substr` | `substr(s, start, len?) → string` | Extract a substring. len is optional. | — |
| `len` | `len(x) → float` | Length of a string, array, or vec-array (element count). | — |
| `repeat` | `repeat(s, n) → string` | Repeat a string N times. | — |
| `str_reverse` | `str_reverse(s) → string` | Reverse a string. | — |
| `matches` | `matches(s, pattern) → float` | Returns 1.0 if the whole string matches the regex pattern, else 0.0. | — |
| `hash` | `hash(s) → string` | Deterministic string hash (SHA-256 hex, first 16 chars). | — |
| `hash_float` | `hash_float(s) → float` | Deterministic hash of a string to a float in [0, 1). | — |
| `hash_int` | `hash_int(s, max?) → int` | Deterministic hash of a string to a non-negative int (optional exclusive max). | — |
| `to_int` | `to_int(s) → int` | Parse a string as an integer. | — |
| `to_float` | `to_float(s) → float` | Parse a string as a float. | — |
| `sanitize_filename` | `sanitize_filename(s) → string` | Remove unsafe characters for use in file paths. | — |
| `format` | `format(fmt, ...) → string` | Printf-style formatting. %d = int, %f = float, %s = string. | — |
| `split` | `split(s, sep) → string[]` | Split string into array by separator. | — |
| `lstrip` | `lstrip(s) → string` | Remove leading whitespace. | — |
| `rstrip` | `rstrip(s) → string` | Remove trailing whitespace. | — |
| `pad_left` | `pad_left(s, width, fill) → string` | Pad string on the left to reach width. | — |
| `pad_right` | `pad_right(s, width, fill) → string` | Pad string on the right to reach width. | — |
| `count` | `count(s, sub) → float` | Count non-overlapping occurrences of sub in s. | — |
| `char_at` | `char_at(s, idx) → string` | Character at index (0-based). | — |

## Arrays

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `sort` | `sort(arr) → array` | Sort array elements in ascending order. | — |
| `reverse` | `reverse(arr) → array` | Reverse array element order. | — |
| `arr_sum` | `arr_sum(arr) → float` | Sum of all array elements. | — |
| `arr_min` | `arr_min(arr) → float` | Minimum value in array. | — |
| `arr_max` | `arr_max(arr) → float` | Maximum value in array. | — |
| `median` | `median(arr) → float` | Median value of array. | — |
| `arr_avg` | `arr_avg(arr) → float` | Average of all array elements. | — |
| `join` | `join(arr, sep) → string` | Concatenate string array with separator. | — |

## Matrix

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `transpose` | `transpose(m) → mat` | Transpose a matrix. | — |
| `determinant` | `determinant(m) → float` | Compute the determinant. | — |
| `inverse` | `inverse(m) → mat` | Compute the matrix inverse. | — |

## Batch / Temporal

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `fetch_frame` | `fetch_frame(img, frame, px, py) → vec` | Nearest-neighbor fetch from a specific batch frame. | spatial, non-local |
| `sample_frame` | `sample_frame(img, frame, u, v) → vec` | Bilinear sample from a specific batch frame. | spatial, non-local |

## Debugging

| Function | Signature | Description | Tags |
|----------|-----------|-------------|------|
| `debug_print` | `debug_print(label, value[, x, y]) → value` | Probe a value at a pixel — records it for the node's HUD and returns the value unchanged (a print-style debug tap). Interpreter-only; a compiled tier falls back so the probe always fires. | sync |
