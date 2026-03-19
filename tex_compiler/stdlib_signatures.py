"""
TEX Standard Library — function signatures for type checking.

Each entry maps a function name to:
  - args: (min_count, max_count)
  - return: TEXType or callable(arg_types) -> TEXType

The callable form allows return type to depend on argument types
(e.g., clamp returns the type of its first argument).
"""
from __future__ import annotations
from .type_checker import TEXType


def _passthrough_type(arg_types: list[TEXType]) -> TEXType:
    """Return the type of the first argument (with promotion for mixed)."""
    if not arg_types:
        return TEXType.FLOAT
    return arg_types[0]


def _promote_args(arg_types: list[TEXType]) -> TEXType:
    """Return the promoted type across all args."""
    if not arg_types:
        return TEXType.FLOAT
    result = arg_types[0]
    for t in arg_types[1:]:
        if result != t:
            if result.is_scalar and t.is_vector:
                result = t
            elif result.is_vector and t.is_scalar:
                pass  # keep vector
            elif result == TEXType.INT and t == TEXType.FLOAT:
                result = TEXType.FLOAT
            elif result == TEXType.FLOAT and t == TEXType.INT:
                pass
            elif {result, t} == {TEXType.VEC3, TEXType.VEC4}:
                result = TEXType.VEC4
    return result


# Function signatures: name -> {args: (min, max), return: type_or_callable}
FUNCTION_SIGNATURES: dict[str, dict] = {
    # Math — scalar or element-wise on vectors
    "sin":       {"args": (1, 1), "return": _passthrough_type},
    "cos":       {"args": (1, 1), "return": _passthrough_type},
    "tan":       {"args": (1, 1), "return": _passthrough_type},
    "asin":      {"args": (1, 1), "return": _passthrough_type},
    "acos":      {"args": (1, 1), "return": _passthrough_type},
    "atan":      {"args": (1, 1), "return": _passthrough_type},
    "atan2":     {"args": (2, 2), "return": _passthrough_type},
    "sqrt":      {"args": (1, 1), "return": _passthrough_type},
    "pow":       {"args": (2, 2), "return": _passthrough_type},
    "exp":       {"args": (1, 1), "return": _passthrough_type},
    "log":       {"args": (1, 1), "return": _passthrough_type},
    "abs":       {"args": (1, 1), "return": _passthrough_type},
    "sign":      {"args": (1, 1), "return": _passthrough_type},
    "floor":     {"args": (1, 1), "return": _passthrough_type},
    "ceil":      {"args": (1, 1), "return": _passthrough_type},
    "round":     {"args": (1, 1), "return": _passthrough_type},
    "fract":     {"args": (1, 1), "return": _passthrough_type},
    "mod":       {"args": (2, 2), "return": _passthrough_type},

    "log2":      {"args": (1, 1), "return": _passthrough_type},
    "log10":     {"args": (1, 1), "return": _passthrough_type},
    "pow2":      {"args": (1, 1), "return": _passthrough_type},
    "pow10":     {"args": (1, 1), "return": _passthrough_type},
    "sinh":      {"args": (1, 1), "return": _passthrough_type},
    "cosh":      {"args": (1, 1), "return": _passthrough_type},
    "tanh":      {"args": (1, 1), "return": _passthrough_type},
    "hypot":     {"args": (2, 2), "return": _passthrough_type},
    "isnan":     {"args": (1, 1), "return": lambda _: TEXType.FLOAT},  # always returns float (0.0/1.0)
    "isinf":     {"args": (1, 1), "return": lambda _: TEXType.FLOAT},  # always returns float (0.0/1.0)
    "degrees":   {"args": (1, 1), "return": _passthrough_type},
    "radians":   {"args": (1, 1), "return": _passthrough_type},
    "spow":      {"args": (2, 2), "return": _passthrough_type},        # safe power — sign(x)*pow(abs(x),y)
    "sdiv":      {"args": (2, 2), "return": _passthrough_type},        # safe division — 0 when b≈0

    # Clamping and interpolation
    "min":       {"args": (2, 2), "return": _promote_args},
    "max":       {"args": (2, 2), "return": _promote_args},
    "clamp":     {"args": (3, 3), "return": _passthrough_type},
    "lerp":      {"args": (3, 3), "return": _promote_args},
    "mix":       {"args": (3, 3), "return": _promote_args},     # alias for lerp
    "fit":       {"args": (5, 5), "return": _passthrough_type},  # fit(val, old_min, old_max, new_min, new_max)
    "smoothstep": {"args": (3, 3), "return": _passthrough_type},
    "step":      {"args": (2, 2), "return": _passthrough_type},

    # Vector operations — always return float (scalar result)
    "dot":       {"args": (2, 2), "return": lambda _: TEXType.FLOAT},
    "length":    {"args": (1, 1), "return": lambda _: TEXType.FLOAT},
    "distance":  {"args": (2, 2), "return": lambda _: TEXType.FLOAT},

    # Vector operations — return vector
    "normalize": {"args": (1, 1), "return": _passthrough_type},
    "cross":     {"args": (2, 2), "return": lambda _: TEXType.VEC3},
    "reflect":   {"args": (2, 2), "return": _passthrough_type},

    # Matrix operations
    "transpose":   {"args": (1, 1), "return": _passthrough_type},                  # transpose(mat) — transpose matrix
    "determinant": {"args": (1, 1), "return": lambda _: TEXType.FLOAT},            # determinant(mat) — scalar
    "inverse":     {"args": (1, 1), "return": _passthrough_type},                  # inverse(mat) — matrix inverse

    # Color operations
    "luma":      {"args": (1, 1), "return": lambda _: TEXType.FLOAT},  # luminance of vec3/vec4
    "hsv2rgb":   {"args": (1, 1), "return": _passthrough_type},
    "rgb2hsv":   {"args": (1, 1), "return": _passthrough_type},

    # Sampling
    "sample":         {"args": (3, 3), "return": lambda _: TEXType.VEC4},  # sample(@A, u, v) — bilinear
    "fetch":          {"args": (3, 3), "return": lambda _: TEXType.VEC4},  # fetch(@A, px, py) — nearest neighbor
    "sample_cubic":   {"args": (3, 3), "return": lambda _: TEXType.VEC4},  # sample_cubic(@A, u, v) — Catmull-Rom
    "sample_lanczos": {"args": (3, 3), "return": lambda _: TEXType.VEC4},  # sample_lanczos(@A, u, v) — Lanczos-3

    # Cross-frame sampling (temporal)
    "fetch_frame":    {"args": (4, 4), "return": lambda _: TEXType.VEC4},  # fetch_frame(@A, frame, px, py) — nearest from specific frame
    "sample_frame":   {"args": (4, 4), "return": lambda _: TEXType.VEC4},  # sample_frame(@A, frame, u, v) — bilinear from specific frame

    # Noise
    "perlin":    {"args": (2, 2), "return": lambda _: TEXType.FLOAT},  # perlin(x, y) — 2D Perlin noise
    "simplex":   {"args": (2, 2), "return": lambda _: TEXType.FLOAT},  # simplex(x, y) — 2D Simplex noise
    "fbm":       {"args": (3, 3), "return": lambda _: TEXType.FLOAT},  # fbm(x, y, octaves) — Fractional Brownian Motion

    # String operations
    "str":               {"args": (1, 1), "return": TEXType.STRING},              # str(number) — number to string
    "len":               {"args": (1, 1), "return": TEXType.FLOAT},               # len(s) — string length
    "replace":           {"args": (3, 4), "return": TEXType.STRING},              # replace(s, old, new, max_count?) — replace
    "strip":             {"args": (1, 1), "return": TEXType.STRING},              # strip(s) — trim whitespace
    "lower":             {"args": (1, 1), "return": TEXType.STRING},              # lower(s) — to lowercase
    "upper":             {"args": (1, 1), "return": TEXType.STRING},              # upper(s) — to uppercase
    "contains":          {"args": (2, 2), "return": TEXType.FLOAT},               # contains(s, sub) — 1.0/0.0
    "startswith":        {"args": (2, 2), "return": TEXType.FLOAT},               # startswith(s, prefix) — 1.0/0.0
    "endswith":          {"args": (2, 2), "return": TEXType.FLOAT},               # endswith(s, suffix) — 1.0/0.0
    "find":              {"args": (2, 2), "return": TEXType.FLOAT},               # find(s, sub) — index or -1.0
    "substr":            {"args": (2, 3), "return": TEXType.STRING},              # substr(s, start, len?) — extract
    "to_int":            {"args": (1, 1), "return": TEXType.INT},                 # to_int(s) — parse integer
    "to_float":          {"args": (1, 1), "return": TEXType.FLOAT},               # to_float(s) — parse float
    "sanitize_filename": {"args": (1, 1), "return": TEXType.STRING},              # sanitize_filename(s) — clean path
    "split":             {"args": (2, 3), "return": lambda _: TEXType.ARRAY},    # split(s, delim, max?) — split string
    "lstrip":            {"args": (1, 1), "return": TEXType.STRING},             # lstrip(s) — trim leading whitespace
    "rstrip":            {"args": (1, 1), "return": TEXType.STRING},             # rstrip(s) — trim trailing whitespace
    "pad_left":          {"args": (2, 3), "return": TEXType.STRING},             # pad_left(s, width, char?) — left-pad
    "pad_right":         {"args": (2, 3), "return": TEXType.STRING},             # pad_right(s, width, char?) — right-pad
    "format":            {"args": (1, 16), "return": TEXType.STRING},            # format(template, ...args) — interpolation
    "repeat":            {"args": (2, 2), "return": TEXType.STRING},             # repeat(s, count) — repeat N times
    "str_reverse":       {"args": (1, 1), "return": TEXType.STRING},             # str_reverse(s) — reverse string
    "count":             {"args": (2, 2), "return": TEXType.FLOAT},              # count(s, sub) — count occurrences
    "matches":           {"args": (2, 2), "return": TEXType.FLOAT},              # matches(s, pattern) — regex match
    "hash":              {"args": (1, 1), "return": TEXType.STRING},             # hash(s) — deterministic SHA-256 prefix
    "hash_float":        {"args": (1, 1), "return": TEXType.FLOAT},             # hash_float(s) — deterministic float [0,1)
    "hash_int":          {"args": (1, 2), "return": TEXType.INT},               # hash_int(s, max?) — deterministic integer
    "char_at":           {"args": (2, 2), "return": TEXType.STRING},            # char_at(s, i) — character at index

    # Array operations
    "sort":      {"args": (1, 1), "return": lambda _: TEXType.ARRAY},             # sort(arr) — ascending sort, returns new array
    "reverse":   {"args": (1, 1), "return": lambda _: TEXType.ARRAY},             # reverse(arr) — reverse order, returns new array
    "arr_sum":   {"args": (1, 1), "return": lambda _: TEXType.FLOAT},             # arr_sum(arr) — sum of elements
    "arr_min":   {"args": (1, 1), "return": lambda _: TEXType.FLOAT},             # arr_min(arr) — minimum element
    "arr_max":   {"args": (1, 1), "return": lambda _: TEXType.FLOAT},             # arr_max(arr) — maximum element
    "median":    {"args": (1, 1), "return": lambda _: TEXType.FLOAT},             # median(arr) — median element
    "arr_avg":   {"args": (1, 1), "return": lambda _: TEXType.FLOAT},             # arr_avg(arr) — mean of elements

    # String array operations
    "join":      {"args": (2, 2), "return": TEXType.STRING},                      # join(arr, sep) — concatenate with separator

    # Image reductions
    "img_sum":    {"args": (1, 1), "return": _passthrough_type},                  # img_sum(@A) — sum of all pixels per channel
    "img_mean":   {"args": (1, 1), "return": _passthrough_type},                  # img_mean(@A) — mean of all pixels per channel
    "img_min":    {"args": (1, 1), "return": _passthrough_type},                  # img_min(@A) — min per channel
    "img_max":    {"args": (1, 1), "return": _passthrough_type},                  # img_max(@A) — max per channel
    "img_median": {"args": (1, 1), "return": _passthrough_type},                  # img_median(@A) — median per channel
}
