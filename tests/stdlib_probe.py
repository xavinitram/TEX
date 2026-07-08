"""
Shared stdlib-call generator — the spine of the Phase-1 safety net.

Given a name from FUNCTION_SIGNATURES, `generate(name)` builds a valid TEX program
that calls it and a bindings dict, so TST-6 (parity), TST-2 (edge matrix), and
TST-1 (fuzzer) all derive their per-function coverage from the ONE signature
table — a new stdlib function is auto-covered the moment it's registered.

Non-numeric families (string / array / matrix / cross-frame) need bespoke fixtures
and are in SKIP; a meta-test asserts SKIP ∪ generated == every signature, so a new
function can't silently escape coverage.
"""
from helpers import make_img
from TEX_Wrangle.tex_compiler.stdlib_signatures import FUNCTION_SIGNATURES

# Bespoke-fixture families — parity covered by dedicated hand-written tests.
SKIP = frozenset({
    "transpose", "determinant", "inverse",                       # matrix args
    "fetch_frame", "sample_frame",                               # need a @frames batch
    # string ops
    "str", "len", "replace", "strip", "lower", "upper", "contains", "startswith",
    "endswith", "find", "substr", "to_int", "to_float", "sanitize_filename", "split",
    "lstrip", "rstrip", "pad_left", "pad_right", "format", "repeat", "str_reverse",
    "count", "matches", "hash", "hash_float", "hash_int", "char_at", "join",
    # array ops
    "sort", "reverse", "arr_sum", "arr_min", "arr_max", "median", "arr_avg",
    "debug_print",   # LX-5: string label + interpreter-only side-effect probe
})

_IMG_UV3 = {"sample", "fetch", "sample_cubic", "sample_lanczos"}      # (img, u, v) -> vec3
_IMG_UV4 = {"sample_mip", "sample_mip_gauss"}                         # (img, u, v, lod)
_NOISE2 = {"perlin", "simplex", "worley_f1", "worley_f2", "voronoi", "alligator"}
_NOISE_OCT = {"fbm", "ridged", "billow", "turbulence"}
_COMPOSITE = {"over", "under", "atop"}                                # (vec4, vec4) -> vec4
_UNARY_VEC4 = {"premultiply", "unpremultiply"}
_BLEND = {"screen", "overlay", "hard_light", "soft_light", "color_dodge",
          "color_burn", "linear_light", "vivid_light"}
_COLOR_VEC = {"hsv2rgb", "rgb2hsv", "srgb_to_linear", "linear_to_srgb",
              "oklab_from_rgb", "oklab_to_rgb", "normalize"}
_IMG_REDUCE = {"img_sum", "img_mean", "img_min", "img_max", "img_median"}

# Functions that may LEGITIMATELY produce non-finite output on the edge matrix
# (e.g. log(0) → -inf). The finiteness invariant is waived for these; cross-tier
# agreement (equal_nan) is still required.
ALLOW_NONFINITE = frozenset({
    "log", "log2", "log10",          # log(0) = -inf on 0-valued pixels
    "pow", "spow", "pow2", "pow10",  # pow with edge exponents
    "atan2", "sdiv", "mod",          # 0/0-ish edge combos
    "asin", "acos",                  # out-of-[-1,1] under NaN/Inf inputs → NaN
    "tan", "sinh", "cosh", "exp",    # overflow to inf on large/Inf inputs
    "sqrt", "hypot",
})


def _wrap(call, kind):
    return {
        "float": f"@OUT = vec4({call}, 0.0, 0.0, 1.0);",
        "vec2":  f"@OUT = vec4({call}, 0.0, 1.0);",
        "vec3":  f"@OUT = vec4({call}, 1.0);",
        "vec4":  f"@OUT = {call};",
    }[kind]


def _call_and_kind(name, lo):
    """Return (call_expr, output_kind) for `name`, or None if unhandled."""
    if name in _IMG_UV3:      return f"{name}(@A, u, v)", "vec3"
    if name in _IMG_UV4:      return f"{name}(@A, u, v, 1.0)", "vec3"
    if name == "gauss_blur":  return "gauss_blur(@A, 2.0)", "vec3"
    if name == "bilateral_filter": return "bilateral_filter(@A, 1.5, 0.2)", "vec3"
    if name in ("erode", "dilate"): return f"{name}(@A.rgb, 2)", "vec3"
    if name in _IMG_REDUCE:   return f"{name}(@A.rgb)", "vec3"
    if name == "sample_grad": return "sample_grad(@A, u, v)", "vec2"
    if name in _NOISE2:       return f"{name}(u*6.0, v*6.0)", "float"
    if name in _NOISE_OCT:    return f"{name}(u*6.0, v*6.0, 4)", "float"
    if name == "flow":        return "flow(u*6.0, v*6.0, 0.5)", "float"
    if name == "curl":        return "curl(u*6.0, v*6.0)", "vec2"
    if name in _COLOR_VEC:    return f"{name}(@A.rgb)", "vec3"
    if name == "luma":        return "luma(@A.rgb)", "float"
    if name in _UNARY_VEC4:   return f"{name}(vec4(@A.rgb, 0.5))", "vec4"
    if name in _COMPOSITE:    return f"{name}(vec4(@A.rgb, 0.7), vec4(@B.rgb, 0.5))", "vec4"
    if name in _BLEND:        return f"{name}(@A.rgb, @B.rgb)", "vec3"
    if name == "length":      return "length(@A.rgb)", "float"
    if name == "distance":    return "distance(@A.rgb, @B.rgb)", "float"
    if name == "dot":         return "dot(@A.rgb, @B.rgb)", "float"
    if name == "cross":       return "cross(@A.rgb, @B.rgb)", "vec3"
    if name == "reflect":     return "reflect(@A.rgb, @B.rgb)", "vec3"
    if name == "sincos":      return "sincos(u)", "vec2"
    if name == "sdf_circle":  return "sdf_circle(u, v, 0.3)", "float"
    if name == "sdf_box":     return "sdf_box(u, v, 0.3, 0.2)", "float"
    if name == "sdf_line":    return "sdf_line(u, v, 0.1, 0.1, 0.8, 0.8)", "float"
    if name == "sdf_polygon": return "sdf_polygon(u, v, 0.3, 5)", "float"
    # Generic numeric: a tensor first arg then scalar literals (mixing a tensor
    # into a later arg makes codegen fall back for a few fns like clamp — scalar
    # later-args keep codegen actually emitting). Values don't matter for cross-
    # tier parity — both tiers compute identically, NaN included.
    if lo <= 5:
        args = ["u", "0.5", "0.25", "0.75", "0.9"][:lo]
        return f"{name}({', '.join(args)})", "float"
    return None


def generate(name):
    """(program, bindings) that calls `name`, or None if skipped/unhandled."""
    if name in SKIP:
        return None
    ck = _call_and_kind(name, FUNCTION_SIGNATURES[name]["args"][0])
    if ck is None:
        return None
    return _wrap(ck[0], ck[1]), {"A": make_img(1, 8, 8, 3, seed=1),
                                 "B": make_img(1, 8, 8, 3, seed=2)}


def all_generated():
    """{name: (program, bindings)} for every non-skip signature we can build."""
    out = {}
    for name in FUNCTION_SIGNATURES:
        g = generate(name)
        if g is not None:
            out[name] = g
    return out
