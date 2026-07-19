"""
TOOL-1 / STOCK-1 — generate the shipped stock `.textool` exemplars into `stock/`.

The first stock-node library, dogfooding strategic bet #8 ("any stock node expressible in
TEX must be one"): Grade, Blur, Merge, Vignette as single-stage tools, plus one fused
2-stage composite (GradeVignette) that exercises the fused round-trip. Stage code is
embedded here (not read from examples/) so a shipped tool is a stable, reviewed artifact
independent of any examples/ edit. Each manifest is validated + preflighted through
tex_tool before it is written, so a broken exemplar can never ship.

Run:   python tools/gen_stock_tools.py            # write stock/*.textool
       python tools/gen_stock_tools.py --check    # drift check (CI): fail if stale
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))  # custom_nodes on path

from TEX_Wrangle import tex_tool  # noqa: E402
from TEX_Wrangle import __version__  # noqa: E402

_OUT_DIR = os.path.join(ROOT, "stock")
_LANG = "0.23"

# ── stage sources ────────────────────────────────────────────────────────────────
_GRADE = """// Grade — Nuke-style lift / gamma / gain / offset color grade.
f$blackpoint = 0.0;
f$whitepoint = 1.0;
f$black = 0.0;
f$white = 1.0;
f$multiply = 1.0;
f$offset = 0.0;
f$gamma = 1.0;
f$saturation = 1.0;
f$mix = 1.0;

vec3 src = @image;
float A = $multiply * ($white - $black) / max($whitepoint - $blackpoint, 0.0001);
float B = $offset + $black - A * $blackpoint;
vec3 linear = src * A + B;
float ex = 1.0 / max($gamma, 0.0001);
float r = linear.r; float g = linear.g; float b = linear.b;
r = r < 0.0 ? r : (r > 1.0 ? (r - 1.0) * ex + 1.0 : spow(r, ex));
g = g < 0.0 ? g : (g > 1.0 ? (g - 1.0) * ex + 1.0 : spow(g, ex));
b = b < 0.0 ? b : (b > 1.0 ? (b - 1.0) * ex + 1.0 : spow(b, ex));
r = max(r, 0.0); g = max(g, 0.0); b = max(b, 0.0);
vec3 graded = vec3(r, g, b);
float lum = luma(graded);
vec3 result = lerp(vec3(lum), graded, $saturation);
@OUT = lerp(src, result, $mix);
"""

_BLUR = """// Blur — separable Gaussian (stdlib gauss_blur, radius ~= 3*sigma px).
f$sigma = 2.0;
@OUT = gauss_blur(@image, $sigma);
"""

_MERGE = """// Merge — blend two images with a selectable mode.
// 0=add 1=subtract 2=multiply 3=screen 4=overlay 5=soft-light 6=min 7=max 8=difference 9=divide
i$operation = 0;
f$mix = 1.0;
vec3 a = @A;
vec3 b = @B;
vec3 result = b;
float eps = 1e-6;
if ($operation == 0) { result = a + b; }
else if ($operation == 1) { result = a - b; }
else if ($operation == 2) { result = a * b; }
else if ($operation == 3) { result = a + b - a * b; }
else if ($operation == 4) {
    float rr = (b.r < 0.5) ? 2.0 * a.r * b.r : 1.0 - 2.0 * (1.0 - a.r) * (1.0 - b.r);
    float rg = (b.g < 0.5) ? 2.0 * a.g * b.g : 1.0 - 2.0 * (1.0 - a.g) * (1.0 - b.g);
    float rb = (b.b < 0.5) ? 2.0 * a.b * b.b : 1.0 - 2.0 * (1.0 - a.b) * (1.0 - b.b);
    result = vec3(rr, rg, rb);
}
else if ($operation == 5) {
    float sr = (1.0 - 2.0 * a.r) * b.r * b.r + 2.0 * a.r * b.r;
    float sg = (1.0 - 2.0 * a.g) * b.g * b.g + 2.0 * a.g * b.g;
    float sb = (1.0 - 2.0 * a.b) * b.b * b.b + 2.0 * a.b * b.b;
    result = vec3(sr, sg, sb);
}
else if ($operation == 6) { result = vec3(min(a.r, b.r), min(a.g, b.g), min(a.b, b.b)); }
else if ($operation == 7) { result = vec3(max(a.r, b.r), max(a.g, b.g), max(a.b, b.b)); }
else if ($operation == 8) { result = vec3(abs(a.r - b.r), abs(a.g - b.g), abs(a.b - b.b)); }
else if ($operation == 9) { result = vec3(a.r / max(b.r, eps), a.g / max(b.g, eps), a.b / max(b.b, eps)); }
@OUT = b + (result - b) * $mix;
"""

_VIGNETTE = """// Vignette — darken image edges with a soft radial falloff.
f$strength = 1.0;
float cx = u - 0.5;
float cy = v - 0.5;
float dist = hypot(cx, cy);
float falloff = 1.0 - smoothstep(0.3, 0.7, dist * $strength);
@darkened = @image * falloff;
m@vignette_mask = falloff;
"""

# Composite (fused, linear 2-stage): a single-output grade feeding a single-output vignette.
_GV_GRADE = """// Grade stage — gain + gamma (single output for chaining).
f$gain = 1.0;
f$gamma = 1.0;
vec3 c = @image * $gain;
float ex = 1.0 / max($gamma, 0.0001);
@OUT = vec3(spow(c.r, ex), spow(c.g, ex), spow(c.b, ex));
"""

_GV_VIGNETTE = """// Vignette terminal stage — single output.
f$strength = 1.0;
float cx = u - 0.5;
float cy = v - 0.5;
float falloff = 1.0 - smoothstep(0.3, 0.7, hypot(cx, cy) * $strength);
@OUT = @image * falloff;
"""


def _p(name, default, mn, mx, label, internal=None, stage=None, thint="f", step=None):
    meta = {"min": mn, "max": mx, "label": label}
    if step is not None:
        meta["step"] = step
    return {"name": name, "internal": internal or name, "stage": stage,
            "type": thint, "default": default, "metadata": meta}


def _manifests() -> dict:
    grade = {
        "manifest_schema": 1, "name": "Grade", "tool_version": "1.0.0",
        "tex_language": _LANG, "min_engine": __version__, "category": "Color",
        "context": "filter", "author": "TEX",
        "doc": "Nuke-style lift / gamma / gain / offset color grade.",
        "code": _GRADE,
        "inputs": [{"name": "image", "type": "IMAGE"}],
        "outputs": [{"name": "OUT", "type": "IMAGE"}],
        "promoted_params": [
            _p("blackpoint", 0.0, 0.0, 1.0, "Black Point"),
            _p("whitepoint", 1.0, 0.0, 2.0, "White Point"),
            _p("black", 0.0, -1.0, 1.0, "Lift"),
            _p("white", 1.0, 0.0, 4.0, "Gain"),
            _p("multiply", 1.0, 0.0, 4.0, "Multiply"),
            _p("offset", 0.0, -1.0, 1.0, "Offset"),
            _p("gamma", 1.0, 0.01, 4.0, "Gamma"),
            _p("saturation", 1.0, 0.0, 2.0, "Saturation"),
            _p("mix", 1.0, 0.0, 1.0, "Mix"),
        ],
    }
    blur = {
        "manifest_schema": 1, "name": "Blur", "tool_version": "1.0.0",
        "tex_language": _LANG, "min_engine": __version__, "category": "Filter",
        "context": "filter", "author": "TEX",
        "doc": "Separable Gaussian blur (radius ~= 3*sigma pixels).",
        "code": _BLUR,
        "inputs": [{"name": "image", "type": "IMAGE"}],
        "outputs": [{"name": "OUT", "type": "IMAGE"}],
        "promoted_params": [_p("sigma", 2.0, 0.0, 50.0, "Sigma", step=0.1)],
    }
    merge = {
        "manifest_schema": 1, "name": "Merge", "tool_version": "1.0.0",
        "tex_language": _LANG, "min_engine": __version__, "category": "Compositing",
        "context": "filter", "author": "TEX",
        "doc": "Blend @A and @B with a selectable mode (add/subtract/multiply/screen/overlay/"
               "soft-light/min/max/difference/divide). Not alpha compositing -- no @A.a channel.",
        "code": _MERGE,
        "inputs": [{"name": "A", "type": "IMAGE"}, {"name": "B", "type": "IMAGE"}],
        "outputs": [{"name": "OUT", "type": "IMAGE"}],
        "promoted_params": [
            _p("operation", 0, 0, 9, "Operation", thint="i", step=1),
            _p("mix", 1.0, 0.0, 1.0, "Mix"),
        ],
    }
    vignette = {
        "manifest_schema": 1, "name": "Vignette", "tool_version": "1.0.0",
        "tex_language": _LANG, "min_engine": __version__, "category": "Effects",
        "context": "filter", "author": "TEX",
        "doc": "Darken image edges with a soft radial falloff (image + mask outputs).",
        "code": _VIGNETTE,
        "inputs": [{"name": "image", "type": "IMAGE"}],
        "outputs": [{"name": "darkened", "type": "IMAGE"},
                    {"name": "vignette_mask", "type": "MASK"}],
        "promoted_params": [_p("strength", 1.0, 0.0, 2.0, "Strength")],
    }
    # Fused composite — linear 2-stage: grade (upstream) -> vignette (terminal).
    grade_vignette = {
        "manifest_schema": 1, "name": "GradeVignette", "tool_version": "1.0.0",
        "tex_language": _LANG, "min_engine": __version__, "category": "Look",
        "context": "filter", "author": "TEX",
        "doc": "A fused look: gain/gamma grade feeding a radial vignette (one compiled block).",
        "graphspec": {
            "schema": 1,
            "stages": [{"code": _GV_GRADE, "image_input": "image", "params": {}}],
            "terminal_image_input": "image",
        },
        "terminal_code": _GV_VIGNETTE,
        "terminal_image_input": "image",
        "terminal_params": {},
        "inputs": [{"name": "image", "type": "IMAGE"}],
        "outputs": [{"name": "OUT", "type": "IMAGE"}],
        "promoted_params": [
            _p("gain", 1.0, 0.0, 4.0, "Gain", stage=0),
            _p("gamma", 1.0, 0.01, 4.0, "Gamma", stage=0),
            _p("strength", 1.0, 0.0, 2.0, "Vignette", stage="terminal"),
        ],
    }
    return {"grade": grade, "blur": blur, "merge": merge, "vignette": vignette,
            "grade_vignette": grade_vignette}


def build() -> dict:
    """Validate + preflight every manifest, return {stem: json_text}. Raises if any is broken."""
    out = {}
    for stem, raw in _manifests().items():
        tex_tool.validate_manifest(raw)
        m = tex_tool.load_tool(raw)
        pf = tex_tool.preflight_tool(m)
        if not pf["ok"]:
            msg = pf["diagnostics"][0].get("message") if pf["diagnostics"] else "unknown"
            raise SystemExit(f"stock tool '{stem}' failed preflight: {msg}")
        out[stem] = json.dumps(raw, indent=2, ensure_ascii=False) + "\n"
    return out


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    check = "--check" in argv
    built = build()
    os.makedirs(_OUT_DIR, exist_ok=True)
    stale = []
    for stem, text in built.items():
        path = os.path.join(_OUT_DIR, stem + ".textool")
        cur = None
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                cur = fh.read()
        if cur != text:
            stale.append(stem)
            if not check:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(text)
    if check and stale:
        raise SystemExit(f"stock tools out of date: {stale} — run python tools/gen_stock_tools.py")
    print(("OK (up to date): " if check else "wrote: ") + ", ".join(sorted(built)))


if __name__ == "__main__":
    main()
