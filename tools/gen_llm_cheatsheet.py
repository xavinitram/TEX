"""S-3 — generate wiki/LLM-Cheatsheet.md: a dense, paste-into-any-LLM context block
that lets a model author a valid TEX program from a node-graph intent.

Like DOC-4's Function-Reference, this is a *view* of the single sources of truth — the
terse signatures come from the REG-1 registry + the editor `TEX_HELP_DATA` (reused via
`gen_function_reference.parse_help`), so it cannot drift off the code. The prose sections
(conventions, the "v is reserved" pitfall class, the worked examples) are the stable
authoring contract. The companion drift test (test_v019_phase3) asserts this file is
current AND that every worked example still compiles — so the cheatsheet can never teach
a convention the compiler has since rejected. Run: python tools/gen_llm_cheatsheet.py [--check].
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_OUT = os.path.join(_PKG, "wiki", "LLM-Cheatsheet.md")
sys.path.insert(0, _HERE)               # gen_function_reference (sibling) on path
sys.path.insert(0, os.path.dirname(os.path.dirname(_PKG)))  # custom_nodes on path


# The worked node-graph→TEX examples — the highest-value part for an LLM. Each MUST
# compile (the drift test compiles all of them), so the cheatsheet can never teach a
# program the compiler rejects. (intent, code).
WORKED = [
    ("Brightness / contrast grade (a Color Grade node)",
     "float b = $brightness;\n"
     "float c = $contrast;\n"
     "vec3 g = (@A.rgb - 0.5) * c + 0.5 + b;\n"
     "@OUT = vec4(clamp(g, 0.0, 1.0), 1.0);"),
    ("Vignette (radial darkening toward the edges)",
     "float d = distance(vec2(u, v), vec2(0.5, 0.5));\n"
     "float falloff = smoothstep(0.8, 0.3, d);\n"
     "@OUT = vec4(@A.rgb * falloff, 1.0);"),
    ("Channel swap + desaturate mix (a hue/mix node)",
     "float y = luma(@A);\n"
     "vec3 swapped = @A.bgr;\n"
     "@OUT = vec4(mix(vec3(y), swapped, 0.5), 1.0);"),
    ("Horizontal box blur via neighbour sampling",
     "vec3 acc = vec3(0.0);\n"
     "for (int i = -2; i <= 2; i = i + 1) {\n"
     "    acc = acc + sample(@A, u + float(i) / iw, v).rgb;\n"
     "}\n"
     "@OUT = vec4(acc / 5.0, 1.0);"),
]


def _signature_block():
    """Terse per-category signature lines, sourced from the registry + editor help."""
    import gen_function_reference as G
    entries, cats = G.parse_help()
    reg_names = {n for e in G.R.REGISTRY for n in e.names}
    lines = []
    for cat, names in cats:
        sigs = [G._unescape(entries[n]["sig"]) for n in names if n in reg_names and n in entries]
        if sigs:
            lines.append(f"- **{cat}:** " + "  ·  ".join(f"`{s}`" for s in sigs))
    return lines, len(reg_names)


def render():
    sigs, n_fns = _signature_block()
    out = [
        "# TEX Wrangle — LLM authoring cheatsheet",
        "",
        "> Paste this whole page into an LLM to have it write TEX programs. **Generated** "
        "(`tools/gen_llm_cheatsheet.py`) from the registry — do not edit by hand.",
        "",
        "## What TEX is",
        "",
        "A GLSL-like per-pixel expression language for ComfyUI. You write one program; it "
        "runs once per pixel over image tensors `[B,H,W,C]`. No `main()`, no explicit loop "
        "over pixels — the per-pixel loop is implicit. Statements end with `;`.",
        "",
        "## Bindings (how data gets in and out)",
        "",
        "- **Inputs:** `@A`, `@B`, … `@H` — the wired IMAGE/MASK/LATENT sockets. `@A.rgb` is a "
        "`vec3`, `@A.r` a `float`, `@A` a `vec4` (a MASK reads as a scalar).",
        "- **Output:** assign `@OUT` (an IMAGE wants a `vec3`/`vec4`; the result is clamped to "
        "`[0,1]` and the 3-channel RGB is what leaves the node). Extra outputs: `@name = expr`.",
        "- **Parameters:** `$name` creates a float widget on the node (e.g. `$contrast`). Great "
        "for anything a user should tweak without editing code.",
        "",
        "## Built-in variables",
        "",
        "- `ix`, `iy` — integer pixel coordinates. `iw`, `ih` — image width/height (floats).",
        "- `u`, `v` — normalized coordinates in `[0,1]` (`u` = x/width, `v` = y/height).",
        "- `PI`, `E` — the constants.",
        "- **⚠ `v` is reserved** (the normalized y-coordinate). This is the #1 pitfall: do NOT "
        "name a variable `v` — it silently shadows the built-in. Same care for `u`/`ix`/`iy`.",
        "",
        "## Types & operators",
        "",
        "- Scalars `float`/`int`; vectors `vec2`/`vec3`/`vec4`; matrices `mat3`/`mat4`; also "
        "`string` and arrays. Swizzle like GLSL: `.xyz`, `.rgb`, `.bgr`, `.xy`, `.wzyx`.",
        "- Arithmetic `+ - * /`, comparisons, `&&`/`||`/`!`. Vectors operate component-wise; "
        "`vec3 * float` broadcasts. `mat3 * vec3` is a transform.",
        "- Control flow: `if (cond) { … } else { … }` (vectorized per-pixel), `for`/`while` "
        "with `break`/`continue`. Loop bounds should be compile-time-bounded for speed.",
        "",
        "## Pitfalls (the failure classes)",
        "",
        "1. **`v` (and `u`/`ix`/`iy`) are reserved built-ins** — never use them as variable names.",
        "2. **IMAGE output is clamped to `[0,1]` and RGB-only** — return `vec3`/`vec4`; alpha and "
        "out-of-range values are dropped/clamped.",
        "3. **Divide safely:** raw `/` by a possibly-zero value gives `NaN` (paints magenta with "
        "`debug_nan_highlight`). Use `sdiv(a, b)` (returns 0 when `b≈0`) when the denominator "
        "can vanish.",
        "4. **Spatial reads use `sample(@A, u, v)`** — indexing another pixel means sampling by "
        "normalized coordinate, not array indexing. Offsets are in `1/iw`, `1/ih` units.",
        "5. **`precision=\"auto\"` is accuracy-safe, not a speedup** — leave `precision=fp32` "
        "unless you know you want the fp16 path.",
        "",
        f"## Function signatures ({n_fns} stdlib functions)",
        "",
        *sigs,
        "",
        "## Worked examples (node intent → TEX)",
        "",
        "Each of these compiles as-is.",
        "",
    ]
    for intent, code in WORKED:
        out.append(f"**{intent}**")
        out.append("")
        out.append("```glsl")
        out.append(code)
        out.append("```")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main():
    check = "--check" in sys.argv
    content = render()
    if check:
        try:
            existing = open(_OUT, encoding="utf-8").read()
        except FileNotFoundError:
            print("LLM-Cheatsheet.md missing — run tools/gen_llm_cheatsheet.py")
            return 1
        if existing != content:
            print("LLM-Cheatsheet.md is stale — regenerate with tools/gen_llm_cheatsheet.py")
            return 1
        print("LLM-Cheatsheet.md up to date")
        return 0
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"wrote {_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
