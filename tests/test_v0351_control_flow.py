"""Per-pixel control flow: the documented rule, pinned on both tiers, and its opt-in advisories.

LANGUAGE.md §7.1 states what the engine does with a condition that can differ from pixel to
pixel: a per-pixel `if` runs BOTH branches on every pixel and merges them, and a `break`,
`continue` or `return` under one raises past that merge, so it acts on every pixel. These
rows hold the sentence to the engine on the interpreter AND codegen (a pin: green at the
base, red only if the semantics move without the doc), hold the doc text itself, and pin
`tex_api.control_flow_advisories` (W7006 / W7007):

  * it flags the shapes the rule is about — including the shipped examples that gather
    behind a per-pixel `if`;
  * it spares the uniform shapes (literals, scalar params, `iw`/`ih`/..., uniformly-bounded
    loop counters, arithmetic on those), so an opt-in host sees no noise;
  * it is INVISIBLE to `check()`: the editor's `/tex_wrangle/check` live lint and `tex_lsp`
    both go through `check()`, and no program gains a diagnostic there.
"""
import os

import torch

from helpers import *

from TEX_Wrangle import tex_api, tex_lsp
from TEX_Wrangle.tex_compiler.types import TEXType

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES = os.path.join(_ROOT, "examples")
_ADVISORY_CODES = ("W7006", "W7007")


def _read(*parts):
    with open(os.path.join(_ROOT, *parts), encoding="utf-8") as f:
        return f.read()


def _both_tiers(code, bindings):
    """(interp OUT, codegen OUT). Codegen declining is a failure: the rule is pinned on both."""
    interp, cg = run_both(code, bindings)
    if cg is None:
        raise AssertionError("codegen declined the program, so only one tier was measured")
    return interp["OUT"], cg["OUT"]


def _codes_by_line(source, binding_types=None):
    out = {}
    for d in tex_api.control_flow_advisories(source, binding_types or {}):
        out.setdefault(d.loc.line, []).append(d.code)
    return out


def _line_of(source, needle):
    for i, line in enumerate(source.splitlines(), 1):
        if needle in line:
            return i
    raise AssertionError(f"{needle!r} not found")


# ── the rule, on both tiers ──────────────────────────────────────────────────

# `COND` is FALSE on every pixel. A uniform condition takes one branch, so the `break`
# never runs and acc counts 3 passes; a per-pixel condition runs the branch on every pixel,
# so the `break` fires on the first pass for all of them and acc stays 0. The body reads @A
# so codegen lowers it as a tensor loop (the path every image program takes).
_BREAK_PROBE = """f$g = 0.5;
i$n = 1;
float lum = luma(@A);
float acc = 0.0;
for (int k = 0; k < 3; k++) {
    if (COND) { break; }
    acc = acc + 1.0 + @A.r * 0.0;
}
@OUT = vec3(acc);"""

_UNIFORM_FALSE = {
    "literal": "0.0 > 1.0",
    "scalar float param": "$g > 1.0",
    "scalar int param": "$n > 5",
    "iw": "iw < 0.0", "ih": "ih < 0.0", "px": "px > 1.0", "py": "py > 1.0",
    "fn": "fn > 100.0", "PI": "PI < 0.0", "TAU": "TAU < 0.0", "E": "E < 0.0",
    "loop counter": "k > 5",
    "arithmetic on uniforms": "float(k) * $g + iw / ih < 0.0",
}
_PER_PIXEL_FALSE = {
    "@ input": "@A.r > 2.0",
    "u": "u > 2.0", "v": "v > 2.0", "ix": "ix < 0.0", "iy": "iy < 0.0",
    "fi": "fi > 5.0",
    "reduction img_min": "img_min(@A).r > 2.0",
    "local computed from @": "lum > 2.0",
}


def test_control_flow_per_pixel_condition_semantics_both_tiers(r: SubTestResult):
    print("\n--- control flow: uniform vs per-pixel conditions, pinned on both tiers ---")
    torch.manual_seed(7)
    bind = {"A": torch.rand(1, 4, 5, 3), "g": 0.5, "n": 1}

    for label, cond, want in ([(k, c, 3.0) for k, c in _UNIFORM_FALSE.items()]
                              + [(k, c, 0.0) for k, c in _PER_PIXEL_FALSE.items()]):
        try:
            oi, oc = _both_tiers(_BREAK_PROBE.replace("COND", cond), bind)
            for tier, out in (("interp", oi), ("codegen", oc)):
                got = sorted(set(round(x, 4) for x in out[..., 0].flatten().tolist()))
                assert got == [want], f"{tier}: acc {got}, expected [{want}]"
            kind = "uniform: one branch" if want else "per-pixel: the break acts on every pixel"
            r.ok(f"{label} ({cond}) -> {kind}, both tiers")
        except Exception as e:
            r.fail(f"control-flow kind {label}", f"{type(e).__name__}: {e}")

    A = torch.zeros(1, 4, 4, 3)
    A[..., 2:, 0] = 1.0                            # right half r=1, left half r=0
    rows = [
        ("a per-pixel if merges each pixel's side",
         "float x = 0.0; if (@A.r > 0.5) { x = 1.0; } else { x = 2.0; } @OUT = vec3(x);",
         [2.0, 2.0, 1.0, 1.0]),
        ("an else-arm break acts on every pixel",
         "float acc = 0.0; for (int k = 0; k < 3; k++) { if (@A.r > -1.0) { acc = acc + 1.0; }"
         " else { break; } } @OUT = vec3(acc);",
         [0.0, 0.0, 0.0, 0.0]),
        ("a continue under a per-pixel if acts on every pixel",
         "float acc = 0.0; for (int k = 0; k < 3; k++) { acc = acc + 1.0 + @A.r * 0.0;"
         " if (@A.r > 2.0) { continue; } acc = acc + 10.0; } @OUT = vec3(acc);",
         [3.0, 3.0, 3.0, 3.0]),
        ("the assignments before a per-pixel break land on every pixel",
         "float x = 0.0; for (int k = 0; k < 3; k++) { if (@A.r > 2.0) { x = 5.0; break; } }"
         " @OUT = vec3(x);",
         [5.0, 5.0, 5.0, 5.0]),
        ("a break nested in a uniform if under a per-pixel if still acts on every pixel",
         "float acc = 0.0; for (int k = 0; k < 3; k++) { if (@A.r > 2.0) { if (1.0 > 0.0)"
         " { break; } } acc = acc + 1.0 + @A.r * 0.0; } @OUT = vec3(acc);",
         [0.0, 0.0, 0.0, 0.0]),
        ("a return under a per-pixel if returns for every pixel",
         "float f(float a) { if (a > 0.5) { return 7.0; } return 3.0; } @OUT = vec3(f(@A.r));",
         [7.0, 7.0, 7.0, 7.0]),
        ("a return under a uniform if takes one branch",
         "f$g = 0.5; float f(float a) { if (a > 0.9) { return 7.0; } return 3.0; }"
         " @OUT = vec3(f($g)) + @A * 0.0;",
         [3.0, 3.0, 3.0, 3.0]),
        ("a return under an fi condition returns for every pixel",
         "float f() { if (fi > 5.0) { return 7.0; } return 3.0; } @OUT = vec3(f()) + @A * 0.0;",
         [7.0, 7.0, 7.0, 7.0]),
        ("a continue in a loop nested inside a per-pixel if stays per-pass",
         "float acc = 0.0; if (@A.r > 0.5) { for (int k = 0; k < 4; k++) { if (k == 1)"
         " { continue; } acc = acc + 1.0; } } @OUT = vec3(acc);",
         [0.0, 0.0, 3.0, 3.0]),
    ]
    for label, code, want in rows:
        try:
            oi, oc = _both_tiers(code, {"A": A, "g": 0.5})
            for tier, out in (("interp", oi), ("codegen", oc)):
                got = [round(x, 4) for x in out[0, 0, :, 0].tolist()]
                assert got == want, f"{tier}: {got}, expected {want}"
            r.ok(f"{label}, both tiers")
        except Exception as e:
            r.fail(f"control-flow row: {label}", f"{type(e).__name__}: {e}")


def test_control_flow_language_md_states_the_rule(r: SubTestResult):
    print("\n--- control flow: LANGUAGE.md §7.1, DEVELOPMENT.md and the zdefocus comment ---")
    try:
        lang = _read("LANGUAGE.md")
        sec = lang[lang.index("### 7.1 Uniform and per-pixel conditions"):]
        sec = sec[:sec.index("\n---")]
        for needle in ("img_min(@A)", "runs **both** branches on every pixel",
                       "Assume `?:` evaluates both operands",
                       "`break`, `continue` and `return` under a per-pixel `if` act on **every** pixel",
                       "iw ih px py fn ic PI TAU E frame fps time",
                       "tex_api.control_flow_advisories", "**W7006**", "**W7007**",
                       "never reports"):
            assert needle in sec, f"LANGUAGE.md §7.1 lacks {needle!r}"
        r.ok("LANGUAGE.md §7.1 states the rule, names reductions and the opt-in advisories")
    except Exception as e:
        r.fail("LANGUAGE.md §7.1", f"{type(e).__name__}: {e}")
    try:
        dev = _read("DEVELOPMENT.md")
        assert "which short-circuits to the taken branch" in dev
        assert "raises past the merge, so it acts on every pixel" in dev
        r.ok("DEVELOPMENT.md names the uniform short-circuit and the escape past the merge")
    except Exception as e:
        r.fail("DEVELOPMENT.md control flow", f"{type(e).__name__}: {e}")
    try:
        zd = _read("examples", "zdefocus.tex")
        assert "Skip blur for in-focus pixels" not in zd, "zdefocus.tex still claims a skip"
        assert "the blur is not skipped" in zd
        r.ok("examples/zdefocus.tex no longer claims a per-pixel skip")
    except Exception as e:
        r.fail("zdefocus comment", f"{type(e).__name__}: {e}")


# ── W7006 ────────────────────────────────────────────────────────────────────

def test_control_flow_w7006_marks_gathers_in_per_pixel_branches(r: SubTestResult):
    print("\n--- control flow: W7006 on a per-pixel if/?: that gathers in a branch ---")
    shipped = {
        "zdefocus.tex": "if (coc < 0.5) {",
        "fix_pixels.tex": "if (is_bad(col) > 0.5) {",
        "binding_access.tex": "if (dist < 0.2) {",
        "matrix_transform.tex": "if (su >= 0.0 && su <= 1.0 && sv >= 0.0 && sv <= 1.0) {",
        "barrel_distortion.tex": "if (du < 0.0 || du > 1.0 || dv < 0.0 || dv > 1.0) {",
    }
    for name, needle in shipped.items():
        try:
            src = _read("examples", name)
            got = _codes_by_line(src).get(_line_of(src, needle), [])
            assert "W7006" in got, f"no W7006 on {needle!r}: {_codes_by_line(src)}"
            r.ok(f"W7006 on examples/{name}: {needle}")
        except Exception as e:
            r.fail(f"W7006 {name}", f"{type(e).__name__}: {e}")
    for name in ("erode_dilate.tex", "fast_defocus.tex"):
        try:
            src = _read("examples", name)
            got = [d.code for d in tex_api.control_flow_advisories(src, {})]
            assert not got, got
            r.ok(f"no advisory on examples/{name} (its skips test uniform values)")
        except Exception as e:
            r.fail(f"no advisory {name}", f"{type(e).__name__}: {e}")

    gathers = {
        "sample()": "@OUT = sample(@A, u, v);",
        "fetch()": "@OUT = fetch(@A, ix + 1, iy);",
        "paren sample": "@OUT = @A(u, v);",
        "bracket fetch": "@OUT = @A[ix, iy];",
        "a blur": "@OUT = gauss_blur(@A, 2.0);",
        "a reduction": "@OUT = img_max(@A).rgb;",
        "a user function holding a gather": "@OUT = pick(u);",
    }
    prelude = "vec3 pick(float x) { return sample(@A, x, 0.5).rgb; }\n"
    for label, stmt in gathers.items():
        code = prelude + f"@OUT = @A;\nif (@A.r > 0.5) {{\n    {stmt}\n}}\n"
        try:
            got = _codes_by_line(code).get(3, [])
            assert got == ["W7006"], _codes_by_line(code)
            r.ok(f"W7006 when a per-pixel branch holds {label}")
        except Exception as e:
            r.fail(f"W7006 {label}", f"{type(e).__name__}: {e}")
    cases = [
        ("a ternary with a gather in an operand",
         "vec3 c = (u > 0.5) ? sample(@A, u, v).rgb : vec3(0.0);\n@OUT = c;", {1: ["W7006"]}),
        ("a per-pixel if with no gather", "vec3 c = @A.rgb;\nif (u > 0.5) {\n    c = c * 2.0;\n}\n"
         "@OUT = c;", {}),
        ("a uniform if with a gather", "f$m = 1.0;\n@OUT = @A;\nif ($m > 0.5) {\n"
         "    @OUT = sample(@A, u, v);\n}\n", {}),
        ("a gather outside any branch", "vec3 s = sample(@A, u, v).rgb;\n"
         "vec3 c = (u > 0.5) ? s : vec3(0.0);\n@OUT = c;", {}),
    ]
    for label, code, want in cases:
        try:
            got = _codes_by_line(code)
            assert got == want, f"{got} != {want}"
            r.ok(f"W7006 as expected for {label}: {want or 'none'}")
        except Exception as e:
            r.fail(f"W7006 {label}", f"{type(e).__name__}: {e}")


# ── W7007 ────────────────────────────────────────────────────────────────────

# Minimal programs with the shapes the rule is about — among them the shapes the shipped
# example programs used to have, before they were rewritten to mean what they say.
_W7007_SHAPES = [
    ("a break under a per-pixel if (a row scan)",
     "int found = -1;\nfor (int sx = 0; sx < 8; sx++) {\n    if (luma(fetch(@A, sx, iy)) > 0.5) {\n"
     "        found = sx;\n        break;\n    }\n}\n@OUT = vec3(float(found));",
     {5: ["W7007"]}),
    ("a break on convergence (a per-pixel error)",
     "float g = @A.r;\nint it = 0;\nwhile (it < 20) {\n    float nx = g * 0.5;\n"
     "    float err = abs(nx - g);\n    g = nx;\n    it = it + 1;\n"
     "    if (err < 0.0001) {\n        break;\n    }\n}\n@OUT = vec3(g);",
     {9: ["W7007"]}),
    ("a return under a per-pixel if in a user function",
     "float ov(float a, float b) {\n    if (a < 0.5) {\n        return 2.0 * a * b;\n    }\n"
     "    return 1.0 - 2.0 * (1.0 - a) * (1.0 - b);\n}\n@OUT = vec3(ov(@A.r, 0.9));",
     {3: ["W7007"]}),
    ("returns in a helper fed a per-pixel argument",
     "float bad(vec3 c) {\n    if (isnan(c.r)) { return 1.0; }\n    return 0.0;\n}\n"
     "@OUT = vec3(bad(@A.rgb));",
     {2: ["W7007"]}),
    ("a per-pixel for bound",
     "int n = int(@A.r * 8.0);\nvec3 s = vec3(0.0);\nfor (int i = 0; i < n; i++) {\n"
     "    s += @A.rgb;\n}\n@OUT = s;",
     {3: ["W7007"]}),
    ("a per-pixel while condition",
     "float x = u;\nwhile (x < 1.0) {\n    x = x + 0.1;\n}\n@OUT = vec3(x);",
     {2: ["W7007"]}),
    ("a continue under a per-pixel if",
     "float acc = 0.0;\nfor (int k = 0; k < 3; k++) {\n    if (v > 0.5) {\n        continue;\n"
     "    }\n    acc += 1.0;\n}\n@OUT = vec3(acc);",
     {4: ["W7007"]}),
    ("a break in the else arm",
     "float acc = 0.0;\nfor (int k = 0; k < 3; k++) {\n    if (u > 0.5) { acc += 1.0; }"
     " else { break; }\n}\n@OUT = vec3(acc);",
     {3: ["W7007"]}),
    ("a break under a uniform if nested in a per-pixel if",
     "for (int k = 0; k < 3; k++) {\n    if (@A.r > 0.5) {\n        if (k == 1) {\n"
     "            break;\n        }\n    }\n}\n@OUT = @A;",
     {4: ["W7007"]}),
    ("a loop bound merged per pixel by an earlier pass",
     "int n = 2;\nint k = 0;\nwhile (k < n) {\n    if (@A.r > 0.5) { n = 4; }\n    k = k + 1;\n}\n"
     "@OUT = @A;",
     {3: ["W7007"]}),
    ("a vector parameter's component",
     "v3$tint = vec3(0.0, 0.0, 0.0);\nfor (int k = 0; k < 3; k++) {\n"
     "    if ($tint.r > 0.5) { break; }\n}\n@OUT = @A;",
     {3: ["W7007"]}),
    ("a condition on fi",
     "float f() {\n    if (fi > 0.5) { return 1.0; }\n    return 0.0;\n}\n@OUT = vec3(f()) + @A;",
     {2: ["W7007"]}),
    ("a reduction-gated break",
     "for (int k = 0; k < 3; k++) {\n    if (img_min(@A).r > 0.5) { break; }\n}\n@OUT = @A;",
     {2: ["W7007"]}),
]


def test_control_flow_w7007_marks_control_flow_on_every_pixel(r: SubTestResult):
    print("\n--- control flow: W7007 on control flow that acts on every pixel ---")
    for label, code, want in _W7007_SHAPES:
        try:
            got = _codes_by_line(code)
            assert got == want, f"{got} != {want}"
            r.ok(f"W7007 for {label}")
        except Exception as e:
            r.fail(f"W7007 {label}", f"{type(e).__name__}: {e}")
    try:
        src = _read("examples", "zdefocus.tex")
        loop_line = _line_of(src, "for (int i = 0; i < num_samples; i++) {")
        got = _codes_by_line(src)
        assert "W7007" not in got.get(loop_line, []), got
        assert not any("W7007" in v for v in got.values()), got
        r.ok("no W7007 on zdefocus.tex's uniform loop inside its per-pixel branch")
    except Exception as e:
        r.fail("W7007 zdefocus uniform loop", f"{type(e).__name__}: {e}")
    try:
        diag = tex_api.control_flow_advisories(_W7007_SHAPES[0][1], {})[0]
        assert diag.severity == "warning" and diag.loc.line == 5 and diag.end_col == diag.loc.col + 5
        assert "every pixel" in diag.render() and "W7007" in diag.render()
        r.ok("the diagnostic is a warning spanning the keyword, with a rendered message")
    except Exception as e:
        r.fail("W7007 diagnostic shape", f"{type(e).__name__}: {e}")


_UNIFORM_SHAPES = [
    ("literal conditions", "for (int k = 0; k < 3; k++) {\n    if (1.0 > 0.5) { break; }\n}\n@OUT = @A;"),
    ("scalar, int and string params",
     "f$gain = 0.5;\ni$count = 3;\ns$mode = \"add\";\nfor (int k = 0; k < $count; k++) {\n"
     "    if ($gain > 0.9 || $mode == \"screen\") { break; }\n}\n@OUT = @A;"),
    ("the uniform built-ins",
     "for (int k = 0; k < 3; k++) {\n    if (iw < 2.0 || ih < 2.0 || px > 1.0 || py > 1.0 || fn > 9.0"
     " || ic > 9.0 || PI < 3.0 || TAU < 6.0 || E < 2.0) { break; }\n}\n@OUT = @A;"),
    ("host time built-ins",
     "for (int k = 0; k < 3; k++) {\n    if (frame < 0.0 || fps < 0.0 || time < 0.0) { continue; }\n}\n@OUT = @A;"),
    ("loop counters and arithmetic on uniforms",
     "i$radius = 2;\nint r = abs($radius);\nfloat t2 = (float(r) + 0.5) * (float(r) + 0.5);\n"
     "vec3 acc = @A.rgb;\nfor (int dy = -r; dy <= r; dy++) {\n    for (int dx = -r; dx <= r; dx++) {\n"
     "        if (float(dx * dx + dy * dy) > t2) { continue; }\n"
     "        acc = max(acc, fetch(@A, ix + dx, iy + dy));\n    }\n}\n@OUT = acc;"),
    ("a uniform while bound", "i$max_iter = 20;\nint it = 0;\nfloat g = @A.r;\n"
     "while (it < $max_iter) {\n    g = g * 0.5;\n    it = it + 1;\n}\n@OUT = vec3(g);"),
    ("a uniform loop inside a per-pixel branch",
     "i$samples = 8;\nif (@A.r < 0.5) {\n    @OUT = @A;\n} else {\n    int n = clamp($samples, 1, 16);\n"
     "    vec3 acc = vec3(0.0);\n    for (int i = 0; i < n; i++) {\n        if (i == 3) { continue; }\n"
     "        acc += @A.rgb;\n    }\n    @OUT = acc;\n}"),
    ("a helper called with uniform arguments",
     "f$gain = 0.5;\nfloat sq(float x) {\n    if (x > 2.0) { return 1.0; }\n    return 0.0;\n}\n"
     "float y = sq($gain);\n@OUT = @A * y;"),
    ("recursion on a uniform depth",
     "float fr(float x, float y, int depth) {\n    if (depth <= 0) { return 0.0; }\n"
     "    return perlin(x, y) + fr(x, y, depth - 1);\n}\n@OUT = vec3(fr(u, v, 4));"),
    ("a local reassigned to a uniform value",
     "float n = @A.r;\nn = 4.0;\nfor (int k = 0; float(k) < n; k++) {\n    if (n > 9.0) { break; }\n}\n@OUT = @A;"),
    ("a variable first declared inside a per-pixel branch",
     "if (@A.r > 0.5) {\n    int m = 3;\n    for (int k = 0; k < m; k++) { if (k == m) { break; } }\n}\n"
     "@OUT = @A;"),
]


def test_control_flow_advisories_spare_uniform_shapes(r: SubTestResult):
    print("\n--- control flow: no advisory on uniform conditions and uniformly-bounded loops ---")
    for label, code in _UNIFORM_SHAPES:
        try:
            got = [(d.code, d.loc.line) for d in tex_api.control_flow_advisories(code, {})]
            assert not got, got
            r.ok(f"no advisory for {label}")
        except Exception as e:
            r.fail(f"uniform shape {label}", f"{type(e).__name__}: {e}")
    try:
        code = "for (int k = 0; k < 3; k++) {\n    if (@label == \"x\") { break; }\n}\n@OUT = @A;"
        assert [d.code for d in tex_api.control_flow_advisories(code, {"label": TEXType.STRING})] == []
        assert [d.code for d in tex_api.control_flow_advisories(code, {})] == ["W7007"]
        assert [d.code for d in tex_api.control_flow_advisories(
            code.replace("@label", "s@label"), {})] == []
        r.ok("a STRING wire is uniform (typed by the map or by its s@ prefix); an untyped one is not")
    except Exception as e:
        r.fail("string wire", f"{type(e).__name__}: {e}")


# ── invisibility: check() never sees them ────────────────────────────────────

def test_control_flow_advisories_are_invisible_to_check(r: SubTestResult):
    print("\n--- control flow: W7006/W7007 never reach check(), the live lint or the LSP ---")
    flagged = 0
    try:
        names = sorted(f for f in os.listdir(_EXAMPLES) if f.endswith(".tex"))
        assert len(names) > 100, len(names)
        for name in names:
            src = _read("examples", name)
            before = [d.to_dict() for d in tex_api.check(src, {})]
            adv = tex_api.control_flow_advisories(src, {})
            after = [d.to_dict() for d in tex_api.check(src, {})]
            assert before == after, f"{name}: check() changed after the advisories ran"
            leaked = [d["code"] for d in before if d["code"] in _ADVISORY_CODES]
            assert not leaked, f"{name}: check() reports {leaked}"
            flagged += bool(adv)
        assert flagged >= 5, f"only {flagged} examples flagged: the canary would be vacuous"
        r.ok(f"check() holds no W7006/W7007 for all {len(names)} examples, {flagged} of which "
             f"the advisories flag, and is unchanged by running them")
    except Exception as e:
        r.fail("check() invisibility", f"{type(e).__name__}: {e}")
    try:
        for _label, code, _want in _W7007_SHAPES:
            assert not [d for d in tex_api.check(code, {}) if d.code in _ADVISORY_CODES]
            lsp = [d.get("code") for d in tex_lsp.diagnostics_for(code, {})]
            assert not set(lsp) & set(_ADVISORY_CODES), lsp
        r.ok("check() and tex_lsp.diagnostics_for stay silent on every W7007 shape")
    except Exception as e:
        r.fail("LSP invisibility", f"{type(e).__name__}: {e}")
    try:
        for bad in ("", "@OUT = ;", "if (u > 0.5) { break; ", "\x00\xff{{{", "for (;;) {}"):
            assert tex_api.control_flow_advisories(bad, {}) == [], bad
        assert tex_api.control_flow_advisories(None, {}) == []
        good = _W7007_SHAPES[0][1]
        assert [d.code for d in tex_api.control_flow_advisories(good, None)] == ["W7007"]
        assert [d.code for d in tex_api.control_flow_advisories(good, "junk")] == ["W7007"]
        r.ok("total: parse failures and junk inputs return [], a missing map is {}")
    except Exception as e:
        r.fail("advisory totality", f"{type(e).__name__}: {e}")


# ── loops whose condition is per-pixel ───────────────────────────────────────

def test_control_flow_per_pixel_loop_bound_semantics_both_tiers(r: SubTestResult):
    """A per-pixel loop condition keeps the loop going while ANY pixel's condition holds and
    does not mask the body, so every pixel runs to the frame's maximum; a uniform bound with
    a guarded body gives each pixel its own count. A pin: green at the base, red only if the
    semantics move without LANGUAGE.md §7.1."""
    print("\n--- control flow: a per-pixel loop bound runs every pixel to the frame's maximum ---")
    A = torch.zeros(1, 2, 4, 3)
    A[..., 0] = torch.tensor([1.0, 2.0, 4.0, 0.0])      # per-column counts n = 1, 2, 4, 0
    rows = [
        ("a per-pixel for bound",
         "int n = int(@A.r);\nfloat acc = 0.0;\nfor (int i = 0; i < n; i++) { acc += 1.0; }\n"
         "@OUT = vec3(acc);", [4.0, 4.0, 4.0, 4.0]),
        ("a per-pixel while condition",
         "int n = int(@A.r);\nfloat acc = 0.0;\nint i = 0;\nwhile (i < n) { acc += 1.0; i = i + 1; }\n"
         "@OUT = vec3(acc);", [4.0, 4.0, 4.0, 4.0]),
        ("a uniform bound with a guarded body",
         "int n = int(@A.r);\nfloat acc = 0.0;\nfor (int i = 0; i < 4; i++) { if (i < n) { acc += 1.0; } }\n"
         "@OUT = vec3(acc);", [1.0, 2.0, 4.0, 0.0]),
        ("a uniform bound with a weighted body",
         "int n = int(@A.r);\nfloat acc = 0.0;\nfor (int i = 0; i < 4; i++) { acc += (i < n) ? 1.0 : 0.0; }\n"
         "@OUT = vec3(acc);", [1.0, 2.0, 4.0, 0.0]),
    ]
    for label, code, want in rows:
        try:
            oi, oc = _both_tiers(code, {"A": A})
            for tier, out in (("interp", oi), ("codegen", oc)):
                got = [round(x, 4) for x in out[0, 0, :, 0].tolist()]
                assert got == want, f"{tier}: {got}, expected {want}"
            r.ok(f"{label} -> {want}, both tiers")
        except Exception as e:
            r.fail(f"loop bound: {label}", f"{type(e).__name__}: {e}")
    try:
        from TEX_Wrangle.tex_compiler.diagnostics import TEXCompileError  # noqa: F401
        from TEX_Wrangle.tex_runtime.interpreter import InterpreterError
        for code in ("float a = 0.0; for (int i = 0; i < 1025; i++) { a += 1.0; } @OUT = vec3(a) + @A;",
                     "float a = 0.0; while (a < 5000.0) { a += 1.0; } @OUT = vec3(a) + @A;"):
            prog = tex_api.compile(code, {"A": TEXType.VEC3})
            try:
                tex_api.execute(prog, {"A": A})
            except InterpreterError as e:
                assert e._code == "E6010", (e._code, e)
            else:
                raise AssertionError(f"no E6010 for {code!r}")
        ok = tex_api.compile("float a = 0.0; for (int i = 0; i < 1024; i++) { a += 1.0; } "
                             "@OUT = vec3(a) + @A * 0.0;", {"A": TEXType.VEC3})
        assert tex_api.execute(ok, {"A": A})["OUT"][0, 0, 0, 0].item() == 1024.0
        r.ok("a loop that needs more than 1024 iterations fails with E6010; 1024 runs")
    except Exception as e:
        r.fail("loop cap E6010", f"{type(e).__name__}: {e}")


def test_control_flow_language_md_states_the_loop_bound(r: SubTestResult):
    print("\n--- control flow: LANGUAGE.md says what a per-pixel loop bound does ---")
    try:
        lang = _read("LANGUAGE.md")
        assert "static ranges for" not in lang, "LANGUAGE.md still claims static `for` ranges"
        assert "Every loop is capped\nat 1024 iterations" in lang.replace("\r\n", "\n")
        sec = lang[lang.index("### 7.1 Uniform and per-pixel conditions"):]
        sec = sec[:sec.index("\n---")]
        for needle in ("whose condition is per-pixel runs **every** pixel for as many passes as",
                       "does not mask the body",
                       "for (int i = 0; i < $max; i++) { if (i < n) { sum += tap; } }"):
            assert needle in sec.replace("\r\n", "\n").replace("\n  ", " "), f"§7.1 lacks {needle!r}"
        dev = _read("DEVELOPMENT.md")
        assert "keeps the loop running while any pixel's condition holds" in dev
        r.ok("LANGUAGE.md states the 1024 cap and the per-pixel loop bullet; DEVELOPMENT.md agrees")
    except Exception as e:
        r.fail("loop-bound docs", f"{type(e).__name__}: {e}")
