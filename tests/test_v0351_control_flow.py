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
_ADVISORY_CODES = ("W7006", "W7007", "W7008")


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
    # TRK-25: a per-pixel loop BOUND also draws W7008 — it is the half of W7007 whose answer
    # depends on which region was cooked, so the engine declines to split the cook for it.
    ("a per-pixel for bound",
     "int n = int(@A.r * 8.0);\nvec3 s = vec3(0.0);\nfor (int i = 0; i < n; i++) {\n"
     "    s += @A.rgb;\n}\n@OUT = s;",
     {3: ["W7007", "W7008"]}),
    ("a per-pixel while condition",
     "float x = u;\nwhile (x < 1.0) {\n    x = x + 0.1;\n}\n@OUT = vec3(x);",
     {2: ["W7007", "W7008"]}),
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
     {3: ["W7007", "W7008"]}),
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


# ── the shipped examples that used to exit or bound per pixel ────────────────
#
# Each example is measured against a per-pixel reference computed here in plain Python —
# one pixel at a time, with an ordinary early `break`/`return` — so the reference cannot
# share the engine's control flow. Both tiers must match it.

import math  # noqa: E402

_EPS_FLOAT = 1e-5


def _grid(t):
    """Batch 0 of a [1,H,W,C] tensor as nested Python floats [H][W][C]."""
    return t[0].tolist()


def _clamped(img, y, x):
    return img[min(max(y, 0), len(img) - 1)][min(max(x, 0), len(img[0]) - 1)]


def _luma(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def _lerp(a, b, t):
    return a + (b - a) * t


def _max_abs_diff(out, ref):
    """Largest |out - ref| over the reference's channels (a NaN anywhere counts as inf)."""
    got = out[0, ..., :len(ref[0][0])]
    d = (got.double() - torch.tensor(ref, dtype=torch.float64)).abs()
    return float("inf") if torch.isnan(d).any() else d.max().item()


def _measure(src, bindings, ref):
    """(interp maxdiff, codegen maxdiff, interp OUT, codegen OUT) against the reference."""
    oi, oc = _both_tiers(src, bindings)
    return _max_abs_diff(oi, ref), _max_abs_diff(oc, ref), oi, oc


def _ref_fix_pixels(img, fallback):
    def bad(c):
        return any(math.isnan(x) or math.isinf(x) for x in c)
    out = []
    for y, row in enumerate(img):
        orow = []
        for x, c in enumerate(row):
            if not bad(c[:3]):
                orow.append(c[:3])
                continue
            acc, n = [0.0, 0.0, 0.0], 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nb = _clamped(img, y + dy, x + dx)[:3]
                    if not bad(nb):
                        acc = [a + b for a, b in zip(acc, nb)]
                        n += 1
            orow.append([a / n for a in acc] if n else list(fallback))
        out.append(orow)
    return out


def _ref_break_search(img, threshold, scan_width):
    H, W = len(img), len(img[0])
    width = scan_width if scan_width > 0 else W
    out = []
    for y in range(H):
        found = -1
        for sx in range(min(width, W)):
            if _luma(img[y][sx]) > threshold:
                found = sx
                break
        orow = []
        for x in range(W):
            s = img[y][x]
            if found >= 0 and x < found:
                orow.append([s[0] * 0.5 + 0.3, s[1] * 0.3, s[2] * 0.3])
            elif found >= 0 and x == found:
                orow.append([0.0, 1.0, 0.0])
            else:
                orow.append(s[:3])
        out.append(orow)
    return out


def _ref_overlay(a, b):
    if a < 0.5:
        return 2.0 * a * b
    return 1.0 - 2.0 * (1.0 - a) * (1.0 - b)


def _ref_soft_light(a, b):
    if b < 0.5:
        return a - (1.0 - 2.0 * b) * a * (1.0 - a)
    d = ((16.0 * a - 12.0) * a + 4.0) * a if a < 0.25 else math.sqrt(a)
    return a + (2.0 * b - 1.0) * (d - a)


def _ref_blend(base, over, fn, t):
    return [[[_lerp(a, fn(a, b), t) if t is not None else fn(a, b) for a, b in zip(ca, cb)]
             for ca, cb in zip(ra, rb)] for ra, rb in zip(base, over)]


def _ref_newton(img, tolerance, max_iter):
    H, W = len(img), len(img[0])
    out = []
    for y in range(H):
        orow = []
        for xi in range(W):
            x = _luma(img[y][xi])
            guess = max(x, 0.001)
            for _ in range(max_iter):
                nxt = (guess + x / max(guess, 0.0001)) * 0.5
                err = abs(nxt - guess)
                guess = nxt
                if err < tolerance:
                    break
            if xi / max(W - 1, 1) < 0.5:
                orow.append([guess] * 3)
            else:
                orow.append([abs(guess - math.sqrt(x)) * 100.0, 0.0, 0.0])
        out.append(orow)
    return out


def _bilinear(img, u, v):
    """sample(): bilinear, pixel centres at u = ix / (W - 1), clamped to the border."""
    H, W = len(img), len(img[0])
    fx, fy = u * (W - 1), v * (H - 1)
    x0, y0 = min(int(math.floor(fx)), W - 1), min(int(math.floor(fy)), H - 1)
    x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
    ax, ay = fx - x0, fy - y0
    return [_lerp(_lerp(img[y0][x0][c], img[y0][x1][c], ax),
                  _lerp(img[y1][x0][c], img[y1][x1][c], ax), ay) for c in range(3)]


def _ref_vector_blur(img, vec, strength, max_samples):
    H, W = len(img), len(img[0])
    out = []
    for y in range(H):
        orow = []
        for x in range(W):
            u, v = x / max(W - 1, 1), y / max(H - 1, 1)
            dx = (vec[y][x][0] - 0.5) * 2.0 * strength
            dy = (vec[y][x][1] - 0.5) * 2.0 * strength
            n = min(max(int(math.floor(math.hypot(dx, dy) * 0.5 + 1.0)), 3), max_samples)
            steps = float(n - 1)
            step_u = (dx / W) / steps if abs(steps) >= 1e-8 else 0.0
            step_v = (dy / H) / steps if abs(steps) >= 1e-8 else 0.0
            acc = [0.0, 0.0, 0.0]
            for i in range(n):                     # this pixel's own count, and no more
                t = i - steps * 0.5
                su = min(max(u + step_u * t, 0.0), 1.0)
                sv = min(max(v + step_v * t, 0.0), 1.0)
                acc = [a + b for a, b in zip(acc, _bilinear(img, su, sv))]
            orow.append([a / n for a in acc])
        out.append(orow)
    return out


# Every case: (label, bindings, reference, tolerance). The same tables drive a measurement
# of any version of an example's source, so a before/after reading uses identical inputs.

def _fix_pixels_cases():
    torch.manual_seed(42)
    clean = torch.rand(1, 8, 8, 3)
    zero = (0.0, 0.0, 0.0)
    torch.manual_seed(3)
    img = torch.rand(1, 7, 9, 3)
    img[0, 0, 0, 1] = float("nan")                  # a corner, clamped neighbours
    img[0, 3, 6, 0] = float("inf")
    img[0, 5, 2, 2] = float("-inf")
    img[0, 1:4, 2:5, :] = float("nan")              # a 3x3 block: its centre has no valid neighbour
    fb = (0.25, 0.5, 0.75)
    return [
        ("a clean image passes through",
         {"image": clean, "fallback_r": 0.0, "fallback_g": 0.0, "fallback_b": 0.0},
         _ref_fix_pixels(_grid(clean), zero), _EPS_FLOAT),
        ("NaN/Inf pixels take their valid neighbours' mean, a fully-bad neighbourhood the fallback",
         {"image": img, "fallback_r": fb[0], "fallback_g": fb[1], "fallback_b": fb[2]},
         _ref_fix_pixels(_grid(img), fb), _EPS_FLOAT),
    ]


def _break_search_cases():
    one = torch.full((1, 8, 8, 3), 0.3)
    one[0, 3, 5, :] = 1.0                           # the one bright pixel: row 3, column 5
    torch.manual_seed(11)
    img = torch.rand(1, 6, 10, 3)
    margin = min(abs(_luma(c) - 0.6) for row in _grid(img) for c in row)
    assert margin > 1e-4, f"a luma sits on the threshold ({margin}): pick another seed"
    return [("one bright pixel at column 5", {"image": one, "threshold": 0.5, "scan_width": 0},
             _ref_break_search(_grid(one), 0.5, 0), _EPS_FLOAT)] + [
        (f"a random image, scan_width={w}", {"image": img, "threshold": 0.6, "scan_width": w},
         _ref_break_search(_grid(img), 0.6, w), _EPS_FLOAT) for w in (0, 4, 25)]


def _custom_blend_cases():
    halves = torch.zeros(1, 4, 8, 3)
    halves[..., :4, :] = 0.2
    halves[..., 4:, :] = 0.8
    over9 = torch.full((1, 4, 8, 3), 0.9)
    torch.manual_seed(5)
    base, over = torch.rand(1, 5, 7, 3), torch.rand(1, 5, 7, 3)
    return [
        ("base 0.2 | 0.8 under overlay 0.9, blend 0.5",
         {"base": halves, "overlay": over9, "blend_amount": 0.5},
         _ref_blend(_grid(halves), _grid(over9), _ref_overlay, 0.5), _EPS_FLOAT),
        ("random base and overlay, blend 0.7", {"base": base, "overlay": over, "blend_amount": 0.7},
         _ref_blend(_grid(base), _grid(over), _ref_overlay, 0.7), _EPS_FLOAT),
    ]


def _while_loop_cases():
    const = torch.full((1, 4, 8, 3), 0.36)
    vals = torch.tensor([0.0, 0.0004, 0.02, 0.15, 0.36, 0.5, 0.81, 1.0])
    ramp = vals.view(1, 1, 8, 1).expand(1, 3, 8, 3).contiguous()
    ramp = torch.cat([ramp, ramp.flip(2)], dim=1)   # each luma lands in both halves
    # The right half is |error| x 100, so a float32-vs-float64 rounding of 1e-7 reads 1e-5.
    return [
        ("luma 0.36", {"image": const, "tolerance": 0.0001, "max_iter": 20},
         _ref_newton(_grid(const), 0.0001, 20), 1e-4),
        ("lumas that converge after different step counts",
         {"image": ramp, "tolerance": 0.0001, "max_iter": 20},
         _ref_newton(_grid(ramp), 0.0001, 20), 1e-4),
    ]


def _vector_blur_cases():
    ones = torch.ones(1, 4, 8, 3)
    one_vec = torch.full((1, 4, 8, 3), 0.5)
    one_vec[0, 2, 5, 0] = 1.0                       # one 11-tap vector among zero motion (3 taps)
    torch.manual_seed(9)
    img = torch.rand(1, 6, 9, 3)
    vec = 0.5 + (torch.rand(1, 6, 9, 3) - 0.5) * 0.9
    vec[0, :2, :3, :2] = 0.5                        # a still patch among streaks
    counts = [math.hypot((c[0] - 0.5) * 30.0, (c[1] - 0.5) * 30.0) * 0.5 + 1.0
              for row in _grid(vec) for c in row]
    assert all(abs(f - round(f)) > 1e-3 or abs(f - 1.0) < 1e-9 for f in counts), \
        "a tap count sits on an integer boundary: pick another seed"
    return [
        ("a constant image with one long vector",
         {"image": ones, "vectors": one_vec, "strength": 20.0, "max_samples": 32},
         _ref_vector_blur(_grid(ones), _grid(one_vec), 20.0, 32), _EPS_FLOAT),
        ("random streaks of 3 to 12 taps",
         {"image": img, "vectors": vec, "strength": 15.0, "max_samples": 12},
         _ref_vector_blur(_grid(img), _grid(vec), 15.0, 12), _EPS_FLOAT),
    ]


def _check_example(r, name, cases, extra=None):
    src = _read("examples", name)
    for label, bindings, ref, tol in cases():
        try:
            di, dc, oi, oc = _measure(src, bindings, ref)
            assert di <= tol and dc <= tol, f"maxdiff interp {di:.3g}, codegen {dc:.3g} > {tol}"
            if extra is not None:
                extra(label, oi, oc)
            r.ok(f"examples/{name}: {label} (maxdiff interp {di:.1e}, codegen {dc:.1e})")
        except Exception as e:
            r.fail(f"{name}: {label}", f"{type(e).__name__}: {e}")


def test_control_flow_fix_pixels_matches_a_per_pixel_reference(r: SubTestResult):
    print("\n--- examples/fix_pixels.tex against a per-pixel reference, both tiers ---")

    def extra(label, oi, oc):
        assert torch.isfinite(oi).all() and torch.isfinite(oc).all(), "a NaN/Inf survived"
    _check_example(r, "fix_pixels.tex", _fix_pixels_cases, extra)


def test_control_flow_break_search_matches_a_per_pixel_reference(r: SubTestResult):
    print("\n--- examples/break_search.tex against a per-pixel reference, both tiers ---")
    _check_example(r, "break_search.tex", _break_search_cases)


def test_control_flow_custom_blend_matches_a_per_pixel_reference(r: SubTestResult):
    print("\n--- examples/custom_blend.tex against a per-pixel reference, both tiers ---")
    _check_example(r, "custom_blend.tex", _custom_blend_cases)
    try:
        # my_soft_light is shown but not wired: call it per channel, the way a reader would.
        src = _read("examples", "custom_blend.tex")
        soft = src[:src.index("vec3 result = vec3(")] + (
            "@OUT = vec3(my_soft_light(base.r, over.r), my_soft_light(base.g, over.g),"
            " my_soft_light(base.b, over.b));\n")
        _label, bindings, _ref, _tol = _custom_blend_cases()[1]
        ref = _ref_blend(_grid(bindings["base"]), _grid(bindings["overlay"]), _ref_soft_light, None)
        di, dc, _oi, _oc = _measure(soft, bindings, ref)
        assert di <= _EPS_FLOAT and dc <= _EPS_FLOAT, (di, dc)
        r.ok(f"examples/custom_blend.tex: my_soft_light called per channel "
             f"(maxdiff interp {di:.1e}, codegen {dc:.1e})")
    except Exception as e:
        r.fail("custom_blend my_soft_light", f"{type(e).__name__}: {e}")


def test_control_flow_while_loop_matches_a_per_pixel_reference(r: SubTestResult):
    print("\n--- examples/while_loop.tex against a per-pixel reference, both tiers ---")
    _check_example(r, "while_loop.tex", _while_loop_cases)


def test_control_flow_vector_blur_matches_a_per_pixel_reference(r: SubTestResult):
    print("\n--- examples/vector_blur.tex against a per-pixel reference, both tiers ---")
    _check_example(r, "vector_blur.tex", _vector_blur_cases)


def _ref_mandelbrot(H, W, zoom, cx, cy, iterations):
    """The escape-time fractal, one pixel at a time, with an ordinary early `break`."""
    out = []
    for y in range(H):
        orow = []
        for x in range(W):
            u, v = x / max(W - 1, 1), y / max(H - 1, 1)
            cr = (u - 0.5) * 3.0 / zoom + cx
            ci = (v - 0.5) * 2.0 / zoom + cy
            r = i = 0.0
            escape = 0.0
            for n in range(iterations):
                r, i = r * r - i * i + cr, 2.0 * r * i + ci
                if r * r + i * i > 4.0:
                    escape = n / iterations
                    break
            if escape > 0.0:            # as the example writes it: n = 0 stays black
                orow.append([0.5 + 0.5 * math.cos(math.tau * (escape + k))
                             for k in (0.0, 0.33, 0.67)])
            else:
                orow.append([0.0, 0.0, 0.0])
        out.append(orow)
    return out


def _recursive_pattern_cases():
    # Purely procedural: an unread image binding is what gives the cook its grid, the way a
    # host wires one in (`_consensus_extent`), and it changes nothing the program computes.
    H, W = 9, 14
    torch.manual_seed(21)
    grid = torch.rand(1, H, W, 3)
    base = {"image": grid, "zoom_level": 1.0, "center_x": -0.5, "center_y": 0.0, "iterations": 50}
    zoomed = dict(base, zoom_level=4.0, center_x=-0.75, center_y=0.1, iterations=30)
    return [
        ("the default view", base, _ref_mandelbrot(H, W, 1.0, -0.5, 0.0, 50), _EPS_FLOAT),
        ("zoomed in on the seahorse valley, 30 iterations", zoomed,
         _ref_mandelbrot(H, W, 4.0, -0.75, 0.1, 30), _EPS_FLOAT),
    ]


def test_control_flow_recursive_pattern_matches_a_per_pixel_reference(r: SubTestResult):
    print("\n--- examples/recursive_pattern.tex against a per-pixel reference, both tiers ---")

    def extra(label, oi, oc):
        lit = int((oi[0].sum(-1) > 0).sum())
        assert lit > 0, "every pixel is black: the escape step is not recorded per pixel"
    _check_example(r, "recursive_pattern.tex", _recursive_pattern_cases, extra)


# The snippet-menu line and the widget surface a user sees must not move with the fix.
_EXAMPLE_SURFACE = {
    "fix_pixels.tex": ("// Fix Pixels — sanitize NaN and Inf values in images",
                       {"fallback_r": 0.0, "fallback_g": 0.0, "fallback_b": 0.0}, {"image"}),
    "break_search.tex": ("// Break Search — scan for the first bright pixel using for + break",
                         {"threshold": 0.5, "scan_width": 0}, {"image"}),
    "custom_blend.tex": ("// Custom Blend — overlay blend mode using user-defined functions",
                         {"blend_amount": 0.5}, {"base", "overlay"}),
    "while_loop.tex": ("// While Loop — Newton's method for square root",
                       {"tolerance": 0.0001, "max_iter": 20}, {"image"}),
    "vector_blur.tex": ("// Vector Blur — directional per-pixel motion blur driven by a vector map",
                        {"strength": 20.0, "max_samples": 32}, {"image", "vectors"}),
    # TRK-24: `center_x`'s default (`f$center_x = -0.5;`) now reads -0.5, not None — the
    # type checker's default-extraction chain previously had no branch for a negative
    # literal (UnaryOp('-') over a NumberLiteral) and silently dropped it. This is the
    # corrected surface, not the pre-fix one this row used to pin.
    "recursive_pattern.tex": ("// Mandelbrot Fractal — escape-time fractal with cosine palette",
                              {"zoom_level": 1.0, "center_x": -0.5, "center_y": 0.0,
                               "iterations": 50}, set()),
}


def test_control_flow_fixed_examples_keep_their_surface(r: SubTestResult):
    print("\n--- the fixed examples keep their menu line, parameters, inputs and outputs ---")
    for name, (line1, params, inputs) in _EXAMPLE_SURFACE.items():
        try:
            src = _read("examples", name)
            assert src.splitlines()[0] == line1, src.splitlines()[0]
            prog = tex_api.compile(src, {n: TEXType.VEC3 for n in inputs})
            got = {n: p.get("default_value") for n, p in prog.params.items()}
            assert got == params, got
            assert set(prog.assigned) == {"OUT"}, prog.assigned
            assert set(prog.referenced) - set(prog.params) - set(prog.assigned) == inputs, \
                prog.referenced
            codes = [d.code for d in tex_api.control_flow_advisories(src, {})]
            assert "W7007" not in codes, codes
            r.ok(f"examples/{name}: same menu line, params, inputs and outputs; no W7007")
        except Exception as e:
            r.fail(f"example surface {name}", f"{type(e).__name__}: {e}")
    try:
        # The curriculum as a whole: no shipped example may rely on control flow acting on
        # every pixel. W7006 (a per-pixel branch that gathers) is a cost note, not a defect,
        # and several examples keep it truthfully.
        flagged = {}
        for name in sorted(f for f in os.listdir(_EXAMPLES) if f.endswith(".tex")):
            codes = sorted({d.code for d in tex_api.control_flow_advisories(_read("examples", name), {})})
            if codes:
                flagged[name] = codes
        w7007 = {n: c for n, c in flagged.items() if "W7007" in c}
        assert not w7007, f"shipped examples still exit or bound per pixel: {w7007}"
        r.ok(f"no shipped example carries W7007; {len(flagged)} carry W7006 (a cost note)")
    except Exception as e:
        r.fail("shipped examples free of W7007", f"{type(e).__name__}: {e}")
