"""Region-dependent programs: the engine declines to split the cook (TRK-25).

THE RULE. A program is *region-dependent* when its output can depend on WHICH REGION was
cooked rather than only on the pixel — because some control decision REDUCES a per-pixel
value over the cooked region. Two shapes do that:

  (a)/(b) a `for` / `while` whose condition is not uniform. The interpreter keeps looping
          while `(cond > 0.5).any()` over the region holds, and does not mask the body, so
          every pixel runs as many passes as the hungriest pixel IN ITS REGION. Split the
          frame into strips and a pixel's pass count changes. A 0-dim condition whose VALUE
          is region-derived (`img_mean(@A)`, `img_width(@A)`) is the same defect and is
          covered by the same non-uniformity rule (a reduction's footprint is not a point).
  (c)     a per-pixel `if` that assigns a STRING. A string has no per-pixel representation,
          so the merge resolves it by a MAJORITY VOTE over the region's pixels, and a strip
          can hold a different majority than the whole frame.

For those, `tex_roi.region_dependent` is True and the three planners that split a cook —
`_tile_plan` (strips), `roi_plan` (windows, and through it the halo strips and the chain
windowing) and `batch_sliceable` (batch strips) — all decline. The executors are untouched:
`run_tiled` / `run_roi` / `run_batch_strips` stay dumb, so the rows below that drive them
DIRECTLY still show the old, region-dependent numbers. That is deliberate — it is the
evidence the gate exists for, and it is what makes these rows characterization pins.

WHAT IS NOT DECLINED, and the reason this is not a blanket disable: `break`, `continue` and
`return` under a per-pixel guard. Those fire on FIRST ARRIVAL at the statement, which is a
structural fact identical in every region, so they tile correctly and keep tiling. So do
static and uniformly-bounded loops. The negative row asserts that over every shipped
`examples/*.tex` and `stock/*.textool` program AS A COUNT, so an example that later acquires
the shape reds this file on purpose rather than silently widening the decline.

Every row runs on the CPU interpreter. No ComfyUI, no CUDA, no compiler, no Windows path, no
embedded interpreter, no numpy — and no row asserts a time.
"""
import glob
import json
import os

import torch

from helpers import *

from TEX_Wrangle import tex_api, tex_memory, tex_roi
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_runtime.interpreter import Interpreter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TRK-25's own repro, in its own orientation: H=8, W=2, read column 0 top to bottom.
# `v` runs 0 … 1 down the rows, so the top pixel needs four passes and the bottom none.
REPRO = ("float x = v;\n"
         "float n = 0.0;\n"
         "while (x < 1.0) { x = x + 0.25; n = n + 1.0; }\n"
         "@OUT = vec4(n,n,n,1.0);\n")

# F2: the majority-vote STRING merge — the second, independent region-dependence class.
STRING_REPRO = ('string s = "lo";\n'
                'if (v > 0.2) { s = "hi"; }\n'
                '@TXT = s;\n'
                '@OUT = @A;\n')

# F3: the same per-pixel loop plus a TOP-LEVEL grounded blur, so `is_tile_safe` is False and
# the HALO strip route (the one that exists precisely for the programs `is_tile_safe` refuses)
# is the route that has to be closed.
HALO_REPRO = ("float x = v;\n"
              "float n = 0.0;\n"
              "while (x < 1.0) { x = x + 0.25; n = n + 1.0; }\n"
              "@OUT = gauss_blur(@A, 2.0) + vec4(n,n,n,0.0);\n")

HALO_UNIFORM_TWIN = ("float n = 0.0;\n"
                     "for (int i = 0; i < 4; i = i + 1) { n = n + 1.0; }\n"
                     "@OUT = gauss_blur(@A, 2.0) + vec4(n,n,n,0.0);\n")

# F4a: a loop bounded by `fi`, which is per-FRAME. The spatial routes do not care; the batch
# route does, and nothing gated it before.
FI_REPRO = ("float n = 0.0;\n"
            "float i = 0.0;\n"
            "while (i < fi + 1.0) { n = n + 1.0; i = i + 1.0; }\n"
            "@OUT = vec4(n,n,n,1.0);\n")

# F4b: a 0-dim loop bound whose VALUE is a whole-image reduction of the narrowed binding.
IMG_MEAN_REPRO = ("float t = img_mean(@A).r;\n"
                  "float n = 0.0;\n"
                  "float i = 0.0;\n"
                  "while (i < t * 8.0) { n = n + 1.0; i = i + 1.0; }\n"
                  "@OUT = vec4(n,n,n,1.0);\n")


def _parse(src):
    """Parse WITHOUT folding or type-checking — what the predicate takes."""
    return Parser(Lexer(src).tokenize(), source=src).parse()


def _img(b, h, w):
    return torch.linspace(0.0, 1.0, b * h * w * 4).reshape(b, h, w, 4)


def _cook(src, *, shape=(1, 8, 2), tiles=None, batch=None, roi=None,
          narrow=None, halo=0, params=None):
    """Cook `src` whole-frame, or through one executor directly. Returns (program, outputs)."""
    b, h, w = shape
    bindings = {"A": _img(b, h, w)}
    btypes = {"A": TEXType.VEC4}
    for name, value in (params or {}).items():
        bindings[name] = value
        btypes[name] = TEXType.FLOAT
    prog = tex_api.compile(src, btypes)
    names = sorted(prog.assigned.keys())
    interp = Interpreter()
    head = (interp, prog.ast, bindings, prog.type_map, "cpu", 0, names,
            prog.used_builtins, "fp32")
    if tiles:
        out = tex_memory.run_tiled(*head, tiles)
    elif batch:
        out = tex_memory.run_batch_strips(*head, batch)
    elif roi is not None:
        out = tex_memory.run_roi(*head, roi,
                                 frozenset() if narrow is None else narrow, halo)
    else:
        out = interp.execute(prog.ast, bindings, prog.type_map, device="cpu",
                             latent_channel_count=0, output_names=names,
                             used_builtins=prog.used_builtins, precision="fp32")
    return prog, out


def _column(t, x=0, ch=0):
    """Column `x` of frame 0, top to bottom — TRK-25's own reading."""
    return [round(float(val), 4) for val in t[0, :, x, ch]]


def _maxdiff(a, b):
    return float((a - b).abs().max())


def _example_sources():
    for path in sorted(glob.glob(os.path.join(_ROOT, "examples", "*.tex"))):
        with open(path, encoding="utf-8") as f:
            yield os.path.basename(path), f.read()


def _stock_sources():
    """Every TEX program inside the shipped `.textool` bundles — a plain tool's `code`, plus
    each stage and the terminal stage of a fused one."""
    for path in sorted(glob.glob(os.path.join(_ROOT, "stock", "*.textool"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            man = json.load(f)
        if man.get("code"):
            yield name, man["code"]
        for i, stage in enumerate((man.get("graphspec") or {}).get("stages") or []):
            if stage.get("code"):
                yield f"{name}#stage{i}", stage["code"]
        if man.get("terminal_code"):
            yield f"{name}#terminal", man["terminal_code"]


# ── T1: the mandatory pin ────────────────────────────────────────────────────

def test_t1_repro_is_region_dependent(r: SubTestResult):
    print("\n--- T1: TRK-25's repro splits wrong, and the predicate says so ---")
    try:
        _p, whole = _cook(REPRO)
        _p2, tiled = _cook(REPRO, tiles=2)
        _p4, tiled4 = _cook(REPRO, tiles=4)
        got_whole, got_2, got_4 = (_column(whole["OUT"]), _column(tiled["OUT"]),
                                   _column(tiled4["OUT"]))
        assert got_whole == [4.0] * 8, got_whole
        assert got_2 == [4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 2.0], got_2
        assert got_4 == [4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0], got_4
        assert _maxdiff(whole["OUT"], tiled["OUT"]) == 2.0
        assert _maxdiff(whole["OUT"], tiled4["OUT"]) == 3.0
        r.ok(f"characterization: whole={got_whole[:2]}… 2 strips={got_2} 4 strips={got_4}")
    except Exception as e:
        r.fail("T1 characterization", f"{type(e).__name__}: {e}")

    try:
        assert tex_roi.region_dependent(_parse(REPRO), code=REPRO) is True
        r.ok("region_dependent(TRK-25 repro) is True")
    except Exception as e:
        r.fail("T1 predicate", f"{type(e).__name__}: {e}")


# ── T7: the negative test — NOT a blanket disable ────────────────────────────

# Each row: (name, source, params). All are region-INDEPENDENT and must keep splitting.
_MUST_STILL_SPLIT = [
    ("static for",
     "float n = 0.0;\nfor (int i = 0; i < 4; i++) { n = n + 1.0; }\n@OUT = vec4(n,n,n,1.0);\n",
     None),
    ("$param-bounded for",
     "float n = 0.0;\nfor (int i = 0; float(i) < $k; i = i + 1) { n = n + 1.0; }\n"
     "@OUT = vec4(n,n,n,1.0);\n",
     {"k": 3.0}),
    ("ih-bounded while with a per-pixel BODY",
     "float n = 0.0;\nfloat i = 0.0;\nwhile (i < ih) { n = n + @A.r; i = i + 1.0; }\n"
     "@OUT = vec4(n,n,n,1.0);\n",
     None),
    ("break under a per-pixel guard",
     "float n = 0.0;\nfor (int i = 0; i < 8; i++) { if (@A.r > 0.5) { break; } n = n + 1.0; }\n"
     "@OUT = vec4(n,n,n,1.0);\n",
     None),
    ("break under a COUNTER-dependent per-pixel guard",
     "float n = 0.0;\n"
     "for (int i = 0; i < 8; i++) { if (float(i) >= @A.r * 8.0) { break; } n = n + 1.0; }\n"
     "@OUT = vec4(n,n,n,1.0);\n",
     None),
    ("continue under a per-pixel guard",
     "float n = 0.0;\nfor (int i = 0; i < 4; i++) { if (@A.r > 0.5) { continue; } n = n + 1.0; }\n"
     "@OUT = vec4(n,n,n,1.0);\n",
     None),
    ("perlin at constant coordinates in the bound",
     "float n = 0.0;\nfloat i = 0.0;\n"
     "while (i < perlin(3.0, 4.0) * 2.0 + 4.0) { n = n + 1.0; i = i + 1.0; }\n"
     "@OUT = vec4(n,n,n,1.0);\n",
     None),
    ("select(cond, a, b) — pointwise, no loop",
     "@OUT = select(@A.r > 0.5, @A, vec4(0.0, 0.0, 0.0, 1.0));\n",
     None),
]


def test_t7_predicate_is_not_a_blanket_disable(r: SubTestResult):
    print("\n--- T7: the shapes that MUST keep splitting (the negative test) ---")
    for name, src, params in _MUST_STILL_SPLIT:
        try:
            prog, whole = _cook(src, params=params)
            _p, tiled = _cook(src, tiles=2, params=params)
            md = _maxdiff(whole["OUT"], tiled["OUT"])
            scalars = {n: v for n, v in (params or {}).items()}
            assert md == 0.0, f"tiled != whole (maxdiff {md})"
            assert tex_roi.region_dependent(_parse(src), code=src) is False, "declined"
            assert tex_memory.is_tile_safe(prog.ast) is True, "is_tile_safe moved"
            assert tex_roi.roi_plan(src, scalars).executable is True, "roi_plan refused"
            assert tex_roi.batch_sliceable(src, scalars) is True, "batch_sliceable refused"
            r.ok(f"still splits: {name}")
        except Exception as e:
            r.fail(f"T7 {name}", f"{type(e).__name__}: {e}")


def test_t7_no_shipped_program_is_declined(r: SubTestResult):
    print("\n--- T7: the count over every shipped program ---")
    examples = list(_example_sources())
    stock = list(_stock_sources())
    try:
        assert len(examples) == 116, f"expected 116 examples/*.tex, found {len(examples)}"
        assert len(stock) == 6, f"expected 6 stock programs, found {len(stock)}"
        r.ok(f"corpus size pinned: {len(examples)} examples + {len(stock)} stock programs")
    except Exception as e:
        r.fail("T7 corpus size", f"{type(e).__name__}: {e}")

    declined = []
    unparsed = []
    for name, src in examples + stock:
        try:
            program = _parse(src)
        except Exception as e:
            unparsed.append(f"{name} ({type(e).__name__})")
            continue
        if tex_roi.region_dependent(program, code=src):
            declined.append(name)
    try:
        assert not unparsed, f"shipped programs failed to parse: {unparsed}"
        assert declined == [], (
            "the predicate declines shipped programs, so it is no longer only about the "
            f"region-dependent class: {declined}")
        r.ok(f"0 of {len(examples) + len(stock)} shipped programs are region-dependent")
    except Exception as e:
        r.fail("T7 shipped-program count", f"{type(e).__name__}: {e}")


# ── T11: fail-closed ─────────────────────────────────────────────────────────

def _budget_buster(depth=2, chain=500):
    """A program whose def-use chain forces the lint's loop fixed point to re-iterate once per
    link, which is quadratic in the chain while the work budget is only linear in the node
    count — so it runs past `_CFBudget`. Nothing about it is region-dependent on its own
    merits (every loop bound is a static literal), which is what makes it a posture test."""
    decls = "".join(f"float a{j} = 0.0;\n" for j in range(chain))
    body = "".join(f"a{j} = a{j + 1}; " for j in range(chain - 1)) + f"a{chain - 1} = u;"
    head = "".join(f"for (int i{i} = 0; i{i} < 2; i{i} = i{i} + 1) {{ " for i in range(depth))
    return decls + head + body + " }" * depth + "\n@OUT = vec4(a0,a0,a0,1.0);\n"


def test_t11_analysis_failure_declines_the_split(r: SubTestResult):
    print("\n--- T11: an analysis that cannot finish declines the split (fail CLOSED) ---")
    src = _budget_buster()
    program = _parse(src)
    try:
        lint = tex_api._ControlFlowLint(program, src, {})
        try:
            lint.region_clauses()
            raise AssertionError("the work budget did not blow — this row is now vacuous")
        except tex_api._CFBudget:
            pass
        r.ok("the probe really does run the lint past its work budget")
    except Exception as e:
        r.fail("T11 budget probe", f"{type(e).__name__}: {e}")

    try:
        assert tex_roi.region_dependent(program, code=src) is True, \
            "a gate that cannot analyse a program must decline the split, not allow it"
        r.ok("region_dependent fails CLOSED on an over-budget analysis")
    except Exception as e:
        r.fail("T11 predicate fails closed", f"{type(e).__name__}: {e}")

    try:
        assert tex_api.control_flow_advisories(src, {}) == [], \
            "the advisory's fail-OPEN posture moved"
        r.ok("control_flow_advisories still fails OPEN on the same program ([])")
    except Exception as e:
        r.fail("T11 advisory fails open", f"{type(e).__name__}: {e}")

    try:
        # Not only the budget: §1.6 says ANY failure declines. A non-Program object cannot be
        # walked, and the answer must still be "decline", never "allow".
        assert tex_roi.region_dependent(object(), code="") is True
        r.ok("region_dependent declines on any analysis failure, not just the budget")
    except Exception as e:
        r.fail("T11 predicate declines on a broken input", f"{type(e).__name__}: {e}")
