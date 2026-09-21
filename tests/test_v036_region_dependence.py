"""Region-dependent programs: the engine declines to split the cook (TRK-25, TRK-32).

THE RULE. A program is *region-dependent* when its output can depend on WHICH REGION was
cooked rather than only on the pixel — because some control decision REDUCES a per-pixel
value over the cooked region. Four shapes do that:

  (a)/(b) a `for` / `while` whose condition is not uniform. The interpreter keeps looping
          while `(cond > 0.5).any()` over the region holds, and does not mask the body, so
          every pixel runs as many passes as the hungriest pixel IN ITS REGION. Split the
          frame into strips and a pixel's pass count changes. A 0-dim condition whose VALUE
          is region-derived (`img_mean(@A)`, `img_width(@A)`) is the same defect and is
          covered by the same non-uniformity rule (a reduction's footprint is not a point).
  (c)     a per-pixel `if` that assigns a STRING, or a per-pixel `?:` that picks one. A
          string has no per-pixel representation, so the merge resolves it by a MAJORITY
          VOTE over the region's pixels, and a strip can hold a different majority than the
          whole frame.
  (d)     a per-pixel value cast STRAIGHT to a STRING — `string(x)`, `str(x)`, or a
          `format()` call that actually fills a placeholder — with no condition or merge
          at all (TRK-32, a sibling of (c) and NOT the same defect). `_scalar_from_tensor`
          has no per-pixel representation either, so it falls back to the MEAN of every
          pixel in the region being cooked, and a strip's mean differs from the frame's.
          `format("%f", x)` is NOT in this class: `%f` is not a `{}`/`{:spec}` placeholder,
          so the template returns unchanged and the reduced value never reaches the output.

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

# TRK-32 clause (d): a per-pixel value cast STRAIGHT to a STRING, no condition/merge at all —
# a sibling of F2, not the same defect. `stdlib._scalar_from_tensor` has no per-pixel
# representation either, so it falls back to the MEAN of every pixel in the region being
# cooked, and a strip's mean differs from the whole frame's.
CAST_REPRO = "@TXT = string(@A.r * 10.0);\n@OUT = @A;\n"
STR_REPRO = "@TXT = str(@A.r * 10.0);\n@OUT = @A;\n"
FORMAT_REPRO = ('float x = @A.r * 10.0;\n'
               '@TXT = format("{}", x);\n'
               '@OUT = @A;\n')

# The documented correction: `format()` only substitutes a Python-style `{}`/`{:spec}`
# placeholder — `%f` is not one and passes through literally, so the reduced value never
# reaches the output. NOT in this class, and the negative control that proves it.
FORMAT_PERCENT_REPRO = ('float x = @A.r * 10.0;\n'
                        '@TXT = format("%f", x);\n'
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
          narrow=None, halo=0, params=None, strings=None):
    """Cook `src` whole-frame, or through one executor directly. Returns (program, outputs)."""
    b, h, w = shape
    bindings = {"A": _img(b, h, w)}
    btypes = {"A": TEXType.VEC4}
    for name, value in (params or {}).items():
        bindings[name] = value
        btypes[name] = TEXType.FLOAT
    for name, value in (strings or {}).items():
        bindings[name] = value
        btypes[name] = TEXType.STRING
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


def _read_repo(*parts):
    with open(os.path.join(_ROOT, *parts), encoding="utf-8") as f:
        return f.read()


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
    ("format('%f', x) — not a real placeholder, TRK-32 clause (d) negative control",
     FORMAT_PERCENT_REPRO,
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
        assert len(examples) == 117, f"expected 117 examples/*.tex, found {len(examples)}"
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
        # DOC-8 corrected `examples/string_format.tex`'s format() calls from dead `%f`/`%s`
        # sequences (never filled — see the CHANGELOG's ASK-9 entry) to real `{}`
        # placeholders. Its five numeric calls now genuinely cast a per-region-varying value
        # straight to a string, so clause (d) correctly declines it — the one legitimate
        # entry the count below should ever gain from a documentation fix.
        assert declined == ["string_format.tex"], (
            "the predicate declines shipped programs, so it is no longer only about the "
            f"region-dependent class: {declined}")
        r.ok(f"1 of {len(examples) + len(stock)} shipped programs is region-dependent "
             f"(string_format.tex)")
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


# ── T3 / T5: the spatial routes (windows, and the halo strips behind them) ───

def test_t3_roi_window_is_declined(r: SubTestResult):
    print("\n--- T3: an ROI window, and the chain windowing built on it ---")
    try:
        plan = tex_roi.roi_plan(REPRO, {})
        assert plan.executable is False, "roi_plan still narrows a region-dependent program"
        assert plan.narrow == frozenset(), f"nothing may be narrowed: {sorted(plan.narrow)}"
        r.ok("roi_plan(repro) is not executable and narrows nothing")
    except Exception as e:
        r.fail("T3 roi_plan", f"{type(e).__name__}: {e}")

    try:
        # `stage_halo`'s inversion: a non-executable plan means UNBOUNDED reach, so a host
        # composing windows over a chain of stages grows every upstream window to the frame.
        assert tex_roi.stage_halo(REPRO, {}) == tex_roi.WHOLE_FRAME
        r.ok("stage_halo(repro) == WHOLE_FRAME, so chain_windows saturates")
    except Exception as e:
        r.fail("T3 stage_halo", f"{type(e).__name__}: {e}")

    try:
        # The evidence the decline exists for: driven directly — the executor stays dumb — a
        # narrowed window over rows 4-7 cooks a DIFFERENT answer than the whole frame does.
        _p, whole = _cook(REPRO)
        _p2, windowed = _cook(REPRO, roi=(0, 4, 2, 4, 2, 8), narrow=frozenset({"A"}), halo=0)
        assert _column(whole["OUT"])[4:] == [4.0, 4.0, 4.0, 4.0]
        assert _column(windowed["OUT"]) == [2.0, 2.0, 2.0, 2.0], _column(windowed["OUT"])
        r.ok("characterization: rows 4-7 cook as 2, not 4, when the window is narrowed")
    except Exception as e:
        r.fail("T3 characterization", f"{type(e).__name__}: {e}")


def test_t5_halo_strip_route_is_declined(r: SubTestResult):
    print("\n--- T5: the HALO strip route (the one is_tile_safe does not close) ---")
    try:
        prog = tex_api.compile(HALO_REPRO, {"A": TEXType.VEC4})
        assert tex_memory.is_tile_safe(prog.ast) is False, \
            "the blur makes this non-pointwise — halo tiling is the route that must be closed"
        assert tex_roi.roi_plan(HALO_REPRO, {}).executable is False, \
            "roi_plan still reports a halo plan, so _halo_tile_plan can still strip it"
        r.ok("roi_plan refuses the blur + per-pixel-loop program, so _halo_tile_plan gets None")
    except Exception as e:
        r.fail("T5 halo repro", f"{type(e).__name__}: {e}")

    try:
        twin = tex_roi.roi_plan(HALO_UNIFORM_TWIN, {})
        assert twin.executable is True, "the uniform twin must still halo-tile"
        assert twin.halo == 6, f"halo moved: {twin.halo}"
        assert twin.narrow == frozenset({"A"}), f"narrow moved: {sorted(twin.narrow)}"
        r.ok("the uniform twin still gets halo=6, narrow={'A'}")
    except Exception as e:
        r.fail("T5 uniform twin", f"{type(e).__name__}: {e}")

    try:
        # F3's own measurement, driven straight at the executor: halo strips of a
        # region-dependent program disagree with the whole frame.
        prog = tex_api.compile(HALO_REPRO, {"A": TEXType.VEC4})
        bindings = {"A": _img(1, 64, 64)}
        names = sorted(prog.assigned.keys())
        interp = Interpreter()
        whole = interp.execute(prog.ast, bindings, prog.type_map, device="cpu",
                               latent_channel_count=0, output_names=names,
                               used_builtins=prog.used_builtins, precision="fp32")
        striped = tex_memory.run_tiled_halo(
            interp, prog.ast, bindings, prog.type_map, "cpu", 0, names,
            prog.used_builtins, "fp32", 2, frozenset({"A"}), 6)
        md = _maxdiff(whole["OUT"], striped["OUT"])
        # The pass-count difference is exactly 1; the blur term adds fp32 slack, so this is a
        # tolerance rather than an equality (never a hash of a float).
        assert abs(md - 1.0) < 1e-5, f"expected the F3 divergence of 1.0, got {md}"
        r.ok("characterization: halo strips disagree with the whole frame by 1.0")
    except Exception as e:
        r.fail("T5 characterization", f"{type(e).__name__}: {e}")


# ── T4: the batch axis ───────────────────────────────────────────────────────

def test_t4_batch_strips_are_declined(r: SubTestResult):
    print("\n--- T4: batch strips (the host-only route nothing gated) ---")
    for name, src in (("the spatial repro", REPRO),
                      ("an fi-bounded loop (F4a)", FI_REPRO),
                      ("an img_mean-bounded loop (F4b)", IMG_MEAN_REPRO)):
        try:
            assert tex_roi.batch_sliceable(src, {}) is False
            r.ok(f"batch_sliceable is False for {name}")
        except Exception as e:
            r.fail(f"T4 {name}", f"{type(e).__name__}: {e}")

    try:
        # F4a's measurement: `fi` in the bound makes the pass count a property of the BATCH
        # strip, and nothing looked at it before (there is no frame op to find).
        _p, whole = _cook(FI_REPRO, shape=(4, 2, 4))
        _p2, striped = _cook(FI_REPRO, shape=(4, 2, 4), batch=2)
        per_frame_whole = [round(float(whole["OUT"][b, 0, 0, 0]), 4) for b in range(4)]
        per_frame_striped = [round(float(striped["OUT"][b, 0, 0, 0]), 4) for b in range(4)]
        assert per_frame_whole == [4.0, 4.0, 4.0, 4.0], per_frame_whole
        assert per_frame_striped == [2.0, 2.0, 4.0, 4.0], per_frame_striped
        r.ok(f"characterization: batch strips give {per_frame_striped}, not {per_frame_whole}")
    except Exception as e:
        r.fail("T4 characterization", f"{type(e).__name__}: {e}")


# ── T6: clause (c) — the majority-vote string merge ─────────────────────────

def test_t6_string_merge_is_region_dependent(r: SubTestResult):
    print("\n--- T6: a string assigned under a per-pixel `if` (clause (c), F2) ---")
    try:
        assert tex_roi.region_dependent(_parse(STRING_REPRO), code=STRING_REPRO) is True
        assert tex_roi.roi_plan(STRING_REPRO, {}).executable is False
        assert tex_roi.batch_sliceable(STRING_REPRO, {}) is False
        r.ok("the string program is region-dependent and all three routes decline it")
    except Exception as e:
        r.fail("T6 predicate and routes", f"{type(e).__name__}: {e}")

    try:
        # The evidence: 6 of 8 rows satisfy the condition whole-frame, but strip 0 holds only
        # 2 of 4 — not a majority — and `run_tiled` takes a non-spatial output from whichever
        # strip produced it first.
        _p, whole = _cook(STRING_REPRO)
        _p2, tiled = _cook(STRING_REPRO, tiles=2)
        assert whole["TXT"] == "hi", repr(whole["TXT"])
        assert tiled["TXT"] == "lo", repr(tiled["TXT"])
        r.ok('characterization: whole frame votes "hi", two strips vote "lo"')
    except Exception as e:
        r.fail("T6 characterization", f"{type(e).__name__}: {e}")


# Clause (c) is about the MERGE, not about one keyword. The `if` merge
# (`_merge_branch_vars`) is one spelling of it; a per-pixel `?:` whose arms are strings is the
# other, and it runs its OWN region-wide vote (`Interpreter._eval_ternary`, and the same
# arithmetic in the codegen tier). Both divergences are pinned here as characterization, so
# the rows below are a gate against something that really happens.
TERNARY_REPRO = ('@TXT = v > 0.2 ? "hi" : "lo";\n'
                 '@OUT = @A;\n')

NESTED_TERNARY_REPRO = ('@TXT = v > 0.2 ? (v > 0.9 ? "hi" : "mid") : "lo";\n'
                        '@OUT = @A;\n')

# The same vote, but the arms arrive as STRING WIRES rather than literals — so the verdict is
# only reachable when the binding types are known. `_ControlFlowLint` has always been able to
# see this class; before this row nothing in the cook path fed it the types.
WIRE_TERNARY_REPRO = ('@TXT = v > 0.2 ? @S : @T;\n'
                      '@OUT = @A;\n')

# The `if` spelling of the same thing: a STRING BINDING assigned under a per-pixel `if`, from
# string wires. `collect_assigned_vars`' binding half sees `TXT`; knowing that `@S` is a string
# is what puts `TXT` in the string-name set.
WIRE_IF_REPRO = ('if (v > 0.2) { @TXT = @S; } else { @TXT = @T; }\n'
                 '@OUT = @A;\n')

_STRING_WIRES = {"S": TEXType.STRING, "T": TEXType.STRING}
_WIRE_VALUES = {"S": "hi", "T": "lo"}


def test_t6_string_ternary_is_region_dependent(r: SubTestResult):
    print("\n--- T6: a string `?:` on a per-pixel condition (clause (c), the other spelling) ---")
    for name, src in (("ternary", TERNARY_REPRO), ("nested ternary", NESTED_TERNARY_REPRO)):
        try:
            assert tex_roi.region_dependent(_parse(src), code=src) is True, "not declined"
            assert tex_roi.roi_plan(src, {}).executable is False, "roi_plan still executable"
            assert tex_roi.batch_sliceable(src, {}) is False, "batch_sliceable still True"
            r.ok(f"{name}: region-dependent, and all three routes decline it")
        except Exception as e:
            r.fail(f"T6 {name} routes", f"{type(e).__name__}: {e}")

    try:
        # The evidence the gate exists for, driven through the executor directly.
        _p, whole = _cook(TERNARY_REPRO)
        _p2, tiled = _cook(TERNARY_REPRO, tiles=2)
        assert whole["TXT"] == "hi", repr(whole["TXT"])
        assert tiled["TXT"] == "lo", repr(tiled["TXT"])
        _p3, nwhole = _cook(NESTED_TERNARY_REPRO)
        _p4, ntiled = _cook(NESTED_TERNARY_REPRO, tiles=2)
        assert nwhole["TXT"] != ntiled["TXT"], (nwhole["TXT"], ntiled["TXT"])
        r.ok(f'characterization: `?:` votes "hi" whole-frame and "lo" at two strips; '
             f'nested votes {nwhole["TXT"]!r} vs {ntiled["TXT"]!r}')
    except Exception as e:
        r.fail("T6 ternary characterization", f"{type(e).__name__}: {e}")


def test_t6_string_wires_need_the_binding_types(r: SubTestResult):
    print("\n--- T6: a string WIRE merged per pixel — only visible with the binding types ---")
    for name, src in (("`?:` over two string wires", WIRE_TERNARY_REPRO),
                      ("`if` assigns a string binding", WIRE_IF_REPRO)):
        try:
            assert tex_roi.region_dependent(
                _parse(src), _STRING_WIRES, code=src) is True, "not declined"
            assert tex_roi.roi_plan(
                src, {}, binding_types=_STRING_WIRES).executable is False, "roi_plan executable"
            assert tex_roi.batch_sliceable(
                src, {}, binding_types=_STRING_WIRES) is False, "batch_sliceable True"
            r.ok(f"{name}: region-dependent once the binding types are supplied")
        except Exception as e:
            r.fail(f"T6 {name}", f"{type(e).__name__}: {e}")

    try:
        _p, whole = _cook(WIRE_IF_REPRO, strings=_WIRE_VALUES)
        _p2, tiled = _cook(WIRE_IF_REPRO, tiles=2, strings=_WIRE_VALUES)
        assert whole["TXT"] == "hi", repr(whole["TXT"])
        assert tiled["TXT"] == "lo", repr(tiled["TXT"])
        r.ok('characterization: the string-binding `if` votes "hi" whole, "lo" at two strips')
    except Exception as e:
        r.fail("T6 wire characterization", f"{type(e).__name__}: {e}")


def test_t6_binding_types_move_the_verdict_and_the_memo(r: SubTestResult):
    print("\n--- T6: the binding types are part of the question, so part of every key ---")
    src = WIRE_TERNARY_REPRO
    try:
        # Identical source, different binding types: the answer MUST differ. Nothing about the
        # source says `@S` holds a string, so without the types there is no string merge to see.
        assert tex_roi.region_dependent(_parse(src), None, code=src) is False, (
            "a program whose string-ness is unknowable from the source alone was declined")
        assert tex_roi.region_dependent(_parse(src), _STRING_WIRES, code=src) is True, (
            "supplying the binding types did not reach the predicate")
        r.ok("the verdict is a function of (source, binding types), not of the source alone")
    except Exception as e:
        r.fail("T6 verdict follows the types", f"{type(e).__name__}: {e}")

    try:
        # …and therefore every memo that can serve the verdict has to key on them. Ask the
        # UNTYPED question first, so a key that ignores the types would answer it again.
        tex_roi.clear_roi_memo()
        assert tex_roi.roi_plan(src, {}).executable is True, "the untyped baseline moved"
        assert tex_roi.roi_plan(src, {}, binding_types=_STRING_WIRES).executable is False, (
            "roi_plan served the untyped verdict for a typed question — `_walk`'s memo key "
            "does not include the binding types")
        assert tex_roi.batch_sliceable(src, {}) is True, "the untyped baseline moved"
        assert tex_roi.batch_sliceable(src, {}, binding_types=_STRING_WIRES) is False, (
            "batch_sliceable served the untyped verdict for a typed question")
        r.ok("`_walk`'s memo distinguishes the two questions rather than serving a stale False")
    except Exception as e:
        r.fail("T6 memo key", f"{type(e).__name__}: {e}")
    finally:
        tex_roi.clear_roi_memo()


# Negative controls for clause (c)'s second spelling — these must STAY False, or the ternary
# rule has become a blanket disable on `?:` (T7's corpus count is the other half of that
# guard: a per-pixel `?:` is one of the commonest shapes in `examples/*.tex`).
_TERNARY_MUST_STILL_SPLIT = [
    ("`?:` over floats on a per-pixel condition",
     "@OUT = v > 0.2 ? @A : vec4(0.0, 0.0, 0.0, 1.0);\n", None),
    ("`?:` over floats, nested",
     "@OUT = v > 0.2 ? (u > 0.5 ? @A : @A * 0.5) : vec4(0.0, 0.0, 0.0, 1.0);\n", None),
    ("a string `?:` on a UNIFORM ($param) condition",
     '@TXT = $k > 0.2 ? "hi" : "lo";\n@OUT = @A;\n', {"k": 0.5}),
    ("a string `?:` on a UNIFORM (iw) condition",
     '@TXT = iw > 4.0 ? "hi" : "lo";\n@OUT = @A;\n', None),
]


def test_t6_ternary_rule_is_not_a_blanket_disable(r: SubTestResult):
    print("\n--- T6: the `?:` shapes that MUST keep splitting ---")
    for name, src, params in _TERNARY_MUST_STILL_SPLIT:
        try:
            scalars = dict(params or {})
            assert tex_roi.region_dependent(
                _parse(src), _STRING_WIRES, code=src) is False, "declined"
            assert tex_roi.roi_plan(
                src, scalars, binding_types=_STRING_WIRES).executable is True, "roi_plan refused"
            assert tex_roi.batch_sliceable(
                src, scalars, binding_types=_STRING_WIRES) is True, "batch_sliceable refused"
            _p, whole = _cook(src, params=params)
            _p2, tiled = _cook(src, tiles=2, params=params)
            for key in whole:
                if isinstance(whole[key], torch.Tensor):
                    md = _maxdiff(whole[key], tiled[key])
                    assert md == 0.0, f"tiled != whole on @{key} (maxdiff {md})"
                else:
                    assert whole[key] == tiled[key], (whole[key], tiled[key])
            r.ok(f"still splits: {name}")
        except Exception as e:
            r.fail(f"T6 negative {name}", f"{type(e).__name__}: {e}")


# ── T2 / T9: the strip planner, and where the gate SITS ─────────────────────

# A uniform twin of the repro: the same shape of loop, bounded by a literal instead of `v`.
UNIFORM_TWIN = ("float n = 0.0;\n"
                "for (int i = 0; i < 4; i = i + 1) { n = n + 1.0; }\n"
                "@OUT = vec4(n,n,n,1.0);\n")


def _tile_plan_for(src, *, free_hint, device="cuda", fingerprint=None, shape=(1, 256, 256)):
    """Drive `_tile_plan` without a GPU: the device is a STRING the planner only compares, and
    `free_hint` is the free-VRAM reading it would otherwise buy from the host, so a low hint is
    memory pressure and a huge one is none."""
    from TEX_Wrangle import tex_engine
    bindings = {"A": _img(*shape)}
    prog = tex_api.compile(src, {"A": TEXType.VEC4})
    return tex_engine._tile_plan(prog.ast, bindings, device, 0, 4, fingerprint,
                                 free_hint=free_hint, code=src)


def test_t2_strip_planner_declines(r: SubTestResult):
    print("\n--- T2: _tile_plan under pressure ---")
    try:
        n = _tile_plan_for(UNIFORM_TWIN, free_hint=1024.0)
        assert isinstance(n, int) and n >= 2, (
            "the harness never reached the gate — a uniform program under pressure must still "
            f"get a strip count, got {n!r}")
        r.ok(f"a uniform-loop twin under pressure still strips ({n} strips)")
    except Exception as e:
        r.fail("T2 uniform twin", f"{type(e).__name__}: {e}")

    try:
        assert _tile_plan_for(REPRO, free_hint=1024.0) is None, \
            "_tile_plan still strips a region-dependent program under pressure"
        r.ok("_tile_plan(repro) under pressure is None — the cook runs whole-frame or OOMs")
    except Exception as e:
        r.fail("T2 repro", f"{type(e).__name__}: {e}")


def test_t9_gate_is_never_reached_on_an_unpressured_cook(r: SubTestResult):
    print("\n--- T9: placement (structural, never timed) ---")
    from TEX_Wrangle.tex_runtime import compiled
    calls = []
    real = tex_roi.region_dependent

    def counting(program, binding_types=None, code=None):
        calls.append(code)
        return real(program, binding_types, code)

    tex_roi.region_dependent = counting
    try:
        tex_roi._region_dep_memo.clear()
        try:
            _tile_plan_for(REPRO, free_hint=1024.0, device="cpu")
            assert calls == [], f"a CPU cook consulted the predicate {len(calls)} time(s)"
            r.ok("a CPU cook never reaches the gate (0 calls)")
        except Exception as e:
            r.fail("T9 cpu", f"{type(e).__name__}: {e}")

        calls.clear()
        try:
            assert _tile_plan_for(REPRO, free_hint=1e13) is None, "the cook should be unpressured"
            assert calls == [], f"an unpressured cook consulted the predicate {len(calls)} time(s)"
            r.ok("an unpressured CUDA-shaped cook never reaches the gate (0 calls)")
        except Exception as e:
            r.fail("T9 unpressured", f"{type(e).__name__}: {e}")

        calls.clear()
        try:
            fp = "trk25-t9-fingerprint"
            assert _tile_plan_for(REPRO, free_hint=1024.0, fingerprint=fp) is None
            assert len(calls) == 1, f"expected one walk, got {len(calls)}"
            assert _tile_plan_for(REPRO, free_hint=1024.0, fingerprint=fp) is None
            assert len(calls) == 1, "the second pressured cook re-walked instead of using the memo"
            r.ok("a pressured cook walks once and is served from the memo thereafter")
        except Exception as e:
            r.fail("T9 pressured + memo", f"{type(e).__name__}: {e}")
    finally:
        tex_roi.region_dependent = real

    try:
        tex_roi._region_dep_memo.clear()
        program = _parse(UNIFORM_TWIN)
        for i in range(tex_roi._REGION_DEP_MEMO_MAX + 40):
            tex_roi.region_dependent_cached(program, f"trk25-cap-{i}", code=UNIFORM_TWIN)
        assert len(tex_roi._region_dep_memo) <= tex_roi._REGION_DEP_MEMO_MAX, \
            f"memo grew past its cap: {len(tex_roi._region_dep_memo)}"
        # PERF-8: the key is `(fingerprint, profile)`, so the eviction check names both.
        assert (("trk25-cap-0", tex_roi._profile_key())
                not in tex_roi._region_dep_memo), "the LRU never evicted"
        r.ok(f"the memo caps at {tex_roi._REGION_DEP_MEMO_MAX} and evicts least-recently-used")
    except Exception as e:
        r.fail("T9 memo cap", f"{type(e).__name__}: {e}")

    try:
        tex_roi.region_dependent_cached(_parse(REPRO), "trk25-clear", code=REPRO)
        assert ("trk25-clear", tex_roi._profile_key()) in tex_roi._region_dep_memo
        compiled.clear_compiled_cache()
        assert len(tex_roi._region_dep_memo) == 0, "the memo survived the test-isolation reset"
        r.ok("compiled.clear_compiled_cache() clears the memo, beside _tile_safe_memo")
    except Exception as e:
        r.fail("T9 memo clear", f"{type(e).__name__}: {e}")


# ── T8: the pragma sunset, per clause ───────────────────────────────────────

def test_t8_pragma_sunsets_the_loop_clause_only(r: SubTestResult):
    print("\n--- T8: masked flow retires clauses (a)/(b), never clause (c) ---")
    try:
        assert tex_roi.MASKED_FLOW_SINCE == (0, 25)
        r.ok("MASKED_FLOW_SINCE is the single (0, 25) constant")
    except Exception as e:
        r.fail("T8 constant", f"{type(e).__name__}: {e}")

    masked = "//!tex 0.25\n" + REPRO

    # A pragma is a REQUEST, not a capability. One naming a language newer than the engine
    # implements does not block (it is advisory only), so the program still cooks under the
    # rules this engine actually has — and must therefore keep the gate. Asserted against the
    # real `LANGUAGE_VERSION` rather than a hardcoded string, so the row documents the
    # coupling instead of quietly going vacuous when that constant moves.
    try:
        assert tex_api._ver_tuple(tex_api.LANGUAGE_VERSION) < tex_roi.MASKED_FLOW_SINCE, \
            (f"this row only means something while the engine "
             f"({tex_api.LANGUAGE_VERSION}) predates masked flow")
        assert tex_roi.region_dependent(_parse(masked), code=masked) is True, \
            "a pragma the engine does not implement must not retire the loop clause"
        assert tex_roi.roi_plan(masked, {}).executable is False, \
            "roi_plan opened a window for a program the engine still cooks under old rules"
        r.ok(f"engine at {tex_api.LANGUAGE_VERSION}: `//!tex 0.25` alone retires nothing")
    except Exception as e:
        r.fail("T8 a pragma alone does not sunset", f"{type(e).__name__}: {e}")

    # …and the day `LANGUAGE_VERSION` reaches masked flow, the sunset starts working exactly
    # as designed, with no further edit here.
    _real_version = tex_api.LANGUAGE_VERSION
    try:
        tex_api.LANGUAGE_VERSION = "0.25"
        tex_roi.clear_roi_memo()
        assert tex_roi.region_dependent(_parse(masked), code=masked) is False, \
            "engine and program both at masked flow: clauses (a)/(b) must retire"
        assert tex_roi.roi_plan(masked, {}).executable is True, \
            "roi_plan still refuses a program both halves agree is pointwise"
        r.ok("engine at 0.25: the sunset fires, with no second migration")

        assert tex_roi.region_dependent(_parse(REPRO), code=REPRO) is True, \
            "a 0.25 engine must not retire the clause for a program that asked for 0.23"
        r.ok("engine at 0.25, no pragma: still declined (the program did not ask)")

        masked_string_both = "//!tex 0.25\n" + STRING_REPRO
        assert tex_roi.region_dependent(_parse(masked_string_both),
                                        code=masked_string_both) is True, \
            "clause (c) must NOT sunset: masked flow keeps the majority vote verbatim"
        r.ok("clause (c) survives both halves, which is why the gate is per-clause")
    except Exception as e:
        r.fail("T8 sunset once the engine implements it", f"{type(e).__name__}: {e}")
    finally:
        tex_api.LANGUAGE_VERSION = _real_version
        tex_roi.clear_roi_memo()

    for label, src in (("an older pragma", "//!tex 0.24\n" + REPRO),
                       ("no pragma", REPRO),
                       ("a pragma after real code", "@OUT = @A;\n//!tex 0.25\n" + REPRO)):
        try:
            assert tex_roi.region_dependent(_parse(src), code=src) is True
            r.ok(f"still declined with {label}")
        except Exception as e:
            r.fail(f"T8 {label}", f"{type(e).__name__}: {e}")

    try:
        masked_string = "//!tex 0.25\n" + STRING_REPRO
        assert tex_roi.region_dependent(_parse(masked_string), code=masked_string) is True, \
            "clause (c) must NOT sunset: the masked-flow rules keep the majority vote verbatim"
        r.ok("the string clause survives the pragma, which is why the gate is per-clause")
    except Exception as e:
        r.fail("T8 clause (c) does not sunset", f"{type(e).__name__}: {e}")


# ── T10: the advisory ────────────────────────────────────────────────────────

def _codes_by_line(source, binding_types=None):
    out = {}
    for d in tex_api.control_flow_advisories(source, binding_types or {}):
        out.setdefault(d.loc.line, []).append(d.code)
    return out


def test_t10_w7008_names_what_the_engine_now_refuses(r: SubTestResult):
    print("\n--- T10: W7008, opt-in, beside W7006/W7007 ---")
    try:
        got = _codes_by_line(REPRO)
        assert got == {3: ["W7007", "W7008"]}, got
        r.ok("W7008 lands on the loop line, beside W7007")
    except Exception as e:
        r.fail("T10 loop clause", f"{type(e).__name__}: {e}")

    try:
        got = _codes_by_line(STRING_REPRO)
        assert got == {2: ["W7008"]}, got
        diag = tex_api.control_flow_advisories(STRING_REPRO, {})[0]
        assert "majority vote" in diag.message, diag.message
        assert diag.severity == "warning", diag.severity
        r.ok("W7008 alone marks the string merge — there is no W7007 for it")
    except Exception as e:
        r.fail("T10 string clause", f"{type(e).__name__}: {e}")

    try:
        # The asymmetry that makes a new code worth having: an escape under a per-pixel guard
        # is W7007 (it acts on every pixel) but NOT W7008 (it still splits correctly). No
        # SHIPPED example has that shape any more — v0.35.1 rewrote the six that did, so
        # `examples/break_search.tex` now breaks on a uniform test and draws nothing — so the
        # row states it on the shape itself and pins the example's silence beside it.
        escape = ("float n = 0.0;\n"
                  "for (int i = 0; i < 8; i++) { if (@A.r > 0.5) { break; } n = n + 1.0; }\n"
                  "@OUT = vec4(n,n,n,1.0);\n")
        codes = {d.code for d in tex_api.control_flow_advisories(escape, {})}
        assert codes == {"W7007"}, codes
        assert tex_roi.region_dependent(_parse(escape), code=escape) is False
        shipped = {d.code for d in tex_api.control_flow_advisories(
            _read_repo("examples", "break_search.tex"), {})}
        assert shipped == set(), shipped
        r.ok("a break under a per-pixel guard is W7007 and never W7008")
    except Exception as e:
        r.fail("T10 W7007 without W7008", f"{type(e).__name__}: {e}")

    try:
        # DOC-8 gave `string_format.tex` real `{}` placeholders (they were `%f`/`%s`, never
        # filled), so it now correctly carries W7008 beside the five W7006 files that were
        # already here: an opt-in host reading the advisories sees exactly that surface.
        by_file = {}
        for name, src in _example_sources():
            codes = sorted({d.code for d in tex_api.control_flow_advisories(src, {})})
            if codes:
                by_file[name] = codes
        w7006_files = {k: v for k, v in by_file.items() if k != "string_format.tex"}
        assert all(c == ["W7006"] for c in w7006_files.values()), w7006_files
        assert len(w7006_files) == 5, w7006_files
        assert by_file.get("string_format.tex") == ["W7008"], by_file
        r.ok(f"the shipped advisory surface: {sorted(w7006_files)} carry W7006 only, plus "
             f"string_format.tex carrying W7008")
    except Exception as e:
        r.fail("T10 shipped advisory surface", f"{type(e).__name__}: {e}")

    try:
        from TEX_Wrangle import tex_lsp
        leaked = []
        for name, src in _example_sources():
            for d in tex_api.check(src, {}):
                if d.code in ("W7007", "W7008"):
                    leaked.append((name, d.code))
        for src in (REPRO, STRING_REPRO):
            leaked += [(src[:12], d.code) for d in tex_api.check(src, {})
                       if d.code in ("W7007", "W7008")]
            leaked += [(src[:12], d.get("code")) for d in tex_lsp.diagnostics_for(src, {})
                       if d.get("code") in ("W7007", "W7008")]
        assert leaked == [], leaked
        r.ok("check() and tex_lsp show neither code, for any shipped example or either repro")
    except Exception as e:
        r.fail("T10 invisibility", f"{type(e).__name__}: {e}")

    try:
        page = _read_repo("Error-Codes.md")
        assert "### W7008" in page, "Error-Codes.md is stale — regenerate it"
        assert "W7008" in _read_repo("LANGUAGE.md"), "LANGUAGE.md does not document the code"
        r.ok("the generated error-code page and LANGUAGE.md both carry W7008")
    except Exception as e:
        r.fail("T10 documented surface", f"{type(e).__name__}: {e}")


# ── T12: the frozen compat corpus ───────────────────────────────────────────

def test_t12_corpus_neutrality(r: SubTestResult):
    print("\n--- T12: the frozen corpus is unmoved, and declines exactly two programs ---")
    import compat_corpus
    try:
        # DOC-8 added `string_format` to this set: its format() calls now fill `{}`
        # placeholders instead of leaving dead `%f`/`%s` sequences unfilled, so its five
        # numeric calls genuinely cast a per-region-varying value to a string.
        declined = sorted(name for name, src in compat_corpus._corpus_programs()
                          if tex_roi.region_dependent(_parse(src), code=src))
        assert declined == ["adv_while_loop", "string_format"], declined
        r.ok("exactly two corpus programs are region-dependent, by name: "
             "adv_while_loop, string_format")
    except Exception as e:
        r.fail("T12 declined set", f"{type(e).__name__}: {e}")

    try:
        # The goldens are WHOLE-FRAME cooks through the interpreter, which no planner gates,
        # so a decline cannot move one. Pinned on the one program that is declined.
        src = compat_corpus._ADVERSARIAL["adv_while_loop"]
        got = compat_corpus._program_hash(src)
        versions = compat_corpus.archived_versions()
        assert versions, "no frozen corpus versions to check against"
        for version in versions:
            frozen = compat_corpus.load_version(version)
            hashes = frozen.get("hashes", frozen)
            want = hashes.get("adv_while_loop")
            assert want is not None, f"{version} has no adv_while_loop golden"
            assert got == want, f"{version}: golden moved ({got} != {want})"
        r.ok(f"adv_while_loop hashes identically against {len(versions)} frozen version(s)")
    except Exception as e:
        r.fail("T12 golden unmoved", f"{type(e).__name__}: {e}")


# ── T13: clause (d) — a per-pixel value cast straight to a STRING (TRK-32) ──────────────────

def test_t13_scalar_cast_is_region_dependent(r: SubTestResult):
    print("\n--- T13: string(x) / str(x) / format('{}', x) on a per-pixel value ---")
    for name, src in (("string(x) cast", CAST_REPRO),
                      ("str(x)", STR_REPRO),
                      ("format('{}', x)", FORMAT_REPRO)):
        try:
            assert tex_roi.region_dependent(_parse(src), code=src) is True
            assert tex_roi.roi_plan(src, {}).executable is False
            assert tex_roi.batch_sliceable(src, {}) is False
            r.ok(f"{name} is region-dependent and all three routes decline it")
        except Exception as e:
            r.fail(f"T13 predicate and routes ({name})", f"{type(e).__name__}: {e}")

    for name, src in (("string(x) cast", CAST_REPRO),
                      ("str(x)", STR_REPRO),
                      ("format('{}', x)", FORMAT_REPRO)):
        try:
            # The evidence: `_scalar_from_tensor` takes the MEAN of whatever region it is
            # handed. The whole frame and a 2-strip split hand it different pixel sets, so
            # the averaged string differs — bit-identical to TRK-32's own measurement.
            _p, whole = _cook(src)
            _p2, tiled = _cook(src, tiles=2)
            assert whole["TXT"] != tiled["TXT"], (
                f"{name}: whole ({whole['TXT']!r}) and tiled ({tiled['TXT']!r}) must differ "
                "for this to be evidence of anything")
            r.ok(f"{name}: whole={whole['TXT']!r} tiled={tiled['TXT']!r}")
        except Exception as e:
            r.fail(f"T13 characterization ({name})", f"{type(e).__name__}: {e}")

    try:
        _p, whole = _cook(CAST_REPRO)
        _p2, tiled = _cook(CAST_REPRO, tiles=2)
        assert whole["TXT"] == "4.761904716491699", whole["TXT"]
        assert tiled["TXT"] == "2.222222328186035", tiled["TXT"]
        r.ok("string(x): pinned to TRK-32's own two numbers")
    except Exception as e:
        r.fail("T13 pinned numbers", f"{type(e).__name__}: {e}")


def test_t13_percent_style_format_is_not_in_the_class(r: SubTestResult):
    print("\n--- T13: format('%f', x) is NOT in this class (the documented correction) ---")
    try:
        assert tex_roi.region_dependent(_parse(FORMAT_PERCENT_REPRO),
                                        code=FORMAT_PERCENT_REPRO) is False
        assert tex_roi.roi_plan(FORMAT_PERCENT_REPRO, {}).executable is True
        assert tex_roi.batch_sliceable(FORMAT_PERCENT_REPRO, {}) is True
        r.ok("format('%f', x) is not declined by any route")
    except Exception as e:
        r.fail("T13 percent-style predicate", f"{type(e).__name__}: {e}")

    try:
        # `%f` is not a `{}`/`{:spec}` placeholder, so fn_format returns the template
        # UNCHANGED on every route — the reduced value never reaches the output, whole or
        # tiled, so there is genuinely nothing to decline.
        _p, whole = _cook(FORMAT_PERCENT_REPRO)
        _p2, tiled = _cook(FORMAT_PERCENT_REPRO, tiles=2)
        assert whole["TXT"] == tiled["TXT"] == "%f", (whole["TXT"], tiled["TXT"])
        r.ok(f"characterization: whole and tiled both give {whole['TXT']!r}")
    except Exception as e:
        r.fail("T13 percent-style characterization", f"{type(e).__name__}: {e}")


def test_t13_pragma_never_sunsets_clause_d(r: SubTestResult):
    print("\n--- T13: clause (d) survives the masked-flow pragma, like clause (c) ---")
    masked_cast = "//!tex 0.25\n" + CAST_REPRO
    try:
        assert tex_api._ver_tuple(tex_api.LANGUAGE_VERSION) < tex_roi.MASKED_FLOW_SINCE, (
            f"this row only means something while the engine "
            f"({tex_api.LANGUAGE_VERSION}) predates masked flow")
        assert tex_roi.region_dependent(_parse(masked_cast), code=masked_cast) is True
        r.ok(f"engine at {tex_api.LANGUAGE_VERSION}: `//!tex 0.25` alone retires nothing")
    except Exception as e:
        r.fail("T13 pragma alone does not sunset", f"{type(e).__name__}: {e}")

    # Masked per-pixel control flow (0.25) changes how a LOOP runs its body — it says nothing
    # about how `_scalar_from_tensor` reduces a tensor to a string, so clause (d) must still
    # fire even once the ENGINE implements 0.25, exactly like clause (c) (§1.5: "the majority
    # rule verbatim"). This is why region_dependent gates clause (d) beside (c), never with
    # the loop clauses (a)/(b) that DO sunset at MASKED_FLOW_SINCE.
    _real_version = tex_api.LANGUAGE_VERSION
    try:
        tex_api.LANGUAGE_VERSION = "0.25"
        tex_roi.clear_roi_memo()
        assert tex_roi.region_dependent(_parse(masked_cast), code=masked_cast) is True, \
            "clause (d) must NOT sunset: masked flow says nothing about the string reduction"
        r.ok("clause (d) survives an engine that implements masked flow too")
    except Exception as e:
        r.fail("T13 sunset once the engine implements it", f"{type(e).__name__}: {e}")
    finally:
        tex_api.LANGUAGE_VERSION = _real_version
        tex_roi.clear_roi_memo()


def test_t13_advisory_and_corpus_are_unaffected(r: SubTestResult):
    print("\n--- T13: W7008 on the cast line; the shipped surface is unmoved ---")
    try:
        got = _codes_by_line(CAST_REPRO)
        assert got == {1: ["W7008"]}, got
        diag = tex_api.control_flow_advisories(CAST_REPRO, {})[0]
        assert "MEAN" in diag.message, diag.message
        assert diag.severity == "warning", diag.severity
        r.ok("W7008 marks the cast line")
    except Exception as e:
        r.fail("T13 advisory", f"{type(e).__name__}: {e}")

    try:
        # `examples/string_format.tex` is the ONLY shipped program that even mentions
        # string(/str(/format(. Before DOC-8 every one of its format() calls was
        # `%f`/`%s`-style — dead sequences `format()` has never filled (CHANGELOG's ASK-9
        # entry) — so clause (d) never touched it. DOC-8 corrected them to real `{}`
        # placeholders so the statistics the example computes actually reach its output,
        # and that means its five numeric calls now genuinely cast a per-region-varying
        # value straight to a string: clause (d) correctly fires, one W7008 per line. The
        # `[{}]`-around-`$label` line does not — a string parameter is not per-pixel-varying.
        src = _read_repo("examples", "string_format.tex")
        assert tex_roi.region_dependent(_parse(src), code=src) is True
        codes = _codes_by_line(src)
        assert codes == {54: ["W7008"], 55: ["W7008"], 56: ["W7008"],
                          57: ["W7008"], 58: ["W7008"]}, codes
        r.ok("string_format.tex: now correctly declined, W7008 on its five numeric "
             "format() lines")
    except Exception as e:
        r.fail("T13 string_format.tex", f"{type(e).__name__}: {e}")
