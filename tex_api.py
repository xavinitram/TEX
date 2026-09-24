"""
PORT-2 — the public, host-agnostic API facade.

A small stable surface over the internal compile/execute machinery, for the `tex run`
CLI (PORT-3) and future non-ComfyUI hosts. It calls straight through — `execute()` is
bit-for-bit identical to `Interpreter.execute`, and `compile()` wraps the cache's
6-tuple. The `Program` dataclass field NAMES are a public contract (a canary test pins
them), so a host can depend on `program.assigned` / `program.type_map` etc.

**Scope (doc 32 F4):** `execute()` runs the INTERPRETER ONLY and returns its RAW
per-output tensors — no tier selection, no fallbacks, no tiling, and deliberately none of
the ComfyUI IMAGE post-formatting (clamp to [0,1], alpha-drop to 3 channels, gray→RGB
broadcast, MASK/LATENT typing). So `execute(compile("@OUT=vec4(@A.rgb*3,1);"),
{"A": half_grey})` returns a `[1,H,W,4]` tensor with values to 1.5, where the node returns
a clamped `[1,H,W,3]` IMAGE.

**Which entry point do you want?** (v0.22 added the middle one, and it is usually the
answer — before ENG-1 the only way to reach the real engine was to import the ComfyUI node.)

    tex_api.execute      the interpreter, nothing else. The oracle. Bit-for-bit reference.
    tex_engine.cook      the ENGINE: tiers, fallbacks, OOM ladder, tiling, precision-auto.
                         What a host should cook with. Returns raw tensors.
    TEXWrangleNode.execute   the ComfyUI adapter. Only if you ARE ComfyUI.

For pixel-identical ComfyUI output without ComfyUI, cook through `tex_engine.cook` and
apply the `comfy` egress profile (ENG-3) — that pairing IS the node's conversion, and it
is what `tex run` does:

    from TEX_Wrangle import tex_engine
    from TEX_Wrangle.tex_marshalling import prepare_output, map_inferred_type
    res = tex_engine.cook(src, {"A": img}, device_mode="cuda")
    img_out = prepare_output(res.outputs["OUT"], map_inferred_type(res.assigned["OUT"], False),
                             profile="comfy")

`cook()` guarantees its outputs do not alias your input bindings; `execute()` makes no such
promise (`@OUT = @A;` hands back the tensor you passed in).

    from TEX_Wrangle.tex_api import compile, execute
    from TEX_Wrangle.tex_compiler.types import TEXType
    prog = compile("@OUT = vec4(@A.rgb * 1.2, 1.0);", {"A": TEXType.VEC3})
    out = execute(prog, {"A": img}, device="cuda", precision="auto")  # raw, unclamped
"""
import re as _re
from dataclasses import dataclass
from typing import Any

# ENG-4: the public error type + its payload, re-exported so a host imports ONE module.
# `diagnostic_from_exc` is the shared per-phase→diagnostic materializer (it lives in
# diagnostics.py so the engine raiser and check() share it without an import cycle).
from .tex_compiler.diagnostics import (  # noqa: F401
    TEXCompileError, TEXDiagnostic, diagnostic_from_exc as _diag_from_exc)

# SCHED-3: the cooperative-cancellation exception + token protocol, re-exported so a host
# catches CookCancelled and wires a CancelToken from the same facade it cooks through.
from .tex_runtime.host import CookCancelled, CancelToken  # noqa: F401

# DATA-1: the buffer-metadata sidecar — a host tags inputs with colour/alpha/frame, passes
# them to `tex_engine.cook(binding_meta=...)`, and reads the merged tags back off
# `CookResult.out_meta`. The tags + merge policy live in tex_marshalling (the wire seam); the
# W7005 gamma-halo lint is `color_advisories` below (an analysis, so it lives beside check()).
from .tex_marshalling import (  # noqa: F401
    BufferMeta, COLORSPACES, PREMULT, merge_buffer_meta,
)

# DATA-4: the engine session — one handle a standalone host holds over the process's cook state
# (program cache, governor, host services, interpreter) with `reset()` / `close()` / `stats()`.
# Phase 1 is a view of the module singletons (ComfyUI byte-identical); see tex_session.
from .tex_session import EngineSession, default_session  # noqa: F401

# LANG-3: the TEX LANGUAGE version — grammar + semantics — versioned SEPARATELY from the
# package `__version__`. A program may declare the language level it targets with a leading
# `//!tex X.Y` pragma; `check()` advises (W7004) when a program targets a NEWER language
# than this engine implements. The frozen compat corpus (tests/) pins that a program keeps
# computing the same pixels across versions. See LANGUAGE.md for the compatibility policy.
LANGUAGE_VERSION = "0.25"


def language_pragma(source: str):
    """Return the language version a program targets via a LEADING `//!tex X.Y` pragma (as
    the string 'X.Y'), or None. Only a pragma in the header run of blank / `//` line-comment
    lines is recognized — one buried after real code or inside a `/* … */` block comment is
    ignored (it would otherwise raise a spurious W7004).

    LANG-L1: the scan itself now lives in `tex_compiler.parser` (what `Parser.parse` calls
    to set `Program.language`); this delegates so there is exactly one implementation. Kept
    here, rather than moved outright, because `check()` needs the pragma before a parse is
    even attempted (a lexer/parser error must not suppress the W7004 advisory below), and
    because this name is documented public API a host may already import from `tex_api`."""
    from .tex_compiler.parser import language_pragma as _parser_language_pragma
    return _parser_language_pragma(source)


def _ver_tuple(v: str) -> tuple:
    """Parse a dotted version string into a tuple of ints, one per `.`-separated component,
    for ordering comparisons (`>`/`<`/`min`/`max`) against another such tuple — never for
    display. Tolerant per component rather than all-or-nothing: a component with no leading
    digit degrades to `0` in place (`"1.abc"` -> `(1, 0)`) instead of collapsing the whole
    result to a sentinel, so a comparison against a well-formed operand (every real call
    site's other side: `LANGUAGE_VERSION`, a package version, a source pragma) still reads
    the well-formed components correctly. TRK-144: this was `tex_tool.py`'s copy (the same
    two use sites, `raw["tex_language"]`/LANGUAGE_VERSION and `min_engine`/the package
    version, both unvalidated-format strings a manifest author writes by hand) folded in
    here as the one definition; `tex_tool.load_tool` now imports it instead of redefining
    it."""
    parts = []
    for chunk in str(v).split("."):
        m = _re.match(r"\d+", chunk)
        parts.append(int(m.group()) if m else 0)
    return tuple(parts)


@dataclass(frozen=True)
class Program:
    """A compiled TEX program — a named view over the compiler's 6-tuple. The field
    names are a public contract (`test_port2_program_shape` pins them)."""
    ast: Any
    type_map: dict
    referenced: Any
    assigned: dict
    params: dict
    used_builtins: Any
    source: str


def _compile_impl(source: str, binding_types: dict, *, fp: str | None = None):
    """Shared body of `compile` and `prewarm`'s per-program compile step (TRK-73): the caller
    may already hold `TEXCache.fingerprint(source, binding_types)` — `prewarm` needs the same
    string again for the codegen sidecar, the background-compile key and the capturability
    verdict — and handing it through here, exactly as `tex_engine.prepare` (LAT-2) already
    forwards its own `fp` into `_compile_or_raise`, means the string is computed once and
    passed down instead of once per caller. Returns `(Program, fp)` so a caller that did not
    already have `fp` gets it back without a second `fingerprint()` call of its own."""
    from . import tex_engine
    from .tex_cache import get_cache
    if fp is None:
        fp = get_cache().fingerprint(source, binding_types)
    ast, type_map, referenced, assigned, params, used_builtins = \
        tex_engine._compile_or_raise(source, binding_types, fp=fp)
    return Program(ast, type_map, referenced, assigned, params, used_builtins, source), fp


def compile(source: str, binding_types: dict) -> Program:  # noqa: A001 (public name)
    """Compile TEX `source` to a `Program`. `binding_types` maps input binding names to
    their `TEXType`.

    Raises `TEXCompileError` (ENG-4) on invalid code — one public type carrying
    `.diagnostics`, instead of the four internal per-phase exceptions a caller used to
    have to know. Everything else propagates unchanged.

    The raw per-phase catch now lives in ONE place — `tex_engine._compile_or_raise` (the
    raiser `prepare()` and the ComfyUI node also flow through) — so this delegates to it
    rather than re-knowing the tuple itself.
    """
    program, _fp = _compile_impl(source, binding_types)
    return program


def execute(program: Program, bindings: dict, *, device: str = "cpu",
            precision: str = "fp32", output_names=None, cancel=None, on_progress=None) -> dict:
    """Execute a compiled `Program` against `bindings` → `{output_name: tensor}`. Calls
    straight through to `Interpreter.execute` (bit-for-bit identical).

    SCHED-3: an optional `cancel` token (`.check()` raising `CookCancelled`) is polled per
    top-level statement, and `on_progress(phase, frac)` reports statement progress — the
    thinnest surface that exercises the interpreter's cancellation seam directly."""
    from .tex_runtime.interpreter import Interpreter
    outs = output_names if output_names is not None else sorted(program.assigned.keys())
    return Interpreter().execute(
        program.ast, bindings, program.type_map, device=device,
        output_names=outs, precision=precision, used_builtins=program.used_builtins,
        cancel=cancel, on_progress=on_progress)


def check(source: str, binding_types: dict) -> list:
    """LANG-2: compile-only diagnostics. Lex, parse and type-check `source` and return a
    `list[TEXDiagnostic]` (errors AND W7xxx warnings) — and NEVER raise. This is the
    backend for the editor's live-lint (`/tex_wrangle/check`) and any future LSP.

    `binding_types` maps @input names to `TEXType`; pass `{}` when the caller does not
    know the wired types yet (undeclared inputs then resolve to VEC4, exactly as at cook
    time). A lexer or parser error is fatal for deeper analysis, so it is returned alone;
    only a clean parse reaches the type checker, which accumulates every type error plus
    the W7xxx advisories in one pass. Diagnostics carry `.severity` ('error' | 'warning')
    so a consumer can render errors and warnings differently.

    Unlike `compile()`, which raises `TEXCompileError`, `check()` is total: it always
    returns a list, empty when the program is clean."""
    from .tex_cache import parse_and_split
    from .tex_compiler.lexer import LexerError
    from .tex_compiler.parser import ParseError
    from .tex_compiler.type_checker import TypeChecker, TypeCheckError
    from .tex_compiler.diagnostics import make_diagnostic, TEXMultiError

    # LANG-3: a program targeting a NEWER language than we implement gets an up-front
    # advisory (independent of whether it then compiles), so a version mismatch is not
    # mistaken for an ordinary syntax error.
    pragma_diags = []
    pragma = language_pragma(source)
    if pragma and _ver_tuple(pragma) > _ver_tuple(LANGUAGE_VERSION):
        pragma_diags.append(make_diagnostic(
            code="W7004",
            message=f"This program targets TEX language {pragma}, newer than this "
                    f"engine's {LANGUAGE_VERSION}; newer features may not compile, and "
                    f"will be cooked under the older rules and may compute differently.",
            loc=None, source=source, phase="compile", severity="warning"))

    try:
        # DATA-6: the one front end (`tex_cache.parse_and_split`) — lex, parse and resolve
        # every dotted `@name.seg` against `binding_types` — so the lint reads a swizzle's BASE
        # as the wire, exactly as the cook does. A private lexer here would report `@image.g`
        # as an unconnected vec4 binding (a false E3200) and `image` as never used (a false
        # W7002). The splitback's own refusals (E2000 / E2002, the swizzle sugar) are
        # TypeCheckErrors raised BEFORE the checker and are returned as the single fatal
        # diagnostic a parse error would be.
        try:
            program = parse_and_split(source, binding_types)
        except TEXMultiError as e:
            return pragma_diags + list(e.diagnostics)
        except (LexerError, ParseError, TypeCheckError) as e:
            return pragma_diags + [_diag_from_exc(e, source)]
        errors, warnings = TypeChecker(
            binding_types=binding_types, source=source).check_collect(program)
        return pragma_diags + errors + warnings
    except Exception as e:  # the contract is absolute: check() must never raise
        return pragma_diags + [make_diagnostic(
            code="E0000", message=f"internal error during check(): {e}",
            loc=None, source=source, phase="compile")]


# W7005 (DATA-1 phase 2): the gamma-space halo hazard. A spatial op — blur, morphology, any
# neighbourhood gather — averages a pixel's neighbours; averaging in a NON-LINEAR space (srgb,
# oklab) darkens edges and shifts hue, the mistake the stdlib docstrings warn about in prose.
# (W7004 above is LANG-3's language-version advisory; this is the next free warning code.)
_NONLINEAR_SPACES = ("srgb", "oklab")


def color_advisories(source: str, param_values: dict | None, binding_meta: dict | None) -> list:
    """W7005 diagnostics (DATA-1): a non-pointwise read of a non-linearly-tagged buffer. A
    host/editor advisory that pairs the program's spatial footprints (`tex_roi.binding_footprints`,
    the ROI-2 substrate) with the host's per-input colour tags (`BufferMeta.colorspace`). Pure
    analysis — no cook, no side effects — and total (never raises). Returns [] when the host
    tagged nothing non-linear (the default path) or the footprint analysis is unavailable; it is
    off the cook path. It lives here beside `check()`, the host-facing diagnostics surface."""
    if not binding_meta:
        return []
    tagged = {n: m for n, m in binding_meta.items() if m.colorspace in _NONLINEAR_SPACES}
    if not tagged:
        return []
    from . import tex_roi
    from .tex_compiler.diagnostics import make_diagnostic
    fps = tex_roi.binding_footprints(source, param_values or {})
    if not fps:
        return []
    out = []
    for name, meta in tagged.items():
        fp = fps.get(name)
        if fp is not None and fp.kind != "point":
            out.append(make_diagnostic(
                "W7005",
                f"A spatial operation reads '@{name}', tagged {meta.colorspace}: averaging a "
                f"neighbourhood in a non-linear space darkens edges and shifts hue. Convert to "
                f"linear first (e.g. srgb_to_linear), operate, then convert back.",
                loc=None, source=source, severity="warning", phase="type_checker"))
    return out


# W7006 / W7007: control flow on a condition that can differ from pixel to pixel
# (LANGUAGE.md §7.1). The engine decides an `if` by the RANK of its condition: a 0-dim value
# takes one branch; anything else runs BOTH branches on every pixel and merges them with
# torch.where. So a gather in a per-pixel branch is paid everywhere (W7006), and a
# break/continue/return under one raises past the merge and acts on every pixel, while a
# per-pixel loop condition runs every pixel to the frame's maximum, unmasked (W7007).
#
# OPT-IN by construction: nothing calls this but a host. It is never reached from `check()`,
# so neither the editor's `/tex_wrangle/check` live lint nor `tex_lsp` shows these codes —
# existing programs gain no new squiggles.
#
# "Can differ" over-approximates: an `@` wire (a FLOAT type cannot tell a mask from a
# scalar) unless typed STRING; `u v ix iy fi`; a vector parameter (staged as a tensor, whose
# component is not 0-dim on the interpreter); a call to a builtin whose ROI footprint is not
# 'point' (a reduction keeps its dims); a call with such an argument or to a user function
# whose result can differ; a local computed from any of these, or merged by a per-pixel `if`;
# and a function parameter some call site feeds such a value. Literals, scalar `$params`,
# `iw ih px py fn ic PI TAU E frame fps time` and counters of loops bounded by those never
# count. The walk is flow-sensitive (a plain `x = …;` re-decides `x`) with each loop iterated
# to a fixed point, and follows the engine's merge rule (`collect_assigned_vars`): a merged
# name broadcasts only when it existed before the `if` or both branches define it.
#
# W7008 (TRK-25) is the subset of that the ENGINE now acts on: a loop whose condition can
# differ per pixel, or a string chosen per pixel by an `if` or a `?:`. Those shapes make the
# output depend on WHICH REGION is cooked (the loop runs to the region's maximum; each string
# merge is a region-wide majority vote), so the planners decline to split the cook —
# see `tex_roi.region_dependent`. W7007 also covers `break`/`continue`/`return` under a
# per-pixel `if`, which is region-INDEPENDENT (the escape fires on first arrival, identically
# in every region) and therefore draws no W7008.
_PER_PIXEL_BUILTINS = frozenset(("u", "v", "ix", "iy", "fi"))
_VEC_PARAM_HINTS = frozenset(("c", "v", "v2", "v3", "v4"))
_CF_KEYWORD_LEN = {"IfElse": 2, "ForLoop": 3, "WhileLoop": 5, "BreakStmt": 5,
                   "ContinueStmt": 8, "ReturnStmt": 6, "TernaryOp": 1}

#: TRK-32 clause (d): a `format()` template only actually substitutes a Python-style
#: `{}` / `{:spec}` placeholder (`stdlib.fn_format`'s own contract) — `%f`/`%s` are not
#: placeholders and pass through literally, so a template without one never lets a
#: reduced argument reach the output. Deliberately over-inclusive (a literal `{{}}`
#: escape would still match) rather than under: the safe direction is to decline.
_FORMAT_PLACEHOLDER_RE = _re.compile(r"\{[^{}]*\}")

_STRING_RET_CACHE: "frozenset | None" = None


def _string_ret_fns() -> frozenset:
    """Stdlib function names whose declared return type is STRING, from the compiler-side
    signature table (the single source). Lazy, like `tex_roi._footmap`."""
    global _STRING_RET_CACHE
    if _STRING_RET_CACHE is None:
        from .tex_compiler.stdlib_signatures import FUNCTION_SIGNATURES
        from .tex_compiler.types import TEXType
        _STRING_RET_CACHE = frozenset(
            name for name, sig in FUNCTION_SIGNATURES.items()
            if isinstance(sig, dict) and sig.get("return") is TEXType.STRING)
    return _STRING_RET_CACHE


def control_flow_advisories(source: str, binding_types: dict) -> list:
    """W7006 / W7007 / W7008 diagnostics — opt-in advisories for control flow on a condition that
    can differ from pixel to pixel (LANGUAGE.md §7.1). `binding_types` is the same
    `{name: TEXType}` map `check()` takes; only a STRING type changes the result (a string
    wire never differs per pixel).

      * **W7006** — an `if` or `?:` whose condition can differ per pixel, with a gather in a
        branch: `@A(u, v)` / `@A[x, y]`, a builtin whose footprint is not a point
        (`sample`, `fetch`, a blur, a reduction), or a user function whose body holds one.
        Both branches run on every pixel, so the gather is never skipped.
      * **W7007** — control flow that acts on every pixel: `break` / `continue` / `return`
        under such an `if`, or a `for` / `while` whose condition can differ per pixel.
        CONDITIONAL since `0.25` (LANG-L7): false, and never fires, for a program actually
        cooked under `0.25`'s masked per-pixel control flow (`min(pragma,
        LANGUAGE_VERSION) >= (0, 25)`) — such a program masks rather than acts on every
        pixel. Still fires exactly as before for a program cooked below `0.25`.
      * **W7008** — control flow whose result depends on WHICH REGION is cooked, so the engine
        declines to split the cook (`tex_roi.region_dependent`): a `for` / `while` whose
        condition can differ per pixel (the loop runs to the region's maximum), or a string
        chosen per pixel by an `if` or a `?:` (either is a region-wide majority vote; a string
        arriving on a wire needs `binding_types` to be seen at all). Strictly the
        subset of W7007 the engine ACTS on — a `break` / `continue` / `return` under a
        per-pixel `if` draws W7007 and no W7008, because it fires on first arrival and so does
        the same thing in every region. The LOOP half sunsets in lockstep with W7007 above
        (a masked loop's pass count is each pixel's own, not the region's maximum); the
        STRING halves (a per-pixel string choice, or a per-pixel value cast straight to a
        string) never sunset at any language level — a string has no per-pixel form.

    Never emitted by `check()`: a host calls this beside it. Pure AST analysis — no compile,
    no cook, no side effects — and total: a program that does not parse, a non-string source,
    or an analysis that cannot finish within its work budget, returns [].

    A lint that CRASHES is a different thing from a lint that declines, and it used to be
    spelled the same way. `except Exception: return []` answered "no problems" for an
    internal failure, so an editor drew a clean gutter and the node then cooked the program
    the editor had called clean — the one failure mode a lint must not have. The three
    DECLINE cases above still return [] (they are answers, and the parse error is already on
    its way to the same editor from `check()`). Anything else now comes back as ONE synthetic
    `E0000` naming the exception type, exactly the code and shape `check()` uses for its own
    internal failure, so the return TYPE is unchanged and a host needs no new branch."""
    if not isinstance(source, str):
        return []
    from .tex_compiler.lexer import LexerError
    from .tex_compiler.parser import ParseError
    from .tex_compiler.type_checker import TypeCheckError
    from .tex_compiler.diagnostics import make_diagnostic, TEXMultiError

    def _internal(e, where):
        return [make_diagnostic(
            code="E0000",
            message=(f"internal error during control_flow_advisories() {where}: "
                     f"{type(e).__name__}: {e}"),
            loc=None, source=source, phase="compile")]

    try:
        # DATA-6: through the one front end, so a swizzled wire is seen by its BASE name —
        # the name `binding_types` (and `string_wires` below) key on.
        from .tex_cache import parse_and_split
        program = parse_and_split(source, binding_types)
    except (TEXMultiError, LexerError, ParseError, TypeCheckError):
        return []          # a program that does not parse has no advisories to give
    except Exception as e:
        return _internal(e, "parse")
    try:
        return _ControlFlowLint(program, source, binding_types).run()
    except _CFBudget:
        return []          # over budget is a DECLINE; the analysis stays total by refusing
    except Exception as e:  # the contract is absolute, as for check(): never raise
        return _internal(e, "analysis")


class _CFBudget(Exception):
    """The analysis ran past its work budget (a pathological nesting of loops)."""


class _CFState:
    """The abstract state at a program point: names whose value can differ per pixel, and
    names that may be defined (the merge rule needs the latter)."""
    __slots__ = ("vary", "defined")

    def __init__(self, vary=(), defined=()):
        self.vary = set(vary)
        self.defined = set(defined)

    def copy(self):
        return _CFState(self.vary, self.defined)

    def join(self, other):
        self.vary |= other.vary
        self.defined |= other.defined

    def same(self, other):
        return self.vary == other.vary and self.defined == other.defined


class _CFLoop:
    """The states that leave a loop body early: at a `break` or a `continue`."""
    __slots__ = ("breaks", "continues")

    def __init__(self):
        self.breaks = []
        self.continues = []


class _CFScope:
    """One analysis scope (the top level, or one function body)."""
    __slots__ = ("returns_vary", "record_calls")

    def __init__(self, record_calls):
        self.returns_vary = False
        self.record_calls = record_calls


class _ControlFlowLint:
    def __init__(self, program, source, binding_types):
        from .tex_compiler import ast_nodes as A
        from . import tex_roi
        self.A = A
        self.source = source
        self.program = program   # LANG-L7: needed for the W7007/W7008 masked-flow sunset
        self.footmap = tex_roi._footmap()
        self._tex_roi = tex_roi   # LANG-L3: reuse tex_roi's own scatter-target unwrap below
                                   # rather than re-deriving it (one definition, not two).
        self.fns = {}
        for n in self._walk(program):
            if type(n) is A.FunctionDef:
                self.fns[n.name] = n
        self.main = [s for s in program.statements if type(s) is not A.FunctionDef]
        types = binding_types if isinstance(binding_types, dict) else {}
        self.string_wires = {n for n, t in types.items() if _is_string_type(t)}
        self.vec_params = set()
        for n in self._walk(program):
            if type(n) is A.ParamDecl and n.type_hint in _VEC_PARAM_HINTS:
                self.vec_params.add(n.name)
        self.params_vary = {name: set() for name in self.fns}   # fed a varying argument
        self.ret_vary = {name: False for name in self.fns}      # varies with uniform arguments
        self.gathers = {name: False for name in self.fns}       # the body holds a gather
        # LANG-L3 (M4/M6): has ANY call site to this function been seen under a per-pixel
        # `if`? Over-approximated per FUNCTION, not per call site -- the same granularity
        # `ret_vary`/`gathers` already use for this class of question. Grows monotonically
        # (like them), so folding it into `_facts()` below is enough to fixed-point it.
        self.pp_called = {name: False for name in self.fns}
        self.all_vary = set()                                   # free names a body may inherit
        self.all_defined = set()
        # TRK-25: which nodes made the program REGION-DEPENDENT, recorded on every pass (the
        # facts only grow, and an id-keyed set makes the repetition idempotent) so the predicate
        # can read them without asking for diagnostics. `tex_roi.region_dependent` reads these.
        self.varying_loops = set()      # id(ForLoop/WhileLoop) — clauses (a) and (b)
        self.string_ifs = set()         # id(IfElse / TernaryOp) — clause (c)
        self.scalar_casts = set()       # id(CastExpr / FunctionCall) — TRK-32 clause (d)
        # LANG-L3: the flow-plan sites `flow_plan()` reports (docs/masked-control-flow.md
        # §8's L3 row), computed on this SAME walk rather than a second one -- a caller that
        # only wants `region_clauses()` or `run()`'s diagnostics simply leaves these unread.
        self.transfer_sites = set()       # id(BreakStmt/ContinueStmt/ReturnStmt) under a
                                          # per-pixel `if` (M1/M4)
        self.scatter_sites = set()        # id(Assignment) — a computed-coordinate write
                                          # (`@T[x,y] op= v`) under one (M5)
        self.probe_sites = set()          # id(FunctionCall) — `debug_print` under one (M7)
        self.binding_write_sites = set()  # id(Assignment) — a plain `@binding` write inside a
                                          # CALLED function, under one (M6)
        # TRK-154: id(FunctionCall) — a call to a user-defined function reached under a
        # per-pixel `if` (or inside a loop already `pp`), the same condition `pp_called`
        # already flags at function granularity below. `_mf_call_user_function`'s M4
        # empty-call skip (`if not m_any(self._live))`) syncs (a `.item()`) exactly here —
        # once the live mask can be a per-pixel TENSOR rather than always-True — and until
        # now no set on this walk named that class of site for a CUDA-graph capture
        # decision to see.
        self.call_sites = set()
        self.sync_points = set()          # id(ForLoop/WhileLoop) needing a per-pass live check
                                          # under masking: its own condition is per-pixel (==
                                          # varying_loops), or it directly encloses a transfer
                                          # gated by one (R-BREAK/R-CONT's shape)
        self.emit = False
        self.diags = {}
        self.budget = 200_000 + 200 * sum(1 for _ in self._walk(program))
        # TRK-25 clause (c): names that can hold a STRING. A per-pixel `if` that assigns one —
        # or a per-pixel `?:` that selects one — is resolved by a region-wide MAJORITY VOTE over
        # the pixels being cooked, so its value depends on how the cook was split. Name-based
        # and deliberately over-approximating (no scoping, no shadowing, no dataflow): a
        # declared `string`, a `s$param`, a STRING-typed wire, a string-typed function parameter
        # or return, or any name assigned a string-producing expression anywhere in the program.
        self.string_names = set(self.string_wires)
        self.string_fns = {n for n, fd in self.fns.items()
                           if (fd.return_type or "").lower() == "string"}
        for n in self._walk(program):
            cls = type(n)
            if cls is A.ParamDecl:
                if n.type_hint == "s":
                    self.string_names.add(n.name)
            elif cls is A.FunctionDef:
                self.string_names.update(p for t, p in n.params
                                         if (t or "").lower() == "string")
        # A FIXED POINT, because string-ness flows along names and returns: `s = @S; t = s;`.
        # Both sets only grow, so it terminates — in at most one round per link of the longest
        # chain — and `_tick` keeps a pathological chain inside the same work budget as the rest
        # of the analysis (over budget → `_CFBudget` → the predicate's fail-closed True).
        grew = True
        while grew:
            grew = False
            for n in self._walk(program):
                self._tick()
                cls = type(n)
                name = None
                if cls is A.VarDecl:
                    if (n.type_name or "").lower() == "string" \
                            or self._is_string_expr(n.initializer):
                        name = n.name
                elif cls is A.ArrayDecl:
                    if (n.element_type_name or "").lower() == "string":
                        name = n.name
                elif cls is A.Assignment and self._is_string_expr(n.value):
                    t = n.target
                    while type(t) is A.ChannelAccess:
                        t = t.object
                    if type(t) is A.ArrayIndexAccess:
                        t = t.array
                    if type(t) is A.Identifier or type(t) is A.BindingRef:
                        name = t.name
                if name is not None and name not in self.string_names:
                    self.string_names.add(name)
                    grew = True
            for fname, fd in self.fns.items():
                if fname in self.string_fns:
                    continue
                if any(type(s) is A.ReturnStmt and self._is_string_expr(s.value)
                       for s in self._walk(fd)):
                    self.string_fns.add(fname)
                    grew = True

    # ── LANG-L7: the W7007/W7008 masked-flow sunset ──────────────────────────
    def _effective_masked(self) -> bool:
        """True when this program is actually COOKED under `0.25`'s masked per-pixel control
        flow — `min(pragma, LANGUAGE_VERSION) >= MASKED_FLOW_SINCE` (the identical rule
        `tex_roi.region_dependent`'s clause (a)/(b) sunset already keys on; a second
        definition here would drift against it). When True, `break`/`continue`/`return`
        under a per-pixel `if` and a per-pixel loop bound no longer act on every pixel, so
        W7007 (all three sites) and W7008's loop half are FALSE and must not fire. W7008's
        string halves (a per-pixel `if`/`?:` choosing a string, or a per-pixel value cast
        straight to a string) never sunset — a string has no per-pixel representation at
        any language level — so they are not gated by this."""
        return self._tex_roi._language_tuple(
            self.program, self.source) >= self._tex_roi.MASKED_FLOW_SINCE

    # ── traversal helpers ────────────────────────────────────────────────────
    def _walk(self, node):
        stack = [node]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(self.A.iter_child_nodes(n))

    def _tick(self):
        self.budget -= 1
        if self.budget < 0:
            raise _CFBudget()

    def _is_string_expr(self, expr) -> bool:
        """Can this expression produce a STRING? Every construct that can carry one is
        enumerated: a literal; a `string(...)` cast; a name already known to hold a string (a
        declared `string`, an `s$param`, a STRING-typed wire, a string-typed function
        parameter, or a name the fixed point above has already learned); a stdlib call or a
        user function whose declared/derived return is STRING; and — recursively — a `?:`, a
        `+` concatenation, or an element of a string array.

        POSITIVE and syntactic. Clause (c) over-approximates by NAME (no scoping, no
        shadowing, no dataflow), which is the §1.5 whitelist posture pointed at the place it
        belongs: an unknown NAME is treated as a string the moment anything in the program can
        make it one. It deliberately does NOT read an unknown VALUE as a string — doing that
        would decline every per-pixel `?:` over floats, which is the commonest shape there is
        and one that splits perfectly."""
        if expr is None:
            return False
        A = self.A
        cls = type(expr)
        if cls is A.StringLiteral:
            return True
        if cls is A.Identifier:
            return expr.name in self.string_names
        if cls is A.BindingRef:
            return expr.type_hint == "s" or expr.name in self.string_names
        if cls is A.CastExpr:
            return (expr.target_type or "").lower() == "string"
        if cls is A.TernaryOp:
            return (self._is_string_expr(expr.true_expr)
                    or self._is_string_expr(expr.false_expr))
        if cls is A.BinOp:                     # `+` is the only string operator, but `==`/`!=`
            return (self._is_string_expr(expr.left)     # over strings is numeric — answering
                    or self._is_string_expr(expr.right))  # True there only over-approximates
        if cls is A.ArrayIndexAccess:
            return self._is_string_expr(expr.array)
        if cls is A.ChannelAccess:
            return False                       # a swizzle is numeric by construction
        if cls is A.FunctionCall:
            return (expr.name in self.string_fns if expr.name in self.fns
                    else expr.name in _string_ret_fns())
        return False

    def _scalar_cast_varies(self, node, st) -> bool:
        """TRK-32 clause (d): does `node` reduce a per-pixel value to a STRING through
        `stdlib._scalar_from_tensor`'s region MEAN — `string(x)`, `str(x)`, or a `format()`
        call whose template actually fills a `{}`/`{:spec}` placeholder with an argument?

        Four sites implement that same reduction — `interpreter.py`'s cast calls
        `stdlib._scalar_from_tensor` directly, as do `stdlib.fn_str` and `stdlib.fn_format`;
        `codegen.py`'s cast inlines the identical `.item()`/`.float().mean().item()` rather
        than calling the shared helper — and this predicate does not re-derive
        which VALUES are per-pixel — it reuses `_varies`, the same rule clauses (a)/(b) and
        the `?:` half of clause (c) already trust (over-approximate by NAME: a call whose
        registry footprint is not `'point'` is treated as non-uniform, never by guessing an
        unresolved VALUE).

        `format("%f", x)` is deliberately NOT in this class: `fn_format` only substitutes a
        Python-style `{}`/`{:spec}` placeholder (its own documented contract), so a literal
        template with none returns itself unchanged on every route and `x`'s reduced value
        never reaches the output — declining it would cost the corpus for nothing. A
        NON-literal template cannot be checked for a placeholder, so — the same whitelist
        posture as everywhere else in this class — it is assumed to carry one."""
        A = self.A
        cls = type(node)
        if cls is A.CastExpr:
            return node.target_type == "string" and self._varies(node.expr, st)
        if cls is not A.FunctionCall:
            return False
        if node.name == "str":
            return bool(node.args) and self._varies(node.args[0], st)
        if node.name == "format":
            if not node.args:
                return False
            template = node.args[0]
            has_placeholder = (type(template) is not A.StringLiteral
                               or _FORMAT_PLACEHOLDER_RE.search(template.value or "") is not None)
            return has_placeholder and any(self._varies(a, st) for a in node.args[1:])
        return False

    def _varies(self, expr, st) -> bool:
        A = self.A
        stack = [expr]
        while stack:
            n = stack.pop()
            cls = type(n)
            if cls is A.Identifier:
                if n.name in st.vary or (n.name in _PER_PIXEL_BUILTINS
                                         and n.name not in st.defined):
                    return True
                continue
            if cls is A.BindingRef:
                if n.kind == "param":
                    if n.name in self.vec_params or n.type_hint in _VEC_PARAM_HINTS:
                        return True
                elif n.type_hint != "s" and n.name not in self.string_wires:
                    return True
                continue
            if cls is A.BindingSampleAccess or cls is A.BindingIndexAccess:
                return True
            if cls is A.ChannelAccess and type(n.object) is A.BindingRef \
                    and n.object.kind == "param":
                return True                       # a component of a vector parameter
            if cls is A.FunctionCall:
                if n.name in self.fns:
                    if self.ret_vary[n.name]:
                        return True
                elif self.footmap.get(n.name, "point") != "point":
                    return True
            stack.extend(A.iter_child_nodes(n))
        return False

    def _has_gather(self, node) -> bool:
        A = self.A
        stack = [node]
        while stack:
            n = stack.pop()
            cls = type(n)
            if cls is A.BindingSampleAccess or cls is A.BindingIndexAccess:
                return True
            if cls is A.FunctionCall:
                if n.name in self.fns:
                    if self.gathers[n.name]:
                        return True
                elif self.footmap.get(n.name, "point") != "point":
                    return True
            elif cls is A.Assignment:             # a write target is not a read
                stack.append(n.value)
                t = n.target
                if type(t) is A.BindingIndexAccess:
                    stack.extend(t.args)
                elif type(t) is A.ArrayIndexAccess:
                    stack.append(t.index)
                continue
            elif cls is A.FunctionDef:
                continue
            stack.extend(A.iter_child_nodes(n))
        return False

    def _warn(self, code, node, message, hint):
        key = (code, id(node))
        if not self.emit or key in self.diags:
            return
        from .tex_compiler.diagnostics import make_diagnostic
        loc = node.loc
        width = _CF_KEYWORD_LEN.get(type(node).__name__)
        end_col = loc.col + width if (width and loc is not None and loc.col) else None
        self.diags[key] = make_diagnostic(code, message, loc, self.source, end_col=end_col,
                                          hint=hint, phase="type_checker", severity="warning")

    # ── the passes ───────────────────────────────────────────────────────────
    def run(self) -> list:
        changed = True
        while changed:                            # transitive "holds a gather"
            changed = False
            for name, fd in self.fns.items():
                if not self.gathers[name] and any(self._has_gather(s) for s in fd.body):
                    self.gathers[name] = changed = True
        while True:                               # facts only grow, so this terminates
            before = self._facts()
            self._pass()
            if self._facts() == before:
                break
        self.emit = True
        self._pass()
        return sorted(self.diags.values(), key=lambda d: (d.loc.line, d.loc.col, d.code))

    def region_clauses(self):
        """TRK-25/TRK-32: run the same fixed point WITHOUT emitting anything, and return
        `(varying_loops, string_ifs, scalar_casts)` — the id sets behind clauses (a)/(b),
        clause (c) and clause (d) of `tex_roi.region_dependent`. No diagnostic is built, and
        the gather fixed point (which only W7006 reads) is skipped. Raises `_CFBudget` past
        the work budget, which the predicate turns into 'region-dependent' — it is a GATE,
        so it fails closed."""
        while True:
            before = self._facts()
            self._pass()
            if self._facts() == before:
                break
        return self.varying_loops, self.string_ifs, self.scalar_casts

    def _facts(self):
        return (tuple(sorted((k, tuple(sorted(v))) for k, v in self.params_vary.items())),
                tuple(sorted(self.ret_vary.items())), len(self.all_vary), len(self.all_defined),
                tuple(sorted(self.pp_called.items())))

    def _pass(self):
        self._in_function = False
        self._block(self.main, _CFState(), _CFScope(True), None, False, False, False, None)
        for name, fd in self.fns.items():
            params = {p for _t, p in fd.params}
            inherited = self.all_vary - params
            defined = self.all_defined | params
            # M4/M6: a call inherits the caller's live mask, over-approximated per function
            # (as True the moment ANY call site has been seen under a per-pixel `if`) rather
            # than per call site — the same whole-function granularity `ret_vary` already
            # uses for "what does this function do".
            self._in_function = True
            called_pp = self.pp_called[name]
            # What the result does with uniform arguments (per call site, a varying argument
            # is added on top) — no emission, no call-site recording.
            emit, self.emit = self.emit, False
            scope = _CFScope(False)
            self._block(fd.body, _CFState(inherited, defined), scope, None, False, False,
                        called_pp, None)
            self.emit = emit
            if scope.returns_vary:
                self.ret_vary[name] = True
            # The body as its call sites actually feed it: this is the pass that warns.
            self._block(fd.body, _CFState(inherited | self.params_vary[name], defined),
                        _CFScope(True), None, False, False, called_pp, None)
        self._in_function = False

    def _block(self, stmts, st, scope, loop, pp_loop, pp_fn, pp=False, loop_node=None):
        for s in stmts:
            st = self._stmt(s, st, scope, loop, pp_loop, pp_fn, pp, loop_node)
        return st

    def _stmt(self, s, st, scope, loop, pp_loop, pp_fn, pp=False, loop_node=None):
        A = self.A
        self._tick()
        cls = type(s)
        if cls is A.VarDecl or cls is A.ArrayDecl:
            init = s.initializer
            v = init is not None and self._expr(init, st, scope, pp)
            self._set(st, s.name, v, strong=True)
        elif cls is A.Assignment:
            v = self._expr(s.value, st, scope, pp)
            t = s.target
            if type(t) is A.Identifier:
                self._set(st, t.name, v, strong=s.op is None)
            elif type(t) is A.ChannelAccess and type(t.object) is A.Identifier:
                self._set(st, t.object.name, v, strong=False)
            elif type(t) is A.ArrayIndexAccess and type(t.array) is A.Identifier:
                v = self._expr(t.index, st, scope, pp) or v
                self._set(st, t.array.name, v, strong=False)
            elif type(t) is A.BindingIndexAccess:
                for a in t.args:
                    self._expr(a, st, scope, pp)
            # LANG-L3 M5/M6: a write to an `@binding`, under a per-pixel condition.
            # `_scatter_target_base` (reused from `tex_roi`, not re-derived) finds a
            # COMPUTED-COORDINATE target (`@T[x,y]=`/`@T(u,v)=`, however wrapped in a
            # channel/array-index suffix) — that is M5's scatter, gated by SOURCE regardless
            # of whether it sits in a function. Anything else that bottoms out at a plain
            # `@binding` (a bare BindingRef, through any ChannelAccess wrapper — `@OUT.r=`)
            # is M6's plain binding write, which only needs recording here INSIDE a called
            # function — a top-level plain `@` write under a per-pixel `if` is already
            # handled by the engine's ordinary per-pixel-if write masking (M1) once that
            # lands, so M6 is specifically the merge gap `collect_assigned_vars` leaves by
            # not descending into calls.
            if pp:
                scatter_base = self._tex_roi._scatter_target_base(t)
                if scatter_base is not None:
                    self.scatter_sites.add(id(s))
                elif self._in_function:
                    tt = t
                    while type(tt) is A.ChannelAccess:
                        tt = tt.object
                    if type(tt) is A.BindingRef and tt.kind == "wire":
                        self.binding_write_sites.add(id(s))
        elif cls is A.ExprStatement:
            self._expr(s.expr, st, scope, pp)
        elif cls is A.IfElse:
            return self._if(s, st, scope, loop, pp_loop, pp_fn, pp, loop_node)
        elif cls is A.ForLoop or cls is A.WhileLoop:
            return self._loop(s, st, scope, loop, pp_fn, pp)
        elif cls is A.BreakStmt or cls is A.ContinueStmt:
            if loop is not None:
                (loop.breaks if cls is A.BreakStmt else loop.continues).append(st.copy())
            if pp_loop:
                kw = "break" if cls is A.BreakStmt else "continue"
                what = ("ends the loop" if cls is A.BreakStmt
                        else "skips the rest of the pass")
                # LANG-L7: under `0.25` masked flow this `{kw}` no longer acts on every
                # pixel — it clears only the pixels live at this branch — so the warning
                # would be false and must not fire (`_effective_masked`).
                if not self._effective_masked():
                    self._warn("W7007", s,
                               f"This `{kw}` sits under an `if` whose condition can differ from pixel "
                               f"to pixel. Such an `if` runs its branches on every pixel, so the "
                               f"`{kw}` {what} for ALL pixels the first time the loop reaches it, "
                               f"whatever the condition says, and the assignments before it in "
                               f"that branch land on every pixel too, unless this program "
                               f"declares `//!tex 0.25`.",
                               "Keep a per-pixel flag the loop body tests instead, e.g. "
                               "`if (found < 0 && hit) { found = i; }`, and let the loop run a "
                               "bound that is the same for every pixel, or declare "
                               "`//!tex 0.25` (LANGUAGE.md §7.1).")
                # LANG-L3 (M1/M3): the site itself, plus the loop it clears bits IN — a
                # break/continue gated by a per-pixel `if` makes THAT loop's own live mask
                # able to narrow mid-loop even when the loop's bound is uniform (R-BREAK's
                # and R-CONT's shape), so it needs the same per-pass live check as a loop
                # whose bound is itself per-pixel (`varying_loops`).
                self.transfer_sites.add(id(s))
                if loop_node is not None:
                    self.sync_points.add(id(loop_node))
        elif cls is A.ReturnStmt:
            if s.value is not None and self._expr(s.value, st, scope, pp):
                scope.returns_vary = True
            if pp_fn:
                # LANG-L7: under `0.25` masked flow this `return` only records and clears
                # the pixels live at this branch, so the warning would be false below
                # `MASKED_FLOW_SINCE` only (`_effective_masked`).
                if not self._effective_masked():
                    self._warn("W7007", s,
                               "This `return` sits under an `if` whose condition can differ from "
                               "pixel to pixel. Such an `if` runs its branches on every pixel, so the "
                               "function returns this value for ALL pixels the first time it reaches "
                               "the `return`, whatever the condition says, unless this program "
                               "declares `//!tex 0.25`.",
                               "Assign the result to a local inside the `if` and return it once at "
                               "the end, or select with `cond ? a : b`, or declare `//!tex 0.25` "
                               "(LANGUAGE.md §7.1).")
                self.transfer_sites.add(id(s))
        elif cls is A.ParamDecl and s.default_expr is not None:
            self._expr(s.default_expr, st, scope, pp)
        return st

    def _set(self, st, name, varies, strong):
        if varies:
            st.vary.add(name)
            self.all_vary.add(name)
        elif strong:
            st.vary.discard(name)
        st.defined.add(name)
        self.all_defined.add(name)

    def _expr(self, expr, st, scope, pp=False) -> bool:
        """Scan an expression — W7006 on a per-pixel `?:` holding a gather, and the call-site
        facts for user functions — and return whether its value can differ per pixel."""
        A = self.A
        stack = [expr]
        while stack:
            n = stack.pop()
            self._tick()
            cls = type(n)
            if pp and cls is A.FunctionCall:
                # LANG-L3 M7/M4: independent of the elif chain below (which decides whether
                # THIS call feeds `params_vary` / falls to `_scalar_cast_varies`) — a call can
                # be a probe, feed a user function's pp_called, AND be one of those, all at
                # once, and none of them should suppress another.
                if n.name == "debug_print":
                    self.probe_sites.add(id(n))
                if n.name in self.fns:
                    self.pp_called[n.name] = True
                    self.call_sites.add(id(n))    # TRK-154: the call SITE, not just the callee
            if cls is A.TernaryOp:
                if self.emit and self._varies(n.condition, st) and (
                        self._has_gather(n.true_expr) or self._has_gather(n.false_expr)):
                    self._warn("W7006", n,
                               "Both operands of this `?:` are evaluated on every pixel, because "
                               "its condition can differ from pixel to pixel, so the gather "
                               "inside (a sample, fetch, blur or reduction) costs the same "
                               "whichever operand a pixel keeps.",
                               "Nothing is skipped per pixel. To skip work for the whole frame, "
                               "test a value that is the same for every pixel (LANGUAGE.md §7.1).")
                # TRK-25 clause (c), the OTHER spelling of the same merge. `if` resolves a
                # string through `_merge_branch_vars`; a `?:` runs its own region-wide vote on
                # the condition (`Interpreter._eval_ternary`, and the same arithmetic in the
                # codegen tier), so it diverges whole-frame vs tiled in exactly the same way.
                # Recorded wherever the `?:` sits, not only where it reaches a name: a string
                # `?:` handed straight to a call is the same vote. The string test is first
                # because it is a shallow syntactic walk and `_varies` is not.
                if (self._is_string_expr(n.true_expr) or self._is_string_expr(n.false_expr)) \
                        and self._varies(n.condition, st):
                    self.string_ifs.add(id(n))
                    self._warn("W7008", n,
                               "A string chosen by a `?:` whose condition can differ from "
                               "pixel to pixel is resolved by a majority vote over the pixels "
                               "of the region being cooked, so which operand wins depends on "
                               "how the cook was split. This program is therefore cooked as "
                               "one whole region.",
                               "Choose the string from a value that is the same for every "
                               "pixel (a parameter, a literal, `iw`/`ih`), or carry the "
                               "per-pixel decision in a number instead (LANGUAGE.md §7.1).")
            elif cls is A.FunctionCall and scope.record_calls and n.name in self.fns:
                fed = self.params_vary[n.name]
                for (_ptype, pname), arg in zip(self.fns[n.name].params, n.args):
                    if pname not in fed and self._varies(arg, st):
                        fed.add(pname)
            elif self._scalar_cast_varies(n, st):
                # TRK-32 clause (d): `string(x)` / `str(x)` / `format("{}", x)` on a
                # per-pixel value has no conditioned MERGE (that is clause (c)) — the
                # cast itself reduces the tensor to one number by taking a MEAN over the
                # cooked region (`stdlib._scalar_from_tensor`, and its interpreter/codegen
                # cast twins), and a strip's mean differs from the whole frame's.
                self.scalar_casts.add(id(n))
                self._warn("W7008", n,
                           "This converts a value that can differ from pixel to pixel "
                           "directly to a STRING. A string has no per-pixel representation, "
                           "so the engine reduces it by taking the MEAN of every pixel in "
                           "the region being cooked, and a strip's mean differs from the "
                           "whole frame's — so this string depends on how the cook was "
                           "split. This program is therefore cooked as one whole region.",
                           "Reduce the value to one number first (an average/min/max over "
                           "the whole image, e.g. `img_mean`), or index one pixel to make "
                           "the choice explicit, before converting it to a string "
                           "(LANGUAGE.md §7.1).")
            stack.extend(A.iter_child_nodes(n))
        return self._varies(expr, st)

    def _if(self, s, st, scope, loop, pp_loop, pp_fn, pp=False, loop_node=None):
        from .tex_compiler.ast_nodes import collect_assigned_vars
        per_pixel = self._expr(s.condition, st, scope, pp)
        if per_pixel and self.emit and (any(self._has_gather(x) for x in s.then_body)
                                        or any(self._has_gather(x) for x in s.else_body)):
            self._warn("W7006", s,
                       "Both branches of this `if` run on every pixel, because its condition "
                       "can differ from pixel to pixel, so the gather inside (a sample, fetch, "
                       "blur or reduction) costs the same whether or not a pixel takes that "
                       "branch.",
                       "Nothing is skipped per pixel. To skip work for the whole frame, test a "
                       "value that is the same for every pixel: a parameter, a literal, "
                       "`iw`/`ih` or a loop counter (LANGUAGE.md §7.1).")
        inner_loop, inner_fn = pp_loop or per_pixel, pp_fn or per_pixel
        # LANG-L3 (M2): unlike `pp_loop` (loop-scoped — reset by `_loop` for its OWN body),
        # `pp` never resets on the way down: an `@`/scatter/probe under a per-pixel `if`
        # stays under it however many loops or nested `if`s sit between them.
        inner_pp = pp or per_pixel
        then_st = self._block(s.then_body, st.copy(), scope, loop, inner_loop, inner_fn,
                              inner_pp, loop_node)
        else_st = (self._block(s.else_body, st.copy(), scope, loop, inner_loop, inner_fn,
                               inner_pp, loop_node)
                   if s.else_body else st.copy())
        out = then_st.copy()
        out.join(else_st)
        if per_pixel:
            # The engine's merge: a name either branch assigns or declares is torch.where-d
            # to the condition's shape when it held a value before the `if` or both branches
            # define it; otherwise the one branch's value is kept as it is.
            then_env, then_bind = collect_assigned_vars(s.then_body)
            else_env, else_bind = collect_assigned_vars(s.else_body)
            names = then_env | else_env
            for name in names:
                if name in st.defined or (name in then_st.defined and name in else_st.defined):
                    self._set(out, name, True, strong=False)
            # TRK-25 clause (c). A STRING has no per-pixel representation, so the merge resolves
            # it by a majority vote over the pixels of the region being cooked — a different
            # region can hold a different majority. `@bindings` are included alongside locals
            # because a called function CAN write a caller-visible binding under this branch.
            if self.string_names & (names | then_bind | else_bind):
                self.string_ifs.add(id(s))
                self._warn("W7008", s,
                           "A string assigned inside an `if` whose condition can differ from "
                           "pixel to pixel is resolved by a majority vote over the pixels of "
                           "the region being cooked, so its value depends on how the cook was "
                           "split. This program is therefore cooked as one whole region.",
                           "Decide the string from a value that is the same for every pixel (a "
                           "parameter, a literal, `iw`/`ih`), or carry the per-pixel decision "
                           "in a number instead (LANGUAGE.md §7.1).")
        return out

    def _loop(self, s, st, scope, outer_loop, pp_fn, pp=False):
        A = self.A
        is_for = type(s) is A.ForLoop
        if is_for and s.init is not None:
            st = self._stmt(s.init, st, scope, outer_loop, False, pp_fn, pp, None)
        head = st
        while True:                               # the head state, to a fixed point
            frame = _CFLoop()
            if s.condition is not None:
                self._expr(s.condition, head, scope, pp)
            # LANG-L3: `s` itself is THIS loop's `loop_node` for everything inside its own
            # body — a break/continue in here narrows THIS loop's live mask (sync_points),
            # never an outer one (that is exactly why `pp_loop` also resets to False here,
            # unchanged from before this lane).
            body = self._block(s.body, head.copy(), scope, frame, False, pp_fn, pp, s)
            for c in frame.continues:
                body.join(c)
            if is_for and s.update is not None:
                body = self._stmt(s.update, body, scope, frame, False, pp_fn, pp, s)
            nxt = head.copy()
            nxt.join(body)
            if nxt.same(head):
                break
            head = nxt
        if s.condition is not None and self._varies(s.condition, head):
            kw = "for" if is_for else "while"
            # LANG-L7: `varying_loops` is the structural FACT `tex_roi.region_dependent`'s
            # own sunset test reads (clauses a/b) — it is recorded unconditionally, at every
            # language level, regardless of whether the WARNING below fires. Only the two
            # diagnostics are gated on `_effective_masked`: under `0.25` this loop no longer
            # runs every pixel to the frame's maximum (each pixel's own live mask exits the
            # loop on its own pass), so W7007 is false; and the pass count is each pixel's
            # own rather than the region's, so W7008's loop half is also false — but a
            # per-pixel STRING choice (clauses c/d, gated elsewhere) never sunsets, which is
            # why only the loop-bound W7008 site is gated here.
            self.varying_loops.add(id(s))
            if not self._effective_masked():
                self._warn("W7007", s,
                           f"This `{kw}` loop's condition can differ from pixel to pixel. The loop "
                           f"keeps running while ANY pixel's condition holds and its body is not "
                           f"masked, so every pixel runs as many passes as the pixel that needs the "
                           f"most, including pixels whose own condition is already false, unless "
                           f"this program declares `//!tex 0.25`.",
                           "Bound the loop by a value that is the same for every pixel and guard or "
                           "weight the per-pixel work, e.g. `for (int i = 0; i < $max; i++) "
                           "{ if (i < n) { ... } }`, or declare `//!tex 0.25` (LANGUAGE.md §7.1).")
                # TRK-25 clauses (a)/(b): the pass count is the region's MAXIMUM, so the output
                # depends on which region was cooked. This is the half of W7007 the engine
                # acts on, and it sunsets in lockstep with W7007 above (both are false once
                # the loop is masked, since a masked loop's pass count is per-pixel, not the
                # region's maximum).
                self._warn("W7008", s,
                           "This loop's bound can differ from pixel to pixel, so the number of "
                           "passes depends on which region is cooked. The engine therefore cooks "
                           "this program as one whole region: windows, strips and batch strips are "
                           "declined, and under memory pressure the cook can run out of memory "
                           "where a split would have fitted.",
                           "Bound the loop by a value that is the same for every pixel and guard or "
                           "weight the per-pixel work, e.g. `for (int i = 0; i < $max; i++) "
                           "{ if (i < n) { ... } }`. A uniformly bounded loop splits again "
                           "(LANGUAGE.md §7.1).")
        out = head.copy()
        for b in frame.breaks:
            out.join(b)
        return out


# ── LANG-L3: the shared structural flow-plan walk ─────────────────────────────
#
# `docs/masked-control-flow.md` §8's L3 row: "the shared structural walk that flags
# per-pixel loops, transfer-bearing regions, scatter/probe/binding-write sites under a
# per-pixel `if`, and the sync points." Later stages (L4's interpreter masking, L5's
# codegen mirror, L6's graph-capture/ROI consumers) all consult ONE `FlowPlan` rather than
# each re-deriving "is this per-pixel" — the same discipline `tex_roi.region_dependent`
# already documents for W7007/W7008 ("a second definition of 'per-pixel' would drift
# against it"), extended to a second question asked of the identical walk.
#
# UNCONDITIONAL by design — NOT gated on `Program.language` or `LANGUAGE_VERSION`. The
# sites named are a structural fact about the program; a caller combines this plan with
# the language gate (`Program.language`, and — once `LANGUAGE_VERSION` reaches `0.25` —
# `tex_roi._language_tuple`) to decide whether to actually mask. Gating the WALK itself on
# the pragma would make `flow_plan` permanently empty for every program until `LANGUAGE_
# VERSION` bumps at L7 — including the very repro programs L4/L5 need it to name sites for
# while they are still being built, before that bump exists.
@dataclass(frozen=True)
class FlowPlan:
    """Every site a masking implementation (L4/L5) or a masking-aware consumer (L6) needs
    to know about, computed ONCE per program by `_ControlFlowLint`'s existing per-pixel
    walk. Nothing here masks anything — the plan only NAMES sites; L4/L5 decide what to do
    with them.

    Every field except `complete` is a frozenset of `id()` of an AST node — the same
    identity-keyed spelling `region_dependent`'s `varying_loops`/`string_ifs`/`scalar_casts`
    already use, so a caller holding the Program can look a site up by walking it once
    (`ast_nodes.iter_child_nodes`) and testing `id(node) in plan.<field>`. Because the ids
    are the program's OWN node identities, a plan is only meaningful against the EXACT
    `Program` instance `flow_plan()` was called with — never against a re-parse of the
    same source, whose nodes get fresh ids.

      * `per_pixel_loops`   — a `for`/`while` whose condition can differ per pixel (M3;
                              IDENTICAL to `region_dependent`'s clauses (a)/(b) — the same
                              set, not a second definition. R-BOUND / R-WBOUND).
      * `transfer_sites`    — a `break`/`continue`/`return` under a per-pixel `if` (M1/M4;
                              R-BREAK / R-CONT / R-RET).
      * `scatter_sites`     — a computed-coordinate write (`@T[x,y] op= v` / `@T(u,v) op= v`,
                              however wrapped in a channel/array-index suffix) under a
                              per-pixel `if` (M5), gated by SOURCE.
      * `probe_sites`       — a `debug_print(...)` call under a per-pixel `if` (M7).
      * `binding_write_sites` — a plain `@binding = v` write (not a scatter) inside a
                              user-defined function, under a per-pixel `if` — either
                              directly in that function's own body, or because the function
                              is called from one anywhere in the program (M6; the latter is
                              over-approximated per FUNCTION, not per call site — the same
                              granularity `ret_vary`/`gathers` already use for this class).
      * `sync_points`       — a `for`/`while` that needs a per-pass live-mask check under
                              masking: its own condition is per-pixel (⊇ `per_pixel_loops`),
                              or it directly encloses a `break`/`continue` gated by a
                              per-pixel `if` (so the loop's OWN live mask can narrow
                              mid-loop even though its bound is uniform — R-BREAK/R-CONT's
                              shape).
      * `call_sites`        — a call to a user-defined function reached under a per-pixel
                              `if` (or from inside a loop already `pp`; TRK-154). M4's
                              empty-call skip (`if not m_any(self._live)`) syncs (a
                              `.item()`) exactly here, once the live mask reaching the call
                              can be a per-pixel TENSOR rather than always-True — the class
                              of sync `sync_points`/`scatter_sites` did not name, so a
                              CUDA-graph capture decision consulting only those two could
                              not see it (it still failed capture loudly and blacklisted
                              the key, the pre-existing net — never served silently wrong).
      * `complete`          — False when the walk could not finish (the work budget was
                              exceeded, or it raised) — the FAIL-CLOSED half of "over-
                              approximate by name, not by value": a caller that sees
                              `complete=False` cannot say WHERE the sites are, so it must
                              treat the program as needing masking everywhere, never as
                              needing none. `is_empty()` enforces this — it is never True
                              on an incomplete walk, even when every set above is empty."""
    per_pixel_loops: frozenset = frozenset()
    transfer_sites: frozenset = frozenset()
    scatter_sites: frozenset = frozenset()
    probe_sites: frozenset = frozenset()
    binding_write_sites: frozenset = frozenset()
    sync_points: frozenset = frozenset()
    call_sites: frozenset = frozenset()
    complete: bool = True

    def is_empty(self) -> bool:
        """No masking-relevant site anywhere, AND the walk that says so finished normally.
        A non-empty plan on a program that does not need masking would make a later stage
        decline or mask work that is fine today (docs/masked-control-flow.md §8) — the
        reason every set above is a NAME-level over-approximation, never a guess."""
        return self.complete and not (self.per_pixel_loops or self.transfer_sites
                                       or self.scatter_sites or self.probe_sites
                                       or self.binding_write_sites or self.sync_points
                                       or self.call_sites)


_INCOMPLETE_FLOW_PLAN = FlowPlan(complete=False)


def flow_plan(program, binding_types: dict | None = None) -> FlowPlan:
    """LANG-L3: the structural per-pixel-control-flow sites `FlowPlan` describes, for
    `program`. Pure and total — an exception or a blown work budget answers the
    INCOMPLETE plan (`complete=False`), never the empty one; see `FlowPlan.complete`.

    `binding_types` is the cook's `{name: TEXType}` map, exactly as `region_dependent`
    takes it — optional, and a STRING-typed `@` wire is the one thing that changes a
    verdict here (a string can never be "per-pixel" in the numeric sense this walk tests).

    One walk, reused rather than duplicated: this constructs the SAME `_ControlFlowLint`
    `region_dependent` already trusts for W7007/W7008 and runs its identical fixed-point
    loop, so `per_pixel_loops` here is `region_dependent`'s `varying_loops` by
    construction, not a second computation that could drift from it."""
    try:
        lint = _ControlFlowLint(
            program, "", binding_types if isinstance(binding_types, dict) else {})
        while True:
            before = lint._facts()
            lint._pass()
            if lint._facts() == before:
                break
    except Exception:
        return _INCOMPLETE_FLOW_PLAN
    return FlowPlan(
        per_pixel_loops=frozenset(lint.varying_loops),
        transfer_sites=frozenset(lint.transfer_sites),
        scatter_sites=frozenset(lint.scatter_sites),
        probe_sites=frozenset(lint.probe_sites),
        binding_write_sites=frozenset(lint.binding_write_sites),
        sync_points=frozenset(lint.sync_points | lint.varying_loops),
        call_sites=frozenset(lint.call_sites),
    )


def _is_string_type(t) -> bool:
    name = getattr(t, "name", None) or getattr(t, "value", None) or t
    return isinstance(name, str) and name.upper() == "STRING"


def prewarm(programs, shapes=None, *, device: str = "cuda", precision: str = "fp32",
            compile_mode: str = "auto", cancel: "CancelToken | None" = None) -> dict:
    """CACHE-3: warm the compile/codegen tiers for a set of programs so the first scrub after
    a project load / relaunch replays instead of trialling ("first scrub doesn't jank").

    `programs` is an iterable of `(source, binding_types)` — the same pair `compile()` takes.
    `shapes` is an optional list of `(B,H,W)` the host expects to cook at; it is accepted for
    forward compatibility (per-resolution timing warm) — the warming below is shape-independent
    (codegen emission, backend compile, and the static capturability verdict don't depend on
    resolution). Per program it: materializes + persists the codegen fn (writes the `.cg`
    sidecar), submits a background `torch.compile` (LAT-1a, non-blocking), and seeds the
    graph-capturability verdict. CUDA graphs cannot be pre-captured (capture must be on the hot
    path), so their *verdict* is persisted via warm_state and the graph re-captures off the hot
    path on first cook. Loads and re-persists `warm_state.json` around the run. Best-effort per
    program (a bad program is skipped, never fatal). Returns a summary of what was warmed.

    `cancel` (v0.42 HOSTAUDIT-2): an optional `CancelToken`, checked once PER PROGRAM — this
    loop had no yield point at all, so a project-load prewarm over many programs used to be one
    un-interruptible span from the first program to the last. Unlike a cook's yield points,
    a cancelled prewarm never raises: every program already warmed keeps its (already-persisted)
    verdict, warming is pure best-effort optimization by contract, and the caller gets its usual
    summary dict back with `summary["cancelled"]` set to how many programs were skipped — never
    `CookCancelled`, which would break every existing caller that does not expect one."""
    import torch
    from .tex_cache import get_cache
    from .tex_runtime import compiled, graphed, warm_state
    from .tex_runtime.host import _cancel_check, CookCancelled as _CookCancelled
    warm_state.ensure_loaded()
    dev_type = "cuda" if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    summary = {"programs": 0, "codegen": 0, "bg_compile": 0, "capturable": 0, "errors": 0,
               "cancelled": 0}
    programs = list(programs)
    for i, (source, binding_types) in enumerate(programs):
        try:
            _cancel_check(cancel)   # HOSTAUDIT-2 yield: abort a stale prewarm between programs
        except _CookCancelled:
            summary["cancelled"] = len(programs) - i
            break
        try:
            # TRK-73: compute the fingerprint ONCE and hand it into `_compile_impl`, which
            # forwards it to `compile_tex` instead of that call recomputing its own — `compile()`
            # itself still computes exactly one per call (via `_compile_impl(fp=None)`), so this
            # moves the second of `prewarm`'s two calls, not just adds a third.
            fp = get_cache().fingerprint(source, binding_types)
            prog, fp = _compile_impl(source, binding_types, fp=fp)
            summary["programs"] += 1
            try:                                   # 1. codegen fn → persisted .cg sidecar
                if compiled._get_or_make_codegen_fn(prog.ast, prog.type_map, fp) is not None:
                    summary["codegen"] += 1
            except Exception:
                pass
            if dev_type == "cuda" and compile_mode in ("auto", "torch_compile"):
                try:                               # 2. background torch.compile (LAT-1a)
                    # Gate on the SAME preconditions the interactive auto path enforces on this
                    # call (compiled.py's auto tier): only submit with comfortable VRAM headroom
                    # and never during a CUDA-graph capture. A mid-session prewarm must not queue
                    # compiles that starve a live cook or collide with an in-flight capture.
                    if compiled._cuda_headroom_ok(device) and not compiled._capture_in_flight():
                        ck = (fp, dev_type, precision)
                        if compiled._submit_bg_compile(ck, prog.ast, prog.type_map, dev_type,
                                                       prog.used_builtins, precision, fp):
                            summary["bg_compile"] += 1
                except Exception:
                    pass
            if dev_type == "cuda":
                try:                               # 3. seed the graph-capturability verdict
                    graphed._capturable_memo[fp] = graphed._capturable(prog.ast)
                    summary["capturable"] += 1
                except Exception:
                    pass
        except Exception:
            summary["errors"] += 1
            continue
    warm_state.persist(force=True)
    return summary
