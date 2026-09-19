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
LANGUAGE_VERSION = "0.23"

# A `//!tex X.Y` pragma on its own comment line (the lexer discards comments, so this is
# recovered from the raw source, not from tokens).
_PRAGMA_RE = _re.compile(r"//!tex\s+(\d+)\.(\d+)\b")


def language_pragma(source: str):
    """Return the language version a program targets via a LEADING `//!tex X.Y` pragma (as
    the string 'X.Y'), or None. Only a pragma in the header run of blank / `//` line-comment
    lines is recognized — one buried after real code or inside a `/* … */` block comment is
    ignored (it would otherwise raise a spurious W7004)."""
    for raw in (source or "").splitlines():
        line = raw.strip()
        if not line:
            continue                      # blank line — keep scanning the header
        m = _PRAGMA_RE.match(line)
        if m:
            return f"{m.group(1)}.{m.group(2)}"
        if line.startswith("//"):
            continue                      # an ordinary leading line comment — keep scanning
        break                             # first real code (or a block comment): no pragma
    return None


def _ver_tuple(v):
    try:
        parts = str(v).split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except (ValueError, IndexError, AttributeError):
        return (0, 0)


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
    from . import tex_engine
    ast, type_map, referenced, assigned, params, used_builtins = \
        tex_engine._compile_or_raise(source, binding_types)
    return Program(ast, type_map, referenced, assigned, params, used_builtins, source)


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
                    f"engine's {LANGUAGE_VERSION}; newer features may not compile.",
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
      * **W7008** — control flow whose result depends on WHICH REGION is cooked, so the engine
        declines to split the cook (`tex_roi.region_dependent`): a `for` / `while` whose
        condition can differ per pixel (the loop runs to the region's maximum), or a string
        chosen per pixel by an `if` or a `?:` (either is a region-wide majority vote; a string
        arriving on a wire needs `binding_types` to be seen at all). Strictly the
        subset of W7007 the engine ACTS on — a `break` / `continue` / `return` under a
        per-pixel `if` draws W7007 and no W7008, because it fires on first arrival and so does
        the same thing in every region.

    Never emitted by `check()`: a host calls this beside it. Pure AST analysis — no compile,
    no cook, no side effects — and total: a program that does not parse, or that the
    analysis cannot finish within its work budget, returns []."""
    try:
        # DATA-6: through the one front end, so a swizzled wire is seen by its BASE name —
        # the name `binding_types` (and `string_wires` below) key on.
        from .tex_cache import parse_and_split
        program = parse_and_split(source, binding_types)
    except Exception:
        return []
    try:
        return _ControlFlowLint(program, source, binding_types).run()
    except Exception:  # the contract is absolute, as for check(): never raise
        return []


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
        self.footmap = tex_roi._footmap()
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
        self.all_vary = set()                                   # free names a body may inherit
        self.all_defined = set()
        # TRK-25: which nodes made the program REGION-DEPENDENT, recorded on every pass (the
        # facts only grow, and an id-keyed set makes the repetition idempotent) so the predicate
        # can read them without asking for diagnostics. `tex_roi.region_dependent` reads these.
        self.varying_loops = set()      # id(ForLoop/WhileLoop) — clauses (a) and (b)
        self.string_ifs = set()         # id(IfElse / TernaryOp) — clause (c)
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
        """TRK-25: run the same fixed point WITHOUT emitting anything, and return
        `(varying_loops, string_ifs)` — the id sets behind clauses (a)/(b) and clause (c) of
        `tex_roi.region_dependent`. No diagnostic is built, and the gather fixed point (which
        only W7006 reads) is skipped. Raises `_CFBudget` past the work budget, which the
        predicate turns into 'region-dependent' — it is a GATE, so it fails closed."""
        while True:
            before = self._facts()
            self._pass()
            if self._facts() == before:
                break
        return self.varying_loops, self.string_ifs

    def _facts(self):
        return (tuple(sorted((k, tuple(sorted(v))) for k, v in self.params_vary.items())),
                tuple(sorted(self.ret_vary.items())), len(self.all_vary), len(self.all_defined))

    def _pass(self):
        self._block(self.main, _CFState(), _CFScope(True), None, False, False)
        for name, fd in self.fns.items():
            params = {p for _t, p in fd.params}
            inherited = self.all_vary - params
            defined = self.all_defined | params
            # What the result does with uniform arguments (per call site, a varying argument
            # is added on top) — no emission, no call-site recording.
            emit, self.emit = self.emit, False
            scope = _CFScope(False)
            self._block(fd.body, _CFState(inherited, defined), scope, None, False, False)
            self.emit = emit
            if scope.returns_vary:
                self.ret_vary[name] = True
            # The body as its call sites actually feed it: this is the pass that warns.
            self._block(fd.body, _CFState(inherited | self.params_vary[name], defined),
                        _CFScope(True), None, False, False)

    def _block(self, stmts, st, scope, loop, pp_loop, pp_fn):
        for s in stmts:
            st = self._stmt(s, st, scope, loop, pp_loop, pp_fn)
        return st

    def _stmt(self, s, st, scope, loop, pp_loop, pp_fn):
        A = self.A
        self._tick()
        cls = type(s)
        if cls is A.VarDecl or cls is A.ArrayDecl:
            init = s.initializer
            v = init is not None and self._expr(init, st, scope)
            self._set(st, s.name, v, strong=True)
        elif cls is A.Assignment:
            v = self._expr(s.value, st, scope)
            t = s.target
            if type(t) is A.Identifier:
                self._set(st, t.name, v, strong=s.op is None)
            elif type(t) is A.ChannelAccess and type(t.object) is A.Identifier:
                self._set(st, t.object.name, v, strong=False)
            elif type(t) is A.ArrayIndexAccess and type(t.array) is A.Identifier:
                v = self._expr(t.index, st, scope) or v
                self._set(st, t.array.name, v, strong=False)
            elif type(t) is A.BindingIndexAccess:
                for a in t.args:
                    self._expr(a, st, scope)
        elif cls is A.ExprStatement:
            self._expr(s.expr, st, scope)
        elif cls is A.IfElse:
            return self._if(s, st, scope, loop, pp_loop, pp_fn)
        elif cls is A.ForLoop or cls is A.WhileLoop:
            return self._loop(s, st, scope, loop, pp_fn)
        elif cls is A.BreakStmt or cls is A.ContinueStmt:
            if loop is not None:
                (loop.breaks if cls is A.BreakStmt else loop.continues).append(st.copy())
            if pp_loop:
                kw = "break" if cls is A.BreakStmt else "continue"
                what = ("ends the loop" if cls is A.BreakStmt
                        else "skips the rest of the pass")
                self._warn("W7007", s,
                           f"This `{kw}` sits under an `if` whose condition can differ from pixel "
                           f"to pixel. Such an `if` runs its branches on every pixel, so the "
                           f"`{kw}` {what} for ALL pixels the first time the loop reaches it, "
                           f"whatever the condition says, and the assignments before it in "
                           f"that branch land on every pixel too.",
                           "Keep a per-pixel flag the loop body tests instead, e.g. "
                           "`if (found < 0 && hit) { found = i; }`, and let the loop run a "
                           "bound that is the same for every pixel (LANGUAGE.md §7.1).")
        elif cls is A.ReturnStmt:
            if s.value is not None and self._expr(s.value, st, scope):
                scope.returns_vary = True
            if pp_fn:
                self._warn("W7007", s,
                           "This `return` sits under an `if` whose condition can differ from "
                           "pixel to pixel. Such an `if` runs its branches on every pixel, so the "
                           "function returns this value for ALL pixels the first time it reaches "
                           "the `return`, whatever the condition says.",
                           "Assign the result to a local inside the `if` and return it once at "
                           "the end, or select with `cond ? a : b` (LANGUAGE.md §7.1).")
        elif cls is A.ParamDecl and s.default_expr is not None:
            self._expr(s.default_expr, st, scope)
        return st

    def _set(self, st, name, varies, strong):
        if varies:
            st.vary.add(name)
            self.all_vary.add(name)
        elif strong:
            st.vary.discard(name)
        st.defined.add(name)
        self.all_defined.add(name)

    def _expr(self, expr, st, scope) -> bool:
        """Scan an expression — W7006 on a per-pixel `?:` holding a gather, and the call-site
        facts for user functions — and return whether its value can differ per pixel."""
        A = self.A
        stack = [expr]
        while stack:
            n = stack.pop()
            self._tick()
            cls = type(n)
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
            stack.extend(A.iter_child_nodes(n))
        return self._varies(expr, st)

    def _if(self, s, st, scope, loop, pp_loop, pp_fn):
        from .tex_compiler.ast_nodes import collect_assigned_vars
        per_pixel = self._expr(s.condition, st, scope)
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
        then_st = self._block(s.then_body, st.copy(), scope, loop, inner_loop, inner_fn)
        else_st = (self._block(s.else_body, st.copy(), scope, loop, inner_loop, inner_fn)
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

    def _loop(self, s, st, scope, outer_loop, pp_fn):
        A = self.A
        is_for = type(s) is A.ForLoop
        if is_for and s.init is not None:
            st = self._stmt(s.init, st, scope, outer_loop, False, pp_fn)
        head = st
        while True:                               # the head state, to a fixed point
            frame = _CFLoop()
            if s.condition is not None:
                self._expr(s.condition, head, scope)
            body = self._block(s.body, head.copy(), scope, frame, False, pp_fn)
            for c in frame.continues:
                body.join(c)
            if is_for and s.update is not None:
                body = self._stmt(s.update, body, scope, frame, False, pp_fn)
            nxt = head.copy()
            nxt.join(body)
            if nxt.same(head):
                break
            head = nxt
        if s.condition is not None and self._varies(s.condition, head):
            kw = "for" if is_for else "while"
            self._warn("W7007", s,
                       f"This `{kw}` loop's condition can differ from pixel to pixel. The loop "
                       f"keeps running while ANY pixel's condition holds and its body is not "
                       f"masked, so every pixel runs as many passes as the pixel that needs the "
                       f"most, including pixels whose own condition is already false.",
                       "Bound the loop by a value that is the same for every pixel and guard or "
                       "weight the per-pixel work, e.g. `for (int i = 0; i < $max; i++) "
                       "{ if (i < n) { ... } }` (LANGUAGE.md §7.1).")
            # TRK-25 clauses (a)/(b): the pass count is the region's MAXIMUM, so the output
            # depends on which region was cooked. This is the half of W7007 the engine acts on.
            self.varying_loops.add(id(s))
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


def _is_string_type(t) -> bool:
    name = getattr(t, "name", None) or getattr(t, "value", None) or t
    return isinstance(name, str) and name.upper() == "STRING"


def prewarm(programs, shapes=None, *, device: str = "cuda", precision: str = "fp32",
            compile_mode: str = "auto") -> dict:
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
    program (a bad program is skipped, never fatal). Returns a summary of what was warmed."""
    import torch
    from .tex_cache import get_cache
    from .tex_runtime import compiled, graphed, warm_state
    warm_state.ensure_loaded()
    dev_type = "cuda" if (str(device).startswith("cuda") and torch.cuda.is_available()) else "cpu"
    summary = {"programs": 0, "codegen": 0, "bg_compile": 0, "capturable": 0, "errors": 0}
    for source, binding_types in programs:
        try:
            prog = compile(source, binding_types)
            fp = get_cache().fingerprint(source, binding_types)
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
