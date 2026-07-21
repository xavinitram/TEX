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
    from .tex_compiler.lexer import Lexer, LexerError
    from .tex_compiler.parser import Parser, ParseError
    from .tex_compiler.type_checker import TypeChecker
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
        try:
            tokens = Lexer(source).tokenize()
        except LexerError as e:
            return pragma_diags + [_diag_from_exc(e, source)]
        try:
            program = Parser(tokens, source=source).parse()
        except TEXMultiError as e:
            return pragma_diags + list(e.diagnostics)
        except ParseError as e:
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
