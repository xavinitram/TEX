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
from .tex_compiler.diagnostics import TEXCompileError, TEXDiagnostic  # noqa: F401

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


def _diag_from_exc(e, source: str):
    """Materialize the structured diagnostic for a per-phase compiler exception (Lexer/
    Parse/TypeCheckError build it lazily), synthesizing an E0000 fallback so the public
    contract is never downgraded to a bare message. Shared by `compile()` and `check()`."""
    from .tex_compiler.diagnostics import make_diagnostic
    if hasattr(e, "_build_diagnostic"):
        e._build_diagnostic()
    diag = getattr(e, "diagnostic", None)
    if diag is None:
        diag = make_diagnostic(code="E0000", message=str(e),
                               loc=getattr(e, "loc", None), source=source, phase="compile")
    return diag


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
    """
    from .tex_cache import get_cache
    from .tex_compiler.lexer import LexerError
    from .tex_compiler.parser import ParseError
    from .tex_compiler.type_checker import TypeCheckError
    from .tex_compiler.diagnostics import TEXMultiError
    try:
        ast, type_map, referenced, assigned, params, used_builtins = \
            get_cache().compile_tex(source, binding_types)
    except TEXMultiError as e:
        raise TEXCompileError(e.diagnostics) from e
    except (LexerError, ParseError, TypeCheckError) as e:
        raise TEXCompileError([_diag_from_exc(e, source)]) from e
    return Program(ast, type_map, referenced, assigned, params, used_builtins, source)


def execute(program: Program, bindings: dict, *, device: str = "cpu",
            precision: str = "fp32", output_names=None) -> dict:
    """Execute a compiled `Program` against `bindings` → `{output_name: tensor}`. Calls
    straight through to `Interpreter.execute` (bit-for-bit identical)."""
    from .tex_runtime.interpreter import Interpreter
    outs = output_names if output_names is not None else sorted(program.assigned.keys())
    return Interpreter().execute(
        program.ast, bindings, program.type_map, device=device,
        output_names=outs, precision=precision, used_builtins=program.used_builtins)


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
