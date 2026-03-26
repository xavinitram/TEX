"""
TEX Diagnostics — structured error reporting with source snippets,
suggestions, and contextual hints.

Every TEX error produces a TEXDiagnostic carrying:
  - Error code (E1xxx–E6xxx) for searchability and documentation
  - Source snippet with caret underline
  - "Try:" suggestions via fuzzy matching
  - Contextual hints for common beginner mistakes

TEXMultiError aggregates multiple diagnostics so the user sees
ALL problems at once, not just the first.

Error code ranges:
  E1xxx  Lexer (syntax / tokenization)
  E2xxx  Parser (grammar / structure)
  E3xxx  Type checker — names and scope
  E4xxx  Type checker — types and coercions
  E5xxx  Type checker — function signatures
  E6xxx  Runtime (interpreter)
  W7xxx  Warnings
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Iterable

from .ast_nodes import SourceLoc

TEX_BUG_REPORT_URL = "https://github.com/xavinitram/TEX/issues"
TEX_WIKI_URL = "https://github.com/xavinitram/TEX/wiki"


def wiki_url_for_code(code: str) -> str:
    """Return the wiki URL for an error code, e.g. E3001 -> wiki/Error-Codes#e3001"""
    return f"{TEX_WIKI_URL}/Error-Codes#{code.lower()}"


# ── Diagnostic dataclass ──────────────────────────────────────────────

@dataclass(slots=True)
class TEXDiagnostic:
    """Structured error/warning produced by the TEX compiler."""
    code: str                         # e.g. "E3001"
    severity: str                     # "error" | "warning"
    message: str                      # Primary human-readable message
    loc: SourceLoc | None             # Line/col (1-based)
    source_line: str                  # The offending line of source code
    end_col: int | None = None        # End column for underline span
    suggestions: list[str] = field(default_factory=list)
    hint: str = ""                    # Contextual help text
    docs_url: str = ""                # Link to wiki page for this error code
    phase: str = ""                   # "lexer" | "parser" | "type_checker" | "runtime"

    def render(self) -> str:
        """Render a human-friendly formatted error message.

        Message leads, error code trails. The user's problem comes first;
        metadata comes last.
        """
        parts = []

        # Lead with the human message — this is what matters
        parts.append(self.message)

        # Source snippet with caret
        if self.loc and self.source_line:
            line_num = str(self.loc.line)
            gutter = " " * len(line_num)
            parts.append(f"  {gutter} |")
            parts.append(f"  {line_num} | {self.source_line}")

            # Underline
            col = max(self.loc.col - 1, 0)
            if self.end_col is not None:
                length = max(self.end_col - self.loc.col, 1)
            else:
                length = 1
            underline = " " * col + "~" * length
            parts.append(f"  {gutter} | {underline}")

        # Suggestions
        if self.suggestions:
            if len(self.suggestions) == 1:
                parts.append(f"  > Try: {self.suggestions[0]}")
            else:
                joined = ", ".join(self.suggestions)
                parts.append(f"  > Try one of: {joined}")

        # Hint
        if self.hint:
            parts.append(f"  > Help: {self.hint}")

        # Error code — at the bottom, for reference
        if self.code:
            parts.append(f"  > Error Code: {self.code}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        """Serialize for JSON transport to the frontend."""
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "line": self.loc.line if self.loc else None,
            "col": self.loc.col if self.loc else None,
            "end_col": self.end_col,
            "source_line": self.source_line,
            "suggestions": self.suggestions,
            "hint": self.hint,
            "phase": self.phase,
            **({"docs_url": self.docs_url} if self.docs_url else {}),
        }


# ── Multi-error wrapper ──────────────────────────────────────────────

class TEXMultiError(Exception):
    """Wraps multiple TEXDiagnostics so all errors surface at once."""

    def __init__(self, diagnostics: list[TEXDiagnostic]):
        self.diagnostics = diagnostics
        rendered = "\n\n".join(d.render() for d in diagnostics)
        super().__init__(rendered)


# ── Source snippet helper ─────────────────────────────────────────────

def get_source_line(source: str, line: int) -> str:
    """Extract a single line from source text (1-indexed). Returns '' if out of range."""
    if not source or line < 1:
        return ""
    lines = source.split("\n")
    if line > len(lines):
        return ""
    return lines[line - 1]


# ── Fuzzy matching ────────────────────────────────────────────────────

def suggest_similar(
    name: str,
    candidates: Iterable[str],
    max_results: int = 3,
    cutoff: float = 0.6,
) -> list[str]:
    """
    Suggest similar names using difflib's sequence matcher.

    Returns up to `max_results` matches above the `cutoff` threshold.
    Handles transpositions well (e.g. 'clampp' → 'clamp').
    """
    matches = difflib.get_close_matches(name, list(candidates), n=max_results, cutoff=cutoff)
    return matches


# ── GLSL / JS / Houdini alias hints ──────────────────────────────────

_FOREIGN_FUNCTION_HINTS: dict[str, str] = {
    # GLSL
    "texture2D":     "In TEX, use sample(@img, u, v) to sample images.",
    "texture":       "In TEX, use sample(@img, u, v) to sample images.",
    "texelFetch":    "In TEX, use fetch(@img, ix, iy) for nearest-neighbor pixel fetch.",
    "frag_color":    "In TEX, assign to @OUT to produce output.",
    "gl_FragColor":  "In TEX, assign to @OUT to produce output.",
    "gl_FragCoord":  "In TEX, use u, v (0–1 normalized) or ix, iy (pixel coordinates).",
    "discard":       "TEX processes every pixel — there is no discard. Use conditional assignment instead.",

    # HLSL
    "float2":        "TEX uses GLSL-style names. There is no float2 — use a plain float for scalars.",
    "float3":        "TEX uses GLSL-style names. Use vec3 instead of float3.",
    "float4":        "TEX uses GLSL-style names. Use vec4 instead of float4.",
    "saturate":      "TEX doesn't have saturate(). Use clamp(x, 0.0, 1.0) instead.",
    "frac":          "TEX uses fract() instead of frac().",
    "tex2D":         "In TEX, use sample(@img, u, v) to sample images.",

    # JavaScript / Python
    "console.log":   "TEX has no print or log function. Use str() to format values.",
    "print":         "TEX has no print function. Assign results to @OUT or another @name.",
    "Math.sin":      "In TEX, just use sin(x) — no Math prefix needed.",
    "Math.cos":      "In TEX, just use cos(x) — no Math prefix needed.",
    "Math.PI":       "In TEX, use the built-in constant PI.",
    "Math.random":   "In TEX, use perlin(x, y) or simplex(x, y) for noise.",
    "random":        "In TEX, use perlin(x, y) or simplex(x, y) for noise.",
    "parseInt":      "In TEX, use to_int(s) to parse a string as an integer.",
    "parseFloat":    "In TEX, use to_float(s) to parse a string as a float.",

    # Houdini VEX
    "getattrib":     "TEX uses @name syntax for inputs. Write @A to read from input A.",
    "setattrib":     "TEX uses @name syntax for outputs. Write @OUT = value; to set output.",
    "chramp":        "TEX doesn't have chramp yet. Use smoothstep() or fit() for remapping.",
    "pcopen":        "TEX doesn't support point clouds. Use sample() for spatial lookups.",
}

_FOREIGN_VARIABLE_HINTS: dict[str, str] = {
    "gl_FragCoord":  "TEX provides u, v (0–1 normalized) and ix, iy (pixel coordinates).",
    "fragCoord":     "TEX provides u, v (0–1 normalized) and ix, iy (pixel coordinates).",
    "uv":            "TEX provides separate u and v variables (not a vec2 uv).",
    "UV":            "TEX provides separate u and v variables (not a vec2 UV). Use u, v instead.",
    "resolution":    "TEX provides iw (width) and ih (height) as separate floats.",
    "iResolution":   "TEX provides iw (width) and ih (height) as separate floats.",
    "iTime":         "TEX provides fi (frame index) and fn (normalized frame 0–1).",
    "time":          "TEX provides fi (frame index) and fn (normalized frame 0–1).",
    "PI":            "PI is a built-in constant — make sure the capitalization is correct.",
}

_FOREIGN_KEYWORD_HINTS: dict[str, str] = {
    "const":   "Use float x = ..., int x = ..., or vec3 x = ... — TEX uses explicit types.",
    "let":     "Use float x = ..., int x = ..., or vec3 x = ... — TEX uses explicit types.",
    "var":     "Use float x = ..., int x = ..., or vec3 x = ... — TEX uses explicit types.",
    "auto":    "Use float x = ..., int x = ..., or vec3 x = ... — TEX uses explicit types.",
    "def":     "TEX uses C-style function syntax: float myFunc(float x) { return x * 2.0; }",
    "fn":      "fn is a built-in variable (normalized frame). Define functions with: float myFunc(float x) { return x; }",
    "func":    "TEX uses C-style function syntax: float myFunc(float x) { return x * 2.0; }",
    "function":"TEX uses C-style function syntax: float myFunc(float x) { return x * 2.0; }",
    "class":   "TEX is expression-based — there are no classes or structs.",
    "struct":  "TEX is expression-based — there are no structs. Use vec3/vec4 for compound values.",
    "import":  "TEX is self-contained — there is no import system.",
    "include": "TEX is self-contained — there is no include system.",
    "void":    "TEX doesn't use void. Assign to @OUT to produce output.",
    "uniform": "TEX uses $param for user parameters and @name for inputs.",
    "varying": "TEX uses @name for inputs. Every pixel runs the same code with different coordinates.",
    "in":      "TEX uses @name for inputs. Write @A to read from input A.",
    "out":     "TEX uses @OUT for output. Assign to @OUT = value; to set it.",
    "inout":   "TEX uses @name for both input and output. Read from @A, write to @OUT.",
    "elif":    "TEX uses else if (two words), not elif.",
    "else if": None,  # valid — no hint needed
    "switch":  "TEX doesn't have switch/case. Use if/else if chains instead.",
    "case":    "TEX doesn't have switch/case. Use if/else if chains instead.",
    "true":    "TEX uses 1.0 for true and 0.0 for false — there are no boolean literals.",
    "false":   "TEX uses 1.0 for true and 0.0 for false — there are no boolean literals.",
    "null":    "TEX has no null. Use 0.0 or vec3(0.0) for default values.",
    "None":    "TEX has no None. Use 0.0 or vec3(0.0) for default values.",
    "bool":    "TEX has no bool type. Use float (0.0 = false, non-zero = true).",
    "vec2":    "TEX doesn't have vec2. Use float for single values, vec3 for RGB, vec4 for RGBA.",
    "double":  "TEX uses float for all decimal numbers — there is no double type.",
}


# ── Hint lookup helpers ──────────────────────────────────────────────

def get_function_hint(name: str) -> str:
    """Return a contextual hint for a foreign/unknown function name, or ''."""
    return _FOREIGN_FUNCTION_HINTS.get(name, "")


def get_variable_hint(name: str) -> str:
    """Return a contextual hint for a foreign/unknown variable name, or ''."""
    return _FOREIGN_VARIABLE_HINTS.get(name, "")


def get_keyword_hint(keyword: str) -> str | None:
    """
    Return a contextual hint for a foreign keyword, or None if not recognized.
    Returns '' (empty string) if keyword is valid TEX (no hint needed).
    """
    return _FOREIGN_KEYWORD_HINTS.get(keyword)


# ── Diagnostic builder helpers ────────────────────────────────────────

def make_diagnostic(
    code: str,
    message: str,
    loc: SourceLoc | None,
    source: str = "",
    *,
    end_col: int | None = None,
    suggestions: list[str] | None = None,
    hint: str = "",
    phase: str = "",
    severity: str = "error",
) -> TEXDiagnostic:
    """Convenience builder for TEXDiagnostic with auto source-line extraction."""
    source_line = get_source_line(source, loc.line) if (loc and source) else ""
    return TEXDiagnostic(
        code=code,
        severity=severity,
        message=message,
        loc=loc,
        source_line=source_line,
        end_col=end_col,
        suggestions=suggestions or [],
        hint=hint,
        docs_url=wiki_url_for_code(code),
        phase=phase,
    )
