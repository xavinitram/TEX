"""
TEX Lexer — tokenizes TEX source code into a stream of tokens.

Supports:
  - C-style identifiers, numbers (int and float), operators
  - @ bindings (@A, @OUT, @p1) and typed bindings (f@threshold, img@result)
  - $ parameter bindings ($strength, f$blend)
  - Keywords: float, int, vec2, vec3, vec4, string, mat3, mat4, if, else, for, while, break, continue, return, const
  - Single-line (//) and multi-line (/* */) comments
  - Dot-access for channel swizzling
"""
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from .ast_nodes import SourceLoc
from .diagnostics import make_diagnostic


class TokenType(Enum):
    # Literals
    INT_LIT = auto()
    FLOAT_LIT = auto()
    STRING_LIT = auto()

    # Identifiers and keywords
    IDENT = auto()
    AT_BINDING = auto()           # @name
    DOLLAR_BINDING = auto()       # $name
    TYPED_AT_BINDING = auto()     # prefix@name (e.g. f@threshold)
    TYPED_DOLLAR_BINDING = auto() # prefix$name (e.g. f$strength)

    # Type keywords
    KW_FLOAT = auto()
    KW_INT = auto()
    KW_VEC2 = auto()
    KW_VEC3 = auto()
    KW_VEC4 = auto()
    KW_STRING = auto()
    KW_MAT3 = auto()
    KW_MAT4 = auto()

    # Control flow keywords
    KW_IF = auto()
    KW_ELSE = auto()
    KW_FOR = auto()
    KW_WHILE = auto()
    KW_BREAK = auto()
    KW_CONTINUE = auto()
    KW_RETURN = auto()
    KW_CONST = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    ASSIGN = auto()           # =
    PLUS_ASSIGN = auto()      # +=
    MINUS_ASSIGN = auto()     # -=
    STAR_ASSIGN = auto()      # *=
    SLASH_ASSIGN = auto()     # /=
    PLUS_PLUS = auto()        # ++
    MINUS_MINUS = auto()      # --
    EQ = auto()               # ==
    NEQ = auto()              # !=
    LT = auto()               # <
    GT = auto()               # >
    LTE = auto()              # <=
    GTE = auto()              # >=
    AND = auto()              # &&
    OR = auto()               # ||
    NOT = auto()              # !
    QUESTION = auto()         # ?
    COLON = auto()            # :

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()           # [
    RBRACKET = auto()           # ]
    COMMA = auto()
    SEMI = auto()
    DOT = auto()

    # Special
    EOF = auto()


KEYWORDS = {
    "float": TokenType.KW_FLOAT,
    "int": TokenType.KW_INT,
    "vec2": TokenType.KW_VEC2,
    "vec3": TokenType.KW_VEC3,
    "vec4": TokenType.KW_VEC4,
    "string": TokenType.KW_STRING,
    "mat3": TokenType.KW_MAT3,
    "mat4": TokenType.KW_MAT4,
    "if": TokenType.KW_IF,
    "else": TokenType.KW_ELSE,
    "for": TokenType.KW_FOR,
    "while": TokenType.KW_WHILE,
    "break": TokenType.KW_BREAK,
    "continue": TokenType.KW_CONTINUE,
    "return": TokenType.KW_RETURN,
    "const": TokenType.KW_CONST,
}

# Two-character operators, checked before SINGLE_CHAR_TOKENS so `<=` beats `<`
# (and comment openers `//`, `/*` are handled before either, preserving
# maximal munch for `/=`).
TWO_CHAR_TOKENS = {
    "++": TokenType.PLUS_PLUS,
    "--": TokenType.MINUS_MINUS,
    "+=": TokenType.PLUS_ASSIGN,
    "-=": TokenType.MINUS_ASSIGN,
    "*=": TokenType.STAR_ASSIGN,
    "/=": TokenType.SLASH_ASSIGN,
    "==": TokenType.EQ,
    "!=": TokenType.NEQ,
    "<=": TokenType.LTE,
    ">=": TokenType.GTE,
    "&&": TokenType.AND,
    "||": TokenType.OR,
}

SINGLE_CHAR_TOKENS = {
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "%": TokenType.PERCENT,
    "=": TokenType.ASSIGN,
    "<": TokenType.LT,
    ">": TokenType.GT,
    "!": TokenType.NOT,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    ",": TokenType.COMMA,
    ";": TokenType.SEMI,
    ".": TokenType.DOT,
    "?": TokenType.QUESTION,
    ":": TokenType.COLON,
}

# Escapes recognized inside string literals. The E1004 message in read_string
# spells out this list — keep the two in sync.
_STRING_ESCAPE_MAP = {"\\": "\\", '"': '"', "n": "\n", "t": "\t", "r": "\r"}

# Type prefixes that can appear immediately before @ or $ to form typed bindings.
# e.g. f@threshold → TYPED_AT_BINDING with prefix="f"
# e.g. img@result → TYPED_AT_BINDING with prefix="img"
# e.g. a@palette → an ARRAY wire (DATA-3; only reachable under the engine profile)
BINDING_TYPE_PREFIXES = {"f", "i", "v", "v2", "v3", "v4", "s", "img", "m", "l", "c", "b", "a"}


def _is_ascii_digit(c: str) -> bool:
    """True only for ASCII digits 0-9 (str.isdigit() accepts Unicode digits)."""
    return "0" <= c <= "9"


def _is_ascii_alpha(c: str) -> bool:
    """True only for ASCII letters a-z/A-Z (str.isalpha() accepts Unicode letters)."""
    return ("a" <= c <= "z") or ("A" <= c <= "Z")


def _is_ident_start(c: str) -> bool:
    """ASCII-only identifier start: letter or underscore."""
    return _is_ascii_alpha(c) or c == "_"


def _is_ident_continue(c: str) -> bool:
    """ASCII-only identifier continuation: letter, digit, or underscore."""
    return _is_ascii_alpha(c) or _is_ascii_digit(c) or c == "_"


@dataclass(slots=True)
class Token:
    type: TokenType
    value: str
    loc: SourceLoc
    prefix: str = ""             # Type prefix for typed bindings (e.g. "f", "img")

    def __repr__(self):
        if self.prefix:
            return f"Token({self.type.name}, {self.value!r}, {self.loc}, prefix={self.prefix!r})"
        return f"Token({self.type.name}, {self.value!r}, {self.loc})"


class LexerError(Exception):
    def __init__(self, message: str, loc: SourceLoc, *, source: str = "",
                 code: str = "E1000", hint: str = "", end_col: int | None = None):
        self.loc = loc
        self.diagnostic = None  # Built lazily
        self._raw_message = message
        self._source = source
        self._code = code
        self._hint = hint
        self._end_col = end_col
        super().__init__(f"[{loc}] {message}")

    def _build_diagnostic(self):
        """Lazily build the diagnostic (only errors that surface need one)."""
        if self.diagnostic is not None:
            return
        self.diagnostic = make_diagnostic(
            code=self._code,
            message=self._raw_message,
            loc=self.loc,
            source=self._source,
            end_col=self._end_col,
            hint=self._hint,
            phase="lexer",
        )


class Lexer:
    """Tokenizes TEX source code."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0  # CT-2: the byte offset is the sole cursor (no line/col)
        self.tokens: list[Token] = []

    def _error(self, message: str, loc: SourceLoc, *,
               code: str = "E1000", hint: str = "", end_col: int | None = None) -> LexerError:
        """Create a LexerError with source context for rich diagnostics."""
        return LexerError(message, loc, source=self.source,
                          code=code, hint=hint, end_col=end_col)

    def loc(self) -> SourceLoc:
        # CT-2: capture only the byte offset; line/col are resolved lazily by
        # SourceLoc, and only if a diagnostic ever renders this location.
        return SourceLoc.from_offset(self.pos, self.source)

    def peek(self) -> str:
        if self.pos >= len(self.source):
            return "\0"
        return self.source[self.pos]

    def peek_ahead(self, offset: int = 1) -> str:
        idx = self.pos + offset
        if idx >= len(self.source):
            return "\0"
        return self.source[idx]

    def advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        return ch

    def _advance_run(self, end: int):
        """Move pos to `end` (batched replacement for repeated advance() calls).
        CT-2: no line/col bookkeeping — the byte offset is the only cursor."""
        self.pos = end

    def skip_whitespace(self):
        src = self.source
        n = len(src)
        i = self.pos
        while i < n and src[i] in " \t\r\n":
            i += 1
        if i > self.pos:
            self._advance_run(i)

    def skip_line_comment(self):
        # Leave the terminating "\n" for skip_whitespace (line stays exact)
        end = self.source.find("\n", self.pos)
        if end == -1:
            end = len(self.source)
        self.pos = end

    def skip_block_comment(self):
        start_loc = self.loc()  # at the `*` of the opening `/*`
        # Search after the opening `*` so `/*/` does not close itself
        end = self.source.find("*/", self.pos + 1)
        if end == -1:
            raise self._error("Unterminated block comment. Did you forget a closing `*/`?",
                              start_loc, code="E1001")
        self._advance_run(end + 2)

    def read_number(self) -> Token:
        # Numbers never contain newlines, so the scan runs on a local index
        # and commits pos/col once at the end (line stays exact).
        start_loc = self.loc()
        start_pos = self.pos
        src = self.source
        n = len(src)
        is_float = False

        # Check for hex: 0x...
        if self.peek() == "0" and self.peek_ahead() in ("x", "X"):
            i = self.pos + 2
            hex_start = i
            while i < n and src[i] in "0123456789abcdefABCDEF":
                i += 1
            if i == hex_start:
                raise self._error("Hex literal '0x' has no digits. Try something like 0xFF.",
                                  start_loc, code="E1002")
            self.pos = i
            return Token(TokenType.INT_LIT, src[start_pos:i], start_loc)

        i = self.pos
        while i < n and _is_ascii_digit(src[i]):
            i += 1

        if i < n and src[i] == ".":
            # Only treat as float if followed by digit (not field access like 1.rgb)
            if i + 1 < n and _is_ascii_digit(src[i + 1]):
                is_float = True
                i += 1  # .
                while i < n and _is_ascii_digit(src[i]):
                    i += 1

        # Scientific notation
        if i < n and src[i] in ("e", "E"):
            j = i + 1
            if j < n and src[j] in ("+", "-"):
                j += 1
            exp_start = j
            while j < n and _is_ascii_digit(src[j]):
                j += 1
            if j > exp_start:
                is_float = True
                i = j
            # else: no digits after 'e' — leave it for a separate identifier

        self.pos = i
        text = src[start_pos:i]
        tok_type = TokenType.FLOAT_LIT if is_float else TokenType.INT_LIT
        return Token(tok_type, text, start_loc)

    def read_identifier(self) -> Token:
        start_loc = self.loc()
        start_pos = self.pos
        src = self.source
        n = len(src)
        i = self.pos
        while i < n and _is_ident_continue(src[i]):
            i += 1
        self.pos = i
        text = src[start_pos:i]

        # Check for typed binding: prefix immediately followed by @ or $
        # e.g. f@threshold, img@result, i$count
        if self.pos < len(self.source) and self.source[self.pos] in ("@", "$"):
            if text in BINDING_TYPE_PREFIXES:
                sigil = self.advance()  # consume @ or $
                name = self._read_binding_name(text, sigil, start_loc)
                tok_type = (TokenType.TYPED_AT_BINDING if sigil == "@"
                            else TokenType.TYPED_DOLLAR_BINDING)
                return Token(tok_type, name, start_loc, prefix=text)

        tok_type = KEYWORDS.get(text, TokenType.IDENT)
        return Token(tok_type, text, start_loc)

    def read_string(self) -> Token:
        """Read a double-quoted string literal with escape sequence support."""
        start_loc = self.loc()
        self.advance()  # skip opening "
        chars: list[str] = []
        src = self.source
        n = len(src)
        while self.pos < n:
            # Batch the plain run up to the next terminator. Runs are
            # newline-free (\n terminates), so col stays exact for esc_loc.
            i = self.pos
            while i < n and src[i] not in '"\\\n':
                i += 1
            if i > self.pos:
                chars.append(src[self.pos:i])
                self.pos = i
            if i >= n:
                break
            ch = src[i]
            if ch == '"':
                self.advance()  # skip closing "
                return Token(TokenType.STRING_LIT, "".join(chars), start_loc)
            if ch == '\\':
                esc_loc = self.loc()
                self.advance()  # skip backslash
                if self.pos >= n:
                    raise self._error("Unterminated string escape at end of input.",
                                      esc_loc, code="E1003")
                esc_ch = src[self.pos]
                if esc_ch not in _STRING_ESCAPE_MAP:
                    raise self._error(
                        f"Invalid escape sequence: \\{esc_ch}. Valid escapes are: \\\\, \\\", \\n, \\t, \\r.",
                        esc_loc, code="E1004"
                    )
                chars.append(_STRING_ESCAPE_MAP[esc_ch])
                self.advance()
                continue
            # ch == '\n'
            raise self._error("Unterminated string — strings can't span multiple lines. Close it with a `\"` before the line break.",
                              start_loc, code="E1005")
        raise self._error("Unterminated string literal. Did you forget a closing `\"`?",
                          start_loc, code="E1006")

    def _read_binding_name(self, prefix_or_sigil: str, sigil: str,
                            start_loc: SourceLoc) -> str:
        """Read the identifier part after a binding sigil (@ or $).

        Used by both read_at_binding/read_dollar_binding and typed binding
        detection in read_identifier.
        """
        name_start = self.pos
        src = self.source
        n = len(src)
        if name_start >= n or not _is_ident_start(src[name_start]):
            raise self._error(
                f"Expected a name after '{prefix_or_sigil}{sigil}'. For example: {sigil}myInput",
                start_loc, code="E1007"
            )
        i = name_start
        while i < n and _is_ident_continue(src[i]):
            i += 1
        self.pos = i
        return src[name_start:i]

    def read_at_binding(self) -> Token:
        start_loc = self.loc()
        self.advance()  # skip @
        name = self._read_binding_name("", "@", start_loc)
        return Token(TokenType.AT_BINDING, name, start_loc)

    def read_dollar_binding(self) -> Token:
        start_loc = self.loc()
        self.advance()  # skip $
        name = self._read_binding_name("", "$", start_loc)
        return Token(TokenType.DOLLAR_BINDING, name, start_loc)

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source, returning a list of tokens ending with EOF."""
        self.tokens = []
        while self.pos < len(self.source):
            self.skip_whitespace()
            if self.pos >= len(self.source):
                break

            ch = self.peek()

            # Comments
            if ch == "/" and self.peek_ahead() == "/":
                self.advance()  # /
                self.advance()  # /
                self.skip_line_comment()
                continue
            if ch == "/" and self.peek_ahead() == "*":
                self.advance()  # /
                self.skip_block_comment()
                continue

            # Numbers
            if _is_ascii_digit(ch):
                self.tokens.append(self.read_number())
                continue

            # Float literal starting with dot (e.g., .5)
            if ch == "." and _is_ascii_digit(self.peek_ahead()):
                start_loc = self.loc()
                start_pos = self.pos
                src = self.source
                n = len(src)
                i = self.pos + 1  # .
                while i < n and _is_ascii_digit(src[i]):
                    i += 1
                self.pos = i
                self.tokens.append(Token(TokenType.FLOAT_LIT, src[start_pos:i], start_loc))
                continue

            # String literals
            if ch == '"':
                self.tokens.append(self.read_string())
                continue

            # Identifiers and keywords
            if _is_ident_start(ch):
                self.tokens.append(self.read_identifier())
                continue

            # @ bindings
            if ch == "@":
                self.tokens.append(self.read_at_binding())
                continue

            # $ parameter bindings
            if ch == "$":
                self.tokens.append(self.read_dollar_binding())
                continue

            # Multi-character operators (check longest match first). At the
            # last char the slice is 1 char long, misses the dict, and falls
            # through to the single-char lookup. No 2-char operator contains
            # a newline, so col += 2 is exact.
            start_loc = self.loc()
            pair = self.source[self.pos:self.pos + 2]
            tt = TWO_CHAR_TOKENS.get(pair)
            if tt is not None:
                self.pos += 2
                self.tokens.append(Token(tt, pair, start_loc))
                continue

            # Single-character operators and delimiters
            if ch in SINGLE_CHAR_TOKENS:
                self.advance()
                self.tokens.append(Token(SINGLE_CHAR_TOKENS[ch], ch, start_loc))
                continue

            raise self._error(f"Unexpected character: {ch!r}. TEX doesn't recognize this symbol.",
                              start_loc, code="E1008")

        self.tokens.append(Token(TokenType.EOF, "", self.loc()))
        return self.tokens
