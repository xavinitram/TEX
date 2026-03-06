"""
TEX Lexer — tokenizes TEX source code into a stream of tokens.

Supports:
  - C-style identifiers, numbers (int and float), operators
  - @ bindings (@A, @OUT, @p1)
  - Keywords: float, int, vec3, vec4, if, else
  - Single-line (//) and multi-line (/* */) comments
  - Dot-access for channel swizzling
"""
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from .ast_nodes import SourceLoc


class TokenType(Enum):
    # Literals
    INT_LIT = auto()
    FLOAT_LIT = auto()
    STRING_LIT = auto()

    # Identifiers and keywords
    IDENT = auto()
    AT_BINDING = auto()       # @name

    # Type keywords
    KW_FLOAT = auto()
    KW_INT = auto()
    KW_VEC3 = auto()
    KW_VEC4 = auto()
    KW_STRING = auto()

    # Control flow keywords
    KW_IF = auto()
    KW_ELSE = auto()
    KW_FOR = auto()

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
    ERROR = auto()


KEYWORDS = {
    "float": TokenType.KW_FLOAT,
    "int": TokenType.KW_INT,
    "vec3": TokenType.KW_VEC3,
    "vec4": TokenType.KW_VEC4,
    "string": TokenType.KW_STRING,
    "if": TokenType.KW_IF,
    "else": TokenType.KW_ELSE,
    "for": TokenType.KW_FOR,
}

SINGLE_CHAR_TOKENS = {
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "%": TokenType.PERCENT,
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


@dataclass
class Token:
    type: TokenType
    value: str
    loc: SourceLoc

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, {self.loc})"


class LexerError(Exception):
    def __init__(self, message: str, loc: SourceLoc):
        self.loc = loc
        super().__init__(f"[{loc}] {message}")


class Lexer:
    """Tokenizes TEX source code."""

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens: list[Token] = []

    def loc(self) -> SourceLoc:
        return SourceLoc(self.line, self.col)

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
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def skip_whitespace(self):
        while self.pos < len(self.source) and self.source[self.pos] in " \t\r\n":
            self.advance()

    def skip_line_comment(self):
        while self.pos < len(self.source) and self.source[self.pos] != "\n":
            self.advance()

    def skip_block_comment(self):
        start_loc = self.loc()
        self.advance()  # skip *
        while self.pos < len(self.source):
            if self.peek() == "*" and self.peek_ahead() == "/":
                self.advance()  # *
                self.advance()  # /
                return
            self.advance()
        raise LexerError("Unterminated block comment", start_loc)

    def read_number(self) -> Token:
        start_loc = self.loc()
        start_pos = self.pos
        is_float = False

        # Check for hex: 0x...
        if self.peek() == "0" and self.peek_ahead() in ("x", "X"):
            self.advance()  # 0
            self.advance()  # x
            while self.pos < len(self.source) and self.source[self.pos] in "0123456789abcdefABCDEF":
                self.advance()
            text = self.source[start_pos:self.pos]
            return Token(TokenType.INT_LIT, text, start_loc)

        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            self.advance()

        if self.pos < len(self.source) and self.source[self.pos] == ".":
            next_ch = self.peek_ahead()
            # Only treat as float if followed by digit (not field access like 1.rgb)
            if next_ch.isdigit():
                is_float = True
                self.advance()  # .
                while self.pos < len(self.source) and self.source[self.pos].isdigit():
                    self.advance()

        # Scientific notation
        if self.pos < len(self.source) and self.source[self.pos] in ("e", "E"):
            is_float = True
            self.advance()
            if self.pos < len(self.source) and self.source[self.pos] in ("+", "-"):
                self.advance()
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self.advance()

        text = self.source[start_pos:self.pos]
        tok_type = TokenType.FLOAT_LIT if is_float else TokenType.INT_LIT
        return Token(tok_type, text, start_loc)

    def read_identifier(self) -> Token:
        start_loc = self.loc()
        start_pos = self.pos
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == "_"):
            self.advance()
        text = self.source[start_pos:self.pos]
        tok_type = KEYWORDS.get(text, TokenType.IDENT)
        return Token(tok_type, text, start_loc)

    def read_string(self) -> Token:
        """Read a double-quoted string literal with escape sequence support."""
        start_loc = self.loc()
        self.advance()  # skip opening "
        chars: list[str] = []
        ESCAPE_MAP = {"\\": "\\", '"': '"', "n": "\n", "t": "\t", "r": "\r"}
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch == '"':
                self.advance()  # skip closing "
                return Token(TokenType.STRING_LIT, "".join(chars), start_loc)
            if ch == '\\':
                esc_loc = self.loc()
                self.advance()  # skip backslash
                if self.pos >= len(self.source):
                    raise LexerError("Unterminated string escape", esc_loc)
                esc_ch = self.source[self.pos]
                if esc_ch not in ESCAPE_MAP:
                    raise LexerError(
                        f"Invalid escape sequence: \\{esc_ch}", esc_loc
                    )
                chars.append(ESCAPE_MAP[esc_ch])
                self.advance()
                continue
            if ch == '\n':
                raise LexerError("Unterminated string (newline in string)", start_loc)
            chars.append(ch)
            self.advance()
        raise LexerError("Unterminated string literal", start_loc)

    def read_at_binding(self) -> Token:
        start_loc = self.loc()
        self.advance()  # skip @
        start_pos = self.pos
        if self.pos >= len(self.source) or not (self.source[self.pos].isalpha() or self.source[self.pos] == "_"):
            raise LexerError("Expected identifier after '@'", start_loc)
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == "_"):
            self.advance()
        name = self.source[start_pos:self.pos]
        return Token(TokenType.AT_BINDING, name, start_loc)

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
            if ch.isdigit():
                self.tokens.append(self.read_number())
                continue

            # Float literal starting with dot (e.g., .5)
            if ch == "." and self.peek_ahead().isdigit():
                start_loc = self.loc()
                start_pos = self.pos
                self.advance()  # .
                while self.pos < len(self.source) and self.source[self.pos].isdigit():
                    self.advance()
                text = self.source[start_pos:self.pos]
                self.tokens.append(Token(TokenType.FLOAT_LIT, text, start_loc))
                continue

            # String literals
            if ch == '"':
                self.tokens.append(self.read_string())
                continue

            # Identifiers and keywords
            if ch.isalpha() or ch == "_":
                self.tokens.append(self.read_identifier())
                continue

            # @ bindings
            if ch == "@":
                self.tokens.append(self.read_at_binding())
                continue

            # Multi-character operators (check longest match first)
            start_loc = self.loc()
            next_ch = self.peek_ahead()

            if ch == "+" and next_ch == "+":
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.PLUS_PLUS, "++", start_loc))
                continue
            if ch == "-" and next_ch == "-":
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.MINUS_MINUS, "--", start_loc))
                continue
            if ch == "+" and next_ch == "=":
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.PLUS_ASSIGN, "+=", start_loc))
                continue
            if ch == "-" and next_ch == "=":
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.MINUS_ASSIGN, "-=", start_loc))
                continue
            if ch == "*" and next_ch == "=":
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.STAR_ASSIGN, "*=", start_loc))
                continue
            if ch == "/" and next_ch == "=":
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.SLASH_ASSIGN, "/=", start_loc))
                continue
            if ch == "=" and next_ch == "=":
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.EQ, "==", start_loc))
                continue
            if ch == "!" and next_ch == "=":
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.NEQ, "!=", start_loc))
                continue
            if ch == "<" and next_ch == "=":
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.LTE, "<=", start_loc))
                continue
            if ch == ">" and next_ch == "=":
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.GTE, ">=", start_loc))
                continue
            if ch == "&" and next_ch == "&":
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.AND, "&&", start_loc))
                continue
            if ch == "|" and next_ch == "|":
                self.advance(); self.advance()
                self.tokens.append(Token(TokenType.OR, "||", start_loc))
                continue

            # Single-character operators and delimiters
            if ch in SINGLE_CHAR_TOKENS:
                self.advance()
                self.tokens.append(Token(SINGLE_CHAR_TOKENS[ch], ch, start_loc))
                continue

            # Standalone = (assignment)
            if ch == "=":
                self.advance()
                self.tokens.append(Token(TokenType.ASSIGN, "=", start_loc))
                continue

            # Standalone < and >
            if ch == "<":
                self.advance()
                self.tokens.append(Token(TokenType.LT, "<", start_loc))
                continue
            if ch == ">":
                self.advance()
                self.tokens.append(Token(TokenType.GT, ">", start_loc))
                continue

            # ! (not)
            if ch == "!":
                self.advance()
                self.tokens.append(Token(TokenType.NOT, "!", start_loc))
                continue

            raise LexerError(f"Unexpected character: {ch!r}", start_loc)

        self.tokens.append(Token(TokenType.EOF, "", self.loc()))
        return self.tokens
