"""
TEX AST Node Definitions.

Every node in the TEX abstract syntax tree is defined here.
Nodes are dataclasses with __slots__ for reduced memory and faster attribute access.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Source location tracking
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SourceLoc:
    """Line/column in the original TEX source (1-based)."""
    line: int
    col: int

    def __repr__(self):
        return f"{self.line}:{self.col}"


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ASTNode:
    loc: SourceLoc = field(default_factory=lambda: SourceLoc(0, 0))


# ---------------------------------------------------------------------------
# Program (top-level)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Program(ASTNode):
    statements: list[ASTNode] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class VarDecl(ASTNode):
    """Variable declaration: `float x = expr;` or `vec4 color;`"""
    type_name: str = ""          # "float", "int", "vec3", "vec4"
    name: str = ""
    initializer: Optional[ASTNode] = None


@dataclass(slots=True)
class Assignment(ASTNode):
    """Assignment: `x = expr;` or `@OUT = expr;` or `@A.r = expr;`"""
    target: ASTNode = None       # Identifier, BindingRef, or ChannelAccess
    value: ASTNode = None


@dataclass(slots=True)
class IfElse(ASTNode):
    """if (cond) { ... } else { ... }"""
    condition: ASTNode = None
    then_body: list[ASTNode] = field(default_factory=list)
    else_body: list[ASTNode] = field(default_factory=list)


@dataclass(slots=True)
class ForLoop(ASTNode):
    """Bounded for loop: `for (int i = 0; i < N; i++) { ... }`

    Executes sequentially (not vectorized). Each iteration runs the body
    as vectorized tensor operations. Hard limit on iterations prevents
    infinite loops.
    """
    init: ASTNode = None             # VarDecl or Assignment
    condition: ASTNode = None        # Expression (must be scalar)
    update: ASTNode = None           # Assignment (e.g. i = i + 1)
    body: list[ASTNode] = field(default_factory=list)


@dataclass(slots=True)
class WhileLoop(ASTNode):
    """While loop: `while (condition) { body }`

    Executes sequentially (like ForLoop). Hard limit on iterations.
    """
    condition: ASTNode = None
    body: list[ASTNode] = field(default_factory=list)


@dataclass(slots=True)
class BreakStmt(ASTNode):
    """Break statement: exits the innermost for/while loop."""
    pass


@dataclass(slots=True)
class ContinueStmt(ASTNode):
    """Continue statement: skips to the next iteration of the innermost for loop."""
    pass


@dataclass(slots=True)
class ExprStatement(ASTNode):
    """A bare expression used as a statement (e.g. function call)."""
    expr: ASTNode = None


@dataclass(slots=True)
class ParamDecl(ASTNode):
    """Parameter declaration: `f$strength = 0.5;` or `i$count;`

    Declares a UI widget parameter with an optional default value.
    At runtime, the actual value comes from the ComfyUI widget.
    The default_expr (if present) must be a literal.
    """
    name: str = ""
    type_hint: str = ""                    # "f", "i", "s", "" (auto -> float)
    default_expr: Optional[ASTNode] = None # Literal for default value


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class BinOp(ASTNode):
    """Binary operation: `a + b`, `x > 0.5`, `a && b`"""
    op: str = ""
    left: ASTNode = None
    right: ASTNode = None


@dataclass(slots=True)
class UnaryOp(ASTNode):
    """Unary operation: `-x`, `!flag`"""
    op: str = ""
    operand: ASTNode = None


@dataclass(slots=True)
class TernaryOp(ASTNode):
    """Ternary: `cond ? a : b`"""
    condition: ASTNode = None
    true_expr: ASTNode = None
    false_expr: ASTNode = None


@dataclass(slots=True)
class FunctionCall(ASTNode):
    """Function call: `clamp(x, 0.0, 1.0)`"""
    name: str = ""
    args: list[ASTNode] = field(default_factory=list)


@dataclass(slots=True)
class Identifier(ASTNode):
    """Variable reference: `x`, `gray`"""
    name: str = ""


@dataclass(slots=True)
class BindingRef(ASTNode):
    """@ or $ binding reference: `@A`, `@OUT`, `$strength`, `f@threshold`"""
    name: str = ""               # The part after @ or $
    kind: str = "wire"           # "wire" (@) or "param" ($)
    type_hint: str = ""          # Explicit type prefix: "f", "i", "v", "v4", "img", "m", "l", "s"


@dataclass(slots=True)
class ChannelAccess(ASTNode):
    """Channel/swizzle access: `@A.r`, `color.rgb`, `v.xy`"""
    object: ASTNode = None       # The base expression
    channels: str = ""           # "r", "g", "b", "a", "rgb", "xyz", etc.


@dataclass(slots=True)
class NumberLiteral(ASTNode):
    """Numeric literal: `1.0`, `42`, `0xFF`"""
    value: float = 0.0
    is_int: bool = False


@dataclass(slots=True)
class StringLiteral(ASTNode):
    """String literal: `"hello world"`"""
    value: str = ""


@dataclass(slots=True)
class VecConstructor(ASTNode):
    """Vector constructor: `vec3(1.0, 0.5, 0.0)` or `vec4(r, g, b, a)`"""
    size: int = 4                # 3 or 4
    args: list[ASTNode] = field(default_factory=list)


@dataclass(slots=True)
class MatConstructor(ASTNode):
    """Matrix constructor: `mat3(...)` or `mat4(...)`"""
    size: int = 3                # 3 or 4 (-> 3x3 or 4x4)
    args: list[ASTNode] = field(default_factory=list)


@dataclass(slots=True)
class CastExpr(ASTNode):
    """Type cast: `float(x)`, `int(y)`, `string(z)`"""
    target_type: str = ""        # "float", "int", "string"
    expr: ASTNode = None


# ---------------------------------------------------------------------------
# Array nodes
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ArrayDecl(ASTNode):
    """Array declaration: `float arr[5];` or `float arr[] = {1.0, 2.0};`"""
    element_type_name: str = ""    # "float" or "int"
    name: str = ""
    size: Optional[int] = None     # None when inferred from initializer
    initializer: Optional[ASTNode] = None  # ArrayLiteral or Identifier (copy)


@dataclass(slots=True)
class ArrayIndexAccess(ASTNode):
    """Array element access: `arr[i]`, `arr[2]`"""
    array: ASTNode = None          # Identifier being indexed
    index: ASTNode = None          # Expression for the index


@dataclass(slots=True)
class ArrayLiteral(ASTNode):
    """Array initializer list: `{1.0, 2.0, 3.0}`"""
    elements: list[ASTNode] = field(default_factory=list)


@dataclass(slots=True)
class ErrorNode(ASTNode):
    """Placeholder for a statement that failed to parse.

    Inserted during parser error recovery so downstream passes
    (type checker, interpreter) can skip it without cascade errors.
    """
    error_message: str = ""
