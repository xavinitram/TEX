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
    """Variable declaration: `float x = expr;` or `const vec4 color = expr;`"""
    type_name: str = ""          # "float", "int", "vec3", "vec4"
    name: str = ""
    initializer: Optional[ASTNode] = None
    is_const: bool = False


@dataclass(slots=True)
class Assignment(ASTNode):
    """Assignment: `x = expr;` or `@OUT = expr;` or `@A.r = expr;`

    For scatter compound assignments (@OUT[x,y] += val), op is set to the
    operator ("+", "-", "*", "/") instead of desugaring, so the interpreter
    can use scatter_add_ / index_put_ with accumulate=True.
    """
    target: ASTNode = None       # Identifier, BindingRef, ChannelAccess, or BindingIndexAccess
    value: ASTNode = None
    op: Optional[str] = None     # None for =, "+" for +=, "-" for -=, "*" for *=, "/" for /=


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
class FunctionDef(ASTNode):
    """User-defined function: `float blend(float a, float b, float t) { ... }`"""
    return_type: str = ""                # "float", "int", "vec3", "vec4", "string"
    name: str = ""
    params: list[tuple[str, str]] = field(default_factory=list)  # [(type, name), ...]
    body: list[ASTNode] = field(default_factory=list)


@dataclass(slots=True)
class ReturnStmt(ASTNode):
    """Return statement inside a user-defined function."""
    value: ASTNode = None


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
    # type_hint controls the ComfyUI widget type and value conversion:
    #   "f"  — float slider (default when omitted)
    #   "i"  — integer slider
    #   "s"  — string input
    #   "b"  — boolean toggle (converted to 0.0/1.0 float)
    #   "c"  — hex color picker (e.g. "#FF8800" → [R, G, B] float list)
    #   "v2" — vec2 comma string ("x, y" → [float, float])
    #   "v3" — vec3 comma string ("x, y, z" → [float, float, float])
    #   "v4" — vec4 comma string ("x, y, z, w" → [float, float, float, float])
    type_hint: str = ""
    default_expr: Optional[ASTNode] = None  # Literal for default value


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
    # type_hint for wire bindings (@) controls type inference:
    #   "f"   — float/mask (scalar per pixel)
    #   "i"   — integer
    #   "v"   — vec3 image (alias for "v3")
    #   "v2"  — vec2 image
    #   "v3"  — vec3 image
    #   "v4"  — vec4 image (RGBA)
    #   "img" — image (alias for "v3")
    #   "m"   — mask (alias for "f")
    #   "l"   — latent (vec4 with latent metadata passthrough)
    #   "s"   — string
    #   ""    — auto-infer from connected input tensor shape
    type_hint: str = ""


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
class BindingIndexAccess(ASTNode):
    """Binding fetch access: `@Image[ix, iy]` or `@Image[ix, iy, frame]`"""
    binding: ASTNode = None              # BindingRef
    args: list[ASTNode] = field(default_factory=list)  # 2 args (x, y) or 3 args (x, y, frame)


@dataclass(slots=True)
class BindingSampleAccess(ASTNode):
    """Binding sample access: `@Image(u, v)` or `@Image(u, v, frame)`"""
    binding: ASTNode = None              # BindingRef
    args: list[ASTNode] = field(default_factory=list)  # 2 args (u, v) or 3 args (u, v, frame)


@dataclass(slots=True)
class ErrorNode(ASTNode):
    """Placeholder for a statement that failed to parse.

    Inserted during parser error recovery so downstream passes
    (type checker, interpreter) can skip it without cascade errors.
    """
    error_message: str = ""


def try_extract_static_range(node: ForLoop) -> tuple[str, int, int, int] | None:
    """Extract a fully static for-loop as (var_name, start, stop, step).

    Matches: for (int VAR = START; VAR < END; VAR = VAR + STEP)
    where START, END, STEP are all NumberLiterals (possibly after constant folding).
    Returns None if the pattern doesn't match.

    Shared by both the interpreter and codegen to avoid duplication.
    """
    init = node.init
    if not isinstance(init, VarDecl) or init.initializer is None:
        return None
    if not isinstance(init.initializer, NumberLiteral):
        return None
    loop_var = init.name
    start = int(init.initializer.value)

    cond = node.condition
    if not isinstance(cond, BinOp) or cond.op not in ("<", "<="):
        return None
    if not isinstance(cond.left, Identifier) or cond.left.name != loop_var:
        return None
    if not isinstance(cond.right, NumberLiteral):
        return None
    end = int(cond.right.value)
    if cond.op == "<=":
        end += 1

    update = node.update
    if not isinstance(update, Assignment):
        return None
    if not isinstance(update.target, Identifier) or update.target.name != loop_var:
        return None
    if not isinstance(update.value, BinOp):
        return None
    upd = update.value
    if not isinstance(upd.left, Identifier) or upd.left.name != loop_var:
        return None
    if not isinstance(upd.right, NumberLiteral):
        return None
    if upd.op == "+":
        step = int(upd.right.value)
    elif upd.op == "-":
        step = -int(upd.right.value)
    else:
        return None

    if step == 0:
        return None
    return (loop_var, start, end, step)


def collect_assigned_vars(stmts: list[ASTNode]) -> tuple[set[str], set[str]]:
    """Collect variable names and binding names assigned in a statement list.

    Returns (env_vars, binding_names). Recursively walks into nested
    if/else and loop bodies to find all possible assignments.
    Shared by both the interpreter and codegen.
    """
    env_vars: set[str] = set()
    binding_names: set[str] = set()
    for stmt in stmts:
        if isinstance(stmt, Assignment):
            t = stmt.target
            if isinstance(t, Identifier):
                env_vars.add(t.name)
            elif isinstance(t, BindingRef):
                binding_names.add(t.name)
            elif isinstance(t, ChannelAccess):
                o = t.object
                if isinstance(o, Identifier):
                    env_vars.add(o.name)
                elif isinstance(o, BindingRef):
                    binding_names.add(o.name)
            elif isinstance(t, ArrayIndexAccess):
                if isinstance(t.array, Identifier):
                    env_vars.add(t.array.name)
            elif isinstance(t, BindingIndexAccess):
                if isinstance(t.binding, BindingRef):
                    binding_names.add(t.binding.name)
        elif isinstance(stmt, VarDecl):
            env_vars.add(stmt.name)
        elif isinstance(stmt, ArrayDecl):
            env_vars.add(stmt.name)
        elif isinstance(stmt, IfElse):
            e1, b1 = collect_assigned_vars(stmt.then_body)
            e2, b2 = collect_assigned_vars(stmt.else_body)
            env_vars |= e1 | e2
            binding_names |= b1 | b2
        elif isinstance(stmt, (ForLoop, WhileLoop)):
            e, b = collect_assigned_vars(stmt.body)
            env_vars |= e
            binding_names |= b
    return env_vars, binding_names
