"""
TEX Type Checker — semantic analysis pass over the AST.

Resolves:
  - Variable types and scopes
  - @ binding types (from input metadata)
  - Expression types with automatic promotion rules
  - Channel access validity
  - Function signature matching

Type promotion rules:
  int -> float (implicit)
  float -> vec3 (broadcast)
  float -> vec4 (broadcast)
  vec3 -> vec4 (adds alpha=1.0)
  vec4 -> vec3 (drops alpha) — only via explicit .rgb/.xyz swizzle

Output types:
  float: scalar per-pixel value
  int: integer per-pixel value (stored as float tensor)
  vec3: 3-channel image (RGB)
  vec4: 4-channel image (RGBA)
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from .ast_nodes import (
    ASTNode, Program, VarDecl, Assignment, IfElse, ForLoop, ExprStatement,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor, MatConstructor,
    CastExpr, SourceLoc, ArrayDecl, ArrayIndexAccess, ArrayLiteral, ParamDecl,
)


class TEXType(Enum):
    INT = "int"
    FLOAT = "float"
    VEC3 = "vec3"
    VEC4 = "vec4"
    MAT3 = "mat3"
    MAT4 = "mat4"
    STRING = "string"
    ARRAY = "array"  # fixed-size array; metadata in TEXArrayType
    VOID = "void"    # for statements

    @property
    def is_scalar(self) -> bool:
        return self in (TEXType.INT, TEXType.FLOAT)

    @property
    def is_vector(self) -> bool:
        return self in (TEXType.VEC3, TEXType.VEC4)

    @property
    def is_matrix(self) -> bool:
        return self in (TEXType.MAT3, TEXType.MAT4)

    @property
    def is_string(self) -> bool:
        return self == TEXType.STRING

    @property
    def is_array(self) -> bool:
        return self == TEXType.ARRAY

    @property
    def is_numeric(self) -> bool:
        return self in (TEXType.INT, TEXType.FLOAT, TEXType.VEC3, TEXType.VEC4,
                        TEXType.MAT3, TEXType.MAT4)

    @property
    def channels(self) -> int:
        if self == TEXType.VEC3:
            return 3
        elif self == TEXType.VEC4:
            return 4
        elif self == TEXType.MAT3:
            return 9
        elif self == TEXType.MAT4:
            return 16
        return 1

    @property
    def mat_size(self) -> int:
        """Matrix dimension (3 for mat3, 4 for mat4). 0 for non-matrix types."""
        if self == TEXType.MAT3:
            return 3
        elif self == TEXType.MAT4:
            return 4
        return 0


@dataclass
class TEXArrayType:
    """Metadata for an array type: element type + fixed size."""
    element_type: TEXType  # FLOAT, INT, VEC3, VEC4, or STRING
    size: int


TYPE_NAME_MAP = {
    "float": TEXType.FLOAT,
    "int": TEXType.INT,
    "vec3": TEXType.VEC3,
    "vec4": TEXType.VEC4,
    "mat3": TEXType.MAT3,
    "mat4": TEXType.MAT4,
    "string": TEXType.STRING,
}

# Valid swizzle characters and their meanings
CHANNEL_MAP = {
    "r": 0, "g": 1, "b": 2, "a": 3,
    "x": 0, "y": 1, "z": 2, "w": 3,
}

# Valid single-channel accesses
SINGLE_CHANNELS = {"r", "g", "b", "a", "x", "y", "z", "w"}

# Valid multi-channel swizzles
VALID_SWIZZLES = {
    "rg", "rb", "ra", "gr", "gb", "ga", "br", "bg", "ba", "ar", "ag", "ab",
    "xy", "xz", "xw", "yx", "yz", "yw", "zx", "zy", "zw", "wx", "wy", "wz",
    "rgb", "rgba", "xyz", "xyzw",
    "bgr", "abgr",
}


# Maps typed-binding prefixes to their TEXType
BINDING_HINT_TYPES = {
    "f": TEXType.FLOAT,
    "i": TEXType.INT,
    "v": TEXType.VEC3,
    "v4": TEXType.VEC4,
    "s": TEXType.STRING,
    "img": TEXType.VEC3,   # IMAGE → vec3 at the tensor level
    "m": TEXType.FLOAT,    # MASK → float at the tensor level
    "l": TEXType.VEC4,     # LATENT → vec4 at the tensor level
}


class TypeCheckError(Exception):
    def __init__(self, message: str, loc: SourceLoc):
        self.loc = loc
        super().__init__(f"[{loc}] {message}")


@dataclass
class BindingInfo:
    """Metadata about an @ binding's type at the ComfyUI level."""
    name: str
    tex_type: TEXType
    is_output: bool = False


@dataclass
class TypeChecker:
    """
    Walks the AST and annotates each expression node with its resolved TEXType.
    Also collects @ binding references for the node to create sockets.
    """
    # Mapping of @ binding names -> their types (set before checking)
    binding_types: dict[str, TEXType] = field(default_factory=dict)

    # Populated during checking: all @ bindings referenced in the code
    referenced_bindings: set[str] = field(default_factory=set)

    # Variable scopes (list of dicts for nested scopes)
    _scopes: list[dict[str, TEXType]] = field(default_factory=list)

    # Array metadata scopes (parallel to _scopes)
    _array_scopes: list[dict[str, TEXArrayType]] = field(default_factory=list)

    # Type annotations stored on AST nodes (node id -> type)
    _types: dict[int, TEXType] = field(default_factory=dict)

    # Errors collected (non-fatal where possible)
    errors: list[TypeCheckError] = field(default_factory=list)

    # Multi-output: tracks all @bindings assigned to and their inferred types
    assigned_bindings: dict[str, TEXType] = field(default_factory=dict)

    # Parameter declarations: name → {type: TEXType, type_hint: str}
    param_declarations: dict[str, dict] = field(default_factory=dict)

    # Legacy compat: whether the old "OUT"-only inference mode was active
    _infer_out: bool = False

    @property
    def inferred_out_type(self) -> TEXType | None:
        """Legacy compat: inferred type for @OUT binding."""
        return self.assigned_bindings.get("OUT")

    def check(self, program: Program) -> dict[int, TEXType]:
        """Run type checking on a program. Returns a map from AST node id to TEXType."""
        self._scopes = [{}]
        self._array_scopes = [{}]
        self._types = {}
        self.referenced_bindings = set()
        self.errors = []
        self.assigned_bindings = {}
        self.param_declarations = {}
        self._infer_out = "OUT" not in self.binding_types

        # Pre-populate built-in variables
        builtins = {
            "ix": TEXType.FLOAT, "iy": TEXType.FLOAT,
            "iw": TEXType.FLOAT, "ih": TEXType.FLOAT,
            "u": TEXType.FLOAT, "v": TEXType.FLOAT,
            "fi": TEXType.FLOAT, "fn": TEXType.FLOAT,
            "PI": TEXType.FLOAT, "E": TEXType.FLOAT,
            "ic": TEXType.FLOAT,
        }
        self._scopes[0].update(builtins)

        for stmt in program.statements:
            self._check_stmt(stmt)

        if self.errors:
            raise self.errors[0]

        return self._types

    def get_type(self, node: ASTNode) -> TEXType:
        return self._types.get(id(node), TEXType.VOID)

    def _set_type(self, node: ASTNode, t: TEXType):
        self._types[id(node)] = t

    def _lookup_var(self, name: str) -> TEXType | None:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def _declare_var(self, name: str, t: TEXType, loc: SourceLoc):
        if name in self._scopes[-1]:
            self.errors.append(TypeCheckError(f"Variable '{name}' already declared in this scope", loc))
            return
        self._scopes[-1][name] = t

    def _push_scope(self):
        self._scopes.append({})
        self._array_scopes.append({})

    def _pop_scope(self):
        self._scopes.pop()
        self._array_scopes.pop()

    def _lookup_array_info(self, name: str) -> TEXArrayType | None:
        for scope in reversed(self._array_scopes):
            if name in scope:
                return scope[name]
        return None

    def _error(self, msg: str, loc: SourceLoc):
        self.errors.append(TypeCheckError(msg, loc))

    # -- Promotion rules ------------------------------------------------

    @staticmethod
    def _promote(a: TEXType, b: TEXType) -> TEXType:
        """Find the common type for a binary operation."""
        if a == b:
            return a
        # String cannot be promoted to/from numeric types
        if a.is_string or b.is_string:
            return TEXType.STRING  # caller must validate context
        # int + float -> float
        if {a, b} == {TEXType.INT, TEXType.FLOAT}:
            return TEXType.FLOAT
        # scalar + vec -> vec (broadcast)
        if a.is_scalar and b.is_vector:
            return b
        if b.is_scalar and a.is_vector:
            return a
        # vec3 + vec4 -> vec4 (promote vec3 to vec4 with alpha=1)
        if {a, b} == {TEXType.VEC3, TEXType.VEC4}:
            return TEXType.VEC4
        return TEXType.FLOAT  # fallback

    # -- Statement checking ---------------------------------------------

    def _check_stmt(self, node: ASTNode):
        if isinstance(node, VarDecl):
            self._check_var_decl(node)
        elif isinstance(node, ArrayDecl):
            self._check_array_decl(node)
        elif isinstance(node, Assignment):
            self._check_assignment(node)
        elif isinstance(node, IfElse):
            self._check_if_else(node)
        elif isinstance(node, ForLoop):
            self._check_for_loop(node)
        elif isinstance(node, ExprStatement):
            self._check_expr(node.expr)
        elif isinstance(node, ParamDecl):
            self._check_param_decl(node)
        else:
            self._error(f"Unknown statement type: {type(node).__name__}", node.loc)

    def _check_var_decl(self, node: VarDecl):
        declared_type = TYPE_NAME_MAP.get(node.type_name)
        if declared_type is None:
            self._error(f"Unknown type: {node.type_name}", node.loc)
            return

        if node.initializer:
            init_type = self._check_expr(node.initializer)
            # Check compatibility
            if not self._is_assignable(declared_type, init_type):
                self._error(
                    f"Cannot initialize '{node.type_name} {node.name}' with expression of type '{init_type.value}'",
                    node.loc,
                )

        self._declare_var(node.name, declared_type, node.loc)
        self._set_type(node, declared_type)

    def _check_array_decl(self, node: ArrayDecl):
        """Type-check an array declaration."""
        elem_type = TYPE_NAME_MAP.get(node.element_type_name)
        if elem_type is None:
            self._error(f"Unknown type: {node.element_type_name}", node.loc)
            return
        if elem_type in (TEXType.VOID, TEXType.ARRAY):
            self._error(
                f"Cannot create array of '{node.element_type_name}'",
                node.loc,
            )
            return

        size = node.size
        if node.initializer:
            if isinstance(node.initializer, ArrayLiteral):
                init_size = len(node.initializer.elements)
                if size is not None and size != init_size:
                    self._error(
                        f"Array size mismatch: declared [{size}] but initializer has {init_size} elements",
                        node.loc,
                    )
                if size is None:
                    size = init_size
                # Type-check each element
                for elem in node.initializer.elements:
                    et = self._check_expr(elem)
                    if not self._is_assignable(elem_type, et):
                        self._error(
                            f"Cannot initialize {node.element_type_name} array with {et.value} element",
                            elem.loc,
                        )
                self._set_type(node.initializer, TEXType.ARRAY)
            else:
                # Array copy from another variable
                init_type = self._check_expr(node.initializer)
                if init_type != TEXType.ARRAY:
                    self._error("Array initializer must be an array", node.loc)
                else:
                    # Check element type + size compatibility
                    if isinstance(node.initializer, Identifier):
                        src_info = self._lookup_array_info(node.initializer.name)
                        if src_info:
                            if src_info.element_type != elem_type:
                                self._error(
                                    f"Cannot copy {src_info.element_type.value} array "
                                    f"to {elem_type.value} array",
                                    node.loc,
                                )
                            if size is not None and size != src_info.size:
                                self._error(
                                    f"Array size mismatch: [{size}] vs [{src_info.size}]",
                                    node.loc,
                                )
                            if size is None:
                                size = src_info.size

        if size is None:
            self._error("Array must have a known size", node.loc)
            size = 1  # fallback

        if size > 1024:
            self._error(f"Array size {size} exceeds maximum (1024)", node.loc)

        arr_type = TEXArrayType(element_type=elem_type, size=size)
        self._declare_var(node.name, TEXType.ARRAY, node.loc)
        self._array_scopes[-1][node.name] = arr_type
        self._set_type(node, TEXType.ARRAY)

    def _check_param_decl(self, node: ParamDecl):
        """Type-check a parameter declaration: f$strength = 0.5;"""
        # Resolve type from hint (default: FLOAT)
        tex_type = BINDING_HINT_TYPES.get(node.type_hint, TEXType.FLOAT)

        # Check for @wire / $param name conflict
        if node.name in self.referenced_bindings:
            self._error(
                f"'{node.name}' is used as both @wire and $parameter", node.loc
            )

        # Check default literal type (if present)
        if node.default_expr is not None:
            default_type = self._check_expr(node.default_expr)
            if not self._is_assignable(tex_type, default_type):
                self._error(
                    f"Default value type mismatch for ${node.name}: "
                    f"expected '{tex_type.value}', got '{default_type.value}'",
                    node.loc,
                )

        # Extract default literal value (for backend fallback)
        default_value = None
        if node.default_expr is not None:
            if isinstance(node.default_expr, NumberLiteral):
                default_value = (
                    int(node.default_expr.value)
                    if node.default_expr.is_int
                    else node.default_expr.value
                )
            elif isinstance(node.default_expr, StringLiteral):
                default_value = node.default_expr.value

        # Register parameter
        self.param_declarations[node.name] = {
            "type": tex_type,
            "type_hint": node.type_hint,
            "default_value": default_value,
        }
        self._set_type(node, TEXType.VOID)

    def _check_assignment(self, node: Assignment):
        target_type = self._check_expr(node.target)
        value_type = self._check_expr(node.value)

        # Track @binding assignments as outputs (multi-output inference)
        binding_target = self._get_binding_target(node.target)
        if binding_target is not None:
            name = binding_target.name

            # Don't allow assigning to $param bindings
            if binding_target.kind == "param":
                self._error(
                    f"Cannot assign to parameter ${name} (parameters are read-only widgets)",
                    node.loc,
                )
            # Don't allow @wire assignment if name was already declared as $param
            elif name in self.param_declarations:
                self._error(
                    f"'{name}' is already declared as a $parameter — "
                    f"cannot also assign to @{name}",
                    node.loc,
                )
            else:
                # Reject matrix/array types as output values (not representable in ComfyUI)
                if value_type.is_matrix:
                    self._error(
                        f"Cannot assign {value_type.value} to @{name} "
                        f"(matrix types cannot be output)",
                        node.loc,
                    )
                    self._set_type(node, TEXType.VOID)
                    return
                if value_type.is_array:
                    self._error(
                        f"Cannot assign array to @{name} "
                        f"(array types cannot be output)",
                        node.loc,
                    )
                    self._set_type(node, TEXType.VOID)
                    return

                # Infer output type from assignment
                if isinstance(node.target, ChannelAccess):
                    effective = self._infer_binding_from_channel(node.target)
                else:
                    effective = value_type

                if name not in self.assigned_bindings:
                    self.assigned_bindings[name] = effective
                else:
                    self.assigned_bindings[name] = self._promote_out(
                        self.assigned_bindings[name], effective, node.loc
                    )
            self._set_type(node, TEXType.VOID)
            return

        # Check target is assignable
        if isinstance(node.target, (Identifier, BindingRef, ChannelAccess, ArrayIndexAccess)):
            # Array-to-array assignment (arr = sort(arr))
            if target_type.is_array and value_type.is_array:
                pass  # compatible — sizes checked at runtime
            elif target_type.is_array and not value_type.is_array:
                self._error(
                    f"Cannot assign '{value_type.value}' to array variable",
                    node.loc,
                )
            elif not target_type.is_array and value_type.is_array:
                self._error(
                    f"Cannot assign array to '{target_type.value}' variable",
                    node.loc,
                )
            elif not self._is_assignable(target_type, value_type):
                self._error(
                    f"Cannot assign '{value_type.value}' to target of type '{target_type.value}'",
                    node.loc,
                )
        else:
            self._error("Invalid assignment target", node.loc)

        self._set_type(node, TEXType.VOID)

    def _check_if_else(self, node: IfElse):
        cond_type = self._check_expr(node.condition)
        # Condition should be scalar (float or int used as boolean)
        if cond_type.is_vector:
            self._error("If condition must be a scalar expression, not a vector", node.loc)

        self._push_scope()
        for stmt in node.then_body:
            self._check_stmt(stmt)
        self._pop_scope()

        if node.else_body:
            self._push_scope()
            for stmt in node.else_body:
                self._check_stmt(stmt)
            self._pop_scope()

        self._set_type(node, TEXType.VOID)

    def _check_for_loop(self, node: ForLoop):
        self._push_scope()

        # Check init
        self._check_stmt(node.init)

        # Check condition (must be scalar)
        cond_type = self._check_expr(node.condition)
        if cond_type.is_vector:
            self._error("For-loop condition must be a scalar expression, not a vector", node.loc)

        # Check update
        self._check_stmt(node.update)

        # Check body
        for stmt in node.body:
            self._check_stmt(stmt)

        self._pop_scope()
        self._set_type(node, TEXType.VOID)

    # -- Expression checking --------------------------------------------

    def _check_expr(self, node: ASTNode) -> TEXType:
        if isinstance(node, NumberLiteral):
            t = TEXType.INT if node.is_int else TEXType.FLOAT
            self._set_type(node, t)
            return t

        if isinstance(node, StringLiteral):
            self._set_type(node, TEXType.STRING)
            return TEXType.STRING

        if isinstance(node, Identifier):
            return self._check_identifier(node)

        if isinstance(node, BindingRef):
            return self._check_binding_ref(node)

        if isinstance(node, ChannelAccess):
            return self._check_channel_access(node)

        if isinstance(node, BinOp):
            return self._check_binop(node)

        if isinstance(node, UnaryOp):
            return self._check_unary(node)

        if isinstance(node, TernaryOp):
            return self._check_ternary(node)

        if isinstance(node, FunctionCall):
            return self._check_function_call(node)

        if isinstance(node, VecConstructor):
            return self._check_vec_constructor(node)

        if isinstance(node, MatConstructor):
            return self._check_mat_constructor(node)

        if isinstance(node, CastExpr):
            return self._check_cast(node)

        if isinstance(node, ArrayIndexAccess):
            return self._check_array_index(node)

        if isinstance(node, ArrayLiteral):
            # Standalone array literals only appear in declarations (handled there)
            self._error("Array literal '{...}' can only appear in array declarations", node.loc)
            self._set_type(node, TEXType.ARRAY)
            return TEXType.ARRAY

        self._error(f"Unknown expression type: {type(node).__name__}", node.loc)
        self._set_type(node, TEXType.FLOAT)
        return TEXType.FLOAT

    def _check_identifier(self, node: Identifier) -> TEXType:
        t = self._lookup_var(node.name)
        if t is None:
            self._error(f"Undefined variable: '{node.name}'", node.loc)
            t = TEXType.FLOAT
        self._set_type(node, t)
        return t

    def _check_binding_ref(self, node: BindingRef) -> TEXType:
        self.referenced_bindings.add(node.name)

        # $ parameter bindings — type from declaration or type hint
        if node.kind == "param":
            param_info = self.param_declarations.get(node.name)
            if param_info:
                t = param_info["type"]
            elif node.type_hint:
                t = BINDING_HINT_TYPES.get(node.type_hint, TEXType.FLOAT)
            else:
                t = TEXType.FLOAT  # undeclared param defaults to float
            self._set_type(node, t)
            return t

        # @ wire bindings — check binding_types (inputs), then type hint, then fallback
        # 1. Pre-set input type (from connected wire)
        t = self.binding_types.get(node.name)
        if t is not None:
            self._set_type(node, t)
            return t

        # 2. Already inferred from a previous assignment (output)
        t = self.assigned_bindings.get(node.name)
        if t is not None:
            self._set_type(node, t)
            return t

        # 3. Type hint from code (e.g. f@threshold → FLOAT)
        if node.type_hint:
            t = BINDING_HINT_TYPES.get(node.type_hint, TEXType.VEC4)
            self._set_type(node, t)
            return t

        # 4. Fallback: assume vec4 (image), will be resolved at runtime
        t = TEXType.VEC4
        self._set_type(node, t)
        return t

    def _check_channel_access(self, node: ChannelAccess) -> TEXType:
        obj_type = self._check_expr(node.object)
        channels = node.channels

        if obj_type.is_string:
            self._error("Channel access is not valid on string type", node.loc)
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

        if obj_type.is_matrix:
            self._error("Channel access is not valid on matrix type", node.loc)
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

        if len(channels) == 1:
            # Single channel -> float
            if channels not in SINGLE_CHANNELS:
                self._error(f"Invalid channel: '.{channels}'", node.loc)
            if obj_type == TEXType.VEC3 and channels in ("a", "w"):
                self._error(f"vec3 has no '.{channels}' channel", node.loc)
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

        # Multi-channel swizzle
        if channels not in VALID_SWIZZLES:
            self._error(f"Invalid swizzle: '.{channels}'", node.loc)

        n = len(channels)
        if n == 2:
            # We don't have a vec2 type; treat as vec3 for now? No — error.
            self._error(f"2-component swizzle '.{channels}' not supported (use vec3 or vec4 swizzles)", node.loc)
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT
        elif n == 3:
            self._set_type(node, TEXType.VEC3)
            return TEXType.VEC3
        elif n == 4:
            self._set_type(node, TEXType.VEC4)
            return TEXType.VEC4
        else:
            self._error(f"Invalid swizzle length: '.{channels}'", node.loc)
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

    def _check_binop(self, node: BinOp) -> TEXType:
        lt = self._check_expr(node.left)
        rt = self._check_expr(node.right)

        # String-specific rules
        if lt.is_string or rt.is_string:
            if lt.is_string and rt.is_string:
                if node.op == "+":
                    self._set_type(node, TEXType.STRING)
                    return TEXType.STRING
                if node.op in ("==", "!="):
                    self._set_type(node, TEXType.FLOAT)
                    return TEXType.FLOAT
                self._error(f"Operator '{node.op}' is not supported for strings", node.loc)
                self._set_type(node, TEXType.STRING)
                return TEXType.STRING
            # Mixed string + numeric
            self._error(
                f"Cannot use operator '{node.op}' between string and {(rt if lt.is_string else lt).value} "
                f"(use str() to convert)",
                node.loc,
            )
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

        # Matrix-specific rules
        if lt.is_matrix or rt.is_matrix:
            result = self._check_matrix_binop(lt, rt, node.op, node.loc)
            self._set_type(node, result)
            return result

        if node.op in ("&&", "||"):
            # Logical ops: result is always float (boolean-like)
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

        if node.op in ("==", "!=", "<", ">", "<=", ">="):
            # Comparison ops: result is float (0.0 or 1.0)
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

        # Arithmetic: promote
        result = self._promote(lt, rt)
        self._set_type(node, result)
        return result

    def _check_unary(self, node: UnaryOp) -> TEXType:
        t = self._check_expr(node.operand)
        if t.is_string:
            self._error(f"Unary operator '{node.op}' is not supported for strings", node.loc)
        if node.op == "!":
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT
        # Negation preserves type
        self._set_type(node, t)
        return t

    def _check_ternary(self, node: TernaryOp) -> TEXType:
        self._check_expr(node.condition)
        tt = self._check_expr(node.true_expr)
        ft = self._check_expr(node.false_expr)
        # Both branches must agree for string
        if tt.is_string != ft.is_string:
            self._error("Ternary branches must both be strings or both be numeric", node.loc)
        result = self._promote(tt, ft)
        self._set_type(node, result)
        return result

    def _check_vec_constructor(self, node: VecConstructor) -> TEXType:
        for arg in node.args:
            arg_type = self._check_expr(arg)
            if arg_type.is_string:
                self._error("Vector constructor arguments cannot be strings", node.loc)
            elif arg_type.is_vector:
                self._error("Vector constructor arguments must be scalar", node.loc)

        expected_args = node.size
        if len(node.args) == 1:
            # Broadcast single value: vec4(0.5) = vec4(0.5, 0.5, 0.5, 0.5)
            pass
        elif len(node.args) != expected_args:
            self._error(
                f"vec{node.size} expects {expected_args} arguments (or 1 for broadcast), got {len(node.args)}",
                node.loc,
            )

        t = TEXType.VEC3 if node.size == 3 else TEXType.VEC4
        self._set_type(node, t)
        return t

    def _check_mat_constructor(self, node: MatConstructor) -> TEXType:
        for arg in node.args:
            arg_type = self._check_expr(arg)
            if arg_type.is_string:
                self._error("Matrix constructor arguments cannot be strings", node.loc)
            elif arg_type.is_vector or arg_type.is_matrix:
                self._error("Matrix constructor arguments must be scalar", node.loc)

        n = node.size * node.size  # 9 for mat3, 16 for mat4
        if len(node.args) == 1:
            pass  # Broadcast: mat3(1.0) → scaled identity
        elif len(node.args) != n:
            self._error(
                f"mat{node.size} expects {n} arguments (or 1 for scaled identity), got {len(node.args)}",
                node.loc,
            )

        t = TEXType.MAT3 if node.size == 3 else TEXType.MAT4
        self._set_type(node, t)
        return t

    def _check_matrix_binop(self, lt: TEXType, rt: TEXType, op: str, loc: SourceLoc) -> TEXType:
        """Type-check binary operations involving matrices."""
        if op == "*":
            # mat * mat → mat (matmul)
            if lt.is_matrix and rt.is_matrix:
                if lt != rt:
                    self._error(f"Cannot multiply {lt.value} by {rt.value}", loc)
                return lt
            # mat * vec → vec (matrix-vector product)
            if lt.is_matrix and rt.is_vector:
                if lt == TEXType.MAT3 and rt not in (TEXType.VEC3, TEXType.VEC4):
                    self._error(f"mat3 * requires vec3 or vec4 operand, got {rt.value}", loc)
                if lt == TEXType.MAT4 and rt != TEXType.VEC4:
                    self._error(f"mat4 * requires vec4 operand, got {rt.value}", loc)
                return rt
            # vec * mat → error
            if lt.is_vector and rt.is_matrix:
                self._error(
                    f"Cannot multiply {lt.value} * {rt.value} — use transpose({rt.value}) * {lt.value} instead",
                    loc,
                )
                return lt
            # scalar * mat → mat (element-wise scale)
            if lt.is_scalar and rt.is_matrix:
                return rt
            if lt.is_matrix and rt.is_scalar:
                return lt
        elif op in ("+", "-"):
            # mat +/- mat → mat (element-wise)
            if lt.is_matrix and rt.is_matrix:
                if lt != rt:
                    self._error(f"Cannot {op} {lt.value} and {rt.value}", loc)
                return lt
            # scalar +/- mat or mat +/- scalar → mat (element-wise)
            if lt.is_scalar and rt.is_matrix:
                return rt
            if lt.is_matrix and rt.is_scalar:
                return lt
            self._error(f"Cannot use '{op}' between {lt.value} and {rt.value}", loc)
            return lt
        else:
            self._error(f"Operator '{op}' is not supported for matrix types", loc)
            return lt if lt.is_matrix else rt

    def _check_cast(self, node: CastExpr) -> TEXType:
        expr_type = self._check_expr(node.expr)
        t = TYPE_NAME_MAP.get(node.target_type, TEXType.FLOAT)
        # string(vec3/vec4) is not allowed
        if t.is_string and expr_type.is_vector:
            self._error(f"Cannot cast {expr_type.value} to string", node.loc)
        # numeric casts from string are not allowed (use to_int/to_float)
        if t.is_numeric and expr_type.is_string:
            self._error(
                f"Cannot cast string to {t.value} (use to_int() or to_float())", node.loc
            )
        self._set_type(node, t)
        return t

    def _check_array_index(self, node: ArrayIndexAccess) -> TEXType:
        """Type-check array indexing: arr[i]"""
        array_type = self._check_expr(node.array)
        index_type = self._check_expr(node.index)

        if array_type != TEXType.ARRAY:
            self._error(f"Cannot index non-array type '{array_type.value}'", node.loc)
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

        if not index_type.is_scalar:
            self._error(f"Array index must be int or float, got '{index_type.value}'", node.loc)

        # Resolve element type from array info
        elem_type = TEXType.FLOAT  # default
        if isinstance(node.array, Identifier):
            arr_info = self._lookup_array_info(node.array.name)
            if arr_info:
                elem_type = arr_info.element_type

        self._set_type(node, elem_type)
        return elem_type

    def _check_function_call(self, node: FunctionCall) -> TEXType:
        arg_types = [self._check_expr(arg) for arg in node.args]

        # Validate len() argument type: only STRING and ARRAY allowed
        if node.name == "len" and arg_types:
            if arg_types[0].is_vector:
                self._error(
                    f"len() does not accept {arg_types[0].value} arguments (use on string or array)",
                    node.loc,
                )

        # Array aggregate functions: return element type for vector arrays
        if node.name in ("arr_sum", "arr_min", "arr_max", "median", "arr_avg"):
            if arg_types and arg_types[0] == TEXType.ARRAY:
                if isinstance(node.args[0], Identifier):
                    arr_info = self._lookup_array_info(node.args[0].name)
                    if arr_info:
                        if arr_info.element_type == TEXType.STRING:
                            self._error(f"'{node.name}' does not work on string arrays", node.loc)
                        elif arr_info.element_type.is_vector:
                            result_type = arr_info.element_type
                            self._set_type(node, result_type)
                            return result_type

        # Image reduction validation
        if node.name in ("img_sum", "img_mean", "img_min", "img_max", "img_median"):
            if arg_types and not arg_types[0].is_numeric:
                self._error(f"'{node.name}' expects an image/mask, not {arg_types[0].value}", node.loc)

        # join() validation
        if node.name == "join":
            if arg_types and arg_types[0] != TEXType.ARRAY:
                self._error("join() expects a string array as first argument", node.loc)

        result_type = self._resolve_function_type(node.name, arg_types, node.loc)
        self._set_type(node, result_type)
        return result_type

    def _resolve_function_type(self, name: str, arg_types: list[TEXType], loc: SourceLoc) -> TEXType:
        """Resolve the return type of a built-in function call."""
        from .stdlib_signatures import FUNCTION_SIGNATURES
        sig = FUNCTION_SIGNATURES.get(name)
        if sig is None:
            self._error(f"Unknown function: '{name}'", loc)
            return TEXType.FLOAT

        # Check argument count
        min_args, max_args = sig["args"]
        if not (min_args <= len(arg_types) <= max_args):
            self._error(
                f"Function '{name}' expects {min_args}-{max_args} arguments, got {len(arg_types)}",
                loc,
            )
            return sig["return"](arg_types) if callable(sig["return"]) else sig["return"]

        # Resolve return type
        ret = sig["return"]
        if callable(ret):
            return ret(arg_types)
        return ret

    # -- Assignment compatibility ---------------------------------------

    @staticmethod
    def _is_assignable(target: TEXType, value: TEXType) -> bool:
        """Check if `value` type can be assigned to `target` type."""
        if target == value:
            return True
        # String is only assignable to/from string
        if target.is_string or value.is_string:
            return False
        # Matrix is only assignable to same matrix type
        if target.is_matrix or value.is_matrix:
            return target == value
        # int -> float
        if target == TEXType.FLOAT and value == TEXType.INT:
            return True
        # scalar -> vec (broadcast)
        if target.is_vector and value.is_scalar:
            return True
        # vec3 -> vec4 (add alpha)
        if target == TEXType.VEC4 and value == TEXType.VEC3:
            return True
        # vec4 -> vec3 (drop alpha)
        if target == TEXType.VEC3 and value == TEXType.VEC4:
            return True
        return False

    # -- Binding output inference helpers ---------------------------------

    @staticmethod
    def _get_binding_target(target: ASTNode) -> BindingRef | None:
        """If target is a @binding or @binding.channel, return the BindingRef. Else None."""
        if isinstance(target, BindingRef):
            return target
        if isinstance(target, ChannelAccess):
            obj = target.object
            if isinstance(obj, BindingRef):
                return obj
        return None

    def _infer_binding_from_channel(self, target: ChannelAccess) -> TEXType:
        """Infer binding type from channel assignment like @X.r = expr."""
        binding = target.object
        name = binding.name if isinstance(binding, BindingRef) else "OUT"
        channels = target.channels

        # .a or .w implies vec4 (alpha/w channel)
        if len(channels) == 1 and channels in ("a", "w"):
            return TEXType.VEC4
        # 4-component swizzle implies vec4
        if len(channels) >= 4:
            return TEXType.VEC4
        # If already inferred as VEC4, keep it
        if self.assigned_bindings.get(name) == TEXType.VEC4:
            return TEXType.VEC4
        # Default to VEC3 for .r, .g, .b, .x, .y, .z and 3-component swizzles
        return TEXType.VEC3

    def _promote_out(self, current: TEXType, new: TEXType, loc: SourceLoc) -> TEXType:
        """Promote inferred output type, erroring on string/numeric conflict."""
        if current == new:
            return current
        if current.is_string != new.is_string:
            self._error(
                "Output binding assigned both string and numeric types in different branches",
                loc,
            )
            return current
        return self._promote(current, new)
