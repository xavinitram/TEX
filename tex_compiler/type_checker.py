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
  float -> vec2 (broadcast)
  float -> vec3 (broadcast)
  float -> vec4 (broadcast)
  vec2 -> vec3 (pads with 0)
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
    ASTNode, Program, VarDecl, Assignment, IfElse, ForLoop, WhileLoop, ExprStatement,
    BreakStmt, ContinueStmt, FunctionDef, ReturnStmt,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor, MatConstructor,
    CastExpr, SourceLoc, ArrayDecl, ArrayIndexAccess, ArrayLiteral, ParamDecl,
    BindingIndexAccess, BindingSampleAccess,
)


class TEXType(Enum):
    INT = "int"
    FLOAT = "float"
    VEC2 = "vec2"
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
        return self in (TEXType.VEC2, TEXType.VEC3, TEXType.VEC4)

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
        return self in (TEXType.INT, TEXType.FLOAT, TEXType.VEC2, TEXType.VEC3,
                        TEXType.VEC4, TEXType.MAT3, TEXType.MAT4)

    @property
    def channels(self) -> int:
        if self == TEXType.VEC2:
            return 2
        elif self == TEXType.VEC3:
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
    "vec2": TEXType.VEC2,
    "vec3": TEXType.VEC3,
    "vec4": TEXType.VEC4,
    "mat3": TEXType.MAT3,
    "mat4": TEXType.MAT4,
    "string": TEXType.STRING,
}

_VEC_RANK = {TEXType.VEC2: 0, TEXType.VEC3: 1, TEXType.VEC4: 2}
_VEC_SIZE_TYPE = {2: TEXType.VEC2, 3: TEXType.VEC3, 4: TEXType.VEC4}

# Valid swizzle characters and their meanings
CHANNEL_MAP = {
    "r": 0, "g": 1, "b": 2, "a": 3,
    "x": 0, "y": 1, "z": 2, "w": 3,
}

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
    "v2": TEXType.VEC2,
    "v3": TEXType.VEC3,
    "v4": TEXType.VEC4,
    "s": TEXType.STRING,
    "img": TEXType.VEC3,   # IMAGE → vec3 at the tensor level
    "m": TEXType.FLOAT,    # MASK → float at the tensor level
    "l": TEXType.VEC4,     # LATENT → vec4 at the tensor level
    "c": TEXType.VEC3,     # Color → RGB vec3 (hex string in widget)
    "b": TEXType.INT,      # Boolean → 0/1 checkbox
}


class TypeCheckError(Exception):
    def __init__(self, message: str, loc: SourceLoc, *, source: str = "",
                 code: str = "E3000", hint: str = "", suggestions: list[str] | None = None,
                 end_col: int | None = None):
        self.loc = loc
        self.diagnostic = None  # Built lazily
        self._raw_message = message
        self._source = source
        self._code = code
        self._hint = hint
        self._suggestions = suggestions or []
        self._end_col = end_col
        super().__init__(f"[{loc}] {message}")

    def _build_diagnostic(self):
        if self.diagnostic is not None:
            return
        from .diagnostics import make_diagnostic
        self.diagnostic = make_diagnostic(
            code=self._code,
            message=self._raw_message,
            loc=self.loc,
            source=self._source,
            end_col=self._end_col,
            suggestions=self._suggestions,
            hint=self._hint,
            phase="type_checker",
        )


@dataclass
class TypeChecker:
    """
    Walks the AST and annotates each expression node with its resolved TEXType.
    Also collects @ binding references for the node to create sockets.
    """
    # Mapping of @ binding names -> their types (set before checking)
    binding_types: dict[str, TEXType] = field(default_factory=dict)

    # Source code for diagnostic snippets
    source: str = ""

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

    # Track loop nesting depth for break/continue validation
    _loop_depth: int = 0

    # User-defined function signatures: name -> {return_type, params, node}
    _user_functions: dict[str, dict] = field(default_factory=dict)

    # Return type of the function currently being checked (None if not in a function)
    _current_function_return_type: TEXType | None = None

    # Const variable scopes (parallel to _scopes) — tracks variables declared
    # with 'const' qualifier so reassignment is rejected per-scope.
    _const_scopes: list[set[str]] = field(default_factory=list)

    @property
    def inferred_out_type(self) -> TEXType | None:
        """Convenience: inferred type for @OUT binding."""
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
        self._user_functions = {}
        self._current_function_return_type = None
        self._const_scopes = [set()]

        # Pre-populate built-in variables
        builtins = {
            "ix": TEXType.FLOAT, "iy": TEXType.FLOAT,
            "iw": TEXType.FLOAT, "ih": TEXType.FLOAT,
            "u": TEXType.FLOAT, "v": TEXType.FLOAT,
            "px": TEXType.FLOAT, "py": TEXType.FLOAT,
            "fi": TEXType.FLOAT, "fn": TEXType.FLOAT,
            "PI": TEXType.FLOAT, "TAU": TEXType.FLOAT,
            "E": TEXType.FLOAT,
            "ic": TEXType.FLOAT,
        }
        self._scopes[0].update(builtins)

        for stmt in program.statements:
            self._check_stmt(stmt)

        if self.errors:
            for e in self.errors:
                e._build_diagnostic()
            if len(self.errors) == 1:
                raise self.errors[0]
            from .diagnostics import TEXMultiError
            diagnostics = [e.diagnostic for e in self.errors if e.diagnostic]
            raise TEXMultiError(diagnostics)

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
            self._error(f"Variable '{name}' is already declared in this scope.",
                       loc, code="E3001",
                       hint="Each variable can only be declared once per scope. Choose a different name, or assign to the existing one.")
            return
        self._scopes[-1][name] = t

    def _push_scope(self):
        self._scopes.append({})
        self._array_scopes.append({})
        self._const_scopes.append(set())

    def _pop_scope(self):
        self._scopes.pop()
        self._array_scopes.pop()
        self._const_scopes.pop()

    def _is_const(self, name: str) -> bool:
        """Check if a variable is declared const, respecting shadowing.

        Searches innermost scope first (matching _lookup_var).  If the
        innermost scope that *declares* the variable also marks it const,
        returns True.  A non-const redeclaration in an inner scope shadows
        an outer const.
        """
        for scope, const_set in zip(reversed(self._scopes), reversed(self._const_scopes)):
            if name in scope:
                return name in const_set
        return False

    def _lookup_array_info(self, name: str) -> TEXArrayType | None:
        for scope in reversed(self._array_scopes):
            if name in scope:
                return scope[name]
        return None

    def _error(self, msg: str, loc: SourceLoc, *, code: str = "E3000",
               hint: str = "", suggestions: list[str] | None = None):
        self.errors.append(TypeCheckError(
            msg, loc, source=self.source, code=code,
            hint=hint, suggestions=suggestions,
        ))

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
        if (a == TEXType.INT and b == TEXType.FLOAT) or (a == TEXType.FLOAT and b == TEXType.INT):
            return TEXType.FLOAT
        # scalar + vec -> vec (broadcast)
        if a.is_scalar and b.is_vector:
            return b
        if b.is_scalar and a.is_vector:
            return a
        # vec2 + vec3 -> vec3, vec2 + vec4 -> vec4, vec3 + vec4 -> vec4
        if a in _VEC_RANK and b in _VEC_RANK:
            return a if _VEC_RANK[a] >= _VEC_RANK[b] else b
        return TEXType.FLOAT  # fallback

    # -- Statement checking ---------------------------------------------

    def _check_stmt(self, node: ASTNode):
        from .ast_nodes import ErrorNode
        if isinstance(node, ErrorNode):
            return  # Skip — already reported by parser
        elif isinstance(node, VarDecl):
            self._check_var_decl(node)
        elif isinstance(node, ArrayDecl):
            self._check_array_decl(node)
        elif isinstance(node, Assignment):
            self._check_assignment(node)
        elif isinstance(node, IfElse):
            self._check_if_else(node)
        elif isinstance(node, ForLoop):
            self._check_for_loop(node)
        elif isinstance(node, WhileLoop):
            self._check_while_loop(node)
        elif isinstance(node, ExprStatement):
            self._check_expr(node.expr)
        elif isinstance(node, ParamDecl):
            self._check_param_decl(node)
        elif isinstance(node, (BreakStmt, ContinueStmt)):
            kind = "break" if isinstance(node, BreakStmt) else "continue"
            if self._loop_depth <= 0:
                self._error(f"'{kind}' statement outside of a loop.", node.loc, code="E3002",
                            hint=f"'{kind}' can only be used inside for or while loops.")
        elif isinstance(node, FunctionDef):
            self._check_function_def(node)
        elif isinstance(node, ReturnStmt):
            self._check_return_stmt(node)
        else:
            self._error(f"This statement type isn't recognized: {type(node).__name__}.", node.loc,
                        code="E4000", hint="This may be a syntax that TEX doesn't support yet.")

    def _check_var_decl(self, node: VarDecl):
        declared_type = TYPE_NAME_MAP.get(node.type_name)
        if declared_type is None:
            self._error(f"Unknown type name '{node.type_name}'.", node.loc,
                        code="E3100",
                        hint="Try: float, int, vec2, vec3, vec4, mat3, mat4, or string.")
            return

        if node.initializer:
            init_type = self._check_expr(node.initializer)
            # Check compatibility
            if not self._is_assignable(declared_type, init_type):
                self._error(
                    f"Expected '{node.type_name}' for variable '{node.name}', but the initializer is '{init_type.value}'.",
                    node.loc,
                    code="E3200",
                    hint=f"The right-hand side produces a {init_type.value}, which doesn't fit into {node.type_name}.",
                )

        self._declare_var(node.name, declared_type, node.loc)
        self._set_type(node, declared_type)
        if node.is_const:
            self._const_scopes[-1].add(node.name)

    def _check_array_decl(self, node: ArrayDecl):
        """Type-check an array declaration."""
        elem_type = TYPE_NAME_MAP.get(node.element_type_name)
        if elem_type is None:
            self._error(f"Unknown type name '{node.element_type_name}'.", node.loc,
                        code="E3100",
                        hint="Try: float, int, vec2, vec3, vec4, mat3, mat4, or string.")
            return
        if elem_type in (TEXType.VOID, TEXType.ARRAY):
            self._error(
                f"Arrays of '{node.element_type_name}' are not supported.",
                node.loc,
                code="E3101",
                hint="Array elements must be float, int, vec2, vec3, vec4, or string.",
            )
            return

        size = node.size
        if node.initializer:
            if isinstance(node.initializer, ArrayLiteral):
                init_size = len(node.initializer.elements)
                if size is not None and size != init_size:
                    self._error(
                        f"Array size mismatch: declared [{size}] but the initializer has {init_size} elements.",
                        node.loc,
                        code="E3102",
                        hint=f"Either change the size to [{init_size}], or adjust the initializer to have {size} elements.",
                    )
                if size is None:
                    size = init_size
                # Type-check each element
                for elem in node.initializer.elements:
                    et = self._check_expr(elem)
                    if not self._is_assignable(elem_type, et):
                        self._error(
                            f"Expected {node.element_type_name} element, but found {et.value}.",
                            elem.loc,
                            code="E3102",
                            hint=f"Each element in a {node.element_type_name}[] array must be compatible with {node.element_type_name}.",
                        )
                self._set_type(node.initializer, TEXType.ARRAY)
            else:
                # Array copy from another variable
                init_type = self._check_expr(node.initializer)
                if init_type != TEXType.ARRAY:
                    self._error("This array initializer needs to be an array value.",
                                node.loc, code="E3102",
                                hint="Try assigning from another array variable or an array literal {1, 2, 3}.")
                else:
                    # Check element type + size compatibility
                    if isinstance(node.initializer, Identifier):
                        src_info = self._lookup_array_info(node.initializer.name)
                        if src_info:
                            if src_info.element_type != elem_type:
                                self._error(
                                    f"Expected {elem_type.value} array, but the source is a {src_info.element_type.value} array.",
                                    node.loc,
                                    code="E3101",
                                    hint="Both arrays need to have the same element type for copying.",
                                )
                            if size is not None and size != src_info.size:
                                self._error(
                                    f"Array size mismatch: expected [{size}] but the source has [{src_info.size}] elements.",
                                    node.loc,
                                    code="E3102",
                                )
                            if size is None:
                                size = src_info.size

        if size is None:
            # Allow dynamic-size arrays from function calls (e.g. split())
            if node.initializer and isinstance(node.initializer, FunctionCall):
                size = 0  # dynamic — resolved at runtime
            else:
                self._error("This array needs a known size.",
                            node.loc, code="E3101",
                            hint="Try: float[10] arr; or float[] arr = {1, 2, 3};")
                size = 1  # fallback

        if size > 1024:  # 0 = dynamic, skip check
            self._error(f"Array size {size} exceeds the maximum of 1024.",
                        node.loc, code="E3103",
                        hint="Keep array sizes at 1024 or below.")

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
                f"'{node.name}' is used as both @wire and $parameter.",
                node.loc, code="E3202",
                hint="A name must be either @wire or $param, not both. Try renaming one of them.",
            )

        # Check default literal type (if present)
        if node.default_expr is not None:
            default_type = self._check_expr(node.default_expr)
            if not self._is_assignable(tex_type, default_type):
                self._error(
                    f"Default value type mismatch for ${node.name}: "
                    f"expected '{tex_type.value}', but found '{default_type.value}'.",
                    node.loc,
                    code="E3200",
                    hint=f"The default value should match the parameter type ({tex_type.value}).",
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
            elif isinstance(node.default_expr, VecConstructor):
                # Extract component literals for vec/color param defaults
                components = []
                for arg in node.default_expr.args:
                    if isinstance(arg, NumberLiteral):
                        components.append(float(arg.value))
                    else:
                        break
                if len(components) == len(node.default_expr.args):
                    default_value = components

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

        # Reject assignment to const variables
        if isinstance(node.target, Identifier) and self._is_const(node.target.name):
            self._error(
                f"Variable '{node.target.name}' is declared as const and cannot be reassigned.",
                node.loc, code="E3204",
                hint="Remove the 'const' qualifier if you need to modify this variable.",
            )
        elif isinstance(node.target, ChannelAccess) and isinstance(node.target.object, Identifier):
            if self._is_const(node.target.object.name):
                self._error(
                    f"Variable '{node.target.object.name}' is declared as const and cannot be modified.",
                    node.loc, code="E3204",
                    hint="Remove the 'const' qualifier if you need to modify this variable.",
                )

        # Track @binding assignments as outputs (multi-output inference)
        binding_target = self._get_binding_target(node.target)
        if binding_target is not None:
            name = binding_target.name

            # Don't allow assigning to $param bindings
            if binding_target.kind == "param":
                self._error(
                    f"Parameter ${name} is read-only and doesn't support assignment.",
                    node.loc,
                    code="E3201",
                    hint=f"Parameters are widget inputs. Use a local variable instead, e.g.: float my_{name} = ${name};",
                )
            # Don't allow @wire assignment if name was already declared as $param
            elif name in self.param_declarations:
                self._error(
                    f"'{name}' is already declared as a $parameter, so assigning to @{name} isn't allowed.",
                    node.loc,
                    code="E3202",
                    hint="A name must be either @wire or $param, not both. Try renaming one of them.",
                )
            else:
                # Reject matrix/array types as output values (not representable in ComfyUI)
                if value_type.is_matrix:
                    self._error(
                        f"Assigning {value_type.value} to @{name} isn't supported "
                        f"(matrix types are not valid as outputs).",
                        node.loc,
                        code="E3203",
                        hint="Try converting to vec3/vec4 first, or output individual components.",
                    )
                    self._set_type(node, TEXType.VOID)
                    return
                if value_type.is_array:
                    self._error(
                        f"Assigning an array to @{name} isn't supported "
                        f"(array types are not valid as outputs).",
                        node.loc,
                        code="E3203",
                        hint="Try outputting individual elements, or use join() to convert to a string.",
                    )
                    self._set_type(node, TEXType.VOID)
                    return

                # Infer output type from assignment
                if isinstance(node.target, ChannelAccess):
                    effective = self._infer_binding_from_channel(node.target)
                else:
                    effective = value_type

                # Honor explicit type prefix on the binding (e.g. m@mask, img@out).
                # This lets the user force the ComfyUI output type regardless of
                # the inferred value type:
                #   m@mask  = color_vec3  → MASK  (luminance of the vec3)
                #   img@out = float_val   → IMAGE (grayscale image from float)
                # Channel-access targets (e.g. @out.r = ...) are excluded because
                # they specify a component, not the whole output type.
                if not isinstance(node.target, ChannelAccess):
                    hint = binding_target.type_hint
                    if hint and hint in BINDING_HINT_TYPES:
                        effective = BINDING_HINT_TYPES[hint]

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
                    f"Expected an array value, but found '{value_type.value}'.",
                    node.loc,
                    code="E3200",
                    hint="The left side is an array, so the right side needs to be an array too.",
                )
            elif not target_type.is_array and value_type.is_array:
                self._error(
                    f"Expected '{target_type.value}', but found an array.",
                    node.loc,
                    code="E3200",
                    hint="Try indexing into the array (e.g. arr[0]) to get a single value.",
                )
            elif not self._is_assignable(target_type, value_type):
                self._error(
                    f"Expected '{target_type.value}', but found '{value_type.value}'.",
                    node.loc,
                    code="E3200",
                    hint=f"The target is {target_type.value}, which isn't compatible with {value_type.value}.",
                )
        else:
            self._error("This expression doesn't work as an assignment target.",
                        node.loc, code="E4000",
                        hint="The left side of '=' must be a variable, @binding, or array element.")

        self._set_type(node, TEXType.VOID)

    def _check_scalar_condition(self, cond_type: TEXType, keyword: str, loc):
        if cond_type.is_vector:
            self._error(f"This '{keyword}' condition needs a scalar expression (int or float), but found a vector.",
                        loc, code="E3500",
                        hint="Try using a single component like .r or .x, or compare with length().")

    def _check_if_else(self, node: IfElse):
        cond_type = self._check_expr(node.condition)
        self._check_scalar_condition(cond_type, "if", node.loc)

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
        self._check_scalar_condition(cond_type, "for", node.loc)

        # Check update
        self._check_stmt(node.update)

        # Check body (with loop depth tracking for break/continue)
        self._loop_depth += 1
        for stmt in node.body:
            self._check_stmt(stmt)
        self._loop_depth -= 1

        self._pop_scope()
        self._set_type(node, TEXType.VOID)

    def _check_while_loop(self, node: WhileLoop):
        self._push_scope()

        cond_type = self._check_expr(node.condition)
        self._check_scalar_condition(cond_type, "while", node.loc)

        self._loop_depth += 1
        for stmt in node.body:
            self._check_stmt(stmt)
        self._loop_depth -= 1

        self._pop_scope()
        self._set_type(node, TEXType.VOID)

    def _check_function_def(self, node: FunctionDef):
        if self._current_function_return_type is not None:
            self._error("Functions cannot be defined inside other functions.",
                        node.loc, code="E3014",
                        hint="Move this function definition to the top level.")
            return

        name = node.name
        if name in self._user_functions:
            self._error(f"Function '{name}' is already defined.", node.loc, code="E3010")
            return

        from .stdlib_signatures import FUNCTION_SIGNATURES
        if name in FUNCTION_SIGNATURES:
            self._error(f"'{name}' is a built-in function and cannot be redefined.",
                        node.loc, code="E3011",
                        hint="Choose a different name for your function.")
            return

        return_type = TYPE_NAME_MAP.get(node.return_type)
        if return_type is None:
            self._error(f"Unknown return type '{node.return_type}'.", node.loc, code="E3100",
                        hint="Try: float, int, vec2, vec3, vec4, or string.")
            return

        param_types: list[tuple[TEXType, str]] = []
        for pt, pn in node.params:
            ptype = TYPE_NAME_MAP.get(pt)
            if ptype is None:
                self._error(f"Unknown parameter type '{pt}'.", node.loc, code="E3100",
                            hint="Try: float, int, vec2, vec3, vec4, or string.")
                return
            param_types.append((ptype, pn))

        self._user_functions[name] = {
            "return_type": return_type,
            "params": param_types,
        }

        self._push_scope()
        saved_return_type = self._current_function_return_type
        self._current_function_return_type = return_type
        for ptype, pname in param_types:
            self._declare_var(pname, ptype, node.loc)
        for stmt in node.body:
            self._check_stmt(stmt)
        self._current_function_return_type = saved_return_type
        self._pop_scope()

        self._set_type(node, TEXType.VOID)

    def _check_return_stmt(self, node: ReturnStmt):
        if self._current_function_return_type is None:
            self._error("'return' can only appear inside a function body.",
                        node.loc, code="E3012",
                        hint="Assign to @OUT or another @binding instead.")
            self._set_type(node, TEXType.VOID)
            return

        value_type = self._check_expr(node.value)
        expected = self._current_function_return_type
        if not self._is_assignable(expected, value_type):
            self._error(
                f"Function expects to return '{expected.value}', but this returns '{value_type.value}'.",
                node.loc, code="E3013",
                hint=f"The declared return type is {expected.value}.",
            )
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
            self._error("Array literal '{...}' can only appear in array declarations.",
                        node.loc, code="E3900",
                        hint="Try: float[] arr = {1, 2, 3}; — array literals need a declaration.")
            self._set_type(node, TEXType.ARRAY)
            return TEXType.ARRAY

        if isinstance(node, BindingIndexAccess):
            return self._check_binding_index_access(node)

        if isinstance(node, BindingSampleAccess):
            return self._check_binding_sample_access(node)

        self._error(f"This expression type isn't recognized: {type(node).__name__}.",
                    node.loc, code="E4000",
                    hint="This may be a syntax that TEX doesn't support yet.")
        self._set_type(node, TEXType.FLOAT)
        return TEXType.FLOAT

    def _check_identifier(self, node: Identifier) -> TEXType:
        t = self._lookup_var(node.name)
        if t is None:
            from .diagnostics import suggest_similar, get_variable_hint
            # Collect all variables in scope for suggestions
            all_vars = set()
            for scope in self._scopes:
                all_vars.update(scope.keys())
            suggestions = suggest_similar(node.name, all_vars)
            hint = get_variable_hint(node.name)
            if not hint and not suggestions:
                hint = f"Check the spelling, or declare it first with a type: float {node.name} = ...;"
            self._error(f"I can't find a variable named '{node.name}'.",
                        node.loc, code="E3003", suggestions=suggestions, hint=hint)
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

    def _check_binding_index_access(self, node: BindingIndexAccess) -> TEXType:
        """Type-check @Image[ix, iy] or @Image[ix, iy, frame]."""
        self._check_expr(node.binding)
        arg_types = [self._check_expr(a) for a in node.args]
        if len(arg_types) < 2 or len(arg_types) > 3:
            self._error(
                f"@binding[...] expects 2 or 3 arguments (x, y [, frame]), got {len(arg_types)}.",
                node.loc, code="E5002",
                hint="Use @Image[ix, iy] or @Image[ix, iy, frame].",
            )
        self._set_type(node, TEXType.VEC4)
        return TEXType.VEC4

    def _check_binding_sample_access(self, node: BindingSampleAccess) -> TEXType:
        """Type-check @Image(u, v) or @Image(u, v, frame)."""
        self._check_expr(node.binding)
        arg_types = [self._check_expr(a) for a in node.args]
        if len(arg_types) < 2 or len(arg_types) > 3:
            self._error(
                f"@binding(...) expects 2 or 3 arguments (u, v [, frame]), got {len(arg_types)}.",
                node.loc, code="E5002",
                hint="Use @Image(u, v) or @Image(u, v, frame).",
            )
        self._set_type(node, TEXType.VEC4)
        return TEXType.VEC4

    def _check_channel_access(self, node: ChannelAccess) -> TEXType:
        obj_type = self._check_expr(node.object)
        channels = node.channels

        if obj_type.is_string:
            self._error("Channel access (.rgb, .x, etc.) doesn't work on strings.",
                        node.loc, code="E3300",
                        hint="Strings don't have channels. Try len() or substr() instead.")
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

        if obj_type.is_matrix:
            self._error("Channel access (.rgb, .x, etc.) doesn't work on matrix types.",
                        node.loc, code="E3300",
                        hint="Use matrix indexing or multiply by a vector instead.")
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

        if len(channels) == 1:
            # Single channel -> float
            if channels not in CHANNEL_MAP:
                self._error(f"'.{channels}' isn't a recognized channel name.",
                            node.loc, code="E3301",
                            hint="Try: .r, .g, .b, .a (color) or .x, .y, .z, .w (position).")
            if obj_type == TEXType.VEC3 and channels in ("a", "w"):
                self._error(f"vec3 doesn't have a '.{channels}' channel (only 3 components: rgb/xyz).",
                            node.loc, code="E3301",
                            hint="Use vec4 if you need a 4th channel, or access .r, .g, .b instead.")
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

        # Multi-channel swizzle
        if channels not in VALID_SWIZZLES:
            self._error(f"'.{channels}' isn't a recognized swizzle pattern.",
                        node.loc, code="E3302",
                        hint="Try common swizzles like .rgb, .xyz, .rgba, or .xyzw.")

        n = len(channels)
        if n == 2:
            self._set_type(node, TEXType.VEC2)
            return TEXType.VEC2
        elif n == 3:
            self._set_type(node, TEXType.VEC3)
            return TEXType.VEC3
        elif n == 4:
            self._set_type(node, TEXType.VEC4)
            return TEXType.VEC4
        else:
            self._error(f"Swizzle '.{channels}' has too many components ({len(channels)}).",
                        node.loc, code="E3303",
                        hint="Swizzles must be 1–4 components (e.g. .r, .xy, .rgb, .rgba).")
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
                self._error(f"Operator '{node.op}' is not supported for strings.",
                            node.loc, code="E3401",
                            hint="Strings only support + (concatenation) and == / != (comparison).")
                self._set_type(node, TEXType.STRING)
                return TEXType.STRING
            # Mixed string + numeric
            self._error(
                f"Operator '{node.op}' doesn't work between string and {(rt if lt.is_string else lt).value}.",
                node.loc,
                code="E3401",
                hint="Try: str() to convert the numeric side, e.g. str(x) + myString.",
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
            self._error(f"Unary operator '{node.op}' is not supported for strings.",
                        node.loc, code="E3401",
                        hint="Unary operators only work on numeric types (int, float, vec, mat).")
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
            self._error("Both branches of a ternary (? :) need to be the same kind: both strings or both numeric.",
                        node.loc, code="E3400",
                        hint="Make sure the true and false branches return the same general type.")
        result = self._promote(tt, ft)
        self._set_type(node, result)
        return result

    def _check_vec_constructor(self, node: VecConstructor) -> TEXType:
        arg_types: list[TEXType] = []
        for arg in node.args:
            arg_type = self._check_expr(arg)
            if arg_type.is_string:
                self._error("Vector constructors don't accept string arguments.",
                            node.loc, code="E3600",
                            hint="Use numeric values (int, float, vec) for vector construction.")
            elif arg_type.is_matrix:
                self._error("Vector constructors don't accept matrix arguments.",
                            node.loc, code="E3600",
                            hint="Use numeric values (int, float, vec) for vector construction.")
            arg_types.append(arg_type)

        if len(node.args) == 1:
            # Broadcast single value: vec4(0.5) = vec4(0.5, 0.5, 0.5, 0.5)
            # Also allows vec4(vec4_val) — identity / type cast
            if arg_types[0].is_vector and arg_types[0].channels != node.size:
                self._error(
                    f"vec{node.size}() with one argument needs a scalar or vec{node.size}, but found {arg_types[0].value}.",
                    node.loc,
                    code="E3600",
                    hint=f"Try: vec{node.size}(scalar_value) to broadcast, or pass {node.size} separate components.",
                )
        else:
            # Count total components: scalars contribute 1, vec3 → 3, vec4 → 4
            total = sum(t.channels for t in arg_types)
            if total != node.size:
                self._error(
                    f"vec{node.size}() needs exactly {node.size} components, but got {total}.",
                    node.loc,
                    code="E3601",
                    hint=f"Each scalar counts as 1, vec2 as 2, vec3 as 3, vec4 as 4. Adjust to total {node.size}.",
                )

        t = _VEC_SIZE_TYPE[node.size]
        self._set_type(node, t)
        return t

    def _check_mat_constructor(self, node: MatConstructor) -> TEXType:
        for arg in node.args:
            arg_type = self._check_expr(arg)
            if arg_type.is_string:
                self._error("Matrix constructors don't accept string arguments.",
                            node.loc, code="E3600",
                            hint="Use numeric scalar values for matrix construction.")
            elif arg_type.is_vector or arg_type.is_matrix:
                self._error("Matrix constructor arguments need to be scalar values.",
                            node.loc, code="E3600",
                            hint="Pass individual float/int values, not vectors or matrices.")

        n = node.size * node.size  # 9 for mat3, 16 for mat4
        if len(node.args) == 1:
            pass  # Broadcast: mat3(1.0) → scaled identity
        elif len(node.args) != n:
            self._error(
                f"mat{node.size}() needs {n} arguments (or 1 for scaled identity), but got {len(node.args)}.",
                node.loc,
                code="E3601",
                hint=f"A mat{node.size} is {node.size}x{node.size}, so it takes {n} values or 1 for a diagonal matrix.",
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
                    self._error(f"Multiplying {lt.value} by {rt.value} isn't supported (matrix sizes must match).",
                                loc, code="E3402")
                return lt
            # mat * vec → vec (matrix-vector product)
            if lt.is_matrix and rt.is_vector:
                if lt == TEXType.MAT3 and rt not in (TEXType.VEC3, TEXType.VEC4):
                    self._error(f"mat3 * needs a vec3 or vec4 operand, but found {rt.value}.",
                                loc, code="E3402",
                                hint="Try converting the right side to vec3 or vec4 first.")
                if lt == TEXType.MAT4 and rt != TEXType.VEC4:
                    self._error(f"mat4 * needs a vec4 operand, but found {rt.value}.",
                                loc, code="E3402",
                                hint="Try converting the right side to vec4 first.")
                return rt
            # vec * mat → error
            if lt.is_vector and rt.is_matrix:
                self._error(
                    f"Multiplying {lt.value} * {rt.value} isn't supported in this order.",
                    loc,
                    code="E3402",
                    hint=f"Try: {rt.value} * {lt.value} (matrix on the left), or transpose() the matrix first.",
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
                    self._error(f"'{op}' between {lt.value} and {rt.value} isn't supported (matrix sizes must match).",
                                loc, code="E3402")
                return lt
            # scalar +/- mat or mat +/- scalar → mat (element-wise)
            if lt.is_scalar and rt.is_matrix:
                return rt
            if lt.is_matrix and rt.is_scalar:
                return lt
            self._error(f"'{op}' between {lt.value} and {rt.value} isn't supported.",
                        loc, code="E3402",
                        hint="Matrix +/- only works with same-size matrices or scalars.")
            return lt
        else:
            self._error(f"Operator '{op}' is not supported for matrix types.",
                        loc, code="E3402",
                        hint="Matrices support *, +, and - operators.")
            return lt if lt.is_matrix else rt

    def _check_cast(self, node: CastExpr) -> TEXType:
        expr_type = self._check_expr(node.expr)
        t = TYPE_NAME_MAP.get(node.target_type, TEXType.FLOAT)
        # string(vec3/vec4) is not allowed
        if t.is_string and expr_type.is_vector:
            self._error(f"Casting {expr_type.value} to string isn't supported.",
                        node.loc, code="E3700",
                        hint="Try: str() to convert scalars, or access individual channels first.")
        # numeric casts from string are not allowed (use to_int/to_float)
        if t.is_numeric and expr_type.is_string:
            self._error(
                f"Casting string to {t.value} isn't supported directly.",
                node.loc, code="E3700",
                hint="Try: to_int() or to_float() to parse a string as a number.",
            )
        self._set_type(node, t)
        return t

    def _check_array_index(self, node: ArrayIndexAccess) -> TEXType:
        """Type-check array indexing: arr[i]"""
        array_type = self._check_expr(node.array)
        index_type = self._check_expr(node.index)

        if array_type != TEXType.ARRAY:
            self._error(f"Indexing with [] doesn't work on '{array_type.value}' (only arrays support indexing).",
                        node.loc, code="E3800",
                        hint="Make sure the variable is declared as an array, e.g. float[] arr = ...;")
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

        if not index_type.is_scalar:
            self._error(f"Array index needs to be int or float, but found '{index_type.value}'.",
                        node.loc, code="E3800",
                        hint="Use a numeric expression for the index, e.g. arr[0] or arr[i].")

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
                    f"len() doesn't work on {arg_types[0].value} (it only accepts string or array).",
                    node.loc,
                    code="E5003",
                    hint="For vectors, the component count is fixed (3 for vec3, 4 for vec4).",
                )

        # Array aggregate functions: return element type for vector arrays
        if node.name in ("arr_sum", "arr_min", "arr_max", "median", "arr_avg"):
            if arg_types and arg_types[0] == TEXType.ARRAY:
                if isinstance(node.args[0], Identifier):
                    arr_info = self._lookup_array_info(node.args[0].name)
                    if arr_info:
                        if arr_info.element_type == TEXType.STRING:
                            self._error(f"'{node.name}' doesn't work on string arrays.",
                                        node.loc, code="E5003",
                                        hint="This function only works on numeric arrays (int, float, vec).")
                        elif arr_info.element_type.is_vector:
                            result_type = arr_info.element_type
                            self._set_type(node, result_type)
                            return result_type

        # Image reduction validation
        if node.name in ("img_sum", "img_mean", "img_min", "img_max", "img_median"):
            if arg_types and not arg_types[0].is_numeric:
                self._error(f"'{node.name}' expects an image or mask input, but found {arg_types[0].value}.",
                            node.loc, code="E5003",
                            hint="Pass an @binding that contains image data (vec3/vec4) or a mask (float).")

        # join() validation
        if node.name == "join":
            if arg_types and arg_types[0] != TEXType.ARRAY:
                self._error("join() expects a string array as its first argument.",
                            node.loc, code="E5003",
                            hint="Try: join(string_array, separator).")

        # split() validation — first arg must be string
        if node.name == "split":
            if arg_types and arg_types[0] != TEXType.STRING:
                self._error("split() expects a string as its first argument.",
                            node.loc, code="E5003",
                            hint="Try: split(myString, delimiter).")

        result_type = self._resolve_function_type(node.name, arg_types, node.loc)
        self._set_type(node, result_type)
        return result_type

    def _resolve_function_type(self, name: str, arg_types: list[TEXType], loc: SourceLoc) -> TEXType:
        """Resolve the return type of a built-in function call."""
        from .stdlib_signatures import FUNCTION_SIGNATURES
        sig = FUNCTION_SIGNATURES.get(name)
        if sig is None:
            # Check user-defined functions
            user_fn = self._user_functions.get(name)
            if user_fn is not None:
                expected_count = len(user_fn["params"])
                if len(arg_types) != expected_count:
                    self._error(
                        f"'{name}()' expects {expected_count} argument{'s' if expected_count != 1 else ''}, but got {len(arg_types)}.",
                        loc, code="E5002",
                        hint=f"Check the function definition: {name}() takes {expected_count} argument{'s' if expected_count != 1 else ''}.",
                    )
                else:
                    for i, ((ptype, pname), atype) in enumerate(zip(user_fn["params"], arg_types)):
                        if not self._is_assignable(ptype, atype):
                            self._error(
                                f"Argument {i + 1} of '{name}()' expects '{ptype.value}', but got '{atype.value}'.",
                                loc, code="E5003",
                            )
                return user_fn["return_type"]

            from .diagnostics import suggest_similar, get_function_hint
            all_names = set(FUNCTION_SIGNATURES.keys()) | set(self._user_functions.keys())
            suggestions = suggest_similar(name, all_names)
            hint = get_function_hint(name)
            if not hint:
                hint = "Check the spelling, or see the ? help panel for the full function list."
            self._error(f"I can't find a function named '{name}'.",
                        loc, code="E5001", suggestions=suggestions, hint=hint)
            return TEXType.FLOAT

        # Check argument count
        min_args, max_args = sig["args"]
        if not (min_args <= len(arg_types) <= max_args):
            self._error(
                f"'{name}()' expects {min_args}-{max_args} arguments, but got {len(arg_types)}.",
                loc,
                code="E5002",
                hint=f"Check the function signature: {name}() takes {min_args} to {max_args} arguments.",
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
        # Matrix is only assignable to same matrix type (already checked target == value above)
        if target.is_matrix or value.is_matrix:
            return False
        # int -> float
        if target == TEXType.FLOAT and value == TEXType.INT:
            return True
        # scalar -> vec (broadcast)
        if target.is_vector and value.is_scalar:
            return True
        # Any vector -> any vector (promotion or truncation via swizzle)
        if target.is_vector and value.is_vector:
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
        if isinstance(target, BindingIndexAccess):
            if isinstance(target.binding, BindingRef):
                return target.binding
        return None

    def _infer_binding_from_channel(self, target: ChannelAccess) -> TEXType:
        """Infer binding type from channel assignment like @X.r = expr or @X.rgb = expr."""
        binding = target.object
        name = binding.name if isinstance(binding, BindingRef) else "OUT"
        channels = target.channels

        # .a or .w implies vec4 (alpha/w channel)
        if any(ch in ("a", "w") for ch in channels):
            return TEXType.VEC4
        # 4-component swizzle implies vec4
        if len(channels) >= 4:
            return TEXType.VEC4
        # If already inferred as VEC4, keep it
        if self.assigned_bindings.get(name) == TEXType.VEC4:
            return TEXType.VEC4
        # 3-component → vec3, 2-component → vec3 (at minimum, since we need a displayable output)
        if self.assigned_bindings.get(name) == TEXType.VEC3:
            return TEXType.VEC3
        if len(channels) >= 3:
            return TEXType.VEC3
        # Default to VEC3 for single/2-component channel assigns (.r, .xy, etc.)
        return TEXType.VEC3

    def _promote_out(self, current: TEXType, new: TEXType, loc: SourceLoc) -> TEXType:
        """Promote inferred output type, erroring on string/numeric conflict."""
        if current == new:
            return current
        if current.is_string != new.is_string:
            self._error(
                "This output binding is assigned both string and numeric types in different branches.",
                loc,
                code="E3200",
                hint="Make sure all branches assign the same kind of value (all string or all numeric).",
            )
            return current
        return self._promote(current, new)
