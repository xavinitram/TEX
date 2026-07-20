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
  vec4 -> vec3 (drops extra channels) — implicit on assignment/declaration,
                or explicit via .rgb/.xyz swizzle

Output types:
  float: scalar per-pixel value
  int: integer per-pixel value (stored as float tensor)
  vec3: 3-channel image (RGB)
  vec4: 4-channel image (RGBA)
"""
from __future__ import annotations
from dataclasses import dataclass, field
# STR-1: the type vocabulary lives in the dependency-free `types` leaf. Re-imported
# here so `type_checker`'s own logic — and any legacy `from .type_checker import
# TEXType` — keep working, but the pipeline's other modules now depend on `.types`.
from .types import (
    TEXType, TEXArrayType, TYPE_NAME_MAP, CHANNEL_MAP, VALID_SWIZZLES,
    _VEC_RANK, _VEC_SIZE_TYPE, array_wires_enabled,
)
from .ast_nodes import (
    ASTNode, Program, VarDecl, Assignment, IfElse, ForLoop, WhileLoop, ExprStatement,
    BreakStmt, ContinueStmt, FunctionDef, ReturnStmt,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor, MatConstructor,
    CastExpr, SourceLoc, ArrayDecl, ArrayIndexAccess, ArrayLiteral, ParamDecl,
    BindingIndexAccess, BindingSampleAccess, ErrorNode,
)
from .diagnostics import (
    TEXMultiError, get_builtin_var_hint, get_function_hint, get_keyword_hint,
    get_type_hint, get_variable_hint, make_diagnostic, suggest_similar,
)


# TEXType / TEXArrayType / TYPE_NAME_MAP / _VEC_RANK / _VEC_SIZE_TYPE / CHANNEL_MAP
# / VALID_SWIZZLES now live in `.types` (STR-1) and are imported at the top of this
# module. `type_checker` keeps only the checking logic below.


# STR-1 broke the old stdlib_signatures↔type_checker cycle (stdlib_signatures now
# takes its TEXType from `.types`, not from here). The table is still bound lazily
# so importing the checker alone doesn't eagerly drag in the whole signature table,
# and to stay robust to import order.
_FUNCTION_SIGNATURES: dict | None = None


def _function_signatures() -> dict:
    global _FUNCTION_SIGNATURES
    if _FUNCTION_SIGNATURES is None:
        from .stdlib_signatures import FUNCTION_SIGNATURES
        _FUNCTION_SIGNATURES = FUNCTION_SIGNATURES
    return _FUNCTION_SIGNATURES


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
    "a": TEXType.ARRAY,    # DATA-3: an ARRAY wire (curve / palette / histogram; engine profile)
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


# LANG-2: the built-in variable names pre-populated into the top scope (all FLOAT).
# A single source so the scope seed and the param-shadow warning (W7003) can't drift.
# Keep in step with interpreter._TIME_BUILTIN_NAMES for the timeline trio.
_BUILTIN_VAR_NAMES = frozenset({
    "ix", "iy", "iw", "ih", "u", "v", "px", "py", "fi", "fn",
    "PI", "TAU", "E", "ic", "frame", "fps", "time",
})
# The {name: FLOAT} top-scope seed, built once at import rather than per check().
_BUILTIN_VAR_SEED = {n: TEXType.FLOAT for n in _BUILTIN_VAR_NAMES}


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

    # When False, re-declaring a variable in the same scope overwrites the prior
    # type instead of raising E3001. Used ONLY for the internal re-type-check of
    # an already-optimized AST: loop unrolling flattens N copies of a body that
    # declares locals into one scope (benign same-type redeclarations). The
    # original (strict) check already rejected any genuine user redeclaration.
    strict_redeclare: bool = True

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

    # LANG-2: opt-in diagnostic collection for tex_api.check() — a non-raising lint pass.
    # OFF on the compile path (invariant #7: no warning bookkeeping during a normal cook).
    _collect_warnings: bool = False
    warnings: list = field(default_factory=list)            # list[TEXDiagnostic]
    _used_var_names: set = field(default_factory=set)       # local names read (W7001)
    _declared_locals: list = field(default_factory=list)    # [(name, loc)] declared (W7001)

    @property
    def inferred_out_type(self) -> TEXType | None:
        """Convenience: inferred type for @OUT binding."""
        return self.assigned_bindings.get("OUT")

    def _run(self, program: Program) -> dict[int, TEXType]:
        """Type-check WITHOUT raising: reset state, walk every statement, compute W7xxx
        warnings (if armed), and materialize a diagnostic for each error. The walk always
        completes — `_error` only appends. `check()` raises afterwards; `check_collect()`
        returns. (`_error` never raises, so there is no exception to catch here.)"""
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
        self.warnings = []
        self._used_var_names = set()
        self._declared_locals = []

        # Pre-populate built-in variables (ix/iy/u/v/…, and ENG-7's timeline trio
        # frame/fps/time — HOST time fed as builtin VALUES per cook, never $params,
        # which would churn the lazy memo + fingerprint every frame). All FLOAT.
        self._scopes[0].update(_BUILTIN_VAR_SEED)

        for stmt in program.statements:
            self._check_stmt(stmt)

        if self._collect_warnings:
            self._compute_warnings()   # W7xxx advisories (opt-in — off on the cook path)

        for e in self.errors:          # no-op on the success path (errors is empty)
            e._build_diagnostic()
        return self._types

    def check(self, program: Program) -> dict[int, TEXType]:
        """Run type checking. Returns node-id → TEXType; raises on the first (or, for
        several, the aggregated TEXMultiError) error — a contract the cook path depends on."""
        types = self._run(program)
        if self.errors:
            if len(self.errors) == 1:
                raise self.errors[0]
            raise TEXMultiError([e.diagnostic for e in self.errors if e.diagnostic])
        return types

    def _set_type(self, node: ASTNode, t: TEXType):
        self._types[id(node)] = t

    def _lookup_var(self, name: str) -> TEXType | None:
        for scope in reversed(self._scopes):
            if name in scope:
                return scope[name]
        return None

    def _declare_var(self, name: str, t: TEXType, loc: SourceLoc):
        if name in self._scopes[-1]:
            if self.strict_redeclare:
                # UX-1: a collision with a built-in (the `v` gotcha) gets a specific
                # explanation; a genuine re-declaration gets the generic one.
                builtin_hint = get_builtin_var_hint(name)
                self._error(f"Variable '{name}' is already declared in this scope.",
                           loc, code="E3001",
                           hint=builtin_hint or "Each variable can only be declared once "
                           "per scope. Choose a different name, or assign to the existing one.")
                return False   # LANG-2: a rejected redeclaration is NOT a new local decl
            # Lenient (post-optimization re-check): overwrite with the new type.
            # Unrolled loop bodies legitimately redeclare the same local.
        self._scopes[-1][name] = t
        return True

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

    # -- LANG-2: non-fatal advisories (W7xxx), collected only in check() ----

    def _warn(self, code: str, msg: str, loc: SourceLoc, *, hint: str = ""):
        """Record a W7xxx warning. No-op unless the lint pass is armed, so a normal
        cook's type-check does zero warning work (invariant #7)."""
        if not self._collect_warnings:
            return
        self.warnings.append(make_diagnostic(
            code=code, message=msg, loc=loc, source=self.source,
            hint=hint, phase="type_checker", severity="warning"))

    def _note_local_decl(self, name: str, loc: SourceLoc):
        """Register a genuine LOCAL declaration (var/array) for the unused-variable
        (W7001) and shadow (W7003) checks. Deliberately NOT called for function params
        or the timeline builtins — those are not 'unused local' candidates."""
        if not self._collect_warnings:
            return
        self._declared_locals.append((name, loc))
        # W7003 shadow: a local hiding an OUTER-scope name or a builtin (top-of-scope
        # collisions are already the E3001 error; this catches the nested-scope case).
        if any(name in s for s in self._scopes[:-1]):
            self._warn("W7003", f"'{name}' shadows a variable from an outer scope.",
                       loc, hint="Shadowing is legal but easy to misread; a distinct "
                       "name avoids the ambiguity.")

    def _compute_warnings(self):
        """Emit the whole-program W7xxx advisories after the walk (opt-in)."""
        # W7002 unused input: a wired @input the code never reads (and never writes).
        for name in self.binding_types:
            if name not in self.referenced_bindings and name not in self.assigned_bindings:
                self._warn("W7002", f"Input '@{name}' is connected but never used.",
                           SourceLoc(1, 1),
                           hint=f"Reference it with @{name}, or disconnect the wire.")
        # W7001 unused variable: a declared local that is never read.
        for name, loc in self._declared_locals:
            if name not in self._used_var_names:
                self._warn("W7001", f"Variable '{name}' is declared but never used.",
                           loc, hint="Remove it, or use it — an unused local is usually "
                           "a typo or leftover.")

    def check_collect(self, program: Program):
        """LANG-2: type-check WITHOUT raising, for tex_api.check()/live lint. Returns
        (error_diagnostics, warning_diagnostics) as two lists of TEXDiagnostic — the same
        errors check() would raise, plus the W7xxx warnings, in one non-raising pass."""
        self._collect_warnings = True
        self._run(program)
        err_diags = [e.diagnostic for e in self.errors if e.diagnostic is not None]
        return err_diags, list(self.warnings)

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
        """Dispatch a statement node to its type-checking handler."""
        handler = _STMT_HANDLERS.get(node.__class__)
        if handler is None:
            self._error(f"This statement type isn't recognized: {type(node).__name__}.", node.loc,
                        code="E4000", hint="This may be a syntax that TEX doesn't support yet.")
            return
        handler(self, node)

    def _check_error_node(self, node: ErrorNode):
        """An ErrorNode (parser recovery) type-checks as VOID; its error was already reported."""
        return  # Skip — already reported by parser

    def _check_expr_statement(self, node: ExprStatement):
        """Type-check a bare-expression statement (evaluated for its effect)."""
        self._check_expr(node.expr)

    def _check_break_continue(self, node: ASTNode):
        """Validate that break/continue appears inside a loop."""
        kind = "break" if node.__class__ is BreakStmt else "continue"
        if self._loop_depth <= 0:
            self._error(f"'{kind}' statement outside of a loop.", node.loc, code="E3002",
                        hint=f"'{kind}' can only be used inside for or while loops.")

    def _check_var_decl(self, node: VarDecl):
        """Type-check a variable declaration: the initializer must be assignable to the declared type."""
        declared_type = TYPE_NAME_MAP.get(node.type_name)
        if declared_type is None:
            hint = (get_type_hint(node.type_name)
                    or "Try: float, int, vec2, vec3, vec4, mat3, mat4, or string.")
            self._error(f"Unknown type name '{node.type_name}'.", node.loc,
                        code="E3100", hint=hint)
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

        if self._declare_var(node.name, declared_type, node.loc) and self._collect_warnings:
            self._note_local_decl(node.name, node.loc)   # LANG-2 W7001/W7003 (only if declared)
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
            size = self._check_array_initializer(node, elem_type, size)

        if size is None:
            size = self._resolve_array_size(node)

        if size > 1024:  # 0 = dynamic, skip check
            self._error(f"Array size {size} exceeds the maximum of 1024.",
                        node.loc, code="E3103",
                        hint="Keep array sizes at 1024 or below.")

        arr_type = TEXArrayType(element_type=elem_type, size=size)
        if self._declare_var(node.name, TEXType.ARRAY, node.loc) and self._collect_warnings:
            self._note_local_decl(node.name, node.loc)   # LANG-2 W7001/W7003 (only if declared)
        self._array_scopes[-1][node.name] = arr_type
        self._set_type(node, TEXType.ARRAY)
        if node.is_const:  # LX-8: forbid reassignment / element writes of a const array
            self._const_scopes[-1].add(node.name)

    def _check_array_initializer(
        self, node: ArrayDecl, elem_type: TEXType, size: int | None,
    ) -> int | None:
        """Validate an array initializer (literal or copy). Returns resolved size."""
        if isinstance(node.initializer, ArrayLiteral):
            return self._check_array_literal(node, elem_type, size)
        return self._check_array_copy(node, elem_type, size)

    def _check_array_literal(
        self, node: ArrayDecl, elem_type: TEXType, size: int | None,
    ) -> int | None:
        """Validate an array literal initializer {1, 2, 3}."""
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
        return size

    def _check_array_copy(
        self, node: ArrayDecl, elem_type: TEXType, size: int | None,
    ) -> int | None:
        """Validate an array copy initializer (from another variable)."""
        init_type = self._check_expr(node.initializer)
        if init_type != TEXType.ARRAY:
            self._error("This array initializer needs to be an array value.",
                        node.loc, code="E3102",
                        hint="Try assigning from another array variable or an array literal {1, 2, 3}.")
            return size
        if not isinstance(node.initializer, Identifier):
            return size
        src_info = self._lookup_array_info(node.initializer.name)
        if src_info is None:
            return size
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
        return size

    def _resolve_array_size(self, node: ArrayDecl) -> int:
        """Resolve size for an array without explicit size from initializer."""
        if node.initializer and isinstance(node.initializer, FunctionCall):
            return 0  # dynamic — resolved at runtime
        self._error("This array needs a known size.",
                    node.loc, code="E3101",
                    hint="Try: float arr[10]; or float arr[] = {1, 2, 3};")
        return 1  # fallback

    def _check_param_decl(self, node: ParamDecl):
        """Type-check a parameter declaration: f$strength = 0.5;"""
        # Resolve type from hint (default: FLOAT)
        tex_type = BINDING_HINT_TYPES.get(node.type_hint, TEXType.FLOAT)

        # W7003 (LANG-2): a $param sharing a name with a built-in variable is legal (the
        # $ sigil keeps `$time` and the `time` builtin distinct) but easy to misread.
        if self._collect_warnings and node.name in _BUILTIN_VAR_NAMES:
            self._warn("W7003",
                       f"Parameter '${node.name}' shares its name with the built-in "
                       f"'{node.name}'.", node.loc,
                       hint=f"They stay distinct ($'{node.name}' vs '{node.name}'), but "
                       "the shared name is easy to confuse — consider renaming the param.")

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
        """Type-check an assignment: the value must be assignable to the target's type."""
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
        elif isinstance(node.target, ArrayIndexAccess) and isinstance(node.target.array, Identifier):
            if self._is_const(node.target.array.name):   # LX-8: const-array element write
                self._error(
                    f"Array '{node.target.array.name}' is declared as const and cannot be modified.",
                    node.loc, code="E3204",
                    hint="Remove the 'const' qualifier if you need to write to this array.",
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
                # DATA-3: an array output is rejected under the ComfyUI wire (no ARRAY type),
                # but ALLOWED when a host enables array wires (the engine profile) — a curve /
                # palette / histogram flowing to another tool. The comfy EGRESS still refuses it
                # (the always-on guard), so this compile-time gate only sharpens the error.
                if value_type.is_array and not array_wires_enabled():
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
        """Require a condition expression to be a scalar (float/int) value."""
        if cond_type.is_vector:
            self._error(f"This '{keyword}' condition needs a scalar expression (int or float), but found a vector.",
                        loc, code="E3500",
                        hint="Try using a single component like .r or .x, or compare with length().")

    def _check_if_else(self, node: IfElse):
        """Type-check an if/else: a scalar condition plus both branch bodies."""
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
        """Type-check a for-loop: init / scalar condition / update plus the loop-scoped body."""
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
        """Type-check a while-loop: a scalar condition plus the body."""
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
        """Type-check a user function definition: parameters, body, and return-type consistency."""
        if self._current_function_return_type is not None:
            self._error("Functions cannot be defined inside other functions.",
                        node.loc, code="E3014",
                        hint="Move this function definition to the top level.")
            return

        name = node.name
        if name in self._user_functions:
            self._error(f"Function '{name}' is already defined.", node.loc, code="E3010")
            return

        if name in _function_signatures():
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
        """Type-check a return: the value's type must match the enclosing function's return type."""
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
        """Dispatch an expression node to its type-inference handler; returns its TEXType."""
        handler = _EXPR_HANDLERS.get(node.__class__)
        if handler is None:
            self._error(f"This expression type isn't recognized: {type(node).__name__}.",
                        node.loc, code="E4000",
                        hint="This may be a syntax that TEX doesn't support yet.")
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT
        return handler(self, node)

    def _check_number_literal(self, node: NumberLiteral) -> TEXType:
        """A numeric literal is INT when integer-valued, else FLOAT."""
        t = TEXType.INT if node.is_int else TEXType.FLOAT
        self._set_type(node, t)
        return t

    def _check_string_literal(self, node: StringLiteral) -> TEXType:
        """A string literal has type STRING."""
        self._set_type(node, TEXType.STRING)
        return TEXType.STRING

    def _check_array_literal_expr(self, node: ArrayLiteral) -> TEXType:
        # Standalone array literals only appear in declarations (handled there)
        """Infer an array literal's element type and fixed size from its elements."""
        self._error("Array literal '{...}' can only appear in array declarations.",
                    node.loc, code="E3900",
                    hint="Try: float arr[] = {1, 2, 3}; — array literals need a declaration.")
        self._set_type(node, TEXType.ARRAY)
        return TEXType.ARRAY

    def _check_identifier(self, node: Identifier) -> TEXType:
        """Resolve an identifier to its declared variable/builtin type in the current scope."""
        t = self._lookup_var(node.name)
        if self._collect_warnings and t is not None:
            self._used_var_names.add(node.name)   # LANG-2: mark the local as read (W7001)
        if t is None:
            # Collect all variables in scope for suggestions
            all_vars = set()
            for scope in self._scopes:
                all_vars.update(scope.keys())
            suggestions = suggest_similar(node.name, all_vars)
            # Prefer a foreign-language hint (e.g. `true`/`let`/`float3` written
            # where a value was expected) over the generic spelling tip.
            hint = (get_variable_hint(node.name)
                    or get_keyword_hint(node.name)
                    or get_type_hint(node.name))
            if not hint and not suggestions:
                hint = f"Check the spelling, or declare it first with a type: float {node.name} = ...;"
            self._error(f"I can't find a variable named '{node.name}'.",
                        node.loc, code="E3003", suggestions=suggestions, hint=hint)
            t = TEXType.FLOAT
        self._set_type(node, t)
        return t

    def _check_binding_ref(self, node: BindingRef) -> TEXType:
        """Resolve an @/$ binding reference to its input/parameter type."""
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
        binding_type = self._check_expr(node.binding)
        arg_types = [self._check_expr(a) for a in node.args]
        if len(arg_types) < 2 or len(arg_types) > 3:
            self._error(
                f"@binding[...] expects 2 or 3 arguments (x, y [, frame]), got {len(arg_types)}.",
                node.loc, code="E5002",
                hint="Use @Image[ix, iy] or @Image[ix, iy, frame].",
            )
        # Return the binding's actual type (VEC3 for IMAGE, FLOAT for MASK, etc.)
        ret = binding_type if binding_type != TEXType.STRING else TEXType.VEC4
        self._set_type(node, ret)
        return ret

    def _check_binding_sample_access(self, node: BindingSampleAccess) -> TEXType:
        """Type-check @Image(u, v) or @Image(u, v, frame)."""
        binding_type = self._check_expr(node.binding)
        arg_types = [self._check_expr(a) for a in node.args]
        if len(arg_types) < 2 or len(arg_types) > 3:
            self._error(
                f"@binding(...) expects 2 or 3 arguments (u, v [, frame]), got {len(arg_types)}.",
                node.loc, code="E5002",
                hint="Use @Image(u, v) or @Image(u, v, frame).",
            )
        ret = binding_type if binding_type != TEXType.STRING else TEXType.VEC4
        self._set_type(node, ret)
        return ret

    def _check_channel_access(self, node: ChannelAccess) -> TEXType:
        """Type-check a swizzle/channel access (.rgb/.xy…): validate channels against the base width."""
        obj_type = self._check_expr(node.object)
        channels = node.channels

        # Types with NO addressable channels reject a swizzle up front (E3300). An ARRAY base
        # (an `a@name` wire, DATA-3) matters most: the bounds checks below gate only vectors/
        # scalars, so a swizzle here would fall through, claim a VECn from the swizzle width, and
        # the interpreter would slice an [...,N,C]/[...,N] array by a channel index — silently
        # wrong. (Swizzling a reduction's SCALAR result — `arr_avg(a@pal).r` — is FLOAT, not an
        # array, so it never reaches this guard.)
        for has_no_channels, noun, hint in (
            (obj_type.is_string, "strings", "Strings don't have channels. Try len() or substr() instead."),
            (obj_type.is_matrix, "matrix types", "Use matrix indexing or multiply by a vector instead."),
            (obj_type.is_array, "an array", "Index the array with arr[i], then swizzle the element."),
        ):
            if has_no_channels:
                self._error(f"Channel access (.rgb, .x, etc.) doesn't work on {noun}.",
                            node.loc, code="E3300", hint=hint)
                self._set_type(node, TEXType.FLOAT)
                return TEXType.FLOAT

        if len(channels) == 1:
            # Single channel -> float
            if channels not in CHANNEL_MAP:
                self._error(f"'.{channels}' isn't a recognized channel name.",
                            node.loc, code="E3301",
                            hint="Try: .r, .g, .b, .a (color) or .x, .y, .z, .w (position).")
            # A scalar (FLOAT/INT) has 1 component (.channels == 1), so this bounds check must
            # cover it too — else `.g`/`.b`/`.a` on a scalar was silently accepted and the
            # interpreter mis-sliced the spatial axis. `.r`/`.x` (index 0) stays valid on a
            # scalar: it's a reduction's own channel (e.g. `arr_avg(a@pal).r`) and, being a
            # count-preserving no-op, isn't the wrong-channel-count bug this closes. (A genuine
            # spatial-scalar `.r` mis-slice is a narrower pre-existing gap not fixable here
            # without also rejecting that legitimate reduction pattern.)
            elif (obj_type.is_vector or obj_type.is_scalar) and CHANNEL_MAP[channels] >= obj_type.channels:
                self._error(
                    f"{obj_type.value} doesn't have a '.{channels}' channel "
                    f"(only {obj_type.channels} components).",
                    node.loc, code="E3301",
                    hint="Access a channel within range, or use a wider vector type.")
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT

        # Multi-channel swizzle
        if channels not in VALID_SWIZZLES:
            self._error(f"'.{channels}' isn't a recognized swizzle pattern.",
                        node.loc, code="E3302",
                        hint="Try common swizzles like .rgb, .xyz, .rgba, or .xyzw.")
        elif (obj_type.is_vector or obj_type.is_scalar) and any(
            CHANNEL_MAP[ch] >= obj_type.channels for ch in channels if ch in CHANNEL_MAP
        ):
            # A scalar base (C=1) has no channel past .r/.x, so a multi-swizzle like `.rgb` on a
            # FLOAT must error here — the old is_vector-only gate let it fall through and silently
            # claim vec3 (the interpreter then sliced pixels, not channels). Covers the scalar
            # exactly as the single-channel branch above.
            self._error(
                f"Swizzle '.{channels}' references channels {obj_type.value} doesn't have "
                f"(only {obj_type.channels} components).",
                node.loc, code="E3301",
                hint="Only swizzle channels that exist on the source vector type.")

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
        """Type-check a binary operation: promote the operands and compute the result type."""
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

        if node.op in ("&&", "||", "==", "!=", "<", ">", "<=", ">="):
            # Logical and comparison ops evaluate element-wise on both
            # backends, so a vector operand yields a per-component vector
            # (0.0/1.0 values), not a scalar. Mirror that here so the checker
            # agrees with runtime shape.
            result = self._promote(lt, rt) if (lt.is_vector or rt.is_vector) else TEXType.FLOAT
            self._set_type(node, result)
            return result

        # Arithmetic: promote
        result = self._promote(lt, rt)
        self._set_type(node, result)
        return result

    def _check_unary(self, node: UnaryOp) -> TEXType:
        """Type-check a unary operation (-/!): the operand must be numeric/scalar."""
        t = self._check_expr(node.operand)
        if t.is_string:
            self._error(f"Unary operator '{node.op}' is not supported for strings.",
                        node.loc, code="E3401",
                        hint="Unary operators only work on numeric types (int, float, vec, mat).")
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT
        if node.op == "!":
            self._set_type(node, TEXType.FLOAT)
            return TEXType.FLOAT
        # Negation preserves type
        self._set_type(node, t)
        return t

    def _check_ternary(self, node: TernaryOp) -> TEXType:
        """Type-check a ternary: a scalar condition plus the promoted common type of both arms."""
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
        """Type-check a vecN constructor: the component widths must sum to N."""
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
        """Type-check a matN constructor (scalar-diagonal or full component list)."""
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
            # mat * vec → vec (matrix-vector product).
            #   mat3 * vec3 -> vec3;  mat3 * vec4 -> vec4 (transforms xyz, keeps w)
            #   mat4 * vec4 -> vec4;  mat4 * vec3 -> vec4 (vec3 treated as a point, w=1)
            if lt.is_matrix and rt.is_vector:
                if lt == TEXType.MAT3:
                    if rt not in (TEXType.VEC3, TEXType.VEC4):
                        self._error(f"mat3 * needs a vec3 or vec4 operand, but found {rt.value}.",
                                    loc, code="E3402",
                                    hint="A mat3 transforms vec3 values; with a vec4 it transforms xyz and keeps w.")
                        return TEXType.VEC3
                    return rt  # vec3 -> vec3, vec4 -> vec4 (xyz transformed, w preserved)
                # mat4
                if rt not in (TEXType.VEC3, TEXType.VEC4):
                    self._error(f"mat4 * needs a vec3 or vec4 operand, but found {rt.value}.",
                                loc, code="E3402",
                                hint="A mat4 transforms vec4 values; a vec3 is treated as a point (w = 1).")
                return TEXType.VEC4  # vec3 promoted to a point, vec4 transformed
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
            # Any other matrix operand combination (e.g. mat * array) is
            # unsupported — report it cleanly instead of falling through to an
            # implicit None return, which later crashes with AttributeError
            # ('NoneType' has no attribute 'is_string') in _is_assignable.
            self._error(
                f"'*' between {lt.value} and {rt.value} isn't supported.",
                loc, code="E3402",
                hint="Matrix '*' needs a matrix, vector, or scalar operand.",
            )
            return lt if lt.is_matrix else rt
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
        """Type-check an explicit cast to a numeric/vector target type."""
        expr_type = self._check_expr(node.expr)
        t = TYPE_NAME_MAP.get(node.target_type, TEXType.FLOAT)
        # string(vec3/vec4/mat3/mat4) is not allowed — the str conversion code
        # is written for scalars only.
        if t.is_string and (expr_type.is_vector or expr_type.is_matrix):
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
                        hint="Make sure the variable is declared as an array, e.g. float arr[] = ...;")
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
        """Type-check a stdlib/user call: argument count plus the signature's return-type rule."""
        arg_types = [self._check_expr(arg) for arg in node.args]

        # Validate len() argument type: only STRING and ARRAY allowed
        if node.name == "len" and arg_types:
            if not (arg_types[0].is_string or arg_types[0] == TEXType.ARRAY):
                self._error(
                    f"len() doesn't work on {arg_types[0].value} (it only accepts string or array).",
                    node.loc,
                    code="E5003",
                    hint="Vectors have a fixed component count (3 for vec3, 4 for vec4); scalars have no length.",
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

        # cross() requires 3-component vectors (vec3, or vec4 which is trimmed
        # to xyz). vec2 / scalar args crash torch.cross at runtime, so fail fast
        # at compile time — this covers both the interpreter and codegen paths.
        if node.name == "cross":
            for i, at in enumerate(arg_types):
                if at not in (TEXType.VEC3, TEXType.VEC4):
                    self._error(
                        f"cross() argument {i + 1} must be a vec3 (or vec4), but got {at.value}.",
                        node.loc, code="E5003",
                        hint="The cross product is only defined for 3-component vectors.")
                    break

        # Matrix/color built-ins crash on the wrong operand types at runtime
        # (raw torch internals). Validate at compile time, in cross()'s voice.
        if node.name in ("determinant", "inverse", "transpose") and arg_types:
            if not arg_types[0].is_matrix:
                self._error(
                    f"{node.name}() needs a matrix (mat3 or mat4), but got {arg_types[0].value}.",
                    node.loc, code="E5003",
                    hint="Build a matrix first with mat3(...) or mat4(...).")
        elif node.name in ("hsv2rgb", "rgb2hsv") and arg_types:
            if not arg_types[0].is_vector:
                self._error(
                    f"{node.name}() needs a color vector (vec3 or vec4), but got {arg_types[0].value}.",
                    node.loc, code="E5003",
                    hint="Pass an RGB/HSV color, e.g. vec3(r, g, b).")
        elif node.name == "dot":
            for i, at in enumerate(arg_types):
                if not at.is_vector:
                    self._error(
                        f"dot() needs vectors, but argument {i + 1} is {at.value}.",
                        node.loc, code="E5003",
                        hint="Both arguments should be vec2, vec3, or vec4.")
                    break
        elif node.name in ("length", "normalize", "distance", "reflect"):
            # These reduce/operate over the channel axis, so a scalar argument
            # would reduce over a non-existent dimension at runtime. Fail fast
            # at compile time, matching the dot()/cross() pattern.
            for i, at in enumerate(arg_types):
                if not at.is_vector:
                    self._error(
                        f"{node.name}() needs a vector, but argument {i + 1} is {at.value}.",
                        node.loc, code="E5003",
                        hint="Pass a vec2, vec3, or vec4.")
                    break

        result_type = self._resolve_function_type(node.name, arg_types, node.loc)
        self._set_type(node, result_type)
        return result_type

    def _resolve_function_type(self, name: str, arg_types: list[TEXType], loc: SourceLoc) -> TEXType:
        """Resolve the return type of a built-in function call."""
        FUNCTION_SIGNATURES = _function_signatures()
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
                                hint=f"Parameter '{pname}' is declared as {ptype.value}. "
                                     f"Convert with to_float()/to_int()/str(), or a vec constructor.",
                            )
                return user_fn["return_type"]

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


# Dispatch tables for _check_stmt/_check_expr, built once at import (the
# checker is instantiated several times per compile, so per-instance tables
# would be rebuilt each time). Keyed on the exact node class — every AST node
# subclasses ASTNode directly, and a future subclass needs its own entry here
# (the interpreter's dispatch dicts impose the same constraint). Handlers are
# unbound methods, called as handler(self, node); classes missing from a table
# fall through to the E4000 diagnostic.
_STMT_HANDLERS: dict[type, object] = {
    ErrorNode: TypeChecker._check_error_node,
    VarDecl: TypeChecker._check_var_decl,
    ArrayDecl: TypeChecker._check_array_decl,
    Assignment: TypeChecker._check_assignment,
    IfElse: TypeChecker._check_if_else,
    ForLoop: TypeChecker._check_for_loop,
    WhileLoop: TypeChecker._check_while_loop,
    ExprStatement: TypeChecker._check_expr_statement,
    ParamDecl: TypeChecker._check_param_decl,
    BreakStmt: TypeChecker._check_break_continue,
    ContinueStmt: TypeChecker._check_break_continue,
    FunctionDef: TypeChecker._check_function_def,
    ReturnStmt: TypeChecker._check_return_stmt,
}

_EXPR_HANDLERS: dict[type, object] = {
    NumberLiteral: TypeChecker._check_number_literal,
    StringLiteral: TypeChecker._check_string_literal,
    Identifier: TypeChecker._check_identifier,
    BindingRef: TypeChecker._check_binding_ref,
    ChannelAccess: TypeChecker._check_channel_access,
    BinOp: TypeChecker._check_binop,
    UnaryOp: TypeChecker._check_unary,
    TernaryOp: TypeChecker._check_ternary,
    FunctionCall: TypeChecker._check_function_call,
    VecConstructor: TypeChecker._check_vec_constructor,
    MatConstructor: TypeChecker._check_mat_constructor,
    CastExpr: TypeChecker._check_cast,
    ArrayIndexAccess: TypeChecker._check_array_index,
    ArrayLiteral: TypeChecker._check_array_literal_expr,
    BindingIndexAccess: TypeChecker._check_binding_index_access,
    BindingSampleAccess: TypeChecker._check_binding_sample_access,
}
