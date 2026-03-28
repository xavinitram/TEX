"""
TEX Codegen — compile TEX AST to Python functions for zero-overhead execution.

Instead of tree-walking the AST on every frame, this module generates a Python
function string, compiles it via exec(), and caches the callable. Subsequent
executions call the function directly, eliminating:
  - Per-node dispatch table lookups
  - Per-node Python function call overhead
  - Redundant dict lookups for env/bindings

Falls back to None (caller uses tree-walking interpreter) for unsupported
patterns: string operations, spatial if/else with complex merging.

Break/continue handling: the codegen uses exception-based control flow
(matching the interpreter) because Python's native 'continue' would skip
the for-loop update statement. _CgBreak/_CgContinue are raised in generated
code and caught by the generated loop wrapper.
"""
from __future__ import annotations

from typing import Any

from ..tex_compiler.ast_nodes import (
    ASTNode, Program, VarDecl, Assignment, IfElse, ForLoop, WhileLoop,
    ExprStatement, BreakStmt, ContinueStmt, ParamDecl, ArrayDecl,
    FunctionDef, ReturnStmt,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor,
    MatConstructor, CastExpr, ArrayIndexAccess, ArrayLiteral,
    BindingIndexAccess, BindingSampleAccess,
    try_extract_static_range,
)
from ..tex_compiler.type_checker import TEXType, CHANNEL_MAP, TYPE_NAME_MAP
from .interpreter import MAX_CALL_DEPTH


class _Unsupported(Exception):
    """Raised when codegen encounters a pattern it cannot handle."""
    pass


class _CgBreak(Exception):
    """Signal for break in codegen-generated code."""
    pass


class _CgContinue(Exception):
    """Signal for continue in codegen-generated code."""
    pass


# Operators that map directly to Python infix ops
_INFIX_OPS = {"+", "-", "*"}

# Comparison operators
_CMP_OPS = {
    "==": "==", "!=": "!=", "<": "<", ">": ">", "<=": "<=", ">=": ">=",
}


def try_compile(program: Program, type_map: dict[int, TEXType]) -> Any | None:
    """Try to compile a TEX program AST to a Python function.

    Returns a callable with signature:
        fn(env, bindings, functions, device, spatial_shape) -> None

    The function mutates env and bindings dicts in-place.
    Returns None if the program uses unsupported features.
    """
    try:
        gen = _CodeGen(type_map)
        gen.emit_program(program)
        return gen.build()
    except _Unsupported:
        return None
    except Exception:
        return None


# Stdlib functions that are simple torch.XXX(arg) wrappers.
# In codegen, arguments are always tensors so _to_tensor is a no-op.
# We emit _torch.XXX(arg) directly, avoiding dict lookup + function call + _to_tensor.
_INLINE_TORCH_1ARG: dict[str, str] = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "asin": "asin", "acos": "acos", "atan": "atan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "exp": "exp", "abs": "abs",
    "floor": "floor", "ceil": "ceil", "round": "round", "trunc": "trunc",
    "sign": "sign",
    "degrees": "rad2deg", "radians": "deg2rad",
}

# 2-arg torch functions: fn(a, b) -> torch.XXX(a, b)
_INLINE_TORCH_2ARG: dict[str, str] = {
    "max": "maximum", "min": "minimum",
    "atan2": "atan2", "hypot": "hypot",
}


def _body_has_break_continue(stmts: list[ASTNode]) -> bool:
    """Check if a list of statements contains break or continue (not in nested loops)."""
    for stmt in stmts:
        if isinstance(stmt, (BreakStmt, ContinueStmt)):
            return True
        if isinstance(stmt, IfElse):
            if _body_has_break_continue(stmt.then_body):
                return True
            if stmt.else_body and _body_has_break_continue(stmt.else_body):
                return True
        # Don't recurse into nested loops — their break/continue is local to them
    return False


class _CodeGen:
    """Generates Python source code from a TEX AST."""

    def __init__(self, type_map: dict[int, TEXType]):
        self.type_map = type_map
        self._tmp_counter = 0
        self._lines: list[str] = []
        self._preamble: list[str] = []  # hoisted constant assignments
        self._indent = 1  # Start at 1 (inside function body)
        self._const_cache: dict[float, str] = {}  # value → variable name
        self._vec_const_cache: dict[tuple, str] = {}  # (v1, v2, ...) → variable name
        self._range_cache: dict[tuple, str] = {}  # (start, stop, step) → variable name
        # Local variable mapping: TEX var name → Python local var name.
        # When set, Identifier emission uses locals instead of _env[name].
        self._local_vars: dict[str, str] = {}
        # User-defined function names (populated during FunctionDef emission)
        self._user_functions: set[str] = set()
        # Track if we're inside a user function body (for return statements)
        self._in_user_function: bool = False

    def _tmp(self) -> str:
        """Generate a unique temporary variable name."""
        self._tmp_counter += 1
        return f"_t{self._tmp_counter}"

    def _emit(self, line: str):
        """Emit a line of code at the current indentation level."""
        self._lines.append("    " * self._indent + line)

    def _emit_spatial_branch(self, spatial_code: str, non_spatial_code: str):
        """Emit an if _sp: / else: block with the given code for each branch."""
        self._emit("if _sp:")
        self._indent += 1
        self._emit(spatial_code)
        self._indent -= 1
        self._emit("else:")
        self._indent += 1
        self._emit(non_spatial_code)
        self._indent -= 1

    def _get_const(self, value: float) -> str:
        """Return a variable name for a numeric constant tensor.

        Constants are hoisted to the function preamble (before any loops),
        so each unique value is created once per execution regardless of
        how many times or where it appears in the source.
        """
        cached = self._const_cache.get(value)
        if cached is not None:
            return cached
        var = self._tmp()
        self._preamble.append(f"    {var} = _torch.scalar_tensor({value!r}, dtype=_torch.float32, device=_dev)")
        self._const_cache[value] = var
        return var

    def emit_program(self, program: Program):
        """Emit code for the entire program.

        All env variables are pre-registered as Python locals to avoid
        _env dict lookups on every read/write. This produces cleaner code
        for TorchInductor (no dict guard overhead) and is faster in eager
        mode too (local variable access is faster than dict access).
        """
        # Register all program-level env vars as locals
        all_env_vars, _ = self._collect_modified_vars(program.statements)
        for vname in sorted(all_env_vars):
            self._local_vars[vname] = f"_lv_{vname}"

        # Initialize locals from _env (for builtins like u, v, ix, iy that
        # the caller pre-populates in the env dict)
        for vname in sorted(all_env_vars):
            local = self._local_vars[vname]
            self._preamble.append(f"    {local} = _env.get({vname!r})")

        for stmt in program.statements:
            self._emit_stmt(stmt)

        # Write back all locals to _env so the caller can read results
        for vname in sorted(all_env_vars):
            local = self._local_vars[vname]
            self._emit(f"_env[{vname!r}] = {local}")

    def build(self) -> Any:
        """Compile the generated source to a callable function."""
        preamble = "\n".join(self._preamble)
        body = "\n".join(self._lines)
        # Hoisted constants go before the main body so they're available
        # inside loops without per-iteration tensor creation.
        func_body = f"{preamble}\n{body}" if preamble else body
        func_src = (
            "def _tex_fn(_env, _bind, _fns, _dev, _sp, "
            "_torch, _bp, _es, _tw, _math, _SAFE_EPS, _CMAP, _MAX_ITER, "
            "_CgBreak, _CgContinue):\n"
            f"{func_body}\n"
        )
        namespace: dict[str, Any] = {}
        code_obj = compile(func_src, "<tex_codegen>", "exec")
        exec(code_obj, namespace)
        return namespace["_tex_fn"]

    # ── Statement emission ──────────────────────────────────────────────

    def _emit_stmt(self, stmt: ASTNode):
        if isinstance(stmt, VarDecl):
            self._emit_var_decl(stmt)
        elif isinstance(stmt, Assignment):
            self._emit_assignment(stmt)
        elif isinstance(stmt, ExprStatement):
            expr = self._emit_expr(stmt.expr)
            self._emit(f"{expr}")
        elif isinstance(stmt, ParamDecl):
            pass  # no-op at runtime
        elif isinstance(stmt, IfElse):
            self._emit_if_else(stmt)
        elif isinstance(stmt, ForLoop):
            self._emit_for_loop(stmt)
        elif isinstance(stmt, WhileLoop):
            self._emit_while_loop(stmt)
        elif isinstance(stmt, (BreakStmt, ContinueStmt)):
            # Use exception-based control flow (matching interpreter semantics).
            # Python's native 'continue' would skip the for-loop update, so we
            # use _CgBreak/_CgContinue caught by the generated loop wrapper.
            if isinstance(stmt, BreakStmt):
                self._emit("raise _CgBreak()")
            else:
                self._emit("raise _CgContinue()")
        elif isinstance(stmt, ArrayDecl):
            self._emit_array_decl(stmt)
        elif isinstance(stmt, FunctionDef):
            self._emit_function_def(stmt)
        elif isinstance(stmt, ReturnStmt):
            self._emit_return_stmt(stmt)
        else:
            raise _Unsupported(f"Unknown statement: {type(stmt).__name__}")

    def _var_target(self, name: str) -> str:
        """Return the Python expression for writing to a TEX variable."""
        local = self._local_vars.get(name)
        return local if local is not None else f"_env[{name!r}]"

    def _emit_var_decl(self, stmt: VarDecl):
        if stmt.initializer:
            expr = self._emit_expr(stmt.initializer)
            self._emit(f"{self._var_target(stmt.name)} = {expr}")
        else:
            # Default initialization based on type
            tgt = self._var_target(stmt.name)
            declared = self.type_map.get(id(stmt), TEXType.FLOAT)
            if declared == TEXType.STRING:
                self._emit(f"{tgt} = ''")
            elif declared.is_vector:
                ch = declared.channels
                self._emit_spatial_branch(
                    f"{tgt} = _torch.zeros(*_sp, {ch}, dtype=_torch.float32, device=_dev)",
                    f"{tgt} = _torch.zeros({ch}, dtype=_torch.float32, device=_dev)",
                )
            elif declared.is_matrix:
                n = declared.mat_size
                self._emit(f"{tgt} = _torch.zeros({n}, {n}, dtype=_torch.float32, device=_dev)")
            else:
                self._emit(f"{tgt} = _torch.scalar_tensor(0.0, dtype=_torch.float32, device=_dev)")

    def _emit_array_decl(self, stmt: ArrayDecl):
        """Emit array declaration matching interpreter's tensor layout."""
        tgt = self._var_target(stmt.name)
        elem_type = stmt.element_type_name
        tex_type = TYPE_NAME_MAP.get(elem_type)
        is_vec = tex_type is not None and tex_type.is_vector
        is_string = elem_type == "string"

        if is_string:
            if stmt.initializer and isinstance(stmt.initializer, ArrayLiteral):
                elems = [self._emit_expr(e) for e in stmt.initializer.elements]
                self._emit(f"{tgt} = [{', '.join(elems)}]")
            elif stmt.initializer:
                src = self._emit_expr(stmt.initializer)
                self._emit(f"{tgt} = list({src}) if isinstance({src}, list) else [str({src})]")
            else:
                self._emit(f"{tgt} = [''] * {stmt.size}")
            return

        if stmt.initializer and isinstance(stmt.initializer, ArrayLiteral):
            elems = [self._emit_expr(e) for e in stmt.initializer.elements]
            es_list = ", ".join(f"_es({e}, _sp)" for e in elems)
            ns_list = ", ".join(elems)
            suffix = ".transpose(-2, -1)" if is_vec else ""
            self._emit_spatial_branch(
                f"{tgt} = _torch.stack([{es_list}], dim=-1){suffix}",
                f"{tgt} = _torch.stack([{ns_list}], dim=-1){suffix}",
            )
        elif stmt.initializer:
            src = self._emit_expr(stmt.initializer)
            self._emit(f"{tgt} = {src}.clone()")
        else:
            size = stmt.size
            if is_vec:
                channels = TYPE_NAME_MAP[elem_type].channels
                self._emit_spatial_branch(
                    f"{tgt} = _torch.zeros(*_sp, {size}, {channels}, dtype=_torch.float32, device=_dev)",
                    f"{tgt} = _torch.zeros({size}, {channels}, dtype=_torch.float32, device=_dev)",
                )
            else:
                self._emit_spatial_branch(
                    f"{tgt} = _torch.zeros(*_sp, {size}, dtype=_torch.float32, device=_dev)",
                    f"{tgt} = _torch.zeros({size}, dtype=_torch.float32, device=_dev)",
                )

    def _emit_array_index_read(self, node: ArrayIndexAccess) -> str:
        """Emit array[index] read with clamped bounds."""
        arr = self._emit_expr(node.array)
        idx = self._emit_expr(node.index)
        tmp = self._tmp()

        # String arrays are Python lists
        self._emit(f"if isinstance({arr}, list):")
        self._indent += 1
        self._emit(f"{tmp} = {arr}[max(0, min(int(round({idx}.item() if _torch.is_tensor({idx}) else float({idx}))), len({arr}) - 1))]")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        # Tensor arrays — check dim for vec (dim 2 or 5) vs scalar (dim 1 or 4)
        self._emit(f"if {arr}.dim() in (2, 5):")
        self._indent += 1
        self._emit(f"_ai = _torch.clamp(_torch.floor({idx}).long(), 0, {arr}.shape[-2] - 1)")
        self._emit(f"if _ai.dim() == 0:")
        self._indent += 1
        self._emit(f"{tmp} = {arr}[..., _ai.item(), :]")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_C = {arr}.shape[-1]")
        self._emit(f"_idx_exp = _ai.unsqueeze(-1).unsqueeze(-1).expand(*_ai.shape, 1, _C)")
        self._emit(f"if _idx_exp.shape[:3] != {arr}.shape[:3]:")
        self._indent += 1
        self._emit(f"_idx_exp = _idx_exp.expand({arr}.shape[:3] + (1, _C))")
        self._indent -= 1
        self._emit(f"{tmp} = _torch.gather({arr}, dim=-2, index=_idx_exp).squeeze(-2)")
        self._indent -= 1
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_ai = _torch.clamp(_torch.floor({idx}).long(), 0, {arr}.shape[-1] - 1)")
        self._emit(f"if _ai.dim() == 0:")
        self._indent += 1
        self._emit(f"{tmp} = {arr}[..., _ai.item()]")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_idx_exp = _ai.unsqueeze(-1)")
        self._emit(f"if _idx_exp.shape[:3] != {arr}.shape[:3]:")
        self._indent += 1
        self._emit(f"_idx_exp = _idx_exp.expand({arr}.shape[:3] + (1,))")
        self._indent -= 1
        self._emit(f"{tmp} = _torch.gather({arr}, dim=-1, index=_idx_exp).squeeze(-1)")
        self._indent -= 1
        self._indent -= 1
        self._indent -= 1

        return tmp

    def _emit_array_index_assign(self, target: ArrayIndexAccess, value_expr: str):
        """Emit array[index] = value with clamped bounds."""
        arr = self._emit_expr(target.array)
        idx = self._emit_expr(target.index)

        # String arrays
        self._emit(f"if isinstance({arr}, list):")
        self._indent += 1
        self._emit(f"_al = list({arr})")
        self._emit(f"_al[max(0, min(int(round({idx}.item() if _torch.is_tensor({idx}) else float({idx}))), len(_al) - 1))] = {value_expr} if isinstance({value_expr}, str) else str({value_expr})")
        if isinstance(target.array, Identifier):
            self._emit(f"{self._var_target(target.array.name)} = _al")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        # Clone for copy-on-write semantics
        self._emit(f"_arr_c = {arr}.clone()")
        self._emit(f"if _arr_c.dim() in (2, 5):")
        self._indent += 1
        self._emit(f"_ai = _torch.clamp(_torch.floor({idx}).long(), 0, _arr_c.shape[-2] - 1)")
        self._emit(f"if _ai.dim() == 0:")
        self._indent += 1
        self._emit(f"_arr_c[..., _ai.item(), :] = _es({value_expr}, _arr_c.shape[:-2]) if _arr_c.dim() > 2 else {value_expr}")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_C = _arr_c.shape[-1]")
        self._emit(f"_idx_exp = _ai.unsqueeze(-1).unsqueeze(-1).expand(*_ai.shape, 1, _C)")
        self._emit(f"if _idx_exp.shape[:3] != _arr_c.shape[:3]:")
        self._indent += 1
        self._emit(f"_idx_exp = _idx_exp.expand(_arr_c.shape[:3] + (1, _C))")
        self._indent -= 1
        self._emit(f"_val_exp = _es({value_expr}, _arr_c.shape[:-2]).unsqueeze(-2)")
        self._emit(f"_arr_c.scatter_(-2, _idx_exp, _val_exp)")
        self._indent -= 1
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_ai = _torch.clamp(_torch.floor({idx}).long(), 0, _arr_c.shape[-1] - 1)")
        self._emit(f"if _ai.dim() == 0:")
        self._indent += 1
        self._emit(f"_arr_c[..., _ai.item()] = _es({value_expr}, _arr_c.shape[:-1]) if _arr_c.dim() > 1 else {value_expr}")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_idx_exp = _ai.unsqueeze(-1)")
        self._emit(f"if _idx_exp.shape[:3] != _arr_c.shape[:3]:")
        self._indent += 1
        self._emit(f"_idx_exp = _idx_exp.expand(_arr_c.shape[:3] + (1,))")
        self._indent -= 1
        self._emit(f"_val_exp = _es({value_expr}, _arr_c.shape[:-1]).unsqueeze(-1)")
        self._emit(f"_arr_c.scatter_(-1, _idx_exp, _val_exp)")
        self._indent -= 1
        self._indent -= 1
        if isinstance(target.array, Identifier):
            self._emit(f"{self._var_target(target.array.name)} = _arr_c")
        self._indent -= 1

    def _emit_array_literal(self, node: ArrayLiteral) -> str:
        """Emit array literal {a, b, c} as a tensor stack."""
        elems = [self._emit_expr(e) for e in node.elements]
        tmp = self._tmp()
        es_list = ", ".join(f"_es({e}, _sp) if _sp else {e}" for e in elems)
        self._emit(f"{tmp} = _torch.stack([{es_list}], dim=-1)")
        return tmp

    def _emit_scatter_write(self, target: BindingIndexAccess, value_expr: str, op=None):
        """Emit @OUT[px, py] = value scatter write."""
        name = target.binding.name
        arg_exprs = [self._emit_expr(a) for a in target.args]
        px, py = arg_exprs[0], arg_exprs[1]
        has_frame = len(arg_exprs) == 3

        # Get or create output buffer
        self._emit(f"if {name!r} not in _bind or not _torch.is_tensor(_bind[{name!r}]):")
        self._indent += 1
        self._emit(f"_sv = {value_expr}")
        self._emit(f"_sc = _sv.shape[-1] if _torch.is_tensor(_sv) and _sv.dim() >= 1 and _sv.shape[-1] in (2,3,4) else 1")
        self._emit(f"if _sp:")
        self._indent += 1
        self._emit(f"_bind[{name!r}] = _torch.zeros(*_sp, _sc, dtype=_torch.float32, device=_dev) if _sc > 1 else _torch.zeros(*_sp, dtype=_torch.float32, device=_dev)")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_bind[{name!r}] = _torch.zeros(1, 1, 1, _sc, dtype=_torch.float32, device=_dev) if _sc > 1 else _torch.zeros(1, 1, 1, dtype=_torch.float32, device=_dev)")
        self._indent -= 1
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_sv = {value_expr}")
        self._indent -= 1

        self._emit(f"_sb = _bind[{name!r}]")
        self._emit(f"_sB, _sH, _sW = _sb.shape[0], _sb.shape[1], _sb.shape[2]")

        # Clamp coordinates
        self._emit(f"_six = _torch.clamp(_torch.floor(_es({px}, (_sB,_sH,_sW)) if _sp else {px}).long(), 0, _sW - 1)")
        self._emit(f"_siy = _torch.clamp(_torch.floor(_es({py}, (_sB,_sH,_sW)) if _sp else {py}).long(), 0, _sH - 1)")
        self._emit(f"_sval = _es(_sv, (_sB,_sH,_sW)) if _sp else _sv")

        if has_frame:
            self._emit(f"_sbi = _torch.clamp((_es({arg_exprs[2]}, (_sB,_sH,_sW)) if _sp else {arg_exprs[2]}).long(), 0, _sB - 1)")
        else:
            self._emit(f"_sbi = _torch.arange(_sB, device=_dev).view(_sB, 1, 1).expand(_sB, _sH, _sW)")

        self._emit(f"_fb = _sbi.reshape(-1)")
        self._emit(f"_fy = _siy.reshape(-1)")
        self._emit(f"_fx = _six.reshape(-1)")

        # Reshape value to match buffer layout
        self._emit(f"if _sb.dim() == 4:")
        self._indent += 1
        self._emit(f"_fv = _sval.reshape(-1, _sb.shape[-1])")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"_fv = _sval.reshape(-1) if _torch.is_tensor(_sval) and _sval.dim() > 0 else _sval")
        self._indent -= 1

        # Apply scatter operation
        if op is None:
            self._emit(f"_sb[_fb, _fy, _fx] = _fv")
        elif op == "+":
            self._emit(f"_sb.index_put_((_fb, _fy, _fx), _fv, accumulate=True)")
        elif op == "-":
            self._emit(f"_sb.index_put_((_fb, _fy, _fx), -_fv, accumulate=True)")
        elif op == "*":
            self._emit(f"_sb[_fb, _fy, _fx] = _sb[_fb, _fy, _fx] * _fv")
        elif op == "/":
            self._emit(f"_sb[_fb, _fy, _fx] = _sb[_fb, _fy, _fx] / _tw(_fv == 0, _SAFE_EPS, _fv)")

    def _emit_function_def(self, stmt: FunctionDef):
        """Emit a user-defined function as a nested Python def."""
        self._user_functions.add(stmt.name)
        params = [f"_p_{pname}" for _, pname in stmt.params]
        params_str = ", ".join(params + ["_depth=0"])
        self._emit(f"def _uf_{stmt.name}({params_str}):")
        self._indent += 1
        self._emit(f"if _depth > {MAX_CALL_DEPTH}: raise RuntimeError('Maximum function call depth exceeded in {stmt.name}()')")

        # Save and swap local vars context for function body
        saved_locals = self._local_vars
        self._local_vars = {}
        # Register params as locals
        for _, pname in stmt.params:
            self._local_vars[pname] = f"_p_{pname}"

        # Collect all vars declared anywhere in function body (including nested blocks)
        body_vars, _ = self._collect_modified_vars(stmt.body)
        for vname in body_vars:
            if vname not in self._local_vars:  # don't overwrite params
                self._local_vars[vname] = f"_uf_lv_{vname}"

        saved_in_fn = self._in_user_function
        self._in_user_function = True

        for s in stmt.body:
            self._emit_stmt(s)

        # Default return if no explicit return
        self._emit(f"return _torch.scalar_tensor(0.0, dtype=_torch.float32, device=_dev)")
        self._in_user_function = saved_in_fn
        self._local_vars = saved_locals
        self._indent -= 1

    def _emit_return_stmt(self, stmt: ReturnStmt):
        """Emit a return statement inside a user function."""
        value = self._emit_expr(stmt.value)
        self._emit(f"return {value}")

    def _emit_assignment(self, stmt: Assignment):
        target = stmt.target
        value_expr = self._emit_expr(stmt.value)
        if isinstance(target, Identifier):
            self._emit(f"{self._var_target(target.name)} = {value_expr}")
        elif isinstance(target, BindingRef):
            self._emit(f"_bind[{target.name!r}] = {value_expr}")
        elif isinstance(target, ChannelAccess):
            self._emit_channel_assign(target, value_expr)
        elif isinstance(target, ArrayIndexAccess):
            self._emit_array_index_assign(target, value_expr)
        elif isinstance(target, BindingIndexAccess):
            self._emit_scatter_write(target, value_expr, op=stmt.op)
        else:
            raise _Unsupported(f"Unsupported assignment target: {type(target).__name__}")

    def _emit_channel_assign(self, target: ChannelAccess, value_expr: str):
        channels = target.channels
        base_expr = self._emit_expr(target.object)
        tmp = self._tmp()
        self._emit(f"{tmp} = {base_expr}.clone()")

        if len(channels) == 1:
            idx = CHANNEL_MAP.get(channels)
            if idx is None:
                raise _Unsupported(f"Invalid channel: {channels}")
            self._emit(f"{tmp}[..., {idx}] = _es({value_expr}, {tmp}.shape[:-1])")
        else:
            # Multi-channel assignment: .rgb, .xy, .rgba, etc.
            indices = [CHANNEL_MAP.get(ch) for ch in channels]
            if any(i is None for i in indices):
                raise _Unsupported(f"Invalid swizzle: {channels}")
            val_tmp = self._tmp()
            self._emit(f"{val_tmp} = _es({value_expr}, {tmp}.shape[:-1])")
            multi_flag = self._tmp()
            self._emit(f"{multi_flag} = {val_tmp}.dim() >= 1 and {val_tmp}.shape[-1] > 1")
            for i, idx in enumerate(indices):
                self._emit(f"{tmp}[..., {idx}] = {val_tmp}[..., {i}] if {multi_flag} else {val_tmp}")

        if isinstance(target.object, Identifier):
            self._emit(f"{self._var_target(target.object.name)} = {tmp}")
        elif isinstance(target.object, BindingRef):
            self._emit(f"_bind[{target.object.name!r}] = {tmp}")
        else:
            raise _Unsupported("Channel assign on non-variable target")

    def _emit_if_else(self, stmt: IfElse):
        """Emit if/else.

        Scalar conditions use Python if/else (supports break/continue).
        Spatial conditions use vectorized both-branch with torch.where merging,
        same logic as the interpreter's selective cloning.
        """
        cond_expr = self._emit_expr(stmt.condition)
        cond_tmp = self._tmp()
        self._emit(f"{cond_tmp} = {cond_expr}")

        # Check if this could be a scalar condition
        # We emit both paths: scalar short-circuit and spatial vectorized
        self._emit(f"if {cond_tmp}.dim() == 0:")
        self._indent += 1
        self._emit(f"if {cond_tmp}.item() > 0.5:")
        self._indent += 1
        for s in stmt.then_body:
            self._emit_stmt(s)
        self._indent -= 1
        if stmt.else_body:
            self._emit(f"else:")
            self._indent += 1
            for s in stmt.else_body:
                self._emit_stmt(s)
            self._indent -= 1
        self._indent -= 1

        # Spatial path: this is complex (selective cloning + torch.where merge)
        # Fall back to unsupported to let the interpreter handle it
        # UNLESS there's no else body and the then body is simple
        self._emit(f"else:")
        self._indent += 1

        # For spatial if/else, we need the full clone/merge logic.
        # Emit it inline.
        self._emit_spatial_if_else(stmt, cond_tmp)

        self._indent -= 1

    def _emit_spatial_if_else(self, stmt: IfElse, cond_var: str):
        """Emit spatial if/else with selective cloning and torch.where merge.

        Uses local variables for env vars when available (program-level locals),
        falls back to _env dict for any vars not in _local_vars.
        """
        # Collect modified variables (same analysis as interpreter)
        then_mods = self._collect_modified_vars(stmt.then_body)
        else_mods = self._collect_modified_vars(stmt.else_body) if stmt.else_body else (set(), set())
        all_env_mods = then_mods[0] | else_mods[0]
        all_bind_mods = then_mods[1] | else_mods[1]

        if not all_env_mods and not all_bind_mods:
            # Nothing modified — just execute both branches for side effects
            for s in stmt.then_body:
                self._emit_stmt(s)
            if stmt.else_body:
                for s in stmt.else_body:
                    self._emit_stmt(s)
            return

        # Snapshot modified env vars (using locals where available)
        # Some vars may be None if only declared inside one branch (not yet assigned)
        snap_vars: dict[str, str] = {}  # env_var_name -> snapshot_tmp_name
        for k in all_env_mods:
            snap = self._tmp()
            snap_vars[k] = snap
            src = self._var_target(k)
            self._emit(f"{snap} = {src}.clone() if ({src} is not None and _torch.is_tensor({src})) else {src}")

        snap_bind = self._tmp()
        bind_mod_repr = repr(all_bind_mods)
        self._emit(f"{snap_bind} = {{k: _bind[k].clone() if _torch.is_tensor(_bind[k]) else _bind[k] for k in {bind_mod_repr} if k in _bind}}")

        # Execute then-branch
        for s in stmt.then_body:
            self._emit_stmt(s)

        # Capture then-state
        then_vars: dict[str, str] = {}
        for k in all_env_mods:
            then_tmp = self._tmp()
            then_vars[k] = then_tmp
            src = self._var_target(k)
            self._emit(f"{then_tmp} = {src}")

        then_bind = self._tmp()
        self._emit(f"{then_bind} = {{k: _bind.get(k) for k in {bind_mod_repr}}}")

        # Restore snapshot
        for k in all_env_mods:
            tgt = self._var_target(k)
            self._emit(f"{tgt} = {snap_vars[k]}")
        self._emit(f"_bind.update({snap_bind})")

        if stmt.else_body:
            for s in stmt.else_body:
                self._emit_stmt(s)

        # Merge with torch.where
        cond_bool = self._tmp()
        self._emit(f"{cond_bool} = ({cond_var} > 0.5)")

        for k in all_env_mods:
            tv = then_vars[k]
            tgt = self._var_target(k)
            ev = self._tmp()
            self._emit(f"{ev} = {tgt}")
            self._emit(f"if {tv} is not None and {ev} is not None and _torch.is_tensor({tv}) and _torch.is_tensor({ev}):")
            self._indent += 1
            self._emit(f"{tgt} = _tw({cond_bool}, {tv}, {ev})")
            self._indent -= 1
            self._emit(f"elif {tv} is not None:")
            self._indent += 1
            self._emit(f"{tgt} = {tv}")
            self._indent -= 1

        for k in all_bind_mods:
            tv = self._tmp()
            ev = self._tmp()
            self._emit(f"{tv} = {then_bind}.get({k!r})")
            self._emit(f"{ev} = _bind.get({k!r})")
            self._emit(f"if {tv} is not None and {ev} is not None and _torch.is_tensor({tv}) and _torch.is_tensor({ev}):")
            self._indent += 1
            self._emit(f"_bind[{k!r}] = _tw({cond_bool}, {tv}, {ev})")
            self._indent -= 1
            self._emit(f"elif {tv} is not None:")
            self._indent += 1
            self._emit(f"_bind[{k!r}] = {tv}")
            self._indent -= 1

    def _collect_modified_vars(self, stmts: list[ASTNode]) -> tuple[set[str], set[str]]:
        """Collect env var names and binding names assigned in a statement list."""
        env_vars: set[str] = set()
        bind_names: set[str] = set()
        for stmt in stmts:
            if isinstance(stmt, Assignment):
                t = stmt.target
                if isinstance(t, Identifier):
                    env_vars.add(t.name)
                elif isinstance(t, BindingRef):
                    bind_names.add(t.name)
                elif isinstance(t, ChannelAccess):
                    o = t.object
                    if isinstance(o, Identifier):
                        env_vars.add(o.name)
                    elif isinstance(o, BindingRef):
                        bind_names.add(o.name)
                elif isinstance(t, ArrayIndexAccess):
                    if isinstance(t.array, Identifier):
                        env_vars.add(t.array.name)
                elif isinstance(t, BindingIndexAccess):
                    if isinstance(t.binding, BindingRef):
                        bind_names.add(t.binding.name)
            elif isinstance(stmt, VarDecl):
                env_vars.add(stmt.name)
            elif isinstance(stmt, ArrayDecl):
                env_vars.add(stmt.name)
            elif isinstance(stmt, IfElse):
                e1, b1 = self._collect_modified_vars(stmt.then_body)
                if stmt.else_body:
                    e2, b2 = self._collect_modified_vars(stmt.else_body)
                else:
                    e2, b2 = set(), set()
                env_vars |= e1 | e2
                bind_names |= b1 | b2
            elif isinstance(stmt, (ForLoop, WhileLoop)):
                body = stmt.body
                e, b = self._collect_modified_vars(body)
                env_vars |= e
                bind_names |= b
        return env_vars, bind_names

    def _collect_read_vars(self, stmts: list[ASTNode]) -> set[str]:
        """Collect all Identifier names read in a list of statements."""
        names: set[str] = set()
        for stmt in stmts:
            self._collect_reads_node(stmt, names)
        return names

    def _collect_reads_node(self, node: ASTNode, names: set[str]):
        """Recursively collect Identifier names read in an AST node."""
        if isinstance(node, Identifier):
            names.add(node.name)
        elif isinstance(node, BinOp):
            self._collect_reads_node(node.left, names)
            self._collect_reads_node(node.right, names)
        elif isinstance(node, UnaryOp):
            self._collect_reads_node(node.operand, names)
        elif isinstance(node, TernaryOp):
            self._collect_reads_node(node.condition, names)
            self._collect_reads_node(node.true_expr, names)
            self._collect_reads_node(node.false_expr, names)
        elif isinstance(node, FunctionCall):
            for a in node.args:
                self._collect_reads_node(a, names)
        elif isinstance(node, VecConstructor):
            for a in node.args:
                self._collect_reads_node(a, names)
        elif isinstance(node, CastExpr):
            self._collect_reads_node(node.expr, names)
        elif isinstance(node, ChannelAccess):
            self._collect_reads_node(node.object, names)
        elif isinstance(node, Assignment):
            self._collect_reads_node(node.value, names)
            # Also collect reads from target (e.g., channel access on identifier)
            if isinstance(node.target, ChannelAccess):
                self._collect_reads_node(node.target.object, names)
        elif isinstance(node, VarDecl):
            if node.initializer:
                self._collect_reads_node(node.initializer, names)
        elif isinstance(node, IfElse):
            self._collect_reads_node(node.condition, names)
            for s in node.then_body:
                self._collect_reads_node(s, names)
            if node.else_body:
                for s in node.else_body:
                    self._collect_reads_node(s, names)
        elif isinstance(node, ExprStatement):
            self._collect_reads_node(node.expr, names)
        elif isinstance(node, ForLoop):
            # Recurse into nested loops so outer scope can localize all vars
            for s in node.body:
                self._collect_reads_node(s, names)
        elif isinstance(node, WhileLoop):
            self._collect_reads_node(node.condition, names)
            for s in node.body:
                self._collect_reads_node(s, names)

    def _emit_body_with_flow(self, body: list[ASTNode]):
        """Emit loop body wrapped in try/except for break/continue signals."""
        self._emit("try:")
        self._indent += 1
        for s in body:
            self._emit_stmt(s)
        self._indent -= 1
        self._emit("except _CgBreak:")
        self._indent += 1
        self._emit("break")
        self._indent -= 1
        self._emit("except _CgContinue:")
        self._indent += 1
        self._emit("pass")
        self._indent -= 1

    def _emit_cond_break(self, cond_node: ASTNode):
        """Emit scalar/spatial condition check that breaks if false."""
        cond_expr = self._emit_expr(cond_node)
        cc = self._tmp()
        self._emit(f"{cc} = {cond_expr}")
        self._emit(f"if {cc}.dim() == 0:")
        self._indent += 1
        self._emit(f"if {cc}.item() <= 0.5: break")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"if ({cc} > 0.5).float().sum().item() == 0: break")
        self._indent -= 1

    def _emit_iter_limit(self, iter_var: str, label: str):
        """Emit post-loop iteration limit check."""
        self._emit(f"if {iter_var} >= _MAX_ITER:")
        self._indent += 1
        self._emit(f"raise RuntimeError('{label} loop exceeded maximum iteration limit (' + str(_MAX_ITER) + '). Check your loop condition.')")
        self._indent -= 1

    def _emit_for_loop(self, stmt: ForLoop):
        """Emit a for loop, optimized for static ranges."""
        # Try fully static loop: pre-compute Python range() — zero per-iteration overhead
        static_range = try_extract_static_range(stmt)
        if static_range is not None:
            loop_var, start, stop, step = static_range
            n = abs(stop - start) // max(abs(step), 1)
            if n > 1024:
                self._emit(f"raise RuntimeError('For loop would exceed {1024} iterations')")
                return

            has_flow_control = _body_has_break_continue(stmt.body)

            # Collect vars modified AND read in the loop body so we can use
            # local Python variables instead of _env[name] dict lookups.
            modified_vars, _ = self._collect_modified_vars(stmt.body)
            read_vars = self._collect_read_vars(stmt.body)
            # All vars touched in the loop (modified + read-only + loop var)
            all_vars = modified_vars | read_vars
            all_vars.add(loop_var)
            # Vars that need writeback after the loop
            writeback_vars = modified_vars | {loop_var}

            # Register local variables for all vars used in this loop.
            # If a var is already a local (from an outer loop), reuse its name.
            saved_locals = {}
            for vname in all_vars:
                prev = self._local_vars.get(vname)
                saved_locals[vname] = prev
                if prev is None:
                    self._local_vars[vname] = f"_lv_{vname}"
                # else: keep the outer loop's local — already a Python local

            # Initialize local vars from _env ONLY for vars not already locals
            loop_var_local = self._local_vars[loop_var]
            for vname in all_vars:
                if vname != loop_var and saved_locals[vname] is None:
                    local = self._local_vars[vname]
                    self._emit(f"{local} = _env.get({vname!r})")

            # Pre-allocate loop variable tensors via arange + unbind.
            # Hoisted to preamble so nested loops don't re-create each outer iteration.
            range_key = (start, stop, step)
            vals_tmp = self._range_cache.get(range_key)
            if vals_tmp is None:
                vals_tmp = self._tmp()
                self._preamble.append(
                    f"    {vals_tmp} = _torch.arange({start}, {stop}, {step}, dtype=_torch.float32, device=_dev).unbind(0)"
                )
                self._range_cache[range_key] = vals_tmp
            self._emit(f"for _i_idx in range({n}):")
            self._indent += 1
            self._emit(f"{loop_var_local} = {vals_tmp}[_i_idx]")

            if has_flow_control:
                self._emit_body_with_flow(stmt.body)
            else:
                for s in stmt.body:
                    self._emit_stmt(s)

            self._indent -= 1

            # Write back modified vars to _env ONLY if they weren't already locals
            # from an outer scope (outer locals are still accessible directly).
            for vname in writeback_vars:
                if saved_locals[vname] is None:
                    local = self._local_vars[vname]
                    self._emit(f"_env[{vname!r}] = {local}")

            # Restore previous local var mappings
            for vname, prev in saved_locals.items():
                if prev is None:
                    if vname in self._local_vars:
                        del self._local_vars[vname]
                else:
                    self._local_vars[vname] = prev
            return

        # General case: emit initializer + while loop
        self._emit_stmt(stmt.init)

        static = self._try_static_bound(stmt)
        iter_var = self._tmp()

        self._emit(f"{iter_var} = 0")
        self._emit(f"while {iter_var} < _MAX_ITER:")
        self._indent += 1

        # Condition check
        if static is not None:
            loop_var, bound, is_le = static
            cv = self._tmp()
            self._emit(f"{cv} = {self._var_target(loop_var)}")
            self._emit(f"if {cv}.dim() == 0:")
            self._indent += 1
            ci = self._tmp()
            self._emit(f"{ci} = int({cv}.item())")
            if is_le:
                self._emit(f"if {ci} > {bound}: break")
            else:
                self._emit(f"if {ci} >= {bound}: break")
            self._indent -= 1
            self._emit(f"else:")
            self._indent += 1
            cond_expr = self._emit_expr(stmt.condition)
            self._emit(f"if ({cond_expr} > 0.5).float().sum().item() == 0: break")
            self._indent -= 1
        else:
            self._emit_cond_break(stmt.condition)

        self._emit_body_with_flow(stmt.body)

        # Update always executes (even after continue)
        self._emit_stmt(stmt.update)
        self._emit(f"{iter_var} += 1")
        self._indent -= 1

        self._emit_iter_limit(iter_var, "For")

    def _try_static_bound(self, stmt: ForLoop) -> tuple[str, int, bool] | None:
        """Try to extract static loop bounds."""
        cond = stmt.condition
        if not isinstance(cond, BinOp):
            return None
        if cond.op not in ("<", "<="):
            return None
        if not isinstance(cond.left, Identifier):
            return None
        if not isinstance(cond.right, NumberLiteral):
            return None
        return (cond.left.name, int(cond.right.value), cond.op == "<=")

    def _emit_while_loop(self, stmt: WhileLoop):
        """Emit a while loop with exception-based break/continue."""
        iter_var = self._tmp()
        self._emit(f"{iter_var} = 0")
        self._emit(f"while {iter_var} < _MAX_ITER:")
        self._indent += 1

        self._emit_cond_break(stmt.condition)
        self._emit_body_with_flow(stmt.body)

        self._emit(f"{iter_var} += 1")
        self._indent -= 1

        self._emit_iter_limit(iter_var, "While")

    # ── Expression emission ─────────────────────────────────────────────

    def _emit_expr(self, node: ASTNode) -> str:
        """Emit an expression, returning a Python expression string.

        For complex expressions, emits assignment to a temp var and
        returns the temp var name. For simple leaves, returns inline.
        """
        if isinstance(node, NumberLiteral):
            return self._get_const(node.value)

        if isinstance(node, StringLiteral):
            return repr(node.value)

        if isinstance(node, Identifier):
            local = self._local_vars.get(node.name)
            if local is not None:
                return local
            return f"_env[{node.name!r}]"

        if isinstance(node, BindingRef):
            return f"_bind[{node.name!r}]"

        if isinstance(node, ChannelAccess):
            return self._emit_channel_access(node)

        if isinstance(node, BinOp):
            return self._emit_binop(node)

        if isinstance(node, UnaryOp):
            return self._emit_unary(node)

        if isinstance(node, TernaryOp):
            return self._emit_ternary(node)

        if isinstance(node, FunctionCall):
            return self._emit_function_call(node)

        if isinstance(node, VecConstructor):
            return self._emit_vec_constructor(node)

        if isinstance(node, MatConstructor):
            return self._emit_mat_constructor(node)

        if isinstance(node, CastExpr):
            return self._emit_cast(node)

        if isinstance(node, ArrayIndexAccess):
            return self._emit_array_index_read(node)

        if isinstance(node, ArrayLiteral):
            return self._emit_array_literal(node)

        if isinstance(node, BindingIndexAccess):
            binding = self._emit_expr(node.binding)
            args = [self._emit_expr(a) for a in node.args]
            tmp = self._tmp()
            if len(args) == 3:
                self._emit(f"{tmp} = _fns['fetch_frame']({binding}, {args[2]}, {args[0]}, {args[1]})")
            else:
                self._emit(f"{tmp} = _fns['fetch']({binding}, {args[0]}, {args[1]})")
            return tmp

        if isinstance(node, BindingSampleAccess):
            binding = self._emit_expr(node.binding)
            args = [self._emit_expr(a) for a in node.args]
            tmp = self._tmp()
            if len(args) == 3:
                self._emit(f"{tmp} = _fns['sample_frame']({binding}, {args[2]}, {args[0]}, {args[1]})")
            else:
                self._emit(f"{tmp} = _fns['sample']({binding}, {args[0]}, {args[1]})")
            return tmp

        raise _Unsupported(f"Unknown expression: {type(node).__name__}")

    def _emit_channel_access(self, node: ChannelAccess) -> str:
        base = self._emit_expr(node.object)
        channels = node.channels

        if len(channels) == 1:
            idx = CHANNEL_MAP.get(channels)
            if idx is None:
                raise _Unsupported(f"Invalid channel: {channels}")
            return f"{base}[..., {idx}]"

        # Multi-channel swizzle
        indices = [CHANNEL_MAP.get(ch) for ch in channels]
        if any(i is None for i in indices):
            raise _Unsupported(f"Invalid swizzle: {channels}")

        # Contiguous slice optimization: .rgb (0,1,2) or .rg (0,1) → [..., :N]
        # This is a zero-copy view instead of 3 index ops + torch.stack.
        if indices == list(range(indices[0], indices[0] + len(indices))):
            start = indices[0]
            end = start + len(indices)
            if start == 0:
                return f"{base}[..., :{end}]"
            else:
                return f"{base}[..., {start}:{end}]"

        parts = ", ".join(f"{base}[..., {i}]" for i in indices)
        tmp = self._tmp()
        self._emit(f"{tmp} = _torch.stack([{parts}], dim=-1)")
        return tmp

    def _emit_binop(self, node: BinOp) -> str:
        left = self._emit_expr(node.left)
        right = self._emit_expr(node.right)

        # String concatenation check — fall back
        # (can't easily detect at codegen time without type info on values)

        # Matrix operations
        lt = self.type_map.get(id(node.left))
        rt = self.type_map.get(id(node.right))
        if node.op == "*" and lt is not None and rt is not None:
            if lt.is_matrix and rt.is_matrix:
                tmp = self._tmp()
                self._emit(f"{tmp} = _torch.matmul({left}, {right})")
                return tmp
            if lt.is_matrix and rt.is_vector:
                tmp = self._tmp()
                self._emit(f"{tmp} = _torch.matmul({left}, {right}.unsqueeze(-1)).squeeze(-1)")
                return tmp

        # Check if broadcasting is needed (scalar vs vector mixed)
        needs_broadcast = False
        needs_channel_pad = False
        if lt is not None and rt is not None:
            l_vec = lt.is_vector
            r_vec = rt.is_vector
            l_scalar = not l_vec and not lt.is_matrix
            r_scalar = not r_vec and not rt.is_matrix
            if (l_vec and r_scalar) or (l_scalar and r_vec):
                needs_broadcast = True
            elif l_vec and r_vec and lt.channels != rt.channels:
                needs_channel_pad = True

        if needs_broadcast:
            tl = self._tmp()
            tr = self._tmp()
            self._emit(f"{tl}, {tr} = _bp({left}, {right})")
            left, right = tl, tr
        elif needs_channel_pad:
            lc, rc = lt.channels, rt.channels
            if lc < rc:
                tl = self._tmp()
                self._emit(f"{tl} = _torch.nn.functional.pad({left}, (0, {rc - lc}))")
                left = tl
            else:
                tr = self._tmp()
                self._emit(f"{tr} = _torch.nn.functional.pad({right}, (0, {lc - rc}))")
                right = tr

        op = node.op
        tmp = self._tmp()

        if op in _INFIX_OPS:
            self._emit(f"{tmp} = ({left} {op} {right})")
        elif op == "/":
            self._emit(f"{tmp} = ({left} / _tw({right} == 0, _SAFE_EPS, {right}))")
        elif op == "%":
            self._emit(f"{tmp} = _torch.fmod({left}, _tw({right} == 0, _SAFE_EPS, {right}))")
        elif op in _CMP_OPS:
            py_op = _CMP_OPS[op]
            self._emit(f"{tmp} = ({left} {py_op} {right}).float()")
        elif op == "&&":
            self._emit(f"{tmp} = (({left} > 0.5) & ({right} > 0.5)).float()")
        elif op == "||":
            self._emit(f"{tmp} = (({left} > 0.5) | ({right} > 0.5)).float()")
        else:
            raise _Unsupported(f"Unknown operator: {op}")

        return tmp

    def _emit_unary(self, node: UnaryOp) -> str:
        operand = self._emit_expr(node.operand)
        tmp = self._tmp()
        if node.op == "-":
            self._emit(f"{tmp} = (-{operand})")
        elif node.op == "!":
            self._emit(f"{tmp} = ({operand} <= 0.5).float()")
        else:
            raise _Unsupported(f"Unknown unary op: {node.op}")
        return tmp

    def _emit_ternary(self, node: TernaryOp) -> str:
        cond = self._emit_expr(node.condition)
        true_val = self._emit_expr(node.true_expr)
        false_val = self._emit_expr(node.false_expr)
        tmp = self._tmp()

        # Handle string ternary
        self._emit(f"if isinstance({true_val}, str) or isinstance({false_val}, str):")
        self._indent += 1
        self._emit(f"_cs = {cond}.float().mean().item() if _torch.is_tensor({cond}) and {cond}.dim() > 0 else (float({cond}.item()) if _torch.is_tensor({cond}) else float({cond}))")
        self._emit(f"{tmp} = {true_val} if _cs > 0.5 else {false_val}")
        self._indent -= 1
        self._emit(f"else:")
        self._indent += 1
        self._emit(f"{tmp} = _tw({cond} > 0.5, {true_val}, {false_val})")
        self._indent -= 1

        return tmp

    def _emit_function_call(self, node: FunctionCall) -> str:
        name = node.name
        args = [self._emit_expr(a) for a in node.args]
        tmp = self._tmp()

        # User-defined function call
        if name in self._user_functions:
            args_str = ", ".join(args)
            depth_arg = ", _depth=_depth+1" if self._in_user_function else ""
            self._emit(f"{tmp} = _uf_{name}({args_str}{depth_arg})")
            return tmp

        # Try to inline simple torch functions directly (avoids dict lookup +
        # wrapper function call + _to_tensor overhead per call).
        if len(args) == 1:
            torch_fn = _INLINE_TORCH_1ARG.get(name)
            if torch_fn is not None:
                self._emit(f"{tmp} = _torch.{torch_fn}({args[0]})")
                return tmp
        elif len(args) == 2:
            # pow(base, exp): use exp-log trick (~3x faster on spatial tensors)
            if name == "pow":
                self._emit(f"{tmp} = _torch.exp(_torch.log({args[0]}.clamp(min=_SAFE_EPS)) * {args[1]})")
                return tmp
            torch_fn = _INLINE_TORCH_2ARG.get(name)
            if torch_fn is not None:
                self._emit(f"{tmp} = _torch.{torch_fn}({args[0]}, {args[1]})")
                return tmp

        # Inline lerp: torch.lerp(a, b, t) — very common in TEX
        if name == "lerp" and len(args) == 3:
            self._emit(f"{tmp} = _torch.lerp({args[0]}, {args[1]}, {args[2]})")
            return tmp

        # Inline clamp: torch.clamp(x, min, max)
        if name == "clamp" and len(args) == 3:
            self._emit(f"{tmp} = _torch.clamp({args[0]}, {args[1]}, {args[2]})")
            return tmp

        # Inline dot: einsum is 4× faster than mul+sum on CPU
        if name == "dot" and len(args) == 2:
            self._emit(f"{tmp} = _torch.einsum('...c,...c->...', {args[0]}, {args[1]})")
            return tmp

        # Inline distance: linalg.vector_norm(a - b, dim=-1)
        if name == "distance" and len(args) == 2:
            self._emit(f"{tmp} = _torch.linalg.vector_norm({args[0]} - {args[1]}, dim=-1)")
            return tmp

        # Inline normalize: a / (norm(a) + eps)
        if name == "normalize" and len(args) == 1:
            self._emit(f"{tmp} = {args[0]} / (_torch.linalg.vector_norm({args[0]}, dim=-1, keepdim=True) + _SAFE_EPS)")
            return tmp

        # Inline length: linalg.vector_norm(a, dim=-1)
        if name == "length" and len(args) == 1:
            self._emit(f"{tmp} = _torch.linalg.vector_norm({args[0]}, dim=-1)")
            return tmp

        # Inline sincos: stack [sin, cos] into vec2
        if name == "sincos" and len(args) == 1:
            self._emit(f"{tmp} = _torch.stack([_torch.sin({args[0]}), _torch.cos({args[0]})], dim=-1)")
            return tmp

        # Inline sqrt: clamp to avoid NaN on negative input
        if name == "sqrt" and len(args) == 1:
            self._emit(f"{tmp} = _torch.sqrt(_torch.clamp({args[0]}, min=0.0))")
            return tmp

        # Inline log variants: clamp to SAFE_EPSILON to avoid -inf
        if name == "log" and len(args) == 1:
            self._emit(f"{tmp} = _torch.log(_torch.clamp({args[0]}, min=_SAFE_EPS))")
            return tmp
        if name == "log2" and len(args) == 1:
            self._emit(f"{tmp} = _torch.log2(_torch.clamp({args[0]}, min=_SAFE_EPS))")
            return tmp
        if name == "log10" and len(args) == 1:
            self._emit(f"{tmp} = _torch.log10(_torch.clamp({args[0]}, min=_SAFE_EPS))")
            return tmp

        # Inline fract: x - floor(x)
        if name == "fract" and len(args) == 1:
            self._emit(f"{tmp} = {args[0]} - _torch.floor({args[0]})")
            return tmp

        # Inline isnan/isinf: return float mask
        if name == "isnan" and len(args) == 1:
            self._emit(f"{tmp} = _torch.isnan({args[0]}).float()")
            return tmp
        if name == "isinf" and len(args) == 1:
            self._emit(f"{tmp} = _torch.isinf({args[0]}).float()")
            return tmp

        # Inline pow2/pow10: 2^x, 10^x
        if name == "pow2" and len(args) == 1:
            self._emit(f"{tmp} = _torch.pow(2.0, {args[0]})")
            return tmp
        if name == "pow10" and len(args) == 1:
            self._emit(f"{tmp} = _torch.pow(10.0, {args[0]})")
            return tmp

        # Inline luma: weighted sum of RGB channels (guard for non-vector input)
        if name == "luma" and len(args) == 1:
            ret_type = self.type_map.get(id(node.args[0]))
            if ret_type is not None and ret_type.is_vector and ret_type.channels >= 3:
                self._emit(f"{tmp} = 0.2126 * {args[0]}[..., 0] + 0.7152 * {args[0]}[..., 1] + 0.0722 * {args[0]}[..., 2]")
                return tmp

        # Inline smoothstep: t*t*(3-2t) with clamped t
        if name == "smoothstep" and len(args) == 3:
            tt = self._tmp()
            self._emit(f"{tt} = _torch.clamp(({args[2]} - {args[0]}) / ({args[1]} - {args[0]} + _SAFE_EPS), 0.0, 1.0)")
            self._emit(f"{tmp} = {tt} * {tt} * (3.0 - 2.0 * {tt})")
            return tmp

        # Inline step: (x >= edge).float()
        if name == "step" and len(args) == 2:
            self._emit(f"{tmp} = ({args[1]} >= {args[0]}).float()")
            return tmp

        # Inline fit: remap from [old_min, old_max] to [new_min, new_max]
        if name == "fit" and len(args) == 5:
            tt = self._tmp()
            self._emit(f"{tt} = ({args[0]} - {args[1]}) / ({args[2]} - {args[1]} + _SAFE_EPS)")
            self._emit(f"{tmp} = _torch.lerp({args[3]}, {args[4]}, {tt})")
            return tmp

        # Inline mod: safe fmod
        if name == "mod" and len(args) == 2:
            self._emit(f"{tmp} = _torch.fmod({args[0]}, _tw({args[1]} == 0, _SAFE_EPS, {args[1]}))")
            return tmp

        # Inline cross: cross product (truncate vec4 to vec3)
        if name == "cross" and len(args) == 2:
            self._emit(f"{tmp} = _torch.cross({args[0]}[..., :3], {args[1]}[..., :3], dim=-1)")
            return tmp

        # Inline reflect: i - 2 * dot(i, n) * n
        if name == "reflect" and len(args) == 2:
            dt = self._tmp()
            self._emit(f"{dt} = ({args[0]} * {args[1]}).sum(dim=-1, keepdim=True)")
            self._emit(f"{tmp} = {args[0]} - 2.0 * {dt} * {args[1]}")
            return tmp

        # Inline spow: sign(x) * pow(abs(x), y) — safe power for negative bases
        if name == "spow" and len(args) == 2:
            at = self._tmp()
            self._emit(f"{at} = _torch.abs({args[0]})")
            mask = self._tmp()
            self._emit(f"{mask} = {at} < _SAFE_EPS")
            self._emit(f"{tmp} = _tw({mask}, _torch.zeros_like({args[0]}), _torch.sign({args[0]}) * _torch.pow(_torch.clamp({at}, min=_SAFE_EPS), {args[1]}))")
            return tmp

        # Inline sdiv: safe division — returns 0 where |b| < eps
        if name == "sdiv" and len(args) == 2:
            mask = self._tmp()
            self._emit(f"{mask} = _torch.abs({args[1]}) < _SAFE_EPS")
            self._emit(f"{tmp} = _tw({mask}, _torch.zeros_like({args[0]}), {args[0]} / _tw({mask}, _torch.ones_like({args[1]}), {args[1]}))")
            return tmp

        # General case: call through stdlib dict
        args_str = ", ".join(args)
        self._emit(f"{tmp} = _fns[{name!r}]({args_str})")
        # Skip isinstance guard when the return type is known to be numeric
        # (all numeric stdlib functions return tensors directly).
        ret_type = self.type_map.get(id(node))
        if ret_type is None or ret_type == TEXType.STRING:
            self._emit(f"if not isinstance({tmp}, (str, list, _torch.Tensor)): {tmp} = _torch.scalar_tensor(float({tmp}), dtype=_torch.float32, device=_dev)")
        return tmp

    def _emit_vec_constructor(self, node: VecConstructor) -> str:
        n = node.size

        # Optimization: hoist all-literal vec constructors to preamble.
        # vec3(0.2126, 0.7152, 0.0722) → single pre-computed tensor reused on every call.
        if len(node.args) > 1 and all(isinstance(a, NumberLiteral) for a in node.args) and len(node.args) == n:
            values = tuple(a.value for a in node.args)
            cached = self._vec_const_cache.get(values)
            if cached is not None:
                return cached
            var = self._tmp()
            vals_repr = ", ".join(repr(v) for v in values)
            self._preamble.append(
                f"    {var} = _torch.tensor([{vals_repr}], dtype=_torch.float32, device=_dev)"
                f".reshape(" + ", ".join(["1"] * 3) + f", {n}).expand(*_sp, {n})"
                f" if _sp else _torch.tensor([{vals_repr}], dtype=_torch.float32, device=_dev)"
            )
            self._vec_const_cache[values] = var
            return var

        args = [self._emit_expr(a) for a in node.args]
        tmp = self._tmp()

        if len(args) == 1:
            # Check compile-time type for identity case
            arg_type = self.type_map.get(id(node.args[0]))
            if arg_type is not None and arg_type.is_vector and arg_type.channels == n:
                # vec4(vec4_val) — identity / type cast
                self._emit(f"{tmp} = {args[0]}")
            else:
                # Broadcast: vec3(scalar) → call _es once, then expand
                es_tmp = self._tmp()
                self._emit(f"{es_tmp} = _es({args[0]}, _sp)")
                components = ", ".join([es_tmp] * n)
                self._emit(f"{tmp} = _torch.stack([{components}], dim=-1)")
        else:
            # Flatten composite args using compile-time types:
            # e.g. vec4(vec3_val, float_val) → [v[...,0], v[...,1], v[...,2], f]
            component_exprs: list[str] = []
            for i, a in enumerate(args):
                arg_type = self.type_map.get(id(node.args[i]))
                if arg_type is not None and arg_type.is_vector:
                    for ch in range(arg_type.channels):
                        component_exprs.append(f"_es({a}[..., {ch}], _sp)")
                else:
                    component_exprs.append(f"_es({a}, _sp)")

            components = ", ".join(component_exprs)
            self._emit(f"{tmp} = _torch.stack([{components}], dim=-1)")

        return tmp

    def _emit_mat_constructor(self, node: MatConstructor) -> str:
        n = node.size
        args = [self._emit_expr(a) for a in node.args]
        tmp = self._tmp()

        if len(args) == 1:
            self._emit(f"{tmp} = _torch.eye({n}, dtype=_torch.float32, device=_dev) * {args[0]}")
        elif len(args) == n * n:
            components = ", ".join(f"_es({a}, _sp)" for a in args)
            flat = self._tmp()
            self._emit(f"{flat} = _torch.stack([{components}], dim=-1)")
            self._emit(f"{tmp} = {flat}.reshape({flat}.shape[:-1] + ({n}, {n}))")
        else:
            raise _Unsupported(f"mat{n} wrong arg count")

        return tmp

    def _emit_cast(self, node: CastExpr) -> str:
        value = self._emit_expr(node.expr)

        # Fast path: float cast on already-float/int source is a no-op
        if node.target_type == "float":
            src_type = self.type_map.get(id(node.expr))
            if src_type is not None and src_type in (TEXType.FLOAT, TEXType.INT):
                return value

        tmp = self._tmp()

        if node.target_type == "string":
            self._emit(f"if _torch.is_tensor({value}):")
            self._indent += 1
            self._emit(f"_cv = {value}.item() if {value}.dim() == 0 else {value}.float().mean().item()")
            self._emit(f"{tmp} = str(int(_cv)) if _cv == int(_cv) else str(_cv)")
            self._indent -= 1
            self._emit(f"else:")
            self._indent += 1
            self._emit(f"{tmp} = str({value})")
            self._indent -= 1
        elif node.target_type == "int":
            self._emit(f"{tmp} = _torch.floor({value})")
        elif node.target_type == "float":
            self._emit(f"{tmp} = {value}.float()")
        else:
            self._emit(f"{tmp} = {value}")

        return tmp
