"""
TEX Codegen — compile TEX AST to Python functions for zero-overhead execution.

Instead of tree-walking the AST on every frame, this module generates a Python
function string, compiles it via exec(), and caches the callable. Subsequent
executions call the function directly, eliminating:
  - Per-node dispatch table lookups
  - Per-node Python function call overhead
  - Redundant dict lookups for env/bindings

Falls back to None (caller uses tree-walking interpreter) for unsupported
patterns: arrays, string operations, spatial if/else with complex merging.

Break/continue handling: the codegen uses exception-based control flow
(matching the interpreter) because Python's native 'continue' would skip
the for-loop update statement. _CgBreak/_CgContinue are raised in generated
code and caught by the generated loop wrapper.
"""
from __future__ import annotations

import math
from typing import Any

from ..tex_compiler.ast_nodes import (
    ASTNode, Program, VarDecl, Assignment, IfElse, ForLoop, WhileLoop,
    ExprStatement, BreakStmt, ContinueStmt, ParamDecl, ArrayDecl,
    BinOp, UnaryOp, TernaryOp, FunctionCall, Identifier, BindingRef,
    ChannelAccess, NumberLiteral, StringLiteral, VecConstructor,
    MatConstructor, CastExpr, ArrayIndexAccess, ArrayLiteral,
)
from ..tex_compiler.type_checker import TEXType, CHANNEL_MAP


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


class _CodeGen:
    """Generates Python source code from a TEX AST."""

    def __init__(self, type_map: dict[int, TEXType]):
        self.type_map = type_map
        self._tmp_counter = 0
        self._lines: list[str] = []
        self._indent = 1  # Start at 1 (inside function body)
        self._constants: dict[float, str] = {}  # value -> const var name
        self._const_lines: list[str] = []  # constant initialization lines

    def _tmp(self) -> str:
        """Generate a unique temporary variable name."""
        self._tmp_counter += 1
        return f"_t{self._tmp_counter}"

    def _emit(self, line: str):
        """Emit a line of code at the current indentation level."""
        self._lines.append("    " * self._indent + line)

    def _get_const(self, value: float) -> str:
        """Get or create a constant tensor variable for a number literal."""
        if value not in self._constants:
            name = f"_c{len(self._constants)}"
            self._constants[value] = name
            self._const_lines.append(
                f"    {name} = _torch.tensor({value!r}, dtype=_torch.float32, device=_dev)"
            )
        return self._constants[value]

    def emit_program(self, program: Program):
        """Emit code for the entire program."""
        for stmt in program.statements:
            self._emit_stmt(stmt)

    def build(self) -> Any:
        """Compile the generated source to a callable function."""
        body = "\n".join(self._const_lines + self._lines)
        func_src = (
            "def _tex_fn(_env, _bind, _fns, _dev, _sp, "
            "_torch, _bp, _es, _tw, _math, _SAFE_EPS, _CMAP, _MAX_ITER, "
            "_CgBreak, _CgContinue):\n"
            f"{body}\n"
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
            raise _Unsupported("ArrayDecl")
        else:
            raise _Unsupported(f"Unknown statement: {type(stmt).__name__}")

    def _emit_var_decl(self, stmt: VarDecl):
        if stmt.initializer:
            expr = self._emit_expr(stmt.initializer)
            self._emit(f"_env[{stmt.name!r}] = {expr}")
        else:
            # Default initialization based on type
            declared = self.type_map.get(id(stmt), TEXType.FLOAT)
            if declared == TEXType.STRING:
                self._emit(f"_env[{stmt.name!r}] = ''")
            elif declared == TEXType.VEC3:
                self._emit(f"if _sp:")
                self._indent += 1
                self._emit(f"_env[{stmt.name!r}] = _torch.zeros(*_sp, 3, dtype=_torch.float32, device=_dev)")
                self._indent -= 1
                self._emit(f"else:")
                self._indent += 1
                self._emit(f"_env[{stmt.name!r}] = _torch.zeros(3, dtype=_torch.float32, device=_dev)")
                self._indent -= 1
            elif declared == TEXType.VEC4:
                self._emit(f"if _sp:")
                self._indent += 1
                self._emit(f"_env[{stmt.name!r}] = _torch.zeros(*_sp, 4, dtype=_torch.float32, device=_dev)")
                self._indent -= 1
                self._emit(f"else:")
                self._indent += 1
                self._emit(f"_env[{stmt.name!r}] = _torch.zeros(4, dtype=_torch.float32, device=_dev)")
                self._indent -= 1
            elif declared.is_matrix:
                n = declared.mat_size
                self._emit(f"_env[{stmt.name!r}] = _torch.zeros({n}, {n}, dtype=_torch.float32, device=_dev)")
            else:
                self._emit(f"_env[{stmt.name!r}] = _torch.tensor(0.0, dtype=_torch.float32, device=_dev)")

    def _emit_assignment(self, stmt: Assignment):
        value_expr = self._emit_expr(stmt.value)
        target = stmt.target
        if isinstance(target, Identifier):
            self._emit(f"_env[{target.name!r}] = {value_expr}")
        elif isinstance(target, BindingRef):
            self._emit(f"_bind[{target.name!r}] = {value_expr}")
        elif isinstance(target, ChannelAccess):
            self._emit_channel_assign(target, value_expr)
        elif isinstance(target, ArrayIndexAccess):
            raise _Unsupported("ArrayIndexAccess assignment")
        else:
            raise _Unsupported(f"Unsupported assignment target: {type(target).__name__}")

    def _emit_channel_assign(self, target: ChannelAccess, value_expr: str):
        channels = target.channels
        if len(channels) != 1:
            raise _Unsupported("Multi-channel assignment")
        idx = CHANNEL_MAP.get(channels)
        if idx is None:
            raise _Unsupported(f"Invalid channel: {channels}")

        base_expr = self._emit_expr(target.object)
        tmp = self._tmp()
        self._emit(f"{tmp} = {base_expr}.clone()")
        self._emit(f"{tmp}[..., {idx}] = _es({value_expr}, {tmp}.shape[:-1])")

        if isinstance(target.object, Identifier):
            self._emit(f"_env[{target.object.name!r}] = {tmp}")
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
        """Emit spatial if/else with selective cloning and torch.where merge."""
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

        snap_env = self._tmp()
        snap_bind = self._tmp()
        then_env = self._tmp()
        then_bind = self._tmp()

        # Snapshot modified variables
        env_mod_repr = repr(all_env_mods)
        bind_mod_repr = repr(all_bind_mods)
        self._emit(f"{snap_env} = {{k: _env[k].clone() if _torch.is_tensor(_env[k]) else _env[k] for k in {env_mod_repr} if k in _env}}")
        self._emit(f"{snap_bind} = {{k: _bind[k].clone() if _torch.is_tensor(_bind[k]) else _bind[k] for k in {bind_mod_repr} if k in _bind}}")

        # Execute then-branch
        for s in stmt.then_body:
            self._emit_stmt(s)

        # Capture then-state
        self._emit(f"{then_env} = {{k: _env.get(k) for k in {env_mod_repr}}}")
        self._emit(f"{then_bind} = {{k: _bind.get(k) for k in {bind_mod_repr}}}")

        # Restore snapshot
        self._emit(f"_env.update({snap_env})")
        self._emit(f"_bind.update({snap_bind})")

        if stmt.else_body:
            for s in stmt.else_body:
                self._emit_stmt(s)

        # Merge with torch.where
        cond_bool = self._tmp()
        self._emit(f"{cond_bool} = ({cond_var} > 0.5)")

        for k in all_env_mods:
            tv = self._tmp()
            ev = self._tmp()
            self._emit(f"{tv} = {then_env}.get({k!r})")
            self._emit(f"{ev} = _env.get({k!r})")
            self._emit(f"if {tv} is not None and {ev} is not None and _torch.is_tensor({tv}) and _torch.is_tensor({ev}):")
            self._indent += 1
            self._emit(f"_env[{k!r}] = _tw({cond_bool}, {tv}, {ev})")
            self._indent -= 1
            self._emit(f"elif {tv} is not None:")
            self._indent += 1
            self._emit(f"_env[{k!r}] = {tv}")
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
            elif isinstance(stmt, VarDecl):
                env_vars.add(stmt.name)
            elif isinstance(stmt, IfElse):
                e1, b1 = self._collect_modified_vars(stmt.then_body)
                e2, b2 = self._collect_modified_vars(stmt.else_body)
                env_vars |= e1 | e2
                bind_names |= b1 | b2
            elif isinstance(stmt, (ForLoop, WhileLoop)):
                body = stmt.body
                e, b = self._collect_modified_vars(body)
                env_vars |= e
                bind_names |= b
        return env_vars, bind_names

    def _emit_for_loop(self, stmt: ForLoop):
        """Emit a for loop with exception-based break/continue."""
        # Emit initializer
        self._emit_stmt(stmt.init)

        # Try static bound extraction (same as interpreter)
        static = self._try_static_bound(stmt)
        iter_var = self._tmp()

        self._emit(f"{iter_var} = 0")
        self._emit(f"while {iter_var} < _MAX_ITER:")
        self._indent += 1

        # Condition check
        if static is not None:
            loop_var, bound, is_le = static
            cv = self._tmp()
            self._emit(f"{cv} = _env[{loop_var!r}]")
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
            cond_expr = self._emit_expr(stmt.condition)
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

        # Body wrapped in try/except for break/continue
        self._emit(f"try:")
        self._indent += 1
        for s in stmt.body:
            self._emit_stmt(s)
        self._indent -= 1
        self._emit(f"except _CgBreak:")
        self._indent += 1
        self._emit(f"break")
        self._indent -= 1
        self._emit(f"except _CgContinue:")
        self._indent += 1
        self._emit(f"pass")
        self._indent -= 1

        # Update always executes (even after continue)
        self._emit_stmt(stmt.update)
        self._emit(f"{iter_var} += 1")
        self._indent -= 1

        # Check iteration limit after loop
        self._emit(f"if {iter_var} >= _MAX_ITER:")
        self._indent += 1
        self._emit(f"raise RuntimeError('For loop exceeded maximum iteration limit (' + str(_MAX_ITER) + '). Check your loop condition.')")
        self._indent -= 1

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

        cond_expr = self._emit_expr(stmt.condition)
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

        # Body wrapped in try/except for break/continue
        self._emit(f"try:")
        self._indent += 1
        for s in stmt.body:
            self._emit_stmt(s)
        self._indent -= 1
        self._emit(f"except _CgBreak:")
        self._indent += 1
        self._emit(f"break")
        self._indent -= 1
        self._emit(f"except _CgContinue:")
        self._indent += 1
        self._emit(f"pass")
        self._indent -= 1

        self._emit(f"{iter_var} += 1")
        self._indent -= 1

        # Check iteration limit after loop
        self._emit(f"if {iter_var} >= _MAX_ITER:")
        self._indent += 1
        self._emit(f"raise RuntimeError('While loop exceeded maximum iteration limit (' + str(_MAX_ITER) + '). Check your loop condition.')")
        self._indent -= 1

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
            raise _Unsupported("ArrayIndexAccess")

        if isinstance(node, ArrayLiteral):
            raise _Unsupported("ArrayLiteral")

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
        if lt is not None and rt is not None:
            l_vec = lt.is_vector or lt == TEXType.VEC3 or lt == TEXType.VEC4
            r_vec = rt.is_vector or rt == TEXType.VEC3 or rt == TEXType.VEC4
            l_scalar = not l_vec and not lt.is_matrix
            r_scalar = not r_vec and not rt.is_matrix
            if (l_vec and r_scalar) or (l_scalar and r_vec):
                needs_broadcast = True

        if needs_broadcast:
            tl = self._tmp()
            tr = self._tmp()
            self._emit(f"{tl}, {tr} = _bp({left}, {right})")
            left, right = tl, tr

        op = node.op
        tmp = self._tmp()

        if op in _INFIX_OPS:
            self._emit(f"{tmp} = ({left} {op} {right})")
        elif op == "/":
            self._emit(f"{tmp} = ({left} / ({right} + _SAFE_EPS * ({right} == 0).float()))")
        elif op == "%":
            self._emit(f"{tmp} = _torch.fmod({left}, {right} + _SAFE_EPS * ({right} == 0).float())")
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
        args = [self._emit_expr(a) for a in node.args]
        args_str = ", ".join(args)
        tmp = self._tmp()
        self._emit(f"{tmp} = _fns[{node.name!r}]({args_str})")
        # Ensure result is tensor if not string/list
        self._emit(f"if not isinstance({tmp}, (str, list, _torch.Tensor)): {tmp} = _torch.tensor(float({tmp}), dtype=_torch.float32, device=_dev)")
        return tmp

    def _emit_vec_constructor(self, node: VecConstructor) -> str:
        args = [self._emit_expr(a) for a in node.args]
        tmp = self._tmp()
        n = node.size

        if len(args) == 1:
            # Check compile-time type for identity case
            arg_type = self.type_map.get(id(node.args[0]))
            if arg_type is not None and arg_type.is_vector and arg_type.channels == n:
                # vec4(vec4_val) — identity / type cast
                self._emit(f"{tmp} = {args[0]}")
            else:
                # Broadcast: vec4(0.5) -> [0.5, 0.5, 0.5, 0.5]
                components = ", ".join([f"_es({args[0]}, _sp)"] * n)
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
