"""Interpreter control-flow execution — SPLIT-I (v0.44 Phase A1).

Split mechanically out of `interpreter.py`, following the STR-7 codegen split's mixin
pattern: `Interpreter` still inherits this class, so `self._exec_if_else(...)` etc. resolve
exactly as before — no new call layer, no cross-module lookup added to any per-pixel path.
This module owns if/else (scalar short-circuit AND the vectorized `torch.where` spatial-if),
for-loops (including the static-range and UC-3 uniform-range fast paths) and while-loops.

`_Break`/`_Continue`, `InterpreterError` and `MAX_LOOP_ITERATIONS` are `interpreter.py`
module-level names; `_tensor_where`, `_collect_expr_names` and `_int_valued_scalar` are
`interpreter.py` module-level helpers defined after the `Interpreter` class. Both are
imported back lazily (inside the methods that need them) rather than at module load time —
`interpreter.py` imports this module before any of those names exist in its own namespace,
so a top-level `from .interpreter import ...` here would be a load-time cycle. This is the
same deferred-import shape `masked_flow.py` already uses for its own back-references.

No behaviour changed by this move: every body below is byte-identical to the code it replaced
in `interpreter.py`.
"""
from __future__ import annotations

from typing import Any

import torch

from ..tex_compiler.ast_nodes import (
    ASTNode, Assignment, BinOp, ForLoop, Identifier, IfElse, VarDecl, WhileLoop,
    collect_assigned_vars, try_extract_static_range,
)
from .interpreter import MAX_LOOP_ITERATIONS, InterpreterError, _Break, _Continue
# TRK-143: `cond_mask` is masked_flow's (language-0.25, TRK-152) existing single
# definition of "is this pixel on" — module-level, no cycle (masked_flow.py never
# imports interpreter.py/interpreter_control_flow.py at load time, only lazily inside
# its own methods). The UNMASKED `0.23` spatial-if below now calls it too, instead of
# repeating the same formula inline, so there is exactly one spelling for both language
# tiers and both runtime backends (see codegen.py's mirroring edit for the other half).
from . import masked_flow as _masked_flow_mod


class _ControlFlowMixin:
    """Mixin supplying `Interpreter`'s if/for/while execution methods (SPLIT-I).

    Composed onto `Interpreter` alongside `MaskedFlowMixin` / `_SpatialContextMixin` /
    `_BindingExecMixin` — every attribute referenced below (`self.env`, `self._eval`,
    `self._exec_stmt`, `self._assigned_vars_cache`, ...) is set on the instance by
    `Interpreter.__init__` or supplied by a sibling mixin.
    """

    _collect_assigned_vars = staticmethod(collect_assigned_vars)

    def _exec_if_else(self, node: IfElse):
        """
        Execute if/else.

        Scalar conditions (0-dim tensors, e.g. loop counters): use short-circuit
        evaluation — only execute the true branch. This is required for
        break/continue to work correctly inside if blocks.

        Spatial conditions (B, H, W tensors): use vectorized both-branch
        evaluation with torch.where merging. Uses selective cloning — only
        variables actually assigned in branches are cloned/merged.
        """
        cond = self._eval(node.condition)

        # Scalar condition: short-circuit (supports break/continue)
        if cond.dim() == 0:
            if cond.item() > 0.5:
                for stmt in node.then_body:
                    self._exec_stmt(stmt)
            elif node.else_body:
                for stmt in node.else_body:
                    self._exec_stmt(stmt)
            return

        # Spatial condition: vectorized both-branch evaluation
        self._exec_spatial_if(node, cond)

    def _snapshot_vars(
        self, keys: set[str], source: dict,
    ) -> dict[str, Any]:
        """Clone tensor values (or copy non-tensors) for the given keys."""
        snap = {}
        for k in keys:
            v = source.get(k)
            if v is not None:
                snap[k] = v.clone() if isinstance(v, torch.Tensor) else v
        return snap

    @staticmethod
    def _merge_branch_vars(
        cond_bool: torch.Tensor, cond_scalar_box: list,
        target: dict, keys: set[str],
        then_vals: dict, else_vals: dict,
    ) -> None:
        """Merge then/else branch values into *target* using torch.where.

        *cond_scalar_box* is a single-element list used as a lazy cache for
        the majority-vote scalar (needed for string merges).
        """
        from .interpreter import _tensor_where
        for key in keys:
            then_val = then_vals.get(key)
            else_val = else_vals.get(key)
            if then_val is not None and else_val is not None:
                if isinstance(then_val, torch.Tensor) and isinstance(else_val, torch.Tensor):
                    target[key] = _tensor_where(cond_bool, then_val, else_val)
                elif isinstance(then_val, str) or isinstance(else_val, str):
                    if not cond_scalar_box:
                        cond_scalar_box.append(
                            cond_bool.float().mean().item() > 0.5
                        )
                    target[key] = then_val if cond_scalar_box[0] else else_val
                else:
                    target[key] = then_val
            elif then_val is not None:
                target[key] = then_val
            elif else_val is not None:
                target[key] = else_val

    def _exec_spatial_if(self, node: IfElse, cond: torch.Tensor):
        """Execute a spatial (vectorized) if/else with torch.where merging."""
        # Selective cloning: only snapshot variables that are assigned in branches.
        # Cache by node id — AST is immutable, so results never change.
        node_id = id(node)
        cached = self._assigned_vars_cache.get(node_id)
        if cached is not None:
            modified_env, modified_bindings = cached
        else:
            modified_env, modified_bindings = self._collect_assigned_vars(node.then_body)
            if node.else_body:
                e2, b2 = self._collect_assigned_vars(node.else_body)
                modified_env |= e2
                modified_bindings |= b2
            self._assigned_vars_cache[node_id] = (modified_env, modified_bindings)

        # Snapshot only modified variables that already exist
        env_snapshot = self._snapshot_vars(modified_env, self.env)
        bindings_snapshot = self._snapshot_vars(modified_bindings, self.bindings)
        has_arrays = bool(self._array_meta)
        if has_arrays:
            meta_snapshot = dict(self._array_meta)

        # Execute then-branch and capture modified state
        for stmt in node.then_body:
            self._exec_stmt(stmt)
        then_env = {k: self.env.get(k) for k in modified_env}
        then_bindings = {k: self.bindings.get(k) for k in modified_bindings}
        if has_arrays:
            then_meta = dict(self._array_meta)

        # Restore snapshot and execute else-branch
        self.env.update(env_snapshot)
        self.bindings.update(bindings_snapshot)
        if has_arrays:
            self._array_meta = dict(meta_snapshot)

        if node.else_body:
            for stmt in node.else_body:
                self._exec_stmt(stmt)
            else_env = {k: self.env.get(k) for k in modified_env}
            else_bindings = {k: self.bindings.get(k) for k in modified_bindings}
        else:
            else_env = env_snapshot
            else_bindings = bindings_snapshot

        # Merge using torch.where (tensors) or scalar majority-vote (strings)
        # TRK-143: was the inline `(cond > 0.5) if cond.is_floating_point() else
        # cond.bool()` — now routed through the one shared `cond_mask` definition so
        # this reading can never drift from what `masked_flow.cond_mask` (and codegen's
        # mirror of it) computes. Byte-for-byte the same formula for every dtype this
        # method has ever been called with — see `cond_mask`'s own docstring for why
        # THIS spelling (the interpreter's) is the one both tiers now share.
        cond_bool = _masked_flow_mod.cond_mask(cond)
        cond_scalar_box: list = []  # lazy cache for string merge
        self._merge_branch_vars(
            cond_bool, cond_scalar_box,
            self.env, modified_env, then_env, else_env,
        )
        self._merge_branch_vars(
            cond_bool, cond_scalar_box,
            self.bindings, modified_bindings, then_bindings, else_bindings,
        )

        # Merge array metadata from then-branch
        if has_arrays:
            self._array_meta.update(then_meta)

    def _loop_cond_true(self, cond_node: ASTNode) -> bool:
        """Evaluate a loop condition and return True if the loop should continue."""
        cond = self._eval(cond_node)
        if cond.dim() == 0:
            return cond.item() > 0.5
        return (cond > 0.5).any().item()

    def _exec_loop_body(self, body: list[ASTNode]) -> bool:
        """Execute loop body statements. Returns True if break was hit."""
        try:
            for stmt in body:
                self._exec_stmt(stmt)
        except _Break:
            return True
        except _Continue:
            pass
        return False

    def _raise_loop_limit(self, loop_type: str, loc):
        raise InterpreterError(
            f"This {loop_type} loop ran {MAX_LOOP_ITERATIONS} iterations without finishing",
            loc, source=self._source, code="E6010",
            hint=f"Loops are capped at {MAX_LOOP_ITERATIONS} iterations to prevent hangs. "
                 "Make sure your loop condition will eventually become false.",
        )

    def _exec_for_loop(self, node: ForLoop):
        """
        Execute a bounded for loop sequentially.

        Each iteration runs the loop body as vectorized tensor operations.
        The loop variable is a scalar that gets updated each iteration.
        Hard limit of MAX_LOOP_ITERATIONS to prevent infinite loops.

        Optimization: for fully static loops like `for (int i = 0; i < N; i++)`
        where init, condition, and update are all literal-based, pre-compute
        the iteration range as Python range() — zero .item() GPU→CPU syncs.
        """
        # Try fully static loop: pre-compute range() from init/cond/update literals
        static_range = self._try_extract_static_range(node)
        # UC-3: else try a *uniform* range — same shape but with scalar-expression
        # bounds (e.g. `for (dy = -$radius; dy <= $radius; ...)`), resolved once at
        # loop entry instead of an `.item()` sync every iteration.
        if static_range is None:
            static_range = self._try_resolve_uniform_range(node)

        if static_range is not None:
            loop_var, iter_range = static_range

            if len(iter_range) > MAX_LOOP_ITERATIONS:
                raise InterpreterError(
                    f"This for loop would run {len(iter_range)} iterations, which exceeds the limit of {MAX_LOOP_ITERATIONS}",
                    node.loc, source=self._source, code="E6010",
                    hint=f"Loops are capped at {MAX_LOOP_ITERATIONS} iterations to prevent hangs. "
                         "Consider reducing your range or processing in smaller batches.",
                )

            # Pre-allocate all loop variable tensors at once.
            # torch.arange + unbind is faster than per-iteration torch.tensor().
            dtype = self._dtype
            device = self.device
            n = len(iter_range)
            start = iter_range.start
            step = iter_range.step
            loop_tensors = torch.arange(start, start + n * step, step,
                                        dtype=dtype, device=device)
            # unbind(0) returns a tuple of 0-d tensors — no per-iteration allocation
            loop_values = loop_tensors.unbind(0)
            env = self.env
            inplace_discard = self._inplace_ready.discard

            for val in loop_values:
                env[loop_var] = val
                inplace_discard(loop_var)
                if self._exec_loop_body(node.body):
                    break
        else:
            # General case — execute init, evaluate condition each iteration
            self._exec_stmt(node.init)
            iteration = 0
            while iteration < MAX_LOOP_ITERATIONS:
                if not self._loop_cond_true(node.condition):
                    break
                if self._exec_loop_body(node.body):
                    break
                self._exec_stmt(node.update)
                iteration += 1

            if iteration >= MAX_LOOP_ITERATIONS:
                self._raise_loop_limit("for", node.loc)

    def _try_extract_static_range(self, node: ForLoop) -> tuple[str, range] | None:
        """Try to extract a fully static loop as a Python range().

        Delegates to the shared ``try_extract_static_range`` in ast_nodes
        and wraps the (var, start, stop, step) tuple into a range object.
        """
        result = try_extract_static_range(node)
        if result is None:
            return None
        loop_var, start, end, step = result
        try:
            return (loop_var, range(start, end, step))
        except ValueError:
            return None

    def _analyze_uniform_range(self, node: ForLoop):
        """Structural check for a uniform (scalar-expression-bounded) for-loop —
        same shape as try_extract_static_range but the start/end/step may be any
        expressions, provided they don't reference the loop var or any variable
        assigned in the loop body (which would make them non-uniform). Returns
        (loop_var, start_expr, cond_op, end_expr, step_expr, step_sign) or False.
        """
        init = node.init
        if not isinstance(init, VarDecl) or init.initializer is None:
            return False
        loop_var = init.name
        start_e = init.initializer

        cond = node.condition
        if (not isinstance(cond, BinOp) or cond.op not in ("<", "<=")
                or not isinstance(cond.left, Identifier) or cond.left.name != loop_var):
            return False
        end_e = cond.right

        upd = node.update
        if (not isinstance(upd, Assignment) or not isinstance(upd.target, Identifier)
                or upd.target.name != loop_var or not isinstance(upd.value, BinOp)):
            return False
        ub = upd.value
        if (not isinstance(ub.left, Identifier) or ub.left.name != loop_var
                or ub.op not in ("+", "-")):
            return False
        step_e = ub.right
        step_sign = 1 if ub.op == "+" else -1

        # Bounds must not depend on the loop var or anything mutated in the body —
        # for BOTH env vars AND bindings (UC-3b: a bound reading @A while the body
        # reassigns @A resolves once and diverges from per-iteration semantics).
        body_assigned, body_bindings = self._collect_assigned_vars(node.body)
        forbidden = body_assigned | {loop_var}
        names: set[str] = set()
        bind_names: set[str] = set()
        from .interpreter import _collect_expr_names
        for e in (start_e, end_e, step_e):
            _collect_expr_names(e, names, bind_names)
        if (names & forbidden) or (bind_names & body_bindings):
            return False
        return (loop_var, start_e, cond.op, end_e, step_e, step_sign)

    def _try_resolve_uniform_range(self, node: ForLoop) -> tuple[str, range] | None:
        """Resolve a uniform-range for-loop to a Python range() by evaluating its
        scalar bound expressions once (UC-3). Structural eligibility is memoized."""
        from .interpreter import _int_valued_scalar
        elig = self._uniform_range_cache.get(id(node))
        if elig is None:
            elig = self._analyze_uniform_range(node)
            self._uniform_range_cache[id(node)] = elig
        if elig is False:
            return None
        loop_var, start_e, cond_op, end_e, step_e, step_sign = elig
        try:
            # UC-3a: only resolve to a Python range() when start/end/step are all
            # INTEGER-VALUED. The general per-iteration path evaluates the float
            # condition with true fractional values, so flooring a fractional
            # bound/step (e.g. i=0.5, step 1.5) silently changes the loop values
            # and trip count vs v0.14.1. A fractional bound → fall back.
            start = _int_valued_scalar(self._eval(start_e))
            end = _int_valued_scalar(self._eval(end_e))
            step_mag = _int_valued_scalar(self._eval(step_e))
        except (ValueError, TypeError, RuntimeError):
            return None  # non-scalar / non-numeric bound → fall back to general path
        if start is None or end is None or step_mag is None:
            return None  # fractional or non-scalar bound → general path
        step = step_sign * step_mag
        if step == 0:
            return None
        if cond_op == "<=":
            end += 1
        try:
            return (loop_var, range(start, end, step))
        except ValueError:
            return None

    def _exec_while_loop(self, node: WhileLoop):
        """Execute a bounded while loop. Hard limit of MAX_LOOP_ITERATIONS."""
        iteration = 0
        while iteration < MAX_LOOP_ITERATIONS:
            if not self._loop_cond_true(node.condition):
                break
            if self._exec_loop_body(node.body):
                break
            iteration += 1

        if iteration >= MAX_LOOP_ITERATIONS:
            self._raise_loop_limit("while", node.loc)

