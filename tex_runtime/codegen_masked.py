"""Masked per-pixel control flow, emitted — the codegen tier's language-`0.25` rules.

`docs/masked-control-flow.md` §1 is the specification and `tex_runtime/masked_flow.py` is
the interpreter's implementation of it. **The interpreter is the oracle; this module is its
mirror at emit time**, and the contract it has to meet is bitwise, not approximate: §5
concluded that no approximation exists anywhere in the rule, so a tolerance here would be
hiding something.

**How the mirror is kept honest.** Every mask edit the emitted source performs is a CALL
into `masked_flow` — `cond_mask`, `m_and`, `m_sub`, `m_any`, `frames_dead`, `apply_transfer`,
`record_return`, `merge_write`, `scatter_keep`, and the interpreter's own
`_merge_branch_vars` — reached through the `_MF` global that `codegen.build` seeds into the
generated module's namespace. So the two tiers do not have two spellings of the mask algebra
that must be *argued* to agree: they have one, called from two places. That is
`docs/masked-control-flow.md` §5's prescription for divergence site 1 applied to the whole
of §5's list rather than only to the predicate.

**What is emitted is the interpreter's shape, statement for statement.** A loop pushes a
`"loop"` frame and each pass a `"pass"` frame; a call pushes a `"call"` frame inside the
emitted `def`; a transfer edits the frame's `dead` set instead of leaving the block, so the
statements after it still run for the pixels that stayed; a loop runs while ANY pixel is
live. The region depth and the "declared inside this region" question of M1 are resolved at
EMIT time (regions are lexical, so the depth a statement runs at is a property of where it
is written), which is the one place the mirror is a translation rather than a copy.

**Byte-identity below `0.25` is by construction, not by care.** Nothing in this module runs
unless `codegen.try_compile`'s language gate opened, and the gate installs a SECOND
statement-dispatch table rather than putting a branch in the existing handlers — the same
shape L4 took on the interpreter. A program without a `0.25` pragma therefore emits source
that does not contain a single character from this file, which is what makes the
emitted-source digest acceptance in §8's `L5` row provable rather than measured.

**Two deliberate over-declines**, both `0.25`-only and both costing performance rather than
answers (`docs/masked-control-flow.md` §5, precondition 1, and §10's "a masked scalar-loop
path in codegen" entry):

* the **scalar-loop path declines a flagged program outright** — `_setup_scalar_loop` /
  `_is_scalar_body` run a loop body in Python scalars and cannot hold a per-pixel mask;
* **stencil specialisation declines too** — rewriting a loop nest into one `conv2d` deletes
  the passes whose masks the rule is about, and the interpreter has no such rewrite to
  match.
"""
from __future__ import annotations

from ..tex_compiler.ast_nodes import (
    ArrayDecl, ArrayIndexAccess, Assignment, BindingIndexAccess, BindingRef, BreakStmt,
    ChannelAccess, ContinueStmt, ExprStatement, ForLoop, FunctionDef, Identifier, IfElse,
    ParamDecl, ReturnStmt, VarDecl, WhileLoop, try_extract_static_range,
)

__all__ = ["MaskedEmitMixin"]

#: The name the generated source binds its `masked_flow.CgFlow` to.
_STATE = "_mf"


def _root_of(target):
    """The name an assignment ultimately stores into, and which store holds it.

    The emit-time twin of `masked_flow.MaskedFlowMixin._mf_root`; it answers the same
    question about the same AST node, one tier earlier."""
    t = target
    while True:
        cls = type(t)
        if cls is ChannelAccess:
            t = t.object
        elif cls is ArrayIndexAccess:
            t = t.array
        else:
            break
    cls = type(t)
    if cls is Identifier:
        return t.name, True
    if cls is BindingRef:
        return t.name, False
    return None, None


class MaskedEmitMixin:
    """The `0.25` statement emitters, mixed into `_CodeGen`.

    Nothing here overrides a `0.23` emitter. Each is reached only through the second
    dispatch table `_mf_begin` binds, so the default emission path is untouched."""

    # ── set-up ──────────────────────────────────────────────────────────────
    def _mf_begin(self) -> None:
        """Bind the `0.25` emitters for this compile, and the runtime mask carrier."""
        self._mf_on = True
        self._mf_depth = 0
        self._mf_decl_depth = {}
        self._preamble.append(f"    {_STATE} = _MF.CgFlow()")
        self._stmt_dispatch = dict(self._stmt_dispatch)
        self._stmt_dispatch.update({
            VarDecl: self._mfe_var_decl,
            ArrayDecl: self._mfe_array_decl,
            Assignment: self._mfe_assignment,
            IfElse: self._mfe_if_else,
            ForLoop: self._mfe_for_loop,
            WhileLoop: self._mfe_while_loop,
            FunctionDef: self._mfe_function_def,
            ReturnStmt: self._mfe_return,
            BreakStmt: self._mfe_break,
            ContinueStmt: self._mfe_continue,
            ExprStatement: self._stmt_expr,
            ParamDecl: self._stmt_noop,
        })

    # ── reading and writing a name ──────────────────────────────────────────
    def _mf_read_ref(self, root: str, in_env: bool) -> str:
        """A read of *root* that is safe before the name is known to exist — the emit-time
        equivalent of the interpreter's `store.get(root)`, which answers `None` rather
        than raising for a name no statement has written yet."""
        if not in_env:
            return f"_bind.get({root!r})"
        local = self._local_vars.get(root)
        return local if local is not None else f"_env.get({root!r})"

    def _mf_write_ref(self, root: str, in_env: bool) -> str:
        return self._var_target(root) if in_env else f"_bind[{root!r}]"

    # ── declarations (M1's "declared inside this region") ───────────────────
    def _mfe_var_decl(self, stmt: VarDecl):
        self._emit_var_decl(stmt)
        self._mf_decl_depth[stmt.name] = self._mf_depth

    def _mfe_array_decl(self, stmt: ArrayDecl):
        self._emit_array_decl(stmt)
        self._mf_decl_depth[stmt.name] = self._mf_depth

    # ── M1: writes ──────────────────────────────────────────────────────────
    def _mfe_assignment(self, stmt: Assignment):
        """`target := where(live, new_value, target)` for a target declared OUTSIDE the
        innermost region containing the write; an unmasked store otherwise.

        Snapshot-then-select around the `0.23` write, exactly as the interpreter does it
        and for the same reason: a channel write, an array-index write and a plain store
        all end by putting one whole tensor back under one name, so there is exactly one
        place for the selection to be wrong.

        The "declared inside this region" test is resolved HERE, at emit time, against
        `_mf_decl_depth` — the emit-time shadow of the interpreter's per-cook `_decl_depth`.
        Regions are lexical, so the two always agree; what they agree on is that a
        loop-header counter and a body-local temporary write unmasked."""
        target = stmt.target
        if type(target) is BindingIndexAccess:
            # M5 gates a scatter by SOURCE, not by destination: `_emit_scatter_write`
            # compacts the live sources itself.
            value_expr = self._emit_expr(stmt.value)
            self._emit_scatter_write(target, value_expr, op=stmt.op)
            return

        root, in_env = _root_of(target)
        if root is None or (in_env and self._mf_decl_depth.get(root, -1) >= self._mf_depth):
            self._emit_assignment(stmt)
            return

        read_ref = self._mf_read_ref(root, in_env)
        write_ref = self._mf_write_ref(root, in_env)
        before = self._tmp()
        self._emit(f"{before} = {read_ref}")
        self._emit(f"{before} = {before}.clone() if _torch.is_tensor({before}) else {before}")
        self._emit_assignment(stmt)
        self._emit(f"{write_ref} = _MF.merge_write({_STATE}.live, {write_ref}, {before})")
        # The name now holds a `torch.where` result rather than the buffer an in-place
        # channel write may have claimed; re-asserting ownership across a masked write
        # would licence eliding the next clone on a tensor this statement did not allocate.
        self._owned.discard(root)

    # ── M2: `if` ────────────────────────────────────────────────────────────
    def _mfe_if_else(self, stmt: IfElse):
        self._owned.clear()
        self._mf_emit_if_else(stmt)
        self._owned.clear()

    def _mf_emit_if_else(self, stmt: IfElse):
        """A 0-dim condition short-circuits exactly as in `0.23`; a per-pixel condition
        keeps `0.23`'s both-branches-then-merge model and adds the branch live masks.

        Both paths are emitted because which one runs is a run-time property of the
        condition's rank, which is how the `0.23` emitter already handles it."""
        cond_expr = self._emit_expr(stmt.condition)
        cond_tmp = self._tmp()
        self._emit(f"{cond_tmp} = {cond_expr}")
        hoist_snap = dict(self._hoisted_bchw)

        self._emit(f"if not _torch.is_tensor({cond_tmp}) or {cond_tmp}.dim() == 0:")
        self._indent += 1
        self._emit(f"if float({cond_tmp}) > 0.5:")
        self._indent += 1
        self._hoisted_bchw = dict(hoist_snap)
        if stmt.then_body:
            for s in stmt.then_body:
                self._emit_stmt(s)
        else:
            self._emit("pass")
        self._indent -= 1
        if stmt.else_body:
            self._emit("else:")
            self._indent += 1
            self._hoisted_bchw = dict(hoist_snap)
            for s in stmt.else_body:
                self._emit_stmt(s)
            self._indent -= 1
        self._indent -= 1

        self._emit("else:")
        self._indent += 1
        self._mf_emit_spatial_if(stmt, cond_tmp, hoist_snap)
        self._indent -= 1
        self._hoisted_bchw = hoist_snap

    def _mf_emit_spatial_if(self, stmt: IfElse, cond_var: str,
                            hoist_snap: dict):
        saved = self._tmp()
        cm = self._tmp()
        self._emit(f"{saved} = {_STATE}.live")
        self._emit(f"{cm} = _MF.cond_mask({cond_var})")

        then_mods = self._collect_modified_vars(stmt.then_body)
        else_mods = (self._collect_modified_vars(stmt.else_body)
                     if stmt.else_body else (set(), set()))
        # Sorted so emission order — and the repr() baked into the source — is
        # deterministic across processes.
        env_keys = sorted(then_mods[0] | else_mods[0])
        bind_keys = sorted(then_mods[1] | else_mods[1])

        # Snapshot (the interpreter's `_snapshot_vars`: clone a tensor, copy anything
        # else, and leave a name that does not exist out of the dict altogether).
        snap: dict[str, str] = {}
        for k in env_keys:
            s = self._tmp()
            snap[k] = s
            src = self._mf_read_ref(k, True)
            self._emit(f"{s} = {src}.clone() if ({src} is not None and _torch.is_tensor({src})) else {src}")
        snap_bind = self._tmp()
        bind_repr = repr(bind_keys)
        self._emit(f"{snap_bind} = {{k: _bind[k].clone() if _torch.is_tensor(_bind[k]) "
                   f"else _bind[k] for k in {bind_repr} if k in _bind}}")

        # then-branch, under `live & cond`
        self._emit(f"{_STATE}.live = _MF.m_and({saved}, {cm})")
        self._hoisted_bchw = dict(hoist_snap)
        for s in stmt.then_body:
            self._emit_stmt(s)
        then_env = self._tmp()
        self._emit(f"{then_env} = {{" + ", ".join(
            f"{k!r}: {self._mf_read_ref(k, True)}" for k in env_keys) + "}")
        then_bind = self._tmp()
        self._emit(f"{then_bind} = {{k: _bind.get(k) for k in {bind_repr}}}")

        # restore the snapshot
        for k in env_keys:
            self._emit(f"{self._var_target(k)} = {snap[k]}")
        self._emit(f"_bind.update({snap_bind})")

        if stmt.else_body:
            # `m_sub(saved, dead_now())`: a transfer taken in the THEN arm has already
            # left the enclosing region, so those pixels are not offered to the else arm.
            self._emit(f"{_STATE}.live = _MF.m_and(_MF.m_sub({saved}, {_STATE}.dead_now()), ~{cm})")
            self._hoisted_bchw = dict(hoist_snap)
            for s in stmt.else_body:
                self._emit_stmt(s)
            else_env = self._tmp()
            self._emit(f"{else_env} = {{" + ", ".join(
                f"{k!r}: {self._mf_read_ref(k, True)}" for k in env_keys) + "}")
            else_bind = self._tmp()
            self._emit(f"{else_bind} = {{k: _bind.get(k) for k in {bind_repr}}}")
        else:
            else_env = self._tmp()
            self._emit(f"{else_env} = {{" + ", ".join(
                f"{k!r}: {snap[k]}" for k in env_keys) + "}")
            else_bind = snap_bind

        self._emit(f"{_STATE}.restore({saved})")

        # The merge is `Interpreter._merge_branch_vars` itself (see
        # `masked_flow.cg_merge_branch`) — the string majority vote and the `torch.where`
        # have one implementation, not two.
        box = self._tmp()
        self._emit(f"{box} = []")
        if env_keys:
            merged = self._tmp()
            self._emit(f"{merged} = {{}}")
            self._emit(f"_MF.cg_merge_branch({cm}, {box}, {merged}, {env_keys!r}, "
                       f"{then_env}, {else_env})")
            for k in env_keys:
                self._emit(f"if {k!r} in {merged}: {self._var_target(k)} = {merged}[{k!r}]")
            self._spatial_vars.update(env_keys)
        if bind_keys:
            self._emit(f"_MF.cg_merge_branch({cm}, {box}, _bind, {bind_keys!r}, "
                       f"{then_bind}, {else_bind})")
        if not env_keys and not bind_keys:
            self._emit("pass")

    # ── M3: loops ───────────────────────────────────────────────────────────
    def _mf_emit_pass(self, body, live_var: str):
        """One pass of a loop body, under *live_var*. Its own `"pass"` frame, so a
        `continue` clears for the rest of THIS pass and the bit is restored at the next
        condition evaluation (M3.4)."""
        saved_live = self._tmp()
        self._emit(f"{_STATE}.push('pass')")
        self._emit(f"{saved_live} = {_STATE}.live")
        self._emit(f"{_STATE}.live = {live_var}")
        self._emit("try:")
        self._indent += 1
        self._mf_depth += 1
        if body:
            for s in body:
                self._emit_stmt(s)
        else:
            self._emit("pass")
        self._mf_depth -= 1
        self._indent -= 1
        self._emit("finally:")
        self._indent += 1
        self._emit(f"{_STATE}.pop()")
        self._emit(f"{_STATE}.live = {saved_live}")
        self._indent -= 1

    def _mfe_for_loop(self, stmt: ForLoop):
        self._owned.clear()
        self._mf_emit_for_loop(stmt)
        self._owned.clear()

    def _mfe_while_loop(self, stmt: WhileLoop):
        self._owned.clear()
        self._mf_emit_while_loop(stmt)
        self._owned.clear()

    def _mf_loop_prologue(self) -> str:
        entry = self._tmp()
        self._emit(f"{entry} = {_STATE}.live")
        self._emit(f"{_STATE}.push('loop')")
        self._emit("try:")
        self._indent += 1
        return entry

    def _mf_loop_epilogue(self, entry: str):
        self._indent -= 1
        self._emit("finally:")
        self._indent += 1
        self._emit(f"{_STATE}.pop()")
        self._emit(f"{_STATE}.restore({entry})")
        self._indent -= 1

    def _mf_emit_for_loop(self, stmt: ForLoop):
        """`docs/masked-control-flow.md` M3, for a `for`.

        No stencil specialisation and no scalar-loop mode (see the module docstring); the
        static-range split is the SAME `try_extract_static_range` the interpreter's masked
        driver consults first, so the two tiers take the same branch for the same loop."""
        static_range = try_extract_static_range(stmt)
        entry = self._mf_loop_prologue()
        if static_range is not None:
            self._mf_emit_static_for(stmt, static_range, entry)
        else:
            self._mf_emit_general_for(stmt, entry)
        self._mf_loop_epilogue(entry)

    def _mf_emit_static_for(self, stmt: ForLoop, static_range: tuple, entry: str):
        loop_var, start, stop, step = static_range
        n = abs(stop - start) // max(abs(step), 1)
        if n > 1024:
            self._emit("raise RuntimeError('For loop would exceed 1024 iterations')")
            return

        modified_vars, _ = self._collect_modified_vars(stmt.body)
        read_vars = self._collect_read_vars(stmt.body)
        all_vars = sorted(modified_vars | read_vars | {loop_var})
        writeback_vars = sorted(modified_vars | {loop_var})
        saved_locals = {}
        for vname in all_vars:
            prev = self._local_vars.get(vname)
            saved_locals[vname] = prev
            if prev is None:
                self._local_vars[vname] = f"_lv_{vname}"
        loop_var_local = self._local_vars[loop_var]
        for vname in all_vars:
            if vname != loop_var and saved_locals[vname] is None:
                self._emit(f"{self._local_vars[vname]} = _env.get({vname!r})")

        self._setup_tensor_loop(start, stop, step)
        vals_tmp = self._range_cache[(start, stop, step)]

        # M3.5: the loop-header counter is declared BY the loop, so it stays uniform —
        # an unmasked store at the loop's own region depth.
        saved_decl = self._mf_decl_depth.get(loop_var, None)
        self._mf_decl_depth[loop_var] = self._mf_depth

        live = self._tmp()
        self._emit(f"{live} = {entry}")
        self._emit(f"for _i_idx in range({n}):")
        self._indent += 1
        self._emit(f"{loop_var_local} = {vals_tmp}[_i_idx]")
        # `dead_now()`, not just this loop's own frame: a `return` taken inside the body
        # clears the pixel for the rest of the CALL, which includes every later pass here.
        self._emit(f"{live} = _MF.m_sub({live}, {_STATE}.dead_now())")
        self._emit(f"if not _MF.m_any({live}): break")
        self._mf_emit_pass(stmt.body, live)
        self._indent -= 1

        if saved_decl is None:
            self._mf_decl_depth.pop(loop_var, None)
        else:
            self._mf_decl_depth[loop_var] = saved_decl

        for vname in writeback_vars:
            if saved_locals[vname] is None:
                self._emit(f"_env[{vname!r}] = {self._local_vars[vname]}")
        for vname, prev in saved_locals.items():
            if prev is None:
                self._local_vars.pop(vname, None)
            else:
                self._local_vars[vname] = prev

    def _mf_emit_general_for(self, stmt: ForLoop, entry: str):
        self._emit_stmt(stmt.init)
        live = self._tmp()
        iter_var = self._tmp()
        self._emit(f"{live} = {entry}")
        self._emit(f"{iter_var} = 0")
        self._emit(f"while {iter_var} < _MAX_ITER:")
        self._indent += 1
        self._mf_emit_cond_narrow(stmt.condition, live)
        self._emit(f"{live} = _MF.m_sub({live}, {_STATE}.dead_now())")
        self._emit(f"if not _MF.m_any({live}): break")
        self._mf_emit_pass(stmt.body, live)
        self._emit(f"{live} = _MF.m_sub({live}, {_STATE}.dead_now())")
        self._emit_stmt(stmt.update)
        self._emit(f"{iter_var} += 1")
        self._indent -= 1
        self._emit_iter_limit(iter_var, "For")

    def _mf_emit_while_loop(self, stmt: WhileLoop):
        entry = self._mf_loop_prologue()
        live = self._tmp()
        iter_var = self._tmp()
        self._emit(f"{live} = {entry}")
        self._emit(f"{iter_var} = 0")
        self._emit(f"while {iter_var} < _MAX_ITER:")
        self._indent += 1
        self._mf_emit_cond_narrow(stmt.condition, live)
        self._emit(f"{live} = _MF.m_sub({live}, {_STATE}.dead_now())")
        self._emit(f"if not _MF.m_any({live}): break")
        self._mf_emit_pass(stmt.body, live)
        self._emit(f"{live} = _MF.m_sub({live}, {_STATE}.dead_now())")
        self._emit(f"{iter_var} += 1")
        self._indent -= 1
        # The cap still fires for a pixel that never terminates. Masking must not turn a
        # runaway loop into a silent one.
        self._emit_iter_limit(iter_var, "While")
        self._mf_loop_epilogue(entry)

    def _mf_emit_cond_narrow(self, cond_node, live_var: str):
        """M3.1 — evaluate the condition and narrow the live mask by it.

        A uniform (0-dim) bound inside a `0.25` program keeps `0.23`'s reading exactly:
        no mask is narrowed and no tensor work is added. A per-pixel bound narrows through
        `cond_mask`, the shared predicate."""
        cond_expr = self._emit_expr(cond_node)
        cc = self._tmp()
        self._emit(f"{cc} = {cond_expr}")
        self._emit(f"if not _torch.is_tensor({cc}) or {cc}.dim() == 0:")
        self._indent += 1
        self._emit(f"if not (float({cc}) > 0.5): break")
        self._indent -= 1
        self._emit("else:")
        self._indent += 1
        self._emit(f"{live_var} = _MF.m_and({live_var}, _MF.cond_mask({cc}))")
        self._indent -= 1

    # ── M3.4 / M4: the transfers ────────────────────────────────────────────
    def _mfe_break(self, stmt):
        self._emit(f"_MF.cg_break({_STATE})")

    def _mfe_continue(self, stmt):
        self._emit(f"_MF.cg_continue({_STATE})")

    def _mfe_return(self, stmt: ReturnStmt):
        """M4: record the value for the pixels live at this statement and clear their bits
        for the remainder of the call body — NOT a native Python `return`, which would take
        the pixels that stayed with it."""
        value = self._emit_expr(stmt.value) if stmt.value is not None else "None"
        self._emit(f"_MF.cg_return({_STATE}, {value})")

    # ── M4: calls ───────────────────────────────────────────────────────────
    def _mfe_function_def(self, stmt: FunctionDef):
        self._owned.clear()
        self._mf_emit_function_def(stmt)
        self._owned.clear()

    def _mf_emit_function_def(self, stmt: FunctionDef):
        """The `0.23` nested `def`, plus the call region M4 makes it.

        The `"call"` frame is pushed INSIDE the emitted function rather than at the call
        site, which is the one structural difference from the interpreter — and it is a
        difference of where the code is written, not of what happens: the frame's lifetime
        is still exactly the call's, and the caller's live mask is inherited untouched."""
        from .interpreter import MAX_CALL_DEPTH

        self._user_functions.add(stmt.name)
        params = [f"_p_{pname}" for _, pname in stmt.params]
        params_str = ", ".join(params + ["_depth=0"])
        self._emit(f"def _uf_{stmt.name}({params_str}):")
        self._indent += 1
        self._emit(f"if _depth > {MAX_CALL_DEPTH}: raise RuntimeError("
                   f"'Maximum function call depth exceeded in {stmt.name}()')")

        saved_locals = self._local_vars
        self._local_vars = {}
        for _, pname in stmt.params:
            self._local_vars[pname] = f"_p_{pname}"
        body_vars, _ = self._collect_modified_vars(stmt.body)
        for vname in sorted(body_vars):
            if vname not in self._local_vars:
                self._local_vars[vname] = f"_uf_lv_{vname}"

        saved_in_fn = self._in_user_function
        self._in_user_function = True
        saved_native_flow = self._use_native_flow_control
        self._use_native_flow_control = False
        saved_scalar_loop = self._scalar_loop
        self._scalar_loop = False

        # The body is its own declaration scope (the type checker gives it one), so the
        # inherited depths are shadowed rather than shared; the params are declared AT the
        # body's depth, which is what makes a write to a parameter unmasked.
        saved_decl = self._mf_decl_depth
        self._mf_decl_depth = dict(saved_decl)
        saved_depth = self._mf_depth
        self._mf_depth = saved_depth + 1
        for _, pname in stmt.params:
            self._mf_decl_depth[pname] = self._mf_depth

        frame = self._tmp()
        saved_live = self._tmp()
        self._emit(f"{frame} = {_STATE}.push('call')")
        self._emit(f"{saved_live} = {_STATE}.live")
        self._emit("try:")
        self._indent += 1
        for s in stmt.body:
            self._emit_stmt(s)
        self._emit(f"return _MF.cg_call_result({frame}, _dev)")
        self._indent -= 1
        self._emit("finally:")
        self._indent += 1
        self._emit(f"{_STATE}.pop()")
        self._emit(f"{_STATE}.restore({saved_live})")
        self._indent -= 1

        self._mf_depth = saved_depth
        self._mf_decl_depth = saved_decl
        self._scalar_loop = saved_scalar_loop
        self._use_native_flow_control = saved_native_flow
        self._in_user_function = saved_in_fn
        self._local_vars = saved_locals
        self._indent -= 1

    def _mf_emit_user_call(self, node) -> str:
        """M4's empty-call skip: a call with NO live pixel is skipped entirely.

        The ARGUMENTS are emitted inside the live branch, not before it, because the
        interpreter evaluates them after its own skip test — and an argument can carry a
        scatter or an `@` write, which is the only way the skip is observable at all."""
        tmp = self._tmp()
        self._fresh_temps.clear()
        self._emit(f"if _MF.m_any({_STATE}.live):")
        self._indent += 1
        args = [self._emit_expr(a) for a in node.args]
        depth_arg = ", _depth=_depth+1" if self._in_user_function else ""
        self._emit(f"{tmp} = _uf_{node.name}({', '.join(args)}{depth_arg})")
        self._indent -= 1
        self._emit("else:")
        self._indent += 1
        self._emit(f"{tmp} = _MF.cg_skip_call(_dev)")
        self._indent -= 1
        self._fresh_temps.clear()
        return tmp
