"""Masked per-pixel control flow — the interpreter's language-`0.25` rules (M1–M7).

`docs/masked-control-flow.md` §1 is the specification; this module is its interpreter-side
implementation, and the one that a later codegen tier must reproduce bit-exactly.

**The rule, in one sentence.** Every active region carries a per-pixel *live* mask, and a
write to anything declared outside that region becomes a masked select, so a pixel that has
left keeps the value it left with. A per-pixel `break`/`continue`/`return` clears mask bits
instead of unwinding; a loop keeps running while ANY pixel is live, so the pass count stays
the region's maximum while the values stop changing.

**Why it lives beside the interpreter rather than inside it.** Everything here runs only
for a program the language gate has flagged, so it is a second, parallel statement
dispatch — the shape `docs/masked-control-flow.md` §5 asks for, bound per cook and restored
in a `finally`. Keeping it out of `interpreter.py` means the default tree-walk keeps every
one of its handlers byte-identical: a program below `0.25` never reaches a line of this
file, and the module the hot path lives in does not grow.

**The gate is the ENGINE's capability, not the source's request** (`docs/masked-control-flow.md`
§4, and the v0.36.0 review that had to be corrected before it shipped). `enabled_for` asks
`tex_roi._language_tuple`, which is `min(what the program asks for, what the engine
implements)` — the SAME single definition the region-dependence sunset keys on, never a
second one. While `tex_api.LANGUAGE_VERSION` is below `0.25` that minimum is below `0.25`
for every program that exists, so nothing reaches the masked path at all.

**Mask representation.** `True` (the Python bool) means every pixel is live and is the
program's starting mask; anything else is a `torch.bool` tensor broadcastable to the cook
grid. The `True` spelling is not a micro-optimisation — it is what keeps a region with a
statically-full mask doing exactly the tensor work `0.23` does, which is the same tri-state
the codegen mirror needs so that such a region can emit the source it emits today.
"""
from __future__ import annotations

import torch

from ..tex_compiler.ast_nodes import (
    Assignment, ArrayDecl, BindingIndexAccess, BindingRef, BreakStmt, ChannelAccess,
    ContinueStmt, ExprStatement, ForLoop, FunctionDef, Identifier, IfElse,
    ArrayIndexAccess, ParamDecl, ReturnStmt, VarDecl, WhileLoop,
)

__all__ = ["cond_mask", "enabled_for", "MaskedFlowMixin"]


# ── the mask algebra ────────────────────────────────────────────────────────────

def cond_mask(cond: torch.Tensor) -> torch.Tensor:
    """THE single definition of "which pixels take this condition".

    `docs/masked-control-flow.md` §5 names the predicate's spelling as the highest-risk
    line in the release: the interpreter's per-pixel `if` has always read
    `(cond > 0.5) if cond.is_floating_point() else cond.bool()` while codegen emits
    `(cond > 0.5)` unconditionally, and the two disagree for an integer condition whose
    value is non-zero and below `1` (a negative int is the reachable case). The design's
    fix is one shared helper imported by both tiers; this is it, and every mask in this
    module is derived through it.

    It adopts the INTERPRETER's existing spelling deliberately, for two reasons. The
    masked `if` uses one mask both to gate the branch and to select the merge, so a
    different predicate here would make a `0.25` per-pixel `if` merge differently from a
    `0.23` one for no reason the language states. And the interpreter is the oracle:
    changing its reading would move a `0.23` program's answer, which invariant 7 forbids.
    A float condition — which is what every TEX comparison produces (`_eval_binop`
    answers `(l > r).float()`) — takes `> 0.5` on both tiers, so NaN reads False on both.
    """
    if cond.dtype is torch.bool:
        return cond
    if cond.is_floating_point():
        return cond > 0.5
    return cond.bool()


def m_and(a, b):
    """Narrow mask `a` by mask `b`."""
    if a is True:
        return b
    if b is True:
        return a
    return a & b


def m_or(a, b):
    """Union of two masks, either of which may be `None` (no pixels)."""
    if a is None:
        return b
    if b is None:
        return a
    if a is True or b is True:
        return True
    return a | b


def m_sub(a, dead):
    """Mask `a` with the pixels in `dead` removed. `dead is None` means none left."""
    if dead is None:
        return a
    if dead is True:
        return False
    if a is True:
        return ~dead
    if a is False:
        return a
    return a & ~dead


def m_any(a) -> bool:
    """Is ANY pixel live? This is `docs/masked-control-flow.md` §5's divergence site 2 —
    the loop-exit test — so the codegen mirror must test the same expression on the same
    tensor, not an equivalent one."""
    if a is True:
        return True
    if a is False:
        return False
    return bool(a.any().item())


class _Frame:
    """One region whose exit a transfer can cause.

    `kind` is `"loop"` (a `break` clears for the rest of it), `"pass"` (a `continue`
    clears for the rest of it), or `"call"` (a `return` clears for the rest of it, and it
    is also the boundary a mask restoration stops at — a callee's returns must never kill
    a caller's pixels). `dead` accumulates the pixels that have left."""

    __slots__ = ("kind", "dead", "ret")

    def __init__(self, kind: str):
        self.kind = kind
        self.dead = None
        self.ret = None


# ── the language gate ───────────────────────────────────────────────────────────

def enabled_for(program, source: str) -> bool:
    """Does this cook run under the `0.25` rules?

    Two independent conditions, and BOTH must hold:

      1. the effective language level — `min(pragma, LANGUAGE_VERSION)`, read from
         `tex_roi._language_tuple`, which is the one definition the region-dependence
         sunset already keys on — is at least `MASKED_FLOW_SINCE`; and
      2. `tex_api.flow_plan` names at least one masking-relevant site.

    (2) is what keeps the cost off programs that need no masking: the plan is empty for
    129 of the 130 corpus programs, and an empty plan means the masked dispatch table is
    never bound, so such a program runs the `0.23` handlers unchanged even when it
    declares the pragma. An INCOMPLETE plan (`complete=False`) is never empty, so an
    analysis that could not finish masks rather than skips — the fail-closed direction.

    Pure and total: anything unexpected answers False, which is `0.23`'s behaviour and
    therefore the only safe direction while the engine's own `LANGUAGE_VERSION` is below
    `0.25` anyway.

    THE `Program.language is None` FAST-OUT IS LOAD-BEARING FOR INVARIANT 7, not tidiness.
    `tex_roi._language_tuple` falls back to `tex_api.language_pragma(code)` when the field
    is absent, and that scan calls `source.splitlines()` — building a list of every line of
    the program, on every cook, for a cook that can never mask. A `Program` whose
    `.language` is None asked for nothing, so its effective level is `(0, 0)`, which is
    below `MASKED_FLOW_SINCE` whatever `LANGUAGE_VERSION` says; answering False from the
    attribute alone is the same answer for a fraction of the work. LANG-L1's acceptance is
    what makes it the same answer: the pragma round-trips onto `Program.language` through
    every source→AST path there is, so a program that declares one never arrives here with
    the field unset."""
    if getattr(program, "language", None) is None:
        return False
    try:
        from .. import tex_api, tex_roi
        if tex_roi._language_tuple(program, source) < tex_roi.MASKED_FLOW_SINCE:
            return False
        return not tex_api.flow_plan(program).is_empty()
    except Exception:
        return False


class MaskedFlowMixin:
    """The `0.25` statement handlers, mixed into `Interpreter`.

    Nothing here overrides a `0.23` method. Each handler is reached only through the
    second dispatch table `_mf_enter` binds, so the unmasked tree-walk is untouched."""

    # ── per-cook set-up / tear-down ─────────────────────────────────────────
    def _mf_enter(self, program):
        """Bind the `0.25` dispatch table for this cook. Returns the token `_mf_leave`
        needs, so the caller can restore in a `finally` exactly as `_cancel` does."""
        table = getattr(self, "_masked_dispatch", None)
        if table is None:
            table = {
                VarDecl: self._mf_var_decl,
                ArrayDecl: self._mf_array_decl,
                Assignment: self._mf_assignment,
                IfElse: self._mf_if_else,
                ForLoop: self._mf_for_loop,
                WhileLoop: self._mf_while_loop,
                ExprStatement: lambda node: self._eval(node.expr),
                ParamDecl: lambda node: None,
                FunctionDef: self._exec_function_def,
                ReturnStmt: self._mf_return,
                BreakStmt: self._mf_break,
                ContinueStmt: self._mf_continue,
            }
            self._masked_dispatch = table
        token = (self._stmt_dispatch, self.functions, self._masked)
        self._stmt_dispatch = table
        self._masked = True
        self._live = True
        self._frames = []
        self._decl_depth = {}
        self._region_depth = 0
        # M7: a probe records only if its probe pixel is live. Swapping ONE entry of a
        # per-cook copy of the registry keeps `_eval_function_call` byte-identical — the
        # probe is the rare case and must not put a branch on the call path.
        if "debug_print" in self.functions:
            fns = dict(self.functions)
            fns["debug_print"] = self._mf_debug_print
            self.functions = fns
        return token

    def _mf_leave(self, token):
        self._stmt_dispatch, self.functions, self._masked = token

    # ── mask plumbing ───────────────────────────────────────────────────────
    def _mf_dead_now(self):
        """Every pixel that has left a region enclosing the current statement, up to and
        including the innermost call. Stops at the call because a callee's `return` bits
        belong to the callee: the caller's pixels are live again the moment the call
        returns."""
        dead = None
        for f in reversed(self._frames):
            if f.dead is not None:
                dead = f.dead if dead is None else (dead | f.dead)
            if f.kind == "call":
                break
        return dead

    def _mf_restore(self, saved):
        """Leave a nested region: go back to `saved`, minus whatever left while inside.
        Subtracting bits already absent from `saved` is a no-op, so this is correct
        however many transfers fired at whatever depth."""
        self._live = m_sub(saved, self._mf_dead_now())

    def _mf_frame(self, kind):
        for f in reversed(self._frames):
            if f.kind == kind:
                return f
            if f.kind == "call" and kind != "call":
                return None          # E3015 makes this unreachable; answer honestly anyway
        return None

    # ── declarations (M1's "declared inside this region") ───────────────────
    def _mf_var_decl(self, node):
        self._exec_var_decl(node)
        self._decl_depth[node.name] = self._region_depth

    def _mf_array_decl(self, node):
        self._exec_array_decl(node)
        self._decl_depth[node.name] = self._region_depth

    # ── M1: writes ──────────────────────────────────────────────────────────
    @staticmethod
    def _mf_root(target):
        """The name an assignment ultimately stores into, and which store holds it."""
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

    def _mf_assignment(self, node):
        """`target := where(live, new_value, target)` for a target declared OUTSIDE the
        innermost region containing the write; an unmasked store otherwise.

        Implemented as snapshot-then-select around the `0.23` write rather than as a
        per-target-shape masked store. That is deliberate: a channel write, an array-index
        write and a plain store all end by putting a whole tensor back under one name, so
        selecting on that one tensor is the same rule for all three and there is exactly
        one place for it to be wrong. It costs one clone per masked write, on `0.25`
        programs only.

        A scatter is the exception and gets its own path, because M5 gates it by SOURCE:
        selecting on the DESTINATION buffer would keep the wrong pixels."""
        target = node.target
        if type(target) is BindingIndexAccess:
            value = self._eval(node.value)
            return self._exec_scatter_write(target, value, op=node.op, live=self._live)
        live = self._live
        if live is True:
            return self._exec_assignment(node)
        root, in_env = self._mf_root(target)
        if root is None:
            return self._exec_assignment(node)
        if in_env and self._decl_depth.get(root, -1) >= self._region_depth:
            # Declared inside this very region: a loop-header counter or a body-local
            # temporary, dead on exit. M1 leaves it unmasked.
            return self._exec_assignment(node)
        store = self.env if in_env else self.bindings
        before = store.get(root)
        if not isinstance(before, torch.Tensor):
            # M7: a string (or a string array) has no per-pixel representation and keeps
            # `0.23`'s majority-vote reading verbatim. A name that does not exist yet has
            # nothing for a departed pixel to keep.
            return self._exec_assignment(node)
        before = before.clone()
        self._exec_assignment(node)
        after = store.get(root)
        if isinstance(after, torch.Tensor):
            from .interpreter import _tensor_where
            store[root] = _tensor_where(live, after, before)
            self._inplace_ready.discard(root)
        return None

    # ── M2: `if` ────────────────────────────────────────────────────────────
    def _mf_if_else(self, node):
        """A 0-dim condition short-circuits exactly as in `0.23`. A per-pixel condition
        keeps `0.23`'s both-branches-then-merge model and adds the branch live masks, so
        a transfer taken inside a branch clears bits in the enclosing region's mask
        instead of unwinding past the merge."""
        cond = self._eval(node.condition)
        if cond.dim() == 0:
            body = node.then_body if cond.item() > 0.5 else (node.else_body or ())
            for stmt in body:
                self._exec_stmt(stmt)
            return None

        saved = self._live
        cm = cond_mask(cond)

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

        env_snapshot = self._snapshot_vars(modified_env, self.env)
        bindings_snapshot = self._snapshot_vars(modified_bindings, self.bindings)
        has_arrays = bool(self._array_meta)
        if has_arrays:
            meta_snapshot = dict(self._array_meta)

        self._live = m_and(saved, cm)
        for stmt in node.then_body:
            self._exec_stmt(stmt)
        then_env = {k: self.env.get(k) for k in modified_env}
        then_bindings = {k: self.bindings.get(k) for k in modified_bindings}
        if has_arrays:
            then_meta = dict(self._array_meta)

        self.env.update(env_snapshot)
        self.bindings.update(bindings_snapshot)
        if has_arrays:
            self._array_meta = dict(meta_snapshot)

        if node.else_body:
            self._live = m_and(m_sub(saved, self._mf_dead_now()), ~cm)
            for stmt in node.else_body:
                self._exec_stmt(stmt)
            else_env = {k: self.env.get(k) for k in modified_env}
            else_bindings = {k: self.bindings.get(k) for k in modified_bindings}
        else:
            else_env = env_snapshot
            else_bindings = bindings_snapshot

        self._mf_restore(saved)

        cond_scalar_box: list = []
        self._merge_branch_vars(cm, cond_scalar_box,
                                self.env, modified_env, then_env, else_env)
        self._merge_branch_vars(cm, cond_scalar_box,
                                self.bindings, modified_bindings,
                                then_bindings, else_bindings)
        if has_arrays:
            self._array_meta.update(then_meta)
        return None

    # ── M3: loops ───────────────────────────────────────────────────────────
    def _mf_pass(self, body, live):
        """One pass of a loop body, under `live`. Its own frame, so a `continue` clears
        for the rest of THIS pass and the bit is restored at the next condition."""
        frame = _Frame("pass")
        self._frames.append(frame)
        saved_live, saved_depth = self._live, self._region_depth
        self._live = live
        self._region_depth = saved_depth + 1
        try:
            for stmt in body:
                self._exec_stmt(stmt)
        finally:
            self._frames.pop()
            self._region_depth = saved_depth
            self._live = saved_live

    def _mf_for_loop(self, node):
        from .interpreter import MAX_LOOP_ITERATIONS, InterpreterError
        entry = self._live
        frame = _Frame("loop")
        self._frames.append(frame)
        try:
            static_range = self._try_extract_static_range(node)
            if static_range is None:
                static_range = self._try_resolve_uniform_range(node)

            if static_range is not None:
                loop_var, iter_range = static_range
                if len(iter_range) > MAX_LOOP_ITERATIONS:
                    raise InterpreterError(
                        f"This for loop would run {len(iter_range)} iterations, which "
                        f"exceeds the limit of {MAX_LOOP_ITERATIONS}",
                        node.loc, source=self._source, code="E6010",
                        hint=f"Loops are capped at {MAX_LOOP_ITERATIONS} iterations to "
                             "prevent hangs. Consider reducing your range or processing "
                             "in smaller batches.",
                    )
                n = len(iter_range)
                loop_tensors = torch.arange(
                    iter_range.start, iter_range.start + n * iter_range.step,
                    iter_range.step, dtype=self._dtype, device=self.device)
                live = entry
                # M3.5: the loop-header counter is declared BY the loop, so it stays
                # uniform — an unmasked store at the loop's own region depth.
                self._decl_depth[loop_var] = self._region_depth
                for val in loop_tensors.unbind(0):
                    self.env[loop_var] = val
                    self._inplace_ready.discard(loop_var)
                    # `_mf_dead_now()`, not just this loop's own `dead`: a `return` taken
                    # inside the body clears the pixel for the rest of the CALL, which
                    # includes every later pass of this loop. Reading only the loop frame
                    # let a returned pixel keep running the body (measured: `return`
                    # inside a loop diverged from the oracle by a whole count).
                    live = m_sub(live, self._mf_dead_now())
                    if not m_any(live):
                        break
                    self._mf_pass(node.body, live)
                return None

            self._exec_stmt(node.init)
            if type(node.init) is VarDecl:
                self._decl_depth[node.init.name] = self._region_depth
            live = entry
            iteration = 0
            while iteration < MAX_LOOP_ITERATIONS:
                cond = self._eval(node.condition)
                if cond.dim() == 0:
                    # A uniform bound inside a `0.25` program keeps `0.23`'s reading
                    # exactly: no mask is narrowed and no tensor work is added.
                    if not (cond.item() > 0.5):
                        break
                else:
                    live = m_and(live, cond_mask(cond))
                live = m_sub(live, self._mf_dead_now())
                if not m_any(live):
                    break
                self._mf_pass(node.body, live)
                live = m_sub(live, self._mf_dead_now())
                self._exec_stmt(node.update)
                iteration += 1
            if iteration >= MAX_LOOP_ITERATIONS:
                self._raise_loop_limit("for", node.loc)
            return None
        finally:
            self._frames.pop()
            self._mf_restore(entry)

    def _mf_while_loop(self, node):
        from .interpreter import MAX_LOOP_ITERATIONS
        entry = self._live
        frame = _Frame("loop")
        self._frames.append(frame)
        try:
            live = entry
            iteration = 0
            while iteration < MAX_LOOP_ITERATIONS:
                cond = self._eval(node.condition)
                if cond.dim() == 0:
                    if not (cond.item() > 0.5):
                        break
                else:
                    live = m_and(live, cond_mask(cond))
                live = m_sub(live, self._mf_dead_now())
                if not m_any(live):
                    break
                self._mf_pass(node.body, live)
                live = m_sub(live, self._mf_dead_now())
                iteration += 1
            if iteration >= MAX_LOOP_ITERATIONS:
                # The cap still fires for a pixel that never terminates. Masking must not
                # turn a runaway loop into a silent one.
                self._raise_loop_limit("while", node.loc)
            return None
        finally:
            self._frames.pop()
            self._mf_restore(entry)

    # ── M3.4 / M4: the transfers ────────────────────────────────────────────
    def _mf_break(self, node):
        frame = self._mf_frame("loop")
        if frame is None:
            from .interpreter import _Break
            raise _Break()
        frame.dead = m_or(frame.dead, self._live)
        self._live = m_sub(self._live, frame.dead)
        return None

    def _mf_continue(self, node):
        frame = self._mf_frame("pass")
        if frame is None:
            from .interpreter import _Continue
            raise _Continue()
        frame.dead = m_or(frame.dead, self._live)
        self._live = m_sub(self._live, frame.dead)
        return None

    def _mf_return(self, node):
        """M4: record `e` for the pixels live at this statement and clear their bits for
        the remainder of the call body."""
        value = self._eval(node.value) if node.value is not None else None
        frame = self._mf_frame("call")
        if frame is None:
            from .interpreter import _ReturnSignal
            raise _ReturnSignal(value)
        live = self._live
        if isinstance(value, str) or not isinstance(value, torch.Tensor):
            # M7: no per-pixel representation for a string — first writer wins, which is
            # what `0.23` gives for the same source with a uniform condition.
            if frame.ret is None:
                frame.ret = value
        elif live is True:
            frame.ret = value
        else:
            from .interpreter import _tensor_where
            base = frame.ret
            if base is None or not isinstance(base, torch.Tensor):
                base = torch.zeros_like(value)
            frame.ret = _tensor_where(live, value, base)
        frame.dead = m_or(frame.dead, live)
        self._live = m_sub(self._live, frame.dead)
        return None

    # ── M4: calls ───────────────────────────────────────────────────────────
    def _mf_call_user_function(self, func_def, call_node):
        """A call inherits the caller's live mask. A call with NO live pixel is skipped
        entirely — that skip is what lets a per-pixel recursion terminate, and it is
        observable only through M6/M7 side effects, never through the returned value."""
        from .interpreter import InterpreterError, MAX_CALL_DEPTH, _ReturnSignal
        if not m_any(self._live):
            return torch.scalar_tensor(0.0, dtype=self._dtype, device=self.device)

        self._call_depth += 1
        if self._call_depth > MAX_CALL_DEPTH:
            self._call_depth -= 1
            raise InterpreterError(
                f"Maximum function call depth ({MAX_CALL_DEPTH}) exceeded — "
                f"possible infinite recursion in '{func_def.name}()'.",
                call_node.loc, source=self._source, code="E6060",
                hint="Check for functions that call themselves without a base case.",
            )

        args = [self._eval(arg) for arg in call_node.args]

        saved_env = self.env
        saved_ready = self._inplace_ready
        saved_decl = self._decl_depth
        saved_live = self._live
        saved_depth = self._region_depth
        self.env = dict(saved_env)
        self._inplace_ready = set()
        self._decl_depth = dict(saved_decl)
        self._region_depth = saved_depth + 1

        frame = _Frame("call")
        self._frames.append(frame)
        for (ptype, pname), arg_val in zip(func_def.params, args):
            self.env[pname] = arg_val
            self._decl_depth[pname] = self._region_depth

        try:
            for stmt in func_def.body:
                self._exec_stmt(stmt)
        except _ReturnSignal as ret:
            result = ret.value
        else:
            result = frame.ret
            if result is None:
                result = torch.scalar_tensor(0.0, dtype=self._dtype, device=self.device)
        finally:
            self._frames.pop()
            self.env = saved_env
            self._inplace_ready = saved_ready
            self._decl_depth = saved_decl
            self._region_depth = saved_depth
            self._mf_restore(saved_live)
            self._call_depth -= 1

        return result

    # ── M7: probes ──────────────────────────────────────────────────────────
    def _mf_debug_print(self, label, value, x=0.0, y=0.0):
        """`debug_print` records only if its probe pixel is live."""
        live = self._live
        if live is not True:
            try:
                xi = int(x.item()) if isinstance(x, torch.Tensor) else int(x)
                yi = int(y.item()) if isinstance(y, torch.Tensor) else int(y)
                if live is False:
                    return value
                m = live
                if m.dim() >= 3:
                    h, w = m.shape[-2], m.shape[-1]
                    ok = bool(m[..., min(max(yi, 0), h - 1),
                                min(max(xi, 0), w - 1)].reshape(-1)[0].item())
                else:
                    ok = bool(m.reshape(-1)[0].item())
            except Exception:
                ok = True
            if not ok:
                return value
        return self._get_stdlib()["debug_print"](label, value, x, y)
