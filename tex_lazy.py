"""
Lazy input cooking — static analysis of which inputs a TEX program can need.

Given the source and the widget parameter values, `lazy_required_bindings`
returns the set of @/$ names the program can reference at runtime, so
`check_lazy_status` (tex_node.py) can tell ComfyUI to skip cooking the
upstream subgraphs of everything else.

Tiers covered:
  T1  input wired but never referenced             -> not in the set
  T2  referenced only inside statically-dead flow  -> pruned by const-fold
  T3  usage gated on a $param widget value         -> param folded as a
      literal, the dead branch pruned (the sweet spot)
  T4  usage gated on another *input's* value       -> deliberately NOT here;
      wired scalar params get a limited form via check_lazy_status's
      iterative protocol (params cook first, then fold like T3).

What deliberately does NOT sever a dependency (correctness, verified):
  * `@A * 0.0`   — NaN*0 = NaN in IEEE, and the optimizer itself refuses
                   x*0->0 (shape-unsafe, optimizer.py). Only *both-literal*
                   BinOps fold, so @A survives — by construction.
  * `&&` / `||`  — the interpreter evaluates both sides (no short-circuit
                   on tensors); folding only reaches them via literal operands.
  * spatial ifs  — torch.where evaluates both branches; only conditions that
                   fold to a compile-time NumberLiteral are pruned, which is
                   exactly the class the interpreter short-circuits as scalars.

The analysis is syntactic and over-approximating: it may KEEP a binding that
is dead (missed optimisation, never a bug). If it ever wrongly DROPPED one,
the interpreter raises the loud "Input '@X' is not connected" error — never
silent wrong output.

Precision note: folding runs comparisons in Python floats, the runtime in
fp32 tensors. Substituted params are pre-rounded to fp32 to close that gap;
the residual window (a literal-vs-param straddling one fp32 ulp inside a
comparison) fails loud per the above, not silently.

Cache #14: a module-level LRU keyed on (code-hash, folded-param fp32 bits).
Distinct key + lifecycle from the 13 existing caches (ARCHITECTURE.md), and
shared by check_lazy_status and execute() so the per-cook cost is a dict hit.
"""
from __future__ import annotations

import hashlib
import struct
from collections import OrderedDict

from .tex_compiler.lexer import Lexer
from .tex_compiler.parser import Parser
from .tex_compiler.ast_nodes import (
    ASTNode, BindingRef, NumberLiteral, IfElse, WhileLoop, ForLoop,
    FunctionDef, iter_child_nodes,
)
from .tex_compiler.optimizer import _propagate_literal_locals, _fold_all

# Wire types that can carry a spatial tensor (participate in first-wins shape
# derivation). STRING/INT/FLOAT/BOOLEAN wires marshal to non-spatial values.
SPATIAL_WIRE_TYPES = frozenset({"IMAGE", "MASK", "LATENT", "*"})
# Wire types R1 accepts as a shape anchor (known [B,H,W,C] tensors post-marshal).
SHAPE_ANCHOR_TYPES = frozenset({"IMAGE", "MASK"})
# Wire types whose cooked value is a foldable scalar (T4-lite candidates).
SCALAR_WIRE_TYPES = frozenset({"INT", "FLOAT", "BOOLEAN"})

_MEMO_MAX = 256
_memo: "OrderedDict[tuple, frozenset | None]" = OrderedDict()


def _fp32(v: float) -> float:
    """Round a Python float to fp32 so folded comparisons match the runtime's
    fp32 tensors (numpy-free per the torch-only invariant)."""
    return struct.unpack("f", struct.pack("f", float(v)))[0]


def _substitute_params(node: ASTNode, subs: dict[str, NumberLiteral]) -> None:
    """Replace $param BindingRefs with their literal values, in place.

    Walks every child field; a BindingRef child whose kind is "param" and
    whose name is in *subs* is swapped for a fresh NumberLiteral.
    """
    for field_name in node.__dataclass_fields__:
        val = getattr(node, field_name, None)
        if isinstance(val, BindingRef):
            if val.kind == "param" and val.name in subs:
                lit = subs[val.name]
                setattr(node, field_name, NumberLiteral(
                    loc=val.loc, value=lit.value, is_int=lit.is_int))
        elif isinstance(val, ASTNode):
            _substitute_params(val, subs)
        elif isinstance(val, list):
            for i, item in enumerate(val):
                if isinstance(item, BindingRef):
                    if item.kind == "param" and item.name in subs:
                        lit = subs[item.name]
                        val[i] = NumberLiteral(
                            loc=item.loc, value=lit.value, is_int=lit.is_int)
                elif isinstance(item, ASTNode):
                    _substitute_params(item, subs)


def _prune_static_flow(stmts: list) -> list:
    """Remove statically-dead control flow after const-folding.

    * IfElse with a NumberLiteral condition -> splice the taken body
      (`value > 0.5`, matching the interpreter's scalar short-circuit and the
      optimizer's ternary fold).
    * WhileLoop with a literal-false condition -> dropped.
    * ForLoop is left intact (zero-trip literal ranges are rare; keeping the
      body only over-approximates).
    Recurses into surviving bodies and FunctionDefs.
    """
    out: list = []
    for stmt in stmts:
        cls = stmt.__class__
        if cls is IfElse and isinstance(stmt.condition, NumberLiteral):
            taken = stmt.then_body if stmt.condition.value > 0.5 else stmt.else_body
            out.extend(_prune_static_flow(taken or []))
            continue
        if cls is WhileLoop and isinstance(stmt.condition, NumberLiteral) \
                and not stmt.condition.value > 0.5:
            continue
        if cls is IfElse:
            stmt.then_body = _prune_static_flow(stmt.then_body or [])
            stmt.else_body = _prune_static_flow(stmt.else_body or [])
        elif cls in (WhileLoop, ForLoop, FunctionDef):
            stmt.body = _prune_static_flow(stmt.body or [])
        out.append(stmt)
    return out


def _collect_binding_refs(stmts: list) -> frozenset[str]:
    """All @/$ names syntactically reachable in the surviving statements."""
    names: set[str] = set()
    stack = list(stmts)
    while stack:
        node = stack.pop()
        if node.__class__ is BindingRef:
            names.add(node.name)
        stack.extend(iter_child_nodes(node))
    return frozenset(names)


def _param_key(param_values: dict) -> tuple:
    """Stable memo key component: fp32 bit patterns for numerics, raw for the
    rest (bools fold as 0/1; strings never fold but distinguish programs)."""
    items = []
    for name in sorted(param_values):
        v = param_values[name]
        if isinstance(v, bool):
            items.append((name, "b", int(v)))
        elif isinstance(v, (int, float)):
            items.append((name, "f", struct.pack("f", float(v))))
        else:
            items.append((name, "s", str(v)))
    return tuple(items)


def lazy_required_bindings(code: str,
                           param_values: dict | None = None,
                           ) -> frozenset[str] | None:
    """The set of @/$ names the program can reference at runtime given these
    widget values, or None when the analysis fails (caller keeps everything).
    Never raises. Over-approximates: extra names are a missed skip, never a bug.

    Only float/int/bool params fold (T3); strings and vectors stay symbolic,
    so conditions gated on them conservatively keep both branches.
    """
    param_values = param_values or {}
    key = (hashlib.sha256(code.encode()).hexdigest(), _param_key(param_values))
    hit = _memo.get(key)
    if hit is not None or key in _memo:
        _memo.move_to_end(key)
        return hit
    try:
        # Fresh parse: the analysis mutates its AST, and no type info is
        # needed (references are syntactic).
        program = Parser(Lexer(code).tokenize(), source=code).parse()
        subs = {
            name: NumberLiteral(value=_fp32(v), is_int=isinstance(v, (bool, int)))
            for name, v in param_values.items()
            if isinstance(v, (bool, int, float))
        }
        stmts = program.statements
        if subs:
            for stmt in stmts:
                _substitute_params(stmt, subs)
        stmts = _propagate_literal_locals(stmts)
        stmts = _fold_all(stmts)
        stmts = _prune_static_flow(stmts)
        result: frozenset | None = _collect_binding_refs(stmts)
    except Exception:
        result = None
    _memo[key] = result
    if len(_memo) > _MEMO_MAX:
        _memo.popitem(last=False)
    return result


def clear_lazy_memo() -> None:
    """Test hook."""
    _memo.clear()
