"""CG-1 — the emitter reads a type map whose `id()` keys can only mean the nodes it emits.

`type_map` is keyed by `id()`, and an id names an object only while it lives. Before this
change the map the emitter was handed could carry entries for nodes already dead by emission
time, so a live node whose address was recycled onto one read a type that was not its own —
and the emitted codegen source of 3–6 of the 130 corpus programs differed from one PROCESS
to the next (`_emit_binop`'s runtime-broadcast branch appearing and disappearing; the matrix
branch reads the same lookup, so nothing bounded it). Two halves close it:

* `tex_compiler.types.TypeMap` — the checker's map pins every node it types (`record`), so a
  key can never be recycled while the map lives, and `is_own` asks the identity form of the
  question a bare `get(id(node))` could only approximate.
* `tex_runtime.codegen._live_type_map` — `try_compile` narrows the map it was handed to the
  nodes the program reaches AND that the map recorded for that very object, holding the
  program for the whole emission so nothing reachable can recycle meanwhile.

The cross-process claim itself needs separate processes and lives in an out-of-tree sweep;
what lives here is the deterministic construction of the collision (an entry under the id of
a node the checker never typed), the pin, and the watched-file half (the STR-7 emitters
`codegen_stencil.py` / `codegen_persist.py` now key the `.cg` epoch).

The CG-1 symbols are imported inside the rows that need them, so a tree without them reds
row by row instead of failing to import the module.
"""
import gc

from helpers import *   # noqa: F403

from TEX_Wrangle import tex_cache
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_compiler.ast_nodes import (ASTNode, BinOp, BindingRef, Identifier,
                                                NumberLiteral, iter_child_nodes)
from TEX_Wrangle.tex_compiler.optimizer import optimize
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_runtime.codegen import try_compile


def _check(src, binding_types):
    program = parse_and_split(src, binding_types)
    checker = TypeChecker(binding_types=binding_types, source=src)
    return program, checker.check(program)


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(iter_child_nodes(n))


_VEC3_AB = {"A": TEXType.VEC3, "B": TEXType.VEC3}


def test_emitter_ignores_an_entry_recorded_for_another_node(r):
    """The collision, constructed deterministically: an entry under the `id()` of a node the
    checker never typed. A recycled address reads exactly like this — a bare `type_map[id] = t`
    for an object that is not the one the entry was made for. The originals are kept alive so
    no id can recycle during the test; the fresh operands are untyped, so the reference is the
    answer a map that knows only the checked tree gives."""
    label = "CG-1: a stale MAT3 entry under a live vec3 operand's id is not read"
    try:
        program, tm = _check("@OUT = @A * @B;\n", _VEC3_AB)
        binop = program.statements[0].value
        assert isinstance(binop, BinOp) and binop.op == "*"
        keep = (binop.left, binop.right)                     # alive: nothing recycles here
        binop.left = BindingRef(name="A", loc=keep[0].loc)
        binop.right = BindingRef(name="B", loc=keep[1].loc)
        ref = try_compile(program, tm)._tex_src
        assert "matmul" not in ref
        # what a live node reads through a recycled id: the dead node's type, under its own id
        tm[id(binop.left)] = TEXType.MAT3
        tm[id(binop.right)] = TEXType.MAT3
        got = try_compile(program, tm)._tex_src
        assert "matmul" not in got, "the emitter read a matrix type the checker never gave this node"
        assert got == ref, "the stale entry changed the emitted source"
        r.ok(label)
    except Exception as e:
        r.fail(label, f"{type(e).__name__}: {e}")


def test_the_narrowed_map_is_reachable_and_own(r):
    """`_live_type_map` keeps exactly the entries a reachable node owns: nothing for a node
    outside the program, nothing under an id the map did not record for that object."""
    label = "CG-1: _live_type_map = reachable ∩ recorded-for-this-object"
    try:
        from TEX_Wrangle.tex_runtime.codegen import _live_type_map
        program, tm = _check("float a = 2.0;\n@OUT = @A * a + @B;\n", _VEC3_AB)
        stray = NumberLiteral(value=7.0)                    # typed, but not in the program
        tm.record(stray, TEXType.FLOAT)
        forged = Identifier(name="ghost")                   # in the map by a bare write only
        dropped = program.statements[0].initializer         # kept alive: typed, now unreachable
        program.statements[0].initializer = forged
        tm[id(forged)] = TEXType.MAT4
        live = _live_type_map(program, tm)
        assert id(stray) not in live, "an unreachable node's entry survived narrowing"
        assert id(dropped) not in live, "a node the program no longer reaches survived narrowing"
        assert id(forged) not in live, "an entry not recorded for that object survived narrowing"
        reach = {id(n): n for n in _walk(program)}
        want = {i: tm[i] for i, n in reach.items() if i in tm and tm.is_own(n)}
        assert want and live == want, "narrowing is not reachable ∩ own"
        # a bare dict (a map a caller built by hand) is narrowed by reachability alone
        plain = dict(tm)
        assert _live_type_map(program, plain) == {i: plain[i] for i in reach if i in plain}
        r.ok(label)
    except Exception as e:
        r.fail(label, f"{type(e).__name__}: {e}")


def test_the_checker_map_outlives_the_nodes_it_typed(r):
    """The pin. Every entry the checker writes is recorded for that object; and once the
    program is gone, no freshly built node can land on a typed id — the symptom the corpus
    sweep showed, which a pinned key makes impossible rather than merely unlikely."""
    label = "CG-1: TypeMap pins what it types; a freed program's ids are never recycled"
    try:
        from TEX_Wrangle.tex_compiler.types import TypeMap
        program, tm = _check(
            "float a = 2.0;\nfloat b = a * 3.0 + 1.0;\nvec3 c = @A * b;\n@OUT = c + @B;\n",
            _VEC3_AB)
        assert isinstance(tm, TypeMap)
        typed = [n for n in _walk(program) if id(n) in tm]
        assert typed and all(tm.is_own(n) for n in typed)
        assert len(tm._pins) == len(tm), "an entry without its pin"
        ids = set(tm)
        del program, typed
        gc.collect()
        fresh = []
        for _ in range(500):
            fresh.append(NumberLiteral(value=1.0))
            fresh.append(Identifier(name="x"))
            fresh.append(BindingRef(name="A"))
            fresh.append(BinOp(op="+", left=fresh[-3], right=fresh[-2]))
        hits = [id(n) for n in fresh if id(n) in ids]
        assert not hits, f"{len(hits)} fresh nodes landed on typed ids"
        assert all(isinstance(n, ASTNode) for n in tm._pins.values())
        r.ok(label)
    except Exception as e:
        r.fail(label, f"{type(e).__name__}: {e}")


def test_optimizer_registered_nodes_are_pinned_too(r):
    """The optimizer registers the CSE/LICM temps it synthesizes in the map it is given; those
    entries carry the same pin, so a temp a later pass drops cannot leave a recyclable key.
    (`sin(u * 2.0)` twice: depth 2 is the CSE threshold, so the pass synthesizes a temp.)"""
    label = "CG-1: optimizer-registered temps are recorded for their object"
    try:
        program, tm = _check("@OUT = vec3(sin(u * 2.0) + sin(u * 2.0));\n", {})
        n0 = len(tm)
        optimize(program, tm)
        assert len(tm) > n0, "the optimizer registered nothing (CSE did not fire?)"
        assert len(tm._pins) == len(tm), "an optimizer entry without its pin"
        assert all(tm.is_own(n) for n in _walk(program) if id(n) in tm)
        r.ok(label)
    except Exception as e:
        r.fail(label, f"{type(e).__name__}: {e}")


def test_stencil_and_persist_emitters_key_the_cg_epoch(r):
    """L5-F3: the STR-7 split's other two emitters were not watched, so a stencil-lowering
    edit left every existing `.cg` sidecar passing its version check."""
    label = "CG-1: codegen_stencil.py and codegen_persist.py are in _CODEGEN_FILES"
    try:
        watched = {p.name for p in tex_cache._CODEGEN_FILES}
        for name in ("codegen_stencil.py", "codegen_persist.py"):
            assert name in watched, f"{name} does not key the codegen epoch"
        for p in tex_cache._CODEGEN_FILES:
            assert p.exists(), f"watched file missing on disk: {p}"
        r.ok(label)
    except Exception as e:
        r.fail(label, f"{type(e).__name__}: {e}")
