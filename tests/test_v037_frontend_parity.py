"""DATA-6, "one front end": every consumer agrees on the WIRES a program reads.

THE CLAIM. The lexer reads `@name.seg` as ONE binding token (the language rule since planes),
and `tex_cache.splitback_dotted_bindings` puts every dotted binding that is not a plane read
back to the `ChannelAccess(BindingRef(base), seg)` swizzle the parser built before planes. That
is only safe if NOTHING parses privately: a consumer that lexes and type-checks (or walks) the
raw stream reads `image.r` where the cook reads `image`, and the drift is silent on every
program without a dot. So there is one owner — `tex_cache.parse_and_split` — and this file
pins that every reader of a binding's name as its wire agrees with the production seam:

    tex_api.check                     the editor lint (no false E3200, no false W7002)
    TEXCache.compile_tex              the production seam (the reference reading)
    tex_fusion.compile_fused          the per-stage checker + the name-keyed wire maps
    tex_roi.binding_footprints        the ROI read set
    tex_lazy.lazy_required_bindings   the lazy-input set (invariant 11: the WIRE is requested)
    tex_marshalling.sigil_names       the sigil scan (greedy by contract; bases compared)
    failure_harness.compile_program   the suite's tier harness
    helpers.check_code                the suite's checker harness
    test_integration._prepare_example the corpus harness (pass 1 splits with `{}`)

THE PROOF IS A MUTATION, NOT A GREEN RUN: at the base every consumer was non-greedy and agreed
by construction, so this file is green there too. What it buys is that reverting ANY ONE
consumer to a private `Lexer(...)` call reds it, naming that consumer — the lane that wired it
ran that mutation against several consumers and restored them. Keep it that way: a consumer
added later must either parse through `parse_and_split` or join this file with a reason.

THE CORPUS IS DERIVED, never typed: every `examples/*.tex` whose default token stream carries a
dotted `@` form (the twelve the compat scan reports), plus inline programs with a `p@` hint.

Every row runs on the compiler and the CPU interpreter alone. No ComfyUI, no CUDA, no
compiler toolchain, no Windows path, no embedded interpreter, no numpy, no timing.
"""
import contextlib
import os
import re

from helpers import *

import failure_harness as fh
import test_integration as ti
from TEX_Wrangle import tex_api, tex_fusion, tex_lazy, tex_roi
from TEX_Wrangle.tex_cache import get_cache, parse_and_split
from TEX_Wrangle.tex_compiler.types import array_wires_enabled, set_array_wires
from TEX_Wrangle.tex_marshalling import infer_binding_type, param_only_names, sigil_names
from TEX_Wrangle.tex_runtime.interpreter import _collect_binding_reads

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_W7002_NAME = re.compile(r"Input '@([^']+)'")


def _dotted_examples() -> list:
    """`[(name, source)]` for every shipped example whose DEFAULT token stream carries a dotted
    `@` binding (one AT_BINDING / TYPED_AT_BINDING whose value holds a dot)."""
    exdir = os.path.join(_ROOT, "examples")
    out = []
    for fn in sorted(os.listdir(exdir)):
        if not fn.endswith(".tex"):
            continue
        with open(os.path.join(exdir, fn), encoding="utf-8") as f:
            src = f.read()
        toks = Lexer(src).tokenize()
        if any(t.type in (TokenType.AT_BINDING, TokenType.TYPED_AT_BINDING) and "." in t.value
               for t in toks):
            out.append((fn[:-4], src))
    return out


def _base(name: str) -> str:
    return name.rsplit(".", 1)[0] if "." in name else name


def _has_planes_hint(src: str) -> bool:
    """True when the program carries a `p@` hint — a PLANES wire exists only on the engine
    profile, so such a program is read there; every other program keeps the default profile."""
    return any(t.type is TokenType.TYPED_AT_BINDING and t.prefix == "p"
               for t in Lexer(src).tokenize())


class _planes_enabled:
    def __init__(self, on=True):
        self._on = on

    def __enter__(self):
        self._prev = array_wires_enabled()
        set_array_wires(self._on)

    def __exit__(self, *a):
        set_array_wires(self._prev)


# ── the consumers, each reduced to "the wire names this program reads" ───────────────────────

def _reference(src):
    """The production seam's reading. Bindings come from the corpus harness (itself a
    consumer, checked below); the seam's `referenced - assigned - params` is the reference
    wire set every other consumer must reproduce."""
    _program, bindings, _tm, _outs = ti._prepare_example(src, 1, 8, 8)
    bt = {n: infer_binding_type(v) for n, v in bindings.items()}
    _prog, _tm2, referenced, assigned, params, _used = get_cache().compile_tex(src, bt)
    params = frozenset(params)
    assigned = frozenset(assigned)
    wires = frozenset(referenced) - assigned - params
    return bt, bindings, wires, assigned, params


def _check_reads(src, bt, assigned, params):
    """`tex_api.check`: no error-severity diagnostic on a program the seam compiles, and the
    wires it did NOT flag W7002 ("connected but never used") are the wires it read."""
    diags = tex_api.check(src, bt)
    errors = [(d.code, d.message) for d in diags if d.severity == "error"]
    assert not errors, f"tex_api.check reports errors on a program the seam compiles: {errors}"
    unused = set()
    for d in diags:
        if d.code == "W7002":
            m = _W7002_NAME.search(d.message)
            assert m, d.message
            unused.add(m.group(1))
    return frozenset(n for n in bt if n not in params and n not in assigned) - unused


def _fusion_reads(src, bindings, wires, assigned, params):
    """`tex_fusion.compile_fused` with the program as the TERMINAL of a two-stage chain, its
    first wire fed by a trivial upstream. The fused program's referenced names carry the
    stage-1 user prefix; stripping it back must give the same wires (the chain input is read
    through the handoff local, so it is added back by construction)."""
    chain_in = sorted(wires)[0]
    up_src = {k: v for k, v in bindings.items() if k == chain_in}
    rest = {k: v for k, v in bindings.items() if k != chain_in}
    nonce = f"\n// parity {os.getpid()}\n"          # defeat the fused memo across processes
    stages = [
        {"code": f"@OUT = @src;{nonce}", "chain_input": None, "bindings": {"src": up_src[chain_in]}},
        {"code": src + nonce, "chain_input": chain_in, "bindings": rest},
    ]
    _prog, _tm, refs, _asg, _params, _used, _merged = tex_fusion.compile_fused(
        stages, infer_binding_type)
    pref = tex_fusion._user_prefix("_s1_")
    got = frozenset(n[len(pref):] for n in refs if n.startswith(pref))
    return (got | {chain_in}) - assigned - params


def _roi_reads(src, assigned, params):
    fps = tex_roi.binding_footprints(src, {})
    assert fps is not None, "tex_roi.binding_footprints declined the program"
    return frozenset(fps) - assigned - params


def _lazy_reads(src, assigned, params):
    lazy = tex_lazy.lazy_required_bindings(src, {})
    assert lazy is not None, "tex_lazy.lazy_required_bindings declined the program"
    return frozenset(lazy) - assigned - params


def _harness_reads(src, bindings, assigned, params):
    prog, _tm, _outs = fh.compile_program(src, bindings)
    return _collect_binding_reads(prog) - assigned - params


def _check_code_reads(src, bt, assigned, params):
    _tm, checker = check_code(src, bt)
    return frozenset(checker.referenced_bindings) - assigned - params


def _consumers(src, bt, bindings, wires, assigned, params):
    """`[(consumer name, thunk -> the wire names it reads)]` — thunks, so a consumer that
    RAISES on a swizzle it read as a binding (a false type error from a private parse) is
    named individually instead of aborting the row."""
    return [
        ("tex_api.check", lambda: _check_reads(src, bt, assigned, params)),
        ("tex_fusion.compile_fused (per-stage checker + name maps)",
         lambda: _fusion_reads(src, bindings, wires, assigned, params)),
        ("tex_roi.binding_footprints", lambda: _roi_reads(src, assigned, params)),
        ("tex_lazy.lazy_required_bindings", lambda: _lazy_reads(src, assigned, params)),
        ("tex_marshalling.sigil_names (bases)",
         lambda: frozenset(_base(n) for n in sigil_names(src)[0]) - assigned - params),
        ("failure_harness.compile_program", lambda: _harness_reads(src, bindings, assigned, params)),
        ("helpers.check_code", lambda: _check_code_reads(src, bt, assigned, params)),
        ("test_integration._prepare_example", lambda: frozenset(bindings) - params),
    ]


def test_every_front_end_agrees_on_the_wires_a_program_reads(r: SubTestResult):
    print("\n--- DATA-6: one front end — every consumer names the same wires ---")
    corpus = _dotted_examples()
    try:
        assert len(corpus) >= 12, [n for n, _ in corpus]
        r.ok(f"corpus derived from examples/: {len(corpus)} dotted programs")
    except Exception as e:
        r.fail("corpus derived from examples/", str(e))
    for name, src in corpus:
        # DATA-6 L-E: a program with a `p@` hint reads a PLANES wire, which exists only on the
        # engine profile (`test_v037_planes_wire` holds the same switch); it is read there, and
        # its reads are compared at the WIRE level — the seam reads the per-plane rows
        # (`beauty.diffuse`), `sigil_names` reports bases by design, and "the wires a program
        # reads" is the base either way. Every other program keeps the default profile and the
        # exact comparison, so the parity claim for the twelve swizzle programs is unchanged.
        hinted = _has_planes_hint(src)
        norm = (lambda names: frozenset(_base(n) for n in names)) if hinted else (lambda names: names)
        with (_planes_enabled(True) if hinted else contextlib.nullcontext()):
            try:
                bt, bindings, wires, assigned, params = _reference(src)
                assert wires and not any("." in w for w in norm(wires)), (name, wires)
                assert wires == frozenset(bindings) - params, \
                    f"seam {sorted(wires)} vs corpus harness {sorted(frozenset(bindings) - params)}"
            except Exception as e:
                r.fail(f"{name}: reference reading", str(e))
                continue
            disagree = []
            # Each consumer is asked in its own try, so the one that RAISES on a swizzle it read
            # as a binding (a false type error from a private parse) is named, not just the row.
            for consumer, read in _consumers(src, bt, bindings, wires, assigned, params):
                try:
                    got = read()
                except Exception as e:                    # noqa: BLE001 — the name is the point
                    disagree.append(f"{consumer}: raised {type(e).__name__}: {str(e).splitlines()[0]}")
                    continue
                if norm(got) != norm(wires):
                    disagree.append(f"{consumer}: reads {sorted(got)}, seam reads {sorted(wires)}")
        try:
            assert not disagree, "\n      ".join(disagree)
            r.ok(f"{name}: {len(wires)} wire(s) {sorted(wires)} — every consumer agrees")
        except Exception as e:
            r.fail(f"{name}: consumers disagree with the seam", str(e))


# ── the `p@` hint: the same base everywhere, on both profiles ────────────────────────────────

_HINTED = {
    # (a bare `@OUT = ...`: with plane wires ON the `p@` hint types the kept read PLANES, which
    # a typed declaration refuses (E3200, ARRAY-parity) — these rows are about NAMES, so the
    # program must compile on both profiles)
    "p@ read": "@OUT = p@beauty.diffuse;",
    "p@ read swizzled + an ordinary swizzle": "@OUT = vec4(p@beauty.diffuse.rgb, 1.0) * @image.r;",
}


def test_hinted_plane_reads_resolve_the_same_base_everywhere(r: SubTestResult):
    print("\n--- DATA-6: a `p@beauty.diffuse` read names `beauty` in every wire-keyed consumer ---")
    for label, src in _HINTED.items():
        # Plane wires OFF (the ComfyUI default): the read is a swizzle of `beauty` on every path.
        # The checker refuses `.diffuse` (E3302: not a swizzle) — the lint reports exactly that,
        # never an internal error and never "beauty is unused"; the wire-keyed analyses name
        # `beauty`, never `beauty.diffuse`.
        try:
            bt = {"beauty": TEXType.VEC4, "image": TEXType.VEC4}
            diags = tex_api.check(src, bt)
            codes = {d.code for d in diags}
            assert "E3302" in codes and "E0000" not in codes, codes
            assert not any(d.code == "W7002" and "beauty" in d.message for d in diags), \
                [d.message for d in diags]
            lazy = tex_lazy.lazy_required_bindings(src, {})
            assert lazy is not None and "beauty" in lazy and not any("." in n for n in lazy), lazy
            reads = _collect_binding_reads(parse_and_split(src, {}))
            assert "beauty" in reads and not any("." in n for n in reads), reads
            fps = tex_roi.binding_footprints(src, {})
            assert fps is not None and "beauty" in fps and not any("." in n for n in fps), fps
            assert "beauty" in _collect_binding_reads(parse_and_split(src, bt)), "typed map"
            r.ok(f"{label}, planes OFF: lint E3302 on `.diffuse`; lazy/roi/reads name `beauty`")
        except Exception as e:
            r.fail(f"{label}, planes OFF", str(e))
        # Plane wires ON: the `p@` hint types the base PLANES on every path, so the read is
        # KEPT as the dotted binding `beauty.diffuse` everywhere — the seam's referenced set,
        # the lint (no E3302 on `.diffuse`), the lazy set and the ROI read set all name the
        # same dotted wire, which is the name an expanded bindings map carries. (The source
        # gets a per-profile comment because the lazy / ROI / sigil memos are keyed on the
        # source alone, not on the profile switch — a host sets the profile once.)
        try:
            with _planes_enabled(True):
                on = src + "// planes-on\n"
                bt = {"beauty": TEXType.PLANES, "image": TEXType.VEC4}
                diags = tex_api.check(on, bt)
                codes = {d.code for d in diags}
                assert "E3302" not in codes and "E0000" not in codes, codes
                _p, _t, referenced, assigned, _pr, _u = get_cache().compile_tex(on, bt)
                seam = frozenset(referenced) - frozenset(assigned)
                assert "beauty.diffuse" in seam and "beauty" not in seam, seam
                lazy = tex_lazy.lazy_required_bindings(on, {})
                assert lazy is not None and frozenset(lazy) - frozenset(assigned) == seam, lazy
                fps = tex_roi.binding_footprints(on, {})
                assert fps is not None and frozenset(fps) == seam, fps
                reads = _collect_binding_reads(parse_and_split(on, {})) - frozenset(assigned)
                assert reads == seam, reads
            assert "beauty.diffuse" in sigil_names(src)[0], sigil_names(src)[0]
            r.ok(f"{label}, planes ON: every path keeps the plane read `beauty.diffuse`")
        except Exception as e:
            r.fail(f"{label}, planes ON", str(e))


# ── the sigil scan and its two wire-keyed consumers ─────────────────────────────────────────

def test_sigil_names_is_greedy_and_the_wire_keyed_consumers_keep_the_base(r: SubTestResult):
    print("\n--- DATA-6: sigil_names reports per-plane demand; param_only_names / ROI keep the base ---")
    try:
        ats, dollars = sigil_names("@OUT = vec4(@beauty.diffuse, 1.0) * @image.r + $k * $beauty;")
        assert ats == {"OUT", "beauty.diffuse", "image.r"}, ats
        assert dollars == {"k", "beauty"}, dollars
        r.ok("sigil_names: `beauty.diffuse` and `image.r` verbatim (per-plane demand)")
    except Exception as e:
        r.fail("sigil_names greedy", str(e))
    try:
        # `$beauty` beside `@beauty.x` uses `beauty` BOTH ways: it must NOT be param-only, or a
        # real wire's type would leave the identity map.
        only = param_only_names("@OUT = vec4(@beauty.x) * $k * $beauty;")
        assert only == {"k"}, only
        assert param_only_names("@OUT = @k.rgb * $k;") == frozenset(), "the pre-planes case"
        r.ok("param_only_names: the dotted BASE is subtracted (`beauty` stays in the key)")
    except Exception as e:
        r.fail("param_only_names keeps the base", str(e))
    try:
        # The ROI read set names the wire, and the fold-erased set does too: `mix(@A, @B.rgb, $k)`
        # with k = 0 folds `@B.rgb` away; the plan must still narrow `B`, the wire the cook
        # passes — with only `B.rgb` in that set nothing matches and `B` goes in WHOLE. The
        # verbatim `B.rgb` rides along (the ROI cooker tests membership, so a name that is
        # not a binding is inert) and stays right for a plane read on an expanded map.
        src = "@OUT = mix(@A, vec4(@B.rgb, 1.0), $k);"
        plan = tex_roi.roi_plan(src, {"k": 0.0})
        assert plan.executable, plan
        assert "B" in plan.narrow, plan.narrow
        assert "B" in tex_roi._referenced_at_bindings(src), tex_roi._referenced_at_bindings(src)
        r.ok("tex_roi: a fold-erased `@B.rgb` still narrows the wire `B`")
    except Exception as e:
        r.fail("tex_roi fold-erased base", str(e))
