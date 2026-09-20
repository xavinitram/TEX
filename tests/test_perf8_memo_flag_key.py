"""PERF-8 — the analysis memos key on the egress profile, not just on the source text.

THE HAZARD. `tex_cache.parse_and_split` is THE front end, and what it builds from a source is
not a function of that source alone: while plane wires are enabled, `p@beauty.diffuse` stays
ONE dotted `BindingRef` — a plane read — and while they are disabled the splitback puts it
back to `ChannelAccess(@beauty, "diffuse")`, the swizzle it meant before planes existed. The
per-source parse memos (`tex_roi._parse_memo`, `tex_lazy._parse_memo`) held a whole AST under
the SOURCE TEXT alone, so an entry minted under one profile was served under the other, and
the three answer memos above them (`tex_roi._walk_memo`, `tex_roi._region_dep_memo`,
`tex_lazy._memo`) inherited the same blindness through their own keys.

The symptom is a wrong REACH, not a wrong picture — but in the direction that matters: a
plane-reading host asking after a parse taken while planes were off is told the program needs
the base wire `beauty`, and a required-binding set that is too SMALL is the one thing the lazy
analysis may never produce (invariant #11). `tex_lazy._profile_key` is now the one element
every one of those five keys carries.

WHY IT IS NOT A REGRESSION AND NOT A USER-FACING BUG TODAY. The flag is process-global and
set once by the host before its first cook (`tex_compiler/types.py`, the set-once posture), and
it is deliberately absent from the program fingerprint. Under either shipped host the new key
element is a constant, so no hit rate and no answer moves — which is exactly what makes closing
the hazard free. What it buys is a host that can change profile mid-process: an engine toggling
a planes capability, or an editor previewing both.

WHAT EACH ROW PROVES:
  1. THE CANARY. Mint a parse under plane wires OFF, flip them ON, ask again WITHOUT clearing
     anything — the answer must be the ON parse, and flipping back must return the OFF parse.
     Both parse memos, on a source with a dotted binding. Red at the base sha, both memos.
  2. THE ANSWERS, not just the parses. The same canary through the two public entry points
     (`tex_lazy.lazy_required_bindings`, `tex_roi.binding_footprints`), because a flag-keyed
     parse under a flag-blind answer memo would still serve the stale set.
  3. THE MUTATION. Drop the flag from the key — `_profile_key` pinned to a constant, which is
     precisely the pre-change key — and rows 1 and 2 must fail. A canary that cannot die is
     decoration.
  4. THE DISCIPLINE IS UNCHANGED. One shared body behind both parse memos (so the two cannot
     drift the way five hand-spelled keys did), the bounded LRU still bounded, and
     `clear_roi_memo` / `clear_lazy_memo` still clear.

PORTABILITY: CPU, no ComfyUI, no CUDA, no compiler, no numpy, no timing assertion. Every row
restores the flag and drops the memos on the way in AND on the way out, so this file can
neither inherit pollution nor create it — it flips the very flag whose blindness it pins.
"""
import functools

from helpers import *

from TEX_Wrangle import tex_lazy, tex_roi
from TEX_Wrangle.tex_compiler.ast_nodes import BindingRef, iter_child_nodes
from TEX_Wrangle.tex_compiler.types import array_wires_enabled, set_array_wires
from test_perf1_roi_walk_memo import _drop_analysis_memos

#: A dotted read whose MEANING is the flag: `p@` hints PLANES, so with plane wires on the
#: splitback keeps `beauty.diffuse` as one binding, and with them off it is a swizzle of
#: `@beauty`. The analyses answer `beauty.diffuse` on one side and `beauty` on the other.
DOTTED = "@OUT = vec4(p@beauty.diffuse, 1.0);\n"

_PLANE_PARSE = frozenset({"beauty.diffuse"})    # what the ON parse leaves dotted
_SWIZZLE_PARSE = frozenset()                    # what the OFF splitback leaves dotted


def profile_canary(fn):
    """Save/restore the flag and drop every analysis memo on both edges of the row."""
    @functools.wraps(fn)
    def wrapper(r):
        prev = array_wires_enabled()
        _drop_analysis_memos()
        try:
            return fn(r)
        finally:
            set_array_wires(prev)
            _drop_analysis_memos()
    return wrapper


def _dotted_bindings(program) -> frozenset:
    """The `@name.seg` reads this AST still spells as ONE binding — the plane-read shape.
    Empty when the splitback put every dotted read back to a `ChannelAccess` swizzle."""
    names, stack = set(), list(program.statements)
    while stack:
        node = stack.pop()
        if node.__class__ is BindingRef and node.kind == "wire" and "." in node.name:
            names.add(node.name)
        stack.extend(iter_child_nodes(node))
    return frozenset(names)


def _parse_shapes(pristine) -> tuple:
    """`(off, on, off-again)` from one memo, flipping the flag and clearing NOTHING between —
    the hazard's exact shape. `pristine` is a module's `_pristine_program`."""
    set_array_wires(False)
    off = _dotted_bindings(pristine(DOTTED))
    set_array_wires(True)
    on = _dotted_bindings(pristine(DOTTED))
    set_array_wires(False)
    back = _dotted_bindings(pristine(DOTTED))
    return off, on, back


def _answer_shapes(ask) -> tuple:
    """`(off, on, off-again)` from a public analysis entry point, clearing nothing between."""
    set_array_wires(False)
    off = ask()
    set_array_wires(True)
    on = ask()
    set_array_wires(False)
    back = ask()
    return off, on, back


def _lazy_names() -> frozenset:
    return frozenset(tex_lazy.lazy_required_bindings(DOTTED, {}) or ())


def _roi_names() -> frozenset:
    return frozenset((tex_roi.binding_footprints(DOTTED, {}) or {}).keys())


# ── 1. the canary ─────────────────────────────────────────────────────────────

@profile_canary
def test_perf8_a_parse_memo_entry_is_not_served_across_a_profile_flip(r: SubTestResult):
    """Red at the base sha, in both modules: the ON ask is served the OFF parse."""
    print("\n--- PERF-8: the parse memos follow the flag ---")
    for name, pristine in (("tex_roi", tex_roi._pristine_program),
                           ("tex_lazy", tex_lazy._pristine_program)):
        _drop_analysis_memos()
        off, on, back = _parse_shapes(pristine)
        if off == _SWIZZLE_PARSE and on == _PLANE_PARSE and back == _SWIZZLE_PARSE:
            r.ok(f"{name}._parse_memo re-parses across a profile flip and back "
                 f"(swizzle -> {sorted(on)} -> swizzle)")
        else:
            r.fail(f"PERF-8 parse memo {name}",
                   f"off={sorted(off)} on={sorted(on)} back={sorted(back)} — expected "
                   f"[], {sorted(_PLANE_PARSE)}, []; a memo entry outlived the profile "
                   f"that produced it")


# ── 2. the answers, not just the parses ───────────────────────────────────────

@profile_canary
def test_perf8_the_analysis_answers_follow_the_profile(r: SubTestResult):
    """A flag-keyed parse under a flag-blind ANSWER memo would still serve the stale set, so
    the canary is repeated at the two public entry points the hosts actually call."""
    print("\n--- PERF-8: the answer memos follow the flag too ---")
    for name, ask, on_want, off_want in (
            ("tex_lazy.lazy_required_bindings", _lazy_names,
             frozenset({"OUT", "beauty.diffuse"}), frozenset({"OUT", "beauty"})),
            ("tex_roi.binding_footprints", _roi_names,
             frozenset({"beauty.diffuse"}), frozenset({"beauty"}))):
        _drop_analysis_memos()
        off, on, back = _answer_shapes(ask)
        if off == off_want and on == on_want and back == off_want:
            r.ok(f"{name} answers {sorted(on)} with plane wires on and {sorted(off)} with "
                 f"them off, in either order, with no memo cleared between")
        else:
            r.fail(f"PERF-8 answer memo {name}",
                   f"off={sorted(off)} on={sorted(on)} back={sorted(back)} — expected "
                   f"{sorted(off_want)}, {sorted(on_want)}, {sorted(off_want)}")


# ── 3. the mutation ───────────────────────────────────────────────────────────

@profile_canary
def test_perf8_dropping_the_flag_from_the_key_brings_the_hazard_back(r: SubTestResult):
    """Pin `_profile_key` to a constant — which IS the pre-change key — and rows 1 and 2 must
    fail. Both modules are patched: `tex_roi` imports the helper by name, so its keys read
    `tex_roi._profile_key` while both parse memos read `tex_lazy._profile_key` through the
    shared body."""
    print("\n--- PERF-8: the flag in the key is load-bearing (mutation) ---")
    def blind():
        return False

    real_lazy, real_roi = tex_lazy._profile_key, tex_roi._profile_key
    tex_lazy._profile_key = tex_roi._profile_key = blind
    try:
        _drop_analysis_memos()
        _, parse_on, _ = _parse_shapes(tex_roi._pristine_program)
        _drop_analysis_memos()
        _, lazy_on, _ = _answer_shapes(_lazy_names)
        _drop_analysis_memos()
        _, roi_on, _ = _answer_shapes(_roi_names)
    finally:
        tex_lazy._profile_key, tex_roi._profile_key = real_lazy, real_roi
    stale = (parse_on == _SWIZZLE_PARSE
             and lazy_on == frozenset({"OUT", "beauty"})
             and roi_on == frozenset({"beauty"}))
    r.ok("with the flag dropped from the key every ON ask is served the OFF answer — the "
         "canary above is testing the key and nothing else") if stale else \
        r.fail("PERF-8 mutation",
               f"a flag-blind key did NOT reproduce the stale answer: parse={sorted(parse_on)} "
               f"lazy={sorted(lazy_on)} roi={sorted(roi_on)} — either the canary passes for "
               f"some other reason or a second cache is doing the work")


# ── 4. the discipline the key change may not move ─────────────────────────────

@profile_canary
def test_perf8_one_shared_body_bounded_and_clearable(r: SubTestResult):
    """One helper behind both parse memos (five hand-spelled keys is how the blindness got in),
    the LRU still bounded, and both clear hooks still clear."""
    print("\n--- PERF-8: one body, same bounds, same clear hooks ---")
    set_array_wires(False)

    _drop_analysis_memos()
    tex_roi._pristine_program(DOTTED)
    after_roi = (len(tex_roi._parse_memo), len(tex_lazy._parse_memo))
    tex_lazy._pristine_program(DOTTED)
    after_lazy = (len(tex_roi._parse_memo), len(tex_lazy._parse_memo))
    one_body = (tex_roi._pristine_parse is tex_lazy._pristine_parse
                and tex_roi._profile_key is tex_lazy._profile_key)
    if one_body and after_roi == (1, 0) and after_lazy == (1, 1):
        r.ok("both parse memos are the one shared body keyed by the one shared profile key, "
             "each still filling only its own store")
    else:
        r.fail("PERF-8 shared body",
               f"shared body/key={one_body}, stores after roi={after_roi} lazy={after_lazy} — "
               f"the two memos can drift in key or in eviction order")

    _drop_analysis_memos()
    for i in range(tex_roi._PARSE_MEMO_MAX + 20):
        tex_roi._pristine_program(f"@OUT = vec4({i}.0);")
        tex_lazy._pristine_program(f"@OUT = vec4({i}.0);")
    held = (len(tex_roi._parse_memo), len(tex_lazy._parse_memo))
    tex_roi.clear_roi_memo()
    tex_lazy.clear_lazy_memo()
    emptied = (len(tex_roi._parse_memo), len(tex_lazy._parse_memo),
               len(tex_roi._walk_memo), len(tex_roi._region_dep_memo), len(tex_lazy._memo))
    if held == (tex_roi._PARSE_MEMO_MAX, tex_lazy._PARSE_MEMO_MAX) and emptied == (0,) * 5:
        r.ok(f"both parse memos still cap at {tex_roi._PARSE_MEMO_MAX} (held {held}) and "
             f"clear_roi_memo / clear_lazy_memo still empty all five stores")
    else:
        r.fail("PERF-8 bounds", f"held={held} after-clear={emptied}")
