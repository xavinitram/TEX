"""v0.45 TIMEREADS-45 — `Program.time_reads: frozenset[str]`.

The host's ask: which of `frame`/`fps`/`time`/`fetch_time`/`sample_time` a compiled program
reads, so a host can decide staleness/refresh policy without walking TEX's AST itself. The
host already does that walk today, because `used_builtins` (`_collect_identifiers`) is an
Identifier-only scan and `fetch_time`/`sample_time` are `FunctionCall` names — invisible to
it, per the host's own report. A miss there is a wrong-pixel stale serve.

`time_reads` is derived from the SAME two facts `codegen._reads_time_builtin` and
`graphed._capturable`'s time-decline branch already check, run together over one generic
`iter_child_nodes` walk (`tex_api._collect_time_reads`) — so this file is mostly a table over
the AST-level function, plus one row proving the public `tex_api.Program.time_reads` wiring
reaches it, and one differential row proving the two truths (`time_reads` and graphed's
capture bar) can only ever agree, not merely correlate.
"""
from TEX_Wrangle import tex_api
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_api import _collect_time_reads
from TEX_Wrangle.tex_runtime.graphed import _capturable

# (name, source, expected time_reads)
_CASES = [
    ("none", "@OUT = vec4(0.5, 0.5, 0.5, 1.0);", frozenset()),
    ("frame_only", "@OUT = vec4(vec3(frame * 0.01), 1.0);", frozenset({"frame"})),
    ("fps_only", "@OUT = vec4(vec3(fps * 0.01), 1.0);", frozenset({"fps"})),
    ("time_only", "@OUT = vec4(vec3(time), 1.0);", frozenset({"time"})),
    ("fetch_time_only",
     '@OUT = fetch_time("plate", 0.5, ix, iy);', frozenset({"fetch_time"})),
    ("sample_time_only",
     '@OUT = sample_time("plate", 0.5, u, v);', frozenset({"sample_time"})),
    ("frame_and_fetch_time",
     'float t = frame * 0.1;\n@OUT = fetch_time("plate", t, ix, iy);',
     frozenset({"frame", "fetch_time"})),
    ("all_five",
     'float t = frame + fps + time;\n'
     '@OUT = fetch_time("a", t, ix, iy) + sample_time("b", t, u, v);',
     frozenset({"frame", "fps", "time", "fetch_time", "sample_time"})),
    ("time_inside_user_function",
     "float readTime() {\n"
     "    return time;\n"
     "}\n"
     "@OUT = vec4(vec3(readTime()), 1.0);",
     frozenset({"time"})),
    ("frame_inside_for_loop",
     "float acc = 0.0;\n"
     "for (int i = 0; i < 3; i = i + 1) {\n"
     "    acc = acc + frame;\n"
     "}\n"
     "@OUT = vec4(vec3(acc), 1.0);",
     frozenset({"frame"})),
    ("frame_inside_while_loop",
     "float acc = 0.0;\n"
     "int i = 0;\n"
     "while (i < 3) {\n"
     "    acc = acc + frame;\n"
     "    i = i + 1;\n"
     "}\n"
     "@OUT = vec4(vec3(acc), 1.0);",
     frozenset({"frame"})),
    ("fps_inside_conditional",
     "float pick = 0.0;\n"
     "if (u > 0.5) {\n"
     "    pick = fps;\n"
     "} else {\n"
     "    pick = 0.0;\n"
     "}\n"
     "@OUT = vec4(vec3(pick), 1.0);",
     frozenset({"fps"})),
    ("fetch_time_inside_nested_loop_and_function",
     "vec4 grab(float t) {\n"
     "    return fetch_time(\"plate\", t, ix, iy);\n"
     "}\n"
     "vec4 acc = vec4(0.0);\n"
     "for (int i = 0; i < 2; i = i + 1) {\n"
     "    acc = acc + grab(float(i));\n"
     "}\n"
     "@OUT = acc;",
     frozenset({"fetch_time"})),
    # A non-time sync builtin, and no time read at all — the case the differential test
    # below turns into an explicit assertion: graphed declines this program (erode syncs),
    # but NOT for reading time, so time_reads must stay empty.
    ("non_time_sync_only", "@OUT = erode(@A, 2.0);", frozenset()),
]


def test_timereads45_table(r):
    print("\n--- TIMEREADS-45: _collect_time_reads table (identifiers + calls, incl. "
          "functions/loops/conditionals) ---")
    fails = []
    for name, src, expected in _CASES:
        try:
            program = parse_and_split(src, {})
            got = _collect_time_reads(program)
            if got != expected:
                fails.append(f"{name}: got {sorted(got)}, want {sorted(expected)}")
        except Exception as e:
            fails.append(f"{name}: {type(e).__name__}: {e}")
    if fails:
        r.fail("TIMEREADS-45 table", "; ".join(fails))
    else:
        r.ok(f"{len(_CASES)} programs: identifiers, calls, functions, loops, conditionals, "
             "none, and all five together")


def test_timereads45_program_attribute_is_wired(r):
    print("\n--- TIMEREADS-45: tex_api.Program.time_reads matches the AST-level answer ---")
    try:
        fails = []
        for name, src, expected in _CASES:
            prog = tex_api.compile(src, {})
            if not isinstance(prog.time_reads, frozenset):
                fails.append(f"{name}: time_reads is a {type(prog.time_reads).__name__}, "
                             "not a frozenset")
            elif prog.time_reads != expected:
                fails.append(f"{name}: Program.time_reads={sorted(prog.time_reads)}, "
                             f"want {sorted(expected)}")
        if fails:
            r.fail("TIMEREADS-45 Program wiring", "; ".join(fails))
        else:
            r.ok("tex_api.Program.time_reads agrees with _collect_time_reads(ast) on every row")
    except Exception as e:
        r.fail("TIMEREADS-45 Program wiring", f"{type(e).__name__}: {e}")


def test_timereads45_does_not_move_the_fingerprint(r):
    """A derived, read-only attribute must not perturb the compile fingerprint or any cache
    key — the ask is explicit about this. Nothing in `_compile_impl` computes `time_reads`
    before `fp` is minted, but pin it directly: the SAME source through `TEXCache.fingerprint`
    (independently of `tex_api.compile`) must equal the `fp` `_compile_impl` used."""
    print("\n--- TIMEREADS-45: time_reads does not move the fingerprint ---")
    try:
        from TEX_Wrangle.tex_cache import get_cache
        fails = []
        for name, src, _expected in _CASES:
            prog, fp = tex_api._compile_impl(src, {})
            direct_fp = get_cache().fingerprint(src, {})
            if fp != direct_fp:
                fails.append(f"{name}: fp={fp!r} != direct fingerprint {direct_fp!r}")
            # Re-compiling must reproduce the identical time_reads value (a pure function of
            # the AST alone) without touching the fingerprint either.
            prog2, fp2 = tex_api._compile_impl(src, {})
            if fp2 != fp or prog2.time_reads != prog.time_reads:
                fails.append(f"{name}: re-compile drifted (fp {fp!r}->{fp2!r}, "
                             f"time_reads {sorted(prog.time_reads)}->{sorted(prog2.time_reads)})")
        if fails:
            r.fail("TIMEREADS-45 fingerprint", "; ".join(fails))
        else:
            r.ok("fingerprint unmoved by time_reads; recompute is stable")
    except Exception as e:
        r.fail("TIMEREADS-45 fingerprint", f"{type(e).__name__}: {e}")


def test_timereads45_differential_against_graphed_capture_bar(r):
    """time_reads is non-empty EXACTLY when graphed's capture bar declines for READING TIME —
    not merely whenever it declines at all. `non_time_sync_only` is the row that separates the
    two claims: `erode` syncs (graphed declines it), but it never reads time, so `time_reads`
    must stay empty even though the program is not capturable."""
    print("\n--- TIMEREADS-45: differential vs. graphed._capturable's time-decline branch ---")
    fails = []
    for name, src, expected in _CASES:
        try:
            program = parse_and_split(src, {})
            time_reads = _collect_time_reads(program)
            capturable, _ops = _capturable(program)
            reads_time = bool(time_reads)
            if name == "non_time_sync_only":
                # The isolating case: graphed declines (a sync builtin), time_reads is empty.
                if reads_time:
                    fails.append(f"{name}: time_reads non-empty on a non-time program")
                if capturable:
                    fails.append(f"{name}: expected graphed to decline (erode syncs)")
                continue
            # Every other row in the table has NOTHING ELSE that would make graphed decline
            # (no loops with non-static bounds, no other sync builtin) except a while-loop
            # row, which graphed bars for its OWN reason (not statically-bounded) — exclude
            # it from the direct capturable<->reads_time comparison for that reason, since it
            # would decline even with time_reads empty.
            if name == "frame_inside_while_loop":
                if not reads_time:
                    fails.append(f"{name}: expected a time read")
                continue
            if reads_time and capturable:
                fails.append(f"{name}: time_reads={sorted(time_reads)} but graphed capturable")
            if not reads_time and not capturable:
                fails.append(f"{name}: time_reads empty but graphed declined anyway "
                              "(unexpected decline reason)")
        except Exception as e:
            fails.append(f"{name}: {type(e).__name__}: {e}")
    if fails:
        r.fail("TIMEREADS-45 differential", "; ".join(fails))
    else:
        r.ok("time_reads non-empty exactly when graphed declines for reading time "
             "(isolated from every other decline reason)")


if __name__ == "__main__":
    from helpers import SubTestResult
    r = SubTestResult()
    test_timereads45_table(r)
    test_timereads45_program_attribute_is_wired(r)
    test_timereads45_does_not_move_the_fingerprint(r)
    test_timereads45_differential_against_graphed_capture_bar(r)
    r.summary()
