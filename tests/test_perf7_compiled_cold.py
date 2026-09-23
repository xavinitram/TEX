"""PERF-7 — what a COLD cook on the compiled tier is allowed to do, pinned as counts.

WHAT A "COLD COMPILED COOK" IS. `benchmarks/eight_config_bench.py`'s `cpu_on_cold` /
`cuda_on_cold` configs time exactly this shape, once per program, in an isolated
subprocess::

    clear_compiled_cache()
    execute_compiled(program, bindings, type_map, device, fingerprint, ...)

`clear_compiled_cache()` drops the compiled-callable cache, the route memos and the
codegen memo's MEMORY tier — but not the parse, not the type check (they happened once,
outside the timed region) and not the marshalled `.cg` sidecar on disk. So the work a
cold cook is SUPPOSED to do is: re-enter the compile route once, fetch the already
generated function back from the codegen cache, rebuild the coordinate-grid env, and
run. It is not supposed to re-lex, re-parse, re-emit, re-fold or re-tag anything.

WHY THIS FILE EXISTS. A `cpu_on_cold` geomean below the bench's 0.95 stop-ship threshold
is not, on its own, evidence of anything: `docs/roadmap.md` §10 item 3 records a null
control on a BYTE-IDENTICAL tree returning 1.105 for this very config with individual
rows spanning 0.88–2.32, and a measured `cpu_on_cold` of **0.926** — signature: a fixed
few-millisecond addition on the SHORT compiled programs only — that turned out to be a
test suite running in another shell. Timings on shared hardware cannot settle the
question "did the cold compiled path grow work?". Counts can, and these are they. The
rows that matter are exact integers that repeat cook after cook and process after
process: a named function either runs on this path or it does not. (The grand TOTAL is
the one quantity that wobbles by a frame or two between cooks, which is the second
reason it is gated as a ceiling.)

WHAT EACH ROW IS FOR. `_MUST_NOT_RUN` names, one by one, the mechanisms a perf lane could
plausibly leak onto this path — and each entry is a thing that has been proposed or
shipped nearby, not a hypothetical:

  * `Lexer.tokenize` / `Parser.parse` / `TEXCache.parse_and_split` — a re-parse per cook.
  * `ast_nodes.clone_tree` / `tex_roi._pristine_program` / `tex_roi._fold_program` /
    `tex_roi._walk` — PERF-1's ROI-walk parse memo and the AST copy it hands out. That
    copy is correct where it lives (the interactive ROI walk) and would be pure loss
    here; the cook path must never reach it.
  * `codegen._CodeGen.build` / `codegen.try_compile` — a re-EMIT per cold cook. This is
    the expensive one, and the one a change that defeated the codegen cache would show:
    the sidecar is what makes a cold cook cost milliseconds instead of tens of them.
  * `stdlib_core._tag_host_scalar` (LIB-1: moved off `stdlib.py` onto its per-domain leaf,
    still reached through `stdlib._tag_host_scalar` — the facade re-export) — PERF-2's
    host-scalar tag. The tag is paid once per tensor
    MINTED, and a hoisted constant is minted by the generated preamble's own assignment
    (the emitter writes `_t1._tex_host_scalar = 2.0` as a literal store), so no call
    reaches this function on a cook. A count here means the tag became per-cook work.
  * `compiled._params_on_device` — the opt-in device-param placement. It is a learned,
    non-CPU route; on the default CPU cook it must not run at all.

`_EXACTLY_ONCE` is the other half: the seams that legitimately run once per cold cook.
Pinning them at 1 keeps the zeros above from being satisfied by a cook that never
happened, and catches the opposite failure — a cook that re-entered the compile route
twice.

WHY THE TOTAL IS A CEILING AND NOT AN EQUALITY. The grand total of Python frames is a
claim partly about CPython, not about TEX: 3.12 inlines comprehensions, so a
`<listcomp>` frame that exists on 3.10 does not exist on 3.12. Three releases in a row
were blocked by tests that pinned something whose contract was never exact
(`CHANGELOG.md` / the release record), so the total is gated as an upper bound with the
measured readings written down beside it, while the exact equalities are reserved for
named TEX functions, which either run or do not.

PORTABILITY. CPU only, no ComfyUI, no CUDA, no compiler, no numpy, no timing assertion.
The programs are declared here rather than loaded from `benchmarks/`, so the file has no
dependency on a `.comfyignore`d directory. If the box cannot persist a `.cg` sidecar the
codegen rows are not measurable and the affected rows SKIP rather than pass.
"""
import os
import sys

from helpers import *
# By name, not by star: `helpers.__all__` is pinned to its v0.35.0 set (HOOK-4), because
# `from helpers import *` is a surface a host's own suite binds.
from helpers import load_counts_harness

from TEX_Wrangle.tex_runtime import compiled as _compiled
from TEX_Wrangle.tex_runtime.compiled import execute_compiled, clear_compiled_cache


#: Every spelling `co_filename` can carry for a file in this package, from the ONE
#: implementation the counts harness owns. This used to be
#: `normcase(Path(__file__).resolve().parents[1]) + os.sep` — a single RESOLVED
#: spelling — and `resolve()` follows the `custom_nodes\TEX_Wrangle -> TEX` junction
#: while imported modules keep the junction spelling in `co_filename`. Run from
#: `custom_nodes` (the canonical location) the filter therefore matched NOTHING and
#: every row below read zero, so the two tests errored there and were green in any
#: worktree. A second, hand-spelled copy of the harness's filter is what made that
#: possible, so there is no second copy any more.
_counts = load_counts_harness()
_PKG_PREFIXES = _counts.path_prefixes(str(Path(__file__).parents[1]))
#: Bound once: `_hook` runs on every call event, so it may not do a lookup per frame.
_package_relpath = _counts.package_relpath
#: And the ONE qualified-name resolver, for the same reason a second copy of the path filter
#: is not allowed here: `co_qualname` is 3.11+, so below that a method has to be NAMED from
#: its frame or every `Class.method` row in this file counts zero (see `frame_qualname`).
_frame_qualname = _counts.frame_qualname

#: Two programs that reach the compiled tier on CPU and take the codegen-only adapter
#: (they call stdlib functions, so `_has_fn_calls` is set and `torch.compile` is never
#: entered — see `compiled._try_compile`). One of them calls `gauss_blur`, which is the
#: only class of program whose emitted preamble carries PERF-2's host-scalar stores, so
#: the `_tag_host_scalar` row below is measured where it could actually fire.
_PROGRAMS = {
    "blur_chain": """
vec3 base = @A.rgb;
vec3 soft = gauss_blur(@A, 2.0).rgb;
vec3 detail = base - soft;
vec3 sharp = base + detail * 1.5;
vec3 mixed = lerp(soft, sharp, 0.5);
vec3 toned = clamp(mixed * 1.2 - 0.05, 0.0, 1.0);
@OUT = vec4(toned, 1.0);
""",
    "fetch_stencil": """
vec3 c = fetch(@A, ix, iy).rgb;
vec3 t = fetch(@A, ix, iy - 1).rgb;
vec3 b = fetch(@A, ix, iy + 1).rgb;
vec3 l = fetch(@A, ix - 1, iy).rgb;
vec3 rr = fetch(@A, ix + 1, iy).rgb;
vec3 s = c * 5.0 - t - b - l - rr;
@OUT = vec4(clamp(lerp(c, s, 0.75), 0.0, 1.0), 1.0);
""",
}

#: `module/path:qualname -> 0`. Every one of these is a mechanism that has a right to
#: exist somewhere else and no right to exist on a cold compiled cook. See the docstring.
_MUST_NOT_RUN = (
    "tex_compiler/lexer:Lexer.tokenize",
    "tex_compiler/parser:Parser.parse",
    "tex_cache:parse_and_split",
    "tex_compiler/ast_nodes:clone_tree",
    "tex_roi:_pristine_program",
    "tex_roi:_fold_program",
    "tex_roi:_walk",
    "tex_runtime/codegen:_CodeGen.build",
    "tex_runtime/codegen:try_compile",
    "tex_runtime/stdlib_core:_tag_host_scalar",  # LIB-1: `_tag_host_scalar` now lives on this leaf
    "tex_runtime/compiled:_params_on_device",
)

#: The seams a cold cook runs EXACTLY once. A 0 here means the cook did not happen (which
#: would make every zero above vacuous); a 2 means the compile route was re-entered.
_EXACTLY_ONCE = (
    "tex_runtime/compiled:execute_compiled",
    "tex_runtime/compiled:_try_compile",
    "tex_runtime/compiled:_build_codegen_env",
    "tex_runtime/codegen:_invoke_cg",
)

#: Total TEX Python frames on one cold cook at 64^2. A CEILING, not an equality — see the
#: docstring. The margin (~23 %) is generous enough to absorb a CPython frame-model
#: difference and far too small to absorb a re-emit, which the mutation guard below
#: measures at roughly five times the pinned reading.
_FRAME_CEILING = {"blur_chain": 340, "fetch_stencil": 400}
#: v0.37.0 (`dfe7c38`) read 276 / 321; head reads 280 / 324 (RTX 5070 Ti Laptop, this box,
#: `python tests/test_perf7_compiled_cold.py -q -s`, each row prints its own). One extra
#: frame on `blur_chain` is PERF-2's `_host_scalar` call inside `fn_gauss_blur` — one Python
#: frame in place of a device readback, which is the trade that lane recorded; the rest of
#: the drift is later, unrelated frame growth the ceiling (340/400) already has room for.
_MEASURED_AT_V0370 = {"blur_chain": 276, "fetch_stencil": 321}

REDERIVE = ("python -m pytest tests/test_perf7_compiled_cold.py -q  "
            "(each row prints its reading; the ceilings are in _FRAME_CEILING)")


class _Frames:
    """Count `call` events in files under the package, keyed `module/path:qualname`.

    `sys.setprofile`, not `settrace`: one event per call instead of one per line. The same
    hook `benchmarks/host_path_counts.py::FrameCounter` installs, and now the same FILTER
    too: the row keys differ (`module/path:qualname` here, dotted there) but deciding
    whether a frame belongs to the package is one rule, in one place, because two copies
    of it is exactly how this hook came to count zero without failing."""

    def __init__(self):
        self.counts = {}

    def _hook(self, frame, event, arg):
        if event != "call":
            return
        code = frame.f_code
        rel = _package_relpath(code.co_filename, _PKG_PREFIXES)
        if rel is None:
            return
        mod = rel[:-3] if rel.endswith(".py") else rel
        key = mod + ":" + _frame_qualname(code, frame)
        self.counts[key] = self.counts.get(key, 0) + 1

    def __enter__(self):
        sys.setprofile(self._hook)
        return self

    def __exit__(self, *exc):
        sys.setprofile(None)
        return False


class _FakeCode:
    """The three attributes a profile hook reads, with a `co_filename` the caller chooses.

    `co_name` and `co_qualname` are spelled the SAME on purpose: the filter probes below run
    under the native resolver and under the forced pre-3.11 fallback, and a synthetic frame
    carries no `self`, so both paths must land on one row name for the probe to mean the same
    thing on either. It is also what a module-level function looks like on 3.10."""

    co_firstlineno = 1
    co_name = "probe"
    co_qualname = "probe"

    def __init__(self, filename):
        self.co_filename = filename


class _FakeFrame:
    def __init__(self, filename):
        self.f_code = _FakeCode(filename)


def _fake_frame(filename):
    """A frame the counter's filter must decide about, without running anything real."""
    return _FakeFrame(filename)


class _InlinePool:
    """Run the compile lifecycle on THIS thread.

    `execute_compiled` hands `_compile_and_run` to a worker thread for dynamo-TLS
    isolation, and `sys.setprofile` is per-thread — so a counter installed on the caller
    sees 51 frames of a cook that does several hundred. Running it inline does the same
    work in the same order and makes it visible. This is a measurement device, not a
    behaviour change: it is installed for the duration of one cook and removed."""

    def submit(self, fn, *a, **kw):
        import concurrent.futures
        f = concurrent.futures.Future()
        try:
            f.set_result(fn(*a, **kw))
        except BaseException as e:   # noqa: BLE001 — the future is the caller's to raise
            f.set_exception(e)
        return f


def _front_end(code: str):
    """Lex + parse + type-check, through the production seam (`tex_cache.parse_and_split`
    and `TypeChecker`), which is what a bench worker does once per program OUTSIDE its
    timed region. Returns (program, type_map, binding_types, fingerprint)."""
    from TEX_Wrangle.tex_cache import get_cache, parse_and_split
    btypes = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    program = parse_and_split(code, btypes)
    type_map = TypeChecker(binding_types=btypes, source=code).check(program)
    return program, type_map, btypes, get_cache().fingerprint(code, btypes)


def _prepare(name: str, res: int = 64):
    """Everything `execute_compiled` needs for one program, front end already spent."""
    code = _PROGRAMS[name]
    program, type_map, _btypes, fp = _front_end(code)
    bindings = {"A": torch.rand(1, res, res, 3, dtype=torch.float32)}
    return program, bindings, type_map, fp, None, None


def _cook(prepared, count: bool):
    """One cook of a prepared program. With `count`, it is a COLD cook under the frame
    counter and the counts come back; without, it is a plain warm-up cook."""
    program, bindings, type_map, fp, out_names, used = prepared
    if not count:
        execute_compiled(program, dict(bindings), type_map, "cpu", fp,
                         output_names=out_names, used_builtins=used)
        return None
    pool = _compiled._COMPILE_POOL
    _compiled._COMPILE_POOL = _InlinePool()
    try:
        clear_compiled_cache()
        f = _Frames()
        with f:
            execute_compiled(program, dict(bindings), type_map, "cpu", fp,
                             output_names=out_names, used_builtins=used)
        return f.counts
    finally:
        _compiled._COMPILE_POOL = pool


def _cold_counts(name: str):
    """Warm the codegen sidecar the way a bench worker's warm-up leg does, then take the
    counts of ONE cold cook. Returns (counts, status) where status is the bench's own
    honest classification."""
    prepared = _prepare(name)
    fp = prepared[3]
    clear_compiled_cache()
    _cook(prepared, count=False)
    status = ("compiled" if (fp, "cpu", "fp32") in _compiled._compiled_cache
              else "fallback" if fp in _compiled._compile_blacklist else "plain")
    entry = _compiled._compiled_cache.get((fp, "cpu", "fp32"))
    backend = entry[1] if entry is not None else "n/a"
    return _cold_counts_only(prepared), status, backend


def _cold_counts_only(prepared):
    return _cook(prepared, count=True)


def _sidecar_live(prepared) -> bool:
    """True when the marshalled `.cg` sidecar can actually serve this fingerprint.

    `clear_compiled_cache()` drops the codegen memo's memory tier, so a cold cook reaches
    the sidecar. On a box that cannot write one (read-only cache dir, a concurrent
    holder) every cold cook legitimately re-emits, and the codegen rows below would red
    for an environment reason rather than a tree one. Detected, not assumed."""
    from TEX_Wrangle.tex_cache import get_cache
    cache = get_cache()
    cache._codegen_memory.clear()
    try:
        return cache._load_codegen_from_disk(prepared[3]) is not None
    except Exception:
        return False


# ── the pins ────────────────────────────────────────────────────────────────

def test_perf7_cold_compiled_cook_does_no_front_end_work(r: SubTestResult):
    """A cold cook on the compiled tier re-lexes, re-parses, re-folds, re-emits and
    re-tags NOTHING. Each name is a mechanism with a legitimate home elsewhere."""
    print("\n--- PERF-7: a cold compiled cook does no front-end work ---")
    for name in _PROGRAMS:
        try:
            prepared = _prepare(name)
            clear_compiled_cache()
            _cook(prepared, count=False)
            key = (prepared[3], "cpu", "fp32")
            if key not in _compiled._compiled_cache:
                r.skip(f"PERF-7 {name}", "this program did not reach the compiled tier on "
                                         "this box — the cold-compiled path is not measurable")
                continue
            live = _sidecar_live(prepared)
            counts = _cold_counts_only(prepared)
            bad = []
            for row in _MUST_NOT_RUN:
                n = counts.get(row, 0)
                if n == 0:
                    continue
                if not live and row.startswith("tex_runtime/codegen:"):
                    continue     # no persistable sidecar on this box — see _sidecar_live
                bad.append(f"{row}: 0 -> {n} per cold cook")
            if not live:
                r.skip(f"PERF-7 {name} codegen rows",
                       "no `.cg` sidecar could be read back on this box, so a cold cook "
                       "legitimately re-emits; the re-emit rows are not measurable here")
            if bad:
                r.fail(f"PERF-7 {name} cold cook", "; ".join(bad) + f" || the cold compiled "
                       f"path grew work that belongs elsewhere; re-derive with `{REDERIVE}`")
            else:
                r.ok(f"{name}: none of the {len(_MUST_NOT_RUN)} front-end seams ran on a "
                     f"cold compiled cook")
        except Exception as e:
            r.fail(f"PERF-7 {name}", f"{type(e).__name__}: {e}")


def test_perf7_cold_compiled_cook_runs_the_route_once(r: SubTestResult):
    """The seams a cold cook DOES run, pinned at one apiece — so the zeros above cannot be
    passed by a cook that never happened, and a doubled compile-route entry is caught."""
    print("\n--- PERF-7: the cold compiled route is entered exactly once ---")
    for name in _PROGRAMS:
        try:
            prepared = _prepare(name)
            clear_compiled_cache()
            _cook(prepared, count=False)
            if (prepared[3], "cpu", "fp32") not in _compiled._compiled_cache:
                r.skip(f"PERF-7 once {name}", "not on the compiled tier on this box")
                continue
            counts = _cold_counts_only(prepared)
            bad = [f"{row}: 1 -> {counts.get(row, 0)}"
                   for row in _EXACTLY_ONCE if counts.get(row, 0) != 1]
            if bad:
                r.fail(f"PERF-7 once {name}", "; ".join(bad) +
                       f" || re-derive with `{REDERIVE}`")
            else:
                r.ok(f"{name}: execute_compiled / _try_compile / _build_codegen_env / "
                     f"_invoke_cg all exactly 1 per cold cook")
        except Exception as e:
            r.fail(f"PERF-7 once {name}", f"{type(e).__name__}: {e}")


def test_perf7_cold_compiled_cook_frame_ceiling(r: SubTestResult):
    """Total TEX Python frames on one cold cook, gated as a ceiling.

    The zeros above name the mechanisms known today. This row is what catches work that
    arrives under a name nobody thought to list — it is deliberately loose, because the
    exact total is partly a property of the CPython that runs it."""
    print("\n--- PERF-7: total TEX frames on a cold compiled cook (ceiling) ---")
    for name, ceiling in _FRAME_CEILING.items():
        try:
            prepared = _prepare(name)
            clear_compiled_cache()
            _cook(prepared, count=False)
            if (prepared[3], "cpu", "fp32") not in _compiled._compiled_cache:
                r.skip(f"PERF-7 frames {name}", "not on the compiled tier on this box")
                continue
            if not _sidecar_live(prepared):
                r.skip(f"PERF-7 frames {name}",
                       "no `.cg` sidecar on this box — a cold cook re-emits, so the total "
                       "is not comparable to the pinned reading")
                continue
            total = sum(_cold_counts_only(prepared).values())
            if total > ceiling:
                r.fail(f"PERF-7 frames {name}",
                       f"{total} TEX python frames on one cold compiled cook, ceiling "
                       f"{ceiling} (v0.37.0 read {_MEASURED_AT_V0370[name]}): work was "
                       f"added to the cold path; re-derive with `{REDERIVE}`")
            else:
                r.ok(f"{name}: {total} frames per cold cook (ceiling {ceiling}, "
                     f"v0.37.0 read {_MEASURED_AT_V0370[name]})")
        except Exception as e:
            r.fail(f"PERF-7 frames {name}", f"{type(e).__name__}: {e}")


def test_perf7_the_counter_is_not_inert(r: SubTestResult):
    """The mutation guard, both halves.

    Most rows above assert that a named function ran ZERO times. A counter that could not
    see those functions — a renamed target, a module reached under a second name, a
    qualname spelled wrong — satisfies every one of them, and the file would report a
    contract it is not measuring. That is the ANIM-1 failure, verbatim
    (`tests/test_bench2_counts.py` carries the same guard for the same reason).

    Half one: drive each forbidden mechanism under the counter and require it to be SEEN.
    Half two: defeat the codegen cache and require the cold-cook pin to notice — the
    mutation that stands in for the regression this lane was sent to find."""
    print("\n--- PERF-7 mutation guard: the counter sees what it forbids ---")

    # Half zero: the FILTER itself. Every row in this file is "X ran N times", and the filter
    # decides which frames are even looked at — so a filter that accepts one spelling of this
    # package's path and not another zeroes every row at once, silently, and every "must not
    # run" row then passes vacuously. It did: run from `custom_nodes`, through the
    # `TEX_Wrangle -> TEX` junction, this file read `4 passed, 2 errors` while any worktree
    # read green. Drive a synthetic frame under EACH accepted spelling and require it counted,
    # and one from outside the package and require it ignored.
    bad = []
    for pref in _PKG_PREFIXES:
        probe = _Frames()
        probe._hook(_fake_frame(pref + "tex_engine.py"), "call", None)
        if probe.counts.get("tex_engine:probe") != 1:
            bad.append(f"a frame spelled {pref!r} was NOT counted ({probe.counts})")
    outside = _Frames()
    outside._hook(_fake_frame(os.path.join(tempfile.gettempdir(), "not_tex.py")), "call", None)
    if outside.counts:
        bad.append(f"a frame OUTSIDE the package was counted: {outside.counts}")
    r.fail("PERF-7 frame filter", "; ".join(bad)) if bad else \
        r.ok(f"the frame filter accepts all {len(_PKG_PREFIXES)} spelling(s) of this package "
             f"and nothing outside it: {', '.join(_PKG_PREFIXES)}")

    from TEX_Wrangle.tex_cache import get_cache, parse_and_split
    from TEX_Wrangle.tex_compiler import ast_nodes as _ast
    from TEX_Wrangle import tex_roi as _roi
    from TEX_Wrangle.tex_runtime import stdlib as _stdlib
    from TEX_Wrangle.tex_runtime import codegen as _codegen

    src = _PROGRAMS["blur_chain"]
    program, type_map, _bt, _fp = _front_end(src)
    f = _Frames()
    with f:
        _ast.clone_tree(parse_and_split(src, {}))
        _roi._fold_program(src, {})
        _stdlib._tag_host_scalar(torch.zeros((), dtype=torch.float32), 0.5)
        _codegen.try_compile(program, type_map)
    seen = f.counts
    for row in _MUST_NOT_RUN:
        if row in ("tex_roi:_walk", "tex_runtime/compiled:_params_on_device"):
            continue     # driven by their own owners' suites; not reachable from here
        n = seen.get(row, 0)
        if n > 0:
            r.ok(f"the counter sees {row} ({n}): the zero pinned above is not vacuous")
        else:
            r.fail("PERF-7 inert spy", f"{row} counted 0 while being called directly: the "
                   f"row name is wrong, so its zero above is vacuous")

    # Half two: a cold cook with the codegen cache defeated MUST red the pin.
    prepared = _prepare("blur_chain")
    clear_compiled_cache()
    _cook(prepared, count=False)
    if (prepared[3], "cpu", "fp32") not in _compiled._compiled_cache:
        r.skip("PERF-7 defeated-cache mutation", "not on the compiled tier on this box")
        return
    cache = get_cache()
    real = cache.get_codegen_fn
    cache.get_codegen_fn = lambda fp: None      # every cold cook re-emits
    try:
        counts = _cold_counts_only(prepared)
    finally:
        cache.get_codegen_fn = real
    emitted = counts.get("tex_runtime/codegen:_CodeGen.build", 0)
    total = sum(counts.values())
    if emitted > 0 and total > _FRAME_CEILING["blur_chain"]:
        r.ok(f"with the codegen cache defeated a cold cook emits again "
             f"(_CodeGen.build = {emitted}, {total} frames > ceiling "
             f"{_FRAME_CEILING['blur_chain']}): the pin would catch it")
    else:
        r.fail("PERF-7 mutation", f"defeating the codegen cache did not move the pin "
               f"(_CodeGen.build = {emitted}, {total} frames vs ceiling "
               f"{_FRAME_CEILING['blur_chain']}): the cold-cook rows cannot fail")


#: An attribute name no code object carries, on any interpreter. Setting the harness's
#: `_QUALNAME_ATTR` to it makes `frame_qualname` take the branch Python 3.10 takes, so the
#: pre-3.11 keying can be measured on a box that has no 3.10 — which is every box here.
_FORCED_ATTR = "co_qualname_absent_before_python_3_11"

#: `(the row this file pins, the bare-name spelling a naive 3.10 hook would write instead)`.
#: These three are the only `Class.method` rows in `_MUST_NOT_RUN`; every other row there and
#: in `_EXACTLY_ONCE` is a module-level function, whose `co_name` IS its qualified name.
_QUALNAME_ROWS = (
    ("tex_compiler/lexer:Lexer.tokenize", "tex_compiler/lexer:tokenize"),
    ("tex_compiler/parser:Parser.parse", "tex_compiler/parser:parse"),
    ("tex_runtime/codegen:_CodeGen.build", "tex_runtime/codegen:build"),
)


def _drive_qualname_rows(tag: str, forced: bool) -> dict:
    """Call the three pinned METHODS under the counter and return its rows.

    Two distinct sources, both unique to `tag`: the program `try_compile` is handed is lexed
    and parsed OUTSIDE the counted region (that is what `_front_end` is for), and PERF-5's
    token offer would otherwise let the counted `parse_and_split` claim the stream that lex
    produced — `Lexer.tokenize` would then legitimately not run and the row would read zero
    for a reason that has nothing to do with the keying this row is about."""
    from TEX_Wrangle.tex_cache import parse_and_split
    from TEX_Wrangle.tex_runtime import codegen as _codegen

    program, type_map, _bt, _fp = _front_end(_PROGRAMS["blur_chain"] + f"// prepared {tag}\n")
    counted_src = _PROGRAMS["blur_chain"] + f"// counted {tag}\n"
    saved = _counts._QUALNAME_ATTR
    if forced:
        _counts._QUALNAME_ATTR = _FORCED_ATTR
    try:
        f = _Frames()
        with f:
            parse_and_split(counted_src, {})
            _codegen.try_compile(program, type_map)
        return f.counts
    finally:
        _counts._QUALNAME_ATTR = saved


def test_perf7_the_row_keys_survive_a_missing_co_qualname(r: SubTestResult):
    """PY-3.10 — the row keys may not depend on `co_qualname`, which is 3.11+.

    CI runs 3.10, 3.11 and 3.12. `co_qualname` arrived in 3.11, so on the 3.10 leg a method
    frame used to key `tex_compiler/lexer:tokenize` while every pin in this file reads
    `tex_compiler/lexer:Lexer.tokenize`. Those rows then counted ZERO whatever the tree did:
    the three `Class.method` entries in `_MUST_NOT_RUN` passed vacuously and the mutation
    guard above reported them "counted 0 while being called directly" — 3.10 red, 3.11 and
    3.12 green, which is the shape of the run this row was written for.

    NEITHER development box has a 3.10 interpreter, so the fallback cannot be run here by
    running it. It is forced instead, through the one seam the resolver reads its attribute
    name from, and two things are asserted: that the three rows are counted at all with the
    fallback taken (the half that was red), and that the key it derives is the SAME string
    the native path produces on this interpreter (the half that stops the fallback drifting
    away from 3.11+ unnoticed). The real proof is the Linux 3.10 leg of CI.

    ON a 3.10 interpreter both legs take the fallback and the comparison degenerates into the
    measurement itself — the three rows are still required to be counted, which is precisely
    the thing that was red there. So this row is not guarded by an interpreter check and does
    not skip: a row that cannot run on the interpreter it was written for is not a witness."""
    print("\n--- PERF-7: the row keys survive a missing co_qualname (Python 3.10) ---")
    try:
        native = _drive_qualname_rows("native", forced=False)
        forced = _drive_qualname_rows("forced", forced=True)
    except Exception as e:
        r.fail("PERF-7 qualname fallback", f"{type(e).__name__}: {e}")
        return
    if _counts._QUALNAME_ATTR != "co_qualname":
        r.fail("PERF-7 qualname seam", f"the seam was left at {_counts._QUALNAME_ATTR!r}: "
               f"every later row in this process would be measured through the fallback")
        return
    for row, bare in _QUALNAME_ROWS:
        n, fb = native.get(row, 0), forced.get(row, 0)
        if n < 1:
            r.fail("PERF-7 qualname drive", f"{row} counted 0 under the NATIVE resolver — "
                   f"the drive does not reach it, so this row proves nothing about 3.10")
        elif fb < 1:
            r.fail("PERF-7 qualname fallback", f"{row} counted {fb} with `co_qualname` forced "
                   f"missing while being called directly ({bare!r} read "
                   f"{forced.get(bare, 0)} instead): on Python 3.10 this row's zero above is "
                   f"vacuous")
        elif forced.get(bare, 0):
            r.fail("PERF-7 qualname fallback", f"the fallback also wrote the bare-name row "
                   f"{bare!r} ({forced[bare]}): two spellings of one function is the drift "
                   f"this row exists to catch")
        else:
            r.ok(f"{row}: {n} native / {fb} with the pre-3.11 fallback forced, same key, and "
                 f"no bare-name {bare!r} row")


if __name__ == "__main__":   # derivation helper: print the readings this file pins
    for _name in _PROGRAMS:
        _c, _st, _b = _cold_counts(_name)
        print(f"{_name}: status={_st} backend={_b} total={sum(_c.values())}")
        for _k, _v in sorted(_c.items(), key=lambda kv: -kv[1])[:14]:
            print(f"    {_v:6d}  {_k}")
