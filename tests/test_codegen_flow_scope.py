"""Native break/continue belongs to the loop that licensed it, not to the loops inside it.

A static-range `for` is emitted as a Python `for` over a precomputed range, and a `while` keeps
its iteration counter at the top of its body, so in both a TEX `break`/`continue` can be emitted
as the Python statement — there is no update or counter for a native `continue` to skip. A
general `for` (a non-static header) is emitted as `while <counter> < _MAX_ITER:` with the user's
update statement AND the counter increment at the BOTTOM of the body, so its transfers must be
exception-based (`raise _CgContinue()` caught below the body) or a native `continue` skips both.

`_CodeGen._use_native_flow_control` says which form to emit. The static and `while` emitters set
it for their own bodies, but the general-`for` emitter did not, so the flag LEAKED into a nested
general loop: its `continue` was emitted natively and skipped the update and the counter, and the
loop never terminated — the counter that exists to cap a runaway loop is exactly what the leaked
`continue` jumps over. Neither tier's iteration limit can fire, so the cook hangs rather than
raising E6010, and only under codegen (the interpreter's `_exec_loop_body` catches `_Continue`
and runs the update either way). Each emitter now pins the mode for its own body and restores it.

`break` is deliberately untouched: inside a general loop a native `break` leaves the emitted
`while` exactly as `except _CgBreak: break` does, so it stays native and those programs emit
byte-identical code (the general emitter pins the mode only when its body holds a `continue`).

Rows. `test_flow_scope_emission` is the in-process pin: no bare `continue` may be emitted inside
a general loop's body. The other two rows RUN programs, in a CHILD PROCESS with a per-program
timeout, because the defect's signature is non-termination: a timeout is the failure, and it can
never hang the suite. Every row reads the serving tier from `tier_trace` before comparing pixels
(a parity check against a silent interpreter fallback compares the interpreter with itself), then
compares bit-exactly (invariant 2). The matrix covers static/general/while as outer and inner
loop, with break/continue in either, the outer transfer placed AFTER the nested loop so a missing
RESTORE is caught too, on the raw path and on the engine path (optimizer included). Loop counts
are 9, above the optimizer's small-loop unroll bound, so the engine path keeps the shapes; each
raw row asserts the emitted loop kinds it claims to exercise, so a header that silently stops
being general cannot make the row vacuous.
"""
from helpers import *

import itertools
import json
import queue
import subprocess
import threading

from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_runtime.codegen import try_compile

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(_PKG)                 # the directory that holds TEX_Wrangle/

# A 2x2 cook of a scalar loop program is microseconds of work; 30 s is ~5 orders of margin, so a
# row that hits it did not terminate. Tunable for a slow/contended runner (and for the red-first
# reading, where every hanging row costs its whole budget).
_TIMEOUT = float(os.environ.get("TEX_FLOW_TIMEOUT", "30"))
_IMPORT_TIMEOUT = float(os.environ.get("TEX_FLOW_IMPORT_TIMEOUT", "300"))

_N = 9   # above the optimizer's small-loop unroll bound, so the engine path keeps each loop

# F2, verbatim from the control-flow engine finding: a static outer loop with a break (which
# licenses native flow control) around a general inner loop whose `continue` skips its update.
# The interpreter gives 4.0; codegen did not terminate.
_T15 = """float acc = 0.0;
int j = 0;
for (int i = 0; i < 3; i++) {
    if (i == 2) { break; }
    for (j = 0; j < 3; j = j + 1) {
        if (j == 1) { continue; }
        acc += 1.0;
    }
}
@OUT = vec3(acc, 0.0, 0.0) + @A * 0.0;
"""

# The same program with a STATIC inner header (`int j` declared in it): native flow control is
# correct there, so this one always terminated. It guards the fix from over-reaching.
_T15_STATIC_INNER = """float acc = 0.0;
for (int i = 0; i < 3; i++) {
    if (i == 2) { break; }
    for (int j = 0; j < 3; j = j + 1) {
        if (j == 1) { continue; }
        acc += 1.0;
    }
}
@OUT = vec3(acc, 0.0, 0.0) + @A * 0.0;
"""


def _loop(kind, var, body):
    if kind == "static":
        return f"for (int {var} = 0; {var} < {_N}; {var}++) {{\n{body}}}\n"
    if kind == "general":
        return f"for ({var} = 0; {var} < {_N}; {var} = {var} + 1) {{\n{body}}}\n"
    # `while` advances its counter at the top of the body, so a `continue` cannot skip it
    return f"{var} = 0;\nwhile ({var} < {_N}) {{\n{var} = {var} + 1;\n{body}}}\n"


def _matrix_program(outer, inner, o_xfer, i_xfer):
    inner_body = ""
    if i_xfer == "break":
        inner_body += "if (j == 2) { break; }\n"
    elif i_xfer == "continue":
        inner_body += "if (j == 1) { continue; }\n"
    inner_body += "acc += 1.0;\n"
    outer_body = ""
    if o_xfer == "break":
        outer_body += "if (i == 2) { break; }\n"
    outer_body += _loop(inner, "j", inner_body)
    if o_xfer == "continue":
        # after the nested loop: emitted once the inner loop has restored the flow mode
        outer_body += "if (i == 1) { continue; }\n"
    outer_body += "acc += 100.0;\n"
    decls = "float acc = 0.0;\n"
    decls += "" if outer == "static" else "int i = 0;\n"
    decls += "" if inner == "static" else "int j = 0;\n"
    return decls + _loop(outer, "i", outer_body) + "@OUT = vec3(acc, 0.0, 0.0) + @A * 0.0;\n"


def _expect(levels):
    """What the raw emission must look like for a shape, as (kind, transfer) per loop.

    Both directions of the fix: a `continue` in a general loop MUST be emitted as a signal
    (or it skips that loop's update), and a `continue` in a static or `while` loop must NOT
    be (those keep native flow control, so the fix may not reach them).
    """
    return {
        "range": sum(1 for kind, _x in levels if kind == "static"),
        "counted": sum(1 for kind, _x in levels if kind != "static"),
        "signal_continue": any(x == "continue" and kind == "general" for kind, x in levels),
        "native_continue": any(x == "continue" and kind != "general" for kind, x in levels),
    }


def _matrix():
    """(name, code, expected emission) for every nested shape in the matrix."""
    kinds = ("static", "general", "while")
    rows = []
    for outer, inner in itertools.product(kinds, kinds):
        for o_xfer, i_xfer in itertools.product((None, "break", "continue"), repeat=2):
            if o_xfer is None and i_xfer is None:
                continue
            name = f"{outer}[{o_xfer or '-'}] > {inner}[{i_xfer or '-'}]"
            rows.append((name, _matrix_program(outer, inner, o_xfer, i_xfer),
                         _expect(((outer, o_xfer), (inner, i_xfer)))))
    return rows


# Shapes the matrix's two levels cannot express: a leak through an intermediate loop, a transfer
# under a PER-PIXEL condition (codegen's spatial branch), a bound the general emitter cannot read
# statically, and the variant that TERMINATES with the wrong answer instead of hanging.
_EXTRA = {
    "static[break] > static[-] > general[continue]": """float acc = 0.0;
int k = 0;
for (int i = 0; i < 9; i++) {
    if (i == 2) { break; }
    for (int j = 0; j < 9; j++) {
        for (k = 0; k < 9; k = k + 1) {
            if (k == 1) { continue; }
            acc += 1.0;
        }
        acc += 10.0;
    }
    acc += 100.0;
}
@OUT = vec3(acc, 0.0, 0.0) + @A * 0.0;
""",
    "while > general[break] > general[continue]": """float acc = 0.0;
int i = 0; int j = 0; int k = 0;
while (i < 9) {
    i = i + 1;
    for (j = 0; j < 9; j = j + 1) {
        if (j == 2) { break; }
        for (k = 0; k < 9; k = k + 1) {
            if (k == 1) { continue; }
            acc += 1.0;
        }
        acc += 10.0;
    }
    acc += 100.0;
}
@OUT = vec3(acc, 0.0, 0.0) + @A * 0.0;
""",
    "general[continue] after a nested while": """float acc = 0.0;
int i = 0; int j = 0;
for (i = 0; i < 9; i = i + 1) {
    j = 0;
    while (j < 9) {
        j = j + 1;
        if (j == 2) { break; }
        acc += 1.0;
    }
    if (i == 1) { continue; }
    acc += 100.0;
}
@OUT = vec3(acc, 0.0, 0.0) + @A * 0.0;
""",
    "static[break] > general[per-pixel continue]": """float acc = 0.0;
int j = 0;
for (int i = 0; i < 9; i++) {
    if (i == 2) { break; }
    for (j = 0; j < 9; j = j + 1) {
        if (u < 0.5) { continue; }
        acc += 1.0;
    }
    acc += 100.0;
}
@OUT = vec3(acc, 0.0, 0.0) + @A * 0.0;
""",
    "static[break] > general[continue], non-literal bound": """float acc = 0.0;
int j = 0;
for (int i = 0; i < 9; i++) {
    if (i == 2) { break; }
    for (j = 0; float(j) < 9.0; j = j + 1) {
        if (j == 1) { continue; }
        acc += 1.0;
    }
    acc += 100.0;
}
@OUT = vec3(acc, 0.0, 0.0) + @A * 0.0;
""",
    "static[break] > general[continue], body advances the counter": """float acc = 0.0;
int j = 0;
for (int i = 0; i < 9; i++) {
    if (i == 2) { break; }
    for (j = 0; j < 9; j = j + 1) {
        if (j == 1) { j = j + 1; continue; }
        acc += 1.0;
    }
    acc += 100.0;
}
@OUT = vec3(acc, 0.0, 0.0) + @A * 0.0;
""",
}


# ── the child: one program per line of stdout, so a hang is attributable ──────────────
_CHILD = r'''
import json, os, sys
jobs_path, root, cache_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.environ["TEX_CACHE_DIR"] = cache_dir          # before any TEX import (CACHE-0)
sys.path.insert(0, root)
with open(jobs_path, encoding="utf-8") as f:
    jobs = json.load(f)
import torch
torch.set_num_threads(1)
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_cache import get_cache
from TEX_Wrangle.tex_marshalling import infer_binding_type
from TEX_Wrangle.tex_runtime.interpreter import Interpreter
from TEX_Wrangle.tex_runtime.codegen import try_compile
from TEX_Wrangle.tex_runtime.compiled import _codegen_only_execute
from TEX_Wrangle.tex_runtime import tier_trace


def compile_one(code, bindings, mode):
    bt = {n: infer_binding_type(v) for n, v in bindings.items()}
    program = Parser(Lexer(code).tokenize(), source=code).parse()
    if mode == "engine":     # the pipeline a cook runs: optimize + re-type-check
        program, tm, _refs, assigned, _params, _used = get_cache().compile_ast(
            program, bt, source=code)
        return program, tm, sorted(assigned.keys())
    checker = TypeChecker(binding_types=bt, source=code)
    tm = checker.check(program)
    return program, tm, sorted(checker.assigned_bindings.keys())


def run(job):
    dev = torch.device(job["device"])
    img = torch.zeros(1, 2, 2, 3, device=dev)
    program, tm, outs = compile_one(job["code"], {"A": img}, job["mode"])
    ref = Interpreter().execute(program, {"A": img.clone()}, tm, device=dev,
                                output_names=outs)[outs[0]]
    fn = try_compile(program, tm)
    src = None if fn is None else fn._tex_src
    tier_trace.reset()
    got = _codegen_only_execute(program, {"A": img.clone()}, tm, dev, output_names=outs,
                                fingerprint=None, time_context=None)[outs[0]]
    rec = tier_trace.last()
    return {
        "tier": None if rec is None else rec.tier,
        "fell_back_from": None if rec is None else rec.fallback_from,
        "reason": None if rec is None else (str(rec.reason)[:160] if rec.reason else None),
        "declined": fn is None,
        "equal": bool(torch.equal(ref.float(), got.float())),
        "maxdiff": float((ref.float() - got.float()).abs().max()),
        "ref": [round(float(x), 6) for x in ref.reshape(-1).tolist()[:4]],
        "got": [round(float(x), 6) for x in got.reshape(-1).tolist()[:4]],
        "native_continue": None if src is None else any(
            ln.strip() == "continue" for ln in src.splitlines()),
        "signal_continue": None if src is None else ("raise _CgContinue()" in src),
        "range_loops": None if src is None else src.count("for _i_idx in range("),
        "counted_loops": None if src is None else src.count("< _MAX_ITER:"),
    }


sys.stdout.write("READY\n")
sys.stdout.flush()
for _i, _job in enumerate(jobs):
    try:
        _res = run(_job)
    except Exception as _e:
        _res = {"error": type(_e).__name__ + ": " + str(_e)[:200]}
    _res["i"] = _i
    sys.stdout.write(json.dumps(_res) + "\n")
    sys.stdout.flush()
'''


def _kill(proc):
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except Exception:
        pass
    for stream in (proc.stdout, proc.stderr):
        try:
            stream.close()
        except Exception:
            pass


def _run_jobs(jobs):
    """Run each job in a child process; return one result dict per job, in order.

    A job that produces no line within `_TIMEOUT` is recorded as {"timeout": True} and its
    child is killed — the hang stays in a process the suite can end, and the jobs after it
    still run (a fresh child picks them up). One child serves the whole list when nothing
    hangs, so the green path pays a single interpreter start-up.
    """
    results = [None] * len(jobs)
    tmp = tempfile.mkdtemp(prefix="texflow_")
    try:
        script = os.path.join(tmp, "flow_child.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(_CHILD)
        remaining = list(range(len(jobs)))
        while remaining:
            jobs_file = os.path.join(tmp, f"jobs_{remaining[0]}.json")
            with open(jobs_file, "w", encoding="utf-8") as f:
                json.dump([jobs[i] for i in remaining], f)
            env = dict(os.environ, PYTHONIOENCODING="utf-8",
                       TEX_CACHE_DIR=os.path.join(tmp, "cache"))
            proc = subprocess.Popen(
                [sys.executable, "-X", "utf8", script, jobs_file, _ROOT,
                 os.path.join(tmp, "cache")],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="replace", env=env)
            lines: "queue.Queue" = queue.Queue()

            def _pump(stream=proc.stdout):
                try:
                    for line in stream:
                        lines.put(line)
                finally:
                    lines.put(None)

            threading.Thread(target=_pump, daemon=True).start()

            done_through = -1          # index (within `remaining`) of the last finished job
            budget = _IMPORT_TIMEOUT   # the first line waits for the child's imports
            stalled = False
            while done_through + 1 < len(remaining):
                try:
                    line = lines.get(timeout=budget)
                except queue.Empty:
                    results[remaining[done_through + 1]] = {"timeout": True, "after": budget}
                    done_through += 1
                    stalled = True
                    break
                if line is None:       # the child exited without reporting this job
                    err = ""
                    try:
                        err = proc.stderr.read() or ""
                    except Exception:
                        pass
                    results[remaining[done_through + 1]] = {
                        "error": f"child exited (rc={proc.poll()}): {err.strip()[-300:]}"}
                    done_through += 1
                    stalled = True
                    break
                line = line.strip()
                if line == "READY":
                    budget = _TIMEOUT       # imports are done; the rest is per-program work
                    continue
                if not line.startswith("{"):
                    continue                # anything else a library printed to stdout
                rec = json.loads(line)
                done_through = rec.pop("i")
                results[remaining[done_through]] = rec
            _kill(proc)
            remaining = remaining[done_through + 1:] if stalled else []
        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _jobs_for(rows, devices, modes=("raw", "engine")):
    """rows: [(name, code, expect|None)] -> ([(label, expect, mode), ...], [job, ...])."""
    labels, jobs = [], []
    for name, code, expect in rows:
        for device, mode in itertools.product(devices, modes):
            labels.append((f"{name} [{device}/{mode}]", expect, mode))
            jobs.append({"code": code, "device": device, "mode": mode})
    return labels, jobs


def _check(r, label, expect, mode, res):
    """One row: it terminated, codegen served it, and its pixels are the interpreter's."""
    if res is None:
        r.fail(label, "no result recorded")
        return
    if res.get("timeout"):
        r.fail(label, f"did NOT terminate under codegen within {res['after']:.0f}s "
                      "(native flow control leaked into a general loop)")
        return
    if res.get("error"):
        r.fail(label, res["error"])
        return
    if res.get("declined") or res.get("tier") != "codegen":
        r.fail(label, f"codegen did not serve it (tier={res.get('tier')}, "
                      f"fell_back_from={res.get('fell_back_from')}, reason={res.get('reason')}) "
                      "— a parity check the interpreter served twice proves nothing")
        return
    # The emission expectations read the RAW path: the engine path is free to rewrite a shape.
    if expect is not None and mode == "raw":
        got = {"range": res.get("range_loops"), "counted": res.get("counted_loops"),
               "signal_continue": res.get("signal_continue"),
               "native_continue": res.get("native_continue")}
        if (got["range"], got["counted"]) != (expect["range"], expect["counted"]):
            r.fail(label, f"emitted {got['range']} range loop(s) / {got['counted']} counted "
                          f"loop(s), expected {expect['range']}/{expect['counted']} — the row "
                          "is not exercising the shape it names")
            return
        if got["signal_continue"] != expect["signal_continue"]:
            r.fail(label, "a `continue` in a general loop must be emitted as a signal and one "
                          f"elsewhere must not: signalled={got['signal_continue']}, "
                          f"expected {expect['signal_continue']}")
            return
        if got["native_continue"] != expect["native_continue"]:
            r.fail(label, "a `continue` in a static or while loop must stay native: "
                          f"native={got['native_continue']}, expected "
                          f"{expect['native_continue']}")
            return
    if not res.get("equal"):
        r.fail(label, f"codegen != interpreter (maxdiff {res.get('maxdiff')}, "
                      f"interp {res.get('ref')} vs codegen {res.get('got')})")
        return
    r.ok(f"{label} -> {res.get('ref')}")


def test_flow_scope_emission(r: SubTestResult):
    """In-process pin: a general loop's body must never carry a native `continue`."""
    print("\n--- flow scope: emitted flow-control form ---")
    for name, code in (("F2 (general inner)", _T15), ("static inner", _T15_STATIC_INNER)):
        try:
            program = Parser(Lexer(code).tokenize(), source=code).parse()
            tm = TypeChecker(binding_types={"A": TEXType.VEC3}, source=code).check(program)
            fn = try_compile(program, tm)
            assert fn is not None, "codegen declined the program"
            src = fn._tex_src
            native = [k + 1 for k, ln in enumerate(src.splitlines()) if ln.strip() == "continue"]
            if name.startswith("F2"):
                assert "raise _CgContinue()" in src, \
                    "the nested general loop's `continue` was not emitted as a signal"
                assert not native, (
                    f"native `continue` emitted at line(s) {native} — it skips the general "
                    "loop's update statement and its iteration counter")
                # the outer static loop keeps its native break (no update to skip)
                assert any(ln.strip() == "break" for ln in src.splitlines()), \
                    "the static outer loop's `break` stopped being native"
            else:
                assert native and "raise _CgContinue()" not in src, \
                    "a static loop's `continue` must stay native (no update to skip)"
            r.ok(f"emission: {name}")
        except Exception as e:
            r.fail(f"emission: {name}", f"{type(e).__name__}: {e}")


def test_flow_scope_t15_nested_general_loop_terminates(r: SubTestResult):
    """T15: the F2 program terminates under codegen and equals the interpreter (4.0)."""
    print("\n--- flow scope T15: a general loop nested in a native-flow loop ---")
    rows = [("F2 (general inner)", _T15,
             _expect((("static", "break"), ("general", "continue")))),
            ("F2 with a static inner loop", _T15_STATIC_INNER,
             _expect((("static", "break"), ("static", "continue"))))]
    labels, jobs = _jobs_for(rows, devices())
    results = _run_jobs(jobs)
    for (label, expect, mode), res in zip(labels, results):
        _check(r, label, expect, mode, res)
        if res and not res.get("timeout") and not res.get("error") and res.get("ref"):
            if abs(res["ref"][0] - 4.0) > 1e-6:
                r.fail(label, f"the interpreter's own answer moved: {res['ref'][0]} != 4.0")


def test_flow_scope_nested_loop_matrix(r: SubTestResult):
    """Every nested (outer x inner) loop shape with a transfer in either loop."""
    print("\n--- flow scope: nested loop matrix, codegen == interpreter ---")
    rows = _matrix()
    labels, jobs = _jobs_for(rows, devices())
    results = _run_jobs(jobs)
    for (label, expect, mode), res in zip(labels, results):
        _check(r, label, expect, mode, res)


def test_flow_scope_deeper_and_per_pixel_shapes(r: SubTestResult):
    """Leaks through an intermediate loop, a per-pixel transfer, and the divergence variant."""
    print("\n--- flow scope: deeper nestings and per-pixel transfers ---")
    rows = [(name, code, None) for name, code in sorted(_EXTRA.items())]
    labels, jobs = _jobs_for(rows, devices())
    results = _run_jobs(jobs)
    for (label, expect, mode), res in zip(labels, results):
        _check(r, label, expect, mode, res)
