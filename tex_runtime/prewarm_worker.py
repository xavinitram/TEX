"""
PREWARM-481 — an out-of-process compiled-tier warm worker.

`tex_api.prewarm_async()`'s own background thread (`compiled_capability._get_prewarm_pool()`)
keeps every CHEAP step of `prewarm()` in-process, exactly as it always has (parse/typecheck,
codegen emission + the `.cg` disk persist, the graph-capturability verdict — all measured
bounded). Only the ONE expensive, GIL-sharing step — `compiled._try_compile`'s one-time
toolchain probe plus the first Dynamo/Inductor/Triton lowering for each program — moves here,
into a throwaway CHILD process that shares no GIL with the host at all.

The child does nothing a warm cook would not already do: it calls the ordinary, unmodified
`tex_api.prewarm()` (in its own default `bg_compile_mode="thread"` — this module's subprocess
boundary IS the isolation; nothing inside the child needs to know it is one) against the SAME
on-disk caches this process is using (`TEX_CACHE_DIR`'s `.cg` sidecar, `warm_state.json`, and
PyTorch/Triton's own on-disk kernel caches) — every one of those is inherited through the
child's environment, never special-cased here, so a later real cook in THIS process still finds
a warm on-disk cache, byte-identical to what an un-warmed cook would have produced (invariant 7).

What does NOT cross the process boundary is the parent's in-memory `compiled._compiled_cache`
entry (not picklable, and not the point of this module) or `graphed._capturable_memo` (never
disk-backed). A program warmed this way still pays its own comparatively-cheap Dynamo trace and
capturability probe on this process's first real cook — the same as `"thread"` mode's own
cold-in-THIS-process case for any program the warm never reached at all.

HOUSE-50/H6 (TRK-232, doc-only — no behaviour below changed): an embedding host's own
tooling asked whether this child's VRAM footprint is observable per-process. It is NOT,
from OUTSIDE this process, on every GPU/driver: an external `nvidia-smi`-style query
attributes VRAM to a PID, and that attribution is a driver/OS feature that is not
guaranteed everywhere (WDDM-shared or virtualized/MIG contexts can under- or
mis-attribute), and by the time `warm_in_subprocess` returns, this child process has
already exited (this module's own subprocess call blocks until it does), so there is
nothing left for a host to query even where attribution works. The only point that can
ever answer this reliably is the
child itself, self-reporting BEFORE it exits — `torch.cuda.memory_allocated()`/
`memory_reserved()`/`max_memory_allocated()` on the device it just compiled for, already
available with NO new dependency (`torch` is imported here regardless, via
`tex_api.prewarm()`). This module does not currently thread any such figure into the JSON
summary `_run_worker_main` prints — a real, ADDITIVE field a future ask could add there,
never a fix to what runs today. Recorded here, beside the code it concerns, rather than
only in an orchestration document this repository never ships, so the next reader who
reaches for this exact answer finds it."""
from __future__ import annotations

import json
import os
import subprocess
import sys

# One prewarm batch may pay a first-ever toolchain probe plus a Dynamo/Inductor lowering per
# program; two minutes covers a normal batch. On expiry the child is killed and the caller gets
# {"error": "worker timed out ..."}, after which it warms the batch in-process instead.
_WORKER_TIMEOUT_S = 120.0


def _child_source(pkg_name: str, pkg_dir: str) -> str:
    """The child interpreter's `-c` program. It loads the package from `pkg_dir` under the
    SAME module name this process imported it as (a plain folder name, or, under ComfyUI's
    loader, the folder's absolute path), so it never depends on that name being importable
    from `sys.path`, and pickles written by the child name the same classes as the parent's."""
    init = os.path.join(pkg_dir, "__init__.py")
    return (
        "import sys, importlib, importlib.util\n"
        f"_spec = importlib.util.spec_from_file_location({pkg_name!r}, {init!r},"
        f" submodule_search_locations=[{pkg_dir!r}])\n"
        "_pkg = importlib.util.module_from_spec(_spec)\n"
        f"sys.modules[{pkg_name!r}] = _pkg\n"
        "_spec.loader.exec_module(_pkg)\n"
        f"_m = importlib.import_module({pkg_name + '.tex_runtime.prewarm_worker'!r})\n"
        "sys.exit(_m._run_worker_main())\n"
    )


def warm_in_subprocess(jobs, *, device: str, precision: str, compile_mode: str) -> dict:
    """Warm `jobs` — a list of `(source, binding_types_as_value_map, fingerprint)` triples,
    `binding_types_as_value_map` a plain `{name: TEXType.value}` dict (the caller's own
    serialization; this boundary never guesses a richer shape) — in a child process that
    inherits this one's environment and working directory, so whichever `TEX_CACHE_DIR`/
    `TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` this process is using (relative or not), the
    child uses too.

    Best-effort, like every step of `prewarm()` itself: a launch failure, a non-zero exit, a
    timeout or unparsable output all come back as `{"error": ...}` (with `"bg_compile": 0`)
    rather than raising — a warm-ahead job is an optimization, never load-bearing, and
    `tex_api.prewarm()`'s caller already treats every step this way. On success returns the
    child's own `prewarm()` summary dict (so `summary["bg_compile"]` reports how many of
    `jobs` it actually warmed). With no jobs it returns
    `{"programs": 0, "bg_compile": 0, "error": None}` without starting a child."""
    if not jobs:
        return {"programs": 0, "bg_compile": 0, "error": None}
    payload = {
        "jobs": [{"source": src, "binding_types": bt, "fingerprint": fp}
                 for src, bt, fp in jobs],
        "device": device, "precision": precision, "compile_mode": compile_mode,
    }
    # `__package__` is always correct here: this module is only reached through a relative
    # import, so its root names the package as this process actually imported it.
    _pkg_name = (__package__ or __name__.rsplit(".", 1)[0]).split(".")[0]
    _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../<pkg_name>
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-c", _child_source(_pkg_name, _pkg_dir)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
        try:
            out, err = proc.communicate(json.dumps(payload), timeout=_WORKER_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:   # a grandchild (cl.exe, an Inductor worker) may still hold the pipes: bounded wait
                proc.communicate(timeout=5)
            except Exception:
                pass
            return {"bg_compile": 0, "error": f"worker timed out after {_WORKER_TIMEOUT_S:g}s"}
    except Exception as exc:
        return {"bg_compile": 0, "error": f"worker launch failed: {exc!r}"}
    if proc.returncode != 0:
        detail = err[-2000:]
        try:   # the child prints its own exception as one JSON line before exiting non-zero
            detail = json.loads(out.strip().splitlines()[-1]).get("error") or detail
        except Exception:
            pass
        return {"bg_compile": 0, "error": f"worker exited {proc.returncode}: {detail}"}
    try:
        line = out.strip().splitlines()[-1]
        return json.loads(line)
    except Exception as exc:
        return {"bg_compile": 0, "error": f"worker output unparsable: {exc!r}"}


def _run_worker_main() -> int:
    """Entry point for the child `warm_in_subprocess` spawns (`_child_source`, under whatever
    the real top-level package name is; `-m <pkg>.tex_runtime.prewarm_worker` still reaches the
    same code for manual/debug invocation). Reads a job payload on stdin, warms it via the ordinary `tex_api.prewarm()`,
    prints exactly one JSON line to stdout. This process itself never decides it is "the
    subprocess" for anything — it just runs `prewarm()` the way any other embedding host
    would, which is the whole point."""
    try:
        payload = json.loads(sys.stdin.read())
        from .. import tex_api
        from ..tex_compiler.types import TEXType
        programs = []
        for job in payload["jobs"]:
            bt = {name: TEXType(value) for name, value in job["binding_types"].items()}
            programs.append((job["source"], bt))
        summary = tex_api.prewarm(programs, device=payload["device"],
                                   precision=payload["precision"],
                                   compile_mode=payload["compile_mode"])
        sys.stdout.write(json.dumps(summary) + "\n")
        return 0
    except Exception as exc:
        sys.stdout.write(json.dumps({"bg_compile": 0, "error": repr(exc)}) + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(_run_worker_main())
