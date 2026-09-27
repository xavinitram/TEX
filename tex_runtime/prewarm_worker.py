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
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_WORKER_TIMEOUT_S = 120.0


def warm_in_subprocess(jobs, *, device: str, precision: str, compile_mode: str) -> dict:
    """Warm `jobs` — a list of `(source, binding_types_as_value_map, fingerprint)` triples,
    `binding_types_as_value_map` a plain `{name: TEXType.value}` dict (the caller's own
    serialization; this boundary never guesses a richer shape) — in a child process that
    inherits this one's environment unchanged, so whichever `TEX_CACHE_DIR`/
    `TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` this process is using, the child uses too.

    Best-effort, like every step of `prewarm()` itself: a launch failure, a non-zero exit, a
    timeout or unparsable output all come back as `{"error": ...}` rather than raising — a
    warm-ahead job is an optimization, never load-bearing, and `tex_api.prewarm()`'s caller
    already treats every step this way. Returns the child's own `prewarm()` summary dict on
    success (so `summary["bg_compile"]` reports how many of `jobs` it actually warmed)."""
    if not jobs:
        return {"programs": 0, "bg_compile": 0, "error": None}
    payload = {
        "jobs": [{"source": src, "binding_types": bt, "fingerprint": fp}
                 for src, bt, fp in jobs],
        "device": device, "precision": precision, "compile_mode": compile_mode,
    }
    # `-m TEX_Wrangle...` resolves only with the directory that CONTAINS `TEX_Wrangle` as the
    # child's cwd -- a fresh interpreter does not inherit the PARENT's `sys.path`, so this
    # cannot rely on however THIS process ended up able to import `TEX_Wrangle` (an embedding
    # host's own loader, a test runner's cwd, ...).
    # Derived from `__file__` rather than assumed, so it is correct regardless of the caller.
    _pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../TEX_Wrangle
    _parent_dir = os.path.dirname(_pkg_dir)                                  # .../custom_nodes
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "TEX_Wrangle.tex_runtime.prewarm_worker"],
            input=json.dumps(payload), capture_output=True, text=True,
            timeout=_WORKER_TIMEOUT_S, env=os.environ.copy(), cwd=_parent_dir,
        )
    except Exception as exc:
        return {"bg_compile": 0, "error": f"worker launch failed: {exc!r}"}
    if proc.returncode != 0:
        return {"bg_compile": 0, "error": f"worker exited {proc.returncode}: {proc.stderr[-2000:]}"}
    try:
        line = proc.stdout.strip().splitlines()[-1]
        return json.loads(line)
    except Exception as exc:
        return {"bg_compile": 0, "error": f"worker output unparsable: {exc!r}"}


def _run_worker_main() -> int:
    """`python -m TEX_Wrangle.tex_runtime.prewarm_worker`: read a job payload on stdin,
    warm it via the ordinary `tex_api.prewarm()`, print exactly one JSON line to stdout. This
    process itself never decides it is "the subprocess" for anything — it just runs `prewarm()`
    the way any other embedding host would, which is the whole point."""
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
