"""Threads in one long process: cooking does not grow them, and nothing blocks exit.

`compiled._COMPILE_POOL` / `_WARM_POOL` were `ThreadPoolExecutor`s, whose exit hook joins
every worker ever started. K5 (`compiled._pool_for`) abandons a pool whose job is stuck, and
that stuck worker was still joined at exit, so after one abandonment the process could never
exit. The pools now drain only what `compiled` currently holds, and a live pool still runs its
queued jobs at exit (a prewarm child relies on that).
"""
import subprocess
import sys
import threading
from pathlib import Path

import torch

from TEX_Wrangle import tex_engine
from TEX_Wrangle.tex_cookqueue import CookQueue

_CUSTOM_NODES = str(Path(__file__).resolve().parents[2])

_HEAD = r'''
import os, sys, threading, time
sys.path.insert(0, sys.argv[1])
os.environ["TEX_CACHE_DIR"] = sys.argv[2]
from TEX_Wrangle.tex_runtime import compiled as C
'''


def _run(body, tmp_path, timeout):
    """(finished, returncode, stdout+stderr) of a fresh interpreter running `body`."""
    try:
        proc = subprocess.run([sys.executable, "-X", "utf8", "-c", _HEAD + body, _CUSTOM_NODES,
                               str(tmp_path / "cache"), str(tmp_path)],
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, None, ""
    return True, proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def test_an_abandoned_stuck_pool_does_not_block_exit(tmp_path):
    body = r'''
never = threading.Event()
def stuck():
    C._mark_pool_busy("warm")
    never.wait()
C._pool_for("warm").submit(stuck)
time.sleep(0.3)
C._POOL_STUCK_BOUND_S = 0.1
assert C._pool_for("warm") is C._WARM_POOL
print("ABANDONED", flush=True)
'''
    finished, rc, out = _run(body, tmp_path, timeout=60)
    assert finished, "the process never exited: the abandoned pool's stuck job was joined"
    assert rc == 0 and "ABANDONED" in out, out[-800:]


def test_a_live_pool_still_runs_its_queue_at_exit(tmp_path):
    body = r'''
out = os.path.join(sys.argv[3], "ran.txt")
def job(i):
    def run():
        time.sleep(0.3)
        with open(out, "a") as f:
            f.write(f"{i}\n")
    return run
for i in range(2):
    C._pool_for("compile").submit(job(i))
'''
    finished, rc, out = _run(body, tmp_path, timeout=60)
    assert finished and rc == 0, out[-800:]
    assert (tmp_path / "ran.txt").read_text().split() == ["0", "1"]


_PROGRAMS = (
    "@OUT = @A * 0.5;",
    "@OUT = vec4(fbm(u * 8.0, v * 8.0, 3));",
    "float s = 0.0; for (int i = 0; i < 4; i++) { s += sin(u * i); } @OUT = vec4(s);",
    "@OUT = vec4(worley_f1(u * 5.0, v * 5.0));",
    "@OUT = vec4(simplex(u * 3.0, v * 3.0) + @A.r);",
)


def _cook(i):
    img = torch.rand(1, 12 + i % 5, 14 + i % 3, 4)
    tex_engine.cook(_PROGRAMS[i % len(_PROGRAMS)], {"A": img}, device_mode="cpu",
                    precision="fp32")


def test_many_cooks_keep_the_thread_count_bounded():
    """No thread per cook: after a warm-up, 60 more cooks of assorted programs and shapes
    leave `threading.active_count()` where it was (the lazily started pools are already up)."""
    for i in range(len(_PROGRAMS)):
        _cook(i)
    before = threading.active_count()
    for i in range(60):
        _cook(i)
    assert threading.active_count() <= before + 1, [t.name for t in threading.enumerate()]


def test_a_closed_cook_queue_leaves_no_thread_behind():
    before = threading.active_count()
    for _ in range(3):
        q = CookQueue(name="tex-thread-bound")
        jobs = [q.submit(lambda cancel, i=i: _cook(i)) for i in range(5)]
        for j in jobs:
            j.result(timeout=60)
        q.close(timeout=10)
    alive = [t.name for t in threading.enumerate() if t.name == "tex-thread-bound"]
    assert not alive and threading.active_count() <= before, alive
