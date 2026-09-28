"""Auto-tier: a promotion job serves only its own cook; transient failures never persist a rejection; a lost artifact recompiles."""
import json
import os
import threading

import torch

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_runtime import autotier as AT
from TEX_Wrangle.tex_runtime import compiled as C
from TEX_Wrangle.tex_runtime.interpreter import _collect_identifiers


def _tiny():
    code = "vec3 c=@A.rgb*1.3 - 0.1; c=clamp(c,0.0,1.0); @OUT=vec4(c,1.0);"
    bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
    prog = parse_and_split(code, bt)
    tm = TypeChecker(binding_types=bt, source=code).check(prog)
    return prog, tm, _collect_identifiers(prog)


def _out(res):
    return res["OUT"] if isinstance(res, dict) else res


def _trial_setup(fp, stand_in, size):
    prog, tm, used = _tiny()
    a = torch.rand(1, size, size, 3)
    cache_key = (fp, "cpu", "fp32")
    key = AT.make_key(fp, "cpu", "fp32", C._consensus_extent({"A": a}, prog))
    AT._get(key).state = AT.TRIAL
    C._compiled_cache[cache_key] = (stand_in, "inductor")
    return prog, tm, used, cache_key, key


def _cook(prog, tm, used, fp, img):
    return C.run_auto(prog, {"A": img}, tm, "cpu", fp, output_names=["OUT"], used_builtins=used)


def test_trial_output_is_not_served_to_a_later_cook():
    with cold_engine_state():
        AT.reset()
        release = threading.Event()

        def stand_in(program, binds, tm, dev, lcc, names, scale=None):
            release.wait(5)
            n = binds["A"].shape[1]
            return {"OUT": torch.full((1, n, n, 4), 7.0)}

        fp = "cache52_t11"
        prog, tm, used, cache_key, key = _trial_setup(fp, stand_in, 40)
        try:
            _cook(prog, tm, used, fp, torch.rand(1, 40, 40, 3))       # submits; still pending
            release.set()
            C._trial_futures[cache_key].result(5)
            later = torch.rand(1, 44, 44, 3)
            out = _out(_cook(prog, tm, used, fp, later))
            assert tuple(out.shape[1:3]) == (44, 44)
            assert not bool((out == 7.0).all())
        finally:
            release.set()
            C._drain_bg_for_test()
            C._compiled_cache.pop(cache_key, None)


def _persisted_keys():
    p = AT._persist_path()
    if not p or not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [list(r["key"]) for r in json.load(f).get("verdicts", [])]


def test_transient_trial_failure_never_persists_a_rejection():
    with cold_engine_state():
        AT.reset()

        def stand_in(program, binds, tm, dev, lcc, names, scale=None):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

        fp = "cache52_t33_oom"
        prog, tm, used, cache_key, key = _trial_setup(fp, stand_in, 40)
        try:
            img = torch.rand(1, 40, 40, 3)
            _cook(prog, tm, used, fp, img)      # the job fails inside the bounded wait
            assert AT.verdict(key) == AT.REJECTED
            assert list(key) not in _persisted_keys()
        finally:
            C._drain_bg_for_test()
            C._compiled_cache.pop(cache_key, None)


def test_genuine_trial_failure_still_persists():
    with cold_engine_state():
        AT.reset()

        def stand_in(program, binds, tm, dev, lcc, names, scale=None):
            raise ValueError("the compiled callable is wrong")

        fp = "cache52_t33_bug"
        prog, tm, used, cache_key, key = _trial_setup(fp, stand_in, 40)
        try:
            img = torch.rand(1, 40, 40, 3)
            _cook(prog, tm, used, fp, img)      # the job fails inside the bounded wait
            assert AT.verdict(key) == AT.REJECTED
            assert list(key) in _persisted_keys()
        finally:
            C._drain_bg_for_test()
            C._compiled_cache.pop(cache_key, None)


def test_non_durable_verdict_is_not_carried_to_disk_by_a_later_write():
    with cold_engine_state():
        AT.reset()
        k1 = ("cache52_nd1", "cpu", "fp32", 10)
        k2 = ("cache52_nd2", "cpu", "fp32", 10)
        AT.record_trial(k1, None, persist=False)
        AT.record_trial(k2, None)
        stored = [k[0] for k in _persisted_keys()]
        assert "cache52_nd2" in stored and "cache52_nd1" not in stored


def test_committed_verdict_with_lost_artifact_measures_again():
    with cold_engine_state():
        AT.reset()
        prog, tm, used = _tiny()
        img = torch.rand(1, 40, 40, 3)
        fp = "cache52_t312"
        key = AT.make_key(fp, "cpu", "fp32", C._consensus_extent({"A": img}, prog))
        AT._get(key).state = AT.COMMITTED
        C._compiled_cache.pop((fp, "cpu", "fp32"), None)
        _cook(prog, tm, used, fp, img)
        assert AT.verdict(key) == AT.MEASURING
