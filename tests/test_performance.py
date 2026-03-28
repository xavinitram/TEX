import tempfile
import shutil
import time
from pathlib import Path

import pytest
from helpers import *


@pytest.mark.slow
def test_performance(r: SubTestResult):
    print("\n--- Performance Tests ---")

    B, H, W = 1, 512, 512
    perf_img = torch.rand(B, H, W, 3)

    # Cold compile (fresh cache)
    try:
        tmp_dir = tempfile.mkdtemp()
        cold_cache = TEXCache(cache_dir=Path(tmp_dir))
        code = "float g = luma(@A);\n@OUT = vec3(g, g, g);"

        tokens = Lexer(code).tokenize()
        prog = Parser(tokens).parse()
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
        checker = TypeChecker(binding_types=bt)
        type_map = checker.check(prog)

        start = time.perf_counter()
        interp = Interpreter()
        interp.execute(prog, {"A": perf_img}, type_map, device="cpu")
        cold_ms = (time.perf_counter() - start) * 1000

        shutil.rmtree(tmp_dir, ignore_errors=True)
        assert cold_ms < 2000, f"Cold compile took {cold_ms:.1f}ms (limit: 2000ms)"
        r.ok(f"perf: cold compile ({cold_ms:.1f}ms)")
    except Exception as e:
        r.fail("perf: cold compile", f"{e}\n{traceback.format_exc()}")

    # Hot cached (median of 10)
    try:
        code = "float g = luma(@A);\n@OUT = vec3(g, g, g);"
        # Warm up
        compile_and_run(code, {"A": perf_img})

        times = []
        for _ in range(10):
            start = time.perf_counter()
            compile_and_run(code, {"A": perf_img})
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        median_ms = times[len(times) // 2]
        assert median_ms < 50, f"Hot execution median {median_ms:.1f}ms (limit: 50ms)"
        r.ok(f"perf: hot cached ({median_ms:.1f}ms)")
    except Exception as e:
        r.fail("perf: hot cached", f"{e}\n{traceback.format_exc()}")

    # Complex expression
    try:
        code = "float x = smoothstep(0.0, 1.0, sin(u * PI) * 0.5 + 0.5);\n@OUT = vec3(clamp(x, 0.0, 1.0), lerp(0.0, 1.0, x), x);"
        # Warm up
        compile_and_run(code, {"A": perf_img})

        times = []
        for _ in range(10):
            start = time.perf_counter()
            compile_and_run(code, {"A": perf_img})
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        median_ms = times[len(times) // 2]
        assert median_ms < 100, f"Complex expression median {median_ms:.1f}ms (limit: 100ms)"
        r.ok(f"perf: complex expression ({median_ms:.1f}ms)")
    except Exception as e:
        r.fail("perf: complex expression", f"{e}\n{traceback.format_exc()}")

    # For-loop 10 iterations
    try:
        code = "vec3 sum = vec3(0.0, 0.0, 0.0);\nfor (int i = 0; i < 10; i++) {\n    sum += @A * 0.1;\n}\n@OUT = sum;"
        small_img = torch.rand(1, 256, 256, 3)
        # Warm up
        compile_and_run(code, {"A": small_img})

        times = []
        for _ in range(10):
            start = time.perf_counter()
            compile_and_run(code, {"A": small_img})
            times.append((time.perf_counter() - start) * 1000)
        times.sort()
        median_ms = times[len(times) // 2]
        assert median_ms < 500, f"For-loop median {median_ms:.1f}ms (limit: 500ms)"
        r.ok(f"perf: for-loop 10 iters ({median_ms:.1f}ms)")
    except Exception as e:
        r.fail("perf: for-loop 10 iters", f"{e}\n{traceback.format_exc()}")
