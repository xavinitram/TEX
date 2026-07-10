"""
TST-8 — release gate.

(1) version-consistency: __init__.__version__ == pyproject version == the top
    CHANGELOG heading (a real drift risk — three files must agree at release).
(2) codegen-emission determinism: the generated source must be byte-identical
    across two PYTHONHASHSEEDs, enforcing the `sorted()`-over-sets invariant that
    keeps compiled artifacts reproducible (persistable, cache-stable).

The publish workflow (`.github/workflows/publish_action.yml`) runs the test suite
before publishing, so a red suite can't ship — the CI half of the gate.
"""
import os
import re
import subprocess
import sys
import tempfile
from helpers import SubTestResult

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # TEX_Wrangle/
_CUSTOM_NODES = os.path.dirname(_PKG)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def test_version_consistency(r: SubTestResult):
    print("\n--- TST-8: version consistency ---")
    try:
        init_v = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']',
                           _read(os.path.join(_PKG, "__init__.py"))).group(1)
        pyproj_v = re.search(r'(?m)^version\s*=\s*["\']([^"\']+)["\']',
                             _read(os.path.join(_PKG, "pyproject.toml"))).group(1)
        chlog_v = re.search(r'(?m)^##\s*\[([0-9]+\.[0-9]+\.[0-9]+)\]',
                            _read(os.path.join(_PKG, "CHANGELOG.md"))).group(1)
        assert init_v == pyproj_v == chlog_v, \
            f"version drift: __init__={init_v} pyproject={pyproj_v} CHANGELOG={chlog_v}"
        r.ok(f"version consistent across __init__/pyproject/CHANGELOG ({init_v})")
    except Exception as e:
        r.fail("version consistency", f"{type(e).__name__}: {e}")


_PROBE = f'''
import sys
sys.path.insert(0, r"{_CUSTOM_NODES}")
from TEX_Wrangle.tex_compiler.lexer import Lexer
from TEX_Wrangle.tex_compiler.parser import Parser
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker
from TEX_Wrangle.tex_compiler.types import TEXType
from TEX_Wrangle.tex_runtime.codegen import try_compile
code = ("float a = @A.r * u + @B.g * v; float b = @A.b + float(ix) * py; "
        "@OUT = vec4(a, b, u * v, 1.0);")
prog = Parser(Lexer(code).tokenize(), source=code).parse()
tm = TypeChecker(binding_types={{"A": TEXType.VEC3, "B": TEXType.VEC3, "OUT": TEXType.VEC4}},
                 source=code).check(prog)
fn = try_compile(prog, tm, fingerprint="det")
sys.stdout.write(getattr(fn, "_tex_src", "NONE") if fn is not None else "NONE")
'''


def _emit_under_seed(seed: str, script_path: str) -> str:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    out = subprocess.run([sys.executable, script_path], capture_output=True,
                         text=True, env=env, timeout=180)
    return out.stdout


def test_codegen_determinism(r: SubTestResult):
    print("\n--- TST-8: codegen-emission determinism (PYTHONHASHSEED) ---")
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as tf:
            tf.write(_PROBE)
            script = tf.name
        try:
            src0 = _emit_under_seed("0", script)
            src1 = _emit_under_seed("1", script)
        finally:
            os.unlink(script)
        assert src0 and src0 != "NONE", f"probe produced no source: {src0!r}"
        assert src0 == src1, ("codegen emission is NOT hash-seed-stable — a set/dict "
                              "is iterated without sorted(). First diff:\n"
                              f"seed0:\n{src0[:400]}\nseed1:\n{src1[:400]}")
        r.ok("codegen source byte-identical across PYTHONHASHSEED 0/1 (sorted() invariant holds)")
    except Exception as e:
        r.fail("codegen determinism", f"{type(e).__name__}: {e}")


def test_scatter_determinism_band(r: SubTestResult):
    print("\n--- TST-8 / A1-4: CUDA scatter-determinism release band ---")
    # A1-4: read the value the PR-LP5 pin recorded and gate a release on out-of-band
    # drift, even if the pin itself was in SOFT (WARN) mode. Must run AFTER the pin.
    try:
        import test_determinism_pin as pin
        val = pin.LAST_CUDA_DET_VAR
        if val is None:
            # No CUDA on this runner (the CPU CI lane) — can't gate what wasn't measured.
            # This is the honest hardware limitation S-4 (validate-hw) exists to close.
            r.ok("scatter-determinism band: not measured (no CUDA on this runner) — SKIPPED")
        elif val <= pin._CUDA_DET_BAND:
            r.ok(f"scatter-determinism within release band ({val:.1e} <= {pin._CUDA_DET_BAND:.0e})")
        else:
            r.fail("scatter-determinism band",
                   f"recorded CUDA run-to-run {val:.2e} > band {pin._CUDA_DET_BAND:.0e} "
                   "— do not publish; resolve the atomic-add ordering regression (A1-4)")
    except Exception as e:
        r.fail("scatter-determinism band", f"{type(e).__name__}: {e}")
