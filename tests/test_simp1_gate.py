"""The gate's verdict cache may not serve one interpreter's answer for another.

`tools/gate.py` skips a re-run when the tree has not changed. That is the whole point of it —
a full tier costs about thirteen minutes — but it makes the KEY load-bearing: anything the
verdict depends on and the key does not carry is a wrong answer served instantly and with a
timestamp on it, which is worse than no cache at all.

The key used to end in `os.path.basename(ci_python)`. On Windows that is `python.exe` for every
interpreter there has ever been, so two different Pythons produced the same key and a verdict
measured with one was handed back for the other — observed: a RED belonging to a different
interpreter, which only `--no-cache` got past.

So this file pins the property rather than the spelling: **two interpreters that differ in
anything the run depends on must produce different keys, and the same interpreter must produce
the same one** (a key that never collides is trivially achievable and useless — it would simply
turn the cache off, so both directions are checked).

PORTABILITY. Pure stdlib: no torch, no CUDA, no ComfyUI, no compiler, no network. The fake
interpreter paths are built under a temporary directory, so nothing here names a real machine
and the basenames collide on every operating system.
"""
import importlib.util
import os
import pathlib
import sys
import tempfile

from helpers import SubTestResult

_PKG = pathlib.Path(__file__).resolve().parent.parent


def _gate():
    """Load `tools/gate.py` by path, once per process.

    `tools/` is not a package (it is `.comfyignore`d, like `tests/` and `benchmarks/`), so
    there is no import name to use; loading the file keeps this honest about testing the tool
    the law tells an implementer to run rather than a copy of its logic."""
    mod = sys.modules.get("_simp1_gate")
    if mod is not None:
        return mod
    path = _PKG / "tools" / "gate.py"
    spec = importlib.util.spec_from_file_location("_simp1_gate", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_simp1_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _key_of(g, tree, tier, interpreters, counts=False):
    """The key `g` computes — or, on a build that has no `cache_key`, the key it USED to.

    The fallback is the base sha's expression, spelled out so this row FAILS on that build
    instead of erroring on a missing attribute. A red-first row that dies with an
    AttributeError proves the function is absent; this one proves what its absence cost."""
    fn = getattr(g, "cache_key", None)
    if fn is not None:
        return fn(tree, tier, interpreters, counts)
    return f"{tree}:{tier}:{os.path.basename(interpreters[-1][1])}:{bool(counts)}"


def test_simp1_gate_cache_key_separates_interpreters(r: SubTestResult):
    print("\n--- SIMP-1: the gate's verdict cache key distinguishes interpreters ---")
    g = _gate()
    tree = "0" * 64

    with tempfile.TemporaryDirectory(prefix="tex-gate-key-") as tmp:
        # Two interpreters whose BASENAMES are identical -- the shape every Windows box has,
        # where each one is `python.exe`. Neither exists, which is deliberate: an interpreter
        # the tool cannot query must still key uniquely, or an unqueryable path would collapse
        # every verdict onto one entry.
        a = os.path.join(tmp, "a", "python.exe")
        b = os.path.join(tmp, "b", "python.exe")
        assert os.path.basename(a) == os.path.basename(b), "the fixture must collide on name"

        ka = _key_of(g, tree, "full", [("python", sys.executable), ("ci-python", a)])
        kb = _key_of(g, tree, "full", [("python", sys.executable), ("ci-python", b)])
        if ka == kb:
            r.fail("SIMP-1 cache key",
                   "two interpreters with the same basename produce the SAME cache key, so a "
                   "verdict measured with one is served for the other: "
                   f"{a} and {b} both key {str(ka)[:16]}...")
        else:
            r.ok("two interpreters with the same basename key differently")

        # The other direction: a key that collides with nothing is a cache that never hits.
        again = _key_of(g, tree, "full", [("python", sys.executable), ("ci-python", a)])
        r.ok("the same tree, tier and interpreters key identically (the cache can still hit)") \
            if again == ka else \
            r.fail("SIMP-1 cache key", "the same inputs produced two different keys -- the "
                                       "cache would never hit and the tier is simply off")

        # And the fields that were already load-bearing must still separate.
        if _key_of(g, tree, "cheap", [("python", sys.executable)]) == \
                _key_of(g, tree, "full", [("python", sys.executable)]):
            r.fail("SIMP-1 cache key", "the tier is not in the key")
        elif _key_of(g, "1" * 64, "full", [("python", sys.executable)]) == \
                _key_of(g, tree, "full", [("python", sys.executable)]):
            r.fail("SIMP-1 cache key", "the tree hash is not in the key")
        else:
            r.ok("the tier and the tree hash still separate keys")


def test_simp1_gate_interpreter_identity_is_real(r: SubTestResult):
    """The non-vacuous half: the identity must actually READ the interpreter.

    A version field that was always the same string would satisfy the row above -- the paths
    alone would carry it -- while silently failing to notice a Python upgraded underneath one
    path, which is the other half of what the key is for."""
    print("\n--- SIMP-1: the gate reads the interpreter it is keying on ---")
    g = _gate()
    if not hasattr(g, "interpreter_identity"):
        r.fail("SIMP-1 identity", "tools/gate.py exposes no interpreter_identity()")
        return

    real, version = g.interpreter_identity(sys.executable)
    if os.path.normcase(os.path.realpath(sys.executable)) != real:
        r.fail("SIMP-1 identity", f"resolved path is not this interpreter's: {real}")
    elif sys.version.split(" ")[0] not in version:
        r.fail("SIMP-1 identity",
               f"the version was not read from the interpreter (got {version!r}, this one is "
               f"{sys.version.split(' ')[0]}) -- a constant here cannot see an upgrade")
    else:
        r.ok(f"identity reads the live interpreter: {version.split(' ')[0]} at {real}")

    with tempfile.TemporaryDirectory(prefix="tex-gate-id-") as tmp:
        missing = os.path.join(tmp, "not-an-interpreter")
        path, note = g.interpreter_identity(missing)
        r.ok("an interpreter that cannot be run still yields a unique identity, marked "
             f"unqueryable ({note})") \
            if os.path.normcase(os.path.realpath(missing)) == path and "unqueryable" in note else \
            r.fail("SIMP-1 identity",
                   f"an unqueryable interpreter must still key uniquely, got {(path, note)}")
