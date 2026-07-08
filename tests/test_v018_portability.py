"""
v0.18.0 portability ladder (PORT-1 HostServices seam + import lint; PORT-2 API facade).
"""
from helpers import *
import re

_PKG = Path(__file__).resolve().parent.parent
_IMPORT_RE = re.compile(r"^\s*(import\s+comfy\.model_management|from\s+comfy\.model_management)",
                        re.M)


def test_port1_import_lint(r: SubTestResult):
    print("\n--- PORT-1: comfy.model_management import is pinned to host.py ---")
    offenders = []
    for path in _PKG.rglob("*.py"):
        rel = path.relative_to(_PKG).as_posix()
        if rel == "tex_runtime/host.py" or "/tests/" in f"/{rel}" or rel.startswith("tests/"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in _IMPORT_RE.finditer(text):
            line = text[:m.start()].count("\n") + 1
            offenders.append(f"{rel}:{line}")
    if offenders:
        r.fail("PORT-1 import lint",
               "comfy.model_management imported outside host.py (re-scatter): "
               + ", ".join(offenders))
    else:
        r.ok("comfy.model_management is imported ONLY in tex_runtime/host.py (seam intact)")


def test_port1_host_services(r: SubTestResult):
    print("\n--- PORT-1: HostServices Null contract + Comfy forwarding ---")
    from TEX_Wrangle.tex_runtime import host
    fails = []

    # NullHostServices: unknown free memory, no-op hooks, torch-only OOM detection
    null = host.NullHostServices()
    if null.get_free_memory("cuda") is not None:
        fails.append("Null.get_free_memory should be None")
    null.free_memory(1, "cuda"); null.soft_empty_cache()  # must not raise
    if null.is_oom(ValueError("x")) is not False:
        fails.append("Null.is_oom(non-OOM) should be False")

    # ComfyHostServices forwards to the module (a fake stands in for comfy.model_management)
    class _FakeMM:
        def __init__(self): self.freed = None; self.emptied = False
        def get_free_memory(self, dev): return 4242
        def free_memory(self, amt, dev): self.freed = (amt, dev)
        def soft_empty_cache(self): self.emptied = True
        def is_oom(self, e): return isinstance(e, KeyError)
    fake = _FakeMM()
    comfy = host.ComfyHostServices(fake)
    if comfy.get_free_memory("cuda") != 4242:
        fails.append("Comfy.get_free_memory did not forward")
    comfy.free_memory(99, "cuda")
    if fake.freed != (99, "cuda"):
        fails.append("Comfy.free_memory did not forward")
    comfy.soft_empty_cache()
    if not fake.emptied:
        fails.append("Comfy.soft_empty_cache did not forward")
    if comfy.is_oom(KeyError()) is not True or comfy.is_oom(ValueError()) is not False:
        fails.append("Comfy.is_oom did not forward")

    # a host whose API raises degrades to Null behaviour (never propagates)
    class _BrokenMM:
        def get_free_memory(self, dev): raise RuntimeError("boom")
        def free_memory(self, amt, dev): raise RuntimeError("boom")
        def soft_empty_cache(self): raise RuntimeError("boom")
        def is_oom(self, e): raise RuntimeError("boom")
    broken = host.ComfyHostServices(_BrokenMM())
    try:
        if broken.get_free_memory("cuda") is not None:
            fails.append("broken host free-memory should degrade to None")
        broken.free_memory(1, "cuda"); broken.soft_empty_cache()  # must not raise
        broken.is_oom(ValueError())  # must not raise
    except Exception as e:
        fails.append(f"broken host propagated: {e}")

    # resolver + override
    host.set_host_services(null)
    if host.get_host_services() is not null:
        fails.append("set_host_services override not honored")
    host.reset_host_services()

    if fails:
        r.fail("PORT-1 host services", "; ".join(fails))
    else:
        r.ok("Null no-ops (torch-only OOM); Comfy forwards; a broken host degrades; "
             "resolver override honored")
