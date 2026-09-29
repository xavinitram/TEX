"""
`tex doctor`: the CPU inductor prerequisite is reported from the toolchain that is actually
present, not assumed.
"""
import sys

from TEX_Wrangle import tex_doctor as D


def _which(found):
    return lambda name: (name if name in found else None)


def test_posix_without_a_c_compiler_is_unavailable(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(D.shutil, "which", _which(()))
    ok, why = D._inductor_prereq("cpu")
    assert ok is False and "C compiler" in why


def test_posix_with_a_c_compiler_holds(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(D.shutil, "which", _which(("gcc",)))
    assert D._inductor_prereq("cpu") == (True, None)


def test_windows_with_cl_on_path_holds(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(D.shutil, "which", _which(("cl",)))
    assert D._inductor_prereq("cpu") == (True, None)


def test_windows_include_alone_is_not_a_compiler(monkeypatch):
    from TEX_Wrangle.tex_runtime import compiled
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(D.shutil, "which", _which(()))
    monkeypatch.setenv("INCLUDE", "C:\\some\\include")
    monkeypatch.setattr(compiled, "_msvc_env_initialized", False)
    assert D._inductor_prereq("cpu") == (None, None)          # not knowable, and not "holds"
    monkeypatch.setattr(compiled, "_msvc_env_initialized", True)
    ok, why = D._inductor_prereq("cpu")
    assert ok is False and "cl.exe" in why
