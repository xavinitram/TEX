"""
PORT-2 — the public, host-agnostic API facade.

A small stable surface over the internal compile/execute machinery, for the `tex run`
CLI (PORT-3) and future non-ComfyUI hosts. It calls straight through — `execute()` is
bit-for-bit identical to `Interpreter.execute`, and `compile()` wraps the cache's
6-tuple. The `Program` dataclass field NAMES are a public contract (a canary test pins
them), so a host can depend on `program.assigned` / `program.type_map` etc.

    from TEX_Wrangle.tex_api import compile, execute
    from TEX_Wrangle.tex_compiler.types import TEXType
    prog = compile("@OUT = vec4(@A.rgb * 1.2, 1.0);", {"A": TEXType.VEC3})
    out = execute(prog, {"A": img}, device="cuda", precision="auto")
"""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Program:
    """A compiled TEX program — a named view over the compiler's 6-tuple. The field
    names are a public contract (`test_port2_program_shape` pins them)."""
    ast: Any
    type_map: dict
    referenced: Any
    assigned: dict
    params: dict
    used_builtins: Any
    source: str


def compile(source: str, binding_types: dict) -> Program:  # noqa: A001 (public name)
    """Compile TEX `source` to a `Program`. `binding_types` maps input binding names to
    their `TEXType`. Raises LexerError / ParseError / TypeCheckError on invalid code."""
    from .tex_cache import get_cache
    ast, type_map, referenced, assigned, params, used_builtins = \
        get_cache().compile_tex(source, binding_types)
    return Program(ast, type_map, referenced, assigned, params, used_builtins, source)


def execute(program: Program, bindings: dict, *, device: str = "cpu",
            precision: str = "fp32", output_names=None) -> dict:
    """Execute a compiled `Program` against `bindings` → `{output_name: tensor}`. Calls
    straight through to `Interpreter.execute` (bit-for-bit identical)."""
    from .tex_runtime.interpreter import Interpreter
    outs = output_names if output_names is not None else sorted(program.assigned.keys())
    return Interpreter().execute(
        program.ast, bindings, program.type_map, device=device,
        output_names=outs, precision=precision, used_builtins=program.used_builtins)
