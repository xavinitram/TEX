"""
TRK-236 (v0.51, "COMPILE-51" item 2) — `_get_param_local`'s emitted preamble minted a
`$param` scalar into a CPU tensor with `_torch.as_tensor(value)`, carrying no host reading
forward (unlike `_stage_wire_scalars`, `compiled.py::_params_on_device` and `Interpreter`'s
own bind loop, which all already tag — PERF-2, `stdlib_core._tag_host_scalar`'s own module
comment). `gauss_blur`/`bilateral_filter`'s own `_host_scalar` kernel-radius read then found
no tag on that tensor and fell through to a raw `.item()` -- reproduced as a genuine graph
break under a real `torch.compile`d cook (CPU, `backend="aot_eager"`, no CUDA/Triton needed
to prove a graph break exists — mirrors `test_compiletry50_gaussblur_dynamo.py`'s own
portability posture).

FIXED: `_stage_codegen_param` (seeded into the generated module as `_THS`, `codegen.py`'s
`build()` / `codegen_persist.py`'s `materialize_codegen()`, same pattern as `_MF`/`_SCM`),
called from the SAME preamble line the pre-existing `_torch.as_tensor(` call already
occupies — `test_codegen_value_parity.py::test_codegen_vec_param_staging_leaves_emitted_code_alone`
pins that exact substring, so the fix could not remove it; it wraps it instead
(`_THS(local, _torch.as_tensor(local))`). Tagging must happen INSIDE the traced preamble
text, not by pre-staging `bindings` before the codegen'd function is invoked: an earlier
draft tried pre-staging (the `_stage_wire_scalars` shape) and measured it inert under a real
`torch.compile` trace — a value already bound in `bindings` when handed to a
`torch.compile`-wrapped callable is treated by Dynamo as a traced GRAPH INPUT, and a custom
Python instance attribute like `_tex_host_scalar` is not visible from inside the trace even
though the real object carries it. A tensor minted from a Python constant INSIDE the traced
preamble (Dynamo constant-folds `torch.as_tensor(raw_python_float)`) does not have this
problem — confirmed directly below.

PORTABILITY. CPU only, `backend="aot_eager"` (traces for real with Dynamo, no C++ compiler,
no CUDA/Triton) — same posture as `test_compiletry50_gaussblur_dynamo.py`. This file does
NOT reproduce the SEPARATE, deeper `caching_precompile` guard-state pickling crash found
alongside this fix (`TypeError: cannot pickle '_thread._local' object`, at ANY
`@torch._dynamo.disable()` boundary reached under `mode="reduce-overhead"` + real CUDA
Inductor) — that needs a real CUDA box and is a torch-internal defect this ask did not fix
(filed separately); this file only pins the ONE graph break that IS this ask's own fix,
which is CPU-reproducible and does not depend on that other defect.
"""
import pytest

torch = pytest.importorskip("torch")
dynamo = pytest.importorskip("torch._dynamo")

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_runtime.codegen import _stage_codegen_param
from TEX_Wrangle.tex_runtime.stdlib_core import _host_scalar, _HOST_SCALAR_ATTR


def _preamble_like(raw):
    """The EXACT shape `_get_param_local`'s emitted preamble now runs, for one `$param`
    read: `{local} = _bind[name]` (here: the argument itself) then
    `{local} = _THS({local}, _torch.as_tensor({local}))` — reproduced as a plain Python
    function so `torch.compile`/`torch._dynamo.explain` can trace it directly, the same
    "probe the mechanism in isolation" technique `test_compiletry50_gaussblur_dynamo.py`
    already uses for `_get_gauss_kernels`."""
    local = raw
    local = _stage_codegen_param(local, torch.as_tensor(local))
    return local


class TestTRK236PreambleTagging:
    def test_scalar_param_traces_with_zero_breaks_and_reads_back_without_item(self):
        """RED before this ask (measured manually against the pre-fix preamble shape:
        `torch.as_tensor(raw)` alone, no `_THS` — a genuine `$param` float minted this way
        carries no tag, so `_host_scalar` must fall through to `.item()`, itself a graph
        break under `torch.compile`); GREEN at head."""
        dynamo.reset()
        r = dynamo.explain(_preamble_like)(2.0)
        assert r.graph_break_count == 0, r.break_reasons
        out = _preamble_like(2.0)
        assert getattr(out, _HOST_SCALAR_ATTR, None) == 2.0
        assert _host_scalar(out) == 2.0

        def _reads_without_item(raw):
            staged = _preamble_like(raw)
            v = _host_scalar(staged)
            # A trailing TENSOR op, so a real break here (unlike the naive
            # "nothing follows" shape the sibling sanity test below had to correct
            # for) would force a genuine second graph/resume frame — this 0 is
            # meaningful, not an artefact of there being nothing left to break.
            return torch.full((2, 2), max(v, 0.0))

        dynamo.reset()
        r2 = dynamo.explain(_reads_without_item)(2.0)
        assert r2.graph_break_count == 0, r2.break_reasons

    def test_preamble_pre_fix_shape_does_break(self):
        """Sanity: the OLD preamble shape (bare `as_tensor`, no tagging) genuinely breaks
        on `_host_scalar` — proves the fix closes a REAL gap, not an already-empty one.
        A trailing op after the read (mirroring `fn_gauss_blur`'s own further statements,
        e.g. `max(sigma_val, 0.0)`) is needed for `dynamo.explain` to count this as a break
        rather than a same-as-return boundary event — `_host_scalar` at the tail of a
        function with nothing after it needs no RESUME continuation, so explain's own
        counter reads 0 there even though the identical `.item()` warning still fires
        (the SAME quirk `test_compiletry50_gaussblur_dynamo.py` already documents for a
        `torch._dynamo.disable()` boundary)."""
        def _old_preamble_like(raw):
            local = raw
            local = torch.as_tensor(local)
            v = _host_scalar(local)
            # A further TENSOR op is what forces a genuine second graph/resume frame for
            # the continuation (plain-Python code after `.item()` needs no new graph at
            # all, which is why an earlier draft of this test saw graph_break_count == 0
            # even with the real `.item()` warning firing).
            return torch.full((2, 2), max(v, 0.0))

        dynamo.reset()
        r = dynamo.explain(_old_preamble_like)(2.0)
        assert r.graph_break_count >= 1, "expected the pre-fix shape to still break"


class TestStageCodegenParamValueShapes:
    """Unit coverage for `_stage_codegen_param`'s four value shapes — no Dynamo needed."""

    def test_genuine_scalar_gets_tagged_with_the_rounded_value(self):
        # 0.1 is not exactly representable in fp32, so a tag of the raw Python float would
        # differ from the rounded value the tensor really carries.
        rounded = torch.scalar_tensor(0.1, dtype=torch.float32).item()
        assert rounded != 0.1
        minted = torch.as_tensor(0.1)
        out = _stage_codegen_param(0.1, minted)
        assert out is minted
        assert getattr(out, "_tex_host_scalar", None) == rounded

    def test_bool_scalar_gets_tagged(self):
        minted = torch.as_tensor(True)
        out = _stage_codegen_param(True, minted)
        assert getattr(out, "_tex_host_scalar", None) == 1.0

    def test_already_a_tensor_is_returned_unchanged_and_untouched(self):
        raw = torch.tensor(3.0)
        minted = torch.as_tensor(raw)  # as_tensor on a tensor: same object
        out = _stage_codegen_param(raw, minted)
        assert out is minted
        assert getattr(out, "_tex_host_scalar", None) is None

    def test_vec_param_list_is_returned_unchanged_and_untagged(self):
        raw = [0.1, 0.2, 0.3]
        minted = torch.as_tensor(raw)
        out = _stage_codegen_param(raw, minted)
        assert out is minted
        assert getattr(out, "_tex_host_scalar", None) is None


def test_gauss_blur_via_param_binding_is_bit_identical_interp_vs_codegen():
    """Invariant #2: the fix only changes WHICH tensor `_get_param_local` hands the
    generated function (an already-tagged one instead of a fresh untagged one) — the
    VALUE never moves. `run_both` compares interp vs. codegen on a real `$sigma`-bound
    gauss_blur program, exactly the shape TRK-236 fixes."""
    code = "f$sigma = 2.0;\n@OUT = gauss_blur(@A, $sigma);\n"
    for sigma_val in (0.0, 0.29, 0.3, 1.0, 2.0, 5.5):
        bindings = {"A": make_img(B=1, H=9, W=11, C=3), "sigma": sigma_val}
        interp_out, cg_out = run_both(code, dict(bindings))
        assert cg_out is not None, "codegen declined this program"
        assert torch.equal(interp_out["OUT"], cg_out["OUT"]), sigma_val
