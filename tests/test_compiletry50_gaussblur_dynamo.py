"""
COMPILETRY-50 (v0.50, "Build compile D1" item 2) — `gauss_blur`'s RLock-guarded kernel
cache off the traced path.

BEFORE: `stdlib_core._get_gauss_kernels`'s cache (`_gauss_kernel_cache_budget.touch`/`put`)
opens a `threading.RLock` as a context manager, which Dynamo cannot trace ("Unsupported
context manager") -- measured as the SOLE structural graph-break source for `gauss_blur`:
`torch._dynamo.explain(fn_gauss_blur)` (CPU) reports Graph Count 4 / Graph Break Count 3 /
Op Count 11. This file does not restate that reading as a live assertion against unpatched
code, since there is no "unpatched" branch to import here; the RED state was confirmed
manually against base sha b79ce82 before this file's assertions were written.

AFTER: `@torch._dynamo.disable()` on `_get_gauss_kernels` (COMPILETRY-50) takes the WHOLE
function -- lock, cache dict, kernel build -- off the traced path. `_gauss_blur_bchw` (the
function that both fetches the kernel AND applies it via the two pad+conv2d passes) now
traces as ONE graph with ZERO breaks. Nothing about `_get_gauss_kernels` itself changes, so
results stay bit-identical (checked directly here, eager vs. a real `torch.compile`d call,
across a sweep of sigma values spanning both branches of the `sigma < 0.3` no-op gate).

PORTABILITY. CPU only. `torch._dynamo.explain` and `torch.compile(backend="aot_eager")`
both trace with Dynamo but need NO C++ compiler and NO CUDA/Triton (unlike the real
`inductor` backend, which this box cannot use at all -- `cl` is not on PATH here, confirmed
manually) -- `aot_eager` just replays the traced FX graph in eager ops, so this file proves
graph-break count and end-to-end numeric parity without depending on a toolchain the box
does not have. Guarded with `pytest.importorskip` per the standing portability rule: a box
with no `torch._dynamo` at all (never expected on this project's own torch 2.x floor) skips
rather than errors.
"""
import pytest

torch = pytest.importorskip("torch")
dynamo = pytest.importorskip("torch._dynamo")

from helpers import *  # noqa: F401,F403
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib
from TEX_Wrangle.tex_runtime.stdlib_core import _gauss_blur_bchw, _get_bchw


def test_gauss_blur_bchw_traces_with_zero_breaks():
    """RED at base sha b79ce82 (measured manually: Graph Break Count 3, the RLock);
    GREEN at head. `_gauss_blur_bchw` is the function that both resolves the kernel
    (through the disabled, lock-guarded cache) AND applies it -- the two pad+conv2d
    passes BUILTINS-50a-design.md's own §1 already found Dynamo captures cleanly on
    their own ("Graph 3" in that document's own breakdown)."""
    with cold_engine_state():
        img = make_img(B=1, H=8, W=8, C=3)
        bchw = _get_bchw(img)
        dynamo.reset()
        r = dynamo.explain(_gauss_blur_bchw)(bchw, 2.0)
        assert r.graph_break_count == 0, r.break_reasons
        assert r.graph_count == 1
        # pad, conv2d, pad, conv2d -- the separable blur's own two passes, nothing else.
        assert r.op_count == 4


def test_fn_gauss_blur_no_longer_breaks_on_the_rlock():
    """The outer builtin (`TEXStdlib.fn_gauss_blur`, what a real cook calls) must not
    report "Unsupported context manager" any more -- this ask's OWN, narrowly-scoped
    fix (only `_get_gauss_kernels`'s lock). `fn_gauss_blur`'s own sigma-resolution
    prefix (`_host_scalar`) has an unrelated, pre-existing break-counting quirk when
    probed directly with a raw, untagged tensor; asserting the ABSENCE of the RLock
    reason specifically -- not a bare zero -- is what actually pins this ask's fix
    without also pinning that unrelated, out-of-scope prefix."""
    with cold_engine_state():
        img = make_img(B=1, H=8, W=8, C=3)
        dynamo.reset()
        r = dynamo.explain(TEXStdlib.fn_gauss_blur)(img, torch.tensor(2.0))
        reasons = [str(getattr(b, "reason", b)) for b in r.break_reasons]
        assert not any("context manager" in reason.lower() for reason in reasons), reasons


@pytest.mark.parametrize("sigma_val", [0.0, 0.29, 0.3, 1.0, 2.0, 5.5])
def test_gauss_blur_stays_bit_identical_under_torch_compile(sigma_val):
    """Nothing about `_get_gauss_kernels`'s OWN behaviour changed -- `@torch._dynamo.disable()`
    only changes what Dynamo traces, never what the function computes -- so a real
    `torch.compile`d call (backend `aot_eager`: traces for real, needs no C++ compiler)
    must return the EXACT same tensor as the plain eager call, for both the no-op branch
    (`sigma < 0.3`) and the real-blur branch, and across a batch/channel-count sigma
    boundary case."""
    with cold_engine_state():
        img = make_img(B=2, H=17, W=23, C=4)
        sigma = torch.tensor(sigma_val)
        out_eager = TEXStdlib.fn_gauss_blur(img, sigma)
        dynamo.reset()
        compiled_fn = torch.compile(TEXStdlib.fn_gauss_blur, backend="aot_eager",
                                    fullgraph=False)
        out_compiled = compiled_fn(img, sigma)
        assert torch.equal(out_eager, out_compiled)
        dynamo.reset()
