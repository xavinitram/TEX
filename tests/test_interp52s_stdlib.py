"""Stdlib helpers: strings, host arguments, mip sampling, budgets, bilateral and noise
arguments, the registry views, and the noise tier cache lock."""
import threading

import pytest
import torch

from helpers import *
from TEX_Wrangle.tex_runtime.interpreter import Interpreter, InterpreterError
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib as S
from TEX_Wrangle.tex_runtime import stdlib_core as C
from TEX_Wrangle.tex_runtime import stdlib_registry as R
from TEX_Wrangle.tex_runtime import noise as N
from TEX_Wrangle.tex_cache import parse_and_split
from TEX_Wrangle.tex_compiler.type_checker import TypeChecker


def _img(H=8, W=8, C_=3, seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, H, W, C_, generator=g)


def _cook(src, bindings):
    bt = {n: _infer_binding_type(v) for n, v in bindings.items()}
    prog = parse_and_split(src, bt)
    ch = TypeChecker(binding_types=bt, source=src)
    tm = ch.check(prog)
    return Interpreter().execute(prog, dict(bindings), tm, device="cpu", output_names=["OUT"])["OUT"]


# -- strings -----------------------------------------------------------------------------

def test_str_prints_the_number_that_was_typed():
    assert S.fn_str(torch.tensor(0.1)) == "0.1"
    assert S.fn_str(torch.tensor(3.14159274)) == "3.14159"
    assert S.fn_str(torch.tensor(-2.5)) == "-2.5"
    assert S.fn_str(torch.tensor(42.0)) == "42"
    assert S.fn_str(torch.tensor(0.1)) == S.fn_format("{}", torch.tensor(0.1))


def test_substr_clamps_a_negative_start_and_length():
    assert S.fn_substr("hello", torch.tensor(-1.0), torch.tensor(3.0)) == "hel"
    assert S.fn_substr("hello", torch.tensor(1.0), torch.tensor(-2.0)) == ""
    assert S.fn_substr("hello", torch.tensor(-3.0)) == "hello"
    assert S.fn_substr("hello", torch.tensor(1.0), torch.tensor(3.0)) == "ell"


@pytest.mark.parametrize("name,want", [
    ("CON", "_CON"), ("nul.png", "_nul.png"), ("Com1", "_Com1"), ("LPT9.txt", "_LPT9.txt"),
    ("console", "console"), ("a:b", "ab"), ("", "unnamed"), ("aux ", "_aux"),
])
def test_sanitize_filename_avoids_device_names(name, want):
    assert S.fn_sanitize_filename(name) == want


def test_a_per_pixel_string_index_is_a_diagnostic():
    img = _img(H=4, W=5)
    with pytest.raises(InterpreterError) as ei:
        _cook('string s[] = {"a", "bb", "ccc"}; float n = float(len(s[int(@A.r * 3.0)])); '
              '@OUT = vec4(n, 0.0, 0.0, 1.0);', {"A": img})
    assert ei.value.code == "E6005"
    with pytest.raises(InterpreterError, match="varies per pixel"):
        _cook('string t = "hello"; string c = char_at(t, int(@A.r * 5.0)); '
              '@OUT = vec4(float(len(c)), 0.0, 0.0, 1.0);', {"A": img})


def test_scalar_from_an_empty_tensor_is_a_clear_error():
    with pytest.raises(ValueError, match="empty"):
        C._scalar_from_tensor(torch.zeros(0), "str")


# -- host readings -----------------------------------------------------------------------

def test_scale_pixel_arg_tag_equals_what_item_reads():
    import random
    rnd = random.Random(7)
    for _ in range(2000):
        v = rnd.uniform(0.0, 10.0)
        sc = rnd.uniform(0.01, 3.0)                  # a double, not fp32-exact
        t = C._tag_host_scalar(torch.scalar_tensor(v), v, torch.float32)
        out = C._scale_pixel_arg(t, sc)
        assert C._host_scalar(out) == out.item()


# -- mip sampling ------------------------------------------------------------------------

def test_sample_mip_with_a_nan_lod_reads_level_zero():
    img = _img(H=8, W=8, C_=3)
    u = torch.rand(1, 8, 8)
    v = torch.rand(1, 8, 8)
    a = S.fn_sample_mip(img, u, v, torch.tensor(float("nan")))
    b = S.fn_sample_mip(img, u, v, torch.tensor(0.0))
    assert torch.equal(a, b)
    lod = torch.full((1, 8, 8), float("nan"))
    assert torch.equal(S.fn_sample_mip(img, u, v, lod), b)


def test_mip_cache_bytes_count_shared_storage_once():
    img = _img(H=16, W=16, C_=4)
    C._mip_cache_budget.clear(C._mip_cache)
    C._get_mip_pyramid(img)
    pyr = list(C._mip_cache.values())[0][2]
    # level 0 is a permute view of the cached source image: one allocation
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in pyr}
    storages[img.untyped_storage().data_ptr()] = img.untyped_storage().nbytes()
    assert C._mip_cache_budget.total() == sum(storages.values())
    C._mip_cache_budget.clear(C._mip_cache)


# -- bilateral and patch_dist arguments --------------------------------------------------

def test_bilateral_nan_range_sigma_is_a_diagnostic():
    img = _img()
    with pytest.raises(InterpreterError) as ei:
        S.fn_bilateral_filter(img, torch.tensor(1.5), torch.tensor(float("nan")))
    assert ei.value.code == "E6052"
    out = S.fn_bilateral_filter(img, torch.tensor(1.5), torch.tensor(float("inf")))
    assert torch.isfinite(out).all()      # an infinite range sigma is a plain blur


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_patch_dist_non_finite_argument_is_a_diagnostic(bad):
    img = _img()
    for args in [(bad, 0.0, 1.0), (0.0, bad, 1.0), (0.0, 0.0, bad)]:
        with pytest.raises(InterpreterError) as ei:
            S.fn_patch_dist(img, *[torch.tensor(a) for a in args])
        assert ei.value.code == "E6052"
    grid = torch.full((1, 8, 8), bad)
    with pytest.raises(InterpreterError):
        S.fn_patch_dist(img, grid, torch.tensor(0.0), torch.tensor(1.0))


def _separable_reference(bchw, ss, sr, radius):
    def one_pass(x, dim, ref):
        n = x.shape[dim]
        d = torch.arange(-radius, radius + 1, dtype=torch.float32)
        w1 = torch.exp(-0.5 * (d * d) / max(ss * ss, 1e-10))
        inv = -0.5 / max(sr * sr, 1e-10)
        base = torch.arange(n)
        src, r32 = x.float(), ref.float()
        acc = torch.zeros_like(src)
        shp = list(x.shape)
        shp[1] = 1
        wsum = torch.zeros(shp)
        for i, off in enumerate(range(-radius, radius + 1)):
            idx = (base + off).clamp(0, n - 1)
            tap = torch.index_select(src, dim, idx)
            tref = torch.index_select(r32, dim, idx)          # the gather the fast path skips
            diff = r32 - tref
            w = w1[i] * torch.exp((diff * diff).sum(dim=1, keepdim=True) * inv)
            acc = acc + tap * w
            wsum = wsum + w
        return (acc / wsum.clamp(min=1e-10)).to(x.dtype)
    row = one_pass(bchw, 3, bchw)
    return one_pass(row, 2, bchw)


def test_separable_bilateral_row_pass_is_unchanged():
    bchw = _img(H=12, W=12, C_=3).permute(0, 3, 1, 2)
    out = S._bilateral_separable_bchw(bchw, 20.0, 0.3, 50)
    assert torch.equal(out, _separable_reference(bchw, 20.0, 0.3, 50))


# -- noise arguments ---------------------------------------------------------------------

@pytest.mark.parametrize("fn", ["fbm", "ridged", "billow", "turbulence", "alligator"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_noise_octave_argument_must_be_finite(fn, bad):
    x = torch.rand(1, 4, 4)
    with pytest.raises(InterpreterError) as ei:
        getattr(S, f"fn_{fn}")(x, x, torch.tensor(bad))
    assert ei.value.code == "E6052"


def test_flow_time_must_be_finite():
    x = torch.rand(1, 4, 4)
    with pytest.raises(InterpreterError) as ei:
        S.fn_flow(x, x, torch.tensor(float("inf")))
    assert ei.value.code == "E6052"
    assert torch.isfinite(S.fn_flow(x, x, torch.tensor(0.5))).all()


def test_tiered_cache_forget_settled_takes_the_lock():
    c = N._TieredCache("probe")
    c._settled.add(("k", (1,), (1,), torch.float32))
    done = threading.Event()

    def worker():
        c.forget_settled("k")
        done.set()

    with c._lock:
        t = threading.Thread(target=worker)
        t.start()
        assert not done.wait(0.2)         # blocked while another thread holds the lock
    assert done.wait(2.0)
    t.join()
    assert not c._settled


# -- registry views ----------------------------------------------------------------------

def test_registry_by_name_views_are_never_seen_empty_while_rebuilt():
    view = R.non_spatial_args_by_name()
    assert view
    seen_empty = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            if not view:
                seen_empty.append(True)
                return

    t = threading.Thread(target=reader)
    t.start()
    try:
        for _ in range(300):
            R._NON_SPATIAL_CACHE_READY = False
            assert R.non_spatial_args_by_name() is view
    finally:
        stop.set()
        t.join()
    assert not seen_empty
    assert R.arg_footprint_by_name()["convolve"][1] == "image"
    assert R.pixel_args_by_name()["gauss_blur"] == (1,)
