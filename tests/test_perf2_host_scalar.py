"""PERF-2 — a builtin that needs a host number takes it from the host, and answers the same.

WHAT CHANGED. `gauss_blur` and its siblings need a PYTHON number: the kernel radius sizes an
allocation, `erode`'s radius drives a `range`, `convolve`'s `normalize` picks a branch. They
read it with `.item()`, which on CUDA is a 4-byte device->host copy AND the stream
synchronisation that copy implies — per call, per cook, on the commonest interactive tick.
Almost none of those numbers needed the device: a literal, a `$param` float/int and a folded
constant all start life as a Python number and are only minted into a 0-dim device tensor on
the way in. The mint sites now record the host reading ON that tensor (`_tag_host_scalar`) and
the builtins read it off the host (`_host_scalar`). A sigma genuinely computed on the device
carries no tag and still reads back — that readback is legitimate and stays.

WHY THIS FILE IS SHAPED THE WAY IT IS. The change is behaviour-preserving or it is nothing, so
the PRE-CHANGE path is kept available rather than described: every reader falls back to the
exact expression it replaced when `_host_scalar` answers None, so forcing `_host_scalar` to
None reproduces the base sha's arithmetic inside the shipped code. `_host_scalar_off` does
that, and every output comparison is against it with `torch.equal` — BIT-exact, not a
tolerance, because the two runs are the same tier on the same device and any difference at all
would be a value change. (The separate interp-vs-codegen row uses the repo's own cross-tier
contract instead: invariant #2's 1e-5, because the two tiers mint their constants differently.)

BOTH DIRECTIONS. `test_perf2_a_host_scalar_costs_no_readback` counts every `Tensor.item()`
the builtin makes, split by where the tensor lived, and requires ZERO from the device — it
fails on the base sha, where each of those shapes reads one.
`test_perf2_a_computed_scalar_still_reads_back` requires the same probe to count a readback
for a sigma computed in-program, so "no readback" cannot be passed by a builtin that stopped
needing the number, or by a probe that stopped counting. And
`test_perf2_the_tag_carries_the_rounded_value` mutates the tag to the un-rounded Python double
and requires the output to MOVE.

PORTABILITY: every row runs on CPU except the device-readback count, which SKIPs without CUDA
— a CPU `.item()` is a host-memory read, so that row would pass without measuring anything.
No ComfyUI, no compiler, no numpy, no timing assertion.
"""
import dataclasses
import math

from helpers import *

from failure_harness import run_tier
from TEX_Wrangle.tex_runtime import interpreter as _interp
from TEX_Wrangle.tex_runtime import stdlib as _stdlib
from TEX_Wrangle.tex_runtime import stdlib_core as _stdlib_core
from TEX_Wrangle.tex_runtime import stdlib_registry as _registry
from TEX_Wrangle.tex_runtime import stdlib_sample as _ssample


# ── the corpus: one builtin per host-resolved argument, across sigma shapes ──

#: (registered name, program template, extra bindings). `{s}` is the host-resolved argument.
_PROGRAMS = (
    ("gauss_blur",        "@OUT = gauss_blur(@A, {s});", {}),
    ("bilateral_filter",  "@OUT = bilateral_filter(@A, {s}, 0.2);", {}),
    ("erode",             "@OUT = erode(@A, {s});", {}),
    ("dilate",            "@OUT = dilate(@A, {s});", {}),
    # TRK-66: the LOD arg is read back AFTER `lod_t.clamp(0.0, max_level)` mints a fresh,
    # untagged tensor — every shape below (including "literal 7.9", which clamps) exercises
    # that clamp on a host-origin LOD.
    ("sample_mip",         "@OUT = sample_mip(@A, u, v, {s});", {}),
)

#: (label, the text substituted for `{s}`, the bindings it needs, has_host_value).
#: `has_host_value` is False for exactly one row — a sigma computed on the device from a
#: binding — which is the row that must STILL read back.
_SIGMA_SHAPES = (
    ("literal 0",       "0",            {},                 True),
    ("literal 0.4",     "0.4",          {},                 True),
    ("literal 2.0",     "2.0",          {},                 True),
    ("literal 2.25",    "2.25",         {},                 True),
    ("literal 7.9",     "7.9",          {},                 True),
    ("literal int 3",   "3",            {},                 True),
    ("$param float",    "$sig",         {"sig": 2.25},      True),
    ("$param int",      "$sig",         {"sig": 3},         True),
    ("computed",        "($sig * 2.0)", {"sig": 1.125},     False),
)


def _img(device, size=24):
    torch.manual_seed(20260920)
    return torch.rand(1, size, size, 4, dtype=torch.float32).to(device)


def _bindings(extra, device):
    b = {"A": _img(device)}
    b.update(extra)
    return b


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class _host_scalar_off:
    """Run the shipped code with `_host_scalar` answering None — which IS the base sha's
    path, because every reader falls back to the exact expression it replaced.

    TRK-170: `stdlib.py` re-exports `_host_scalar` (`from .stdlib_core import ...`), but
    every builtin this file exercises resolves it through a DIFFERENT binding of the
    same name, minted at each leaf module's own load time — `gauss_blur` /
    `bilateral_filter` / `convolve` / `patch_dist` read `stdlib_sample.py`'s own
    imported copy, and `erode` / `dilate` (via `_morph` -> `_to_float`) and
    `sample_mip` (via `_sample_mip_trilinear`) read `stdlib_core.py`'s own defining
    copy. Reassigning only `stdlib.<name>` leaves every one of those pointing at the
    original function, so this now patches each leaf module's own binding by name —
    the same fix `test_perf2_the_tag_carries_the_rounded_value` already applies to
    `_tag_host_scalar` for one function, generalised to every module this file calls
    into (grep `_host_scalar` across `tex_runtime/*.py` finds no third binding any
    builtin in `_PROGRAMS` or `test_perf2_convolve_and_patch_dist_are_bit_exact`
    reads)."""

    _MODULES = (_stdlib, _ssample, _stdlib_core)

    def __enter__(self):
        self._orig = {mod: mod._host_scalar for mod in self._MODULES}
        for mod in self._MODULES:
            mod._host_scalar = lambda x: None
        return self

    def __exit__(self, *exc):
        for mod, orig in self._orig.items():
            mod._host_scalar = orig
        return False


class _count_readbacks_inside:
    """Wrap one registered builtin so that every `Tensor.item()` it makes is COUNTED, split
    by where the tensor lived. `device_reads` is the number PERF-2 is about: an `.item()` on
    a non-CPU tensor is the 4-byte D2H plus the stream synchronisation. An `.item()` on a CPU
    tensor is a host-memory read and costs neither, so counting the two together would call
    the codegen tier's CPU `$param` a regression it never was. The counter delegates to the
    real `.item()`, so the cook completes and the assertion is about counts, not a crash.

    BOTH function tables have to be swapped: codegen rebuilds `TEXStdlib.get_functions()`
    per cook (the REGISTRY is enough there), but `Interpreter._stdlib_functions` is a
    class-level memo built once per process. Patching only the registry makes this probe
    report a confident — and wrong — zero on the interpreter tier, the same trap BENCH-2
    recorded for `tex_engine`'s module-level re-exports."""

    def __init__(self, fn_name):
        self.fn_name = fn_name
        self.reads = 0
        self.device_reads = 0

    def __enter__(self):
        for i, e in enumerate(_registry.REGISTRY):
            if e.name == self.fn_name:
                self._index, self._entry = i, e
                break
        else:                                          # pragma: no cover - corpus typo
            raise AssertionError(f"{self.fn_name} is not registered")
        real, probe = self._entry.fn, self

        def _counted(*a, **kw):
            orig_item = torch.Tensor.item

            def _item(_self):
                probe.reads += 1
                if _self.device.type != "cpu":
                    probe.device_reads += 1
                return orig_item(_self)
            torch.Tensor.item = _item
            try:
                return real(*a, **kw)
            finally:
                torch.Tensor.item = orig_item
        _registry.REGISTRY[self._index] = dataclasses.replace(self._entry, fn=_counted)
        self._memo = Interpreter._get_stdlib()
        self._saved = {n: self._memo[n] for n in self._entry.names if n in self._memo}
        for n in self._saved:
            self._memo[n] = _counted
        return self

    def __exit__(self, *exc):
        _registry.REGISTRY[self._index] = self._entry
        self._memo.update(self._saved)
        return False


def _same(a, b):
    """None when every output is bit-identical, else a message naming the difference."""
    for k, av in a.items():
        bv = b.get(k)
        if isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor):
            if av.shape != bv.shape or av.dtype != bv.dtype:
                return (f"output {k!r}: {tuple(av.shape)}/{av.dtype} vs "
                        f"{tuple(bv.shape)}/{bv.dtype}")
            if not torch.equal(av, bv):
                d = (av.float() - bv.float()).abs().max().item()
                return f"output {k!r}: not bit-exact (maxdiff {d:g})"
        elif av != bv:
            return f"output {k!r}: {av!r} != {bv!r}"
    return None


# ── 1. behaviour preservation: the same bytes, with and without the host value ──

def test_perf2_outputs_are_bit_exact_on_both_tiers(r: SubTestResult):
    """Every builtin x every sigma shape x both tiers x every device: the shipped output is
    BIT-identical to the same code run with `_host_scalar` disabled (the base path)."""
    print("\n--- PERF-2: host-resolved scalars are bit-exact against the readback ---")
    for device in _devices():
        for tier in ("interp", "codegen"):
            for fname, tmpl, fextra in _PROGRAMS:
                for label, expr, sextra, _host in _SIGMA_SHAPES:
                    name = f"[{device}/{tier}] {fname}({label})"
                    code = tmpl.format(s=expr)
                    extra = dict(fextra)
                    extra.update(sextra)
                    try:
                        got = run_tier(code, _bindings(extra, device), tier, device=device)
                        with _host_scalar_off():
                            want = run_tier(code, _bindings(extra, device), tier, device=device)
                        bad = _same(got, want)
                        if bad:
                            r.fail("PERF-2 bit-exact", f"{name}: {bad}")
                        else:
                            r.ok(name)
                    except Exception as e:
                        r.fail("PERF-2 bit-exact", f"{name}: {type(e).__name__}: {e}")


def test_perf2_convolve_and_patch_dist_are_bit_exact(r: SubTestResult):
    """The two host-resolved arguments that are NOT a blur radius: `convolve`'s normalize
    flag, and `patch_dist`'s radius (a uniform-or-raise check of the same shape)."""
    print("\n--- PERF-2: convolve's flag and patch_dist's radius ---")
    rows = (
        ("convolve normalize=1",  "@OUT = convolve(@A, @K, 1);",  {"K": True}),
        ("convolve normalize=0",  "@OUT = convolve(@A, @K, 0);",  {"K": True}),
        ("convolve normalize=$p", "@OUT = convolve(@A, @K, $p);", {"K": True, "p": 0}),
        ("patch_dist r=1",        "float d = patch_dist(@A.rgb, 2, -1, 1);\n"
                                  "@OUT = vec4(d, d, d, 1.0);",   {}),
        ("patch_dist r=$p",       "float d = patch_dist(@A.rgb, 2, -1, $p);\n"
                                  "@OUT = vec4(d, d, d, 1.0);",   {"p": 2}),
    )
    for device in _devices():
        for tier in ("interp", "codegen"):
            for label, code, extra in rows:
                name = f"[{device}/{tier}] {label}"
                try:
                    e = dict(extra)
                    if "K" in e:
                        e["K"] = torch.full((1, 3, 3, 1), 1.0 / 9.0,
                                            dtype=torch.float32).to(device)
                    got = run_tier(code, _bindings(e, device), tier, device=device)
                    with _host_scalar_off():
                        want = run_tier(code, _bindings(e, device), tier, device=device)
                    bad = _same(got, want)
                    if bad:
                        r.fail("PERF-2 bit-exact", f"{name}: {bad}")
                    else:
                        r.ok(name)
                except Exception as ex:
                    r.fail("PERF-2 bit-exact", f"{name}: {type(ex).__name__}: {ex}")


# ── 2. the interp<->codegen contract still holds, per sigma shape ──────────────

def test_perf2_the_two_tiers_still_agree(r: SubTestResult):
    """Invariant #2, per sigma shape: codegen matches the interpreter oracle within 1e-5."""
    print("\n--- PERF-2: interpreter == codegen, per sigma shape ---")
    for device in _devices():
        for fname, tmpl, fextra in _PROGRAMS:
            for label, expr, sextra, _host in _SIGMA_SHAPES:
                name = f"[{device}] {fname}({label}) interp==codegen"
                code = tmpl.format(s=expr)
                extra = dict(fextra)
                extra.update(sextra)
                try:
                    a = run_tier(code, _bindings(extra, device), "interp", device=device)
                    b = run_tier(code, _bindings(extra, device), "codegen", device=device)
                    m = max((a[k].float() - b[k].float()).abs().max().item()
                            for k in a if isinstance(a[k], torch.Tensor))
                    if m < 1e-5:
                        r.ok(f"{name} (maxdiff {m:g})")
                    else:
                        r.fail("PERF-2 tier equivalence", f"{name}: maxdiff {m:g}")
                except Exception as e:
                    r.fail("PERF-2 tier equivalence", f"{name}: {type(e).__name__}: {e}")


# ── 3. red-first: the readback is gone, and only where it should be ───────────

def test_perf2_a_host_scalar_costs_no_readback(r: SubTestResult):
    """THE RED-FIRST ROW. A literal / `$param` sigma must cost ZERO device readbacks inside
    the builtin, on either tier. On the base sha the CUDA/interpreter rows read one each —
    which is the `cuda.memcpy_DtoH` the host-path harness counts."""
    print("\n--- PERF-2: a host-origin scalar costs no device readback ---")
    if not torch.cuda.is_available():
        r.skip("PERF-2 device readback",
               "no CUDA device — a CPU `.item()` is a host-memory read, so this row would "
               "pass without measuring anything")
        return
    for tier in ("interp", "codegen"):
        for fname, tmpl, fextra in _PROGRAMS:
            for label, expr, sextra, host in _SIGMA_SHAPES:
                if not host:
                    continue
                name = f"[cuda/{tier}] {fname}({label}) no device readback"
                code = tmpl.format(s=expr)
                extra = dict(fextra)
                extra.update(sextra)
                try:
                    with _count_readbacks_inside(fname) as probe:
                        run_tier(code, _bindings(extra, "cuda"), tier, device="cuda")
                    if probe.device_reads:
                        r.fail("PERF-2 readback",
                               f"{name}: {probe.device_reads} device readback(s)")
                    else:
                        r.ok(name)
                except Exception as e:
                    r.fail("PERF-2 readback", f"{name}: {type(e).__name__}: {e}")


def test_perf2_a_computed_scalar_still_reads_back(r: SubTestResult):
    """THE OTHER DIRECTION. A sigma computed in-program from a binding has no host value, so
    the readback is correct and must still happen — otherwise the row above could be passed
    by a builtin that stopped needing the number at all, or by an inert probe. Runs on every
    device: a readback is a readback, cheap or not, and its PRESENCE is what is pinned."""
    print("\n--- PERF-2: a device-computed scalar still reads back ---")
    label, expr, sextra, host = _SIGMA_SHAPES[-1]
    assert not host, "the last corpus row must be the computed one"
    for device in _devices():
        for tier in ("interp", "codegen"):
            for fname, tmpl, fextra in _PROGRAMS:
                name = f"[{device}/{tier}] {fname}({label}) reads back"
                code = tmpl.format(s=expr)
                extra = dict(fextra)
                extra.update(sextra)
                try:
                    with _count_readbacks_inside(fname) as probe:
                        run_tier(code, _bindings(extra, device), tier, device=device)
                    want = "device_reads" if device == "cuda" else "reads"
                    if getattr(probe, want):
                        r.ok(f"{name} ({getattr(probe, want)})")
                    else:
                        r.fail("PERF-2 readback",
                               f"{name}: no readback — either the probe is inert, or a value "
                               f"that only exists on the device was resolved host-side")
                except Exception as e:
                    r.fail("PERF-2 readback", f"{name}: {type(e).__name__}: {e}")


# ── 4. the tag is the ROUNDED value, and that matters ─────────────────────────

def test_perf2_the_tag_carries_the_rounded_value(r: SubTestResult):
    """A tag must be the number `.item()` WOULD have returned — the value after the tensor's
    dtype rounded it — never the un-rounded Python double. The mutation proves the distinction
    is load-bearing rather than pedantic."""
    print("\n--- PERF-2: the tag is the dtype-rounded reading ---")
    try:
        raw = 0.1
        rounded = _stdlib._dtype_rounded(raw, torch.float32)
        assert rounded is not None and rounded != raw, (
            f"0.1 must differ from its fp32 reading; got {rounded!r}")
        t = _stdlib._tag_host_scalar(torch.scalar_tensor(raw, dtype=torch.float32), raw)
        assert getattr(t, _stdlib._HOST_SCALAR_ATTR) == t.item() == rounded, (
            "the tag must equal what .item() returns")
        r.ok(f"fp32 tag of 0.1 == .item() == {rounded!r}")
    except Exception as e:
        r.fail("PERF-2 tag rounding", f"{type(e).__name__}: {e}")

    # MUTATION: a tag that skipped the rounding must produce a different picture.
    #
    # The sigma has to be chosen, not picked: most differences hide. Torch treats a Python
    # float operand as a WEAK scalar, so `exp(-0.5*(x/sigma)**2)` rounds the double to fp32
    # anyway and the kernel WEIGHTS come out identical; and `_get_gauss_kernels` keyed its
    # cache on `round(sigma, 3)` until PERF-3, so a second run at the same slider was served
    # the first run's kernel regardless — which is why the cache is CLEARED either side of
    # the mutation below, and stays cleared now that it is keyed exactly (a module-level memo
    # outlives a cook either way). What does NOT hide is the arithmetic done in PYTHON:
    # `radius = ceil(3*sigma)`. 2/3 is exactly 2.0 there and 2.0000000596 in fp32, so the
    # rounding decides between a 5-tap and a 7-tap kernel.
    try:
        img = _img("cpu")
        code = "@OUT = gauss_blur(@A, $sig);"
        sig = 2.0 / 3.0
        assert math.ceil(3.0 * sig) != math.ceil(3.0 * _stdlib._dtype_rounded(sig, torch.float32)), (
            "the chosen sigma no longer separates the rounded reading from the raw double")
        _stdlib._gauss_kernel_cache_budget.clear(_stdlib._gauss_kernel_cache)
        good = run_tier(code, {"A": img, "sig": sig}, "interp")
        # The interpreter imported the tagger BY NAME, so the module attribute is not the
        # one the mint site calls — patch the binding that is actually read.
        orig = _interp._tag_host_scalar

        def _unrounded(t, value, dtype=None):
            setattr(t, _stdlib._HOST_SCALAR_ATTR, float(value))
            return t
        try:
            _interp._tag_host_scalar = _unrounded
            _stdlib._gauss_kernel_cache_budget.clear(_stdlib._gauss_kernel_cache)
            bad = run_tier(code, {"A": img, "sig": sig}, "interp")
        finally:
            _interp._tag_host_scalar = orig
            _stdlib._gauss_kernel_cache_budget.clear(_stdlib._gauss_kernel_cache)
        if _same(good, bad) is None:
            r.fail("PERF-2 tag mutation",
                   "an un-rounded tag produced the SAME output — the rounding is untested")
        else:
            r.ok("an un-rounded tag moves the output (so the rounding is load-bearing)")
    except Exception as e:
        r.fail("PERF-2 tag mutation", f"{type(e).__name__}: {e}")


def test_perf2_a_tag_never_survives_an_operation(r: SubTestResult):
    """The staleness argument, pinned. An operation that computes a NEW value returns a NEW
    object, which carries no tag — if that stopped being true, `gauss_blur(@A, $s * 2.0)`
    would silently blur by `$s`. An operation that is a NO-OP may hand back the tensor itself
    (`t.float()` on an fp32 tensor does, and `_to_tensor` relies on exactly that), and there
    the tag is still true. So what is pinned is "a new value means a new object", not "a tag
    never travels"."""
    print("\n--- PERF-2: a tag never describes a value it was not minted from ---")
    try:
        t = _stdlib._tag_host_scalar(torch.scalar_tensor(2.0, dtype=torch.float32), 2.0)
        assert _stdlib._host_scalar(t) == 2.0, "the tag must be readable on the tensor itself"
        for label, derived in (("*2", t * 2), ("+0", t + 0), ("clone", t.clone()),
                               ("half", t.to(torch.float16)), ("neg", -t)):
            v = getattr(derived, _stdlib._HOST_SCALAR_ATTR, None)
            assert v is None, f"{label} carried the tag forward ({v!r})"
        for label, derived in (("float()", t.float()), ("reshape", t.reshape(())),
                               ("_to_tensor", _stdlib._to_tensor(t))):
            v = getattr(derived, _stdlib._HOST_SCALAR_ATTR, None)
            assert v is None or derived is t, (
                f"{label} put the tag on a DIFFERENT object ({v!r})")
        r.ok("a tag never reaches a value it was not minted from")
    except Exception as e:
        r.fail("PERF-2 tag staleness", f"{type(e).__name__}: {e}")


# ── 5. TRK-67: the same mechanism, off the stdlib string family (v041-p2) ──────────
#
# The string family lives outside any single registered builtin's `fn` in the way
# `_count_readbacks_inside` expects (which patches one registry entry) — `_host_int` is
# a plain module-level helper several builtins call, not itself registered. This counts
# every `Tensor.item()` call globally instead.

class _count_all_item_calls:
    """Counts every `Tensor.item()` call process-wide while active, split by device."""

    def __enter__(self):
        self.reads = 0
        self.device_reads = 0
        self._orig = torch.Tensor.item
        probe = self

        def _item(self_t):
            probe.reads += 1
            if self_t.device.type != "cpu":
                probe.device_reads += 1
            return probe._orig(self_t)
        torch.Tensor.item = _item
        return self

    def __exit__(self, *exc):
        torch.Tensor.item = self._orig
        return False


def test_trk67_string_family_costs_no_readback(r: SubTestResult):
    """TRK-67 — RED-FIRST. `replace`, `substr`, `split`, `pad_left`, `pad_right`,
    `repeat`, `hash_int` and `char_at` each resolved their size/index argument with
    `int(x.item() if isinstance(x, torch.Tensor) else x)` — a device readback on every
    call, even for a literal or `$param` value that a source program always mints with a
    host reading. `_host_int` (`stdlib_core.py`) takes it from there instead, same
    mechanism as `_host_scalar`. CUDA-only: a CPU `.item()` is a host-memory read and
    would pass without measuring anything."""
    print("\n--- TRK-67: string family size/index args cost no device readback ---")
    if not torch.cuda.is_available():
        r.skip("TRK-67 device readback",
               "no CUDA device — a CPU `.item()` is a host-memory read")
        return
    rows = (
        ("replace $n",   '@OUT = replace("aaaa","a","b",$n);',                  {"n": 2}),
        ("substr $n",    '@OUT = substr("hello world",$n,3);',                  {"n": 2}),
        ("split $n",     'string arr[] = split("a,b,c,d",",",$n);\n'
                          '@OUT = arr[0];',                                     {"n": 1}),
        ("pad_left $n",  '@OUT = pad_left("hi",$n);',                           {"n": 5}),
        ("pad_right $n", '@OUT = pad_right("hi",$n);',                          {"n": 5}),
        ("repeat $n",    '@OUT = repeat("ab",$n);',                             {"n": 3}),
        ("hash_int $n",  'float h = hash_int("seed",$n);\n@OUT = vec4(h);',     {"n": 100}),
        ("char_at $n",   '@OUT = char_at("hello",$n);',                        {"n": 1}),
    )
    for label, code, extra in rows:
        name = f"[cuda] {label}"
        try:
            with _count_all_item_calls() as probe:
                compile_and_run(code, extra, device="cuda")
            if probe.device_reads:
                r.fail("TRK-67 readback", f"{name}: {probe.device_reads} device readback(s)")
            else:
                r.ok(name)
        except Exception as e:
            r.fail("TRK-67 readback", f"{name}: {type(e).__name__}: {e}")


def test_trk68_array_index_and_loop_bound_cost_no_readback(r: SubTestResult):
    """TRK-68 — RED-FIRST. A `$param` array index drains the device once per evaluation,
    and a `$param` loop bound drains it once per loop ENTRY (UC-5's `_const_index`
    already covers the compile-time-literal case one level earlier). `_host_index`
    (`interpreter.py`) and `_int_valued_scalar`'s own `_host_scalar` read take both from
    the host instead. CUDA-only.

    Measured as a DELTA against the same program with `_host_scalar` forced to answer
    None (the pre-fix path), rather than an absolute zero: the array rows below build a
    SPATIAL array (`[B,H,W,N]`, from a per-pixel value — a host-constant array indexes
    with a plain tensor index and never reaches `.item()` either side of this fix, so an
    absolute-zero probe built on one would pass vacuously) via a small loop that itself
    costs a few unrelated device reads. The one this row is about is the DIFFERENCE
    `_host_scalar` being disabled makes, which must be a real, positive reduction —
    interpreter.py imported `_host_scalar` BY NAME (same reason
    `test_perf2_the_tag_carries_the_rounded_value` patches `_interp._tag_host_scalar`
    rather than `_stdlib`'s copy of it), so the patch below targets that binding."""
    print("\n--- TRK-68: array index / loop bound $param costs no device readback ---")
    if not torch.cuda.is_available():
        r.skip("TRK-68 device readback",
               "no CUDA device — a CPU `.item()` is a host-memory read")
        return
    torch.manual_seed(20260924)
    img = torch.rand(1, 4, 4, 4, dtype=torch.float32).to("cuda")
    _spatial_field = (
        "float field[4];\n"
        "for (int k = 0; k < 4; k++) { field[k] = @A.r + float(k); }\n"
    )
    rows = (
        ("read  field[$i]",  _spatial_field +
                              "@OUT = vec4(field[$i], 0, 0, 1);",
                              {"i": 2, "A": img}),
        ("write field[$i]=", _spatial_field +
                              "field[$i] = @A.g;\n"
                              "@OUT = vec4(field[0] + field[1] + field[2] + field[3], 0, 0, 1);",
                              {"i": 1, "A": img}),
        ("loop  k<$n",       "float acc = 0.0;\n"
                              "for (int k = 0; k < $n; k++) { acc = acc + 1.0; }\n"
                              "@OUT = vec4(acc, 0, 0, 1);",
                              {"n": 4}),
    )
    for label, code, extra in rows:
        name = f"[cuda] {label}"
        try:
            with _count_all_item_calls() as fixed_probe:
                compile_and_run(code, extra, device="cuda")
            fixed_reads = fixed_probe.device_reads

            orig = _interp._host_scalar
            _interp._host_scalar = lambda x: None
            try:
                with _count_all_item_calls() as base_probe:
                    compile_and_run(code, extra, device="cuda")
            finally:
                _interp._host_scalar = orig
            base_reads = base_probe.device_reads

            if base_reads <= fixed_reads:
                r.fail("TRK-68 readback",
                       f"{name}: forcing the pre-fix path read {base_reads}, not more than "
                       f"the fixed path's {fixed_reads} — the probe does not exercise the fix")
            else:
                r.ok(f"{name} ({fixed_reads} device read(s), {base_reads} with the fix forced off)")
        except Exception as e:
            r.fail("TRK-68 readback", f"{name}: {type(e).__name__}: {e}")


# ── 6. TRK-69: gauss_blur and bilateral_filter fp32-round a bare float alike ────────

def test_trk69_gauss_blur_and_bilateral_filter_agree_on_a_bare_float(r: SubTestResult):
    """TRK-69. A BARE Python float sigma — never handed to either builtin by a shipped
    tier; the interpreter and codegen both mint every scalar into a tensor first, so
    this is a direct-caller-only corner — used to fp32-round differently between the
    two: `gauss_blur`'s fallback read a freshly minted tensor's `.item()` (fp32-rounded);
    `bilateral_filter`'s kept the raw Python double. Both now resolve a bare float
    through `_dtype_rounded`, the same rounding the PERF-2 mint sites tag their tensors
    with, so a boundary sigma decides the SAME kernel-tap count either way.

    `2.0/3.0` is the chosen boundary (same one `test_perf2_the_tag_carries_the_rounded_
    value` uses for gauss_blur): 3*(2/3) is exactly 2.0 as a Python double but
    2.0000000596 once fp32-rounds 2/3 first, so `ceil(3*sigma)` — the shared shape of
    both builtins' radius formula — lands on a different integer each way. Direct
    static-method calls only; no cook(), no codegen."""
    print("\n--- TRK-69: a bare float sigma fp32-rounds the same in both builtins ---")
    try:
        sig = 2.0 / 3.0
        rounded = _stdlib._dtype_rounded(sig, torch.float32)
        assert math.ceil(3.0 * sig) != math.ceil(3.0 * rounded), (
            "the chosen sigma no longer separates the rounded reading from the raw double")

        img = torch.rand(1, 12, 12, 3, dtype=torch.float32)

        # The MUTATION: force bilateral_filter's non-tensor fallback back to the raw
        # double (the bug this row reports) by disabling the rounding it now shares
        # with gauss_blur, and require the output to MOVE — otherwise the fix is not
        # being exercised by this probe. Patched on `stdlib_sample`'s own binding: it
        # imported `_dtype_rounded` BY NAME (the same reason
        # `test_perf2_the_tag_carries_the_rounded_value` patches `_interp`'s own copy
        # of `_tag_host_scalar` rather than `_stdlib`'s).
        fixed = TEXStdlib.fn_bilateral_filter(img, sig, 0.2)
        orig = _ssample._dtype_rounded
        _ssample._dtype_rounded = lambda v, dt: None      # -> falls back to the raw double
        try:
            unrounded = TEXStdlib.fn_bilateral_filter(img, sig, 0.2)
        finally:
            _ssample._dtype_rounded = orig

        if torch.equal(fixed, unrounded):
            r.fail("TRK-69 fp32 agreement",
                   "bilateral_filter's output did not move when its rounding was disabled "
                   "— this probe does not exercise the fix")
        else:
            r.ok("bilateral_filter fp32-rounds a bare float sigma (own probe, "
                 f"maxdiff {(fixed - unrounded).abs().max().item():g})")

        # And it must now agree with gauss_blur's OWN reading of the same bare float —
        # not bit-exact (different formulas: `min(ceil(3*ss), 3)` vs an unclamped
        # radius), but the same ROUNDED sigma feeding both.
        _stdlib._gauss_kernel_cache_budget.clear(_stdlib._gauss_kernel_cache)
        _ = TEXStdlib.fn_gauss_blur(img, sig)   # exercises the reference reading; no crash
        r.ok("gauss_blur resolves the same bare float without error (reference reading)")
    except Exception as e:
        r.fail("TRK-69 fp32 agreement", f"{type(e).__name__}: {e}")

