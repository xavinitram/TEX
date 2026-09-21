"""v0.31 — NOISE-TIER: the cold frame must render what every later frame renders.

    The FIRST cook of a process and the Nth cook of the identical program with
    identical inputs must be bit-identical ON THE SAME TIER. The one tier change the
    engine makes on purpose — jit.trace -> Inductor on a key's 4th call — is held to a
    recorded envelope instead (see "The promotion envelope" below).

`_TieredCache` (tex_runtime/noise.py) runs eager → jit.trace → torch.compile. Its cold
frame returned the EAGER result and cached a trace for every call after it, so call #1 of
a process ran a *different tier* than calls #2+. On CUDA those tiers are not bit-identical:
the TorchScript fuser reassociates the pointwise chain, and `simplex` at u*800 measured
5.3e-4 apart from eager over 56% of pixels — `x0 = x - X0` cancels ~800 against ~800 where
the fp32 quantum is ~6e-5, and the (0.5-d²)⁴ falloff amplifies what survives. Measured on
an RTX 2080 SUPER / torch 2.5.0+cu118, `simplex` and `fbm` each rendered 2 distinct images
across 6 cooks; `examples/film_grain.tex` was non-reproducible on first cook. CPU never
showed it (its two tiers agree exactly), which inverts ARCHITECTURE.md's marketed
"bitwise run-to-run deterministic on CUDA … the CPU is the honest caveat".

The defect predates v0.30.0 — it is not a release regression, and the fix does not move
the steady state: cook #1 now returns what cook #2+ always returned.

The promotion envelope. `try_upgrade` swaps the jit.trace tier for an Inductor-compiled one
on a key's fourth CALL, and its docstring left that swap knowingly open because no box it
was fixed on could compile. The first box that can (RTX 5070 Ti Laptop / sm_120, torch
2.12.0+cu130 with Triton; MSVC for CPU) answered the question `cold_frame_parity` was written
to ask: on CUDA the swap is NOT bit-safe — simplex moved by up to 6.56e-07 across it, and
three rows here went red although each tier, taken alone, did exactly what they demand. It
is not a hash-cell flip either: every tier renders one image per signature, a region cooked
on either tier crops bit-exactly out of that tier's whole frame, and the difference is the
float tail Inductor fuses differently. So equality across the swap is no longer demanded.
Each tier's own claims stay EXACT, and the swap is held to a per-builtin band measured
against the fp32 interpreter control — the default fp32 cook on the incumbent jit.trace
tier, which is what cooks #1-3 render. Measured on CUDA, promoted vs that control, maximum
over 64² to 2048² frames on grid and random coordinates:

    builtin              measured maximum                                 band
    simplex              7.75e-07, alike at coordinate scales 8 to 8000   2e-6
    fbm (4/6/8 octaves)  1.79e-07                                         5e-7
    worley_f1 / f2       1-2 fp32 ulps OF THE COORDINATE: 3.6e-07 at 8,   4 ulps of the
                         3.9e-06 at 64, 6.07e-05 at 512, 2.44e-04 at      coordinate
                         pixel-space coordinates up to 4080

Three facts a host needs, kept beside the numbers. (1) The swap is not gated by
`compile_mode`: it engages on the DEFAULT cook path wherever Triton (CUDA) or a host C++
compiler (CPU) is present, and `tier_trace.noise_compiles()` is where it is reported. (2) It
counts CALLS, not cooks: a program calling one tiered builtin four times (layered noise,
`alligator`'s octaves) crosses it inside cook #1, so that cold frame differs from its warm
frames by the same envelope. (3) Worley's band grows with its coordinates, and a
thresholding program can turn any envelope into a flipped pixel — so a region recook
composited over a frame cooked on the other tier is bounded by these numbers only where
they apply. On CPU the same sweep (64² to 720p) measured the swap bit-exact for simplex and
worley and 1.19e-07 for fbm; `cold_frame_parity` keeps CPU simplex bit-exact, as v0.31
shipped it. The per-builtin row stays CUDA-only: whether CPU promotes at all depends on the
host's C++ toolchain (on Windows, even on the cache path's length), and a band recorded
against one compiler is not a fact about another.

What each row pins:
  * cold_frame_parity      — NEVER-SEVER. A FRESH PROCESS cooks simplex 5x; every cook on
                             one tier must be bit-equal, and a second process must render
                             the same cold frame. Fresh-process is the honest form: the
                             effect is per-process first-use, and nothing in
                             `cold_engine_state` resets the module-level tier caches. 5 cooks
                             also cross `_COMPILE_AFTER_CALLS`, a boundary that could NOT be
                             measured where this was fixed (no Triton for CUDA, and the CPU
                             Inductor build fails with CppCompileError). Where Inductor
                             engages the row checks it: one promotion at most, jit.trace ->
                             compiled, on cook #4, reported as one noise-compile event, and
                             within the simplex envelope of cook #1 (on CPU that envelope is
                             0.0, so CPU stays bit-exact across it). MUTATION: the same
                             child with the promotion installing "incumbent + delta" (no
                             Inductor needed, so every CUDA box runs it) must judge 2x the
                             band red and a quarter of it green.
  * resolution_dance       — NEVER-SEVER. The SECOND half of the same defect, and the one
                             a fixed-shape test cannot see. A traced module is not one
                             numeric object: torch's profiling executor runs
                             `_jit_get_num_profiled_runs()` (=1) UNOPTIMIZED passes per new
                             (shape, dtype) and only then installs the fused plan. Measured:
                             a module traced at 24x32 and called at 48x64 returned the EAGER
                             value on that shape's first call (4d40b133) and the fused value
                             forever after (952dbe87). So pinning the cold frame alone fixes
                             the process-first cook and silently reopens the identical bug
                             at every new resolution — 512→1024→512 is ordinary ComfyUI use.
                             This row cooks 24x32 → 48x64 → back, twice, with the tier HELD
                             at jit.trace, then dances again on the promoted tier, and
                             demands one image per resolution on EACH. Left unheld, the
                             promotion lands mid-dance and reads as a reopened window.
  * cold_equals_warm       — the same claim as a direct unit, for all three _TieredCache
                             users (simplex, fbm, worley) on every available device.
  * stride_signature       — NEVER-SEVER, split the same way: each strided layout settles to
                             one image on jit.trace, and again on the promoted tier.
  * cold_path_shape        — CANARY. The cold path must ROUTE THROUGH the shared callable
                             rather than compute its own result. The previous fix here
                             tried to hold parity by making the eager body textually mirror
                             the traced body; identical source is exactly what the fuser
                             reassociates, so re-introducing a separate eager result is a
                             regression even when it looks bit-for-bit right.
  * promotion_envelope     — ENVELOPE, per builtin: simplex, fbm and worley at two coordinate
                             scales, promoted vs the fp32 interpreter control. Plus the
                             mixed-tier pair a region recook creates — a window cooked on
                             one tier against the whole frame cooked on the other, both
                             ways — bounded by the same band, while a window on its OWN tier
                             stays bit-exact against the crop. MUTATION: a promotion 2x past
                             the simplex band must turn the envelope and both pair rows red.

Where no band applies, a host declines the mix instead, and for that it needs to know which tier
cooked a frame: `tex_engine.cook(..., want_noise_tiers=True)` fills `CookResult.noise_tiers`
(the contract is in `tex_runtime/tier_trace.py`). Three rows pin that record:
  * tier_record_across_the_promotion — PROVENANCE, per device, in a fresh process: each cook's
                             record names the tier that served it, read off the cache with
                             `_tier_of`, and flips where the promotion lands; a program calling
                             simplex four times straddles the promotion inside one cook and
                             reads None, with a reason naming both tiers; four tiered keys file
                             four labels; and the same cooks unasked render the same digests on
                             the same tiers with the field left None.
  * tier_record_forced     — the same record with each tier FORCED on CPU, so every box runs it:
                             promoted from cook #4, a straddling cook None, a promotion that
                             raises "promotion_failed" in the cook it failed in (a failure is not
                             a mix), a trace that never settles "eager", and a strategy other than
                             "default" None.
  * tier_record_default_path — INVARIANT 7: unasked or False the field is None, asked with no noise
                             it is {}, the pixels are identical either way, a cook that raised
                             while asked leaves nothing armed for the next one, and `tex_node`
                             never asks.
"""
import inspect
import json
import subprocess

from helpers import *

from TEX_Wrangle.tex_runtime.noise import _COMPILE_AFTER_CALLS as _COMPILE_AFTER

_CUDA = torch.cuda.is_available()

# Large coords on purpose: the divergence lives in the `x - X0` cancellation, so a
# small-coordinate probe would pass on a build that still has the bug.
_SIMPLEX_PROG = "@OUT = vec4(vec3(simplex(u*800.0, v*800.0)), 1.0);"
_N_COOKS = 5           # > _COMPILE_AFTER_CALLS (3), so the row spans the tier-3 promotion

# The promotion envelope (module docstring), per builtin: promoted tier vs the fp32 interpreter
# control. Each band is ~2.5-4x the maximum its sweep measured — the headroom v0.30's ROI
# envelope pinned (2.98e-08 -> 1e-7). CPU's simplex swap measured bit-exact, so its band is
# 0.0: v0.31's exact claim, unmoved. A band is a recorded fact about a torch/driver/Triton
# build; a build that blows one is a decision to re-measure and re-band, never a tolerance.
_ENVELOPE_SIMPLEX = {"cpu": 0.0, "cuda": 2e-6}
_ENVELOPE_FBM = 5e-7


def _envelope_worley(scale):
    """Worley's swap moves the COORDINATE's last ulp, not the output's: measured 1-2 fp32 ulps
    of the coordinate magnitude at every scale tried. Coordinates in [scale/2, scale) have an
    ulp of scale * 2**-24, and the band is four of them."""
    return 4.0 * scale * 2.0 ** -24


def _tier_of(cache, key):
    """The callable a `_TieredCache` holds for `key`, in words — read off the cache itself,
    never inferred from the output, so a same-tier claim is only ever made about one tier."""
    held = cache.cache.get(key)
    if held is None:
        return "cold"
    if held is False:
        return "eager"
    return "trace" if isinstance(held, torch.jit.ScriptFunction) else "promoted"


# Shared by every child below. `_tier_of` is copied in from its source so the children and the
# in-process rows cannot drift apart on what a tier is.
_CHILD_HEAD = r'''
import os, sys
sys.path.insert(0, sys.argv[1])                       # .../custom_nodes
os.environ["TEX_CACHE_DIR"] = sys.argv[2]             # before any TEX import
import hashlib, json, struct, torch
from TEX_Wrangle import tex_engine, tex_roi
from TEX_Wrangle.tex_runtime import noise, tier_trace
from TEX_Wrangle.tex_runtime.noise import _COMPILE_AFTER_CALLS

def digest(t):
    # Whole-tensor and numpy-free: .tolist() gives Python doubles, struct.pack("f")
    # narrows back to the exact fp32 bits. tensor_fingerprint would NOT do here — it
    # samples 256 strided elements, so a localized divergence can slip through.
    v = t.detach().float().cpu().flatten().tolist()
    return hashlib.sha256(struct.pack(str(len(v)) + "f", *v)).hexdigest()[:16]

def maxdiff(a, b):
    return float((a.float() - b.float()).abs().max())

def device_key(dev):
    # The key the simplex cache files a device under: the coordinate tensor's own device.
    return torch.device("cuda", torch.cuda.current_device()) if dev == "cuda" else torch.device(dev)

def promote_to_mutant(delta):
    # THE MUTATION, kept here rather than reachable from the tree: the simplex promotion
    # installs "the incumbent tier + delta" instead of an Inductor kernel, through the cache's
    # own try_upgrade. Forcing the backend flag makes it box-independent — it needs no Triton,
    # because what it exercises is the envelope judgement downstream, not Inductor.
    for dev_type in ("cpu", "cuda"):
        noise._inductor_available[dev_type] = True
    def _factory(device, _delta=delta):
        incumbent = noise._simplex_cache.cache[device]
        return lambda x, y: incumbent(x, y) + _delta
    noise._compile_simplex = _factory

''' + inspect.getsource(_tier_of)


_PARITY_CHILD = _CHILD_HEAD + r'''
dev, n, prog = sys.argv[3], int(sys.argv[4]), sys.argv[5]
mutants = [float(d) for d in sys.argv[6:]]
key, cache = device_key(dev), noise._simplex_cache
torch.manual_seed(5)
img = torch.rand(1, 24, 32, 4, device=dev)

def run(label):
    first = None
    for i in range(1, n + 1):
        out = tex_engine.cook(prog, {"A": img}, device_mode=dev, precision="fp32").outputs["OUT"]
        first = out if first is None else first
        events = sum(1 for e in tier_trace.noise_compiles() if e["noise"] == "simplex")
        print("COOK", label, i, _tier_of(cache, key), events, digest(out),
              repr(maxdiff(out, first)))

run("real")
for delta in mutants:
    cache.cache.clear(); cache._settled.clear()
    cache._compile_attempted.clear(); cache._call_count.clear()
    promote_to_mutant(delta)
    run("mutant:" + repr(delta))
'''


def _run_child(script, args, stdin=None):
    """(stdout, error) from a fresh interpreter — the defects here are per-process first-use."""
    custom_nodes = str(Path(__file__).resolve().parents[2])
    try:
        proc = subprocess.run([sys.executable, "-c", script, custom_nodes] + [str(a) for a in args],
                              input=stdin, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None, "the child never finished"
    if proc.returncode != 0:
        return None, f"child exited {proc.returncode}: {(proc.stderr or '')[-400:]}"
    return proc.stdout, None


def _cook_rows_in_fresh_process(dev, cache_dir, n=_N_COOKS, prog=_SIMPLEX_PROG, mutants=()):
    """Return ({sequence: [(cook, tier, compile events, digest, maxdiff vs cook #1)]}, error).

    Sequence "real" is the program as any host would cook it; "mutant:<delta>" sequences re-run
    it in the same process after clearing the simplex cache, with the mutation installed."""
    out, err = _run_child(_PARITY_CHILD, [cache_dir, dev, n, prog, *mutants])
    if err:
        return None, err
    seqs = {}
    for line in out.splitlines():
        if line.startswith("COOK "):
            _, label, i, tier, events, dg, md = line.split()
            seqs.setdefault(label, []).append((int(i), tier, int(events), dg, float(md)))
    if len(seqs.get("real", ())) != n:
        return None, f"child emitted {len(seqs.get('real', ()))} cooks, expected {n}"
    return seqs, None


def _split_at_promotion(rows):
    """Judge one fresh-process cook sequence: EXACT on each tier, at most one legal promotion.

    Returns (problems, envelope, epochs). `envelope` is the promoted cooks' maxdiff against
    cook #1 — the fp32 interpreter control on the incumbent tier — or None when nothing was
    promoted, in which case every cook shares one tier and the exact rule covers all of them:
    this row verbatim as v0.31 shipped it."""
    epochs = []
    for cook, tier, events, dg, md in rows:
        if not epochs or epochs[-1][0] != tier:
            epochs.append((tier, []))
        epochs[-1][1].append((cook, events, dg, md))
    problems = []
    for tier, cooks in epochs:
        images = sorted({dg for _, _, dg, _ in cooks})
        if len(images) != 1:
            cold = (" — cook #1 differs from cook #2 on the SAME cached tier: the cold frame ran "
                    "a different tier (the v0.31 defect)"
                    if cooks[0][0] == 1 and len(cooks) > 1 and cooks[0][2] != cooks[1][2] else "")
            problems.append(f"{len(images)} distinct images on ONE tier ({tier}, cooks "
                            f"{[c for c, _, _, _ in cooks]}){cold}: {images}")
    if len(epochs) == 1:
        return problems, None, epochs
    if len(epochs) > 2 or (epochs[0][0], epochs[1][0]) != ("trace", "promoted"):
        problems.append("unexpected tier sequence " + " -> ".join(t for t, _ in epochs))
        return problems, None, epochs
    (_, before), (_, after) = epochs
    if after[0][0] != _COMPILE_AFTER + 1:
        problems.append(f"the promotion landed on cook #{after[0][0]}, not "
                        f"#{_COMPILE_AFTER + 1} (_COMPILE_AFTER_CALLS={_COMPILE_AFTER})")
    first_events = before[0][1]
    if (any(e != first_events for _, e, _, _ in before)
            or any(e != first_events + 1 for _, e, _, _ in after)):
        problems.append("the promotion was not reported as exactly one noise-compile event "
                        "(tier_trace.noise_compiles)")
    return problems, max(md for _, _, _, md in after), epochs


def test_v031_noise_cold_frame_parity(r: SubTestResult):
    """NEVER-SEVER: N cooks of one program in one fresh process, bit-identical on each tier.

    The one tier change the engine makes on purpose (jit.trace -> Inductor at cook #4) is not
    a same-tier claim. It is held to the recorded simplex envelope of cook #1, the fp32
    interpreter control, and that judgement carries its own mutation."""
    print("\n--- v0.31 NOISE-TIER: the first cook must equal every later cook on its tier ---")
    devices = ["cpu"] + (["cuda"] if _CUDA else [])
    if not _CUDA:
        r.skip("cold-frame parity on CUDA",
               "no CUDA on this box — CPU's tiers agree bitwise, so only the CUDA row has teeth")

    for dev in devices:
        band = _ENVELOPE_SIMPLEX[dev]
        mutants = (2.0 * band, 0.25 * band) if band else ()      # (must go red, must stay green)
        with cold_engine_state() as cold:
            seqs, err = _cook_rows_in_fresh_process(dev, cold.dir, mutants=mutants)
        if err:
            r.fail(f"cold-frame parity ({dev})", err)
            continue
        problems, envelope, epochs = _split_at_promotion(seqs["real"])
        digests = [dg for _, _, _, dg, _ in seqs["real"]]
        if problems:
            r.fail(f"cold-frame parity ({dev})", "; ".join(problems) + f". digests={digests}")
            continue
        r.ok(f"simplex renders one image per tier across {_N_COOKS} cooks on {dev} (" +
             ", ".join(f"{tier} x{len(cooks)} = {cooks[0][2]}" for tier, cooks in epochs) + ")")

        # …and a SECOND independent process must land on the same image: the settled
        # fused value has to be a property of the program, not of one session.
        with cold_engine_state() as cold2:
            again, err2 = _cook_rows_in_fresh_process(dev, cold2.dir, n=2)
        if err2:
            r.fail(f"cross-process parity ({dev})", err2)
        elif again["real"][0][3] == digests[0]:
            r.ok(f"a second process renders the identical image on {dev} ({digests[0]})")
        else:
            r.fail(f"cross-process parity ({dev})",
                   f"process A rendered {digests[0]}, process B rendered {again['real'][0][3]}")

        if envelope is None:
            if band:
                r.skip(f"promotion envelope ({dev})",
                       f"no promotion in {_N_COOKS} cooks — the compile tier does not engage here")
        elif envelope <= band:
            r.ok(f"the promotion at cook #{_COMPILE_AFTER + 1} moved simplex by {envelope:.3e} "
                 f"on {dev}, within the recorded {band:.0e} of the fp32 interpreter control")
        else:
            r.fail(f"promotion envelope ({dev})",
                   f"cooks #{_COMPILE_AFTER + 1}+ differ from cook #1 by {envelope:.3e}, past "
                   f"the recorded {band:.0e}" +
                   (" — the compile tier moved: re-measure and re-band deliberately" if band
                    else " — CPU's simplex swap measured bit-exact and is held there"))

        # MUTATION, both directions: the same judgement over a promotion KNOWN to sit 2x past
        # the band must say red, and one a quarter inside it must say green. Without this the
        # envelope row above could pass by never comparing anything.
        for delta, must_be_red in zip(mutants, (True, False)):
            rows = seqs.get("mutant:" + repr(float(delta)), [])
            m_problems, m_env, _ = _split_at_promotion(rows) if rows else (["no rows"], None, [])
            name = f"MUTATION promotion + {delta:.1e} ({dev})"
            if m_problems or m_env is None:
                r.fail(name, "the mutant did not promote cleanly, so the row measures nothing: "
                             + "; ".join(m_problems or ["no promotion"]))
            elif (m_env > band) != must_be_red:
                r.fail(name, f"maxdiff {m_env:.3e} against the band {band:.0e} was judged "
                             f"{'green' if must_be_red else 'red'} — the envelope row is decorative")
            else:
                r.ok(f"{name}: maxdiff {m_env:.3e} judged {'red' if must_be_red else 'green'} "
                     f"against the band {band:.0e}")


_DANCE_CHILD = _CHILD_HEAD + r'''
dev, prog = sys.argv[3], sys.argv[4]
key, cache = device_key(dev), noise._simplex_cache

def cook(h, w):
    torch.manual_seed(5)
    img = torch.rand(1, h, w, 4, device=dev)
    return tex_engine.cook(prog, {"A": img}, device_mode=dev, precision="fp32").outputs["OUT"]

def dance(leg, control):
    # Two passes over the same two resolutions. Pass 2 proves a settled signature STAYS
    # settled after the other resolution has run through the same module.
    for _pass in range(2):
        for (h, w) in [(24, 32), (48, 64), (24, 32)]:
            out = cook(h, w)
            ref = control.setdefault((h, w), out)
            print("DIGEST", leg, h, w, _tier_of(cache, key), digest(out), repr(maxdiff(out, ref)))

# Leg 1: the tier HELD at jit.trace by the cache's own "compile already attempted" mark — the
# dance exactly as it runs wherever Inductor never engages.
cache._compile_attempted.add(key)
control = {}
dance("trace", control)
# Leg 2: release the mark, let try_upgrade promote on its own at a third size, and dance again.
# The maxdiff column is against leg 1's image for the same size: the fp32 interpreter control.
cache._compile_attempted.discard(key)
cache._call_count.pop(key, None)
if noise._can_inductor_compile(key):
    for _ in range(_COMPILE_AFTER_CALLS):
        cook(16, 16)
if _tier_of(cache, key) == "promoted":
    dance("promoted", control)
'''


def test_v031_noise_resolution_dance(r: SubTestResult):
    """NEVER-SEVER: each resolution must render exactly one image on each tier, across a dance.

    A fixed-shape parity test cannot see this — the profiling window is per (shape, dtype),
    so it reopens at every new size even when the cold frame is correctly pinned. The dance
    runs with the tier held at jit.trace, then again on the promoted tier: left unheld, the
    promotion lands on cook #4 mid-dance, and a swapped callable is float reassociation (the
    envelope's business), not a reopened window.
    """
    print("\n--- v0.31 NOISE-TIER: 24x32 -> 48x64 -> back renders one image per size ---")
    if not _CUDA:
        r.skip("resolution dance", "no CUDA — the profiling-window gap is a CUDA fuser effect")
        return

    with cold_engine_state() as cold:
        out, err = _run_child(_DANCE_CHILD, [cold.dir, "cuda", _SIMPLEX_PROG])
    if err:
        r.fail("resolution dance", err)
        return

    legs = {}
    for line in out.splitlines():
        if line.startswith("DIGEST "):
            _, leg, h, w, tier, d, md = line.split()
            legs.setdefault(leg, {}).setdefault(f"{h}x{w}", []).append((tier, d, float(md)))
    if "trace" not in legs:
        r.fail("resolution dance", f"child emitted no digests: {out[-300:]}")
        return

    for leg, by_res in legs.items():
        tiers = sorted({tier for rows in by_res.values() for tier, _, _ in rows})
        bad = {res: [d for _, d, _ in rows] for res, rows in by_res.items()
               if len({d for _, d, _ in rows}) != 1}
        if tiers != [leg]:
            r.fail(f"resolution dance ({leg})", f"the leg did not hold its tier: saw {tiers}")
        elif bad:
            r.fail(f"resolution dance ({leg})",
                   "a resolution rendered more than one image on one tier — the profiling "
                   "window reopened per shape: " + "; ".join(f"{res}={ds}" for res, ds in bad.items()))
        else:
            r.ok(f"{leg}: one image per resolution across the dance "
                 f"({', '.join(f'{res}={rows[0][1]}' for res, rows in sorted(by_res.items()))})")

    if "promoted" not in legs:
        r.skip("resolution dance on the promoted tier", "the compile tier does not engage here")
        return
    band = _ENVELOPE_SIMPLEX["cuda"]
    moved = max(md for rows in legs["promoted"].values() for _, _, md in rows)
    if moved <= band:
        r.ok(f"promoted vs jit.trace, per resolution: maxdiff {moved:.3e}, within the recorded {band:.0e}")
    else:
        r.fail("resolution dance envelope",
               f"the promoted dance moved {moved:.3e} from the jit.trace dance, past the "
               f"recorded {band:.0e}")


def test_v031_noise_cold_equals_warm(r: SubTestResult):
    """The same claim as a direct unit, across all three _TieredCache users.

    In-process, so it clears the module-level tier caches by hand — `cold_engine_state`
    does not reach them (they are plain dicts on `noise._simplex_cache` &c., not engine
    state). Clearing is safe mid-suite: they are pure caches and rebuild on next touch.
    """
    print("\n--- v0.31 NOISE-TIER: cold frame == warm frame (simplex / fbm / worley) ---")
    from TEX_Wrangle.tex_runtime import noise

    devices = ["cpu"] + (["cuda"] if _CUDA else [])
    cases = [
        ("simplex", noise._simplex_cache, lambda x, y: noise._simplex2d(x, y)),
        ("fbm",     noise._fbm_cache,     lambda x, y: noise._fbm2d(x, y, 4)),
        ("worley",  noise._worley_cache,  lambda x, y: noise._worley2d(x, y)),
    ]
    bad = []
    for dev in devices:
        H, W = 24, 32
        yy, xx = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                                torch.arange(W, device=dev, dtype=torch.float32), indexing='ij')
        # simplex/fbm want the cancelling large coords; worley's grid is unit-cell.
        for name, cache, fn in cases:
            scale = 8.0 if name == "worley" else 800.0
            x, y = (xx / W) * scale, (yy / H) * scale
            cache.cache.clear()
            cache._compile_attempted.clear()
            cache._call_count.clear()
            cold = fn(x, y)
            warm = fn(x, y)
            if not torch.equal(cold.float(), warm.float()):
                bad.append(f"{name}@{dev} maxdiff="
                           f"{float((cold.float() - warm.float()).abs().max()):.2e}")
    if bad:
        r.fail("cold==warm", "the cold frame ran a different tier: " + "; ".join(bad))
    else:
        r.ok(f"simplex/fbm/worley cold frame == warm frame on {devices}")


def test_v031_noise_stride_signature(r: SubTestResult):
    """The settle signature must key on STRIDES, not just shape+dtype — on each tier.

    torch's profiling guards are stride-aware, so two tensors a shape-only signature
    calls identical can be different guard classes. Measured before this was keyed:
    after a contiguous 64x64 had settled, a TRANSPOSED 64x64 view (same shape, same
    dtype, stride (1,64)) returned ff73847f then b45a711d — the hole reopened on a
    tensor the cache believed was already settled.

    Each layout is called four times on ONE tier: first with the tier held at jit.trace, then
    on the promoted tier. Unheld, the fourth contiguous call IS the promotion, and a swapped
    callable is float reassociation (held to the envelope here), not a reopened window.
    """
    print("\n--- v0.31 NOISE-TIER: the settle signature is stride-aware ---")
    if not _CUDA:
        r.skip("stride signature", "no CUDA — the profiling-window gap is a CUDA fuser effect")
        return
    from TEX_Wrangle.tex_runtime import noise

    c = noise._simplex_cache
    key = torch.device("cuda", torch.cuda.current_device())
    c.cache.clear(); c._settled.clear()
    c._compile_attempted.clear(); c._call_count.clear()

    n = 64
    base = (torch.arange(n * n, device="cuda", dtype=torch.float32).reshape(n, n)
            / (n * n)) * 800.0
    layouts = (("contiguous", base.contiguous()),
               ("transposed", base.t()),                  # same shape+dtype, stride (1,64)
               ("expanded", base[0:1, :].expand(n, n)))   # stride (0,1)

    def leg(tier):
        bad, settled = [], {}
        for label, t in layouts:
            outs = [noise._simplex2d(t, t) for _ in range(4)]
            if _tier_of(c, key) != tier:
                bad.append(f"{label}: the leg left its tier ({tier} -> {_tier_of(c, key)})")
            # lnt2-ok: the arm above is the guard — this elif is only reached once the cache
            # has been read and found still on `tier`, so all four `outs` came off ONE
            # callable. That is what this row asserts: the profiling window closes WITHIN a
            # tier. Cross-tier agreement is never asserted here, only banded (see _envelope_*).
            elif not all(torch.equal(outs[0], o) for o in outs[1:]):
                bad.append(f"{label} stride={t.stride()} "
                           f"maxdiff={float((outs[0] - outs[-1]).abs().max()):.2e}")
            settled[label] = outs[0]
        if bad:
            r.fail(f"stride signature ({tier})",
                   "a strided layout reopened the profiling gap: " + "; ".join(bad))
        else:
            r.ok(f"{tier}: contiguous / transposed / expanded layouts each settle independently")
        return settled

    held = True
    c._compile_attempted.add(key)          # hold at jit.trace: the cache's own mark
    try:
        traced = leg("trace")
        c._compile_attempted.discard(key)
        c._call_count.pop(key, None)
        held = False
        if noise._can_inductor_compile(key):
            probe = torch.rand(8, 8, device="cuda")    # an unrelated signature takes the swap
            for _ in range(_COMPILE_AFTER):
                noise._simplex2d(probe, probe)
        if _tier_of(c, key) != "promoted":
            r.skip("stride signature (promoted)", "the compile tier does not engage here")
            return
        promoted = leg("promoted")
        band = _ENVELOPE_SIMPLEX["cuda"]
        moved = {label: float((promoted[label] - traced[label]).abs().max()) for label, _ in layouts}
        if max(moved.values()) <= band:
            r.ok("promoted vs jit.trace, per layout: " +
                 ", ".join(f"{label} {md:.3e}" for label, md in moved.items()) +
                 f", within the recorded {band:.0e}")
        else:
            r.fail("stride signature envelope",
                   ", ".join(f"{label} {md:.3e}" for label, md in moved.items()) +
                   f" — past the recorded {band:.0e}")
    finally:
        if held:                           # never leak the hold into a later row
            c._compile_attempted.discard(key)


def test_v031_noise_cold_path_shape(r: SubTestResult):
    """CANARY: the cold path must ROUTE through the shared callable, not recompute.

    Textual mirroring of the eager body against the traced body is what this replaced,
    and it cannot hold — the fuser reassociates the very source being mirrored. So a
    re-introduced standalone eager result is a regression by shape, not by value.
    """
    print("\n--- v0.31 NOISE-TIER: every call routes through _TieredCache.call ---")
    from TEX_Wrangle.tex_runtime import noise

    missing = [name for name, fn in (("_simplex2d", noise._simplex2d),
                                     ("_fbm2d", noise._fbm2d),
                                     ("_worley2d", noise._worley2d))
               if "_cache.call(" not in inspect.getsource(fn)]
    if missing:
        r.fail("cold-path shape",
               f"{', '.join(missing)}: no longer routed through _TieredCache.call — the "
               f"cold frame and each new (shape, dtype) can diverge from the settled tier")
    else:
        r.ok("_simplex2d / _fbm2d / _worley2d all route every call through _TieredCache.call")

    # A failed trace must fall back to eager, not propagate the False sentinel store()
    # writes — a raising noise fn would take down every cook.
    def _boom():
        raise RuntimeError("trace unavailable")
    probe = noise._TieredCache("probe-trace-failure")
    got = probe.call("k", (torch.ones(3), torch.ones(3)), device=torch.device("cpu"),
                     trace_fn=_boom, compile_fn=_boom, eager_fn=lambda a, b: a + b)
    if probe.cache.get("k") is False and torch.equal(got, torch.full((3,), 2.0)):
        r.ok("a failed trace falls back to the eager body instead of raising")
    else:
        r.fail("trace-failure fallback",
               f"expected the False sentinel and an eager result, got "
               f"cache={probe.cache.get('k')!r} result={got!r}")

    # A signature that never settles must demote to eager — never spin, and never serve a
    # value that depends on the call index.
    calls = {"n": 0}
    def _never_settles(a):
        calls["n"] += 1
        return a * float(calls["n"])
    probe2 = noise._TieredCache("probe-no-settle")
    probe2.cache["k2"] = _never_settles
    out = probe2._settle("k2", _never_settles, lambda a: a * -1.0, (torch.ones(3),))
    if probe2.cache["k2"] is False and torch.equal(out, torch.full((3,), -1.0)):
        r.ok(f"a signature that will not settle in {noise._SETTLE_MAX_RUNS} runs demotes "
             f"the key to eager")
    else:
        r.fail("settle non-convergence",
               f"expected a permanent eager demotion, got "
               f"cache={probe2.cache['k2']!r} result={out!r}")

    # _bitwise_same must call NaN equal to itself (so a NaN-producing program can settle)
    # and +0.0 unequal to -0.0 (reproducibility is a claim about bits). It must also
    # survive a 0-DIM tensor: `Tensor.view(dtype)` rejects those outright, and scalar
    # coordinates are ordinary TEX — `simplex(2.0, 3.0)` produces exactly that shape.
    nan = torch.tensor([float("nan")])
    checks = [(noise._bitwise_same(nan, nan.clone()), True, "NaN == NaN"),
              (noise._bitwise_same(torch.tensor([0.0]), torch.tensor([-0.0])), False,
               "+0.0 != -0.0"),
              (noise._bitwise_same(torch.tensor(1.5), torch.tensor(1.5)), True, "0-dim =="),
              (noise._bitwise_same(torch.tensor(1.5), torch.tensor(2.5)), False, "0-dim !="),
              (noise._bitwise_same(torch.zeros(0), torch.zeros(0)), True, "empty ==")]
    wrong = [lbl for got_, want, lbl in checks if got_ is not want]
    r.ok("_bitwise_same: NaN settles, signed zero does not, 0-dim and empty are safe") \
        if not wrong else \
        r.fail("_bitwise_same", "wrong verdict for " + ", ".join(wrong))

    # End-to-end: a scalar-coordinate noise call must still cook. This is the shape that
    # broke seven example programs when _bitwise_same viewed a 0-dim tensor as bytes.
    from TEX_Wrangle import tex_engine
    devices = ["cpu"] + (["cuda"] if _CUDA else [])
    broke = []
    for dev in devices:
        img = torch.rand(1, 16, 16, 4, device=dev)
        for fn_call in ("simplex(2.0, 3.0)", "fbm(2.0, 3.0, 4)", "worley_f1(2.0, 3.0)"):
            try:
                tex_engine.cook(f"float n = {fn_call};\n@OUT = vec4(vec3(n), 1.0);",
                                {"A": img}, device_mode=dev)
            except Exception as e:
                broke.append(f"{fn_call}@{dev}: {type(e).__name__} {str(e)[:80]}")
    if broke:
        r.fail("scalar coords", "a 0-dim noise call failed to cook: " + "; ".join(broke))
    else:
        r.ok(f"scalar-coordinate simplex/fbm/worley_f1 cook on {devices}")


# (label, family, program, band) — one row per tiered builtin, worley at two coordinate scales
# because its band is a law in the coordinate rather than a number (see _envelope_worley).
_ENVELOPE_PROBES = [
    ("simplex", "simplex",
     "@OUT = vec4(vec3(simplex(u*800.0, v*800.0)), 1.0);", _ENVELOPE_SIMPLEX["cuda"]),
    ("fbm, 6 octaves", "fbm6",
     "@OUT = vec4(vec3(fbm(u*4.0, v*4.0, 6)), 1.0);", _ENVELOPE_FBM),
    ("worley_f1 at 8", "worley_f1",
     "@OUT = vec4(vec3(worley_f1(u*8.0, v*8.0)), 1.0);", _envelope_worley(8.0)),
    ("worley_f2 at 8", "worley_f2",
     "@OUT = vec4(vec3(worley_f2(u*8.0, v*8.0)), 1.0);", _envelope_worley(8.0)),
    ("worley_f1 at 512", "worley_f1",
     "@OUT = vec4(vec3(worley_f1(u*512.0, v*512.0)), 1.0);", _envelope_worley(512.0)),
    ("worley_f2 at 512", "worley_f2",
     "@OUT = vec4(vec3(worley_f2(u*512.0, v*512.0)), 1.0);", _envelope_worley(512.0)),
]

_ENVELOPE_CHILD = _CHILD_HEAD + r'''
mutant = float(sys.argv[3])                    # 0.0 = the real promotion
probes = json.loads(sys.stdin.read())          # [[label, family, program], ...]
if mutant:
    promote_to_mutant(mutant)
dev = device_key("cuda")
KEYS = {"simplex": (noise._simplex_cache, dev), "fbm6": (noise._fbm_cache, (6, dev)),
        "worley_f1": (noise._worley_cache, (False, dev)),
        "worley_f2": (noise._worley_cache, (True, dev))}
H, W = 256, 256
WIN = (64, 51, 128, 128, W, H)
torch.manual_seed(5)
A = torch.rand(1, H, W, 4, device="cuda")

def cook(code, roi=None):
    tex_roi.clear_roi_memo()
    res = tex_engine.cook(code, {"A": A}, device_mode="cuda", precision="fp32",
                          roi=roi, roi_exec=True if roi else None)
    if roi is not None and res.cooked_roi != roi:
        raise RuntimeError(f"the window was not served: {tier_trace.last_roi()}")
    return res.outputs["OUT"]

def crop(t):
    x0, y0, w, h, _, _ = WIN
    return t[:, y0:y0 + h, x0:x0 + w]

# 1. The control: every key HELD at jit.trace, each program cooked whole and through the window.
for family in {family for _, family, _ in probes}:
    cache, key = KEYS[family]
    cache._compile_attempted.add(key)
control = {label: (cook(code), cook(code, WIN), _tier_of(*KEYS[family]))
           for label, family, code in probes}
# 2. Release each key once and let try_upgrade promote it through ordinary cooks.
released = set()
for label, family, code in probes:
    if family in released:
        continue
    released.add(family)
    cache, key = KEYS[family]
    cache._compile_attempted.discard(key)
    cache._call_count.pop(key, None)
    if noise._can_inductor_compile(dev):
        for _ in range(_COMPILE_AFTER_CALLS):
            cook(code)
# 3. The promoted tier against the control, and the mixed-tier window pair both ways.
for label, family, code in probes:
    whole_t, window_t, control_tier = control[label]
    row = {"label": label, "control_tier": control_tier, "tier": _tier_of(*KEYS[family])}
    if row["tier"] == "promoted":
        whole_p, whole_p2, window_p = cook(code), cook(code), cook(code, WIN)
        row.update(repeat=maxdiff(whole_p, whole_p2),
                   window_on_trace=maxdiff(window_t, crop(whole_t)),
                   window_on_promoted=maxdiff(window_p, crop(whole_p)),
                   envelope=maxdiff(whole_p, whole_t),
                   promoted_window_on_trace_frame=maxdiff(window_p, crop(whole_t)),
                   trace_window_on_promoted_frame=maxdiff(window_t, crop(whole_p)))
    print("PROBE " + json.dumps(row))
'''

_EXACT_COLUMNS = ("repeat", "window_on_trace", "window_on_promoted")
_ENVELOPE_COLUMNS = ("envelope", "promoted_window_on_trace_frame", "trace_window_on_promoted_frame")


def _envelope_rows(cache_dir, probes, mutant=0.0):
    """({label: row}, error) for one fresh process measuring `probes` (label, family, code)."""
    out, err = _run_child(_ENVELOPE_CHILD, [cache_dir, mutant],
                          stdin=json.dumps([list(p) for p in probes]))
    if err:
        return None, err
    rows = {}
    for line in out.splitlines():
        if line.startswith("PROBE "):
            row = json.loads(line[len("PROBE "):])
            rows[row["label"]] = row
    missing = [label for label, _, _ in probes if label not in rows]
    return (None, f"child emitted no row for {missing}") if missing else (rows, None)


def test_v031_noise_promotion_envelope(r: SubTestResult):
    """ENVELOPE, per builtin: the promoted tier against the fp32 interpreter control, and the
    mixed-tier pair a region recook creates, bounded by the same recorded band.

    Two kinds of column, judged differently. EXACT: a tier renders one image (`repeat`), and a
    window cooked on a tier crops bit-exactly out of that tier's whole frame — v0.30's CUDA
    row, now asserted per tier. ENVELOPE: the promoted whole frame against the jit.trace one,
    and the two ways a host can composite across the swap — a window cooked after it laid over
    a frame cooked before it, and the reverse. Worley is probed at two coordinate scales
    because its band follows its coordinates. CUDA only (module docstring: a CPU band would
    be a fact about one C++ compiler); CPU simplex stays bit-exact in `cold_frame_parity`.
    """
    print("\n--- NOISE-TIER: the promotion envelope, per builtin, and the mixed-tier window pair ---")
    if not _CUDA:
        r.skip("promotion envelope", "no CUDA — the envelope is recorded for CUDA; CPU simplex "
                                     "stays bit-exact in cold_frame_parity")
        return
    probes = [(label, family, code) for label, family, code, _ in _ENVELOPE_PROBES]
    bands = {label: band for label, _, _, band in _ENVELOPE_PROBES}

    with cold_engine_state() as cold:
        rows, err = _envelope_rows(cold.dir, probes)
    if err:
        r.fail("promotion envelope", err)
        return
    for label, row in rows.items():
        band = bands[label]
        if row["control_tier"] != "trace":
            r.fail(f"envelope control ({label})", f"the control was cooked on "
                   f"{row['control_tier']!r}, not jit.trace — the row compares nothing it names")
            continue
        if row["tier"] != "promoted":
            r.skip(f"promotion envelope ({label})", "the compile tier does not engage here")
            continue
        loose = [f"{col} {row[col]:.3e}" for col in _EXACT_COLUMNS if row[col] != 0.0]
        if loose:
            r.fail(f"promotion envelope exact ({label})", "a same-tier claim is no longer bit-exact: "
                   + ", ".join(loose))
        else:
            r.ok(f"{label}: each tier renders one image and crops its window bit-exactly")
        past = [f"{col} {row[col]:.3e}" for col in _ENVELOPE_COLUMNS if row[col] > band]
        measured = ", ".join(f"{col} {row[col]:.3e}" for col in _ENVELOPE_COLUMNS)
        if past:
            r.fail(f"promotion envelope ({label})", f"past the recorded {band:.2e}: " + ", ".join(past)
                   + " — the compile tier moved: re-measure and re-band deliberately")
        else:
            r.ok(f"{label}: {measured} — within the recorded {band:.2e}")

    # MUTATION: a simplex promotion KNOWN to sit 2x past its band must turn the envelope column
    # and both window-pair columns red while every exact column stays green. Box-independent,
    # so the judgement above is proven on every CUDA box, including the ones that skip it.
    label, family, code, band = _ENVELOPE_PROBES[0]
    with cold_engine_state() as cold:
        mrows, err = _envelope_rows(cold.dir, [(label, family, code)], mutant=2.0 * band)
    name = f"MUTATION {label} promotion + {2.0 * band:.1e}"
    row = (mrows or {}).get(label, {})
    if err or row.get("tier") != "promoted":
        r.fail(name, f"the mutant did not promote, so the row measures nothing: {err or row}")
    elif any(row[col] != 0.0 for col in _EXACT_COLUMNS):
        r.fail(name, f"the mutant broke a same-tier column: "
                     f"{ {col: row[col] for col in _EXACT_COLUMNS} }")
    elif not all(row[col] > band for col in _ENVELOPE_COLUMNS):
        r.fail(name, f"judged green against {band:.0e}: "
                     f"{ {col: row[col] for col in _ENVELOPE_COLUMNS} } — the envelope row is decorative")
    else:
        r.ok(f"{name}: every envelope column red against {band:.0e} "
             f"({', '.join(f'{row[col]:.3e}' for col in _ENVELOPE_COLUMNS)}), exact columns green")


# BRIEF-4 C6 — a torch.compile PROMOTION failure used to vanish into `try_upgrade`'s bare
# `except Exception: pass`. Fresh child, CPU-only and box-independent: seeding
# `_inductor_available["cpu"]=True` makes the 4th-call attempt fire with no real MSVC
# needed, and replacing `_compile_simplex` with one that always raises stands in for a
# genuine compiler failure (a C1083 is MSVC's real "cannot open compiler generated file").
_PROMOTION_FAILURE_CHILD = _CHILD_HEAD + r'''
dev = "cpu"
key, cache = device_key(dev), noise._simplex_cache
noise._inductor_available["cpu"] = True

def _boom(device):
    raise RuntimeError("C1083: cannot open compiler generated file")
noise._compile_simplex = _boom

torch.manual_seed(5)
img = torch.rand(1, 24, 32, 4, device=dev)
prog = ''' + repr(_SIMPLEX_PROG) + r'''
for _ in range(4):
    tex_engine.cook(prog, {"A": img}, device_mode=dev, precision="fp32")

from TEX_Wrangle.tex_doctor import capabilities
print(json.dumps({
    "tier": _tier_of(cache, key),
    "compile_attempted": key in cache._compile_attempted,
    "failures": tier_trace.noise_compile_failures(),
    "compiles": tier_trace.noise_compiles(),
    "row": capabilities()["rows"]["noise_promotion@cpu"],
}))
'''


def test_v031_noise_promotion_failure_recorded(r: SubTestResult):
    """BRIEF-4 C6 — a torch.compile PROMOTION failure (`try_upgrade`'s previously-swallowed
    exception, `noise.py`) is now RECORDED instead of vanishing: one entry naming the real
    exception; `noise_compiles()` untouched (a failure must never read as a compile event —
    the same discipline that ring's own docstring states); the incumbent jit.trace tier still
    serving the key (a failed promotion must not disturb what already works — the simplex
    calls above returned normal results throughout); and `tex_doctor.capabilities()`'s
    `noise_promotion@cpu` row reads unavailable/measured, naming the failure.
    """
    print("\n--- BRIEF-4 C6: a noise promotion failure is recorded, not swallowed ---")
    with cold_engine_state() as cold:
        out, err = _run_child(_PROMOTION_FAILURE_CHILD, [cold.dir])
    if err:
        r.fail("BRIEF-4 C6 promotion failure", err)
        return
    try:
        result = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError) as e:
        r.fail("BRIEF-4 C6 promotion failure", f"child printed no parseable JSON: {e}\n{out}")
        return

    fails = []
    if result["tier"] != "trace":
        fails.append(f"incumbent tier disturbed by the failed promotion: {result['tier']!r}")
    if not result["compile_attempted"]:
        fails.append("the 4th call never attempted the promotion")
    if result["compiles"]:
        fails.append(f"a failure was recorded on noise_compiles() as a success: "
                     f"{result['compiles']}")
    failures = result["failures"]
    if len(failures) != 1:
        fails.append(f"expected exactly one failure entry, got {failures}")
    else:
        f0 = failures[0]
        if (f0.get("noise") != "simplex" or f0.get("device") != "cpu"
                or "C1083" not in f0.get("error", "")):
            fails.append(f"failure entry wrong shape/content: {f0}")
    row = result["row"]
    if (row.get("status") != "unavailable" or row.get("evidence") != "measured"
            or "C1083" not in (row.get("why_not") or "")):
        fails.append(f"noise_promotion@cpu should be unavailable/measured naming C1083: {row}")

    if fails:
        r.fail("BRIEF-4 C6 promotion failure", "; ".join(fails))
    else:
        r.ok("a swallowed torch.compile promotion failure is now recorded (naming the real "
             "exception), never counted as a compile, leaves the incumbent trace tier "
             "serving the key, and reads unavailable/measured in capabilities()")


# ── The per-cook tier record (`want_noise_tiers`) ───────────────────────────────────────────────
#
# A host compositing a region recook over a cached frame patches only when the two cooks' records
# are dicts and equal, and cooks whole otherwise. So the record must name the tier that ACTUALLY
# served — read off the cache with `_tier_of`, never inferred from pixels — and asking for it must
# move neither a pixel nor the promotion. Programs are embedded into the children with repr(),
# as the promotion-failure child above does.

# Four calls to ONE key in one cook: in a fresh cache the fourth call is the promotion.
_FOUR_CALL_PROG = ("float a = simplex(u*800.0, v*800.0);\n"
                   "float b = simplex(u*400.0, v*400.0);\n"
                   "float c = simplex(u*200.0, v*200.0);\n"
                   "float d = simplex(u*100.0, v*100.0);\n"
                   "@OUT = vec4(vec3(a + b + c + d), 1.0);")
# One call to each of four tiered keys: two fbm octave counts and both worley flavours.
_LABELS_PROG = ("float a = fbm(u*4.0, v*4.0, 4);\n"
                "float b = fbm(u*4.0, v*4.0, 6);\n"
                "float c = worley_f1(u*8.0, v*8.0);\n"
                "float d = worley_f2(u*8.0, v*8.0);\n"
                "@OUT = vec4(a, b, c, d);")

_RECORD_HEAD = _CHILD_HEAD + r'''
dev = sys.argv[3]
key, cache = device_key(dev), noise._simplex_cache
SIMPLEX = ''' + repr(_SIMPLEX_PROG) + r'''
FOUR_CALLS = ''' + repr(_FOUR_CALL_PROG) + r'''
LABELS = ''' + repr(_LABELS_PROG) + r'''
torch.manual_seed(5)
img = torch.rand(1, 24, 32, 4, device=dev)          # the parity child's frame

def clear_simplex():
    cache.cache.clear(); cache._settled.clear()
    cache._compile_attempted.clear(); cache._call_count.clear()

def emit(leg, n, prog, keys=None, **kw):
    # One ROW per cook: the record; why a requested record is None; the tier each key's cache
    # holds afterwards, under the name the record files it by; the promotion failures this cook
    # recorded; the frame's digest.
    keys = keys or {"simplex": (cache, key)}
    for i in range(1, n + 1):
        failures = len(tier_trace.noise_compile_failures())
        res = tex_engine.cook(prog, {"A": img}, device_mode=dev, precision="fp32", **kw)
        print("ROW " + json.dumps({
            "leg": leg, "cook": i, "device": str(key), "record": res.noise_tiers,
            "reason": tier_trace.last_noise_tiers()[1],
            "tiers": {name: _tier_of(c, k) for name, (c, k) in keys.items()},
            "failures": len(tier_trace.noise_compile_failures()) - failures,
            "digest": digest(res.outputs["OUT"])}), flush=True)
'''

_RECORD_REAL_CHILD = _RECORD_HEAD + r'''
emit("armed", 5, SIMPLEX, want_noise_tiers=True)      # first in a fresh process: the honest form
clear_simplex()
emit("unarmed", 5, SIMPLEX)                           # the same five cooks, never asked
emit("declined", 1, SIMPLEX, want_noise_tiers=False)
clear_simplex()
emit("four_calls", 2, FOUR_CALLS, want_noise_tiers=True)
emit("labels", 1, LABELS, want_noise_tiers=True,
     keys={"fbm/4": (noise._fbm_cache, (4, key)), "fbm/6": (noise._fbm_cache, (6, key)),
           "worley/False": (noise._worley_cache, (False, key)),
           "worley/True": (noise._worley_cache, (True, key))})
'''

_RECORD_FORCED_CHILD = _RECORD_HEAD + r'''
# 1. A strategy other than "default" reads None even when noise ran on this thread: the
#    torch_compile strategy is routed to the default body here, so only the gate can say None.
clear_simplex()
select_tier, method = tex_engine.select_tier, tex_engine._TIER_METHOD["torch_compile"]
tex_engine.select_tier = lambda *a, **k: "torch_compile"
tex_engine._TIER_METHOD["torch_compile"] = tex_engine._run_default
try:
    emit("gated", 1, SIMPLEX, want_noise_tiers=True)
finally:
    tex_engine.select_tier, tex_engine._TIER_METHOD["torch_compile"] = select_tier, method
emit("ungated", 1, SIMPLEX, want_noise_tiers=True)

# 2. A signature that never settles demotes the key, and the eager body serves.
clear_simplex()
bitwise_same = noise._bitwise_same
noise._bitwise_same = lambda a, b: False
try:
    emit("eager", 2, SIMPLEX, want_noise_tiers=True)
finally:
    noise._bitwise_same = bitwise_same

# 3. The promotion installing "incumbent + 0": no toolchain needed, so every box runs it.
clear_simplex()
promote_to_mutant(0.0)
emit("forced", 5, SIMPLEX, want_noise_tiers=True)
clear_simplex()
emit("forced_four_calls", 2, FOUR_CALLS, want_noise_tiers=True)

# 4. The promotion RAISING — the stand-in for a compiler failure the promotion-failure row uses.
def _boom(device):
    raise RuntimeError("C1083: cannot open compiler generated file")
noise._compile_simplex = _boom
clear_simplex()
emit("failed", 5, SIMPLEX, want_noise_tiers=True)
clear_simplex()
emit("failed_four_calls", 2, FOUR_CALLS, want_noise_tiers=True)
'''


def _record_legs(script, dev):
    """({leg: [row, ...]}, error) from one fresh process running `script` on `dev`."""
    with cold_engine_state() as cold:
        out, err = _run_child(script, [cold.dir, dev])
    if err:
        return None, err
    legs = {}
    for line in out.splitlines():
        if line.startswith("ROW "):
            row = json.loads(line[len("ROW "):])
            legs.setdefault(row["leg"], []).append(row)
    return legs, None


def _served_record(row):
    """The record a cook must carry when every call it made to a key served the tier that key's
    cache holds afterwards — true of any cook that does not straddle a promotion. A cook whose
    promotion attempt raised served jit.trace, and says which of the two it was."""
    return {f"{name}@{row['device']}": ("promotion_failed" if tier == "trace" and row["failures"]
                                        else tier)
            for name, tier in row["tiers"].items()}


def _straddle_problem(row, label):
    """Why a cook that crossed the promotion mid-cook is misreported, or None: its record must be
    None (no single tier served it), with a reason naming the label and both tiers."""
    reason = row["reason"] or ""
    if row["record"] is None and all(s in reason for s in (label, "trace", "promoted")):
        return None
    return (f"cook #{row['cook']} served trace and promoted but recorded {row['record']!r} "
            f"(reason {reason!r})")


def test_v031_noise_tier_record_across_the_promotion(r: SubTestResult):
    """PROVENANCE, per device, in a fresh process with the real tiers: `want_noise_tiers=True`.

    Each of five cooks records the tier that served it, and the record flips exactly where the
    cache's own tier flips — at cook #4 wherever the compile tier engages, so the last jit.trace
    record and the first promoted one differ and a host declines that pair. On CPU the row states
    whatever this box's toolchain made of the promotion. A program calling simplex four times
    crosses the promotion inside its first cook: that record is None, with a reason naming both
    tiers, and the next cook's is promoted. Four tiered keys in one cook file four labels. And the
    same five cooks, unasked and in the same process from a cleared cache, render the same digests
    on the same tiers with `noise_tiers` left None.
    """
    print("\n--- NOISE-TIER record: which tier served each cook (want_noise_tiers) ---")
    for dev in ["cpu"] + (["cuda"] if _CUDA else []):
        legs, err = _record_legs(_RECORD_REAL_CHILD, dev)
        if err:
            r.fail(f"tier record ({dev})", err)
            continue
        armed = legs.get("armed", [])
        if len(armed) != _N_COOKS:
            r.fail(f"tier record ({dev})", f"the child emitted {len(armed)} armed cooks")
            continue
        label = f"simplex@{armed[0]['device']}"
        tiers = [row["tiers"]["simplex"] for row in armed]
        records = [row["record"] for row in armed]

        wrong = [f"cook #{row['cook']} recorded {row['record']!r}, the cache served "
                 f"{_served_record(row)!r}" for row in armed if row["record"] != _served_record(row)]
        if wrong:
            r.fail(f"tier record per cook ({dev})", "; ".join(wrong))
        else:
            r.ok(f"{dev}: each of {_N_COOKS} cooks records the tier that served it: "
                 + ", ".join(str(rec[label]) for rec in records))
            if "promoted" in tiers:
                flip = tiers.index("promoted") + 1
                if flip != _COMPILE_AFTER + 1 or set(tiers[flip - 1:]) != {"promoted"}:
                    r.fail(f"tier record across the promotion ({dev})",
                           f"the tiers ran {tiers}; the promotion belongs on cook "
                           f"#{_COMPILE_AFTER + 1} and holds from there")
                elif records[flip - 2] == records[flip - 1]:
                    r.fail(f"tier record across the promotion ({dev})",
                           f"cooks #{flip - 1} and #{flip} straddle the promotion but their records "
                           f"compare equal ({records[flip - 1]!r}) — a host could not decline the mix")
                else:
                    r.ok(f"{dev}: the record flips {records[flip - 2][label]} -> promoted at cook "
                         f"#{flip}, where the promotion lands, so a host declines that pair")
            elif dev == "cuda":
                r.skip(f"tier record across the promotion ({dev})",
                       "the compile tier does not engage here, so there is no flip to record")
            else:
                r.ok(f"cpu: the promotion did not install on this box's toolchain; the record "
                     f"states {sorted({rec[label] for rec in records})} throughout")

        # INVARIANT 7: unasked and asked-False, the field stays None and nothing else moves.
        unarmed, declined = legs.get("unarmed", []), legs.get("declined", [])
        moved = [f"cook #{a['cook']}: asked {a['tiers']['simplex']}/{a['digest']}, unasked "
                 f"{u['tiers']['simplex']}/{u['digest']}"
                 for a, u in zip(armed, unarmed) if (a["tiers"], a["digest"]) != (u["tiers"], u["digest"])]
        filled = [row["record"] for row in unarmed + declined if row["record"] is not None]
        if len(unarmed) != _N_COOKS or len(declined) != 1:
            r.fail(f"tier record unasked ({dev})",
                   f"the child emitted {len(unarmed)} unasked and {len(declined)} declined cooks")
        elif moved or filled or declined[0]["digest"] != armed[-1]["digest"]:
            r.fail(f"tier record unasked ({dev})",
                   "; ".join(moved + [f"unasked records {filled}"] * bool(filled)) or
                   f"want_noise_tiers=False rendered {declined[0]['digest']}, asked "
                   f"{armed[-1]['digest']}")
        else:
            r.ok(f"{dev}: unasked (and want_noise_tiers=False) the field stays None, and the same "
                 f"{_N_COOKS} cooks render the same digests on the same tiers as when asked")

        four = legs.get("four_calls", [])
        if len(four) != 2:
            r.fail(f"four calls in one cook ({dev})", f"the child emitted {len(four)} cooks")
        elif four[0]["tiers"]["simplex"] == "promoted":
            problem = _straddle_problem(four[0], label)
            if problem or four[1]["record"] != {label: "promoted"}:
                r.fail(f"four calls in one cook ({dev})",
                       problem or f"the cook after the straddle recorded {four[1]['record']!r}")
            else:
                r.ok(f"{dev}: four simplex calls cross the promotion inside cook #1, which reads "
                     f"None ({four[0]['reason']}); cook #2 records promoted")
        elif any(row["record"] != _served_record(row) for row in four):
            r.fail(f"four calls in one cook ({dev})",
                   f"no promotion engaged, yet the records read {[row['record'] for row in four]}")
        elif dev == "cuda":
            r.skip(f"four calls in one cook ({dev})", "the compile tier does not engage here")
        else:
            r.ok(f"cpu: no promotion engaged inside the four-call cook; its records state "
                 f"{[row['record'] for row in four]}")

        labels = legs.get("labels", [])
        want = _served_record(labels[0]) if len(labels) == 1 else None
        if want is None or labels[0]["record"] != want or set(want.values()) != {"trace"}:
            r.fail(f"tier record labels ({dev})",
                   f"expected four jit.trace labels {want!r}, recorded "
                   f"{labels[0]['record'] if labels else None!r}")
        else:
            r.ok(f"{dev}: fbm(4), fbm(6), worley_f1 and worley_f2 in one cook file four labels, "
                 f"each on jit.trace: {sorted(want)}")


def test_v031_noise_tier_record_forced(r: SubTestResult):
    """PROVENANCE with each tier FORCED, on CPU in a fresh process, so every box runs it.

    The promotion forced to install "incumbent + 0" records jit.trace for cooks #1-3 and promoted
    from #4, and a four-call program straddling it reads None, then promoted. The promotion forced
    to RAISE — through the same failure hook `tex_doctor.capabilities()` reads — records
    "promotion_failed" in the cook it failed in and "trace" after it, and a four-call cook that
    fails mid-cook reads "promotion_failed", not None: jit.trace served every call, so nothing
    mixed. A trace that never settles records "eager". A strategy other than "default" reads None
    with its reason, where the same cook on the default strategy records jit.trace.
    """
    print("\n--- NOISE-TIER record: forced promotion, failure, eager fallback and strategy gate ---")
    legs, err = _record_legs(_RECORD_FORCED_CHILD, "cpu")
    if err:
        r.fail("tier record (forced)", err)
        return
    label = "simplex@cpu"

    def records(leg):
        return [row["record"] for row in legs.get(leg, [])]

    gated = legs.get("gated", [])
    if (records("gated") == [None] and "torch_compile" in (gated[0]["reason"] or "")
            and records("ungated") == [{label: "trace"}]):
        r.ok(f"a strategy other than default reads None ({gated[0]['reason']}); the same cook on "
             f"the default strategy records trace")
    else:
        r.fail("tier record strategy gate",
               f"gated {records('gated')} (reason {gated[0]['reason'] if gated else None!r}), "
               f"ungated {records('ungated')}")

    eager = legs.get("eager", [])
    if records("eager") == [{label: "eager"}] * 2 and {row["tiers"]["simplex"] for row in eager} == {"eager"}:
        r.ok("a trace that never settles records eager, the tier that served")
    else:
        r.fail("tier record eager", f"recorded {records('eager')}, cache "
                                    f"{[row['tiers']['simplex'] for row in eager]}")

    want = ([{label: "trace"}] * _COMPILE_AFTER
            + [{label: "promoted"}] * (_N_COOKS - _COMPILE_AFTER))
    forced = legs.get("forced", [])
    if records("forced") == want and all(row["record"] == _served_record(row) for row in forced):
        r.ok(f"forced promotion: trace for cooks #1-{_COMPILE_AFTER}, promoted from "
             f"#{_COMPILE_AFTER + 1}, each agreeing with the cache")
    else:
        r.fail("tier record forced promotion",
               f"recorded {records('forced')}, cache {[row['tiers']['simplex'] for row in forced]}")

    straddle = legs.get("forced_four_calls", [])
    problem = (_straddle_problem(straddle[0], label) if len(straddle) == 2
               and straddle[0]["tiers"]["simplex"] == "promoted" else "the four-call cook did not promote")
    if problem or straddle[1]["record"] != {label: "promoted"}:
        r.fail("tier record forced straddle", problem or f"then {straddle[1]['record']!r}")
    else:
        r.ok(f"forced straddle: the four-call cook reads None ({straddle[0]['reason']}), the next "
             f"records promoted")

    failed = legs.get("failed", [])
    want = ([{label: "trace"}] * _COMPILE_AFTER + [{label: "promotion_failed"}]
            + [{label: "trace"}] * (_N_COOKS - _COMPILE_AFTER - 1))
    if (records("failed") == want and [row["failures"] for row in failed] == [0] * _COMPILE_AFTER
            + [1] + [0] * (_N_COOKS - _COMPILE_AFTER - 1)
            and {row["tiers"]["simplex"] for row in failed} == {"trace"}):
        r.ok(f"a promotion that raises records promotion_failed in cook #{_COMPILE_AFTER + 1}, the "
             f"cook its one failure was reported in, and trace before and after")
    else:
        r.fail("tier record failed promotion",
               f"recorded {records('failed')}, failures {[row['failures'] for row in failed]}, "
               f"cache {[row['tiers']['simplex'] for row in failed]}")

    failed_four = legs.get("failed_four_calls", [])
    if (records("failed_four_calls") == [{label: "promotion_failed"}, {label: "trace"}]
            and [row["failures"] for row in failed_four] == [1, 0]):
        r.ok("a four-call cook whose promotion raises mid-cook records promotion_failed, not None "
             "— jit.trace served every call — and the next cook records trace")
    else:
        r.fail("tier record failed straddle",
               f"recorded {records('failed_four_calls')}, failures "
               f"{[row['failures'] for row in failed_four]}")


def test_v031_noise_tier_record_default_path(r: SubTestResult):
    """INVARIANT 7 for `want_noise_tiers`, in-process and noise-free, so no tier state moves.

    The keyword defaults False and `CookResult.noise_tiers` defaults None; a cook that does not
    ask, or asks False, reads None, and one that asks with no tiered builtin reads {} — with the
    same pixels all three ways. A cook that raises while asking must not leave the thread's record
    armed past the next cook, or every later noise call would pay the armed path. And the ComfyUI
    node never asks (the source canary every opt-in engine feature ships with).
    """
    print("\n--- NOISE-TIER record: invisible unless a host asks ---")
    import dataclasses
    from TEX_Wrangle import tex_engine
    from TEX_Wrangle.tex_runtime import tier_trace
    from TEX_Wrangle.tex_runtime.host import CookCancelled

    fails = []
    try:
        keyword = inspect.signature(tex_engine.prepare).parameters.get("want_noise_tiers")
        field = {f.name: f for f in dataclasses.fields(tex_engine.CookResult)}.get("noise_tiers")
        if keyword is None or keyword.default is not False:
            fails.append("prepare() has no want_noise_tiers keyword defaulting to False "
                         f"(found {keyword})")
        if field is None or field.default is not None:
            fails.append("CookResult has no noise_tiers field defaulting to None "
                         f"(found {getattr(field, 'default', field)!r})")

        code, img = "@OUT = vec4(@A.rgb * 0.5, 1.0);", make_img(1, 16, 16, 4, seed=7)

        def cook(**kw):
            return tex_engine.cook(code, {"A": img}, device_mode="cpu", **kw)

        plain, declined, asked = cook(), cook(want_noise_tiers=False), cook(want_noise_tiers=True)
        if plain.noise_tiers is not None or declined.noise_tiers is not None:
            fails.append(f"unasked {plain.noise_tiers!r}, False {declined.noise_tiers!r}: not None")
        if asked.noise_tiers != {}:
            fails.append(f"asked with no tiered builtin: {asked.noise_tiers!r}, not {{}}")
        if not all(torch.equal(plain.outputs["OUT"], other.outputs["OUT"]) for other in (declined, asked)):
            fails.append("asking for the record changed the pixels")

        class _CancelOnSecondPoll:           # run() polls once on entry; the interpreter polls next
            polls = 0

            def check(self):
                self.polls += 1
                if self.polls > 1:
                    raise CookCancelled("cancelled mid-cook")

        try:
            cook(want_noise_tiers=True, cancel=_CancelOnSecondPoll())
            fails.append("the cancel token never fired, so the leak check below measures nothing")
        except CookCancelled:
            pass
        after = cook()
        if after.noise_tiers is not None:
            fails.append(f"an unasked cook after the raising one read {after.noise_tiers!r}")
        if tier_trace.take_noise_tiers() is not None:
            fails.append("this thread's record was still armed after an unasked cook: the cook "
                         "that raised while asking leaked it, or the unasked cook armed it")
    except Exception as e:
        fails.append(f"{type(e).__name__}: {e}")

    node = (Path(__file__).resolve().parents[1] / "tex_node.py").read_text(encoding="utf-8")
    if "noise_tiers" in node:
        fails.append("tex_node.py mentions noise_tiers — the ComfyUI path must never ask")

    if fails:
        r.fail("tier record default path", "; ".join(fails))
    else:
        r.ok("want_noise_tiers defaults False and noise_tiers None; unasked/False read None, asked "
             "with no noise {} — identical pixels; a raising cook leaves nothing armed; tex_node "
             "never asks")


def test_eng16_noise_tiers_compatible(r: SubTestResult):
    """CANARY: `tier_trace.noise_tiers_compatible(a, b)` — the callable form of the module
    docstring's compositing rule — pins all six arms named in the ENG-16 ask, plus the "never
    raise on a malformed record" clause. A shape change to `CookResult.noise_tiers` (see
    `tex_engine.CookResult.noise_tiers`, `take_noise_tiers`) that this function stops covering
    reds one of the rows below rather than surfacing as a silent wrong-pixel composite downstream.
    """
    print("\n--- ENG-16: noise_tiers_compatible pins the compositing rule as one callable ---")
    from TEX_Wrangle.tex_runtime import tier_trace
    compat = tier_trace.noise_tiers_compatible

    cases = [
        # (label, a, b, want)
        ("both None -> refuse (record unknown on both sides)", None, None, False),
        ("one None, one empty dict -> refuse (record unknown on one side)",
         None, {}, False),
        ("one None, one non-empty dict -> refuse (record unknown on one side)",
         {"simplex@cpu": "trace"}, None, False),
        ("both empty dicts -> compatible (neither cook used a tiered builtin)",
         {}, {}, True),
        ("both non-empty, fully agreeing -> compatible",
         {"simplex@cpu": "trace", "fbm/6@cpu": "trace"},
         {"simplex@cpu": "trace", "fbm/6@cpu": "trace"}, True),
        ("both non-empty, a shared label disagrees -> refuse",
         {"simplex@cuda:0": "trace"}, {"simplex@cuda:0": "promoted"}, False),
        ("both non-empty, a label named in only one -> refuse",
         {"simplex@cpu": "trace", "fbm/6@cpu": "trace"},
         {"simplex@cpu": "trace"}, False),
        ("asymmetric in the other direction -> refuse (the question is symmetric)",
         {"simplex@cpu": "trace"},
         {"simplex@cpu": "trace", "fbm/6@cpu": "trace"}, False),
        ("same label, different device suffix -> refuse (no shared label at all)",
         {"simplex@cpu": "trace"}, {"simplex@cuda:0": "trace"}, False),
        ("both cooks recorded promotion_failed on the same label -> compatible "
         "(tier words are compared for equality only, never parsed)",
         {"simplex@cpu": "promotion_failed"}, {"simplex@cpu": "promotion_failed"}, True),
    ]
    wrong = []
    for label, a, b, want in cases:
        got = compat(a, b)
        if got is not want:
            wrong.append(f"{label}: compat(a, b)={got!r}, want {want!r}")
        got_rev = compat(b, a)
        if got_rev is not want:
            wrong.append(f"{label}: compat(b, a)={got_rev!r} disagrees with compat(a, b) — "
                         f"the rule is supposed to be symmetric")

    # Never raise: the wrong type entirely reads as refuse, and so does a dict whose own
    # comparison misbehaves — a shape this function has never been handed but must not crash
    # a host's compositing decision over.
    class _RaisingEq:
        def __eq__(self, other):
            raise RuntimeError("a value type from a future record shape that compares unsafely")
        __hash__ = object.__hash__

    malformed = [
        ("a list, not a dict", ["simplex@cpu", "trace"], {}),
        ("a bare string", "simplex@cpu", {}),
        ("an int", 0, {}),
        ("a dict whose value's __eq__ raises", {"simplex@cpu": _RaisingEq()},
         {"simplex@cpu": _RaisingEq()}),
    ]
    for label, a, b in malformed:
        try:
            got = compat(a, b)
        except Exception as e:
            wrong.append(f"malformed input {label} raised {type(e).__name__}: {e} — "
                         f"must return False, never raise")
            continue
        if got is not False:
            wrong.append(f"malformed input {label}: compat returned {got!r}, want False "
                         f"(a value it cannot prove safe)")

    if wrong:
        r.fail("noise_tiers_compatible", "; ".join(wrong))
    else:
        r.ok(f"noise_tiers_compatible judged all {len(cases)} record-shape arms (both "
             f"directions) and {len(malformed)} malformed inputs correctly, never raising")
