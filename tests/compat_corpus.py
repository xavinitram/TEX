"""
LANG-3 — the frozen language-compatibility corpus. R2-archive (v0.34) — append-only.

Runs every `examples/*.tex` plus a set of adversarial grammar programs through the
INTERPRETER on CPU and hashes the quantized outputs. The hashes are frozen per LANGUAGE
VERSION in `compat_corpus_goldens/<lang_version>.json`. A mismatch means a language change
(parser / type checker / interpreter / optimizer) altered what an EXISTING program
computes — the regression PM-4 guards against ("v0.25 runs a v0.22 program the same way").

R2-ARCHIVE, and why the shape changed (doc 40 §1, doc 41 §3.4). Until v0.34 there was ONE
golden file and `regen()` **overwrote the whole thing**. "Old goldens are immutable" was
therefore a convention enforced by review — and a convention is not a compatibility proof,
because the one action that breaks it (re-freezing after an unintended pixel change) is
also the one action that makes the suite go green again. So:

  * the single file becomes a per-version **archive directory**, one file per frozen
    language version;
  * `regen()` is GONE. `freeze(version)` may only ADD a version that is not there yet, and
    refuses to rewrite one that is — the immutability is machinery now, not manners;
  * the corpus test runs current behavior against **every** archived version.

It lands in v0.34 rather than v0.35 (where the first new freeze is due) precisely so it
lands while it is neutral: there is exactly ONE archived version (0.23), so the test does
exactly what it did before, and that is the proof the mechanism changed nothing.

Correcting an archived version is deliberately awkward and deliberately possible: delete
the file and `freeze()` it again, in a commit whose message argues the pixel change. The
awkwardness is the review gate.

Determinism / portability:
  * single-threaded CPU + the example harness's per-binding fixed seed makes each run
    reproducible;
  * outputs are quantized to the 8-bit visible quantum (`round(x*255)`) before hashing,
    so sub-1e-3 float noise — and cross-machine variance — never flips a hash, while any
    real semantic change (which moves pixels by whole levels) does;
  * NaN/Inf are mapped to sentinels so a divergent-but-finite change stays detectable.
CPU-pinned by design (the roadmap: the corpus is CPU-pinned, never GPU).

Freeze a NEW language version (only ever after the surface change that earned the bump):
    python -X utf8 -c "import texboot, compat_corpus; compat_corpus.freeze()"
(run from tests/, with the scratchpad on sys.path for texboot). It writes
`compat_corpus_goldens/<tex_api.LANGUAGE_VERSION>.json` and refuses if that file exists.
"""
import hashlib
import json
import os

import torch

import test_integration as _ti   # reuse the frozen dummy-input harness (_prepare_example)
from TEX_Wrangle.tex_runtime.interpreter import Interpreter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ARCHIVE = os.path.join(_HERE, "compat_corpus_goldens")
_EXAMPLES = os.path.join(os.path.dirname(_HERE), "examples")

_B, _H, _W = 2, 16, 16   # match test_example_files (B=2 exercises batch/temporal paths)

# Adversarial grammar programs — small, self-contained (builtin coords only, no @inputs),
# each exercising a grammar/semantics corner whose output must stay stable across versions.
_ADVERSARIAL = {
    "adv_ternary_ops":
        "@OUT = vec4((u > 0.5 ? 1.0 : (v > 0.5 ? 0.5 : 0.0)), u*v, u+v-1.0, 1.0);",
    "adv_for_accumulate":
        "float s = 0.0; for (int i=0;i<4;i=i+1){ s = s + float(i)*0.1; }\n"
        "@OUT = vec4(s, s*0.5, s*0.25, 1.0);",
    "adv_while_loop":
        "float x = u; int n = 0; while (x < 1.0 && n < 8){ x = x + 0.1; n = n + 1; }\n"
        "@OUT = vec4(x, float(n)*0.1, 0.0, 1.0);",
    "adv_array_index":
        "float arr[4]; for(int i=0;i<4;i=i+1){ arr[i] = float(i)*u; }\n"
        "@OUT = vec4(arr[0], arr[1], arr[2], 1.0);",
    "adv_swizzle_vecops":
        "vec4 c = vec4(u, v, u*v, 1.0);\nvec2 p = c.xy;\n"
        "@OUT = vec4(c.z, p.y, c.x * 2.0 - c.z, c.w);",
    "adv_user_function":
        "float sq(float x){ return x*x; }\n@OUT = vec4(sq(u), sq(v), sq(u*v), 1.0);",
    "adv_math_builtins":
        "@OUT = vec4(sin(u*PI), cos(v*TAU), sqrt(abs(u-v)), 1.0);",
    "adv_mix_clamp_smoothstep":
        "@OUT = vec4(mix(0.2, 0.8, u), clamp(v*2.0-0.5, 0.0, 1.0), smoothstep(0.2, 0.8, u), 1.0);",
    "adv_param_metadata":   # LANG-1 grammar: metadata block, default used at cook time
        "f$gain = 1.5 [min: 0, max: 2, label: \"Gain\"];\n"
        "@OUT = vec4(u*$gain, v*$gain, 0.0, 1.0);",
    "adv_matrix_mul":
        "mat3 m = mat3(1,0,0, 0,1,0, 0,0,1);\nvec3 r = m * vec3(u, v, 1.0);\n@OUT = vec4(r, 1.0);",
    "adv_mod_floor_fract":
        "@OUT = vec4(mod(u*10.0, 1.0), floor(v*4.0)/4.0, fract(u+v), 1.0);",
    "adv_const_compound":
        "const float k = 0.3; float a = u; a += k; a *= 2.0;\n@OUT = vec4(a, a-k, a*0.5, 1.0);",
    "adv_pragma_current":   # LANG-3: a language pragma is an inert comment to the compiler
        "//!tex 0.23\n@OUT = vec4(u, v, u*v, 1.0);",
}


def _hash_outputs(result, output_names) -> str:
    """SHA-256 over each output's 8-bit-quantized values (+ name + shape). NaN/Inf are
    mapped to fixed sentinels so a change that produces non-finite pixels is still a
    stable, comparable hash rather than undefined bytes."""
    h = hashlib.sha256()
    for name in sorted(output_names):
        val = result[name]
        h.update(name.encode("utf-8"))
        if not torch.is_tensor(val):
            # STRING (and any other non-tensor) output — hash its repr verbatim.
            h.update(("str:" + repr(val)).encode("utf-8"))
            continue
        t = val.detach().to(torch.float64).cpu()
        t = torch.nan_to_num(t, nan=-999.0, posinf=998.0, neginf=-998.0)
        q = torch.round(t * 255.0).to(torch.int64)
        h.update(repr(tuple(q.shape)).encode("utf-8"))
        h.update(repr(q.flatten().tolist()).encode("utf-8"))
    return h.hexdigest()


def _program_hash(code: str) -> str:
    """Compile + run one program on CPU and hash its outputs. The caller pins
    single-threaded CPU (compute_all) for determinism and restores it after."""
    program, bindings, type_map, output_names = _ti._prepare_example(code, _B, _H, _W)
    if not output_names:
        raise ValueError("no output bindings")
    result = Interpreter().execute(program, bindings, type_map, device="cpu",
                                   output_names=output_names, source=code)
    return _hash_outputs(result, output_names)


def _corpus_programs():
    """Yield (name, source) for the whole corpus: every example + the adversarial set."""
    if os.path.isdir(_EXAMPLES):
        for fn in sorted(os.listdir(_EXAMPLES)):
            if fn.endswith(".tex"):
                with open(os.path.join(_EXAMPLES, fn), encoding="utf-8") as f:
                    yield fn[:-4], f.read()
    for name, src in sorted(_ADVERSARIAL.items()):
        yield name, src


def compute_all() -> dict:
    """Name → output hash for every corpus program. Pins single-threaded CPU for
    determinism (and restores the prior thread count so the rest of the suite is
    unaffected). A program that fails to compile is recorded as ERROR:<type>."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        out = {}
        for name, src in _corpus_programs():
            try:
                out[name] = _program_hash(src)
            except Exception as e:
                out[name] = f"ERROR:{type(e).__name__}"
        return out
    finally:
        torch.set_num_threads(prev_threads)


def _ver_key(v: str) -> tuple:
    """Sort key for a language version. Numeric per component so `0.9` sorts BELOW `0.23`
    — `tex_api._ver_tuple` int-parses the same way, and a lexical sort would silently make
    the newest archive the wrong file exactly once, at the 0.9 → 0.10 boundary."""
    try:
        return tuple(int(p) for p in v.split("."))
    except ValueError:
        return (-1,)          # an unparseable name sorts first and is never "the newest"


def archived_versions() -> list:
    """Every frozen language version, oldest first. Empty if the archive is missing."""
    if not os.path.isdir(_ARCHIVE):
        return []
    return sorted((fn[:-5] for fn in os.listdir(_ARCHIVE) if fn.endswith(".json")),
                  key=_ver_key)


def archive_path(version: str) -> str:
    return os.path.join(_ARCHIVE, f"{version}.json")


def load_version(version: str) -> dict:
    """One archived version's payload: `{"language_version": ..., "hashes": {...}}`."""
    with open(archive_path(version), encoding="utf-8") as f:
        return json.load(f)


def load_archive() -> dict:
    """`{language_version: payload}` for every frozen version, oldest first."""
    return {v: load_version(v) for v in archived_versions()}


def load_goldens(version: str | None = None) -> dict:
    """One version's payload — the NEWEST archived version by default.

    Kept at its old name and old return shape so a caller that only ever wanted "the
    goldens" is unaffected by the archive split."""
    versions = archived_versions()
    if not versions:
        raise FileNotFoundError(f"no frozen corpus versions in {_ARCHIVE}")
    return load_version(version or versions[-1])


def freeze(version: str | None = None) -> dict:
    """Freeze current behavior as a NEW archived version. Defaults to `LANGUAGE_VERSION`.

    **May only ADD.** Re-freezing an existing version raises, because that is the single
    action the archive exists to prevent: a real pixel regression on a frozen program is
    also, from the suite's point of view, one overwrite away from green. Correcting an
    archived version means deleting the file in a commit that argues the change — visible
    in review, which a silent rewrite was not.
    """
    from TEX_Wrangle.tex_api import LANGUAGE_VERSION
    version = str(version or LANGUAGE_VERSION)
    path = archive_path(version)
    if os.path.exists(path):
        raise FileExistsError(
            f"language version {version} is already frozen at {path}. The archive is "
            f"append-only: bump tex_api.LANGUAGE_VERSION for a new surface, or delete "
            f"that file deliberately if a frozen golden is genuinely wrong.")
    os.makedirs(_ARCHIVE, exist_ok=True)
    data = {"language_version": version, "hashes": compute_all()}
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"froze {len(data['hashes'])} goldens as language version {version} -> {path}")
    return data
