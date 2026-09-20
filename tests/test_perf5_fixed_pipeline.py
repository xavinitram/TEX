"""PERF-5 — the fixed front-end costs a cook paid whatever it was cooking.

TWO THINGS WERE PAID PER COOK THAT ARE PURE FUNCTIONS OF THE SOURCE.

1. `TEXCache.fingerprint` ran TWICE per warm cook and THREE times per cold one. `prepare`
   needs the value for `_preflight_memory`'s memo and computed it beside the compile; the
   compile computed it again in `get`, and once more in `put` on a miss. Each call builds a
   sorted binding key, asks `param_only_names` which names to drop, and probes a memo.
2. A never-seen program was LEXED TWICE. The `param_only_names` under (1) tokenizes the
   source; the compile behind it tokenizes the same source again for the parse.

Neither is a correctness bug, which is exactly why this file is shaped the way it is. The
risk in fixing them is not that a cook breaks — it is that **a fingerprint string moves**.
A fingerprint names an on-disk `.pkl`/`.cg` and seeds a lineage key, so a recipe that drifts
by one byte silently invalidates every user's compiled-program cache and every persisted
codegen sidecar. So the first rows here are a GOLDEN, recorded at the base sha over every
shipped `examples/*.tex` and the ten `examples/host_demo.py::_COMP_STAGES` programs under two
binding maps each, with a mutation row requiring that a one-character change to the recipe —
or to the param-dropping rule the recipe rests on — reds it.

THE COUNT ROWS ARE THE ASK. `TEXCache.fingerprint` is spied through a cook and required to be
entered exactly ONCE per cook, warm or cold, and `Lexer.tokenize` is spied through
`tex_api.prewarm` and required to be entered exactly once per never-seen program. Both are
also pinned per interactive tick in `tests/test_bench2_counts.py`; the rows here name the
mechanism, so a failure says which of the two seams regressed rather than only that a count
moved.

THE HANDOFF'S OWN SAFETY ROW. Sharing a lex means sharing `Token` objects, and `Parser` puts
`tok.loc` — the Token's own `SourceLoc` OBJECT — straight into the nodes it builds, while the
fused-chain tagger later WRITES `SourceLoc.stage`. So the handoff is one-shot: claiming
consumes. `test_perf5_the_token_handoff_is_consumed` proves two parses of one source share no
`SourceLoc`, and its mutation makes the handoff non-consuming and requires the check to fail.

PORTABILITY: compiler + CPU interpreter only. No ComfyUI, no CUDA, no compiler toolchain, no
numpy, no timing assertion. The cache rows run inside `cold_engine_state`.
"""
import hashlib
import importlib.util
import json
import os

from helpers import *

from TEX_Wrangle import tex_cache, tex_engine
from TEX_Wrangle.tex_cache import get_cache
from TEX_Wrangle.tex_marshalling import param_only_names

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "perf5_goldens", "fingerprints.json")
#: Minted at the sha before the fingerprint call-count change, by running `TEXCache.fingerprint`
#: and `param_only_names` over the corpus `_sources()` builds and writing the answers out. A row
#: carries the source's sha256 so a golden can never be checked against a source that moved under
#: it, and the binding maps VERBATIM so the question is not re-derived on head. To re-mint one
#: DELIBERATELY (a source really did change): the `wires` map types every `@` base VEC4 and every
#: `$`-only name FLOAT; the `typed` map types every name by `sha256(name)[:8] % 4` over
#: `(vec4, vec3, float, int)`, so the binding VALUES differ row to row.
_MIN_ROWS = 120


def _golden():
    with open(_GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)["corpus"]


def _sources() -> dict:
    """`key -> source` for the golden's corpus, read from the tree."""
    out = {}
    exdir = os.path.join(_ROOT, "examples")
    for fn in sorted(os.listdir(exdir)):
        if fn.endswith(".tex"):
            with open(os.path.join(exdir, fn), encoding="utf-8") as fh:
                out[f"examples/{fn}"] = fh.read()
    spec = importlib.util.spec_from_file_location(
        "_perf5_host_demo", os.path.join(exdir, "host_demo.py"))
    demo = importlib.util.module_from_spec(spec)
    sys.modules["_perf5_host_demo"] = demo
    spec.loader.exec_module(demo)
    for n, code, _d in demo._COMP_STAGES:
        out[f"_COMP_STAGES/{n}"] = code
    return out


def _types(m: dict) -> dict:
    return {k: TEXType(v) for k, v in m.items()}


# ── the golden ──────────────────────────────────────────────────────────────

def test_perf5_fingerprint_and_param_only_names_are_byte_stable(r: SubTestResult):
    """Every fingerprint string and every `param_only_names` answer in the corpus is the one
    the base sha produced.

    This is the invariant the whole lane is constrained by, not a nicety: `fingerprint` keys
    the on-disk `.pkl` and `.cg` artifacts and seeds CACHE-1 lineage keys, so a changed string
    throws away every user's compiled-program cache once and silently. The corpus is every
    shipped example plus the ten demo comp stages, each under a `wires` map (every `@` base
    VEC4, every `$` FLOAT) and a `typed` map (types hashed from the NAME, so the binding
    values differ row to row and the key's ENCODING is exercised, not only its length)."""
    rows = _golden()
    src = _sources()
    bad, checked = [], 0
    if len(rows) < _MIN_ROWS:
        bad.append(f"the golden carries only {len(rows)} rows (< {_MIN_ROWS}) — it was "
                   f"regenerated against a shrunken corpus, so it proves much less")
    for row in rows:
        code = src.get(row["key"])
        if code is None:
            bad.append(f"{row['key']}: in the golden, not in the tree")
            continue
        if hashlib.sha256(code.encode()).hexdigest() != row["sha256_source"]:
            bad.append(f"{row['key']}: the SOURCE moved under the golden — re-derive the "
                       f"golden deliberately (it invalidates that program's cache) rather "
                       f"than loosening this row")
            continue
        got = sorted(param_only_names(code))
        if got != row["param_only_names"]:
            bad.append(f"{row['key']}: param_only_names {row['param_only_names']} -> {got}")
        for label in ("wires", "typed"):
            want = row[label]["fingerprint"]
            fp = TEXCache.fingerprint(code, _types(row[label]["binding_types"]))
            checked += 1
            if fp != want:
                bad.append(f"{row['key']} [{label}]: fingerprint {want[:12]}… -> {fp[:12]}…")
    if bad:
        r.fail("PERF-5 fingerprint golden", "; ".join(bad[:8]) +
               (f" (+{len(bad) - 8} more)" if len(bad) > 8 else "") +
               " || a fingerprint names an on-disk cache file; moving one invalidates every "
               "user's cache, so this is a release decision and never a re-pin")
    else:
        r.ok(f"{checked} fingerprints + {len(rows)} param_only_names answers unchanged "
             f"from the base sha")


def test_perf5_the_golden_catches_a_one_character_recipe_change(r: SubTestResult):
    """MUTATION, both halves of the recipe.

    (a) The DIGEST: the same recipe with one character changed — the length prefix written
        big-endian instead of little — must disagree with the golden.
    (b) The PARAM DROP: `fingerprint` excludes param-only names by asking
        `param_only_names`. Neutralise that call and the golden must notice, on a program
        that actually has a param-only name.

    Without both, a golden that had stopped being computed from the shipping recipe would
    look exactly like a stable one."""
    rows, src = _golden(), _sources()

    def mutant(code, binding_types):
        drop = param_only_names(code)
        key = tuple(sorted((k, v.value) for k, v in binding_types.items() if k not in drop))
        h = hashlib.sha256()
        b = code.encode()
        h.update(len(b).to_bytes(8, "big"))          # the one character: "little" -> "big"
        h.update(b)
        h.update(json.dumps(key).encode())
        return h.hexdigest()

    caught = sum(1 for row in rows
                 if row["key"] in src
                 and mutant(src[row["key"]], _types(row["wires"]["binding_types"]))
                 != row["wires"]["fingerprint"])
    r.ok(f"(a) a one-character digest change moves {caught}/{len(rows)} golden rows") \
        if caught == len(rows) else \
        r.fail("PERF-5 golden mutation (digest)",
               f"only {caught}/{len(rows)} rows moved under a changed digest recipe — the "
               f"golden is not computed from the recipe it claims to pin")

    withparams = [row for row in rows if row["param_only_names"] and row["key"] in src]
    import TEX_Wrangle.tex_marshalling as _m
    saved = _m.param_only_names
    moved = 0
    try:
        _m.param_only_names = lambda code: frozenset()
        for row in withparams:
            code = src[row["key"]]
            tex_cache._FINGERPRINT_MEMO.clear()
            if TEXCache.fingerprint(code, _types(row["wires"]["binding_types"])) \
                    != row["wires"]["fingerprint"]:
                moved += 1
    finally:
        _m.param_only_names = saved
        tex_cache._FINGERPRINT_MEMO.clear()
    r.ok(f"(b) dropping the param-drop rule moves {moved}/{len(withparams)} rows that have "
         f"param-only names") if withparams and moved == len(withparams) else \
        r.fail("PERF-5 golden mutation (param drop)",
               f"{moved}/{len(withparams)} rows moved when `param_only_names` was neutralised "
               f"— the golden does not cover ANIM-1's exclusion rule")


# ── the count rows ──────────────────────────────────────────────────────────

class _FpSpy:
    """Count entries to `TEXCache.fingerprint`.

    It is a `@staticmethod`: assigning a plain function over it turns every
    `cache.fingerprint(code, types)` into a BOUND method and the next cook dies on arity, so
    the descriptor kind is re-applied. (The counts harness carries the same note.)"""

    def __init__(self):
        self.n = 0

    def __enter__(self):
        self._orig = TEXCache.__dict__["fingerprint"].__func__
        spy = self

        def wrapper(code, binding_types):
            spy.n += 1
            return spy._orig(code, binding_types)
        TEXCache.fingerprint = staticmethod(wrapper)
        return self

    def __exit__(self, *e):
        TEXCache.fingerprint = staticmethod(self._orig)
        return False


_PROG = ("$gain = 1.25;\n"
         "float g = $gain;\n"
         "@OUT = vec4(@IN.rgb * g, 1.0);\n")


def _cook(code=_PROG, gain=1.25):
    return tex_engine.cook(code, {"IN": make_img(1, 8, 8, 4), "gain": gain},
                           device_mode="cpu", compile_mode="none")


def test_perf5_one_fingerprint_per_cook(r: SubTestResult):
    """A cook enters `TEXCache.fingerprint` exactly ONCE — cold and warm.

    `prepare` computes it for `_preflight_memory`'s memo and hands the value to the compile,
    which uses the same string for the cache probe AND the store. At the base a warm cook read
    2 (here + `get`) and a cold one 3 (+ `put`). Nothing between the probe and the store
    mutates `binding_types` — the splitback and both TypeChecker passes only read it — so the
    single value is the one each site would have computed for itself."""
    with cold_engine_state():
        with _FpSpy() as s:
            _cook()
        cold = s.n
        with _FpSpy() as s:
            _cook(gain=2.5)                      # same program, new param value: a WARM cook
        warm = s.n
    for label, got in (("cold (compile + store)", cold), ("warm (cache hit)", warm)):
        r.ok(f"{label}: 1 TEXCache.fingerprint per cook") if got == 1 else \
            r.fail("PERF-5 fingerprint count",
                   f"{label}: {got} calls to TEXCache.fingerprint in one cook, expected 1")


def test_perf5_the_shared_fingerprint_is_the_cache_key(r: SubTestResult):
    """The single fingerprint is the string the disk tier files under — the half a call count
    cannot see. A cook that computed the key once but stored under a DIFFERENT string would
    satisfy the row above and never hit its own cache again."""
    with cold_engine_state():
        _cook()
        cache = get_cache()
        fp = TEXCache.fingerprint(_PROG, {"IN": TEXType.VEC4, "gain": TEXType.FLOAT})
        on_disk = sorted(p.name for p in Path(cache._cache_dir).glob("*.pkl"))
        r.ok(f"the cook's artifact is filed under the public fingerprint ({fp[:12]}…)") \
            if f"{fp}.pkl" in on_disk else \
            r.fail("PERF-5 fingerprint key",
                   f"no {fp[:12]}….pkl after a cold cook; the dir holds "
                   f"{[n[:12] for n in on_disk]} — the compile stored under a different key")
