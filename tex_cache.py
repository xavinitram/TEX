"""
TEX Cache — two-tier compilation cache for TEX programs.

Tier 1 (memory): OrderedDict with LRU eviction. Stores ready-to-execute
(program, type_map, referenced_bindings, assigned_bindings,
param_declarations, used_builtins) tuples.

Tier 2 (disk): Pickle files in .tex_cache/. Stores a dict
{version, program, binding_types, timestamp}. On load, re-runs the type
checker to regenerate type_map with valid id() keys (~0.1ms, negligible).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import marshal
import os
import pickle
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .tex_compiler.lexer import Lexer
from .tex_compiler.parser import Parser
from .tex_compiler.type_checker import TypeChecker
from .tex_compiler.types import TEXType
from .tex_compiler.optimizer import optimize
from .tex_runtime.interpreter import _collect_identifiers

logger = logging.getLogger("TEX")


# CACHE-4: LAYERED CACHE EPOCHS. A single mono-hash over every source file cold-started every
# artifact on any edit — a comment-only stdlib change threw away the parsed-program (.pkl) tier
# for every user (fatal at monthly app-update cadence). The invalidation is split into a NESTED
# lattice, each epoch gating exactly the artifacts a change to its files can affect:
#
#   AST_EPOCH      parse/typecheck/optimize files          -> the compiled-program .pkl tier
#   CODEGEN_EPOCH  = H(AST_EPOCH, codegen/interpreter files, cgreuse) -> .cg sidecars + inductor
#   VERDICT_EPOCH  = H(CODEGEN_EPOCH, tier-policy files)     -> autotier.json + warm_state.json
#
# The nesting is load-bearing: a .cg blob is emitted FROM the compiled program, so it depends on
# the AST pipeline as well as the codegen files — CODEGEN_EPOCH folds AST_EPOCH in, or an
# AST-file edit would leave a stale .cg passing its version check (codegen drifting from the
# interpreter). THE WIN: a codegen-only edit bumps CODEGEN_EPOCH (invalidating .cg + verdicts)
# while AST_EPOCH is unchanged, so the parse/typecheck/optimize .pkl SURVIVES. The full mono-hash
# is DEMOTED to a completeness tripwire (test_v025_phase1: the partition file-sets must union to
# the watched set and AST/CODEGEN must be disjoint) plus a fail-safe oracle spot-check, so a
# watched file can never silently fall out of every epoch.
_C_DIR = Path(__file__).parent / "tex_compiler"
_R_DIR = Path(__file__).parent / "tex_runtime"
# AST pipeline — a change alters the parsed/optimized program (the .pkl).
_AST_FILES = [_C_DIR / "ast_nodes.py", _C_DIR / "lexer.py", _C_DIR / "parser.py",
              _C_DIR / "type_checker.py", _C_DIR / "optimizer.py",
              _C_DIR / "stdlib_signatures.py"]
# Codegen / interpreter — a change alters emitted code or interpreter semantics (the .cg).
# CT-1: tex_fusion is here — a splicer change must invalidate fused .cg entries too.
_CODEGEN_FILES = [_R_DIR / "interpreter.py", _R_DIR / "codegen.py", _R_DIR / "codegen_stdfns.py",
                  _R_DIR / "stdlib.py", _R_DIR / "noise.py", Path(__file__).parent / "tex_fusion.py"]
# Tier-policy — a change moves a measured win/lose verdict (autotier.json / warm_state.json).
# NEW under CACHE-4: previously a compiled.py tiering change kept stale verdicts.
_VERDICT_FILES = [_R_DIR / "precision_policy.py", _R_DIR / "autotier.py",
                  _R_DIR / "compiled.py", _R_DIR / "graphed.py"]


def _hash_files(files, *extra: bytes) -> str:
    """SHA-256 (16 hex) over a set of source files (missing → hash the name) plus any extra
    byte fragments. The building block of every epoch and the mono-hash tripwire."""
    h = hashlib.sha256()
    for p in sorted(files, key=lambda x: x.name):
        try:
            h.update(p.read_bytes())
        except FileNotFoundError:
            h.update(p.name.encode())
    for e in extra:
        h.update(e)
    return h.hexdigest()[:16]


_AST_EPOCH = _hash_files(_AST_FILES)
# M-5: the out= reuse kill switch changes emitted code without touching a file, so fold it into
# the codegen epoch — else a persisted .cg emitted with reuse ON is served after it toggles OFF.
_CODEGEN_EPOCH = _hash_files(
    _CODEGEN_FILES, b"ast:" + _AST_EPOCH.encode(),
    b"cgreuse:" + os.environ.get("TEX_CODEGEN_NO_OUT_REUSE", "").encode())
_VERDICT_EPOCH = _hash_files(_VERDICT_FILES, b"cg:" + _CODEGEN_EPOCH.encode())


def ast_epoch() -> str:
    """CACHE-4: the epoch gating the compiled-program (.pkl) tier — parse/typecheck/optimize."""
    return _AST_EPOCH


def codegen_epoch() -> str:
    """CACHE-4: the epoch gating codegen .cg sidecars + the inductor dir (nests AST_EPOCH). Also
    the code-identity component of a CACHE-1 result key (tex_results.env_epoch)."""
    return _CODEGEN_EPOCH


def verdict_epoch() -> str:
    """CACHE-4: the epoch gating measured tier verdicts (autotier.json / warm_state.json)."""
    return _VERDICT_EPOCH


def epoch_partitions() -> dict:
    """The three epoch file-sets, for the CACHE-4 completeness tripwire (a test asserts they
    union to the watched set and that AST/CODEGEN are disjoint)."""
    return {"ast": list(_AST_FILES), "codegen": list(_CODEGEN_FILES),
            "verdict": list(_VERDICT_FILES)}


def _compute_compiler_hash() -> str:
    """The full mono-hash over every AST + codegen watched file. Under CACHE-4 NO artifact keys on
    it anymore (the specific epochs do; `_CACHE_VERSION` now merely aliases `_CODEGEN_EPOCH`). It
    survives only so a test can hash the watched set — the completeness tripwire itself lives in
    tests/test_v025_phase1 (it asserts the AST/CODEGEN partitions union to the watched set and are
    disjoint), not in this function."""
    return _hash_files(_AST_FILES + _CODEGEN_FILES,
                       b"cgreuse:" + os.environ.get("TEX_CODEGEN_NO_OUT_REUSE", "").encode())


# Back-compat: a few call sites / external hosts still import _CACHE_VERSION. It now aliases the
# codegen epoch (the broadest program-identity code hash); artifact keys use the specific epochs.
_CACHE_VERSION = _CODEGEN_EPOCH

# Limits
_MEMORY_MAX_ENTRIES = 128
_CODEGEN_MEMORY_MAX_ENTRIES = 128  # in-memory codegen tier (disk sidecar backs evictions)
_DISK_MAX_ENTRIES = 512
# CACHE-0: store_codegen_fn writes a .cg sidecar for ANY fingerprint — paired with
# a .pkl or not (the fused/codegen tiers and the bench harness mint .cg without a
# sibling .pkl). The .pkl-only census below never reclaims those, so orphan .cg
# accumulate without bound. Evict orphan .cg (no sibling .pkl) oldest-first past
# this cap, with a grace so a .cg legitimately preceding its .pkl within a session
# is never nuked.
_CG_DISK_MAX_ENTRIES = 1024
_CG_ORPHAN_GRACE_SEC = 600
_CG_CENSUS_INTERVAL_SEC = 300  # throttle: at most one orphan-.cg glob per 5 min

# Cache directory lives alongside tex_cache.py (inside the TEX_Wrangle package)
_CACHE_DIR_NAME = ".tex_cache"
_TORCH_COMPILE_CACHE_SUBDIR = "torch_compile"

# PC-3: persisted generated-code (marshal) sidecar. CPython bytecode magic
# invalidates blobs across interpreter versions (marshal isn't portable).
_BYTECODE_MAGIC = importlib.util.MAGIC_NUMBER
# Sentinel for "codegen tried and this program is unsupported" — persisted so a
# restart doesn't re-attempt emission every time.
_CG_UNSUPPORTED = object()

# Memoizes computed fingerprints per (code, sorted binding-types tuple) so the
# SHA256 over the full source runs once per unique program, not on every probe.
_FINGERPRINT_MEMO: dict[tuple, str] = {}
_FINGERPRINT_MEMO_MAX = 256


class TEXCache:
    """
    Two-tier compilation cache for TEX programs.

    Memory tier: OrderedDict with LRU eviction (max 128 entries).
    Disk tier:   Pickle files in .tex_cache/ (max 512 entries, LRU by atime).

    Usage:
        cache = get_cache()
        program, type_map, refs, assigned, params, builtins = cache.compile_tex(code, binding_types)
    """

    def __init__(self, cache_dir: Path | None = None):
        self._memory: OrderedDict[str, tuple] = OrderedDict()

        if cache_dir is None:
            # CACHE-0: a TEX_CACHE_DIR env override lets a test/bench harness (or a
            # pip-installed / read-only deployment) point the cache at a scratch dir
            # so harness artifacts never land in the shipping package cache. (This is
            # also the first rung of ENG-11's eventual resolution order.) Unset →
            # the package-local .tex_cache, exactly as before.
            env_dir = os.environ.get("TEX_CACHE_DIR")
            if env_dir:
                cache_dir = Path(env_dir)
            else:
                pkg_dir = Path(__file__).parent
                cache_dir = pkg_dir / _CACHE_DIR_NAME
        self._cache_dir = cache_dir
        self._torch_compile_cache_dir = cache_dir / _TORCH_COMPILE_CACHE_SUBDIR
        # PC-3: fingerprint -> materialized codegen fn (or _CG_UNSUPPORTED).
        # LRU-bounded (was an unbounded dict — a long session with many distinct
        # programs grew it without limit; the disk sidecar backs evicted entries).
        self._codegen_memory: OrderedDict[str, Any] = OrderedDict()
        self._last_cg_census = 0.0  # CACHE-0: throttle timestamp for the orphan sweep

    # ── Public API ────────────────────────────────────────────────────

    @property
    def torch_compile_cache_dir(self) -> Path:
        """Directory for torch.compile / inductor cache artifacts."""
        return self._torch_compile_cache_dir

    @staticmethod
    def fingerprint(code: str, binding_types: dict[str, TEXType]) -> str:
        """Compute cache key from code + binding types (SHA256).

        Uses a length-prefixed/structured encoding so arbitrary user code
        (which may contain '|' or ':') can never collide with the binding-type
        descriptors. Memoized per (code, sorted binding-types) so the SHA256
        over the full source is computed once per unique program rather than on
        every cache probe.
        """
        binding_key = tuple(sorted((k, v.value) for k, v in binding_types.items()))
        memo = _FINGERPRINT_MEMO
        cache_key = (code, binding_key)
        cached = memo.get(cache_key)
        if cached is not None:
            return cached

        h = hashlib.sha256()
        code_bytes = code.encode()
        h.update(len(code_bytes).to_bytes(8, "little"))
        h.update(code_bytes)
        h.update(json.dumps(binding_key).encode())
        fp = h.hexdigest()

        if len(memo) >= _FINGERPRINT_MEMO_MAX:
            memo.clear()
        memo[cache_key] = fp
        return fp

    def get(self, code: str, binding_types: dict[str, TEXType]) -> tuple | None:
        """
        Look up a cached compilation result.

        Returns (program, type_map, referenced_bindings,
                 assigned_bindings, param_declarations, used_builtins) or None.
        Checks memory first, then disk.
        """
        fp = self.fingerprint(code, binding_types)

        # Tier 1: memory
        if fp in self._memory:
            self._memory.move_to_end(fp)
            return self._memory[fp]

        # Tier 2: disk
        result = self._load_from_disk(fp, binding_types)
        if result is not None:
            self._memory_put(fp, result)
            return result

        return None

    def put(
        self,
        code: str,
        binding_types: dict[str, TEXType],
        program: Any,
        type_map: dict,
        referenced_bindings: set[str],
        assigned_bindings: dict[str, TEXType] | None = None,
        param_declarations: dict[str, dict] | None = None,
        used_builtins: frozenset[str] | None = None,
    ):
        """Store a compilation result in both memory and disk caches."""
        fp = self.fingerprint(code, binding_types)
        result = (program, type_map, referenced_bindings,
                  assigned_bindings or {}, param_declarations or {},
                  used_builtins or frozenset())
        self._memory_put(fp, result)
        self._save_to_disk(fp, program, binding_types)

    def compile_tex(
        self, code: str, binding_types: dict[str, TEXType]
    ) -> tuple:
        """
        Compile TEX source: lex -> parse -> type-check, with caching.

        Returns (program_ast, type_map, referenced_bindings,
                 assigned_bindings, param_declarations, used_builtins).

        assigned_bindings: dict mapping output binding names to their inferred TEXType.
        param_declarations: dict mapping parameter names to {type, type_hint}.
        used_builtins: frozenset of builtin names referenced by the program.
        Raises LexerError, ParseError, or TypeCheckError on invalid code.
        """
        cached = self.get(code, binding_types)
        if cached is not None:
            return cached

        # Full compilation pipeline: lex + parse, then the shared post-parse
        # orchestration (STR-8: identical to the fusion path's).
        tokens = Lexer(code).tokenize()
        program = Parser(tokens, source=code).parse()
        program, type_map, referenced, assigned, params, used_builtins = \
            self.compile_ast(program, binding_types, source=code)

        self.put(code, binding_types, program, type_map,
                 referenced, assigned, params, used_builtins)
        return (program, type_map, referenced, assigned, params, used_builtins)

    def compile_ast(self, program, binding_types, *, source: str):
        """STR-8: the shared post-parse compile pipeline (type-check → optimize →
        re-type-check on the optimized AST → collect builtins), used by BOTH the
        normal path (`compile_tex`, from source) and fusion (`compile_fused`, from a
        spliced AST). Returns the same 6-tuple as `compile_tex`. Error-agnostic: a
        `TypeCheckError` propagates so each caller translates it as it sees fit
        (fusion wraps it as `FusionError`).
        """
        checker = TypeChecker(binding_types=binding_types, source=source)
        type_map = checker.check(program)
        # Pass type_map so optimizer-created nodes (CSE/LICM temps + their
        # references) are registered, keeping the AST type-consistent during
        # optimization (id()-keyed lookups would otherwise miss them).
        program = optimize(program, type_map)
        # Re-run the type checker on the OPTIMIZED AST to rebuild a complete,
        # correct type_map. Optimization passes synthesize brand-new nodes
        # (const-folded BinOps, CSE/LICM temps, unrolled-loop bodies) that the
        # original id()-keyed type_map can never cover; a stale lookup would
        # mistype them (e.g. a vec arg counted as one scalar component, crashing
        # vec constructors). This mirrors what the disk-cache reload path already
        # does. strict_redeclare=False: unrolling flattens N copies of a local-
        # declaring loop body into one scope; the strict check above already
        # validated the user's original code, so tolerate benign redeclarations.
        type_map = TypeChecker(binding_types=binding_types, source=source,
                               strict_redeclare=False).check(program)
        # Pre-compute builtin identifiers (avoids AST walk on every execution)
        used_builtins = _collect_identifiers(program)
        return (program, type_map, checker.referenced_bindings,
                checker.assigned_bindings, checker.param_declarations, used_builtins)

    def clear_memory(self):
        """Clear in-memory cache only (disk entries remain)."""
        self._memory.clear()

    def clear_all(self):
        """Clear memory, disk, and torch.compile/inductor caches."""
        self._memory.clear()
        self._codegen_memory.clear()
        try:
            # *.tmp also sweeps autotier.json.tmp; the persisted-verdict/model JSONs
            # (autotier.json = CC-2 tier verdicts, xfer.json = ENG-8 transfer model)
            # are named explicitly.
            for pat in ("*.pkl", "*.cg", "*.tmp", "autotier.json", "xfer.json"):
                for p in self._cache_dir.glob(pat):
                    p.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("[TEX] Disk cache clear failed: %s", e)
        # Inductor artifacts (~30-60 MB/program) live under torch_compile/ now
        # that the cache dir is deliberately owned (PC-1). Remove the tree.
        try:
            if self._torch_compile_cache_dir.exists():
                shutil.rmtree(self._torch_compile_cache_dir, ignore_errors=True)
        except Exception as e:
            logger.warning("[TEX] torch.compile cache clear failed: %s", e)

    # ── Internal: memory tier ─────────────────────────────────────────

    def _memory_put(self, fp: str, result: tuple):
        """Insert into memory cache with LRU eviction."""
        self._memory[fp] = result
        self._memory.move_to_end(fp)
        while len(self._memory) > _MEMORY_MAX_ENTRIES:
            self._memory.popitem(last=False)

    # ── Internal: disk tier ───────────────────────────────────────────

    def _disk_path(self, fp: str) -> Path:
        return self._cache_dir / f"{fp}.pkl"

    @staticmethod
    def _atomic_pickle(path: Path, data: Any) -> None:
        """Pickle *data* to *path* atomically (temp file + os.replace) so a
        concurrent reader — e.g. a second ComfyUI instance sharing the dir —
        can never observe a half-written entry."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)  # atomic on same-volume NTFS

    def _save_to_disk(self, fp: str, program: Any, binding_types: dict[str, TEXType]):
        """Persist compilation artifacts to disk."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "version": _AST_EPOCH,          # CACHE-4: .pkl gated by the AST epoch only
                "program": program,
                "binding_types": {k: v.value for k, v in binding_types.items()},
                "timestamp": time.time(),
            }
            self._atomic_pickle(self._disk_path(fp), data)
            self._evict_disk_if_needed()
        except Exception as e:
            logger.warning("[TEX] Disk cache write failed: %s", e)

    def _load_from_disk(
        self, fp: str, binding_types: dict[str, TEXType]
    ) -> tuple | None:
        """Load from disk and re-run type checker to get valid type_map."""
        path = self._disk_path(fp)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            # Version check — stale entries are deleted (CACHE-4: AST epoch gates the .pkl)
            if data.get("version") != _AST_EPOCH:
                path.unlink(missing_ok=True)
                return None

            program = data["program"]

            # Re-run type checker to regenerate type_map with valid id() keys.
            # The stored program is already optimized, so use lenient redeclare
            # (unrolled loop bodies legitimately redeclare locals) — matching the
            # in-memory compile path's post-optimization re-check.
            checker = TypeChecker(binding_types=binding_types, source="",
                                  strict_redeclare=False)
            type_map = checker.check(program)

            # Touch file to update access time for LRU eviction
            os.utime(path, None)

            return (program, type_map, checker.referenced_bindings,
                    checker.assigned_bindings, checker.param_declarations,
                    _collect_identifiers(program))
        except Exception as e:
            logger.warning("[TEX] Disk cache load failed for %s…: %s", fp[:12], e)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def _evict_disk_if_needed(self):
        """Remove oldest disk entries if over the limit."""
        # CACHE-0: the orphan-.cg sweep runs FIRST, independent of the .pkl cap —
        # if it sat after the early return below it would almost never fire (a
        # session rarely exceeds 512 .pkl, but mints .cg freely).
        self._evict_orphan_cg()
        try:
            entries = list(self._cache_dir.glob("*.pkl"))
            if len(entries) <= _DISK_MAX_ENTRIES:
                return
            # Sort by access time (oldest first)
            entries.sort(key=lambda p: p.stat().st_atime)
            to_remove = len(entries) - _DISK_MAX_ENTRIES
            for p in entries[:to_remove]:
                p.unlink(missing_ok=True)
                p.with_suffix(".pkl.tmp").unlink(missing_ok=True)
                # Drop the paired codegen sidecar (and any orphan .tmp).
                p.with_suffix(".cg").unlink(missing_ok=True)
                p.with_suffix(".cg.tmp").unlink(missing_ok=True)
        except Exception as e:
            logger.warning("[TEX] Disk cache eviction failed: %s", e)

    def _evict_orphan_cg(self):
        """CACHE-0: reclaim orphan .cg sidecars (no sibling .pkl) oldest-first when
        their count exceeds _CG_DISK_MAX_ENTRIES. The .pkl census only drops .cg it
        PAIRS with, so codegen/fused/bench-harness .cg written without a matching
        .pkl leak forever otherwise. Grace-gated so a fresh .cg still awaiting its
        .pkl within the session is never removed (the loader regenerates on a miss,
        so even an over-eager delete is only a recompile, never wrong output)."""
        now = time.time()
        # Throttle: the census does a full glob (+ a stat-storm once over cap), and it
        # runs from every disk save AND every .cg write. Once-per-interval keeps it off
        # the hot path while still reclaiming within a session.
        if now - self._last_cg_census < _CG_CENSUS_INTERVAL_SEC:
            return
        self._last_cg_census = now
        try:
            cgs = list(self._cache_dir.glob("*.cg"))
            over = len(cgs) - _CG_DISK_MAX_ENTRIES
            if over <= 0:
                return
            # (mtime, path) for orphans past the grace window — one stat() per file.
            aged = []
            for p in cgs:
                try:
                    if p.with_suffix(".pkl").exists():
                        continue
                    mt = p.stat().st_mtime
                except OSError:
                    continue  # concurrent unlink / vanished — skip, don't abort census
                if now - mt > _CG_ORPHAN_GRACE_SEC:
                    aged.append((mt, p))
            aged.sort()  # oldest first
            for _mt, p in aged[:over]:
                try:
                    p.unlink(missing_ok=True)
                    p.with_suffix(".cg.tmp").unlink(missing_ok=True)
                except OSError:
                    pass  # a concurrent reader/unlink on one file never aborts the rest
        except Exception as e:
            logger.warning("[TEX] Orphan .cg census failed: %s", e)

    # ── PC-3: generated-code (marshal) persistence ────────────────────

    def _cg_path(self, fp: str) -> Path:
        return self._cache_dir / f"{fp}.cg"

    def _codegen_memory_put(self, fp: str, val) -> None:
        """Insert into the in-memory codegen tier as MRU, then LRU-evict to the
        bound (disk sidecar backs evictions). Mirrors the `_memory` tier's put."""
        self._codegen_memory[fp] = val
        self._codegen_memory.move_to_end(fp)
        while len(self._codegen_memory) > _CODEGEN_MEMORY_MAX_ENTRIES:
            self._codegen_memory.popitem(last=False)

    def get_codegen_fn(self, fp: str):
        """Return the codegen fn for *fp*: a callable, the _CG_UNSUPPORTED
        sentinel, or None (not yet generated — the caller should emit and then
        call store_codegen_fn). Memory tier first, then the marshal sidecar."""
        cached = self._codegen_memory.get(fp)
        if cached is not None:
            self._codegen_memory.move_to_end(fp)
            return cached
        fn = self._load_codegen_from_disk(fp)
        if fn is not None:
            self._codegen_memory_put(fp, fn)
        return fn

    def store_codegen_fn(self, fp: str, fn) -> None:
        """Record a freshly generated codegen fn (callable) or None (unsupported)
        in memory and persist it to a marshal sidecar."""
        if fn is None:
            self._codegen_memory_put(fp, _CG_UNSUPPORTED)
            self._persist_codegen(fp, unsupported=True)
            return
        self._codegen_memory_put(fp, fn)
        code = getattr(fn, "_tex_code", None)
        src = getattr(fn, "_tex_src", None)
        if code is None or src is None:
            return  # nothing to persist (fn not from build())
        try:
            blob = marshal.dumps(code)
        except Exception:
            return
        self._persist_codegen(
            fp, blob=blob, sha=hashlib.sha256(blob).hexdigest(),
            src=src, has_fn_calls=bool(getattr(fn, "_has_fn_calls", False)))

    def _persist_codegen(self, fp: str, *, blob: bytes | None = None,
                         sha: str = "", src: str = "",
                         has_fn_calls: bool = False,
                         unsupported: bool = False) -> None:
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            data = {
                "version": _CODEGEN_EPOCH,      # CACHE-4: .cg gated by the codegen epoch
                "magic": _BYTECODE_MAGIC,
                "unsupported": unsupported,
                "blob": blob,
                "sha": sha,
                "src": src,
                "has_fn_calls": has_fn_calls,
            }
            self._atomic_pickle(self._cg_path(fp), data)
            # CACHE-0: also census here (the .cg-WRITE site), throttled — so a
            # codegen-heavy but .pkl-quiet session still reclaims orphans, not only
            # sessions that write new .pkl (which is what triggers _evict_disk_if_needed).
            self._evict_orphan_cg()
        except Exception:
            # Persistence is best-effort — a concurrent reader (second ComfyUI
            # instance) can PermissionError; codegen simply regenerates next time.
            pass

    def _load_codegen_from_disk(self, fp: str):
        path = self._cg_path(fp)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if (data.get("version") != _CODEGEN_EPOCH
                    or data.get("magic") != _BYTECODE_MAGIC):
                path.unlink(missing_ok=True)
                return None
            if data.get("unsupported"):
                return _CG_UNSUPPORTED
            blob = data.get("blob")
            if not blob or hashlib.sha256(blob).hexdigest() != data.get("sha"):
                path.unlink(missing_ok=True)  # corrupt — never marshal.loads it
                return None
            from .tex_runtime.codegen_persist import materialize_codegen
            return materialize_codegen(blob, data.get("src", ""),
                                       data.get("has_fn_calls", False), fp)
        except Exception as e:
            logger.warning("[TEX] Codegen sidecar load failed for %s…: %s", fp[:12], e)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None


# ── Module-level singleton ────────────────────────────────────────────

_cache_instance: TEXCache | None = None


def get_cache() -> TEXCache:
    """Get or create the global TEXCache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = TEXCache()
    return _cache_instance
