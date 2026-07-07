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
from .tex_compiler.type_checker import TypeChecker, TEXType
from .tex_compiler.optimizer import optimize
from .tex_runtime.interpreter import _collect_identifiers

logger = logging.getLogger("TEX")


def _compute_compiler_hash() -> str:
    """Hash all compiler/runtime source files to auto-invalidate disk cache on code changes."""
    pkg_dir = Path(__file__).parent
    source_files = sorted([
        pkg_dir / "tex_compiler" / "ast_nodes.py",
        pkg_dir / "tex_compiler" / "lexer.py",
        pkg_dir / "tex_compiler" / "parser.py",
        pkg_dir / "tex_compiler" / "type_checker.py",
        pkg_dir / "tex_compiler" / "optimizer.py",
        pkg_dir / "tex_compiler" / "stdlib_signatures.py",
        pkg_dir / "tex_runtime" / "interpreter.py",
        pkg_dir / "tex_runtime" / "codegen.py",
        pkg_dir / "tex_runtime" / "stdlib.py",
        pkg_dir / "tex_runtime" / "noise.py",
        # CT-1: fused programs are now persisted, so a splicer change must
        # invalidate their disk entries too.
        pkg_dir / "tex_fusion.py",
    ])
    h = hashlib.sha256()
    for p in source_files:
        try:
            h.update(p.read_bytes())
        except FileNotFoundError:
            h.update(p.name.encode())
    # M-5: the out= reuse kill switch changes generated code without changing any
    # source file, so fold it in — else a persisted .cg blob emitted with reuse ON
    # is served after the flag is toggled OFF (and vice-versa).
    h.update(b"cgreuse:" + os.environ.get("TEX_CODEGEN_NO_OUT_REUSE", "").encode())
    return h.hexdigest()[:16]


# Auto-computed from compiler/runtime source files — any code change invalidates disk cache.
_CACHE_VERSION = _compute_compiler_hash()

# Limits
_MEMORY_MAX_ENTRIES = 128
_CODEGEN_MEMORY_MAX_ENTRIES = 128  # in-memory codegen tier (disk sidecar backs evictions)
_DISK_MAX_ENTRIES = 512

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
            pkg_dir = Path(__file__).parent
            cache_dir = pkg_dir / _CACHE_DIR_NAME
        self._cache_dir = cache_dir
        self._torch_compile_cache_dir = cache_dir / _TORCH_COMPILE_CACHE_SUBDIR
        # PC-3: fingerprint -> materialized codegen fn (or _CG_UNSUPPORTED).
        # LRU-bounded (was an unbounded dict — a long session with many distinct
        # programs grew it without limit; the disk sidecar backs evicted entries).
        self._codegen_memory: OrderedDict[str, Any] = OrderedDict()

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

        # Full compilation pipeline
        tokens = Lexer(code).tokenize()
        program = Parser(tokens, source=code).parse()
        checker = TypeChecker(binding_types=binding_types, source=code)
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
        # vec constructors). This mirrors what the disk-cache reload path
        # already does — both paths now share identical, correct semantics.
        # strict_redeclare=False: unrolling flattens N copies of a local-declaring
        # loop body into one scope; the strict check above already validated the
        # user's original code, so tolerate those benign redeclarations here.
        type_map = TypeChecker(binding_types=binding_types, source=code,
                               strict_redeclare=False).check(program)

        # Pre-compute builtin identifiers (avoids AST walk on every execution)
        used_builtins = _collect_identifiers(program)

        self.put(code, binding_types, program, type_map,
                 checker.referenced_bindings, checker.assigned_bindings,
                 checker.param_declarations, used_builtins)
        return (program, type_map, checker.referenced_bindings,
                checker.assigned_bindings, checker.param_declarations,
                used_builtins)

    def clear_memory(self):
        """Clear in-memory cache only (disk entries remain)."""
        self._memory.clear()

    def clear_all(self):
        """Clear memory, disk, and torch.compile/inductor caches."""
        self._memory.clear()
        self._codegen_memory.clear()
        try:
            # *.tmp also sweeps autotier.json.tmp; autotier.json itself (CC-2
            # measured-tier verdicts) is named explicitly.
            for pat in ("*.pkl", "*.cg", "*.tmp", "autotier.json"):
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
                "version": _CACHE_VERSION,
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

            # Version check — stale entries are deleted
            if data.get("version") != _CACHE_VERSION:
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
                "version": _CACHE_VERSION,
                "magic": _BYTECODE_MAGIC,
                "unsupported": unsupported,
                "blob": blob,
                "sha": sha,
                "src": src,
                "has_fn_calls": has_fn_calls,
            }
            self._atomic_pickle(self._cg_path(fp), data)
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
            if (data.get("version") != _CACHE_VERSION
                    or data.get("magic") != _BYTECODE_MAGIC):
                path.unlink(missing_ok=True)
                return None
            if data.get("unsupported"):
                return _CG_UNSUPPORTED
            blob = data.get("blob")
            if not blob or hashlib.sha256(blob).hexdigest() != data.get("sha"):
                path.unlink(missing_ok=True)  # corrupt — never marshal.loads it
                return None
            from .tex_runtime.codegen import materialize_codegen
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
