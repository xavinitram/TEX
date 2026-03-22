"""
TEX Cache — two-tier compilation cache for TEX programs.

Tier 1 (memory): OrderedDict with LRU eviction. Stores ready-to-execute
(program, type_map, referenced_bindings, assigned_bindings, param_declarations) tuples.

Tier 2 (disk): Pickle files in .tex_cache/. Stores (program_ast,
binding_types, version). On load, re-runs type checker to regenerate
type_map with valid id() keys (~0.1ms, negligible).
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
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

# Disk cache format version — tracks AST/type-checker compatibility, NOT the
# release version. Bump this when AST node structure or type checker changes
# would make existing .pkl cache files invalid (causes graceful cache miss).
_CACHE_VERSION = "2.4.0"  # v0.7: CSE pass, static for-loop range, reusable interpreter

# Limits
_MEMORY_MAX_ENTRIES = 128
_DISK_MAX_ENTRIES = 512

# Cache directory lives alongside tex_cache.py (inside the TEX_Wrangle package)
_CACHE_DIR_NAME = ".tex_cache"
_TORCH_COMPILE_CACHE_SUBDIR = "torch_compile"


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

    # ── Public API ────────────────────────────────────────────────────

    @property
    def torch_compile_cache_dir(self) -> Path:
        """Directory for torch.compile / inductor cache artifacts."""
        return self._torch_compile_cache_dir

    @staticmethod
    def fingerprint(code: str, binding_types: dict[str, TEXType]) -> str:
        """Compute cache key from code + binding types (SHA256)."""
        key_parts = [code] + [
            f"{k}:{v.value}" for k, v in sorted(binding_types.items())
        ]
        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()

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
        program = Parser(tokens).parse()
        checker = TypeChecker(binding_types=binding_types)
        type_map = checker.check(program)
        program = optimize(program)

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
        """Clear both memory and disk caches."""
        self._memory.clear()
        try:
            for p in self._cache_dir.glob("*.pkl"):
                p.unlink(missing_ok=True)
        except Exception:
            pass

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

    def _save_to_disk(self, fp: str, program: Any, binding_types: dict[str, TEXType]):
        """Persist compilation artifacts to disk."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._disk_path(fp)
            data = {
                "version": _CACHE_VERSION,
                "program": program,
                "binding_types": {k: v.value for k, v in binding_types.items()},
                "timestamp": time.time(),
            }
            with open(path, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
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

            # Re-run type checker to regenerate type_map with valid id() keys
            checker = TypeChecker(binding_types=binding_types)
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
        except Exception as e:
            logger.warning("[TEX] Disk cache eviction failed: %s", e)


# ── Module-level singleton ────────────────────────────────────────────

_cache_instance: TEXCache | None = None


def get_cache() -> TEXCache:
    """Get or create the global TEXCache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = TEXCache()
    return _cache_instance
