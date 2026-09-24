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
import shutil
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .tex_compiler.lexer import Lexer, claim_tokens
from .tex_compiler.parser import Parser
from .tex_compiler.ast_nodes import (BindingRef, ChannelAccess, NodeTransformer, SourceLoc)
from .tex_compiler.type_checker import TypeChecker, TypeCheckError, BINDING_HINT_TYPES
from .tex_compiler.types import TEXType, planes_wires_enabled
from .tex_compiler.optimizer import optimize
from .tex_runtime.interpreter import _collect_identifiers, _collect_identifiers_and_calls

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
# LANG-L5: `masked_flow.py` and `codegen_masked.py` are here for the same reason the two
# above them are — they are the language-0.25 halves of the interpreter's semantics and of
# the emitter, so an edit to either changes what a flagged program computes and must not
# leave a stale `.cg` behind. They are added now, while no program can reach them, rather
# than at the release that makes them reachable.
# LIB-1: `stdlib.py`'s per-domain split. The facade still carries the module-level caches
# and helpers (now re-exported from `stdlib_core.py`), and every `fn_*` impl moved onto one
# of the seven domain leaves — a change to any of them alters emitted code or interpreter
# semantics exactly as a `stdlib.py` edit used to, so each leaf is watched here too.
_CODEGEN_FILES = [_R_DIR / "interpreter.py", _R_DIR / "codegen.py", _R_DIR / "codegen_stdfns.py",
                  _R_DIR / "stdlib.py", _R_DIR / "stdlib_core.py", _R_DIR / "stdlib_math.py",
                  _R_DIR / "stdlib_color.py", _R_DIR / "stdlib_sample.py", _R_DIR / "stdlib_noise.py",
                  _R_DIR / "stdlib_sdf.py", _R_DIR / "stdlib_string.py", _R_DIR / "stdlib_array.py",
                  _R_DIR / "noise.py", Path(__file__).parent / "tex_fusion.py",
                  _R_DIR / "masked_flow.py", _R_DIR / "codegen_masked.py",
                  # CG-1: the STR-7 split's other two emitters. `codegen_stencil.py` owns the whole
                  # stencil detection-and-lowering route, so an edit there changes emitted code.
                  _R_DIR / "codegen_stencil.py",
                  # `codegen_persist.py` writes and authenticates the `.cg` bytes themselves;
                  # watched defensively so a format change can never be read by the old reader.
                  _R_DIR / "codegen_persist.py"]
# Tier-policy — a change moves a measured win/lose verdict (autotier.json / warm_state.json).
# NEW under CACHE-4: previously a compiled.py tiering change kept stale verdicts.
_VERDICT_FILES = [_R_DIR / "precision_policy.py", _R_DIR / "autotier.py",
                  _R_DIR / "compiled.py", _R_DIR / "graphed.py"]


# ── DATA-6: the plane seam ────────────────────────────────────────────────────
#
# The lexer reads `@name.seg` as ONE binding token (one dotted segment, verbatim) so a plane is
# addressed by the name its file gives it and `sigil_names` reports per-plane demand. The
# lexer never sees binding types, so it cannot tell `@beauty.diffuse` (a plane read on a PLANES
# wire) from `@A.rgb` (a swizzle of an ordinary wire) — the pass below does, from the binding
# types, BEFORE the first TypeChecker runs. It lives here and not in the checker because
# `_check_binding_ref` returns a type and cannot replace its own node, and `compile_ast` is the
# one post-parse pipeline both production entries (`compile_tex`, `compile_fused`) share.
#
# It is legal for a resolution to depend on binding types because the compile cache is keyed on
# `(code, binding_types)` (`TEXCache.fingerprint`): a differently-typed compile of the same
# source can never alias this one.
class _DottedBindingSplitback(NodeTransformer):
    """The swizzle-splitback rule, one row per binding-type case (`docs/plane-bindings.md` §1.1).

    For every wire `BindingRef` whose name carries a dot, split on the LAST dot into `base` and
    `seg`, then:

      * the dotted name is itself declared in `binding_types` — a plane the host has already
        expanded (`beauty.diffuse: VEC3`) — or `base` is typed PLANES (by the map or by the
        node's own `p@` hint), AND plane wires are enabled: a PLANE READ. Keep the dotted
        `BindingRef`; the checker types it from the per-plane row.
      * `base` is a vector type: SPLIT BACK to `ChannelAccess(BindingRef(base), seg)` — the
        exact AST the parser built before planes existed. `seg` is then validated by the
        existing swizzle rules (E3301), unchanged.
      * `base` is a non-vector type (FLOAT / INT / MASK / STRING / ARRAY …): split back the
        same way; the existing rules already own what `.r` means on a channel-less value.
      * `base` is ABSENT (an untyped base): split back. This is the compat guarantee in one
        line — an untyped base is a swizzle, so no program that compiled before planes can be
        re-read as a plane access.

    While plane wires are disabled (the ComfyUI default) the first row never fires, so every
    dotted `@` is a swizzle — exactly what it always meant — and PLANES is inert end to end.

    A dotted non-plane binding in a `@X[..]` / `@X(..)` slot is refused with the parse errors
    those spellings drew before (E2000 / E2002): the parser guarantees that slot holds a bare
    `BindingRef`, and `@A.rgb[ix, iy]` was never a program.

    The split node carries the SEGMENT's source location, as the parser's `ChannelAccess` did,
    so an unknown-channel diagnostic lands on the same column it always has.
    """

    def __init__(self, binding_types: dict, source: str):
        self._bt = binding_types or {}
        self._source = source
        self._planes_on = planes_wires_enabled()

    def _is_plane_read(self, node: BindingRef) -> bool:
        if not self._planes_on:
            return False
        bt = self._bt
        if node.name in bt:                       # an already-expanded plane row
            return True
        t = bt.get(node.name.rsplit(".", 1)[0])
        if t is None and node.type_hint:
            t = BINDING_HINT_TYPES.get(node.type_hint)
        return t is TEXType.PLANES

    @staticmethod
    def _is_dotted_wire(node) -> bool:
        return node.__class__ is BindingRef and node.kind == "wire" and "." in node.name

    def _split(self, node: BindingRef) -> ChannelAccess:
        base, seg = node.name.rsplit(".", 1)
        loc = node.loc
        seg_loc = loc
        off = loc._offset
        if off is not None and off >= 0:
            # `[prefix]@base.seg` — the segment starts after the sigil, the base and the dot.
            seg_loc = SourceLoc.from_offset(off + len(node.type_hint) + 1 + len(base) + 1,
                                            loc._source, stage=loc.stage)
        return ChannelAccess(loc=seg_loc, channels=seg,
                             object=BindingRef(loc=loc, name=base, kind=node.kind,
                                               type_hint=node.type_hint))

    def visit_BindingRef(self, node):
        if not self._is_dotted_wire(node) or self._is_plane_read(node):
            return node
        return self._split(node)

    def _refuse_swizzle_sugar(self, node, *, code: str, message: str, hint: str):
        b = node.binding
        if self._is_dotted_wire(b) and not self._is_plane_read(b):
            raise TypeCheckError(message, node.loc, source=self._source, code=code, hint=hint)

    def visit_BindingIndexAccess(self, node):
        self._refuse_swizzle_sugar(
            node, code="E2000",
            message="A swizzle can't be indexed like a binding.",
            hint="`@A.rgb` is a swizzle of @A — fetch first, then swizzle: `@A[ix, iy].rgb`.")
        return self.generic_visit(node)

    def visit_BindingSampleAccess(self, node):
        self._refuse_swizzle_sugar(
            node, code="E2002",
            message="This value can't be called like a function.",
            hint="Only function names and @bindings can be followed by `(...)`. `@A.rgb` is a "
                 "swizzle of @A — sample first, then swizzle: `@A(u, v).rgb`.")
        return self.generic_visit(node)


def splitback_dotted_bindings(program, binding_types: dict, *, source: str = ""):
    """DATA-6: resolve every dotted `@name.seg` in `program` — a plane read stays a dotted
    `BindingRef`, everything else is put back to the `ChannelAccess` swizzle it always was.
    In place; returns `program`. Runs in `compile_ast` ahead of the first `TypeChecker`, and is
    THE SEAM the plane-expansion step shares: expansion rewrites `binding_types` (per-plane rows
    for the planes the source mentions) immediately before this call, and this pass reads the
    expanded map. Rules: `_DottedBindingSplitback`."""
    return _DottedBindingSplitback(binding_types, source).visit(program)


def parse_and_split(source: str, binding_types: dict | None = None):
    """DATA-6: THE front end — `source` -> `Program` with every dotted binding resolved.

    Lexes (the lexer reads `@name.seg` as ONE binding token — the language rule since planes),
    parses, and runs `splitback_dotted_bindings` against `binding_types`, so the returned AST
    names a swizzle's BASE wire exactly as the parser did before planes existed. A caller that
    has no binding types passes `{}` (or nothing): the untyped-base row then splits every
    dotted binding back, which is the pre-planes AST for every program written before planes.

    ONE OWNER, deliberately. Every consumer that reads a binding's name as the wire it is
    connected to — the production seam (`TEXCache.compile_tex`), the editor lint
    (`tex_api.check`), the fused-chain splicer (`tex_fusion`), the ROI walk (`tex_roi`), the
    lazy-input analysis (`tex_lazy`), and the test harnesses — parses through THIS function.
    A private `Lexer(...)` / `Parser(...)` pair in a consumer would read `image.r` where the
    cook reads `image`, and the drift would be silent on every program without a dot; the
    front-end parity test (`tests/test_v037_frontend_parity.py`) pins that they all agree.

    Raises `LexerError` / `ParseError` (or `TEXMultiError`) from the phase that found the
    problem, and `TypeCheckError` (E2000 / E2002) for the swizzle sugar the splitback refuses.
    """
    # The flag is spelled even though it is the lexer's default: this is the one place the
    # greed is REQUIRED (the splitback below is what makes it safe), and it keeps the seam
    # independent of the default should a caller ever want the raw pre-planes stream.
    # PERF-5: `TEXCache.fingerprint` asks `tex_marshalling.param_only_names` which names the
    # source uses with which sigil, and that scan lexes — immediately before this one, with the
    # same flag, over the same characters. It OFFERS its stream; claiming it is what makes a
    # never-seen program cost one lex instead of two. The claim consumes the offer, so no two
    # parses ever share a token (and so no two ASTs ever share a `SourceLoc`); a miss — no
    # offer, an evicted one, a second parse of the same source — lexes here exactly as before.
    tokens = claim_tokens(source, dotted_bindings=True)
    if tokens is None:
        tokens = Lexer(source, dotted_bindings=True).tokenize()
    program = Parser(tokens, source=source).parse()
    return splitback_dotted_bindings(program, binding_types or {}, source=source)


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


# LANG-L7 (L6-F1): `tex_api.py` — the file holding `LANGUAGE_VERSION` — is not, and should not
# become, a member of `_AST_FILES`: it is the language-satellite bump's OWN file, not a
# parser/typechecker/optimizer file, and adding it would fold in every unrelated tex_api.py
# edit (a docstring, a new host-facing helper) as a spurious AST-epoch bump. `Program.language`
# IS parser-set output, though, and a `//!tex 0.25` program's masked-vs-unmasked reading depends
# on `LANGUAGE_VERSION` at COOK time, not at compile time of the untouched AST/codegen files —
# so without this line, a `.cg` sidecar for a `//!tex 0.25` program compiled while the engine
# was still below 0.25 would be served, UNMASKED, after the bump, under the SAME fingerprint and
# the SAME (unmoved) epoch: a silently wrong picture through the cache, not the language gate
# (`docs/masked-control-flow.md` §4's failure class, arriving a different way). Folding the
# version STRING directly into the AST epoch's hash input — the same "extra byte fragment"
# mechanism `_CODEGEN_EPOCH` already uses for `TEX_CODEGEN_NO_OUT_REUSE` below — moves every
# epoch (AST ⊑ CODEGEN ⊑ VERDICT all nest it) exactly when, and only when, the version moves;
# every cache tier goes cold once on adoption, the same one-time cost any `codegen.py` edit
# already causes on a release.
from .tex_api import LANGUAGE_VERSION as _LANGUAGE_VERSION_AT_IMPORT  # noqa: E402
# REG-1e: `Program.non_spatial_calls` (set by `compile_ast` below) is baked into the SAME
# .pkl this epoch gates, and its correctness depends on WHICH builtin names
# `stdlib_registry.non_spatial_args_by_name()` currently reports -- a fact declared on
# `@stdlib(...)` decorators in the stdlib_*.py leaves, none of which are (or should become)
# `_AST_FILES` members: they are CODEGEN_FILES, and a file cannot sit in both partitions
# (the AST/CODEGEN disjointness the completeness tripwire checks). Folding the CURRENT name
# set into the hash, the same "extra byte fragment" mechanism the language version above
# already uses, moves the AST epoch exactly when that set moves, without relocating any
# file between partitions.
from .tex_runtime.stdlib_registry import non_spatial_args_by_name as _non_spatial_args_by_name  # noqa: E402
_AST_EPOCH = _hash_files(
    _AST_FILES, b"lang:" + _LANGUAGE_VERSION_AT_IMPORT.encode(),
    b"nonspatial:" + ",".join(sorted(_non_spatial_args_by_name())).encode())
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

        ANIM-1: `$param` names are dropped from the key HERE, by construction, rather than by
        each caller remembering to filter its map first. `binding_types` has two jobs — it types
        `@` wires for the TypeChecker, and it identifies the program — and only the second one
        must exclude params. Filtering at the callers got the first two of four right: this
        release shipped with `cook_stage_list`'s single-stage branch and `tex_tool`'s warm key
        still unfiltered, which meant a CACHE-6 sub-chain recompiled per param value and
        `install_tool(warm=True)` warmed a fingerprint no cook would ever probe. Doing it in the
        key means the rule cannot be forgotten, and the checker keeps the whole map it wants.

        (Why a param has no business in a program's identity: its type comes from its
        DECLARATION in the code — already hashed below — not from the bound value, so keeping it
        made `$k = 2` and `$k = 2.0` two programs. See tex_marshalling.param_only_names.)"""
        from .tex_marshalling import param_only_names
        drop = param_only_names(code)
        binding_key = tuple(sorted((k, v.value) for k, v in binding_types.items()
                                   if k not in drop))
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

    def get(self, code: str, binding_types: dict[str, TEXType],
            *, fp: str | None = None) -> tuple | None:
        """
        Look up a cached compilation result.

        Returns (program, type_map, referenced_bindings,
                 assigned_bindings, param_declarations, used_builtins) or None.
        Checks memory first, then disk.

        `fp` is an already-computed `fingerprint(code, binding_types)` for THIS pair — the
        caller's, when it needed the value for something else anyway (PERF-5). It is a pure
        function of the arguments, so passing it is an optimisation and never a semantic
        choice; omit it and it is computed here exactly as before.
        """
        if fp is None:
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
        fp: str | None = None,
    ):
        """Store a compilation result in both memory and disk caches.

        `fp`: see `get` — an already-computed fingerprint for this exact `(code,
        binding_types)` pair. Nothing between the probe and the store mutates
        `binding_types` (the splitback and both TypeChecker passes only read it), so the
        value a caller probed with is the value this would recompute."""
        if fp is None:
            fp = self.fingerprint(code, binding_types)
        result = (program, type_map, referenced_bindings,
                  assigned_bindings or {}, param_declarations or {},
                  used_builtins or frozenset())
        self._memory_put(fp, result)
        self._save_to_disk(fp, program, binding_types)

    def compile_tex(
        self, code: str, binding_types: dict[str, TEXType], *, fp: str | None = None
    ) -> tuple:
        """
        Compile TEX source: lex -> parse -> type-check, with caching.

        Returns (program_ast, type_map, referenced_bindings,
                 assigned_bindings, param_declarations, used_builtins).

        assigned_bindings: dict mapping output binding names to their inferred TEXType.
        param_declarations: dict mapping parameter names to {type, type_hint}.
        used_builtins: frozenset of builtin names referenced by the program.
        Raises LexerError, ParseError, or TypeCheckError on invalid code.

        PERF-5: the probe and the store share ONE fingerprint. `fp` lets the caller share
        it too — `tex_engine.prepare` needs the value for `_preflight_memory`'s memo and
        used to compute it a second time beside this call, which is why a warm cook read
        two `fingerprint` calls per cook and a cold one read three.
        """
        if fp is None:
            fp = self.fingerprint(code, binding_types)
        cached = self.get(code, binding_types, fp=fp)
        if cached is not None:
            return cached

        # Full compilation pipeline: the one front end (lex + parse + splitback), then the
        # shared post-parse orchestration (STR-8: identical to the fusion path's).
        # DATA-6: the AST arriving at `compile_ast` is already split against THIS map; the
        # splitback there is an identity on it (a kept plane read is kept again, a swizzle
        # has no dot left to split) and is the hook the expansion step will re-read once it
        # adds per-plane rows.
        program = parse_and_split(code, binding_types)
        program, type_map, referenced, assigned, params, used_builtins = \
            self.compile_ast(program, binding_types, source=code)

        self.put(code, binding_types, program, type_map,
                 referenced, assigned, params, used_builtins, fp=fp)
        return (program, type_map, referenced, assigned, params, used_builtins)

    def compile_ast(self, program, binding_types, *, source: str):
        """STR-8: the shared post-parse compile pipeline (type-check → optimize →
        re-type-check on the optimized AST → collect builtins), used by BOTH the
        normal path (`compile_tex`, from source) and fusion (`compile_fused`, from a
        spliced AST). Returns the same 6-tuple as `compile_tex`. Error-agnostic: a
        `TypeCheckError` propagates so each caller translates it as it sees fit
        (fusion wraps it as `FusionError`).

        DATA-6: the plane seam sits at the top, BEFORE the first TypeChecker — the parser
        never sees binding types, and the checker cannot replace a node it is typing. Two
        steps, one owner, in this order: (1) [expansion — lands with the wire value] a PLANES
        wire's per-plane rows are added to `binding_types` for the planes the source mentions;
        (2) the splitback below resolves every dotted `@name.seg` against that map. Both
        production entries and the test harnesses converge here, so nothing compiles an
        unresolved dotted binding.
        """
        program = splitback_dotted_bindings(program, binding_types, source=source)
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
        # Pre-compute builtin identifiers (avoids AST walk on every execution). REG-1e: the
        # SAME pass also names every function CALLED, so `Program.non_spatial_calls` (the
        # single source `_consensus_extent` reads, tex_runtime/interpreter.py) is set HERE,
        # once per compile, never a second walk -- `compile_fused` reaches this same method
        # with the FULL spliced AST, so a fused program's flag is sound by construction.
        used_builtins, called_fns = _collect_identifiers_and_calls(program)
        _ns_names = _non_spatial_args_by_name()
        program.non_spatial_calls = bool(_ns_names) and not called_fns.isdisjoint(_ns_names)
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
        """Pickle *data* to *path* atomically AND durably, so a concurrent reader — e.g. a
        second ComfyUI instance sharing the dir — can never observe a half-written entry, and
        a crash never leaves a torn artifact the next launch would load. ENG-13 routes every
        persisted engine file through the one `tex_recovery.atomic_write`."""
        from .tex_recovery import sign_pickle
        # Streamed, not blobbed: a compiled artifact is small next to a frame, but there is no
        # reason to build a second copy of it in memory to reach the same write.
        #
        # BRIEF-10: routed through `sign_pickle`, which appends a keyed-MAC trailer so
        # `_load_from_disk`/`_load_codegen_from_disk` can authenticate the bytes BEFORE
        # `pickle.load` runs a crafted `__reduce__`. The signing is HMAC-SHA256 at memcpy speed
        # on a ≤~200 KB artifact — off the per-frame path and lost in the recompile it guards.
        #
        # NOT fsynced, and this is the load-bearing half. This write is INLINE ON THE COOK
        # THREAD, on the first cook of every distinct program — i.e. on every ComfyUI code edit
        # and every re-queue. An fsync there measured **+44.4% CPU / +45.6% CUDA on cold
        # compile** (proven causally by stubbing `os.fsync`, 4/4 interleaved rounds with no
        # distribution overlap). The release's own invariant-#7 measurement covered steady-state
        # cooks only, so it missed a regression on the path a user hits every time they type.
        #
        # The durability it bought was not worth having: an artifact is pure cache, losing one
        # costs a single ~2.5 ms recompile, and `_load_from_disk` already unlinks-and-recompiles
        # on a bad load — so a torn file is a case this code handles by design rather than a case
        # the fsync was protecting against. Atomicity (temp + rename) is what matters here and
        # is unaffected.
        sign_pickle(str(path), data)

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
            # BRIEF-10: AUTHENTICATE before deserialise. `load_verified` reads the file once,
            # checks the keyed-MAC trailer, and unpickles the SAME buffer — so a `.pkl` with no
            # valid trailer (a pre-integrity file, a foreign one, or a crafted one) is a MISS and
            # its `__reduce__` never runs, with no re-read a writer could swap under. The
            # version/epoch checks below are unchanged; the MAC is a gate in front, not a reorder.
            from .tex_recovery import load_verified, _UNVERIFIED, _FUTURE_TRAILER
            data = load_verified(path)
            if data is _UNVERIFIED:
                try:
                    path.unlink(missing_ok=True)     # unsigned/foreign/corrupt: recompile fresh
                except OSError:
                    pass                             # F6: an undeletable file is a silent miss
                return None
            if data is _FUTURE_TRAILER:
                return None                          # a newer TEX's file — leave it, just miss

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
            # BRIEF-10: the OUTER pickle is itself an execution sink, so authenticate the file
            # before deserialising — the inner-blob sha below only ever guarded corruption of the
            # marshal blob (and is attacker-recomputable), so it cannot stand in for this.
            # `load_verified` reads once, checks the MAC, and unpickles that one buffer (no
            # re-read window — F1).
            from .tex_recovery import load_verified, _UNVERIFIED, _FUTURE_TRAILER
            data = load_verified(path)
            if data is _UNVERIFIED:
                try:
                    path.unlink(missing_ok=True)     # unsigned/foreign/corrupt: regenerate
                except OSError:
                    pass                             # F6: an undeletable file is a silent miss
                return None
            if data is _FUTURE_TRAILER:
                return None                          # a newer TEX's sidecar — leave it, just miss
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
