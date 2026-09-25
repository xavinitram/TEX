# Security Policy

## TEX Sandboxing

TEX Wrangle executes user-written TEX code through a **sandboxed pipeline**. The security model is designed to prevent arbitrary code execution:

- **No user-supplied Python** — TEX code is parsed into an AST and executed either by a tree-walking interpreter or by a codegen backend that emits Python from the AST. The codegen uses `exec()` internally but only on code it generates itself from the validated AST — user strings are never interpolated into the generated source.
- **No file I/O** — TEX cannot read or write files, access the filesystem, or interact with the network.
- **No imports** — TEX has no import mechanism. All available functions are hardcoded in `stdlib_signatures.py`.
- **No reflection** — No access to `__import__`, `globals()`, `getattr()`, or any Python introspection.
- **Resource limits** — For loops are capped at 1024 iterations. Arrays are capped at 1024 elements.

TEX code can only perform PyTorch tensor operations through the predefined standard library functions.

## What the registry scanner reports, and why each finding is there

The Comfy registry scans every published archive with YARA rules and holds a version for
manual review on any finding. TEX's shipped code produces a small, stable set; they are the
product's real behaviour, documented here so a reviewer does not have to rediscover them.
(`tests/`, `benchmarks/`, `tools/`, `docs/` and `editor_build/` are excluded from the archive
by `.comfyignore`; `tests/test_pub1_archive.py` pins this census so it cannot grow unnoticed. The
scanner reads prose as well as code — this table therefore *describes* each mechanism rather than
quoting the call it matches, which is not evasion: the code sites themselves stay declared below.)

| Finding | Where | What it is | Reachable from outside? |
|---|---|---|---|
| `compile()` / `exec()` | `tex_runtime/codegen.py` | The codegen tier: the DSL's typed AST is emitted as one Python function and executed. Identifiers are ASCII-only at the lexer, unknown names are rejected by the type checker, string literals are emitted with `repr()`, bindings as `_bind[{name!r}]` — no user text reaches the source. Proven by `tests/test_v026_phase1.py::test_tool_emitter_fuzz`. | Only through a program's own AST |
| `marshal.loads()` + `exec()` | `tex_runtime/codegen_persist.py` | Rehydrates a persisted codegen function. The `.cg` sidecar is HMAC-SHA256-verified with a per-user key (`tex_recovery.load_verified`), then checked for codegen epoch, `MAGIC_NUMBER` and blob SHA-256; any failure deletes it and regenerates from source. | Only TEX's own verified output |
| a subprocess call to `vcvarsall.bat` | `tex_runtime/compiled.py` | Windows only: runs `vcvarsall.bat` once so torch inductor can find `cl.exe`. Constant argv; the path comes from probing known Visual Studio locations; called lazily on the first `torch.compile` attempt, never at import, never from a route. | No |
| two environment writes | `tex_runtime/compiled.py`, `tex_testkit.py` | The INCLUDE/LIB/LIBPATH/PATH output of that probe; and the opt-in host test harness pointing `TEX_CACHE_DIR` at a temp dir (restored afterwards). `tex_testkit` is not imported by the node, routes or CLI. | No |
| environment reads | several modules | Read-only `TEX_*` tuning knobs (cache dir and byte budgets, opt-in thread count, the codegen out-reuse kill switch, opt-in ROI paths, offline docs), plus `INCLUDE` and the per-user state directory. | Reads only |
| LiteGraph's node-link `connect` call in `js/` | `js/tex_extension.js`, `js/tex_cm6_bundle.js` | LiteGraph's node-link API and the editor bundle — a Python rule matching JavaScript. The package makes no network call anywhere. | No |

## The HTTP routes

Twelve, under `/tex_wrangle/`, unauthenticated as every node's are. None accepts a URL, spawns
a process, or reaches the codegen tier. The three that write do so only under the user
directory: `user_snippets` to one fixed JSON file (snippet names are keys, never paths);
`publish_tool` to a filename sanitised to `[A-Za-z0-9_.-]` in the tool store, with the
destination not settable from the request and a different-tool collision refused;
`free_caches` drops TEX's own caches. `docs/{page}` resolves through a whitelist of three
shipped files. `check`, `chain_preflight` and `detect_regions` lex/parse/type-check or splice
ASTs and never emit or execute code.

## Persisted state and deserialization

Everything TEX writes to disk lives under one root — `TEX_CACHE_DIR` if set, else a
`.tex_cache` folder beside the package (`tex_cache.py:TEXCache.__init__`) — with two
exceptions that live in the *user's own* data, not the cache dir. Every writer uses the
one atomic-write primitive (`tex_recovery.py:atomic_write`, temp + `os.replace`, optionally
`fsync`), so a crash never leaves a torn file.

| File(s) | Writer (file:symbol) | Format | Location control | Host can relocate/disable? |
|---|---|---|---|---|
| `<fp>.pkl` (program cache) | `tex_cache.py:TEXCache._save_to_disk` / `_atomic_pickle` | pickle + HMAC-SHA256 trailer | `TEX_CACHE_DIR` env var | Yes — env var, or point it at a non-writable/absent dir to disable |
| `<fp>.cg` (codegen sidecar) | `tex_cache.py:TEXCache._persist_codegen` / `store_codegen_fn` | pickle-wrapped `{marshal blob, sha256, src, magic}` + HMAC-SHA256 trailer | `TEX_CACHE_DIR` | Yes, same as above |
| `results/<key>.frame` (RAM-budget spill) | `tex_results.py:ResultCache._spill` via the module-level `_atomic_pickle` | pickle (tensor + metadata) + HMAC-SHA256 trailer | `TEX_CACHE_DIR` (`ResultCache._spill_dir`) | Yes |
| `autotier.json` | `tex_runtime/autotier.py:_persist` | JSON, plain | `TEX_CACHE_DIR` (`_persist_path`) | Yes |
| `warm_state.json` + `warm_state.json.journal` | `tex_runtime/warm_state.py:persist` / `note_update` (journal via `tex_recovery.py:Journal.append`) | JSON snapshot; JSONL journal | `TEX_CACHE_DIR` | Yes |
| `xfer.json` | `tex_runtime/xfer.py:_persist` | JSON, plain | `TEX_CACHE_DIR` | Yes |
| `%LOCALAPPDATA%/XDG_STATE_HOME/TEX_Wrangle/cache_mac.key` | `tex_recovery.py:_publish_new_key` | raw 32-byte key | Not `TEX_CACHE_DIR` — deliberately outside it, in the OS per-user profile | No (by design: the whole point is that the cache-dir writer cannot reach it) |
| `<user_dir>/tex_wrangle/user_snippets.json` | `tex_snippets.py:save_user_snippets` | JSON, plain (`{name: code}`) | Host's `get_user_dir()` (`tex_runtime/host.py:ComfyHostServices.get_user_dir`), else `TEX_CACHE_DIR/user` | Yes, via the host's user-dir setting or `TEX_CACHE_DIR` |
| `<user_dir>/tex_wrangle/tools/*.textool` | `tex_tool.py:write_tool` | JSON manifest (may carry inline TEX-DSL source in `code`/`terminal_code`) | Same as snippets (`tex_tool.py:tools_dir`) | Yes, same as above |
| `tools/gate.py`'s verdict cache, `tests/known_reds.json` | `tools/gate.py:_cache_write`, hand-authored | JSON, plain | Dev-tooling only; cache lives outside the repo (`tools/gate.py:_cache_path`); excluded from the shipped archive by `.comfyignore` | N/A — never reaches a ComfyUI install |

Deserialize sites, every one that reads bytes back off disk:

| Site (file:symbol) | Reads | Protection | Key source | On verification failure |
|---|---|---|---|---|
| `tex_recovery.py:load_verified` — the ONE unpickling call in the tree, called by `tex_cache.py:TEXCache._load_from_disk` (`.pkl`) and `_load_codegen_from_disk` (`.cg`), and by `tex_results.py:ResultCache._restore` (`.frame`) | The pickled payload | HMAC-SHA256 trailer (`MAGIC + tag`, `_MAC_MAGIC`/`_MAC_TAG_LEN`), checked with `hmac.compare_digest` BEFORE the payload is ever unpickled, on the same verified buffer (no re-read/TOCTOU) | `tex_recovery.py:_mac_key` — a 32-byte key minted once per user, stored outside any cache dir (see table above), memoised per process | Returns `_UNVERIFIED`: treated as a cache miss, file unlinked and regenerated. A newer-version trailer (`_FUTURE_TRAILER`) is left on disk, untouched, never unpickled |
| `tex_runtime/codegen_persist.py:materialize_codegen` — un-marshals `blob` then executes the resulting code object | A marshalled Python code object | Reached only after its caller (`tex_cache.py:_load_codegen_from_disk`) has already run the `.cg` file through `load_verified` (HMAC-gate above) AND rechecked the codegen epoch, `_BYTECODE_MAGIC`, and a SHA-256 over the inner blob | Same MAC key, plus the inner SHA-256 (corruption-only, attacker-recomputable — it is not the security boundary, `load_verified` is) | Any mismatch deletes the `.cg` and falls back to a fresh compile; `materialize_codegen` itself is never reached |
| `tex_runtime/codegen.py:_CodeGen.build` — compiles and runs freshly generated source (the site already declared above under "What the registry scanner reports") | Not a deserialize site: the source is emitted THIS process, from the validated AST, never read back off disk. Identifiers are ASCII-only, string literals are `repr()`'d, no user text is interpolated (§"TEX Sandboxing" above) | n/a | n/a | n/a |
| `tex_runtime/autotier.py:load`, `tex_runtime/xfer.py:_load`, `tex_runtime/warm_state.py:load` (+ `Journal.replay`) | `json.load`/`json.loads` of the three JSON files above | None — no MAC, no signature. `json.load` cannot execute code, so the exposure is data-poisoning (a wrong tier/bandwidth verdict or capturability flag), not RCE; each is additionally gated by a version/arch tag that a stale or foreign file simply fails, dropping to a cold recompute | n/a | Wrapped in a bare `except Exception: pass` (or, for `warm_state`, a version-tag mismatch): the record is skipped and the value in question is re-derived from scratch |
| `tex_snippets.py:load_user_snippets`, `tex_tool.py:load_tool`/`write_tool`'s existing-file check | `json.load` of the user's own snippets / tool-store files | None — no MAC. Data is user-authored strings (snippet code, tool manifest fields); a `.textool`'s `code` is validated (`tex_tool.py:validate_manifest`) and later runs through the SAME sandboxed AST pipeline as any TEX program a user types — it never reaches Python `exec` un-parsed | n/a | Malformed JSON raises `SnippetStoreError` (surfaced as a 503, never treated as an empty store) or, for the tool-store collision check, is caught and the write proceeds |
| `tex_engine.py:prepare`'s `chain_payload`, `tex_node.py:_parse_time_context`/`_parse_slot_map`, `tex_lsp.py`'s request body | `json.loads` of ComfyUI-graph widget values / LSP protocol bytes | None, but out of THIS scope: none of these are files TEX persists — they are per-call host/editor input, already un-trusted by construction, and are only ever read into plain dicts of numbers/strings, never `exec`'d | n/a | Wrapped in `try/except`, falls back to `None`/`[]`/zeros |
| `benchmarks/cache_capacity_bench.py`'s `torch.load(f, weights_only=True)` | Its own just-written temp `.frame` reference sample | `weights_only=True`; the file is `mkstemp`'d and removed in the same process run, never a shared or long-lived path | n/a | n/a — dev benchmark, excluded from the shipped archive |

**No unprotected RCE-capable site exists at head.** The three unpickling call sites BRIEF-10
found (the program cache, the codegen sidecar, and the frame spill) are now one call site
(`tex_recovery.py:load_verified`), and every one of them is reached only after its HMAC-SHA256
trailer verifies. The JSON-only sites (`autotier.json`, `warm_state.json` + journal, `xfer.json`,
`user_snippets.json`, `*.textool`) carry no integrity check, but `json.load` cannot execute code —
the worst a forged file buys is a bad tier verdict or a bogus snippet, always inside a
version/arch-tagged, best-effort, miss-on-failure load. These are tracked as findings for a
future hardening pass, not fixed here.

## Reporting a Vulnerability

If you discover a security issue in TEX Wrangle, please report it via [GitHub Issues](https://github.com/xavinitram/TEX/issues) with the label "security", or contact the maintainer directly.

Please include:
- A description of the vulnerability
- Steps to reproduce (TEX code that demonstrates the issue)
- The potential impact

We will acknowledge reports within 72 hours and work to address confirmed vulnerabilities promptly.
