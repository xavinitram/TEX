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

## Reporting a Vulnerability

If you discover a security issue in TEX Wrangle, please report it via [GitHub Issues](https://github.com/xavinitram/TEX/issues) with the label "security", or contact the maintainer directly.

Please include:
- A description of the vulnerability
- Steps to reproduce (TEX code that demonstrates the issue)
- The potential impact

We will acknowledge reports within 72 hours and work to address confirmed vulnerabilities promptly.
