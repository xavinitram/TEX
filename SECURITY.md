# Security Policy

## TEX Sandboxing

TEX Wrangle executes user-written TEX code through a **sandboxed pipeline**. The security model is designed to prevent arbitrary code execution:

- **No user-supplied Python** — TEX code is parsed into an AST and executed either by a tree-walking interpreter or by a codegen backend that emits Python from the AST. The codegen uses `exec()` internally but only on code it generates itself from the validated AST — user strings are never interpolated into the generated source.
- **No file I/O** — TEX cannot read or write files, access the filesystem, or interact with the network.
- **No imports** — TEX has no import mechanism. All available functions are hardcoded in `stdlib_signatures.py`.
- **No reflection** — No access to `__import__`, `globals()`, `getattr()`, or any Python introspection.
- **Resource limits** — For loops are capped at 1024 iterations. Arrays are capped at 1024 elements.

TEX code can only perform PyTorch tensor operations through the predefined standard library functions.

## Reporting a Vulnerability

If you discover a security issue in TEX Wrangle, please report it via [GitHub Issues](https://github.com/xavinitram/TEX/issues) with the label "security", or contact the maintainer directly.

Please include:
- A description of the vulnerability
- Steps to reproduce (TEX code that demonstrates the issue)
- The potential impact

We will acknowledge reports within 72 hours and work to address confirmed vulnerabilities promptly.
