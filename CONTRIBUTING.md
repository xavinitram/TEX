# Contributing to TEX Wrangle

Thanks for your interest in contributing to TEX Wrangle. This guide covers the essentials for getting started.

## Dev Setup

1. Clone the repo into your ComfyUI custom nodes directory:
   ```
   cd <ComfyUI>/custom_nodes
   git clone https://github.com/xavinitram/TEX.git TEX_Wrangle
   ```
2. Python 3.10+ is required. The only dependency beyond the standard library is PyTorch (provided by ComfyUI).
3. Run the full test suite:
   ```
   cd TEX_Wrangle/tests
   python run_all.py
   ```
   `run_all.py` is the canonical runner (no extra dependencies; it prints its own sub-test
   count on each run rather than a number pinned here, which drifts as rows are added).
   `python -m pytest tests/ -v` also works if you have pytest installed.

No additional packages or build steps are needed for the core compiler and runtime.

## Code Style

- Follow the patterns already established in the codebase.
- Do not introduce external dependencies. TEX intentionally relies only on PyTorch.
- Keep functions focused and well-named. Add docstrings where the intent is not obvious.

## Adding a Standard Library Function

1. **Declare the signature** in `tex_compiler/stdlib_signatures.py` -- specify the function name, parameter types, and return type.
2. **Implement the function** as `fn_<name>` in `tex_runtime/stdlib.py` using PyTorch tensor operations.
3. **Register it** in the `get_functions()` dispatch dict in `tex_runtime/stdlib.py` (maps the TEX name to your `fn_<name>`), so the interpreter and codegen can resolve it.
4. **Add tests** in the appropriate file under `tests/` (see `tests/README.md` for where each type of test belongs).

## Error Codes & Diagnostics

User-facing errors flow through `tex_compiler/diagnostics.py` and render as: message → source-line caret → a `Try:` suggestion → a `Help:` hint → a trailing `Error Code: Exxxx`. Codes are grouped by stage:

| Range | Stage |
|-------|-------|
| `E0xxx` | Internal (a phase raised without a structured diagnostic) |
| `E1xxx` | Lexer |
| `E2xxx` | Parser |
| `E3xxx` | Type checker (names, scope, types & coercions) |
| `E4xxx` | Type checker — unrecognized construct (catch-all) |
| `E5xxx` | Type checker — function signatures |
| `E6xxx` | Runtime / interpreter (e.g. `E6050` unknown function, `E6051` a function's runtime failure) |
| `E7xxx` | Host I/O (a host `FrameProvider` binding TEX cannot type) |
| `E9xxx` | Tools (building or preflighting a `.textool` bundle) |
| `W7xxx` | Warnings |

Assign a **new** code for a new condition rather than reusing one — codes are stable anchors that map to documentation, so never renumber an existing one. This table, `DEVELOPMENT.md`'s
own copy and `tools/gen_error_codes.py`'s family list are required to agree.

## Log Lines

Python `logging` calls (as opposed to the user-facing diagnostics above) follow their own
guideline — subject, event, consequence, values, no first person — in `DEVELOPMENT.md`
§"Log Lines".

## Editor Build

The `editor_build/` directory contains the source for the CodeMirror 6 editor component. You only need to touch this if you are modifying the code editor UI. Building it requires Node.js and npm:

```
cd editor_build
npm install
npm run build
```

`npm run build` writes `js/tex_cm6_bundle.js`; commit it with the source change. The editor's keyword, built-in variable and stdlib function lists are generated: after adding a stdlib function or a keyword, run `python tools/gen_editor_lexicon.py` before building.

For all other contributions, the editor build is not required.

## Pull Request Expectations

- All existing tests must pass (`python -m pytest tests/ -v`).
- Remove any debug output (`print()` statements, `console.log()` calls) before submitting.
- Describe what your PR changes and why. Reference related issues where applicable.
- If your change affects user-facing behavior, update the README accordingly.

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/xavinitram/TEX/issues) for bug reports and feature requests. Issue templates are provided to help structure your report.
