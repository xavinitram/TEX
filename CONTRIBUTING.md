# Contributing to TEX Wrangle

Thanks for your interest in contributing to TEX Wrangle. This guide covers the essentials for getting started.

## Dev Setup

1. Clone the repo into your ComfyUI custom nodes directory:
   ```
   cd <ComfyUI>/custom_nodes
   git clone https://github.com/xavinitram/TEX.git TEX_Wrangle
   ```
2. Python 3.10+ is required. The only dependency beyond the standard library is PyTorch (provided by ComfyUI).
3. Run tests from the project root:
   ```
   cd TEX_Wrangle
   python -m pytest tests/test_tex.py -v
   ```

No additional packages or build steps are needed for the core compiler and runtime.

## Code Style

- Follow the patterns already established in the codebase.
- Do not introduce external dependencies. TEX intentionally relies only on PyTorch.
- Keep functions focused and well-named. Add docstrings where the intent is not obvious.

## Adding a Standard Library Function

1. **Declare the signature** in `tex_compiler/stdlib_signatures.py` -- specify the function name, parameter types, and return type.
2. **Implement the function** in `tex_runtime/stdlib.py` using PyTorch tensor operations.
3. **Add tests** in `tests/test_tex.py` covering typical usage and edge cases.

## Editor Build

The `editor_build/` directory contains the source for the CodeMirror 6 editor component. You only need to touch this if you are modifying the code editor UI. Building it requires Node.js and npm:

```
cd editor_build
npm install
npm run build
```

For all other contributions, the editor build is not required.

## Pull Request Expectations

- All existing tests must pass (`python -m pytest tests/test_tex.py -v`).
- Remove any debug output (`print()` statements, `console.log()` calls) before submitting.
- Describe what your PR changes and why. Reference related issues where applicable.
- If your change affects user-facing behavior, update the README accordingly.

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/xavinitram/TEX/issues) for bug reports and feature requests. Issue templates are provided to help structure your report.
