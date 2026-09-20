"""The canonical local harness: pytest with the ComfyUI v3 NodeOutput wrapper disarmed.

Why this file exists at all. The embedded interpreter this repository is developed against
puts the host application's package directory on `sys.path`, so `tex_node._V3_AVAILABLE`
is True locally and False in CI (CI has no host installed). A handful of rows then wrap
their result in the host's v3 `NodeOutput` and error where CI passes — a difference in the
environment, not in the tree. Disarming the flag before collection makes the local reading
comparable with the CI one.

It used to be six lines retyped from prose by every reader, which is how a harness drifts.
Run it from the directory that CONTAINS the package, with the same arguments as pytest:

    python -X utf8 TEX_Wrangle/tools/canonical_harness.py TEX_Wrangle/tests -q -m "not slow"

`tools/` is excluded from the published archive (`.comfyignore`), so nothing here ships.
"""
import os
import sys

import pytest

sys.path.insert(0, os.getcwd())
import TEX_Wrangle.tex_node as t          # noqa: E402

t._V3_AVAILABLE = False
sys.exit(pytest.main(sys.argv[1:]))
