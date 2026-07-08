"""
LNT-1 — enforce the numpy ban (a tribal invariant that has already bitten).

CI runs PyTorch WITHOUT numpy, so `import numpy` or `Tensor.numpy()` raises there
even though it works locally. This lint scans the shipped package + tests for those
patterns so the ban is machine-enforced, not doc-only. Use `struct.pack` for raw
bytes and `.tolist()` for arrays instead.
"""
import os
import re
from helpers import SubTestResult

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # TEX_Wrangle/
_BANNED = re.compile(r"\bimport\s+numpy\b|\bfrom\s+numpy\b|\.numpy\s*\(")
_SKIP_DIRS = ("__pycache__", "editor_build", "node_modules", ".git")
_SELF = os.path.basename(__file__)


def _code_only(line: str) -> str:
    """Strip a trailing `# comment` so a comment mentioning the pattern doesn't
    trip the lint (crude but sufficient — no `#` appears inside our string lits
    on the lines that matter)."""
    return line.split("#", 1)[0]


def test_no_numpy_ban(r: SubTestResult):
    print("\n--- LNT-1: numpy-ban lint ---")
    hits = []
    for root, dirs, files in os.walk(_PKG):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fn in files:
            if not fn.endswith(".py") or fn == _SELF:
                continue
            path = os.path.join(root, fn)
            try:
                with open(path, encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        if _BANNED.search(_code_only(line)):
                            rel = os.path.relpath(path, _PKG)
                            hits.append(f"{rel}:{i}: {line.strip()}")
            except (OSError, UnicodeDecodeError):
                continue
    if hits:
        r.fail("numpy-ban lint", "numpy usage found (CI has no numpy):\n  " +
               "\n  ".join(hits[:20]))
    else:
        r.ok("no numpy import / .numpy() anywhere in the package or tests")
