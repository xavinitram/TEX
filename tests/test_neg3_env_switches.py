"""NEG-3 — the environment switches nobody was watching.

Two size switches shipped with no test and no user-facing document:
`TEX_GOVERNOR_BUDGET_MB` (`tex_memory.governor_budget`) and `TEX_RESULTS_DISK_MB`
(`tex_results._budget_bytes`). Both were read as "whatever this parses to", so `0` and `-8`
were accepted as *sizes* — and a zero byte budget is not "no limit" and not "off", it is
"every entry is over budget on every check", i.e. a silent cache-off switch wearing the
spelling of a size. The verdict for both is **keep and harden**, not freeze: they are the
only way an embedding host can state a budget, and a host that states one deserves to be
told when its value was ignored.

What is pinned here, per switch: the UNSET default (the ComfyUI path — nothing may move),
a SET-to-a-sane-value reading, and a SET-to-garbage reading that must equal the unset one.
Garbage is enumerated rather than sampled: empty, non-numeric, zero, negative, and — for the
whole-MiB switch — a fractional value.

The third row is a drift gate: every environment switch the shipped product reads must be
named in `README.md`'s switch table. A switch missing from the document is the one an audit
proposes deleting next year, which is exactly how these two got here.
"""
import os
import re
from pathlib import Path

import torch

from helpers import SubTestResult

_PKG = Path(__file__).resolve().parent.parent


class _env:
    """Set (or unset, with value None) one variable for the duration of the block."""

    def __init__(self, name, value):
        self.name, self.value = name, value

    def __enter__(self):
        self._prev = os.environ.get(self.name)
        if self.value is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self.value
        return self

    def __exit__(self, *exc):
        if self._prev is None:
            os.environ.pop(self.name, None)
        else:
            os.environ[self.name] = self._prev
        return False


# The values that must ALL be refused. "" is included because the reader's `if override:`
# already skipped it — pinning that keeps a future "is not None" rewrite honest.
_GARBAGE = ("", "0", "-8", "0.0", "-0.5", "none", "512MB", "  ")


def test_neg3_governor_budget_env_is_hardened(r: SubTestResult):
    """`TEX_GOVERNOR_BUDGET_MB`: set / unset / garbage, all three states."""
    from TEX_Wrangle.tex_memory import governor_budget
    cpu = torch.device("cpu")

    try:
        with _env("TEX_GOVERNOR_BUDGET_MB", None):
            default = governor_budget(cpu)
        assert isinstance(default, int) and default > 0, default
        r.ok(f"unset: the computed CPU governor budget stands ({default >> 20} MiB)")
    except Exception as e:
        r.fail("governor budget unset", f"{type(e).__name__}: {e}")
        return

    try:
        for mb in ("1", "512", "4096"):
            with _env("TEX_GOVERNOR_BUDGET_MB", mb):
                got = governor_budget(cpu)
            assert got == int(mb) * (1 << 20), (mb, got)
        r.ok("set: a positive whole number of MiB is honoured exactly")
    except Exception as e:
        r.fail("governor budget set", f"{type(e).__name__}: {e}")

    try:
        bad = []
        for v in _GARBAGE + ("1.5",):
            with _env("TEX_GOVERNOR_BUDGET_MB", v):
                got = governor_budget(cpu)
            if got != default:
                bad.append((v, got))
        assert not bad, f"accepted as a budget instead of falling back to {default}: {bad}"
        r.ok("garbage: every non-positive / unparseable value is refused, default stands")
    except Exception as e:
        r.fail("governor budget garbage", f"{type(e).__name__}: {e}")

    try:
        # The specific hazard the audit named: zero is not a size, and the governor holding a
        # zero budget arbitrates every pool away on the next cook.
        with _env("TEX_GOVERNOR_BUDGET_MB", "0"):
            assert governor_budget(cpu) != 0, \
                "a zero coordinated budget evicts the stdlib, graph and frame pools to nothing"
        r.ok("zero is refused (the coordinated budget can never be 0 by typo)")
    except Exception as e:
        r.fail("governor budget zero", f"{type(e).__name__}: {e}")


def test_neg3_results_budget_envs_are_hardened(r: SubTestResult):
    """`TEX_RESULTS_DISK_MB` and its RAM twin, through their one reader and end to end."""
    from TEX_Wrangle.tex_results import ResultCache, _budget_bytes

    default = 4 << 30
    try:
        for name in ("TEX_RESULTS_DISK_MB", "TEX_RESULTS_BUDGET_MB"):
            with _env(name, None):
                assert _budget_bytes(name, default) == default, name
        r.ok("unset: the caller's default budget stands for both results switches")
    except Exception as e:
        r.fail("results budgets unset", f"{type(e).__name__}: {e}")

    try:
        for name in ("TEX_RESULTS_DISK_MB", "TEX_RESULTS_BUDGET_MB"):
            for v, want in (("16", 16 << 20), ("0.5", (1 << 20) // 2), ("2048", 2048 << 20)):
                with _env(name, v):
                    assert _budget_bytes(name, default) == want, (name, v)
        r.ok("set: a positive size in MiB is honoured (fractional MiB included)")
    except Exception as e:
        r.fail("results budgets set", f"{type(e).__name__}: {e}")

    try:
        bad = []
        for name in ("TEX_RESULTS_DISK_MB", "TEX_RESULTS_BUDGET_MB"):
            for v in _GARBAGE:
                with _env(name, v):
                    got = _budget_bytes(name, default)
                if got != default:
                    bad.append((name, v, got))
        assert not bad, f"accepted instead of falling back to the default: {bad}"
        r.ok("garbage: zero, negative and unparseable values are refused for both")
    except Exception as e:
        r.fail("results budgets garbage", f"{type(e).__name__}: {e}")

    try:
        # End to end: the cache the switch exists to size must read the refusal too, and a
        # host argument still wins over the environment (the switch is a fallback, not a lock).
        with _env("TEX_RESULTS_DISK_MB", "0"):
            with _env("TEX_RESULTS_BUDGET_MB", "-1"):
                c = ResultCache()
                assert c._disk_budget == default, c._disk_budget
                assert c._budget > 0, c._budget
                c2 = ResultCache(budget_mb=8, disk_budget_mb=64)
                assert c2._budget == 8 << 20 and c2._disk_budget == 64 << 20
        r.ok("ResultCache: a refused env value never becomes a zero-byte tier")
    except Exception as e:
        r.fail("ResultCache budgets", f"{type(e).__name__}: {e}")


# ── the drift gate ────────────────────────────────────────────────────────────────────

# Directories under the package that are NOT the shipped product (tests, benchmarks, tools and
# the design notes may read whatever they like — their switches are not a user's business).
_NOT_PRODUCT = {"tests", "benchmarks", "tools", "docs", "examples", "stock", "js",
                "editor_build", "assets", ".git", "__pycache__", "wiki"}

_ENV_NAME = re.compile(r"[\"'](TEX_[A-Z0-9_]+)[\"']")


def _product_env_names() -> set:
    """Every `TEX_*` switch name the shipped product spells as a string literal.

    Deliberately NOT filtered to lines that mention the environment: `tex_results.py` names
    both of its switches as ARGUMENTS to a shared reader, on lines that mention neither, and a
    census that missed those two would have missed the very switch this lane is here for.
    """
    found = set()
    for path in _PKG.rglob("*.py"):
        rel = path.relative_to(_PKG)
        if rel.parts[0] in _NOT_PRODUCT:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        found.update(_ENV_NAME.findall(text))
    return found


def test_neg3_every_product_env_switch_is_documented(r: SubTestResult):
    """README must name every switch the product reads. A gate, not a snapshot: a new
    switch lands with its documentation line or this reds with the switch's name."""
    try:
        names = _product_env_names()
        assert len(names) >= 10, f"the census found almost nothing -- it is broken: {names}"
        readme = (_PKG / "README.md").read_text(encoding="utf-8")
        assert "## Environment switches" in readme, \
            "README.md has no environment-switch section for a new switch to land in"
        missing = sorted(n for n in names if n not in readme)
        assert not missing, f"read by the product, absent from README.md: {missing}"
        r.ok(f"all {len(names)} product environment switches are named in README.md")
    except Exception as e:
        r.fail("env-switch documentation drift", f"{type(e).__name__}: {e}")
