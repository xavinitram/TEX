"""PUB-1 — the archive is the product.

`comfy node publish` uploads `git ls-files` minus the root `.comfyignore` (comfy-cli 1.20.0,
`comfy_cli/file_utils.py::zip_files`, gitwildmatch via pathspec). The registry then scans that
archive and holds a version out of `Active` — the only status users receive — on ANY finding
until an admin approves it. Most of the findings that held three releases back were in
directories that do not ship a node: the suite's own NEGATIVE security tests, the benchmarks,
the generators. This file keeps the archive honest from the tree side:

  (a) every top-level tracked directory is either ignored or named in the SHIP allowlist, so a
      new directory reds by name instead of shipping by accident; and `.comfyignore` holds only
      plain directory-prefix patterns, the one subset this file can mirror provably;
  (b) a surface ratchet over EVERY shipped text file — `.py`, `.js`, `.md`, `.json`, `.toml`,
      `.txt`, `.tex`, `.textool` and anything else that decodes as UTF-8 (fonts and images are
      skipped) — for the scanner's families, pinned at the measured counts: a new site reds with
      file:line, a decrease reds until the pin follows it down (the REG-2 headroom-floor
      discipline — a pin that only ever rises is decoration). Prose IS in scope: the scanner
      reads it (PUB-2; the measurement is above `_FAMILIES`);
  (c) no shipped module imports from an ignored directory (an AST scan, not a grep).

Plus the one runtime seam the split exposed: `tex_validate_hw`'s triton lane delegated to
`benchmarks/`, which an installed node no longer has, so it now SKIPs like its GPU-only
siblings instead of raising.
"""
import ast
import collections
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from helpers import SubTestResult

_PKG = Path(__file__).resolve().parent.parent
_COMFYIGNORE = _PKG / ".comfyignore"

# The top-level directories that SHIP, by name. A new top-level directory must be added here
# or to .comfyignore — that decision is the point of arm (a). Every name must exist as a tracked
# directory (a stale entry reds too).
_SHIP_DIRS = {"examples", "js", "stock", "tex_compiler", "tex_io", "tex_runtime"}

# Files a shipped reader opens by path (verified readers, so a `.comfyignore` edit cannot
# silently drop one): `__init__.py` serves examples/ on /tex_wrangle/snippets and the three
# root docs on /tex_wrangle/docs/{page}; WEB_DIRECTORY is js/; the stock `.textool`s are the
# shipped exemplars `tex build` and the tool loader read by path; the registry reads
# pyproject.toml; the help data is loaded by the frontend.
_MUST_SHIP = (
    "__init__.py", "pyproject.toml", "tex_help.json", "LICENSE", "README.md",
    "Function-Reference.md", "Error-Codes.md", "LANGUAGE.md",
    "js/tex_extension.js", "examples/grade.tex", "stock/grade.textool",
)

# The only pattern form comfy-cli's gitwildmatch and this file agree on by construction:
# `name/` = "a directory called `name`, at any depth" (no leading slash, no glob, no negation).
_DIR_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+/$")

# The scanner's families, as line regexes over the RAW text of EVERY shipped file that decodes
# as UTF-8 — docstrings, comments and prose included, because the scanner reads bytes, not ASTs
# (0.36.1 was flagged on two DOCSTRINGS in tex_runtime/compiled.py). Markdown IS censused, and
# the reason is a measurement that overturned the previous one: 0.36.3 (2026-09-19, read via
# /versions?include_status_reason=true): 4 of 23 registry findings were in Markdown —
# CHANGELOG.md:16 (`contains_rm_rf`, prose quoting a literal `rm -rf /`) and SECURITY.md:27/28/30
# (`subprocess.run(` / `os.environ[...] =` / `node.connect(` quoted in the finding table's
# cells). The scanner reads prose; 0.36.1's `.md` files came back clean ONLY because they lacked
# the exact byte patterns, not because `.md` was out of the scanner's reach — the inference that
# narrowed this census to `.py`/`.js` was wrong. So a shipped document may DESCRIBE a mechanism
# ("a subprocess call", "an environment write") but must not spell a matched call shape or an
# attack string. Each call-shaped family is anchored so an identifier that merely contains the
# word does not count: `compile(` must not match `compile_program(` / `recompile(` /
# `re.compile(`, and `marshal.loads(` is spelled out so `tex_marshalling` does not match.
# `rm_rf` and `dunder_import` are the two families the registry matched (`contains_rm_rf`,
# `$import_func_direct`) that this census did not cover before PUB-2. Known, deliberate gap:
# `network` does not census the scanner's bare `.connect(` / `.bind(` strings (`$socket3` /
# `$socket4`), which match LiteGraph's link API in js/ on every version — those are answered in
# SECURITY.md's table, not by a pin.
_FAMILIES = {
    "env_read":      re.compile(r"os\.environ\.get\(|os\.environ\[|os\.getenv\("),
    "subprocess":    re.compile(r"subprocess\."),
    "os_system":     re.compile(r"os\.system\("),
    "exec":          re.compile(r"(?<![A-Za-z0-9_.])exec\("),
    "eval":          re.compile(r"(?<![A-Za-z0-9_.])eval\("),
    "compile":       re.compile(r"(?<![A-Za-z0-9_.])compile\("),
    "marshal_loads": re.compile(r"marshal\.loads\("),
    "pickle_load":   re.compile(r"pickle\.load"),
    "network":       re.compile(r"urlopen\(|\brequests\.[a-z_]+\(|http\.client|\bsocket\.[a-z_]+\("),
    "rm_rf":         re.compile(r"rm\s+-rf\b"),
    "dunder_import": re.compile(r"__import__\("),
}

# Measured over EVERY shipped UTF-8 text file (git-tracked minus .comfyignore; binaries skipped).
# These pins are HIGHER than the `.py`/`.js`-only pins that preceded them (env_read 24→26,
# subprocess 1→2, exec 5→11, compile 11→15, marshal_loads 1→2, pickle_load 4→8; `rm_rf` and
# `dunder_import` are new) because Markdown came back into scope — every added count is a prose
# site in AGENTS.md, CHANGELOG.md, DEVELOPMENT.md or SECURITY.md, and NO code site was added. That
# is a scope correction with the 0.36.3 measurement behind it (see `_FAMILIES`), not a raised bar:
# the pins were re-measured, not copied. The pin moves DOWN freely and reds until it does; it
# moves UP only as a release decision that names the new finding — every shipped finding is
# justified to the registry reviewer in writing, so a new one is a new paragraph there, never a
# reflex here.
_SURFACE_PINS = {
    "env_read": 26,
    "subprocess": 2,
    "os_system": 0,
    "exec": 11,
    "eval": 0,
    "compile": 15,
    "marshal_loads": 2,
    "pickle_load": 8,
    "network": 0,
    "rm_rf": 1,
    "dunder_import": 1,
}

# Arm (c)'s one allowed reach into an ignored directory: the triton lane's optional delegate,
# guarded by the directory check `test_pub1_validate_hw_triton_lane_skips_without_benchmarks`
# pins. A second such site is a second entry here, with its own guard and its own pin.
_ALLOWED_IGNORED_IMPORTS = {("tex_validate_hw.py", "triton_validation")}


# ── the mirror ───────────────────────────────────────────────────────────────

def _git_tracked() -> list[str]:
    """`git ls-files` under the package, `/`-separated, exactly what comfy-cli zips from."""
    out = subprocess.check_output(["git", "-C", str(_PKG), "ls-files", "-z"])
    return [p for p in out.decode("utf-8").split("\0") if p]


def _parse_comfyignore() -> tuple[list[str], list[str]]:
    """(directory names, offending lines). Same line filter as comfy-cli's
    `_load_comfyignore_spec`: strip, drop blanks, drop `#` comments."""
    names, bad = [], []
    for raw in _COMFYIGNORE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if _DIR_PATTERN.match(line):
            names.append(line[:-1])
        else:
            bad.append(line)
    return names, bad


def _is_ignored(rel_path: str, names: list[str]) -> bool:
    """gitwildmatch `name/`: the path has a DIRECTORY component called `name`, at any depth."""
    return any(part in names for part in rel_path.split("/")[:-1])


def _shipped(names: list[str]) -> list[str]:
    return [p for p in _git_tracked() if not _is_ignored(p, names)]


# ── (a) coverage of the tree ─────────────────────────────────────────────────

def test_pub1_every_top_level_directory_is_ignored_or_allowlisted(r: SubTestResult):
    print("\n--- PUB-1 (a): every top-level directory is ignored or allowlisted; .comfyignore is mirrorable ---")
    try:
        if not _COMFYIGNORE.is_file():
            r.fail("PUB-1 .comfyignore", f"missing: {_COMFYIGNORE} — without it the whole tree "
                   "(tests/, benchmarks/, tools/, ...) is the archive")
            return
        names, bad = _parse_comfyignore()
        if bad:
            r.fail("PUB-1 .comfyignore form", "pattern(s) not of the form `name/` — this file "
                   "mirrors only that subset, so it cannot vouch for: " + ", ".join(repr(b) for b in bad))
            return
        tracked = _git_tracked()
        top_dirs = sorted({p.split("/", 1)[0] for p in tracked if "/" in p})
        ignored = set(names)
        problems = []
        for d in top_dirs:
            if d in ignored and d in _SHIP_DIRS:
                problems.append(f"{d}/ is both ignored and allowlisted")
            elif d not in ignored and d not in _SHIP_DIRS:
                problems.append(f"{d}/ is a tracked top-level directory that is neither in "
                                ".comfyignore nor in _SHIP_DIRS — it would ship")
        for d in sorted(_SHIP_DIRS - set(top_dirs)):
            problems.append(f"_SHIP_DIRS names {d}/ but no tracked file lives there (stale)")
        for d in sorted(ignored - set(top_dirs)):
            problems.append(f".comfyignore names {d}/ but no tracked file lives there (stale)")
        shipped = set(_shipped(names))
        for must in _MUST_SHIP:
            if must not in shipped:
                problems.append(f"{must} has a shipped reader and is not in the archive")
        if problems:
            r.fail("PUB-1 tree coverage", "\n  ".join(problems))
        else:
            r.ok(f"{len(top_dirs)} top-level dirs: {len(ignored)} ignored, "
                 f"{len(_SHIP_DIRS)} shipped; {len(shipped)}/{len(tracked)} tracked files ship")
    except Exception as e:
        r.fail("PUB-1 (a)", f"{type(e).__name__}: {e}")


# ── (b) the surface ratchet ──────────────────────────────────────────────────

def _census(paths: list[str]) -> dict[str, list[str]]:
    """Every shipped file the scanner can read as text: no suffix filter (the scanner has none —
    measured above), only a UTF-8 decode; a file that does not decode is a binary (fonts,
    images) and is skipped."""
    sites = collections.defaultdict(list)
    for rel in paths:
        try:
            text = (_PKG / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue   # binary (fonts, images) or gone
        for i, line in enumerate(text.splitlines(), 1):
            for fam, rx in _FAMILIES.items():
                if rx.search(line):
                    sites[fam].append(f"{rel}:{i}: {line.strip()[:100]}")
    return sites


def test_pub1_shipped_surface_ratchet(r: SubTestResult):
    print("\n--- PUB-1 (b): scanner-family sites over EVERY shipped text file stay at their pins ---")
    try:
        names, bad = _parse_comfyignore() if _COMFYIGNORE.is_file() else ([], [])
        if bad:
            r.fail("PUB-1 ratchet", "cannot census: .comfyignore is not mirrorable (see arm a)")
            return
        sites = _census(_shipped(names))
        if set(_SURFACE_PINS) != set(_FAMILIES):
            r.fail("PUB-1 ratchet", "every family needs a pin and every pin a family")
            return
        up, down = [], []
        for fam, pin in _SURFACE_PINS.items():
            n = len(sites[fam])
            if n > pin:
                up.append(f"{fam}: {n} sites, pinned {pin} — a NEW scanner finding in the "
                          f"archive. Every site of the family:\n      " + "\n      ".join(sites[fam]))
            elif n < pin:
                down.append(f"{fam}: {n} sites, pinned {pin} — re-pin DOWN to {n}")
        if up:
            r.fail("PUB-1 surface ratchet (new site)", "\n  ".join(up))
        elif down:
            r.fail("PUB-1 surface ratchet (stale pin)", "\n  ".join(down))
        else:
            r.ok("shipped surface at its pins: "
                 + ", ".join(f"{k}={v}" for k, v in _SURFACE_PINS.items()))
    except Exception as e:
        r.fail("PUB-1 (b)", f"{type(e).__name__}: {e}")


# ── (c) no shipped module imports from an ignored directory ──────────────────

def _import_targets(tree: ast.AST):
    """Yield (top-level module name, lineno) for every import in a module, with a relative
    import resolved to its first component and `TEX_Wrangle.x` reduced to `x`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node.module.split(".")[0], node.lineno
            elif node.level:            # `from . import x` — x is the module
                for alias in node.names:
                    yield alias.name.split(".")[0], node.lineno


def test_pub1_no_shipped_module_imports_an_ignored_directory(r: SubTestResult):
    print("\n--- PUB-1 (c): no shipped module imports from an ignored directory (AST) ---")
    try:
        names, bad = _parse_comfyignore() if _COMFYIGNORE.is_file() else ([], [])
        if bad:
            r.fail("PUB-1 imports", "cannot scan: .comfyignore is not mirrorable (see arm a)")
            return
        tracked = _git_tracked()
        ignored_dirs = set(names)
        # a bare module name that lives in an ignored directory (`import triton_validation`
        # after a sys.path insert is still an import from benchmarks/)
        ignored_modules = {Path(p).stem for p in tracked
                           if p.endswith(".py") and _is_ignored(p, names)}
        hits = []
        for rel in _shipped(names):
            if not rel.endswith(".py"):
                continue
            tree = ast.parse((_PKG / rel).read_text(encoding="utf-8"), filename=rel)
            for name, lineno in _import_targets(tree):
                if name == "TEX_Wrangle":
                    continue   # absolute self-imports are checked by their next component below
                if name in ignored_dirs or name in ignored_modules:
                    if (rel, name) in _ALLOWED_IGNORED_IMPORTS:
                        continue
                    hits.append(f"{rel}:{lineno}: imports `{name}` (an ignored directory / a "
                                "module that lives in one)")
            # `TEX_Wrangle.tests.x` / `from TEX_Wrangle.benchmarks import`
            for node in ast.walk(tree):
                mod = None
                if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    mod = node.module
                elif isinstance(node, ast.Import):
                    mod = ".".join(a.name for a in node.names)
                if mod and mod.startswith("TEX_Wrangle."):
                    nxt = mod.split(".")[1]
                    if nxt in ignored_dirs:
                        hits.append(f"{rel}:{node.lineno}: imports `{mod}`")
        if hits:
            r.fail("PUB-1 ignored-directory import", "\n  ".join(hits))
        else:
            r.ok(f"no shipped module imports from {sorted(ignored_dirs)} "
                 f"({len(_ALLOWED_IGNORED_IMPORTS)} named, guarded exception)")
    except Exception as e:
        r.fail("PUB-1 (c)", f"{type(e).__name__}: {e}")


# ── the runtime seam: validate-hw's triton lane without benchmarks/ ──────────

def test_pub1_validate_hw_triton_lane_skips_without_benchmarks(r: SubTestResult):
    print("\n--- PUB-1: validate-hw's triton lane SKIPs when benchmarks/ is absent (installed node) ---")
    from TEX_Wrangle import tex_validate_hw as vh
    real_root = vh._ROOT
    try:
        with tempfile.TemporaryDirectory() as td:
            # an installed node: the package root exists, benchmarks/ under it does not
            vh._ROOT = Path(td)
            path_before = list(sys.path)
            try:
                out = vh._lane_triton()
            except Exception as e:
                r.fail("PUB-1 triton lane", f"raised {type(e).__name__}: {e} — an installed "
                       "node has no benchmarks/; the lane must SKIP like the GPU-only lanes")
                return
            finally:
                vh._ROOT = real_root
            if not (isinstance(out, dict) and out.get("status") == "skipped"
                    and "benchmarks" in str(out.get("reason", ""))):
                r.fail("PUB-1 triton lane shape", f"expected {{'status': 'skipped', 'reason': "
                       f"...benchmarks...}}, got {out!r}")
                return
            if sys.path != path_before:
                r.fail("PUB-1 triton lane sys.path", "the lane inserted a path it did not use")
                return
        # control: the development checkout still reaches the delegate (a dict with a status,
        # 'skipped' without Triton — test_hw3_triton_validation_skips owns that verdict).
        ctl = vh._lane_triton()
        if not (isinstance(ctl, dict) and "status" in ctl):
            r.fail("PUB-1 triton lane control", f"development checkout: {ctl!r}")
            return
        r.ok("triton lane SKIPs with a reason when benchmarks/ is absent; the checkout still delegates")
    finally:
        vh._ROOT = real_root
