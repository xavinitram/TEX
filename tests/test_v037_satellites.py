"""v0.37 DATA-6 — the language-version satellites a machine can read track `LANGUAGE_VERSION`.

The language version lives in ten places across nine files. v0.35's CF-7 pinned the one in
the JS publish manifest; this pins the rest a test can read: every shipped `stock/*.textool`
manifest's `tex_language`, and the `_LANG` literal in the generator that writes those five
files (a bump that edits the manifests by hand and not the generator regresses on the next
regeneration). Eight of the ten had no test at the 0.23 -> 0.24 bump. The failure is quiet:
a stock tool declaring an older language than the engine that shipped it is exactly the
compatibility claim LANG-3 makes, silently wrong. `LANGUAGE.md`'s two prose copies stay on
review.
"""
import glob
import json
import os
import pathlib
import re

_PKG = pathlib.Path(__file__).resolve().parent.parent


def test_v037_language_version_satellites(r):
    from TEX_Wrangle.tex_api import LANGUAGE_VERSION

    try:
        # Strictly numeric: `tex_api._ver_tuple` int-parses each component (tolerant of a
        # non-numeric suffix, TRK-144), so "0.24-planes" would silently parse as (0, 24) —
        # indistinguishable from a clean "0.24" — and break every version comparison quietly.
        assert re.fullmatch(r"[0-9]+(\.[0-9]+)+", LANGUAGE_VERSION), LANGUAGE_VERSION

        stock = sorted(glob.glob(str(_PKG / "stock" / "*.textool")))
        assert stock, "no stock/*.textool found"
        bad = {}
        for path in stock:
            with open(path, encoding="utf-8") as f:
                declared = json.load(f).get("tex_language")
            if declared != LANGUAGE_VERSION:
                bad[os.path.basename(path)] = declared
        assert not bad, (f"stock tools declare tex_language {bad}, "
                         f"but tex_api.LANGUAGE_VERSION is {LANGUAGE_VERSION!r}")

        gen = (_PKG / "tools" / "gen_stock_tools.py").read_text(encoding="utf-8")
        m = re.search(r'^_LANG\s*=\s*"([0-9.]+)"', gen, re.M)
        assert m, "no `_LANG = \"X.Y\"` literal in tools/gen_stock_tools.py"
        assert m.group(1) == LANGUAGE_VERSION, (
            f"tools/gen_stock_tools.py _LANG is {m.group(1)!r}, "
            f"but tex_api.LANGUAGE_VERSION is {LANGUAGE_VERSION!r}")
        r.ok(f"{len(stock)} stock tools + the generator's _LANG track "
             f"LANGUAGE_VERSION ({LANGUAGE_VERSION})")
    except Exception as e:
        r.fail("v037 language-version satellites", f"{type(e).__name__}: {e}")
