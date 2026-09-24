"""TRK-126 — `compat_corpus.freeze(version, only={name, ...})` corrects specific
existing rows in an archived version without touching anything else.

Before this: the documented correction procedure was "delete the file and `freeze()`
it again." Followed on any version but the newest, that recomputes hashes for EVERY
program currently in `examples/*.tex` -- today's tree, not the tree that version was
frozen against. Tried on an old archive it silently ADDED a key (`aov_relight`) that
did not exist when that version was genuinely frozen, and
`test_v023_phase1.py::test_lang3_compat_corpus` cannot catch it (an older version is
explicitly allowed to cover FEWER programs, so a silently WIDENED snapshot passes).

`only=` closes it: it requires the file to already exist (never re-creates from
scratch), requires every named row to already be a key in it (never adds a row), and
recomputes ONLY those rows, carrying everything else over byte-for-byte. This file
pins that write discipline the same way `test_v034_r2_archive.py` pins the
append-only discipline of the plain path, stubbing the expensive hashing half
(`_compute_selected` / `compute_all`) so the row is about the archive's file
discipline, not about running the interpreter.
"""
import json
import os
import shutil
import tempfile

import compat_corpus


def _isolated_archive():
    tmp = tempfile.mkdtemp(prefix="tex_trk126_")
    return tmp


def test_trk126_only_requires_the_file_to_already_exist(r):
    tmp = _isolated_archive()
    real_archive = compat_corpus._ARCHIVE
    try:
        compat_corpus._ARCHIVE = tmp
        raised = None
        try:
            compat_corpus.freeze("0.99", only={"p0"})
        except FileNotFoundError as e:
            raised = e
        if raised is None:
            r.fail("TRK-126 only= requires an existing file",
                   "freeze(only=...) SUCCEEDED against a version that was never frozen "
                   "-- only= must never be a back door to creating a fresh archive")
            return
        assert not os.path.exists(os.path.join(tmp, "0.99.json")), \
            "a refused only= correction must not have written anything"
        r.ok("TRK-126: only= refuses when the archived version does not exist yet")
    except Exception as e:
        r.fail("TRK-126 only= existing-file guard", f"{type(e).__name__}: {e}")
    finally:
        compat_corpus._ARCHIVE = real_archive
        shutil.rmtree(tmp, ignore_errors=True)


def test_trk126_only_requires_every_named_row_to_already_exist(r):
    tmp = _isolated_archive()
    real_archive, real_compute = compat_corpus._ARCHIVE, compat_corpus.compute_all
    try:
        compat_corpus._ARCHIVE = tmp
        compat_corpus.compute_all = lambda: {"p0": "h0", "p1": "h1"}
        compat_corpus.freeze("0.99")

        raised = None
        try:
            compat_corpus.freeze("0.99", only={"aov_relight"})   # never in the archive
        except KeyError as e:
            raised = e
        if raised is None:
            r.fail("TRK-126 only= row-must-exist guard",
                   "freeze(only={'aov_relight'}) SUCCEEDED -- only= must never ADD a "
                   "row that was not already frozen, which is exactly the silent "
                   "widening this filter exists to prevent")
            return
        assert "aov_relight" in str(raised)
        payload = json.load(open(os.path.join(tmp, "0.99.json"), encoding="utf-8"))
        assert set(payload["hashes"]) == {"p0", "p1"}, \
            "the refused correction must not have widened the archive"
        r.ok("TRK-126: only= refuses to add a row that isn't already frozen")
    except Exception as e:
        r.fail("TRK-126 only= row-must-exist guard", f"{type(e).__name__}: {e}")
    finally:
        compat_corpus._ARCHIVE, compat_corpus.compute_all = real_archive, real_compute
        shutil.rmtree(tmp, ignore_errors=True)


def test_trk126_only_corrects_named_rows_and_preserves_every_other_row(r):
    tmp = _isolated_archive()
    real_archive, real_compute, real_selected = (
        compat_corpus._ARCHIVE, compat_corpus.compute_all, compat_corpus._compute_selected)
    try:
        compat_corpus._ARCHIVE = tmp
        compat_corpus.compute_all = lambda: {f"p{i}": f"h{i}" for i in range(20)}
        compat_corpus.freeze("0.99")

        compat_corpus._compute_selected = lambda names: {n: f"CORRECTED-{n}" for n in names}
        data = compat_corpus.freeze("0.99", only={"p5", "p9"})

        assert data["hashes"]["p5"] == "CORRECTED-p5"
        assert data["hashes"]["p9"] == "CORRECTED-p9"
        for i in range(20):
            name = f"p{i}"
            if name in ("p5", "p9"):
                continue
            assert data["hashes"][name] == f"h{i}", \
                f"{name} moved but was not named in only= -- unrelated growth leaked in"
        assert len(data["hashes"]) == 20, "only= must never change the row COUNT"
        assert data["language_version"] == "0.99"

        on_disk = json.load(open(os.path.join(tmp, "0.99.json"), encoding="utf-8"))
        assert on_disk == data, "the corrected payload must be what was written to disk"
        r.ok("TRK-126: only= corrects exactly the named rows and leaves every other "
             "row byte-for-byte unchanged")
    except Exception as e:
        r.fail("TRK-126 only= scoped correction", f"{type(e).__name__}: {e}")
    finally:
        (compat_corpus._ARCHIVE, compat_corpus.compute_all,
         compat_corpus._compute_selected) = real_archive, real_compute, real_selected
        shutil.rmtree(tmp, ignore_errors=True)


def test_trk126_only_can_correct_a_version_that_is_not_the_newest(r):
    """The tracker row's own motivating example corrected `0.23.json` while `0.24.json`
    already existed -- i.e. the version being fixed was NOT the newest archive. `only=`
    must keep that legitimate use working, and must never touch a sibling version."""
    tmp = _isolated_archive()
    real_archive, real_compute, real_selected = (
        compat_corpus._ARCHIVE, compat_corpus.compute_all, compat_corpus._compute_selected)
    try:
        compat_corpus._ARCHIVE = tmp
        compat_corpus.compute_all = lambda: {"p0": "old0", "p1": "old1"}
        compat_corpus.freeze("0.23")
        compat_corpus.compute_all = lambda: {"p0": "new0", "p1": "new1", "p2": "new2"}
        compat_corpus.freeze("0.24")

        compat_corpus._compute_selected = lambda names: {n: "FIXED" for n in names}
        compat_corpus.freeze("0.23", only={"p0"})   # correct the OLDER, non-newest file

        older = json.load(open(os.path.join(tmp, "0.23.json"), encoding="utf-8"))
        newer = json.load(open(os.path.join(tmp, "0.24.json"), encoding="utf-8"))
        assert older["hashes"]["p0"] == "FIXED"
        assert older["hashes"]["p1"] == "old1", "the untouched row in 0.23 must survive"
        assert set(older["hashes"]) == {"p0", "p1"}, \
            "correcting 0.23 must not pull in p2, which only exists in 0.24's snapshot"
        assert newer["hashes"] == {"p0": "new0", "p1": "new1", "p2": "new2"}, \
            "a sibling version must be completely untouched by another version's correction"
        r.ok("TRK-126: only= corrects a non-newest archived version without touching "
             "a sibling version or pulling in its rows")
    except Exception as e:
        r.fail("TRK-126 only= on a non-newest version", f"{type(e).__name__}: {e}")
    finally:
        (compat_corpus._ARCHIVE, compat_corpus.compute_all,
         compat_corpus._compute_selected) = real_archive, real_compute, real_selected
        shutil.rmtree(tmp, ignore_errors=True)


def test_trk126_plain_freeze_without_only_is_still_append_only(r):
    """`only=` is additive: the pre-existing R2-archive guarantee (freeze() may only
    ADD a version that is not there yet) is unchanged when `only` is not given."""
    tmp = _isolated_archive()
    real_archive, real_compute = compat_corpus._ARCHIVE, compat_corpus.compute_all
    try:
        compat_corpus._ARCHIVE = tmp
        compat_corpus.compute_all = lambda: {"p0": "h0"}
        compat_corpus.freeze("0.99")
        raised = None
        try:
            compat_corpus.freeze("0.99")
        except FileExistsError as e:
            raised = e
        if raised is None:
            r.fail("TRK-126 plain freeze still append-only",
                   "re-freezing an already-archived version with no only= SUCCEEDED")
            return
        assert "append-only" in str(raised)
        r.ok("TRK-126: freeze() with no only= is still append-only")
    except Exception as e:
        r.fail("TRK-126 plain freeze still append-only", f"{type(e).__name__}: {e}")
    finally:
        compat_corpus._ARCHIVE, compat_corpus.compute_all = real_archive, real_compute
        shutil.rmtree(tmp, ignore_errors=True)
