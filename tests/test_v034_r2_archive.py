"""v0.34 R2-archive — the compat-corpus freeze machinery becomes append-only.

Doc 40 §1 / doc 41 §3.4. Before this, `regen()` overwrote the whole golden file, so "old
goldens are immutable" was a convention enforced by review — and the action that breaks it
(re-freezing after an unintended pixel change) is also the action that turns the suite
green again. These rows pin the machinery that replaces the convention.

The item lands a release EARLY, in v0.34 rather than with v0.35's first new freeze,
precisely so it lands while it is neutral: there is exactly one archived version, so the
corpus test does what it always did. `test_v034_r2_neutrality` is that proof.
"""
import json
import os
import shutil
import tempfile

import compat_corpus


def test_v034_r2_freeze_may_only_add(r):
    """`freeze()` refuses to rewrite a version that is already frozen.

    THE ROW THAT IS THE ITEM. Everything else here is plumbing; this is the property
    `regen()` did not have, and the reason a corpus can now be called a proof."""
    tmp = tempfile.mkdtemp(prefix="tex_r2_")
    real_archive, real_compute = compat_corpus._ARCHIVE, compat_corpus.compute_all
    try:
        compat_corpus._ARCHIVE = tmp
        # The corpus itself is 129 programs on the interpreter; this row is about the
        # archive's write discipline, not about hashing, so stub the expensive half.
        compat_corpus.compute_all = lambda: {f"p{i}": f"h{i}" for i in range(120)}

        compat_corpus.freeze("0.99")
        assert os.path.exists(os.path.join(tmp, "0.99.json")), "freeze wrote nothing"

        raised = None
        try:
            compat_corpus.freeze("0.99")
        except FileExistsError as e:
            raised = e
        if raised is None:
            r.fail("R2: freeze may only add",
                   "re-freezing an already-archived version SUCCEEDED — the archive is "
                   "still rewritable, which is the whole defect R2-archive closes")
            return
        assert "append-only" in str(raised), str(raised)

        # ...and the refusal left the archived bytes alone, rather than truncating and
        # then failing (the failure mode that would be worse than the rewrite).
        payload = json.load(open(os.path.join(tmp, "0.99.json"), encoding="utf-8"))
        assert len(payload["hashes"]) == 120, payload["hashes"]

        compat_corpus.freeze("1.0")
        assert compat_corpus.archived_versions() == ["0.99", "1.0"], \
            compat_corpus.archived_versions()
        r.ok("R2: freeze ADDS a new version and refuses to rewrite an existing one")
    except Exception as e:
        r.fail("R2 freeze discipline", f"{type(e).__name__}: {e}")
    finally:
        compat_corpus._ARCHIVE, compat_corpus.compute_all = real_archive, real_compute
        shutil.rmtree(tmp, ignore_errors=True)


def test_v034_r2_versions_sort_numerically(r):
    """`0.9` sorts BELOW `0.10`, and `load_goldens()` returns the NEWEST.

    A lexical sort is right for every version string the archive will hold for years and
    then wrong exactly once, at 0.9 -> 0.10 — at which point "the newest archive" silently
    becomes the wrong file and the coverage assertion checks the wrong set.
    `tex_api._ver_tuple` int-parses components for the same reason."""
    tmp = tempfile.mkdtemp(prefix="tex_r2_")
    real_archive = compat_corpus._ARCHIVE
    try:
        compat_corpus._ARCHIVE = tmp
        for v in ("0.10", "0.9", "0.23", "1.0"):
            with open(os.path.join(tmp, f"{v}.json"), "w", encoding="utf-8") as f:
                json.dump({"language_version": v, "hashes": {"only": v}}, f)
        got = compat_corpus.archived_versions()
        assert got == ["0.9", "0.10", "0.23", "1.0"], got
        assert compat_corpus.load_goldens()["language_version"] == "1.0"
        assert compat_corpus.load_goldens("0.9")["language_version"] == "0.9"
        r.ok("R2: archived versions sort numerically; load_goldens() defaults to the newest")
    except Exception as e:
        r.fail("R2 version ordering", f"{type(e).__name__}: {e}")
    finally:
        compat_corpus._ARCHIVE = real_archive
        shutil.rmtree(tmp, ignore_errors=True)


def test_v034_r2_neutrality(r):
    """The shipped archive holds exactly ONE version, and it is the language's own.

    This is the neutrality proof doc 41 §3.4 asks for: with one archived version the corpus
    test runs the same single comparison it ran before R2-archive, so the mechanism changed
    the machinery and nothing else. When v0.35 freezes 0.24 this row's count becomes 2 and
    the assertion below is what makes someone update it deliberately."""
    try:
        from TEX_Wrangle.tex_api import LANGUAGE_VERSION
        versions = compat_corpus.archived_versions()
        assert versions == [LANGUAGE_VERSION], \
            f"archived={versions}, LANGUAGE_VERSION={LANGUAGE_VERSION}"
        payload = compat_corpus.load_goldens()
        assert payload["language_version"] == LANGUAGE_VERSION, payload["language_version"]
        n = len(payload["hashes"])
        assert n >= 100, f"only {n} goldens"
        # The flat file the archive replaced must be GONE, not shadowed: two sources of
        # goldens is how one of them silently stops being read.
        legacy = os.path.join(compat_corpus._HERE, "compat_corpus_goldens.json")
        assert not os.path.exists(legacy), f"the pre-R2 flat golden file is still there: {legacy}"
        r.ok(f"R2: one frozen version ({LANGUAGE_VERSION}, {n} programs), flat file retired")
    except Exception as e:
        r.fail("R2 neutrality", f"{type(e).__name__}: {e}")


def test_v034_r2_regen_is_gone(r):
    """`regen()` no longer exists.

    Leaving it as an alias for `freeze()` would be kinder to a stale caller and would also
    leave the overwrite one keystroke away under the name everyone already knows. An
    AttributeError naming `freeze` is the better failure."""
    try:
        assert not hasattr(compat_corpus, "regen"), \
            "compat_corpus.regen still exists — the overwrite path is still reachable"
        assert callable(compat_corpus.freeze)
        r.ok("R2: regen() is gone; freeze() is the only writer")
    except Exception as e:
        r.fail("R2 regen removal", f"{type(e).__name__}: {e}")
