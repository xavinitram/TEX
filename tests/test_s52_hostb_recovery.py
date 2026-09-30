"""v0.52 sweep: journal compaction against concurrent appends, and the temp reclaim's age gate."""
import os
import threading
import time

from TEX_Wrangle import tex_recovery as R


def test_a_record_appended_during_compaction_survives(tmp_path, monkeypatch):
    j = R.Journal(str(tmp_path / "snap.json"))
    for i in range(3):
        assert j.append({"i": i})
    real = R.atomic_write
    late = threading.Thread(target=lambda: j.append({"i": "late"}))

    def write_with_a_racing_append(path, data, **kw):
        late.start()
        late.join(0.3)               # unlocked, the append lands here, before the replace
        return real(path, data, **kw)

    monkeypatch.setattr(R, "atomic_write", write_with_a_racing_append)
    j.drop_prefix(2)
    late.join(10)
    assert [r["i"] for r in j.replay()] == [2, "late"]


def test_clear_and_append_share_the_lock(tmp_path):
    j = R.Journal(str(tmp_path / "snap.json"))
    j.append({"i": 1})
    j.drop_prefix(1)                 # empty remainder clears through the re-entrant lock
    assert j.replay() == [] and not j.exists()
    assert j.append({"i": 2}) and j.count() == 1


def test_sweep_temps_leaves_a_fresh_temp_alone(tmp_path):
    fresh = tmp_path / (R.TMP_PREFIX + "live.tmp")
    old = tmp_path / (R.TMP_PREFIX + "dead.tmp")
    other = tmp_path / "keep.tmp"
    for f in (fresh, old, other):
        f.write_bytes(b"x")
    long_ago = time.time() - 3600
    os.utime(old, (long_ago, long_ago))
    os.utime(other, (long_ago, long_ago))
    assert R.sweep_temps(str(tmp_path)) == 1
    assert fresh.exists() and not old.exists() and other.exists()
    assert R.sweep_temps(str(tmp_path), min_age_s=0.0) == 1 and not fresh.exists()


def test_sweep_temps_with_no_age_floor_takes_a_temp_stamped_ahead_of_the_clock(tmp_path):
    # A just-written file's mtime can read a tick past time.time() on Windows; with no age
    # floor it must still go.
    ahead = tmp_path / (R.TMP_PREFIX + "ahead.tmp")
    ahead.write_bytes(b"x")
    soon = time.time() + 5.0
    os.utime(ahead, (soon, soon))
    assert R.sweep_temps(str(tmp_path), min_age_s=0.0) == 1 and not ahead.exists()
