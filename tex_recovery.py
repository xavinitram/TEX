"""
ENG-13 — the engine recovery contract.

**The guarantee.** An engine-process crash loses **at most the in-flight cook**. Everything
the engine learned before that cook — compile artifacts, tier verdicts, graph-capturability
verdicts, spilled frames — is on disk and is adopted by the next process (or, without a
restart at all, by `EngineSession.reattach()`).

**What was already true, and what was not.** Doc 39 §2 is right that "everything is already
disk-backed": the program cache, `autotier.json`, `xfer.json`, `warm_state.json` and the
CACHE-2 spill directory all persist, and all five wrote through their own private
`tmp + os.replace`, which is atomic against a torn read. Two things were missing, and they are
this item:

1. **A throttle is not a durability boundary.** `warm_state` coalesces writes over a 5-second
   window and relies on `atexit` to flush the tail. `atexit` does not run on `os._exit`, a
   SIGKILL, or a hard crash — so up to five seconds of learned verdicts were lost, which is
   not "at most the in-flight cook". Fixed with a JOURNAL (below): the coalescing window keeps
   its performance job, and durability moves to an append that costs microseconds.

2. **`os.replace` is atomic, not durable.** Atomicity survives a process crash (the page cache
   outlives the process). It does not survive a machine crash: the rename can land while the
   data is still dirty, leaving the new name pointing at a short or zero-length file. One
   `fsync` before the rename closes that, and `atomic_write` below is the single place it now
   happens — replacing five near-identical private copies, and applying the flush where it
   earns its cost (see its `fsync=` note: a re-derivable measurement cache does not).

   SCOPE OF "the single place", stated exactly because it was once broader than it was true:
   `atomic_write` is the single place the *fsync-before-rename* discipline happens. Temp-file
   CREATION has two further callers that do not route through it (`tex_snippets`, `tex_tool`) —
   they now share `bounded_mkstemp` instead, which is the single place the P0-7 retry bound
   lives. Two disciplines, two single places; neither claims the other's callers.

   HONEST PLATFORM NOTE: the fsync makes the *file's* bytes durable. Making the *rename*
   durable additionally needs an fsync of the containing directory, which POSIX supports and
   Windows does not. So on Windows the machine-crash guarantee is "the file is never torn",
   not "the rename is never lost". The process-crash guarantee — the one ENG-13 states — holds
   everywhere, and did before the fsync.

**Journal, then compact.** The pattern, for state that is learned incrementally and rewritten
wholesale:

    load()     read the snapshot, then replay the journal on top of it
    learn()    append one line to the journal (write + flush; ~µs, off the cook's hot path)
    persist()  write the snapshot atomically, THEN clear the journal

The ordering in `persist()` is the whole correctness argument, and it is deliberately the
lossy-safe one: a crash between the snapshot and the clear replays already-snapshotted
records, which is idempotent; the reverse order would drop them. Verdicts are idempotent by
construction — `fp -> (capturable, op_count)` is a pure function of the AST and the arch — so
double-replay is a no-op, not a merge conflict.

**Reattach.** `reattach()` restores an engine's warm state without a process restart: tier
verdicts, capturability verdicts, and a `ResultCache`'s view of its own disk tier. A host
whose engine thread died, or which is adopting a cache directory another process wrote, calls
it and keeps going. Reached as `EngineSession.reattach()`.
"""
from __future__ import annotations

import json
import os
import tempfile

#: Every temp this module mints carries this prefix, so a crash-orphaned one is findable.
#: The private writers it replaced used derived names (`<key>.frame.tmp`), which were
#: self-limiting — at most one stale temp per key, overwritten by the next write. A random
#: `mkstemp` name leaks a NEW file per crash, and the reclaim paths all filter on the real
#: suffix, so without a sweepable prefix they accumulate without bound and invisibly to the
#: disk budget. `sweep_temps` is the reclaim.
TMP_PREFIX = ".tex-tmp-"


def sweep_temps(directory: str) -> int:
    """Delete this module's orphaned temps under `directory`. Returns how many went.

    Only ever removes files this module minted (both the prefix AND the suffix must match),
    so it can be pointed at a shared cache directory without touching a peer's data."""
    n = 0
    try:
        with os.scandir(directory) as it:
            for entry in it:
                if entry.name.startswith(TMP_PREFIX) and entry.name.endswith(".tmp"):
                    try:
                        os.remove(entry.path)
                        n += 1
                    except OSError:
                        pass
    except OSError:
        pass
    return n


# ── the one durable atomic write ─────────────────────────────────────────────


#: How many names `bounded_mkstemp` will try before giving up. `tempfile.TMP_MAX` is
#: 2,147,483,647 — a number chosen for "we will never realistically collide", which is the
#: right bound for COLLISIONS and catastrophically wrong for a directory that rejects every
#: name. Two is enough to ride out a genuine collision; anything beyond that is a wall.
_MKSTEMP_TRIES = 2


def bounded_mkstemp(**kw):
    """`tempfile.mkstemp(**kw)` with a BOUNDED retry. Returns `(fd, path)`; raises otherwise.

    THE BUG THIS EXISTS FOR. On Windows, `mkstemp` retries `TMP_MAX` times on `PermissionError`,
    because on nt that error usually means "name collided with a directory". But an existing
    directory whose ACL denies write raises `PermissionError` for EVERY name — and
    `os.access(dir, W_OK)`, the obvious pre-check, returns True there because it only consults
    FILE_ATTRIBUTE_READONLY and never the ACL. So the loop runs ~2.1 billion times: `put()`
    hangs effectively forever, on the ONE path a ComfyUI user can reach.

    The spill contract is best-effort — "a failed spill just drops the frame, the cook
    reproduces it". A hang is not a degraded spill; it is the opposite of one. Bounding the
    retry converts the unreachable directory into the exception the callers already handle,
    and names the directory so the failure is diagnosable rather than mysterious.
    """
    last = None
    for _ in range(_MKSTEMP_TRIES):
        try:
            return tempfile.mkstemp(**kw)
        except PermissionError as e:
            last = e
    raise PermissionError(
        f"could not create a temp file in {kw.get('dir', '.')!r} after {_MKSTEMP_TRIES} "
        f"attempts — the directory exists but rejects writes (an ACL denial reads as "
        f"PermissionError on every candidate name, and os.access() cannot see it)") from last


def atomic_write(path: str, write, *, fsync: bool = False) -> bool:
    """Write to `path` atomically, and durably when `fsync`.

    `write` is either the BYTES to write, or a CALLABLE taking the open binary file — the
    callable form is for a producer that can stream (`pickle.dump`), so a caller never has to
    materialize a whole frame as a `bytes` blob just to reach this function. That distinction
    is worth a parameter: the CACHE-2 spill path runs *precisely when the RAM budget is already
    exceeded*, and forcing a blob there transiently doubles the footprint of the frame the
    spill exists to shed (measured 2.5× at 1024², ~100 MB extra at 4K).

    The single implementation for every persisted engine file — the program cache, the tier
    verdicts, the transfer model, the warm state, the spilled frames. Best-effort by contract:
    every caller is persisting a cache, and a failed cache write must degrade to a recook,
    never to a raised cook. Returns True on success, so a caller keeping a running byte total
    can tell.

    `fsync` DEFAULTS TO FALSE, because durability is the minority case here and it should be
    visible at the one place that wants it. Four of the five callers persist state the engine
    re-derives on its own — a spilled frame (a recook), a compile artifact (~2.5 ms), a tier
    verdict, a bandwidth probe — and for those a flush costs real time on the cook thread and
    buys nothing: 14 ms per frame eviction, and +44% on cold compile when it sat on that path.
    Only `warm_state` asks for `fsync=True`, and it says so at its call site.

    TEMP NAMING: `mkstemp` in the target directory, not a fixed `path + ".tmp"`. Two of the
    callers document their motivating case as "a second ComfyUI instance sharing the dir", and
    with a shared temp name two writers race on one file — a corrupt promote on POSIX, a
    sharing violation on Windows, and a failure-path `remove` that deletes the *other* writer's
    in-flight temp. `tex_snippets` had already worked this out and used `mkstemp`; unifying on
    the weaker form would have generalized downward."""
    fd = tmp = None
    try:
        fd, tmp = bounded_mkstemp(dir=os.path.dirname(path) or ".",
                                  prefix=TMP_PREFIX, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            fd = None                   # fdopen owns it now; closing twice is an error
            if callable(write):
                write(f)
            else:
                f.write(write)
            if fsync:
                f.flush()
                os.fsync(f.fileno())    # the bytes are on the platter BEFORE the rename
        os.replace(tmp, path)
        return True
    except Exception:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp is not None:
            try:
                os.remove(tmp)          # our OWN temp — never a shared name, never a peer's
            except OSError:
                pass
        return False


def atomic_write_json(path: str, obj, *, fsync: bool = False) -> bool:
    """`atomic_write` of `json.dumps(obj)`. The guard is for `dumps` (an unserializable value);
    `atomic_write` already returns False rather than raising for anything past it."""
    try:
        payload = json.dumps(obj).encode("utf-8")
    except Exception:
        return False
    return atomic_write(path, payload, fsync=fsync)


# ── the journal ──────────────────────────────────────────────────────────────


class Journal:
    """An append-only sidecar next to a snapshot file, so incrementally-learned state is
    durable the moment it is learned instead of when the snapshot next happens to be written.

    JSONL rather than a binary log on purpose: a truncated tail (the crash case) costs exactly
    the one malformed line, which `replay()` skips, and the file stays readable by a human
    debugging a recovery.

    `append` flushes but does NOT fsync. HONEST COST, because an earlier draft of this docstring
    said "microseconds" and that was wrong: open + write + flush + close measures **186 µs** on
    this box, against 5.4 µs through a held handle. It runs once per newly-learned capturability
    verdict (from `graphed`, on the first cook of a program), not per cook — so it is a
    once-per-program cost on the cook thread, not a per-frame one. A held handle would recover
    the difference but would block a peer process's `os.replace` over the same path on Windows,
    which is the compaction this class exists to allow. The bound ENG-13 states is a PROCESS
    crash, and a flushed write survives that; `persist()`'s snapshot is where the fsync lands."""

    __slots__ = ("path",)

    def __init__(self, snapshot_path: str):
        self.path = snapshot_path + ".journal"

    def append(self, record: dict) -> bool:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                # LEADING newline as well as trailing: a crash mid-append leaves a partial
                # line with no terminator, and appending straight onto it merges the torn
                # record with this one — losing BOTH while returning True. The extra byte
                # is inert: `count()`, `replay()` and `drop_prefix` all skip blank lines,
                # consistently (verified across LF/CRLF/torn/BOM/NUL file shapes).
                f.write("\n" + json.dumps(record, separators=(",", ":")) + "\n")
                f.flush()
            return True
        except Exception:
            return False

    def count(self) -> int:
        """How many records are on disk, by LINE — no JSON parse.

        Compaction needs a count and a tail, and neither needs parsed records. Parsing 200
        records just to `len()` them measured 442 µs, of which 340 µs was `json.loads` on values
        nobody read — paid twice per compaction, since `drop_prefix` then parsed again."""
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    def replay(self) -> list:
        """Every well-formed record, oldest first. A malformed line — the signature of a crash
        mid-append — costs exactly that record.

        `errors="replace"` is load-bearing, not defensive dressing: a single bad byte anywhere in
        the file (a partial multi-byte write torn by the crash this exists to survive) raised
        `UnicodeDecodeError` straight out of here and **discarded every good record permanently**,
        turning a one-record loss into total loss of the journal. Replacing the byte instead
        confines the damage to the line it is on, which `json.loads` then skips.

        The `except Exception` around the whole loop is the same argument one level up: recovery
        must degrade to "fewer records", never to "raise at load"."""
        out = []
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue        # a torn or corrupt line: the one record the crash cost
        except OSError:
            pass
        except Exception:               # never let a recovery read raise at the caller
            pass
        return out

    def clear(self) -> None:
        """Drop the journal entirely. Only safe when nothing can have been appended since the
        snapshot was taken — prefer `drop_prefix`, which is safe unconditionally."""
        try:
            os.remove(self.path)
        except OSError:
            pass

    def drop_prefix(self, n: int) -> None:
        """Discard the first `n` records and KEEP the rest.

        This is what compaction needs, and a plain `clear()` is not. A snapshot is taken from
        the live table, then written — and a verdict learned during that write appends to the
        journal but is NOT in the snapshot. Clearing wholesale loses it (reproduced 2/5). So the
        compactor counts what it is superseding, and only that many records go.

        Rewrites via the shared atomic write, so a crash mid-compaction leaves either the old
        journal or the trimmed one, never a torn one."""
        if n <= 0:
            return
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                rest = [line for line in f if line.strip()][n:]
        except OSError:
            return
        if not rest:
            self.clear()
            return
        # LINES, not records: no JSON round-trip in either direction, and a malformed line is
        # one the snapshot never adopted, so dropping it with the prefix is exactly right.
        atomic_write(self.path, "".join(rest).encode("utf-8"))

    def exists(self) -> bool:
        return os.path.exists(self.path)


# ── reattach ─────────────────────────────────────────────────────────────────


def reattach(*, result_cache=None) -> dict:
    """Re-adopt the engine's persisted warm state in a LIVE process, and report what came back.

    The recovery path for a host whose engine died, or which is picking up a cache directory
    another process wrote. Not a restart: nothing is torn down, the caches keep whatever they
    already hold, and every restored fact is merged with `setdefault` semantics so anything
    learned in THIS session (which is fresher) wins.

    Returns `{"verdicts": n, "capturable": n, "frames": n, "frame_bytes": n, "errors": [...]}`.
    Best-effort per component: a corrupt `autotier.json` must not stop the warm state from
    coming back, so each restore is independently guarded and named in `errors`."""
    report = {"verdicts": 0, "capturable": 0, "frames": 0, "frame_bytes": 0, "errors": []}

    # Each component owns its own reload — this function aggregates and names failures, it does
    # not know how any of them latch or where they keep their table. `ResultCache.reindex_disk`
    # established that shape; `autotier.reload` and `warm_state.reload` complete it, so a fourth
    # persisted thing joins by implementing the protocol rather than by growing a stanza here.
    try:
        from .tex_runtime import autotier
        report["verdicts"] = autotier.reload()
    except Exception as e:
        report["errors"].append(f"autotier: {e}")

    try:
        from .tex_runtime import warm_state
        report["capturable"] = warm_state.reload()      # snapshot + the journal tail
    except Exception as e:
        report["errors"].append(f"warm_state: {e}")

    # (3) the CACHE-2 disk tier. A restored `ResultCache` already SERVES spilled frames — `get`
    # falls through to `_restore`, which reads by key and re-checks `env_epoch` — so nothing
    # needs re-loading. What a fresh cache lacks is the byte accounting, and a None total forces
    # a full reconciling scan on the next spill. Do that scan here instead, off the cook path,
    # and report the tier's size so a host can show it.
    if result_cache is not None:
        try:
            report["frames"], report["frame_bytes"] = result_cache.reindex_disk()
        except Exception as e:
            report["errors"].append(f"result_cache: {e}")

    return report
