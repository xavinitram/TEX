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


# ── BRIEF-10: integrity BEFORE deserialise, for the on-disk pickle caches ──────
#
# The program cache (`.pkl`/`.cg`) and the frame spill tier (`.frame`) reload with `pickle`, and
# a pickle's `__reduce__` executes DURING the load, before any field the reader could inspect.
# The cache dir is writable by more than this process by design (the writers say so: "a second
# instance sharing the dir"), so anyone who can drop a file there gets code execution on the next
# cook that hits its fingerprint — the pickle-CVE class exactly.
#
# The fix AUTHENTICATES the bytes before `pickle` sees them, with a key the dir-writer cannot
# read. A digest stored in — or beside — a file the attacker can write is recomputable by the
# attacker: it catches corruption, not a crafted file. A keyed MAC does not recompute without the
# key, so a forged file fails the check and is a cache MISS (recompile / re-cook), never an error
# and never a served frame.
#
# READ ONCE, then verify, then unpickle the SAME buffer (`load_verified`). Verifying a streamed
# read and then re-reading the file to `pickle.load` would be a TOCTOU: on Windows another handle
# can rewrite the file between the hash and the re-read, so the bytes checked would not be the
# bytes deserialised. The one buffer closes that; the `.frame` tier pays one transient copy on
# RESTORE for it, which is off the default ComfyUI path (the frame cache is host-armed) — the
# WRITE side still streams (`sign_pickle`), so the RAM-pressure spill-out path is unchanged.
#
# KEY LOCATION is the security argument: OUTSIDE the cache dir, in the user's own profile
# (`%LOCALAPPDATA%` / `$XDG_STATE_HOME`), reachable by THIS user but not the OTHER principal who
# can write a shared cache dir. Same-user is out of scope (that account can already run code as
# itself). SCOPE, stated honestly: this covers TEX's OWN `.pkl`/`.cg`/`.frame`. It does NOT cover
# torch's inductor cache under the same dir (`TORCHINDUCTOR_CACHE_DIR`), which torch reads with
# its own unauthenticated `pickle`/codegen — so a shared cache dir stays unsafe while
# torch.compile runs, and on the DEFAULT layout (cache inside the package dir) the MAC adds
# nothing, since whoever can plant a pickle there can edit the source beside it. It earns its
# keep only when `TEX_CACHE_DIR` points somewhere more exposed than the code.
#
# INVISIBLE (invariant #7): the tag is a 37-byte trailer `pickle` ignores, HMAC-SHA256 runs at
# memcpy speed off the per-frame path (disk loads are memoised in RAM after the first), and a
# pre-integrity file is a one-time miss+recook. No new setting: the key is minted on first use.
import hashlib     # noqa: E402  — co-located with the integrity code it serves
import hmac        # noqa: E402
import pickle      # noqa: E402
import threading   # noqa: E402

#: Trailer = MAGIC + tag. `_MAC_FAMILY` is the version-independent prefix: a trailer that starts
#: with it but is not exactly `_MAC_MAGIC` was written by a NEWER TEX and is declined without
#: being destroyed (F7). The MAGIC is folded into the MAC input too, so a truncated or
#: wrong-version trailer cannot be made to verify against a v1 tag.
_MAC_FAMILY = b"TEXm"
_MAC_MAGIC = b"TEXm1"
_MAC_TAG_LEN = 32                       # SHA-256
_MAC_TRAILER_LEN = len(_MAC_MAGIC) + _MAC_TAG_LEN
_MAC_KEY_FILE = "cache_mac.key"
_MAC_KEY_LEN = 32

#: `load_verified`'s three non-object verdicts. UNVERIFIED = the file OPENED and READ, but its
#: content is missing/unsigned/foreign/tampered/corrupt — a miss, and the caller may delete it,
#: because there is no way to tell a genuine pre-integrity or crafted file from one that will
#: never become valid. FUTURE_TRAILER = a newer `TEXm<n>` wrote it — a miss, but leave it on disk
#: so a downgrade never destroys a readable frame. UNREADABLE = the file could not even be OPENED
#: or READ (a transient OS-level failure — a Windows sharing violation from a real-time scanner or
#: indexer, a momentary EMFILE, a flaky network/cloud-synced cache dir — the cause is unknowable
#: from here and irrelevant): this says NOTHING about the file's content, so it is a miss that
#: leaves the file untouched, exactly like FUTURE_TRAILER. Collapsing this into UNVERIFIED was
#: the RESTORE-462 defect: a transient "could not open it right now" was indistinguishable from
#: "opened fine and failed the MAC", so every caller deleted a perfectly valid, previously-spilled
#: frame on a passing lock/scan/hiccup. The module already drew this line once, for the MAC key
#: file itself (`_probe_key`'s "unreadable" branch, which never removes what it could not read) —
#: this extends the same discipline to the content the key protects.
_UNVERIFIED = object()
_FUTURE_TRAILER = object()
_UNREADABLE = object()


def _is_decline_quietly(verdict) -> bool:
    """True for a `load_verified` verdict that means "leave the file on disk — it is a miss,
    never a delete": `_FUTURE_TRAILER` or `_UNREADABLE`. `_UNVERIFIED` is the odd one out (a
    caller MAY delete it) and is deliberately not part of this.

    R1 (v0.46.2 Phase C, altitude): before this, all three callers of `load_verified`
    (`tex_cache._load_from_disk`, `tex_cache._load_codegen_from_disk`, `tex_results._restore`)
    hand-spelled `is _FUTURE_TRAILER or is _UNREADABLE` — RESTORE-462's own diff already landed
    the identical line twice, and a caller had no way to tell it was re-deriving a grouping this
    module already owns. One place now owns the verdict→action mapping; each call site just
    asks it and keeps its own return shape (`None`, or a wider tuple of `None`s).

    Identity-only (`is`, never `==`), because `verdict` can be an arbitrary deserialised object
    (a cache record dict, a tensor) and `==` against one of those is not guaranteed to be a
    plain bool — a tensor's `__eq__` returns an elementwise tensor, which is exactly the
    ambiguous-truth-value trap a membership test (`verdict in (...)`) would walk into."""
    return verdict is _FUTURE_TRAILER or verdict is _UNREADABLE

_mac_key_cache: bytes | None = None
_mac_key_lock = threading.Lock()


def _mac_key_home() -> str | None:
    """The directory the MAC key lives in — deliberately OUTSIDE any cache dir, in a location the
    other principal who can write a shared cache dir cannot read. Returns None if no per-user
    writable home resolves (then the caller uses an ephemeral key)."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    else:
        base = os.environ.get("XDG_STATE_HOME")
        if not base:
            home = os.environ.get("HOME") or os.path.expanduser("~")
            base = os.path.join(home, ".local", "state") if home and home != "~" else None
    if not base:
        return None
    return os.path.join(base, "TEX_Wrangle")


def _mac_key() -> bytes:
    """Resolve-or-create the per-user cache MAC key, memoised for the process under a lock so two
    threads' first use cannot memoise different keys.

    Persistence (`_resolve_or_create_key`) is CREATE-ONLY and atomic: the key is minted into a
    temp and published with a link/rename that FAILS if the name exists, so a racing peer never
    reads a half-written key and exactly one process becomes the creator — everyone else reads the
    winner's key, so same-user instances share one key and verify each other's files. A malformed
    key (a crash or full disk mid-mint) is removed and re-minted rather than dooming every later
    process. Only when no per-user home is writable, or persistence fails outright, does this
    fall back to a PROCESS-EPHEMERAL key: still unforgeable by another principal (the crafted-file
    defence holds), at the cost that this process's disk cache neither persists across a restart
    nor mixes with another instance's — recorded churn, not an error."""
    global _mac_key_cache
    if _mac_key_cache is not None:
        return _mac_key_cache
    with _mac_key_lock:
        if _mac_key_cache is not None:
            return _mac_key_cache
        key = _resolve_or_create_key()
        if key is None:
            key = os.urandom(_MAC_KEY_LEN)
        _mac_key_cache = key
        return key


def _resolve_or_create_key() -> bytes | None:
    """A persistent per-user key, or None to signal the ephemeral fallback. Converges under a
    race: read a well-formed key; else be the single atomic creator (or read the winner if a peer
    created it first); repair a malformed key ONLY when the file on disk is still the exact
    malformed one probed — matched by (inode, size, mtime). Between the probe and here a peer may
    have replaced it with a good key; its stat then differs and we leave it, re-probe, and adopt
    it (N1). An unreadable file is never removed."""
    home = _mac_key_home()
    if home is None:
        return None
    path = os.path.join(home, _MAC_KEY_FILE)
    try:
        os.makedirs(home, exist_ok=True)
        if os.name != "nt":
            try:
                os.chmod(home, 0o700)
            except OSError:
                pass
    except OSError:
        return None
    for _ in range(16):
        kind, info = _probe_key(path)
        if kind == "ok":
            return info
        if kind == "absent":
            published = _publish_new_key(home, path)
            if published is not None:
                return published
            continue                     # lost the create race: loop and read the winner's key
        if kind == "malformed":
            # A short/empty/corrupt key (a crash mid-write, a full disk) must not doom every
            # later process to ephemeral (F3) — but remove it ONLY if it is STILL the exact file
            # we probed. A peer that republished a good key between the probe and now changes the
            # file's (inode, size, mtime), so the guard fails and we leave that key to adopt on
            # the next probe (N1). Never remove on an open failure (kind == "unreadable").
            try:
                st = os.stat(path)
                if (st.st_ino, st.st_size, st.st_mtime_ns) == info:
                    os.remove(path)
            except OSError:
                pass
            continue
        # kind == "unreadable": the file exists but cannot be read, and must not be removed —
        # there is nothing safe to do with it, so fall back to an ephemeral key.
        return None
    return None


def _publish_new_key(home: str, path: str) -> bytes | None:
    """Mint a key and publish it ATOMICALLY and CREATE-ONLY: write a temp, then link/rename it
    into place with a primitive that FAILS if the name already exists, so a reader never sees a
    half-written key and exactly one racer wins (F3). Returns the key on success, None if a peer
    won (caller re-reads) or on any error (caller falls to ephemeral)."""
    new_key = os.urandom(_MAC_KEY_LEN)
    fd = tmp = None
    try:
        fd, tmp = bounded_mkstemp(dir=home, prefix=".macgen-", suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            fd = None
            f.write(new_key)
            if os.name != "nt":
                try:
                    os.fchmod(f.fileno(), 0o600)
                except (OSError, AttributeError):
                    pass
        if os.name == "nt":
            os.rename(tmp, path)     # Windows: raises FileExistsError if `path` exists
        else:
            os.link(tmp, path)       # POSIX: raises FileExistsError if `path` exists
            os.remove(tmp)           # drop the temp link; `path` is the durable name
        tmp = None
        return new_key
    except FileExistsError:
        return None                  # a peer created it first — caller reads the winner's key
    except OSError:
        return None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _probe_key(path: str):
    """Classify the key file from a SINGLE descriptor (read and stamp the same open file, so a
    swap cannot slip between the read and the stat). Returns:
      ("ok", key)                          — a well-formed `_MAC_KEY_LEN`-byte key;
      ("malformed", (ino, size, mtime_ns)) — a wrong-length file, STAMPED so the caller can
                                             remove only this exact file (N1);
      ("absent", None)                     — it does not exist (go create);
      ("unreadable", None)                 — any other open/read error (never remove it).

    RESTORE-462: BINARY, explicitly. Every OTHER writer in this module mints its fd through
    `tempfile.mkstemp` (binary-mode by default) and never hits this; this is the one place a raw
    `os.open` reads binary state, and with no `O_BINARY` Windows opens it in TEXT mode — `os.read`
    then stops at the first 0x1A (Ctrl-Z, the legacy text-mode EOF marker) and folds every 0x0D
    0x0A pair to 0x0A. A uniformly random 32-byte key contains a 0x1A byte on ~11.8% of mints
    (1-(255/256)**32); such a key reads back SHORT here, `_probe_key` classes it "malformed" below,
    and the caller deletes and re-mints it — silently invalidating every earlier process's signed
    spill/`.pkl`/`.cg`, which then fail the MAC in every later process with an intact trailer and a
    matching epoch (the key changed under them, nothing was tampered)."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
    except FileNotFoundError:
        return ("absent", None)
    except OSError:
        return ("unreadable", None)
    try:
        st = os.fstat(fd)
        data = os.read(fd, _MAC_KEY_LEN + 1)
    except OSError:
        return ("unreadable", None)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    if len(data) == _MAC_KEY_LEN:
        return ("ok", data)
    return ("malformed", (st.st_ino, st.st_size, st.st_mtime_ns))


def _mac_init(name: bytes):
    key = _mac_key()
    m = hmac.new(key, digestmod=hashlib.sha256)
    m.update(_MAC_MAGIC)
    m.update(len(name).to_bytes(4, "little"))
    m.update(name)                      # bind the tag to the fingerprint filename
    return m


class _HashingWriter:
    """Tees `pickle.dump`'s streamed bytes into the running MAC and the file in one pass — so the
    frame spill's whole-frame streaming write is preserved (no second in-memory copy)."""

    __slots__ = ("_f", "_m")

    def __init__(self, f, m):
        self._f, self._m = f, m

    def write(self, b):
        self._m.update(b)
        return self._f.write(b)


def sign_pickle(path, data, *, fsync: bool = False) -> bool:
    """`atomic_write` of `pickle.dump(data)` with a keyed-MAC trailer appended. The one write
    helper the on-disk pickle caches call, so authentication is spelled once. Streamed, so a
    frame is never blobbed a second time in memory. Returns `atomic_write`'s success verdict."""
    name = os.path.basename(str(path)).encode("utf-8", "surrogatepass")

    def _body(f):
        m = _mac_init(name)
        pickle.dump(data, _HashingWriter(f, m), protocol=pickle.HIGHEST_PROTOCOL)
        f.write(_MAC_MAGIC)
        f.write(m.digest())

    return atomic_write(str(path), _body, fsync=fsync)


def load_verified(path):
    """Read `path` ONCE, authenticate the keyed-MAC trailer, and unpickle from the SAME in-memory
    buffer — so the bytes `pickle` deserialises are byte-for-byte the bytes the MAC verified, with
    no second read a concurrent writer could swap under (the F1 TOCTOU). THE GATE: the only
    deserialiser for the on-disk pickle caches.

    Returns the deserialised object, or `_UNVERIFIED` (the file opened and read, but is too short /
    unsigned / foreign / tampered / corrupt — the caller treats it as a miss and may delete the
    file), or `_FUTURE_TRAILER` (a newer `TEXm<n>` wrote it — decline WITHOUT deleting, so a
    downgrade never destroys a frame the newer build can still read), or `_UNREADABLE` (the file
    could not even be opened/read — a transient OS-level failure that says nothing about the
    file's content — decline WITHOUT deleting, exactly like `_FUTURE_TRAILER`; RESTORE-462)."""
    try:
        with open(str(path), "rb") as f:
            buf = f.read()
    except OSError:
        return _UNREADABLE
    if len(buf) < _MAC_TRAILER_LEN:
        return _UNVERIFIED
    magic = buf[-_MAC_TRAILER_LEN:-_MAC_TAG_LEN]
    if magic != _MAC_MAGIC:
        # A newer TEXm<n> trailer is declined, not destroyed; anything else is a plain miss.
        return _FUTURE_TRAILER if magic[:len(_MAC_FAMILY)] == _MAC_FAMILY else _UNVERIFIED
    tag = buf[-_MAC_TAG_LEN:]
    payload = memoryview(buf)[:-_MAC_TRAILER_LEN]
    name = os.path.basename(str(path)).encode("utf-8", "surrogatepass")
    m = _mac_init(name)
    m.update(payload)
    if not hmac.compare_digest(m.digest(), tag):
        return _UNVERIFIED
    try:
        return pickle.loads(payload)
    except Exception:
        return _UNVERIFIED


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

    Returns `{"verdicts": n, "capturable": n, "frames": n, "frame_bytes": n,
    "media_frames": n, "media_bytes": n, "errors": [...]}`.
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

    # (4) DATA-7's media pool. There is nothing to RESTORE — providers are host-side and
    # process-global (DATA-4 phase 2 is still deferred), so a host must re-register its
    # providers and re-declare its prefetch windows after a reattach; that is the contract,
    # not an omission. What the report adds is the count, and after a restart the count is
    # ZERO — printing the zero is the point. A host that assumed its pool survived and
    # finds it did not currently has no way to tell.
    try:
        from .tex_provider import _media_cache
        st = {"frames": 0, "bytes": 0} if _media_cache is None else _media_cache.stats()
        report["media_frames"], report["media_bytes"] = st["frames"], st["bytes"]
    except Exception as e:
        report["media_frames"], report["media_bytes"] = 0, 0
        report["errors"].append(f"media_pool: {e}")

    return report
