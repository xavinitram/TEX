"""
LANG-5 — the server-side user snippet store.

User snippets are a `{name: code}` map persisted as JSON in the host's per-user data
directory (the `get_user_dir()` host seam, PORT-1). The editor's localStorage becomes an
*offline cache* layered under this: the frontend syncs from the server on menu-open and
writes through to the server on save, so snippets are a real, git-versionable asset that
survives a browser/cache wipe and follows the user across machines.

This module is pure `os`/`json` (no comfy) so it is unit-testable; the aiohttp routes in
`__init__.py` are thin callers. When the host exposes no user dir (CLI / tests), the store
falls back to a `user/` folder under the TEX cache dir (TEX_CACHE_DIR-controlled, so tests
stay isolated).
"""
import json
import os
import tempfile

_FILE = "user_snippets.json"


class SnippetStoreError(Exception):
    """The user-snippet store EXISTS on disk but could not be read as a `{name: code}`
    map — a locked / permission-denied file (e.g. AV scanning it right after another
    tab's `os.replace`), a truncated / corrupt JSON, or a non-object payload. Distinct
    from a genuinely ABSENT store, which is simply empty. The GET route turns this into a
    503 + `{"read_error": true}` (NOT an empty 200) so the frontend keeps its offline
    cache instead of syncing an empty map over real snippets (LANG-5 BUG 2)."""


def _snippets_path():
    """The absolute path to the user-snippets JSON, or None if no location resolves.
    `<user_dir>/tex_wrangle/user_snippets.json`, else `<cache_dir>/user/tex_wrangle/…`."""
    base = None
    try:
        from .tex_runtime.host import get_host_services
        base = get_host_services().get_user_dir()
    except Exception:
        base = None
    if not base:
        try:
            from .tex_cache import get_cache
            base = os.path.join(get_cache()._cache_dir, "user")
        except Exception:
            return None
    return os.path.join(base, "tex_wrangle", _FILE)


def load_user_snippets() -> dict:
    """The saved `{name: code}` map.

    Returns `{}` when the store is genuinely ABSENT (never saved, or no location
    resolves) — a legitimately empty store. RAISES `SnippetStoreError` when the store
    file EXISTS but cannot be read as a `{name: code}` object (locked by another process /
    AV, permission denied, truncated / corrupt JSON, or a non-object payload).

    This read-failure-vs-empty distinction is load-bearing (LANG-5 BUG 2): the frontend
    syncs its offline cache from this over the wire, so an *unreadable* store must NOT be
    reported as an empty one, or a transient lock would wipe every saved snippet. An
    explicitly-saved empty map (`{}` on disk) is a dict, so it returns `{}` without
    raising — the user clearing their last snippet is a legitimate empty store."""
    p = _snippets_path()
    if not p:
        return {}
    # Open directly rather than gating on os.path.isfile: isfile() swallows every OSError
    # (a stat that fails on a permission-denied / locked path) and reports False, which
    # would misclassify a present-but-unreadable store as ABSENT — the very BUG-2 wipe this
    # is meant to prevent, moved onto the stat gate. Split FileNotFoundError (genuinely
    # absent -> empty) from any other read fault (present but unreadable -> read error).
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise SnippetStoreError(
            f"snippet store at {p} is present but unreadable: {type(e).__name__}: {e}") from e
    if not isinstance(data, dict):
        raise SnippetStoreError(
            f"snippet store at {p} is not a JSON object (got {type(data).__name__})")
    return {str(k): str(v) for k, v in data.items()}


def save_user_snippets(snippets) -> bool:
    """Replace the whole `{name: code}` map (matches the frontend's load-modify-save of
    the localStorage cache). Atomic write; returns True on success, never raises. String
    values are coerced to `str`; a non-string KEY is dropped (not coerced) — a hostile
    payload can't smuggle a non-string name in."""
    p = _snippets_path()
    if not p or not isinstance(snippets, dict):
        return False
    clean = {str(k): str(v) for k, v in snippets.items() if isinstance(k, str)}
    try:
        d = os.path.dirname(p)
        os.makedirs(d, exist_ok=True)
        # A UNIQUE temp per writer (not the shared `p + ".tmp"`): two concurrent saves must
        # not write the same temp then both os.replace it (a corrupt-promote risk on POSIX;
        # a spurious sharing-violation on Windows). mkstemp gives each writer its own file.
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".snip_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                json.dump(clean, f, indent=2, ensure_ascii=False)
            os.replace(tmp, p)
            return True
        except Exception:
            try:
                os.unlink(tmp)   # don't leave a temp turd behind on failure
            except OSError:
                pass
            return False
    except Exception:
        return False


# ── Route response builders ───────────────────────────────────────────────────
# The aiohttp handlers in `__init__.py` are thin JSON-encoders over these. Keeping the
# `(body, status)` policy here (pure os/json, no aiohttp/comfy — PORT-1/S-1) makes the
# GET/POST contract — including the read-failure 503 and the failed-save 503 — unit-
# testable off the event loop, without a live PromptServer.

def user_snippets_get_payload():
    """The GET `/tex_wrangle/user_snippets` response as `(body, status)`.

    Healthy read → `({"snippets": {name: code}}, 200)`. Read failure (`SnippetStoreError`
    or any unexpected fault) → `({"snippets": {}, "read_error": True, "error": ...}, 503)`.
    The 503 + `read_error` flag is what lets the frontend tell "the store is empty" (safe
    to sync over the cache) from "the store couldn't be read" (keep the cache) — LANG-5
    BUG 2. Never raises."""
    try:
        return {"snippets": load_user_snippets()}, 200
    except SnippetStoreError as e:
        return {"snippets": {}, "read_error": True, "error": str(e)}, 503
    except Exception as e:   # defensive backstop — an unexpected read fault is still a read error
        return {"snippets": {}, "read_error": True,
                "error": f"{type(e).__name__}: {e}"}, 503


def user_snippets_post_payload(body):
    """The POST `/tex_wrangle/user_snippets` response as `(body, status)`. `body` is the
    parsed request JSON, expected shape `{"snippets": {name: code}}`.

    A malformed body (not an object, or `snippets` absent / not an object) is a 400 that
    does NOT touch the store — a garbled request must never wipe saved snippets. A save
    that does not become durable (read-only disk, unresolved path) → `({"ok": False}, 503)`
    so the frontend keeps the edit marked *pending* and retries on the next sync (LANG-5
    BUG 1), instead of treating a rejected write as committed. Success → `({"ok": True},
    200)`. Never raises."""
    if not isinstance(body, dict) or not isinstance(body.get("snippets"), dict):
        return {"ok": False, "error": "bad request body (expected {'snippets': {name: code}})"}, 400
    if save_user_snippets(body["snippets"]):
        return {"ok": True}, 200
    return {"ok": False, "error": "snippet store is not writable"}, 503
