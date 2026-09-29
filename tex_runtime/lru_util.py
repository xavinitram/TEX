"""tex_runtime/lru_util.py — K6 (v0.50.0 Phase C, R1#2): the shared bounded-LRU idiom.

Three independent hand-copies of the identical `OrderedDict` LRU idiom existed before
this module: `graphed._blacklist_add` (`_blacklist[key] = None; _blacklist.move_to_end
(key); while len(_blacklist) > _BLACKLIST_MAX: _blacklist.popitem(last=False)`),
`compiled._blacklist_add` (the same shape against `_compile_blacklist`/`_BLACKLIST_MAX`),
and `fncalls_compile.record` (which wrote the memo insert WITHOUT the `move_to_end` call
the other two always make on insert — a fingerprint that is `record()`-ed but never
subsequently `verdict()`-read was a STRONGER eviction candidate than the equivalent case
in the other two copies, a small behavioural difference that was a direct symptom of
copy-by-hand rather than shared code). A future fix to the eviction/touch policy is now
one edit, not three, and the drift above cannot recur.

A pure leaf: no state of its own, no import of any sibling `tex_runtime` module."""
from __future__ import annotations
from collections import OrderedDict
from typing import Any


def lru_put(d: "OrderedDict[Any, Any]", key: Any, value: Any, max_size: int) -> None:
    """Insert `key: value`, mark it MOST-recently-used, and evict the LEAST-recently-used
    entry/entries until `len(d) <= max_size`. The one insert shape every bounded-LRU
    store in this codebase wants: an ordered SET (`graphed._blacklist`/
    `compiled._compile_blacklist`, value always `None`) and an ordered-dict-with-real-
    values MEMO (`fncalls_compile._memo`) are the same idiom with a different value, not
    two different idioms."""
    d[key] = value
    try:
        d.move_to_end(key)
        while len(d) > max_size:
            d.popitem(last=False)
    except KeyError:
        pass        # another thread evicted `key` (or emptied the memo) mid-insert: the bound holds


def lru_get(d: "OrderedDict[Any, Any]", key: Any, default: Any = None) -> Any:
    """`d[key]` marked most-recently-used, or `default`. Another thread may evict `key`
    between the lookup and the touch; that reads as a miss instead of raising."""
    try:
        value = d[key]
        d.move_to_end(key)
    except KeyError:
        return default
    return value
