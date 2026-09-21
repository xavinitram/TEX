"""
TEX Standard Library — string builtins (one domain leaf of `stdlib.py`).

The `TEXStdlib` class-body section(s) String functions moved here verbatim, onto the `_StdlibString`
mixin. `stdlib.py` composes the leaves' mixins into `TEXStdlib` in the class body's original
section order, which is the REG-1 registration order (`help_entries()`, the generated
reference and the help panel all read it) — so import this leaf THROUGH `stdlib`, not
directly, unless registering only this domain is what you want.
"""
from __future__ import annotations
import hashlib
import re
import torch
from .stdlib_registry import stdlib
from .stdlib_core import (
    _scalar_from_tensor,
)


class _StdlibString:
    """string builtins: the `fn_*` methods `stdlib.py` mixes into `TEXStdlib`."""

    # -- String functions -----------------------------------------------

    @stdlib("str", sig='str(x) \\u2192 string', category='Strings', doc='Convert a number to a string.', ex='string s = str(42);')
    @staticmethod
    def fn_str(x) -> str:
        """Convert number to string."""
        if isinstance(x, str):
            return x
        if isinstance(x, torch.Tensor):
            v = _scalar_from_tensor(x, "str")
            return str(int(v)) if v == int(v) else str(v)
        return str(x)

    @stdlib("len", sig='len(x) \\u2192 float', category='Strings', doc='Length of a string, array, or vec-array (element count).', ex='float n = len("hello");')
    @staticmethod
    def fn_len(s) -> torch.Tensor:
        """String length, array length, or vec array element count -> float tensor."""
        if isinstance(s, str):
            return torch.scalar_tensor(float(len(s)), dtype=torch.float32)
        if isinstance(s, list):
            return torch.scalar_tensor(float(len(s)), dtype=torch.float32)
        if isinstance(s, torch.Tensor):
            # Vec array [B,H,W,N,C] or [N,C]: element count is dim -2
            if s.dim() in (2, 5):
                return torch.scalar_tensor(float(s.shape[-2]), dtype=torch.float32)
            return torch.scalar_tensor(float(s.shape[-1]), dtype=torch.float32)
        raise ValueError("len() expects a string or array argument")

    @stdlib("replace", sig='replace(s, old, new) \\u2192 string', category='Strings', doc='Replace all occurrences of old with new.', ex='string r = replace(s, "foo", "bar");')
    @staticmethod
    def fn_replace(s, old, new, max_count=None) -> str:
        """Replace occurrences of old with new. Optional max_count limits replacements."""
        if not all(isinstance(x, str) for x in (s, old, new)):
            raise ValueError("replace() expects string arguments for s, old, new")
        if max_count is not None:
            n = int(max_count.item() if isinstance(max_count, torch.Tensor) else max_count)
            return s.replace(old, new, n)
        return s.replace(old, new)

    @stdlib("strip", sig='strip(s) \\u2192 string', category='Strings', doc='Remove leading/trailing whitespace.', ex='string clean = strip(s);')
    @staticmethod
    def fn_strip(s) -> str:
        """Trim leading/trailing whitespace."""
        if not isinstance(s, str):
            raise ValueError("strip() expects a string argument")
        return s.strip()

    @stdlib("lower", sig='lower(s) \\u2192 string', category='Strings', doc='Convert to lowercase.', ex='string lc = lower("Hello");')
    @staticmethod
    def fn_lower(s) -> str:
        """Convert to lowercase."""
        if not isinstance(s, str):
            raise ValueError("lower() expects a string argument")
        return s.lower()

    @stdlib("upper", sig='upper(s) \\u2192 string', category='Strings', doc='Convert to uppercase.', ex='string uc = upper("hello");')
    @staticmethod
    def fn_upper(s) -> str:
        """Convert to uppercase."""
        if not isinstance(s, str):
            raise ValueError("upper() expects a string argument")
        return s.upper()

    @stdlib("contains", sig='contains(s, sub) \\u2192 float', category='Strings', doc='Returns 1.0 if s contains sub, 0.0 otherwise.', ex='float has = contains(s, "test");')
    @staticmethod
    def fn_contains(s, sub) -> torch.Tensor:
        """Check if s contains sub. Returns 1.0 or 0.0."""
        if not (isinstance(s, str) and isinstance(sub, str)):
            raise ValueError("contains() expects two string arguments")
        return torch.scalar_tensor(1.0 if sub in s else 0.0, dtype=torch.float32)

    @stdlib("startswith", sig='startswith(s, prefix) \\u2192 float', category='Strings', doc='Returns 1.0 if s starts with prefix.', ex='float sw = startswith(s, "img_");')
    @staticmethod
    def fn_startswith(s, prefix) -> torch.Tensor:
        """Check if s starts with prefix. Returns 1.0 or 0.0."""
        if not (isinstance(s, str) and isinstance(prefix, str)):
            raise ValueError("startswith() expects two string arguments")
        return torch.scalar_tensor(1.0 if s.startswith(prefix) else 0.0, dtype=torch.float32)

    @stdlib("endswith", sig='endswith(s, suffix) \\u2192 float', category='Strings', doc='Returns 1.0 if s ends with suffix.', ex='float ew = endswith(s, ".png");')
    @staticmethod
    def fn_endswith(s, suffix) -> torch.Tensor:
        """Check if s ends with suffix. Returns 1.0 or 0.0."""
        if not (isinstance(s, str) and isinstance(suffix, str)):
            raise ValueError("endswith() expects two string arguments")
        return torch.scalar_tensor(1.0 if s.endswith(suffix) else 0.0, dtype=torch.float32)

    @stdlib("find", sig='find(s, sub) \\u2192 float', category='Strings', doc='Index of first occurrence, or -1.0 if not found.', ex='float idx = find(s, "world");')
    @staticmethod
    def fn_find(s, sub) -> torch.Tensor:
        """Find index of sub in s. Returns -1.0 if not found."""
        if not (isinstance(s, str) and isinstance(sub, str)):
            raise ValueError("find() expects two string arguments")
        return torch.scalar_tensor(float(s.find(sub)), dtype=torch.float32)

    @stdlib("substr", sig='substr(s, start, len?) \\u2192 string', category='Strings', doc='Extract a substring. len is optional.', ex='string sub = substr(s, 0, 5);')
    @staticmethod
    def fn_substr(s, start, length=None) -> str:
        """Extract substring. start is 0-based index."""
        if not isinstance(s, str):
            raise ValueError("substr() expects a string first argument")
        start_i = int(start.item() if isinstance(start, torch.Tensor) else start)
        if length is not None:
            len_i = int(length.item() if isinstance(length, torch.Tensor) else length)
            return s[start_i:start_i + len_i]
        return s[start_i:]

    @stdlib("to_int", sig='to_int(s) \\u2192 int', category='Strings', doc='Parse a string as an integer.', ex='int n = to_int("42");')
    @staticmethod
    def fn_to_int(s) -> torch.Tensor:
        """Parse integer from string."""
        if not isinstance(s, str):
            raise ValueError("to_int() expects a string argument")
        try:
            return torch.scalar_tensor(float(int(s.strip())), dtype=torch.float32)
        except ValueError:
            raise ValueError(f"to_int(): cannot parse '{s}' as integer")

    @stdlib("to_float", sig='to_float(s) \\u2192 float', category='Strings', doc='Parse a string as a float.', ex='float f = to_float("3.14");')
    @staticmethod
    def fn_to_float(s) -> torch.Tensor:
        """Parse float from string."""
        if not isinstance(s, str):
            raise ValueError("to_float() expects a string argument")
        try:
            return torch.scalar_tensor(float(s.strip()), dtype=torch.float32)
        except ValueError:
            raise ValueError(f"to_float(): cannot parse '{s}' as float")

    @stdlib("sanitize_filename", sig='sanitize_filename(s) \\u2192 string', category='Strings', doc='Remove unsafe characters for use in file paths.', ex='string safe = sanitize_filename(s);')
    @staticmethod
    def fn_sanitize_filename(s) -> str:
        """Remove characters illegal in filenames."""
        if not isinstance(s, str):
            raise ValueError("sanitize_filename() expects a string argument")
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', s)
        cleaned = cleaned.strip('. ')
        return cleaned if cleaned else "unnamed"

    @stdlib("split", sig='split(s, sep) \\u2192 string[]', category='Strings', doc='Split string into array by separator.', ex='string parts[4] = split(s, ",");')
    @staticmethod
    def fn_split(s, delimiter, max_splits=None) -> list:
        """Split string by delimiter. Returns a list of strings."""
        if not isinstance(s, str):
            raise ValueError("split() expects a string first argument")
        if not isinstance(delimiter, str):
            raise ValueError("split() delimiter must be a string")
        if max_splits is not None:
            n = int(max_splits.item() if isinstance(max_splits, torch.Tensor) else max_splits)
            return s.split(delimiter, n)
        return s.split(delimiter)

    @stdlib("lstrip", sig='lstrip(s) \\u2192 string', category='Strings', doc='Remove leading whitespace.', ex='string clean = lstrip(s);')
    @staticmethod
    def fn_lstrip(s) -> str:
        """Trim leading whitespace."""
        if not isinstance(s, str):
            raise ValueError("lstrip() expects a string argument")
        return s.lstrip()

    @stdlib("rstrip", sig='rstrip(s) \\u2192 string', category='Strings', doc='Remove trailing whitespace.', ex='string clean = rstrip(s);')
    @staticmethod
    def fn_rstrip(s) -> str:
        """Trim trailing whitespace."""
        if not isinstance(s, str):
            raise ValueError("rstrip() expects a string argument")
        return s.rstrip()

    @stdlib("pad_left", sig='pad_left(s, width, fill) \\u2192 string', category='Strings', doc='Pad string on the left to reach width.', ex='string n = pad_left(str(fi), 4, "0");')
    @staticmethod
    def fn_pad_left(s, width, char=None) -> str:
        """Pad string on the left to reach target width. Default pad char is space."""
        if not isinstance(s, str):
            raise ValueError("pad_left() expects a string first argument")
        w = int(width.item() if isinstance(width, torch.Tensor) else width)
        fill = " "
        if char is not None:
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError("pad_left() fill character must be a single character string")
            fill = char
        return s.rjust(w, fill)

    @stdlib("pad_right", sig='pad_right(s, width, fill) \\u2192 string', category='Strings', doc='Pad string on the right to reach width.', ex='string n = pad_right(s, 20, " ");')
    @staticmethod
    def fn_pad_right(s, width, char=None) -> str:
        """Pad string on the right to reach target width. Default pad char is space."""
        if not isinstance(s, str):
            raise ValueError("pad_right() expects a string first argument")
        w = int(width.item() if isinstance(width, torch.Tensor) else width)
        fill = " "
        if char is not None:
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError("pad_right() fill character must be a single character string")
            fill = char
        return s.ljust(w, fill)

    @stdlib("format", sig='format(fmt, ...) \\u2192 string', category='Strings', doc='Python-style {} placeholders, filled in order; specs like {:04d} and {:.2f} work. A whole number formats as an int. % sequences are not placeholders.', ex='string s = format("{} x {}", iw, ih);')
    @staticmethod
    def fn_format(template, *args) -> str:
        """String interpolation. Replaces {} placeholders with arguments.
        format("frame_{}_v{}", 42, 3) produces "frame_42_v3".
        """
        if not isinstance(template, str):
            raise ValueError("format() expects a string template as first argument")
        # Convert tensor args to Python values for formatting
        converted = []
        for a in args:
            if isinstance(a, torch.Tensor):
                v = _scalar_from_tensor(a, "format")
                # Round to 6 significant digits to counteract float32 noise
                if v == int(v):
                    converted.append(int(v))
                else:
                    converted.append(float(f"{v:.6g}"))
            else:
                converted.append(a)
        try:
            return template.format(*converted)
        except KeyError as e:
            raise ValueError(
                f"format() only understands plain {{}} placeholders, but the template uses "
                f"a named one ({{{e.args[0]}}}). Replace it with {{}} and pass values in "
                f"order, e.g. format(\"hi {{}}\", x).") from e
        except IndexError as e:
            n_ph = template.count("{}")
            raise ValueError(
                f"format() template has {n_ph} {{}} placeholder{'s' if n_ph != 1 else ''}, "
                f"but {len(converted)} value{'s' if len(converted) != 1 else ''} "
                f"{'were' if len(converted) != 1 else 'was'} given.") from e

    @stdlib("repeat", sig='repeat(s, n) \\u2192 string', category='Strings', doc='Repeat a string N times.', ex='string bar = repeat("=", 10);')
    @staticmethod
    def fn_repeat(s, count) -> str:
        """Repeat a string N times."""
        if not isinstance(s, str):
            raise ValueError("repeat() expects a string first argument")
        n = int(count.item() if isinstance(count, torch.Tensor) else count)
        if n < 0:
            n = 0
        return s * n

    @stdlib("str_reverse", sig='str_reverse(s) \\u2192 string', category='Strings', doc='Reverse a string.', ex='string r = str_reverse("abc");')
    @staticmethod
    def fn_str_reverse(s) -> str:
        """Reverse a string."""
        if not isinstance(s, str):
            raise ValueError("str_reverse() expects a string argument")
        return s[::-1]

    @stdlib("count", sig='count(s, sub) \\u2192 float', category='Strings', doc='Count non-overlapping occurrences of sub in s.', ex='float n = count(s, "the");')
    @staticmethod
    def fn_count(s, sub) -> torch.Tensor:
        """Count non-overlapping occurrences of sub in s."""
        if not (isinstance(s, str) and isinstance(sub, str)):
            raise ValueError("count() expects two string arguments")
        return torch.scalar_tensor(float(s.count(sub)), dtype=torch.float32)

    @stdlib("matches", sig='matches(s, pattern) \\u2192 float', category='Strings', doc='Returns 1.0 if the whole string matches the regex pattern, else 0.0.', ex='float ok = matches(s, "[0-9]+");')
    @staticmethod
    def fn_matches(s, pattern) -> torch.Tensor:
        """Test if string matches a regex pattern. Returns 1.0 if the full string matches, 0.0 otherwise."""
        if not (isinstance(s, str) and isinstance(pattern, str)):
            raise ValueError("matches() expects two string arguments")
        try:
            return torch.scalar_tensor(1.0 if re.fullmatch(pattern, s) else 0.0, dtype=torch.float32)
        except re.error as e:
            raise ValueError(f"matches() invalid regex: {e}")

    @stdlib("hash", sig='hash(s) \\u2192 string', category='Strings', doc='Deterministic string hash (SHA-256 hex, first 16 chars).', ex='string h = hash("seed");')
    @staticmethod
    def fn_hash(s) -> str:
        """Deterministic hash of a string. Returns a stable string hash (SHA-256 hex, first 16 chars)."""
        if not isinstance(s, str):
            raise ValueError("hash() expects a string argument")
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    @stdlib("hash_float", sig='hash_float(s) \\u2192 float', category='Strings', doc='Deterministic hash of a string to a float in [0, 1).', ex='float r = hash_float("seed");')
    @staticmethod
    def fn_hash_float(s) -> torch.Tensor:
        """Deterministic hash of a string to a float in [0, 1).
        Useful for procedural seeding and per-pixel variation."""
        if not isinstance(s, str):
            raise ValueError("hash_float() expects a string argument")
        h = hashlib.sha256(s.encode("utf-8")).digest()
        # Use first 8 bytes as unsigned 64-bit integer, normalize to [0, 1)
        value = int.from_bytes(h[:8], "big") / (2**64)
        return torch.scalar_tensor(value, dtype=torch.float32)

    @stdlib("hash_int", sig='hash_int(s, max?) \\u2192 int', category='Strings', doc='Deterministic hash of a string to a non-negative int (optional exclusive max).', ex='int i = hash_int("seed", 100);')
    @staticmethod
    def fn_hash_int(s, max_val=None) -> torch.Tensor:
        """Deterministic hash of a string to a non-negative integer.
        If max_val is provided, result is in [0, max_val).
        Otherwise returns a large positive integer."""
        if not isinstance(s, str):
            raise ValueError("hash_int() expects a string first argument")
        h = hashlib.sha256(s.encode("utf-8")).digest()
        value = int.from_bytes(h[:8], "big")
        if max_val is not None:
            m = int(max_val.item() if isinstance(max_val, torch.Tensor) else max_val)
            if m > 0:
                value = value % m
                if m <= 2**24:
                    # Modulo already bounds the value within float32's exact-int
                    # range — don't clamp it down and break the [0, max_val) contract.
                    return torch.scalar_tensor(float(value), dtype=torch.float32)
        # No max_val (or a range beyond float32's exact-int range): clamp so the
        # value stays exactly representable as float32.
        value = min(value, 2**24 - 1)
        return torch.scalar_tensor(float(value), dtype=torch.float32)

    @stdlib("char_at", sig='char_at(s, idx) \\u2192 string', category='Strings', doc='Character at index (0-based).', ex='string c = char_at(s, 0);')
    @staticmethod
    def fn_char_at(s, index) -> str:
        """Get character at index. Returns empty string if out of bounds."""
        if not isinstance(s, str):
            raise ValueError("char_at() expects a string first argument")
        i = int(index.item() if isinstance(index, torch.Tensor) else index)
        if 0 <= i < len(s):
            return s[i]
        return ""
