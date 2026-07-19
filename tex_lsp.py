"""LANG-7 — a thin stdio JSON-RPC Language Server for TEX.

LANG-2 already built 90% of an LSP's backend: `tex_api.check()` returns structured
diagnostics and never raises, and the REG-1 stdlib registry carries every function's
signature / doc / example. This wraps them in the Language Server Protocol so a standalone
editor (VS Code, Neovim, …) gets live squiggles, autocompletion, and hover — the same
experience the ComfyUI editor gets from the `/tex_wrangle/check` route.

Run:  python -m TEX_Wrangle.tex_lsp        # speaks LSP over stdin/stdout

The protocol handling is split from I/O: `LSPServer.handle(method, params)` is a pure
dispatch returning (result, notifications), so it is unit-testable without a live stdio
pipe (the stdio loop in `main()` is a thin frame reader/writer around it). Host-agnostic:
no comfy, no aiohttp.
"""
from __future__ import annotations

import json
import re
import sys

LSP_VERSION = "0.1.0"
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# LSP DiagnosticSeverity: 1=Error 2=Warning 3=Information 4=Hint
_SEVERITY = {"error": 1, "warning": 2, "info": 3, "hint": 4}


def _registry():
    """Populate + return the stdlib registry module (help entries source)."""
    from .tex_runtime.stdlib import TEXStdlib  # noqa: F401  (import populates REGISTRY)
    from .tex_runtime import stdlib_registry as R
    return R


def diagnostics_for(source: str) -> list[dict]:
    """Run LANG-2 check() and convert each TEXDiagnostic to an LSP Diagnostic. check() never
    raises, so this never raises. LSP positions are 0-based; TEXDiagnostic is 1-based."""
    from .tex_api import check
    out = []
    for d in check(source, {}):
        dd = d.to_dict()
        line = max(0, (dd.get("line") or 1) - 1)
        col = max(0, (dd.get("col") or 1) - 1)
        end_line = max(0, (dd.get("end_line") or dd.get("line") or 1) - 1)
        end_col = dd.get("end_col")
        end_col = col + 1 if end_col is None else max(0, end_col - 1)
        item = {
            "range": {"start": {"line": line, "character": col},
                      "end": {"line": end_line, "character": end_col}},
            "severity": _SEVERITY.get(dd.get("severity", "error"), 1),
            "code": dd.get("code", ""),
            "source": "tex",
            "message": dd.get("message", ""),
        }
        if dd.get("docs_url"):
            item["codeDescription"] = {"href": dd["docs_url"]}
        out.append(item)
    return out


_COMPLETION_CACHE: list | None = None


def completion_items() -> list[dict]:
    """CompletionItems for every stdlib function (name + signature + doc), from the registry.
    Memoized -- the stdlib registry is static within a process, so re-decoding all ~143 entries
    on every keystroke is wasted work."""
    global _COMPLETION_CACHE
    if _COMPLETION_CACHE is None:
        R = _registry()
        _COMPLETION_CACHE = [{
            "label": e["name"], "kind": 3,       # CompletionItemKind.Function
            "detail": e.get("sig", ""), "documentation": e.get("desc", ""),
        } for e in R.help_entries(decode=True)]
    return _COMPLETION_CACHE


def hover_for(source: str, line: int, character: int) -> dict | None:
    """Hover text for the identifier under (line, character) (0-based), if it names a
    stdlib function. Returns an LSP Hover or None."""
    lines = source.split("\n")
    if line < 0 or line >= len(lines):
        return None
    text = lines[line]
    word = None
    for m in _WORD_RE.finditer(text):
        if m.start() <= character <= m.end():
            word = m.group()
            break
    if not word:
        return None
    e = _registry().help_lookup(word)
    if e is None:
        return None
    md = f"```\n{e['sig']}\n```\n\n{e.get('desc', '')}"
    if e.get("example"):
        md += f"\n\n*example:* `{e['example']}`"
    return {"contents": {"kind": "markdown", "value": md}}


class LSPServer:
    """Holds open-document text and dispatches LSP methods. `handle` is pure (no I/O)."""

    def __init__(self):
        self.docs: dict[str, str] = {}
        self.shutdown_requested = False

    def handle(self, method: str, params: dict) -> tuple:
        """Return (result, notifications). `result` is the JSON-RPC result for a request
        (None for notifications); `notifications` is a list of (method, params) to push."""
        params = params or {}
        if method == "initialize":
            return self._initialize(), []
        if method in ("initialized", "$/setTrace"):
            return None, []
        if method == "shutdown":
            self.shutdown_requested = True
            return None, []
        if method == "textDocument/didOpen":
            doc = params.get("textDocument", {})
            uri, text = doc.get("uri"), doc.get("text", "")
            self.docs[uri] = text
            return None, [self._publish(uri, text)]
        if method == "textDocument/didChange":
            uri = params.get("textDocument", {}).get("uri")
            changes = params.get("contentChanges", [])
            if changes:
                last = changes[-1]
                # We advertise full-document sync (textDocumentSync=1), so a conformant client
                # sends the whole document as `.text`. A non-conformant client that sends a
                # RANGE-scoped (incremental) change gives only the edited span — storing that as
                # the whole doc would silently corrupt every later diagnostic/hover, so ignore it.
                if "range" in last:
                    return None, []
                text = last.get("text", "")
                if self.docs.get(uri) == text:   # unchanged — skip a redundant full re-analysis
                    return None, []
                self.docs[uri] = text
                return None, [self._publish(uri, text)]
            return None, []
        if method == "textDocument/didClose":
            uri = params.get("textDocument", {}).get("uri")
            self.docs.pop(uri, None)
            return None, [{"method": "textDocument/publishDiagnostics",
                           "params": {"uri": uri, "diagnostics": []}}]
        if method == "textDocument/completion":
            return {"isIncomplete": False, "items": completion_items()}, []
        if method == "textDocument/hover":
            uri = params.get("textDocument", {}).get("uri")
            pos = params.get("position", {})
            text = self.docs.get(uri, "")
            return hover_for(text, pos.get("line", 0), pos.get("character", 0)), []
        # Unknown method: a JSON-RPC null result is a safe no-op for optional features.
        return None, []

    def _initialize(self) -> dict:
        return {
            "capabilities": {
                "textDocumentSync": 1,        # full document sync
                "completionProvider": {"triggerCharacters": ["."]},
                "hoverProvider": True,
                "diagnosticProvider": {"interFileDependencies": False,
                                       "workspaceDiagnostics": False},
            },
            "serverInfo": {"name": "tex-lsp", "version": LSP_VERSION},
        }

    def _publish(self, uri: str, text: str) -> dict:
        return {"method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": diagnostics_for(text)}}


# ── stdio JSON-RPC framing ────────────────────────────────────────────────────────
class _BadFrame(Exception):
    """A frame was received but its header/body couldn't be parsed. Distinct from a dead
    stream: the caller SKIPS a bad frame (the server keeps running) but STOPS on a stream
    error, so a garbled message can't kill the session and a closed pipe can't busy-spin."""


def _read_message(stream) -> dict | None:
    """Read one Content-Length-framed JSON-RPC message. Returns the parsed message, or None on
    a clean EOF (empty read). Raises `_BadFrame` on a malformed header/body (skippable) and lets
    stream errors (OSError / closed-stream ValueError from readline/read) propagate (fatal)."""
    headers = {}
    while True:
        line = stream.readline()          # stream errors propagate → main() stops (no busy-spin)
        if not line:
            return None                   # clean EOF
        line = line.decode("ascii", "replace").strip()
        if line == "":
            break
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    try:
        length = int(headers.get("content-length", 0))
    except (TypeError, ValueError):
        raise _BadFrame("bad Content-Length header")
    if length <= 0:
        # A headerless/zero-length frame is malformed, NOT EOF — skip it rather than shutting the
        # server down mid-session (an EOF is signalled by an empty readline above).
        raise _BadFrame("missing or non-positive Content-Length")
    body = stream.read(length)            # stream errors propagate
    try:
        obj = json.loads(body.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise _BadFrame("malformed JSON body")
    # A valid-JSON non-object body (42, "x", true, a JSON-RPC batch array, or null) is still a
    # malformed message — route it through the skip path so `msg.get(...)` in main() can't raise
    # (and `null` doesn't masquerade as EOF). main() reads method OUTSIDE its handler try, so an
    # unguarded non-dict here would tear the whole session down.
    if not isinstance(obj, dict):
        raise _BadFrame("JSON-RPC message must be an object")
    return obj


def _write_message(stream, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    stream.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii"))
    stream.write(data)
    stream.flush()


def main(argv=None) -> None:
    server = LSPServer()
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    while True:
        try:
            msg = _read_message(stdin)
        except _BadFrame:
            continue                           # malformed frame must not kill the server (#12)
        except Exception:
            break                              # dead/closed stream — stop, don't busy-spin (#11)
        if msg is None:
            break                              # clean EOF
        method = msg.get("method")
        if method == "exit":
            break
        try:
            result, notifications = server.handle(method, msg.get("params"))
            if "id" in msg:                    # a request expects a response
                _write_message(stdout, {"jsonrpc": "2.0", "id": msg["id"], "result": result})
            for note in notifications:
                _write_message(stdout, {"jsonrpc": "2.0", "method": note["method"],
                                        "params": note["params"]})
        except Exception as e:                 # one bad request must not tear down the session
            if isinstance(msg, dict) and "id" in msg:
                _write_message(stdout, {"jsonrpc": "2.0", "id": msg["id"],
                                        "error": {"code": -32603, "message": str(e)}})


if __name__ == "__main__":
    main()
