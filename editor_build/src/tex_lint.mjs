/**
 * TEX Error Diagnostic Bridge for CodeMirror 6
 *
 * Converts TEX compilation errors (from the WebSocket execution_error event)
 * into CodeMirror 6 Diagnostic objects for inline error display.
 *
 * A message carrying a structured `TEX_DIAG:` JSON payload (multi-error, suggestions,
 * hints, error codes) is read from that. Any other message (a runtime error, or a compile
 * error without the payload) is parsed for a "line N, column M" or "[N:M]" location.
 */
import { setDiagnostics } from "@codemirror/lint";

export { setDiagnostics };

/**
 * Parse a TEX error message and produce CodeMirror diagnostics.
 *
 * @param {EditorView} view - The CodeMirror editor view
 * @param {object} errorData - { message, type, traceback }
 * @returns {Diagnostic[]} Array of CM6 diagnostic objects
 */
export function texErrorToDiagnostics(view, errorData) {
    if (!errorData || !errorData.message) return [];

    const msg = errorData.message;

    // ── Structured diagnostics ──
    const diagPos = msg.indexOf("TEX_DIAG:");
    if (diagPos >= 0) {
        try {
            const diagnostics = JSON.parse(msg.slice(diagPos + 9));
            return diagnostics.map(d => structuredToCM6(view, d)).filter(Boolean);
        } catch {
            // JSON parse failed: fall through to the message-text parser
        }
    }

    // ── Message text: regex-based line:col extraction ──
    return legacyToCM6(view, msg, errorData);
}

// The server counts columns in code points (a Python string); CodeMirror offsets are
// UTF-16 code units, so a non-BMP character (an emoji in a string or comment) earlier on the
// line is two units wide. Returns the offset in `lineText` of the 1-based code-point column
// `col`, clamped to the end of the line.
function colToOffset(lineText, col) {
    let off = 0;
    for (let n = col - 1; n > 0 && off < lineText.length; n--) {
        off += lineText.codePointAt(off) > 0xFFFF ? 2 : 1;
    }
    return off;
}

// A diagnostic for a message with no usable line: mark the first line.
function firstLineDiagnostic(doc, message, severity, source) {
    return { from: 0, to: doc.lines > 0 ? doc.line(1).to : 0, severity, message, source };
}

/**
 * Convert a structured TEXDiagnostic dict to a CM6 Diagnostic.
 */
function structuredToCM6(view, d) {
    const doc = view.state.doc;
    const line = d.line;
    const col = d.col || 1;

    if (!line || line < 1 || line > doc.lines) {
        return firstLineDiagnostic(doc, formatStructuredMessage(d), d.severity || "error", d.code || "TEX");
    }

    const lineInfo = doc.line(line);
    const from = lineInfo.from + colToOffset(lineInfo.text, col);
    let to;
    // LANG-2: a multi-line span (end_line > line) underlines through to end_col on the
    // last line; otherwise fall back to end_col on the same line, else to end-of-line.
    if (d.end_line && d.end_line > line && d.end_line <= doc.lines) {
        const endInfo = doc.line(d.end_line);
        to = d.end_col ? endInfo.from + colToOffset(endInfo.text, d.end_col) : endInfo.to;
    } else if (d.end_col && d.end_col > col) {
        to = lineInfo.from + colToOffset(lineInfo.text, d.end_col);
    } else {
        to = lineInfo.to;
    }

    return {
        from,
        to,
        severity: d.severity || "error",
        message: formatStructuredMessage(d),
        source: d.code || "TEX",
    };
}

/**
 * Format a structured diagnostic into a human-readable CM6 message.
 */
function formatStructuredMessage(d) {
    let parts = [d.message];
    if (d.suggestions && d.suggestions.length > 0) {
        if (d.suggestions.length === 1) {
            parts.push(`> Try: ${d.suggestions[0]}`);
        } else {
            parts.push(`> Try one of: ${d.suggestions.join(", ")}`);
        }
    }
    if (d.hint) {
        parts.push(`> Help: ${d.hint}`);
    }
    return parts.join("\n");
}

/**
 * Regex-based parsing of a message that has no TEX_DIAG payload.
 */
function legacyToCM6(view, msg, errorData) {
    let line = null;
    let col = null;

    // Try to extract line:col from "Error at line N, column M: ..."
    const match1 = msg.match(/(?:at |)line\s+(\d+)(?:,\s*col(?:umn)?\s+(\d+))?/i);
    if (match1) {
        line = parseInt(match1[1]);
        col = match1[2] ? parseInt(match1[2]) : 1;
    }

    // Try compact format "[N:M]"
    if (line === null) {
        const match2 = msg.match(/\[(\d+):(\d+)\]/);
        if (match2) {
            line = parseInt(match2[1]);
            col = parseInt(match2[2]);
        }
    }

    const doc = view.state.doc;

    if (line !== null && line >= 1 && line <= doc.lines) {
        // Heuristic: for "Expected ';'" errors at col 1, the real problem is
        // the end of the previous line (the semicolon was missing there).
        const isMissingSemicolon = /expected\s*['"`;]/i.test(msg) || /unexpected\s*(EOF|token|end)/i.test(msg);
        if (isMissingSemicolon && col <= 1 && line > 1) {
            const prevLine = doc.line(line - 1);
            const trimmedLen = prevLine.text.trimEnd().length;
            return [{
                from: prevLine.from + Math.max(0, trimmedLen - 1),
                to: prevLine.to,
                severity: "error",
                message: cleanErrorMessage(msg),
                source: errorData.type || "TEX",
            }];
        }

        const lineInfo = doc.line(line);
        const from = lineInfo.from + colToOffset(lineInfo.text, col || 1);
        const to = lineInfo.to;
        return [{
            from,
            to,
            severity: "error",
            message: cleanErrorMessage(msg),
            source: errorData.type || "TEX",
        }];
    }

    // No line info: mark the first line
    return [firstLineDiagnostic(doc, msg, "error", errorData.type || "TEX")];
}

/**
 * Remove redundant location prefix from error message for inline display.
 */
function cleanErrorMessage(msg) {
    const clean = msg
        .replace(/^(?:Lex|Parse|Type|Interpret)(?:er)?Error:\s*/i, "")
        .replace(/(?:at |)line\s+\d+(?:,\s*col(?:umn)?\s+\d+)?:?\s*/i, "")
        .replace(/^\[?\d+:\d+\]?\s*/, "")
        .trim();
    return clean || msg;
}
