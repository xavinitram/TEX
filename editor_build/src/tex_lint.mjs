/**
 * TEX Error Diagnostic Bridge for CodeMirror 6
 *
 * Converts TEX compilation errors (from the WebSocket execution_error event)
 * into CodeMirror 6 Diagnostic objects for inline error display.
 */
import { setDiagnostics } from "@codemirror/lint";

export { setDiagnostics };

/**
 * Parse a TEX error message and produce CodeMirror diagnostics.
 *
 * TEX errors include line/column info in two formats:
 *   1. "Error at line 3, column 5: ..."  (type checker / parser)
 *   2. "[3:5] ..."  (compact format)
 *
 * Heuristic: When the error is about a missing semicolon or unexpected token
 * and points to column 1, the actual problem is at the END of the previous
 * line. We adjust the highlight to point there instead.
 *
 * @param {EditorView} view - The CodeMirror editor view
 * @param {object} errorData - { message, type, traceback }
 * @returns {Diagnostic[]} Array of CM6 diagnostic objects
 */
export function texErrorToDiagnostics(view, errorData) {
    if (!errorData || !errorData.message) return [];

    const msg = errorData.message;
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

    // Fallback: try just "line N"
    if (line === null) {
        const match3 = msg.match(/line\s+(\d+)/i);
        if (match3) {
            line = parseInt(match3[1]);
            col = 1;
        }
    }

    const doc = view.state.doc;

    if (line !== null && line >= 1 && line <= doc.lines) {
        // Heuristic: for "Expected ';'" errors at col 1, the real problem is
        // the end of the previous line (the semicolon was missing there).
        // Also handle unexpected token / unexpected EOF at start of line.
        const isMissingSemicolon = /expected\s*['"`;]/i.test(msg) || /unexpected\s*(EOF|token|end)/i.test(msg);
        if (isMissingSemicolon && col <= 1 && line > 1) {
            // Point to the end of the previous line instead
            const prevLine = doc.line(line - 1);
            const trimmedLen = prevLine.text.trimEnd().length;

            // Clean up the error message
            let cleanMsg = cleanErrorMessage(msg);

            return [{
                from: prevLine.from + Math.max(0, trimmedLen - 1),
                to: prevLine.to,
                severity: "error",
                message: cleanMsg,
                source: errorData.type || "TEX",
            }];
        }

        // Normal case: underline from the reported position to end of line
        const lineInfo = doc.line(line);
        const from = lineInfo.from + Math.min((col || 1) - 1, lineInfo.length);
        const to = lineInfo.to;

        let cleanMsg = cleanErrorMessage(msg);

        return [{
            from,
            to,
            severity: "error",
            message: cleanMsg,
            source: errorData.type || "TEX",
        }];
    }

    // No line info — mark the first line
    return [{
        from: 0,
        to: doc.lines > 0 ? doc.line(1).to : 0,
        severity: "error",
        message: msg,
        source: errorData.type || "TEX",
    }];
}

/**
 * Remove redundant location prefix from error message for inline display.
 */
function cleanErrorMessage(msg) {
    let clean = msg
        .replace(/^(?:Lex|Parse|Type|Interpret)(?:er)?Error:\s*/i, "")
        .replace(/(?:at |)line\s+\d+(?:,\s*col(?:umn)?\s+\d+)?:?\s*/i, "")
        .replace(/^\[?\d+:\d+\]?\s*/, "")
        .trim();
    return clean || msg;
}
