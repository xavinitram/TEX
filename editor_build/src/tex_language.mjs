/**
 * TEX Language Definition for CodeMirror 6
 *
 * Uses StreamLanguage (line-by-line tokenizer) with a custom tokenTable
 * to map TEX token types to proper CM6 highlight tags.
 */
import { StreamLanguage } from "@codemirror/language";
import { LEX_KEYWORDS, LEX_CONSTANTS, LEX_COORD_VARS, LEX_BINDING_PREFIXES,
         LEX_FUNCTIONS, LEX_ALIASES } from "./tex_lexicon.mjs";

// ─── Token sets ──────────────────────────────────────────────────────
// Generated into tex_lexicon.mjs by tools/gen_editor_lexicon.py from the lexer's
// KEYWORDS and BINDING_TYPE_PREFIXES, the type checker's built-in variable names and
// the stdlib registry. Regenerate it instead of editing a list here.

const TEX_KEYWORDS = new Set(LEX_KEYWORDS);
const TEX_BUILTINS = new Set([...LEX_FUNCTIONS.map(f => f[0]), ...LEX_ALIASES.map(a => a[0])]);
const TEX_CONSTANTS = new Set(LEX_CONSTANTS);
const TEX_COORD_VARS = new Set(LEX_COORD_VARS);

// Type prefixes for typed bindings: f@threshold, i$count, etc.
const BINDING_TYPE_PREFIXES = new Set(LEX_BINDING_PREFIXES);

// ─── Token name strategy ─────────────────────────────────────────────
// StreamLanguage.define() accepts ONE argument (the spec); there is NO
// second options argument. Token names returned by token() are resolved
// via a built-in default table that maps CM5 names to CM6 tags:
//
//   "builtin"    → tags.standard(tags.variableName)   → blue
//   "variable-2" → tags.special(tags.variableName)    → orange
//   "def"        → tags.definition(tags.variableName) → cyan
//
// We use these standard names and match them in the theme.

// ─── StreamLanguage parser ───────────────────────────────────────────

// Consume up to and including the closing `*/` on this line, or to the end of the line
// when the comment continues on the next one.
function blockCommentBody(stream, state) {
    if (stream.skipTo("*/")) {
        stream.next(); // *
        stream.next(); // /
        state.inBlockComment = false;
    } else {
        stream.skipToEnd();
    }
    return "blockComment";
}

const texStreamParser = {
    name: "tex-wrangle",

    startState() {
        return { inBlockComment: false };
    },

    copyState(state) {
        return { inBlockComment: state.inBlockComment };
    },

    token(stream, state) {
        // ── Block comment continuation ──
        if (state.inBlockComment) return blockCommentBody(stream, state);

        // ── Whitespace ──
        if (stream.eatSpace()) return null;

        // ── Line comment: // ──
        if (stream.match("//")) {
            stream.skipToEnd();
            return "lineComment";
        }

        // ── Block comment start: /* ──
        if (stream.match("/*")) {
            state.inBlockComment = true;
            return blockCommentBody(stream, state);
        }

        // ── String literal: "..." with escape sequences ──
        if (stream.peek() === '"') {
            stream.next(); // opening quote
            let escaped = false;
            while (!stream.eol()) {
                const ch = stream.next();
                if (escaped) {
                    escaped = false;
                    continue;
                }
                if (ch === "\\") {
                    escaped = true;
                    continue;
                }
                if (ch === '"') break;
            }
            return "string";
        }

        // ── @ wire bindings (@A, @OUT, @base_image) and $ parameter bindings ($strength) ──
        if (stream.eat(/[@$]/)) {
            stream.eatWhile(/[A-Za-z0-9_]/);
            return "variable-2";   // → default table → tags.special(tags.variableName)
        }

        // ── Numbers: hex, float, int, scientific ──
        // Hex: 0xFF
        if (stream.match(/^0[xX][0-9a-fA-F]+/)) return "number";
        // Float/int with optional scientific: 3.14, .5, 1e10, 2.5e-3
        if (stream.match(/^\d+\.?\d*(?:[eE][+-]?\d+)?/) ||
            stream.match(/^\.\d+(?:[eE][+-]?\d+)?/)) return "number";

        // ── Identifiers: keywords, builtins, constants, coord vars ──
        if (stream.match(/^[A-Za-z_]\w*/)) {
            const word = stream.current();

            // Typed binding prefix: f@threshold, i$count, img@result, etc.
            if (BINDING_TYPE_PREFIXES.has(word)) {
                const next = stream.peek();
                if (next === "@" || next === "$") {
                    stream.next();                // consume @ or $
                    stream.eatWhile(/[A-Za-z0-9_]/);
                    return "variable-2";          // highlight entire typed binding
                }
            }

            if (TEX_KEYWORDS.has(word)) return "keyword";
            if (TEX_BUILTINS.has(word)) return "builtin";    // → default table → blue
            if (TEX_CONSTANTS.has(word)) return "atom";
            if (TEX_COORD_VARS.has(word)) return "def";     // → default table → cyan
            return "variableName";
        }

        // ── Multi-char operators ──
        if (stream.match("&&") || stream.match("||") ||
            stream.match("==") || stream.match("!=") ||
            stream.match("<=") || stream.match(">=") ||
            stream.match("+=") || stream.match("-=") ||
            stream.match("*=") || stream.match("/=") ||
            stream.match("++") || stream.match("--")) {
            return "operator";
        }

        // ── Single-char operators and punctuation ──
        const ch = stream.next();
        if ("+-*/%=<>!?:".includes(ch)) return "operator";
        if ("(){}[];,.".includes(ch)) return "punctuation";

        return null;
    },

    languageData: {
        commentTokens: { line: "//", block: { open: "/*", close: "*/" } },
        closeBrackets: { brackets: ["(", "[", "{", '"'] },
    },
};

export const texLanguageDef = StreamLanguage.define(texStreamParser);

// Exported through the TEX_CM6 bundle API (tex_cm6.mjs).
export { TEX_KEYWORDS, TEX_BUILTINS, TEX_CONSTANTS, TEX_COORD_VARS };
