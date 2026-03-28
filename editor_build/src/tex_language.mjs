/**
 * TEX Language Definition for CodeMirror 6
 *
 * Uses StreamLanguage (line-by-line tokenizer) with a custom tokenTable
 * to map TEX token types to proper CM6 highlight tags.
 */
import { StreamLanguage } from "@codemirror/language";

// ─── Token sets (must match tex_extension.js exactly) ────────────────

const TEX_KEYWORDS = new Set([
    "float", "int", "vec3", "vec4", "mat3", "mat4", "string",
    "if", "else", "for", "while", "break", "continue",
]);

const TEX_BUILTINS = new Set([
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sincos",
    "sinh", "cosh", "tanh",
    "pow", "sqrt", "exp", "log", "log2", "log10", "abs", "sign",
    "pow2", "pow10", "hypot",
    "floor", "ceil", "round", "fract", "mod",
    "isnan", "isinf", "degrees", "radians",
    "spow", "sdiv",
    "min", "max", "clamp", "lerp", "mix", "step", "smoothstep",
    "length", "normalize", "dot", "cross", "reflect",
    "luma", "rgb2hsv", "hsv2rgb", "fit", "rand",
    "sample", "fetch", "sample_cubic", "sample_lanczos",
    "fetch_frame", "sample_frame",
    "distance",
    "perlin", "simplex", "fbm",
    "str", "len", "replace", "strip", "lstrip", "rstrip", "lower", "upper",
    "contains", "startswith", "endswith", "find", "substr",
    "to_int", "to_float", "sanitize_filename",
    "split", "pad_left", "pad_right", "format", "repeat", "str_reverse",
    "count", "matches", "hash", "hash_float", "hash_int", "char_at",
    "sort", "reverse", "arr_sum", "arr_min", "arr_max", "median", "arr_avg",
    "join",
    "img_sum", "img_mean", "img_min", "img_max", "img_median",
    "transpose", "determinant", "inverse",
]);

const TEX_CONSTANTS = new Set(["PI", "TAU", "E"]);

const TEX_COORD_VARS = new Set([
    "u", "v", "ix", "iy", "iw", "ih", "px", "py", "ic", "fi", "fn",
]);

// Type prefixes for typed bindings: f@threshold, i$count, etc.
const BINDING_TYPE_PREFIXES = new Set(["f", "i", "v", "v2", "v3", "v4", "s", "img", "m", "l", "c", "b"]);

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
        if (state.inBlockComment) {
            if (stream.skipTo("*/")) {
                stream.next(); // *
                stream.next(); // /
                state.inBlockComment = false;
            } else {
                stream.skipToEnd();
            }
            return "blockComment";
        }

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
            // Consume rest of this line within the comment
            if (stream.skipTo("*/")) {
                stream.next();
                stream.next();
                state.inBlockComment = false;
            } else {
                stream.skipToEnd();
            }
            return "blockComment";
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

        // ── @ bindings: @A, @OUT, @base_image ──
        if (stream.eat("@")) {
            stream.eatWhile(/[A-Za-z0-9_]/);
            return "variable-2";   // → default table → tags.special(tags.variableName)
        }

        // ── $ parameter bindings: $strength, $count ──
        if (stream.eat("$")) {
            stream.eatWhile(/[A-Za-z0-9_]/);
            return "variable-2";   // same highlight as @ bindings
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
            if (BINDING_TYPE_PREFIXES.has(word) && !stream.eol()) {
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

// Re-export sets for use in completions
export { TEX_KEYWORDS, TEX_BUILTINS, TEX_CONSTANTS, TEX_COORD_VARS };
