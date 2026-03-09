/**
 * TEX Custom Theme for CodeMirror 6
 *
 * Matches ComfyUI's dark aesthetic and preserves the existing TEX syntax
 * color scheme from tex_extension.js.
 */
import { EditorView } from "@codemirror/view";
import { HighlightStyle, syntaxHighlighting } from "@codemirror/language";
import { tags } from "@lezer/highlight";

// ─── Editor theme (layout, chrome, colors) ───────────────────────────

export const texEditorTheme = EditorView.theme({
    // Root editor — desaturated dark grey to match ComfyUI's neutral aesthetic
    "&": {
        backgroundColor: "#1a1a1f",
        color: "#d4d4d4",
        fontSize: "13px",
        fontFamily: "'Monaspace Neon', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace",
        fontFeatureSettings: '"calt" 1, "liga" 1, "ss01" 1, "ss02" 1, "ss03" 1, "ss06" 1, "cv01" 2',
        fontVariationSettings: '"wght" 380',
        WebkitFontSmoothing: "antialiased",
        textRendering: "optimizeLegibility",
        borderRadius: "4px",
        border: "1px solid #333",
    },

    // Focused outline
    "&.cm-focused": {
        outline: "1px solid rgba(79, 195, 247, 0.5)",
    },

    // Content area
    ".cm-content": {
        caretColor: "#fff",
        lineHeight: "1.5",
        padding: "4px 0",
    },

    // Cursor
    ".cm-cursor, .cm-dropCursor": {
        borderLeftColor: "#fff",
        borderLeftWidth: "2px",
    },

    // Selection
    "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, ::selection": {
        backgroundColor: "rgba(79, 195, 247, 0.2) !important",
    },

    // Active line
    ".cm-activeLine": {
        backgroundColor: "rgba(255, 255, 255, 0.03)",
    },

    // Gutters (line numbers) — ultra-compact, semi-transparent, fade in on hover
    ".cm-gutters": {
        backgroundColor: "transparent",
        color: "#444",
        border: "none",
        opacity: "0.5",
        transition: "opacity 0.2s",
        minWidth: "0",
    },
    "&:hover .cm-gutters": {
        opacity: "1",
    },
    ".cm-activeLineGutter": {
        backgroundColor: "transparent",
        color: "#888",
    },
    ".cm-lineNumbers .cm-gutterElement": {
        padding: "0 0px",
        minWidth: "0px",
        fontSize: "9px",
        textAlign: "right",
    },

    // Matching brackets
    "&.cm-focused .cm-matchingBracket": {
        backgroundColor: "rgba(79, 195, 247, 0.25)",
        outline: "1px solid rgba(79, 195, 247, 0.5)",
    },
    "&.cm-focused .cm-nonmatchingBracket": {
        backgroundColor: "rgba(255, 50, 50, 0.25)",
        outline: "1px solid rgba(255, 50, 50, 0.5)",
    },

    // Fold gutter — hidden by default, too narrow to be useful inline
    ".cm-foldGutter": {
        width: "0px",
        display: "none",
    },

    // ── Autocomplete popup ──
    ".cm-tooltip": {
        backgroundColor: "#1e1e2e",
        border: "1px solid #4FC3F7",
        borderRadius: "6px",
        boxShadow: "0 4px 16px rgba(0, 0, 0, 0.5)",
    },
    ".cm-tooltip.cm-tooltip-autocomplete": {
        "& > ul": {
            fontFamily: "'Monaspace Neon', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace",
            fontSize: "12px",
            maxHeight: "200px",
        },
        "& > ul > li": {
            padding: "3px 8px",
            lineHeight: "1.4",
        },
        "& > ul > li[aria-selected]": {
            backgroundColor: "rgba(79, 195, 247, 0.2)",
            color: "#fff",
        },
    },
    ".cm-completionLabel": {
        color: "#d4d4d4",
    },
    ".cm-completionDetail": {
        color: "#888",
        fontStyle: "italic",
        marginLeft: "8px",
    },
    ".cm-completionInfo": {
        color: "#aaa",
        padding: "4px 8px",
        borderLeft: "1px solid #333",
    },

    // Completion icon badges
    ".cm-completionIcon-function::after": { content: "'f'", color: "#82aaff" },
    ".cm-completionIcon-variable::after": { content: "'v'", color: "#89ddff" },
    ".cm-completionIcon-keyword::after": { content: "'k'", color: "#c792ea" },
    ".cm-completionIcon-constant::after": { content: "'c'", color: "#ff5370" },

    // ── Lint / Error diagnostics ──
    ".cm-diagnostic": {
        padding: "4px 8px",
        borderRadius: "4px",
        fontSize: "12px",
        fontFamily: "'Monaspace Neon', 'Cascadia Code', 'Fira Code', 'Consolas', monospace",
    },
    ".cm-diagnostic-error": {
        backgroundColor: "rgba(180, 40, 40, 0.3)",
        borderLeft: "3px solid #f44336",
        color: "#ffcccc",
    },
    ".cm-lintRange-error": {
        backgroundImage: "none",
        textDecoration: "underline wavy #f44336",
        textUnderlineOffset: "3px",
    },
    // Lint gutter markers — tiny to save horizontal space
    ".cm-gutter-lint": {
        width: "6px",
        padding: "0",
    },
    ".cm-lint-marker": {
        width: "5px",
        height: "5px",
    },
    ".cm-lint-marker-error": {
        content: "''",
    },

    // Lint tooltip
    ".cm-tooltip-lint": {
        backgroundColor: "#1e1e2e",
        border: "1px solid #f44336",
    },

    // ── Scrollbar ──
    "&::-webkit-scrollbar, & *::-webkit-scrollbar": {
        width: "6px",
        height: "6px",
    },
    "&::-webkit-scrollbar-track, & *::-webkit-scrollbar-track": {
        backgroundColor: "#1a1a1f",
    },
    "&::-webkit-scrollbar-thumb, & *::-webkit-scrollbar-thumb": {
        backgroundColor: "#4FC3F7",
        borderRadius: "3px",
    },

    // ── Search panel ──
    ".cm-panels": {
        backgroundColor: "#1e1e2e",
        borderTop: "1px solid #333",
        color: "#d4d4d4",
    },
    ".cm-searchMatch": {
        backgroundColor: "rgba(79, 195, 247, 0.3)",
        outline: "1px solid rgba(79, 195, 247, 0.5)",
    },
    ".cm-searchMatch.cm-searchMatch-selected": {
        backgroundColor: "rgba(79, 195, 247, 0.5)",
    },

}, { dark: true });

// ─── Syntax highlight style (token colors) ───────────────────────────
// Maps CodeMirror tags to the existing TEX color scheme.

export const texHighlightStyle = syntaxHighlighting(HighlightStyle.define([
    // Keywords: purple bold — float, int, vec3, vec4, mat3, mat4, string, if, else, for
    { tag: tags.keyword, color: "#c792ea", fontWeight: "bold" },

    // Built-in functions: blue — sin, cos, lerp, sample, luma, etc.
    // "builtin" token → default table → tags.standard(tags.variableName)
    { tag: tags.standard(tags.variableName), color: "#82aaff" },

    // @ bindings: orange bold — @A, @OUT, @base_image
    { tag: tags.special(tags.variableName), color: "#f78c6c", fontWeight: "bold" },

    // Numbers: amber — 0.5, 42, 0xFF
    { tag: tags.number, color: "#f9ae58" },

    // Constants: red — PI, E
    { tag: tags.atom, color: "#ff5370" },

    // Coord variables: cyan — u, v, ix, iy, iw, ih, ic, fi, fn
    { tag: tags.definition(tags.variableName), color: "#89ddff" },

    // String literals: green — "hello"
    { tag: tags.string, color: "#c3e88d" },

    // Comments: grey italic — // ..., /* ... */
    { tag: tags.lineComment, color: "#6a6a8a", fontStyle: "italic" },
    { tag: tags.blockComment, color: "#6a6a8a", fontStyle: "italic" },

    // Operators: light grey
    { tag: tags.operator, color: "#c0c0c0" },

    // Punctuation: dim
    { tag: tags.punctuation, color: "#999" },

    // Regular variables: default text
    { tag: tags.variableName, color: "#d4d4d4" },
]));
