/**
 * TEX CodeMirror 6 Bundle Entry Point
 *
 * This file is the Rollup input. It imports all CM6 modules and TEX-specific
 * extensions, then registers them as globalThis.TEX_CM6 for cross-module access.
 *
 * Build: cd editor_build && npm run build
 * Output: ../js/tex_cm6_bundle.js
 */

// ── Core CodeMirror ──
import { EditorView, keymap, lineNumbers, highlightActiveLineGutter,
         highlightSpecialChars, drawSelection, dropCursor,
         rectangularSelection, crosshairCursor, highlightActiveLine,
         tooltips } from "@codemirror/view";
import { EditorState, Compartment } from "@codemirror/state";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";

// ── Language support ──
import { syntaxHighlighting, defaultHighlightStyle, indentOnInput,
         bracketMatching, foldGutter, foldKeymap, StreamLanguage } from "@codemirror/language";

// ── Autocomplete ──
import { autocompletion, completionKeymap, closeBrackets, closeBracketsKeymap,
         startCompletion, closeCompletion, completionStatus } from "@codemirror/autocomplete";

// ── Lint (error diagnostics) ──
import { lintGutter, lintKeymap } from "@codemirror/lint";

// ── Search ──
import { searchKeymap, highlightSelectionMatches } from "@codemirror/search";

// ── TEX-specific modules ──
import { texLanguageDef, TEX_KEYWORDS, TEX_BUILTINS, TEX_CONSTANTS, TEX_COORD_VARS } from "./tex_language.mjs";
import { createTexCompletions } from "./tex_completions.mjs";
import { texEditorTheme, texHighlightStyle } from "./tex_theme.mjs";
import { texErrorToDiagnostics, setDiagnostics } from "./tex_lint.mjs";

// ── Assemble a TEX-specific setup (like basicSetup but customized) ──

function texSetup() {
    return [
        lineNumbers(),
        highlightActiveLineGutter(),
        highlightSpecialChars(),
        history(),
        // Fold gutter omitted — takes horizontal space; code folding still works
        // via keyboard shortcuts but no gutter markers shown.
        drawSelection(),
        dropCursor(),
        EditorState.allowMultipleSelections.of(true),
        indentOnInput(),
        bracketMatching(),
        closeBrackets(),
        rectangularSelection(),
        crosshairCursor(),
        highlightActiveLine(),
        highlightSelectionMatches(),
        keymap.of([
            ...closeBracketsKeymap,
            ...defaultKeymap,
            ...searchKeymap,
            ...historyKeymap,
            ...foldKeymap,
            ...completionKeymap,
            ...lintKeymap,
            indentWithTab,
        ]),
    ];
}

// ── Build the public API object ──

const TEX_CM6_API = {
    // Core
    EditorView,
    EditorState,
    Compartment,
    keymap,

    // Setup
    texSetup,

    // TEX language
    texLanguageDef,
    TEX_KEYWORDS,
    TEX_BUILTINS,
    TEX_CONSTANTS,
    TEX_COORD_VARS,

    // Autocomplete
    autocompletion,
    createTexCompletions,
    startCompletion,
    closeCompletion,
    completionStatus,
    tooltips,

    // Theme
    texEditorTheme,
    texHighlightStyle,

    // Lint
    lintGutter,
    setDiagnostics,
    texErrorToDiagnostics,
};

// ── Register as global ──────────────────────────────────────────────
// ComfyUI loads JS files as ES modules via import(). Module-scoped vars
// are NOT visible to other modules. We must explicitly set a global so
// tex_extension.js can access the CM6 API.

globalThis.TEX_CM6 = TEX_CM6_API;

console.log("[TEX] CodeMirror 6 bundle registered (globalThis.TEX_CM6)");

// Also keep the named exports for Rollup's IIFE return value (belt + suspenders)
export {
    EditorView,
    EditorState,
    Compartment,
    keymap,
    texSetup,
    texLanguageDef,
    TEX_KEYWORDS,
    TEX_BUILTINS,
    TEX_CONSTANTS,
    TEX_COORD_VARS,
    autocompletion,
    createTexCompletions,
    startCompletion,
    closeCompletion,
    completionStatus,
    texEditorTheme,
    texHighlightStyle,
    tooltips,
    lintGutter,
    setDiagnostics,
    texErrorToDiagnostics,
};
