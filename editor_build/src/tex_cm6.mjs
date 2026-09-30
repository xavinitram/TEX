/**
 * TEX CodeMirror 6 Bundle Entry Point
 *
 * This file is the Rollup input. It imports all CM6 modules and TEX-specific
 * extensions, then registers them as globalThis.TEX_CM6 for cross-module access. The
 * bundle has no exports: the global is the whole interface, and it carries a few members
 * (keymap, closeCompletion, the TEX_* word sets) that js/tex_extension.js does not use
 * because the bundle is also redistributed to a second embedding host.
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
import { indentOnInput, bracketMatching } from "@codemirror/language";

// ── Autocomplete ──
import { autocompletion, completionKeymap, closeBrackets, closeBracketsKeymap,
         startCompletion, closeCompletion, completionStatus } from "@codemirror/autocomplete";

// ── Lint (error diagnostics) ──
import { lintGutter, lintKeymap } from "@codemirror/lint";

// ── Search ──
import { searchKeymap, highlightSelectionMatches } from "@codemirror/search";

// ── TEX-specific modules ──
import { texLanguageDef, TEX_KEYWORDS, TEX_BUILTINS, TEX_CONSTANTS, TEX_COORD_VARS } from "./tex_language.mjs";
import { createTexCompletions, createTexHover, texHoverAt } from "./tex_completions.mjs";
import { texEditorTheme, texHighlightStyle } from "./tex_theme.mjs";
import { texErrorToDiagnostics, setDiagnostics } from "./tex_lint.mjs";

// ── Assemble a TEX-specific setup (like basicSetup but customized) ──

function texSetup() {
    return [
        lineNumbers(),
        highlightActiveLineGutter(),
        highlightSpecialChars(),
        history(),
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
    createTexHover,
    texHoverAt,   // the tooltip lookup on its own; tests/js_editor_bundle_checks.cjs calls it
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
