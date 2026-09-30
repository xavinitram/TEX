/**
 * TEX Autocomplete Provider for CodeMirror 6
 *
 * One completion source with three branches, chosen by what precedes the cursor:
 *   - `$`: parameter names
 *   - `@`: wire binding names
 *   - a word: the stdlib functions (with signatures and descriptions), keywords and
 *     built-in variables in ALL_COMPLETIONS
 * Plus (LX-6) a hover-tooltip provider reusing the same completion data.
 */
import { hoverTooltip } from "@codemirror/view";
import { ensureSyntaxTree, syntaxTree } from "@codemirror/language";
import { LEX_FUNCTIONS, LEX_ALIASES } from "./tex_lexicon.mjs";

// ─── Stdlib function completions ─────────────────────────────────────
// Generated with the rest of the lexicon (tools/gen_editor_lexicon.py) from the stdlib
// registry, so the list is every registered function and alias and cannot drift.

const STDLIB_COMPLETIONS = [
    ...LEX_FUNCTIONS.map(([label, detail, info]) => ({ label, detail, info })),
    ...LEX_ALIASES.map(([label, target]) => {
        const [, detail, info] = LEX_FUNCTIONS.find(f => f[0] === target);
        return { label, detail, info: `${info} (alias of ${target})` };
    }),
].map(c => ({ ...c, type: "function" }));

// ─── Keyword completions (types and control flow) ────────────────────

const KEYWORD_COMPLETIONS = [
    { label: "float", type: "keyword", info: "Scalar floating-point" },
    { label: "int", type: "keyword", info: "Integer value" },
    { label: "vec2", type: "keyword", detail: "(x, y)", info: "2-component vector" },
    { label: "vec3", type: "keyword", detail: "(r, g, b)", info: "3-component vector (RGB)" },
    { label: "vec4", type: "keyword", detail: "(r, g, b, a)", info: "4-component vector (RGBA)" },
    { label: "mat3", type: "keyword", detail: "(...)", info: "3x3 matrix (internal only)" },
    { label: "mat4", type: "keyword", detail: "(...)", info: "4x4 matrix (internal only)" },
    { label: "string", type: "keyword", info: "Text value (scalar-only)" },
    { label: "if", type: "keyword", info: "Conditional (vectorized via torch.where)" },
    { label: "else", type: "keyword", info: "Else branch" },
    { label: "for", type: "keyword", detail: "(int i = 0; i < n; i++)", info: "Bounded loop" },
    { label: "while", type: "keyword", detail: "(condition)", info: "Conditional loop" },
    { label: "break", type: "keyword", info: "Exit innermost loop" },
    { label: "continue", type: "keyword", info: "Skip to next iteration" },
    { label: "return", type: "keyword", info: "Return a value from a user function" },
    { label: "const", type: "keyword", info: "Declare a value that cannot be reassigned" },
];

// ─── Built-in variable completions ───────────────────────────────────

const VARIABLE_COMPLETIONS = [
    { label: "ix", type: "variable", info: "Pixel x-coordinate (integer)" },
    { label: "iy", type: "variable", info: "Pixel y-coordinate (integer)" },
    { label: "u", type: "variable", info: "Normalized x-coordinate [0, 1]" },
    { label: "v", type: "variable", info: "Normalized y-coordinate [0, 1]" },
    { label: "iw", type: "variable", info: "Image width (pixels)" },
    { label: "ih", type: "variable", info: "Image height (pixels)" },
    { label: "ic", type: "variable", info: "Latent channel count (0 for images)" },
    { label: "fi", type: "variable", info: "Frame/batch index (0 to B-1)" },
    { label: "fn", type: "variable", info: "Total frame/batch count" },
    { label: "px", type: "variable", info: "Pixel step in x: 1.0 / iw" },
    { label: "py", type: "variable", info: "Pixel step in y: 1.0 / ih" },
    { label: "frame", type: "variable", info: "Host timeline frame number (0 in ComfyUI, which has no playhead)" },
    { label: "fps", type: "variable", info: "Host timeline frames per second" },
    { label: "time", type: "variable", info: "Host timeline time in seconds" },
    { label: "PI", type: "constant", info: "3.14159..." },
    { label: "TAU", type: "constant", info: "6.28318... (2 * PI)" },
    { label: "E", type: "constant", info: "2.71828..." },
];

// ─── All non-binding completions ─────────────────────────────────────

const ALL_COMPLETIONS = [...STDLIB_COMPLETIONS, ...KEYWORD_COMPLETIONS, ...VARIABLE_COMPLETIONS];

// ─── LX-6: hover tooltips (reuse the completion data) ─────────────────

const _HOVER_INDEX = new Map(ALL_COMPLETIONS.map(c => [c.label, c]));

// True when `pos` sits inside a comment or a string literal, where a word is prose and
// neither completion nor hover documentation applies. `side` picks which token to read
// when `pos` is a token boundary (-1: the one ending there, 1: the one starting there).
function inCommentOrString(state, pos, side) {
    const tree = ensureSyntaxTree(state, pos, 20) || syntaxTree(state);
    const name = tree.resolveInner(pos, side).name;
    return name === "lineComment" || name === "blockComment" || name === "string";
}

/**
 * The documentation entry under document position `pos`, as `{ from, to, entry }`, or null.
 * A word that is a binding or parameter name (`@length`, `$mix`, `f@u`), a member (`.x`) or
 * part of a comment or string is not the built-in of the same name and has no entry.
 */
export function texHoverAt(state, pos) {
    const line = state.doc.lineAt(pos);
    const text = line.text, base = line.from;
    const isWord = c => c && /[A-Za-z0-9_]/.test(c);
    let start = pos, end = pos;
    while (start > base && isWord(text[start - base - 1])) start--;
    while (end < line.to && isWord(text[end - base])) end++;
    if (start >= end) return null;
    const before = text[start - base - 1], after = text[end - base];
    if (before === "@" || before === "$" || before === "." || after === "@" || after === "$") return null;
    if (inCommentOrString(state, start, 1)) return null;
    const entry = _HOVER_INDEX.get(text.slice(start - base, end - base));
    return entry ? { from: start, to: end, entry } : null;
}

/**
 * Create a hover-tooltip provider: hovering a known TEX token shows its signature
 * (detail) + description (info) from the same ALL_COMPLETIONS the autocomplete uses,
 * so docs are one hover away and can't drift from the completion list.
 */
export function createTexHover() {
    return hoverTooltip((view, pos) => {
        const hit = texHoverAt(view.state, pos);
        if (!hit) return null;
        const { from: start, to: end, entry } = hit;
        return {
            pos: start, end, above: true,
            create() {
                const dom = document.createElement("div");
                dom.className = "cm-tooltip-tex-hover";
                dom.style.cssText = "padding:4px 8px;max-width:360px;font-size:12px";
                const sig = document.createElement("div");
                sig.style.fontWeight = "bold";
                sig.textContent = entry.label + (entry.detail || "");
                dom.appendChild(sig);
                if (entry.info) {
                    const desc = document.createElement("div");
                    desc.style.opacity = "0.85";
                    desc.style.marginTop = "2px";
                    desc.textContent = entry.info;
                    dom.appendChild(desc);
                }
                return { dom };
            },
        };
    });
}

// ─── Completion function factory ─────────────────────────────────────

/**
 * Create a completion source for a specific TEX node.
 * @param {Function} getBindings — returns current @ binding names (inputs + outputs)
 * @param {Function} [getParams] — returns current $ parameter names
 */
export function createTexCompletions(getBindings, getParams) {
    return function texCompletionSource(context) {
        if (inCommentOrString(context.state, context.pos, -1)) return null;

        // ── $ parameter trigger ──
        const dollarMatch = context.matchBefore(/\$\w*/);
        if (dollarMatch) {
            const paramOptions = [];
            try {
                const params = getParams ? getParams() : [];
                for (const name of params) {
                    paramOptions.push({
                        label: "$" + name,
                        type: "variable",
                        info: `Parameter "${name}"`,
                    });
                }
            } catch (_) {}
            return {
                from: dollarMatch.from,
                options: paramOptions,
                validFor: /^\$\w*$/,
            };
        }

        // ── @ binding trigger ──
        // If the user just typed "@", show all available bindings
        const atMatch = context.matchBefore(/@\w*/);
        if (atMatch) {
            const bindingOptions = [];
            // Add dynamic bindings from connected sockets (inputs + outputs)
            try {
                const bindings = getBindings();
                if (bindings && bindings.length) {
                    for (const name of bindings) {
                        bindingOptions.push({
                            label: "@" + name,
                            type: "variable",
                            info: `Binding "${name}"`,
                        });
                    }
                }
            } catch (err) {
                // Ignore binding lookup errors
            }
            return {
                from: atMatch.from,
                options: bindingOptions,
                validFor: /^@\w*$/,
            };
        }

        // ── General word completions ──
        // Activate on 1+ typed characters; a token that starts with a digit is a number
        // (`1.5`, `1e3`, `0xFF`), not a name.
        const word = context.matchBefore(/\w+/);
        if (!word || /^\d/.test(word.text)) return null;

        return {
            from: word.from,
            options: ALL_COMPLETIONS,
            validFor: /^\w*$/,
        };
    };
}
