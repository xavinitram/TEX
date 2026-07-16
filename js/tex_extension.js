/**
 * TEX Wrangle — ComfyUI frontend extension (v5.0 — Nodes v3)
 *
 * Features:
 *   1. Dynamic input sockets: auto-created from @name reads in TEX code.
 *   2. Dynamic output sockets: auto-created from @name = assignments.
 *   3. Parameter widgets ($): f$strength = 0.5 creates a FLOAT widget.
 *   4. CodeMirror 6 editor with syntax highlighting, autocomplete, and
 *      bracket matching (replaces the old overlay-based textarea).
 *   5. Inline error diagnostics: red squiggly underlines at the exact
 *      line/column where errors occur, plus gutter markers.
 *   6. Floating error banner above the node (DOM on document.body,
 *      positioned each frame via onDrawForeground / RAF fallback).
 *   7. Floating "?" help button and "TEX" badge on the node title bar.
 *
 * Implementation notes:
 *   - The CM6 bundle is pre-built by Rollup into tex_cm6_bundle.js and
 *     exposed as globalThis.TEX_CM6. This file accesses that API lazily.
 *   - The original code textarea widget is spliced out of the widgets
 *     array and replaced by a DOM widget named "code" hosting the CM6
 *     EditorView.  This ensures neither the canvas renderer nor the
 *     Vue layer draw a phantom textarea alongside the editor.
 *   - Code changes in the editor are synced back to the original widget
 *     object (kept alive in the updateListener closure) for the debounced
 *     socket scanner, while ComfyUI serialization reads from the DOM
 *     widget's getValue().
 *   - Output slots are sorted alphabetically to match the backend's index
 *     ordering (backend fills slots 0..N-1, pads rest with None).
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// DBG-4: expose the `tex doctor` environment report from the browser console —
// `await texDoctor()` fetches /tex_wrangle/doctor and logs torch/CUDA/Triton/MSVC/cache/
// tier facts. The minimal API-accessible surface the plan called for (a full panel is a
// live-session follow-up); the route itself is tested + never-500.
globalThis.texDoctor = async function () {
    try {
        const resp = await api.fetchApi("/tex_wrangle/doctor");
        const facts = await resp.json();
        console.log("[TEX doctor]", facts);
        return facts;
    } catch (e) {
        console.warn("[TEX doctor] route unavailable:", e);
    }
};

const TEX_NODE_TYPE = "TEX_Wrangle";
// Resolve font URL at module scope where import.meta.url is reliable
const TEX_FONT_URL = (() => {
    try {
        return new URL("MonaspaceNeon.woff2", import.meta.url).href;
    } catch (_) {
        return "/extensions/TEX_Wrangle/MonaspaceNeon.woff2";
    }
})();
// Names reserved by the system — NOT treated as TEX wire or output bindings
const RESERVED_NAMES = new Set(["code", "device", "compile_mode", "_tex_any"]);
const DEBOUNCE_MS = 400;

// ─── Snippet System ─────────────────────────────────────────────────
// Built-in example snippets are fetched from the backend (/tex_wrangle/snippets)
// which reads them from the examples/ directory.  Cached after first fetch.
// User snippets are stored in localStorage.

let _builtinSnippetsCache = null;

async function _fetchBuiltinSnippets() {
    if (_builtinSnippetsCache !== null) return _builtinSnippetsCache;
    try {
        const resp = await api.fetchApi("/tex_wrangle/snippets");
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        _builtinSnippetsCache = await resp.json();
    } catch (e) {
        console.warn("TEX Wrangle: failed to fetch snippets:", e);
        return {};  // don't cache — allow retry on next hover
    }
    return _builtinSnippetsCache;
}

const SNIPPET_STORAGE_KEY = "tex_wrangle_snippets";
const _cmpLocale = (a, b) => a.localeCompare(b, undefined, { sensitivity: "base" });

function _loadUserSnippets() {
    try { return JSON.parse(localStorage.getItem(SNIPPET_STORAGE_KEY) || "{}"); }
    catch (_) { return {}; }
}

function _saveUserSnippets(snippets) {
    localStorage.setItem(SNIPPET_STORAGE_KEY, JSON.stringify(snippets));
}

/** Build a nested tree from flat "folder/folder/name" → content map. */
function _buildSnippetTree(flat) {
    const root = {};
    for (const key of Object.keys(flat).sort(_cmpLocale)) {
        const parts = key.split("/");
        let node = root;
        for (let i = 0; i < parts.length - 1; i++) {
            if (!node[parts[i]] || typeof node[parts[i]] === "string") node[parts[i]] = {};
            node = node[parts[i]];
        }
        node[parts[parts.length - 1]] = flat[key];
    }
    return root;
}

/** Create a standard context-menu row (label + shortcut). */
function _createMenuRow(label, shortcut) {
    const row = document.createElement("div");
    row.className = "tex-context-menu-item";
    const lbl = document.createElement("span");
    lbl.textContent = label;
    row.appendChild(lbl);
    const sc = document.createElement("span");
    sc.className = "tex-context-menu-shortcut";
    sc.textContent = shortcut || "";
    row.appendChild(sc);
    return row;
}

/** Create a menu separator line. */
function _createSeparator() {
    const sep = document.createElement("div");
    sep.style.cssText = "height:1px; background:#333; margin:4px 8px;";
    return sep;
}

/** Position a submenu relative to its anchor row, clamped to viewport. */
function _positionSubmenu(el, anchorRow) {
    const pr = anchorRow.getBoundingClientRect();
    let left = pr.right - 2;
    let top = pr.top;
    const sr = el.getBoundingClientRect();
    if (left + sr.width > window.innerWidth) left = pr.left - sr.width + 2;
    if (top + sr.height > window.innerHeight) top = window.innerHeight - sr.height - 4;
    if (top < 0) top = 4;
    el.style.left = left + "px";
    el.style.top = top + "px";
}

/** All open cascade submenus (for cleanup). */
let _openSubmenus = [];

function _closeAllSubmenus() {
    for (const el of _openSubmenus) {
        for (const row of el.querySelectorAll(":scope > .tex-context-menu-item")) {
            row._childSub = null;
        }
        el.remove();
    }
    _openSubmenus = [];
}

/**
 * Build a cascade submenu DOM element from a snippet tree node.
 *
 * Uses a single hover-timeout per submenu level to avoid races when the
 * mouse moves quickly between rows.  Every mouseenter (folder OR leaf)
 * cancels the pending timer and closes stale sibling subs, so the menu
 * never gets "stuck".
 *
 * @param {object} tree  - nested tree from _buildSnippetTree
 * @param {function} onInsert - callback(content) when a leaf snippet is clicked
 * @param {Element} anchorRow - the row that spawned this submenu (for positioning)
 * @param {Element[]} [extraItems] - optional extra DOM elements appended after tree entries
 * @returns {HTMLElement} the submenu element (already appended to body)
 */
function _createCascadeSubmenu(tree, onInsert, anchorRow, extraItems) {
    const sub = document.createElement("div");
    sub.className = "tex-context-menu tex-snippet-submenu";

    // Shared per-level state — only one timer can be pending at a time
    let pendingTimeout = null;

    /** Close all child submenus except the one belonging to `keepRow`. */
    function closeSiblings(keepRow) {
        for (const ch of sub.querySelectorAll(":scope > .tex-context-menu-item")) {
            if (ch._childSub && ch !== keepRow) {
                _removeSubmenuChain(ch._childSub);
                ch._childSub = null;
            }
        }
    }

    for (const key of Object.keys(tree).sort(_cmpLocale)) {
        const val = tree[key];
        const isFolder = typeof val === "object" && val !== null;
        const row = _createMenuRow(key, isFolder ? "\u25B8" : "");

        if (isFolder) {
            row._childSub = null;

            row.addEventListener("mouseenter", () => {
                clearTimeout(pendingTimeout);
                closeSiblings(row);
                pendingTimeout = setTimeout(() => {
                    if (row._childSub) return;
                    row._childSub = _createCascadeSubmenu(val, onInsert, row);
                }, 50);
            });
            row.addEventListener("mouseleave", () => {
                clearTimeout(pendingTimeout);
            });
        } else {
            row.addEventListener("mouseenter", () => {
                // Leaf row: cancel any pending folder open & close stale subs
                clearTimeout(pendingTimeout);
                closeSiblings(null);
            });
            row.addEventListener("click", (e) => {
                e.stopPropagation();
                onInsert(val);
            });
        }

        sub.appendChild(row);
    }

    // Append optional extra items (separator, action rows, etc.)
    if (extraItems) {
        for (const el of extraItems) sub.appendChild(el);
    }

    document.body.appendChild(sub);
    _openSubmenus.push(sub);
    _positionSubmenu(sub, anchorRow);
    return sub;
}

/** Remove a submenu and all its children recursively. */
function _removeSubmenuChain(el) {
    if (!el) return;
    // Find children
    for (const row of el.querySelectorAll(":scope > .tex-context-menu-item")) {
        if (row._childSub) {
            _removeSubmenuChain(row._childSub);
            row._childSub = null;
        }
    }
    el.remove();
    const idx = _openSubmenus.indexOf(el);
    if (idx !== -1) _openSubmenus.splice(idx, 1);
}

/**
 * Show the "Save Snippet" dialog — prompts for a name, saves content.
 */
function _showSaveSnippetDialog(content) {
    // Remove any existing dialog overlay
    const old = document.querySelector(".tex-snippet-dialog-overlay");
    if (old) old.remove();

    const overlay = document.createElement("div");
    overlay.className = "tex-snippet-dialog-overlay";

    const dialog = document.createElement("div");
    dialog.className = "tex-snippet-dialog";

    dialog.innerHTML = `
        <div class="tex-snippet-dialog-title">Save Snippet</div>
        <div class="tex-snippet-dialog-hint">Use / to create folders, e.g. "My Snippets/Blur"</div>
        <input type="text" class="tex-snippet-dialog-input" placeholder="Snippet name..." spellcheck="false" />
        <div class="tex-snippet-dialog-preview"></div>
        <div class="tex-snippet-dialog-buttons">
            <button class="tex-snippet-btn tex-snippet-btn-cancel">Cancel</button>
            <button class="tex-snippet-btn tex-snippet-btn-save">Save</button>
        </div>
    `;

    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    const input = dialog.querySelector(".tex-snippet-dialog-input");
    const preview = dialog.querySelector(".tex-snippet-dialog-preview");
    const btnCancel = dialog.querySelector(".tex-snippet-btn-cancel");
    const btnSave = dialog.querySelector(".tex-snippet-btn-save");

    // Show preview of snippet content (first 3 lines)
    const lines = content.split("\n");
    preview.textContent = lines.slice(0, 3).join("\n") + (lines.length > 3 ? "\n..." : "");

    function doClose() { overlay.remove(); }

    function doSave() {
        const name = input.value.trim();
        if (!name) { input.focus(); return; }
        const snippets = _loadUserSnippets();
        snippets[name] = content;
        _saveUserSnippets(snippets);
        doClose();
    }

    btnCancel.addEventListener("click", doClose);
    btnSave.addEventListener("click", doSave);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) doClose(); });
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") doSave();
        if (e.key === "Escape") doClose();
        e.stopPropagation();
    });

    requestAnimationFrame(() => input.focus());
}

// Modal factory — the snippet-dialog overlay chrome (backdrop-close + Escape) as one home,
// so new modals don't re-copy it. Returns {overlay, dialog, close}. (The pre-existing
// snippet dialogs still inline their own chrome; they can adopt this later.)
function _texMakeModal(innerHTML) {
    const old = document.querySelector(".tex-snippet-dialog-overlay");
    if (old) old.remove();
    const overlay = document.createElement("div");
    overlay.className = "tex-snippet-dialog-overlay";
    const dialog = document.createElement("div");
    dialog.className = "tex-snippet-dialog";
    dialog.innerHTML = innerHTML;
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);
    const close = () => { overlay.remove(); document.removeEventListener("keydown", onKey); };
    function onKey(e) { if (e.key === "Escape") close(); }
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    document.addEventListener("keydown", onKey);
    return { overlay, dialog, close };
}

// C2-ux — "TEX Doctor" modal: renders the /tex_wrangle/doctor JSON as a key-value table.
// The S-5 arch caveat is surfaced at the top.
function _showDoctorDialog() {
    const { dialog, close } = _texMakeModal(`
        <div class="tex-snippet-dialog-title">TEX Doctor</div>
        <div class="tex-doctor-body">Loading…</div>
        <div class="tex-snippet-dialog-buttons">
            <button class="tex-snippet-btn tex-snippet-btn-cancel">Close</button>
        </div>`);
    dialog.querySelector(".tex-snippet-btn-cancel").addEventListener("click", close);
    const body = dialog.querySelector(".tex-doctor-body");
    Promise.resolve(globalThis.texDoctor ? globalThis.texDoctor() : null).then((facts) => {
        body.innerHTML = facts ? _renderDoctorFacts(facts)
            : "<div class='tex-doctor-row'>Doctor route unavailable — is the server running?</div>";
    });
}

function _renderDoctorFacts(facts) {
    const rows = [];
    const arch = facts.arch || {};
    if (arch.note) rows.push(`<div class="tex-doctor-caveat">⚠ ${arch.note}</div>`);
    const fmt = (v) => typeof v === "boolean"
        ? `<span style="color:${v ? "#66BB6A" : "#FFB300"}">${v}</span>`
        : (Array.isArray(v) ? JSON.stringify(v) : String(v));
    const walk = (obj, prefix) => {
        for (const [k, v] of Object.entries(obj || {})) {
            if (v && typeof v === "object" && !Array.isArray(v)) walk(v, prefix + k + ".");
            else rows.push(`<div class="tex-doctor-row"><span class="tex-doctor-key">`
                + `${prefix}${k}</span><span>${fmt(v)}</span></div>`);
        }
    };
    walk(facts, "");
    return rows.join("");
}

/**
 * Show the "Manage Snippets" dialog — rename / delete user snippets.
 */
function _showManageSnippetsDialog() {
    const old = document.querySelector(".tex-snippet-dialog-overlay");
    if (old) old.remove();

    const overlay = document.createElement("div");
    overlay.className = "tex-snippet-dialog-overlay";

    const dialog = document.createElement("div");
    dialog.className = "tex-snippet-dialog tex-snippet-manage";

    function render() {
        const snippets = _loadUserSnippets();
        const keys = Object.keys(snippets).sort(_cmpLocale);

        dialog.innerHTML = `
            <div class="tex-snippet-dialog-title">Manage Snippets</div>
            <div class="tex-snippet-manage-list"></div>
            <div class="tex-snippet-dialog-buttons">
                <button class="tex-snippet-btn tex-snippet-btn-cancel">Close</button>
            </div>
        `;

        const list = dialog.querySelector(".tex-snippet-manage-list");

        if (keys.length === 0) {
            const empty = document.createElement("div");
            empty.className = "tex-snippet-manage-empty";
            empty.textContent = "No user snippets yet. Select text and use Save Snippet to create one.";
            list.appendChild(empty);
        } else {
            for (const key of keys) {
                const row = document.createElement("div");
                row.className = "tex-snippet-manage-row";

                const name = document.createElement("span");
                name.className = "tex-snippet-manage-name";
                name.textContent = key;
                row.appendChild(name);

                const actions = document.createElement("span");
                actions.className = "tex-snippet-manage-actions";

                const btnRename = document.createElement("button");
                btnRename.className = "tex-snippet-btn-sm";
                btnRename.textContent = "Rename";
                btnRename.addEventListener("click", (e) => {
                    e.stopPropagation();
                    const newName = prompt("New name for snippet:", key);
                    if (newName && newName.trim() && newName.trim() !== key) {
                        const snips = _loadUserSnippets();
                        snips[newName.trim()] = snips[key];
                        delete snips[key];
                        _saveUserSnippets(snips);
                        render();
                    }
                });

                const btnDelete = document.createElement("button");
                btnDelete.className = "tex-snippet-btn-sm tex-snippet-btn-danger";
                btnDelete.textContent = "Delete";
                btnDelete.addEventListener("click", (e) => {
                    e.stopPropagation();
                    if (confirm(`Delete snippet "${key}"?`)) {
                        const snips = _loadUserSnippets();
                        delete snips[key];
                        _saveUserSnippets(snips);
                        render();
                    }
                });

                actions.appendChild(btnRename);
                actions.appendChild(btnDelete);
                row.appendChild(actions);
                list.appendChild(row);
            }
        }

        const btnClose = dialog.querySelector(".tex-snippet-btn-cancel");
        btnClose.addEventListener("click", () => overlay.remove());
    }

    overlay.appendChild(dialog);
    document.body.appendChild(overlay);
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
    render();
}

// ─── CM6 API (lazy resolution from pre-built bundle) ─────────────────
// The bundle (tex_cm6_bundle.js) sets globalThis.TEX_CM6 on load, but load
// order between extension files is not guaranteed. We resolve lazily so
// it works regardless of which file loads first.

let _cm6Cache = null;

function getCM6() {
    if (!_cm6Cache) {
        _cm6Cache = globalThis.TEX_CM6 || null;
    }
    return _cm6Cache;
}

// ─── Editor Tracking ─────────────────────────────────────────────────

const texEditors    = new WeakMap();  // node → EditorView
const texEditorMeta = new WeakMap();  // node → { autocompleteCompartment, completionSource }

// ─── Code Parser (inputs, outputs, params) ──────────────────────────

function parseCode(code) {
    const stripped = code
        .replace(/\/\/[^\n]*/g, "")
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/"(?:[^"\\]|\\.)*"/g, '""');

    // 1) Count ALL @name occurrences (including typed prefixes like f@name)
    const refCounts = new Map();
    const AT_RE = /(?:[a-z]\d{0,2})?@([A-Za-z_]\w*)/g;
    let m;
    while ((m = AT_RE.exec(stripped)) !== null) {
        if (!RESERVED_NAMES.has(m[1])) {
            refCounts.set(m[1], (refCounts.get(m[1]) || 0) + 1);
        }
    }

    // 2) Count simple assignment targets (@name = ..., not ==)
    const simpleAssignCounts = new Map();
    const SIMPLE_RE = /(?:[a-z]\d{0,2})?@([A-Za-z_]\w*)(?:\.[a-zA-Z]+)?\s*=(?!=)/g;
    while ((m = SIMPLE_RE.exec(stripped)) !== null) {
        if (!RESERVED_NAMES.has(m[1])) {
            simpleAssignCounts.set(m[1], (simpleAssignCounts.get(m[1]) || 0) + 1);
        }
    }

    // 3) Find compound assignment targets (+=, -=, *=, /=) — always both read+write
    const compoundTargets = new Set();
    const COMPOUND_RE = /(?:[a-z]\d{0,2})?@([A-Za-z_]\w*)(?:\.[a-zA-Z]+)?\s*[+\-*/]=/g;
    while ((m = COMPOUND_RE.exec(stripped)) !== null) {
        if (!RESERVED_NAMES.has(m[1])) {
            compoundTargets.add(m[1]);
        }
    }

    // 4) Detect parameter bindings ($name with optional type prefix)
    const params = new Map(); // name → { typeHint, defaultValue }
    // Explicit declarations: f$strength = 0.5; or i$count; (supports all type prefixes)
    // Lookbehind requires statement boundary (^, ;, {, }) to avoid matching
    // $name inside expressions like for-loop headers: for (int dy = -$radius; ...)
    const PARAM_DECL_RE = /(?<=(?:^|[;{}])\s*)(?:(img|v[234]|[fismvlcb]))?\$([A-Za-z_]\w*)\s*(?:=\s*([^;]+))?\s*;/gm;
    while ((m = PARAM_DECL_RE.exec(stripped)) !== null) {
        params.set(m[2], {
            typeHint: m[1] || "f",
            defaultValue: m[3]?.trim() || null,
        });
    }
    // Also find $name references not yet captured (preserve type prefix from reference)
    const PARAM_REF_RE = /([a-z]\d{0,2})?\$([A-Za-z_]\w*)/g;
    while ((m = PARAM_REF_RE.exec(stripped)) !== null) {
        if (!params.has(m[2]) && !RESERVED_NAMES.has(m[2])) {
            params.set(m[2], { typeHint: m[1] || "f", defaultValue: null });
        }
    }

    // 5) Build input and output sets
    const outputs = new Set();
    const inputs = new Set();

    for (const [name, total] of refCounts) {
        if (params.has(name)) continue;

        const simpleAssigns = simpleAssignCounts.get(name) || 0;
        const isCompound = compoundTargets.has(name);
        const isAssigned = simpleAssigns > 0 || isCompound;

        if (isAssigned) outputs.add(name);

        // It's an input if compound (read-modify-write) or has reads beyond assignments
        if (isCompound || total > simpleAssigns) {
            inputs.add(name);
        }
    }

    // Remove param names from outputs (defensive)
    for (const name of params.keys()) {
        outputs.delete(name);
    }

    return { inputs, outputs, params };
}

// ─── Debounce ────────────────────────────────────────────────────────

function debounce(fn, ms) {
    let timer = null;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), ms);
    };
}

// ─── Auto-Socket Management ─────────────────────────────────────────

function syncInputs(node, usedBindings, paramNames) {
    if (!node.graph) return;
    const ANY_TYPE = "*";
    // paramNames: Set of $param names that have their own input sockets
    // managed by syncParams — don't touch them here.
    const paramSet = paramNames || new Set();

    // Build map of current TEX binding inputs
    const currentInputs = new Map();
    if (node.inputs) {
        for (let i = 0; i < node.inputs.length; i++) {
            currentInputs.set(node.inputs[i].name, i);
        }
    }

    // Desired set — sorted for stable ordering
    const desired = [...usedBindings].sort();

    // Remove unneeded inputs (reverse order for index stability)
    // Skip param-owned sockets — those are managed by syncParams
    const toRemove = [];
    for (const [name, idx] of currentInputs) {
        if (!usedBindings.has(name) && !paramSet.has(name)) {
            toRemove.push(idx);
        }
    }
    toRemove.sort((a, b) => b - a);
    for (const idx of toRemove) node.removeInput(idx);

    // Rebuild map after removals
    const afterRemove = new Set(node.inputs ? node.inputs.map(i => i.name) : []);

    // Add missing inputs
    for (const name of desired) {
        if (!afterRemove.has(name)) {
            node.addInput(name, ANY_TYPE);
            const newInput = node.inputs[node.inputs.length - 1];
            newInput.tooltip = `Use in TEX: @${name}\nAccepts any type \u2014 connects to images, masks, scalars, latents, or strings.`;
        }
    }

    node.setDirtyCanvas(true, true);
}

// ─── Dynamic Output Sockets ─────────────────────────────────────────

function syncOutputs(node, outputNames) {
    if (!node.graph) return;
    const ANY_TYPE = "*";

    // Desired: sorted alphabetically (must match backend index order)
    const desired = [...outputNames].sort();

    // Quick check: already matching?
    const currentNames = node.outputs ? node.outputs.map(o => o.name) : [];
    if (currentNames.length === desired.length &&
        currentNames.every((n, i) => n === desired[i])) {
        return;
    }

    // Save existing connections by output name: name → [{targetNodeId, targetSlot}]
    const savedConnections = new Map();
    if (node.outputs) {
        for (const out of node.outputs) {
            if (out.links && out.links.length > 0) {
                const conns = [];
                for (const linkId of out.links) {
                    const link = node.graph.links?.[linkId];
                    if (link) {
                        conns.push({
                            targetNodeId: link.target_id,
                            targetSlot: link.target_slot,
                        });
                    }
                }
                if (conns.length > 0) {
                    savedConnections.set(out.name, conns);
                }
            }
        }
    }

    // Remove all existing outputs (reverse order)
    if (node.outputs) {
        for (let i = node.outputs.length - 1; i >= 0; i--) {
            node.removeOutput(i);
        }
    }

    // Re-add in sorted order (indices must match backend's sorted output slots)
    for (const name of desired) {
        node.addOutput(name, ANY_TYPE);
        const newOutput = node.outputs[node.outputs.length - 1];
        newOutput.tooltip = `TEX output: @${name}\nType auto-inferred from code.\nWrite to this with: @${name} = expr;`;
    }

    // Restore connections for outputs that still exist (at their new indices)
    for (const [name, conns] of savedConnections) {
        const newIdx = desired.indexOf(name);
        if (newIdx < 0) continue; // output was removed

        for (const conn of conns) {
            try {
                const targetNode = node.graph.getNodeById(conn.targetNodeId);
                if (targetNode) {
                    node.connect(newIdx, targetNode, conn.targetSlot);
                }
            } catch (_) {
                // Connection failed — target may have been removed
            }
        }
    }

    node.setDirtyCanvas(true, true);
}

// ─── Dynamic Param Schema Registry ──────────────────────────────────
// nodeData.input.optional is TYPE-GLOBAL: every TEX node shares the one
// schema object, so dynamic $param entries are refcounted per node here
// and only dropped once no node declares the name anymore.

const _texParamRegistry = new Map();        // paramName → Map(nodeId → typeHint)
const _texParamConflictWarned = new Set();  // param names already warned about

function _texParamSchemaEntry(typeHint) {
    if (typeHint === "i") return ["INT", { default: 0, min: -9999, max: 9999 }];
    if (typeHint === "b") return ["BOOLEAN", { default: false }];
    if (typeHint === "s") return ["STRING", { default: "" }];
    if (typeHint === "c") return ["STRING", { default: "#000000" }];
    if (typeHint === "v2") return ["STRING", { default: "0.0, 0.0" }];
    if (typeHint === "v3") return ["STRING", { default: "0.0, 0.0, 0.0" }];
    return ["FLOAT", { default: 0.0, min: -9999, max: 9999, step: 0.1, round: 0.001 }];
}

/**
 * Replace `node`'s declarations in the registry (params = null drops them
 * all, e.g. on node removal) and rebuild the shared schema from the union.
 * Only names the registry itself added are ever deleted from the schema.
 */
function _texSyncParamSchema(node, params) {
    if (typeof LiteGraph === "undefined") return;
    const optional = LiteGraph.registered_node_types?.[TEX_NODE_TYPE]?.nodeData?.input?.optional;
    if (!optional) return;

    for (const [name, owners] of _texParamRegistry) {
        owners.delete(node.id);
        if (owners.size === 0) {
            _texParamRegistry.delete(name);
            delete optional[name];
        }
    }
    if (params) {
        for (const [name, info] of params) {
            let owners = _texParamRegistry.get(name);
            if (!owners) {
                owners = new Map();
                _texParamRegistry.set(name, owners);
            }
            owners.set(node.id, info.typeHint || "f");
        }
    }

    for (const [name, owners] of _texParamRegistry) {
        // Same name on several nodes: the schema is type-global, so pick a
        // deterministic winner (lowest node id) and warn once on conflict.
        let winId = null;
        for (const id of owners.keys()) {
            if (winId === null || String(id) < String(winId)) winId = id;
        }
        const typeHint = owners.get(winId);
        if (!_texParamConflictWarned.has(name) && new Set(owners.values()).size > 1) {
            _texParamConflictWarned.add(name);
            console.warn(`[TEX] $${name} is declared with different types on multiple nodes; the shared schema uses "${typeHint}" (node ${winId}).`);
        }
        optional[name] = _texParamSchemaEntry(typeHint);
    }
}

// ─── Parameter Widgets ($) ──────────────────────────────────────────

function syncParams(node, params) {
    // params: Map of name → { typeHint, defaultValue }
    // Creates/removes parameter widgets with linked input sockets.
    // In modern ComfyUI, widgets and input sockets co-exist: the widget
    // provides manual value entry, and the linked socket accepts wires.
    // When a wire connects, the wired value overrides the widget value.
    // The graphToPrompt hook injects widget values for unconnected params.
    if (!node.widgets) node.widgets = [];

    // Build map of current param widgets: name → { widget, index, typeHint }
    const currentParams = new Map();
    for (let i = 0; i < node.widgets.length; i++) {
        const w = node.widgets[i];
        if (w._texParam) {
            currentParams.set(w.name, { widget: w, index: i, typeHint: w._texTypeHint || "f" });
        }
    }

    // Desired param names
    const desiredNames = new Set(params.keys());

    // Identify widgets to remove: stale names OR type changed
    const toRemove = new Set();
    for (const [name, cur] of currentParams) {
        if (!desiredNames.has(name)) {
            toRemove.add(name);
        } else {
            // Check if type hint changed (e.g. f$x → i$x)
            const desired = params.get(name);
            if (desired && (desired.typeHint || "f") !== cur.typeHint) {
                toRemove.add(name);
            }
        }
    }

    // Remove stale/changed widgets (reverse order for index stability)
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        if (node.widgets[i]._texParam && toRemove.has(node.widgets[i].name)) {
            node.widgets.splice(i, 1);
        }
    }

    // Remove stale param input sockets (reverse order for index stability)
    if (node.inputs) {
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            if (node.inputs[i]._texParam && toRemove.has(node.inputs[i].name)) {
                node.removeInput(i);
            }
        }
    }

    // Rebuild current set after removals
    const remaining = new Set();
    for (const w of node.widgets) {
        if (w._texParam) remaining.add(w.name);
    }

    // Map type hints to ComfyUI type strings for input sockets
    function paramComfyType(typeHint) {
        if (typeHint === "i") return "INT";
        if (typeHint === "b") return "BOOLEAN";
        if (typeHint === "s") return "STRING";
        if (typeHint === "c") return "STRING";   // hex color string
        if (typeHint === "v2") return "STRING";   // comma-separated floats
        if (typeHint === "v3") return "STRING";   // comma-separated floats
        return "FLOAT";
    }

    // Add new / recreated param widgets with linked input sockets
    for (const [name, info] of params) {
        if (remaining.has(name)) continue;

        const typeHint = info.typeHint || "f";
        const typeName = paramComfyType(typeHint);
        let widget;
        if (typeHint === "b") {
            // Boolean → toggle checkbox
            const raw = info.defaultValue != null ? String(info.defaultValue).trim() : "0";
            const def = raw === "1" || raw === "true";
            widget = node.addWidget("toggle", name, def, () => {});
        } else if (typeHint === "c") {
            // Color → text input with hex default
            let def = "#000000";
            if (info.defaultValue != null) {
                const raw = String(info.defaultValue).trim().replace(/^["']|["']$/g, "");
                if (raw.startsWith("#")) def = raw;
            }
            widget = node.addWidget("text", name, def, () => {});
        } else if (typeHint === "v2" || typeHint === "v3") {
            // Vec2/Vec3 → text input with comma-separated defaults
            const n = typeHint === "v2" ? 2 : 3;
            let def = Array(n).fill("0.0").join(", ");
            if (info.defaultValue != null) {
                const raw = String(info.defaultValue).trim();
                const vecMatch = raw.match(new RegExp(`vec${n}\\s*\\(\\s*([^)]+)\\)`));
                if (vecMatch) def = vecMatch[1].trim();
                else {
                    const cleaned = raw.replace(/^["']|["']$/g, "");
                    if (cleaned) def = cleaned;
                }
            }
            widget = node.addWidget("text", name, def, () => {});
        } else if (typeHint === "i") {
            const parsed = info.defaultValue != null ? parseInt(info.defaultValue) : NaN;
            const def = Number.isFinite(parsed) ? parsed : 0;
            widget = node.addWidget("number", name, def, () => {}, {
                min: -9999, max: 9999, step: 10, precision: 0,
            });
        } else if (typeHint === "s") {
            const raw = info.defaultValue || "";
            const def = raw.replace(/^["']|["']$/g, "");
            widget = node.addWidget("text", name, def, () => {});
        } else {
            // float (default)
            const parsed = info.defaultValue != null ? parseFloat(info.defaultValue) : NaN;
            const def = Number.isFinite(parsed) ? parsed : 0.0;
            widget = node.addWidget("number", name, def, () => {}, {
                min: -9999, max: 9999, step: 0.1, precision: 3,
            });
        }
        if (widget) {
            widget._texParam = true;
            widget._texTypeHint = typeHint;
            const prefixMap = { i: "i", s: "s", b: "b", c: "c", v2: "v2", v3: "v3" };
            const prefixChar = prefixMap[typeHint] || "f";
            const defaultStr = info.defaultValue != null ? String(info.defaultValue) : (typeHint === "s" ? '""' : typeHint === "b" ? "0" : typeHint === "c" ? '"#000000"' : "0");
            widget.tooltip = `TEX parameter: $${name} (${typeName})\nDeclare in code: ${prefixChar}$${name} = ${defaultStr};\nUse in expressions: $${name}\nConnect a wire or adjust the widget value.`;
        }

        // Create a linked input socket for the widget.
        // Setting input.widget = { name } tells ComfyUI's modern frontend
        // that this socket is the wired counterpart of the widget above.
        // They co-exist as a single unified control — no double rendering.
        const existingInput = node.inputs?.find(inp => inp.name === name);
        if (!existingInput) {
            // Vec/color params accept any wire type (IMAGE, MASK, etc.) to override widget
            const socketType = (typeHint === "c" || typeHint === "v2" || typeHint === "v3") ? "*" : typeName;
            node.addInput(name, socketType);
            const newInput = node.inputs[node.inputs.length - 1];
            newInput._texParam = true;
            newInput.widget = { name: name };
            newInput.tooltip = `Wire input for parameter $${name} (${typeName})\nOverrides widget value when connected.`;
        }
    }

    // Register dynamic params in nodeData.input.optional so ComfyUI's Vue
    // overlay recognizes them and enables click/drag interaction.
    _texSyncParamSchema(node, params);

    node.setDirtyCanvas(true, true);
}

// ─── Code → Sockets ─────────────────────────────────────────────────
// Parse the code and bring inputs, outputs and param widgets in sync.
// Shared by the debounced editor updater and onConfigure.

function applyCodeToSockets(node, code) {
    const { inputs, outputs, params } = parseCode(code);
    node._texBindings = inputs;
    node._texOutputs = outputs;
    node._texParams = params;
    syncInputs(node, inputs, new Set(params.keys()));
    syncOutputs(node, outputs);
    syncParams(node, params);
}

// ─── Error Cache (per-node, from WebSocket events) ───────────────────

const texErrorCache = new Map(); // nodeId -> { message, type, traceback }

// ─── Floating DOM Overlays ───────────────────────────────────────────
// Error panel and ? button are positioned as fixed divs on document.body,
// placed at the node's screen coordinates each frame via onDrawForeground.
// This works in BOTH Nodes 1.0 and 2.0 because:
//   - onDrawForeground fires in both (proven by TEX badge visibility)
//   - Fixed DOM elements on document.body capture clicks in both

function showDOMErrorBanner(node, errMsg) {
    clearDOMErrorBanner(node);

    const diagIdx = errMsg.indexOf("TEX_DIAG:");
    if (diagIdx < 0) {
        const banner = _createBannerElement(errMsg);
        document.body.appendChild(banner);
        node._texErrorBanner = banner;
        node._texOverlaySig = null; // force the RAF loop to position it
        return;
    }

    let diags;
    try { diags = JSON.parse(errMsg.slice(diagIdx + 9)); } catch { return; }
    if (!diags?.length) return;

    const banner = document.createElement("div");
    banner.className = "tex-floating-error";
    banner.style.display = "none"; // Hidden until onDrawForeground positions it

    const copyParts = [];

    for (let di = 0; di < diags.length; di++) {
        const d = diags[di];
        if (di > 0) {
            banner.appendChild(document.createElement("hr"));
            copyParts.push("");
        }

        // Main message — coral
        const msgEl = document.createElement("div");
        msgEl.style.cssText = "color: #f0a0a0; font-weight: bold; margin-bottom: 2px;";
        msgEl.textContent = `\u{1F914}  ${d.message}`;
        banner.appendChild(msgEl);
        copyParts.push(d.message);

        // Source snippet — grey
        if (d.source_line && d.line) {
            const srcEl = document.createElement("div");
            srcEl.style.cssText = "color: #cccccc; padding-left: 8px; white-space: pre;";
            srcEl.textContent = `${d.line} | ${d.source_line}`;
            banner.appendChild(srcEl);
            copyParts.push(`  ${d.line} | ${d.source_line}`);

            if (d.col && d.col > 0) {
                const pointerEl = document.createElement("div");
                pointerEl.style.cssText = "color: #ff9966; padding-left: 8px; white-space: pre;";
                const gutter = `${d.line} | `;
                const pad = " ".repeat(gutter.length + d.col - 1);
                pointerEl.textContent = `${pad}\u261D\uFE0F`;
                banner.appendChild(pointerEl);
            }
        }

        // Suggestions — amber, with quick-fix buttons
        if (d.suggestions?.length) {
            const sugEl = document.createElement("div");
            sugEl.style.cssText = "color: #e0c090; padding-left: 8px; margin-top: 2px;";
            const sugRaw = d.suggestions.join(", ");

            sugEl.appendChild(document.createTextNode(`\u{1F6E0}\uFE0F  Try: `));

            for (let si = 0; si < d.suggestions.length; si++) {
                const suggestion = d.suggestions[si];
                if (si > 0) sugEl.appendChild(document.createTextNode(", "));
                sugEl.appendChild(document.createTextNode(suggestion));

                const fixLink = document.createElement("span");
                fixLink.textContent = " [Fix]";
                fixLink.style.cssText = "color: #88bbdd; cursor: pointer; font-size: 10px; margin-left: 2px;";
                fixLink.addEventListener("mouseenter", () => { fixLink.style.textDecoration = "underline"; });
                fixLink.addEventListener("mouseleave", () => { fixLink.style.textDecoration = "none"; });
                fixLink.addEventListener("click", ((sug, diag) => (e) => {
                    e.stopPropagation();
                    if (!diag.source_line || !diag.col) return;
                    const wrongStart = diag.col - 1;
                    let wrongEnd = wrongStart;
                    while (wrongEnd < diag.source_line.length && /[\w]/.test(diag.source_line[wrongEnd])) wrongEnd++;
                    const wrongText = diag.source_line.substring(wrongStart, wrongEnd);
                    const editor = texEditors.get(node);
                    if (editor && wrongText) {
                        const line = editor.state.doc.line(diag.line);
                        const from = line.from + wrongStart;
                        const to = from + wrongText.length;
                        editor.dispatch({ changes: { from, to, insert: sug } });
                        clearDOMErrorBanner(node);
                    }
                })(suggestion, d));
                sugEl.appendChild(fixLink);
            }

            banner.appendChild(sugEl);
            copyParts.push(`  > Try: ${sugRaw}`);
        }

        if (d.hint) {
            const hintEl = document.createElement("div");
            hintEl.style.cssText = "color: #e0c090; padding-left: 8px;";
            hintEl.textContent = `\u{1F9E9}  Help: ${d.hint}`;
            banner.appendChild(hintEl);
            copyParts.push(`  > Help: ${d.hint}`);
        }

        if (d.code) {
            const codeEl = document.createElement("div");
            codeEl.style.cssText = "color: #888888; padding-left: 8px; margin-top: 2px;";
            codeEl.textContent = `Error Code: ${d.code}`;
            banner.appendChild(codeEl);
            copyParts.push(`  > Error Code: ${d.code}`);
        }
    }

    // Copy icon
    const copyIcon = document.createElement("span");
    copyIcon.style.cssText = "position: absolute; top: 4px; right: 6px; color: #777; cursor: pointer; font-size: 12px;";
    copyIcon.textContent = "\u{1F4CB}";
    copyIcon.title = "Copy error";
    const fullCopyText = copyParts.join("\n");
    const doCopy = () => {
        navigator.clipboard.writeText(fullCopyText).then(() => {
            copyIcon.textContent = "\u2713 copied";
            copyIcon.style.color = "#88dd88";
            setTimeout(() => { copyIcon.textContent = "\u{1F4CB}"; copyIcon.style.color = "#777"; }, 1200);
        }).catch(() => {});
    };
    copyIcon.addEventListener("click", (e) => { e.stopPropagation(); doCopy(); });
    banner.appendChild(copyIcon);
    banner.addEventListener("click", doCopy);
    banner.addEventListener("mousedown", (e) => e.stopPropagation());
    banner.addEventListener("pointerdown", (e) => e.stopPropagation());

    document.body.appendChild(banner);
    node._texErrorBanner = banner;
    node._texOverlaySig = null; // force the RAF loop to position it
}

function _createBannerElement(text) {
    const banner = document.createElement("div");
    banner.className = "tex-floating-error";
    banner.style.color = "#f0a0a0";
    banner.style.whiteSpace = "pre-wrap";
    banner.style.display = "none"; // Hidden until onDrawForeground positions it
    banner.textContent = text;
    banner.addEventListener("mousedown", (e) => e.stopPropagation());
    banner.addEventListener("pointerdown", (e) => e.stopPropagation());
    return banner;
}

function clearDOMErrorBanner(node) {
    if (node._texErrorBanner) {
        node._texErrorBanner.remove();
        node._texErrorBanner = null;
        node._texOverlaySig = null;
    }
}

// ─── Floating Overlay Positioning ─────────────────────────────────────
// Shared logic for positioning error banner + ? button above a node.
// Called from onDrawForeground (Nodes 1.0) and from a RAF loop (Nodes 2.0).

function _texEnsureHelpBtn(node) {
    if (node._texHelpBtn) return node._texHelpBtn;
    const btn = document.createElement("button");
    btn.className = "tex-floating-help-btn";
    btn.textContent = "?";
    btn.title = "TEX Reference (F1)";
    btn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        showHelpPopup(e.clientX || 200, e.clientY || 200);
    });
    btn.addEventListener("mousedown", (e) => { e.stopPropagation(); });
    btn.addEventListener("pointerdown", (e) => { e.stopPropagation(); });
    document.body.appendChild(btn);
    node._texHelpBtn = btn;
    return btn;
}

// C1-ux — the perf HUD as a floating DOM badge (the Nodes-1.0/2.0 dual-path guarantee;
// the canvas draw stays for Nodes-1.0 aesthetics). Populated from the same tex_perf/
// tex_probes payload; the hover title carries the C7-ux reason/precision_reason.
function _texEnsurePerfBadge(node) {
    if (node._texPerfBadge) return node._texPerfBadge;
    const b = document.createElement("div");
    b.className = "tex-floating-perf-badge";
    b.addEventListener("mousedown", (e) => e.stopPropagation());
    b.addEventListener("pointerdown", (e) => e.stopPropagation());
    document.body.appendChild(b);
    node._texPerfBadge = b;
    return b;
}

function _texUpdatePerfBadge(node) {
    const p = node._texPerf;
    if (!(_texPerfHudEnabled && p)) {
        if (node._texPerfBadge) node._texPerfBadge.style.display = "none";
        return;
    }
    const b = _texEnsurePerfBadge(node);
    const prec = p.precision && p.precision !== "fp32" ? " · " + p.precision : "";
    const fb = p.fallback_from ? " (←" + p.fallback_from + ")" : "";
    let txt = p.tier + fb + " · " + Number(p.elapsed_ms).toFixed(1) + "ms" + prec;
    if (p.near_singularities) txt += " · ⬤" + p.near_singularities;  // C4-ux count
    const probes = node._texProbes;                                  // debug_print taps
    if (probes && probes.length) {
        txt += probes.slice(0, 4).map((pr) => `\n${pr.label}=${pr.value} @${pr.x},${pr.y}`).join("");
    }
    b.textContent = txt;
    b.style.color = p.fallback_from ? "#FFB300" : "#66BB6A";
    b.title = [p.reason && ("tier: " + p.reason),
               p.precision_reason && ("precision: " + p.precision_reason)]
        .filter(Boolean).join("\n") || "TEX perf";
    b.style.display = "";
}

function _texPositionOverlaysFromRect(node, rect, scale, titleInside) {
    // rect: { left, top, width } in screen (client) coordinates
    // scale: zoom level — overlays scale proportionally with the node
    // titleInside: true when rect.top is the node's OUTER top (title bar included,
    //   as getBoundingClientRect returns in Nodes 2.0); false when rect.top is the
    //   node BODY top with the title bar above it (the Nodes 1.0 canvas convention).
    const tooSmall = scale < 0.3;

    // ── Error banner ──
    if (node._texErrorBanner) {
        const el = node._texErrorBanner;
        if (tooSmall) {
            el.style.display = "none";
        } else {
            el.style.display = "";
            // Width matches the node; CSS transform scales the content
            el.style.width = (rect.width / scale) + "px";
            el.style.left = rect.left + "px";
            el.style.transform = `scale(${scale})`;
            el.style.transformOrigin = "bottom left";
            // Position above the node title bar
            el.style.top = (rect.top - el.offsetHeight * scale - 4) + "px";
        }
    }

    // ── Help ? button ──
    const btn = _texEnsureHelpBtn(node);
    if (tooSmall) {
        btn.style.display = "none";
    } else {
        btn.style.display = "";
        const btnSize = 20 * scale;
        btn.style.width = btnSize + "px";
        btn.style.height = btnSize + "px";
        btn.style.fontSize = (12 * scale) + "px";
        // Nodes 1.0 draws a "TEX" badge at the far right, so the ? goes to its left
        // (36px in). Nodes 2.0 has no badge, so the ? can sit nearer the edge.
        btn.style.left = (rect.left + rect.width - btnSize - (titleInside ? 14 : 36) * scale) + "px";
        // Nodes 1.0: rect.top is the body top, so the button sits just above it (in
        // the title bar). Nodes 2.0: rect.top already includes the title bar, so
        // place the button down inside that top edge — otherwise it floats above.
        btn.style.top = (titleInside ? rect.top + 6 * scale : rect.top - btnSize - 2) + "px";
    }

    // ── C1-ux perf badge (below the node) ──
    const badge = node._texPerfBadge;
    if (badge && badge.style.display !== "none") {
        if (tooSmall) {
            badge.style.display = "none";
        } else {
            const nodeH = (node.size ? node.size[1] : 0) * scale;
            badge.style.transform = `scale(${scale})`;
            badge.style.transformOrigin = "top left";
            badge.style.left = rect.left + "px";
            // Nodes 1.0: rect.top is the body top → node bottom is +nodeH. Nodes 2.0:
            // rect.top includes the title bar, so add a little for it. Tuned live.
            badge.style.top = (rect.top + nodeH + (titleInside ? 34 * scale : 6)) + "px";
        }
    }
}

// Called from onDrawForeground (Nodes 1.0 — fires every frame)
function _texPositionOverlays(node, ctx) {
    const t = ctx.getTransform();
    const canvas = ctx.canvas;
    const canvasRect = canvas.getBoundingClientRect();
    const scale = t.a;
    const screenX = canvasRect.left + t.e;
    const screenY = canvasRect.top + t.f;
    const scaledWidth = node.size[0] * scale;

    _texPositionOverlaysFromRect(node, {
        left: screenX,
        top: screenY,
        width: scaledWidth,
    }, scale);
}

// ─── Nodes 2.0 Fallback: RAF-based positioning via Vue DOM element ───
// In Nodes 2.0 (vueNodesMode), onDrawForeground is NOT called per-node.
// Instead, we find the Vue-rendered [data-node-id="X"] element and use
// its bounding rect to position overlays.

const _texOverlayNodes = new Set(); // all TEX nodes needing overlay positioning

function _texStartV2Positioning(node) {
    _texOverlayNodes.add(node);
}

function _texStopV2Positioning(node) {
    _texOverlayNodes.delete(node);
}

// Single RAF loop — handles positioning for nodes where onDrawForeground
// didn't fire (Nodes 2.0). In Nodes 1.0, onDrawForeground sets
// _texV1Positioned=true each frame; the RAF loop clears it after skipping.
let _texRafRunning = false;
function _texOverlayRafLoop() {
    if (_texOverlayNodes.size === 0) {
        _texRafRunning = false;
        return;
    }
    // Nodes 1.0 vs 2.0 is a global, live-togglable setting. false means the
    // Vue node layer is off ([data-node-id] elements don't exist), so the DOM
    // lookup below can never match; undefined (older frontend / renamed id)
    // falls through to the lookup so behavior there is unchanged.
    let vueMode;
    try { vueMode = app.ui.settings.getSettingValue("Comfy.VueNodes.Enabled"); }
    catch (_) { vueMode = undefined; }
    for (const node of _texOverlayNodes) {
        // In Nodes 1.0, onDrawForeground already positioned overlays.
        // Clear the flag so next frame we can detect whether it fired again.
        if (node._texV1Positioned) {
            node._texV1Positioned = false;
            node._texOverlaySig = null; // V1 places overlays with its own conventions
            continue;
        }

        if (vueMode === false) continue;

        const collapsed = node.flags?.collapsed;
        if (collapsed) {
            if (node._texErrorBanner) node._texErrorBanner.style.display = "none";
            if (node._texHelpBtn) node._texHelpBtn.style.display = "none";
            node._texOverlaySig = null;
            continue;
        }

        // Find the Vue-rendered DOM element for this node (Nodes 2.0)
        let el = node._texVueEl;
        if (!el || !el.isConnected) {
            el = document.querySelector(`[data-node-id="${node.id}"]`);
            node._texVueEl = el;
        }
        if (!el) continue;

        const rect = el.getBoundingClientRect();
        if (rect.width === 0 && rect.height === 0) continue;

        // Estimate scale from node size vs rendered size
        const scale = node.size[0] > 0 ? rect.width / node.size[0] : 1;

        // Skip the style writes when nothing moved. Banner create/remove and
        // collapse reset the signature so fresh overlays position immediately.
        const sig = rect.left + "," + rect.top + "," + rect.width + "," + scale + "," + (node._texErrorBanner ? "b" : "");
        if (sig === node._texOverlaySig) continue;

        _texPositionOverlaysFromRect(node, {
            left: rect.left,
            top: rect.top,
            width: rect.width,
        }, scale, true);   // Nodes 2.0: rect.top includes the title bar
        node._texOverlaySig = sig;
    }
    requestAnimationFrame(_texOverlayRafLoop);
}

function _texEnsureV2Raf() {
    if (!_texRafRunning) {
        _texRafRunning = true;
        requestAnimationFrame(_texOverlayRafLoop);
    }
}

api.addEventListener("execution_error", (ev) => {
    const d = ev.detail;
    if (!d?.node_id) return;

    const errData = {
        message: d.exception_message || "Unknown error",
        type: d.exception_type || "",
        traceback: d.traceback || "",
    };

    texErrorCache.set(String(d.node_id), errData);

    // Log human-readable error to browser console
    const errMsg = errData.message || "";
    const diagIdx = errMsg.indexOf("TEX_DIAG:");
    if (diagIdx >= 0) {
        try {
            const diags = JSON.parse(errMsg.slice(diagIdx + 9));
            for (const dg of diags) {
                let readable = `[TEX] ${dg.message}`;
                if (dg.source_line && dg.line) readable += `\n  ${dg.line} | ${dg.source_line}`;
                if (dg.suggestions?.length) readable += `\n  > Try: ${dg.suggestions.join(", ")}`;
                if (dg.hint) readable += `\n  > Help: ${dg.hint}`;
                if (dg.code) readable += `\n  > Error Code: ${dg.code}`;
                console.warn(readable);
            }
        } catch { /* ignore parse failures */ }
    }

    // Push inline diagnostics to the CM6 editor + DOM banner
    const node = app.graph?.getNodeById?.(d.node_id);
    if (node) {
        // DOM error banner — works in ALL frontend modes
        showDOMErrorBanner(node, errMsg);

        // CM6 inline diagnostics (squiggles)
        const CM6 = getCM6();
        const editor = texEditors.get(node);
        if (CM6 && editor) {
            try {
                const diagnostics = CM6.texErrorToDiagnostics(editor, errData);
                editor.dispatch(CM6.setDiagnostics(editor.state, diagnostics));
            } catch (err) {
                console.warn("[TEX] Failed to push error diagnostics:", err);
            }
        }
    }
});

api.addEventListener("execution_start", () => {
    // Clear errors at the start of a new prompt
    texErrorCache.clear();

    // Clear DOM banners + inline diagnostics from all TEX nodes
    const CM6 = getCM6();
    if (app.graph?._nodes) {
        for (const node of app.graph._nodes) {
            // Clear DOM banner (works for any node type)
            clearDOMErrorBanner(node);

            // Clear CM6 diagnostics
            if (node.type === TEX_NODE_TYPE) {
                const editor = texEditors.get(node);
                if (CM6 && editor) {
                    try {
                        editor.dispatch(CM6.setDiagnostics(editor.state, []));
                    } catch (err) {
                        console.warn("[TEX] Failed to clear diagnostics:", err);
                    }
                }
            }
        }
    }
});

// ─── CodeMirror 6 Editor Creation ────────────────────────────────────

function createTexEditor(node, codeWidget, updateSockets) {
    const CM6 = getCM6();
    if (!CM6) {
        console.warn("[TEX] CM6 bundle not available — falling back to default textarea");
        return null;
    }

    const container = document.createElement("div");
    container.className = "tex-cm-container";

    // Dynamic @ binding completions — reads from node's current sockets
    function getBindings() {
        const names = [];
        if (node.inputs) {
            for (const inp of node.inputs) {
                if (inp.name && !RESERVED_NAMES.has(inp.name)) {
                    names.push(inp.name);
                }
            }
        }
        if (node.outputs) {
            for (const out of node.outputs) {
                if (out.name && !RESERVED_NAMES.has(out.name)) {
                    if (!names.includes(out.name)) names.push(out.name);
                }
            }
        }
        // Also parse code for bindings not yet synced
        const code = codeWidget.value || "";
        const { inputs, outputs } = parseCode(code);
        for (const name of inputs) {
            if (!names.includes(name)) names.push(name);
        }
        for (const name of outputs) {
            if (!names.includes(name)) names.push(name);
        }
        return names;
    }

    // Dynamic $ parameter completions
    function getParams() {
        const names = [];
        if (node._texParams && node._texParams instanceof Map) {
            for (const name of node._texParams.keys()) {
                names.push(name);
            }
        }
        // Also parse code for params not yet synced
        const code = codeWidget.value || "";
        const { params } = parseCode(code);
        for (const name of params.keys()) {
            if (!names.includes(name)) names.push(name);
        }
        return names;
    }

    // Build the editor
    try {
        const autocompleteCompartment = new CM6.Compartment();
        const completionSource = CM6.createTexCompletions(getBindings, getParams);

        const editor = new CM6.EditorView({
            state: CM6.EditorState.create({
                doc: codeWidget.value || "",
                extensions: [
                    // Core setup (line numbers, history, brackets, keymaps, etc.)
                    CM6.texSetup(),
                    // TEX language (syntax highlighting)
                    CM6.texLanguageDef,
                    // Theme
                    CM6.texEditorTheme,
                    CM6.texHighlightStyle,
                    // Autocomplete (wrapped in Compartment for live toggling)
                    autocompleteCompartment.of(
                        app.ui.settings.getSettingValue("TEX.Editor.autocomplete", true)
                            ? CM6.autocompletion({ override: [completionSource], activateOnTyping: true })
                            : []
                    ),
                    // Tooltip positioning: use absolute to avoid the
                    // position:fixed-inside-transform:scale() bug in
                    // ComfyUI Desktop's widget containers.
                    CM6.tooltips({
                        position: "absolute",
                    }),
                    // LX-6: hover docs — signature + description on hover, from the same
                    // completion data. Guarded so an older bundle (no createTexHover)
                    // still loads the editor.
                    ...(CM6.createTexHover ? [CM6.createTexHover()] : []),
                    // Lint gutter (error markers)
                    CM6.lintGutter(),
                    // Listen for doc changes → sync to widget + update sockets
                    CM6.EditorView.updateListener.of((update) => {
                        if (update.docChanged) {
                            const newCode = update.state.doc.toString();
                            codeWidget.value = newCode;
                            // Call original callback if any (for ComfyUI internals)
                            if (codeWidget.callback) {
                                codeWidget.callback(newCode);
                            }
                            updateSockets();

                            // ── Manual completion trigger (Electron compatibility) ──
                            // CM6's activateOnTyping relies on transactions being tagged
                            // as "input.type", which may not happen in Electron's Chromium
                            // renderer. We manually trigger completion when the cursor is
                            // after a word character or "@".
                            try {
                                const pos = update.state.selection.main.head;
                                if (pos > 0) {
                                    const line = update.state.doc.lineAt(pos);
                                    const col = pos - line.from;
                                    if (col > 0) {
                                        const charBefore = line.text.charAt(col - 1);
                                        if (/[@$\w]/.test(charBefore)) {
                                            // Short delay so the state settles before querying
                                            setTimeout(() => {
                                                try {
                                                    CM6.startCompletion(update.view);
                                                } catch (_) { /* view may be gone */ }
                                            }, 50);
                                        }
                                    }
                                }
                            } catch (_) { /* ignore */ }
                        }
                    }),
                    // Prevent litegraph from stealing keystrokes when editor is focused
                    CM6.EditorView.domEventHandlers({
                        keydown(e) {
                            // Let Escape bubble up to deselect the node
                            if (e.key === "Escape") return false;
                            // F1 opens TEX Reference
                            if (e.key === "F1") {
                                e.preventDefault();
                                e.stopPropagation();
                                showHelpPopup(
                                    window.innerWidth / 2,
                                    window.innerHeight * 0.1
                                );
                                return true;
                            }
                            // All other keys stay in the editor
                            e.stopPropagation();
                            return false;
                        },
                        // Stop paste/copy/cut from bubbling to LiteGraph —
                        // prevents ComfyUI from pasting nodes onto the canvas
                        // when the user pastes text into the editor (Chrome).
                        paste(e) { e.stopPropagation(); return false; },
                        copy(e) { e.stopPropagation(); return false; },
                        cut(e) { e.stopPropagation(); return false; },
                    }),
                ],
            }),
            parent: container,
        });

        // ── Right-click context menu ──
        // Custom lightweight menu with Cut / Copy / Paste / Select All.
        // Works in both Desktop (Electron) and Chrome browser.
        container.addEventListener("contextmenu", (e) => {
            e.stopPropagation();
            e.preventDefault();
            showTexContextMenu(e.clientX, e.clientY, editor);
        });
        container.addEventListener("mousedown", (e) => {
            if (e.button === 2) {
                e.stopPropagation();
            }
        });

        texEditors.set(node, editor);
        texEditorMeta.set(node, { autocompleteCompartment, completionSource });

        // Store container and editor references on node for external access
        node._texEditorContainer = container;
        node._texEditor = editor;

        return { container, editor };

    } catch (err) {
        console.error("[TEX] Failed to create CM6 editor:", err);
        return null;
    }
}

// ─── Context Menu ───────────────────────────────────────────────────

let activeContextMenu = null;
let _ctxMenuDismiss = null;
let _ctxMenuDismissKey = null;

function closeContextMenu() {
    _closeAllSubmenus();
    if (_ctxMenuDismiss) {
        document.removeEventListener("mousedown", _ctxMenuDismiss, true);
        _ctxMenuDismiss = null;
    }
    if (_ctxMenuDismissKey) {
        document.removeEventListener("keydown", _ctxMenuDismissKey, true);
        _ctxMenuDismissKey = null;
    }
    if (activeContextMenu) {
        activeContextMenu.remove();
        activeContextMenu = null;
    }
}

function showTexContextMenu(x, y, editorView) {
    closeContextMenu();

    const menu = document.createElement("div");
    menu.className = "tex-context-menu";

    const items = [
        { label: "Cut", shortcut: "Ctrl+X", action: () => document.execCommand("cut") },
        { label: "Copy", shortcut: "Ctrl+C", action: () => document.execCommand("copy") },
        { label: "Paste", shortcut: "Ctrl+V", action: async () => {
            try {
                const text = await navigator.clipboard.readText();
                editorView.dispatch(editorView.state.replaceSelection(text));
            } catch (_) {
                document.execCommand("paste");
            }
        }},
        { label: "Select All", shortcut: "Ctrl+A", action: () => {
            editorView.dispatch({
                selection: { anchor: 0, head: editorView.state.doc.length },
            });
        }},
        { label: "---" },  // separator
        { label: "TEX Help", shortcut: "F1", action: () => {
            showHelpPopup(x, y);
        }},
        { label: "TEX Doctor", action: () => { _showDoctorDialog(); } },  // C2-ux
        { label: "Snippets", shortcut: "\u25B8", cascade: true },
    ];


    for (const item of items) {
        if (item.label === "---") {
            menu.appendChild(_createSeparator());
            continue;
        }

        const row = _createMenuRow(item.label, item.shortcut);

        if (item.cascade) {
            // Snippets cascade item — opens submenu on hover
            row._childSub = null;
            let hoverTimeout = null;

            row.addEventListener("mouseenter", () => {
                clearTimeout(hoverTimeout);
                hoverTimeout = setTimeout(async () => {
                    if (row._childSub) return;

                    const builtins = await _fetchBuiltinSnippets();
                    if (row._childSub) return;  // re-check after await
                    const combined = { ...builtins, ..._loadUserSnippets() };
                    const tree = _buildSnippetTree(combined);

                    const onInsert = (content) => {
                        closeContextMenu();
                        editorView.focus();
                        editorView.dispatch(editorView.state.replaceSelection(content));
                    };

                    // Build action rows for the bottom of the submenu
                    const saveRow = _createMenuRow("Save Snippet\u2026");
                    saveRow.addEventListener("click", (e) => {
                        e.stopPropagation();
                        const sel = editorView.state.sliceDoc(
                            editorView.state.selection.main.from,
                            editorView.state.selection.main.to
                        );
                        closeContextMenu();
                        _showSaveSnippetDialog(sel || editorView.state.doc.toString());
                    });

                    const manageRow = _createMenuRow("Manage Snippets\u2026");
                    manageRow.addEventListener("click", (e) => {
                        e.stopPropagation();
                        closeContextMenu();
                        _showManageSnippetsDialog();
                    });

                    row._childSub = _createCascadeSubmenu(
                        tree, onInsert, row,
                        [_createSeparator(), saveRow, manageRow]
                    );
                }, 60);
            });
            row.addEventListener("mouseleave", () => {
                clearTimeout(hoverTimeout);
            });
        } else {
            row.addEventListener("click", (e) => {
                e.stopPropagation();
                closeContextMenu();
                editorView.focus();
                item.action();
            });
        }
        menu.appendChild(row);
    }

    // Position: ensure it stays within the viewport
    menu.style.left = x + "px";
    menu.style.top = y + "px";
    document.body.appendChild(menu);

    const rect = menu.getBoundingClientRect();
    if (rect.right > window.innerWidth) menu.style.left = (window.innerWidth - rect.width - 4) + "px";
    if (rect.bottom > window.innerHeight) menu.style.top = (window.innerHeight - rect.height - 4) + "px";

    activeContextMenu = menu;

    // Close on any click outside or Escape
    setTimeout(() => {
        _ctxMenuDismiss = (e) => {
            if (!menu.contains(e.target) && !_openSubmenus.some(s => s.contains(e.target))) closeContextMenu();
        };
        _ctxMenuDismissKey = (e) => {
            if (e.key === "Escape") closeContextMenu();
        };
        document.addEventListener("mousedown", _ctxMenuDismiss, true);
        document.addEventListener("keydown", _ctxMenuDismissKey, true);
    }, 0);
}

// ─── Help Popup Data ────────────────────────────────────────────────

const TEX_HELP_DATA = [
    {
        title: "Getting Started",
        icon: "\u{1F680}",
        entries: [
            { name: "Wire Bindings", sig: "@name", desc: "Read inputs and write outputs via named sockets. Read with @name, write with @name = expr.", example: "@OUT = @A * 0.5;" },
            { name: "Type Prefixes", sig: "f@ i@ v@ v4@ img@ m@ l@ s@", desc: "Prefix a binding to set its type: f@ float, i@ int, v@ vec3, v4@ vec4, img@ IMAGE, m@ MASK, l@ LATENT, s@ string.", example: "img@photo = ...;" },
            { name: "Parameters ($)", sig: "f$name = val;  i$name = val;  s$name = val;", desc: "Declare adjustable widgets on the node. f$ = FLOAT, i$ = INT, s$ = STRING. Use $name in expressions.", example: "f$strength = 0.75;\n@OUT = @A * $strength;" },
            { name: "Output", sig: "@OUT = expr;  @name = expr;", desc: "Assign to @OUT for a single output, or use named outputs for multiple results.", example: "@result = lerp(@base, @overlay, 0.5);\n@mask = luma(@base);" },
            { name: "Built-in Variables", sig: "ix iy u v iw ih px py fi fn ic PI TAU E", desc: "ix/iy = pixel coords. u/v = normalized [0,1]. iw/ih = image dimensions. px/py = pixel step (1/iw, 1/ih). fi = frame index. fn = frame count. ic = latent channels. PI, TAU, E constants.", example: "float cx = u - 0.5;\nfloat cy = v - 0.5;" },
        ]
    },
    {
        title: "Types & Variables",
        icon: "\u{1F4DD}",
        entries: [
            { name: "float", sig: "float x = 1.0;", desc: "Floating-point number. The default numeric type.", example: "float brightness = 0.5;" },
            { name: "int", sig: "int n = 5;", desc: "Integer number.", example: "int count = 10;" },
            { name: "vec2", sig: "vec2 p = vec2(x, y);", desc: "2-component vector. Used for 2D coordinates and UV pairs.", example: "vec2 uv = vec2(u, v);" },
            { name: "vec3", sig: "vec3 c = vec3(r, g, b);", desc: "3-component vector. Used for RGB colors and 3D coordinates. Standard image type in ComfyUI.", example: "vec3 color = vec3(1.0, 0.0, 0.0);" },
            { name: "vec4", sig: "vec4 c = vec4(r, g, b, a);", desc: "4-component vector. Used for RGBA colors.", example: "vec4 pixel = vec4(1.0, 1.0, 1.0, 0.5);" },
            { name: "mat3", sig: "mat3 m = mat3(1.0);", desc: "3x3 matrix for internal computation (cannot assign to @OUT).", example: "mat3 identity = mat3(1.0);" },
            { name: "mat4", sig: "mat4 m = mat4(1.0);", desc: "4x4 matrix for internal computation (cannot assign to @OUT).", example: "mat4 transform = mat4(1.0);" },
            { name: "string", sig: "string s = \"hello\";", desc: "Text string. Concat with +. String output auto-detected when @OUT is a string.", example: "string greeting = \"hello\" + \" world\";" },
            { name: "Pixel Coords", sig: "ix, iy", desc: "Current pixel coordinates (integers).", example: "float val = ix + iy;" },
            { name: "Normalized Coords", sig: "u, v", desc: "Normalized coordinates in [0, 1] range.", example: "float gradient = u;" },
            { name: "Dimensions", sig: "iw, ih", desc: "Image width and height in pixels.", example: "float aspect = iw / ih;" },
            { name: "Pixel Step", sig: "px, py", desc: "Pixel step in UV space: px = 1/iw, py = 1/ih. Use for neighbor sampling.", example: "vec4 right = sample(@A, u + px, v);" },
            { name: "Constants", sig: "PI, TAU, E", desc: "Mathematical constants: PI = 3.14159..., TAU = 2*PI = 6.28318..., E = 2.71828...", example: "float circle = sin(u * TAU);" },
        ]
    },
    {
        title: "Math",
        icon: "\u{1F4D0}",
        entries: [
            { name: "sin", sig: "sin(x) \u2192 float", desc: "Sine (radians).", example: "float s = sin(u * PI * 2.0);" },
            { name: "cos", sig: "cos(x) \u2192 float", desc: "Cosine (radians).", example: "float c = cos(v * PI);" },
            { name: "tan", sig: "tan(x) \u2192 float", desc: "Tangent (radians).", example: "float t = tan(u);" },
            { name: "asin", sig: "asin(x) \u2192 float", desc: "Arcsine. Returns radians.", example: "float angle = asin(0.5);" },
            { name: "acos", sig: "acos(x) \u2192 float", desc: "Arccosine. Returns radians.", example: "float angle = acos(0.5);" },
            { name: "atan", sig: "atan(x) \u2192 float", desc: "Arctangent. Returns radians.", example: "float angle = atan(1.0);" },
            { name: "atan2", sig: "atan2(y, x) \u2192 float", desc: "Two-argument arctangent. Returns radians.", example: "float angle = atan2(v - 0.5, u - 0.5);" },
            { name: "sincos", sig: "sincos(x) \u2192 vec2", desc: "Returns vec2(sin(x), cos(x)). More efficient than separate sin/cos calls.", example: "vec2 sc = sincos(angle);\nfloat s = sc.x;\nfloat c = sc.y;" },
            { name: "sinh", sig: "sinh(x) \u2192 float", desc: "Hyperbolic sine.", example: "float s = sinh(u);" },
            { name: "cosh", sig: "cosh(x) \u2192 float", desc: "Hyperbolic cosine.", example: "float c = cosh(u);" },
            { name: "tanh", sig: "tanh(x) \u2192 float", desc: "Hyperbolic tangent.", example: "float t = tanh(u * 2.0);" },
            { name: "pow", sig: "pow(x, y) \u2192 float", desc: "Raise x to the power y.", example: "float p = pow(u, 2.2);" },
            { name: "sqrt", sig: "sqrt(x) \u2192 float", desc: "Square root.", example: "float s = sqrt(u * u + v * v);" },
            { name: "exp", sig: "exp(x) \u2192 float", desc: "e raised to the power x.", example: "float e = exp(-u * 5.0);" },
            { name: "log", sig: "log(x) \u2192 float", desc: "Natural logarithm (base e).", example: "float l = log(u + 1.0);" },
            { name: "log2", sig: "log2(x) \u2192 float", desc: "Logarithm base 2.", example: "float l = log2(256.0);" },
            { name: "log10", sig: "log10(x) \u2192 float", desc: "Logarithm base 10.", example: "float l = log10(1000.0);" },
            { name: "abs", sig: "abs(x) \u2192 float", desc: "Absolute value.", example: "float a = abs(u - 0.5);" },
            { name: "sign", sig: "sign(x) \u2192 float", desc: "Returns -1, 0, or 1.", example: "float s = sign(u - 0.5);" },
            { name: "pow2", sig: "pow2(x) \u2192 float", desc: "2 raised to the power x.", example: "float p = pow2(8.0);" },
            { name: "pow10", sig: "pow10(x) \u2192 float", desc: "10 raised to the power x.", example: "float p = pow10(3.0);" },
            { name: "hypot", sig: "hypot(x, y) \u2192 float", desc: "Hypotenuse: sqrt(x*x + y*y).", example: "float d = hypot(u - 0.5, v - 0.5);" },
            { name: "floor", sig: "floor(x) \u2192 float", desc: "Round down to nearest integer.", example: "float f = floor(u * 10.0);" },
            { name: "ceil", sig: "ceil(x) \u2192 float", desc: "Round up to nearest integer.", example: "float c = ceil(u * 10.0);" },
            { name: "round", sig: "round(x) \u2192 float", desc: "Round to nearest integer.", example: "float r = round(u * 10.0) / 10.0;" },
            { name: "trunc", sig: "trunc(x) \u2192 float", desc: "Truncate toward zero (drop fractional part).", example: "float t = trunc(u * 10.0);" },
            { name: "fract", sig: "fract(x) \u2192 float", desc: "Fractional part: x - floor(x).", example: "float f = fract(u * 5.0);" },
            { name: "mod", sig: "mod(x, y) \u2192 float", desc: "Modulo (remainder).", example: "float m = mod(u * 10.0, 1.0);" },
            { name: "degrees", sig: "degrees(x) \u2192 float", desc: "Convert radians to degrees.", example: "float d = degrees(PI);" },
            { name: "radians", sig: "radians(x) \u2192 float", desc: "Convert degrees to radians.", example: "float r = radians(180.0);" },
            { name: "spow", sig: "spow(x, y) \u2192 float", desc: "Sign-preserving power. Safe for negative x.", example: "float s = spow(u - 0.5, 2.0);" },
            { name: "sdiv", sig: "sdiv(a, b) \u2192 float", desc: "Safe divide. Returns 0 when b is zero.", example: "float d = sdiv(1.0, u);" },
            { name: "isnan", sig: "isnan(x) \u2192 float", desc: "Returns 1.0 if x is NaN, 0.0 otherwise.", example: "float check = isnan(x);" },
            { name: "isinf", sig: "isinf(x) \u2192 float", desc: "Returns 1.0 if x is infinite, 0.0 otherwise.", example: "float check = isinf(x);" },
        ]
    },
    {
        title: "Interpolation",
        icon: "\u{1F39A}\uFE0F",
        entries: [
            { name: "min", sig: "min(a, b) \u2192 float", desc: "Returns the smaller value.", example: "float m = min(u, 0.5);" },
            { name: "max", sig: "max(a, b) \u2192 float", desc: "Returns the larger value.", example: "float m = max(u, 0.0);" },
            { name: "clamp", sig: "clamp(x, lo, hi) \u2192 float", desc: "Clamp x to [lo, hi] range.", example: "float c = clamp(u * 2.0, 0.0, 1.0);" },
            { name: "lerp", sig: "lerp(a, b, t) \u2192 float", desc: "Linear interpolation from a to b by t.", example: "@OUT = lerp(@A, @B, 0.5);" },
            { name: "mix", sig: "mix(a, b, t) \u2192 float", desc: "Alias for lerp. Linear interpolation.", example: "@OUT = mix(@A, @B, $blend);" },
            { name: "fit", sig: "fit(x, inLo, inHi, outLo, outHi) \u2192 float", desc: "Remap x from [inLo, inHi] to [outLo, outHi].", example: "float y = fit(u, 0.2, 0.8, 0.0, 1.0);" },
            { name: "step", sig: "step(edge, x) \u2192 float", desc: "Returns 0 if x < edge, 1 otherwise.", example: "float s = step(0.5, u);" },
            { name: "smoothstep", sig: "smoothstep(lo, hi, x) \u2192 float", desc: "Smooth Hermite interpolation between lo and hi.", example: "float s = smoothstep(0.3, 0.7, u);" },
        ]
    },
    {
        title: "Vector",
        icon: "\u{1F4CF}",
        entries: [
            { name: "dot", sig: "dot(a, b) \u2192 float", desc: "Dot product of two vectors.", example: "float d = dot(normal, lightDir);" },
            { name: "length", sig: "length(v) \u2192 float", desc: "Length (magnitude) of a vector.", example: "float len = length(vec3(u, v, 0.0));" },
            { name: "distance", sig: "distance(a, b) \u2192 float", desc: "Distance between two points.", example: "float d = distance(vec3(u,v,0), vec3(0.5,0.5,0));" },
            { name: "normalize", sig: "normalize(v) \u2192 vec", desc: "Unit vector in the same direction.", example: "vec3 dir = normalize(vec3(u-0.5, v-0.5, 1.0));" },
            { name: "cross", sig: "cross(a, b) \u2192 vec3", desc: "Cross product of two vec3 vectors.", example: "vec3 n = cross(tangent, bitangent);" },
            { name: "reflect", sig: "reflect(v, n) \u2192 vec", desc: "Reflect vector v around normal n.", example: "vec3 r = reflect(incoming, normal);" },
        ]
    },
    {
        title: "Color",
        icon: "\u{1F3A8}",
        entries: [
            { name: "luma", sig: "luma(rgb) \u2192 float", desc: "Perceptual luminance of an RGB color.", example: "float gray = luma(@image);" },
            { name: "hsv2rgb", sig: "hsv2rgb(hsv) \u2192 vec3", desc: "Convert HSV color to RGB.", example: "vec3 rgb = hsv2rgb(vec3(u, 1.0, 1.0));" },
            { name: "rgb2hsv", sig: "rgb2hsv(rgb) \u2192 vec3", desc: "Convert RGB color to HSV.", example: "vec3 hsv = rgb2hsv(@image);" },
            { name: "srgb_to_linear", sig: "srgb_to_linear(c) \u2192 vec", desc: "Gamma-encoded sRGB \u2192 linear-light. Blur/blend in linear to avoid halos.", example: "vec3 lin = srgb_to_linear(@image.rgb);" },
            { name: "linear_to_srgb", sig: "linear_to_srgb(c) \u2192 vec", desc: "Linear-light \u2192 gamma-encoded sRGB (inverse of srgb_to_linear).", example: "@OUT = vec4(linear_to_srgb(lin), 1.0);" },
            { name: "oklab_from_rgb", sig: "oklab_from_rgb(c) \u2192 vec3", desc: "Linear RGB \u2192 OKLab. Mix/interpolate in OKLab for perceptually-even gradients.", example: "vec3 lab = oklab_from_rgb(srgb_to_linear(@image.rgb));" },
            { name: "oklab_to_rgb", sig: "oklab_to_rgb(lab) \u2192 vec3", desc: "OKLab \u2192 linear RGB (inverse of oklab_from_rgb).", example: "vec3 rgb = oklab_to_rgb(lab);" },
            { name: "premultiply", sig: "premultiply(rgba) \u2192 vec4", desc: "Straight \u2192 premultiplied alpha (rgb *= a).", example: "vec4 p = premultiply(@image);" },
            { name: "unpremultiply", sig: "unpremultiply(rgba) \u2192 vec4", desc: "Premultiplied \u2192 straight alpha (rgb /= a).", example: "vec4 s = unpremultiply(p);" },
            { name: "over", sig: "over(fg, bg) \u2192 vec4", desc: "Porter-Duff 'over': composite fg atop bg (straight-alpha RGBA).", example: "@OUT = over(@A, @B);" },
            { name: "under", sig: "under(fg, bg) \u2192 vec4", desc: "Composite fg under bg (= over(bg, fg)).", example: "@OUT = under(@A, @B);" },
            { name: "atop", sig: "atop(fg, bg) \u2192 vec4", desc: "'atop': fg confined to bg's coverage.", example: "@OUT = atop(@A, @B);" },
            { name: "screen", sig: "screen(a, b) \u2192 vec", desc: "Screen blend: 1 - (1-a)(1-b). Brightens.", example: "@OUT = vec4(screen(@A.rgb, @B.rgb), 1.0);" },
            { name: "overlay", sig: "overlay(a, b) \u2192 vec", desc: "Overlay blend (multiply/screen by base).", example: "@OUT = vec4(overlay(@A.rgb, @B.rgb), 1.0);" },
            { name: "hard_light", sig: "hard_light(a, b) \u2192 vec", desc: "Hard-light blend (overlay with operands swapped).", example: "@OUT = vec4(hard_light(@A.rgb, @B.rgb), 1.0);" },
            { name: "soft_light", sig: "soft_light(a, b) \u2192 vec", desc: "Soft-light blend (Pegtop, smooth).", example: "@OUT = vec4(soft_light(@A.rgb, @B.rgb), 1.0);" },
            { name: "color_dodge", sig: "color_dodge(a, b) \u2192 vec", desc: "Color-dodge: brightens base by blend.", example: "@OUT = vec4(color_dodge(@A.rgb, @B.rgb), 1.0);" },
            { name: "color_burn", sig: "color_burn(a, b) \u2192 vec", desc: "Color-burn: darkens base by blend.", example: "@OUT = vec4(color_burn(@A.rgb, @B.rgb), 1.0);" },
            { name: "linear_light", sig: "linear_light(a, b) \u2192 vec", desc: "Linear-light blend: clamp(a + 2b - 1).", example: "@OUT = vec4(linear_light(@A.rgb, @B.rgb), 1.0);" },
            { name: "vivid_light", sig: "vivid_light(a, b) \u2192 vec", desc: "Vivid-light blend (burn/dodge by blend).", example: "@OUT = vec4(vivid_light(@A.rgb, @B.rgb), 1.0);" },
        ]
    },
    {
        title: "Sampling",
        icon: "\u{1F5BC}\uFE0F",
        entries: [
            { name: "sample", sig: "sample(img, u, v) \u2192 vec", desc: "Bilinear sample at normalized UV coordinates.", example: "@OUT = sample(@A, u + 0.01, v);" },
            { name: "fetch", sig: "fetch(img, px, py) \u2192 vec", desc: "Nearest-neighbor fetch at pixel coordinates.", example: "@OUT = fetch(@A, ix, iy);" },
            { name: "sample_cubic", sig: "sample_cubic(img, u, v) \u2192 vec", desc: "Bicubic (Catmull-Rom) sampling.", example: "@OUT = sample_cubic(@A, u, v);" },
            { name: "sample_lanczos", sig: "sample_lanczos(img, u, v) \u2192 vec", desc: "Lanczos-3 high-quality sampling.", example: "@OUT = sample_lanczos(@A, u * 0.5, v * 0.5);" },
            { name: "sample_mip", sig: "sample_mip(img, u, v, lod) \u2192 vec", desc: "Mipmap sampling with LOD. 0 = full res, 1 = half, etc. Trilinear between levels.", example: "@OUT = sample_mip(@A, u, v, 2.5);" },
            { name: "sample_mip_gauss", sig: "sample_mip_gauss(img, u, v, lod) \u2192 vec", desc: "Gaussian-prefiltered mipmap sampling. Smoother pyramid (sigma=1.13) gives ~5 dB better exponential blur accuracy vs sample_mip.", example: "@OUT = sample_mip_gauss(@A, u, v, 2.5);" },
            { name: "gauss_blur", sig: "gauss_blur(img, sigma) \u2192 vec", desc: "Separable Gaussian blur. Kernel radius \u2248 3\u00d7sigma pixels. Replicate border padding.", example: "@OUT = gauss_blur(@A, 2.0);" },
            { name: "bilateral_filter", sig: "bilateral_filter(img, spatial_sigma, range_sigma) \u2192 vec", desc: "Edge-preserving smoothing: blurs within regions but keeps edges. Window capped at 7\u00d77.", example: "@OUT = bilateral_filter(@A, 1.5, 0.2);" },
            { name: "erode", sig: "erode(img, radius) \u2192 vec", desc: "Morphological erosion (local min over a (2r+1)\u00b2 square). Shrinks bright regions.", example: "@OUT = erode(@mask, 3);" },
            { name: "dilate", sig: "dilate(img, radius) \u2192 vec", desc: "Morphological dilation (local max). Grows bright regions.", example: "@OUT = dilate(@mask, 3);" },
            { name: "fetch_frame", sig: "fetch_frame(img, frame, px, py) \u2192 vec", desc: "Nearest-neighbor fetch from a specific batch frame.", example: "@OUT = fetch_frame(@A, 0, ix, iy);" },
            { name: "sample_frame", sig: "sample_frame(img, frame, u, v) \u2192 vec", desc: "Bilinear sample from a specific batch frame.", example: "@OUT = sample_frame(@A, fi-1, u, v);" },
            { name: "sample_grad", sig: "sample_grad(img, u, v) \u2192 vec2", desc: "Image gradient (Sobel) at UV. Returns vec2(dI/dx, dI/dy) of luminance.", example: "vec2 grad = sample_grad(@A, u, v);" },
        ]
    },
    {
        title: "Noise",
        icon: "\u{1F30A}",
        entries: [
            { name: "perlin", sig: "perlin(x, y) \u2192 float", desc: "2D Perlin noise. Returns value in [-1, 1].", example: "float n = perlin(u * 10.0, v * 10.0);" },
            { name: "simplex", sig: "simplex(x, y) \u2192 float", desc: "2D Simplex noise. Returns value in [-1, 1].", example: "float n = simplex(u * 8.0, v * 8.0);" },
            { name: "fbm", sig: "fbm(x, y, octaves) \u2192 float", desc: "Fractal Brownian Motion (multi-octave Perlin).", example: "float n = fbm(u * 4.0, v * 4.0, 6);" },
            { name: "worley_f1", sig: "worley_f1(x, y) \u2192 float", desc: "Worley (cellular) noise — distance to nearest cell center.", example: "float n = worley_f1(u * 5.0, v * 5.0);" },
            { name: "worley_f2", sig: "worley_f2(x, y) \u2192 float", desc: "Worley noise — distance to second-nearest cell center.", example: "float n = worley_f2(u * 5.0, v * 5.0);" },
            { name: "voronoi", sig: "voronoi(x, y) \u2192 float", desc: "Voronoi cell ID noise. Returns a unique value per cell.", example: "float cell = voronoi(u * 8.0, v * 8.0);" },
            { name: "billow", sig: "billow(x, y, octaves) \u2192 float", desc: "Billowy noise — abs(fbm). Puffy cloud shapes.", example: "float n = billow(u * 4.0, v * 4.0, 6);" },
            { name: "turbulence", sig: "turbulence(x, y, octaves) \u2192 float", desc: "Turbulence — sum of abs(noise) per octave. Veiny patterns.", example: "float n = turbulence(u * 4.0, v * 4.0, 6);" },
            { name: "ridged", sig: "ridged(x, y, octaves) \u2192 float", desc: "Ridged multifractal — sharp ridges, good for mountains.", example: "float n = ridged(u * 4.0, v * 4.0, 6);" },
            { name: "flow", sig: "flow(x, y, angle) \u2192 float", desc: "Flow noise — Perlin rotated by angle per octave. Avoids static patterns.", example: "float n = flow(u * 6.0, v * 6.0, fi * 0.1);" },
            { name: "curl", sig: "curl(x, y) \u2192 vec2", desc: "Curl of 2D noise field. Returns a divergence-free vector.", example: "vec2 c = curl(u * 5.0, v * 5.0);" },
            { name: "alligator", sig: "alligator(x, y) \u2192 float", desc: "Alligator noise — cellular crack patterns.", example: "float n = alligator(u * 5.0, v * 5.0);" },
        ]
    },
    {
        title: "SDF & Smooth",
        icon: "\u{1F7E2}",
        entries: [
            { name: "sdf_circle", sig: "sdf_circle(x, y, cx, cy, r) \u2192 float", desc: "Signed distance to circle. Negative inside, positive outside.", example: "float d = sdf_circle(u, v, 0.5, 0.5, 0.3);" },
            { name: "sdf_box", sig: "sdf_box(x, y, cx, cy, hw, hh) \u2192 float", desc: "Signed distance to axis-aligned box.", example: "float d = sdf_box(u, v, 0.5, 0.5, 0.2, 0.15);" },
            { name: "sdf_line", sig: "sdf_line(x, y, x1, y1, x2, y2) \u2192 float", desc: "Distance to line segment.", example: "float d = sdf_line(u, v, 0.2, 0.2, 0.8, 0.8);" },
            { name: "sdf_polygon", sig: "sdf_polygon(x, y, cx, cy, r, n) \u2192 float", desc: "Signed distance to regular polygon with n sides.", example: "float d = sdf_polygon(u, v, 0.5, 0.5, 0.3, 6);" },
            { name: "smin", sig: "smin(a, b, k) \u2192 float|vec", desc: "Smooth minimum. Polynomial blending with radius k. Works on scalars and vectors.", example: "float d = smin(d1, d2, 0.1);" },
            { name: "smax", sig: "smax(a, b, k) \u2192 float|vec", desc: "Smooth maximum. Polynomial blending with radius k. Works on scalars and vectors.", example: "float d = smax(d1, d2, 0.1);" },
        ]
    },
    {
        title: "Image Stats",
        icon: "\u{1F4CA}",
        entries: [
            { name: "img_min", sig: "img_min(img) \u2192 vec", desc: "Per-channel minimum across the entire image.", example: "vec3 lo = img_min(@A);" },
            { name: "img_max", sig: "img_max(img) \u2192 vec", desc: "Per-channel maximum across the entire image.", example: "vec3 hi = img_max(@A);" },
            { name: "img_mean", sig: "img_mean(img) \u2192 vec", desc: "Per-channel mean (average) of the image.", example: "vec3 avg = img_mean(@A);" },
            { name: "img_sum", sig: "img_sum(img) \u2192 vec", desc: "Per-channel sum of all pixel values.", example: "vec3 total = img_sum(@A);" },
            { name: "img_median", sig: "img_median(img) \u2192 vec", desc: "Per-channel median of the image.", example: "vec3 mid = img_median(@A);" },
            { name: "Auto-levels Example", sig: "img_min, img_max", desc: "Normalize an image to full [0,1] range using stats.", example: "@OUT = (@A - img_min(@A)) / max(img_max(@A) - img_min(@A), 0.001);" },
        ]
    },
    {
        title: "Strings",
        icon: "\u{1F524}",
        entries: [
            { name: "str", sig: "str(x) \u2192 string", desc: "Convert a number to a string.", example: "string s = str(42);" },
            { name: "len (string)", sig: "len(s) \u2192 float", desc: "Length of a string.", example: "float n = len(\"hello\");" },
            { name: "replace", sig: "replace(s, old, new) \u2192 string", desc: "Replace all occurrences of old with new.", example: "string r = replace(s, \"foo\", \"bar\");" },
            { name: "strip", sig: "strip(s) \u2192 string", desc: "Remove leading/trailing whitespace.", example: "string clean = strip(s);" },
            { name: "lower", sig: "lower(s) \u2192 string", desc: "Convert to lowercase.", example: "string lc = lower(\"Hello\");" },
            { name: "upper", sig: "upper(s) \u2192 string", desc: "Convert to uppercase.", example: "string uc = upper(\"hello\");" },
            { name: "contains", sig: "contains(s, sub) \u2192 float", desc: "Returns 1.0 if s contains sub, 0.0 otherwise.", example: "float has = contains(s, \"test\");" },
            { name: "startswith", sig: "startswith(s, prefix) \u2192 float", desc: "Returns 1.0 if s starts with prefix.", example: "float sw = startswith(s, \"img_\");" },
            { name: "endswith", sig: "endswith(s, suffix) \u2192 float", desc: "Returns 1.0 if s ends with suffix.", example: "float ew = endswith(s, \".png\");" },
            { name: "find", sig: "find(s, sub) \u2192 float", desc: "Index of first occurrence, or -1.0 if not found.", example: "float idx = find(s, \"world\");" },
            { name: "substr", sig: "substr(s, start, len?) \u2192 string", desc: "Extract a substring. len is optional.", example: "string sub = substr(s, 0, 5);" },
            { name: "len", sig: "len(x) \u2192 float", desc: "Length of a string, array, or vec-array (element count).", example: "float n = len(\"hello\");" },
            { name: "repeat", sig: "repeat(s, n) \u2192 string", desc: "Repeat a string N times.", example: "string bar = repeat(\"=\", 10);" },
            { name: "str_reverse", sig: "str_reverse(s) \u2192 string", desc: "Reverse a string.", example: "string r = str_reverse(\"abc\");" },
            { name: "matches", sig: "matches(s, pattern) \u2192 float", desc: "Returns 1.0 if the whole string matches the regex pattern, else 0.0.", example: "float ok = matches(s, \"[0-9]+\");" },
            { name: "hash", sig: "hash(s) \u2192 string", desc: "Deterministic string hash (SHA-256 hex, first 16 chars).", example: "string h = hash(\"seed\");" },
            { name: "hash_float", sig: "hash_float(s) \u2192 float", desc: "Deterministic hash of a string to a float in [0, 1).", example: "float r = hash_float(\"seed\");" },
            { name: "hash_int", sig: "hash_int(s, max?) \u2192 int", desc: "Deterministic hash of a string to a non-negative int (optional exclusive max).", example: "int i = hash_int(\"seed\", 100);" },
            { name: "to_int", sig: "to_int(s) \u2192 int", desc: "Parse a string as an integer.", example: "int n = to_int(\"42\");" },
            { name: "to_float", sig: "to_float(s) \u2192 float", desc: "Parse a string as a float.", example: "float f = to_float(\"3.14\");" },
            { name: "sanitize_filename", sig: "sanitize_filename(s) \u2192 string", desc: "Remove unsafe characters for use in file paths.", example: "string safe = sanitize_filename(s);" },
            { name: "format", sig: "format(fmt, ...) \u2192 string", desc: "Printf-style formatting. %d = int, %f = float, %s = string.", example: "string s = format(\"Frame %d of %d\", fi, fn);" },
            { name: "split", sig: "split(s, sep) \u2192 string[]", desc: "Split string into array by separator.", example: "string parts[4] = split(s, \",\");" },
            { name: "lstrip", sig: "lstrip(s) \u2192 string", desc: "Remove leading whitespace.", example: "string clean = lstrip(s);" },
            { name: "rstrip", sig: "rstrip(s) \u2192 string", desc: "Remove trailing whitespace.", example: "string clean = rstrip(s);" },
            { name: "pad_left", sig: "pad_left(s, width, fill) \u2192 string", desc: "Pad string on the left to reach width.", example: "string n = pad_left(str(fi), 4, \"0\");" },
            { name: "pad_right", sig: "pad_right(s, width, fill) \u2192 string", desc: "Pad string on the right to reach width.", example: "string n = pad_right(s, 20, \" \");" },
            { name: "count", sig: "count(s, sub) \u2192 float", desc: "Count non-overlapping occurrences of sub in s.", example: "float n = count(s, \"the\");" },
            { name: "char_at", sig: "char_at(s, idx) \u2192 string", desc: "Character at index (0-based).", example: "string c = char_at(s, 0);" },
            { name: "String Concat", sig: "\"a\" + \"b\"", desc: "Concatenate strings with the + operator.", example: "string full = \"frame_\" + str(fi) + \".png\";" },
        ]
    },
    {
        title: "Arrays",
        icon: "\u{1F4E6}",
        entries: [
            { name: "Declaration", sig: "type name[size];", desc: "Declare an array. Supports float, int, vec3, vec4, string. Max size: 1024.", example: "float arr[5];\narr[0] = 1.0;" },
            { name: "sort", sig: "sort(arr) \u2192 array", desc: "Sort array elements in ascending order.", example: "sort(arr);" },
            { name: "reverse", sig: "reverse(arr) \u2192 array", desc: "Reverse array element order.", example: "reverse(arr);" },
            { name: "arr_sum", sig: "arr_sum(arr) \u2192 float", desc: "Sum of all array elements.", example: "float total = arr_sum(arr);" },
            { name: "arr_min", sig: "arr_min(arr) \u2192 float", desc: "Minimum value in array.", example: "float lo = arr_min(arr);" },
            { name: "arr_max", sig: "arr_max(arr) \u2192 float", desc: "Maximum value in array.", example: "float hi = arr_max(arr);" },
            { name: "median", sig: "median(arr) \u2192 float", desc: "Median value of array.", example: "float mid = median(arr);" },
            { name: "arr_avg", sig: "arr_avg(arr) \u2192 float", desc: "Average of all array elements.", example: "float avg = arr_avg(arr);" },
            { name: "join", sig: "join(arr, sep) \u2192 string", desc: "Concatenate string array with separator.", example: "string csv = join(names, \", \");" },
            { name: "len (array)", sig: "len(arr) \u2192 int", desc: "Number of elements in array.", example: "int n = len(arr);" },
        ]
    },
    {
        title: "Matrix",
        icon: "\u{1F9EE}",
        entries: [
            { name: "mat3 constructor", sig: "mat3(s)  mat3(a,b,...,i)", desc: "Create 3x3 matrix. Single value = scaled identity. 9 values = row-major.", example: "mat3 m = mat3(1.0);" },
            { name: "mat4 constructor", sig: "mat4(s)  mat4(a,b,...,p)", desc: "Create 4x4 matrix. Single value = scaled identity. 16 values = row-major.", example: "mat4 m = mat4(1.0);" },
            { name: "Matrix * Vector", sig: "mat * vec \u2192 vec", desc: "Matrix-vector multiplication.", example: "vec3 result = m * myVec;" },
            { name: "Matrix * Matrix", sig: "mat * mat \u2192 mat", desc: "Matrix-matrix multiplication.", example: "mat3 combined = a * b;" },
            { name: "Scalar * Matrix", sig: "scalar * mat \u2192 mat", desc: "Element-wise scale.", example: "mat3 scaled = 2.0 * m;" },
            { name: "Matrix +/-", sig: "mat + mat  mat - mat", desc: "Element-wise addition and subtraction.", example: "mat3 sum = a + b;" },
            { name: "transpose", sig: "transpose(m) \u2192 mat", desc: "Transpose a matrix.", example: "mat3 mt = transpose(m);" },
            { name: "determinant", sig: "determinant(m) \u2192 float", desc: "Compute the determinant.", example: "float det = determinant(m);" },
            { name: "inverse", sig: "inverse(m) \u2192 mat", desc: "Compute the matrix inverse.", example: "mat3 inv = inverse(m);" },
        ]
    },
    {
        title: "Control Flow",
        icon: "\u{1F501}",
        entries: [
            { name: "if / else", sig: "if (cond) { ... } else { ... }", desc: "Conditional branching. Vectorized via torch.where for images.", example: "if (u > 0.5) {\n  @OUT = @A;\n} else {\n  @OUT = @B;\n}" },
            { name: "for loop", sig: "for (int i = 0; i < n; i++) { ... }", desc: "Bounded loop. The loop count must be deterministic.", example: "float sum = 0.0;\nfor (int i = 0; i < 10; i++) {\n  sum += i;\n}" },
            { name: "while loop", sig: "while (cond) { ... }", desc: "Loop while condition is true.", example: "int i = 0;\nwhile (i < 10) {\n  i++;\n}" },
            { name: "break", sig: "break;", desc: "Exit the current loop early.", example: "for (int i=0; i<100; i++) {\n  if (arr[i] < 0) break;\n}" },
            { name: "continue", sig: "continue;", desc: "Skip to the next loop iteration.", example: "for (int i=0; i<10; i++) {\n  if (i == 5) continue;\n}" },
            { name: "User Functions", sig: "type name(args) { return expr; }", desc: "Define reusable functions. Must be declared before use.", example: "float remap(float x, float lo, float hi) {\n  return (x - lo) / (hi - lo);\n}" },
            { name: "const", sig: "const type name = value;", desc: "Compile-time constant. Must be initialized with a literal.", example: "const float GOLDEN = 1.618034;" },
        ]
    },
    {
        title: "Batch / Temporal",
        icon: "\u{1F3AC}",
        entries: [
            { name: "fi", sig: "fi \u2192 int", desc: "Current frame index (0 to batch_size - 1).", example: "float t = fi / max(fn - 1, 1);" },
            { name: "fn", sig: "fn \u2192 int", desc: "Total number of frames in the batch.", example: "float progress = fi / max(fn - 1, 1);" },
            { name: "fetch_frame", sig: "fetch_frame(img, frame, px, py) \u2192 vec", desc: "Nearest-neighbor fetch from a specific batch frame.", example: "@OUT = fetch_frame(@A, fi-1, ix, iy);" },
            { name: "sample_frame", sig: "sample_frame(img, frame, u, v) \u2192 vec", desc: "Bilinear sample from a specific batch frame.", example: "@OUT = sample_frame(@A, 0, u, v);" },
            { name: "Fade Example", sig: "fi, fn", desc: "Fade an image in over the batch.", example: "@OUT = @A * (fi / max(fn-1, 1));" },
            { name: "Temporal Blend", sig: "fetch_frame + lerp", desc: "Blend adjacent frames together.", example: "@OUT = lerp(\n  fetch_frame(@A, fi-1, ix, iy),\n  fetch_frame(@A, fi+1, ix, iy),\n  0.5);" },
        ]
    },
    {
        title: "Operators",
        icon: "\u{1F527}",
        entries: [
            { name: "Arithmetic", sig: "+ - * / %", desc: "Basic arithmetic: add, subtract, multiply, divide, modulo.", example: "float x = (u + v) * 0.5;" },
            { name: "Comparison", sig: "== != < > <= >=", desc: "Comparison operators. Return 1.0 (true) or 0.0 (false) for images.", example: "float mask = (u > 0.5) ? 1.0 : 0.0;" },
            { name: "Logical", sig: "&& || !", desc: "Logical AND, OR, NOT.", example: "if (u > 0.2 && v < 0.8) { ... }" },
            { name: "Ternary", sig: "cond ? a : b", desc: "Inline conditional expression.", example: "float x = (u > 0.5) ? 1.0 : 0.0;" },
            { name: "Compound", sig: "+= -= *= /= ++ --", desc: "Compound assignment and increment/decrement.", example: "float x = 0.0;\nx += 1.0;\nx++;" },
            { name: "Channel Access", sig: ".r .g .b .a  /  .x .y .z .w", desc: "Access individual channels of a vector.", example: "float red = @image.r;\nfloat alpha = @image.a;" },
            { name: "Swizzle", sig: ".rgb .bgra .xy etc.", desc: "Rearrange/extract multiple channels at once.", example: "vec3 bgr = @image.bgr;\nvec3 gray3 = @image.rrr;" },
        ]
    },
    {
        title: "Latent",
        icon: "\u{1F4E1}",
        entries: [
            { name: "Latent Support", sig: "l@ prefix / auto-detect", desc: "Connect latent data to any input. Output auto-detects as LATENT when inputs are latent. Values are NOT clamped \u2014 latent range is typically [-4, 4].", example: "@OUT = @A * 0.5;" },
            { name: "ic", sig: "ic \u2192 int", desc: "Number of latent channels (4 for SD1.5/SDXL, 16 for SD3).", example: "// ic tells you the latent space width" },
        ]
    },
    {
        title: "Debugging",
        icon: "\u{1F41E}",
        entries: [
            { name: "debug_print", sig: "debug_print(label, value[, x, y]) \u2192 value", desc: "Probe a value at pixel (x, y): records it for the node's tier/timing HUD and returns the value unchanged. Interpreter-only \u2014 a compiled tier falls back so the probe always fires.", example: "float g = debug_print(\"luma\", luma(@A.rgb), 0, 0);" },
        ]
    },
];

// ─── Help Popup Show/Hide ────────────────────────────────────────────

let activeHelpPopup = null;
let outsideClickHandler = null;

function buildHelpDOM() {
    const container = document.createElement("div");

    // Header area
    const header = document.createElement("div");
    header.className = "tex-help-header";
    header.innerHTML = `
        <div class="tex-help-title">TEX Reference</div>
        <div class="tex-help-links">
            <a href="https://github.com/xavinitram/TEX/wiki" target="_blank">Wiki</a>
            <span class="tex-help-link-sep">\u00b7</span>
            <a href="https://github.com/xavinitram/TEX/issues" target="_blank">Report a Bug</a>
        </div>
    `;
    container.appendChild(header);

    // Search bar
    const searchWrap = document.createElement("div");
    searchWrap.className = "tex-help-search-wrap";
    const searchInput = document.createElement("input");
    searchInput.type = "text";
    searchInput.className = "tex-help-search";
    searchInput.placeholder = "Search functions, types, concepts...";
    searchInput.spellcheck = false;
    searchWrap.appendChild(searchInput);
    container.appendChild(searchWrap);

    // Build categories
    const categoriesContainer = document.createElement("div");
    categoriesContainer.className = "tex-help-categories";

    const categoryEls = [];

    for (let ci = 0; ci < TEX_HELP_DATA.length; ci++) {
        const cat = TEX_HELP_DATA[ci];
        const catEl = document.createElement("div");
        catEl.className = "tex-help-category";

        // Category header
        const catHeader = document.createElement("div");
        catHeader.className = "tex-help-cat-header";
        const expanded = ci === 0; // First category starts expanded
        catHeader.innerHTML = `<span class="tex-help-cat-arrow">${expanded ? "\u25BE" : "\u25B8"}</span><span class="tex-help-cat-icon">${cat.icon}</span> ${cat.title}`;
        catHeader.dataset.expanded = expanded ? "1" : "0";
        catEl.appendChild(catHeader);

        // Entries container
        const entriesEl = document.createElement("div");
        entriesEl.className = "tex-help-entries";
        entriesEl.style.display = expanded ? "block" : "none";

        const entryEls = [];
        for (const entry of cat.entries) {
            const card = document.createElement("div");
            card.className = "tex-help-card";
            // Store searchable text for filtering
            card.dataset.searchText = `${entry.name} ${entry.sig} ${entry.desc} ${entry.example}`.toLowerCase();

            const sigHTML = entry.sig.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
            const descHTML = entry.desc.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
            const exampleHTML = entry.example.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

            card.innerHTML = `
                <div class="tex-help-card-name">${entry.name.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</div>
                <div class="tex-help-card-sig"><code>${sigHTML}</code></div>
                <div class="tex-help-card-desc">${descHTML}</div>
                <pre class="tex-help-card-example">${exampleHTML}</pre>
            `;
            entriesEl.appendChild(card);
            entryEls.push(card);
        }

        catEl.appendChild(entriesEl);
        categoriesContainer.appendChild(catEl);
        categoryEls.push({ el: catEl, header: catHeader, entries: entriesEl, cards: entryEls });

        // Toggle expand/collapse
        catHeader.addEventListener("mousedown", (e) => {
            e.stopPropagation();
            const isExpanded = catHeader.dataset.expanded === "1";
            catHeader.dataset.expanded = isExpanded ? "0" : "1";
            catHeader.querySelector(".tex-help-cat-arrow").textContent = isExpanded ? "\u25B8" : "\u25BE";
            entriesEl.style.display = isExpanded ? "none" : "block";
        });
    }

    container.appendChild(categoriesContainer);

    // Search logic
    searchInput.addEventListener("input", () => {
        const query = searchInput.value.toLowerCase().trim();
        if (!query) {
            // Reset: show all categories, restore collapse state (collapse all except first)
            for (let i = 0; i < categoryEls.length; i++) {
                const c = categoryEls[i];
                c.el.style.display = "";
                for (const card of c.cards) card.style.display = "";
                const expand = i === 0;
                c.header.dataset.expanded = expand ? "1" : "0";
                c.header.querySelector(".tex-help-cat-arrow").textContent = expand ? "\u25BE" : "\u25B8";
                c.entries.style.display = expand ? "block" : "none";
            }
            return;
        }
        // Filter
        const terms = query.split(/\s+/);
        for (const c of categoryEls) {
            let anyVisible = false;
            for (const card of c.cards) {
                const text = card.dataset.searchText;
                const match = terms.every(t => text.includes(t));
                card.style.display = match ? "" : "none";
                if (match) anyVisible = true;
            }
            c.el.style.display = anyVisible ? "" : "none";
            if (anyVisible) {
                // Expand categories with matches
                c.header.dataset.expanded = "1";
                c.header.querySelector(".tex-help-cat-arrow").textContent = "\u25BE";
                c.entries.style.display = "block";
            }
        }
    });

    return { container, searchInput };
}

let _helpDocked = false; // persists across open/close within session

function showHelpPopup(screenX, screenY) {
    // Close any existing popup first
    hideHelpPopup();

    const popup = document.createElement("div");
    popup.className = "tex-help-popup";

    // ── Title bar (draggable) ──
    const titleBar = document.createElement("div");
    titleBar.className = "tex-help-titlebar";

    // Close button
    const closeBtn = document.createElement("button");
    closeBtn.className = "tex-help-popup-close";
    closeBtn.textContent = "\u00d7";
    closeBtn.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        hideHelpPopup();
    });

    // Dock/undock button
    const dockBtn = document.createElement("button");
    dockBtn.className = "tex-help-dock-btn";
    dockBtn.title = _helpDocked ? "Undock (floating)" : "Dock to right side";
    dockBtn.textContent = _helpDocked ? "\u{1F5D7}" : "\u{1F4CC}";  // 🗗 or 📌
    dockBtn.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        _helpDocked = !_helpDocked;
        applyDockState(popup);
        dockBtn.title = _helpDocked ? "Undock (floating)" : "Dock to right side";
        dockBtn.textContent = _helpDocked ? "\u{1F5D7}" : "\u{1F4CC}";
    });

    titleBar.appendChild(dockBtn);
    titleBar.appendChild(closeBtn);
    popup.appendChild(titleBar);

    // ── Dragging (only when floating, not docked) ──
    let dragStartX = 0, dragStartY = 0, popupStartX = 0, popupStartY = 0;
    let isDragging = false;

    titleBar.addEventListener("mousedown", (e) => {
        if (e.target === closeBtn || e.target === dockBtn) return;
        if (_helpDocked) return; // no drag when docked
        isDragging = true;
        dragStartX = e.clientX;
        dragStartY = e.clientY;
        popupStartX = parseInt(popup.style.left) || 0;
        popupStartY = parseInt(popup.style.top) || 0;
        e.preventDefault();
        e.stopPropagation();
    });

    const onMouseMove = (e) => {
        if (!isDragging) return;
        const dx = e.clientX - dragStartX;
        const dy = e.clientY - dragStartY;
        popup.style.left = (popupStartX + dx) + "px";
        popup.style.top = (popupStartY + dy) + "px";
    };
    const onMouseUp = () => { isDragging = false; };
    document.addEventListener("mousemove", onMouseMove);
    document.addEventListener("mouseup", onMouseUp);

    // Store cleanup refs
    popup._dragCleanup = () => {
        document.removeEventListener("mousemove", onMouseMove);
        document.removeEventListener("mouseup", onMouseUp);
    };

    const { container, searchInput } = buildHelpDOM();
    popup.appendChild(container);

    document.body.appendChild(popup);
    activeHelpPopup = popup;

    // Apply dock or floating position
    if (_helpDocked) {
        applyDockState(popup);
    } else {
        // Position near click, clamped to viewport
        const rect = popup.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        let x = screenX + 10;
        let y = screenY - 20;
        if (x + rect.width > vw - 10) x = vw - rect.width - 10;
        if (y + rect.height > vh - 10) y = vh - rect.height - 10;
        if (x < 10) x = 10;
        if (y < 10) y = 10;
        popup.style.left = x + "px";
        popup.style.top = y + "px";
    }

    // Focus search bar
    setTimeout(() => searchInput.focus(), 50);

    // Outside-click dismissal (delayed; skip when docked to avoid accidental close)
    setTimeout(() => {
        outsideClickHandler = (e) => {
            if (_helpDocked) return; // docked panel stays open
            if (activeHelpPopup && !activeHelpPopup.contains(e.target)) {
                hideHelpPopup();
            }
        };
        document.addEventListener("mousedown", outsideClickHandler);
    }, 100);
}

function applyDockState(popup) {
    if (_helpDocked) {
        popup.classList.add("tex-help-docked");
        popup.style.left = "";
        popup.style.top = "";
    } else {
        popup.classList.remove("tex-help-docked");
        // Position in center of viewport as a reasonable default after undocking
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        popup.style.left = Math.max(10, (vw - 520) / 2) + "px";
        popup.style.top = Math.max(10, (vh * 0.1)) + "px";
    }
}

function hideHelpPopup() {
    if (activeHelpPopup) {
        if (activeHelpPopup._dragCleanup) activeHelpPopup._dragCleanup();
        activeHelpPopup.remove();
        activeHelpPopup = null;
    }
    if (outsideClickHandler) {
        document.removeEventListener("mousedown", outsideClickHandler);
        outsideClickHandler = null;
    }
}

// ─── Cross-node fusion: detect TEX-only chains, draw a compile-block bubble,
//     and collapse the chain into its terminal at submit time (only the
//     terminal cooks). All gated behind the TEX.Fusion.* settings. ───────────

let _texFusionEnabled = true;    // TEX.Fusion.enabled  (master toggle, default on)
let _texShowBubble    = true;    // TEX.Fusion.showBubble
let _texHullShape     = true;    // TEX.Fusion.bubbleHull (convex hull vs rectangle)

// Litegraph's app.graph.links is a Map — ALWAYS use .get(), never bracket-index
// (bracket-index silently returns undefined → empty adjacency → silent no-fuse).
function _texGetLink(id) {
    const links = app.graph?.links;
    if (!links) return null;
    return (typeof links.get === "function") ? links.get(id) : links[id];
}

// {name: value} for this node's $param widgets (used for both fusion payload + injection).
function _texParamWidgets(node) {
    const params = {};
    for (const w of (node.widgets || [])) if (w._texParam) params[w.name] = w.value;
    return params;
}

// All out-links of a node, to any node type: [{targetNode, fromSlotIndex}].
// fromSlotIndex matters: @OUT is always output slot 0 (output_names = sorted
// assigned bindings, "OUT" sorts first), so a chain handoff must come from slot 0.
function _texOutLinks(node) {
    const out = [];
    const outputs = node.outputs || [];
    for (let oi = 0; oi < outputs.length; oi++) {
        for (const linkId of (outputs[oi].links || [])) {
            const link = _texGetLink(linkId);
            if (!link) continue;
            const tn = app.graph.getNodeById(link.target_id);
            if (tn) out.push({ targetNode: tn, fromSlotIndex: oi });
        }
    }
    return out;
}

// The node's single wired IMAGE input (a $param wire is not the chain link), or
// null if it has 0 or >1. Returns {bindingName, origin: [originIdString, originSlot]}.
function _texSingleWiredInput(node) {
    const wired = [];
    const inputs = node.inputs || [];
    for (let s = 0; s < inputs.length; s++) {
        const inp = inputs[s];
        if (inp.link == null) continue;
        if (inp._texParam) continue;   // a wire into a $param socket is not the image chain
        const link = _texGetLink(inp.link);
        if (!link) continue;
        wired.push({ bindingName: inp.name,
                     origin: [String(link.origin_id), link.origin_slot] });
    }
    return wired.length === 1 ? wired[0] : null;
}

// The single upstream TEX node feeding this node (via its sole wired image input), or null.
function _texImageUpstream(node) {
    const w = _texSingleWiredInput(node);
    if (!w) return null;
    const src = app.graph.getNodeById(parseInt(w.origin[0]));
    if (!src || src.type !== TEX_NODE_TYPE) return null;
    return { node: src, bindingName: w.bindingName };
}

// Maximal linear TEX-only chains. A node is an internal link only when its
// output goes to EXACTLY ONE consumer and that consumer is a TEX node reached
// through that node's sole wired input (no fan-out, no external/Preview tap,
// no multi-input). Each chain is [head, ..., terminal] with length >= 2.
function _texDetectChains() {
    // Exclude muted (mode 2) / bypassed (mode 4) nodes: ComfyUI omits them from
    // the prompt, so they can't be fused and shouldn't be boxed.
    const texNodes = (app.graph?._nodes || []).filter(
        n => n.type === TEX_NODE_TYPE && n.mode !== 2 && n.mode !== 4);
    const soleTexConsumer = new Map();   // nodeId -> the single downstream TEX node, or null
    for (const n of texNodes) {
        const outs = _texOutLinks(n);
        soleTexConsumer.set(n.id,
            (outs.length === 1 && outs[0].targetNode.type === TEX_NODE_TYPE
                && outs[0].fromSlotIndex === 0)   // handoff must be the @OUT slot
                ? outs[0].targetNode : null);
    }
    const chains = [];
    for (const term of texNodes) {
        if (soleTexConsumer.get(term.id)) continue;   // internal to some chain — skip
        const chain = [term];
        let current = term;
        const seen = new Set([term.id]);
        while (true) {
            const up = _texImageUpstream(current);
            if (!up) break;
            // up must collapse specifically into `current` (its sole consumer)
            if (soleTexConsumer.get(up.node.id) !== current) break;
            if (seen.has(up.node.id)) break;           // cycle guard
            chain.unshift(up.node);
            seen.add(up.node.id);
            current = up.node;
        }
        if (chain.length >= 2) chains.push(chain);
    }
    return chains;
}

// Collapse one chain in the prompt dict. CHECK-then-MUTATE: nothing is written
// unless the whole rewrite is provably safe (no dangling [deletedId, slot] ref
// can survive — that would KeyError validate_inputs and reject the prompt).
function _texCollapseOne(out, chain) {
    const terminal = chain[chain.length - 1];
    const upstream = chain.slice(0, -1);
    const chainIds = new Set(chain.map(n => String(n.id)));
    const termId = String(terminal.id);

    for (const n of chain) if (!out[String(n.id)]) return false;   // all must be in the prompt

    // The terminal's chain-link input (the binding that reads the chain).
    const termUp = _texImageUpstream(terminal);
    if (!termUp || termUp.node.id !== upstream[upstream.length - 1].id) return false;
    const terminalImageInput = termUp.bindingName;

    // Defense in depth: every upstream node's ONLY consumer must be inside the
    // chain, reached from its @OUT slot (slot 0).
    for (const n of upstream) {
        const outs = _texOutLinks(n);
        if (outs.length !== 1 || outs[0].fromSlotIndex !== 0
            || !chainIds.has(String(outs[0].targetNode.id))) return false;
    }

    // Build the upstream payload (source-first) + find the chain source origin.
    const stages = [];
    for (const n of upstream) {
        const nid = String(n.id);
        const code = out[nid].inputs?.code;
        if (typeof code !== "string") return false;     // a wired `code` input — can't fuse
        // A $param driven by a WIRE (not its widget) can't be folded away: this
        // upstream node is DELETED, taking its param wire with it, and the fused
        // stage would bake the STALE widget value (② silent-wrong). Mirror the
        // region path's param_wired guard (_grow_region). The terminal is not
        // deleted, so its own wired $params survive and are fine.
        if ((n.inputs || []).some(i => i._texParam && i.link != null)) return false;
        const wi = _texSingleWiredInput(n);
        if (!wi) return false;                          // must have exactly one image input
        stages.push({ code, image_input: wi.bindingName, params: _texParamWidgets(n) });
    }
    const headSource = _texSingleWiredInput(upstream[0]);   // [originId, slot] feeding the chain head
    if (!headSource) return false;
    // Origin comes from LITEGRAPH but is spliced into the SERIALIZED prompt, which
    // omits bypassed/muted (mode 2/4) nodes — so a raw origin can name a node that
    // isn't in `out`, and that dangling ref makes ComfyUI reject the WHOLE prompt.
    if (!out[String(headSource.origin[0])]) return false;

    const delSet = new Set(upstream.map(n => String(n.id)));
    // Up-front integrity scan: no SURVIVING node may reference a to-be-deleted node.
    for (const [id, data] of Object.entries(out)) {
        if (delSet.has(id) || id === termId) continue;
        for (const v of Object.values(data.inputs || {})) {
            if (Array.isArray(v) && v.length === 2 && delSet.has(String(v[0]))) return false;
        }
    }

    // All safe → mutate: rewire terminal input to the source, attach payload, delete upstream.
    // The terminal's chain socket carries the source AND is read as the chain — one key.
    out[termId].inputs[terminalImageInput] = headSource.origin;
    out[termId].inputs["_tex_chain"] = JSON.stringify({
        stages,
        terminal_image_input: terminalImageInput,
    });
    for (const id of delSet) delete out[id];
    return true;
}

function _texCollapseChains(result) {
    if (!_texFusionEnabled || !result?.output) return;
    let chains;
    try { chains = _texDetectChains(); } catch (e) { console.warn("[TEX] chain detect failed", e); return; }
    for (const chain of chains) {
        try {
            if (!_texCollapseOne(result.output, chain)) {
                console.info("[TEX] a linked chain was left unfused (an intermediate " +
                             "result is used elsewhere, or a node has a wired code input) " +
                             "— running those nodes separately.");
            }
        } catch (e) { console.warn("[TEX] fuse skipped (left unfused)", e); }
    }
}

// FUS-1: DAG-region fusion (fan-out / diamonds). Detection is the Python authority
// (/tex_wrangle/detect_regions — same legality every host uses); the frontend only
// serializes the graph and applies the returned collapse plans. ADDITIVE to the linear
// pass above, which already ran: purely-linear regions are the route's to skip
// (_region_is_linear), so they never arrive here. The "all region nodes present" check
// below therefore has ONE real effect — a BRANCHING region whose cone overlaps a chain
// the linear pass just deleted is rejected wholesale (we serialize litegraph, which
// still shows those nodes): costs fusion, never correctness (roadmap v0.21.1).
// Fail-safe — a route miss / timeout / unsafe plan leaves those nodes unfused.

// {nodes: [{id, code, params, code_wired}], edges: [{from, from_slot, to, to_binding}]}
// for the detector. Image handoffs only (a $param / code wire is not a fusable edge);
// a consumer that is non-TEX or a non-image socket is marked EXTERNAL (synthetic id)
// so the detector treats the producer as a boundary instead of folding across it.
function _texSerializeGraph() {
    const texNodes = (app.graph?._nodes || []).filter(
        n => n.type === TEX_NODE_TYPE && n.mode !== 2 && n.mode !== 4);
    const nodes = [], edges = [], seen = new Map();
    const addEdge = (from, fromSlot, to, toBinding, fromType) => {
        const k = from + ":" + fromSlot + "->" + to + ":" + toBinding;
        const prev = seen.get(k);
        if (prev) {
            // Let a typed add upgrade an untyped one rather than let iteration order
            // decide. Inert today (only TEX->TEX edges collide, and a TEX output is
            // ANY_TYPE "*" -> IMAGE either way); guards concrete TEX output types later.
            if (fromType && !prev.from_type) prev.from_type = fromType;
            return;
        }
        const e = { from: String(from), from_slot: fromSlot, to: String(to), to_binding: toBinding };
        if (fromType) e.from_type = fromType;
        seen.set(k, e);
        edges.push(e);
    };
    const isImageSocket = (node, slotIdx) => {
        const inp = (node.inputs || [])[slotIdx];
        return !!(inp && !inp._texParam && inp.name !== "code");
    };
    for (const n of texNodes) {
        const nid = String(n.id);
        const codeInput = (n.inputs || []).find(i => i.name === "code");
        const codeWired = !!(codeInput && codeInput.link != null);
        // A $param driven by a WIRE (not its widget) can't be folded away — the
        // serializer only captures widget values, so folding would sever the dynamic
        // value and bake the stale widget (② silent-wrong). Mark such a node so the
        // detector keeps it a region boundary, mirroring code_wired.
        const paramWired = (n.inputs || []).some(i => i._texParam && i.link != null);
        const codeW = (n.widgets || []).find(w => w.name === "code");
        nodes.push({ id: nid, code: codeWired ? "" : (codeW ? String(codeW.value) : ""),
                     params: _texParamWidgets(n), code_wired: codeWired, param_wired: paramWired });
        for (const inp of (n.inputs || [])) {   // image in-edges (producer -> n)
            if (inp.link == null || inp._texParam || inp.name === "code") continue;
            const link = _texGetLink(inp.link);
            if (!link) continue;
            // The producer's declared socket type ("IMAGE"/"MASK"/"LATENT"): for a
            // region-EXTERNAL source it is the only clue to the binding's TEX type at
            // cook, so the preflight can test the family it will really be handed
            // (a MASK is a float, not a vec3/vec4). See _preflight_samples.
            const srcNode = app.graph.getNodeById(link.origin_id);
            const ftype = srcNode?.outputs?.[link.origin_slot]?.type;
            addEdge(link.origin_id, link.origin_slot, nid, inp.name,
                    typeof ftype === "string" ? ftype : null);
        }
        const outputs = n.outputs || [];        // out-edges (escape detection)
        for (let oi = 0; oi < outputs.length; oi++) {
            for (const linkId of (outputs[oi].links || [])) {
                const link = _texGetLink(linkId);
                if (!link) continue;
                const tn = app.graph.getNodeById(link.target_id);
                if (tn && tn.type === TEX_NODE_TYPE && isImageSocket(tn, link.target_slot)) {
                    addEdge(nid, oi, String(link.target_id), tn.inputs[link.target_slot].name);
                } else {
                    addEdge(nid, oi, "ext_" + link.target_id + "_" + link.target_slot, "-");
                }
            }
        }
    }
    return { nodes, edges };
}

// Apply ONE region collapse plan to the prompt dict. CHECK-then-MUTATE: nothing is
// written unless the whole rewrite is provably safe (mirrors _texCollapseOne).
function _texCollapseRegion(out, plan) {
    const termId = String(plan.terminal);
    const delIds = (plan.delete || []).map(String);
    const delSet = new Set(delIds);
    const srcOrigin = plan.source_origin;   // [originIdString, slot]
    if (!out[termId] || !Array.isArray(srcOrigin) || delSet.has(termId)) return false;
    for (const id of delIds) if (!out[id]) return false;         // whole region present?
    if (delSet.has(String(srcOrigin[0]))) return false;          // source not in THIS delete set
    // C1/C2: the source node must still EXIST in the prompt. The linear pass runs first
    // and can delete it (a boundary node it collapses anyway), and ComfyUI strips
    // bypassed/muted (mode 2/4) nodes — either way rewiring the terminal to a vanished
    // origin makes the prompt reference a deleted node and ComfyUI rejects the WHOLE
    // queue. Leaving the region unfused is always correct.
    if (!out[String(srcOrigin[0])]) return false;

    // Terminal input sockets currently fed by a to-be-deleted region node.
    const termInputs = out[termId].inputs || {};
    const upstream = [];
    for (const [binding, v] of Object.entries(termInputs))
        if (Array.isArray(v) && v.length === 2 && delSet.has(String(v[0]))) upstream.push(binding);
    if (!upstream.length) return false;

    // No SURVIVING node outside the region may reference a to-be-deleted node.
    for (const [id, data] of Object.entries(out)) {
        if (delSet.has(id) || id === termId) continue;
        for (const v of Object.values(data.inputs || {}))
            if (Array.isArray(v) && v.length === 2 && delSet.has(String(v[0]))) return false;
    }

    // Safe → mutate. Reuse the first upstream socket as the SOURCE transport (rewired
    // to the source origin); the fused program routes it to the source stage and the
    // terminal's own reads come from its chain_inputs. Remove the other upstream
    // sockets — their values are internal handoffs now.
    const transport = upstream[0];
    out[termId].inputs[transport] = [String(srcOrigin[0]), srcOrigin[1]];
    for (let i = 1; i < upstream.length; i++) delete out[termId].inputs[upstream[i]];
    out[termId].inputs["_tex_chain"] = JSON.stringify(
        Object.assign({}, plan.payload, { terminal_image_input: transport }));
    for (const id of delSet) delete out[id];
    return true;
}

// Memoize the route result by the serialized-graph signature: re-queuing an
// unchanged graph (or auto-queue) shouldn't re-serialize + round-trip to localhost
// every time (the sibling preflight route memoizes the same way). The signature must
// include param VALUES — the collapse payload bakes them, so a stale plan would fuse
// the wrong values — so a widget tweak correctly re-detects.
let _texRegionSig = null;
let _texRegionPlans = [];
async function _texCollapseRegions(result) {
    if (!_texFusionEnabled || !result?.output) return;
    let graph;
    try { graph = _texSerializeGraph(); } catch (e) { return; }
    if (!graph.nodes.length) return;
    const sig = JSON.stringify(graph);
    if (sig !== _texRegionSig) {
        try {
            const ctrl = new AbortController();
            const timer = setTimeout(() => ctrl.abort(), 800);   // never stall the queue
            const resp = await fetch("/tex_wrangle/detect_regions", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: sig, signal: ctrl.signal });
            clearTimeout(timer);
            _texRegionPlans = ((await resp.json()) || {}).plans || [];
            _texRegionSig = sig;
        } catch (e) { _texRegionSig = null; return; }   // route absent/slow → unfused; retry next queue
    }
    for (const plan of _texRegionPlans) {
        try { _texCollapseRegion(result.output, plan); }
        catch (e) { console.warn("[TEX] region fuse skipped", e); }
    }
}

// DBG-1: per-node tier/timing HUD badge toggle (TEX.Debug.perfHud).
let _texPerfHudEnabled = true;

// ─── Q-5: chain preflight + perf HUD ────────────────────────────────────────
let _texPreflightEnabled = true;              // TEX.Fusion.preflight
const _texPreflight = new Map();              // termId -> {ok, error, stage_of_error, stats}
const _texPreflightSig = new Map();           // termId -> last-checked signature
const _texPreflightBusy = new Set();          // termIds with an in-flight request

function _texCodeOf(node) {
    const w = (node.widgets || []).find(w => w.name === "code");
    return typeof w?.value === "string" ? w.value : "";
}

// Serialize a chain's fusability-relevant shape (codes + wiring) so preflight
// only re-runs when the structure actually changes — not on every pan/repaint.
function _texChainSig(chain) {
    return chain.map(n => {
        const up = _texSingleWiredInput(n);
        return n.id + ":" + _texCodeOf(n).length + ":" + (up ? up.bindingName : "-");
    }).join("|") + "#" + chain.map(n => _texCodeOf(n)).join("\x00");
}

// Build the preflight spec (source-first stages + terminal code) from live node
// widgets — mirrors _texCollapseOne's payload but at draw time (no prompt dict).
function _texPreflightSpec(chain) {
    const upstream = chain.slice(0, -1);
    const terminal = chain[chain.length - 1];
    const stages = [];
    for (const n of upstream) {
        const wi = _texSingleWiredInput(n);
        if (!wi) return null;
        stages.push({ code: _texCodeOf(n), image_input: wi.bindingName,
                      params: _texParamWidgets(n) });
    }
    const termUp = _texSingleWiredInput(terminal);
    if (!termUp) return null;
    return { stages, terminal_image_input: termUp.bindingName,
             terminal_code: _texCodeOf(terminal) };
}

// Debounced-per-signature preflight of one chain. Never throws; a network/route
// error simply leaves the bubble in its neutral (fusable) state.
function _texMaybePreflight(chain) {
    if (!_texPreflightEnabled) return;
    const termId = chain[chain.length - 1].id;
    let sig;
    try { sig = _texChainSig(chain); } catch (e) { return; }
    if (_texPreflightSig.get(termId) === sig || _texPreflightBusy.has(termId)) return;
    const spec = (() => { try { return _texPreflightSpec(chain); } catch (e) { return null; } })();
    if (!spec) return;
    _texPreflightSig.set(termId, sig);
    _texPreflightBusy.add(termId);
    fetch("/tex_wrangle/chain_preflight", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(spec),
    }).then(r => r.json()).then(res => {
        _texPreflight.set(termId, res);
        app.canvas?.setDirty(true, true);   // repaint the bubble with the verdict
    }).catch(() => { /* endpoint absent / offline → neutral bubble */ })
      .finally(() => _texPreflightBusy.delete(termId));
}

function _texRoundRect(ctx, x, y, w, h, r) {
    if (typeof ctx.roundRect === "function") { ctx.beginPath(); ctx.roundRect(x, y, w, h, r); return; }
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

// Convex hull (Andrew's monotone chain) of a set of {x, y} points.
function _texConvexHull(points) {
    const pts = points.slice().sort((a, b) => a.x - b.x || a.y - b.y);
    if (pts.length < 3) return pts;
    const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    const lower = [];
    for (const p of pts) {
        while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
        lower.push(p);
    }
    const upper = [];
    for (let i = pts.length - 1; i >= 0; i--) {
        const p = pts[i];
        while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
        upper.push(p);
    }
    lower.pop(); upper.pop();
    return lower.concat(upper);
}

// Trace a rounded polygon (smooth corners) through the given points.
function _texRoundedPolyPath(ctx, pts, radius) {
    const n = pts.length;
    if (n < 3) return;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
        const prev = pts[(i - 1 + n) % n], curr = pts[i], next = pts[(i + 1) % n];
        const v1x = curr.x - prev.x, v1y = curr.y - prev.y;
        const v2x = next.x - curr.x, v2y = next.y - curr.y;
        const len1 = Math.hypot(v1x, v1y) || 1, len2 = Math.hypot(v2x, v2y) || 1;
        const r = Math.min(radius, len1 / 2, len2 / 2);
        const p1x = curr.x - v1x / len1 * r, p1y = curr.y - v1y / len1 * r;
        const p2x = curr.x + v2x / len2 * r, p2y = curr.y + v2y / len2 * r;
        if (i === 0) ctx.moveTo(p1x, p1y); else ctx.lineTo(p1x, p1y);
        ctx.quadraticCurveTo(curr.x, curr.y, p2x, p2y);
    }
    ctx.closePath();
}

// Is the point (x, y) inside the convex polygon `pts`?
function _texPointInConvex(pts, x, y) {
    if (pts.length < 3) return false;   // empty / degenerate hull contains nothing
    let sign = 0;
    for (let i = 0; i < pts.length; i++) {
        const a = pts[i], b = pts[(i + 1) % pts.length];
        const cr = (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x);
        if (cr !== 0) {
            const s = cr > 0 ? 1 : -1;
            if (sign === 0) sign = s; else if (s !== sign) return false;
        }
    }
    return true;
}

// Compile-block accent colour, shared by the fill, stroke, and hover label.
const BUBBLE_COLOR = "#4FC3F7";
const BUBBLE_ERROR_COLOR = "#EF5350";   // Q-5: red bubble for an unfusable chain

// Per-frame cache of the visible bubbles (shape + label), populated by
// _texDrawBubbles on the background pass and read by the hover-label foreground
// pass. Each entry: { hull: [{x,y}], text, lx, ly }.
let _texBubbleCache = [];
// Index of the bubble the mouse is currently inside (-1 = none) — drives the
// foreground repaint throttle so we only repaint when the hovered bubble changes.
let _texBubbleHover = -1;

// Draw the faint compile-block bubble behind each detected chain (canvas space).
// A rounded convex hull (Houdini-style) when TEX.Fusion.bubbleHull is on, else a
// rounded rectangle — both with a faint fill and a fine continuous outline. The
// label is NOT drawn here; it's drawn on hover by _texDrawBubbleLabels.
function _texDrawBubbles(ctx) {
    _texBubbleCache = [];
    if (!_texFusionEnabled || !_texShowBubble) return;
    // Skip detection when zoomed out — the label isn't legible and detection on
    // every zoomed-out pan repaint isn't worth it.
    if ((app.canvas?.ds?.scale ?? 1) < 0.4) return;
    let chains;
    try { chains = _texDetectChains(); } catch (e) { return; }
    const TH = (typeof LiteGraph !== "undefined" && LiteGraph.NODE_TITLE_HEIGHT) || 24;
    const alpha = app.canvas?.editor_alpha ?? 1;
    const pad = 16;
    for (const chain of chains) {
        // Q-5: debounced preflight of this chain (validates fusability + stats).
        _texMaybePreflight(chain);
        const termId = chain[chain.length - 1].id;
        const pf = _texPreflight.get(termId);
        const bad = pf && pf.ok === false;
        // Padded corner points of every node (title bar included) + the group's
        // bounding box (used for the rectangle fallback and label placement).
        const pts = [];
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const n of chain) {
            const ax = n.pos[0] - pad, ay = n.pos[1] - TH - pad;
            const bx = n.pos[0] + n.size[0] + pad, by = n.pos[1] + n.size[1] + pad;
            pts.push({ x: ax, y: ay }, { x: bx, y: ay }, { x: bx, y: by }, { x: ax, y: by });
            x0 = Math.min(x0, ax); y0 = Math.min(y0, ay);
            x1 = Math.max(x1, bx); y1 = Math.max(y1, by);
        }
        const useHull = _texHullShape && pts.length >= 3;
        const poly = useHull ? _texConvexHull(pts)
            : [{ x: x0, y: y0 }, { x: x1, y: y0 }, { x: x1, y: y1 }, { x: x0, y: y1 }];
        const color = bad ? BUBBLE_ERROR_COLOR : BUBBLE_COLOR;
        ctx.save();
        if (useHull) _texRoundedPolyPath(ctx, poly, 18);
        else _texRoundRect(ctx, x0, y0, x1 - x0, y1 - y0, 12);
        ctx.globalAlpha = (bad ? 0.14 : 0.10) * alpha; ctx.fillStyle = color; ctx.fill();
        ctx.globalAlpha = 0.6 * alpha; ctx.strokeStyle = color;
        ctx.lineWidth = 1; ctx.setLineDash(bad ? [6, 4] : []);
        ctx.stroke();
        ctx.restore();
        if (poly.length >= 3) {
            // Q-5 HUD: red bubble names the blocking reason; green bubble shows
            // the fused stage count and (when preflight reported it) the op count.
            let text;
            if (bad) {
                const where = (pf.stage_of_error != null) ? ` (node #${chain[pf.stage_of_error]?.id ?? "?"})` : "";
                text = `⚠ chain not fusable${where}: ${pf.error}`;
            } else {
                const ops = pf?.stats?.tensor_ops;
                text = `TEX fused · ${chain.length} stages · 1 cook`
                     + (ops != null ? ` · ~${ops} ops` : "");
            }
            _texBubbleCache.push({ hull: poly, text, lx: x0 + 10, ly: y0 - 6 });
        }
    }
}

// Foreground pass: draw a bubble's label only while the mouse hovers its area.
function _texDrawBubbleLabels(ctx) {
    if (!_texFusionEnabled || !_texShowBubble || _texBubbleCache.length === 0) return;
    const m = app.canvas?.graph_mouse;
    if (!m) return;
    const alpha = app.canvas?.editor_alpha ?? 1;
    ctx.save();
    ctx.globalAlpha = 0.9 * alpha; ctx.fillStyle = BUBBLE_COLOR;
    ctx.textAlign = "left"; ctx.font = "12px 'Cascadia Code', monospace";
    for (const b of _texBubbleCache) {
        if (_texPointInConvex(b.hull, m[0], m[1])) ctx.fillText(b.text, b.lx, b.ly);
    }
    ctx.restore();
}

// ─── Extension Registration ──────────────────────────────────────────

app.registerExtension({
    name: "TEX.Wrangle",

    // ── Prompt hook: inject $param widget values into the execution prompt ──
    // ComfyUI only serializes widget values for inputs defined in the schema.
    // Since TEX params are dynamic (parsed from code), they aren't in the schema.
    // This hook adds their widget values so accept_all_inputs passes them to execute().
    setup() {
        // Lazy input cooking: ComfyUI only honours `lazy` on schema-declared
        // input names, so wired user inputs are renamed onto the schema's
        // lazy slot pool (in_0..in_15) IN THE QUEUED PROMPT ONLY — the graph,
        // saved workflow, slots, and labels all keep the user names. The
        // `_tex_slot_map` constant lets the backend map values back and
        // drives check_lazy_status. Runs AFTER fusion collapse so it maps
        // the post-collapse wiring. Fail-safe: on any doubt (pool-name
        // collision, >16 wires) inputs stay user-named and simply cook
        // eagerly, exactly as before v0.18.
        const TEX_SYS_INPUTS = new Set([
            "code", "device", "compile_mode", "precision",
            "debug_nan_highlight", "_tex_any", "_tex_chain",
            "_tex_preview", "_tex_slot_map",
        ]);
        function _texLazyRename(result) {
            if (!app.ui.settings.getSettingValue("TEX.Lazy.enabled", true)) return;
            for (const [nodeId, nodeData] of Object.entries(result.output)) {
                if (nodeData.class_type !== TEX_NODE_TYPE) continue;
                const node = app.graph.getNodeById(parseInt(nodeId));
                if (!node) continue;
                // Collision guard: a user input literally named like a pool
                // slot would collide after renaming — leave the node eager.
                const names = (node.inputs || []).map(i => i?.name).filter(Boolean);
                if (names.some(n => /^in_\d+$/.test(n))) continue;
                const map = [];
                let idx = 0;
                for (const input of node.inputs || []) {
                    if (idx >= 16) break; // pool size; extras stay eager
                    const name = input?.name;
                    if (!name || TEX_SYS_INPUTS.has(name)) continue;
                    const val = nodeData.inputs[name];
                    // Only wired links ([nodeId, slotIdx] arrays) move to the
                    // pool; widget constants keep their user names.
                    if (!Array.isArray(val)) continue;
                    const slot = `in_${idx++}`;
                    nodeData.inputs[slot] = val;
                    delete nodeData.inputs[name];
                    let t = input.type || "*";
                    if (input.link != null) {
                        const link = app.graph.links[input.link];
                        if (link?.type) t = link.type;
                    }
                    map.push({ name, slot, type: String(t) });
                }
                if (map.length) {
                    nodeData.inputs._tex_slot_map = JSON.stringify(map);
                }
            }
        }

        const origGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function (...args) {
            const result = await origGraphToPrompt.apply(this, args);
            if (result?.output) {
                for (const [nodeId, nodeData] of Object.entries(result.output)) {
                    if (nodeData.class_type === TEX_NODE_TYPE) {
                        const node = app.graph.getNodeById(parseInt(nodeId));
                        if (node?.widgets) {
                            for (const w of node.widgets) {
                                if (w._texParam && !(w.name in nodeData.inputs)) {
                                    nodeData.inputs[w.name] = w.value;
                                }
                            }
                        }
                    }
                }
                // Cross-node fusion: collapse linked TEX chains into their
                // terminal (gated by TEX.Fusion.enabled; fail-safe — never
                // produces a prompt with a dangling reference).
                try { _texCollapseChains(result); }
                catch (e) { console.warn("[TEX] fusion collapse skipped", e); }
                // FUS-1: DAG-region fusion (fan-out the linear pass skips). Awaited
                // (Python detection via route) but timeout-bounded + fail-safe, so it
                // never stalls or breaks a queue.
                try { await _texCollapseRegions(result); }
                catch (e) { console.warn("[TEX] region fusion skipped", e); }
                // Lazy input cooking (gated by TEX.Lazy.enabled; fail-safe —
                // any error leaves the prompt user-named and fully eager).
                try { _texLazyRename(result); }
                catch (e) { console.warn("[TEX] lazy rename skipped", e); }
            }
            return result;
        };

        // ── Dynamic styles: font size + line numbers ──────────────────────
        const dynStyle = document.createElement("style");
        dynStyle.id = "tex-dynamic-styles";
        document.head.appendChild(dynStyle);

        let _texFontSize    = 14;
        let _texLineNumbers = true;

        function _updateDynStyles() {
            let css = `
        .tex-cm-container .cm-editor,
        .tex-cm-container .cm-editor .cm-content,
        .tex-cm-container .cm-editor .cm-line,
        .tex-cm-container .cm-editor .cm-content * {
            font-size: ${_texFontSize}px !important;
        }`;
            if (!_texLineNumbers) {
                css += `
        .tex-cm-container .cm-lineNumbers,
        .tex-cm-container .cm-gutter.cm-lineNumbers {
            display: none !important;
        }`;
            }
            dynStyle.textContent = css;
        }

        // Apply saved values immediately on load
        _texFontSize    = app.ui.settings.getSettingValue("TEX.Editor.fontSize",    10);
        _texLineNumbers = app.ui.settings.getSettingValue("TEX.Editor.lineNumbers", true);
        _updateDynStyles();

        // Font size slider (4 – 20 px)
        app.ui.settings.addSetting({
            id:           "TEX.Editor.fontSize",
            name:         "TEX Editor: Font Size",
            type:         "slider",
            defaultValue: 10,
            min:  4,
            max:  20,
            step: 1,
            onChange(val) {
                _texFontSize = val;
                _updateDynStyles();
            },
        });

        // Lazy input cooking toggle (see _texLazyRename)
        app.ui.settings.addSetting({
            id:           "TEX.Lazy.enabled",
            name:         "TEX Lazy: Skip cooking unused inputs",
            type:         "boolean",
            defaultValue: true,
            tooltip:      "When TEX code doesn't use a wired input (including " +
                          "branches disabled by a $param), the upstream nodes " +
                          "feeding it are never cooked. Disable to always cook " +
                          "every wired input.",
        });

        // Line-number gutter toggle
        app.ui.settings.addSetting({
            id:           "TEX.Editor.lineNumbers",
            name:         "TEX Editor: Show Line Numbers",
            type:         "boolean",
            defaultValue: true,
            onChange(val) {
                _texLineNumbers = val;
                _updateDynStyles();
            },
        });

        // Autocomplete toggle — reconfigures every open editor live
        app.ui.settings.addSetting({
            id:           "TEX.Editor.autocomplete",
            name:         "TEX Editor: Autocomplete",
            type:         "boolean",
            defaultValue: true,
            onChange(val) {
                const CM6 = getCM6();
                if (!CM6 || !app.graph?._nodes) return;
                for (const n of app.graph._nodes) {
                    const meta   = texEditorMeta.get(n);
                    const editor = texEditors.get(n);
                    if (!meta || !editor) continue;
                    editor.dispatch({
                        effects: meta.autocompleteCompartment.reconfigure(
                            val
                                ? CM6.autocompletion({ override: [meta.completionSource], activateOnTyping: true })
                                : []
                        ),
                    });
                }
            },
        });

        // ── Cross-node fusion settings ────────────────────────────────────
        _texFusionEnabled = app.ui.settings.getSettingValue("TEX.Fusion.enabled", true);
        _texShowBubble    = app.ui.settings.getSettingValue("TEX.Fusion.showBubble", true);
        _texHullShape     = app.ui.settings.getSettingValue("TEX.Fusion.bubbleHull", true);
        _texPreflightEnabled = app.ui.settings.getSettingValue("TEX.Fusion.preflight", true);

        app.ui.settings.addSetting({
            id:           "TEX.Fusion.enabled",
            name:         "TEX Fusion: Compile linked TEX nodes together",
            tooltip:      "At queue time, collapse a chain of linked TEX nodes into one " +
                          "compiled program so only the last node cooks (no intermediate " +
                          "images are materialized or cached). A chain splits wherever an " +
                          "intermediate output is previewed or used elsewhere.",
            type:         "boolean",
            defaultValue: true,
            onChange(val) { _texFusionEnabled = val; app.canvas?.setDirty(true, true); },
        });

        app.ui.settings.addSetting({
            id:           "TEX.Fusion.showBubble",
            name:         "TEX Fusion: Show grouping bubble",
            tooltip:      "Draw a faint outline around TEX nodes that will be compiled together.",
            type:         "boolean",
            defaultValue: true,
            onChange(val) { _texShowBubble = val; app.canvas?.setDirty(true, true); },
        });

        app.ui.settings.addSetting({
            id:           "TEX.Fusion.bubbleHull",
            name:         "TEX Fusion: Bubble as convex hull",
            tooltip:      "Wrap the fused nodes in a Houdini-style rounded convex hull. " +
                          "Turn off for a plain rounded rectangle.",
            type:         "boolean",
            defaultValue: true,
            onChange(val) { _texHullShape = val; app.canvas?.setDirty(true, true); },
        });

        app.ui.settings.addSetting({
            id:           "TEX.Debug.perfHud",
            name:         "TEX Debug: Per-node tier/timing HUD",
            tooltip:      "Show a small badge under each TEX node after it cooks: which " +
                          "acceleration tier served it (interpreter/codegen/cuda_graph), " +
                          "the cook time in ms, and the precision — amber if a tier fell back.",
            type:         "boolean",
            defaultValue: true,
            onChange(val) { _texPerfHudEnabled = val; app.canvas?.setDirty(true, true); },
        });

        app.ui.settings.addSetting({
            id:           "TEX.Fusion.preflight",
            name:         "TEX Fusion: Preflight chains + HUD",
            tooltip:      "Validate a linked TEX chain's fusability as you edit (a red " +
                          "dashed bubble names the blocking node/reason before you queue), " +
                          "and show the fused stage count and op count on the bubble.",
            type:         "boolean",
            defaultValue: true,
            onChange(val) { _texPreflightEnabled = val; app.canvas?.setDirty(true, true); },
        });

        // ── Compile-block bubble: shape behind nodes (onDrawBackground), label
        //    on hover above nodes (onDrawForeground). ───
        const _installBubble = () => {
            const canvas = app.canvas;
            if (!canvas || canvas._texBubbleHooked) return false;
            // Wrap a canvas draw hook, preserving any handler already installed.
            const wrap = (name, draw) => {
                const orig = canvas[name];
                canvas[name] = function (ctx, ...rest) {
                    if (orig) orig.call(this, ctx, ...rest);
                    try { draw(ctx); } catch (e) { /* never break canvas paint */ }
                };
            };
            wrap("onDrawBackground", _texDrawBubbles);       // shape, behind nodes
            wrap("onDrawForeground", _texDrawBubbleLabels);  // hover label, above nodes
            // Repaint the foreground only when the mouse crosses a bubble boundary,
            // so the hover label appears/clears without a repaint on every move.
            canvas.canvas?.addEventListener("mousemove", () => {
                if (!_texFusionEnabled || !_texShowBubble || !_texBubbleCache.length) return;
                const m = canvas.graph_mouse;
                const idx = m ? _texBubbleCache.findIndex(b => _texPointInConvex(b.hull, m[0], m[1])) : -1;
                if (idx !== _texBubbleHover) { _texBubbleHover = idx; canvas.setDirty(true); }
            });
            canvas._texBubbleHooked = true;
            return true;
        };
        if (!_installBubble()) {
            // Canvas not ready yet — retry on the next frames until it exists.
            const iv = setInterval(() => { if (_installBubble()) clearInterval(iv); }, 200);
            setTimeout(() => clearInterval(iv), 5000);
        }
    },

    async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
        if (nodeData.name !== TEX_NODE_TYPE) return;

        // ── onNodeCreated ──
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);

            const node = this;
            const codeWidget = node.widgets?.find(w => w.name === "code");
            if (!codeWidget) return;

            // Remove any initial inputs/outputs from the backend schema.
            // We manage sockets dynamically based on code content.
            if (node.inputs) {
                for (let i = node.inputs.length - 1; i >= 0; i--) {
                    node.removeInput(i);
                }
            }
            if (node.outputs) {
                for (let i = node.outputs.length - 1; i >= 0; i--) {
                    node.removeOutput(i);
                }
            }

            // Auto-socket updater (shared by both CM6 and fallback paths)
            const updateSockets = debounce(() => {
                applyCodeToSockets(node, codeWidget.value || "");
            }, DEBOUNCE_MS);

            // ── Try CM6 Editor Integration ──
            const result = createTexEditor(node, codeWidget, updateSockets);

            if (result) {
                try {
                    const { container, editor } = result;

                    // ── Suppress the original code textarea widget ──
                    // Two independent mechanisms ensure the textarea is
                    // invisible in every rendering mode:
                    //
                    // 1. Splice from widgets array — prevents the Vue layer
                    //    from creating a ComponentWidget for it.
                    // 2. Mark hidden + poll to hide/remove the DOM element
                    //    — catches the textarea that ComfyUI's canvas
                    //    renderer may have already inserted before onNodeCreated fires.
                    //
                    // The original JS object stays alive (referenced by the
                    // CM6 updateListener closure) so value sync still works.
                    const codeIdx = node.widgets.indexOf(codeWidget);
                    if (codeIdx >= 0) node.widgets.splice(codeIdx, 1);

                    codeWidget.hidden = true;
                    codeWidget.computeSize = function () {
                        return [0, -4]; // -4 absorbs inter-widget spacing
                    };

                    // Poll for the textarea DOM element and hide it + its
                    // .dom-widget container.  The element may not exist yet
                    // at this point (created lazily by the draw loop).
                    const hideOriginal = () => {
                        const el = codeWidget.element || codeWidget.inputEl;
                        if (el) {
                            el.style.display = "none";
                            el.style.height = "0";
                            el.style.overflow = "hidden";
                            const parentEl = el.parentElement;
                            if (parentEl) {
                                parentEl.style.display = "none";
                                parentEl.style.height = "0";
                                parentEl.style.overflow = "hidden";
                            }
                            return true;
                        }
                        return false;
                    };
                    if (!hideOriginal()) {
                        const poll = setInterval(() => {
                            if (hideOriginal()) clearInterval(poll);
                        }, 200);
                        setTimeout(() => clearInterval(poll), 10000);
                    }

                    // Add the CM6 editor as a DOM widget named "code".
                    // Re-using the input name lets ComfyUI's serialization
                    // find this widget for the "code" schema input.
                    const domWidget = node.addDOMWidget("code", "customwidget", container, {
                        getValue() {
                            return editor.state.doc.toString();
                        },
                        setValue(v) {
                            if (v == null) return;
                            const current = editor.state.doc.toString();
                            if (v !== current) {
                                editor.dispatch({
                                    changes: { from: 0, to: editor.state.doc.length, insert: v },
                                });
                            }
                        },
                        // Tell the layout engine this widget is "growable":
                        // it has a small minimum and will expand to fill
                        // remaining vertical space.  DOM widgets with
                        // getMinHeight (fed to computeLayoutSize) participate
                        // in LiteGraph's distributeSpace() allocation instead
                        // of being treated as fixed-height.  This avoids the
                        // old computeSize-based approach which either caused
                        // infinite growth or prevented downward resizing.
                        getMinHeight() { return 100; },
                    });
                    if (domWidget) {
                        // Remove any inherited computeSize so the layout
                        // engine treats this widget as growable (via
                        // computeLayoutSize) rather than fixed-height.
                        domWidget.computeSize = null;

                        // The DOM widget's parent container (.dom-widget)
                        // has pointer-events:auto (reset by ComfyUI's draw loop
                        // every frame) and can overlap canvas-rendered param
                        // widgets below it.  We add a CSS class and use
                        // !important to keep pointer events disabled on the
                        // container while allowing the editor content to remain
                        // interactive.
                        const markContainer = () => {
                            const el = domWidget.element;
                            const parentContainer = el?.parentElement;
                            if (parentContainer && parentContainer.classList.contains("dom-widget")) {
                                parentContainer.classList.add("tex-editor-container");
                                return true;
                            }
                            return false;
                        };
                        if (!markContainer()) {
                            const check = setInterval(() => {
                                if (markContainer()) clearInterval(check);
                            }, 200);
                            setTimeout(() => clearInterval(check), 10000);
                        }

                    }

                    // Size the default node for a better editor experience
                    if (node.size[0] < 400) {
                        node.size[0] = 400;
                    }
                    if (node.size[1] < 280) {
                        node.size[1] = 280;
                    }
                } catch (err) {
                    console.error("[TEX] Failed to mount CM6 editor on node:", err);
                }
            } else {
                // Fallback: plain textarea (CM6 not available)
                console.warn("[TEX] Using fallback textarea for node", node.id);

                const origCallback = codeWidget.callback;
                codeWidget.callback = function (...args) {
                    if (origCallback) origCallback.apply(this, args);
                    updateSockets();
                };
            }

            // Initial socket scan (both paths)
            setTimeout(updateSockets, 200);

            // Register for Nodes 2.0 RAF-based overlay positioning.
            // In Nodes 1.0, onDrawForeground sets _texV1Positioned=true
            // each frame so the RAF loop skips this node.
            _texStartV2Positioning(node);
            _texEnsureV2Raf();
        };

        // ── onConfigure: restore dynamic inputs from saved workflow ──
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origConfigure) origConfigure.apply(this, arguments);
            const node = this;
            setTimeout(() => {
                const codeWidget = node.widgets?.find(w => w.name === "code");
                if (codeWidget) {
                    const code = codeWidget.value || "";
                    applyCodeToSockets(node, code);

                    // Sync CM6 editor content if it exists
                    const editor = texEditors.get(node);
                    if (editor) {
                        const current = editor.state.doc.toString();
                        if (current !== code) {
                            editor.dispatch({
                                changes: { from: 0, to: editor.state.doc.length, insert: code },
                            });
                        }
                    }
                }
            }, 300);
        };

        // ── Custom drawing: TEX badge + error overlay ──
        const origDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origDrawForeground) origDrawForeground.apply(this, arguments);
            if (this.flags?.collapsed) {
                // Hide floating overlays when collapsed
                if (this._texErrorBanner) this._texErrorBanner.style.display = "none";
                if (this._texHelpBtn) this._texHelpBtn.style.display = "none";
                return;
            }

            // "TEX" badge
            ctx.save();
            ctx.font = "bold 10px 'Monaspace Neon', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace";
            ctx.fillStyle = "#4FC3F7";
            ctx.textAlign = "right";
            ctx.fillText("TEX", this.size[0] - 8, -6);
            ctx.restore();

            // DBG-1: per-cook tier/timing HUD badge below the node (green normally,
            // amber when a tier fell back). Feature-flagged; drawn from the ui= facts
            // captured in onExecuted.
            if (_texPerfHudEnabled && this._texPerf) {
                const p = this._texPerf;
                ctx.save();
                ctx.font = "9px 'Cascadia Code', 'Consolas', monospace";
                ctx.textAlign = "left";
                ctx.fillStyle = p.fallback_from ? "#FFB300" : "#66BB6A";
                const prec = p.precision && p.precision !== "fp32" ? " · " + p.precision : "";
                const fb = p.fallback_from ? " (←" + p.fallback_from + ")" : "";
                ctx.fillText(p.tier + fb + " · " + Number(p.elapsed_ms).toFixed(1) + "ms" + prec,
                             8, this.size[1] + 12);
                ctx.restore();
            }

            // ── Position floating DOM overlays ──
            // Nodes 1.0: use canvas transform (ctx.getTransform())
            // Nodes 2.0: onDrawForeground doesn't fire — handled by _texOverlayRafLoop
            this._texV1Positioned = true; // Signal RAF loop to skip (Nodes 1.0 handled it)
            _texPositionOverlays(this, ctx);
        };

        // Clear error after successful execution
        const origExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            if (origExecuted) origExecuted.apply(this, arguments);
            // DBG-1: capture this cook's tier/timing facts from the ui= payload for the
            // per-node perf badge (drawn in onDrawForeground). Defensive — any shape
            // change just leaves the previous badge.
            try {
                const perf = output?.tex_perf?.[0];
                if (perf) this._texPerf = perf;
                this._texProbes = output?.tex_probes || null;  // C1-ux: debug_print taps
                _texUpdatePerfBadge(this);                     // C1-ux: DOM dual-path badge
                _texStartV2Positioning(this);                  // ensure the RAF positions it
            } catch { /* ignore */ }
            texErrorCache.delete(String(this.id));
            // Clear DOM error banner
            clearDOMErrorBanner(this);
            // Clear CM6 inline diagnostics
            const CM6 = getCM6();
            const editor = texEditors.get(this);
            if (editor && CM6) {
                try {
                    editor.dispatch(CM6.setDiagnostics(editor.state, []));
                } catch (err) {
                    console.warn("[TEX] Failed to clear diagnostics:", err);
                }
            }
            this.setDirtyCanvas(true, true);
        };

        // Clean up floating DOM elements + editor when node is removed
        const origOnRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            if (origOnRemoved) origOnRemoved.apply(this, arguments);
            clearDOMErrorBanner(this);
            if (this._texHelpBtn) {
                this._texHelpBtn.remove();
                this._texHelpBtn = null;
            }
            if (this._texPerfBadge) {   // C1-ux
                this._texPerfBadge.remove();
                this._texPerfBadge = null;
            }
            _texStopV2Positioning(this);
            this._texVueEl = null;

            // Destroy the CM6 EditorView (detaches its global listeners and
            // observers). Fallback-textarea nodes have no editor entry.
            const editor = texEditors.get(this);
            if (editor) {
                try { editor.destroy(); } catch (_) { /* already detached */ }
                texEditors.delete(this);
                texEditorMeta.delete(this);
            }
            this._texEditor = null;
            this._texEditorContainer = null;

            // Drop this node's $param entries from the shared schema
            _texSyncParamSchema(this, null);
        };
    },
});

// ─── Styles (injected once) ──────────────────────────────────────────

(function injectStyles() {
    if (document.getElementById("tex-highlight-styles")) return;

    // Preload font via FontFace API — this forces the browser to
    // actually fetch the WOFF2 immediately rather than waiting for
    // a DOM element to use it (which may never trigger if CM6's
    // inline styles resolve a system font first).
    try {
        const face = new FontFace(
            "Monaspace Neon",
            `url('${TEX_FONT_URL}') format('woff2')`,
            { weight: "100 900", style: "normal" }
        );
        face.load().then((loaded) => {
            document.fonts.add(loaded);
        }).catch((_) => { /* font load failed — CSS @font-face fallback active */ });
    } catch (_) { /* FontFace API unavailable — CSS @font-face handles it */ }

    const style = document.createElement("style");
    style.id = "tex-highlight-styles";
    style.textContent = `
        /* ── Monaspace Neon Font ──────────────────────── */
        @font-face {
            font-family: 'Monaspace Neon';
            src: url('${TEX_FONT_URL}') format('woff2');
            font-weight: 100 900;
            font-style: normal;
            font-display: swap;
        }

        /* ── DOM widget container overrides ────────────── */
        /* In both the canvas renderer and the Vue layer, DOM widget
           containers capture pointer events by default.  We disable
           pointer-events on the outer containers so canvas-rendered
           widgets (params, compile_mode, device) remain interactive,
           then re-enable events only on the .cm-editor element. */
        .tex-editor-container {
            pointer-events: none !important;
            overflow: hidden !important;
        }

        /* ── CM6 Editor Container ──────────────────────── */
        /* The container itself is pointer-events:none so that any area
           of the .dom-widget that extends beyond the visible editor
           (above or below) passes events through to the canvas.
           Only .cm-editor gets pointer-events:auto and fills the
           container so the editor background extends to the bottom. */
        .tex-cm-container {
            width: 100%;
            pointer-events: none !important;
            overflow: hidden !important;
        }
        .tex-cm-container .cm-editor {
            height: 100%;
            pointer-events: auto !important;
        }
        .tex-cm-container .cm-editor,
        .tex-cm-container .cm-editor .cm-content,
        .tex-cm-container .cm-editor .cm-line,
        .tex-cm-container .cm-editor .cm-content * {
            font-family: 'Monaspace Neon', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace !important;
            font-feature-settings: "calt" 1, "liga" 1, "ss01" 1, "ss02" 1, "ss03" 1, "ss06" 1, "cv01" 2 !important;
            font-variation-settings: "wght" 380 !important;
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
        }
        .tex-cm-container .cm-scroller {
            overflow: auto !important;
        }

        /* ── Tooltip z-index boost ── */
        .cm-tooltip {
            z-index: 10000 !important;
        }

        /* ── Context Menu ──────────────────────────────── */
        .tex-context-menu {
            position: fixed;
            z-index: 10001;
            background: #1e1e1e;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 4px 0;
            min-width: 160px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
            font: 12px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .tex-context-menu-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 14px;
            color: #cdd6f4;
            cursor: pointer;
            user-select: none;
        }
        .tex-context-menu-item:hover {
            background: #333348;
        }
        .tex-context-menu-shortcut {
            color: #666;
            font-size: 11px;
            margin-left: 24px;
        }
        .tex-snippet-submenu {
            max-height: 70vh;
            overflow-y: auto;
        }
        .tex-snippet-submenu::-webkit-scrollbar {
            width: 6px;
        }
        .tex-snippet-submenu::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 3px;
        }

        /* ── Snippet Dialog ────────────────────────────── */
        .tex-snippet-dialog-overlay {
            position: fixed;
            inset: 0;
            z-index: 10002;
            background: rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .tex-snippet-dialog {
            background: #1e1e1e;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 16px 20px;
            min-width: 340px;
            max-width: 480px;
            color: #cdd6f4;
            font: 13px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        }
        .tex-snippet-dialog-title {
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .tex-snippet-dialog-hint {
            color: #888;
            font-size: 11px;
            margin-bottom: 10px;
        }
        .tex-snippet-dialog-input {
            width: 100%;
            box-sizing: border-box;
            padding: 6px 10px;
            background: #111;
            border: 1px solid #555;
            border-radius: 4px;
            color: #cdd6f4;
            font-size: 13px;
            outline: none;
        }
        .tex-snippet-dialog-input:focus {
            border-color: #7aa2f7;
        }
        .tex-snippet-dialog-preview {
            margin-top: 10px;
            padding: 8px;
            background: #111;
            border-radius: 4px;
            font: 11px/1.4 'Monaspace Neon', 'Consolas', monospace;
            color: #888;
            white-space: pre;
            max-height: 60px;
            overflow: hidden;
        }
        .tex-snippet-dialog-buttons {
            display: flex;
            justify-content: flex-end;
            gap: 8px;
            margin-top: 14px;
        }
        .tex-snippet-btn {
            padding: 5px 14px;
            border: 1px solid #555;
            border-radius: 4px;
            background: #2a2a3a;
            color: #cdd6f4;
            font-size: 12px;
            cursor: pointer;
        }
        .tex-snippet-btn:hover {
            background: #333348;
        }
        .tex-snippet-btn-save {
            background: #2d4f7c;
            border-color: #7aa2f7;
        }
        .tex-snippet-btn-save:hover {
            background: #3a6299;
        }

        /* ── Manage Snippets ───────────────────────────── */
        .tex-snippet-manage {
            min-width: 400px;
            max-width: 540px;
        }
        .tex-snippet-manage-list {
            max-height: 300px;
            overflow-y: auto;
            margin: 8px 0;
        }
        .tex-snippet-manage-list::-webkit-scrollbar {
            width: 6px;
        }
        .tex-snippet-manage-list::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 3px;
        }
        .tex-snippet-manage-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 5px 8px;
            border-radius: 4px;
        }
        .tex-snippet-manage-row:hover {
            background: #2a2a3a;
        }
        .tex-snippet-manage-name {
            flex: 1;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
            font-size: 12px;
        }
        .tex-snippet-manage-actions {
            display: flex;
            gap: 4px;
            margin-left: 12px;
        }
        .tex-snippet-btn-sm {
            padding: 2px 8px;
            border: 1px solid #555;
            border-radius: 3px;
            background: #2a2a3a;
            color: #aaa;
            font-size: 11px;
            cursor: pointer;
        }
        .tex-snippet-btn-sm:hover {
            background: #333348;
            color: #cdd6f4;
        }
        .tex-snippet-btn-danger:hover {
            background: #5c2020;
            border-color: #e06060;
            color: #f88;
        }
        .tex-snippet-manage-empty {
            padding: 16px 8px;
            color: #666;
            font-size: 12px;
            text-align: center;
        }

        /* ── Help Popup ─────────────────────────────────── */
        .tex-help-popup {
            position: fixed;
            z-index: 10000;
            background: #1e1e1e;
            color: #c8c0b8;
            font: 12px/1.5 'Monaspace Neon', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace;
            font-feature-settings: "calt" 1, "liga" 1, "ss01" 1, "ss02" 1, "ss03" 1, "ss06" 1, "cv01" 2;
            font-variation-settings: "wght" 380;
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 0;
            width: 520px;
            max-width: 90vw;
            max-height: 75vh;
            overflow-y: auto;
            box-shadow: 0 12px 40px rgba(0,0,0,0.5);
        }
        /* Title bar — draggable handle */
        .tex-help-titlebar {
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 4px;
            padding: 6px 8px 0 8px;
            cursor: grab;
            user-select: none;
        }
        .tex-help-titlebar:active { cursor: grabbing; }
        .tex-help-popup-close,
        .tex-help-dock-btn {
            background: none;
            border: none;
            color: #776;
            font-size: 16px;
            cursor: pointer;
            line-height: 1;
            padding: 2px 6px;
            border-radius: 4px;
        }
        .tex-help-popup-close:hover,
        .tex-help-dock-btn:hover {
            color: #ddd;
            background: #333;
        }
        .tex-help-popup-close { font-size: 20px; }

        /* Docked state — right side of viewport */
        .tex-help-docked {
            position: fixed !important;
            right: 0 !important;
            top: 0 !important;
            left: auto !important;
            width: 380px !important;
            max-width: 40vw !important;
            height: 100vh !important;
            max-height: 100vh !important;
            border-radius: 0 !important;
            border-right: none !important;
            border-top: none !important;
            border-bottom: none !important;
            box-shadow: -4px 0 20px rgba(0,0,0,0.4) !important;
        }
        .tex-help-docked .tex-help-titlebar { cursor: default; }

        .tex-help-header {
            padding: 10px 20px 0 20px;
        }
        .tex-help-title {
            font-size: 15px;
            font-weight: 600;
            color: #d4c4a8;
            margin-bottom: 4px;
        }
        .tex-help-links {
            font-size: 11px;
            margin-bottom: 2px;
        }
        .tex-help-links a {
            color: #9a8a70;
            text-decoration: none;
        }
        .tex-help-links a:hover {
            color: #d4c4a8;
            text-decoration: underline;
        }
        .tex-help-link-sep {
            color: #555;
            margin: 0 6px;
        }
        .tex-help-search-wrap {
            padding: 10px 20px 8px 20px;
            position: sticky;
            top: 0;
            background: #1e1e1e;
            z-index: 1;
        }
        .tex-help-search {
            width: 100%;
            box-sizing: border-box;
            background: #282828;
            border: 1px solid #3a3a3a;
            border-radius: 6px;
            color: #c8c0b8;
            font: 12px/1.5 inherit;
            font-family: inherit;
            padding: 7px 10px;
            outline: none;
        }
        .tex-help-search::placeholder {
            color: #6a6050;
        }
        .tex-help-search:focus {
            border-color: #5a5040;
            background: #2c2a26;
        }
        .tex-help-categories {
            padding: 4px 20px 16px 20px;
        }
        .tex-help-category {
            margin-bottom: 4px;
        }
        .tex-help-cat-header {
            padding: 7px 8px;
            cursor: pointer;
            color: #c0a080;
            font-size: 12px;
            font-weight: 600;
            border-radius: 5px;
            user-select: none;
        }
        .tex-help-cat-header:hover {
            background: #262420;
        }
        .tex-help-cat-arrow {
            display: inline-block;
            width: 14px;
            font-size: 10px;
            color: #776;
        }
        .tex-help-cat-icon {
            margin-right: 4px;
        }
        .tex-help-entries {
            padding: 2px 0 4px 10px;
        }
        .tex-help-card {
            background: #252320;
            border: 1px solid #333028;
            border-radius: 6px;
            padding: 8px 10px;
            margin: 4px 0;
        }
        .tex-help-card:hover {
            border-color: #4a4535;
        }
        .tex-help-card-name {
            font-size: 12px;
            font-weight: 600;
            color: #d4c4a8;
            margin-bottom: 2px;
        }
        .tex-help-card-sig code {
            background: #2a2826;
            color: #c09060;
            padding: 1px 5px;
            border-radius: 3px;
            font-size: 11px;
            font-family: inherit;
        }
        .tex-help-card-desc {
            color: #958a78;
            font-size: 11px;
            margin: 3px 0;
            line-height: 1.4;
        }
        .tex-help-card-example {
            background: #1a1918;
            color: #8a8070;
            padding: 5px 8px;
            border-radius: 4px;
            margin: 4px 0 0 0;
            font-size: 10.5px;
            line-height: 1.4;
            font-family: inherit;
            white-space: pre-wrap;
            word-break: break-all;
            border: 1px solid #2a2825;
        }
        .tex-help-popup::-webkit-scrollbar { width: 6px; }
        .tex-help-popup::-webkit-scrollbar-track { background: #1e1e1e; }
        .tex-help-popup::-webkit-scrollbar-thumb {
            background: #4a4030;
            border-radius: 3px;
        }
        .tex-help-popup::-webkit-scrollbar-thumb:hover {
            background: #6a5840;
        }

        /* ── Floating Error Banner (above node, on document.body) ── */
        .tex-floating-error {
            position: fixed;
            z-index: 9999;
            background: rgba(30, 20, 20, 0.95);
            border: 1px solid #663333;
            border-radius: 8px;
            padding: 8px 32px 8px 10px;
            font: 11px/1.4 'Monaspace Neon', 'Cascadia Code', 'Fira Code', 'JetBrains Mono', 'Consolas', monospace;
            color: #f0a0a0;
            box-shadow: 0 4px 16px rgba(0,0,0,0.5);
            pointer-events: auto;
            max-height: 300px;
            overflow-y: auto;
            box-sizing: border-box;
        }
        .tex-floating-error hr {
            border: none;
            border-top: 1px solid #443333;
            margin: 6px 0;
        }
        .tex-floating-error::-webkit-scrollbar { width: 4px; }
        .tex-floating-error::-webkit-scrollbar-track { background: transparent; }
        .tex-floating-error::-webkit-scrollbar-thumb { background: #553333; border-radius: 2px; }

        /* ── Floating Help "?" Button (top-right of node, on document.body) ── */
        /* C1-ux — perf HUD DOM badge (dual-path with the canvas draw) */
        .tex-floating-perf-badge {
            position: fixed;
            z-index: 9998;
            font: 9px 'Cascadia Code', 'Consolas', monospace;
            background: rgba(20, 20, 20, 0.82);
            padding: 2px 5px;
            border-radius: 3px;
            pointer-events: auto;
            white-space: pre;
        }
        /* C2-ux — doctor modal rows (reuses the snippet dialog chrome) */
        .tex-doctor-body { max-height: 60vh; overflow-y: auto; font: 12px 'Cascadia Code', monospace; }
        .tex-doctor-row { display: flex; justify-content: space-between; gap: 16px;
            padding: 2px 0; border-bottom: 1px solid rgba(255,255,255,0.06); }
        .tex-doctor-key { color: #9E9E9E; }
        .tex-doctor-caveat { color: #FFB300; background: rgba(255,179,0,0.1);
            padding: 6px 8px; border-radius: 4px; margin-bottom: 8px; }
        .tex-floating-help-btn {
            position: fixed;
            z-index: 9999;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 1.5px solid #4FC3F7;
            background: rgba(30, 30, 30, 0.85);
            color: #4FC3F7;
            font: bold 12px sans-serif;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
            padding: 0;
            opacity: 0.6;
            transition: opacity 0.15s;
            pointer-events: auto;
        }
        .tex-floating-help-btn:hover {
            opacity: 1;
            background: rgba(79, 195, 247, 0.15);
        }
    `;
    document.head.appendChild(style);
})();

