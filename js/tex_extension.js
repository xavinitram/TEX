/**
 * TEX Wrangle — ComfyUI frontend extension (v2.1 — CodeMirror 6)
 *
 * Features:
 *   1. True auto-socket creation: dynamically adds/removes LiteGraph input
 *      slots based on @ bindings found in the TEX code.
 *   2. CodeMirror 6 editor with syntax highlighting, autocomplete, and
 *      bracket matching (replaces the old overlay-based textarea).
 *   3. Inline error diagnostics: red squiggly underlines at the exact
 *      line/column where errors occur, plus gutter markers.
 *   4. Error overlay above the node title bar (kept as visual backup).
 *   5. "?" help popup and "TEX" badge in the title bar.
 *
 * Implementation notes:
 *   - The CM6 bundle is pre-built by Rollup into tex_cm6_bundle.js and
 *     exposed as globalThis.TEX_CM6. This file accesses that API lazily.
 *   - The original code widget is suppressed (zero height). A CM6 EditorView
 *     is mounted in a container div added via node.addDOMWidget().
 *   - Code changes in the editor are synced back to codeWidget.value so
 *     ComfyUI's serialization/deserialization works unchanged.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TEX_NODE_TYPE = "TEX_Wrangle";
// Resolve font URL at module scope where import.meta.url is reliable
const TEX_FONT_URL = (() => {
    try {
        return new URL("MonaspaceNeon.woff2", import.meta.url).href;
    } catch (_) {
        return "/extensions/TEX_Wrangle/MonaspaceNeon.woff2";
    }
})();
const BINDING_REGEX = /@([A-Za-z_][A-Za-z0-9_]*)/g;
// Names reserved by the system — NOT treated as TEX bindings
const RESERVED_NAMES = new Set(["OUT", "code", "output_type", "device", "compile_mode"]);
const DEBOUNCE_MS = 400;

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

const texEditors = new WeakMap();  // node → EditorView

// ─── Binding Parser ──────────────────────────────────────────────────

function parseBindings(code) {
    const stripped = code
        .replace(/\/\/[^\n]*/g, "")
        .replace(/\/\*[\s\S]*?\*\//g, "");
    const bindings = new Set();
    let match;
    BINDING_REGEX.lastIndex = 0;
    while ((match = BINDING_REGEX.exec(stripped)) !== null) {
        const name = match[1];
        if (!RESERVED_NAMES.has(name)) {
            bindings.add(name);
        }
    }
    return bindings;
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

function syncInputs(node, usedBindings) {
    if (!node.graph) return;
    const ANY_TYPE = "*";

    // Build map of current TEX binding inputs (all inputs are bindings)
    const currentInputs = new Map();
    if (node.inputs) {
        for (let i = 0; i < node.inputs.length; i++) {
            currentInputs.set(node.inputs[i].name, i);
        }
    }

    // Desired set — sorted for stable ordering
    const desired = [...usedBindings].sort();

    // Remove unneeded inputs (reverse order for index stability)
    const toRemove = [];
    for (const [name, idx] of currentInputs) {
        if (!usedBindings.has(name)) toRemove.push(idx);
    }
    toRemove.sort((a, b) => b - a);
    for (const idx of toRemove) node.removeInput(idx);

    // Rebuild map after removals
    const afterRemove = new Set(node.inputs ? node.inputs.map(i => i.name) : []);

    // Add missing inputs
    for (const name of desired) {
        if (!afterRemove.has(name)) node.addInput(name, ANY_TYPE);
    }

    node.setDirtyCanvas(true, true);
}

// ─── Error Cache (per-node, from WebSocket events) ───────────────────

const texErrorCache = new Map(); // nodeId -> { message, type, traceback }

api.addEventListener("execution_error", (ev) => {
    const d = ev.detail;
    if (!d?.node_id) return;

    const errData = {
        message: d.exception_message || "Unknown error",
        type: d.exception_type || "",
        traceback: d.traceback || "",
    };

    texErrorCache.set(String(d.node_id), errData);

    // Push inline diagnostics to the CM6 editor if available
    const CM6 = getCM6();
    if (CM6 && app.graph?._nodes) {
        for (const node of app.graph._nodes) {
            if (String(node.id) === String(d.node_id) && node.type === TEX_NODE_TYPE) {
                const editor = texEditors.get(node);
                if (editor) {
                    try {
                        const diagnostics = CM6.texErrorToDiagnostics(editor, errData);
                        editor.dispatch(CM6.setDiagnostics(editor.state, diagnostics));
                    } catch (err) {
                        console.warn("[TEX] Failed to push error diagnostics:", err);
                    }
                }
                break;
            }
        }
    }
});

api.addEventListener("execution_start", () => {
    // Clear errors at the start of a new prompt
    texErrorCache.clear();

    // Clear inline diagnostics from all CM6 editors
    const CM6 = getCM6();
    if (CM6 && app.graph?._nodes) {
        for (const node of app.graph._nodes) {
            if (node.type === TEX_NODE_TYPE) {
                const editor = texEditors.get(node);
                if (editor) {
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

function createTexEditor(node, codeWidget) {
    const CM6 = getCM6();
    if (!CM6) {
        console.warn("[TEX] CM6 bundle not available — falling back to default textarea");
        return null;
    }

    const container = document.createElement("div");
    container.className = "tex-cm-container";

    // Auto-socket updater
    const updateSockets = debounce(() => {
        const code = codeWidget.value || "";
        const usedBindings = parseBindings(code);
        node._texBindings = usedBindings;
        syncInputs(node, usedBindings);
    }, DEBOUNCE_MS);

    // Dynamic @ binding completions — reads from node's current input sockets
    function getBindings() {
        const names = [];
        if (node.inputs) {
            for (const inp of node.inputs) {
                if (inp.name && !RESERVED_NAMES.has(inp.name)) {
                    names.push(inp.name);
                }
            }
        }
        // Also parse code for potential bindings not yet synced
        const code = codeWidget.value || "";
        const parsed = parseBindings(code);
        for (const name of parsed) {
            if (!names.includes(name)) names.push(name);
        }
        return names;
    }

    // Build the editor
    try {
        const completionSource = CM6.createTexCompletions(getBindings);

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
                    // Autocomplete
                    CM6.autocompletion({
                        override: [completionSource],
                        activateOnTyping: true,
                    }),
                    // Tooltip positioning: use absolute to avoid the
                    // position:fixed-inside-transform:scale() bug in
                    // ComfyUI Desktop's widget containers.
                    CM6.tooltips({
                        position: "absolute",
                    }),
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
                                        if (/[@\w]/.test(charBefore)) {
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

        // Store container and editor references on node for external access
        node._texEditorContainer = container;
        node._texEditor = editor;

        // Initial socket scan
        setTimeout(updateSockets, 200);

        return { container, editor };

    } catch (err) {
        console.error("[TEX] Failed to create CM6 editor:", err);
        return null;
    }
}

// ─── Context Menu ───────────────────────────────────────────────────

let activeContextMenu = null;

function closeContextMenu() {
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
    ];

    for (const item of items) {
        const row = document.createElement("div");
        row.className = "tex-context-menu-item";

        const lbl = document.createElement("span");
        lbl.textContent = item.label;
        row.appendChild(lbl);

        const key = document.createElement("span");
        key.className = "tex-context-menu-shortcut";
        key.textContent = item.shortcut;
        row.appendChild(key);

        row.addEventListener("click", (e) => {
            e.stopPropagation();
            closeContextMenu();
            editorView.focus();
            item.action();
        });
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
    const dismiss = (e) => {
        if (!menu.contains(e.target)) {
            closeContextMenu();
            document.removeEventListener("mousedown", dismiss, true);
            document.removeEventListener("keydown", dismissKey, true);
        }
    };
    const dismissKey = (e) => {
        if (e.key === "Escape") {
            closeContextMenu();
            document.removeEventListener("mousedown", dismiss, true);
            document.removeEventListener("keydown", dismissKey, true);
        }
    };
    setTimeout(() => {
        document.addEventListener("mousedown", dismiss, true);
        document.addEventListener("keydown", dismissKey, true);
    }, 0);
}

// ─── Help Popup Content ─────────────────────────────────────────────

const TEX_HELP_HTML = `
<h3>TEX Quick Reference</h3>

<p><b>Types:</b> <code>float</code> <code>int</code> <code>vec3</code> <code>vec4</code> <code>mat3</code> <code>mat4</code> <code>string</code></p>

<p><b>Bindings:</b> <code>@name</code> inputs (e.g. <code>@A</code>, <code>@base_image</code>), <code>@OUT</code> output</p>

<p><b>Built-in Vars:</b>
<code>ix</code> <code>iy</code> pixel coords &nbsp;
<code>u</code> <code>v</code> normalized [0,1] &nbsp;
<code>iw</code> <code>ih</code> dimensions &nbsp;
<code>fi</code> frame index &nbsp; <code>fn</code> frame count &nbsp;
<code>ic</code> latent channels &nbsp;
<code>PI</code> <code>E</code></p>

<p><b>Channel Access:</b>
<code>.r .g .b .a</code> &nbsp; <code>.x .y .z .w</code> &nbsp;
swizzle: <code>.rgb</code> <code>.bgra</code></p>

<p><b>Operators:</b>
<code>+ - * / %</code> &nbsp; <code>== != < > <= >=</code> &nbsp;
<code>&amp;&amp; || !</code> &nbsp; <code>? :</code><br>
Compound: <code>+= -= *= /= ++ --</code></p>

<p><b>Control Flow:</b><br>
<code>if (cond) { ... } else { ... }</code> — vectorized via torch.where<br>
<code>for (int i = 0; i < n; i++) { ... }</code> — bounded loops</p>

<p><b>Math:</b> <code>sin cos tan asin acos atan atan2</code>
<code>sinh cosh tanh</code>
<code>pow sqrt exp log log2 log10 abs sign</code>
<code>pow2 pow10 hypot</code>
<code>floor ceil round fract mod</code>
<code>degrees radians</code></p>

<p><b>Safe Ops:</b> <code>spow(x,y)</code> — sign-safe power
<code>sdiv(a,b)</code> — divide-by-zero safe</p>

<p><b>Classification:</b> <code>isnan(x)</code> <code>isinf(x)</code> → 0.0/1.0</p>

<p><b>Interpolation:</b> <code>min max clamp lerp mix fit step smoothstep</code></p>

<p><b>Vector:</b> <code>dot length distance normalize cross reflect</code></p>

<p><b>Color:</b> <code>luma hsv2rgb rgb2hsv</code></p>

<p><b>Sampling:</b><br>
<code>sample(@A, u, v)</code> — bilinear, UV coords<br>
<code>fetch(@A, px, py)</code> — nearest, pixel coords<br>
<code>sample_cubic(@A, u, v)</code> — bicubic (Catmull-Rom)<br>
<code>sample_lanczos(@A, u, v)</code> — Lanczos-3<br>
<code>fetch_frame(@A, frame, px, py)</code> — nearest from specific frame<br>
<code>sample_frame(@A, frame, u, v)</code> — bilinear from specific frame</p>

<p><b>Noise:</b><br>
<code>perlin(x, y)</code> — 2D Perlin noise [-1, 1]<br>
<code>simplex(x, y)</code> — 2D Simplex noise [-1, 1]<br>
<code>fbm(x, y, octaves)</code> — FBM (multi-octave Perlin)</p>

<p><b>Latent Support:</b><br>
Connect latent data to @A. Output auto-detects as LATENT when inputs are latent.<br>
Values are NOT clamped — latent range is typically [-4, 4].<br>
<code>ic</code> = channel count (4 for SD1.5/SDXL, 16 for SD3).</p>

<p><b>String:</b><br>
<code>str(x)</code> — number to string<br>
<code>len(s)</code> — length &nbsp; <code>replace(s, old, new)</code><br>
<code>strip(s)</code> &nbsp; <code>lower(s)</code> / <code>upper(s)</code><br>
<code>contains(s, sub)</code> — 1.0/0.0<br>
<code>startswith(s, pre)</code> / <code>endswith(s, suf)</code><br>
<code>find(s, sub)</code> — index or -1.0<br>
<code>substr(s, start, len?)</code> — extract<br>
<code>to_int(s)</code> / <code>to_float(s)</code> — parse<br>
<code>sanitize_filename(s)</code> — clean path<br>
Literals: <code>"hello"</code>. Concat: <code>"a" + "b"</code>. String output auto-detected when @OUT is a string.</p>

<p><b>Arrays:</b><br>
<code>float arr[5];</code> <code>int arr[3];</code> <code>vec3 arr[4];</code> <code>vec4 arr[9];</code> <code>string arr[3];</code><br>
<code>sort(arr)</code> <code>reverse(arr)</code> — reorder<br>
<code>arr_sum</code> <code>arr_min</code> <code>arr_max</code> <code>median</code> <code>arr_avg</code> — aggregate (numeric arrays)<br>
<code>join(arr, sep)</code> — concatenate string array with separator<br>
<code>len(arr)</code> — array length. Max size: 1024.</p>

<p><b>Image Reductions:</b><br>
<code>img_min(@A)</code> <code>img_max(@A)</code> — per-channel min/max<br>
<code>img_mean(@A)</code> <code>img_sum(@A)</code> <code>img_median(@A)</code> — per-channel stats<br>
Results broadcast with per-pixel expressions.<br>
Auto-levels: <code>@OUT = (@A - img_min(@A)) / max(img_max(@A) - img_min(@A), 0.001);</code></p>

<p><b>Matrix:</b><br>
<code>mat3</code> (3×3) / <code>mat4</code> (4×4) — internal computation only (cannot assign to @OUT)<br>
<code>mat3(1.0)</code> — scaled identity &nbsp; <code>mat3(a,b,...,i)</code> — 9 values row-major<br>
<code>mat * vec</code> — matrix-vector multiply &nbsp; <code>mat * mat</code> — matrix multiply<br>
<code>scalar * mat</code> — element-wise scale &nbsp; <code>mat + mat</code> / <code>mat - mat</code><br>
<code>transpose(m)</code> &nbsp; <code>determinant(m)</code> &nbsp; <code>inverse(m)</code></p>

<p><b>Batch / Temporal:</b><br>
<code>fi</code> = frame index (0 to B-1), <code>fn</code> = total frame count<br>
<code>fetch_frame(@A, frame, px, py)</code> — read from any frame<br>
<code>sample_frame(@A, frame, u, v)</code> — bilinear from any frame<br>
Fade: <code>@OUT = @A * (fi / max(fn-1, 1))</code><br>
Blend: <code>@OUT = lerp(fetch_frame(@A, fi-1, ix, iy), fetch_frame(@A, fi+1, ix, iy), 0.5)</code></p>

<p style="margin-top:10px; padding-top:8px; border-top:1px solid #333;"><b>Example:</b></p>
<pre style="background:#2a2a3e; padding:8px; border-radius:4px; margin:4px 0; font-size:11px; line-height:1.4;">float gray = luma(@A);
@OUT = vec3(gray, gray, gray);</pre>

<p style="margin-top:10px; padding-top:8px; border-top:1px solid #333; text-align:center; font-size:11px;">
<a href="https://github.com/xavinitram/TEX/issues" target="_blank" style="color:#4FC3F7; text-decoration:none;">Report a bug or request a feature</a></p>
`;

// ─── Help Popup Show/Hide ────────────────────────────────────────────

let activeHelpPopup = null;
let outsideClickHandler = null;

function showHelpPopup(screenX, screenY) {
    // Close any existing popup first
    hideHelpPopup();

    const popup = document.createElement("div");
    popup.className = "tex-help-popup";
    popup.innerHTML = TEX_HELP_HTML;

    // Close button
    const closeBtn = document.createElement("button");
    closeBtn.className = "tex-help-popup-close";
    closeBtn.textContent = "\u00d7";  // × character
    closeBtn.addEventListener("mousedown", (e) => {
        e.stopPropagation();
        hideHelpPopup();
    });
    popup.prepend(closeBtn);

    document.body.appendChild(popup);
    activeHelpPopup = popup;

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

    // Outside-click dismissal (delayed to avoid catching the opening click)
    setTimeout(() => {
        outsideClickHandler = (e) => {
            if (activeHelpPopup && !activeHelpPopup.contains(e.target)) {
                hideHelpPopup();
            }
        };
        document.addEventListener("mousedown", outsideClickHandler);
    }, 100);
}

function hideHelpPopup() {
    if (activeHelpPopup) {
        activeHelpPopup.remove();
        activeHelpPopup = null;
    }
    if (outsideClickHandler) {
        document.removeEventListener("mousedown", outsideClickHandler);
        outsideClickHandler = null;
    }
}

// ─── Extension Registration ──────────────────────────────────────────

app.registerExtension({
    name: "TEX.Wrangle",

    async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
        if (nodeData.name !== TEX_NODE_TYPE) return;

        // ── onNodeCreated ──
        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);

            const node = this;
            const codeWidget = node.widgets?.find(w => w.name === "code");
            if (!codeWidget) return;

            // Remove any initial inputs from the backend schema.
            // We manage sockets dynamically based on code content.
            if (node.inputs) {
                for (let i = node.inputs.length - 1; i >= 0; i--) {
                    node.removeInput(i);
                }
            }

            // Auto-socket updater (shared by both CM6 and fallback paths)
            const updateSockets = debounce(() => {
                const code = codeWidget.value || "";
                const usedBindings = parseBindings(code);
                node._texBindings = usedBindings;
                syncInputs(node, usedBindings);
            }, DEBOUNCE_MS);

            // ── Try CM6 Editor Integration ──
            const result = createTexEditor(node, codeWidget);

            if (result) {
                try {
                    const { container, editor } = result;

                    // Suppress the original code widget so it takes zero layout space.
                    // We keep it alive for serialization but make it invisible.
                    // Method 1: Override computeSize to return zero height
                    const origComputeSize = codeWidget.computeSize;
                    codeWidget.computeSize = function () {
                        return [0, -4];  // -4 absorbs ComfyUI's inter-widget spacing
                    };

                    // Method 2: Also hide the DOM element when it appears
                    const hideOriginal = () => {
                        const el = codeWidget.element || codeWidget.inputEl;
                        if (el) {
                            el.style.display = "none";
                            el.style.height = "0";
                            el.style.overflow = "hidden";
                            return true;
                        }
                        return false;
                    };

                    if (!hideOriginal()) {
                        const checkInterval = setInterval(() => {
                            if (hideOriginal()) clearInterval(checkInterval);
                        }, 200);
                        setTimeout(() => clearInterval(checkInterval), 10000);
                    }

                    // Add the CM6 editor as a DOM widget.
                    // serialize: false — the original "code" widget handles save/load.
                    const domWidget = node.addDOMWidget("tex_editor", "customwidget", container, {
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
                    });
                    if (domWidget) domWidget.serialize = false;

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
                    const usedBindings = parseBindings(code);
                    node._texBindings = usedBindings;
                    syncInputs(node, usedBindings);

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
            if (this.flags?.collapsed) return;

            // "TEX" badge
            ctx.save();
            ctx.font = "bold 10px monospace";
            ctx.fillStyle = "#4FC3F7";
            ctx.textAlign = "right";
            ctx.fillText("TEX", this.size[0] - 8, -6);
            ctx.restore();

            // Help "?" icon — circular button left of the "TEX" badge
            const helpR = 7;
            const helpCX = this.size[0] - 42;
            const helpCY = -LiteGraph.NODE_TITLE_HEIGHT * 0.5;
            ctx.save();
            ctx.strokeStyle = "#4FC3F7";
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(helpCX, helpCY, helpR, 0, Math.PI * 2);
            ctx.stroke();
            ctx.fillStyle = "#4FC3F7";
            ctx.font = "bold 11px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("?", helpCX, helpCY + 0.5);
            ctx.restore();
            // Save hit-test rect for onMouseDown
            this._texHelpBtn = [helpCX - helpR, helpCY - helpR, helpR * 2, helpR * 2];

            // Inline error display from execution_error events
            // Rendered ABOVE the node title bar so it doesn't overlap the editor
            const errData = texErrorCache.get(String(this.id));
            if (errData) {
                const errText = errData.message;
                const padding = 6;
                const lineHeight = 14;
                const maxWidth = this.size[0] - 16;

                ctx.save();
                ctx.font = "11px monospace";

                const lines = wrapText(ctx, errText, maxWidth);
                const boxH = lines.length * lineHeight + padding * 2;
                // Position above the node title (negative Y = above the node)
                const boxY = -boxH - 8;

                // Red background
                ctx.fillStyle = "rgba(180, 40, 40, 0.95)";
                ctx.beginPath();
                if (ctx.roundRect) {
                    ctx.roundRect(4, boxY, this.size[0] - 8, boxH, 4);
                } else {
                    ctx.rect(4, boxY, this.size[0] - 8, boxH);
                }
                ctx.fill();

                // Subtle border
                ctx.strokeStyle = "rgba(255, 100, 100, 0.6)";
                ctx.lineWidth = 1;
                ctx.stroke();

                // Error text
                ctx.fillStyle = "#ffcccc";
                ctx.textAlign = "left";
                for (let i = 0; i < lines.length; i++) {
                    ctx.fillText(lines[i], 8, boxY + padding + (i + 1) * lineHeight - 2);
                }
                ctx.restore();
            }
        };

        // ── Help button click handler ──
        const origMouseDown = nodeType.prototype.onMouseDown;
        nodeType.prototype.onMouseDown = function (e, localPos, graphCanvas) {
            if (this._texHelpBtn && !this.flags?.collapsed) {
                const [bx, by, bw, bh] = this._texHelpBtn;
                if (localPos[0] >= bx && localPos[0] <= bx + bw &&
                    localPos[1] >= by && localPos[1] <= by + bh) {
                    showHelpPopup(e.clientX || e.canvasX || 200, e.clientY || e.canvasY || 200);
                    return true;
                }
            }
            if (origMouseDown) return origMouseDown.apply(this, arguments);
        };

        // Clear error after successful execution
        const origExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            if (origExecuted) origExecuted.apply(this, arguments);
            texErrorCache.delete(String(this.id));
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

        /* ── CM6 Editor Container ──────────────────────── */
        .tex-cm-container {
            width: 100%;
        }
        .tex-cm-container .cm-editor {
            height: 100%;
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
            background: #1e1e2e;
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

        /* ── Help Popup ─────────────────────────────────── */
        .tex-help-popup {
            position: fixed;
            z-index: 10000;
            background: #1e1e2e;
            color: #cdd6f4;
            font: 12px/1.5 monospace;
            border: 2px solid #4FC3F7;
            border-radius: 8px;
            padding: 16px 18px;
            max-width: 480px;
            max-height: 70vh;
            overflow-y: auto;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
        }
        .tex-help-popup h3 {
            color: #4FC3F7;
            margin: 0 0 10px 0;
            font-size: 14px;
            font-weight: bold;
        }
        .tex-help-popup p {
            margin: 4px 0;
            line-height: 1.6;
        }
        .tex-help-popup code {
            background: #2a2a3e;
            color: #f78c6c;
            padding: 1px 4px;
            border-radius: 3px;
            font-size: 11px;
        }
        .tex-help-popup b {
            color: #c792ea;
        }
        .tex-help-popup-close {
            position: absolute;
            top: 6px;
            right: 10px;
            background: none;
            border: none;
            color: #888;
            font-size: 18px;
            cursor: pointer;
            line-height: 1;
            padding: 2px 4px;
        }
        .tex-help-popup-close:hover {
            color: #fff;
        }
        .tex-help-popup::-webkit-scrollbar { width: 6px; }
        .tex-help-popup::-webkit-scrollbar-track { background: #1e1e2e; }
        .tex-help-popup::-webkit-scrollbar-thumb {
            background: #4FC3F7;
            border-radius: 3px;
        }
    `;
    document.head.appendChild(style);
})();

// ─── Utility: text wrapping for canvas ───────────────────────────────

function wrapText(ctx, text, maxWidth) {
    const words = text.split(/\s+/);
    const lines = [];
    let line = "";
    for (const word of words) {
        const test = line ? line + " " + word : word;
        if (ctx.measureText(test).width > maxWidth && line) {
            lines.push(line);
            line = word;
        } else {
            line = test;
        }
    }
    if (line) lines.push(line);
    return lines.length ? lines : [""];
}
