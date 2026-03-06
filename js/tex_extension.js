/**
 * TEX Wrangle — ComfyUI frontend extension (v1.1)
 *
 * Features:
 *   1. True auto-socket creation: dynamically adds/removes LiteGraph input
 *      slots based on @ bindings found in the TEX code.
 *   2. Syntax highlighting overlay behind the code textarea.
 *   3. Inline error display when execution fails.
 *   4. Custom styling and keyboard handling for the code editor.
 *
 * Implementation notes:
 *   - Syntax highlighting uses the "sibling overlay" pattern: a <div> is
 *     inserted before the <textarea> in the DOM, positioned identically via
 *     getComputedStyle(), and the textarea gets transparent background.
 *   - Textareas are discovered via DOM polling (MutationObserver) since
 *     ComfyUI creates them lazily and the widget.element / widget.inputEl
 *     property may not be available at onNodeCreated time.
 *   - Execution errors are captured via the ComfyUI api WebSocket
 *     "execution_error" event and rendered in onDrawForeground.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TEX_NODE_TYPE = "TEX_Wrangle";
const BINDING_REGEX = /@([A-Za-z_][A-Za-z0-9_]*)/g;
// Names reserved by the system — NOT treated as TEX bindings
const RESERVED_NAMES = new Set(["OUT", "code", "output_type", "device", "compile_mode"]);
const DEBOUNCE_MS = 400;

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

// ─── Syntax Highlighting ─────────────────────────────────────────────

const TEX_KEYWORDS = new Set([
    "float", "int", "vec3", "vec4", "string", "if", "else", "for",
]);
const TEX_BUILTINS = new Set([
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
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
    "str", "len", "replace", "strip", "lower", "upper",
    "contains", "startswith", "endswith", "find", "substr",
    "to_int", "to_float", "sanitize_filename",
    "sort", "reverse", "arr_sum", "arr_min", "arr_max", "median", "arr_avg",
    "join",
    "img_sum", "img_mean", "img_min", "img_max", "img_median",
]);
const TEX_CONSTANTS = new Set(["PI", "E"]);
const TEX_COORD_VARS = new Set(["u", "v", "ix", "iy", "iw", "ih", "ic", "fi", "fn"]);

// ─── Help Popup Content ─────────────────────────────────────────────

const TEX_HELP_HTML = `
<h3>TEX Quick Reference</h3>

<p><b>Types:</b> <code>float</code> <code>int</code> <code>vec3</code> <code>vec4</code> <code>string</code></p>

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

<p><b>Batch / Temporal:</b><br>
<code>fi</code> = frame index (0 to B-1), <code>fn</code> = total frame count<br>
<code>fetch_frame(@A, frame, px, py)</code> — read from any frame<br>
<code>sample_frame(@A, frame, u, v)</code> — bilinear from any frame<br>
Fade: <code>@OUT = @A * (fi / max(fn-1, 1))</code><br>
Blend: <code>@OUT = lerp(fetch_frame(@A, fi-1, ix, iy), fetch_frame(@A, fi+1, ix, iy), 0.5)</code></p>

<p style="margin-top:10px; padding-top:8px; border-top:1px solid #333;"><b>Example:</b></p>
<pre style="background:#2a2a3e; padding:8px; border-radius:4px; margin:4px 0; font-size:11px; line-height:1.4;">float gray = luma(@A);
@OUT = vec3(gray, gray, gray);</pre>
`;

function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/**
 * Produce syntax-highlighted HTML from TEX source code.
 */
function highlightTEX(code) {
    let html = "";
    let i = 0;
    const len = code.length;

    while (i < len) {
        // Line comments
        if (code[i] === "/" && code[i + 1] === "/") {
            let end = code.indexOf("\n", i);
            if (end === -1) end = len;
            html += `<span class="tex-hl-comment">${escapeHtml(code.slice(i, end))}</span>`;
            i = end;
            continue;
        }
        // Block comments
        if (code[i] === "/" && code[i + 1] === "*") {
            let end = code.indexOf("*/", i + 2);
            if (end === -1) end = len; else end += 2;
            html += `<span class="tex-hl-comment">${escapeHtml(code.slice(i, end))}</span>`;
            i = end;
            continue;
        }
        // String literals
        if (code[i] === '"') {
            let j = i + 1;
            while (j < len && code[j] !== '"') {
                if (code[j] === '\\' && j + 1 < len) j++; // skip escaped char
                j++;
            }
            if (j < len) j++; // include closing quote
            html += `<span class="tex-hl-string">${escapeHtml(code.slice(i, j))}</span>`;
            i = j;
            continue;
        }
        // @ bindings
        if (code[i] === "@") {
            let j = i + 1;
            while (j < len && /[A-Za-z0-9_]/.test(code[j])) j++;
            html += `<span class="tex-hl-binding">${escapeHtml(code.slice(i, j))}</span>`;
            i = j;
            continue;
        }
        // Numbers
        if (/[0-9]/.test(code[i]) || (code[i] === "." && i + 1 < len && /[0-9]/.test(code[i + 1]))) {
            let j = i;
            if (code[j] === "0" && (code[j + 1] === "x" || code[j + 1] === "X")) {
                j += 2;
                while (j < len && /[0-9a-fA-F]/.test(code[j])) j++;
            } else {
                while (j < len && /[0-9]/.test(code[j])) j++;
                if (j < len && code[j] === ".") { j++; while (j < len && /[0-9]/.test(code[j])) j++; }
                if (j < len && (code[j] === "e" || code[j] === "E")) {
                    j++;
                    if (j < len && (code[j] === "+" || code[j] === "-")) j++;
                    while (j < len && /[0-9]/.test(code[j])) j++;
                }
            }
            html += `<span class="tex-hl-number">${escapeHtml(code.slice(i, j))}</span>`;
            i = j;
            continue;
        }
        // Identifiers / keywords
        if (/[A-Za-z_]/.test(code[i])) {
            let j = i;
            while (j < len && /[A-Za-z0-9_]/.test(code[j])) j++;
            const word = code.slice(i, j);
            if (TEX_KEYWORDS.has(word)) {
                html += `<span class="tex-hl-keyword">${escapeHtml(word)}</span>`;
            } else if (TEX_BUILTINS.has(word)) {
                html += `<span class="tex-hl-builtin">${escapeHtml(word)}</span>`;
            } else if (TEX_CONSTANTS.has(word)) {
                html += `<span class="tex-hl-constant">${escapeHtml(word)}</span>`;
            } else if (TEX_COORD_VARS.has(word)) {
                html += `<span class="tex-hl-coordvar">${escapeHtml(word)}</span>`;
            } else {
                html += escapeHtml(word);
            }
            i = j;
            continue;
        }
        // Everything else (whitespace, operators, braces, etc.)
        html += escapeHtml(code[i]);
        i++;
    }
    // Trailing newline ensures overlay matches textarea height
    if (!html.endsWith("\n")) html += "\n";
    return html;
}

// ─── Overlay Management (follows ComfyUI-Syntax-Highlighting pattern) ─

const enhancedTextareas = new WeakSet();

/**
 * Copy computed position/size from textarea to overlay each frame.
 */
function syncOverlayPosition(textarea, overlay) {
    if (!textarea || !overlay || !document.contains(textarea)) return;
    const cs = window.getComputedStyle(textarea);
    requestAnimationFrame(() => {
        const props = {
            left: cs.left,
            top: cs.top,
            width: cs.width,
            height: cs.height,
            display: cs.display,
            transform: cs.transform,
            transformOrigin: cs.transformOrigin,
        };
        for (const [k, v] of Object.entries(props)) {
            if (overlay.style[k] !== v) overlay.style[k] = v;
        }
    });
}

/**
 * Copy font metrics & box-model from textarea to overlay.
 */
function syncOverlayStyle(textarea, overlay) {
    const cs = window.getComputedStyle(textarea);
    Object.assign(overlay.style, {
        position: "absolute",
        fontFamily: cs.fontFamily,
        fontSize: cs.fontSize,
        fontWeight: cs.fontWeight,
        lineHeight: cs.lineHeight,
        letterSpacing: cs.letterSpacing,
        padding: cs.padding,
        boxSizing: cs.boxSizing,
        // Clip overlay content to its bounds — the textarea handles scroll UI,
        // and we sync scroll position programmatically via scrollTop/scrollLeft.
        overflow: "hidden",
        whiteSpace: cs.whiteSpace || "pre",
        wordWrap: cs.wordWrap || "break-word",
        tabSize: cs.tabSize || "4",
        zIndex: "1",
        pointerEvents: "none",
        color: "#d4d4d4",
        backgroundColor: "var(--comfy-input-bg, #1a1a2e)",
        borderRadius: cs.borderRadius,
        border: "1px solid transparent",
    });
}

/**
 * Sync highlighted HTML content from textarea value.
 */
function syncOverlayContent(textarea, overlay) {
    overlay.innerHTML = highlightTEX(textarea.value || "");
    overlay.scrollTop = textarea.scrollTop;
    overlay.scrollLeft = textarea.scrollLeft;
}

/**
 * Enhance a TEX textarea with syntax highlighting overlay.
 */
function enhanceTexTextarea(textarea) {
    if (enhancedTextareas.has(textarea)) return;
    enhancedTextareas.add(textarea);

    // Create the overlay div
    const overlay = document.createElement("div");
    overlay.className = "tex-highlight-overlay";
    textarea.parentNode.insertBefore(overlay, textarea);

    // Make textarea transparent so overlay shows through
    textarea.style.background = "transparent";
    textarea.style.position = "relative";
    textarea.style.zIndex = "2";
    textarea.style.color = "transparent";
    textarea.style.caretColor = "#fff";
    textarea.spellcheck = false;

    // Initial sync
    syncOverlayStyle(textarea, overlay);
    syncOverlayPosition(textarea, overlay);
    syncOverlayContent(textarea, overlay);

    // Scroll sync
    textarea.addEventListener("scroll", () => {
        overlay.scrollTop = textarea.scrollTop;
        overlay.scrollLeft = textarea.scrollLeft;
    });

    // Content sync on input
    textarea.addEventListener("input", () => {
        syncOverlayContent(textarea, overlay);
        syncOverlayStyle(textarea, overlay);
    });

    // Paste sync (delayed to capture post-paste value)
    textarea.addEventListener("paste", () => {
        setTimeout(() => {
            if (overlay && document.contains(overlay)) {
                syncOverlayContent(textarea, overlay);
                syncOverlayStyle(textarea, overlay);
            }
        }, 10);
    });

    // Tab key for indentation
    textarea.addEventListener("keydown", (e) => {
        if (e.key === "Tab") {
            e.preventDefault();
            const start = textarea.selectionStart;
            const end = textarea.selectionEnd;
            textarea.value = textarea.value.substring(0, start) + "    " + textarea.value.substring(end);
            textarea.selectionStart = textarea.selectionEnd = start + 4;
            textarea.dispatchEvent(new Event("input"));
        }
    });

    // MutationObserver: resync position when textarea style attribute changes
    const styleObserver = new MutationObserver(() => {
        syncOverlayPosition(textarea, overlay);
        syncOverlayStyle(textarea, overlay);
    });
    styleObserver.observe(textarea, {
        attributes: true,
        attributeFilter: ["style"],
    });

    // ResizeObserver: track textarea size changes from node resizing.
    // ComfyUI resizes widgets via layout (not style attribute), so
    // MutationObserver alone won't catch all resize events.
    const resizeObserver = new ResizeObserver(() => {
        syncOverlayPosition(textarea, overlay);
        syncOverlayStyle(textarea, overlay);
        // Re-sync scroll position after resize (content may reflow)
        overlay.scrollTop = textarea.scrollTop;
        overlay.scrollLeft = textarea.scrollLeft;
    });
    resizeObserver.observe(textarea);

    // Cleanup: remove overlay if textarea is removed from DOM
    const parentObserver = new MutationObserver(() => {
        if (!document.contains(textarea)) {
            overlay.remove();
            styleObserver.disconnect();
            resizeObserver.disconnect();
            parentObserver.disconnect();
            enhancedTextareas.delete(textarea);
        }
    });
    if (textarea.parentNode) {
        parentObserver.observe(textarea.parentNode, { childList: true });
    }
}

// ─── DOM Polling: find TEX textareas and enhance them ────────────────

// We tag TEX node textareas with a data attribute in onNodeCreated,
// then poll the DOM for those textareas to add highlighting overlays.

function pollForTexTextareas() {
    const textareas = document.querySelectorAll('textarea[data-tex-node="true"].comfy-multiline-input');
    for (const ta of textareas) {
        enhanceTexTextarea(ta);
    }
}

// Poll periodically and also on DOM mutations
setInterval(pollForTexTextareas, 1000);

const domObserver = new MutationObserver(() => {
    pollForTexTextareas();
});
domObserver.observe(document.body, { childList: true, subtree: true });

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
    // Store error for any node (we filter in draw)
    texErrorCache.set(String(d.node_id), {
        message: d.exception_message || "Unknown error",
        type: d.exception_type || "",
        traceback: d.traceback || "",
    });
});

api.addEventListener("execution_start", () => {
    // Clear errors at the start of a new prompt
    texErrorCache.clear();
});

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

            // Tag the textarea element for discovery by our DOM poll.
            // Handle both widget.element and widget.inputEl (ComfyUI version compat).
            const tagTextarea = () => {
                const el = codeWidget.element || codeWidget.inputEl;
                if (el && el.tagName === "TEXTAREA") {
                    el.setAttribute("data-tex-node", "true");
                    el.setAttribute("data-tex-node-id", String(node.id));
                    return true;
                }
                return false;
            };

            if (!tagTextarea()) {
                // Element not ready yet — poll until it appears
                const checkInterval = setInterval(() => {
                    if (tagTextarea()) clearInterval(checkInterval);
                }, 200);
                setTimeout(() => clearInterval(checkInterval), 10000);
            }

            // Auto-socket: update inputs based on code content
            const updateSockets = debounce(() => {
                const code = codeWidget.value || "";
                const usedBindings = parseBindings(code);
                node._texBindings = usedBindings;
                syncInputs(node, usedBindings);
            }, DEBOUNCE_MS);

            // Watch for code changes
            const origCallback = codeWidget.callback;
            codeWidget.callback = function (...args) {
                if (origCallback) origCallback.apply(this, args);
                updateSockets();
            };

            // Initial scan
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
            // Rendered ABOVE the node title bar so it doesn't overlap the textarea
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
            this.setDirtyCanvas(true, true);
        };
    },
});

// ─── Highlight CSS (injected once) ───────────────────────────────────

(function injectStyles() {
    if (document.getElementById("tex-highlight-styles")) return;
    const style = document.createElement("style");
    style.id = "tex-highlight-styles";
    style.textContent = `
        .tex-highlight-overlay {
            font-variant-ligatures: none;
            -webkit-font-smoothing: antialiased;
        }
        .tex-hl-comment   { color: #6a6a8a; font-style: italic; }
        .tex-hl-keyword   { color: #c792ea; font-weight: bold; }
        .tex-hl-builtin   { color: #82aaff; }
        .tex-hl-binding   { color: #f78c6c; font-weight: bold; }
        .tex-hl-number    { color: #f9ae58; }
        .tex-hl-constant  { color: #ff5370; }
        .tex-hl-coordvar  { color: #89ddff; }
        .tex-hl-string    { color: #c3e88d; }

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
