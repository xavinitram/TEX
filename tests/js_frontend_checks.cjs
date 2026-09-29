// Behavioural checks for the pure logic in js/tex_extension.js, run under plain node.
// The file is a ComfyUI ES module, so the helpers under test are cut out by name and
// evaluated in a sandbox with a fake graph; nothing here needs a browser.
// Usage: node tests/js_frontend_checks.cjs [path-to-tex_extension.js]
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const jsPath = process.argv[2] || path.join(__dirname, "..", "js", "tex_extension.js");
const src = fs.readFileSync(jsPath, "utf8").split("\r\n").join("\n");

function cutFunction(name) {
    // Top-level functions in the file close with a brace in column 0.
    let at = src.indexOf(`function ${name}(`);
    if (at < 0) return null;
    if (src.slice(at - 6, at) === "async ") at -= 6;
    return src.slice(at, src.indexOf("\n}\n", at) + 2);
}

function cutConst(name) {
    const at = src.indexOf(`const ${name} =`);
    return at < 0 ? null : src.slice(at, src.indexOf(";\n", at) + 1);
}

const names = ["_parseParamMetadata", "_socketTypeForPrefix", "_texMaskSource", "parseCode",
    "_texLinkIn", "_texNodeByPromptId", "syncInputs", "syncOutputs", "syncParams",
    "_texSyncParamSchema", "_texParamSchemaEntry", "_texPublishDefault", "_texIdBefore",
    "applyCodeToSockets", "_escHtml", "_renderDoctorFacts", "_buildSnippetTree",
    "_texLintTypes", "_texRequestLint", "_texCodeOf", "_texSingleWiredInput", "_texGetLink",
    "_texParamWidgets", "_texChainSig", "_texPreflightSpec", "_texMaybePreflight"];
let code = cutConst("RESERVED_NAMES") + "\n" + cutConst("_cmpLocale") + "\n"
    + cutConst("_TEX_LINT_TYPES") + "\n"
    + "const _texParamRegistry = new Map(); const _texParamConflictWarned = new Set();\n"
    + "const _lintSeq = new WeakMap(); const _texPreflight = new Map();\n"
    + "const _texPreflightSig = new Map(); const _texPreflightBusy = new Set();\n"
    + "const _texPreflightRetryAt = new Map(); let _texPreflightEnabled = true;\n"
    + "const getCM6 = () => CM6;\n"
    + 'const TEX_NODE_TYPE = "TEX_Wrangle";\n';
for (const n of names) {
    const f = cutFunction(n);
    if (f) code += f + "\n";
}
const opt = (n) => `typeof ${n} === "function" ? ${n} : null`;
code += `this.T = { parseCode, syncOutputs, syncParams, _texPublishDefault: ${opt("_texPublishDefault")},
    _texNodeByPromptId: ${opt("_texNodeByPromptId")}, _texParamSchemaEntry, RESERVED_NAMES,
    _buildSnippetTree: ${opt("_buildSnippetTree")}, _renderDoctorFacts: ${opt("_renderDoctorFacts")},
    _texIdBefore: ${opt("_texIdBefore")}, applyCodeToSockets: ${opt("applyCodeToSockets")},
    _texSyncParamSchema, _texLintTypes: ${opt("_texLintTypes")},
    _texRequestLint: ${opt("_texRequestLint")}, _texMaybePreflight: ${opt("_texMaybePreflight")},
    _texPreflightSig, _texPreflight };`;

// The module-level `api` (fetchApi), `CM6` and `LiteGraph` are stand-ins the checks replace.
const sandbox = { console, app: { graph: null }, LiteGraph: undefined, CM6: null,
    api: { fetchApi: async () => { throw new Error("no fetch"); } }, Date, setTimeout };
vm.createContext(sandbox);
vm.runInContext(code, sandbox);
const api = sandbox.T;

let failed = 0;
function check(label, cond, extra) {
    if (!cond) { failed++; console.log("FAIL " + label + (extra ? " :: " + extra : "")); }
    else console.log("ok   " + label);
}
const sorted = (set) => [...set].sort().join(",");

// --- T14: scatter writes, compound ops and ++/-- are output assignments ---------------
{
    let r = api.parseCode("@OUT[ix, iy] = 1.0;");
    check("T14 scatter write is an output", sorted(r.outputs) === "OUT", sorted(r.outputs));
    check("T14 scatter write is not an input", !r.inputs.has("OUT"), sorted(r.inputs));
    r = api.parseCode("@OUT[ix, iy] += 1.0;");
    check("T14 scatter += is output and input", r.outputs.has("OUT") && r.inputs.has("OUT"));
    r = api.parseCode("@count++;");
    check("T14 @x++ is a read-modify-write output", r.outputs.has("count") && r.inputs.has("count"));
    r = api.parseCode("@a.rgb = @b.rgb;");
    check("T14 swizzle write still works", sorted(r.outputs) === "a" && sorted(r.inputs) === "b");
    r = api.parseCode("float t = @a[0] == 1.0 ? 1.0 : 0.0; @o = t;");
    check("T14 subscripted == is a read, not a write", !r.outputs.has("a") && r.inputs.has("a"),
        sorted(r.outputs));
}

// --- T15: string / hex defaults survive; // inside a string is not a comment ----------
{
    let r = api.parseCode('s$prefix = "image"; c$tint = "#ff8800"; f$k = 0.5; @o = @i;');
    check("T15 string default", r.params.get("prefix").defaultValue === '"image"',
        r.params.get("prefix").defaultValue);
    check("T15 hex default", r.params.get("tint").defaultValue === '"#ff8800"',
        r.params.get("tint").defaultValue);
    check("T15 float default", r.params.get("k").defaultValue === "0.5");
    r = api.parseCode('s$url = "http://x/y"; // note\n@o = @i;');
    check("T15 // inside a string kept", r.params.get("url").defaultValue === '"http://x/y"',
        r.params.get("url").defaultValue);
    check("T15 code after comment still parsed", r.outputs.has("o") && r.inputs.has("i"));
    r = api.parseCode('f$a = 1.0 [min: 0, max: 2, label: "A"]; @o = $a;');
    check("T15 metadata still parsed", r.params.get("a").metadata.max === 2
        && r.params.get("a").metadata.label === "A");
    r = api.parseCode("/* @ghost = 1; */ @o = @i; // @nope = 2\n");
    check("T15 comments do not create bindings", sorted(r.outputs) === "o" && sorted(r.inputs) === "i");
}

// --- fake LiteGraph node for syncOutputs / syncParams ---------------------------------
function makeNode() {
    const graph = { links: new Map(), nodes: new Map(), getNodeById(id) { return this.nodes.get(id) || null; } };
    const node = {
        id: 1, graph, outputs: [], inputs: [], widgets: [],
        setDirtyCanvas() {},
        addOutput(name, type) { this.outputs.push({ name, type, links: [] }); },
        removeOutput(i) {
            for (const id of this.outputs[i].links) {
                const l = graph.links.get(id);
                if (l) { const t = graph.nodes.get(l.target_id); if (t) t.inputs[l.target_slot].link = null; graph.links.delete(id); }
            }
            this.outputs.splice(i, 1);
        },
        addInput(name, type) { this.inputs.push({ name, type, link: null }); },
        removeInput(i) { this.inputs.splice(i, 1); },
        addWidget(type, name, value) { const w = { type, name, value }; this.widgets.push(w); return w; },
        connect(slot, target, tslot) {
            const id = graph.links.size + 100;
            graph.links.set(id, { id, origin_id: this.id, origin_slot: slot, target_id: target.id, target_slot: tslot });
            this.outputs[slot].links.push(id);
            target.inputs[tslot].link = id;
        },
    };
    graph.nodes.set(1, node);
    return node;
}

// --- T13: output set change keeps the surviving output's wires --------------------------
{
    const node = makeNode();
    api.syncOutputs(node, new Set(["b"]));
    const sink = { id: 2, inputs: [{ name: "x", link: null }] };
    node.graph.nodes.set(2, sink);
    node.connect(0, sink, 0);
    api.syncOutputs(node, new Set(["a", "b"]));   // b moves from slot 0 to slot 1
    const b = node.outputs.find((o) => o.name === "b");
    check("T13 wire restored after output-set change",
        b && b.links.length === 1 && sink.inputs[0].link != null,
        JSON.stringify(node.outputs.map((o) => [o.name, o.links])));
    const l = node.graph.links.get(sink.inputs[0].link);
    check("T13 restored wire leaves from the new slot", l && l.origin_slot === 1);
}

// --- T16: reloaded param sockets are adopted -------------------------------------------
{
    const node = makeNode();
    node.inputs.push({ name: "strength", type: "FLOAT", link: 7, widget: { name: "strength" } });
    api.syncParams(node, new Map([["strength", { typeHint: "f", defaultValue: "0.5" }]]));
    const inp = node.inputs.find((i) => i.name === "strength");
    check("T16 existing param socket flagged", inp && inp._texParam === true);
    check("T16 one socket only", node.inputs.filter((i) => i.name === "strength").length === 1);
}

// --- T34: v4 gets a vector widget; composite prompt ids do not resolve ------------------
{
    const node = makeNode();
    api.syncParams(node, new Map([["tint", { typeHint: "v4", defaultValue: "vec4(1.0, 0.5, 0.25, 1.0)" }]]));
    const w = node.widgets.find((x) => x.name === "tint");
    check("T34 v4 param is a text widget", w && w.type === "text", w && w.type);
    check("T34 v4 default parsed", w && w.value === "1.0, 0.5, 0.25, 1.0", w && w.value);
    check("T34 v4 schema entry is STRING", api._texParamSchemaEntry("v4")[0] === "STRING");
    if (api._texNodeByPromptId) {
        sandbox.app.graph = { getNodeById: (id) => ({ id }) };
        check("T34 plain id resolves", api._texNodeByPromptId("12").id === 12);
        check("T34 subgraph id does not resolve to its prefix", api._texNodeByPromptId("12:3") === null);
    } else {
        check("T34 prompt-id resolver exists", false);
    }
}

// --- publish defaults keep their type ---------------------------------------------------
{
    const d = api._texPublishDefault;
    if (d) {
        check("pub string default", d("s", '"abc"') === "abc");
        check("pub color default", d("c", '"#ff8800"') === "#ff8800");
        check("pub vec3 default", d("v3", "vec3(1.0, 2.0, 3.0)") === "1.0, 2.0, 3.0");
        check("pub bool default", d("b", "true") === 1 && d("b", "0") === 0);
        check("pub int default", d("i", "3") === 3 && d("f", null) === 0);
    } else {
        check("publish default helper exists", false);
    }
}

// --- system kwargs are never bindings or params -------------------------------------------
{
    const sys = ["code", "device", "compile_mode", "precision", "_tex_any", "_tex_chain",
        "_tex_preview", "debug_nan_highlight", "_tex_slot_map", "_tex_time"];
    check("reserved names mirror the backend system kwargs",
        sys.every((n) => api.RESERVED_NAMES.has(n)) && api.RESERVED_NAMES.size === sys.length,
        [...api.RESERVED_NAMES].join(","));
    let r = api.parseCode("@precision = @debug_nan_highlight; @o = @i;");
    check("a system kwarg is not a wire or output", sorted(r.outputs) === "o" && sorted(r.inputs) === "i",
        sorted(r.outputs) + "|" + sorted(r.inputs));
    r = api.parseCode("i$device = 3; f$code = 1.0; f$k = 0.5; @o = $k;");
    check("an explicit $decl of a reserved name is ignored", sorted(r.params.keys()) === "k",
        sorted(r.params.keys()));
}

// --- snippet tree: a leaf sharing a folder's name, and __proto__ segments ---------------------
{
    if (!api._buildSnippetTree) check("snippet tree helper exists", false);
    else {
        const t = api._buildSnippetTree({ "a": "leaf", "a/b": "inner", "z/__proto__/x": "p", "__proto__/y": "q" });
        check("folder keeps its children when a leaf shares its name",
            typeof t.a === "object" && t.a.b === "inner", JSON.stringify(t.a));
        check("the same-named leaf is still reachable",
            Object.values(t.a).includes("leaf"), JSON.stringify(t.a));
        check("__proto__ segments stay in the tree", Object.prototype.x === undefined
            && Object.prototype.y === undefined && Object.keys(t).includes("__proto__"),
            Object.keys(t).join(","));
        const u = api._buildSnippetTree({ "a/b": "inner", "a": "leaf" });
        check("leaf after folder is kept too", u.a.b === "inner" && Object.values(u.a).includes("leaf"));
    }
}

// --- doctor dialog escapes server facts ---------------------------------------------------
{
    const html = api._renderDoctorFacts({ arch: { note: "<img src=x onerror=1>" },
        "k<b>": "<script>x</script>", list: ["<i>"] });
    check("doctor facts are escaped", !/<(img|script|b|i)[ >]/.test(html) && html.includes("&lt;img")
        && html.includes("&lt;script&gt;"), html);
}

// --- lowest-node-id tie-break is numeric -------------------------------------------------------
{
    check("9 sorts before 10", api._texIdBefore && api._texIdBefore(9, 10) === true
        && api._texIdBefore(10, 9) === false);
    const optional = {};
    sandbox.LiteGraph = { registered_node_types: { TEX_Wrangle: { nodeData: { input: { optional } } } } };
    const warn = console.warn; console.warn = () => {};
    api._texSyncParamSchema({ id: 10 }, new Map([["k", { typeHint: "i" }]]));
    api._texSyncParamSchema({ id: 9 }, new Map([["k", { typeHint: "b" }]]));
    console.warn = warn;
    check("the lowest numeric node id wins the shared schema", optional.k && optional.k[0] === "BOOLEAN",
        JSON.stringify(optional.k));
    api._texSyncParamSchema({ id: 10 }, null);
    api._texSyncParamSchema({ id: 9 }, null);
    check("dropping every owner clears the schema entry", !("k" in optional));
}

// --- a removed node's deferred socket sync does nothing ------------------------------------
{
    const optional = {};
    sandbox.LiteGraph = { registered_node_types: { TEX_Wrangle: { nodeData: { input: { optional } } } } };
    const node = makeNode();
    node.graph = null;
    api.applyCodeToSockets(node, "f$leak = 1.0; @o = @i;");
    check("no params are registered for a removed node", !("leak" in optional) && node.widgets.length === 0,
        JSON.stringify(Object.keys(optional)));
    sandbox.LiteGraph = undefined;
}

// --- live lint sends the wired types and drops stale responses ----------------------------------
{
    const graph = { links: new Map([[5, { origin_id: 10, origin_slot: 0 }], [6, { origin_id: 11, origin_slot: 0 }],
        [7, { origin_id: 12, origin_slot: 0 }]]),
        getNodeById(id) { return this.nodes[id]; },
        nodes: { 10: { outputs: [{ type: "MASK" }] }, 11: { outputs: [{ type: "IMAGE" }] },
                 12: { outputs: [{ type: "FLOAT" }] } } };
    const node = { graph, inputs: [{ name: "m", link: 5 }, { name: "img", link: 6 },
        { name: "k", link: 7, _texParam: true }, { name: "free", link: null }] };
    if (api._texLintTypes) {
        check("lint types: a MASK wire is a float, an IMAGE wire is left to the default",
            JSON.stringify(api._texLintTypes(node)) === '{"m":"FLOAT"}', JSON.stringify(api._texLintTypes(node)));
    } else check("lint types helper exists", false);

    let text = "a";
    const dispatched = [];
    const view = { state: { doc: { toString: () => text } }, dispatch: (x) => dispatched.push(x) };
    sandbox.CM6 = { texErrorToDiagnostics: (v, e) => [e.message], setDiagnostics: (s, d) => d };
    const pending = [];
    const bodies = [];
    sandbox.api = { fetchApi: (url, opts) => new Promise((res) => {
        bodies.push(JSON.parse(opts.body));
        pending.push((diag) => res({ ok: true, json: async () => ({ diagnostics: [diag] }) }));
    }) };
    const tick = () => new Promise((r) => setTimeout(r, 0));
    (async () => {
        const first = api._texRequestLint(view, "a", node);
        const second = api._texRequestLint(view, "a", node);
        check("lint request carries the binding types", bodies[0] && bodies[0].types
            && bodies[0].types.m === "FLOAT", JSON.stringify(bodies[0]));
        pending[1]("new"); await second;
        pending[0]("old"); await first; await tick();
        check("only the latest lint response is painted", dispatched.length === 1
            && JSON.stringify(dispatched[0]).includes("new") && !JSON.stringify(dispatched).includes("old"),
            JSON.stringify(dispatched));
        dispatched.length = 0;
        const third = api._texRequestLint(view, "a", node);
        text = "ab";   // edited while the request was in flight
        pending[2]("late"); await third; await tick();
        check("a response for an edited document is dropped", dispatched.length === 0,
            JSON.stringify(dispatched));
        await preflightChecks();
        finish();
    })().catch((e) => { console.log("FAIL lint/preflight checks threw :: " + e.stack); failed++; finish(); });
}

// --- preflight: a failed request is retried, a stale verdict is dropped ---------------------------
async function preflightChecks() {
    if (!api._texMaybePreflight) { check("preflight helper exists", false); return; }
    const codeOf = { 1: "x", 2: "y" };
    const mk = (id, link) => ({ id, widgets: [{ name: "code", get value() { return codeOf[id]; } }],
        inputs: [{ name: "i", link }], outputs: [] });
    const a = mk(1, 50), b = mk(2, 51);
    sandbox.app.graph = { links: new Map([[50, { origin_id: 99, origin_slot: 0 }], [51, { origin_id: 1, origin_slot: 0 }]]),
        getNodeById: (id) => ({ 1: a, 2: b })[id] };
    let now = 1000;
    sandbox.Date = class extends Date { static now() { return now; } };
    const tick = () => new Promise((r) => setTimeout(r, 0));
    let calls = 0, mode = "fail", release = null;
    sandbox.api = { fetchApi: () => { calls++;
        if (mode === "fail") return Promise.reject(new Error("offline"));
        if (mode === "http") return Promise.resolve({ ok: false, status: 500, json: async () => ({}) });
        return new Promise((res) => { release = () => res({ ok: true, json: async () => ({ ok: true }) }); }); } };
    const chain = [a, b];
    api._texMaybePreflight(chain); await tick(); await tick();
    check("a failed preflight leaves no recorded signature", !api._texPreflightSig.has(2));
    api._texMaybePreflight(chain); await tick();
    check("a failed preflight is not re-sent on the very next repaint", calls === 1, "calls=" + calls);
    now += 6000; mode = "http";
    api._texMaybePreflight(chain); await tick(); await tick();
    check("a failed preflight is retried after the pause", calls === 2 && !api._texPreflightSig.has(2), "calls=" + calls);
    now += 6000; mode = "ok";
    api._texMaybePreflight(chain);
    codeOf[1] = "x2";   // the chain is edited while the request is in flight
    release(); await tick(); await tick();
    check("the verdict of an edited chain is not kept", !api._texPreflight.has(2) && !api._texPreflightSig.has(2));
    api._texMaybePreflight(chain); release(); await tick(); await tick();
    check("a current verdict is kept with its signature", api._texPreflight.has(2) && api._texPreflightSig.has(2));
}

function finish() { process.exit(failed ? 1 : 0); }
