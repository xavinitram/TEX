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
    const at = src.indexOf(`function ${name}(`);
    if (at < 0) return null;
    return src.slice(at, src.indexOf("\n}\n", at) + 2);
}

function cutConst(name) {
    const at = src.indexOf(`const ${name} =`);
    return at < 0 ? null : src.slice(at, src.indexOf(";\n", at) + 1);
}

const names = ["_parseParamMetadata", "_socketTypeForPrefix", "_texMaskSource", "parseCode",
    "_texLinkIn", "_texNodeByPromptId", "syncOutputs", "syncParams", "_texSyncParamSchema",
    "_texParamSchemaEntry", "_texPublishDefault"];
let code = cutConst("RESERVED_NAMES") + "\n"
    + "const _texParamRegistry = new Map(); const _texParamConflictWarned = new Set();\n"
    + 'const TEX_NODE_TYPE = "TEX_Wrangle";\n';
for (const n of names) {
    const f = cutFunction(n);
    if (f) code += f + "\n";
}
code += `this.api = { parseCode, syncOutputs, syncParams, _texPublishDefault: typeof _texPublishDefault === "function" ? _texPublishDefault : null,
    _texNodeByPromptId: typeof _texNodeByPromptId === "function" ? _texNodeByPromptId : null,
    _texParamSchemaEntry };`;

const sandbox = { console, app: { graph: null }, LiteGraph: undefined };
vm.createContext(sandbox);
vm.runInContext(code, sandbox);
const api = sandbox.api;

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

process.exit(failed ? 1 : 0);
