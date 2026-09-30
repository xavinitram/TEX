// Behavioural checks for the CodeMirror bundle (js/tex_cm6_bundle.js), run under plain node.
// The bundle is loaded the way the host loads it, by import(); nothing here needs a browser
// or editor_build/node_modules. Rebuild the bundle with `npm run build` in editor_build/
// after changing editor_build/src.
// Usage: node tests/js_editor_bundle_checks.cjs [path-to-bundle]   (add --dump for the word lists)
const path = require("path");
const { pathToFileURL } = require("url");

const bundle = process.argv.find((a, i) => i > 1 && !a.startsWith("--"))
    || path.join(__dirname, "..", "js", "tex_cm6_bundle.js");
const dump = process.argv.includes("--dump");

let failed = 0;
function check(label, cond, extra) {
    if (!cond) { failed++; console.log("FAIL " + label + (extra ? " :: " + extra : "")); }
    else console.log("ok   " + label);
}

// A completion context over a document with the cursor at `pos` (default: the end).
function context(C, doc, pos = doc.length) {
    const state = C.EditorState.create({ doc, extensions: [C.texLanguageDef] });
    return {
        state, pos, explicit: false,
        matchBefore(re) {
            const line = state.doc.lineAt(pos);
            const m = new RegExp("(?:" + re.source + ")$").exec(line.text.slice(0, pos - line.from));
            return m ? { from: line.from + m.index, to: pos, text: m[0] } : null;
        },
    };
}

// The leaf tokens of `doc` as "name:text".
function tokens(C, doc) {
    const out = [];
    C.texLanguageDef.parser.parse(doc).iterate({
        enter(n) { if (n.name !== "Document") out.push(n.name + ":" + doc.slice(n.from, n.to)); },
    });
    return out;
}

const docOf = (C, text) => ({ state: C.EditorState.create({ doc: text }) });
const diag = (C, text, d) => C.texErrorToDiagnostics(docOf(C, text), { message: "TEX_DIAG:" + JSON.stringify([d]) })[0];

async function main() {
    globalThis.window = globalThis;                           // the browser global an older build touches
    const realLog = console.log;
    console.log = () => {};                                   // the bundle announces itself on load
    try { await import(pathToFileURL(bundle).href); } finally { console.log = realLog; }
    const C = globalThis.TEX_CM6;

    if (dump) {
        const source = C.createTexCompletions(() => [], () => []);
        const opts = source(context(C, "s")).options;
        console.log(JSON.stringify({
            keywords: [...C.TEX_KEYWORDS], builtins: [...C.TEX_BUILTINS], constants: [...C.TEX_CONSTANTS],
            coordVars: [...C.TEX_COORD_VARS], labels: opts.map(o => o.label),
        }));
        return;
    }
    check("the bundle registers globalThis.TEX_CM6 on import", C && typeof C.texSetup === "function");

    // --- highlighting follows the language -------------------
    let t = tokens(C, "vec2 a; const float b; return a; frame time fps gauss_blur select over mix rand");
    check("vec2, const and return are keywords",
        ["keyword:vec2", "keyword:const", "keyword:return"].every(x => t.includes(x)), t.join(" "));
    check("frame, fps and time are built-in variables",
        ["frame", "fps", "time"].every(x => t.includes("variableName.definition:" + x)), t.join(" "));
    check("gauss_blur, select, over and the alias mix are builtins",
        ["gauss_blur", "select", "over", "mix"].every(x => t.includes("variableName.standard:" + x)), t.join(" "));
    check("rand is not a function, so it is a plain name", t.includes("variableName:rand"), t.join(" "));
    t = tokens(C, "a@x p$y f@z img@q @w $v");
    check("a and p are typed-binding prefixes",
        t.length === 6 && t.every(x => x.startsWith("variableName.special:")), t.join(" "));
    t = tokens(C, "/* one\ntwo */ float x; // tail\n@a = 1;");
    check("a block comment continues across lines and ends",
        t[0] === "blockComment:/* one" && t[1] === "blockComment:two */" && t[2] === "keyword:float",
        t.join(" | "));

    // --- no word completion on numbers, comments or strings --------------------
    const source = C.createTexCompletions(() => ["img"], () => ["amount"]);
    const labels = (doc, pos) => { const r = source(context(C, doc, pos)); return r ? r.options.map(o => o.label) : null; };
    const l = labels("float x = gau");
    check("a word completes to every stdlib function, keyword and variable",
        l && ["gauss_blur", "vec2", "const", "return", "frame", "mix", "sin"].every(x => l.includes(x)) && !l.includes("rand"));
    check("a digit-led token is not completed", labels("float x = 1") === null && labels("x = 1e") === null
        && labels("x = 1.5") === null && labels("x = 0xF") === null);
    check("a name that contains digits is completed", labels("float x2") !== null);
    check("no completion inside a line comment", labels("float x; // gau") === null);
    check("no completion inside a block comment", labels("/* one\n gau") === null);
    check("no completion inside a string", labels('string s = "gau') === null);
    check("completion resumes after the comment ends", labels("/* c */ gau") !== null);
    check("@ still completes bindings", (labels("@i") || [])[0] === "@img");
    check("$ still completes parameters", (labels("$a") || [])[0] === "$amount");

    // --- hover documentation ---------------------------------------------------
    const hover = (doc, at) => {
        if (typeof C.texHoverAt !== "function") return "no texHoverAt";
        const s = C.EditorState.create({ doc, extensions: [C.texLanguageDef] });
        const h = C.texHoverAt(s, doc.indexOf(at) + 1);
        return h ? h.entry.label : null;
    };
    check("hover documents a built-in call", hover("float x = length(v);", "length") === "length");
    check("hover documents a keyword and a built-in variable",
        hover("vec2 p;", "vec2") === "vec2" && hover("float t = time;", "time") === "time");
    check("hover skips a binding named like a built-in", hover("@length = 1.0;", "length") === null
        && hover("float a = $mix;", "mix") === null && hover("float a = f@u;", "u") === null);
    check("hover skips the typed-binding prefix", hover("float a = v@x;", "v@x") === null);
    check("hover skips a member access", hover("float a = c.length;", "length") === null);
    check("hover skips words in comments and strings",
        hover("// length here", "length") === null && hover('string s = "sin";', "sin") === null);

    // --- server columns count code points, the editor counts UTF-16 units ------
    const smile = "\u{1F600}";
    let text = `string s = "${smile}${smile}"; float bad = ;`;
    const cpCol = [...text].indexOf("b") + 1;                  // 1-based code-point column of "bad"
    let d = diag(C, text, { line: 1, col: cpCol, end_col: cpCol + 3, message: "m" });
    check("a diagnostic after non-BMP characters starts on the right token",
        text.slice(d.from, d.to) === "bad", JSON.stringify(text.slice(d.from, d.to)));
    text = `${smile}\nx = ${smile} + y;`;
    d = diag(C, text, { line: 1, col: 1, end_line: 2, end_col: [..."x = " + smile + " + y"].length + 1, message: "m" });
    check("a multi-line span ends on the right token", text.slice(d.from, d.to) === `${smile}\nx = ${smile} + y`,
        JSON.stringify(text.slice(d.from, d.to)));
    d = diag(C, "float x;", { line: 1, col: 99, message: "m" });
    check("a column past the end of the line is clamped", d.from === 8 && d.to === 8, d.from + "," + d.to);
    d = diag(C, "float x;", { line: 9, col: 1, message: "m" });
    check("a diagnostic without a valid line marks the first line", d.from === 0 && d.to === 8);
    d = C.texErrorToDiagnostics(docOf(C, `${smile}${smile} bad`), { message: "Parse error at line 1, column 4: oops" })[0];
    check("a message-text location is converted the same way", d.from === 5, String(d.from));
    d = C.texErrorToDiagnostics(docOf(C, "a\nb\nc"), { message: "runtime failure at line 2" })[0];
    check("a bare 'line N' message marks that line", d.from === 2 && d.to === 3, d.from + "," + d.to);
    d = C.texErrorToDiagnostics(docOf(C, "a\nb"), { message: "no location at all" })[0];
    check("a message with no location marks the first line", d.from === 0 && d.to === 1);

    if (failed) { console.log(failed + " check(s) failed"); process.exit(1); }
}

main().catch((e) => { console.log("FAIL " + (e && e.stack || e)); process.exit(1); });
