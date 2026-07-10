"""S-2 — generate drag-and-drop ComfyUI workflow JSONs into examples/workflows/.

Each workflow embeds a REAL `examples/*.tex` snippet as the node's code widget, so the
curated set can't drift off the shipped examples. The set here is deliberately the
*procedural* snippets (they synthesise output from `u`/`v`/noise and need no wired IMAGE
input), so the graph is an unambiguous `TEX_Wrangle → PreviewImage` that loads and cooks
without the frontend's dynamic input-socket reconstruction. Image-input workflows
(grade/blur/compositing) and their dynamic `@A` sockets are validated and added in the
LIVE session (doc 35 S-2: "validated live") — the socket serialization can only be
confirmed by round-tripping through a real ComfyUI canvas.

The companion smoke test (test_v019_phase3) asserts every emitted file parses as JSON,
carries a `TEX_Wrangle` node whose code is a real shipped snippet AND compiles, and sinks
into a PreviewImage. Run: python tools/gen_workflows.py [--check].
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_EXAMPLES = os.path.join(_PKG, "examples")
_OUT_DIR = os.path.join(_EXAMPLES, "workflows")

# (filename, snippet-stem, title, debug_nan_highlight). Procedural snippets only —
# each uses just @OUT so the graph needs no input wiring.
_CURATED = [
    ("01_gradient.json", "gradient", "Linear gradient (procedural)", False),
    ("02_radial_gradient.json", "radial_gradient", "Radial gradient / vignette base", False),
    ("03_voronoi_cells.json", "voronoi_cells", "Voronoi cells noise", False),
    ("04_sdf_shapes.json", "sdf_shapes", "Signed-distance-field shapes", False),
    ("05_marble.json", "marble", "Marble texture (fbm)", False),
    ("06_perlin_clouds.json", "perlin_clouds", "Perlin clouds", False),
    ("07_wood_grain.json", "wood_grain", "Wood-grain texture", False),
    ("08_billow_texture.json", "billow_texture", "Billow noise texture", False),
]


def _read_snippet(stem: str) -> str:
    with open(os.path.join(_EXAMPLES, f"{stem}.tex"), encoding="utf-8") as f:
        return f.read()


def build_workflow(code: str, title: str, debug_nan: bool) -> dict:
    """A minimal, loadable litegraph UI-format graph: TEX_Wrangle → PreviewImage.
    widgets_values order matches the node's widget inputs: code, device, compile_mode,
    precision, debug_nan_highlight."""
    return {
        "last_node_id": 2,
        "last_link_id": 1,
        "nodes": [
            {"id": 1, "type": "TEX_Wrangle", "pos": [96, 128], "size": [420, 320],
             "flags": {}, "order": 0, "mode": 0, "inputs": [],
             "outputs": [{"name": "OUT", "type": "IMAGE", "links": [1], "slot_index": 0}],
             "properties": {"Node name for S&R": "TEX_Wrangle"},
             "widgets_values": [code, "auto", "none", "fp32", debug_nan]},
            {"id": 2, "type": "PreviewImage", "pos": [560, 128], "size": [320, 320],
             "flags": {}, "order": 1, "mode": 0,
             "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
             "outputs": [], "properties": {"Node name for S&R": "PreviewImage"},
             "widgets_values": []},
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"]],
        "version": 0.4,
        "extra": {"tex_workflow": title, "tex_snippet_source": True},
    }


def generate() -> dict:
    """Return {filename: json-string} for every curated workflow."""
    files = {}
    for fname, stem, title, dbg in _CURATED:
        wf = build_workflow(_read_snippet(stem), title, dbg)
        files[fname] = json.dumps(wf, indent=2) + "\n"
    return files


def main():
    check = "--check" in sys.argv
    files = generate()
    if check:
        for fname, content in files.items():
            path = os.path.join(_OUT_DIR, fname)
            try:
                existing = open(path, encoding="utf-8").read()
            except FileNotFoundError:
                print(f"{fname} missing — run tools/gen_workflows.py")
                return 1
            if existing != content:
                print(f"{fname} is stale — regenerate with tools/gen_workflows.py")
                return 1
        print(f"workflows up to date ({len(files)})")
        return 0
    os.makedirs(_OUT_DIR, exist_ok=True)
    for fname, content in files.items():
        with open(os.path.join(_OUT_DIR, fname), "w", encoding="utf-8") as f:
            f.write(content)
    print(f"wrote {len(files)} workflows to {_OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
