"""
PORT-3 — `tex run`: a standalone CLI that runs a .tex program on an image file with NO
ComfyUI. Proves TEX is host-agnostic and doubles as a ComfyUI-free regression harness.

I/O is **torchvision-only** (no numpy, no PIL — a bit-exact uint8<->float round-trip is
verified). The cook wires through the real engine — `tex_engine.cook` (ENG-1, v0.22) —
so no ComfyUI node, schema or slot protocol is in the call path at all; the host seam
degrades to NullHostServices. Before v0.22 this called `TEXWrangleNode.execute`, which
made the host-agnostic CLI depend on a ComfyUI v3 node classmethod to cook one frame.

    python -m TEX_Wrangle.tex_cli run examples/grayscale.tex --in in.png --out out.png
    python -m TEX_Wrangle.tex_cli run prog.tex --in a.png --out b.png --device cuda --precision auto

Bit-depth policy (PORT-4)
-------------------------
I/O is **8-bit** (uint8 PNG): TEX cooks in fp32 (or fp16), but the CLI quantises to 8-bit
on write — matching ComfyUI's IMAGE convention and the 3.9e-3 "8-bit quantum" that gates
the whole precision story (doc 22/28). The uint8<->float round-trip is bit-exact
(round, not truncate). A 16-bit PNG variant (`--bit-depth 16`) is a deliberate NON-goal
this cycle: there is no 16-bit consumer in the ComfyUI pipeline, and torchvision's
`encode_png` is 8-bit; a future host with an HDR sink can add it against this same
`load_image`/`save_image` seam.
"""
import argparse
import sys

import torch


def _require_torchvision():
    try:
        import torchvision  # noqa: F401
        from torchvision.io import decode_image  # noqa: F401
    except Exception:
        sys.exit("tex run needs torchvision for image I/O (no numpy/PIL): "
                 "pip install torchvision")


def load_image(path: str, device: str = "cpu") -> torch.Tensor:
    """PNG/JPG -> [1, H, W, 3] float32 in [0,1] (ComfyUI IMAGE layout), torchvision-only.
    Branches on BIT DEPTH: a uint16 (16-bit) PNG decodes to 0-65535, so dividing by 255
    (audit) would make it 257x too bright — normalise by the dtype's max.

    DATA-2: an `.exr` input is read by the pure-torch EXR reader instead — scene-linear fp32
    [1,H,W,C] with ALL channels + HDR kept (no [0,1] normalize, no 3-channel force), so
    `tex run --in a.exr --out b.exr` round-trips values the PNG path would clamp/drop."""
    if path.lower().endswith(".exr"):
        from .tex_io import exr
        px = exr.read_exr(path).pixels                # [H, W, C] fp32, HDR + all channels
        if px.shape[-1] not in (3, 4):
            # `tex run` cooks one IMAGE; the engine reads C as RGB(A). A 1-/2-/5+-channel EXR
            # (depth, AOV, multi-plane) would be silently reinterpreted downstream (a mask-output
            # program crushes it to luma; a <3-ch input can crash a mask egress), so refuse it
            # with a clean message instead — named multi-plane EXR is a future host feature (DATA-6).
            raise ValueError(
                f"tex run: EXR input has {px.shape[-1]} channels; only 3 (RGB) or 4 (RGBA) are "
                f"supported. A depth/AOV/multi-plane EXR needs a host that carries named planes.")
        return px.unsqueeze(0).to(device)             # [1, H, W, C] fp32, HDR + alpha kept
    from torchvision.io import read_file, decode_image
    from .tex_io import BufferDesc, decode_to_fp32
    img = decode_image(read_file(path))          # [C, H, W], uint8 or uint16
    if img.shape[0] == 1:
        img = img.expand(3, -1, -1)              # grayscale -> RGB
    img = img[:3]                                # drop alpha
    storage = {torch.uint8: "uint8", torch.uint16: "uint16"}.get(img.dtype)
    if storage is None:
        raise ValueError(f"tex run: unsupported image dtype {img.dtype} (expected uint8/uint16)")
    t = decode_to_fp32(img, BufferDesc(storage))        # [C, H, W] in [0,1] (DATA-2 seam)
    return t.permute(1, 2, 0).unsqueeze(0).to(device)   # [1, H, W, 3]


def save_image(tensor: torch.Tensor, path: str, *, bit_depth: int = 8) -> None:
    """[1, H, W, C] float [0,1] (or [1, H, W] / [H, W] mask) -> PNG. Round (not truncate)
    for a bit-exact round-trip: round(u/255*255) == u. `bit_depth=16` (DATA-2) writes a
    16-bit PNG via `tex_io` (torchvision's encoder is uint8-only) — a higher-fidelity
    normalized-integer sink than 8-bit; still [0,1], so EXR remains the HDR path."""
    from torchvision.io import encode_png, write_file
    t = tensor.detach().float().cpu()
    if t.dim() == 4:                              # [1, H, W, C] -> [H, W, C]
        t = t[0]
    elif t.dim() == 3 and t.shape[0] == 1:        # doc 32 B1: [1, H, W] batched mask/scalar
        t = t[0].unsqueeze(-1)                    #   -> [H, W, 1] (NOT slice the width axis)
    if t.dim() == 2:                              # [H, W] mask -> [H, W, 1]
        t = t.unsqueeze(-1)
    if t.shape[-1] > 4:                           # a stray non-channels-last shape
        raise ValueError(f"tex run: cannot save a tensor of shape {tuple(tensor.shape)} as a "
                         f"PNG (expected [1,H,W,C], [1,H,W] or [H,W])")
    if t.shape[-1] == 4:                          # doc 32 S6: PNG output is RGB (alpha dropped)
        print("tex run: note: dropping alpha channel (PNG output is RGB)", file=sys.stderr)
    t = t[..., :3]
    if t.shape[-1] == 1:
        t = t.expand(-1, -1, 3)
    from .tex_io import BufferDesc, encode_from_fp32
    if bit_depth == 16:
        from .tex_io.png import write_png16
        write_png16(path, encode_from_fp32(t, BufferDesc("uint16")))   # clamp+round to uint16
        return
    u8 = encode_from_fp32(t, BufferDesc("uint8")).permute(2, 0, 1)     # [C, H, W] (DATA-2 seam)
    write_file(path, encode_png(u8))


def run_program(code: str, image: torch.Tensor, device="cpu", precision="fp32",
                compile_mode="none", profile="comfy") -> torch.Tensor:
    """Run a TEX program on one image; return the primary IMAGE output (the first
    vec3/vec4 output, else the first output). Compiles ONCE for its binding sets — the
    compiler's own `referenced`/`assigned` give the input bindings and output types (no
    regex to drift from the grammar); the engine then re-infers types from the tensor.

    `profile` (ENG-3): 'comfy' clamps for a PNG sink; 'engine' preserves the raw fp32 values
    (unclamped, alpha kept) for the EXR sink (DATA-2) — the whole point of a float format.

    ENG-1 (v0.22): cooks through `tex_engine.cook`. The CLI exists to prove TEX is
    host-agnostic, and until v0.22 it had to import a ComfyUI v3 node classmethod to cook
    one frame — dragging in the node schema, the lazy slot-map protocol and the HUD
    payload for a job with no host at all. Output is unchanged: the engine returns raw
    tensors, and the 'comfy' egress profile (ENG-3) is the same conversion the node
    applied — verified byte-identical against the node path across clamped, gray,
    alpha-bearing and MASK outputs.
    """
    from . import tex_engine
    from .tex_api import compile as tex_compile
    from .tex_marshalling import prepare_output, map_inferred_type
    from .tex_compiler.types import TEXType

    prog = tex_compile(code, {})            # authoritative binding sets from the compiler
    # `referenced` mixes @wire bindings AND $param names (the type checker adds both), so a
    # legit single-image program with a $param (e.g. examples/vignette.tex's f$strength) would
    # count as >1 input. Exclude the params (prog.params is keyed by $-name) — they carry their
    # own widget defaults, they are not wires. Assigned names are the outputs.
    inputs = sorted(set(prog.referenced) - set(prog.assigned) - set(prog.params))
    # S1 (doc 33): `tex run` has a single --in image. Binding it to EVERY referenced input
    # would silently alias a multi-input program (@A == @B) and satisfy a typo'd @binding,
    # bypassing the engine's E6003 "not connected" guard. Refuse >1 distinct wire input.
    if len(inputs) > 1:
        raise ValueError(
            f"tex run takes one --in image, but this program wires {len(inputs)} inputs "
            f"({', '.join('@' + n for n in inputs)}). Multi-input programs need ComfyUI "
            f"(a --bind NAME=path option is a future addition).")
    res = tex_engine.cook(code, {name: image for name in inputs},
                          device_mode=device, precision=precision,
                          compile_mode=compile_mode)
    # The profile is PINNED per call, not inherited from the process-wide global, so a host
    # that flipped the profile can't change `tex run`. A PNG sink pins 'comfy' (the clamp +
    # alpha-drop + gray-expand save_image's quantisation assumes); an EXR sink pins 'engine'
    # so the float format actually carries the scene-linear values TEX cooked (DATA-2).
    results = [prepare_output(res.outputs[n],
                              map_inferred_type(res.assigned[n], False), profile=profile)
               for n in res.output_names]

    idx = next((i for i, n in enumerate(res.output_names)
                if prog.assigned[n] in (TEXType.VEC3, TEXType.VEC4)), 0)
    return results[idx]


def run(args) -> None:
    _require_torchvision()
    with open(args.program, encoding="utf-8") as f:
        code = f.read()
    image = load_image(args.in_path, args.device)
    is_exr = args.out_path.lower().endswith(".exr")
    # EXR is the value-preserving sink → cook under the 'engine' profile (raw fp32, alpha kept,
    # unclamped); a PNG sink stays 'comfy' (clamped, the quantiser's contract) (DATA-2, ENG-3).
    result = run_program(code, image, device=args.device, precision=args.precision,
                         compile_mode=args.compile_mode,
                         profile="engine" if is_exr else "comfy")
    if is_exr:
        from .tex_io import exr
        exr.write_exr(args.out_path, result, half=args.half)
        fmt = "exr/half" if args.half else "exr/float"
    else:
        save_image(result, args.out_path, bit_depth=args.bit_depth)
        fmt = f"png/{args.bit_depth}bit"
    print(f"tex run: {args.program} on {args.in_path} -> {args.out_path} "
          f"({args.device}/{args.precision}/{fmt})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tex", description="TEX Wrangle standalone runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run", help="run a .tex program on an image file")
    rp.add_argument("program", help="path to a .tex program")
    rp.add_argument("--in", dest="in_path", required=True,
                    help="input image (png/jpg, or .exr — scene-linear fp32, HDR/alpha kept)")
    rp.add_argument("--out", dest="out_path", required=True,
                    help="output image — .exr writes a float/half EXR (HDR, value-preserving), "
                         "otherwise a PNG (DATA-2)")
    rp.add_argument("--device", default="cpu", help="cpu | cuda")
    rp.add_argument("--precision", default="fp32", choices=["fp32", "auto", "fp16"])
    rp.add_argument("--compile-mode", dest="compile_mode", default="none",
                    choices=["none", "auto", "torch_compile", "cuda_graph"])
    rp.add_argument("--bit-depth", dest="bit_depth", type=int, default=8, choices=[8, 16],
                    help="PNG bit depth (DATA-2; 16 = higher-fidelity normalized integer). "
                         "Ignored for .exr output")
    rp.add_argument("--half", action="store_true",
                    help="for .exr output, store HALF instead of FLOAT (compact, ~1e-3 error)")
    sub.add_parser("validate-hw", help="measure whether TEX's Turing-calibrated perf "
                   "gates hold on THIS GPU; emit a shareable report (S-4)")
    hp = sub.add_parser("help", help="show the signature, description and example for a "
                        "built-in function (LANG-4), or list all functions")
    hp.add_argument("fn", nargs="?", help="function name (omit to list every function)")
    bp = sub.add_parser("build", help="validate + type-check a .textool tool manifest, "
                        "report diagnostics, optionally warm-compile it (TOOL-4)")
    bp.add_argument("tool", help="path to a .textool file")
    bp.add_argument("--warm", action="store_true",
                    help="warm-compile the tool at its promoted-param signature (TOOL-3; "
                         "OFF by default — validate-only, TOOL-5)")
    bp.add_argument("--emit", dest="emit", metavar="PATH",
                    help="write a normalized/refreshed manifest to PATH")
    bp.add_argument("--json", dest="as_json", action="store_true",
                    help="emit the build report as JSON")
    return p


def help_fn(args) -> None:
    """`tex help [<fn>]` — print help for one function, or list all (LANG-4). Reads the
    stdlib registry (the single source), not the JS."""
    from .tex_runtime.stdlib import TEXStdlib  # noqa: F401  (populates REGISTRY)
    from .tex_runtime import stdlib_registry as R
    if not args.fn:
        by_cat = {}
        for e in R.help_entries(decode=True):
            by_cat.setdefault(e["category"] or "Other", []).append(e["name"])
        for cat in sorted(by_cat):
            print(f"{cat}: {', '.join(sorted(by_cat[cat]))}")
        return
    e = R.help_lookup(args.fn)
    if e is None:
        sys.exit(f"tex help: no function named '{args.fn}' "
                 f"(try `tex help` to list all)")
    tags = f"  [{', '.join(e['tags'])}]" if e["tags"] else ""
    alias = f"  (aliases: {', '.join(e['aliases'])})" if e["aliases"] else ""
    print(f"{e['sig']}{tags}{alias}")
    if e["desc"]:
        print(f"\n  {e['desc']}")
    if e["example"]:
        print(f"\n  example: {e['example']}")


def build_fn(args) -> None:
    """`tex build <tool.textool>` (TOOL-4) — validate + type-check + report. Validate-only
    by default (TOOL-5): no code is generated unless --warm is passed. Exits non-zero on a
    preflight error so it can gate a CI/install step."""
    import json as _json
    from .tex_tool import load_tool, preflight_tool, tool_warm_keys, TEXToolError
    try:
        manifest = load_tool(args.tool)      # parse -> schema -> language pin -> engine gate
    except TEXToolError as e:
        sys.exit(f"tex build: error: {e}")
    pf = preflight_tool(manifest)
    shape = "fused" if manifest.is_fused else "single-stage"
    if args.as_json:
        print(_json.dumps({
            "name": manifest.name, "tool_version": manifest.tool_version,
            "tex_language": manifest.tex_language, "min_engine": manifest.min_engine,
            "category": manifest.category, "context": manifest.context, "shape": shape,
            "inputs": manifest.inputs, "promoted_params": [p.name for p in manifest.promoted_params],
            "warnings": manifest.warnings, "ok": pf["ok"], "diagnostics": pf["diagnostics"],
            "output_warnings": pf.get("output_warnings", []),
        }, indent=2))
    else:
        print(f"{manifest.name} v{manifest.tool_version}  [{shape}, {manifest.context}]")
        print(f"  language: {manifest.tex_language}   min-engine: {manifest.min_engine}")
        print(f"  inputs: {', '.join(i['name'] for i in manifest.inputs)}")
        print(f"  params: {', '.join(p.name for p in manifest.promoted_params) or '(none)'}")
        for w in manifest.warnings:
            print(f"  warning: {w}")
        for w in pf.get("output_warnings", []):
            print(f"  warning: {w}")
        if pf["ok"]:
            print("  preflight: OK")
        else:
            for d in pf["diagnostics"]:
                print(f"  ERROR {d.get('code', '')}: {d.get('message', '')}")
    if not pf["ok"]:
        sys.exit(1)
    if args.warm:
        from .tex_tool import warm_tool
        print(f"  warm keys: {tool_warm_keys(manifest)}")
        try:
            print(f"  warmed: {warm_tool(manifest)}")
        except Exception as e:
            print(f"  warm-compile skipped: {e}")
    if args.emit:
        with open(args.emit, "w", encoding="utf-8") as fh:
            _json.dump(manifest.to_dict(), fh, indent=2, ensure_ascii=False)
        print(f"  wrote: {args.emit}")


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    # F3 (doc 32): a bad path / syntax error / cook failure must exit with a one-line
    # message, not a raw Python traceback — the flagship host-agnostic runner should fail
    # like a CLI, not dump a stack. Function-local imports keep the module cheap to import
    # as a library (load_image/save_image) without pulling the whole compiler chain.
    from .tex_compiler.diagnostics import TEXCompileError
    from .tex_runtime.interpreter import InterpreterError
    try:
        if args.cmd == "run":
            run(args)
        elif args.cmd == "validate-hw":
            from .tex_validate_hw import main as validate_hw_main
            validate_hw_main()
        elif args.cmd == "help":
            help_fn(args)
        elif args.cmd == "build":
            build_fn(args)
    # TEXCompileError (ENG-4) is what tex_api.compile / tex_engine both raise on a compile
    # failure, and run_program calls compile BEFORE the cook — so without it here every
    # syntax error in `tex run` dumps a stack, breaking the F3 contract this function keeps.
    # No HOST module catches the raw per-phase types any more — the compile raisers wrap them.
    except (OSError, TEXCompileError, InterpreterError, ValueError, RuntimeError) as e:
        # S2 (doc 33): the node appends a machine-readable "\nTEX_DIAG:{json}" blob to some
        # errors for the frontend — strip it from the human CLI message.
        msg = str(e).split("\nTEX_DIAG:", 1)[0].strip()
        sys.exit(f"tex {getattr(args, 'cmd', None) or 'run'}: error: {msg}")


if __name__ == "__main__":
    main()
