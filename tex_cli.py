"""
PORT-3 — `tex run`: a standalone CLI that runs a .tex program on an image file with NO
ComfyUI. Proves TEX is host-agnostic and doubles as a ComfyUI-free regression harness.

I/O is **torchvision-only** (no numpy, no PIL — a bit-exact uint8<->float round-trip is
verified). The cook wires through the REAL `TEXWrangleNode.execute()` (which runs
correctly with ComfyUI absent — the host seam degrades to NullHostServices).

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
    """PNG/JPG -> [1, H, W, 3] float32 in [0,1] (ComfyUI IMAGE layout), torchvision-only."""
    from torchvision.io import read_file, decode_image
    img = decode_image(read_file(path))          # [C, H, W] uint8
    if img.shape[0] == 1:
        img = img.expand(3, -1, -1)              # grayscale -> RGB
    img = img[:3]                                # drop alpha
    t = img.to(torch.float32) / 255.0            # [C, H, W] in [0,1]
    return t.permute(1, 2, 0).unsqueeze(0).to(device)   # [1, H, W, 3]


def save_image(tensor: torch.Tensor, path: str) -> None:
    """[1, H, W, C] float [0,1] (or [1,H,W]) -> PNG. Round (not truncate) for a bit-exact
    round-trip: round(u/255*255) == u."""
    from torchvision.io import encode_png, write_file
    t = tensor.detach().float().cpu()
    if t.dim() == 4:
        t = t[0]
    if t.dim() == 2:                              # [H, W] mask -> [H, W, 1]
        t = t.unsqueeze(-1)
    t = t[..., :3]
    if t.shape[-1] == 1:
        t = t.expand(-1, -1, 3)
    u8 = (t.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(2, 0, 1)  # [C, H, W]
    write_file(path, encode_png(u8))


def run_program(code: str, image: torch.Tensor, device="cpu", precision="fp32",
                compile_mode="none") -> torch.Tensor:
    """Run a TEX program on one image through the real node; return the primary IMAGE
    output (the first vec3/vec4 output, else the first output). Compiles ONCE — the
    compiler's own `referenced`/`assigned` give the input bindings and output types (no
    regex to drift from the grammar); the node then re-infers types from the tensor."""
    from .tex_node import TEXWrangleNode
    from .tex_api import compile as tex_compile
    from .tex_compiler.types import TEXType

    prog = tex_compile(code, {})            # authoritative binding sets from the compiler
    inputs = set(prog.referenced) - set(prog.assigned)
    kwargs = {name: image for name in inputs}
    kwargs.update(code=code, device=device, precision=precision, compile_mode=compile_mode)
    out = TEXWrangleNode.execute(**kwargs)
    results = out if isinstance(out, tuple) else out.result

    names = sorted(prog.assigned.keys())
    idx = next((i for i, n in enumerate(names)
                if prog.assigned[n] in (TEXType.VEC3, TEXType.VEC4)), 0)
    return results[idx]


def run(args) -> None:
    _require_torchvision()
    with open(args.program, encoding="utf-8") as f:
        code = f.read()
    image = load_image(args.in_path, args.device)
    result = run_program(code, image, device=args.device, precision=args.precision,
                         compile_mode=args.compile_mode)
    save_image(result, args.out_path)
    print(f"tex run: {args.program} on {args.in_path} -> {args.out_path} "
          f"({args.device}/{args.precision})")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tex", description="TEX Wrangle standalone runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run", help="run a .tex program on an image file")
    rp.add_argument("program", help="path to a .tex program")
    rp.add_argument("--in", dest="in_path", required=True, help="input image (png/jpg)")
    rp.add_argument("--out", dest="out_path", required=True, help="output PNG")
    rp.add_argument("--device", default="cpu", help="cpu | cuda")
    rp.add_argument("--precision", default="fp32", choices=["fp32", "auto", "fp16"])
    rp.add_argument("--compile-mode", dest="compile_mode", default="none",
                    choices=["none", "auto", "torch_compile", "cuda_graph"])
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.cmd == "run":
        run(args)


if __name__ == "__main__":
    main()
