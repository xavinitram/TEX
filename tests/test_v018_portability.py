"""
v0.18.0 portability ladder (PORT-1 HostServices seam + import lint; PORT-2 API facade).
"""
from helpers import *
import re

_PKG = Path(__file__).resolve().parent.parent
_IMPORT_RE = re.compile(
    r"^\s*(import\s+comfy\.model_management"        # import comfy.model_management [as mm]
    r"|from\s+comfy\.model_management"              # from comfy.model_management import ...
    r"|from\s+comfy\s+import\s+[^#\n]*\bmodel_management\b)",  # doc 33: from comfy import model_management [as mm]
    re.M)


def test_port1_import_lint(r: SubTestResult):
    print("\n--- PORT-1: comfy.model_management import is pinned to host.py ---")
    offenders = []
    for path in _PKG.rglob("*.py"):
        rel = path.relative_to(_PKG).as_posix()
        if rel == "tex_runtime/host.py" or "/tests/" in f"/{rel}" or rel.startswith("tests/"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for m in _IMPORT_RE.finditer(text):
            line = text[:m.start()].count("\n") + 1
            offenders.append(f"{rel}:{line}")
    if offenders:
        r.fail("PORT-1 import lint",
               "comfy.model_management imported outside host.py (re-scatter): "
               + ", ".join(offenders))
    else:
        r.ok("comfy.model_management is imported ONLY in tex_runtime/host.py (seam intact)")


def test_port1_host_services(r: SubTestResult):
    print("\n--- PORT-1: HostServices Null contract + Comfy forwarding ---")
    from TEX_Wrangle.tex_runtime import host
    fails = []

    # NullHostServices: unknown free memory, no-op hooks, torch-only OOM detection
    null = host.NullHostServices()
    if null.get_free_memory("cuda") is not None:
        fails.append("Null.get_free_memory should be None")
    null.free_memory(1, "cuda"); null.soft_empty_cache()  # must not raise
    if null.is_oom(ValueError("x")) is not False:
        fails.append("Null.is_oom(non-OOM) should be False")

    # ComfyHostServices forwards to the module (a fake stands in for comfy.model_management)
    class _FakeMM:
        def __init__(self): self.freed = None; self.emptied = False
        def get_free_memory(self, dev): return 4242
        def free_memory(self, amt, dev): self.freed = (amt, dev)
        def soft_empty_cache(self): self.emptied = True
        def is_oom(self, e): return isinstance(e, KeyError)
    fake = _FakeMM()
    comfy = host.ComfyHostServices(fake)
    if comfy.get_free_memory("cuda") != 4242:
        fails.append("Comfy.get_free_memory did not forward")
    comfy.free_memory(99, "cuda")
    if fake.freed != (99, "cuda"):
        fails.append("Comfy.free_memory did not forward")
    comfy.soft_empty_cache()
    if not fake.emptied:
        fails.append("Comfy.soft_empty_cache did not forward")
    if comfy.is_oom(KeyError()) is not True or comfy.is_oom(ValueError()) is not False:
        fails.append("Comfy.is_oom did not forward")

    # a host whose API raises degrades to Null behaviour (never propagates)
    class _BrokenMM:
        def get_free_memory(self, dev): raise RuntimeError("boom")
        def free_memory(self, amt, dev): raise RuntimeError("boom")
        def soft_empty_cache(self): raise RuntimeError("boom")
        def is_oom(self, e): raise RuntimeError("boom")
    broken = host.ComfyHostServices(_BrokenMM())
    try:
        if broken.get_free_memory("cuda") is not None:
            fails.append("broken host free-memory should degrade to None")
        broken.free_memory(1, "cuda"); broken.soft_empty_cache()  # must not raise
        broken.is_oom(ValueError())  # must not raise
    except Exception as e:
        fails.append(f"broken host propagated: {e}")

    # resolver + override
    host.set_host_services(null)
    if host.get_host_services() is not null:
        fails.append("set_host_services override not honored")
    host.reset_host_services()

    if fails:
        r.fail("PORT-1 host services", "; ".join(fails))
    else:
        r.ok("Null no-ops (torch-only OOM); Comfy forwards; a broken host degrades; "
             "resolver override honored")


def test_port2_facade(r: SubTestResult):
    print("\n--- PORT-2: public API facade (compile/execute) ---")
    from TEX_Wrangle import tex_api
    from TEX_Wrangle.tex_cache import get_cache
    from TEX_Wrangle.tex_runtime.interpreter import Interpreter
    fails = []
    code = "vec3 c = @A.rgb; @OUT = vec4(pow(c, vec3(0.4545)) * 1.1, 1.0);"
    bt = {"A": TEXType.VEC3}

    # (a) compile() is structurally the compiler's 6-tuple, named
    prog = tex_api.compile(code, bt)
    direct = get_cache().compile_tex(code, bt)
    if prog.assigned != direct[3]:                 # {name: TEXType} — comparable
        fails.append(f"Program.assigned {prog.assigned} != {direct[3]}")
    if prog.used_builtins != direct[5]:
        fails.append("Program.used_builtins differs from compile_tex")
    if prog.source != code:
        fails.append("Program.source not preserved")
    if prog.ast is None or prog.type_map is None:
        fails.append("Program.ast/type_map missing")

    # (b) execute() is bit-for-bit vs Interpreter.execute (same call path, tol=0)
    img = make_img(1, 16, 16, 3, seed=9)
    facade = tex_api.execute(prog, {"A": img}, device="cpu")["OUT"]
    ref = Interpreter().execute(direct[0], {"A": img}, direct[1], device="cpu",
                                output_names=["OUT"], precision="fp32",
                                used_builtins=direct[5])["OUT"]
    if (facade.float() - ref.float()).abs().max().item() != 0.0:
        fails.append("execute() diverged from Interpreter.execute (not bit-for-bit)")

    if fails:
        r.fail("PORT-2 facade", "; ".join(fails))
    else:
        r.ok("compile() wraps the 6-tuple (named); execute() bit-for-bit vs Interpreter")


def test_port3_cli(r: SubTestResult):
    print("\n--- PORT-3: tex run CLI (torchvision I/O, ComfyUI-free) ---")
    try:
        import torchvision  # noqa: F401
        from torchvision.io import decode_image  # noqa: F401
    except Exception:
        r.ok("PORT-3 CLI (no torchvision, SKIPPED)")
        return
    from TEX_Wrangle import tex_cli, tex_api
    from TEX_Wrangle.tex_node import _V3_AVAILABLE
    fails = []
    tmp = Path(tempfile.mkdtemp())
    in_png, out_png = str(tmp / "in.png"), str(tmp / "out.png")
    try:
        # (a) I/O round-trip is BIT-EXACT after the first quantization (uint8<->float)
        img = make_img(1, 32, 48, 3, seed=5)
        tex_cli.save_image(img, in_png)
        load1 = tex_cli.load_image(in_png, "cpu")
        tex_cli.save_image(load1, out_png)
        load2 = tex_cli.load_image(out_png, "cpu")
        if (load1 - load2).abs().max().item() != 0.0:
            fails.append("uint8<->float round-trip not bit-exact")
        if (load1 - img).abs().max().item() > 1.0 / 255 + 1e-6:
            fails.append("load(save(img)) lost more than a quantization step")

        # (b) the CLI runs a real example ComfyUI-free (v3 API absent in this env) and its
        #     output matches a direct execute within 1/255 (quantization only)
        _EX = Path(__file__).resolve().parent.parent / "examples"
        code = (_EX / "grayscale.tex").read_text(encoding="utf-8")
        cli_out = tex_cli.run_program(code, load1, device="cpu")   # node path
        prog = tex_api.compile(code, {"image": TEXType.VEC3})
        direct = tex_api.execute(prog, {"image": load1}, device="cpu")["gray_image"]
        if (cli_out.float() - direct.float()).abs().max().item() > 1.0 / 255:
            fails.append("CLI output diverged from direct execute beyond quantization")
        # writing the output produces a real PNG file
        tex_cli.save_image(cli_out, out_png)
        if not Path(out_png).exists() or Path(out_png).stat().st_size == 0:
            fails.append("CLI did not write a PNG")

        # (c) a small corpus of SINGLE-input examples runs clean through the CLI core
        # (ComfyUI-free harness). Multi-input examples (brightness_contrast wires
        # @contrast/@exposure) now correctly error via the CLI — that path is asserted in
        # test_port3_cli_edges (doc 33 S1), so the corpus here stays single-input.
        for ex in ("invert.tex", "grayscale.tex"):
            p = _EX / ex
            if p.exists():
                try:
                    tex_cli.run_program(p.read_text(encoding="utf-8"), load1, device="cpu")
                except Exception as e:
                    fails.append(f"{ex}: {type(e).__name__}: {e}")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    note = "" if _V3_AVAILABLE else " (v3 API absent — proven ComfyUI-free)"
    if fails:
        r.fail("PORT-3 CLI", "; ".join(fails))
    else:
        r.ok("bit-exact uint8<->float I/O; CLI matches direct execute within 1/255; "
             "corpus runs clean" + note)


def test_port3_cli_edges(r: SubTestResult):
    print("\n--- PORT-3: tex run CLI edges (doc 33 B1 mask-save, S1 multi-input) ---")
    try:
        from torchvision.io import decode_image  # noqa: F401
    except Exception:
        r.ok("PORT-3 CLI edges (no torchvision, SKIPPED)")
        return
    from torchvision.io import decode_image
    from TEX_Wrangle import tex_cli
    fails = []
    tmp = Path(tempfile.mkdtemp())
    try:
        # B1: a [1,H,W] MASK output round-trips to (H,W) — NOT a width-sliced 1px garbage PNG.
        out = tex_cli.run_program("@OUT = luma(@image);", make_img(1, 17, 23, 3, seed=2), device="cpu")
        if out.dim() != 3:
            fails.append(f"mask program output dim {out.dim()} (expected 3, [1,H,W])")
        mp = str(tmp / "m.png")
        tex_cli.save_image(out, mp)
        mh, mw = (int(x) for x in decode_image(mp).shape[1:])
        if (mh, mw) != (17, 23):
            fails.append(f"mask PNG shape ({mh},{mw}) != (17,23) — B1 width-slice corruption")
        # an IMAGE output must remain correct (no regression from the dim==3 branch)
        ip = str(tmp / "i.png")
        tex_cli.save_image(make_img(1, 11, 13, 3, seed=3), ip)
        if tuple(int(x) for x in decode_image(ip).shape[1:]) != (11, 13):
            fails.append("image PNG shape regressed")
        # S1: a two-input program must ERROR (not silently alias @A == @B to one --in image).
        try:
            tex_cli.run_program("@OUT = vec4(mix(@A.rgb,@B.rgb,0.5),1.0);",
                                make_img(1, 8, 8, 3), device="cpu")
            fails.append("multi-input program did NOT error (silent single-image aliasing)")
        except ValueError as e:
            if "2 inputs" not in str(e):
                fails.append(f"multi-input error message unclear: {e}")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    if fails:
        r.fail("PORT-3 CLI edges", "; ".join(fails))
    else:
        r.ok("mask output round-trips to (H,W) [B1]; multi-input errors clearly [S1]")


def test_port2_program_shape(r: SubTestResult):
    print("\n--- PORT-2: Program field-name stability (canary) ---")
    from TEX_Wrangle.tex_api import Program
    import dataclasses
    # the CLI + future hosts depend on these names — a rename is a breaking change
    expected = ["ast", "type_map", "referenced", "assigned", "params",
                "used_builtins", "source"]
    got = [f.name for f in dataclasses.fields(Program)]
    if got != expected:
        r.fail("PORT-2 Program shape",
               f"Program fields changed: {got} != {expected} (breaks the CLI/host API)")
    else:
        r.ok(f"Program fields stable: {', '.join(expected)}")


def test_hw2_multi_gpu_device_context(r: SubTestResult):
    print("\n--- HW-2: multi-GPU device-index correctness ---")
    from TEX_Wrangle.tex_runtime import graphed as G
    fails = []
    # (a) the capture key includes the device INDEX — a cuda:0 and a cuda:1 cook of the
    #     same program never collide on one captured graph (works CPU-side; the device
    #     objects are just labels).
    k0 = G._capture_key("fp", torch.device("cuda:0"), "fp32", {}, ["OUT"], 0)
    k1 = G._capture_key("fp", torch.device("cuda:1"), "fp32", {}, ["OUT"], 0)
    if k0 == k1:
        fails.append("capture key does not distinguish device index")

    # (b) RNG-recovery targets the COOK's device index (mock torch.cuda.device to record
    #     the target and abort before any real CUDA op). Recovering on the wrong device
    #     would leave the real generator poisoned and trip the process-wide kill switch.
    entered = []
    real = torch.cuda.device

    class _Rec:
        def __init__(self, idx): entered.append(idx)
        def __enter__(self): raise RuntimeError("abort before CUDA ops")
        def __exit__(self, *a): return False
    torch.cuda.device = lambda idx: _Rec(idx)
    try:
        G._recover_from_capture_failure(1)
    except Exception:
        pass
    finally:
        torch.cuda.device = real
    if 1 not in entered:
        fails.append(f"recovery did not target device index 1 (targets: {set(entered)})")

    # (c) _cuda_headroom_ok now accepts a full device (queries the cook's index, not GPU 0)
    from TEX_Wrangle.tex_runtime.compiled import _cuda_headroom_ok
    if _cuda_headroom_ok("cpu") is not True:
        fails.append("_cuda_headroom_ok('cpu') should be True")
    if not isinstance(_cuda_headroom_ok(torch.device("cuda:0")), bool):
        fails.append("_cuda_headroom_ok(device) did not accept a full device")

    if fails:
        r.fail("HW-2 multi-GPU", "; ".join(fails))
    else:
        r.ok("capture key carries device index; RNG-recovery pins the cook's device; "
             "_cuda_headroom_ok takes the cook device (index-0 unchanged on a single GPU)")


def test_port3_16bit_png(r: SubTestResult):
    print("\n--- PORT-3: 16-bit PNG normalized by bit depth, not 257x too bright (audit) ---")
    try:
        import torchvision.io as tvio
    except Exception:
        r.ok("PORT-3 16-bit (no torchvision, SKIPPED)")
        return
    from TEX_Wrangle import tex_cli
    saved_read, saved_decode = tvio.read_file, tvio.decode_image
    fails = []
    try:
        tvio.read_file = lambda p: b""
        # uint16 mid-gray 32768/65535 ~= 0.5 (the bug divided by 255 -> ~128, 257x bright)
        tvio.decode_image = lambda data: torch.full((3, 8, 8), 32768, dtype=torch.uint16)
        mid16 = tex_cli.load_image("x.png", "cpu").mean().item()
        if not (0.49 < mid16 < 0.51):
            fails.append(f"uint16 mid-gray -> {mid16:.3f} (expected ~0.5)")
        # uint8 still normalized by 255
        tvio.decode_image = lambda data: torch.full((3, 8, 8), 128, dtype=torch.uint8)
        mid8 = tex_cli.load_image("x.png", "cpu").mean().item()
        if not (0.49 < mid8 < 0.51):
            fails.append(f"uint8 mid-gray -> {mid8:.3f} (expected ~0.5)")
        # an unsupported dtype errors clearly (not silently wrong)
        tvio.decode_image = lambda data: torch.full((3, 8, 8), 0.5, dtype=torch.float32)
        try:
            tex_cli.load_image("x.png", "cpu")
            fails.append("float dtype did not raise")
        except ValueError:
            pass
    finally:
        tvio.read_file, tvio.decode_image = saved_read, saved_decode
    if fails:
        r.fail("PORT-3 16-bit", "; ".join(fails))
    else:
        r.ok("uint16 normalized by 65535 (~0.5); uint8 by 255; unsupported dtype errors")


def test_hw4_cpu_threads(r: SubTestResult):
    print("\n--- HW-4: TEX_CPU_THREADS opt-in knob ---")
    from TEX_Wrangle.tex_node import _apply_cpu_threads_env
    saved_env = os.environ.get("TEX_CPU_THREADS")
    saved_threads = torch.get_num_threads()
    fails = []
    try:
        # unset -> untouched
        os.environ.pop("TEX_CPU_THREADS", None)
        before = torch.get_num_threads()
        _apply_cpu_threads_env()
        if torch.get_num_threads() != before:
            fails.append("unset TEX_CPU_THREADS changed the thread count (must never auto-set)")
        # set -> reflected (use a small, safe target distinct from `before`)
        target = 2 if before != 2 else 3
        os.environ["TEX_CPU_THREADS"] = str(target)
        _apply_cpu_threads_env()
        if torch.get_num_threads() != target:
            fails.append(f"set TEX_CPU_THREADS={target} not reflected "
                         f"(got {torch.get_num_threads()})")
    finally:
        torch.set_num_threads(saved_threads)
        if saved_env is None:
            os.environ.pop("TEX_CPU_THREADS", None)
        else:
            os.environ["TEX_CPU_THREADS"] = saved_env
    if fails:
        r.fail("HW-4 cpu threads", "; ".join(fails))
    else:
        r.ok("unset -> untouched (never auto-set); set -> reflected")
