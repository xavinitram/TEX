"""
Aliasing / copy-on-write regression tests.

Covers the interpreter's clone-on-first-write elision for channel and
array-element assignment, the persistent literal cache, the per-execution
scatter-buffer ownership model, and the stdlib clamp / grid-buffer fixes.
"""
from helpers import *
from TEX_Wrangle.tex_runtime import stdlib as _stdlib_mod


def _compile(code: str, bindings: dict):
    """Lex/parse/typecheck; returns (program, type_map, output_names)."""
    tokens = Lexer(code).tokenize()
    program = Parser(tokens, source=code).parse()
    binding_types = {name: _infer_binding_type(val) for name, val in bindings.items()}
    checker = TypeChecker(binding_types=binding_types, source=code)
    type_map = checker.check(program)
    return program, type_map, sorted(checker.assigned_bindings.keys())


def test_cow_channel_array_writes(r: SubTestResult):
    print("\n--- Copy-on-write channel/array assignment ---")
    img = make_img(1, 4, 4, 3)

    # Overlapping self-swizzle must read pre-write values (c.b gets OLD c.g)
    try:
        result = compile_and_run("""
            vec4 c = vec4(0.1, 0.2, 0.3, 0.4);
            c = c * 1.0;
            c.gb = c.rg;
            @OUT = c;
        """, {"A": img})
        v = result[0, 0, 0]
        assert abs(v[0].item() - 0.1) < 1e-6 and abs(v[1].item() - 0.1) < 1e-6
        assert abs(v[2].item() - 0.2) < 1e-6 and abs(v[3].item() - 0.4) < 1e-6
        r.ok("overlapping self-swizzle c.gb = c.rg")
    except Exception as e:
        r.fail("overlapping self-swizzle c.gb = c.rg", str(e))

    # Swap must also see simultaneous-read semantics
    try:
        result = compile_and_run("""
            vec4 c = vec4(0.1, 0.2, 0.3, 0.4);
            c = c * 1.0;
            c.rg = c.gr;
            @OUT = c;
        """, {"A": img})
        v = result[0, 0, 0]
        assert abs(v[0].item() - 0.2) < 1e-6 and abs(v[1].item() - 0.1) < 1e-6
        r.ok("channel swap c.rg = c.gr")
    except Exception as e:
        r.fail("channel swap c.rg = c.gr", str(e))

    # A plain alias must not see later channel writes to its source
    try:
        result = compile_and_run("""
            vec4 c = vec4(0.5, 0.5, 0.5, 0.5);
            c = c + 0.0;
            vec4 d = c;
            c.r = 0.9;
            @a = d;
            @b = c;
        """, {"A": img})
        assert abs(result["a"][0, 0, 0, 0].item() - 0.5) < 1e-6, "alias d was corrupted"
        assert abs(result["b"][0, 0, 0, 0].item() - 0.9) < 1e-6
        r.ok("alias survives channel write to source")
    except Exception as e:
        r.fail("alias survives channel write to source", str(e))

    # Fill loop: clone-once-then-in-place must produce sequential semantics
    try:
        result = compile_and_run("""
            float arr[8];
            for (int i = 0; i < 8; i = i + 1) { arr[i] = float(i) * 2.0; }
            @OUT = vec3((arr[0] + arr[3] + arr[7]) / 30.0);
        """, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - (0.0 + 6.0 + 14.0) / 30.0) < 1e-5
        r.ok("array fill loop")
    except Exception as e:
        r.fail("array fill loop", str(e))

    # Recurrence reading the previous element
    try:
        result = compile_and_run("""
            float arr[4];
            arr[0] = 1.0;
            for (int i = 1; i < 4; i = i + 1) { arr[i] = arr[i - 1] * 0.5; }
            @OUT = vec3(arr[3]);
        """, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 0.125) < 1e-6
        r.ok("array recurrence arr[i] = arr[i-1]*0.5")
    except Exception as e:
        r.fail("array recurrence arr[i] = arr[i-1]*0.5", str(e))

    # Channel writes inside a spatial if/else
    try:
        result = compile_and_run("""
            vec4 c = vec4(0.0, 0.0, 0.0, 1.0);
            c = c * 1.0;
            if (u > 0.5) { c.r = 1.0; } else { c.g = 1.0; }
            @OUT = c;
        """, {"A": img})
        left = result[0, 0, 0]   # u=0 -> else branch
        right = result[0, 0, 3]  # u=1 -> then branch
        assert abs(left[0].item()) < 1e-6 and abs(left[1].item() - 1.0) < 1e-6
        assert abs(right[0].item() - 1.0) < 1e-6 and abs(right[1].item()) < 1e-6
        r.ok("channel writes in spatial if/else")
    except Exception as e:
        r.fail("channel writes in spatial if/else", str(e))


def test_cow_binding_and_function_holes(r: SubTestResult):
    print("\n--- Aliasing holes: bindings, user functions ---")
    img = make_img(1, 4, 4, 3)

    # Output binding must keep the pre-write value after a channel write
    try:
        result = compile_and_run("""
            vec4 c = vec4(0.2, 0.2, 0.2, 1.0);
            c = c + 0.0;
            @OUT = c;
            c.r = 0.9;
        """, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 0.2) < 1e-6, "@OUT was mutated by a later channel write"
        r.ok("@OUT = c; c.r = ... keeps output intact")
    except Exception as e:
        r.fail("@OUT = c; c.r = ... keeps output intact", str(e))

    # Pre-existing hole: @OUT = c; c = c + 1.0 (in-place binop after binding)
    try:
        result = compile_and_run("""
            vec4 c = vec4(0.2, 0.2, 0.2, 1.0);
            c = c + 0.0;
            @OUT = c;
            c = c + 1.0;
        """, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 0.2) < 1e-6, "@OUT was mutated by a later in-place op"
        r.ok("@OUT = c; c = c + 1.0 keeps output intact")
    except Exception as e:
        r.fail("@OUT = c; c = c + 1.0 keeps output intact", str(e))

    # Multi-output: first output must differ from second after interleaved write
    try:
        result = compile_and_run("""
            vec4 c = vec4(0.3, 0.3, 0.3, 1.0);
            c = c * 1.0;
            @outA = c;
            c.r = 0.8;
            @outB = c;
        """, {"A": img})
        assert abs(result["outA"][0, 0, 0, 0].item() - 0.3) < 1e-6
        assert abs(result["outB"][0, 0, 0, 0].item() - 0.8) < 1e-6
        r.ok("multi-output interleaved channel write")
    except Exception as e:
        r.fail("multi-output interleaved channel write", str(e))

    # Pre-existing hole: param name shadowing a ready caller variable
    try:
        result = compile_and_run("""
            float f(float x) {
                x = x + 1.0;
                return x;
            }
            float x = 2.0;
            x = x + 0.0;
            float y = f(x);
            @OUT = vec3(x / 4.0);
        """, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 0.5) < 1e-5, "caller variable mutated through parameter"
        r.ok("user function cannot mutate caller var via shadowed param")
    except Exception as e:
        r.fail("user function cannot mutate caller var via shadowed param", str(e))

    # Function passthrough return aliases its argument
    try:
        result = compile_and_run("""
            vec4 pass(vec4 p) { return p; }
            vec4 x = vec4(0.3, 0.3, 0.3, 1.0);
            x = x + 0.0;
            vec4 y = pass(x);
            x.r = 0.9;
            @a = x;
            @b = y;
        """, {"A": img})
        assert abs(result["a"][0, 0, 0, 0].item() - 0.9) < 1e-6
        assert abs(result["b"][0, 0, 0, 0].item() - 0.3) < 1e-6, "passthrough alias was corrupted"
        r.ok("function passthrough alias protected")
    except Exception as e:
        r.fail("function passthrough alias protected", str(e))

    # A variable literally named OUT must not authorize in-place @OUT.r writes
    try:
        result = compile_and_run("""
            float OUT = 1.0;
            OUT = OUT + 0.0;
            @OUT = vec4(0.1, 0.2, 0.3, 0.4);
            @OUT.r = 0.9;
            @keep = OUT;
        """, {"A": img})
        v = result["OUT"].reshape(-1)  # constant program -> non-spatial [4] vec
        assert abs(v[0].item() - 0.9) < 1e-6 and abs(v[1].item() - 0.2) < 1e-6
        assert abs(result["keep"].reshape(-1)[0].item() - 1.0) < 1e-6
        r.ok("variable named OUT does not leak readiness to @OUT")
    except Exception as e:
        r.fail("variable named OUT does not leak readiness to @OUT", str(e))


def test_literal_cache_persistence(r: SubTestResult):
    print("\n--- Persistent literal cache ---")
    img = make_img(1, 4, 4, 3)

    # Cached literal must survive a user function mutating its parameter twice
    try:
        result = compile_and_run("""
            float f(float x) {
                x = x + 1.0;
                return x;
            }
            float a = f(2.0);
            float b = f(2.0);
            @OUT = vec3(2.0 / 2.0, a / 4.0, b / 4.0);
        """, {"A": img})
        v = result[0, 0, 0]
        assert abs(v[0].item() - 1.0) < 1e-6, f"literal 2.0 was corrupted (got {v[0].item() * 2.0})"
        assert abs(v[1].item() - 0.75) < 1e-6 and abs(v[2].item() - 0.75) < 1e-6
        r.ok("literal survives repeated param mutation")
    except Exception as e:
        r.fail("literal survives repeated param mutation", str(e))

    # Same instance, two executions: cache persists and results are identical
    try:
        code = "@OUT = @A * 0.25 + 0.5;"
        program, type_map, out_names = _compile(code, {"A": img})
        interp = Interpreter()
        r1 = interp.execute(program, {"A": img}, type_map, device="cpu",
                            output_names=out_names, source=code)
        assert len(interp._literal_cache) > 0, "literal cache was cleared between runs"
        r2 = interp.execute(program, {"A": img}, type_map, device="cpu",
                            output_names=out_names, source=code)
        assert torch.equal(r1["OUT"], r2["OUT"])
        r.ok("literal cache persists across executions")
    except Exception as e:
        r.fail("literal cache persists across executions", str(e))


def test_scatter_ownership(r: SubTestResult):
    print("\n--- Scatter-write buffer ownership ---")

    # Scatter into an aliased input binding must not mutate the caller's tensor
    try:
        img = make_img(1, 4, 4, 3)
        pristine = img.clone()
        _ = compile_and_run("""
            @OUT = @A;
            @OUT[0.0, 0.0] = vec3(9.0);
        """, {"A": img})
        assert torch.equal(img, pristine), "input tensor was mutated by a scatter write"
        r.ok("scatter never mutates the input binding tensor")
    except Exception as e:
        r.fail("scatter never mutates the input binding tensor", str(e))

    # Scatter through a 1x1 expand-view of a literal must not poison the cache
    try:
        img1 = make_img(1, 1, 1, 3)
        result = compile_and_run("""
            @OUT = vec3(0.5).r;
            @OUT[0.0, 0.0] = 9.0;
            @keep = 0.5;
        """, {"A": img1})
        keep = result["keep"]
        kv = keep[0, 0, 0].item() if keep.dim() >= 3 else float(keep)
        assert abs(kv - 0.5) < 1e-6, f"literal 0.5 was corrupted via expand-view scatter (got {kv})"
        r.ok("1x1 expand-view scatter does not poison the literal cache")
    except Exception as e:
        r.fail("1x1 expand-view scatter does not poison the literal cache", str(e))

    # Scatter through a builtin view must not poison the builtins cache
    try:
        code_a = "@OUT = ix;\n@OUT[1.0, 0.0] = 99.0;"
        code_b = "@OUT = ix;"
        img_row = make_img(1, 1, 4, 3)
        prog_a, tm_a, out_a = _compile(code_a, {"A": img_row})
        prog_b, tm_b, out_b = _compile(code_b, {"A": img_row})
        interp = Interpreter()
        interp.execute(prog_a, {"A": img_row}, tm_a, device="cpu",
                       output_names=out_a, source=code_a)
        rb = interp.execute(prog_b, {"A": img_row}, tm_b, device="cpu",
                            output_names=out_b, source=code_b)
        assert rb["OUT"].max().item() <= 3.0, "builtin ix grid was corrupted by a scatter write"
        r.ok("builtin coordinate grid survives scatter writes")
    except Exception as e:
        r.fail("builtin coordinate grid survives scatter writes", str(e))

    # Batched scatter uses the shared flat batch index
    try:
        img2 = make_img(2, 4, 4, 3)
        result = compile_and_run("""
            @OUT = @A;
            @OUT[ix, iy] = vec3(0.5);
        """, {"A": img2})
        assert torch.allclose(result, torch.full_like(result, 0.5))
        r.ok("batched scatter (B=2) full-image write")
    except Exception as e:
        r.fail("batched scatter (B=2) full-image write", str(e))


def test_clamp_and_gridbuf(r: SubTestResult):
    print("\n--- stdlib clamp / grid buffer ---")
    fns = TEXStdlib.get_functions()
    clamp = fns["clamp"]

    # 0-dim tensor bounds must equal Python-number bounds
    try:
        torch.manual_seed(3)
        x = torch.randn(1, 4, 4, 3) * 2.0
        a = clamp(x, 0.0, 1.0)
        b = clamp(x, torch.tensor(0.0), torch.tensor(1.0))
        assert torch.equal(a, b)
        # NaN propagates identically
        xn = x.clone(); xn[0, 0, 0, 0] = float("nan")
        an = clamp(xn, 0.0, 1.0); bn = clamp(xn, torch.tensor(0.0), torch.tensor(1.0))
        assert torch.isnan(an[0, 0, 0, 0]) and torch.isnan(bn[0, 0, 0, 0])
        assert torch.equal(an[0, 0, 0, 1:], bn[0, 0, 0, 1:])
        # lo > hi resolves to hi in both forms
        assert torch.equal(clamp(x, 1.0, 0.0), clamp(x, torch.tensor(1.0), torch.tensor(0.0)))
        # int-dtype 0-dim bounds
        assert torch.equal(clamp(x, 0.0, 1.0),
                           clamp(x, torch.tensor(0, dtype=torch.int64), torch.tensor(1, dtype=torch.int64)))
        r.ok("clamp: tensor bounds match number bounds")
    except Exception as e:
        r.fail("clamp: tensor bounds match number bounds", str(e))

    # On CUDA, 0-dim tensor bounds must not call _to_float (a GPU sync per call)
    try:
        if torch.cuda.is_available():
            orig = _stdlib_mod._to_float
            def _boom(x):
                raise AssertionError("_to_float called — hidden sync on the clamp hot path")
            _stdlib_mod._to_float = _boom
            try:
                _ = clamp(torch.rand(2, 2, device="cuda"),
                          torch.tensor(0.0, device="cuda"), torch.tensor(1.0, device="cuda"))
            finally:
                _stdlib_mod._to_float = orig
            r.ok("clamp: no .item() sync for tensor bounds on CUDA")
        else:
            r.ok("clamp: no .item() sync for tensor bounds on CUDA (no GPU, SKIPPED)")
    except Exception as e:
        r.fail("clamp: no .item() sync for tensor bounds on CUDA", str(e))

    # Grid buffer keeps its allocate-and-hold pattern (see the PERF TRAP note
    # in _get_grid_buf: true reuse measured ~30% slower on CPU)
    try:
        sample = fns["sample"]
        img = make_img(1, 8, 8, 3)
        u = torch.rand(1, 8, 8)
        v = torch.rand(1, 8, 8)
        _stdlib_mod._grid_buf.clear()
        with torch.inference_mode():
            sample(img, u, v)
            sample(img, u, v)
        assert len(_stdlib_mod._grid_buf) == 1, "grid buffer dict should hold one entry per shape"
        r.ok("grid buffer allocate-and-hold pattern intact")
    except Exception as e:
        r.fail("grid buffer allocate-and-hold pattern intact", str(e))

    # Mixed-mode regression: populate inside inference mode, call outside
    try:
        sample = fns["sample"]
        img = make_img(1, 8, 8, 3)
        u = torch.rand(1, 8, 8)
        v = torch.rand(1, 8, 8)
        with torch.inference_mode():
            sample(img, u, v)
        _ = sample(img, u, v)  # must not raise "Inplace update to inference tensor"
        r.ok("grid buffer safe when called outside inference mode")
    except Exception as e:
        r.fail("grid buffer safe when called outside inference mode", str(e))


def test_noise_backend_gate(r: SubTestResult):
    print("\n--- Noise inductor gate (per-device) ---")
    from TEX_Wrangle.tex_runtime import noise as _noise

    # CUDA availability of inductor must be gated on Triton, not MSVC
    try:
        import importlib.util as _ilu
        has_triton = _ilu.find_spec("triton") is not None
        got = _noise._can_inductor_compile(torch.device("cuda"))
        assert got == has_triton, f"cuda gate said {got}, triton present={has_triton}"
        r.ok("inductor gate: cuda keyed on triton")
    except Exception as e:
        r.fail("inductor gate: cuda keyed on triton", str(e))

    # fbm must execute on CUDA even when the compile tier is unavailable
    try:
        if torch.cuda.is_available():
            x = torch.rand(1, 32, 32, device="cuda")
            y = torch.rand(1, 32, 32, device="cuda")
            for _ in range(6):  # enough calls to cross the upgrade threshold
                out = _noise._fbm2d(x, y, 4)
            assert torch.isfinite(out).all()
            r.ok("fbm runs on CUDA without Triton (tier fallback)")
        else:
            r.ok("fbm runs on CUDA without Triton (no GPU, SKIPPED)")
    except Exception as e:
        r.fail("fbm runs on CUDA without Triton (tier fallback)", str(e))


def test_fp16_guards(r: SubTestResult):
    print("\n--- fp16 zero-divisor guards / scalar-mode dtype ---")
    img = make_img(1, 4, 4, 3)

    # mod(x, 0) must stay finite under fp16 (1e-8 underflows to 0 in fp16)
    try:
        code = "@OUT = mod(@A, @B);"
        zeros = torch.zeros(1, 4, 4, 3)
        program, type_map, out_names = _compile(code, {"A": img, "B": zeros})
        interp = Interpreter()
        result = interp.execute(program, {"A": img, "B": zeros}, type_map,
                                device="cpu", output_names=out_names,
                                precision="fp16", source=code)
        assert torch.isfinite(result["OUT"]).all(), "mod(x, 0) produced NaN under fp16"
        r.ok("mod by zero finite under fp16")
    except Exception as e:
        r.fail("mod by zero finite under fp16", str(e))

    # Division by zero must stay finite under fp16 on the operator path
    try:
        code = "@OUT = @A / @B;"
        zeros = torch.zeros(1, 4, 4, 3)
        program, type_map, out_names = _compile(code, {"A": img, "B": zeros})
        interp = Interpreter()
        result = interp.execute(program, {"A": img, "B": zeros}, type_map,
                                device="cpu", output_names=out_names,
                                precision="fp16", source=code)
        assert torch.isfinite(result["OUT"]).all(), "x / 0 produced non-finite under fp16"
        r.ok("division by zero finite under fp16")
    except Exception as e:
        r.fail("division by zero finite under fp16", str(e))

    # Scalar-mode builtins must follow the requested precision dtype
    try:
        code = "@OUT = u + 0.1;"
        program, type_map, out_names = _compile(code, {"s": "x"})
        interp = Interpreter()
        result = interp.execute(program, {"s": "x"}, type_map, device="cpu",
                                output_names=out_names, precision="fp16", source=code)
        val = result["OUT"]
        expected = (torch.tensor(0.0, dtype=torch.float16) + torch.tensor(0.1, dtype=torch.float16)).float()
        assert abs(val.item() - expected.item()) < 1e-7, "scalar-mode builtin did not use fp16"
        r.ok("scalar-mode builtins use the precision dtype")
    except Exception as e:
        r.fail("scalar-mode builtins use the precision dtype", str(e))
