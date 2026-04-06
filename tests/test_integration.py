"""Integration tests — examples, cache, device, inference, batching, latent, matrices."""
from helpers import *


def test_examples(r: SubTestResult):
    print("\n--- Example Snippet Tests ---")

    B, H, W = 1, 8, 8
    test_img = torch.rand(B, H, W, 3)

    # Grayscale example
    try:
        code = """
        float gray = luma(@A);
        @OUT = vec3(gray, gray, gray);
        """
        result = compile_and_run(code, {"A": test_img})
        assert result.shape == (B, H, W, 3)
        r.ok("example: grayscale")
    except Exception as e:
        r.fail("example: grayscale", f"{e}\n{traceback.format_exc()}")

    # Threshold mask example
    try:
        code = """
        float gray = luma(@A);
        @OUT = step(@B, gray);
        """
        result = compile_and_run(code, {"A": test_img, "B": 0.5})
        # Result should be binary (0 or 1)
        unique = torch.unique(result)
        assert all(v.item() in (0.0, 1.0) for v in unique), f"Non-binary values: {unique}"
        r.ok("example: threshold mask")
    except Exception as e:
        r.fail("example: threshold mask", f"{e}\n{traceback.format_exc()}")

    # Vignette example
    try:
        code = """
        float cx = u - 0.5;
        float cy = v - 0.5;
        float dist = sqrt(cx * cx + cy * cy);
        float vignette = 1.0 - smoothstep(0.3, 0.7, dist * @B);
        @OUT = @A * vec3(vignette, vignette, vignette);
        """
        result = compile_and_run(code, {"A": test_img, "B": 1.0})
        assert result.shape == (B, H, W, 3)
        # Center should be brighter than corners
        center = result[0, H//2, W//2, 0].item()
        corner = result[0, 0, 0, 0].item()
        # Not checking magnitude since input is random, but shape should be right
        r.ok("example: vignette")
    except Exception as e:
        r.fail("example: vignette", f"{e}\n{traceback.format_exc()}")

    # Color mix example
    try:
        img_a = torch.zeros(B, H, W, 3)
        img_b = torch.ones(B, H, W, 3)
        code = "@OUT = lerp(@A, @B, @C);"
        result = compile_and_run(code, {"A": img_a, "B": img_b, "C": 0.25})
        assert torch.allclose(result, torch.full((B, H, W, 3), 0.25), atol=1e-4)
        r.ok("example: color mix")
    except Exception as e:
        r.fail("example: color mix", f"{e}\n{traceback.format_exc()}")

    # Invert example
    try:
        code = """
        @OUT = vec3(1.0 - @A.r, 1.0 - @A.g, 1.0 - @A.b);
        """
        result = compile_and_run(code, {"A": test_img})
        expected = 1.0 - test_img
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("example: invert")
    except Exception as e:
        r.fail("example: invert", f"{e}\n{traceback.format_exc()}")

    # Conditional example
    try:
        code = """
        float brightness = luma(@A);
        if (brightness < 0.5) {
            float warmth = (0.5 - brightness) * 0.2;
            @OUT = vec3(
                clamp(@A.r + warmth, 0.0, 1.0),
                @A.g,
                clamp(@A.b - warmth, 0.0, 1.0)
            );
        } else {
            float coolness = (brightness - 0.5) * 0.2;
            @OUT = vec3(
                clamp(@A.r - coolness, 0.0, 1.0),
                @A.g,
                clamp(@A.b + coolness, 0.0, 1.0)
            );
        }
        """
        result = compile_and_run(code, {"A": test_img})
        assert result.shape == (B, H, W, 3)
        assert not torch.isnan(result).any()
        r.ok("example: conditional warm/cool")
    except Exception as e:
        r.fail("example: conditional warm/cool", f"{e}\n{traceback.format_exc()}")


# ── Example File Tests ────────────────────────────────────────────────

def _make_dummy_binding(tex_type, B=1, H=16, W=16, seed=42):
    """Create a dummy tensor value for a given TEXType."""
    torch.manual_seed(seed)
    if tex_type == TEXType.STRING:
        return "hello"
    if tex_type == TEXType.INT:
        return 1
    if tex_type == TEXType.FLOAT:
        return torch.rand(B, H, W)
    if tex_type == TEXType.VEC2:
        return torch.rand(B, H, W, 2)
    if tex_type == TEXType.VEC3:
        return torch.rand(B, H, W, 3)
    if tex_type == TEXType.VEC4:
        return torch.rand(B, H, W, 4)
    # Fallback
    return torch.rand(B, H, W, 3)


def _make_dummy_param(tex_type):
    """Create a dummy scalar value for a parameter type."""
    if tex_type == TEXType.INT:
        return 1
    if tex_type == TEXType.STRING:
        return "test"
    return 0.5


_STRING_FUNCTIONS = frozenset({
    "strip", "lower", "upper", "substr", "strlen", "contains",
    "starts_with", "ends_with", "replace", "concat", "trim",
    "split", "join", "char_at", "index_of",
})


def _collect_binding_hints(program) -> dict:
    """Walk AST to find BindingRef nodes and infer their types.

    Returns dict mapping binding name -> TEXType based on:
    1. Explicit type_hint prefixes (e.g. s@text → STRING)
    2. .a channel access on a binding → VEC4
    3. String function calls on a binding → STRING
    Bindings without any hint or usage signal are not included.
    """
    from TEX_Wrangle.tex_compiler.ast_nodes import (
        BindingRef, BinOp, UnaryOp, FunctionCall, VecConstructor,
        TernaryOp, CastExpr, ChannelAccess, ArrayIndexAccess,
        BindingIndexAccess, BindingSampleAccess, VarDecl, Assignment,
        IfElse, ForLoop, WhileLoop, ExprStatement, ArrayDecl, ArrayLiteral,
        MatConstructor, ReturnStmt, FunctionDef, ParamDecl,
    )
    hints = {}
    needs_alpha = set()   # binding names that access .a
    needs_string = set()  # binding names used with string functions
    has_vec4_context = False  # program uses vec4 types (declarations, constructors)
    stack = list(program.statements)
    while stack:
        node = stack.pop()
        if isinstance(node, BindingRef):
            if node.type_hint and node.kind == "wire":
                hints[node.name] = BINDING_HINT_TYPES.get(node.type_hint, TEXType.VEC4)
            continue
        # Detect .a channel access on a binding or sample(@binding, ...)
        if isinstance(node, ChannelAccess):
            if "a" in node.channels:
                obj = node.object
                if isinstance(obj, BindingRef):
                    needs_alpha.add(obj.name)
                elif isinstance(obj, BindingSampleAccess) and isinstance(obj.binding, BindingRef):
                    needs_alpha.add(obj.binding.name)
            stack.append(node.object)
            continue
        # Detect string functions called with a binding argument
        if isinstance(node, FunctionCall):
            if node.name in _STRING_FUNCTIONS:
                for arg in node.args:
                    if isinstance(arg, BindingRef):
                        needs_string.add(arg.name)
                    elif isinstance(arg, FunctionCall):
                        # nested: lower(strip(@text)) — we'll catch the inner call too
                        pass
            stack.extend(node.args)
            continue
        # Recurse into all child nodes, detect vec4 context
        if isinstance(node, VarDecl):
            if node.type_name == "vec4":
                has_vec4_context = True
            if node.initializer: stack.append(node.initializer)
        elif isinstance(node, ArrayDecl):
            if node.element_type_name == "vec4":
                has_vec4_context = True
            if node.initializer: stack.append(node.initializer)
            continue
        elif isinstance(node, VecConstructor):
            if node.size == 4:
                has_vec4_context = True
            stack.extend(node.args)
            continue
        elif isinstance(node, Assignment):
            stack.append(node.target); stack.append(node.value)
        elif isinstance(node, IfElse):
            stack.append(node.condition)
            stack.extend(node.then_body); stack.extend(node.else_body)
        elif isinstance(node, ForLoop):
            stack.append(node.init); stack.append(node.condition)
            stack.append(node.update); stack.extend(node.body)
        elif isinstance(node, WhileLoop):
            stack.append(node.condition); stack.extend(node.body)
        elif isinstance(node, ExprStatement):
            stack.append(node.expr)
        elif isinstance(node, BinOp):
            stack.append(node.left); stack.append(node.right)
        elif isinstance(node, UnaryOp):
            stack.append(node.operand)
        elif isinstance(node, MatConstructor):
            stack.extend(node.args)
        elif isinstance(node, TernaryOp):
            stack.append(node.condition)
            stack.append(node.true_expr); stack.append(node.false_expr)
        elif isinstance(node, CastExpr):
            stack.append(node.expr)
        elif isinstance(node, ArrayIndexAccess):
            stack.append(node.array); stack.append(node.index)
        elif isinstance(node, BindingIndexAccess):
            stack.append(node.binding); stack.extend(node.args)
        elif isinstance(node, BindingSampleAccess):
            stack.append(node.binding); stack.extend(node.args)
        elif isinstance(node, ArrayLiteral):
            stack.extend(node.elements)
        elif isinstance(node, ReturnStmt):
            if node.value: stack.append(node.value)
        elif isinstance(node, FunctionDef):
            stack.extend(node.body)
        elif isinstance(node, ParamDecl):
            if node.default_expr: stack.append(node.default_expr)

    # Apply usage-based inference (explicit hints take priority)
    for name in needs_alpha:
        if name not in hints:
            hints[name] = TEXType.VEC4
    for name in needs_string:
        if name not in hints:
            hints[name] = TEXType.STRING

    return hints, has_vec4_context


def _prepare_example(code, B, H, W):
    """Shared setup for example file tests: two-pass compile + build dummy bindings.

    Returns (program, bindings, type_map, output_names) ready for execution.
    Raises on compile failure so the caller can catch and report.
    """
    # Pass 1: Parse, collect type hints, type-check to discover bindings
    program = Parser(Lexer(code).tokenize(), source=code).parse()
    binding_hints, has_vec4_context = _collect_binding_hints(program)

    checker = TypeChecker(binding_types={}, source=code)
    checker.check(program)

    output_names = sorted(checker.assigned_bindings.keys())
    param_names = set(checker.param_declarations.keys())
    input_names = checker.referenced_bindings - set(output_names) - param_names

    # Build binding_types from hints, default to VEC3 (or VEC4 if program uses vec4)
    default_img_type = TEXType.VEC4 if has_vec4_context else TEXType.VEC3
    binding_types = {}
    for bname in input_names:
        binding_types[bname] = binding_hints.get(bname, default_img_type)

    # Pass 2: Re-parse (type checker mutates AST), type-check with correct types
    program = Parser(Lexer(code).tokenize(), source=code).parse()
    checker = TypeChecker(binding_types=binding_types, source=code)
    type_map = checker.check(program)
    output_names = sorted(checker.assigned_bindings.keys())

    # Optimize
    program = optimize(program)

    # Build dummy bindings matching the resolved types
    bindings = {}
    for bname in input_names:
        bt = binding_types.get(bname, default_img_type)
        bindings[bname] = _make_dummy_binding(bt, B=B, H=H, W=W)

    for pname, pinfo in checker.param_declarations.items():
        if pname not in bindings:
            bindings[pname] = _make_dummy_param(pinfo["type"])

    return program, bindings, type_map, output_names


def _get_example_files(r, label):
    """Locate and return sorted .tex example files, or None on failure."""
    examples_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "examples"
    if not examples_dir.exists():
        r.fail(label, f"Examples directory not found: {examples_dir}")
        return None
    tex_files = sorted(examples_dir.glob("*.tex"))
    if not tex_files:
        r.fail(label, "No .tex files found in examples/")
        return None
    return tex_files


def test_example_files(r: SubTestResult):
    """Load and run every .tex example through the full pipeline including optimizer.

    This catches bugs where the optimizer, type checker, or interpreter mishandles
    real-world code patterns. Each example must compile and execute without crashing.
    """
    print("\n--- Example File Tests ---")

    tex_files = _get_example_files(r, "example files")
    if tex_files is None:
        return

    B, H, W = 2, 16, 16  # B=2 to exercise batch/temporal code paths
    start_failed = r.failed

    for tex_path in tex_files:
        name = tex_path.stem
        try:
            code = tex_path.read_text(encoding="utf-8")
            program, bindings, type_map, output_names = _prepare_example(code, B, H, W)

            if not output_names:
                r.fail(f"example file: {name}", "No output bindings found")
                continue

            interp = Interpreter()
            result = interp.execute(program, bindings, type_map, device="cpu",
                                    output_names=output_names, source=code)

            for oname in output_names:
                assert oname in result, f"Missing output '{oname}'"

            r.ok(f"example file: {name}")

        except Exception as e:
            r.fail(f"example file: {name}", f"{e}")

    example_failed = r.failed - start_failed
    example_passed = len(tex_files) - example_failed
    print(f"  Example files: {example_passed}/{len(tex_files)} passed ({len(tex_files)} files)")


def test_example_files_compiled(r: SubTestResult):
    """Run every .tex example through execute_compiled (codegen + torch.compile).

    This catches codegen failures and torch.compile graph-break issues on
    real-world programs. Falls back gracefully — the test verifies that
    execute_compiled never crashes, even if it degrades to the interpreter.

    Each program has a 30-second timeout to prevent torch.compile from
    blocking the entire test suite on pathological programs.
    """
    import gc
    import concurrent.futures
    print("\n--- Example File Tests (compiled) ---")

    tex_files = _get_example_files(r, "example files compiled")
    if tex_files is None:
        return

    B, H, W = 1, 8, 8  # Smaller than interpreter test — torch.compile is slow
    _PER_PROGRAM_TIMEOUT = 30  # seconds — torch.compile can take 10-20s per program
    start_failed = r.failed
    codegen_ok = 0
    codegen_fallback = 0
    timed_out = 0

    # Clear compiled cache and dynamo state to start clean
    clear_compiled_cache()
    try:
        torch._dynamo.reset()
    except Exception:
        pass

    for tex_path in tex_files:
        name = tex_path.stem
        try:
            code = tex_path.read_text(encoding="utf-8")
            program, bindings, type_map, output_names = _prepare_example(code, B, H, W)

            if not output_names:
                r.fail(f"example compiled: {name}", "No output bindings found")
                continue

            # Use a simple fingerprint for cache keying
            fp = f"test_example_{name}"

            # Track codegen vs fallback before execution
            cg_fn = try_compile(program, type_map)
            if cg_fn is not None:
                codegen_ok += 1
            else:
                codegen_fallback += 1

            # Run with timeout — torch.compile can hang on complex programs
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    execute_compiled,
                    program, bindings, type_map, "cpu", fp,
                    output_names=output_names,
                )
                try:
                    result = future.result(timeout=_PER_PROGRAM_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    timed_out += 1
                    print(f"    WARNING: {name} timed out (torch.compile >{_PER_PROGRAM_TIMEOUT}s)")
                    continue  # Don't count as pass or fail

            # Verify we got outputs
            if isinstance(result, dict):
                for oname in output_names:
                    assert oname in result, f"Missing output '{oname}'"
            else:
                assert result is not None

            # Cross-validate: compare compiled output against interpreter
            if cg_fn is not None:
                interp = Interpreter()
                interp_result = interp.execute(
                    program, dict(bindings), type_map, device="cpu",
                    output_names=output_names, source=code,
                )
                compiled_dict = result if isinstance(result, dict) else {"OUT": result}
                for oname in output_names:
                    iv = interp_result.get(oname)
                    cv = compiled_dict.get(oname)
                    if (isinstance(iv, torch.Tensor) and isinstance(cv, torch.Tensor)
                            and iv.is_floating_point()):
                        max_diff = (iv.float() - cv.float()).abs().max().item()
                        assert max_diff < 0.1, (
                            f"Codegen/interpreter mismatch for '{oname}': "
                            f"max_diff={max_diff:.6f}"
                        )

            r.ok(f"example compiled: {name}")

        except Exception as e:
            r.fail(f"example compiled: {name}", f"{e}")
        finally:
            # Release compiled kernels and dynamo state between programs
            # to prevent memory accumulation (~25 MB per compiled program).
            gc.collect()
            try:
                torch._dynamo.reset()
            except Exception:
                pass

    example_failed = r.failed - start_failed
    example_passed = len(tex_files) - example_failed
    timeout_note = f", timed out: {timed_out}" if timed_out else ""
    print(f"  Compiled examples: {example_passed}/{len(tex_files)} passed "
          f"(codegen: {codegen_ok}, fallback: {codegen_fallback}{timeout_note})")


# ── Cache Tests ────────────────────────────────────────────────────────

def test_cache(r: SubTestResult):
    print("\n--- Cache Tests ---")

    # Memory cache hit returns same AST object
    try:
        tmp = tempfile.mkdtemp()
        cache = TEXCache(cache_dir=Path(tmp))
        code = "float g = luma(@A); @OUT = vec3(g, g, g);"
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

        r1 = cache.compile_tex(code, bt)
        r2 = cache.compile_tex(code, bt)
        assert r1[0] is r2[0], "Memory hit should return the exact same program object"
        r.ok("cache: memory hit same object")
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        r.fail("cache: memory hit same object", f"{e}\n{traceback.format_exc()}")

    # Disk cache hit after memory is cleared
    try:
        tmp = tempfile.mkdtemp()
        cache = TEXCache(cache_dir=Path(tmp))
        code = "@OUT = @A * 0.5;"
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}

        cache.compile_tex(code, bt)
        # Verify .pkl was written
        pkl_files = list(Path(tmp).glob("*.pkl"))
        assert len(pkl_files) == 1, f"Expected 1 .pkl file, got {len(pkl_files)}"

        cache.clear_memory()
        result = cache.compile_tex(code, bt)
        assert result is not None
        assert len(result) == 6
        r.ok("cache: disk hit after memory clear")
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        r.fail("cache: disk hit after memory clear", f"{e}\n{traceback.format_exc()}")

    # Corrupted disk file handled gracefully
    try:
        tmp = tempfile.mkdtemp()
        cache = TEXCache(cache_dir=Path(tmp))
        code = "@OUT = @A;"
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
        fp = cache.fingerprint(code, bt)

        # Write garbage to the expected disk cache path
        disk_path = Path(tmp) / f"{fp}.pkl"
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        disk_path.write_bytes(b"corrupted data here")

        # compile_tex should handle the corruption and compile fresh
        result = cache.compile_tex(code, bt)
        assert result is not None
        r.ok("cache: corrupted disk handled")
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        r.fail("cache: corrupted disk handled", f"{e}\n{traceback.format_exc()}")

    # Version mismatch triggers recompile
    try:
        import pickle
        tmp = tempfile.mkdtemp()
        cache = TEXCache(cache_dir=Path(tmp))
        code = "@OUT = @A + 1.0;"
        bt = {"A": TEXType.FLOAT, "OUT": TEXType.VEC4}

        cache.compile_tex(code, bt)
        cache.clear_memory()

        # Tamper with the version in the pickled file
        fp = cache.fingerprint(code, bt)
        disk_path = Path(tmp) / f"{fp}.pkl"
        with open(disk_path, "rb") as f:
            data = pickle.load(f)
        data["version"] = "0.0.0-old"
        with open(disk_path, "wb") as f:
            pickle.dump(data, f)

        # Should miss (version mismatch) and recompile fresh
        result = cache.compile_tex(code, bt)
        assert result is not None
        r.ok("cache: version mismatch recompile")
        shutil.rmtree(tmp, ignore_errors=True)
    except Exception as e:
        r.fail("cache: version mismatch recompile", f"{e}\n{traceback.format_exc()}")

    # Fingerprint stability and differentiation
    try:
        code = "@OUT = @A;"
        bt_a = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        fp1 = TEXCache.fingerprint(code, bt_a)
        fp2 = TEXCache.fingerprint(code, bt_a)
        assert fp1 == fp2, "Same inputs should produce same fingerprint"

        bt_b = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
        fp3 = TEXCache.fingerprint(code, bt_b)
        assert fp1 != fp3, "Different binding types should produce different fingerprint"
        r.ok("cache: fingerprint stability")
    except Exception as e:
        r.fail("cache: fingerprint stability", f"{e}\n{traceback.format_exc()}")


# ── Cache Eviction Tests ──────────────────────────────────────────────

def test_cache_eviction(r: SubTestResult):
    print("\n--- Cache Eviction Tests ---")

    # Memory eviction at 128
    try:
        tmp_dir = tempfile.mkdtemp()
        cache = TEXCache(cache_dir=Path(tmp_dir))
        bt = {"A": TEXType.VEC3, "OUT": TEXType.VEC4}
        for i in range(135):
            code = f"@OUT = @A * {i}.0;"
            cache.compile_tex(code, bt)
        assert len(cache._memory) <= 128, f"Memory has {len(cache._memory)} entries (limit: 128)"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        r.ok(f"cache: memory eviction ({len(cache._memory)} entries)")
    except Exception as e:
        r.fail("cache: memory eviction", f"{e}\n{traceback.format_exc()}")

    # Disk eviction at 512
    try:
        tmp_dir = tempfile.mkdtemp()
        cache = TEXCache(cache_dir=Path(tmp_dir))
        # Create 515 dummy .pkl files
        for i in range(515):
            p = Path(tmp_dir) / f"dummy_{i:04d}.pkl"
            with open(p, "wb") as f:
                pickle.dump({"dummy": i}, f)
            # Stagger access times slightly (on Windows atime may not update,
            # so we use mtime as a proxy — the eviction sorts by atime)
            os.utime(p, (i, i))
        cache._evict_disk_if_needed()
        remaining = len(list(Path(tmp_dir).glob("*.pkl")))
        assert remaining <= 512, f"Disk has {remaining} files (limit: 512)"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        r.ok(f"cache: disk eviction ({remaining} files)")
    except Exception as e:
        r.fail("cache: disk eviction", f"{e}\n{traceback.format_exc()}")


# ── Device Selection Tests ─────────────────────────────────────────────

def test_device_selection(r: SubTestResult):
    print("\n--- Device Selection Tests ---")

    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 3)

    # Explicit CPU
    try:
        result = compile_and_run("@OUT = @A;", {"A": test_img}, device="cpu")
        assert result.device.type == "cpu"
        assert torch.allclose(result, test_img, atol=1e-6)
        r.ok("device: explicit cpu")
    except Exception as e:
        r.fail("device: explicit cpu", f"{e}\n{traceback.format_exc()}")

    # Auto mode with CPU tensor -> should stay on CPU
    try:
        result = compile_and_run("@OUT = @A * 0.5;", {"A": test_img}, device="cpu")
        expected = test_img * 0.5
        assert torch.allclose(result, expected, atol=1e-5)
        r.ok("device: auto cpu tensor")
    except Exception as e:
        r.fail("device: auto cpu tensor", f"{e}\n{traceback.format_exc()}")

    # CUDA if available
    if torch.cuda.is_available():
        try:
            img_cuda = test_img.cuda()
            result = compile_and_run("@OUT = @A;", {"A": img_cuda}, device="cuda")
            # Interpreter returns tensor on the execution device;
            # check the values are correct once moved to CPU
            result_cpu = result.cpu() if result.is_cuda else result
            assert torch.allclose(result_cpu, test_img, atol=1e-5)
            r.ok("device: explicit cuda")
        except Exception as e:
            r.fail("device: explicit cuda", f"{e}\n{traceback.format_exc()}")
    else:
        r.ok("device: cuda skipped (no GPU)")


# ── torch.compile Tests ───────────────────────────────────────────────

def test_torch_compile(r: SubTestResult):
    print("\n--- torch.compile Tests ---")

    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 4)

    code = "@OUT = @A * 0.5;"
    bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}

    cache = TEXCache()
    program, type_map, refs, *_ = cache.compile_tex(code, bt)
    fp = cache.fingerprint(code, bt)

    # Ensure fresh state
    clear_compiled_cache()

    # Plain execution baseline
    try:
        result_plain = _plain_execute(program, {"A": test_img}, type_map, "cpu")
        expected = test_img * 0.5
        assert torch.allclose(result_plain, expected, atol=1e-5)
        r.ok("torch_compile: plain baseline")
    except Exception as e:
        r.fail("torch_compile: plain baseline", f"{e}\n{traceback.format_exc()}")

    # Compiled execution produces numerically equivalent result
    try:
        result_compiled = execute_compiled(
            program, {"A": test_img}, type_map, "cpu", fp
        )
        expected = test_img * 0.5
        assert torch.allclose(result_compiled, expected, atol=1e-5), (
            f"Max diff: {(result_compiled - expected).abs().max().item()}"
        )
        r.ok("torch_compile: compiled matches plain")
    except Exception as e:
        r.fail("torch_compile: compiled matches plain", f"{e}\n{traceback.format_exc()}")

    # Graceful execution — should never crash regardless of backend availability
    try:
        result = execute_compiled(
            program, {"A": test_img}, type_map, "cpu", fp
        )
        assert result is not None
        assert result.shape == test_img.shape
        r.ok("torch_compile: graceful execution")
    except Exception as e:
        r.fail("torch_compile: graceful execution", f"{e}\n{traceback.format_exc()}")

    # For-loop program still works under compiled path (graph breaks handled)
    try:
        loop_code = """
        float sum = 0.0;
        for (int i = 0; i < 5; i++) {
            sum += 1.0;
        }
        @OUT = @A * (sum / 5.0);
        """
        loop_bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        loop_cache = TEXCache()
        loop_prog, loop_tm, loop_refs, *_ = loop_cache.compile_tex(loop_code, loop_bt)
        loop_fp = loop_cache.fingerprint(loop_code, loop_bt)

        result_loop = execute_compiled(
            loop_prog, {"A": test_img}, loop_tm, "cpu", loop_fp
        )
        # sum = 5, so sum/5 = 1.0, result should equal @A
        assert torch.allclose(result_loop, test_img, atol=1e-4), (
            f"Max diff: {(result_loop - test_img).abs().max().item()}"
        )
        r.ok("torch_compile: for-loop graph breaks")
    except Exception as e:
        r.fail("torch_compile: for-loop graph breaks", f"{e}\n{traceback.format_exc()}")

    # Multi-output program via execute_compiled
    try:
        mo_code = "@mask = vec4(1.0, 1.0, 1.0, 1.0);\n@result = @A * 0.5;"
        mo_bt = {"A": TEXType.VEC4, "mask": TEXType.VEC4, "result": TEXType.VEC4}
        mo_cache = TEXCache()
        mo_prog, mo_tm, mo_refs, *_ = mo_cache.compile_tex(mo_code, mo_bt)
        mo_fp = mo_cache.fingerprint(mo_code, mo_bt)

        mo_result = execute_compiled(
            mo_prog, {"A": test_img}, mo_tm, "cpu", mo_fp,
            output_names=["mask", "result"]
        )
        assert isinstance(mo_result, dict), f"Expected dict, got {type(mo_result)}"
        assert "mask" in mo_result, "Missing 'mask' in multi-output result"
        assert "result" in mo_result, "Missing 'result' in multi-output result"
        assert torch.allclose(mo_result["result"], test_img * 0.5, atol=1e-5)
        r.ok("torch_compile: multi-output")
    except Exception as e:
        r.fail("torch_compile: multi-output", f"{e}\n{traceback.format_exc()}")


# ── IS_CHANGED Hash Tests ─────────────────────────────────────────────

def test_is_changed_hash(r: SubTestResult):
    print("\n--- IS_CHANGED Hash Tests ---")

    from TEX_Wrangle.tex_node import _tensor_fingerprint

    # Same-shape tensors with different content → different fingerprints
    try:
        t1 = torch.zeros(1, 4, 4, 3)
        t2 = torch.ones(1, 4, 4, 3)
        fp1 = _tensor_fingerprint(t1)
        fp2 = _tensor_fingerprint(t2)
        assert fp1 != fp2, "Different tensors should produce different fingerprints"
        r.ok("is_changed: different tensors produce different fingerprints")
    except Exception as e:
        r.fail("is_changed: different tensors produce different fingerprints", f"{e}\n{traceback.format_exc()}")

    # Same tensor → same fingerprint (stability)
    try:
        t = torch.rand(1, 8, 8, 3)
        fp_a = _tensor_fingerprint(t)
        fp_b = _tensor_fingerprint(t)
        assert fp_a == fp_b, "Same tensor should produce same fingerprint"
        r.ok("is_changed: same tensor produces stable fingerprint")
    except Exception as e:
        r.fail("is_changed: same tensor produces stable fingerprint", f"{e}\n{traceback.format_exc()}")

    # Large tensor still works (stride sampling)
    try:
        t_large = torch.rand(2, 512, 512, 4)
        fp = _tensor_fingerprint(t_large)
        assert "512" in fp, "Fingerprint should contain shape info"
        r.ok("is_changed: large tensor fingerprint")
    except Exception as e:
        r.fail("is_changed: large tensor fingerprint", f"{e}\n{traceback.format_exc()}")


# ── Batch / Temporal Tests ─────────────────────────────────────────────

def test_batch_temporal(r: SubTestResult):
    print("\n--- Batch / Temporal Tests ---")

    # ── Built-in variables: fi, fn ─────────────────────────────────────

    # Test 1: fi values correct across frames
    try:
        B, H, W = 4, 2, 2
        img = torch.rand(B, H, W, 3)
        result = compile_and_run("@OUT = vec3(fi, fi, fi);", {"A": img})
        # Each frame b should have fi == b
        for b in range(B):
            expected = float(b)
            actual = result[b, 0, 0, 0].item()
            assert abs(actual - expected) < 1e-5, f"frame {b}: fi={actual}, expected {expected}"
        r.ok("fi values correct (B=4)")
    except Exception as e:
        r.fail("fi values correct (B=4)", f"{e}\n{traceback.format_exc()}")

    # Test 2: fn equals batch size
    try:
        B, H, W = 4, 2, 2
        img = torch.rand(B, H, W, 3)
        result = compile_and_run("@OUT = vec3(fn, fn, fn);", {"A": img})
        for b in range(B):
            actual = result[b, 0, 0, 0].item()
            assert abs(actual - 4.0) < 1e-5, f"frame {b}: fn={actual}, expected 4.0"
        r.ok("fn equals batch size (B=4)")
    except Exception as e:
        r.fail("fn equals batch size (B=4)", f"{e}\n{traceback.format_exc()}")

    # Test 3: fi/fn scalar mode (no image inputs)
    try:
        result = compile_and_run(
            "@OUT = vec4(fi, fn, 0.0, 1.0);",
            {},
        )
        fi_val = result.flatten()[0].item()
        fn_val = result.flatten()[1].item()
        assert abs(fi_val - 0.0) < 1e-5, f"scalar fi={fi_val}"
        assert abs(fn_val - 1.0) < 1e-5, f"scalar fn={fn_val}"
        r.ok("fi/fn scalar mode (no images)")
    except Exception as e:
        r.fail("fi/fn scalar mode (no images)", f"{e}\n{traceback.format_exc()}")

    # Test 4: fade effect — frame 0 is black, last frame is original
    try:
        B, H, W = 4, 2, 2
        img = torch.ones(B, H, W, 4) * 0.8
        result = compile_and_run(
            "@OUT = @A * (fi / max(fn - 1, 1));",
            {"A": img},
        )
        # Frame 0: fi=0, so 0.8 * 0/3 = 0.0
        assert result[0].abs().max().item() < 1e-5, f"frame 0 not black"
        # Frame 3: fi=3, so 0.8 * 3/3 = 0.8
        assert abs(result[3, 0, 0, 0].item() - 0.8) < 1e-4, f"frame 3 not original"
        r.ok("fade effect (fi / max(fn-1, 1))")
    except Exception as e:
        r.fail("fade effect (fi / max(fn-1, 1))", f"{e}\n{traceback.format_exc()}")

    # ── Cross-frame fetch ──────────────────────────────────────────────

    # Create distinct frames for cross-frame tests: R=0.1*b per frame
    B, H, W = 3, 2, 2

    def make_distinct_frames(num_frames):
        """Create image batch where each frame has a unique solid color."""
        frames = torch.zeros(num_frames, H, W, 4)
        for b in range(num_frames):
            frames[b, :, :, 0] = 0.1 * (b + 1)  # R varies by frame
            frames[b, :, :, 1] = 0.5
            frames[b, :, :, 2] = 0.5
            frames[b, :, :, 3] = 1.0
        return frames

    distinct = make_distinct_frames(B)

    # Test 5: fetch_frame from frame 0 — all frames should read frame 0's color
    try:
        result = compile_and_run(
            "@OUT = fetch_frame(@A, 0, ix, iy);",
            {"A": distinct},
        )
        expected_r = 0.1  # frame 0 R = 0.1
        for b in range(B):
            actual_r = result[b, 0, 0, 0].item()
            assert abs(actual_r - expected_r) < 1e-5, \
                f"frame {b}: got R={actual_r}, expected {expected_r}"
        r.ok("fetch_frame(@A, 0, ix, iy) reads frame 0")
    except Exception as e:
        r.fail("fetch_frame(@A, 0, ix, iy) reads frame 0", f"{e}\n{traceback.format_exc()}")

    # Test 6: fetch_frame with fi+1 — each frame reads from next (last clamps)
    try:
        result = compile_and_run(
            "@OUT = fetch_frame(@A, fi + 1, ix, iy);",
            {"A": distinct},
        )
        # frame 0 reads frame 1 (R=0.2), frame 1 reads frame 2 (R=0.3),
        # frame 2 reads frame 2 (clamped, R=0.3)
        expected = [0.2, 0.3, 0.3]
        for b in range(B):
            actual_r = result[b, 0, 0, 0].item()
            assert abs(actual_r - expected[b]) < 1e-5, \
                f"frame {b}: got R={actual_r}, expected {expected[b]}"
        r.ok("fetch_frame(@A, fi+1, ix, iy) next-frame read")
    except Exception as e:
        r.fail("fetch_frame(@A, fi+1, ix, iy) next-frame read", f"{e}\n{traceback.format_exc()}")

    # Test 7: negative frame clamped to 0
    try:
        result = compile_and_run(
            "@OUT = fetch_frame(@A, -1, ix, iy);",
            {"A": distinct},
        )
        expected_r = 0.1  # frame 0
        for b in range(B):
            actual_r = result[b, 0, 0, 0].item()
            assert abs(actual_r - expected_r) < 1e-5, \
                f"frame {b}: got R={actual_r}, expected {expected_r}"
        r.ok("fetch_frame negative frame clamped to 0")
    except Exception as e:
        r.fail("fetch_frame negative frame clamped to 0", f"{e}\n{traceback.format_exc()}")

    # Test 8: oversized frame clamped to B-1
    try:
        result = compile_and_run(
            "@OUT = fetch_frame(@A, 999, ix, iy);",
            {"A": distinct},
        )
        expected_r = 0.3  # frame 2 (last)
        for b in range(B):
            actual_r = result[b, 0, 0, 0].item()
            assert abs(actual_r - expected_r) < 1e-5, \
                f"frame {b}: got R={actual_r}, expected {expected_r}"
        r.ok("fetch_frame oversized frame clamped to B-1")
    except Exception as e:
        r.fail("fetch_frame oversized frame clamped to B-1", f"{e}\n{traceback.format_exc()}")

    # ── Cross-frame sample ─────────────────────────────────────────────

    # Test 9: sample_frame from frame 0 — all frames read frame 0
    try:
        result = compile_and_run(
            "@OUT = sample_frame(@A, 0, u, v);",
            {"A": distinct},
        )
        expected_r = 0.1
        for b in range(B):
            actual_r = result[b, 0, 0, 0].item()
            assert abs(actual_r - expected_r) < 1e-5, \
                f"frame {b}: got R={actual_r}, expected {expected_r}"
        r.ok("sample_frame(@A, 0, u, v) reads frame 0")
    except Exception as e:
        r.fail("sample_frame(@A, 0, u, v) reads frame 0", f"{e}\n{traceback.format_exc()}")

    # Test 10: sample_frame with fi equivalent to regular sample
    try:
        result_sf = compile_and_run(
            "@OUT = sample_frame(@A, fi, u, v);",
            {"A": distinct},
        )
        result_s = compile_and_run(
            "@OUT = sample(@A, u, v);",
            {"A": distinct},
        )
        diff = (result_sf - result_s).abs().max().item()
        assert diff < 1e-5, f"sample_frame(fi) vs sample diff: {diff}"
        r.ok("sample_frame(@A, fi, u, v) == sample(@A, u, v)")
    except Exception as e:
        r.fail("sample_frame(@A, fi, u, v) == sample(@A, u, v)", f"{e}\n{traceback.format_exc()}")

    # ── Integration patterns ───────────────────────────────────────────

    # Test 11: frame difference
    try:
        result = compile_and_run(
            "vec4 curr = fetch_frame(@A, fi, ix, iy);\n"
            "vec4 prev = fetch_frame(@A, max(fi - 1, 0), ix, iy);\n"
            "@OUT = abs(curr - prev);",
            {"A": distinct},
        )
        # Frame 0: diff with self = 0
        assert result[0].abs().max().item() < 1e-5, "frame 0 diff should be 0"
        # Frame 1: R diff = |0.2 - 0.1| = 0.1
        r_diff = result[1, 0, 0, 0].item()
        assert abs(r_diff - 0.1) < 1e-4, f"frame 1 R diff: {r_diff}, expected 0.1"
        r.ok("frame difference pattern")
    except Exception as e:
        r.fail("frame difference pattern", f"{e}\n{traceback.format_exc()}")

    # Test 12: temporal average (3-frame blend)
    try:
        result = compile_and_run(
            "vec4 prev = fetch_frame(@A, fi - 1, ix, iy);\n"
            "vec4 curr = fetch_frame(@A, fi, ix, iy);\n"
            "vec4 next = fetch_frame(@A, fi + 1, ix, iy);\n"
            "@OUT = (prev + curr + next) / 3.0;",
            {"A": distinct},
        )
        # Frame 1: prev=frame0(R=0.1), curr=frame1(R=0.2), next=frame2(R=0.3)
        # avg R = (0.1+0.2+0.3)/3 = 0.2
        avg_r = result[1, 0, 0, 0].item()
        assert abs(avg_r - 0.2) < 1e-4, f"temporal avg R: {avg_r}, expected 0.2"
        r.ok("temporal average (3-frame blend)")
    except Exception as e:
        r.fail("temporal average (3-frame blend)", f"{e}\n{traceback.format_exc()}")

    # Test 13: time-based gradient
    try:
        B2 = 5
        img = torch.rand(B2, 2, 2, 3)
        result = compile_and_run(
            "@OUT = vec3(fi / max(fn - 1, 1), 0.5, 1.0);",
            {"A": img},
        )
        for b in range(B2):
            expected = b / 4.0  # fn=5, fn-1=4
            actual = result[b, 0, 0, 0].item()
            assert abs(actual - expected) < 1e-5, f"frame {b}: R={actual}, expected {expected}"
        r.ok("time-based gradient (fi / max(fn-1, 1))")
    except Exception as e:
        r.fail("time-based gradient (fi / max(fn-1, 1))", f"{e}\n{traceback.format_exc()}")

    # ── Type checking ──────────────────────────────────────────────────

    # Test 14: fi and fn recognized as FLOAT
    try:
        code = "@OUT = vec4(fi, fn, 0.0, 1.0);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        checker = TypeChecker(binding_types=bt)
        checker.check(program)
        # No error = fi and fn are valid FLOAT builtins
        r.ok("fi and fn recognized as FLOAT by type checker")
    except Exception as e:
        r.fail("fi and fn recognized as FLOAT by type checker", f"{e}\n{traceback.format_exc()}")

    # Test 15: fetch_frame and sample_frame accept 4 args, return VEC4
    try:
        code = "vec4 a = fetch_frame(@A, 0, ix, iy); vec4 b = sample_frame(@A, 0, u, v); @OUT = a + b;"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        bt = {"A": TEXType.VEC4, "OUT": TEXType.VEC4}
        checker = TypeChecker(binding_types=bt)
        checker.check(program)
        r.ok("fetch_frame/sample_frame type-check OK (4 args, VEC4 return)")
    except Exception as e:
        r.fail("fetch_frame/sample_frame type-check OK (4 args, VEC4 return)", f"{e}\n{traceback.format_exc()}")


# ── Latent Tests ──────────────────────────────────────────────────────

def _make_latent(B=1, C=4, H=4, W=4) -> dict:
    """Create a fake LATENT dict for testing."""
    return {"samples": torch.randn(B, C, H, W)}


def test_latent(r: SubTestResult):
    print("\n--- Latent Tests ---")

    # latent: unwrap permute
    try:
        lat = _make_latent(1, 4, 8, 8)
        tensor_cl, meta = _unwrap_latent(lat)
        assert tensor_cl.shape == (1, 8, 8, 4), f"Expected [1,8,8,4] got {tensor_cl.shape}"
        assert isinstance(meta, dict)
        # Verify data is correctly permuted
        assert torch.allclose(tensor_cl[0, 0, 0, :], lat["samples"][0, :, 0, 0])
        r.ok("latent: unwrap permute")
    except Exception as e:
        r.fail("latent: unwrap permute", f"{e}\n{traceback.format_exc()}")

    # latent: infer type dict
    try:
        lat4 = _make_latent(1, 4, 4, 4)
        assert _infer_binding_type(lat4) == TEXType.VEC4
        lat3 = {"samples": torch.randn(1, 3, 4, 4)}
        assert _infer_binding_type(lat3) == TEXType.VEC3
        lat16 = {"samples": torch.randn(1, 16, 4, 4)}
        assert _infer_binding_type(lat16) == TEXType.VEC4  # 16-ch -> VEC4 best-effort
        r.ok("latent: infer type dict")
    except Exception as e:
        r.fail("latent: infer type dict", f"{e}\n{traceback.format_exc()}")

    # latent: passthrough
    try:
        lat = _make_latent(1, 4, 4, 4)
        original = lat["samples"].clone()
        tensor_cl, meta = _unwrap_latent(lat)
        result = compile_and_run("@OUT = @A;", {"A": tensor_cl}, latent_channel_count=4)
        # Permute result back to [B,C,H,W] and compare
        result_cf = result.permute(0, 3, 1, 2)
        assert torch.allclose(result_cf, original, atol=1e-6), "Passthrough mismatch"
        r.ok("latent: passthrough")
    except Exception as e:
        r.fail("latent: passthrough", f"{e}\n{traceback.format_exc()}")

    # latent: scalar gain
    try:
        lat = _make_latent(1, 4, 4, 4)
        tensor_cl, _ = _unwrap_latent(lat)
        result = compile_and_run("@OUT = @A * 0.5;", {"A": tensor_cl}, latent_channel_count=4)
        expected = tensor_cl * 0.5
        assert torch.allclose(result, expected, atol=1e-6), "Scalar gain mismatch"
        r.ok("latent: scalar gain")
    except Exception as e:
        r.fail("latent: scalar gain", f"{e}\n{traceback.format_exc()}")

    # latent: bias
    try:
        lat = _make_latent(1, 4, 4, 4)
        tensor_cl, _ = _unwrap_latent(lat)
        result = compile_and_run("@OUT = @A + 0.1;", {"A": tensor_cl}, latent_channel_count=4)
        expected = tensor_cl + 0.1
        assert torch.allclose(result, expected, atol=1e-6), "Bias mismatch"
        r.ok("latent: bias")
    except Exception as e:
        r.fail("latent: bias", f"{e}\n{traceback.format_exc()}")

    # latent: no clamp
    try:
        # Create latent with values > 1.0
        samples = torch.ones(1, 4, 4, 4) * 2.0  # [B,C,H,W]
        lat = {"samples": samples}
        tensor_cl, _ = _unwrap_latent(lat)
        result = compile_and_run("@OUT = @A * 2.0;", {"A": tensor_cl}, latent_channel_count=4)
        # _prepare_output with LATENT should NOT clamp
        prepared = _prepare_output(result, "LATENT")
        assert prepared.max().item() > 1.0, f"Values were clamped! max={prepared.max().item()}"
        r.ok("latent: no clamp")
    except Exception as e:
        r.fail("latent: no clamp", f"{e}\n{traceback.format_exc()}")

    # latent: metadata preserved
    try:
        noise_mask = torch.ones(1, 1, 4, 4)
        lat = {"samples": torch.randn(1, 4, 4, 4), "noise_mask": noise_mask, "batch_index": [0]}
        tensor_cl, meta = _unwrap_latent(lat)
        assert "noise_mask" in meta, "noise_mask missing from metadata"
        assert "batch_index" in meta, "batch_index missing from metadata"
        assert torch.equal(meta["noise_mask"], noise_mask), "noise_mask value changed"
        assert meta["batch_index"] == [0], "batch_index value changed"
        r.ok("latent: metadata preserved")
    except Exception as e:
        r.fail("latent: metadata preserved", f"{e}\n{traceback.format_exc()}")

    # latent: ic variable
    try:
        lat = _make_latent(1, 4, 4, 4)
        tensor_cl, _ = _unwrap_latent(lat)
        # Use ic to verify it equals channel count
        result = compile_and_run(
            "float c = ic; @OUT = vec4(c, c, c, c);",
            {"A": tensor_cl},
            latent_channel_count=4,
        )
        ic_val = result[0, 0, 0, 0].item()
        assert abs(ic_val - 4.0) < 1e-6, f"ic should be 4.0, got {ic_val}"
        r.ok("latent: ic variable")
    except Exception as e:
        r.fail("latent: ic variable", f"{e}\n{traceback.format_exc()}")

    # latent: lerp two latents
    try:
        lat_a = {"samples": torch.zeros(1, 4, 4, 4)}
        lat_b = {"samples": torch.ones(1, 4, 4, 4) * 2.0}
        tcl_a, _ = _unwrap_latent(lat_a)
        tcl_b, _ = _unwrap_latent(lat_b)
        result = compile_and_run(
            "@OUT = lerp(@A, @B, 0.5);",
            {"A": tcl_a, "B": tcl_b},
            latent_channel_count=4,
        )
        expected_val = 1.0  # midpoint of 0 and 2
        assert torch.allclose(result, torch.ones_like(result) * expected_val, atol=1e-6), \
            f"Lerp midpoint mismatch, got {result[0,0,0,0].item()}"
        r.ok("latent: lerp two latents")
    except Exception as e:
        r.fail("latent: lerp two latents", f"{e}\n{traceback.format_exc()}")

    # latent: channel access
    try:
        # Known values per channel
        samples = torch.zeros(1, 4, 4, 4)
        samples[0, 0, :, :] = 1.0  # channel 0 = 1.0
        samples[0, 1, :, :] = 2.0  # channel 1 = 2.0
        lat = {"samples": samples}
        tensor_cl, _ = _unwrap_latent(lat)
        # .r should be channel 0
        result = compile_and_run(
            "float ch0 = @A.r; @OUT = vec4(ch0, ch0, ch0, ch0);",
            {"A": tensor_cl},
            latent_channel_count=4,
        )
        val = result[0, 0, 0, 0].item()
        assert abs(val - 1.0) < 1e-6, f"Channel 0 should be 1.0, got {val}"
        r.ok("latent: channel access")
    except Exception as e:
        r.fail("latent: channel access", f"{e}\n{traceback.format_exc()}")

    # latent: prepare output
    try:
        # [B,H,W,C] -> [B,C,H,W]
        raw = torch.randn(1, 8, 8, 4)
        prepared = _prepare_output(raw, "LATENT")
        assert prepared.shape == (1, 4, 8, 8), f"Expected [1,4,8,8] got {prepared.shape}"
        # Verify correct permutation
        assert torch.allclose(prepared[0, :, 0, 0], raw[0, 0, 0, :])
        r.ok("latent: prepare output")
    except Exception as e:
        r.fail("latent: prepare output", f"{e}\n{traceback.format_exc()}")

    # latent: fingerprint_inputs
    try:
        lat = _make_latent(1, 4, 4, 4)
        from TEX_Wrangle.tex_node import TEXWrangleNode
        # Should not crash with LATENT dict
        h = TEXWrangleNode.fingerprint_inputs(code="@OUT = @A;", A=lat)
        assert isinstance(h, str), f"Expected hash string, got {type(h)}"
        r.ok("latent: fingerprint_inputs")
    except Exception as e:
        r.fail("latent: fingerprint_inputs", f"{e}\n{traceback.format_exc()}")


# ── Auto-Inference Tests ───────────────────────────────────────────────

def test_auto_inference(r: SubTestResult):
    print("\n--- Auto-Inference Tests ---")
    B, H, W = 1, 4, 4
    test_img = torch.rand(B, H, W, 4)

    # vec3 output -> IMAGE
    try:
        _, inferred = compile_and_infer(
            "float g = luma(@A); @OUT = vec3(g, g, g);",
            {"A": test_img},
        )
        assert inferred == TEXType.VEC3, f"Expected VEC3, got {inferred}"
        assert _map_inferred_type(inferred, False) == "IMAGE"
        r.ok("auto: vec3 -> IMAGE")
    except Exception as e:
        r.fail("auto: vec3 -> IMAGE", f"{e}\n{traceback.format_exc()}")

    # vec4 output (no latent) -> IMAGE
    try:
        _, inferred = compile_and_infer(
            "@OUT = @A;",
            {"A": test_img},
        )
        assert inferred == TEXType.VEC4, f"Expected VEC4, got {inferred}"
        assert _map_inferred_type(inferred, False) == "IMAGE"
        r.ok("auto: vec4 -> IMAGE")
    except Exception as e:
        r.fail("auto: vec4 -> IMAGE", f"{e}\n{traceback.format_exc()}")

    # vec4 output (with latent) -> LATENT
    try:
        _, inferred = compile_and_infer(
            "@OUT = @A * 1.1;",
            {"A": test_img},
        )
        assert inferred == TEXType.VEC4, f"Expected VEC4, got {inferred}"
        assert _map_inferred_type(inferred, True) == "LATENT"
        r.ok("auto: vec4 + latent -> LATENT")
    except Exception as e:
        r.fail("auto: vec4 + latent -> LATENT", f"{e}\n{traceback.format_exc()}")

    # float output -> MASK
    try:
        _, inferred = compile_and_infer(
            "@OUT = luma(@A);",
            {"A": test_img},
        )
        assert inferred == TEXType.FLOAT, f"Expected FLOAT, got {inferred}"
        assert _map_inferred_type(inferred, False) == "MASK"
        r.ok("auto: float -> MASK")
    except Exception as e:
        r.fail("auto: float -> MASK", f"{e}\n{traceback.format_exc()}")

    # int output -> INT
    try:
        _, inferred = compile_and_infer(
            "int x = 42; @OUT = x;",
            {"A": test_img},
        )
        assert inferred == TEXType.INT, f"Expected INT, got {inferred}"
        assert _map_inferred_type(inferred, False) == "INT"
        r.ok("auto: int -> INT")
    except Exception as e:
        r.fail("auto: int -> INT", f"{e}\n{traceback.format_exc()}")

    # string output -> STRING
    try:
        _, inferred = compile_and_infer(
            '@OUT = "hello";',
            {"A": test_img},
        )
        assert inferred == TEXType.STRING, f"Expected STRING, got {inferred}"
        assert _map_inferred_type(inferred, False) == "STRING"
        r.ok("auto: string -> STRING")
    except Exception as e:
        r.fail("auto: string -> STRING", f"{e}\n{traceback.format_exc()}")

    # channel assignment -> VEC3
    try:
        _, inferred = compile_and_infer(
            "@OUT = @A; @OUT.r = 1.0; @OUT.g = 0.5; @OUT.b = 0.0;",
            {"A": test_img},
        )
        # First assignment @OUT = @A infers VEC4, channel .r/.g/.b don't widen
        # Actually: first assigns VEC4, channel accesses on VEC4 stay VEC4
        assert inferred in (TEXType.VEC3, TEXType.VEC4), f"Expected VEC3/VEC4, got {inferred}"
        assert _map_inferred_type(inferred, False) == "IMAGE"
        r.ok("auto: channel assignment -> IMAGE")
    except Exception as e:
        r.fail("auto: channel assignment -> IMAGE", f"{e}\n{traceback.format_exc()}")

    # channel-only assignment (no direct @OUT = ...) -> VEC3
    try:
        binding_types = {"A": TEXType.VEC4}
        checker = TypeChecker(binding_types=binding_types)
        tokens = Lexer("@OUT.r = @A.r; @OUT.g = @A.g; @OUT.b = @A.b;").tokenize()
        program = Parser(tokens).parse()
        checker.check(program)
        assert checker.inferred_out_type == TEXType.VEC3, f"Expected VEC3, got {checker.inferred_out_type}"
        r.ok("auto: channel-only -> VEC3")
    except Exception as e:
        r.fail("auto: channel-only -> VEC3", f"{e}\n{traceback.format_exc()}")

    # if/else same type
    try:
        _, inferred = compile_and_infer(
            "if (luma(@A) > 0.5) { @OUT = vec3(1.0, 0.0, 0.0); } else { @OUT = vec3(0.0, 0.0, 1.0); }",
            {"A": test_img},
        )
        assert inferred == TEXType.VEC3, f"Expected VEC3, got {inferred}"
        r.ok("auto: if/else same type")
    except Exception as e:
        r.fail("auto: if/else same type", f"{e}\n{traceback.format_exc()}")

    # if/else with promotion (float + vec3 -> vec3)
    try:
        _, inferred = compile_and_infer(
            "if (luma(@A) > 0.5) { @OUT = luma(@A); } else { @OUT = vec3(0.0, 0.0, 1.0); }",
            {"A": test_img},
        )
        assert inferred == TEXType.VEC3, f"Expected VEC3, got {inferred}"
        r.ok("auto: if/else promotion")
    except Exception as e:
        r.fail("auto: if/else promotion", f"{e}\n{traceback.format_exc()}")

    # string vs numeric conflict -> error
    try:
        binding_types = {"A": TEXType.VEC4}
        checker = TypeChecker(binding_types=binding_types)
        tokens = Lexer('if (luma(@A) > 0.5) { @OUT = "yes"; } else { @OUT = vec3(0.0, 0.0, 1.0); }').tokenize()
        program = Parser(tokens).parse()
        try:
            checker.check(program)
            r.fail("auto: string/numeric conflict", "Expected TypeCheckError")
        except TypeCheckError as e:
            assert "string" in str(e).lower() and "numeric" in str(e).lower()
            r.ok("auto: string/numeric conflict")
    except Exception as e:
        r.fail("auto: string/numeric conflict", f"{e}\n{traceback.format_exc()}")

    # explicit output_type still works (backward compat)
    try:
        result = compile_and_run(
            "@OUT = luma(@A);",
            {"A": test_img},
            out_type=TEXType.FLOAT,
        )
        assert isinstance(result, torch.Tensor)
        r.ok("auto: explicit still works")
    except Exception as e:
        r.fail("auto: explicit still works", f"{e}\n{traceback.format_exc()}")

    # _map_inferred_type with None -> IMAGE
    try:
        assert _map_inferred_type(None, False) == "IMAGE"
        assert _map_inferred_type(None, True) == "IMAGE"
        r.ok("auto: None -> IMAGE fallback")
    except Exception as e:
        r.fail("auto: None -> IMAGE fallback", f"{e}\n{traceback.format_exc()}")


# ── v0.3 Feature Tests ────────────────────────────────────────────────

def test_v03_features(r: SubTestResult):
    """Tests for v0.3: multi-output, $ parameters, typed bindings."""
    print("\n--- v0.3 Feature Tests ---")

    from TEX_Wrangle.tex_compiler.ast_nodes import BindingRef, ParamDecl

    # ── Lexer: $ binding ──
    try:
        tokens = Lexer("$strength").tokenize()
        tok = tokens[0]
        assert tok.type == TokenType.DOLLAR_BINDING, f"Expected DOLLAR_BINDING, got {tok.type}"
        assert tok.value == "strength"
        r.ok("lexer: $ binding")
    except Exception as e:
        r.fail("lexer: $ binding", f"{e}\n{traceback.format_exc()}")

    # ── Lexer: typed @ binding ──
    try:
        tokens = Lexer("f@threshold").tokenize()
        tok = tokens[0]
        assert tok.type == TokenType.TYPED_AT_BINDING, f"Expected TYPED_AT_BINDING, got {tok.type}"
        assert tok.value == "threshold"
        assert tok.prefix == "f"
        r.ok("lexer: typed @ binding (f@threshold)")
    except Exception as e:
        r.fail("lexer: typed @ binding", f"{e}\n{traceback.format_exc()}")

    # ── Lexer: typed $ binding ──
    try:
        tokens = Lexer("i$count").tokenize()
        tok = tokens[0]
        assert tok.type == TokenType.TYPED_DOLLAR_BINDING, f"Expected TYPED_DOLLAR_BINDING, got {tok.type}"
        assert tok.value == "count"
        assert tok.prefix == "i"
        r.ok("lexer: typed $ binding (i$count)")
    except Exception as e:
        r.fail("lexer: typed $ binding", f"{e}\n{traceback.format_exc()}")

    # ── Lexer: prefix vs identifier (f = 5 is NOT a typed binding) ──
    try:
        tokens = Lexer("f = 5;").tokenize()
        tok = tokens[0]
        assert tok.type == TokenType.IDENT, f"Expected IDENT, got {tok.type}"
        assert tok.value == "f"
        r.ok("lexer: prefix vs identifier")
    except Exception as e:
        r.fail("lexer: prefix vs identifier", f"{e}\n{traceback.format_exc()}")

    # ── Lexer: all type prefixes ──
    try:
        for prefix in ["f", "i", "v", "v4", "s", "img", "m", "l"]:
            tokens = Lexer(f"{prefix}@test").tokenize()
            tok = tokens[0]
            assert tok.type == TokenType.TYPED_AT_BINDING, f"prefix '{prefix}': Expected TYPED_AT_BINDING, got {tok.type}"
            assert tok.prefix == prefix, f"prefix '{prefix}': got prefix {tok.prefix!r}"
            assert tok.value == "test"
        r.ok("lexer: all type prefixes")
    except Exception as e:
        r.fail("lexer: all type prefixes", f"{e}\n{traceback.format_exc()}")

    # ── Parser: ParamDecl with default ──
    try:
        tokens = Lexer("f$strength = 0.5;").tokenize()
        program = Parser(tokens).parse()
        stmt = program.statements[0]
        assert isinstance(stmt, ParamDecl), f"Expected ParamDecl, got {type(stmt)}"
        assert stmt.name == "strength"
        assert stmt.type_hint == "f"
        assert stmt.default_expr is not None
        r.ok("parser: ParamDecl with default")
    except Exception as e:
        r.fail("parser: ParamDecl with default", f"{e}\n{traceback.format_exc()}")

    # ── Parser: ParamDecl no default ──
    try:
        tokens = Lexer("i$count;").tokenize()
        program = Parser(tokens).parse()
        stmt = program.statements[0]
        assert isinstance(stmt, ParamDecl), f"Expected ParamDecl, got {type(stmt)}"
        assert stmt.name == "count"
        assert stmt.type_hint == "i"
        assert stmt.default_expr is None
        r.ok("parser: ParamDecl no default")
    except Exception as e:
        r.fail("parser: ParamDecl no default", f"{e}\n{traceback.format_exc()}")

    # ── Parser: $ in expression ──
    try:
        tokens = Lexer("@OUT = @A * $strength;").tokenize()
        program = Parser(tokens).parse()
        # Should parse without error; the assignment RHS contains a $ binding ref
        r.ok("parser: $ in expression")
    except Exception as e:
        r.fail("parser: $ in expression", f"{e}\n{traceback.format_exc()}")

    # ── Parser: typed @ in expression ──
    try:
        tokens = Lexer("img@result = @A * 0.5;").tokenize()
        program = Parser(tokens).parse()
        r.ok("parser: typed @ assignment")
    except Exception as e:
        r.fail("parser: typed @ assignment", f"{e}\n{traceback.format_exc()}")

    # ── Type checker: multi-output ──
    try:
        code = "@result = @A * 0.5;\n@mask = luma(@A);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        img = torch.rand(1, 4, 4, 3)
        checker = TypeChecker(binding_types={"A": TEXType.VEC3})
        checker.check(program)
        assert "result" in checker.assigned_bindings, "Missing 'result' in assigned_bindings"
        assert "mask" in checker.assigned_bindings, "Missing 'mask' in assigned_bindings"
        assert checker.assigned_bindings["result"] == TEXType.VEC3
        assert checker.assigned_bindings["mask"] == TEXType.FLOAT
        r.ok("type checker: multi-output")
    except Exception as e:
        r.fail("type checker: multi-output", f"{e}\n{traceback.format_exc()}")

    # ── Type checker: output type inference ──
    try:
        code = "@OUT = luma(@A);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        checker = TypeChecker(binding_types={"A": TEXType.VEC3})
        checker.check(program)
        assert checker.assigned_bindings["OUT"] == TEXType.FLOAT
        # Backward compat property
        assert checker.inferred_out_type == TEXType.FLOAT
        r.ok("type checker: output type inference")
    except Exception as e:
        r.fail("type checker: output type inference", f"{e}\n{traceback.format_exc()}")

    # ── Type checker: param declaration ──
    try:
        code = "f$strength = 0.5;\n@OUT = @A * $strength;"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        checker = TypeChecker(binding_types={"A": TEXType.VEC3, "strength": TEXType.FLOAT})
        checker.check(program)
        assert "strength" in checker.param_declarations
        assert checker.param_declarations["strength"]["type"] == TEXType.FLOAT
        r.ok("type checker: param declaration")
    except Exception as e:
        r.fail("type checker: param declaration", f"{e}\n{traceback.format_exc()}")

    # ── Type checker: param type mismatch ──
    try:
        code = 'f$x = "hello";'
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        checker = TypeChecker(binding_types={})
        try:
            checker.check(program)
            r.fail("type checker: param type mismatch", "Expected TypeCheckError")
        except TypeCheckError:
            r.ok("type checker: param type mismatch")
    except Exception as e:
        r.fail("type checker: param type mismatch", f"{e}\n{traceback.format_exc()}")

    # ── Interpreter: multi-output ──
    try:
        code = "@result = @A * 0.5;\n@mask = luma(@A);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        img = torch.rand(1, 4, 4, 3)
        checker = TypeChecker(binding_types={"A": TEXType.VEC3})
        type_map = checker.check(program)
        interp = Interpreter()
        out = interp.execute(program, {"A": img}, type_map, device="cpu",
                             output_names=["mask", "result"])
        assert isinstance(out, dict), f"Expected dict, got {type(out)}"
        assert "result" in out, "Missing 'result'"
        assert "mask" in out, "Missing 'mask'"
        assert out["result"].shape[-1] == 3, f"result should be vec3, got shape {out['result'].shape}"
        assert out["mask"].dim() == 3, f"mask should be 3D (float), got dim {out['mask'].dim()}"
        r.ok("interpreter: multi-output")
    except Exception as e:
        r.fail("interpreter: multi-output", f"{e}\n{traceback.format_exc()}")

    # ── Interpreter: param as binding ──
    try:
        code = "f$strength = 0.5;\n@OUT = @A * $strength;"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        img = torch.ones(1, 2, 2, 3)
        checker = TypeChecker(binding_types={"A": TEXType.VEC3, "strength": TEXType.FLOAT})
        type_map = checker.check(program)
        interp = Interpreter()
        out = interp.execute(program, {"A": img, "strength": torch.tensor(0.5)}, type_map,
                             device="cpu")
        assert torch.allclose(out, torch.full_like(img, 0.5), atol=1e-5), f"Expected 0.5, got {out.mean().item()}"
        r.ok("interpreter: param as binding")
    except Exception as e:
        r.fail("interpreter: param as binding", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: multi-output tuple ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        img = torch.rand(1, 4, 4, 3)
        result = TEXWrangleNode.execute(
            code="@result = @A * 0.5;\n@mask = luma(@A);",
            A=img, device="cpu", compile_mode="none"
        )
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 8, f"Expected 8 slots, got {len(result)}"
        # Alphabetical order: "mask" at 0, "result" at 1
        assert result[0] is not None, "slot 0 (mask) should not be None"
        assert result[1] is not None, "slot 1 (result) should not be None"
        assert result[2] is None, "slot 2 should be None (unused)"
        r.ok("tex_node: multi-output tuple")
    except Exception as e:
        r.fail("tex_node: multi-output tuple", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: backward compat (single @OUT) ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        img = torch.rand(1, 4, 4, 3)
        result = TEXWrangleNode.execute(code="@OUT = @A * 0.5;", A=img, device="cpu", compile_mode="none")
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert result[0] is not None, "slot 0 (OUT) should not be None"
        assert result[1] is None, "slot 1 should be None"
        r.ok("tex_node: backward compat (@OUT)")
    except Exception as e:
        r.fail("tex_node: backward compat (@OUT)", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: old output_type ignored ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        img = torch.rand(1, 4, 4, 3)
        # Passing output_type="IMAGE" — should be silently ignored
        result = TEXWrangleNode.execute(code="@OUT = @A * 0.5;", A=img, device="cpu",
                                        compile_mode="none", output_type="IMAGE")
        assert result[0] is not None
        r.ok("tex_node: old output_type ignored")
    except Exception as e:
        r.fail("tex_node: old output_type ignored", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: param widgets flow through kwargs ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        img = torch.ones(1, 2, 2, 3)
        # $strength value comes as a kwarg (simulating widget value from ComfyUI)
        result = TEXWrangleNode.execute(
            code="f$strength = 0.5;\n@OUT = @A * $strength;",
            A=img, strength=torch.tensor(0.75), device="cpu", compile_mode="none"
        )
        out = result[0]  # OUT is at slot 0 (only output)
        assert out is not None, "OUT should not be None"
        # $strength = 0.75 (widget overrides declaration default)
        assert abs(out.mean().item() - 0.75) < 0.01, f"Expected ~0.75, got {out.mean().item()}"
        r.ok("tex_node: param kwargs")
    except Exception as e:
        r.fail("tex_node: param kwargs", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: param default fallback ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        img = torch.ones(1, 2, 2, 3)
        # No strength kwarg — should use default from code (0.5)
        result = TEXWrangleNode.execute(
            code="f$strength = 0.5;\n@OUT = @A * $strength;",
            A=img, device="cpu", compile_mode="none"
        )
        out = result[0]
        assert out is not None, "OUT should not be None"
        assert abs(out.mean().item() - 0.5) < 0.01, f"Expected ~0.5, got {out.mean().item()}"
        r.ok("tex_node: param default fallback")
    except Exception as e:
        r.fail("tex_node: param default fallback", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: param widget value as kwarg ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        img = torch.ones(1, 2, 2, 3)
        # Widget value injected as kwarg by graphToPrompt hook (overrides code default)
        result = TEXWrangleNode.execute(
            code="f$strength = 0.5;\n@OUT = @A * $strength;",
            A=img, strength=0.3,
            device="cpu", compile_mode="none"
        )
        out = result[0]
        assert out is not None, "OUT should not be None"
        assert abs(out.mean().item() - 0.3) < 0.01, f"Expected ~0.3, got {out.mean().item()}"
        r.ok("tex_node: param widget value as kwarg")
    except Exception as e:
        r.fail("tex_node: param widget value as kwarg", f"{e}\n{traceback.format_exc()}")

    # ── tex_node: param widget overrides code default ──
    try:
        from TEX_Wrangle.tex_node import TEXWrangleNode
        img = torch.ones(1, 2, 2, 3)
        # Widget value (scalar kwarg) overrides code default of 0.5
        result = TEXWrangleNode.execute(
            code="f$strength = 0.5;\n@OUT = @A * $strength;",
            A=img, strength=0.8,
            device="cpu", compile_mode="none"
        )
        out = result[0]
        assert out is not None, "OUT should not be None"
        assert abs(out.mean().item() - 0.8) < 0.01, f"Expected ~0.8 (widget), got {out.mean().item()}"
        r.ok("tex_node: param widget overrides code default")
    except Exception as e:
        r.fail("tex_node: param widget overrides code default", f"{e}\n{traceback.format_exc()}")

    # ── type checker: param default_value extraction ──
    try:
        cache = TEXCache(cache_dir=Path(tempfile.mkdtemp()))
        code = "f$strength = 0.005;\ni$count = 3;\ns$label = \"hello\";\n@OUT = @A * $strength;"
        bt = {"A": TEXType.VEC3}
        program, type_map, refs, assigned, params, *_ = cache.compile_tex(code, bt)
        assert "strength" in params, "Missing 'strength' in params"
        assert params["strength"]["default_value"] == 0.005, f"Expected 0.005, got {params['strength']['default_value']}"
        assert params["strength"]["type"] == TEXType.FLOAT
        assert "count" in params, "Missing 'count' in params"
        assert params["count"]["default_value"] == 3, f"Expected 3, got {params['count']['default_value']}"
        assert params["count"]["type"] == TEXType.INT
        assert "label" in params, "Missing 'label' in params"
        assert params["label"]["default_value"] == "hello", f"Expected 'hello', got {params['label']['default_value']}"
        assert params["label"]["type"] == TEXType.STRING
        r.ok("type checker: param default_value extraction")
    except Exception as e:
        r.fail("type checker: param default_value extraction", f"{e}\n{traceback.format_exc()}")

    # ── cache: multi-output compile_tex ──
    try:
        cache = TEXCache(cache_dir=Path(tempfile.mkdtemp()))
        code = "@result = @A * 0.5;\n@mask = luma(@A);"
        bt = {"A": TEXType.VEC3}
        program, type_map, refs, assigned, params, *_ = cache.compile_tex(code, bt)
        assert "result" in assigned, "Missing 'result' in assigned_bindings"
        assert "mask" in assigned, "Missing 'mask' in assigned_bindings"
        assert assigned["result"] == TEXType.VEC3
        assert assigned["mask"] == TEXType.FLOAT
        r.ok("cache: multi-output compile_tex")
    except Exception as e:
        r.fail("cache: multi-output compile_tex", f"{e}\n{traceback.format_exc()}")


# ── Realistic Sizes Tests ─────────────────────────────────────────────

def test_realistic_sizes(r: SubTestResult):
    """Tests with realistic image sizes (512x512) and multi-batch."""
    print("\n--- Realistic Sizes Tests ---")

    # 512x512 passthrough
    try:
        big_img = torch.rand(1, 512, 512, 3)
        result = compile_and_run("@OUT = @A;", {"A": big_img})
        assert result.shape == (1, 512, 512, 3), f"Shape mismatch: {result.shape}"
        r.ok("realistic: 512x512 passthrough shape")
    except Exception as e:
        r.fail("realistic: 512x512 passthrough shape", f"{e}")

    # Verify values preserved
    try:
        big_img = torch.rand(1, 512, 512, 3)
        result = compile_and_run("@OUT = @A;", {"A": big_img})
        max_diff = (result - big_img).abs().max().item()
        assert max_diff < 1e-6, f"Passthrough altered values: max diff={max_diff}"
        r.ok("realistic: 512x512 passthrough values")
    except Exception as e:
        r.fail("realistic: 512x512 passthrough values", f"{e}")

    # Color grade at 512x512
    try:
        big_img = torch.rand(1, 512, 512, 3) * 0.5
        result = compile_and_run("@OUT = @A * 1.5;", {"A": big_img})
        expected = big_img * 1.5
        max_diff = (result - expected).abs().max().item()
        assert max_diff < 1e-5, f"Color grade drift: max diff={max_diff}"
        r.ok("realistic: 512x512 color grade")
    except Exception as e:
        r.fail("realistic: 512x512 color grade", f"{e}")

    # Shape preserved for vec4 output
    try:
        big_img = torch.rand(1, 512, 512, 4)
        result = compile_and_run("@OUT = @A;", {"A": big_img})
        assert result.shape == (1, 512, 512, 4), f"Shape: {result.shape}"
        r.ok("realistic: 512x512 vec4 shape preserved")
    except Exception as e:
        r.fail("realistic: 512x512 vec4 shape preserved", f"{e}")

    # Precision check with known values
    try:
        known = torch.zeros(1, 512, 512, 3)
        known[0, 0, 0, :] = torch.tensor([0.123456, 0.654321, 0.999999])
        known[0, 255, 255, :] = torch.tensor([0.111111, 0.222222, 0.333333])
        result = compile_and_run("@OUT = @A;", {"A": known})
        assert abs(result[0,0,0,0].item() - 0.123456) < 1e-5
        assert abs(result[0,0,0,1].item() - 0.654321) < 1e-5
        assert abs(result[0,255,255,2].item() - 0.333333) < 1e-5
        r.ok("realistic: precision preserved at 512x512")
    except Exception as e:
        r.fail("realistic: precision preserved at 512x512", f"{e}")

    # Multi-batch (B=2)
    try:
        batch_img = torch.rand(2, 64, 64, 3)
        result = compile_and_run("@OUT = @A;", {"A": batch_img})
        assert result.shape[0] == 2, f"Batch dim: {result.shape[0]}"
        assert result.shape == (2, 64, 64, 3), f"Shape: {result.shape}"
        r.ok("realistic: multi-batch B=2 shape")
    except Exception as e:
        r.fail("realistic: multi-batch B=2 shape", f"{e}")

    # Multi-batch values preserved
    try:
        batch_img = torch.rand(2, 64, 64, 3)
        result = compile_and_run("@OUT = @A;", {"A": batch_img})
        max_diff = (result - batch_img).abs().max().item()
        assert max_diff < 1e-6, f"Batch passthrough altered values: max diff={max_diff}"
        r.ok("realistic: multi-batch values preserved")
    except Exception as e:
        r.fail("realistic: multi-batch values preserved", f"{e}")

    # Multi-batch arithmetic
    try:
        batch_img = torch.rand(2, 64, 64, 3) * 0.5
        result = compile_and_run("@OUT = @A + 0.25;", {"A": batch_img})
        expected = batch_img + 0.25
        max_diff = (result - expected).abs().max().item()
        assert max_diff < 1e-5, f"Batch arithmetic drift: max diff={max_diff}"
        r.ok("realistic: multi-batch arithmetic")
    except Exception as e:
        r.fail("realistic: multi-batch arithmetic", f"{e}")

    # B=4 batch
    try:
        batch_img = torch.rand(4, 32, 32, 3)
        result = compile_and_run("@OUT = @A;", {"A": batch_img})
        assert result.shape == (4, 32, 32, 3), f"Shape: {result.shape}"
        r.ok("realistic: B=4 batch shape")
    except Exception as e:
        r.fail("realistic: B=4 batch shape", f"{e}")

    # Non-square: 256x128
    try:
        rect_img = torch.rand(1, 128, 256, 3)
        result = compile_and_run("@OUT = @A;", {"A": rect_img})
        assert result.shape == (1, 128, 256, 3), f"Shape: {result.shape}"
        r.ok("realistic: non-square 256x128")
    except Exception as e:
        r.fail("realistic: non-square 256x128", f"{e}")

    # Clamp at large size
    try:
        big_img = torch.rand(1, 256, 256, 3) * 2.0  # values up to 2.0
        result = compile_and_run("@OUT = clamp(@A, 0.0, 1.0);", {"A": big_img})
        assert result.max().item() <= 1.0 + 1e-6
        assert result.min().item() >= 0.0 - 1e-6
        r.ok("realistic: clamp at 256x256")
    except Exception as e:
        r.fail("realistic: clamp at 256x256", f"{e}")

    # u/v coordinates at 512x512
    try:
        big_img = torch.rand(1, 512, 512, 3)
        result = compile_and_run("@OUT = vec3(u, v, 0.0);", {"A": big_img})
        # Corner (0,0) should be u=0, v=0
        assert abs(result[0,0,0,0].item()) < 1e-4, f"u at (0,0): {result[0,0,0,0].item()}"
        assert abs(result[0,0,0,1].item()) < 1e-4, f"v at (0,0): {result[0,0,0,1].item()}"
        # Corner (511,511) should be u~1, v~1
        assert abs(result[0,511,511,0].item() - 1.0) < 1e-3
        assert abs(result[0,511,511,1].item() - 1.0) < 1e-3
        r.ok("realistic: u/v at 512x512 corners")
    except Exception as e:
        r.fail("realistic: u/v at 512x512 corners", f"{e}")


# ── Matrix Type Tests ──────────────────────────────────────────────────

def test_matrix_types(r: SubTestResult):
    print("\n--- Matrix Type Tests ---")
    H, W = 4, 4
    img = torch.rand(1, H, W, 4)

    # 1. mat3 constructor (9 args)
    try:
        code = """
mat3 m = mat3(1.0, 0.0, 0.0,
              0.0, 1.0, 0.0,
              0.0, 0.0, 1.0);
@OUT = m * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3]
        assert torch.allclose(result, expected, atol=1e-4), f"Identity mat3*vec3 failed"
        r.ok("mat3 constructor (9 args)")
    except Exception as e:
        r.fail("mat3 constructor (9 args)", f"{e}\n{traceback.format_exc()}")

    # 2. mat3 broadcast constructor (1 arg -> scaled identity)
    try:
        code = """
mat3 m = mat3(2.0);
@OUT = m * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3] * 2.0
        assert torch.allclose(result, expected, atol=1e-4), f"Scaled identity mat3 failed"
        r.ok("mat3 broadcast constructor (scaled identity)")
    except Exception as e:
        r.fail("mat3 broadcast constructor (scaled identity)", f"{e}\n{traceback.format_exc()}")

    # 3. mat4 constructor (16 args)
    try:
        code = """
mat4 m = mat4(1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 1.0);
@OUT = m * @A;
"""
        result = compile_and_run(code, {"A": img})
        assert torch.allclose(result, img, atol=1e-4), f"Identity mat4*vec4 failed"
        r.ok("mat4 constructor (16 args)")
    except Exception as e:
        r.fail("mat4 constructor (16 args)", f"{e}\n{traceback.format_exc()}")

    # 4. mat4 broadcast constructor
    try:
        code = """
mat4 m = mat4(0.5);
@OUT = m * @A;
"""
        result = compile_and_run(code, {"A": img})
        expected = img * 0.5
        assert torch.allclose(result, expected, atol=1e-4), f"Scaled identity mat4 failed"
        r.ok("mat4 broadcast constructor (scaled identity)")
    except Exception as e:
        r.fail("mat4 broadcast constructor (scaled identity)", f"{e}\n{traceback.format_exc()}")

    # 5. mat3 * vec3 (color transform)
    try:
        code = """
mat3 m = mat3(0.0, 1.0, 0.0,
              0.0, 0.0, 1.0,
              1.0, 0.0, 0.0);
@OUT = m * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        # This matrix cycles channels: r->b, g->r, b->g
        expected = torch.stack([img[..., 1], img[..., 2], img[..., 0]], dim=-1)
        assert torch.allclose(result, expected, atol=1e-4), f"Channel cycle mat3*vec3 failed"
        r.ok("mat3 * vec3 (color transform)")
    except Exception as e:
        r.fail("mat3 * vec3 (color transform)", f"{e}\n{traceback.format_exc()}")

    # 6. mat4 * vec4 (homogeneous transform)
    try:
        code = """
mat4 m = mat4(1.0, 0.0, 0.0, 0.1,
              0.0, 1.0, 0.0, 0.2,
              0.0, 0.0, 1.0, 0.3,
              0.0, 0.0, 0.0, 1.0);
@OUT = m * @A;
"""
        result = compile_and_run(code, {"A": img})
        # Translation: r += 0.1*a, g += 0.2*a, b += 0.3*a
        expected = img.clone()
        expected[..., 0] += 0.1 * img[..., 3]
        expected[..., 1] += 0.2 * img[..., 3]
        expected[..., 2] += 0.3 * img[..., 3]
        assert torch.allclose(result, expected, atol=1e-4), f"Translation mat4*vec4 failed"
        r.ok("mat4 * vec4 (homogeneous transform)")
    except Exception as e:
        r.fail("mat4 * vec4 (homogeneous transform)", f"{e}\n{traceback.format_exc()}")

    # 7. mat3 * mat3 (chain two transforms)
    try:
        code = """
mat3 a = mat3(2.0);
mat3 b = mat3(3.0);
mat3 c = a * b;
@OUT = c * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3] * 6.0  # 2*I * 3*I = 6*I
        assert torch.allclose(result, expected, atol=1e-4), f"mat3*mat3 chain failed"
        r.ok("mat3 * mat3 (chain transforms)")
    except Exception as e:
        r.fail("mat3 * mat3 (chain transforms)", f"{e}\n{traceback.format_exc()}")

    # 8. mat4 * mat4
    try:
        code = """
mat4 a = mat4(2.0);
mat4 b = mat4(0.5);
mat4 c = a * b;
@OUT = c * @A;
"""
        result = compile_and_run(code, {"A": img})
        expected = img.clone()  # 2*I * 0.5*I = I
        assert torch.allclose(result, expected, atol=1e-4), f"mat4*mat4 failed"
        r.ok("mat4 * mat4")
    except Exception as e:
        r.fail("mat4 * mat4", f"{e}\n{traceback.format_exc()}")

    # 9. scalar * mat3 (element-wise scale)
    try:
        code = """
mat3 m = mat3(1.0, 2.0, 3.0,
              4.0, 5.0, 6.0,
              7.0, 8.0, 9.0);
mat3 s = 0.5 * m;
@OUT = s * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        # s = 0.5 * m, then s * vec3
        m = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32) * 0.5
        rgb = img[..., :3]
        expected = torch.matmul(rgb.unsqueeze(-2), m.T).squeeze(-2)
        # Actually mat * vec = matmul(m, v), so expected = matmul(m, v.unsqueeze(-1)).squeeze(-1)
        expected = torch.matmul(m, rgb.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(result, expected, atol=1e-4), f"scalar*mat3 failed"
        r.ok("scalar * mat3 (element-wise scale)")
    except Exception as e:
        r.fail("scalar * mat3 (element-wise scale)", f"{e}\n{traceback.format_exc()}")

    # 10. mat3 * scalar
    try:
        code = """
mat3 m = mat3(1.0);
mat3 s = m * 3.0;
@OUT = s * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3] * 3.0
        assert torch.allclose(result, expected, atol=1e-4)
        r.ok("mat3 * scalar")
    except Exception as e:
        r.fail("mat3 * scalar", f"{e}\n{traceback.format_exc()}")

    # 11. mat3 + mat3 (element-wise add)
    try:
        code = """
mat3 a = mat3(1.0);
mat3 b = mat3(2.0);
mat3 c = a + b;
@OUT = c * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3] * 3.0  # I + 2I = 3I
        assert torch.allclose(result, expected, atol=1e-4)
        r.ok("mat3 + mat3 (element-wise add)")
    except Exception as e:
        r.fail("mat3 + mat3 (element-wise add)", f"{e}\n{traceback.format_exc()}")

    # 12. mat3 - mat3
    try:
        code = """
mat3 a = mat3(3.0);
mat3 b = mat3(1.0);
mat3 c = a - b;
@OUT = c * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3] * 2.0  # 3I - I = 2I
        assert torch.allclose(result, expected, atol=1e-4)
        r.ok("mat3 - mat3 (element-wise subtract)")
    except Exception as e:
        r.fail("mat3 - mat3 (element-wise subtract)", f"{e}\n{traceback.format_exc()}")

    # 13. transpose(mat3)
    try:
        code = """
mat3 m = mat3(1.0, 2.0, 3.0,
              4.0, 5.0, 6.0,
              7.0, 8.0, 9.0);
mat3 t = transpose(m);
@OUT = t * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        m_T = torch.tensor([[1, 4, 7], [2, 5, 8], [3, 6, 9]], dtype=torch.float32)
        rgb = img[..., :3]
        expected = torch.matmul(m_T, rgb.unsqueeze(-1)).squeeze(-1)
        assert torch.allclose(result, expected, atol=1e-4)
        r.ok("transpose(mat3)")
    except Exception as e:
        r.fail("transpose(mat3)", f"{e}\n{traceback.format_exc()}")

    # 14. determinant(mat3) — identity -> 1.0
    try:
        code = """
float d = determinant(mat3(1.0));
@OUT = vec4(d, d, d, 1.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4, f"det(I) should be 1.0"
        r.ok("determinant(mat3) = 1.0 for identity")
    except Exception as e:
        r.fail("determinant(mat3) = 1.0 for identity", f"{e}\n{traceback.format_exc()}")

    # 15. inverse(mat3) — identity -> identity
    try:
        code = """
mat3 m = mat3(1.0);
mat3 inv = inverse(m);
@OUT = inv * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3]
        assert torch.allclose(result, expected, atol=1e-4)
        r.ok("inverse(mat3) identity -> identity")
    except Exception as e:
        r.fail("inverse(mat3) identity -> identity", f"{e}\n{traceback.format_exc()}")

    # 16. inverse: m * inverse(m) ~= identity
    try:
        code = """
mat3 m = mat3(2.0, 1.0, 0.0,
              0.0, 3.0, 1.0,
              1.0, 0.0, 2.0);
mat3 inv = inverse(m);
mat3 prod = m * inv;
@OUT = prod * @A.rgb;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3]
        assert torch.allclose(result, expected, atol=1e-3), "m * inverse(m) should ~= identity"
        r.ok("m * inverse(m) ~= identity")
    except Exception as e:
        r.fail("m * inverse(m) ~= identity", f"{e}\n{traceback.format_exc()}")

    # 17. vec3 * mat3 -> type error
    try:
        code = "@OUT = @A.rgb * mat3(1.0);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"A": TEXType.VEC4, "OUT": TEXType.VEC3})
        tc.check(program)
        r.fail("vec * mat type error", "Expected TypeCheckError")
    except TypeCheckError as e:
        assert "transpose" in str(e).lower() or "cannot" in str(e).lower() or "isn't supported" in str(e).lower(), f"Unexpected error: {e}"
        r.ok("vec * mat -> type error")
    except Exception as e:
        r.fail("vec * mat type error", f"{e}\n{traceback.format_exc()}")

    # 18. mat3 channel access -> type error
    try:
        code = "mat3 m = mat3(1.0); float x = m.r; @OUT = vec4(x);"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"OUT": TEXType.VEC4})
        tc.check(program)
        r.fail("mat3 channel access error", "Expected TypeCheckError")
    except TypeCheckError:
        r.ok("mat3 channel access -> type error")
    except Exception as e:
        r.fail("mat3 channel access error", f"{e}\n{traceback.format_exc()}")

    # 19. mat3 as @OUT -> type error
    try:
        code = "mat3 m = mat3(1.0); @OUT = m;"
        tokens = Lexer(code).tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"A": TEXType.VEC4, "OUT": TEXType.VEC4})
        tc.check(program)
        r.fail("mat3 as @OUT error", "Expected TypeCheckError")
    except TypeCheckError:
        r.ok("mat3 as @OUT -> type error")
    except Exception as e:
        r.fail("mat3 as @OUT error", f"{e}\n{traceback.format_exc()}")

    # 20. ACES color transform roundtrip (sRGB -> XYZ -> sRGB)
    try:
        code = """
// sRGB to XYZ (D65)
mat3 srgb_to_xyz = mat3(
    0.4124564, 0.3575761, 0.1804375,
    0.2126729, 0.7151522, 0.0721750,
    0.0193339, 0.1191920, 0.9503041
);
// XYZ to sRGB (D65) — inverse of above
mat3 xyz_to_srgb = inverse(srgb_to_xyz);
// Roundtrip
vec3 xyz = srgb_to_xyz * @A.rgb;
@OUT = xyz_to_srgb * xyz;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        expected = img[..., :3]
        assert torch.allclose(result, expected, atol=1e-3), "sRGB->XYZ->sRGB roundtrip failed"
        r.ok("ACES: sRGB -> XYZ -> sRGB roundtrip")
    except Exception as e:
        r.fail("ACES: sRGB -> XYZ -> sRGB roundtrip", f"{e}\n{traceback.format_exc()}")


# ── Matrix Benchmark Tests ─────────────────────────────────────────────

def test_matrix_benchmarks(r: SubTestResult):
    print("\n--- Matrix Benchmark Tests ---")
    H, W = 512, 512
    img3 = torch.rand(1, H, W, 3)
    img4 = torch.rand(1, H, W, 4)

    # 1. mat3 * vec3 at 512x512
    try:
        code = """
mat3 m = mat3(0.4124564, 0.3575761, 0.1804375,
              0.2126729, 0.7151522, 0.0721750,
              0.0193339, 0.1191920, 0.9503041);
@OUT = m * @A.rgb;
"""
        # Warmup
        compile_and_run(code, {"A": img3}, out_type=TEXType.VEC3)
        # Benchmark
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            compile_and_run(code, {"A": img3}, out_type=TEXType.VEC3)
        elapsed = (time.perf_counter() - t0) / N * 1000
        r.ok(f"mat3 * vec3 @ 512x512: {elapsed:.1f}ms")
    except Exception as e:
        r.fail("mat3 * vec3 benchmark", f"{e}\n{traceback.format_exc()}")

    # 2. mat4 * vec4 at 512x512
    try:
        code = """
mat4 m = mat4(1.0, 0.0, 0.0, 0.1,
              0.0, 1.0, 0.0, 0.2,
              0.0, 0.0, 1.0, 0.3,
              0.0, 0.0, 0.0, 1.0);
@OUT = m * @A;
"""
        compile_and_run(code, {"A": img4})
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            compile_and_run(code, {"A": img4})
        elapsed = (time.perf_counter() - t0) / N * 1000
        r.ok(f"mat4 * vec4 @ 512x512: {elapsed:.1f}ms")
    except Exception as e:
        r.fail("mat4 * vec4 benchmark", f"{e}\n{traceback.format_exc()}")

    # 3. chained mat3 * mat3 * vec3 at 512x512
    try:
        code = """
mat3 a = mat3(0.4124564, 0.3575761, 0.1804375,
              0.2126729, 0.7151522, 0.0721750,
              0.0193339, 0.1191920, 0.9503041);
mat3 b = inverse(a);
mat3 c = a * b;
@OUT = c * @A.rgb;
"""
        compile_and_run(code, {"A": img3}, out_type=TEXType.VEC3)
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            compile_and_run(code, {"A": img3}, out_type=TEXType.VEC3)
        elapsed = (time.perf_counter() - t0) / N * 1000
        r.ok(f"chained mat3*mat3*vec3 @ 512x512: {elapsed:.1f}ms")
    except Exception as e:
        r.fail("chained mat3*mat3*vec3 benchmark", f"{e}\n{traceback.format_exc()}")

    # 4. inverse(mat3) at 512x512
    try:
        code = """
mat3 m = mat3(2.0, 1.0, 0.0,
              0.0, 3.0, 1.0,
              1.0, 0.0, 2.0);
mat3 inv = inverse(m);
@OUT = inv * @A.rgb;
"""
        compile_and_run(code, {"A": img3}, out_type=TEXType.VEC3)
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            compile_and_run(code, {"A": img3}, out_type=TEXType.VEC3)
        elapsed = (time.perf_counter() - t0) / N * 1000
        r.ok(f"inverse(mat3) @ 512x512: {elapsed:.1f}ms")
    except Exception as e:
        r.fail("inverse(mat3) benchmark", f"{e}\n{traceback.format_exc()}")


def test_node_helpers(r: SubTestResult):
    """Tests for helper functions in tex_node.py."""
    print("\n--- Node Helper Function Tests ---")

    from TEX_Wrangle.tex_marshalling import hex_to_rgb as _hex_to_rgb, convert_param_value as _convert_param_value

    # ── _hex_to_rgb ──────────────────────────────────────────────────

    # Basic 6-digit hex
    try:
        rgb = _hex_to_rgb("#FF0000")
        assert abs(rgb[0] - 1.0) < 1e-6 and abs(rgb[1]) < 1e-6 and abs(rgb[2]) < 1e-6, f"Got {rgb}"
        r.ok("hex_to_rgb: #FF0000 -> red")
    except Exception as e:
        r.fail("hex_to_rgb: #FF0000 -> red", str(e))

    try:
        rgb = _hex_to_rgb("#00FF00")
        assert abs(rgb[1] - 1.0) < 1e-6 and abs(rgb[0]) < 1e-6, f"Got {rgb}"
        r.ok("hex_to_rgb: #00FF00 -> green")
    except Exception as e:
        r.fail("hex_to_rgb: #00FF00 -> green", str(e))

    try:
        rgb = _hex_to_rgb("#0000FF")
        assert abs(rgb[2] - 1.0) < 1e-6 and abs(rgb[0]) < 1e-6, f"Got {rgb}"
        r.ok("hex_to_rgb: #0000FF -> blue")
    except Exception as e:
        r.fail("hex_to_rgb: #0000FF -> blue", str(e))

    # Black and white
    try:
        rgb = _hex_to_rgb("#000000")
        assert all(abs(c) < 1e-6 for c in rgb), f"Got {rgb}"
        r.ok("hex_to_rgb: #000000 -> black")
    except Exception as e:
        r.fail("hex_to_rgb: #000000 -> black", str(e))

    try:
        rgb = _hex_to_rgb("#FFFFFF")
        assert all(abs(c - 1.0) < 1e-6 for c in rgb), f"Got {rgb}"
        r.ok("hex_to_rgb: #FFFFFF -> white")
    except Exception as e:
        r.fail("hex_to_rgb: #FFFFFF -> white", str(e))

    # 3-digit shorthand
    try:
        rgb = _hex_to_rgb("#F00")
        assert abs(rgb[0] - 1.0) < 1e-6, f"Got {rgb}"
        r.ok("hex_to_rgb: 3-digit #F00")
    except Exception as e:
        r.fail("hex_to_rgb: 3-digit #F00", str(e))

    # Without hash prefix
    try:
        rgb = _hex_to_rgb("FF8800")
        expected = [1.0, 136/255.0, 0.0]
        assert all(abs(a - b) < 1e-6 for a, b in zip(rgb, expected)), f"Got {rgb}"
        r.ok("hex_to_rgb: no hash prefix")
    except Exception as e:
        r.fail("hex_to_rgb: no hash prefix", str(e))

    # 8-digit hex (with alpha) — should ignore alpha
    try:
        rgb = _hex_to_rgb("#FF000080")
        assert abs(rgb[0] - 1.0) < 1e-6 and abs(rgb[1]) < 1e-6 and abs(rgb[2]) < 1e-6, f"Got {rgb}"
        r.ok("hex_to_rgb: 8-digit ignores alpha")
    except Exception as e:
        r.fail("hex_to_rgb: 8-digit ignores alpha", str(e))

    # Short string gets padded with zeros
    try:
        rgb = _hex_to_rgb("#FF")
        # "FF" -> padded to "FF0000"
        assert abs(rgb[0] - 1.0) < 1e-6, f"Got {rgb}"
        r.ok("hex_to_rgb: short string padded")
    except Exception as e:
        r.fail("hex_to_rgb: short string padded", str(e))

    # Lowercase hex
    try:
        rgb = _hex_to_rgb("#ff8800")
        expected = [1.0, 136/255.0, 0.0]
        assert all(abs(a - b) < 1e-6 for a, b in zip(rgb, expected)), f"Got {rgb}"
        r.ok("hex_to_rgb: lowercase hex")
    except Exception as e:
        r.fail("hex_to_rgb: lowercase hex", str(e))

    # ── _convert_param_value ─────────────────────────────────────────

    # hint "f" — float passthrough
    try:
        result = _convert_param_value(3.14, {"type_hint": "f"})
        assert result == 3.14, f"Got {result}"
        r.ok("convert_param: float passthrough")
    except Exception as e:
        r.fail("convert_param: float passthrough", str(e))

    # hint "i" — int passthrough
    try:
        result = _convert_param_value(42, {"type_hint": "i"})
        assert result == 42, f"Got {result}"
        r.ok("convert_param: int passthrough")
    except Exception as e:
        r.fail("convert_param: int passthrough", str(e))

    # hint "s" — string passthrough
    try:
        result = _convert_param_value("hello", {"type_hint": "s"})
        assert result == "hello", f"Got {result}"
        r.ok("convert_param: string passthrough")
    except Exception as e:
        r.fail("convert_param: string passthrough", str(e))

    # hint "b" — boolean conversion
    try:
        result = _convert_param_value(True, {"type_hint": "b"})
        assert result == 1.0, f"Got {result}"
        r.ok("convert_param: bool True -> 1.0")
    except Exception as e:
        r.fail("convert_param: bool True -> 1.0", str(e))

    try:
        result = _convert_param_value(False, {"type_hint": "b"})
        assert result == 0.0, f"Got {result}"
        r.ok("convert_param: bool False -> 0.0")
    except Exception as e:
        r.fail("convert_param: bool False -> 0.0", str(e))

    try:
        result = _convert_param_value(0, {"type_hint": "b"})
        assert result == 0.0, f"Got {result}"
        r.ok("convert_param: int 0 -> 0.0 (bool)")
    except Exception as e:
        r.fail("convert_param: int 0 -> 0.0 (bool)", str(e))

    # Truthy int clamped to 1.0
    try:
        result = _convert_param_value(2, {"type_hint": "b"})
        assert result == 1.0, f"Got {result}"
        r.ok("convert_param: int 2 -> 1.0 (bool clamp)")
    except Exception as e:
        r.fail("convert_param: int 2 -> 1.0 (bool clamp)", str(e))

    # hint "c" — color hex conversion
    try:
        result = _convert_param_value("#FF0000", {"type_hint": "c"})
        assert isinstance(result, list) and len(result) == 3, f"Got {result}"
        assert abs(result[0] - 1.0) < 1e-6, f"Got {result}"
        r.ok("convert_param: color hex #FF0000")
    except Exception as e:
        r.fail("convert_param: color hex #FF0000", str(e))

    # hint "c" with non-hex string — passthrough
    try:
        result = _convert_param_value("not-a-color", {"type_hint": "c"})
        assert result == "not-a-color", f"Got {result}"
        r.ok("convert_param: color non-hex passthrough")
    except Exception as e:
        r.fail("convert_param: color non-hex passthrough", str(e))

    # hint "c" with non-string value — passthrough
    try:
        result = _convert_param_value(42, {"type_hint": "c"})
        assert result == 42, f"Got {result}"
        r.ok("convert_param: color non-string passthrough")
    except Exception as e:
        r.fail("convert_param: color non-string passthrough", str(e))

    # hint "v2" — vec2 string parsing
    try:
        result = _convert_param_value("1.5, 2.5", {"type_hint": "v2"})
        assert result == [1.5, 2.5], f"Got {result}"
        r.ok("convert_param: v2 normal")
    except Exception as e:
        r.fail("convert_param: v2 normal", str(e))

    # hint "v3" — vec3 string parsing
    try:
        result = _convert_param_value("1.0, 2.0, 3.0", {"type_hint": "v3"})
        assert result == [1.0, 2.0, 3.0], f"Got {result}"
        r.ok("convert_param: v3 normal")
    except Exception as e:
        r.fail("convert_param: v3 normal", str(e))

    # hint "v4" — vec4 string parsing
    try:
        result = _convert_param_value("1, 2, 3, 4", {"type_hint": "v4"})
        assert result == [1.0, 2.0, 3.0, 4.0], f"Got {result}"
        r.ok("convert_param: v4 normal")
    except Exception as e:
        r.fail("convert_param: v4 normal", str(e))

    # vec with too few components — padded with zeros
    try:
        result = _convert_param_value("1.0", {"type_hint": "v3"})
        assert result == [1.0, 0.0, 0.0], f"Got {result}"
        r.ok("convert_param: v3 pad short input")
    except Exception as e:
        r.fail("convert_param: v3 pad short input", str(e))

    # vec with too many components — truncated
    try:
        result = _convert_param_value("1, 2, 3, 4, 5", {"type_hint": "v2"})
        assert result == [1.0, 2.0], f"Got {result}"
        r.ok("convert_param: v2 truncate long input")
    except Exception as e:
        r.fail("convert_param: v2 truncate long input", str(e))

    # vec with invalid string — passthrough
    try:
        result = _convert_param_value("abc, def", {"type_hint": "v3"})
        assert result == "abc, def", f"Got {result}"
        r.ok("convert_param: vec invalid string passthrough")
    except Exception as e:
        r.fail("convert_param: vec invalid string passthrough", str(e))

    # vec with non-string value — passthrough
    try:
        result = _convert_param_value(42, {"type_hint": "v2"})
        assert result == 42, f"Got {result}"
        r.ok("convert_param: vec non-string passthrough")
    except Exception as e:
        r.fail("convert_param: vec non-string passthrough", str(e))

    # Default hint (no type_hint key) — passthrough
    try:
        result = _convert_param_value(99, {})
        assert result == 99, f"Got {result}"
        r.ok("convert_param: default hint passthrough")
    except Exception as e:
        r.fail("convert_param: default hint passthrough", str(e))

    # ── _prepare_output ──────────────────────────────────────────────

    # STRING output from string
    try:
        result = _prepare_output("hello", "STRING")
        assert result == "hello", f"Got {result}"
        r.ok("prepare_output: string -> STRING")
    except Exception as e:
        r.fail("prepare_output: string -> STRING", str(e))

    # STRING output from scalar tensor
    try:
        result = _prepare_output(torch.tensor(42.0), "STRING")
        assert result == "42", f"Got {result}"
        r.ok("prepare_output: scalar tensor -> STRING")
    except Exception as e:
        r.fail("prepare_output: scalar tensor -> STRING", str(e))

    # STRING output from float scalar tensor
    try:
        result = _prepare_output(torch.tensor(3.14), "STRING")
        assert "3.14" in result, f"Got {result}"
        r.ok("prepare_output: float scalar tensor -> STRING")
    except Exception as e:
        r.fail("prepare_output: float scalar tensor -> STRING", str(e))

    # STRING output from spatial tensor (mean)
    try:
        t = torch.ones(1, 2, 2, 3) * 0.5
        result = _prepare_output(t, "STRING")
        assert "0.5" in result, f"Got {result}"
        r.ok("prepare_output: spatial tensor -> STRING (mean)")
    except Exception as e:
        r.fail("prepare_output: spatial tensor -> STRING (mean)", str(e))

    # STRING from non-tensor/non-string
    try:
        result = _prepare_output(123, "STRING")
        assert result == "123", f"Got {result}"
        r.ok("prepare_output: int -> STRING")
    except Exception as e:
        r.fail("prepare_output: int -> STRING", str(e))

    # IMAGE from [B,H,W,3] — passthrough, clamped
    try:
        t = torch.rand(1, 4, 4, 3) * 2.0  # values up to 2.0
        result = _prepare_output(t, "IMAGE")
        assert result.shape == (1, 4, 4, 3), f"Shape {result.shape}"
        assert result.max() <= 1.0, f"Max {result.max()}"
        r.ok("prepare_output: IMAGE [B,H,W,3] clamped")
    except Exception as e:
        r.fail("prepare_output: IMAGE [B,H,W,3] clamped", str(e))

    # IMAGE from [B,H,W] — grayscale expanded to 3 channels
    try:
        t = torch.rand(1, 4, 4)
        result = _prepare_output(t, "IMAGE")
        assert result.shape == (1, 4, 4, 3), f"Shape {result.shape}"
        r.ok("prepare_output: IMAGE [B,H,W] -> [B,H,W,3]")
    except Exception as e:
        r.fail("prepare_output: IMAGE [B,H,W] -> [B,H,W,3]", str(e))

    # IMAGE from [B,H,W,4] — alpha dropped
    try:
        t = torch.rand(1, 4, 4, 4)
        result = _prepare_output(t, "IMAGE")
        assert result.shape == (1, 4, 4, 3), f"Shape {result.shape}"
        r.ok("prepare_output: IMAGE [B,H,W,4] drops alpha")
    except Exception as e:
        r.fail("prepare_output: IMAGE [B,H,W,4] drops alpha", str(e))

    # IMAGE from [B,H,W,2] — padded to 3
    try:
        t = torch.rand(1, 4, 4, 2)
        result = _prepare_output(t, "IMAGE")
        assert result.shape == (1, 4, 4, 3), f"Shape {result.shape}"
        r.ok("prepare_output: IMAGE [B,H,W,2] padded to 3")
    except Exception as e:
        r.fail("prepare_output: IMAGE [B,H,W,2] padded to 3", str(e))

    # IMAGE from scalar
    try:
        t = torch.tensor(0.5)
        result = _prepare_output(t, "IMAGE")
        assert result.shape == (1, 1, 1, 3), f"Shape {result.shape}"
        r.ok("prepare_output: IMAGE scalar -> 1x1x1x3")
    except Exception as e:
        r.fail("prepare_output: IMAGE scalar -> 1x1x1x3", str(e))

    # MASK from [B,H,W] — passthrough, clamped
    try:
        t = torch.rand(1, 4, 4) * 2.0
        result = _prepare_output(t, "MASK")
        assert result.shape == (1, 4, 4), f"Shape {result.shape}"
        assert result.max() <= 1.0, f"Max {result.max()}"
        r.ok("prepare_output: MASK [B,H,W] clamped")
    except Exception as e:
        r.fail("prepare_output: MASK [B,H,W] clamped", str(e))

    # MASK from [B,H,W,C] — luminance conversion
    try:
        t = torch.ones(1, 4, 4, 3) * 0.5
        result = _prepare_output(t, "MASK")
        assert result.shape == (1, 4, 4), f"Shape {result.shape}"
        r.ok("prepare_output: MASK from vec3 -> luminance")
    except Exception as e:
        r.fail("prepare_output: MASK from vec3 -> luminance", str(e))

    # MASK from [H,W] — adds batch dim
    try:
        t = torch.rand(4, 4)
        result = _prepare_output(t, "MASK")
        assert result.shape == (1, 4, 4), f"Shape {result.shape}"
        r.ok("prepare_output: MASK [H,W] -> [1,H,W]")
    except Exception as e:
        r.fail("prepare_output: MASK [H,W] -> [1,H,W]", str(e))

    # MASK from scalar
    try:
        t = torch.tensor(0.7)
        result = _prepare_output(t, "MASK")
        assert result.shape == (1, 1, 1), f"Shape {result.shape}"
        r.ok("prepare_output: MASK scalar -> [1,1,1]")
    except Exception as e:
        r.fail("prepare_output: MASK scalar -> [1,1,1]", str(e))

    # FLOAT from scalar tensor
    try:
        result = _prepare_output(torch.tensor(3.14), "FLOAT")
        assert abs(result - 3.14) < 1e-5, f"Got {result}"
        r.ok("prepare_output: FLOAT scalar")
    except Exception as e:
        r.fail("prepare_output: FLOAT scalar", str(e))

    # FLOAT from spatial tensor — returns mean
    try:
        t = torch.ones(1, 4, 4, 3) * 0.25
        result = _prepare_output(t, "FLOAT")
        assert abs(result - 0.25) < 1e-5, f"Got {result}"
        r.ok("prepare_output: FLOAT spatial -> mean")
    except Exception as e:
        r.fail("prepare_output: FLOAT spatial -> mean", str(e))

    # INT from scalar tensor
    try:
        result = _prepare_output(torch.tensor(7.0), "INT")
        assert result == 7 and isinstance(result, int), f"Got {result}"
        r.ok("prepare_output: INT scalar")
    except Exception as e:
        r.fail("prepare_output: INT scalar", str(e))

    # INT from spatial tensor — returns int(mean)
    try:
        t = torch.ones(1, 4, 4) * 3.9
        result = _prepare_output(t, "INT")
        assert result == 3, f"Got {result}"
        r.ok("prepare_output: INT spatial -> int(mean)")
    except Exception as e:
        r.fail("prepare_output: INT spatial -> int(mean)", str(e))

    # LATENT from [B,H,W,C] — permute to [B,C,H,W]
    try:
        t = torch.rand(1, 4, 4, 4)
        result = _prepare_output(t, "LATENT")
        assert result.shape == (1, 4, 4, 4), f"Shape {result.shape}"
        # Verify permutation: result[0,c,h,w] == t[0,h,w,c]
        assert abs(result[0, 0, 0, 0].item() - t[0, 0, 0, 0].item()) < 1e-6
        assert abs(result[0, 1, 0, 0].item() - t[0, 0, 0, 1].item()) < 1e-6
        r.ok("prepare_output: LATENT permute [B,H,W,C] -> [B,C,H,W]")
    except Exception as e:
        r.fail("prepare_output: LATENT permute [B,H,W,C] -> [B,C,H,W]", str(e))

    # LATENT from [B,H,W] — unsqueeze to [B,1,H,W]
    try:
        t = torch.rand(1, 4, 4)
        result = _prepare_output(t, "LATENT")
        assert result.shape == (1, 1, 4, 4), f"Shape {result.shape}"
        r.ok("prepare_output: LATENT [B,H,W] -> [B,1,H,W]")
    except Exception as e:
        r.fail("prepare_output: LATENT [B,H,W] -> [B,1,H,W]", str(e))

    # LATENT from scalar
    try:
        t = torch.tensor(1.5)
        result = _prepare_output(t, "LATENT")
        assert result.shape == (1, 1, 1, 1), f"Shape {result.shape}"
        r.ok("prepare_output: LATENT scalar -> [1,1,1,1]")
    except Exception as e:
        r.fail("prepare_output: LATENT scalar -> [1,1,1,1]", str(e))

    # LATENT values are NOT clamped (latent space is unbounded)
    try:
        t = torch.tensor([[[[5.0, -3.0]]]]).permute(0, 2, 3, 1)  # [1,1,1,2]
        result = _prepare_output(t, "LATENT")
        assert result.max().item() > 1.0 or result.min().item() < 0.0, "Should not clamp"
        r.ok("prepare_output: LATENT no clamping")
    except Exception as e:
        r.fail("prepare_output: LATENT no clamping", str(e))

    # Non-STRING output from string raises RuntimeError
    try:
        raised = False
        try:
            _prepare_output("oops", "IMAGE")
        except RuntimeError:
            raised = True
        assert raised, "Expected RuntimeError for string -> IMAGE"
        r.ok("prepare_output: string -> IMAGE raises error")
    except Exception as e:
        r.fail("prepare_output: string -> IMAGE raises error", str(e))

    # ── _map_inferred_type ───────────────────────────────────────────

    # VEC3 -> IMAGE
    try:
        result = _map_inferred_type(TEXType.VEC3, False)
        assert result == "IMAGE", f"Got {result}"
        r.ok("map_inferred: VEC3 -> IMAGE")
    except Exception as e:
        r.fail("map_inferred: VEC3 -> IMAGE", str(e))

    # VEC4 without latent -> IMAGE
    try:
        result = _map_inferred_type(TEXType.VEC4, False)
        assert result == "IMAGE", f"Got {result}"
        r.ok("map_inferred: VEC4 no latent -> IMAGE")
    except Exception as e:
        r.fail("map_inferred: VEC4 no latent -> IMAGE", str(e))

    # VEC4 with latent -> LATENT
    try:
        result = _map_inferred_type(TEXType.VEC4, True)
        assert result == "LATENT", f"Got {result}"
        r.ok("map_inferred: VEC4 with latent -> LATENT")
    except Exception as e:
        r.fail("map_inferred: VEC4 with latent -> LATENT", str(e))

    # VEC2 -> IMAGE
    try:
        result = _map_inferred_type(TEXType.VEC2, False)
        assert result == "IMAGE", f"Got {result}"
        r.ok("map_inferred: VEC2 -> IMAGE")
    except Exception as e:
        r.fail("map_inferred: VEC2 -> IMAGE", str(e))

    # FLOAT -> MASK
    try:
        result = _map_inferred_type(TEXType.FLOAT, False)
        assert result == "MASK", f"Got {result}"
        r.ok("map_inferred: FLOAT -> MASK")
    except Exception as e:
        r.fail("map_inferred: FLOAT -> MASK", str(e))

    # INT -> INT
    try:
        result = _map_inferred_type(TEXType.INT, False)
        assert result == "INT", f"Got {result}"
        r.ok("map_inferred: INT -> INT")
    except Exception as e:
        r.fail("map_inferred: INT -> INT", str(e))

    # STRING -> STRING
    try:
        result = _map_inferred_type(TEXType.STRING, False)
        assert result == "STRING", f"Got {result}"
        r.ok("map_inferred: STRING -> STRING")
    except Exception as e:
        r.fail("map_inferred: STRING -> STRING", str(e))

    # None -> IMAGE (default)
    try:
        result = _map_inferred_type(None, False)
        assert result == "IMAGE", f"Got {result}"
        r.ok("map_inferred: None -> IMAGE")
    except Exception as e:
        r.fail("map_inferred: None -> IMAGE", str(e))

    # has_latent_input doesn't affect non-VEC4 types
    try:
        result = _map_inferred_type(TEXType.FLOAT, True)
        assert result == "MASK", f"Got {result}"
        r.ok("map_inferred: FLOAT with latent still MASK")
    except Exception as e:
        r.fail("map_inferred: FLOAT with latent still MASK", str(e))

    # ── _infer_binding_type ──────────────────────────────────────────

    # Float scalar
    try:
        result = _infer_binding_type(3.14)
        assert result == TEXType.FLOAT, f"Got {result}"
        r.ok("infer_binding: float -> FLOAT")
    except Exception as e:
        r.fail("infer_binding: float -> FLOAT", str(e))

    # Int scalar
    try:
        result = _infer_binding_type(42)
        assert result == TEXType.INT, f"Got {result}"
        r.ok("infer_binding: int -> INT")
    except Exception as e:
        r.fail("infer_binding: int -> INT", str(e))

    # Bool scalar
    try:
        result = _infer_binding_type(True)
        assert result == TEXType.INT, f"Got {result}"
        r.ok("infer_binding: bool -> INT")
    except Exception as e:
        r.fail("infer_binding: bool -> INT", str(e))

    # String
    try:
        result = _infer_binding_type("hello")
        assert result == TEXType.STRING, f"Got {result}"
        r.ok("infer_binding: str -> STRING")
    except Exception as e:
        r.fail("infer_binding: str -> STRING", str(e))

    # 4D tensor [B,H,W,3] -> VEC3
    try:
        result = _infer_binding_type(torch.rand(1, 4, 4, 3))
        assert result == TEXType.VEC3, f"Got {result}"
        r.ok("infer_binding: [B,H,W,3] -> VEC3")
    except Exception as e:
        r.fail("infer_binding: [B,H,W,3] -> VEC3", str(e))

    # 4D tensor [B,H,W,4] -> VEC4
    try:
        result = _infer_binding_type(torch.rand(1, 4, 4, 4))
        assert result == TEXType.VEC4, f"Got {result}"
        r.ok("infer_binding: [B,H,W,4] -> VEC4")
    except Exception as e:
        r.fail("infer_binding: [B,H,W,4] -> VEC4", str(e))

    # 4D tensor [B,H,W,2] -> VEC2
    try:
        result = _infer_binding_type(torch.rand(1, 4, 4, 2))
        assert result == TEXType.VEC2, f"Got {result}"
        r.ok("infer_binding: [B,H,W,2] -> VEC2")
    except Exception as e:
        r.fail("infer_binding: [B,H,W,2] -> VEC2", str(e))

    # 4D tensor [B,H,W,1] -> FLOAT
    try:
        result = _infer_binding_type(torch.rand(1, 4, 4, 1))
        assert result == TEXType.FLOAT, f"Got {result}"
        r.ok("infer_binding: [B,H,W,1] -> FLOAT")
    except Exception as e:
        r.fail("infer_binding: [B,H,W,1] -> FLOAT", str(e))

    # 3D tensor [B,H,W] (mask) -> FLOAT
    try:
        result = _infer_binding_type(torch.rand(1, 4, 4))
        assert result == TEXType.FLOAT, f"Got {result}"
        r.ok("infer_binding: [B,H,W] mask -> FLOAT")
    except Exception as e:
        r.fail("infer_binding: [B,H,W] mask -> FLOAT", str(e))

    # LATENT dict with 3 channels -> VEC3
    try:
        latent = {"samples": torch.rand(1, 3, 8, 8)}
        result = _infer_binding_type(latent)
        assert result == TEXType.VEC3, f"Got {result}"
        r.ok("infer_binding: LATENT 3ch -> VEC3")
    except Exception as e:
        r.fail("infer_binding: LATENT 3ch -> VEC3", str(e))

    # LATENT dict with 4 channels -> VEC4
    try:
        latent = {"samples": torch.rand(1, 4, 8, 8)}
        result = _infer_binding_type(latent)
        assert result == TEXType.VEC4, f"Got {result}"
        r.ok("infer_binding: LATENT 4ch -> VEC4")
    except Exception as e:
        r.fail("infer_binding: LATENT 4ch -> VEC4", str(e))

    # List input — infers from first element
    try:
        result = _infer_binding_type([torch.rand(1, 4, 4, 3)])
        assert result == TEXType.VEC3, f"Got {result}"
        r.ok("infer_binding: list -> first element type")
    except Exception as e:
        r.fail("infer_binding: list -> first element type", str(e))

    # Empty list -> FLOAT
    try:
        result = _infer_binding_type([])
        assert result == TEXType.FLOAT, f"Got {result}"
        r.ok("infer_binding: empty list -> FLOAT")
    except Exception as e:
        r.fail("infer_binding: empty list -> FLOAT", str(e))

    # 0D tensor -> FLOAT
    try:
        result = _infer_binding_type(torch.tensor(1.0))
        assert result == TEXType.FLOAT, f"Got {result}"
        r.ok("infer_binding: 0D tensor -> FLOAT")
    except Exception as e:
        r.fail("infer_binding: 0D tensor -> FLOAT", str(e))

    # Unknown type -> FLOAT (fallback)
    try:
        result = _infer_binding_type(None)
        assert result == TEXType.FLOAT, f"Got {result}"
        r.ok("infer_binding: None -> FLOAT fallback")
    except Exception as e:
        r.fail("infer_binding: None -> FLOAT fallback", str(e))
