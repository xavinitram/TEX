"""String and array tests."""
from helpers import *


def test_string(r: SubTestResult):
    print("\n--- String Tests ---")

    # 1. String literal
    try:
        result = compile_and_run(
            'string s = "hello"; @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello", f"Expected 'hello', got {result!r}"
        r.ok("string: literal")
    except Exception as e:
        r.fail("string: literal", f"{e}\n{traceback.format_exc()}")

    # 2. String concatenation
    try:
        result = compile_and_run(
            'string s = "hello" + " " + "world"; @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello world", f"Expected 'hello world', got {result!r}"
        r.ok("string: concatenation")
    except Exception as e:
        r.fail("string: concatenation", f"{e}\n{traceback.format_exc()}")

    # 3. String escape sequences
    try:
        result = compile_and_run(
            r'string s = "line1\nline2"; @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "line1\nline2", f"Expected newline, got {result!r}"
        r.ok("string: escape sequences")
    except Exception as e:
        r.fail("string: escape sequences", f"{e}\n{traceback.format_exc()}")

    # 4. String equality
    try:
        result = compile_and_run(
            'float eq = ("abc" == "abc"); @OUT = vec4(eq, eq, eq, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 1.0) < 1e-6, f"Expected 1.0, got {val}"
        r.ok("string: equality")
    except Exception as e:
        r.fail("string: equality", f"{e}\n{traceback.format_exc()}")

    # 5. String inequality
    try:
        result = compile_and_run(
            'float neq = ("abc" != "xyz"); @OUT = vec4(neq, neq, neq, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 1.0) < 1e-6, f"Expected 1.0, got {val}"
        r.ok("string: inequality")
    except Exception as e:
        r.fail("string: inequality", f"{e}\n{traceback.format_exc()}")

    # 6. String binding input
    try:
        result = compile_and_run(
            '@OUT = @A + " world";',
            {"A": "hello"}, out_type=TEXType.STRING)
        assert result == "hello world", f"Expected 'hello world', got {result!r}"
        r.ok("string: binding input")
    except Exception as e:
        r.fail("string: binding input", f"{e}\n{traceback.format_exc()}")

    # 7. str() function
    try:
        result = compile_and_run(
            'string s = str(42); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "42", f"Expected '42', got {result!r}"
        r.ok("string: str() function")
    except Exception as e:
        r.fail("string: str() function", f"{e}\n{traceback.format_exc()}")

    # 8. string() cast
    try:
        result = compile_and_run(
            'string s = string(3.14); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert "3.14" in result, f"Expected '3.14' in result, got {result!r}"
        r.ok("string: string() cast")
    except Exception as e:
        r.fail("string: string() cast", f"{e}\n{traceback.format_exc()}")

    # 9. len()
    try:
        result = compile_and_run(
            'float n = len("hello"); @OUT = vec4(n, n, n, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 5.0) < 1e-6, f"Expected 5.0, got {val}"
        r.ok("string: len()")
    except Exception as e:
        r.fail("string: len()", f"{e}\n{traceback.format_exc()}")

    # 10. replace()
    try:
        result = compile_and_run(
            'string s = replace("hello world", "world", "TEX"); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello TEX", f"Expected 'hello TEX', got {result!r}"
        r.ok("string: replace()")
    except Exception as e:
        r.fail("string: replace()", f"{e}\n{traceback.format_exc()}")

    # 11. strip()
    try:
        result = compile_and_run(
            'string s = strip("  hello  "); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello", f"Expected 'hello', got {result!r}"
        r.ok("string: strip()")
    except Exception as e:
        r.fail("string: strip()", f"{e}\n{traceback.format_exc()}")

    # 12. upper()
    try:
        result = compile_and_run(
            'string s = upper("hello"); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "HELLO", f"Expected 'HELLO', got {result!r}"
        r.ok("string: upper()")
    except Exception as e:
        r.fail("string: upper()", f"{e}\n{traceback.format_exc()}")

    # 13. contains()
    try:
        result = compile_and_run(
            'float c = contains("hello world", "world"); @OUT = vec4(c, c, c, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 1.0) < 1e-6, f"Expected 1.0, got {val}"
        r.ok("string: contains()")
    except Exception as e:
        r.fail("string: contains()", f"{e}\n{traceback.format_exc()}")

    # 14. find()
    try:
        result = compile_and_run(
            'float idx = find("hello", "ll"); @OUT = vec4(idx, idx, idx, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 2.0) < 1e-6, f"Expected 2.0 (index of 'll'), got {val}"
        r.ok("string: find()")
    except Exception as e:
        r.fail("string: find()", f"{e}\n{traceback.format_exc()}")

    # 15. find() not found
    try:
        result = compile_and_run(
            'float idx = find("hello", "xyz"); @OUT = vec4(idx, idx, idx, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - (-1.0)) < 1e-6, f"Expected -1.0, got {val}"
        r.ok("string: find() not found")
    except Exception as e:
        r.fail("string: find() not found", f"{e}\n{traceback.format_exc()}")

    # 16. substr()
    try:
        result = compile_and_run(
            'string s = substr("hello world", 6, 5); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "world", f"Expected 'world', got {result!r}"
        r.ok("string: substr()")
    except Exception as e:
        r.fail("string: substr()", f"{e}\n{traceback.format_exc()}")

    # 17. to_float()
    try:
        result = compile_and_run(
            'float n = to_float("3.14"); @OUT = vec4(n, n, n, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        val = result[0, 0, 0, 0].item()
        assert abs(val - 3.14) < 1e-4, f"Expected 3.14, got {val}"
        r.ok("string: to_float()")
    except Exception as e:
        r.fail("string: to_float()", f"{e}\n{traceback.format_exc()}")

    # 18. sanitize_filename()
    try:
        result = compile_and_run(
            'string s = sanitize_filename("my:file<name>.txt"); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert ':' not in result and '<' not in result and '>' not in result, \
            f"Expected sanitized filename, got {result!r}"
        r.ok("string: sanitize_filename()")
    except Exception as e:
        r.fail("string: sanitize_filename()", f"{e}\n{traceback.format_exc()}")

    # 19. String ternary
    try:
        result = compile_and_run(
            'string s = 1.0 > 0.5 ? "yes" : "no"; @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "yes", f"Expected 'yes', got {result!r}"
        r.ok("string: ternary")
    except Exception as e:
        r.fail("string: ternary", f"{e}\n{traceback.format_exc()}")

    # 20. String in if/else
    try:
        result = compile_and_run(
            'string s = "default"; if (1.0 > 0.5) { s = "then"; } else { s = "else"; } @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "then", f"Expected 'then', got {result!r}"
        r.ok("string: if/else")
    except Exception as e:
        r.fail("string: if/else", f"{e}\n{traceback.format_exc()}")

    # 21. String in for loop
    try:
        result = compile_and_run(
            'string s = ""; for (int i = 0; i < 3; i++) { s = s + "x"; } @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "xxx", f"Expected 'xxx', got {result!r}"
        r.ok("string: for loop accumulation")
    except Exception as e:
        r.fail("string: for loop accumulation", f"{e}\n{traceback.format_exc()}")

    # 22. Type error: string + number
    try:
        compile_and_run(
            'string s = "hello" + 1;',
            {}, out_type=TEXType.STRING)
        r.fail("string: type error string+number", "Expected TypeCheckError")
    except (TypeCheckError, TEXMultiError, InterpreterError):
        r.ok("string: type error string+number")
    except Exception as e:
        r.fail("string: type error string+number", f"Wrong error type: {e}")

    # 23. Type error: string in vec constructor
    try:
        compile_and_run(
            'vec3 c = vec3("hello", 0.0, 0.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        r.fail("string: type error vec constructor", "Expected TypeCheckError")
    except (TypeCheckError, TEXMultiError):
        r.ok("string: type error vec constructor")
    except Exception as e:
        r.fail("string: type error vec constructor", f"Wrong error type: {e}")

    # 24. Type error: channel access on string
    try:
        compile_and_run(
            'string s = "abc"; float x = s.r;',
            {}, out_type=TEXType.STRING)
        r.fail("string: type error channel access", "Expected TypeCheckError")
    except TypeCheckError:
        r.ok("string: type error channel access")
    except Exception as e:
        r.fail("string: type error channel access", f"Wrong error type: {e}")

    # 25. startswith / endswith
    try:
        result = compile_and_run(
            'float sw = startswith("hello", "hel"); float ew = endswith("hello", "llo"); '
            '@OUT = vec4(sw, ew, 0.0, 1.0);',
            {"A": torch.zeros(1, 2, 2, 4)})
        sw_val = result[0, 0, 0, 0].item()
        ew_val = result[0, 0, 0, 1].item()
        assert abs(sw_val - 1.0) < 1e-6 and abs(ew_val - 1.0) < 1e-6, \
            f"Expected (1.0, 1.0), got ({sw_val}, {ew_val})"
        r.ok("string: startswith/endswith")
    except Exception as e:
        r.fail("string: startswith/endswith", f"{e}\n{traceback.format_exc()}")


def test_string_functions_v04(r: SubTestResult):
    """Tests for new string functions added in v0.4."""
    print("\n--- String Functions v0.4 Tests ---")
    B, H, W = 1, 2, 2
    img = torch.rand(B, H, W, 4)

    # -- split --
    try:
        code = '''
string s = "a,b,c,d";
string arr[] = split(s, ",");
@OUT = arr[2];
'''
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "c", f"Expected 'c', got {result!r}"
        r.ok("split: basic delimiter")
    except Exception as e:
        r.fail("split: basic delimiter", f"{e}\n{traceback.format_exc()}")

    try:
        code = '''
string s = "a.b.c.d";
string arr[] = split(s, ".", 2);
@OUT = arr[2];
'''
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "c.d", f"Expected 'c.d', got {result!r}"
        r.ok("split: max_splits")
    except Exception as e:
        r.fail("split: max_splits", f"{e}\n{traceback.format_exc()}")

    try:
        code = '''
string s = "hello world foo";
string arr[] = split(s, " ");
@OUT = join(arr, "-");
'''
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "hello-world-foo", f"Expected 'hello-world-foo', got {result!r}"
        r.ok("split: round-trip with join")
    except Exception as e:
        r.fail("split: round-trip with join", f"{e}\n{traceback.format_exc()}")

    # -- lstrip / rstrip --
    try:
        result = compile_and_run(
            'string s = "  hello  "; @OUT = lstrip(s);',
            {}, out_type=TEXType.STRING)
        assert result == "hello  ", f"Expected 'hello  ', got {result!r}"
        r.ok("lstrip: trim leading")
    except Exception as e:
        r.fail("lstrip: trim leading", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            'string s = "  hello  "; @OUT = rstrip(s);',
            {}, out_type=TEXType.STRING)
        assert result == "  hello", f"Expected '  hello', got {result!r}"
        r.ok("rstrip: trim trailing")
    except Exception as e:
        r.fail("rstrip: trim trailing", f"{e}\n{traceback.format_exc()}")

    # -- pad_left / pad_right --
    try:
        result = compile_and_run(
            '@OUT = pad_left("42", 5, "0");',
            {}, out_type=TEXType.STRING)
        assert result == "00042", f"Expected '00042', got {result!r}"
        r.ok("pad_left: zero-pad number")
    except Exception as e:
        r.fail("pad_left: zero-pad number", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = pad_right("hi", 6);',
            {}, out_type=TEXType.STRING)
        assert result == "hi    ", f"Expected 'hi    ', got {result!r}"
        r.ok("pad_right: space-pad default")
    except Exception as e:
        r.fail("pad_right: space-pad default", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = pad_left("long_string", 5, "x");',
            {}, out_type=TEXType.STRING)
        assert result == "long_string", f"Expected no-op for string longer than width, got {result!r}"
        r.ok("pad_left: no-op when already wider")
    except Exception as e:
        r.fail("pad_left: no-op when already wider", f"{e}\n{traceback.format_exc()}")

    # -- format --
    try:
        result = compile_and_run(
            '@OUT = format("frame_{}_v{}", 42, 3);',
            {}, out_type=TEXType.STRING)
        assert result == "frame_42_v3", f"Expected 'frame_42_v3', got {result!r}"
        r.ok("format: integer placeholders")
    except Exception as e:
        r.fail("format: integer placeholders", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = format("value={}", 3.14);',
            {}, out_type=TEXType.STRING)
        assert result == "value=3.14", f"Expected 'value=3.14', got {result!r}"
        r.ok("format: float placeholder")
    except Exception as e:
        r.fail("format: float placeholder", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = format("{}-{}", "hello", "world");',
            {}, out_type=TEXType.STRING)
        assert result == "hello-world", f"Expected 'hello-world', got {result!r}"
        r.ok("format: string placeholders")
    except Exception as e:
        r.fail("format: string placeholders", f"{e}\n{traceback.format_exc()}")

    # -- repeat --
    try:
        result = compile_and_run(
            '@OUT = repeat("ab", 3);',
            {}, out_type=TEXType.STRING)
        assert result == "ababab", f"Expected 'ababab', got {result!r}"
        r.ok("repeat: basic repeat")
    except Exception as e:
        r.fail("repeat: basic repeat", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = repeat("x", 0);',
            {}, out_type=TEXType.STRING)
        assert result == "", f"Expected '', got {result!r}"
        r.ok("repeat: zero count")
    except Exception as e:
        r.fail("repeat: zero count", f"{e}\n{traceback.format_exc()}")

    # -- str_reverse --
    try:
        result = compile_and_run(
            '@OUT = str_reverse("hello");',
            {}, out_type=TEXType.STRING)
        assert result == "olleh", f"Expected 'olleh', got {result!r}"
        r.ok("str_reverse: basic reverse")
    except Exception as e:
        r.fail("str_reverse: basic reverse", f"{e}\n{traceback.format_exc()}")

    # -- count --
    try:
        result = compile_and_run(
            'float n = count("banana", "an"); @OUT = vec4(n);',
            {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 2.0, f"Expected 2.0, got {v}"
        r.ok("count: substring occurrences")
    except Exception as e:
        r.fail("count: substring occurrences", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            'float n = count("hello", "xyz"); @OUT = vec4(n);',
            {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 0.0, f"Expected 0.0, got {v}"
        r.ok("count: no matches")
    except Exception as e:
        r.fail("count: no matches", f"{e}\n{traceback.format_exc()}")

    # -- replace with max_count --
    try:
        result = compile_and_run(
            '@OUT = replace("aaa", "a", "b", 2);',
            {}, out_type=TEXType.STRING)
        assert result == "bba", f"Expected 'bba', got {result!r}"
        r.ok("replace: max_count limits replacements")
    except Exception as e:
        r.fail("replace: max_count limits replacements", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            '@OUT = replace("aaa", "a", "b");',
            {}, out_type=TEXType.STRING)
        assert result == "bbb", f"Expected 'bbb', got {result!r}"
        r.ok("replace: all occurrences (no max)")
    except Exception as e:
        r.fail("replace: all occurrences (no max)", f"{e}\n{traceback.format_exc()}")

    # -- matches --
    try:
        result = compile_and_run(
            'float m = matches("frame_042", "frame_\\\\d+"); @OUT = vec4(m);',
            {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 1.0, f"Expected 1.0, got {v}"
        r.ok("matches: regex match")
    except Exception as e:
        r.fail("matches: regex match", f"{e}\n{traceback.format_exc()}")

    try:
        result = compile_and_run(
            'float m = matches("frame_abc", "frame_\\\\d+"); @OUT = vec4(m);',
            {"A": img})
        v = result[0, 0, 0, 0].item()
        assert v == 0.0, f"Expected 0.0, got {v}"
        r.ok("matches: regex no match")
    except Exception as e:
        r.fail("matches: regex no match", f"{e}\n{traceback.format_exc()}")

    # -- hash --
    try:
        result = compile_and_run(
            '@OUT = hash("hello");',
            {}, out_type=TEXType.STRING)
        assert len(result) == 16, f"Expected 16-char hash, got len={len(result)}"
        assert all(c in "0123456789abcdef" for c in result), f"Expected hex string, got {result!r}"
        r.ok("hash: returns hex string")
    except Exception as e:
        r.fail("hash: returns hex string", f"{e}\n{traceback.format_exc()}")

    try:
        r1 = compile_and_run('@OUT = hash("hello");', {}, out_type=TEXType.STRING)
        r2 = compile_and_run('@OUT = hash("hello");', {}, out_type=TEXType.STRING)
        r3 = compile_and_run('@OUT = hash("world");', {}, out_type=TEXType.STRING)
        assert r1 == r2, f"Same input should produce same hash"
        assert r1 != r3, f"Different input should produce different hash"
        r.ok("hash: deterministic and distinct")
    except Exception as e:
        r.fail("hash: deterministic and distinct", f"{e}\n{traceback.format_exc()}")


def test_string_edge_cases(r: SubTestResult):
    """Tests for string edge cases."""
    print("\n--- String Edge Cases Tests ---")

    B, H, W = 1, 4, 4
    img = torch.rand(B, H, W, 4)

    # Empty string
    try:
        result = compile_and_run(
            'string s = ""; @OUT = s;', {}, out_type=TEXType.STRING)
        assert result == "", f"Expected empty string, got {result!r}"
        r.ok("string edge: empty string")
    except Exception as e:
        r.fail("string edge: empty string", f"{e}")

    # Concatenation with empty
    try:
        result = compile_and_run(
            'string s = "hello" + ""; @OUT = s;', {}, out_type=TEXType.STRING)
        assert result == "hello", f"Expected 'hello', got {result!r}"
        r.ok("string edge: concat with empty")
    except Exception as e:
        r.fail("string edge: concat with empty", f"{e}")

    # Empty + non-empty
    try:
        result = compile_and_run(
            'string s = "" + "world"; @OUT = s;', {}, out_type=TEXType.STRING)
        assert result == "world", f"Expected 'world', got {result!r}"
        r.ok("string edge: empty + non-empty")
    except Exception as e:
        r.fail("string edge: empty + non-empty", f"{e}")

    # Very long string
    try:
        long_str = "a" * 200
        code = f'string s = "{long_str}"; float n = len(s); @OUT = vec4(n,n,n,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 200.0) < 1e-4
        r.ok("string edge: long string (200 chars)")
    except Exception as e:
        r.fail("string edge: long string (200 chars)", f"{e}")

    # Special chars in strings (tab, backslash)
    try:
        result = compile_and_run(
            r'string s = "tab\there"; @OUT = s;', {}, out_type=TEXType.STRING)
        assert "\t" in result, f"Expected tab char, got {result!r}"
        r.ok("string edge: tab character")
    except Exception as e:
        r.fail("string edge: tab character", f"{e}")

    # find() with not-found case
    try:
        code = 'float x = find("hello", "xyz"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - (-1.0)) < 1e-4
        r.ok("string edge: find not found returns -1")
    except Exception as e:
        r.fail("string edge: find not found returns -1", f"{e}")

    # find() with found case
    try:
        code = 'float x = find("hello world", "world"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 6.0) < 1e-4
        r.ok("string edge: find found returns index")
    except Exception as e:
        r.fail("string edge: find found returns index", f"{e}")

    # find() at start
    try:
        code = 'float x = find("hello", "hel"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item()) < 1e-4
        r.ok("string edge: find at start returns 0")
    except Exception as e:
        r.fail("string edge: find at start returns 0", f"{e}")

    # replace() with empty replacement
    try:
        result = compile_and_run(
            'string s = replace("hello world", "world", ""); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello ", f"Expected 'hello ', got {result!r}"
        r.ok("string edge: replace with empty")
    except Exception as e:
        r.fail("string edge: replace with empty", f"{e}")

    # replace() with not-found target
    try:
        result = compile_and_run(
            'string s = replace("hello", "xyz", "abc"); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello", f"Expected 'hello', got {result!r}"
        r.ok("string edge: replace not found")
    except Exception as e:
        r.fail("string edge: replace not found", f"{e}")

    # substr() from start
    try:
        result = compile_and_run(
            'string s = substr("hello world", 0, 5); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "hello", f"Expected 'hello', got {result!r}"
        r.ok("string edge: substr from 0")
    except Exception as e:
        r.fail("string edge: substr from 0", f"{e}")

    # substr() past-end length
    try:
        result = compile_and_run(
            'string s = substr("hello", 3, 100); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "lo", f"Expected 'lo', got {result!r}"
        r.ok("string edge: substr past end")
    except Exception as e:
        r.fail("string edge: substr past end", f"{e}")

    # substr() without length (to end)
    try:
        result = compile_and_run(
            'string s = substr("hello world", 6); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "world", f"Expected 'world', got {result!r}"
        r.ok("string edge: substr no length")
    except Exception as e:
        r.fail("string edge: substr no length", f"{e}")

    # len() on empty string
    try:
        code = 'float n = len(""); @OUT = vec4(n,n,n,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item()) < 1e-4
        r.ok("string edge: len empty string")
    except Exception as e:
        r.fail("string edge: len empty string", f"{e}")

    # len() on long string
    try:
        long_str = "x" * 150
        code = f'float n = len("{long_str}"); @OUT = vec4(n,n,n,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 150.0) < 1e-4
        r.ok("string edge: len long string")
    except Exception as e:
        r.fail("string edge: len long string", f"{e}")

    # Multiple concatenations
    try:
        result = compile_and_run(
            'string s = "a" + "b" + "c" + "d" + "e"; @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "abcde", f"Expected 'abcde', got {result!r}"
        r.ok("string edge: multiple concat")
    except Exception as e:
        r.fail("string edge: multiple concat", f"{e}")

    # String equality comparison
    try:
        code = 'float x = ("abc" == "abc"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 1.0) < 1e-4
        r.ok("string edge: equality comparison")
    except Exception as e:
        r.fail("string edge: equality comparison", f"{e}")

    # String in conditional
    try:
        result = compile_and_run(
            'string s = "a"; if (len(s) > 0.0) { s = s + "b"; } @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "ab", f"Expected 'ab', got {result!r}"
        r.ok("string edge: string in conditional")
    except Exception as e:
        r.fail("string edge: string in conditional", f"{e}")

    # String in loop
    try:
        result = compile_and_run(
            'string s = ""; for (int i = 0; i < 5; i++) { s = s + "x"; } @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert result == "xxxxx", f"Expected 'xxxxx', got {result!r}"
        r.ok("string edge: string in loop")
    except Exception as e:
        r.fail("string edge: string in loop", f"{e}")

    # Numeric to string conversion
    try:
        result = compile_and_run(
            'string s = "value=" + str(42); @OUT = s;',
            {}, out_type=TEXType.STRING)
        assert "42" in result, f"Expected '42' in result, got {result!r}"
        r.ok("string edge: numeric to string concat")
    except Exception as e:
        r.fail("string edge: numeric to string concat", f"{e}")

    # split with single char
    try:
        code = '''
string parts[] = split("a,b,c", ",");
float n = len(parts);
@OUT = vec4(n, 0, 0, 1);
'''
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 3.0) < 1e-4
        r.ok("string edge: split count")
    except Exception as e:
        r.fail("string edge: split count", f"{e}")

    # find in empty string
    try:
        code = 'float x = find("", "abc"); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - (-1.0)) < 1e-4
        r.ok("string edge: find in empty string")
    except Exception as e:
        r.fail("string edge: find in empty string", f"{e}")

    # find empty in string
    try:
        code = 'float x = find("hello", ""); @OUT = vec4(x,x,x,1);'
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item()) < 1e-4  # Python str.find("") returns 0
        r.ok("string edge: find empty in string")
    except Exception as e:
        r.fail("string edge: find empty in string", f"{e}")


def test_arrays(r: SubTestResult):
    """Tests for fixed-size array type."""
    print("\n--- Array Tests ---")
    B, H, W = 1, 4, 4
    img = torch.rand(B, H, W, 4)

    # 1. Lexer: bracket tokens
    try:
        tokens = Lexer("arr[0]").tokenize()
        types = [t.type for t in tokens[:-1]]
        assert TokenType.LBRACKET in types
        assert TokenType.RBRACKET in types
        r.ok("array: bracket tokens")
    except Exception as e:
        r.fail("array: bracket tokens", f"{e}\n{traceback.format_exc()}")

    # 2. Parser: array declaration with size
    try:
        tokens = Lexer("float arr[3];").tokenize()
        from TEX_Wrangle.tex_compiler.ast_nodes import ArrayDecl
        program = Parser(tokens).parse()
        assert len(program.statements) == 1
        decl = program.statements[0]
        assert isinstance(decl, ArrayDecl)
        assert decl.element_type_name == "float"
        assert decl.name == "arr"
        assert decl.size == 3
        r.ok("array: parse declaration with size")
    except Exception as e:
        r.fail("array: parse declaration with size", f"{e}\n{traceback.format_exc()}")

    # 3. Parser: array with initializer
    try:
        tokens = Lexer("float arr[] = {1.0, 2.0, 3.0};").tokenize()
        from TEX_Wrangle.tex_compiler.ast_nodes import ArrayDecl, ArrayLiteral
        program = Parser(tokens).parse()
        decl = program.statements[0]
        assert isinstance(decl, ArrayDecl)
        assert decl.size is None  # inferred
        assert isinstance(decl.initializer, ArrayLiteral)
        assert len(decl.initializer.elements) == 3
        r.ok("array: parse initializer list")
    except Exception as e:
        r.fail("array: parse initializer list", f"{e}\n{traceback.format_exc()}")

    # 4. Parser: error on empty array without size or initializer
    try:
        tokens = Lexer("float arr[];").tokenize()
        Parser(tokens).parse()
        r.fail("array: missing size error", "Expected ParseError")
    except ParseError:
        r.ok("array: missing size error")
    except Exception as e:
        r.fail("array: missing size error", f"{e}\n{traceback.format_exc()}")

    # 5. Type checker: vec3/vec4/string element types now allowed
    try:
        tokens = Lexer("vec3 arr[5]; @OUT = vec4(0.0);").tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"OUT": TEXType.VEC4})
        tc.check(program)
        r.ok("array: vec3 element type allowed")
    except Exception as e:
        r.fail("array: vec3 element type allowed", f"{e}\n{traceback.format_exc()}")

    # 6. Declaration and element access
    try:
        code = """
float arr[3];
arr[0] = 1.0;
arr[1] = 2.0;
arr[2] = 3.0;
@OUT = vec4(arr[0], arr[1], arr[2], 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 2.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 3.0) < 1e-4
        r.ok("array: declaration and access")
    except Exception as e:
        r.fail("array: declaration and access", f"{e}\n{traceback.format_exc()}")

    # 7. Initializer list
    try:
        code = """
float arr[] = {0.25, 0.5, 0.75};
@OUT = vec4(arr[0], arr[1], arr[2], 1.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 0.25) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 0.50) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 0.75) < 1e-4
        r.ok("array: initializer list")
    except Exception as e:
        r.fail("array: initializer list", f"{e}\n{traceback.format_exc()}")

    # 8. Loop population
    try:
        code = """
float arr[5];
for (int i = 0; i < 5; i++) {
    arr[i] = float(i) * 0.25;
}
@OUT = vec4(arr[2], arr[4], 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 0.5) < 1e-4  # arr[2] = 2*0.25
        assert abs(result[0, 0, 0, 1].item() - 1.0) < 1e-4  # arr[4] = 4*0.25
        r.ok("array: loop population")
    except Exception as e:
        r.fail("array: loop population", f"{e}\n{traceback.format_exc()}")

    # 9. Clamped bounds
    try:
        code = """
float arr[] = {1.0, 2.0, 3.0};
float x = arr[10];
float y = arr[-1];
@OUT = vec4(x, y, 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4  # clamped to arr[2]
        assert abs(result[0, 0, 0, 1].item() - 1.0) < 1e-4  # clamped to arr[0]
        r.ok("array: clamped bounds")
    except Exception as e:
        r.fail("array: clamped bounds", f"{e}\n{traceback.format_exc()}")

    # 10. Array copy
    try:
        code = """
float a[] = {1.0, 2.0, 3.0};
float b[3] = a;
b[0] = 99.0;
@OUT = vec4(a[0], b[0], 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4   # a[0] unchanged
        assert abs(result[0, 0, 0, 1].item() - 99.0) < 1e-4  # b[0] modified
        r.ok("array: copy independence")
    except Exception as e:
        r.fail("array: copy independence", f"{e}\n{traceback.format_exc()}")

    # 11. sort()
    try:
        code = """
float arr[] = {3.0, 1.0, 2.0};
arr = sort(arr);
@OUT = vec4(arr[0], arr[1], arr[2], 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 2.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 3.0) < 1e-4
        r.ok("array: sort")
    except Exception as e:
        r.fail("array: sort", f"{e}\n{traceback.format_exc()}")

    # 12. reverse()
    try:
        code = """
float arr[] = {1.0, 2.0, 3.0};
arr = reverse(arr);
@OUT = vec4(arr[0], arr[1], arr[2], 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 2.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 1.0) < 1e-4
        r.ok("array: reverse")
    except Exception as e:
        r.fail("array: reverse", f"{e}\n{traceback.format_exc()}")

    # 13. arr_sum, arr_min, arr_max, median, arr_avg
    try:
        code = """
float arr[] = {1.0, 2.0, 3.0, 4.0, 5.0};
float s = arr_sum(arr);
float mn = arr_min(arr);
float mx = arr_max(arr);
float md = median(arr);
float av = arr_avg(arr);
@OUT = vec4(s, mn, mx, md);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 15.0) < 1e-4  # sum
        assert abs(result[0, 0, 0, 1].item() - 1.0) < 1e-4   # min
        assert abs(result[0, 0, 0, 2].item() - 5.0) < 1e-4   # max
        assert abs(result[0, 0, 0, 3].item() - 3.0) < 1e-4   # median
        r.ok("array: aggregate functions")
    except Exception as e:
        r.fail("array: aggregate functions", f"{e}\n{traceback.format_exc()}")

    # 14. arr_avg separately
    try:
        code = """
float arr[] = {2.0, 4.0, 6.0};
float av = arr_avg(arr);
@OUT = vec4(av, 0.0, 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 4.0) < 1e-4
        r.ok("array: arr_avg")
    except Exception as e:
        r.fail("array: arr_avg", f"{e}\n{traceback.format_exc()}")

    # 15. len() on arrays
    try:
        code = """
float arr[] = {1.0, 2.0, 3.0};
float n = len(arr);
@OUT = vec4(n, 0.0, 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4
        r.ok("array: len")
    except Exception as e:
        r.fail("array: len", f"{e}\n{traceback.format_exc()}")

    # 16. int arrays
    try:
        code = """
int arr[3];
arr[0] = 10;
arr[1] = 20;
arr[2] = 30;
float total = float(arr[0]) + float(arr[1]) + float(arr[2]);
@OUT = vec4(total, 0.0, 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 60.0) < 1e-4
        r.ok("array: int elements")
    except Exception as e:
        r.fail("array: int elements", f"{e}\n{traceback.format_exc()}")

    # 17. Arrays in if/else (per-pixel merge via torch.where)
    try:
        code = """
float arr[] = {1.0, 2.0, 3.0};
if (u > 0.5) {
    arr[0] = 99.0;
} else {
    arr[0] = 0.0;
}
@OUT = vec4(arr[0], arr[1], arr[2], 0.0);
"""
        result = compile_and_run(code, {"A": img})
        # Left half (u<=0.5): arr[0]=0.0; Right half (u>0.5): arr[0]=99.0
        assert abs(result[0, 0, 0, 0].item() - 0.0) < 1e-4     # u=0
        assert abs(result[0, 0, W-1, 0].item() - 99.0) < 1e-4  # u=1
        # arr[1] and arr[2] unchanged everywhere
        assert abs(result[0, 0, 0, 1].item() - 2.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 3.0) < 1e-4
        r.ok("array: if/else merge")
    except Exception as e:
        r.fail("array: if/else merge", f"{e}\n{traceback.format_exc()}")

    # 18. Per-pixel fetch into array (spatial use case)
    try:
        code = """
float samples[3];
for (int i = 0; i < 3; i++) {
    samples[i] = fetch(@A, ix + i - 1, iy).r;
}
samples = sort(samples);
@OUT = vec4(samples[1], samples[1], samples[1], 1.0);
"""
        result = compile_and_run(code, {"A": img})
        assert result.shape == (B, H, W, 4)
        r.ok("array: spatial fetch+sort")
    except Exception as e:
        r.fail("array: spatial fetch+sort", f"{e}\n{traceback.format_exc()}")

    # 19. Size mismatch error
    try:
        tokens = Lexer("float arr[2] = {1.0, 2.0, 3.0}; @OUT = vec4(0.0);").tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"OUT": TEXType.VEC4})
        tc.check(program)
        r.fail("array: size mismatch error", "Expected TypeCheckError")
    except TypeCheckError as e:
        assert "mismatch" in str(e).lower()
        r.ok("array: size mismatch error")
    except Exception as e:
        r.fail("array: size mismatch error", f"{e}\n{traceback.format_exc()}")

    # 20. Indexing non-array error
    try:
        tokens = Lexer("float x = 5.0; float y = x[0]; @OUT = vec4(y);").tokenize()
        program = Parser(tokens).parse()
        tc = TypeChecker(binding_types={"OUT": TEXType.VEC4})
        tc.check(program)
        r.fail("array: index non-array error", "Expected TypeCheckError")
    except TypeCheckError as e:
        assert "doesn't work on" in str(e).lower() or "non-array" in str(e).lower() or "only arrays" in str(e).lower()
        r.ok("array: index non-array error")
    except Exception as e:
        r.fail("array: index non-array error", f"{e}\n{traceback.format_exc()}")


def test_vec_arrays(r: SubTestResult):
    """Tests for vec3/vec4 element arrays."""
    print("\n--- Vec Array Tests ---")
    B, H, W = 1, 4, 4
    img = torch.rand(B, H, W, 4)

    # 1. vec3 zero-init, correct shape
    try:
        code = """
vec3 arr[3];
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        # arr[0] should be vec3(0,0,0)
        assert result.shape == (B, H, W, 3), f"Expected shape {(B,H,W,3)}, got {result.shape}"
        assert result.abs().max().item() < 1e-6, "Zero-init vec3 should be all zeros"
        r.ok("vec3 array: zero-init shape")
    except Exception as e:
        r.fail("vec3 array: zero-init shape", f"{e}\n{traceback.format_exc()}")

    # 2. vec4 initializer list
    try:
        code = """
vec4 arr[] = {vec4(1.0, 0.0, 0.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0)};
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4  # R=1
        assert abs(result[0, 0, 0, 1].item() - 0.0) < 1e-4  # G=0
        assert abs(result[0, 0, 0, 3].item() - 1.0) < 1e-4  # A=1
        r.ok("vec4 array: initializer list")
    except Exception as e:
        r.fail("vec4 array: initializer list", f"{e}\n{traceback.format_exc()}")

    # 3. Element access returns correct values
    try:
        code = """
vec4 arr[] = {vec4(1.0, 2.0, 3.0, 4.0), vec4(5.0, 6.0, 7.0, 8.0)};
@OUT = arr[1];
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 5.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 6.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 7.0) < 1e-4
        assert abs(result[0, 0, 0, 3].item() - 8.0) < 1e-4
        r.ok("vec4 array: element access")
    except Exception as e:
        r.fail("vec4 array: element access", f"{e}\n{traceback.format_exc()}")

    # 4. Element assignment
    try:
        code = """
vec4 arr[2];
arr[0] = vec4(1.0, 0.0, 0.0, 1.0);
arr[1] = vec4(0.0, 1.0, 0.0, 1.0);
@OUT = arr[0] + arr[1];
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4  # 1+0
        assert abs(result[0, 0, 0, 1].item() - 1.0) < 1e-4  # 0+1
        assert abs(result[0, 0, 0, 2].item() - 0.0) < 1e-4  # 0+0
        assert abs(result[0, 0, 0, 3].item() - 2.0) < 1e-4  # 1+1
        r.ok("vec4 array: element assignment")
    except Exception as e:
        r.fail("vec4 array: element assignment", f"{e}\n{traceback.format_exc()}")

    # 5. sort() on vec4 array — per-channel sort
    try:
        code = """
vec4 arr[] = {vec4(3.0, 1.0, 2.0, 1.0), vec4(1.0, 3.0, 1.0, 3.0), vec4(2.0, 2.0, 3.0, 2.0)};
arr = sort(arr);
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img})
        # Per-channel sort: R sorted = [1,2,3], G sorted = [1,2,3], etc.
        assert abs(result[0, 0, 0, 0].item() - 1.0) < 1e-4  # min R
        assert abs(result[0, 0, 0, 1].item() - 1.0) < 1e-4  # min G
        r.ok("vec4 array: sort per-channel")
    except Exception as e:
        r.fail("vec4 array: sort per-channel", f"{e}\n{traceback.format_exc()}")

    # 6. median() on vec4 array — per-channel median
    try:
        code = """
vec4 arr[] = {vec4(1.0, 10.0, 5.0, 1.0), vec4(5.0, 20.0, 3.0, 2.0), vec4(3.0, 15.0, 1.0, 3.0)};
@OUT = median(arr);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4  # median of [1,5,3]
        assert abs(result[0, 0, 0, 1].item() - 15.0) < 1e-4  # median of [10,20,15]
        assert abs(result[0, 0, 0, 2].item() - 3.0) < 1e-4  # median of [5,3,1]
        r.ok("vec4 array: median per-channel")
    except Exception as e:
        r.fail("vec4 array: median per-channel", f"{e}\n{traceback.format_exc()}")

    # 7. arr_sum on vec3 array
    try:
        code = """
vec3 arr[] = {vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0)};
vec3 s = arr_sum(arr);
@OUT = s;
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        assert abs(result[0, 0, 0, 0].item() - 5.0) < 1e-4   # 1+4
        assert abs(result[0, 0, 0, 1].item() - 7.0) < 1e-4   # 2+5
        assert abs(result[0, 0, 0, 2].item() - 9.0) < 1e-4   # 3+6
        r.ok("vec3 array: arr_sum")
    except Exception as e:
        r.fail("vec3 array: arr_sum", f"{e}\n{traceback.format_exc()}")

    # 8. arr_avg on vec4 array
    try:
        code = """
vec4 arr[] = {vec4(2.0, 4.0, 6.0, 8.0), vec4(4.0, 6.0, 8.0, 10.0)};
@OUT = arr_avg(arr);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 5.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 7.0) < 1e-4
        assert abs(result[0, 0, 0, 3].item() - 9.0) < 1e-4
        r.ok("vec4 array: arr_avg")
    except Exception as e:
        r.fail("vec4 array: arr_avg", f"{e}\n{traceback.format_exc()}")

    # 9. arr_min / arr_max on vec4 array
    try:
        code = """
vec4 arr[] = {vec4(1.0, 5.0, 3.0, 2.0), vec4(4.0, 2.0, 6.0, 1.0), vec4(2.0, 3.0, 1.0, 4.0)};
vec4 lo = arr_min(arr);
vec4 hi = arr_max(arr);
@OUT = hi - lo;
"""
        result = compile_and_run(code, {"A": img})
        # R: max=4, min=1, diff=3; G: max=5, min=2, diff=3; B: max=6, min=1, diff=5; A: max=4, min=1, diff=3
        assert abs(result[0, 0, 0, 0].item() - 3.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 3.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 5.0) < 1e-4
        assert abs(result[0, 0, 0, 3].item() - 3.0) < 1e-4
        r.ok("vec4 array: arr_min/arr_max")
    except Exception as e:
        r.fail("vec4 array: arr_min/arr_max", f"{e}\n{traceback.format_exc()}")

    # 10. reverse on vec3 array
    try:
        code = """
vec3 arr[] = {vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)};
arr = reverse(arr);
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.VEC3)
        # reversed: [blue, green, red], arr[0] = blue = (0,0,1)
        assert abs(result[0, 0, 0, 0].item() - 0.0) < 1e-4
        assert abs(result[0, 0, 0, 1].item() - 0.0) < 1e-4
        assert abs(result[0, 0, 0, 2].item() - 1.0) < 1e-4
        r.ok("vec3 array: reverse")
    except Exception as e:
        r.fail("vec3 array: reverse", f"{e}\n{traceback.format_exc()}")

    # 11. len() on vec4 array returns element count (not channels)
    try:
        code = """
vec4 arr[] = {vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0), vec4(0.0)};
float n = len(arr);
@OUT = vec4(n, 0.0, 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 5.0) < 1e-4
        r.ok("vec4 array: len returns element count")
    except Exception as e:
        r.fail("vec4 array: len returns element count", f"{e}\n{traceback.format_exc()}")

    # 12. Loop populate vec4 array (spatial fetch pattern)
    try:
        code = """
vec4 samples[3];
for (int i = 0; i < 3; i++) {
    samples[i] = fetch(@A, ix + i - 1, iy);
}
@OUT = samples[1];
"""
        result = compile_and_run(code, {"A": img})
        # samples[1] = fetch(@A, ix, iy) = current pixel
        diff = (result - img).abs().max().item()
        assert diff < 1e-5, f"samples[1] should equal current pixel, diff={diff}"
        r.ok("vec4 array: loop populate with fetch")
    except Exception as e:
        r.fail("vec4 array: loop populate with fetch", f"{e}\n{traceback.format_exc()}")


def test_string_arrays(r: SubTestResult):
    """Tests for string element arrays."""
    print("\n--- String Array Tests ---")
    B, H, W = 1, 2, 2
    img = torch.rand(B, H, W, 4)

    # 1. String array zero-init
    try:
        code = """
string arr[3];
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "", f"Expected empty string, got '{result}'"
        r.ok("string array: zero-init empty strings")
    except Exception as e:
        r.fail("string array: zero-init empty strings", f"{e}\n{traceback.format_exc()}")

    # 2. Initializer list
    try:
        code = """
string arr[] = {"alpha", "beta", "gamma"};
@OUT = arr[1];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "beta", f"Expected 'beta', got '{result}'"
        r.ok("string array: initializer list")
    except Exception as e:
        r.fail("string array: initializer list", f"{e}\n{traceback.format_exc()}")

    # 3. Element assignment
    try:
        code = """
string arr[3];
arr[0] = "hello";
arr[1] = "world";
arr[2] = "!";
@OUT = arr[0] + " " + arr[1] + arr[2];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "hello world!", f"Expected 'hello world!', got '{result}'"
        r.ok("string array: element assignment")
    except Exception as e:
        r.fail("string array: element assignment", f"{e}\n{traceback.format_exc()}")

    # 4. sort() — alphabetical
    try:
        code = """
string arr[] = {"cherry", "apple", "banana"};
arr = sort(arr);
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "apple", f"Expected 'apple', got '{result}'"
        r.ok("string array: sort alphabetical")
    except Exception as e:
        r.fail("string array: sort alphabetical", f"{e}\n{traceback.format_exc()}")

    # 5. reverse()
    try:
        code = """
string arr[] = {"a", "b", "c"};
arr = reverse(arr);
@OUT = arr[0];
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "c", f"Expected 'c', got '{result}'"
        r.ok("string array: reverse")
    except Exception as e:
        r.fail("string array: reverse", f"{e}\n{traceback.format_exc()}")

    # 6. len() on string array
    try:
        code = """
string arr[] = {"x", "y", "z", "w"};
float n = len(arr);
@OUT = vec4(n, 0.0, 0.0, 0.0);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0, 0, 0, 0].item() - 4.0) < 1e-4
        r.ok("string array: len")
    except Exception as e:
        r.fail("string array: len", f"{e}\n{traceback.format_exc()}")

    # 7. join() — concatenate with separator
    try:
        code = """
string arr[] = {"apple", "banana", "cherry"};
@OUT = join(arr, ", ");
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "apple, banana, cherry", f"Expected 'apple, banana, cherry', got '{result}'"
        r.ok("string array: join")
    except Exception as e:
        r.fail("string array: join", f"{e}\n{traceback.format_exc()}")

    # 8. join() with empty separator
    try:
        code = """
string parts[] = {"pre", "fix", "_", "name"};
@OUT = join(parts, "");
"""
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "prefix_name", f"Expected 'prefix_name', got '{result}'"
        r.ok("string array: join empty separator")
    except Exception as e:
        r.fail("string array: join empty separator", f"{e}\n{traceback.format_exc()}")


def test_array_bounds(r: SubTestResult):
    """Tests for array bounds handling, edge cases."""
    print("\n--- Array Bounds Tests ---")

    B, H, W = 1, 4, 4
    img = torch.rand(B, H, W, 4)

    # Access at index 0 (first element)
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[0], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 10.0) < 1e-4
        r.ok("array bounds: index 0")
    except Exception as e:
        r.fail("array bounds: index 0", f"{e}")

    # Access at last valid index
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[2], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 30.0) < 1e-4
        r.ok("array bounds: last index")
    except Exception as e:
        r.fail("array bounds: last index", f"{e}")

    # Access beyond bounds (should clamp to last element)
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[5], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 30.0) < 1e-4
        r.ok("array bounds: beyond upper clamps to last")
    except Exception as e:
        r.fail("array bounds: beyond upper clamps to last", f"{e}")

    # Access at very large index
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[1000], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 30.0) < 1e-4
        r.ok("array bounds: very large index clamps")
    except Exception as e:
        r.fail("array bounds: very large index clamps", f"{e}")

    # Negative index (should clamp to 0)
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[-1], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 10.0) < 1e-4
        r.ok("array bounds: negative clamps to 0")
    except Exception as e:
        r.fail("array bounds: negative clamps to 0", f"{e}")

    # Very negative index
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[-100], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 10.0) < 1e-4
        r.ok("array bounds: very negative clamps to 0")
    except Exception as e:
        r.fail("array bounds: very negative clamps to 0", f"{e}")

    # Array length with len()
    try:
        code = "float a[] = {1.0, 2.0, 3.0, 4.0, 5.0}; float n = len(a); @OUT = vec4(n, 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 5.0) < 1e-4
        r.ok("array bounds: len()")
    except Exception as e:
        r.fail("array bounds: len()", f"{e}")

    # Single element array
    try:
        code = "float a[] = {42.0}; @OUT = vec4(a[0], len(a), 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 42.0) < 1e-4
        assert abs(result[0,0,0,1].item() - 1.0) < 1e-4
        r.ok("array bounds: single element")
    except Exception as e:
        r.fail("array bounds: single element", f"{e}")

    # Write beyond bounds (should clamp)
    try:
        code = """
float a[] = {1.0, 2.0, 3.0};
a[10] = 99.0;
@OUT = vec4(a[0], a[1], a[2], 1);
"""
        result = compile_and_run(code, {"A": img})
        # Writing at clamped index 2 should change a[2]
        assert abs(result[0,0,0,2].item() - 99.0) < 1e-4
        r.ok("array bounds: write beyond clamps")
    except Exception as e:
        r.fail("array bounds: write beyond clamps", f"{e}")

    # Write at negative index (should clamp to 0)
    try:
        code = """
float a[] = {1.0, 2.0, 3.0};
a[-5] = 99.0;
@OUT = vec4(a[0], a[1], a[2], 1);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 99.0) < 1e-4
        r.ok("array bounds: write negative clamps to 0")
    except Exception as e:
        r.fail("array bounds: write negative clamps to 0", f"{e}")

    # Nested array access (array of arrays via vec)
    try:
        code = """
vec3 a[] = {vec3(1,2,3), vec3(4,5,6)};
float x = a[0].y;
float y = a[1].z;
@OUT = vec4(x, y, 0, 1);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 2.0) < 1e-4
        assert abs(result[0,0,0,1].item() - 6.0) < 1e-4
        r.ok("array bounds: vec array element access")
    except Exception as e:
        r.fail("array bounds: vec array element access", f"{e}")

    # Vec array beyond bounds
    try:
        code = """
vec3 a[] = {vec3(1,2,3), vec3(4,5,6)};
vec3 val = a[10];
@OUT = vec4(val.x, val.y, val.z, 1);
"""
        result = compile_and_run(code, {"A": img})
        # Should clamp to a[1]
        assert abs(result[0,0,0,0].item() - 4.0) < 1e-4
        r.ok("array bounds: vec array beyond bounds clamps")
    except Exception as e:
        r.fail("array bounds: vec array beyond bounds clamps", f"{e}")

    # Dynamic index with loop
    try:
        code = """
float a[] = {10.0, 20.0, 30.0, 40.0, 50.0};
float sum = 0.0;
for (int i = 0; i < 5; i++) {
    sum += a[i];
}
@OUT = vec4(sum, 0, 0, 1);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 150.0) < 1e-4
        r.ok("array bounds: dynamic index in loop")
    except Exception as e:
        r.fail("array bounds: dynamic index in loop", f"{e}")

    # Float index (should be floored)
    try:
        code = "float a[] = {10.0, 20.0, 30.0}; @OUT = vec4(a[1.7], 0, 0, 1);"
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item() - 20.0) < 1e-4  # floor(1.7) = 1
        r.ok("array bounds: float index floors")
    except Exception as e:
        r.fail("array bounds: float index floors", f"{e}")

    # Large array
    try:
        code = """
float a[100];
for (int i = 0; i < 100; i++) {
    a[i] = float(i);
}
@OUT = vec4(a[0], a[50], a[99], 1);
"""
        result = compile_and_run(code, {"A": img})
        assert abs(result[0,0,0,0].item()) < 1e-4
        assert abs(result[0,0,0,1].item() - 50.0) < 1e-4
        assert abs(result[0,0,0,2].item() - 99.0) < 1e-4
        r.ok("array bounds: large array (100)")
    except Exception as e:
        r.fail("array bounds: large array (100)", f"{e}")

    # Empty-sized array (declared with size 0 should error)
    try:
        code = "float a[0]; @OUT = vec4(0,0,0,1);"
        compile_and_run(code, {"A": img})
        r.fail("array bounds: size 0 should error", "Expected error")
    except (ParseError, TypeCheckError):
        r.ok("array bounds: size 0 errors")

    # String array bounds
    try:
        code = '''
string a[] = {"alpha", "beta", "gamma"};
@OUT = a[0];
'''
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "alpha"
        r.ok("array bounds: string array index 0")
    except Exception as e:
        r.fail("array bounds: string array index 0", f"{e}")

    # String array beyond bounds
    try:
        code = '''
string a[] = {"alpha", "beta", "gamma"};
@OUT = a[10];
'''
        result = compile_and_run(code, {"A": img}, out_type=TEXType.STRING)
        assert result == "gamma"  # clamped to last
        r.ok("array bounds: string array clamp to last")
    except Exception as e:
        r.fail("array bounds: string array clamp to last", f"{e}")
