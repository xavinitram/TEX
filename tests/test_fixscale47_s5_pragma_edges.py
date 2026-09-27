"""FIX-SCALE S5 (v0.47 Phase C, B2 finding 5) — `scale_pragma`'s header scan skips over a
LEADING `/* ... */` block comment instead of ending the header run there, and a conflicting
pair of `//!tex scale: safe` / `//!tex scale: never` pragmas resolves to `never` (documented),
never by silent first-match line order.

Before this fix: `scale_pragma` copied `language_pragma`'s exact scan (stop at the first line
that is neither blank nor a `//` comment), so a `/* */` file-header block comment -- common
style -- silently defeated a `//!tex scale: never` placed after it: the author's explicit,
stated override was never seen, and the program fell through to the classifier's own
(possibly permissive) verdict, the OPPOSITE of the author's intent. Separately, two
conflicting pragmas resolved by "whichever line comes first" with no diagnostic."""
from helpers import *
from TEX_Wrangle.tex_compiler.parser import scale_pragma, language_pragma
from TEX_Wrangle import tex_roi as R


def test_s5_never_after_a_block_comment_is_recognized(r: SubTestResult):
    print("\n--- FIX-SCALE S5: //!tex scale: never after a leading /* */ block is recognized ---")
    code = ("/* header block comment\n"
            "   nothing relevant */\n"
            "//!tex scale: never\n"
            "@OUT = gauss_blur(@A, 4.0);")
    got = scale_pragma(code)
    if got != "never":
        r.fail("never after block comment", f"expected 'never', got {got!r} -- the author's "
               f"override was defeated by the preceding block comment")
        return
    r.ok("scale_pragma() sees //!tex scale: never past a leading block comment")


def test_s5_safe_after_a_block_comment_is_recognized(r: SubTestResult):
    print("\n--- FIX-SCALE S5: //!tex scale: safe after a leading /* */ block is recognized ---")
    code = "/* header */\n//!tex scale: safe\n@OUT = vec4(ix * 0.01, 0.0, 0.0, 1.0);"
    got = scale_pragma(code)
    if got != "safe":
        r.fail("safe after block comment", f"expected 'safe', got {got!r}")
        return
    r.ok("scale_pragma() sees //!tex scale: safe past a leading block comment")


def test_s5_multiline_block_comment_is_skipped_entirely(r: SubTestResult):
    print("\n--- FIX-SCALE S5: a multi-line /* */ block is skipped as a whole, not just its first line ---")
    code = "/* line one\nline two\nline three */\n//!tex scale: safe\n@OUT = 1.0;"
    got = scale_pragma(code)
    if got != "safe":
        r.fail("multiline block comment", f"expected 'safe', got {got!r}")
        return
    r.ok("a multi-line block comment does not stop the header scan")


def test_s5_engine_end_to_end_respects_never_past_a_block_comment(r: SubTestResult):
    print("\n--- FIX-SCALE S5: R.scale_safe() end-to-end sees the never override past a block comment ---")
    code = ("/* a plain header note */\n"
            "//!tex scale: never\n"
            "@OUT = gauss_blur(@A, 4.0);")
    if R.scale_safe(code):
        r.fail("end to end never", "a program the classifier would otherwise accept was NOT "
               "declined even though it declares //!tex scale: never past a block comment")
        return
    r.ok("R.scale_safe() honours //!tex scale: never past a leading block comment")


def test_s5_never_wins_over_safe_regardless_of_order(r: SubTestResult):
    print("\n--- FIX-SCALE S5: conflicting pragmas -> never wins, in EITHER order ---")
    safe_then_never = "//!tex scale: safe\n//!tex scale: never\n@OUT = 1.0;"
    never_then_safe = "//!tex scale: never\n//!tex scale: safe\n@OUT = 1.0;"
    got1 = scale_pragma(safe_then_never)
    got2 = scale_pragma(never_then_safe)
    if got1 != "never" or got2 != "never":
        r.fail("never wins", f"expected 'never' both ways, got safe-then-never={got1!r}, "
               f"never-then-safe={got2!r}")
        return
    r.ok("//!tex scale: never wins over a conflicting //!tex scale: safe regardless of order")


def test_s5_language_pragma_still_stops_at_a_block_comment(r: SubTestResult):
    print("\n--- FIX-SCALE S5: language_pragma's OWN posture (block comment ends the header) is unchanged ---")
    code = "/* header */\n//!tex 1.2\n@OUT = 1.0;"
    got = language_pragma(code)
    if got is not None:
        r.fail("language_pragma regression", f"expected None (language_pragma still stops at "
               f"a block comment, unchanged by S5), got {got!r}")
        return
    r.ok("language_pragma() is unaffected: a block comment still ends ITS header run")


def test_s5_language_pragma_plain_leading_pragma_unaffected(r: SubTestResult):
    print("\n--- FIX-SCALE S5: language_pragma's plain leading-pragma case is unaffected ---")
    got = language_pragma("//!tex 0.25\n@OUT = 1.0;")
    if got != "0.25":
        r.fail("language_pragma plain", f"expected '0.25', got {got!r}")
        return
    r.ok("language_pragma() still reads a plain leading //!tex X.Y pragma")
