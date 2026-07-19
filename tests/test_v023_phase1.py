"""
v0.23 Phase 1 — "Authoring".

ROI-1  the boolean `non_local` stdlib tag becomes an access-*footprint* descriptor
       ('point' | ('halo', r) | ('halo_arg', i) | 'image' | ('frame', i)); the M-4
       tiling set `_NON_LOCAL_FNS` is DERIVED from it (footprint != 'point'), so the
       old hand-kept literal cannot drift from the impls. Pinned here: the derivation
       reproduces the historical 18-name set exactly, every footprint is well-formed
       and classified into the expected family, a malformed footprint fails LOUD at
       decoration, and is_tile_safe's verdicts are unchanged (invariant #7).
"""
from helpers import *
from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib  # populates REGISTRY
from TEX_Wrangle.tex_runtime import stdlib_registry as R
from TEX_Wrangle import tex_memory


# The exact set the hand-kept literal carried through v0.22. The derivation must
# reproduce this — a regression here means a real tiling behaviour change.
_HISTORICAL_NON_LOCAL = frozenset({
    "sample", "sample_cubic", "sample_frame", "sample_grad", "sample_lanczos",
    "sample_mip", "sample_mip_gauss", "fetch", "fetch_frame",
    "gauss_blur", "bilateral_filter",
    "img_min", "img_max", "img_mean", "img_median", "img_sum",
    "erode", "dilate",
})


def test_roi1_derivation_matches_historical(r: SubTestResult):
    print("\n--- ROI-1: _NON_LOCAL_FNS derives from footprints, exactly ---")

    # (1) The registry derivation equals the frozen 18-name literal.
    try:
        derived = R.non_local_names()
        assert derived == _HISTORICAL_NON_LOCAL, (
            f"derived non-local set drifted: "
            f"+{sorted(derived - _HISTORICAL_NON_LOCAL)} "
            f"-{sorted(_HISTORICAL_NON_LOCAL - derived)}")
        r.ok(f"non_local_names() == the historical {len(_HISTORICAL_NON_LOCAL)}-name set")
    except Exception as e:
        r.fail("ROI-1 derivation", f"{type(e).__name__}: {e}")

    # (2) The public module attribute (via __getattr__) is that same derived set.
    try:
        assert set(tex_memory._NON_LOCAL_FNS) == _HISTORICAL_NON_LOCAL, \
            "tex_memory._NON_LOCAL_FNS != derived set"
        assert tex_memory._NON_LOCAL_FNS is tex_memory._non_local_fns(), \
            "the public attribute and the helper must return the one cached object"
        r.ok("tex_memory._NON_LOCAL_FNS surfaces the derived set (PEP 562 __getattr__)")
    except Exception as e:
        r.fail("ROI-1 public attribute", f"{type(e).__name__}: {e}")

    # (3) The derived non_local property agrees with footprint != 'point' for EVERY
    #     entry — the invariant that keeps every `.non_local` reader (TST-3,
    #     gen_function_reference) correct without a stored boolean.
    try:
        bad = [e.name for e in R.REGISTRY if e.non_local != (e.footprint != "point")]
        assert not bad, f"non_local property disagrees with footprint for: {bad}"
        r.ok("StdlibEntry.non_local == (footprint != 'point') for all entries")
    except Exception as e:
        r.fail("ROI-1 property", f"{type(e).__name__}: {e}")


def test_roi1_footprints_wellformed_and_classified(r: SubTestResult):
    print("\n--- ROI-1: every footprint is valid and classified into its family ---")

    # (1) Every registered footprint is well-formed (the validator the decorator ran).
    try:
        malformed = [(e.name, e.footprint) for e in R.REGISTRY
                     if not R._valid_footprint(e.footprint)]
        assert not malformed, f"malformed footprints in registry: {malformed}"
        r.ok(f"all {len(R.REGISTRY)} registered footprints validate")
    except Exception as e:
        r.fail("ROI-1 well-formed", f"{type(e).__name__}: {e}")

    # (2) The non-local functions carry a footprint of the EXPECTED family — this is
    #     the "demand a classified footprint" teeth: an img-reduction must be 'image',
    #     morphology/blur must be halo-family, cross-frame samplers must be ('frame', i).
    #     A future edit that mis-files one (e.g. tags gauss_blur 'image' and loses the
    #     tileability ROI-5 will exploit) trips here, not silently at ship.
    def _kind(fp):
        return fp if isinstance(fp, str) else fp[0]

    EXPECT = {
        "img_sum": "image", "img_mean": "image", "img_min": "image",
        "img_max": "image", "img_median": "image",
        "sample": "image", "sample_cubic": "image", "sample_lanczos": "image",
        "sample_mip": "image", "sample_mip_gauss": "image", "sample_grad": "image",
        "fetch": "image",
        "fetch_frame": "frame", "sample_frame": "frame",
        "erode": "halo_arg", "dilate": "halo_arg", "gauss_blur": "halo_arg",
        "bilateral_filter": "halo",
    }
    try:
        by_name = {n: e for e in R.REGISTRY for n in e.names}
        miss = []
        for name, want in EXPECT.items():
            got = _kind(by_name[name].footprint)
            if got != want:
                miss.append(f"{name}: want {want}, got {got}")
        assert not miss, "; ".join(miss)
        r.ok("every non-local fn is classified into its expected footprint family")
    except Exception as e:
        r.fail("ROI-1 classification", f"{type(e).__name__}: {e}")

    # (3) Point-footprint fns stay out of the tiling set (no over-classification).
    try:
        overtagged = sorted(n for e in R.REGISTRY for n in e.names
                            if e.footprint == "point" and n in _HISTORICAL_NON_LOCAL)
        assert not overtagged, f"point-footprint fns wrongly in non-local set: {overtagged}"
        r.ok("no 'point' fn leaked into the non-local set")
    except Exception as e:
        r.fail("ROI-1 no over-classification", f"{type(e).__name__}: {e}")


def test_roi1_malformed_footprint_fails_loud(r: SubTestResult):
    print("\n--- ROI-1: a malformed footprint raises at decoration (fail-loud) ---")
    bad_footprints = [
        ("halo", "x"),        # non-numeric radius
        ("halo", -1),         # non-positive radius
        ("halo", 0),          # zero radius
        ("halo", True),       # bool masquerading as a radius
        ("halo_arg", -1),     # negative arg index
        ("halo_arg", 1.0),    # float arg index
        ("frame",),           # missing arg
        "halo",               # bare kind string
        ("mystery", 3),       # unknown kind
        None,                 # not a descriptor
    ]
    try:
        leaked = [fp for fp in bad_footprints if R._valid_footprint(fp)]
        assert not leaked, f"validator accepted malformed footprints: {leaked}"
        r.ok(f"_valid_footprint rejects all {len(bad_footprints)} malformed descriptors")
    except Exception as e:
        r.fail("ROI-1 validator rejects", f"{type(e).__name__}: {e}")

    # The decorator itself must raise — this is what makes a typo a loud import error.
    try:
        raised = False
        try:
            @R.stdlib("_roi1_probe_", footprint=("halo", "nope"))
            def _probe():
                return None
        except ValueError:
            raised = True
        assert raised, "@stdlib did not raise ValueError on a bad footprint"
        # And it must NOT have polluted the registry.
        assert "_roi1_probe_" not in {n for e in R.REGISTRY for n in e.names}, \
            "a rejected registration still reached REGISTRY"
        r.ok("@stdlib raises ValueError on a bad footprint and does not register it")
    except Exception as e:
        r.fail("ROI-1 decorator raises", f"{type(e).__name__}: {e}")


def test_roi1_is_tile_safe_unchanged(r: SubTestResult):
    print("\n--- ROI-1: is_tile_safe verdicts preserved (invariant #7) ---")
    from TEX_Wrangle.tex_compiler.lexer import Lexer
    from TEX_Wrangle.tex_compiler.parser import Parser

    # is_tile_safe walks the parsed AST for non-local FunctionCall names + index/sample
    # access; no typecheck needed (mirrors the M-4 oracle in test_v015_phase3).
    def _parse(code):
        return Parser(Lexer(code).tokenize(), source=code).parse()

    cases = [
        ("@OUT = vec4(@A.rgb * v + u, 1.0);", True, "pointwise arithmetic"),
        ("@OUT = vec4(sample(@A, u + 0.01, v).rgb, 1.0);", False, "sample -> 'image'"),
        ("@OUT = vec4(gauss_blur(@A, 2.0).rgb, 1.0);", False, "gauss_blur -> ('halo_arg',1,3.0)"),
        ("@OUT = vec4(vec3(img_mean(@A)), 1.0);", False, "img_mean -> 'image'"),
        ("@OUT = vec4(sample_frame(@A, 0, u, v).rgb, 1.0);", False, "sample_frame -> ('frame',1)"),
    ]
    for src, want, label in cases:
        try:
            got = tex_memory.is_tile_safe(_parse(src))
            assert got == want, f"is_tile_safe({label}) = {got}, want {want}"
            r.ok(f"is_tile_safe: {label} -> {want}")
        except Exception as e:
            r.fail(f"ROI-1 is_tile_safe {label}", f"{type(e).__name__}: {e}")


# ─────────────────────────── LANG-1: param UI metadata ───────────────────────────

def _parse_stmt(code):
    from TEX_Wrangle.tex_compiler.lexer import Lexer
    from TEX_Wrangle.tex_compiler.parser import Parser
    return Parser(Lexer(code).tokenize(), source=code).parse().statements[0]


def test_lang1_metadata_grammar(r: SubTestResult):
    print("\n--- LANG-1: param metadata block parses into ParamDecl.metadata ---")
    from TEX_Wrangle.tex_compiler.ast_nodes import ParamDecl

    # (1) default + full metadata block
    try:
        p = _parse_stmt('f$strength = 0.5 [min: 0, max: 2, label: "Strength"];')
        assert isinstance(p, ParamDecl) and p.name == "strength" and p.type_hint == "f"
        assert p.default_expr is not None
        assert p.metadata == {"min": 0, "max": 2, "label": "Strength"}, p.metadata
        r.ok("f$x = 0.5 [min, max, label] -> metadata dict + default parsed")
    except Exception as e:
        r.fail("LANG-1 default+metadata", f"{type(e).__name__}: {e}")

    # (2) metadata WITHOUT a default (new dispatch on LBRACKET), negative + float values
    try:
        p = _parse_stmt('f$gain [min: -1.5, max: 1, step: 0.05];')
        assert p.default_expr is None
        assert p.metadata == {"min": -1.5, "max": 1, "step": 0.05}, p.metadata
        r.ok("f$x [min:-1.5, max:1, step:0.05] (no default) parses")
    except Exception as e:
        r.fail("LANG-1 metadata-no-default", f"{type(e).__name__}: {e}")

    # (3) backward compat — no block means metadata is None (not {})
    try:
        assert _parse_stmt('i$count = 3;').metadata is None
        assert _parse_stmt('f$amt;').metadata is None
        r.ok("declarations with no metadata block keep metadata == None")
    except Exception as e:
        r.fail("LANG-1 backward-compat", f"{type(e).__name__}: {e}")

    # (4) empty block is a valid (empty) metadata dict
    try:
        assert _parse_stmt('f$x [];').metadata == {}
        r.ok("empty block `[]` -> metadata == {}")
    except Exception as e:
        r.fail("LANG-1 empty-block", f"{type(e).__name__}: {e}")


def test_lang1_metadata_literals_only(r: SubTestResult):
    print("\n--- LANG-1: metadata values are literals only; malformed blocks rejected ---")
    from TEX_Wrangle.tex_compiler.parser import ParseError

    bad = [
        ('f$x = 0 [min: $y];', "binding ref as value"),
        ('f$x = 0 [min: 1 + 2];', "expression as value"),
        ('f$x = 0 [min: 0, min: 1];', "duplicate key"),
        ('f$x = 0 [0: 1];', "non-identifier key"),
        ('f$x = 0 [min 0];', "missing colon"),
        ('f$x = 0 [min: 0;', "unclosed block"),
    ]
    for src, label in bad:
        try:
            _parse_stmt(src)
            r.fail(f"LANG-1 reject {label}", "expected a ParseError, got none")
        except ParseError:
            r.ok(f"rejected: {label}")
        except Exception as e:
            r.fail(f"LANG-1 reject {label}", f"wrong error {type(e).__name__}: {e}")


def test_lang1_metadata_ignored_by_typecheck(r: SubTestResult):
    print("\n--- LANG-1: metadata survives compile and the type checker ignores it ---")
    from TEX_Wrangle.tex_api import compile as tex_compile
    from TEX_Wrangle.tex_compiler.types import TEXType
    from TEX_Wrangle.tex_compiler.ast_nodes import ParamDecl

    # A program that USES the param must still type-check identically; the metadata is
    # carried through the optimizer + re-check untouched on Program.ast (for tool manifests).
    try:
        prog = tex_compile(
            'f$strength = 0.5 [min: 0, max: 2, label: "Gain"];\n'
            '@OUT = vec4(@A.rgb * $strength, 1.0);',
            {"A": TEXType.VEC3})
        pds = [s for s in prog.ast.statements if isinstance(s, ParamDecl)]
        assert pds, "no ParamDecl on the compiled AST"
        assert pds[0].metadata == {"min": 0, "max": 2, "label": "Gain"}, pds[0].metadata
        # The param still registers with its type (metadata did not perturb type checking).
        assert "strength" in prog.params, prog.params
        r.ok("metadata carried through compile; param types unchanged (checker ignores it)")
    except Exception as e:
        r.fail("LANG-1 compile-survival", f"{type(e).__name__}: {e}")

    # An out-of-range metadata literal is NOT a type error (checker must not validate it).
    try:
        tex_compile('f$x = 0.5 [min: 100, max: -100];\n@OUT = vec4($x, 0.0, 0.0, 1.0);',
                    {"A": TEXType.VEC3})
        r.ok("nonsensical metadata (min>max) is not a compile error — checker ignores metadata")
    except Exception as e:
        r.fail("LANG-1 checker-ignores", f"{type(e).__name__}: {e}")


# ─────────────────────── LANG-2: check() API + W7xxx warnings ───────────────────────

def _codes(src, bt=None):
    from TEX_Wrangle.tex_api import check
    from TEX_Wrangle.tex_compiler.types import TEXType
    if bt is None:
        bt = {"A": TEXType.VEC3}
    return check(src, bt)


def test_lang2_check_never_raises(r: SubTestResult):
    print("\n--- LANG-2: tex_api.check() is total — always a list, never raises ---")
    from TEX_Wrangle.tex_compiler.diagnostics import TEXDiagnostic

    cases = [
        ('@OUT = vec4(@A.rgb, 1.0);', "clean program"),
        ('@OUT = "hi" + 3;', "type error"),
        ('@OUT = vec4(@A.rgb, 1.0)', "syntax error (missing ;)"),
        ('@@@ %%% $$$ !!!', "lexical garbage"),
        ('', "empty source"),
        ('for (', "truncated statement"),
    ]
    for src, label in cases:
        try:
            out = _codes(src)
            assert isinstance(out, list), f"check() returned {type(out)}, not list"
            assert all(isinstance(d, TEXDiagnostic) for d in out), "non-TEXDiagnostic in list"
            r.ok(f"check() -> list of {len(out)} diagnostics ({label})")
        except Exception as e:
            r.fail(f"LANG-2 never-raises {label}", f"{type(e).__name__}: {e}")

    # A clean program yields ZERO diagnostics.
    try:
        assert _codes('@OUT = vec4(@A.rgb * 1.5, 1.0);') == [], "clean program had diagnostics"
        r.ok("a clean program returns [] (no false diagnostics)")
    except Exception as e:
        r.fail("LANG-2 clean-empty", f"{type(e).__name__}: {e}")


def test_lang2_w7xxx_warnings(r: SubTestResult):
    print("\n--- LANG-2: the first W7xxx advisories ---")

    def has(src, code, bt=None):
        return any(d.code == code and d.severity == "warning" for d in _codes(src, bt))

    from TEX_Wrangle.tex_compiler.types import TEXType

    checks = [
        ("W7001 unused variable",
         has('float unused = 3.0;\n@OUT = vec4(@A.rgb, 1.0);', "W7001")),
        ("W7001 NOT fired for a used variable",
         not has('float g = 2.0;\n@OUT = vec4(@A.rgb * g, 1.0);', "W7001")),
        ("W7002 unused input",
         has('@OUT = vec4(0.0, 0.0, 0.0, 1.0);', "W7002",
             {"A": TEXType.VEC3, "B": TEXType.VEC3})),
        ("W7002 NOT fired when the input is used",
         not has('@OUT = vec4(@A.rgb, 1.0);', "W7002")),
        ("W7003 param shadows a builtin",
         has('f$time = 0.5;\n@OUT = vec4($time, 0,0,1);', "W7003")),
        ("W7003 nested-scope variable shadow",
         has('float x = 1.0;\nfor (int i=0;i<2;i=i+1){ float x=2.0; @OUT=vec4(x,0,0,1);}', "W7003")),
        ("no W7003 when no shadow",
         not has('float y = 1.0;\n@OUT = vec4(@A.rgb * y, 1.0);', "W7003")),
    ]
    for label, ok in checks:
        if ok:
            r.ok(f"LANG-2 {label}")
        else:
            r.fail(f"LANG-2 {label}", "expectation not met")

    # Warnings are severity 'warning'; errors are severity 'error' — a consumer can split.
    try:
        ds = _codes('float dead = 1.0;\n@OUT = "s" + 1;')   # one warning + one error
        sev = {d.severity for d in ds}
        assert "warning" in sev and "error" in sev, f"severities: {sev}"
        r.ok("check() returns both 'warning' and 'error' severities in one pass")
    except Exception as e:
        r.fail("LANG-2 severity-split", f"{type(e).__name__}: {e}")

    # A rejected redeclaration (E3001) must not double-count the unused-variable warning:
    # exactly ONE W7001, not one per decl (the `_note_local_decl` on failed-declare fix).
    try:
        ds = _codes('float x = 1.0; float x = 2.0; @OUT = vec4(0,0,0,1);')
        assert sum(1 for d in ds if d.code == "W7001") == 1, \
            [d.code for d in ds if d.code == "W7001"]
        r.ok("a redeclared-and-unused variable yields exactly one W7001 (not one per decl)")
    except Exception as e:
        r.fail("LANG-2 W7001 no-double", f"{type(e).__name__}: {e}")


def test_lang2_sourceloc_end_line(r: SubTestResult):
    print("\n--- LANG-2: SourceLoc.end_line + to_dict emission ---")
    from TEX_Wrangle.tex_compiler.ast_nodes import SourceLoc
    from TEX_Wrangle.tex_compiler.diagnostics import TEXDiagnostic

    try:
        # Default: end_line falls back to line (single-line span).
        assert SourceLoc(5, 2).end_line == 5, "default end_line != line"
        # Explicit multi-line end.
        assert SourceLoc(5, 2, None, 8).end_line == 8, "explicit end_line not honored"
        # The positional `stage` arg is NOT reinterpreted as end_line.
        loc = SourceLoc(1, 1, 2)
        assert loc.stage == 2 and loc.end_line == 1, f"stage/end_line clash: {loc.stage},{loc.end_line}"
        r.ok("SourceLoc.end_line: defaults to line, honors explicit end, preserves positional stage")
    except Exception as e:
        r.fail("LANG-2 end_line", f"{type(e).__name__}: {e}")

    try:
        d = TEXDiagnostic(code="E1", severity="error", message="m",
                          loc=SourceLoc(3, 1, None, 5), source_line="x")
        assert d.to_dict()["end_line"] == 5, d.to_dict()
        assert TEXDiagnostic(code="E1", severity="error", message="m", loc=None,
                             source_line="").to_dict()["end_line"] is None
        r.ok("to_dict emits end_line (value when present, None when loc is None)")
    except Exception as e:
        r.fail("LANG-2 to_dict end_line", f"{type(e).__name__}: {e}")


# ─────────────── LANG-3: language versioning + the frozen compat corpus ───────────────

def test_lang3_version_and_pragma(r: SubTestResult):
    print("\n--- LANG-3: LANGUAGE_VERSION + //!tex pragma ---")
    from TEX_Wrangle.tex_api import LANGUAGE_VERSION, language_pragma, check
    from TEX_Wrangle.tex_compiler.types import TEXType

    try:
        assert isinstance(LANGUAGE_VERSION, str) and "." in LANGUAGE_VERSION, LANGUAGE_VERSION
        r.ok(f"tex_api.LANGUAGE_VERSION = {LANGUAGE_VERSION!r} (separate from __version__)")
    except Exception as e:
        r.fail("LANG-3 version constant", f"{type(e).__name__}: {e}")

    try:
        assert language_pragma("//!tex 0.20\n@OUT = vec4(u,v,0,1);") == "0.20"
        assert language_pragma("  //!tex 1.5\nmore") == "1.5"
        assert language_pragma("@OUT = vec4(u,v,0,1);") is None
        assert language_pragma("// a normal comment\n@OUT = vec4(u,v,0,1);") is None
        r.ok("language_pragma extracts //!tex X.Y, else None")
    except Exception as e:
        r.fail("LANG-3 pragma parse", f"{type(e).__name__}: {e}")

    try:
        newer = check("//!tex 99.0\n@OUT = vec4(@A.rgb, 1.0);", {"A": TEXType.VEC3})
        assert any(d.code == "W7004" and d.severity == "warning" for d in newer), \
            [d.code for d in newer]
        older = check("//!tex 0.20\n@OUT = vec4(@A.rgb, 1.0);", {"A": TEXType.VEC3})
        assert not any(d.code == "W7004" for d in older), [d.code for d in older]
        none = check("@OUT = vec4(@A.rgb, 1.0);", {"A": TEXType.VEC3})
        assert not any(d.code == "W7004" for d in none)
        r.ok("check() advises W7004 only when the pragma targets a NEWER language")
    except Exception as e:
        r.fail("LANG-3 pragma advisory", f"{type(e).__name__}: {e}")

    # A pragma is only recognized as a LEADING line comment — not one buried inside a
    # block comment or after real code (else a commented-out `//!tex` raises a spurious
    # W7004). Guards the fix for the bug-hunt finding.
    try:
        assert language_pragma("/*\n//!tex 99.0\n*/\n@OUT = vec4(u,v,0,1);") is None
        assert language_pragma("@OUT = vec4(u,v,0,1);\n//!tex 99.0") is None
        assert language_pragma("// a note\n//!tex 0.20\n@OUT = vec4(u,v,0,1);") == "0.20"
        bc = check("/*\n//!tex 99.0\n*/\n@OUT = vec4(@A.rgb, 1.0);", {"A": TEXType.VEC3})
        assert not any(d.code == "W7004" for d in bc), "block-commented pragma warned"
        r.ok("a pragma inside a block comment / after code is ignored (no spurious W7004)")
    except Exception as e:
        r.fail("LANG-3 pragma placement", f"{type(e).__name__}: {e}")


def test_lang3_compat_corpus(r: SubTestResult):
    print("\n--- LANG-3: the frozen compat corpus matches its goldens (PM-4) ---")
    import compat_corpus

    try:
        golden = compat_corpus.load_goldens()["hashes"]
    except Exception as e:
        r.fail("LANG-3 corpus goldens", f"could not load goldens: {e}")
        return

    if len(golden) < 100:
        r.fail("LANG-3 corpus size", f"only {len(golden)} goldens (<100)")
    else:
        r.ok(f"compat corpus covers {len(golden)} programs (116 examples + adversarial)")

    current = compat_corpus.compute_all()

    errs = sorted(k for k, v in current.items()
                  if isinstance(v, str) and v.startswith("ERROR"))
    if errs:
        r.fail("LANG-3 corpus errors", f"programs failing to compile/run: {errs}")
    else:
        r.ok("every corpus program compiles + runs on CPU")

    mism = [f"{n}" for n, gh in golden.items() if current.get(n) != gh]
    missing = sorted(set(golden) - set(current))
    added = sorted(set(current) - set(golden))
    if mism or missing or added:
        r.fail("LANG-3 compat drift",
               f"changed={mism[:6]} missing={missing[:4]} new={added[:4]} — a language "
               f"change altered existing output; if intentional, regenerate goldens")
    else:
        r.ok(f"all {len(golden)} program output-hashes match the frozen goldens")


# ─────────────────────── ENG-6: zero-copy AI handoff (DLPack) ───────────────────────

def test_eng6_dlpack_contract(r: SubTestResult):
    print("\n--- ENG-6: DLPack zero-copy handoff contract ---")
    import torch
    from TEX_Wrangle import tex_engine

    res = tex_engine.cook("@OUT = vec4(@A.rgb * 1.5, 1.0);",
                          {"A": torch.rand(1, 8, 8, 3)}, device_mode="cpu")
    out = res.outputs["OUT"]

    # (1) the pinned output contract: tensor, fp32, 4-D BHWC, on the cook device.
    try:
        assert isinstance(out, torch.Tensor), "output is not a tensor"
        assert out.dtype == torch.float32, f"output dtype {out.dtype}, want fp32"
        assert out.dim() == 4 and out.shape[-1] <= 4, f"not BHWC: {tuple(out.shape)}"
        assert str(out.device) == "cpu", f"output device {out.device}"
        r.ok("cook output is a fp32 BHWC device-resident tensor")
    except Exception as e:
        r.fail("ENG-6 output contract", f"{type(e).__name__}: {e}")

    # (2) copy=True (default): round-trips values, is OWNED (no shared storage) and
    #     grad-ready (materialized out of inference_mode).
    try:
        back = tex_engine.from_dlpack(tex_engine.to_dlpack(out))
        assert torch.allclose(back, out), "values changed through DLPack"
        assert back.data_ptr() != out.data_ptr(), "copy=True still shares storage"
        assert not back.is_inference(), "copy=True handed back an inference tensor"
        g = back.detach().requires_grad_(True)
        g.sum().backward()
        assert g.grad is not None, "handed-off tensor is not grad-ready"
        r.ok("to_dlpack(copy=True) -> owned, grad-ready, value-identical round-trip")
    except Exception as e:
        r.fail("ENG-6 copy round-trip", f"{type(e).__name__}: {e}")

    # (3) copy=False: a genuine zero-copy view (shares storage).
    try:
        view = tex_engine.from_dlpack(tex_engine.to_dlpack(out, copy=False))
        assert view.data_ptr() == out.data_ptr(), "copy=False did not share storage"
        assert torch.allclose(view, out)
        r.ok("to_dlpack(copy=False) -> genuine zero-copy view (shared storage)")
    except Exception as e:
        r.fail("ENG-6 zero-copy view", f"{type(e).__name__}: {e}")

    # (4) layout='bchw' gives an NCHW view matching the permute.
    try:
        bb = tex_engine.from_dlpack(tex_engine.to_dlpack(out, layout="bchw"))
        assert tuple(bb.shape) == (1, out.shape[-1], 8, 8), tuple(bb.shape)
        assert torch.allclose(bb, out.permute(0, 3, 1, 2).contiguous())
        r.ok("to_dlpack(layout='bchw') -> NCHW-shaped, value-correct")
    except Exception as e:
        r.fail("ENG-6 bchw layout", f"{type(e).__name__}: {e}")

    # (5) misuse raises loudly.
    try:
        errs = 0
        for thunk in (lambda: tex_engine.to_dlpack([1, 2, 3]),
                      lambda: tex_engine.to_dlpack(out, layout="nchw"),
                      lambda: tex_engine.to_dlpack(torch.rand(3), layout="bchw")):
            try:
                thunk()
            except (TypeError, ValueError):
                errs += 1
        assert errs == 3, f"only {errs}/3 misuse cases raised"
        r.ok("to_dlpack rejects non-tensors, bad layouts, and bchw on non-4-D")
    except Exception as e:
        r.fail("ENG-6 misuse guards", f"{type(e).__name__}: {e}")


# ─────────────────── ENG-9: concurrency — per-thread interpreters ───────────────────

def test_eng9_two_thread_cpu_cook(r: SubTestResult):
    print("\n--- ENG-9: concurrent CPU cooks are isolated (per-thread interpreters) ---")
    import threading
    import torch
    from TEX_Wrangle import tex_engine

    # The interpreter carries per-instance execution state; a SHARED instance mixes up
    # programs across threads (proven pre-ENG-9). With per-thread interpreters, concurrent
    # cooks must each return exactly their single-threaded reference. Distinct programs +
    # a spread of values also churn the shared LRU memo caches (tile-safe/lazy/fingerprint)
    # under concurrent eviction.
    progs = [
        "@OUT = vec4(@A.rgb * 1.5, 1.0);",
        "@OUT = vec4(sin(u*6.2831)*0.5+0.5, v, u*v, 1.0);",
        "float s=0.0; for(int i=0;i<4;i=i+1){ s=s+float(i)*0.1; } @OUT=vec4(s,s,s,1.0);",
        "@OUT = vec4(gauss_blur(@A, 1.5).rgb, 1.0);",
        "vec3 c=vec3(u,v,u*v); @OUT = vec4(c*2.0-0.5, 1.0);",
    ]
    torch.manual_seed(0)
    ref = []
    for p in progs:
        im = torch.rand(1, 12, 12, 3)
        out = tex_engine.cook(p, {"A": im}, device_mode="cpu").outputs["OUT"]
        ref.append((im.clone(), out.clone()))

    errors = []

    def worker(tid):
        try:
            order = list(enumerate(progs))
            if tid % 2:
                order.reverse()
            for _ in range(25):
                for i, p in order:
                    base, expect = ref[i]
                    got = tex_engine.cook(p, {"A": base.clone()},
                                          device_mode="cpu").outputs["OUT"]
                    if not torch.allclose(got, expect, atol=1e-5):
                        errors.append(f"t{tid} prog{i} mismatch "
                                      f"{float((got-expect).abs().max()):.2e}")
                        return
        except Exception as e:
            errors.append(f"t{tid} {type(e).__name__}: {e}")

    ths = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()

    if errors:
        r.fail("ENG-9 two-thread cook", f"{len(errors)} error(s): {errors[:4]}")
    else:
        r.ok("4 threads x 25 iters x 5 programs cook correctly + independently (no races)")

    # The engine hands each thread a DISTINCT Interpreter instance (the isolation source).
    try:
        seen = {}

        def grab(tid):
            seen[tid] = id(tex_engine._get_interpreter())

        gs = [threading.Thread(target=grab, args=(t,)) for t in range(3)]
        for t in gs:
            t.start()
        for t in gs:
            t.join()
        assert len(set(seen.values())) == len(seen), f"threads shared an interpreter: {seen}"
        # And the SAME thread always gets the SAME instance (thread-local, not per-call).
        assert tex_engine._get_interpreter() is tex_engine._get_interpreter()
        r.ok("each thread gets its own Interpreter; a thread reuses its own instance")
    except Exception as e:
        r.fail("ENG-9 per-thread instances", f"{type(e).__name__}: {e}")


# ─────────────────────── LANG-4: help data from the registry ───────────────────────

def test_lang4_registry_help(r: SubTestResult):
    print("\n--- LANG-4: function help is single-sourced from the registry ---")
    from TEX_Wrangle.tex_runtime.stdlib import TEXStdlib  # noqa: F401 (populate REGISTRY)
    from TEX_Wrangle.tex_runtime import stdlib_registry as R

    # (1) every registered function carries the migrated sig + category (the guard that
    #     keeps the JS from being the only home for help data).
    try:
        empty = [e.name for e in R.REGISTRY if not e.sig.strip() or not e.category.strip()]
        assert not empty, f"functions missing sig/category: {empty}"
        r.ok(f"all {len(R.REGISTRY)} functions carry a sig and a category")
    except Exception as e:
        r.fail("LANG-4 sig/category populated", f"{type(e).__name__}: {e}")

    # (2) help_lookup resolves by primary name AND alias, decoding the sig for display.
    try:
        e = R.help_lookup("sin")
        assert e and e["sig"] == "sin(x) → float", e
        assert "→" in e["sig"], "sig was not decoded"
        m = R.help_lookup("mix")   # alias of lerp
        assert m and m["name"] == "lerp" and "mix" in m["aliases"], m
        assert R.help_lookup("does_not_exist") is None
        r.ok("help_lookup resolves primary + alias, decodes the sig, None on miss")
    except Exception as e:
        r.fail("LANG-4 help_lookup", f"{type(e).__name__}: {e}")

    # (3) the generated help JSON covers every function.
    try:
        import json
        import os
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data = json.load(open(os.path.join(root, "tex_help.json"), encoding="utf-8"))
        assert data["function_count"] == len(R.REGISTRY), \
            f"tex_help.json has {data['function_count']}, registry has {len(R.REGISTRY)}"
        assert len(data["categories"]) >= 10, data["categories"]
        r.ok(f"tex_help.json exposes all {data['function_count']} functions in "
             f"{len(data['categories'])} categories")
    except Exception as e:
        r.fail("LANG-4 help JSON", f"{type(e).__name__}: {e}")

    # (4) every function's `ex=` help snippet COMPILES to the right arity — an arity/
    #     unknown-fn error (E5001/E5002) means the documented signature lies about the
    #     impl (the sdf_circle/box/polygon class the bug-hunt caught). E3003 (undefined
    #     var, an illustration free variable) is fine and ignored.
    try:
        from TEX_Wrangle.tex_api import check
        bad = []
        for e in R.help_entries(decode=True):
            for d in check(e["example"], {}):
                if d.code in ("E5001", "E5002"):
                    bad.append(f"{e['name']}: {d.code} on ex {e['example']!r}")
                    break
        assert not bad, "; ".join(bad[:6])
        r.ok("every function's `ex=` help snippet matches its impl's arity")
    except Exception as e:
        r.fail("LANG-4 help-example arity", f"{type(e).__name__}: {e}")


# ─────────────────────── LANG-5: server-side snippet store ───────────────────────

def test_lang5_snippet_store(r: SubTestResult):
    print("\n--- LANG-5: server-backed user snippet store ---")
    import os
    import shutil
    import tempfile
    from TEX_Wrangle import tex_snippets
    from TEX_Wrangle.tex_runtime import host as H

    # (1) the get_user_dir seam: Null host has none (falls back to the cache dir).
    try:
        assert callable(getattr(H.NullHostServices(), "get_user_dir", None))
        assert H.NullHostServices().get_user_dir() is None
        r.ok("HostServices.get_user_dir() exists; Null host returns None")
    except Exception as e:
        r.fail("LANG-5 get_user_dir seam", f"{type(e).__name__}: {e}")

    # Drive the store against a mock host with a real temp user dir.
    tmp = tempfile.mkdtemp(prefix="tex_snip_")

    class _MockHost(H.NullHostServices):
        def get_user_dir(self):
            return tmp

    prev = H.get_host_services()
    H.set_host_services(_MockHost())
    try:
        # (2) empty before anything is saved.
        assert tex_snippets.load_user_snippets() == {}, "store not initially empty"
        # (3) round-trip through the JSON file under <user_dir>/tex_wrangle/.
        snips = {"Grades/warm": "@OUT = @A * vec4(1.1,1.0,0.9,1.0);", "vig": "@OUT = @A;"}
        assert tex_snippets.save_user_snippets(snips) is True, "save failed"
        assert tex_snippets.load_user_snippets() == snips, "round-trip mismatch"
        assert os.path.isfile(os.path.join(tmp, "tex_wrangle", "user_snippets.json"))
        r.ok("save/load round-trips {name: code} through the user dir")

        # (4) hostile payload is sanitized to {str: str}; non-string keys dropped.
        tex_snippets.save_user_snippets({"a": 123, "b": "ok", 7: "bad-key"})
        got = tex_snippets.load_user_snippets()
        assert got.get("a") == "123" and got.get("b") == "ok", got
        assert 7 not in got and "7" not in got, "non-string key leaked in"
        r.ok("save coerces values to strings and drops non-string keys")

        # (5) a replace overwrites the whole map (mirrors the frontend's load-modify-save).
        tex_snippets.save_user_snippets({"only": "x"})
        assert tex_snippets.load_user_snippets() == {"only": "x"}
        r.ok("save replaces the whole map (frontend load-modify-save semantics)")

        # (6) the read-vs-empty distinction the sync depends on (LANG-5 BUG 2): a store that
        #     EXISTS but can't be read as a dict RAISES (not a silent {}), while a genuinely
        #     absent store stays empty. This is what stops a transient read failure from
        #     being reported as "empty" and wiping the frontend's offline cache.
        p = os.path.join(tmp, "tex_wrangle", "user_snippets.json")
        with open(p, "w", encoding="utf-8") as f:
            f.write("{ not valid json")
        raised = False
        try:
            tex_snippets.load_user_snippets()
        except tex_snippets.SnippetStoreError:
            raised = True
        assert raised, "a corrupt store must raise SnippetStoreError, not return {}"
        os.remove(p)
        assert tex_snippets.load_user_snippets() == {}, "an absent store must read as empty {}"
        # a store PATH that exists but isn't a readable file (a directory here; stands in for
        # a locked / permission-denied / stat-failing path) is a READ ERROR, not "absent":
        # the old os.path.isfile() presence gate swallows the OSError and would mis-report it
        # as an empty {} — the residual BUG 2 on the presence check.
        os.makedirs(p)   # the store path is now a directory -> open() raises, not FileNotFound
        raised_dir = False
        try:
            tex_snippets.load_user_snippets()
        except tex_snippets.SnippetStoreError:
            raised_dir = True
        assert raised_dir, "a non-file store path (dir/locked) must raise, not read as empty {}"
        os.rmdir(p)
        # an explicitly-saved empty map is still a legitimate (non-raising) empty store
        tex_snippets.save_user_snippets({})
        assert tex_snippets.load_user_snippets() == {}, "saved {} must read back as {} (no raise)"
        r.ok("load raises on an unreadable store (corrupt / non-file), empty on absent/{} (BUG 2)")
    except Exception as e:
        r.fail("LANG-5 snippet store", f"{type(e).__name__}: {e}")
    finally:
        H.set_host_services(prev)
        shutil.rmtree(tmp, ignore_errors=True)


def test_lang5_snippet_route(r: SubTestResult):
    print("\n--- LANG-5: user_snippets route contract (round-trip, read-error, failed-save) ---")
    import json as _json
    import os
    import shutil
    import tempfile
    from TEX_Wrangle import tex_snippets
    from TEX_Wrangle.tex_snippets import (
        user_snippets_get_payload, user_snippets_post_payload)
    from TEX_Wrangle.tex_runtime import host as H

    tmp = tempfile.mkdtemp(prefix="tex_snip_route_")
    store = os.path.join(tmp, "tex_wrangle", "user_snippets.json")

    class _MockHost(H.NullHostServices):
        def get_user_dir(self):
            return tmp

    prev = H.get_host_services()
    H.set_host_services(_MockHost())
    try:
        # (1) POST -> GET round-trip through the EXACT functions the aiohttp routes call.
        try:
            snips = {"Grade/warm": "@OUT = @A * vec4(1.1,1.0,0.9,1.0);", "vig": "@OUT = @A;"}
            body, status = user_snippets_post_payload({"snippets": snips})
            assert status == 200 and body == {"ok": True}, (status, body)
            body, status = user_snippets_get_payload()
            assert status == 200 and body == {"snippets": snips} and "read_error" not in body, \
                (status, body)
            r.ok("POST {ok:true,200} then GET returns the same map at 200 (no read_error)")
        except Exception as e:
            r.fail("LANG-5 route round-trip", f"{type(e).__name__}: {e}")

        # (2) EMPTY is not a read error: an explicitly-saved {} and a genuinely-absent store
        #     both GET as 200 {} with NO read_error flag (the distinction BUG 2 hinges on).
        try:
            body, status = user_snippets_post_payload({"snippets": {}})
            assert status == 200 and body == {"ok": True}, (status, body)
            body, status = user_snippets_get_payload()
            assert status == 200 and body == {"snippets": {}} and "read_error" not in body, \
                (status, body)
            os.remove(store)   # now genuinely absent
            body, status = user_snippets_get_payload()
            assert status == 200 and body == {"snippets": {}} and "read_error" not in body, \
                (status, body)
            r.ok("empty store (saved {} AND absent file) GETs 200 {} — never flagged read_error")
        except Exception as e:
            r.fail("LANG-5 empty-not-error", f"{type(e).__name__}: {e}")

        # (3) READ-ERROR path (BUG 2): a store that exists but can't be read as a {name:code}
        #     object must NOT masquerade as empty — GET is 503 + read_error (not 200 {}).
        try:
            os.makedirs(os.path.dirname(store), exist_ok=True)
            with open(store, "w", encoding="utf-8") as f:
                f.write("] not json {")               # corrupt / partial write
            body, status = user_snippets_get_payload()
            assert status == 503 and body.get("read_error") is True and body.get("snippets") == {}, \
                (status, body)
            with open(store, "w", encoding="utf-8") as f:
                _json.dump(["not", "an", "object"], f)  # valid JSON, wrong shape — still a read error
            body, status = user_snippets_get_payload()
            assert status == 503 and body.get("read_error") is True, (status, body)
            r.ok("unreadable/non-object store -> 503 + read_error (never a silent empty 200)")
        except Exception as e:
            r.fail("LANG-5 route read-error", f"{type(e).__name__}: {e}")

        # (4) FAILED SAVE keeps the durable entry (BUG 1): a rejected POST returns non-2xx
        #     {ok:false} AND leaves the existing store intact — the server-side guarantee the
        #     frontend's pending-retry relies on. A malformed body must never wipe the store.
        try:
            good = {"keep": "@OUT = @A;"}
            body, status = user_snippets_post_payload({"snippets": good})
            assert status == 200, (status, body)
            # malformed bodies: snippets not an object, and the key missing entirely.
            for bad in ({"snippets": "oops"}, {}, {"snippets": None}, "not-a-dict"):
                body, status = user_snippets_post_payload(bad)
                assert status == 400 and body.get("ok") is False, (bad, status, body)
            # the durable store still holds the earlier save (no malformed request clobbered it).
            body, status = user_snippets_get_payload()
            assert status == 200 and body == {"snippets": good}, (status, body)
            # an undurable save (no resolvable store path) -> 503 {ok:false}; store preserved.
            real_path = tex_snippets._snippets_path
            tex_snippets._snippets_path = lambda: None
            try:
                body, status = user_snippets_post_payload({"snippets": {"x": "y"}})
                assert status == 503 and body.get("ok") is False, (status, body)
            finally:
                tex_snippets._snippets_path = real_path
            body, status = user_snippets_get_payload()
            assert status == 200 and body == {"snippets": good}, (status, body)
            r.ok("rejected save -> {ok:false} 4xx/503 and the prior store survives (BUG 1 guarantee)")
        except Exception as e:
            r.fail("LANG-5 route failed-save", f"{type(e).__name__}: {e}")
    finally:
        H.set_host_services(prev)
        shutil.rmtree(tmp, ignore_errors=True)
