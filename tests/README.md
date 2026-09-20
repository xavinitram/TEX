# TEX Test Suite

77 test functions containing ~1,215 sub-tests across 14 domain-specific files.

## Running Tests

```bash
cd custom_nodes/TEX_Wrangle

# Full suite
python -m pytest tests/ -v

# Skip slow timing tests
python -m pytest tests/ -v -m 'not slow'

# Single file
python -m pytest tests/test_stdlib.py -v

# Standalone (no pytest dependency)
python tests/run_all.py
```

## File Layout

| File | What it tests |
|------|---------------|
| `test_lexer.py` | Tokenization — token types, source locations, error recovery |
| `test_parser.py` | AST construction — expressions, statements, operator precedence |
| `test_type_checker.py` | Static type analysis — promotions, scope, type errors |
| `test_interpreter.py` | Runtime execution — evaluation, for/while loops, break/continue |
| `test_language.py` | Language features — if/else, ternary, scoping, swizzles, casting |
| `test_stdlib.py` | Built-in functions — math, color, SDF, edge cases, NaN/Inf |
| `test_strings_arrays.py` | String and array operations — indexing, bounds, string functions |
| `test_noise_sampling.py` | Procedural noise and texture sampling — Perlin, Worley, mip, blur |
| `test_bindings_params.py` | Bindings, wireable params, user functions, scatter writes |
| `test_codegen_optimizer.py` | Codegen/interpreter equivalence, optimizer passes, LICM |
| `test_integration.py` | End-to-end — cache, device selection, torch.compile, node helpers |
| `test_diagnostics.py` | Error messages — phrasing, suggestions, E-code correctness |
| `test_performance.py` | Timing benchmarks (marked `@pytest.mark.slow`) |

Supporting files:

| File | Role |
|------|------|
| `helpers.py` | Shared imports, `SubTestResult` accumulator, compilation helpers, test fixtures |
| `conftest.py` | pytest fixture wiring — provides the `r` fixture |
| `run_all.py` | Standalone runner. Discovers every `test_*(r, ...)` in `test_*.py` and calls them in a deterministic order — it names no test, so adding one needs no edit here |

## Sub-Test Pattern

Each `test_*` function receives a `SubTestResult` accumulator (`r`) and runs many sub-tests:

```python
def test_something(r: SubTestResult):
    # Sub-test with try/except
    try:
        result = compile_and_run("@OUT = vec3(1.0);", {"A": img})
        assert result.shape == (1, 8, 8, 3)
        r.ok("basic vec3 output")
    except Exception as e:
        r.fail("basic vec3 output", f"{e}\n{traceback.format_exc()}")

    # Quick scalar check
    check_val(r, "cos(0)", "float x = cos(0.0);\n@OUT = vec3(x,x,x);", 1.0)
```

The `conftest.py` fixture creates the `SubTestResult`, passes it to the test function, and asserts zero failures on teardown. This means pytest reports pass/fail per function, but the console output shows every individual sub-test.

## Adding a New Test

1. Find the right file from the table above (or create a new one for a new domain).
2. Write a function following the pattern: `def test_my_feature(r: SubTestResult):`.
3. Use helpers from `helpers.py` — everything is available via `from helpers import *`:
   - `compile_and_run(code, bindings)` — full pipeline, returns output tensor
   - `check_val(r, name, code, expected)` — compile, run, check scalar at `[0,0,0,0]`
   - `check_code(code, bindings)` — lex/parse/type-check only (no execution)
   - `run_both(code, bindings)` — run through both interpreter and codegen
   - `assert_equiv(r, name, code, bindings)` — verify interpreter/codegen match
   - `make_img(B, H, W, C, seed)` — deterministic test image
   - `make_latent(B, C, H, W, seed)` — fake LATENT dict
4. That is the whole wiring: `run_all.py` derives its call list from the tree, and
   `test_v017_phase1.py::test_tst7_runner_coverage` (TST-7) reds if a row it can see is one
   the runner cannot reach.
5. Run `python -m pytest tests/ -v` to verify.

## Pytest Markers

| Marker | Usage | Command |
|--------|-------|---------|
| `@pytest.mark.slow` | Timing-sensitive tests | `pytest -m 'not slow'` to skip |

## The known-red allowlist (`known_reds.json`)

`tools/gate.py` runs the suite and prints one verdict. A verdict can only be honest if the
failures it forgives are written down, so the reds a run is allowed to show live in
`tests/known_reds.json` as **data** rather than in a paragraph somebody reads and agrees with.

**The file is empty, and empty is the correct state.** Each entry is a failure somebody has
agreed to live with; a register that is never emptied trains the next reader to wave the list
through, which is exactly how the one standing red in this suite survived thirteen rounds.

```json
{
  "schema": 1,
  "entries": [
    {
      "id": "tests/test_example.py::test_something",
      "reason": "one sentence: what is actually failing, and why it is not a tree defect",
      "when": ["cuda"],
      "condition": "the human sentence for `when` — the environment this red is expected in",
      "owner": "who removes this entry, and what event lets them"
    }
  ]
}
```

| field | meaning |
|---|---|
| `id` | the pytest node id as the gate normalises it: `tests/<file>.py::<test>` |
| `reason` | why it is red. A reason that is really "nobody has looked" is a bug report, not an entry |
| `when` | machine-readable applicability, ANDed. Vocabulary: `always`, `cuda`, `no_cuda`, `leg:cheap`, `leg:ci-shape`, `leg:canonical`. An unknown token makes the entry apply to nothing — deliberately, so a typo cannot silently forgive a failure |
| `condition` | the same condition in words, for the reader |
| `owner` | who owns REMOVING it. Every entry has an exit |

Two rules the gate enforces by itself:

* an entry that does not fire, on a leg that actually collected the test it names, is
  reported as a **stale allowlist entry** and the run exits `2` — a list that claims a red
  the tree no longer has is a lie in the opposite direction, and costs the same;
* a failure with no matching entry is `RED`, named by node id.

Adding an entry is therefore a decision with an owner on it, and removing one is how a lane
finishes. If a red turns out to be a test bug rather than an environment fact, fix the test —
that is what emptied this file.
