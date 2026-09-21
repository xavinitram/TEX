# LANG-1 — masked per-pixel control flow (language `0.25`): the design

*The decisions, made once, before any code. **Design only — nothing here is built.** Written
against `main` at base sha `af3e8ae` (`LANGUAGE_VERSION` `0.24`). Every count below was
**measured at that head** with the command named beside it, not recalled from an earlier note;
every pointer was re-resolved, because the two design notes this one supersedes
(`ask-6`, `ask-11`) were written at `1a960bb` and the tree has moved under them.*

> **The decision is taken.** The author has ruled that language `0.25` lands in this run. This
> document is not an argument for *whether*; it is the argument for *how*, precise enough that
> an implementer cannot choose wrongly. Three live defects, an embedding host holding its pin,
> and a sunset waiting on the version are the reasons; §0 restates them as measured facts.

**The item, in one program.** This is what a TEX author writes today, and what the engine does
with it:

```tex
float n = @A.r * 10.0;          // 1, 3, 7, 9 across four pixels
float c = 0.0;
for (int i = 0; float(i) < n; i = i + 1) { c = c + 1.0; }
@OUT = vec4(c, c, c, 1.0);      // measured: 9, 9, 9, 9   — wanted: 1, 3, 7, 9
```

Every pixel runs as many passes as the greediest pixel in the region, and the body is not
masked. `LANGUAGE.md` §7.1 states that faithfully today, because v0.35.1 shipped the
documentation *instead of* the semantics, deliberately. Language `0.25` ships the semantics.

---

## 0. Facts at head, re-measured

All probes are read-only, CPU, `B=1, H=1, W=4`, one `@A` wire carrying
`a = [0.10, 0.30, 0.70, 0.90]` in `.r`, run through `tests/helpers.run_both` — the same
interpreter/codegen pairing the codegen-equivalence tests use. Own program text, own literals.

| id | construct | interpreter | codegen | per-pixel (wanted) | advisories |
|---|---|---|---|---|---|
| R-BREAK | `break` under `if (a > 0.5)` | `[10, 10, 10, 10]` | `[10, 10, 10, 10]` | `[-4, -4, 10, 10]` | `W7007` |
| R-CONT | `continue` under `if (a > 0.5)` | `[0, 0, 0, 0]` | `[0, 0, 0, 0]` | `[3, 3, 0, 0]` | `W7007` |
| R-RET | `return` under `if (a > 0.5)` in a function | `[1, 3, 7, 9]` | `[1, 3, 7, 9]` | `[10, 30, 7, 9]` | `W7007` |
| R-BOUND | `for (i; float(i) < a*10; ...)` | `[9, 9, 9, 9]` | `[9, 9, 9, 9]` | `[1, 3, 7, 9]` | `W7007` `W7008` |
| R-WBOUND | `while (x < 0.8) { x += 0.25; }` | `[3, 3, 3, 3]` | `[3, 3, 3, 3]` | `[3, 2, 1, 0]` | `W7007` `W7008` |

The two tiers agree with each other in every row and neither matches the per-pixel reading.
That is the shape of `TRK-3` (R-BREAK/R-CONT/R-RET) and `TRK-4` (R-BOUND/R-WBOUND), both still
open, both reproducing at this head.

**The third defect, `TRK-28`, re-measured here and worse than its row records.** A function
defined inside a loop, holding a bare `break`:

```tex
float total = 0.0;
for (int i = 0; i < 2; i = i + 1) {
  float nudge(float x) { if (x > 0.5) { break; } return x + 1.0; }
  for (int j = 0; j < 3; j = j + 1) { total = total + nudge(@A.r); }
}
```

`tex_api.check` returns **no diagnostic** — `type_checker._check_function_def` pushes a scope
but never resets `_loop_depth`, so `_check_break_continue`'s `E3002` never fires. The
interpreter then returns `[0, 0, 0, 0]` (`_Break` unwinds to the *outer* loop, the loop the
call sits in) and `try_compile` succeeds and **raises a bare `_CgBreak` out of the cook**
(codegen emits the transfer lexically inside the nested `def`, where the nearest emitted
`except _CgBreak` belongs to a different loop). Two tiers, two answers, one of them an
internal exception name. Invariant 2 is broken silently and the compiled tier believes it
succeeded. *Not measured here:* whether a full engine route catches the `_CgBreak` and falls
back; `TRK-28`'s own row measured the value-divergence spelling through
`compile_mode="auto"` at `b8605ce` and read `909.0` vs `1.0`.

**The corpus's one class-B program proves nothing.** `adv_while_loop`
(`tests/compat_corpus.py::_ADVERSARIAL`) is the only frozen program with a per-pixel loop
bound. Measured: the harness cooks it at a **`(4,)` output — a 1×1 grid**, because the
`_ADVERSARIAL` set is "builtin coords only, no `@inputs`" by design and
`_consensus_extent` has nothing to broadcast from. At one pixel, "as many passes as the
greediest pixel" *is* that pixel's own bound, so masked and unmasked agree by construction.
§6 turns on this.

---

## 1. The `0.25` rules, precisely

A program is cooked under the `0.25` rules **iff it declares `//!tex 0.25` or later in its
header** and the engine implements `0.25` or later. No pragma, or an older one, keeps the
`0.23` rules forever. Everything in this section is scoped by that sentence.

Each **active region** — a loop, one iteration of a loop, a user-function call — carries a
per-pixel boolean **live** mask, shaped like the cook grid or the scalar `True`. A pixel is
live in a region when it has entered that region and has not left it. The whole program starts
with `live = True` (uniform).

### M1 — writes

A write to a variable declared **outside** the innermost region containing the write is

```text
    target := where(live, new_value, target)
```

where `live` is the conjunction of the live masks of every region enclosing the write but
**not** enclosing the target's declaration. A write to a variable declared *inside* the region
is unmasked: a loop-header counter stays uniform, and a body-local temporary is dead on exit.

*A pixel that has left keeps the value it left with.* That one sentence is the whole of M1 and
every worked example below is an instance of it.

### M2 — `if`

A **0-dim (uniform) condition short-circuits**, exactly as in `0.23`: only the taken branch
runs. A **per-pixel condition keeps `0.23`'s model** — both branches run on every pixel, each
reads the pre-`if` frame, and the results merge with `torch.where`. `0.25` adds one thing: the
branch's live mask is `live & cond` (`then`) and `live & ~cond` (`else`), and a transfer taken
inside a branch clears bits in the *enclosing region's* mask rather than unwinding past the
merge.

This is deliberately **not** a change to what a per-pixel `if` costs. Both branches still run.
`W7006` is unaffected by this release.

### M3 — loops

A loop is entered with the enclosing live mask. Each pass:

1. evaluate the condition; `live := live & (cond > 0.5)`;
2. if no pixel is live, leave the loop;
3. run the body under `live`;
4. `break` clears the pixel's bit for the rest of the loop; `continue` clears it for the rest
   of *this pass only* and restores it at the update/condition;
5. run the update (unmasked for the loop-header counter, M1).

A cleared bit stays cleared for the remainder of that loop. An inner loop starts from the
outer live mask and its transfers touch only itself. The 1024-pass cap raises **E6010** when a
*live* pixel still needs pass 1024 — see §4's residue.

### M4 — `return` and calls

A call inherits the caller's live mask. `return e` records `e` for the pixels live at that
statement and clears their bits for the remainder of the call body; a pixel that reaches the
end of the body without returning gets `0.23`'s default (a zero scalar tensor,
`interpreter._exec_function_call`'s tail / `codegen._emit_function_def`'s trailing
`return _torch.scalar_tensor(0.0, ...)`). **A call with no live pixel is skipped entirely** —
that skip is what lets a per-pixel recursion terminate, and it is observable only through M6/M7
side effects, never through the returned value.

### M5 — scatter

`@T[x, y] op= v` is gated **by source**: a source pixel contributes iff it is live on the path
to the statement. Collisions between sources landing on one destination remain unspecified, as
in `0.23`. This is a change of rule, not only of masking: `0.23` gates scatter by
*destination*, which has no per-pixel meaning once a transfer can leave a branch.

### M6 — a binding write inside a called function

Uses M5's path mask. Today an `@` write performed inside a user function called from a
per-pixel branch lands on every pixel while the branch's own locals merge correctly
(`ast_nodes.collect_assigned_vars` does not descend into calls). Under `0.25` it lands only on
the live pixels.

### M7 — probes, whole-frame reads, strings

* `debug_print` records only if its probe pixel is live.
* A whole-frame read of written state (`blur(@OUT)`, `img_mean(x)`) sees the current frame with
  departed pixels frozen at the value they left with. A value that was uniform before a
  per-pixel path becomes per-pixel after it, and a builtin that requires a uniform argument
  raises the error it already raises.
* A **string** written on a per-pixel path keeps `0.23`'s majority-vote merge verbatim
  (`interpreter._merge_branch_vars`). A string has no per-pixel representation; `0.25` does not
  invent one. §4 depends on this.

### Worked examples

Grid: four pixels, `a = [0.10, 0.30, 0.70, 0.90]`. "before" is measured at `af3e8ae` on both
tiers; "after" is the rule above applied by hand and cross-checked against a plain-Python
per-pixel simulation of the same source.

**`break`** — R-BREAK

```tex
float hit = -1.0;
for (int i = 0; i < 3; i = i + 1) {
  if (a > 0.5) { hit = float(i) + 10.0; break; }
  hit = hit - 1.0;
}
```

| | p0 (0.10) | p1 (0.30) | p2 (0.70) | p3 (0.90) |
|---|---|---|---|---|
| before | 10 | 10 | 10 | 10 |
| after | −4 | −4 | 10 | 10 |

Pass 0: `cond = [F,F,T,T]`. `hit := where(cond, 10, hit)` → `[-1,-1,10,10]`; the `break` clears
p2/p3. `hit = hit - 1` runs under `live = [T,T,F,F]` → `[-2,-2,10,10]`. Passes 1 and 2 subtract
one from the two live pixels only. Note what M1 buys: p2 and p3 keep the `10` they left with,
and the `hit = hit - 1` *after* the `if` never touches them — which is exactly the sentence
`LANGUAGE.md` §7.1 today has to warn about ("the assignments before it in that branch land on
every pixel too" becomes false, and the warning goes away).

**`continue`** — R-CONT

```tex
float acc = 0.0;
for (int i = 0; i < 3; i = i + 1) {
  if (a > 0.5) { continue; }
  acc = acc + 1.0;
}
```

| | p0 | p1 | p2 | p3 |
|---|---|---|---|---|
| before | 0 | 0 | 0 | 0 |
| after | 3 | 3 | 0 | 0 |

The bit cleared by `continue` is restored at the next condition evaluation (M3.4), so p2/p3 run
all three passes and accumulate nothing, while p0/p1 accumulate three.

**`return`** — R-RET

```tex
float pick(float a) {
  if (a > 0.5) { return a * 10.0; }
  return a * 100.0;
}
```

| | p0 | p1 | p2 | p3 |
|---|---|---|---|---|
| before | 1 | 3 | 7 | 9 |
| after | 10 | 30 | 7 | 9 |

Before, *every* pixel takes the `then` formula using its own `a` — which is why `before` looks
plausible and is wrong. After, the first `return` records `a*10` for p2/p3 and clears them; the
second records `a*100` for the pixels still live.

**per-pixel `for` bound** — R-BOUND

```tex
float n = a * 10.0;   float c = 0.0;
for (int i = 0; float(i) < n; i = i + 1) { c = c + 1.0; }
```

| | p0 | p1 | p2 | p3 |
|---|---|---|---|---|
| before | 9 | 9 | 9 | 9 |
| after | 1 | 3 | 7 | 9 |

Nine passes run either way — the loop runs while **any** pixel is live, which is unchanged and
is what keeps the frame in one kernel — but from pass 1 onward `c = c + 1` is masked, so each
pixel's count is its own bound. **The pass count is still the region's maximum; only the values
stop being.** That distinction is the whole of §4.

**per-pixel `while` bound** — R-WBOUND

```tex
float x = a; float c = 0.0;
while (x < 0.8) { x = x + 0.25; c = c + 1.0; }
```

| | p0 | p1 | p2 | p3 |
|---|---|---|---|---|
| before | 3 | 3 | 3 | 3 |
| after | 3 | 2 | 1 | 0 |

p3 (`a = 0.90`) never enters: `live` is empty for it from the first condition, so `x` and `c`
keep their initial values. A `while` whose condition is false on arrival for a pixel must
execute **zero** body statements for that pixel — under `0.23` it executes three.

---

## 2. What changes for programs that cook today — measured

This is the part that makes it a version bump, so it is a census and not an estimate.

**The detector.** A program's output moves under `0.25` **iff** it declares `0.25` or later
*and* contains a per-pixel transfer or a per-pixel loop bound — which is exactly what `W7007`
reports. `W7007` is `tex_api._ControlFlowLint`, the flow-sensitive, user-function-aware
analysis the region-dependence gate already consults; using a second definition of "per-pixel"
here would drift against it. The census also runs an independent **structural upper bound** (a
plain AST walk: any transfer lexically under any `if`; any non-static loop condition) so that a
zero from the flow-sensitive pass cannot pass unchallenged.

**The corpus.** `tests/compat_corpus.py::_corpus_programs` = every `examples/*.tex` plus the
`_ADVERSARIAL` set. **117 + 13 = 130**, matching `compat_corpus_goldens/0.24.json`.

| class | detector | examples (117) | stock `.textool` (5) | corpus goldens (130) | host stock tools (33) |
|---|---|---:|---:|---:|---:|
| **A** transfer under a per-pixel `if` | `W7007`, transfer spelling | **0** | **0** | **0** | **0** |
| **B** per-pixel loop bound | `W7007`+`W7008`, loop spelling | **0** | **0** | **1** | **0** |
| **C** bare transfer in a fn defined in a loop | AST walk | **0** | **0** | **0** | **0** |
| **M5** scatter on a per-pixel path | AST walk | **0** | **0** | **0** | **0** |
| **M6** `@` write inside a user function | AST walk | **0** | **0** | **0** | **0** |
| **M7** `debug_print` | AST walk | **0** | **0** | **0** | **0** |
| *(structural upper bound)* transfer under **any** `if` | AST walk | 5 | 0 | 5 | 1 |
| *(structural upper bound)* **any** non-static loop | AST walk | 23 | 0 | 24 | 0 |
| carries a `//!tex` pragma at all | `tex_api.language_pragma` | 0 | n/a | 1 (`0.23`) | 0 |

**The headline: zero shipped programs move.** Not one of the 117 examples, not one of the five
`stock/*.textool` manifests, not one of the 33 stock tools the embedding host publishes
(read-only, linted with TEX's own advisory API).

**Why the upper bound is larger, checked by reading each one.** The 5 examples with a transfer
under some `if` are `break_search.tex` (`if (sx >= int(iw)) { break; }` — loop counter vs
image width, uniform), `fast_defocus.tex` (`norm_r = sqrt(float(i)/N)` against a blade
polygon — uniform in the loop index and parameters), `fix_pixels.tex`
(`if (dx == 0 && dy == 0) { continue; }` — loop counters, and the example's own comment says
so), and `recursive_fractal.tex` / `recursive_subdivision.tex` (`if (depth <= 0) { return n; }`
— a uniform recursion depth). All 23 non-static loops are bounded by parameters, `iw`/`ih` or
loop counters. The flow-sensitive zero is therefore the right answer, and it is the *reason*
the release is cheap: **v0.35.1 already paid this bill**, rewriting six examples onto the
flag-and-uniform-bound pattern §7.1 recommends. Nothing is owed a second time.

**The one class-B corpus golden does not move either, and that is a problem, not a relief.**
`adv_while_loop` cooks at a 1×1 grid (§0), where the two rule sets agree by construction. So
**no frozen golden in the archive changes**, and `0.23.json` and `0.24.json` stay green without
the pragma gate having to do any work for them. §6 says what freeze #3 must do about that.

**What a ComfyUI user sees: nothing.** No shipped program carries a pragma, so no shipped
program changes rules; the default path adds one attribute compare per cook
(`Program.language` against `MASKED_FLOW_SINCE`) and emits byte-identical codegen source for a
program without a `0.25` pragma. Invariant 7 is met by construction rather than by argument,
and §5's T-row makes the byte-identity checkable.

**The derivations.** Advisory census: `tex_api.control_flow_advisories(src, {})` over
`examples/*.tex` + `stock/*.textool["code"]` + `compat_corpus._ADVERSARIAL`, splitting `W7007`
by its message opener (`` This `break` ``/`` continue ``/`` return `` vs `` This `for` ``/`` while ``).
Structural bound: `tex_compiler.parser.Parser` + a recursive walk over `then_body`/`else_body`
and `ast_nodes.try_extract_static_range`. Host tools: the same lint over the 33
published tool manifests' program text. Degenerate-grid finding: `tests/test_integration._prepare_example`
+ `Interpreter.execute`, reading `OUT.shape`. Scripts and raw output are in this ask's worklog
directory.

---

## 3. The bare `break` in a function — refuse it, `E3015`

**Decision: it becomes a compile-time error, in the `E301x` function-definition family, with
its own code `E3015`.**

*Why an error rather than a semantics.* There is no defensible masked meaning for it. The two
tiers disagree today because the construct has two readings and the language never chose:
lexical (leave the loop the `break` is written inside, which is codegen's) or dynamic (leave
the loop the *call* sits in, which is the interpreter's). Choosing either would silently change
the other tier's programs and would hand `0.25` a second, unrelated breaking change to
document. Refusing it makes the program unrepresentable, which closes the invariant-2 divergence
by removing the input rather than by agreeing on an answer — the only fix that cannot rot.

*Why a new code and not `E3002`.* The one-line fix is to reset `_loop_depth` while checking a
function body, after which the existing `E3002` fires. But `E3002`'s text is *"'break' statement
outside of a loop."*, and a reader who wrote this program can see a loop wrapped around their
function; the message would be true only under a scoping rule the reader does not yet know. A
distinct code also lets the release note, and a host's migration scan, grep for exactly this
breaking change. `E3002`'s existing row in `tests/test_simp6_error_code_rows.py` is untouched.
`E3015` is free: the family runs `E3010` (duplicate function), `E3011` (reserved builtin name),
`E3012` (`return` outside a function), `E3013`, `E3014` (function inside a function).

*The wording.* Message: *"'break' cannot leave a loop that is outside this function."* Hint:
*"A function body is its own loop scope — the loop around the function's definition is not
`break`'s to leave. Return a value the caller tests, and put the `break` in the caller's
loop."*

*The implementation shape.* `type_checker._check_function_def` saves `_loop_depth`, sets it to
`0` and sets an `_in_function_body` flag for the body; `_check_break_continue` raises `E3015`
when `_loop_depth <= 0 and _in_function_body`, `E3002` otherwise. Both messages stay true.
`codegen._emit_function_def` separately saves and restores `_use_native_flow_control` **and**
`_scalar_loop` around the nested `def` — the same mechanism, still needed for `return`, which
stays legal.

*Reach: zero.* Class C is 0 in all four populations (§2). This is a breaking change that
breaks nothing shipped, which is the cheapest moment it will ever have.

*It does not wait for `0.25`.* The refusal is unconditional — it is not gated on the pragma —
because the construct has no correct behaviour at any language level and leaving it legal for
`0.23` programs preserves a silent invariant-2 violation. That makes it a **breaking change on
the `0.24` surface as well**, and it is therefore named in the release note in its own right.

---

## 4. The sunset — `region_dependent`'s loop clause

`tex_roi.region_dependent` declines to split a cook whose answer depends on the region. It has
four clauses: **(a)** a loop whose condition can differ per pixel, **(b)** the same on the batch
axis, **(c)** a string chosen per pixel by an `if` or a `?:`, **(d)** a per-pixel value cast
straight to a string. **(a)** and **(b)** sunset at `MASKED_FLOW_SINCE = (0, 25)`; **(c)** and
**(d)** never do, and the code says so by returning `True` for them *before* the version
comparison is reached.

**The sunset keys on the engine's capability, and it already does — confirmed by reading, not
assumed.** `tex_roi._language_tuple(program, code)` returns

```text
    min( what the program asks for , tex_api.LANGUAGE_VERSION )
```

and the clause is `bool(loops) and _language_tuple(...) < MASKED_FLOW_SINCE`. A `//!tex X.Y`
pragma is a **request**: `W7004` advises on a too-new pragma and does not block the compile
(`tex_tool` treats a `.textool`'s `tex_language` pin as advisory too), so a program declaring
`0.25` on a `0.24` engine still runs the `0.23` rules — and the `min` is exactly the rule set it
is cooked under. Keying on the pragma alone would have declared such a program masked while the
engine still ran it unmasked: **a split whose strips disagree with the whole frame, served as a
correct picture.** The v0.36.0 review that approved the pragma spelling and was corrected before
it shipped is the recorded lesson; this design changes nothing about that mechanism and exists
partly to say so in writing.

**No edit to the gate is owed by this release.** Moving `LANGUAGE_VERSION` to `"0.25"` makes
the sunset apply by itself.

**What the gate declines, before and after.**

| program | today (`LANGUAGE_VERSION` `0.24`) | after (`0.25`) |
|---|---|---|
| per-pixel loop bound, **no pragma** | declined — `min((0,0),(0,24)) = (0,0) < (0,25)` | **still declined** — `min` is `(0,0)` |
| per-pixel loop bound, `//!tex 0.23`/`0.24` | declined | **still declined** |
| per-pixel loop bound, `//!tex 0.25` or later | declined — `min` is `(0,24)` | **splits again** |
| string chosen per pixel (`if` or `?:`), any pragma | declined | **still declined** (clause c) |
| per-pixel value cast to string, any pragma | declined | **still declined** (clause d) |
| both a per-pixel loop bound and a string cast, `//!tex 0.25` | declined | **still declined** — (d) returns before the version test |

The five consumers that act on the verdict are `tex_tiling._tile_plan` and
`tex_tiling._halo_tile_plan` (both asking it **last**, after the pressure test, so an unpressured
cook never reaches it), `tex_roi._walk` (the ROI plan), `tex_roi.batch_sliceable`, and
`tex_engine._oom_retry`. None moves. *(The tracker's `TRK-25` row cites the two tile plans at
`tex_engine.py`; they moved to `tex_tiling.py` — see the hand-back's pointer table.)*

**Why the sunset is sound.** Under §1's rules a pixel's value depends only on that pixel's own
condition history. The region still sets the *pass count* (M3 runs while any pixel is live), but
from pass 1 onward every write is masked, so the extra passes a large region forces are no-ops
for the pixels that have left. A strip's answer therefore equals the whole frame's, which is the
premise the sunset was written on.

**The residue the sunset does not remove, named rather than hidden.** Two things stay
region-dependent for a `0.25` program, neither of them a pixel value:

1. **E6010.** The 1024-pass cap fires when a live pixel still needs pass 1024. A whole-frame
   cook containing one such pixel raises; an ROI window that excludes it does not. Every pixel
   the window *does* return is correct, and a full cook still fails, so this is a change in
   *which* cooks fail, not in what a successful cook returns. **Recommendation: accept it and
   document it**, rather than keep clause (a) alive for it — keeping the gate on for an error
   condition would decline the whole class the sunset exists to release.
2. **Pass count, hence time and peak memory.** A strip of cheap pixels runs fewer passes than
   the frame. That is the win, not a defect.

**The executors are still dumb.** `tex_memory.run_tiled` / `run_roi` / `run_batch_strips` are
untouched by the v0.36.0 gate by design ("the gate is in the planners"), and this release does
not change that. A host that drives an executor itself is exactly the caller that asks
`region_dependent` / `batch_sliceable` first.

---

## 5. Both tiers, bit-exactly

**Verdict: both tiers can express the rule.** Neither is asked to do anything it does not
already do — masking adds exact selections (`torch.where`) and boolean algebra over a mask that
broadcasts through the same pair-broadcast helper both tiers share. There is no approximation
anywhere in §1, so invariant 2's `tol=1e-5` is not being spent; the answer should be bitwise.
Two preconditions and six divergence sites, each named so an implementer can test it rather than
hope.

**Interpreter (the oracle; lands first).** A second statement-dispatch table bound per cook only
for a flagged `0.25` program and restored in `finally` — the shape `_cancel` already uses — so a
`0.23`/`0.24` cook runs today's code path untouched. `_exec_spatial_if` keeps its snapshot/merge
and gains live bits; `_Break`/`_Continue`/`_ReturnSignal` are caught at the branch boundary and
turned into mask edits instead of being allowed to unwind; `_loop_cond_true`'s
`(cond > 0.5).any().item()` becomes `live &= (cond > 0.5)` with the loop exiting when
`live.any()` is false; the assignment paths apply `where(live, new, old)` once `live` is not
`True`.

**Codegen.** The mirror at emit time. Live bits are tri-state locals (`None` / `True` / a bool
tensor) so that a region whose mask is statically `True` emits the same source it emits today.
`_stmt_break`/`_stmt_continue`/`_emit_return_stmt` stay native on the 0-dim path and become mask
edits on a spatial one; statements after a transfer in the same block guard on the mask.

**Precondition 1 — the scalar-loop path must decline.** `codegen._setup_scalar_loop` /
`_is_scalar_body` run a loop body in Python scalars. It cannot hold a per-pixel mask and must
not try: a flagged region takes the tensor path. This is an **over-decline** and costs
performance on `0.25` programs only; `0.23`/`0.24` programs keep the scalar path byte-identical.

**Precondition 2 — `_emit_function_def` must scope its flow flags.** It saves
`self._local_vars` and `self._in_user_function` today and saves **neither**
`_use_native_flow_control` **nor** `_scalar_loop`. §3's `E3015` removes the `break` spelling of
the bug; `return` still needs the scoping.

**Where they are most likely to diverge.**

1. **The mask predicate's spelling.** `interpreter._exec_spatial_if` computes
   `(cond > 0.5) if cond.is_floating_point() else cond.bool()`; codegen emits `(cond > 0.5)`
   unconditionally. For an int condition these disagree on any non-zero value below `1`
   — and on NaN they disagree in the float case too (`NaN > 0.5` is `False`;
   `NaN.bool()` is `True`). A probe at this head could not reach the `.bool()` branch with a
   per-pixel condition (conditions arrive as floats), so this is a **latent** hazard, not a live
   bug. **The fix is one shared helper, imported by both tiers**, and the mask must be derived
   through it in every site. This is the single highest-risk line in the release.
2. **The loop-exit test.** Interpreter `live.any()` vs codegen's emitted
   `(cond > 0.5).sum().item() == 0`. An exit one pass early or late moves no *value* (masked
   writes are no-ops) but changes whether a masked **scatter** or a `debug_print` in that pass
   runs. The two tests must be the same expression on the same tensor.
3. **The empty-call skip (M4).** Both tiers must skip a call with no live pixel, or a scatter or
   `@` write inside it lands on one tier and not the other. The *return value* cannot see the
   difference, which is what makes this one easy to miss.
4. **Scatter compaction order (M5).** Both tiers must select live sources in the same order
   (row-major `nonzero`), or an unspecified collision resolves differently on each tier. "Unspecified"
   is a promise to the author, not a licence for the tiers to differ.
5. **Write-vs-merge order inside a per-pixel `if`.** Both must snapshot before the branch and
   merge after, in the same sequence, for a variable written in both arms *and* left by a
   transfer in one of them.
6. **A shared error is invisible to parity.** Two tiers agreeing on the wrong answer is exactly
   what §0's table shows today. Parity testing alone cannot catch it, so the oracle below is
   not optional.

**The oracle that settles it.** The `0.23` interpreter **in scalar mode already computes §1's
answer**: seed one pixel's coordinates into the scalar-builtin defaults, slice the bindings to
that pixel, use a fresh `Interpreter` per pixel. Every condition is then 0-dim, the
short-circuit path runs plain sequential control flow, and the result is by definition "the
program run on that pixel alone". A `0.25` cook must equal a per-pixel sweep of that oracle to
`1e-5` — and, being an independent implementation, it is the only check that catches a defect
both tiers share.

---

## 6. Compat freeze #3

`tests/compat_corpus_goldens/` is append-only: `freeze(version)` may only **add** a version and
refuses to rewrite one (`regen()` was deleted in v0.34 for exactly this reason). The archive
holds `0.23.json` (129) and `0.24.json` (130). Freeze #3 mints `0.25.json`.

**What the test actually checks, and the trap in it.** `test_v023_phase1` runs
`compat_corpus.compute_all()` — **current** behaviour — against **every** archived version, and
separately asserts that the **newest** archive covers every program the corpus runs. So:

* `0.23.json` and `0.24.json` must stay green *under the new engine*. They do, because no frozen
  program carries a `0.25` pragma (§2), so none changes rules. **That is the pragma gate's
  proof, and it is already a gate** — no new machinery is needed to enforce it.
* Adding programs without freezing makes the coverage assertion red, so the new rows and
  `freeze("0.25")` **must land in one commit**.

**What freeze #3 must cover, or it is decoration.** Copying the existing 130 programs forward
mints 130 hashes that are bit-identical to `0.24.json`'s and prove nothing about `0.25`. The new
rows are the point:

1. **Five paired `_ADVERSARIAL` rows**, one per construct — `adv025_break`, `adv025_continue`,
   `adv025_return`, `adv025_for_bound`, `adv025_while_bound` — each the `//!tex 0.25` spelling of
   §1's worked examples.
2. **Five `0.23` twins of the same sources without the pragma**, so the archive records both
   answers for the same text and a future engine cannot quietly converge them.
3. **Each new row must read an `@` binding.** This breaks the `_ADVERSARIAL` set's
   "builtin coords only, no `@inputs`" convention **on purpose**, and the comment on the new rows
   must say why: measured at this head, a program with no `@` wire cooks at a 1×1 grid, where
   masked and unmasked agree by construction — which is how `adv_while_loop` has been sitting in
   the corpus since `0.23` proving nothing about the ANY-pixel rule. A probe confirms the
   difference is visible with a wire: `@A.r`-bounded, `B=2,H=16,W=16`, current engine gives
   `n = 8` on every pixel while the masked reading gives `n ∈ {1,2,7,8}` — every pixel moves a
   whole 8-bit level.
4. **One shipped example**, `examples/per_pixel_control_flow.tex`, with its
   `_EXAMPLE_CATEGORIES` row. It is a corpus program too, so it is frozen with the rest.

**Procedure**, in the bump commit and no earlier:

```text
    cd tests && python -X utf8 -c "import texboot, compat_corpus; compat_corpus.freeze()"
```

It defaults to `tex_api.LANGUAGE_VERSION`, so it must run **after** the version string moves and
**after** the new rows exist; it refuses if `0.25.json` is already there. Expected: `0.25.json`
with **141** hashes (130 + 10 new adversarial rows + 1 new example), `0.23.json` and `0.24.json`
byte-unchanged and still green.

---

## 7. The advisories

| code | today | after `0.25` |
|---|---|---|
| `W7004` | a pragma newer than the engine: *"newer features may not compile"* | **meaning changes** — a too-new pragma now also means *the program will be cooked under older rules and may compute differently*. The text gains that clause. It stays advisory and still does not block, which is what §4's `min` depends on. |
| `W7006` | a gather inside a per-pixel `if`/`?:` costs the same on every pixel | **unchanged.** M2 keeps `0.23`'s both-branches model; masking hides no work. |
| `W7007` | control flow that acts on every pixel | **becomes conditional.** For a program on `0.25` rules the warning is false and must not fire; for a program below `0.25` it is as true as it is today. It becomes "…unless this program declares `//!tex 0.25`". |
| `W7008` | the shapes the engine declines to split | **becomes conditional in the same way**, and for the same reason the gate's clause (a) sunsets — but only its loop half. Its string halves (clauses c and d) keep firing at every language level. |

**The testable consequence**, and the reason this table is worth writing down: once `W7007` is
conditional, *"no `W7007` ⇒ the pragma moves no pixel"* becomes a property a test can assert
over the whole corpus, which is a far stronger statement than §2's census re-run by hand each
release.

---

## 8. The staged plan

Sizes are honest: this is the largest language change since `0.23`. `L4` and `L5` are each a
genuine L and neither can be split further without splitting the oracle from the thing it
proves.

| lane | content | size | depends on | acceptance test |
|---|---|---|---|---|
| **L1** | The pragma moves into `tex_compiler`; `Parser.parse` sets `Program.language`; `tex_api.language_pragma` delegates. Nothing reads the field yet. | S | — | A header pragma round-trips onto `Program.language` through **every** source→AST path (`tex_cache.compile_tex`, fusion's per-stage parse, the corpus harness); a buried pragma still reads `None`. Emitted codegen source byte-identical over the corpus. |
| **L2** | §3: `E3015`, the type-checker scope, `_emit_function_def` scoping its two flow flags. | S | — | §0's `TRK-28` program is refused with `E3015` on both tiers at compile time; `E3002`'s own row unchanged; the mutation harness reds when the `_loop_depth` reset is removed. |
| **L3** | `flow_plan(program)`: the shared structural walk that flags per-pixel loops, transfer-bearing regions, scatter/probe/binding-write sites under a per-pixel `if`, and the sync points. Fusion refuses a mixed-language chain at the per-stage parse. Nothing masks yet. | M | L1 | An empty plan for every corpus program; a non-empty plan for each of §0's five repros, naming the right sites. A mixed-language chain raises `FusionError` and the region is left unfused; a same-language chain fused == unfused. |
| **L4** | The interpreter's `0.25` rules (M1–M7) + the per-pixel scalar oracle harness. | **L** | L3 | §1's five worked tables reproduce exactly on the interpreter; a `0.25` cook equals the per-pixel oracle sweep (`1e-5`) over the control-flow atoms *and* `test_v017_phase1._gen_program`'s generated programs; no-pragma and `//!tex 0.24` still give §0's "before" column; nested loops, `return` inside a loop, a terminating per-pixel recursion, and a never-ending pixel raising `E6010`. |
| **L5** | Codegen emission, the two preconditions in §5, the shared mask helper, fuzzer atoms. | **L** | L4 | Interp == codegen bitwise on every L4 row; **the emitted `_tex_src` digest for every corpus program without a pragma is unchanged from the base sha** (the invariant-7 proof); the differential fuzzer green with control-flow atoms enabled; §0's `TRK-28` neighbour — a nested general loop inside a static one with a transfer — terminates and matches. |
| **L6** | Satellite tiers: graph capture declines a `0.25` program with sync points; `precision="auto"` declines a `0.25` per-pixel `for`; ROI/tiling gain a *characterization* test that a `0.25` split equals the whole frame; lazy analysis proven unchanged (it is syntactic and over-approximates under both rule sets). | M | L5 | `_capturable` False on a flagged program with syncs and still True for a transfer-free `0.25` program; the ROI/strip/batch-strip triple equals the whole frame for `0.25` and still differs for `0.23` (the characterization); the never-sever lazy row green. |
| **L7** | The bump: `LANGUAGE_VERSION` → `"0.25"` and its ten satellites; `LANGUAGE.md` §7.1 rewritten around the pragma; `DEVELOPMENT.md`; the JS help block; `W7004`/`W7007`/`W7008` wording; the new example; the eleven new corpus rows; `freeze("0.25")` **last**. | S | L6 | `test_v037_satellites` green at `0.25`; `0.23.json`/`0.24.json` byte-unchanged and still matching; `0.25.json` mints 141; the five `0.25` rows and their five no-pragma twins hash **differently**. |

**Parallelism.** `L1` and `L2` are disjoint and run together. Everything from `L3` is a chain —
`L5` needs `L4`'s oracle to have something to match, and `L7`'s freeze must be the last thing
that happens, because a freeze taken before the surface is final is a golden nobody can trust.

**A note on `L5`'s acceptance test.** The emitted-source digest comparison is the cheapest
possible proof of invariant 7 for this release and it must be taken against the **base sha**,
not against `L4`'s head, or `L4`'s own interpreter-side changes launder into the baseline.

---

## 9. Version, and what a vendoring host owes

`LANGUAGE_VERSION` `0.24` → **`0.25`**. The rule (`plane-bindings.md` §9): *a checklist-§2
function addition does not bump the language version; a grammar-visible surface does.* The
grammar is untouched here — no new token, no new statement form — but the **semantics of an
existing grammar** change, which is the same class of break for a host that pins by version and
strictly larger than a lexer change for one that hashes programs. It bumps.

The string stays **strictly numeric**: `tex_api._ver_tuple` int-parses each component, so
`"0.25-masked"` would compare as `(0, 0)` and silently break §4's `min`, the corpus archive's
sort key, and `W7004`.

**Four things the release note must name before the tag**, per `brief-conventions.md`:

1. **Reserved names:** *none*. This release reserves nothing. §3's `E3015` is a new refusal but
   not a new name.
2. **`LANGUAGE_VERSION` moves** `0.24` → `0.25`, and with it the ten satellites in nine files:
   `tex_api.LANGUAGE_VERSION`, the five `stock/*.textool` `tex_language` fields, the `_LANG`
   literal in `tools/gen_stock_tools.py` (edit the generator or the next regeneration reverts the
   manifests), the JS publish-manifest literal in `js/tex_extension.js`, and `LANGUAGE.md`'s two
   prose copies. `tests/test_v037_satellites.py` pins eight of the ten.
3. **Defaults that move:** *none*. A program without a `0.25` pragma computes exactly what it
   computes today, and the region-dependence gate declines exactly what it declines today.
4. **New module filenames: TWO, and this entry predicted zero.** Both ship, so both arrive in a
   vendoring host's tree on a directory copy:

   | module | stage | why |
   |---|---|---|
   | `tex_runtime/masked_flow.py` | L4 | the masking rules, which wanted a leaf rather than edits to the interpreter's existing flow control |
   | `tex_runtime/codegen_masked.py` | L5 | the masked-emission leaf, kept out of the emitter to honour the standing "stop splitting here" verdict on that module |

   Neither adds a scanner site to the published archive and no host imports either, but a
   filename is a re-pin delta and is named before the tag regardless — a host is told about a new
   file whether or not it uses it.

   *Corrected twice, by L4 and then by L5, each discovering the prediction was wrong by shipping
   a module.* The design reasoned about the work as edits to existing files, and twice it turned
   out to want a leaf of its own. **The lesson is not to predict harder.** A design's re-pin delta
   is a forecast; the ledger's is a measurement, and only the second is handed to a host. Derive
   that section from `git diff --name-status <pin> HEAD` before the tag.

Plus two breaking changes that are **not** gated on the pragma and must be named as such:
**`E3015`** (§3), and **`W7007`/`W7008` becoming conditional** (§7), which a host that asserts on
advisory output will see.

**The re-pin delta for the embedding host.** Measured read-only against that host's tree: 33
stock tools under `assets/stock_tools/` each declare `"tex_language": "0.23"`, and 35 on-disk
manifests carry the field in total (the 33 plus two published-store snapshots). That host's tool
identity is `manifest_hash` = SHA-256 over the whole manifest with sorted keys, and
`tex_language` is a manifest key read live from the build — so adopting this release **re-signs
every one of them**, and the failure direction is a demoted tool rather than a stale grant. The
host is holding its pin so it pays that once; the release note states the language move in the
entry, not only in the diff. Its stock library is **clean of every class in §2** (measured, §2's
last column), so no pixel of it moves.

**Where the pencil puts it.** `docs/roadmap.md` §9 currently pencils "Masked flow" at
**v0.40.0** (it moved down twice, for Planes and then for Linear light). The author's decision to
land `0.25` in this run moves it up; which release number it takes is the author's call and the
pencil is re-cut at each release anyway. Nothing in this design depends on the number.

---

## 10. Declined, and what reopens each

* **Masking without a pragma.** Moves the pixels of every existing program with a per-pixel
  transfer. Never reopens — it is the rule that makes §2's census read zero.
* **Refusing per-pixel transfers in `0.23` programs.** The same objection, wearing a refusal
  instead of a value change. Never reopens.
* **Desugaring to flag variables and guard `if`s** instead of a live mask. Every masked pass
  would clone the branch snapshot and every guard on a 0-dim flag would `.item()`; it cannot
  express M4's empty-call skip, M5 or M7, and it would give both tiers one untestable shared
  lowering — the exact failure §5's divergence-site 6 warns about. Reopens only if `L5` cannot
  reach oracle parity.
* **A masked scalar-loop path in codegen.** Reopens on a measured cost for a real `0.25`
  program; §5 takes the over-decline deliberately.
* **Keeping clause (a) of the region gate alive for `E6010`.** §4's residue 1. Reopens if a host
  reports a windowed recook succeeding where the frame failed and calls that wrong.
* **Reusing `E3002` for §3.** Reopens if the author prefers one code with two hints; the
  implementation difference is two lines.
* **A per-pixel representation for strings.** M7 keeps the majority vote verbatim, which is why
  clauses (c) and (d) of the gate never sunset. Reopens only with a string tensor type, which is
  not on the roadmap.

## 11. What this document does not decide

* **Which release number carries it** (§9) — the author's, and the pencil's.
* **Whether `TRK-3`/`TRK-4` close on landing.** Their documenting halves landed in v0.35.1; this
  is the masking half. Declaring them discharged is the author's word.
* **The performance envelope.** §8 names no timing gate on purpose: v0.38.0 moved timing from a
  gate to a sitting, and a `0.25` program has no baseline to be neutral against because no such
  program exists yet. What `L5` must prove is that a program *without* the pragma emits
  byte-identical source — a counts-and-bytes claim, not a timing one.
* **`0.25` under fusion beyond "refuse a mixed-language chain"** (`L3`). A fused chain of
  uniformly-`0.25` stages is in scope; anything finer is not.
