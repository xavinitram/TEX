# The TEX language

This is the reference for the **TEX language** — its grammar, types, reserved
words, and (the reason this file exists) its **compatibility policy**. For a gentle
tutorial start with [learn_tex_in_5_minutes.md](learn_tex_in_5_minutes.md); for the
full function catalogue see [Function-Reference.md](Function-Reference.md); for the
diagnostic codes see the generated `wiki/Error-Codes.md`.

TEX is a small, statically-typed, per-pixel expression language. One program is
evaluated once per output pixel (and once per batch frame); the same source runs on
the CPU interpreter and the GPU codegen backend and must produce the same result.

---

## 1. Language version & compatibility

The **language** is versioned separately from the package. `tex_api.LANGUAGE_VERSION`
(currently **`0.25`**) names the grammar + semantics this engine implements; the
package `__version__` tracks the release. They move independently — a release that
only fixes a bug or refactors internals does not bump the language version.

**A program may declare the language level it targets** with a leading pragma on its
own comment line:

```tex
//!tex 0.25
@OUT = vec4(@A.rgb * 1.2, 1.0);
```

The pragma is an ordinary comment to the compiler (it never becomes a token). It is a
*mechanism, not a promise*: `tex_api.check()` emits the advisory **W7004** when a
program targets a language *newer* than the engine implements (a feature it may not
understand), and stays silent for an equal or older target. `tex_api.language_pragma(source)`
returns the declared `"X.Y"` (or `None`).

**Stability contract.** A program that compiled and ran on version *N* keeps computing
the **same pixels** on version *N+1*. This is enforced, not merely intended: the frozen
**compat corpus** (`tests/compat_corpus.py`, goldens in `tests/compat_corpus_goldens/`)
runs every bundled example plus a set of adversarial grammar programs on the CPU
interpreter and hashes their quantized outputs against committed goldens. A drift fails
the suite.

Since v0.34 (R2-archive) the goldens are an **append-only archive**: one file per frozen
language version, and current behavior is checked against *every* frozen version, not just
the latest. `compat_corpus.freeze(version)` may only ADD a version and refuses to rewrite
one — so "old goldens are immutable" is machinery rather than a convention. A frozen
program whose pixels genuinely must move requires deleting that archive file in a commit
that argues the change.

The archive holds `0.23.json` (129 programs), `0.24.json` (130: the same 129, unchanged,
plus the first plane program — §5.3), and, since language `0.25`, `0.25.json` (141: the
same 130, unchanged, plus ten new adversarial rows and one new example exercising masked
per-pixel control flow under `//!tex 0.25` — §7.1). Every current program is checked
against all three.

New grammar is added **additively** (v0.23 added the optional parameter-metadata block,
below) so old programs keep parsing. A genuinely breaking change is called out in the
CHANGELOG with a migration and, where possible, an error that names the fix.

Two consequences worth stating outright:

* **Fingerprints are not a stable identity.** A host must never persist a compile
  fingerprint across versions (see DEVELOPMENT.md's API-stability tiers). The language
  version and the compat corpus are the durable contract; the fingerprint is an internal
  cache key that is free to change.
* **fp32 is the wire and compute type.** Storage dtypes (uint8/half/…) convert to fp32
  at ingestion; coordinate and timeline builtins are always fp32.

---

## 2. Lexical structure

* **Comments:** `// line` and `/* block */`. There is no `#` comment.
* **Number literals:** integer (`10`, `0xFF`) and float (`1.5`, `.5`, `2e-3`).
* **String literals:** `"double quoted"`, with `\\ \" \n \t \r` escapes.
* **Identifiers:** `[A-Za-z_][A-Za-z0-9_]*`.
* **Bindings** carry a sigil: `@name` (a wire — an image/mask/latent input or output)
  and `$name` (a parameter — a scalar/vector/string widget value). Either may carry a
  one-token type prefix: `f@x`, `img@src`, `f$gain`, `v3$tint`. A wire name may be
  followed by exactly one dotted segment, `@beauty.diffuse` — a plane on a PLANES wire
  (§5.3), or the swizzle it always was on any other wire.
* Statements end with `;`. Blocks are `{ … }`.

---

## 3. Types

| Type | Notes |
|------|-------|
| `float`, `int` | scalars (int promotes to float freely) |
| `vec2`, `vec3`, `vec4` | fixed-width float vectors; `.xyzw` / `.rgba` swizzles |
| `mat3`, `mat4` | matrices; `m * v` transforms a vector |
| `string` | a separate domain — no numeric promotion to/from it |
| `float[N]` | fixed-size arrays of any element type |
| `IMAGE`/`MASK`/`LATENT` | wire (`@`) binding types at the host boundary |
| `PLANES` | a wire-only binding type (engine hosts): one `@` wire carrying named planes, each read as `@wire.plane` (§5.3); never an expression type |

Swizzles read components by name (`c.x`, `c.rgb`, `p.xy`); a component set must use one
naming family. Reverse/arbitrary reorders are not all supported — read components you
need and rebuild.

## 4. Type-promotion rules

* `int` → `float` wherever a float is expected.
* A scalar combined with a vector **broadcasts** to every component
  (`vec3 * 2.0`, `1.0 - vec4`).
* Two vectors must share width; the result keeps that width.
* `mat3 * vec3` / `mat4 * vec4` transform; `mat * mat` composes.
* `string` never promotes to or from a numeric type; string operations stay in the
  string domain.
* Coordinate/timeline builtins are forced fp32 so their own value is exact.

---

## 5. Bindings & parameters

```tex
@OUT = vec4(@A.rgb * $strength, 1.0);   // @A wire in, @OUT wire out, $strength param
f$strength = 0.5;                        // a float parameter with a default
```

**Parameter UI metadata (v0.23, LANG-1).** A declaration may carry an optional,
literal-only metadata block used by the frontend to build a widget (and, later, tool
manifests). It is *ignored by the type checker* — a nonsensical range is not a compile
error:

```tex
f$strength = 0.5 [min: 0, max: 2, step: 0.05, label: "Strength"];
i$count [min: 1, max: 16];              // metadata without a default is allowed
```

Values are literals only (a number, optionally negated, or a string); an expression or a
binding reference inside the block is a syntax error.

### 5.1 The animated-parameter guarantee (v0.31, ANIM-1) — normative

> A `$param` value is a **cook-time binding**. Changing it never recompiles, never
> re-optimizes, never re-emits codegen, never recaptures a CUDA graph, and never changes the
> program's cache identity. Sweeping a parameter across *N* values costs *N* cooks and
> nothing else.

This is a **guarantee, not an optimization**: a host may build keyframing, timelines, and
slider scrubbing directly on it, and does not need to consult TEX or cache anything of its
own to avoid a recompile. It is normative from language version 0.23 onward and will not be
withdrawn without a language major bump.

What it rests on, so the scope is unambiguous:

- A program's cache identity is `H(code, binding TYPES)`. Parameter **values** are not in it.
- A CUDA graph's capture key holds parameter names, shapes and dtypes — again not values;
  values are re-staged into the captured buffers per replay.
- Emitted codegen reads parameters through the binding environment at call time, not as
  baked constants.

What is **not** covered, and does recompile — each because it changes what the program *is*,
not what it is *given*:

| Change | Recompiles? | Why |
|---|---|---|
| `$strength` 0.2 → 0.9 | no | a value |
| `$tint` `[1,0,0]` → `[0,1,0]` | no | a value (vec params are staged, not baked) |
| `$mode` `"add"` → `"screen"` | no | a value (string params too) |
| a param of a mid-chain stage in a **fused** chain | no | the fused key folds per-stage code + types |
| a promoted param of a `.textool` | no | same guarantee through the tool seam |
| `@A` wired VEC3 → VEC4 | **yes** | a binding's TYPE is part of the identity |
| any edit to the program text | **yes** | it is a different program |

One clarification, because it is the easy thing to get wrong: **a parameter's type comes from
its declaration in the code** (`f$k`, `i$n`, `s$mode`, `v3$tint`), never from the bound value.
So "change a param's type" is not something a host can do at cook time — it is a code edit,
and code edits recompile. The binding-type axis in the fingerprint is about `@` wires, whose
type *is* read from what is connected.

Pinned by `tests/test_v031_anim_contract.py`, which spies on the compiler, the codegen
emitter and the graph capturer across every tier and both devices — and by a negative
control that edits the code and asserts those same spies *do* fire, so the guarantee cannot
pass vacuously. The standing cost is tracked by `benchmarks/param_scrub_bench.py`.

### 5.2 Uniform outputs

An `f@`/`i@` output written **only** from literals, scalar `f$`/`i$` params, and the 0-dim
builtins `iw ih px py fn ic PI TAU E frame fps time` — through arithmetic, comparisons,
ternaries, uniform `if`s and scalar math functions — never reads a pixel. TEX returns it as
one **0-dim** value, computed once per cook, identical across every tier and route that can
produce it: `compile_mode` `none` / `torch_compile` / `auto` / `cuda_graph`, a tiled,
batch-strip or ROI-windowed cook, and a fused chain's terminal stage — exact at
`precision="fp32"`. (A program reading `frame`/`fps`/`time` always cooks on the interpreter,
an existing caching policy unrelated to this guarantee, so for those three there is only ever
one tier to be identical with.)

```tex
@OUT = @A * 0.5;
f@active_x = max($win_x, 0.0);    // a once-per-cook FLOAT, not a per-pixel one
f@active_w = iw * 0.5;
i@active_h = ih - 1;
```

Like any output, this still costs one of the program's `MAX_OUTPUTS` (8) slots, and a
declaring stage must be a fused chain's **terminal** stage: an upstream stage writing
anything besides `@OUT` already refuses to fuse, so a mid-chain declaration is refused
before it can be spliced into the wrong grid.

**Not covered**, and left as ordinary per-pixel or tier-dependent values: a `v2@`–`v4@`
output (broadcasts to the cook's full extent even when every component is uniform);
anything reading `u v ix iy fi` or an image (grid- or batch-shaped by definition); and a
**vec** `$param` read into a scalar output (rank disagrees across tiers today — filed, not
promised here).

TEX gives a uniform output no meaning of its own — it is an ordinary `@` binding that
happens to hold one value. A host may read one as, say, a declared region; TEX neither
validates nor interprets it either way. The reverse direction needs no new syntax: a value
the host already knows about one of its own inputs reaches the program as an ordinary
`$param` — a cook-time binding like any other (§5.1), so sweeping it moves the output's
result identity but never recompiles the program.

Pinned by `tests/test_v035_hygiene.py`, across the interpreter, codegen, every
`compile_mode`, the tiled/batch-strip/ROI assemblers, and a fused chain's terminal stage.

### 5.3 Plane reads (language 0.24)

One `@` wire may carry many **named planes** — an EXR's layers (`diffuse`, `specular`, `Z`,
`N`), a render's AOVs — and a program reads one by name with the dotted form `@wire.plane`:

```tex
vec3 lit = p@beauty.diffuse + @beauty.specular * $spec;   // two three-channel planes
float depth = @beauty.Z;                                  // a one-channel plane is a float
@OUT = vec4(mix(lit, $fog, smoothstep($near, $far, depth)), 1.0);
```

* **One segment names the plane; what follows is an ordinary swizzle.** `@beauty.diffuse.rgb`
  is the plane `diffuse` swizzled `.rgb`. The lexer takes exactly one dotted segment after a
  wire name. On a wire that is not PLANES the segment is the swizzle it always was — `@image.b`
  still means the blue channel — so no `0.23` program changes meaning; the whole `0.23` archive
  is still checked, unchanged (§1).
* **The `p@` prefix declares the wire PLANES** (`p@beauty.diffuse`), the same one-token type
  prefix as `f@` or `img@` (§2). A host that binds the wire as a plane set needs no prefix. The
  wire itself is not a value — `@beauty` alone cannot appear in an expression — and a plane
  read is typed by the plane it names: a three-channel plane is a `vec3`, a four-channel one a
  `vec4`, and a **one-channel plane is a `float`**, handed over in the `[B,H,W]` shape a MASK
  uses, so it composes wherever a mask does.
* **The collision rule.** A PLANES wire may not declare a plane named after one of the 38
  lowercase channel and swizzle names (`r g b a x y z w`; `rg` … `rgba`, `xyz`, `xyzw`, `bgr`,
  `abgr`): `@src.rgb` could not mean both "the plane called `rgb`" and "the rgb of something".
  The engine refuses the wire at binding time with **E3304** — *plane `rgb` on `@src` collides
  with the swizzle `.rgb` — rename the plane* — whether or not the program reads that plane.
  Never a silent guess.
* **`Z` does not collide.** The rule is case-sensitive and lowercase-only, and the conventional
  EXR data-layer names are uppercase: `Z`, `N`, `RGBA`. `@beauty.Z` is a plane read on day one,
  with no rename and no escape hatch. What collides is a plane spelled `z`, and the error says
  to rename it.
* **Reading a plane the wire does not carry** draws the advisory **W7009** with a did-you-mean
  (`@beauty.diffues` → *Did you mean `@beauty.diffuse`?*), followed by the ordinary E6003
  "not connected" refusal naming the slot, because no cook can satisfy the read. Planes a
  program never mentions are never marshalled: a program reading only `@beauty.N` ingests
  only `N`.

**Not in `0.24`** — planes are read-only and engine-only, and each limit has the gate that
reopens it:

* **Plane writes.** `@OUT.diffuse = …` is a compile error whose hint names the deferral;
  `@OUT` is one whole image. Reopens with a host wire type that can carry a plane set.
* **A ComfyUI wire type.** The ComfyUI node has no PLANES socket. A plane set reaches a program
  through the engine API under the engine egress profile, exactly as ARRAY wires do; under the
  ComfyUI profile the type does not exist and nothing about the node changes. Reopens with a
  host wire that carries one.
* **A fused chain carries whole wires.** A `.textool` input cannot feed one plane, and a fused
  stage cannot export one (the refusal names the export). Reopens on a measured graph where a
  plane edge inside a fused region is the bottleneck.
* **UINT planes** (cryptomatte ids) are not read — the named future customer is cryptomatte.
  **Multipart** and deep EXR are not read — multi-*layer* files are; multipart reopens with a
  fixture that needs it.

---

## 6. Reserved words & built-in variables

**Keywords** (cannot name a variable): `float int vec2 vec3 vec4 string mat3 mat4 if
else for while break continue return const`.

**Built-in variables** (read-only; declaring one is an error or a W7003 shadow advisory):

| Group | Names |
|-------|-------|
| Pixel coords | `ix iy iw ih u v px py ic` |
| Batch | `fi` (frame index), `fn` (frame count) |
| Host time | `frame fps time` — the host playhead (reserved built-in names since v0.22) |
| Constants | `PI TAU E` |

`u`/`v` are pixel-centre coordinates: `u = ix / max(iw-1, 1)`, `v = iy / max(ih-1, 1)` —
`0` at the first pixel, `1` at the last. `px`/`py` are `1/iw`/`1/ih`: one pixel's
width/height as a fraction of the *frame*, not the spacing between neighbouring `u`/`v`
centres — those are `1/(iw-1)` and `1/(ih-1)` apart, so `u + px` is short of a true
one-pixel step by `1/iw` of a pixel. The exact k-pixel step is
`u + k / max(iw - 1.0, 1.0)` (`v`/`py`/`ih` alike), or `fetch(@A, ix + k, iy)` by integer
pixel index.

`frame`, `fps`, and `time` are **hard-reserved**: a program declaring its own
`float time = …;` fails to compile. The `$` parameter namespace is separate — `$time`
(a param) does not collide with the `time` builtin, though `check()` warns (W7003) that
the shared name is easy to confuse. Some words (e.g. `pass`, `stage`) are reserved for
future features and error in block position.

---

## 7. Statements & control flow (grammar summary)

```
program     = statement*
statement   = var_decl | array_decl | param_decl | assignment
            | if_else | for_loop | while_loop | function_def | expr ';'
var_decl    = ['const'] type IDENT ['=' expr] ';'
param_decl  = ('$'|prefix'$') IDENT ['=' expr] [ '[' meta_kv (',' meta_kv)* ']' ] ';'
meta_kv     = IDENT ':' literal
assignment  = ('@'|'$'|IDENT) ['.' swizzle] ('='|'+='|'-='|'*='|'/=') expr ';'
if_else     = 'if' '(' expr ')' block ['else' (block | if_else)]
for_loop    = 'for' '(' [var_decl|expr] ';' expr ';' expr ')' block
while_loop  = 'while' '(' expr ')' block
function_def= type IDENT '(' [param (',' param)*] ')' block   // 'return' expr;
```

Operators, in decreasing precedence: postfix (`.`, `[]`, calls) · unary (`- !`) ·
`* / %` · `+ -` · comparisons · `&& ||` · ternary `?:` · assignment. Every loop is capped
at 1024 iterations; a loop that needs more fails the cook with E6010, so a cook always
terminates.

### 7.1 Uniform and per-pixel conditions

A program runs on every pixel at once, so what a condition does depends on whether its value
can differ from one pixel to the next.

**Uniform** means one value for the whole cook: literals; scalar and string parameters
(`$gain`, `i$count`, `s$mode`); the built-ins `iw ih px py fn ic PI TAU E frame fps time`; the
counter of a loop whose bounds are themselves uniform; and arithmetic, comparisons and scalar
math functions of those.

**Per-pixel** means anything that reads an `@` input, `u v ix iy` or `fi`, or a variable
computed from one, **including a reduction such as `img_min(@A)`**, which holds one value per
frame and is still per-pixel here — and a component of a vector or colour parameter, `$tint.r`,
which like a reduction holds one value for the whole cook and is still per-pixel here (the
parameter itself is bound channel-wise, not as a plain scalar). A variable declared before a
per-pixel `if` and assigned inside it is per-pixel after it.

* A uniform `if` runs only the branch it takes.
* A per-pixel `if` runs **both** branches on every pixel and keeps each pixel's side, so a
  branch costs the same whether a pixel takes it or not: putting a `sample`, a blur or a gather
  loop behind a per-pixel `if` skips nothing. **Unchanged by `//!tex 0.25`** — both branches
  still run either way; only what `break`/`continue`/`return`/writes do inside them changes,
  below.
* Assume `?:` evaluates both operands, at every language level.

**Two rule sets, keyed on the pragma.** What `break`, `continue`, `return` and a per-pixel loop
bound do under a per-pixel `if` depends on whether the program declares `//!tex 0.25` or later
*and* the engine implements it (`min(what the program asks for, what the engine implements) >=
0.25` — a program that asks for `0.25` on an older engine still runs the older rules, and
`W7004` says so). No pragma, or an older one, keeps the `0.23` rules forever: a program that
compiled and ran on an earlier version keeps computing the same pixels.

**Under `0.23`/`0.24` rules** (no pragma, or `//!tex 0.23`/`0.24`):

* `break`, `continue` and `return` under a per-pixel `if` act on **every** pixel. The first time
  the loop or function reaches that `if`, they fire for all pixels whatever the condition says,
  and the assignments before them in that branch land on every pixel too. To stop per pixel,
  keep a flag the body tests, `if (found < 0 && hit) { found = i; }`, and let the loop run a
  uniform bound.
* A `for` or `while` whose condition is per-pixel runs **every** pixel for as many passes as
  the pixel that needs the most, and does not mask the body: a pixel whose own condition is
  already false keeps executing it. Bound the loop uniformly and guard or weight the per-pixel
  work, for example `for (int i = 0; i < $max; i++) { if (i < n) { sum += tap; } }`.
* A per-pixel loop bound also means **the cook is never split**. "As many passes as the pixel
  that needs the most" is counted over the region actually being cooked, so a half-frame strip
  and the whole frame give different answers.

**Under `//!tex 0.25`** (masked per-pixel control flow; `docs/masked-control-flow.md` has the
full rules): each active region — a loop, one pass of a loop, a user-function call — carries a
per-pixel **live** mask. A pixel is live in a region when it has entered that region and has not
left it; a pixel that has left keeps the value it left with. An `if` is **not** a region for
this purpose (a loop pass and a call are); a variable written inside a per-pixel `if` is
selected by that branch's live mask exactly as `0.23` already merges it.

* `break` clears the pixel's bit for the **rest of the loop**; `continue` clears it for the
  rest of *this pass only*. Neither one acts on a pixel that is not live at that `if` — the
  assignments before them in that branch, and everything after the loop, no longer land on
  every pixel.
* `return` **records** its value for the pixels live at that statement and clears their bits for
  the remainder of the function body; a pixel that reaches the end without returning gets the
  same default `0.23` always gave it.
* A per-pixel loop bound still runs to the region's maximum pass count — that part is unchanged,
  because the loop keeps running while *any* pixel is live, which is what keeps the whole frame
  in one kernel — but from the first masked pass onward, each pixel's own value stops updating
  once its own condition goes false. **The pass count is still the region's maximum; only the
  values stop being.** A worked example (TRK-25): a 4-strip cook and the whole frame both take
  the same number of passes, `[4, 4, 3, 3, 2, 2, 1, 1]` read off row-by-row — but that row is the
  *strip's* maximum in each strip, not the per-pixel answer, which under `0.25` is each pixel's
  own: `[4, 4, 3, 3, 2, 2, 1, 0]` (the last pixel's own condition is already false and it never
  runs a fourth pass). This is exactly why the sunset below is sound: the pass *count* a region
  reports can still depend on the region, but the *values* it returns no longer do.
* A **string** written on a per-pixel path, and a per-pixel value cast straight to a string,
  keep `0.23`'s whole-frame behaviour verbatim at every language level — see the next bullet.
* `debug_print` (a probe) records only if its probe pixel is live.

Two things never sunset, at any language level, because neither has a per-pixel
representation: a string chosen per pixel — assigned inside a per-pixel `if`, or picked by a
per-pixel `?:` — is resolved by a majority vote over the region's pixels, and a per-pixel value
cast STRAIGHT to a string — `string(x)`, `str(x)`, or a `format()` call that actually fills a
placeholder — falls back to the MEAN of the region's pixels (`format("%f", x)` is not affected —
`%f` is not a placeholder, so the template comes back unchanged and the value never reaches the
output). Either way a strip's vote or mean differs from the whole frame's, so the engine cooks
such a program as one whole region — no window, no strips, no batch strips — regardless of
pragma. That is always correct, and it costs one thing: on a GPU, a frame too large to cook
whole runs out of memory where a split would have fitted. A uniformly bounded loop, and a
`//!tex 0.25` program whose only per-pixel loop bounds are the kind above, both split again.

A host can ask for these as warnings with `tex_api.control_flow_advisories(source,
binding_types)`: **W7006** marks a per-pixel `if` or `?:` with a gather (`sample`, `fetch`,
`@A(u, v)`, a blur, a reduction) in a branch — unaffected by `0.25`, since both branches still
run either way. **W7007** marks control flow that acts on every pixel, meaning a `break`,
`continue` or `return` under a per-pixel `if`, or a loop whose condition is per-pixel — **since
`0.25`, conditional**: false, and never fires, for a program actually cooked under the masked
rules above, because such a program no longer acts on every pixel; still fires exactly as before
for a program cooked under `0.23`/`0.24` rules. **W7008** marks the shapes whose result depends
on which region is cooked, so the engine declines to split the cook — a per-pixel loop bound, a
string chosen per pixel by an `if` or a `?:`, or a per-pixel value cast straight to a string.
W7008 is the part of W7007 the engine acts on: a `break` under a per-pixel `if` draws W7007 and
no W7008, because it fires on first arrival and so does the same thing in every region. Its LOOP
half sunsets in lockstep with W7007 above (a masked loop's pass count is each pixel's own, not
the region's maximum); its STRING halves — a per-pixel string choice, or a per-pixel value cast
straight to a string — never sunset, at any language level. All three are opt-in: `tex_api.check()`, and so the
editor's live lint, never reports them.

---

## 8. See also

* [learn_tex_in_5_minutes.md](learn_tex_in_5_minutes.md) — the tutorial.
* [Function-Reference.md](Function-Reference.md) — every built-in function (generated
  from the stdlib registry).
* `wiki/Error-Codes.md` — every `ENNNN` / `WNNNN` diagnostic (generated).
* [DEVELOPMENT.md](DEVELOPMENT.md) — API-stability tiers and the rejected-decision register.
