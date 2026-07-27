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
(currently **`0.23`**) names the grammar + semantics this engine implements; the
package `__version__` tracks the release. They move independently — a release that
only fixes a bug or refactors internals does not bump the language version.

**A program may declare the language level it targets** with a leading pragma on its
own comment line:

```tex
//!tex 0.23
@OUT = vec4(@A.rgb * 1.2, 1.0);
```

The pragma is an ordinary comment to the compiler (it never becomes a token). It is a
*mechanism, not a promise*: `tex_api.check()` emits the advisory **W7004** when a
program targets a language *newer* than the engine implements (a feature it may not
understand), and stays silent for an equal or older target. `tex_api.language_pragma(source)`
returns the declared `"X.Y"` (or `None`).

**Stability contract.** A program that compiled and ran on version *N* keeps computing
the **same pixels** on version *N+1*. This is enforced, not merely intended: the frozen
**compat corpus** (`tests/compat_corpus.py`, goldens in `tests/compat_corpus_goldens.json`)
runs every bundled example plus a set of adversarial grammar programs on the CPU
interpreter and hashes their quantized outputs against committed goldens. A drift fails
the suite. Regenerating the goldens is a deliberate, reviewed act reserved for an
**intentional** language change.

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
  one-token type prefix: `f@x`, `img@src`, `f$gain`, `v3$tint`.
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
`* / %` · `+ -` · comparisons · `&& ||` · ternary `?:` · assignment. Loops are bounded
(static ranges for `for`; a guard for `while`) so a cook always terminates.

---

## 8. See also

* [learn_tex_in_5_minutes.md](learn_tex_in_5_minutes.md) — the tutorial.
* [Function-Reference.md](Function-Reference.md) — every built-in function (generated
  from the stdlib registry).
* `wiki/Error-Codes.md` — every `ENNNN` / `WNNNN` diagnostic (generated).
* [DEVELOPMENT.md](DEVELOPMENT.md) — API-stability tiers and the rejected-decision register.
