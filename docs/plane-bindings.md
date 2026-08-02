# DATA-6 — plane bindings (`@src.diffuse`): the design

*Doc 42 §2.1 / doc 40 §4.2. Phase 1 of the v0.35 plan: the decisions, made once, before any
code. Verified read-only against the phase-0 tree; every file:line below was checked, not
recalled.*

**The item.** A host wires ONE image input carrying many named planes — an EXR's
`diffuse`/`specular`/`Z`/`N`, a render's AOVs — and a program reads them by name:

```tex
vec3 lit = @beauty.diffuse + @beauty.specular * $spec;
float depth = @beauty.Z;
@OUT = vec4(mix(lit, $fog, smoothstep($near, $far, depth)), 1.0);
```

The alternative is one wire per plane, which is what a compositor's node graph looks like when
it has lost the argument. TEXType stays ≤ VEC4 (doc 42 §0.2): **planes exist precisely so the
type system does not have to grow.** A plane is a binding; a set of planes is a wire.

---

## 1. Who owns the dot

This is the release's compat tripwire and the reason this phase gets its own session.

**The constraint that decides it:** the parser provably never sees binding types.
`TEXCache.compile_tex` calls `Parser(tokens, source=code).parse()` (`tex_cache.py:310`) and
types enter one call later, at `TypeChecker(binding_types=..., source=source)`
(`tex_cache.py:326`, inside the shared `compile_ast`). So doc 40's "`.name` means a plane *on a
PLANES-typed wire*" cannot live in the grammar. Something upstream of types must tokenize the
dot, and something downstream of types must decide what it meant.

### Decision: (a) lexer-greedy, with a TypeChecker splitback

`read_at_binding` consumes `@`, the identifier, and then **exactly one** dotted segment when
one follows. `@beauty.diffuse` becomes a single `AT_BINDING` token whose value is the verbatim
string `"beauty.diffuse"`.

**Why one segment and not greedy-to-the-end.** It is what makes the whole design collapse into
a single rule. Consider a plane that is itself a vec3:

```
@beauty.diffuse.rgb   →   AT_BINDING("beauty.diffuse")  .  IDENT("rgb")
@A.rgb                →   AT_BINDING("A.rgb")
```

The first is a plane read followed by an *ordinary* `ChannelAccess`, which needs no new
machinery — swizzling a plane works on the day planes ship, for free. The second is an
ordinary swizzle that the lexer has mis-tokenized, and §1.1 puts it back. One segment is the
exact greed at which both fall out of one rule; two would eat the swizzle in the first case.

**Why (a) over the alternative.** Doc 42 §2.1 records the other lawful design — leave the
grammar alone, and rewrite `ChannelAccess(BindingRef)` into a dotted `BindingRef` in a
pre-typecheck pass. It is cheaper in the lexer and worse everywhere else, for one structural
reason: **everything that matters is keyed on token value, not on AST shape.**

* `sigil_names(code)` (`tex_marshalling.py:593`) lexes the source and returns every `@name`.
  Under (a) it returns `beauty.diffuse` — per-plane demand, for free, with no second scan.
  Under (b) it returns `beauty`, and demand-driven expansion (§3) needs a whole new mechanism.
  It has two consumers already (`param_only_names`, `tex_roi._referenced_at_bindings`) and its
  own docstring records what happened last time those two scans were separate copies.
* The sampling sugar `@src.N(u, v)` and the fetch form `@src.N[x, y]` work under (a) because
  the postfix forms are gated on `BindingRef` — and under (a) `@src.N` *is* a `BindingRef`.
  Today `@src.N(u,v)` is a parse-time **E2002** ("This value can't be called like a function…
  Only function names and @bindings can be followed by `(...)`", `parser.py:843-850`) — because
  the dot has already produced a `ChannelAccess`, which is not callable — and `@src.N[x,y]`
  parses as `ArrayIndexAccess` and dies in the TypeChecker. Both are unreachable as plane
  sugar under (b), and both fall out for free under (a).
* `identity_binding_types` (`tex_marshalling.py:659`) and `TEXCache.fingerprint`
  (`tex_cache.py:204`) hash `(name, type)` pairs. `("beauty.diffuse", "vec3")` is just another
  tuple; no identity machinery changes.

The cost of (a) is exactly one thing, and it is the next section.

### 1.1 The swizzle-splitback rule, spelled exactly

Under (a), `@A.r` on an ordinary VEC4 wire lexes as a binding named `A.r`. The TypeChecker must
put it back. This is legal — not a hack — because **the compile cache is keyed on
`(code, binding_types)`** (`tex_cache.fingerprint`, `tex_cache.py:204`), so a resolution that
depends on binding types can never alias a differently-typed compile of the same source.

For an `AT_BINDING` token whose value contains a dot, split on the **last** dot into
`base` and `seg`, then:

| `binding_types[base]` | rule |
|---|---|
| `PLANES` | a **plane read**. Keep the dotted `BindingRef("base.seg")`. If `seg` is not in the wire's declared plane set → **W7xxx** (§3). |
| a vector type (`VEC2`/`VEC3`/`VEC4`) | **split back**: rewrite to `ChannelAccess(BindingRef(base), seg)`. `seg` must be in `CHANNEL_MAP` ∪ `VALID_SWIZZLES` (`types.py:128-131`, `:155-160`) — otherwise the existing unknown-channel error fires, unchanged. |
| a non-vector type (`FLOAT`/`INT`/`MASK`/`STRING`/…) | split back as above; `base_is_vector` (`types.py:134`) already owns what `.r` means on a channel-less value, and that answer does not change. |
| **absent** (untyped base) | **default to swizzle** — split back. This is the existing VEC4 fallback in the TypeChecker, and it is the safe default because it is what every program written before planes existed meant. |

The last row is the compat guarantee in one line: **an untyped base is a swizzle**, so no
program that compiles today can be re-read as a plane access tomorrow.

### 1.2 The collision, and why it is narrower than it looks

A wire cannot be both. If `binding_types[base]` is `PLANES` **and** `seg` is a channel or
swizzle name, `@src.rgb` could mean "the plane called `rgb`" or "the rgb of something", and
once planes can themselves be swizzled (§1) the ambiguity is real rather than theoretical.

**Rule: a PLANES wire may not declare a plane whose name is in the collision set.** Refused at
expansion time with a new **E3xxx**, message form *"plane `rgb` on `@src` collides with the
swizzle `.rgb` — rename the plane"*. Never a silent guess, per doc 40.

The collision set is `CHANNEL_MAP` ∪ `VALID_SWIZZLES` — 8 single channels
(`r g b a x y z w`) plus 30 swizzles (`rg`…`wz`, `rgb`, `rgba`, `xyz`, `xyzw`, `bgr`, `abgr`).

**It is case-sensitive and lowercase-only, and that matters more than it sounds.** The
conventional EXR data-layer names are uppercase — `Z` for depth, `N` for normals, `RGBA` for
the beauty group — and *none of them collide*: `Z ∉ CHANNEL_MAP`, which contains only `z`.
So `@beauty.Z` is a plane read on day one, with no rename and no escape hatch, which is the
single most common thing a user will type. What collides is a lowercase `z` plane, and the
error tells them to rename it. Worth stating because the obvious reading of "depth planes
collide with `.z`" is wrong, and someone will otherwise design an escape hatch for a problem
that does not exist.

*(Escape hatches considered and rejected: a quoted form `@src."rgb"` adds grammar for a case
the naming convention already avoids; a `plane()` builtin re-introduces the function-call
surface planes exist to avoid. If a real file ever forces it, the reopen gate is a foreign EXR
fixture whose layer group is genuinely lowercase-swizzle-named.)*

---

## 2. The wire value: `PlanesValue`

A dedicated wrapper class, following the `Promise` precedent (`tex_engine.py:1072-1074`'s
`_resolve_promise_bindings` is the shape to mirror):

```python
class PlanesValue:
    planes: dict[str, torch.Tensor]     # name -> [B,H,W,C<=4]
    descs:  dict[str, BufferDesc]       # per-plane colour/transfer metadata (§2.2's EXR half)
```

**Deliberately not a tensor subclass.** A subclass would be silently accepted by every
`isinstance(v, torch.Tensor)` test in the tree — of which there are many, including the
`_consensus_extent` grid derivation phase 0 just consolidated — and would size cook grids off
whichever plane happened to be first. A distinct class fails loudly instead, and the R4
neutrality guard becomes one `isinstance` check on the plane-free path, mirroring
`_resolve_promise_bindings`'s existing `any()`.

**A raw `{name: tensor}` dict stays E7005-refused.** v0.34 closed an identity-corruption class
by refusing dicts at the wire; re-opening it for planes would reintroduce exactly that hazard
for the sake of saving hosts one import. The refusal message gains a "did you mean
`PlanesValue`?" hint.

---

## 3. Demand-driven expansion, and the W/E codes

**The pass.** Before type-checking, expand a `PLANES` wire into per-plane bindings — but only
the planes `sigil_names(code)` says the source mentions. A program that reads `@src.diffuse`
never puts `@src.Z` into `bindings`, so the interpreter's ingest loop (which marshals exactly
what is in the dict) never touches it, never moves it to the device, never casts it.

**That is the PM-10 laziness proof, and it is structural rather than measured** — there is no
code path by which an unmentioned plane reaches ingest, so the marshalling spy is confirming a
property the design already guarantees rather than establishing one.

**The declared-plane set rides the `PlanesValue`**, which is what lets the linter tell two
different mistakes apart:

* **Reading an undeclared plane** — `@src.diffues` when the wire carries `diffuse`. A new
  **W7xxx**, with a did-you-mean over the declared set (the existing diagnostics helper
  already does the edit-distance work for builtins).
* **A declared plane the program does not read** — silent. This must **not** fire W7002
  ("unused binding"), and the carve-out is explicit: W7002 reads `binding_types`, which after
  expansion contains only the mentioned planes, so an unread plane is invisible to it by
  construction. Stated because the opposite behaviour would make every 12-AOV EXR emit eleven
  warnings, and someone would then "fix" the expansion to declare all of them.

---

## 4. Plane writes

`@OUT.diffuse = expr` is the same rule as §1 with the same owner: the dotted name is an
assigned binding `OUT.diffuse` on a `PLANES` output.

* **What declares an output PLANES: the first dotted write.** There is no separate
  declaration syntax — the program's shape is the declaration.
* **Mixed `@OUT = …` and `@OUT.N = …` is a compile error.** One output, one shape. Allowing
  both would make the output type depend on statement order, which is the class of bug phase 0
  spent its whole budget removing.
* Egress grows the inverse repack into a `PlanesValue`, and `map_inferred_type`
  (`tex_marshalling.py:670`) gains the PLANES arm.

---

## 5. Naming: the internal binding name is the verbatim dotted string

`beauty.diffuse`, not a mangled `beauty__diffuse`. Three reasons, in order of weight:

1. **E6003 messages stay honest** — "input `@beauty.diffuse` is not connected" is what the user
   typed.
2. **EXR round-trip is trivial** — `layer.channel` is already the file's own spelling.
3. **`strip_user_prefix` composes**: `_s0_u_beauty.diffuse → beauty.diffuse`, so fusion's
   two-namespace trick is unaffected.

**One audit this obliges**, and it must happen in phase 2 before anything else: every place
that assumes a binding name is a bare identifier. The known candidates are codegen's local
variable naming (a dotted name cannot become a Python local verbatim) and graph-capture keys.
This is a *known-unknown* — it is written here so phase 2 opens with the grep rather than
discovering it in a stack trace.

---

## 6. Fusion: v1 refuses PLANES edges

Per-stage expansion runs **before** `_fused_memo_key`, so the memo holds per-plane types and
`tex_fusion`'s dotted names flow untouched.

Whether the *frontend* chain detector admits a PLANES edge in v1 is an honest-scope call, and
the answer is **no**. Refusing keeps the R2 compat lane trivially green, and planes still fuse
through the engine API where a host asks for it explicitly. The gate to revisit is a measured
host graph where a plane edge inside a fused region is the bottleneck.

---

## 7. The three-compile-paths convergence — decided once, here

Three places hand-roll "source → typed AST": `failure_harness.compile_program`,
`test_integration._prepare_example`, and the production `TEXCache.compile_tex`. If expansion
does not reach all three, the R1/R2 lanes silently test *unexpanded* programs — a green suite
over a surface that is not the shipped one.

**Decision: converge, and the seam already exists.** `TEXCache.compile_ast(program,
binding_types, *, source)` (`tex_cache.py:318`) is STR-8's shared post-parse pipeline, already
used by both production entries (`compile_tex` and fusion's `compile_fused`). **Expansion lands
inside `compile_ast`, ahead of the first `TypeChecker` call**, and the two test harnesses are
migrated onto it.

Insert-thrice-with-a-parity-pin was the alternative and is rejected: a parity pin tests that
three copies agree *today*, which is the same bet phase 0 lost six times over on the cook-grid
derivation. One owner, or it drifts.

**COLOR-1 (v0.36) inherits this decision rather than remaking it** — its expansion pass needs
the same three insertions one release later, and this is the whole reason the convergence is
decided in the DATA-6 doc instead of being split across two.

`_prepare_example` additionally needs PLANES dummy synthesis **before** any plane program can
enter the corpus, or freeze #2 cannot contain one.

---

## 8. The compat scan

**Claim: DATA-6 is strictly additive, and no frozen program moves.**

The argument, not just the assertion — a dotted `@name.seg` has exactly three fates before this
change, and the third is the only one that changes:

1. `@A.r` on a typed VEC-ish wire — parsed as `ChannelAccess`, and §1.1 rewrites it back to
   *the same AST*. Bit-identical by construction.
2. `@A.r` on an untyped base — the VEC4-fallback swizzle, which §1.1's last row preserves
   verbatim.
3. `@src.N` where `src` is not vector-typed — a compile error today (E2002 for the call form,
   an unknown-channel or type error otherwise). Turning a compile error into a working feature
   cannot move a frozen golden, because a program that does not compile is not in the corpus.

**The scan was run in phase 1, not deferred to phase 2 — because a dirty result changes the
design rather than the implementation.** Tokenized all 129 frozen corpus programs plus all 116
shipped `examples/*.tex` (245 total; freeze #2 draws from both) and counted every
`AT_BINDING . IDENT` sequence — the exact set the greedy lexer fuses:

> **50 dotted `@` forms exist. All 50 are ordinary swizzles** (`.rgb`, `.r`, `.xy`, …); zero
> have a segment outside the collision set.

The first draft of this section predicted *zero*, which would have made the compat argument a
one-liner. It is wrong, and the correction is the reason §1.1 is specified to the row rather
than described: **50 real programs take the splitback path on every compile**, so it is a hot,
load-bearing rewrite and not an edge case. It also means the "untyped base defaults to swizzle"
row is doing actual work today, not just guarding a hypothetical.

**Phase 2 owes a bit-exactness proof over those 50 specifically**, not a general argument: cook
each pre- and post-change and assert identical output. That is a concrete, bounded gate, which
is what the scan bought.

*(Reproduce: `python tools/planes_compat_scan.py` from `tests/`. It exits non-zero only if a
dotted segment is NOT a swizzle — i.e. it is the tripwire for this design decision, not a
one-shot. The collision set is confirmed programmatically at 38 names — 8 channels + 30
swizzles — and lowercase-only, which is §1.2's `Z` finding.)*

Freeze #2 lands **after** expansion, so the 0.24 corpus snapshots the post-planes truth, and
plane programs can be authored into it (`examples/*.tex`, once `_prepare_example` synthesizes
PLANES dummies — §7).

---

## 9. Version: 0.24, and the rule behind it

`tex_language` bumps 0.23 → 0.24. The rule, written down here because v0.34 shipped
`fetch_time`/`sample_time` on 0.23 and the precedent needs stating rather than re-deriving:

> **A checklist-§2 *function* addition does not bump the language version. A grammar-visible
> surface does.**

`@src.diffuse` is grammar-visible — the lexer changes — so it bumps. `sample_time` was a
stdlib function on unchanged grammar, so it did not.

Six satellite copies of the version string must move together, and CF-7's new JS pin makes the
one with no Python coverage loud. `tex_api.LANGUAGE_VERSION` remains **strictly numeric**: a
`"0.24-planes"` string would make `_ver_tuple` return `(0, 0)` and silently break every version
comparison in the tree.

---

## 10. What this doc does not decide

* **EXR layers/parts** (§2.2) — the file half of PM-10, its own phase.
* **Whether a gather source may be a different resolution than the direct read.** Recorded as a
  phase-0 deferral gated on exactly this doc. The answer now: **a plane is co-extent with its
  wire**, so planes do not create the case, and the deferral stays open on its own terms for
  ordinary wires.
* **UINT planes** — excluded from v0.35, cryptomatte named as the future customer.
* **Per-plane residency/eviction.** `PlanesValue` holds tensors a `ResultCache` entry could in
  principle tier independently. Out of scope; noted because CF-1's home propagation is the
  machinery it would build on.
