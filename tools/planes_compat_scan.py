"""DATA-6 compat scan, run in PHASE 1 rather than phase 2 — because if it comes back dirty the
DESIGN changes, not the implementation.

The claim the design rests on: making the lexer greedy over one dotted segment cannot move a
frozen golden, because no frozen program contains a dotted `@` form at all. If any does, it is
an ordinary swizzle and the splitback must be shown bit-exact on that program specifically.

Scans the frozen corpus AND every shipped example, since freeze #2 will draw from both.
"""
import os
import re
import sys

sys.path.insert(0, r"G:\ComfyUI_Menu\comfyUI\custom_nodes")
sys.path.insert(0, r"G:\ComfyUI_Menu\comfyUI\custom_nodes\TEX_Wrangle\tests")

from TEX_Wrangle.tex_compiler.lexer import Lexer, TokenType
from TEX_Wrangle.tex_compiler.types import CHANNEL_MAP, VALID_SWIZZLES

COLLISION = set(CHANNEL_MAP) | set(VALID_SWIZZLES)
print(f"collision set: {len(COLLISION)} names "
      f"({len(CHANNEL_MAP)} channels + {len(VALID_SWIZZLES)} swizzles)")
print(f"  lowercase-only? {all(n == n.lower() for n in COLLISION)}")
print()

import compat_corpus

sources = []
for name, src in compat_corpus._corpus_programs():
    sources.append((f"corpus:{name}", src))

exdir = r"G:\ComfyUI_Menu\comfyUI\custom_nodes\TEX_Wrangle\examples"
for fn in sorted(os.listdir(exdir)):
    if fn.endswith(".tex"):
        with open(os.path.join(exdir, fn), encoding="utf-8") as f:
            sources.append((f"example:{fn}", f.read()))

print(f"scanning {len(sources)} programs "
      f"({sum(1 for n, _ in sources if n.startswith('corpus'))} corpus, "
      f"{sum(1 for n, _ in sources if n.startswith('example'))} examples)\n")

# A dotted `@` form is `@ident . ident` with no intervening space, per the greedy lexer's rule.
DOTTED = re.compile(r"@[A-Za-z_][A-Za-z_0-9]*\.[A-Za-z_][A-Za-z_0-9]*")

hits, lexfail = [], []
for name, src in sources:
    try:
        toks = Lexer(src).tokenize()
    except Exception as e:
        lexfail.append((name, f"{type(e).__name__}: {e}"))
        continue
    # Token-level: an AT_BINDING immediately followed by DOT then IDENT is what the greedy
    # lexer would fuse. This is the exact set the change alters.
    for i, t in enumerate(toks):
        if t.type is not TokenType.AT_BINDING:
            continue
        if i + 2 < len(toks) and toks[i + 1].type is TokenType.DOT \
                and toks[i + 2].type is TokenType.IDENT:
            seg = toks[i + 2].value
            hits.append((name, f"@{t.value}.{seg}", seg,
                         "SWIZZLE" if seg in COLLISION else "NOT-A-SWIZZLE"))

print(f"dotted @ forms found: {len(hits)}")
by_kind = {}
for _n, _form, _seg, kind in hits:
    by_kind[kind] = by_kind.get(kind, 0) + 1
for k, v in sorted(by_kind.items()):
    print(f"  {k}: {v}")

odd = [h for h in hits if h[3] == "NOT-A-SWIZZLE"]
if odd:
    print("\n*** dotted forms whose segment is NOT a channel/swizzle — these would change "
          "meaning under the greedy lexer, and the design must address them by name:")
    for n, form, seg, _k in odd[:20]:
        print(f"    {n}: {form}  (segment {seg!r})")

if lexfail:
    print(f"\n{len(lexfail)} program(s) failed to lex (pre-existing):")
    for n, e in lexfail[:5]:
        print(f"    {n}: {e}")

print("\n" + "=" * 78)
if odd:
    print("VERDICT: DIRTY — at least one dotted form is not a swizzle. Design must handle it.")
elif hits:
    print("VERDICT: CLEAN-WITH-SWIZZLES — every dotted form is an ordinary swizzle, so the")
    print("  splitback rule's row 2/4 covers them all. Phase 2 must show these bit-exact.")
else:
    print("VERDICT: CLEAN — no frozen or shipped program contains a dotted @ form at all.")
    print("  The greedy lexer cannot alter any existing token stream.")
sys.exit(1 if odd else 0)
