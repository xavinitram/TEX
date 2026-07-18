# TEX Wrangle — live verification checklist

The frontend features can't be render-verified headlessly — they need a real ComfyUI
canvas. This checklist is the **release exit criterion**: run it in a live
ComfyUI, in **both** node modes (classic / Nodes 1.0 **and** Vue / Nodes 2.0), and paste
screenshots into the build log. Re-run each release — the JS is version-sensitive.

## Setup
- [ ] ComfyUI launched with the TEX Wrangle node installed; browser console open.
- [ ] Confirm the node loads with **no console errors** on graph open.
- [ ] Toggle **Settings → TEX → Debug → perfHud** ON.

## C1-ux — perf HUD (DOM dual-path) + probes + tooltip
- [ ] Cook a node. A **perf badge** appears below it (`tier · N.Nms · precision`).
- [ ] **Nodes 1.0:** badge renders (canvas HUD also draws — that's expected, dual-path).
- [ ] **Nodes 2.0:** badge renders via the RAF loop (canvas HUD does *not* fire here — the
      DOM badge is the guarantee). Position looks right below the node body.
- [ ] Drag/zoom the node — the badge tracks position and scales.
- [ ] A program with `debug_print("x", value, 0, 0)` shows the probe line(s) in the badge.
- [ ] Hover the badge — tooltip shows the tier `reason` and `precision_reason` (C7-ux).
- [ ] A fp16-auto or fallback cook shows the badge in **amber** with `(←from)`.
- [ ] Toggle perfHud OFF → badge disappears; results unchanged.

## C4-ux — near-singularity overlay
- [ ] Enable the node's **debug_nan_highlight** toggle.
- [ ] Cook `@OUT = vec4(vec3(sdiv(1.0, u - u)), 1.0);` → pixels are **cyan** (not black/clean).
- [ ] Cook a raw `1.0 / (u - u)` (unguarded) → pixels are **magenta** (NaN), distinct from cyan.
- [ ] The perf badge shows `⬤N` (the near-singularity count).
- [ ] Toggle OFF → no cyan, no perf-cost change.

## C2-ux — TEX Doctor modal
- [ ] Right-click the node → **TEX Doctor** → a modal opens.
- [ ] It renders torch/triton/msvc/cache/tiers/arch as a key-value table (booleans coloured).
- [ ] On an unverified GPU (or force it), the **arch caveat** shows prominently at the top.
- [ ] Close via the button, backdrop click, and Escape.

## C6-ux — snippet discoverability
- [ ] A fresh node's default code shows `// Right-click → TEX Snippets for 114 examples`.
- [ ] Right-click → **Snippets** opens the browser with the examples.

## S-2 — example workflows
- [ ] Each `examples/workflows/*.json` drag-drops into ComfyUI, loads without error, and cooks
      to a visible PreviewImage.

## FUS-1 — DAG-region fusion (v0.21, NEW — the release's live-checklist gate)
The Python core + `/tex_wrangle/detect_regions` route + FUS-3 equivalence gate are
headless-verified; the frontend serialize/collapse (`_texSerializeGraph`,
`_texCollapseRegion`, `_texCollapseRegions`) can only be checked on a live canvas.
The linear-chain collapse is UNCHANGED — verify it still works AND the new fan-out path.
- [ ] **Linear chain still fuses** (regression): a straight A→B→C TEX chain collapses
      to the terminal and cooks correctly (unchanged behavior — this must not break).
- [ ] **Diamond fuses**: A→{B,C}→D (A feeds B and C; D merges both, e.g. `@OUT=(@b+@c)*0.5`).
      Queue it — the whole diamond collapses into D, cooks correctly, and matches the
      same graph run UNFUSED (toggle **Settings → TEX → Fusion → enabled** off to compare).
- [ ] **Fan-out tree fuses**: A feeding 3 downstream nodes that a terminal merges.
- [ ] **Two-source merge is left UNFUSED** (safe): a node reading two external images
      does not collapse — its nodes cook separately, result still correct.
- [ ] **Wired-`$param` member left UNFUSED** (C4, silent-wrong guard): in a fan-out
      region, drive an upstream member's `$param` from a WIRE (not its widget) and
      animate/change that wire's value — the region must NOT collapse that member, and
      the cooked result must track the LIVE wired value (not a baked stale widget value).
- [ ] **Bypass/mute upstream of a region** (C1/C2): Ctrl+B (bypass) or Ctrl+M (mute) a
      node feeding a branching region, then queue — the prompt must still submit and
      cook (never a "node N not found" rejection); that region simply runs unfused.
- [ ] **Bypass/mute upstream of a LINEAR chain** (the same dangling-source class on the
      v0.20 path, which runs on EVERY graph): Ctrl+B the node feeding the FIRST TEX node
      of a plain A→B→C chain, then queue — the prompt must still submit and cook. The
      chain source is read from litegraph but spliced into the serialized prompt, and
      ComfyUI omits bypassed nodes from it; without the existence guard this rewires to a
      node that isn't there and ComfyUI rejects the WHOLE prompt.
- [ ] **MASK-sourced region** ([15], two cases — the preflight now uses the source's
      socket type, so a MASK is checked as the float it actually is):
      - *valid*: a branching region off a MASK doing float-safe math (`@in * 1.1`) must
        still fuse and cook (an IMAGE-only preflight wrongly rejected `@in.a`-style
        programs that are legal on a float — this is a coverage gain).
      - *invalid*: a member calling `length()`/`normalize()` **on the mask source** must
        report the real type error (`length() needs a vector, but argument 1 is float`)
        at that node — NOT the misleading "TEX couldn't fuse this chain … turn off TEX
        Fusion". The graph is broken either way; the fix is that the message is honest.
- [ ] **Route-down fail-safe**: stop the ComfyUI server route (or observe with devtools
      Network throttled) — a `/tex_wrangle/detect_regions` miss/timeout must leave fan-out
      graphs UNFUSED and still cooking (never a stalled or broken queue).
- [ ] **Preview tap on an intermediate** of a fan-out region leaves that region unfused
      (its intermediate escapes) — no dangling reference, prompt still valid.
- [ ] Browser console shows no `[TEX] region fuse skipped` errors on a valid graph.

## LANG-5 — server-backed snippet sync + data-loss hardening (v0.23)
The store round-trip is headless-tested (`test_lang5_snippet_route`); the localStorage
cache / pending-set / merge logic in `_saveUserSnippets`, `_postUserSnippets` and
`_syncUserSnippetsFromServer` can only be checked on a live canvas. Open DevTools →
Application → Local Storage and watch `tex_wrangle_snippets` + `tex_wrangle_snippets_pending`.
- [ ] **Happy path**: select code → right-click → Snippets → *Save Snippet* with a name.
      Reopen the menu — it appears. `tex_wrangle_snippets_pending` is empty (POST confirmed).
      Confirm `<user_dir>/tex_wrangle/user_snippets.json` on disk contains it.
- [ ] **Manage**: *Manage Snippets…* → Rename and Delete each round-trip to disk and the
      pending set clears after each (reopen the menu to trigger a sync).
- [ ] **Offline save survives (BUG 1)**: stop the ComfyUI server (or DevTools → Network →
      Offline). Save a snippet — it shows locally and its name is in
      `tex_wrangle_snippets_pending`. Bring the server back, reopen the Snippets menu (a
      sync). The snippet is now on disk and the pending set clears. It must NOT vanish.
- [ ] **Rejected save survives**: make `user_snippets.json` read-only (or its dir), save a
      snippet → the POST returns 503, the name stays pending, the snippet stays in the menu
      (not discarded). Restore write permission → next sync persists it and clears pending.
- [ ] **Transient read-error doesn't wipe (BUG 2)**: with saved snippets present, force the
      GET to fail (temporarily lock/rename `user_snippets.json`, or DevTools throttle to make
      `/tex_wrangle/user_snippets` 503). Reopen the menu — the existing snippets must STILL be
      there (the cache is not overwritten). Console shows no data loss; restoring the file
      and reopening syncs cleanly.
- [ ] **Cross-tab / cross-machine merge**: with snippet X pending locally (offline), have the
      server store already contain a different snippet Y. On reconnect + sync, the menu shows
      BOTH X and Y (server truth merged, local pending preserved), and X reaches disk.
- [ ] **Rapid saves land in order (single-flight POST)**: save two different snippets in quick
      succession (P then Q). Both end up on disk — the older whole-map POST must not clobber
      the newer one. (DevTools → Network: the second POST is sent only after the first settles.)
- [ ] **A stale sync GET does not revert a durable edit (BUG 3)**: edit a snippet's body, then
      immediately open the Snippets menu (forces a sync GET that may race the save's POST). The
      menu and `tex_wrangle_snippets` must show the NEW body, never the pre-edit one — the merge
      snapshots the pending set at GET-issue time and won't adopt server truth for a name that
      was pending when it asked. (Throttle the GET in DevTools to widen the race.)
- [ ] Browser console shows no errors from snippet save/sync on any of the above.
- [ ] *Known limitation (not a bug):* deleting a snippet on one machine/tab may not passively
      propagate to another that already has it cached (the merge favors preservation over
      delete-propagation). Full convergence is a follow-up needing a versioned server store.

## Also verify (carried from prior cycles)
- [ ] Hover docs (F1 / hover a function) render.
- [ ] Preflight fusion bubble + stats appear for a chained TEX graph.
- [ ] Lazy input rename UX works (wire an input, see `@A`→slot mapping).
- [ ] A fused + lazy + `precision=auto` graph on CUDA cooks correctly.

**Phase 3 / release is not done until every box above is checked and screenshotted into the
release build log.**
