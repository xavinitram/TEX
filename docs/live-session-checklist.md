# TEX Wrangle — live verification checklist

The frontend features can't be render-verified headlessly — they need a real ComfyUI
canvas. This checklist is the **release exit criterion** (doc 35, LIVE): run it in a live
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

## Also verify (carried from prior cycles)
- [ ] Hover docs (F1 / hover a function) render.
- [ ] Preflight fusion bubble + stats appear for a chained TEX graph.
- [ ] Lazy input rename UX works (wire an input, see `@A`→slot mapping).
- [ ] A fused + lazy + `precision=auto` graph on CUDA cooks correctly.

**Phase 3 / release is not done until every box above is checked and screenshotted into the
build log (doc 36).**
