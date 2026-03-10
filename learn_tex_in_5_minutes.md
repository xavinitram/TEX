# Learn TEX in 5 Minutes

TEX lets you write compact code to process images in ComfyUI. Instead of wiring 10 nodes for a simple effect, write a few lines of TEX. Each example below builds on the previous one.

---

## 1. Hello World — Passthrough

```
@OUT = @image;
```

`@image` is your input — the name becomes an input socket on the node. `@OUT` is the output. This copies the input unchanged. You can use any name you like: `@photo`, `@base`, `@input` — TEX creates sockets automatically from `@name` references.

**Try it:** Create a TEX Wrangle node, connect an image to the `image` input, and queue the prompt. You should see the same image come out the other side.

---

## 2. Darken an Image

```
@OUT = @image * 0.5;
```

Multiply every pixel by 0.5 to darken the image. All operations happen per-pixel automatically — you never need to write loops.

---

## 3. Blend Two Images

```
f$blend = 0.5;
@OUT = lerp(@base, @overlay, $blend);
```

`lerp()` linearly interpolates between two values. `f$blend = 0.5;` creates a **FLOAT parameter widget** on the node with a default of 0.5 — you can adjust it without editing code. Connect two images to `base` and `overlay` to try this.

---

## 4. Procedural Gradient

```
@OUT = vec3(u, v, 0.5);
```

`u` and `v` are normalized coordinates that range from 0 to 1 across the image. `vec3(r, g, b)` creates a color. This produces a gradient that goes from red at the top-left to yellow-green at the bottom-right, with a constant blue channel of 0.5. You still need an input connected (e.g. `@ref`) so TEX knows what resolution to use.

---

## 5. Conditional Color

```
float gray = luma(@image);
if (gray > 0.5) {
    @OUT = vec3(1.0, 0.0, 0.0);
} else {
    @OUT = @image;
}
```

`luma()` computes the brightness of each pixel. Bright pixels become red, dark pixels keep their original color. The `if/else` works per-pixel automatically — under the hood it is vectorized via `torch.where`, so there is no performance penalty.

---

## 6. Parameters

```
f$strength = 0.5;
@OUT = @image * $strength;
```

Declare parameters with `$name` to create adjustable widgets on the node. Prefix with `f` for float, `i` for integer, `s` for string:

- `f$strength = 0.5;` → FLOAT slider (default 0.5)
- `i$radius = 2;` → INT spinner (default 2)
- `s$label = "hello";` → STRING text field

Parameter widgets appear directly on the node for easy adjustment at runtime.

---

## 7. Multiple Outputs

```
f$strength = 1.0;
@darkened = @image * vec3(1.0 - $strength);
@brightness_mask = luma(@image);
```

Write to any `@name` to create an output. Here, `darkened` and `brightness_mask` appear as separate output sockets. Connect each to a different downstream node.

---

## What's Next?

- Click the **?** icon on any TEX node for the full function reference (79 functions).
- Browse the `examples/` folder for more patterns: blur, vignette, edge detection, noise, and more.
- Types: `float`, `int`, `vec3` (RGB), `vec4` (RGBA) — TEX infers types automatically.
- Use `@name` for any descriptive input name (e.g., `@base`, `@mask`, `@frames`).
- Output types (IMAGE, MASK, FLOAT, STRING, LATENT) are auto-inferred from your code.
