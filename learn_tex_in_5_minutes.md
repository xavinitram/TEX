# Learn TEX in 5 Minutes

TEX lets you write compact code to process images in ComfyUI. Instead of wiring 10 nodes for a simple effect, write a few lines of TEX. Each example below builds on the previous one.

---

## 1. Hello World — Passthrough

```
@OUT = @A;
```

`@A` is your input image. `@OUT` is the output. This copies the input unchanged.

**Try it:** Create a TEX Wrangle node, connect an image to the `@A` input, and queue the prompt. You should see the same image come out the other side.

---

## 2. Darken an Image

```
@OUT = @A * 0.5;
```

Multiply every pixel by 0.5 to darken the image. All operations happen per-pixel automatically — you never need to write loops.

---

## 3. Blend Two Images

```
@OUT = lerp(@A, @B, 0.5);
```

`lerp()` linearly interpolates between two values. `0.0` gives you all `@A`, `1.0` gives you all `@B`, and `0.5` blends them halfway. Connect two images to `@A` and `@B` to try this.

---

## 4. Procedural Gradient

```
@OUT = vec3(u, v, 0.5);
```

`u` and `v` are normalized coordinates that range from 0 to 1 across the image. `vec3(r, g, b)` creates a color. This produces a gradient that goes from red at the top-left to yellow-green at the bottom-right, with a constant blue channel of 0.5. You still need `@A` connected so TEX knows what resolution to use.

---

## 5. Conditional Color

```
float gray = luma(@A);
if (gray > 0.5) {
    @OUT = vec3(1.0, 0.0, 0.0);
} else {
    @OUT = @A;
}
```

`luma()` computes the brightness of each pixel. Bright pixels become red, dark pixels keep their original color. The `if/else` works per-pixel automatically — under the hood it is vectorized via `torch.where`, so there is no performance penalty.

---

## What's Next?

- Click the **?** icon on any TEX node for the full function reference (79 functions).
- Browse the `examples/` folder for more patterns.
- Types: `float`, `int`, `vec3` (RGB), `vec4` (RGBA) — TEX infers types automatically.
- Use `@name` for any input name (e.g., `@base`, `@mask`, `@strength`).
- Set `output_type` to `auto` (the default) and TEX infers whether the output is IMAGE, MASK, or FLOAT.
