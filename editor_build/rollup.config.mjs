import { nodeResolve } from "@rollup/plugin-node-resolve";
import terser from "@rollup/plugin-terser";

export default {
  input: "src/tex_cm6.mjs",
  output: {
    file: "../js/tex_cm6_bundle.js",
    format: "iife",
    name: "TEX_CM6",
    // ComfyUI loads JS files as ES modules via import(), so var declarations
    // are module-scoped. We must explicitly assign to window for cross-module access.
    footer: "window.TEX_CM6 = TEX_CM6;",
  },
  plugins: [nodeResolve(), terser()],
};
