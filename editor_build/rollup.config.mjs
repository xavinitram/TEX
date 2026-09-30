import { nodeResolve } from "@rollup/plugin-node-resolve";
import terser from "@rollup/plugin-terser";

export default {
  input: "src/tex_cm6.mjs",
  output: {
    file: "../js/tex_cm6_bundle.js",
    // The entry has no exports, so the IIFE needs no name. ComfyUI loads JS files as ES
    // modules via import(), where top-level vars are module-scoped, so the entry itself
    // sets globalThis.TEX_CM6 (src/tex_cm6.mjs).
    format: "iife",
  },
  plugins: [nodeResolve(), terser()],
};
