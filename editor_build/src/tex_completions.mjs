/**
 * TEX Autocomplete Provider for CodeMirror 6
 *
 * Four completion sources:
 *   A. Stdlib functions — with signatures and descriptions
 *   B. Built-in variables — with type and description
 *   C. @ binding completions — triggered by "@" character
 *   D. $ parameter completions — triggered by "$" character
 */

// ─── Stdlib function completions ─────────────────────────────────────

const STDLIB_COMPLETIONS = [
    // Math
    { label: "sin", detail: "(x)", info: "Sine" },
    { label: "cos", detail: "(x)", info: "Cosine" },
    { label: "tan", detail: "(x)", info: "Tangent" },
    { label: "asin", detail: "(x)", info: "Arcsine" },
    { label: "acos", detail: "(x)", info: "Arccosine" },
    { label: "atan", detail: "(x)", info: "Arctangent" },
    { label: "atan2", detail: "(y, x)", info: "Two-argument arctangent" },
    { label: "sinh", detail: "(x)", info: "Hyperbolic sine" },
    { label: "cosh", detail: "(x)", info: "Hyperbolic cosine" },
    { label: "tanh", detail: "(x)", info: "Hyperbolic tangent" },
    { label: "pow", detail: "(x, y)", info: "x raised to power y" },
    { label: "pow2", detail: "(x)", info: "2 raised to power x" },
    { label: "pow10", detail: "(x)", info: "10 raised to power x" },
    { label: "sqrt", detail: "(x)", info: "Square root" },
    { label: "exp", detail: "(x)", info: "e raised to power x" },
    { label: "log", detail: "(x)", info: "Natural logarithm" },
    { label: "log2", detail: "(x)", info: "Base-2 logarithm" },
    { label: "log10", detail: "(x)", info: "Base-10 logarithm" },
    { label: "abs", detail: "(x)", info: "Absolute value" },
    { label: "sign", detail: "(x)", info: "Sign (-1, 0, or 1)" },
    { label: "floor", detail: "(x)", info: "Round down to integer" },
    { label: "ceil", detail: "(x)", info: "Round up to integer" },
    { label: "round", detail: "(x)", info: "Round to nearest integer" },
    { label: "fract", detail: "(x)", info: "Fractional part" },
    { label: "mod", detail: "(x, y)", info: "Modulo (remainder)" },
    { label: "hypot", detail: "(x, y)", info: "Hypotenuse: sqrt(x*x + y*y)" },
    { label: "degrees", detail: "(x)", info: "Radians to degrees" },
    { label: "radians", detail: "(x)", info: "Degrees to radians" },
    // Safe ops
    { label: "spow", detail: "(x, y)", info: "Sign-safe power (no NaN on negatives)" },
    { label: "sdiv", detail: "(a, b)", info: "Safe division (0 when b near 0)" },
    // Classification
    { label: "isnan", detail: "(x)", info: "1.0 if NaN, else 0.0" },
    { label: "isinf", detail: "(x)", info: "1.0 if Inf, else 0.0" },
    // Interpolation
    { label: "min", detail: "(a, b)", info: "Minimum of two values" },
    { label: "max", detail: "(a, b)", info: "Maximum of two values" },
    { label: "clamp", detail: "(x, lo, hi)", info: "Clamp x to [lo, hi]" },
    { label: "lerp", detail: "(a, b, t)", info: "Linear interpolation" },
    { label: "mix", detail: "(a, b, t)", info: "Linear interpolation (alias)" },
    { label: "fit", detail: "(x, omin, omax, nmin, nmax)", info: "Remap range" },
    { label: "step", detail: "(edge, x)", info: "0.0 if x < edge, else 1.0" },
    { label: "smoothstep", detail: "(lo, hi, x)", info: "Smooth Hermite step" },
    // Vector
    { label: "dot", detail: "(a, b)", info: "Dot product" },
    { label: "length", detail: "(v)", info: "Vector length" },
    { label: "distance", detail: "(a, b)", info: "Distance between vectors" },
    { label: "normalize", detail: "(v)", info: "Normalize to unit length" },
    { label: "cross", detail: "(a, b)", info: "Cross product (vec3)" },
    { label: "reflect", detail: "(v, n)", info: "Reflect vector around normal" },
    // Matrix
    { label: "transpose", detail: "(m)", info: "Transpose matrix" },
    { label: "determinant", detail: "(m)", info: "Matrix determinant" },
    { label: "inverse", detail: "(m)", info: "Matrix inverse" },
    // Color
    { label: "luma", detail: "(v)", info: "Luminance (Rec.709 weights)" },
    { label: "rgb2hsv", detail: "(rgb)", info: "RGB to HSV conversion" },
    { label: "hsv2rgb", detail: "(hsv)", info: "HSV to RGB conversion" },
    // Noise
    { label: "perlin", detail: "(x, y)", info: "2D Perlin noise [-1, 1]" },
    { label: "simplex", detail: "(x, y)", info: "2D Simplex noise [-1, 1]" },
    { label: "fbm", detail: "(x, y, octaves)", info: "Fractal Brownian Motion" },
    { label: "rand", detail: "()", info: "Random value [0, 1)" },
    // Sampling
    { label: "sample", detail: "(@in, u, v)", info: "Bilinear sample at UV coords" },
    { label: "fetch", detail: "(@in, px, py)", info: "Nearest pixel at integer coords" },
    { label: "sample_cubic", detail: "(@in, u, v)", info: "Bicubic (Catmull-Rom) sample" },
    { label: "sample_lanczos", detail: "(@in, u, v)", info: "Lanczos-3 sample" },
    { label: "fetch_frame", detail: "(@in, frame, px, py)", info: "Fetch from specific frame" },
    { label: "sample_frame", detail: "(@in, frame, u, v)", info: "Bilinear from specific frame" },
    // String
    { label: "str", detail: "(x)", info: "Number to string" },
    { label: "len", detail: "(s)", info: "String/array length" },
    { label: "replace", detail: "(s, old, new)", info: "Replace all occurrences" },
    { label: "strip", detail: "(s)", info: "Trim whitespace" },
    { label: "lower", detail: "(s)", info: "To lowercase" },
    { label: "upper", detail: "(s)", info: "To uppercase" },
    { label: "contains", detail: "(s, sub)", info: "1.0 if sub found, else 0.0" },
    { label: "startswith", detail: "(s, pre)", info: "1.0 if starts with prefix" },
    { label: "endswith", detail: "(s, suf)", info: "1.0 if ends with suffix" },
    { label: "find", detail: "(s, sub)", info: "Index of substring, or -1" },
    { label: "substr", detail: "(s, start, len?)", info: "Extract substring" },
    { label: "to_int", detail: "(s)", info: "Parse integer from string" },
    { label: "to_float", detail: "(s)", info: "Parse float from string" },
    { label: "sanitize_filename", detail: "(s)", info: "Clean illegal filename chars" },
    // Array
    { label: "sort", detail: "(arr)", info: "Sort ascending" },
    { label: "reverse", detail: "(arr)", info: "Reverse order" },
    { label: "arr_sum", detail: "(arr)", info: "Sum of elements" },
    { label: "arr_min", detail: "(arr)", info: "Minimum element" },
    { label: "arr_max", detail: "(arr)", info: "Maximum element" },
    { label: "median", detail: "(arr)", info: "Median element" },
    { label: "arr_avg", detail: "(arr)", info: "Mean of elements" },
    { label: "join", detail: "(arr, sep)", info: "Join string array" },
    // Image reductions
    { label: "img_sum", detail: "(@in)", info: "Per-channel sum of all pixels" },
    { label: "img_mean", detail: "(@in)", info: "Per-channel mean of all pixels" },
    { label: "img_min", detail: "(@in)", info: "Per-channel minimum" },
    { label: "img_max", detail: "(@in)", info: "Per-channel maximum" },
    { label: "img_median", detail: "(@in)", info: "Per-channel median" },
].map(c => ({ ...c, type: "function" }));

// ─── Type keyword completions ────────────────────────────────────────

const TYPE_COMPLETIONS = [
    { label: "float", type: "keyword", info: "Scalar floating-point" },
    { label: "int", type: "keyword", info: "Integer value" },
    { label: "vec3", type: "keyword", detail: "(r, g, b)", info: "3-component vector (RGB)" },
    { label: "vec4", type: "keyword", detail: "(r, g, b, a)", info: "4-component vector (RGBA)" },
    { label: "mat3", type: "keyword", detail: "(...)", info: "3x3 matrix (internal only)" },
    { label: "mat4", type: "keyword", detail: "(...)", info: "4x4 matrix (internal only)" },
    { label: "string", type: "keyword", info: "Text value (scalar-only)" },
    { label: "if", type: "keyword", info: "Conditional (vectorized via torch.where)" },
    { label: "else", type: "keyword", info: "Else branch" },
    { label: "for", type: "keyword", detail: "(int i = 0; i < n; i++)", info: "Bounded loop" },
];

// ─── Built-in variable completions ───────────────────────────────────

const VARIABLE_COMPLETIONS = [
    { label: "ix", type: "variable", info: "Pixel x-coordinate (integer)" },
    { label: "iy", type: "variable", info: "Pixel y-coordinate (integer)" },
    { label: "u", type: "variable", info: "Normalized x-coordinate [0, 1]" },
    { label: "v", type: "variable", info: "Normalized y-coordinate [0, 1]" },
    { label: "iw", type: "variable", info: "Image width (pixels)" },
    { label: "ih", type: "variable", info: "Image height (pixels)" },
    { label: "ic", type: "variable", info: "Latent channel count (0 for images)" },
    { label: "fi", type: "variable", info: "Frame/batch index (0 to B-1)" },
    { label: "fn", type: "variable", info: "Total frame/batch count" },
    { label: "PI", type: "constant", info: "3.14159..." },
    { label: "E", type: "constant", info: "2.71828..." },
];

// ─── All non-binding completions ─────────────────────────────────────

const ALL_COMPLETIONS = [...STDLIB_COMPLETIONS, ...TYPE_COMPLETIONS, ...VARIABLE_COMPLETIONS];

// ─── Completion function factory ─────────────────────────────────────

/**
 * Create a completion source for a specific TEX node.
 * @param {Function} getBindings — returns current @ binding names (inputs + outputs)
 * @param {Function} [getParams] — returns current $ parameter names
 */
export function createTexCompletions(getBindings, getParams) {
    return function texCompletionSource(context) {
        // ── $ parameter trigger ──
        const dollarMatch = context.matchBefore(/\$\w*/);
        if (dollarMatch) {
            const paramOptions = [];
            try {
                const params = getParams ? getParams() : [];
                for (const name of params) {
                    paramOptions.push({
                        label: "$" + name,
                        type: "variable",
                        info: `Parameter "${name}"`,
                    });
                }
            } catch (_) {}
            return {
                from: dollarMatch.from,
                options: paramOptions,
                validFor: /^\$\w*$/,
            };
        }

        // ── @ binding trigger ──
        // If the user just typed "@", show all available bindings
        const atMatch = context.matchBefore(/@\w*/);
        if (atMatch) {
            const bindingOptions = [];
            // Add dynamic bindings from connected sockets (inputs + outputs)
            try {
                const bindings = getBindings();
                if (bindings && bindings.length) {
                    for (const name of bindings) {
                        bindingOptions.push({
                            label: "@" + name,
                            type: "variable",
                            info: `Binding "${name}"`,
                        });
                    }
                }
            } catch (err) {
                // Ignore binding lookup errors
            }
            return {
                from: atMatch.from,
                options: bindingOptions,
                validFor: /^@\w*$/,
            };
        }

        // ── General word completions ──
        // Activate on 1+ typed characters
        const word = context.matchBefore(/\w+/);
        if (!word) return null;

        // Require at least 1 character typed (word.text.length >= 1)
        if (word.text.length < 1) return null;

        return {
            from: word.from,
            options: ALL_COMPLETIONS,
            validFor: /^\w*$/,
        };
    };
}
