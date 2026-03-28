"""
TEX Wrangle — Tensor Expression Language for ComfyUI.

A per-pixel kernel DSL for writing compact image/mask processing logic
directly inside ComfyUI. Inspired by Houdini VEX and Nuke BlinkScript.
"""

__version__ = "0.10.0"

import os

from .tex_node import TEXWrangleNode

NODE_CLASS_MAPPINGS = {
    "TEX_Wrangle": TEXWrangleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TEX_Wrangle": "TEX Wrangle",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# ─── Snippet API ──────────────────────────────────────────────────────
# Serves built-in example snippets from the examples/ directory.
# The frontend fetches this once on first use instead of embedding copies.

_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "examples")

# Map filename (without .tex) → "Category/Display Name"
_EXAMPLE_CATEGORIES = {
    # ── Color ──────────────────────────────────────────────────────────
    "auto_levels":          "Color/Auto Levels",
    "brightness_contrast":  "Color/Brightness Contrast",
    "channel_shuffle":      "Color/Channel Shuffle",
    "channel_swap":         "Color/Channel Swap",
    "color_functions":      "Color/Color Functions",
    "color_grade":          "Color/Color Grade",
    "color_mix":            "Color/Color Mix",
    "grade":                "Color/Grade",
    "grayscale":            "Color/Grayscale",
    "hue_shift":            "Color/Hue Shift",
    "invert":               "Color/Invert",
    "levels":               "Color/Levels",
    "posterize":            "Color/Posterize",
    "tone_map":             "Color/Tone Map",
    "white_balance":        "Color/White Balance",
    # ── Compositing ────────────────────────────────────────────────────
    "alpha_over":           "Compositing/Alpha Over",
    "composite":            "Compositing/Composite",
    "custom_blend":         "Compositing/Custom Blend",
    "merge":                "Compositing/Merge",
    "premultiply":          "Compositing/Premultiply",
    "soft_clamp":           "Compositing/Soft Clamp",
    # ── Effects ────────────────────────────────────────────────────────
    "barrel_distortion":    "Effects/Barrel Distortion",
    "chromatic_aberration":  "Effects/Chromatic Aberration",
    "corner_pin":           "Effects/Corner Pin",
    "emboss":               "Effects/Emboss",
    "film_chromatic_aberration": "Effects/Film Chromatic Aberration",
    "film_lens_distortion": "Effects/Film Lens Distortion",
    "film_optical_glow":    "Effects/Film Optical Glow",
    "film_vignette":        "Effects/Film Vignette",
    "godrays":              "Effects/Godrays",
    "halftone":             "Effects/Halftone",
    "kaleidoscope":         "Effects/Kaleidoscope",
    "lens_distortion":      "Effects/Lens Distortion",
    "lens_distortion_simple": "Effects/Lens Distortion Simple",
    "pixelate":             "Effects/Pixelate",
    "stmap":                "Effects/STMap",
    "swirl":                "Effects/Swirl",
    "transform_2d":         "Effects/Transform 2D",
    "vignette":             "Effects/Vignette",
    # ── Filter ─────────────────────────────────────────────────────────
    "bilateral_approx":     "Filter/Bilateral Approx",
    "blur":                 "Filter/Blur",
    "box_blur":             "Filter/Box Blur",
    "convolve":             "Filter/Convolve",
    "denoise":              "Filter/Denoise",
    "directional_blur":     "Filter/Directional Blur",
    "edge_detect":          "Filter/Edge Detect",
    "erode_dilate":         "Filter/Erode Dilate",
    "fast_blur":            "Filter/Fast Blur",
    "fast_defocus":         "Filter/Fast Defocus",
    "fast_gaussian":        "Filter/Fast Gaussian",
    "film_exponential_blur": "Filter/Film Exponential Blur",
    "film_grain":           "Filter/Film Grain",
    "film_sharpen":         "Filter/Film Sharpen",
    "film_soften":          "Filter/Film Soften",
    "gaussian_blur":        "Filter/Gaussian Blur",
    "grain":                "Filter/Grain",
    "median_filter":        "Filter/Median Filter",
    "mipmap_blur":          "Filter/Mipmap Blur",
    "sharpen":              "Filter/Sharpen",
    "tilt_shift":           "Filter/Tilt Shift",
    "unsharp_mask":         "Filter/Unsharp Mask",
    "vec4_median":          "Filter/Vec4 Median",
    "zdefocus":             "Filter/ZDefocus",
    # ── Generate ───────────────────────────────────────────────────────
    "billow_texture":       "Generate/Billow Texture",
    "caustics":             "Generate/Caustics",
    "curl_distortion":      "Generate/Curl Distortion",
    "flow_noise":           "Generate/Flow Noise",
    "gradient":             "Generate/Gradient",
    "image_gradient":       "Generate/Image Gradient",
    "marble":               "Generate/Marble",
    "normal_map":           "Generate/Normal Map",
    "perlin_clouds":        "Generate/Perlin Clouds",
    "radial_gradient":      "Generate/Radial Gradient",
    "sdf_shapes":           "Generate/SDF Shapes",
    "simplex_terrain":      "Generate/Simplex Terrain",
    "voronoi_cells":        "Generate/Voronoi Cells",
    "wood_grain":           "Generate/Wood Grain",
    # ── Mask ───────────────────────────────────────────────────────────
    "chroma_keyer":         "Mask/Chroma Keyer",
    "conditional":          "Mask/Conditional",
    "difference_key":       "Mask/Difference Key",
    "fix_pixels":           "Mask/Fix Pixels",
    "luma_keyer":           "Mask/Luma Keyer",
    "luminance_key":        "Mask/Luminance Key",
    "mask_from_color":      "Mask/Mask From Color",
    "normalize_mask":       "Mask/Normalize Mask",
    "threshold_mask":       "Mask/Threshold",
    # ── Distortion ─────────────────────────────────────────────────────
    "distortion_map":       "Distortion/Distortion Map",
    "optical_flow":         "Distortion/Optical Flow",
    "turbulent_displace":   "Distortion/Turbulent Displace",
    "vector_blur":          "Distortion/Vector Blur",
    # ── Latent ─────────────────────────────────────────────────────────
    "latent_blend":         "Latent/Latent Blend",
    "latent_scale":         "Latent/Latent Scale",
    # ── String ─────────────────────────────────────────────────────────
    "string_build":         "String/String Build",
    "string_case":          "String/String Case",
    "string_format":        "String/String Format",
    # ── Video ──────────────────────────────────────────────────────────
    "frame_blend":          "Video/Frame Blend",
    "frame_blend_weighted": "Video/Frame Blend Weighted",
    "motion_detect":        "Video/Motion Detect",
    "temporal_median":      "Video/Temporal Median",
    "time_echo":            "Video/Time Echo",
    # ── Educational ────────────────────────────────────────────────────
    "array_reduce":         "Educational/Array Reduce",
    "binding_access":       "Educational/Binding Access",
    "break_search":         "Educational/Break Search",
    "const_values":         "Educational/Constant Values",
    "matrix_transform":     "Educational/Matrix Transform",
    "multi_output":         "Educational/Multi Output",
    "multi_sample":         "Educational/Multi Sample",
    "recursive_pattern":    "Educational/Recursive Pattern",
    "sample_comparison":    "Educational/Sample Comparison",
    "temporal_functions":   "Educational/Temporal Functions",
    "temporal_ramp":        "Educational/Temporal Ramp",
    "ternary_chain":        "Educational/Ternary Chain",
    "user_function_lib":    "Educational/User Function Lib",
    "while_loop":           "Educational/While Loop",
}

# Cache: built once on first request, invalidated never (examples are static)
_snippets_cache = None


def _load_example_snippets():
    """Read all .tex files from examples/ and return {path: content} dict."""
    global _snippets_cache
    if _snippets_cache is not None:
        return _snippets_cache

    snippets = {}
    if not os.path.isdir(_EXAMPLES_DIR):
        _snippets_cache = snippets
        return snippets

    for fname in sorted(os.listdir(_EXAMPLES_DIR)):
        if not fname.endswith(".tex"):
            continue
        stem = fname[:-4]  # strip .tex
        category_name = _EXAMPLE_CATEGORIES.get(stem)
        if category_name is None:
            # Uncategorized example: auto-generate a name
            display = stem.replace("_", " ").title()
            category_name = f"Uncategorized/{display}"
        path_key = f"Examples/{category_name}"
        try:
            with open(os.path.join(_EXAMPLES_DIR, fname), "r", encoding="utf-8") as f:
                snippets[path_key] = f.read()
        except OSError:
            continue

    _snippets_cache = snippets
    return snippets


try:
    from aiohttp import web
    from server import PromptServer
    routes = PromptServer.instance.routes

    @routes.get("/tex_wrangle/snippets")
    async def get_snippets(request):
        """Return built-in example snippets as JSON."""
        return web.json_response(_load_example_snippets())

except (ImportError, AttributeError):
    # PromptServer or aiohttp not available (e.g. running tests or CLI mode)
    pass
