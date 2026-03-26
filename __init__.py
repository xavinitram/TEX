"""
TEX Wrangle — Tensor Expression Language for ComfyUI.

A per-pixel kernel DSL for writing compact image/mask processing logic
directly inside ComfyUI. Inspired by Houdini VEX and Nuke BlinkScript.
"""

__version__ = "0.9.0"

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
    "auto_levels":          "Color/Auto Levels",
    "brightness_contrast":  "Color/Brightness Contrast",
    "channel_swap":         "Color/Channel Swap",
    "color_grade":          "Color/Color Grade",
    "color_mix":            "Color/Color Mix",
    "grayscale":            "Color/Grayscale",
    "hue_shift":            "Color/Hue Shift",
    "invert":               "Color/Invert",
    "levels":               "Color/Levels",
    "tone_map":             "Color/Tone Map",
    "chromatic_aberration":  "Effects/Chromatic Aberration",
    "lens_distortion":      "Effects/Lens Distortion",
    "pixelate":             "Effects/Pixelate",
    "swirl":                "Effects/Swirl",
    "vignette":             "Effects/Vignette",
    "bilateral_approx":     "Filter/Bilateral Approx",
    "blur":                 "Filter/Blur",
    "edge_detect":          "Filter/Edge Detect",
    "median_filter":        "Filter/Median Filter",
    "sharpen":              "Filter/Sharpen",
    "unsharp_mask":         "Filter/Unsharp Mask",
    "vec4_median":          "Filter/Vec4 Median",
    "gradient":             "Generate/Gradient",
    "normal_map":           "Generate/Normal Map",
    "perlin_clouds":        "Generate/Perlin Clouds",
    "radial_gradient":      "Generate/Radial Gradient",
    "simplex_terrain":      "Generate/Simplex Terrain",
    "conditional":          "Mask/Conditional",
    "mask_from_color":      "Mask/Mask From Color",
    "threshold_mask":       "Mask/Threshold",
    "latent_blend":         "Latent/Latent Blend",
    "latent_scale":         "Latent/Latent Scale",
    "string_build":         "String/String Build",
    "string_case":          "String/String Case",
    "frame_blend":          "Video/Frame Blend",
    "motion_detect":        "Video/Motion Detect",
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
