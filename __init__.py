"""
TEX Wrangle — Tensor Expression Language for ComfyUI.

A per-pixel kernel DSL for writing compact image/mask processing logic
directly inside ComfyUI. Inspired by Houdini VEX and Nuke BlinkScript.
"""

__version__ = "0.3.1"

from .tex_node import TEXWrangleNode

NODE_CLASS_MAPPINGS = {
    "TEX_Wrangle": TEXWrangleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TEX_Wrangle": "TEX Wrangle",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
