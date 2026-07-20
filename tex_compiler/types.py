"""
TEX type vocabulary — the dependency-free core the whole pipeline agrees on.

`TEXType` and its lookup/channel tables live here (not in `type_checker`) so that
the modules that only need to know *what a vec3 is* — optimizer, stdlib_signatures,
interpreter, codegen, marshalling, memory, fusion — depend on this leaf, never on
the type-*checking* logic. This is the stable compiler↔runtime type API: the
runtime imports the IR (`ast_nodes`) + these types, never the checker.

STR-1: relocating these symbols also breaks the `stdlib_signatures ↔ type_checker`
import cycle (stdlib_signatures now imports its `TEXType` from here). Keep this
module dependency-free — it must import nothing from the package (only stdlib
`enum`/`dataclasses`), or the leaf property is lost.
"""
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass


class TEXType(Enum):
    INT = "int"
    FLOAT = "float"
    VEC2 = "vec2"
    VEC3 = "vec3"
    VEC4 = "vec4"
    MAT3 = "mat3"
    MAT4 = "mat4"
    STRING = "string"
    ARRAY = "array"  # fixed-size array; metadata in TEXArrayType
    VOID = "void"    # for statements

    @property
    def is_scalar(self) -> bool:
        return self in (TEXType.INT, TEXType.FLOAT)

    @property
    def is_vector(self) -> bool:
        return self in (TEXType.VEC2, TEXType.VEC3, TEXType.VEC4)

    @property
    def is_matrix(self) -> bool:
        return self in (TEXType.MAT3, TEXType.MAT4)

    @property
    def is_string(self) -> bool:
        return self == TEXType.STRING

    @property
    def is_array(self) -> bool:
        return self == TEXType.ARRAY

    @property
    def is_numeric(self) -> bool:
        return self in (TEXType.INT, TEXType.FLOAT, TEXType.VEC2, TEXType.VEC3,
                        TEXType.VEC4, TEXType.MAT3, TEXType.MAT4)

    @property
    def channels(self) -> int:
        if self == TEXType.VEC2:
            return 2
        elif self == TEXType.VEC3:
            return 3
        elif self == TEXType.VEC4:
            return 4
        elif self == TEXType.MAT3:
            return 9
        elif self == TEXType.MAT4:
            return 16
        return 1

    @property
    def mat_size(self) -> int:
        """Matrix dimension (3 for mat3, 4 for mat4). 0 for non-matrix types."""
        if self == TEXType.MAT3:
            return 3
        elif self == TEXType.MAT4:
            return 4
        return 0


@dataclass
class TEXArrayType:
    """Metadata for an array type: element type + fixed size."""
    element_type: TEXType  # FLOAT, INT, VEC3, VEC4, or STRING
    size: int


# ── DATA-3: ARRAY values on a host wire ───────────────────────────────────────
#
# Whether ARRAY values may cross a host wire — an input `a@name` (curve / palette / histogram)
# or an array output. OFF by default = the ComfyUI convention (its wire has no ARRAY type), so
# the checker rejects an array output (E3203) and marshalling never infers ARRAY — byte-identical
# to before. A standalone/engine host that CAN carry arrays turns it on via
# `tex_marshalling.set_egress_profile("engine")`. Process-global, set once by the host (the
# egress-profile model), read at compile + marshalling time. Deliberately NOT in the program
# fingerprint: the only compile that differs is an array-OUTPUT program, which the comfy egress
# rejects anyway (the always-on guard) — so caches need no separate namespace and existing disk
# caches survive an upgrade (invariant #7). Single-cook-thread, set-once (the ENG-9 IAI posture).
_ARRAY_WIRES = False


def set_array_wires(enabled: bool) -> None:
    """Enable/disable ARRAY host wires (DATA-3). Host-level; the egress profile drives it."""
    global _ARRAY_WIRES
    _ARRAY_WIRES = bool(enabled)


def array_wires_enabled() -> bool:
    """True when ARRAY values may cross a host wire (DATA-3). False under ComfyUI (default)."""
    return _ARRAY_WIRES


TYPE_NAME_MAP = {
    "float": TEXType.FLOAT,
    "int": TEXType.INT,
    "vec2": TEXType.VEC2,
    "vec3": TEXType.VEC3,
    "vec4": TEXType.VEC4,
    "mat3": TEXType.MAT3,
    "mat4": TEXType.MAT4,
    "string": TEXType.STRING,
}

_VEC_RANK = {TEXType.VEC2: 0, TEXType.VEC3: 1, TEXType.VEC4: 2}
_VEC_SIZE_TYPE = {2: TEXType.VEC2, 3: TEXType.VEC3, 4: TEXType.VEC4}

# Valid swizzle characters and their meanings
CHANNEL_MAP = {
    "r": 0, "g": 1, "b": 2, "a": 3,
    "x": 0, "y": 1, "z": 2, "w": 3,
}

# Valid multi-channel swizzles
VALID_SWIZZLES = {
    "rg", "rb", "ra", "gr", "gb", "ga", "br", "bg", "ba", "ar", "ag", "ab",
    "xy", "xz", "xw", "yx", "yz", "yw", "zx", "zy", "zw", "wx", "wy", "wz",
    "rgb", "rgba", "xyz", "xyzw",
    "bgr", "abgr",
}
