"""v0.52 sweep: lazy-analysis key overflow and the `_tex_time` payload."""
import math
import struct

from TEX_Wrangle import tex_lazy


class _StrictStruct:
    """`struct` as CPython builds that raise OverflowError when packing a finite value past
    the fp32 range (Linux does; some Windows builds silently return inf)."""
    unpack = staticmethod(struct.unpack)

    @staticmethod
    def pack(fmt, v):
        if fmt == "f" and math.isfinite(v) and abs(v) > 3.4028235677973366e38:
            raise OverflowError("float too large to pack with f format")
        return struct.pack(fmt, v)


def test_lazy_analysis_survives_a_param_beyond_fp32_range(monkeypatch):
    monkeypatch.setattr(tex_lazy, "struct", _StrictStruct)
    tex_lazy.clear_lazy_memo()
    code = "float k = $big; if ($big > 1.0) { @OUT = @A; } else { @OUT = @B; }"
    for v in (1e39, -1e39, 3e38 * 10, float("inf")):
        assert isinstance(tex_lazy.lazy_required_bindings(code, {"big": v}), frozenset)
    assert tex_lazy._param_key({"big": 1e39}) == tex_lazy._param_key({"big": 5e38 * 4})
    assert tex_lazy._fp32_binop("*", 1e38, 10.0) == math.inf
    assert tex_lazy._fp32(-1e39) == -math.inf


def test_time_context_drops_overflowing_and_non_finite_values():
    from TEX_Wrangle.tex_node import TEXWrangleNode
    parse = TEXWrangleNode._parse_time_context
    assert parse({"frame": 10 ** 400, "fps": 24, "time": 2.5}) == {"fps": 24.0, "time": 2.5}
    assert parse('{"frame": NaN, "fps": Infinity, "time": -Infinity}') is None
    assert parse({"frame": 12, "fps": float("nan")}) == {"frame": 12.0}
