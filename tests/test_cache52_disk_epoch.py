"""A warm disk hit answers what the cold compile answered; epoch lists cover the files that shape artifacts."""
import tempfile
from pathlib import Path

from helpers import *  # noqa: F401,F403
from TEX_Wrangle import tex_cache as TC
from TEX_Wrangle.tex_cache import TEXCache, parse_and_split


def test_disk_hit_returns_the_cold_compiles_sets():
    d = Path(tempfile.mkdtemp())
    bt = {"A": TEXType.VEC4, "B": TEXType.VEC4}
    code = "float dead = @B.r; float unused = $k; @OUT = @A;"
    cold = TEXCache(cache_dir=d).compile_tex(code, bt)
    warm = TEXCache(cache_dir=d).compile_tex(code, bt)
    assert "B" in cold[2], "premise: the dead read is in the pre-optimization reference set"
    assert warm[2] == cold[2]
    assert warm[3] == cold[3]
    assert warm[4] == cold[4]


def test_disk_entry_without_persisted_sets_is_a_miss():
    d = Path(tempfile.mkdtemp())
    bt = {"A": TEXType.VEC4}
    c = TEXCache(cache_dir=d)
    fp = c.fingerprint("@OUT = @A;", bt)
    prog = parse_and_split("@OUT = @A;", bt)
    c._save_to_disk(fp, prog, bt)                     # the older payload shape: no sets
    assert TEXCache(cache_dir=d)._load_from_disk(fp, bt) is None



def test_epoch_lists_cover_the_files_that_shape_artifacts():
    parts = TC.epoch_partitions()
    ast = {p.name for p in parts["ast"]}
    cg = {p.name for p in parts["codegen"]}
    assert "types.py" in ast
    assert "tex_fusion.py" in ast          # fused .pkl entries are gated by the AST epoch
    assert "stdlib_registry.py" in cg
    comp = {p.name for p in (Path(TC.__file__).parent / "tex_compiler").glob("*.py")}
    assert comp - {"__init__.py", "diagnostics.py"} <= ast
