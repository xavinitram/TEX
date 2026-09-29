"""v0.52 sweep: version ordering used by the language-version gates."""
from TEX_Wrangle import tex_api


def test_a_three_part_version_does_not_order_after_its_two_part_base():
    v = tex_api._ver_tuple
    assert not v(tex_api.LANGUAGE_VERSION + ".0") > v(tex_api.LANGUAGE_VERSION)
    assert v("0.25.0") == v("0.25") and v("1.0.0") == v("1")
    assert v("0.25.1") > v("0.25")
    assert v("0.40.2") > v("0.40") and v("0.9") < v("0.25")
