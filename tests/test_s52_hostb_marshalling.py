"""v0.52 sweep: marshalling rows (IS_CHANGED fingerprint, promise device, `v$` alias)."""
import torch

from TEX_Wrangle import tex_marshalling as M


def test_fingerprint_sees_a_small_off_stride_edit_in_a_large_image():
    img = torch.rand(1, 1024, 1024, 3, generator=torch.Generator().manual_seed(1))
    base = M.tensor_fingerprint(img)
    assert M.tensor_fingerprint(img.clone()) == base
    edited = img.clone()
    edited.view(-1)[5] += 0.02           # index 5 is off the 256-sample stride
    assert M.tensor_fingerprint(edited) != base


def test_fingerprint_sees_content_moved_between_segments():
    img = torch.rand(1, 1024, 1024, 3, generator=torch.Generator().manual_seed(2))
    base = M.tensor_fingerprint(img)
    moved = img.clone()
    flat = moved.view(-1)
    a, b = slice(5, 8), slice(2_000_005, 2_000_008)
    tmp = flat[a].clone()
    flat[a] = flat[b]
    flat[b] = tmp
    assert not torch.equal(moved, img)
    assert M.tensor_fingerprint(moved) != base


def test_fingerprint_is_layout_independent_and_handles_tiny_tensors():
    t = torch.rand(2, 8, 8, 4)
    assert M.tensor_fingerprint(t) == M.tensor_fingerprint(t.permute(0, 3, 1, 2).contiguous().permute(0, 2, 3, 1))
    for shape in ((0,), (1,), (3,), (256,), (257,), (2, 129)):
        a = torch.arange(int(torch.Size(shape).numel()), dtype=torch.float32).reshape(shape)
        assert M.tensor_fingerprint(a) == M.tensor_fingerprint(a.clone())
    small = torch.zeros(300)
    other = small.clone()
    other[7] = 1.0
    assert M.tensor_fingerprint(small) != M.tensor_fingerprint(other)
    assert isinstance(M.tensor_fingerprint(torch.ones(4, 300, dtype=torch.bool)), str)


def test_promise_device_accepts_an_unindexed_declaration():
    d = torch.device
    assert M._device_matches("cuda", d("cuda:0"))
    assert M._device_matches("cuda:0", d("cuda:0"))
    assert not M._device_matches("cuda:1", d("cuda:0"))
    assert not M._device_matches("cuda", d("cpu"))
    assert M._device_matches("cpu", d("cpu"))
    assert not M._device_matches("not a device", d("cpu"))


def test_bare_v_param_hint_is_an_alias_of_v3():
    assert M.convert_param_value("1, 2, 3", {"type_hint": "v"}, "p") == [1.0, 2.0, 3.0]
    assert M.convert_param_value("1", {"type_hint": "v"}, "p") == [1.0, 0.0, 0.0]
    assert M.convert_param_value(0.5, {"type_hint": "v"}, "p") == 0.5
