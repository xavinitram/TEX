"""tex_buffers: an owned copy is contiguous, independent and not inference-flagged."""
import torch

from TEX_Wrangle import tex_buffers as B


def test_owned_copy_of_a_permuted_view_is_contiguous_and_independent():
    with torch.inference_mode():
        src = torch.rand(1, 6, 5, 3)
    view = src.permute(0, 3, 1, 2)
    out = B._owned_copy(view)
    assert out.is_contiguous() and not out.is_inference()
    assert out.shape == view.shape and torch.equal(out, view)
    assert out.untyped_storage().data_ptr() != src.untyped_storage().data_ptr()


def test_owned_copy_of_a_contiguous_tensor_is_a_copy():
    src = torch.rand(1, 4, 4, 3)
    out = B._owned_copy(src)
    assert torch.equal(out, src) and out.data_ptr() != src.data_ptr()
