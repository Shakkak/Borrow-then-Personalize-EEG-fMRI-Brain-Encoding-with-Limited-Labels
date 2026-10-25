"""Modern grid features: token layouts of DINOv2, DINOv3 and SigLIP 2, and spatial pooling that keeps layout."""
import torch

from cross_subject_encoding.features import modern as mf


def test_token_layouts():
    assert mf.split_tokens(257) == (1, 16)   # DINOv2 ViT-B/14 at 224: CLS + 16 x 16
    assert mf.split_tokens(201) == (5, 14)   # DINOv3 ViT-B/16: CLS + 4 registers + 14 x 14
    assert mf.split_tokens(196) == (0, 14)   # SigLIP 2 B/16: patches only


def test_grid_features_pool_each_layer_and_keep_layout():
    b, dim = 2, 3
    left_right = torch.zeros(b, 5 + 14 * 14, dim)
    grid = torch.zeros(14, 14); grid[:, 7:] = 1.0  # right half bright
    left_right[:, 5:, 0] = grid.flatten()
    hs = [None] * 13
    for n in (4, 8, 12):
        hs[n] = left_right
    f = mf.grid_features(hs, layers=(4, 8, 12), pool=7)
    assert f.shape == (b, 3 * dim * 49)
    first = f[0, :49].reshape(7, 7)  # layer 4, channel 0
    assert torch.allclose(first[:, :3], torch.zeros(7, 3)) and torch.allclose(first[:, 4:], torch.ones(7, 3))


def test_extract_frees_the_model_and_halves_batches_on_out_of_memory(monkeypatch, tmp_path):
    """One model at a time: extract() drops it before returning; an OOM batch is retried at half the size."""
    import numpy as np
    from PIL import Image
    paths = []
    for i in range(5):
        Image.new("RGB", (8, 8)).save(tmp_path / f"{i}.png")
        paths.append(tmp_path / f"{i}.png")
    calls = {"sizes": []}

    def fake_forward(x):
        calls["sizes"].append(len(x))
        if len(x) > 2:
            raise torch.cuda.OutOfMemoryError("fake")
        return [torch.zeros(len(x), 196, 3)] * 13
    processor = lambda images, return_tensors: {"pixel_values": torch.zeros(len(images), 3, 8, 8)}
    monkeypatch.setattr(mf, "load", lambda name, device, token: (fake_forward, processor))
    out = mf.extract(paths, "siglip2", "cpu", batch=4, layers=(4,), pool=7)
    assert out.shape == (5, 3 * 49) and calls["sizes"][:2] == [4, 2] and max(calls["sizes"][1:]) <= 2
