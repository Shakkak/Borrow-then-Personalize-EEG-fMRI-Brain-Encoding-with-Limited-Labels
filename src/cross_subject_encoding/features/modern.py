"""Spatial patch-grid features from recent vision transformers, for the modern-feature check (09m).

Pooled CLIP and DINOv2 tokens score below the paper's AlexNet maps as the feature base (0.519 against 0.669). Here
each chosen layer's patch tokens
keep their layout: reshaped to the model's patch grid and average-pooled to `pool` x `pool`, then the layers are
concatenated, the way the paper concatenates AlexNet's maps. The paper's own pipeline then follows unchanged
(standardize, polynomial kernel, kernel PCA to 1,000 components; `models.paper_linearizing`).

Models: DINOv2 ViT-B/14 (2023, reference), DINOv3 ViT-B/16 (Meta, August 2025; gated with manual approval on Hugging
Face, so it is skipped if the token has no access), SigLIP 2 ViT-B/16 (Google, February 2025, Apache-2.0).
"""
from __future__ import annotations

import math

import numpy as np
import torch

MODELS = {"dinov2": "facebook/dinov2-base", "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
          "siglip2": "google/siglip2-base-patch16-224"}
LAYERS = (4, 8, 12)  # hidden_states index: 0 is the patch embedding, n the output of block n (12-block models)
POOL = 7


def split_tokens(n_tokens: int) -> tuple[int, int]:
    """(prefix tokens, grid side) for a square patch grid after a few non-patch tokens: DINOv2 has a CLS token (257 = 1 +
    16^2), DINOv3 a CLS and 4 registers (201 = 5 + 14^2), SigLIP none (196 = 14^2)."""
    side = math.isqrt(n_tokens)
    while side > 0 and n_tokens - side * side > 16:  # at most a handful of CLS/register tokens
        side -= 1
    return n_tokens - side * side, side


def grid_features(hidden_states, layers=LAYERS, pool: int = POOL) -> torch.Tensor:
    """hidden_states: sequence of (batch, tokens, dim) tensors. Each chosen layer's patch tokens -> (batch, dim, side, side)
    -> average-pooled to (pool, pool) -> flattened; layers concatenated. Returns (batch, len(layers) * dim * pool^2)."""
    out = []
    for n in layers:
        h = hidden_states[n]
        prefix, side = split_tokens(h.shape[1])
        patches = h[:, prefix:].reshape(h.shape[0], side, side, h.shape[2]).permute(0, 3, 1, 2)
        out.append(torch.nn.functional.adaptive_avg_pool2d(patches.float(), pool).flatten(1))
    return torch.cat(out, 1)


def load(name: str, device: str, token: str | None = None):
    """(forward, processor): forward(pixel_values) returns the vision tower's hidden states."""
    from transformers import AutoImageProcessor, AutoModel
    model = AutoModel.from_pretrained(MODELS[name], token=token).eval().to(device)
    tower = model.vision_model if hasattr(model, "vision_model") else model  # SigLIP 2 holds text and vision towers
    processor = AutoImageProcessor.from_pretrained(MODELS[name], token=token)

    @torch.no_grad()
    def forward(pixel_values):
        return tower(pixel_values=pixel_values, output_hidden_states=True).hidden_states
    return forward, processor


def extract(paths, name: str, device: str, token: str | None = None, batch: int = 64, layers=LAYERS, pool: int = POOL,
            progress=None) -> np.ndarray:
    """(images, features) float16 grid features for the image files in `paths`, in order. Models are used one at a time:
    this call loads one, and frees it (and the GPU cache) before returning, so the next model starts on an empty GPU.
    A batch that runs out of GPU memory is retried at half the size (down to 1)."""
    import gc
    from PIL import Image
    forward, processor = load(name, device, token)
    rows, a = [], 0
    try:
        while a < len(paths):
            images = [Image.open(p).convert("RGB") for p in paths[a:a + batch]]
            try:
                x = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                    rows.append(grid_features(forward(x), layers, pool).cpu().numpy().astype(np.float16))
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
                continue
            a += len(images)
            if progress:
                progress(a, len(paths))
    finally:
        del forward, processor
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(rows)
