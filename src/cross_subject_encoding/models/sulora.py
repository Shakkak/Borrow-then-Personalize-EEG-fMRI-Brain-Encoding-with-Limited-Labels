"""A modern learned-adapter comparator for the personal map, adapted from two decoding papers.

SuLoRA (Klein et al., arXiv 2510.08059): a shared weight plus a rank-r adapter per known subject,
W_s = W + (alpha / r) (A_s B_s)^T, used here as the subject-specific output layer (as SuLoRA replaces BrainMagick's
subject layer) on the shared MLP core of `shared_core`. All donors are trained jointly. A new person is not covered by SuLoRA;
two routes follow the papers: a new adapter fitted on the person's k labelled images with everything shared frozen
(SuLoRA's suggested, untested transfer), and the donor adapter that fits those images best, copied and fine-tuned
(the calibration route of Lopes et al., arXiv 2606.16462). The shared part alone is the zero-shot route.
Adapted, not exact: both papers decode; this predicts responses (encoding).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from cross_subject_encoding.models.shared_core import SharedCore


class SuLoRAHead(nn.Module):
    """Shared linear read-out plus one rank-r adapter per subject (adapter output added, scaled by alpha / r)."""

    def __init__(self, latent: int, n_out: int, subjects: list[str], rank: int = 8, alpha: float | None = None, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.shared = nn.Linear(latent, n_out)
        self.rank, self.scale = rank, (rank if alpha is None else alpha) / rank  # alpha = rank by default: scale 1
        self.a = nn.ParameterDict({s: nn.Parameter(torch.randn(latent, rank, generator=g) * (2.0 / rank) ** 0.5) for s in subjects})
        self.b = nn.ParameterDict({s: nn.Parameter(torch.randn(rank, n_out, generator=g) * 0.01) for s in subjects})

    def forward(self, z: torch.Tensor, subject: str | None) -> torch.Tensor:
        out = self.shared(z)
        if subject is not None:
            out = out + self.scale * (z @ self.a[subject]) @ self.b[subject]
        return out

    def add_subject(self, name: str, like: str | None = None, seed: int = 0) -> None:
        """A new adapter: a copy of donor `like`, or freshly initialized as in SuLoRA."""
        if like is not None:
            self.a[name] = nn.Parameter(self.a[like].detach().clone())
            self.b[name] = nn.Parameter(self.b[like].detach().clone())
        else:
            g = torch.Generator().manual_seed(seed)
            self.a[name] = nn.Parameter(torch.randn(self.a[next(iter(self.a))].shape, generator=g) * (2.0 / self.rank) ** 0.5)
            self.b[name] = nn.Parameter(torch.randn(self.b[next(iter(self.b))].shape, generator=g) * 0.01)


class SuLoRAEncoder(nn.Module):
    def __init__(self, in_dim: int, n_out: int, subjects: list[str], rank: int = 8, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.core = SharedCore(in_dim)
        self.head = SuLoRAHead(self.core.net[-1].out_features, n_out, subjects, rank, seed=seed)

    def forward(self, x: torch.Tensor, subject: str | None) -> torch.Tensor:
        return self.head(self.core(x), subject)


def train_donors(model: SuLoRAEncoder, x: torch.Tensor, ys: dict[str, torch.Tensor], train_idx, hold_idx, epochs: int,
                 lr: float = 1e-3, batch: int = 256, seed: int = 0, log=print) -> dict:
    """All donors jointly on images they all saw: one core pass per batch, every donor's head. Keeps the epoch with the
    best median held-out correlation (as `shared_core.train_shared_images`)."""
    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, hold = {"score": -np.inf, "epoch": -1, "state": None}, torch.as_tensor(np.asarray(hold_idx))
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(np.asarray(train_idx))
        for i in range(0, len(order), batch):
            idx = torch.as_tensor(order[i:i + batch])
            z = model.core(x[idx])
            loss = sum(nn.functional.mse_loss(model.head(z, s), y[idx]) for s, y in ys.items()) / len(ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            z = model.core(x[hold])
            rs = [_corr(model.head(z, s), y[hold]) for s, y in ys.items()]
        score = float(np.median(np.concatenate(rs)))
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        log(f"  epoch {epoch + 1}/{epochs}: held-out median r {score:.4f}")
    model.load_state_dict(best["state"])
    return {"best_epoch": best["epoch"], "best_score": best["score"]}


def _corr(p: torch.Tensor, t: torch.Tensor) -> np.ndarray:
    p, t = p - p.mean(0), t - t.mean(0)
    return np.nan_to_num(((p * t).sum(0) / torch.sqrt((p ** 2).sum(0) * (t ** 2).sum(0) + 1e-12)).cpu().numpy())


def _fit_adapter(model: SuLoRAEncoder, z: torch.Tensor, y: torch.Tensor, name: str, like, wd: float, steps: int,
                 lr: float, seed: int) -> None:
    """Fit only the new person's adapter and an output bias shift on frozen latents z (full batch)."""
    model.head.add_subject(name, like=like, seed=seed)
    shift = nn.Parameter(torch.zeros(y.shape[1]))
    params = [model.head.a[name], model.head.b[name], shift]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    for _ in range(steps):
        loss = nn.functional.mse_loss(model.head(z, name) + shift, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.head.register_buffer(f"shift_{name}", shift.detach().clone(), persistent=False)


def adapt_new_person(model: SuLoRAEncoder, x_k: torch.Tensor, y_k: torch.Tensor, x_test: torch.Tensor, route: str,
                     wds=(1e-4, 1e-2, 1.0), steps: int = 300, lr: float = 1e-2, folds: int = 5, seed: int = 0) -> np.ndarray:
    """Predictions for a new person's test images. route: 'new' (fresh adapter), 'best_donor' (copy the donor adapter
    with the lowest error on the k images, then fine-tune), 'shared' (no adapter). The weight decay of the person's
    adapter is chosen by k-fold cross-validation on the k images; the shared core and read-out stay frozen."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        z_k, z_t = model.core(x_k), model.core(x_test)
    if route == "shared":
        with torch.no_grad():
            return model.head(z_t, None).numpy()
    donors = list(model.head.a.keys())
    like = None
    if route == "best_donor":
        with torch.no_grad():
            like = min(donors, key=lambda s: float(nn.functional.mse_loss(model.head(z_k, s), y_k)))
    elif route != "new":
        raise ValueError(route)
    split = np.array_split(np.random.default_rng(seed).permutation(len(z_k)), folds)
    errs = []
    for wd in wds:
        e = 0.0
        for j, held in enumerate(split):
            keep = np.setdiff1d(np.arange(len(z_k)), held)
            m = copy.deepcopy(model)
            _fit_adapter(m, z_k[keep], y_k[keep], "new_person", like, wd, steps, lr, seed + j)
            with torch.no_grad():
                pred = m.head(z_k[held], "new_person") + getattr(m.head, "shift_new_person")
            e += float(((pred - y_k[held]) ** 2).sum())
        errs.append(e)
    m = copy.deepcopy(model)
    _fit_adapter(m, z_k, y_k, "new_person", like, wds[int(np.argmin(errs))], steps, lr, seed)
    with torch.no_grad():
        return (m.head(z_t, "new_person") + getattr(m.head, "shift_new_person")).numpy()
