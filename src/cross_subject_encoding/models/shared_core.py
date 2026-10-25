"""A shared image-feature network with a full linear readout for each donor."""
from __future__ import annotations

import torch
from torch import nn

from cross_subject_encoding.utils.config import CORE_WIDTH, LATENT_DIM


class SharedCore(nn.Module):
    def __init__(self, in_dim: int, width: int = CORE_WIDTH, latent: int = LATENT_DIM, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, latent))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FullReadout(nn.Module):
    def __init__(self, latent: int, n_out: int):
        super().__init__()
        self.lin = nn.Linear(latent, n_out)

    def forward(self, z):
        return self.lin(z)


class MultiSubjectEncoder(nn.Module):
    """One core and one full readout per source subject, trained jointly."""

    def __init__(self, in_dim: int, n_outputs: dict[str, int]):
        super().__init__()
        self.core = SharedCore(in_dim)
        self.heads = nn.ModuleDict({s: FullReadout(LATENT_DIM, n) for s, n in n_outputs.items()})

    def forward(self, x: torch.Tensor, subject: str) -> torch.Tensor:
        return self.heads[subject](self.core(x))

    def mean_readout(self) -> tuple[torch.Tensor, torch.Tensor]:
        ws = torch.stack([h.lin.weight.detach() for h in self.heads.values()])
        bs = torch.stack([h.lin.bias.detach() for h in self.heads.values()])
        return ws.mean(0), bs.mean(0)


def train_multisubject(model: MultiSubjectEncoder, data: dict[str, tuple[torch.Tensor, torch.Tensor]],
                       epochs: int, lr: float, batch: int, device: str, held_out: str) -> list[float]:
    """MSE on z-scored responses, every source subject in each epoch. Asserts the held-out subject is absent."""
    assert held_out not in data, f"{held_out} must never reach training in its own fold"
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for _ in range(epochs):
        total, n = 0.0, 0
        for s, (x, y) in data.items():
            perm = torch.randperm(len(x))
            for i in range(0, len(x), batch):
                idx = perm[i:i + batch]
                loss = nn.functional.mse_loss(model(x[idx].to(device), s), y[idx].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, n = total + loss.item() * len(idx), n + len(idx)
        history.append(total / n)
    return history


def train_shared_images(model: MultiSubjectEncoder, x: torch.Tensor, ys: dict[str, torch.Tensor], train_idx, hold_idx,
                        epochs: int, lr: float, batch: int, seed: int, log=print, checkpoint_dir=None, tag: str = "mlp",
                        snapshot_every: int = 5) -> dict:
    """Train on images every source person saw: one core pass per batch feeds every head.

    MSE averaged over people, AdamW as in `train_multisubject`. The epoch with the best median held-out
    correlation over all outputs of all people is kept, the rule GNet uses. x and ys are indexed
    by image and live on the training device. With `checkpoint_dir`: the core every
    `snapshot_every` epochs, the full state after every epoch (`_last.pt`) and the kept epoch (`_best.pt`).
    """
    from pathlib import Path

    ckpt = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt is not None:
        ckpt.mkdir(parents=True, exist_ok=True)
    import copy

    import numpy as np

    rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, history = {"score": -float("inf"), "epoch": -1, "state": None}, []
    hold = torch.as_tensor(np.asarray(hold_idx), device=x.device)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(np.asarray(train_idx))
        for i in range(0, len(train_idx), batch):
            idx = torch.as_tensor(order[i:i + batch], device=x.device)
            z = model.core(x[idx])
            loss = sum(nn.functional.mse_loss(model.heads[s](z), y[idx]) for s, y in ys.items()) / len(ys)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            z = model.core(x[hold])
            rs = []
            for s, y in ys.items():
                p, t = model.heads[s](z), y[hold]
                p, t = p - p.mean(0), t - t.mean(0)
                rs.append(((p * t).sum(0) / torch.sqrt((p ** 2).sum(0) * (t ** 2).sum(0) + 1e-12)).cpu().numpy())
        score = float(np.median(np.nan_to_num(np.concatenate(rs))))
        history.append({"epoch": epoch, "holdout_median_r": score})
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        if ckpt is not None:
            if (epoch + 1) % snapshot_every == 0 or epoch == epochs - 1:
                torch.save(model.core.state_dict(), ckpt / f"{tag}_core_e{epoch + 1:03d}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "history": history,
                        "best_score": best["score"], "best_epoch": best["epoch"]}, ckpt / f"{tag}_last.pt")
        log(f"  epoch {epoch + 1}/{epochs}: held-out median r {score:.4f}")
    model.load_state_dict(best["state"])
    if ckpt is not None:
        torch.save(best["state"], ckpt / f"{tag}_best.pt")
    return {"best_epoch": best["epoch"], "best_score": best["score"], "history": history}


@torch.no_grad()
def latent(model: MultiSubjectEncoder, x: torch.Tensor, batch: int = 2048) -> "np.ndarray":
    model.eval()
    return torch.cat([model.core(x[i:i + batch]) for i in range(0, len(x), batch)]).cpu().numpy()


def adapter_predictions(z_k, y_k, z_test, w_mean, b_mean, alphas, n_folds: int, seed: int) -> dict:
    """A new person's read-outs on a frozen core's latent, four ways.

    - `fresh`: ridge from the latent to the person's outputs, fitted on the k images alone;
    - `mean`: the nine source people's average read-out, no fitting (the core's own k = 0 prediction);
    - `norm`: the average read-out's prediction rescaled and shifted per output by least squares
      (two numbers per output, the normalization-only adapter);
    - `prior_ridge`: ridge on what the average read-out leaves unexplained, so the read-out shrinks
      towards the group's instead of towards zero (a small personal change to a shared map).
    z_k (k, d), y_k (k, outputs), z_test (n, d); w_mean (outputs, d), b_mean (outputs,) as numpy arrays.
    """
    import numpy as np

    from cross_subject_encoding.models.ridge import RidgePerOutput

    m_k, m_t = z_k @ w_mean.T + b_mean, z_test @ w_mean.T + b_mean
    folds = min(n_folds, len(z_k))
    mk_c = m_k - m_k.mean(0)
    a = (mk_c * (y_k - y_k.mean(0))).sum(0) / np.maximum((mk_c ** 2).sum(0), 1e-12)
    c = y_k.mean(0) - a * m_k.mean(0)
    return {"fresh": RidgePerOutput(alphas, folds, seed).fit(z_k, y_k).predict(z_test),
            "mean": m_t,
            "norm": a * m_t + c,
            "prior_ridge": m_t + RidgePerOutput(alphas, folds, seed).fit(z_k, y_k - m_k).predict(z_test)}


def prior_feature_ridge(m_k, m_t, f_k, f_t, y_k, alphas, n_folds: int, seed: int):
    """The group's prediction through the core as an offset, plus ridge on other features for the rest.

    m_k, m_t: the nine people's average read-out applied to the core's latent (k and test images);
    f_k, f_t: the features the correction is fitted on (the paper's 1,000 components, or those and the
    latent). With few images the ridge shrinks to the group's prediction; with many it can use what the
    256-wide core leaves out, so it should approach the person's own ridge at large k.
    """
    from cross_subject_encoding.models.ridge import RidgePerOutput

    return m_t + RidgePerOutput(alphas, min(n_folds, len(f_k)), seed).fit(f_k, y_k - m_k).predict(f_t)
