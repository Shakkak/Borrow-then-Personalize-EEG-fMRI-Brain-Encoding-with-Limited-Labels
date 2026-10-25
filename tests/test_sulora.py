import numpy as np
import torch

from cross_subject_encoding.models import sulora as su


def people(g, n_img=600, d=20, n_out=12, n_people=6, rank=2):
    """Shared nonlinear map plus a low-rank deviation per person."""
    x = torch.as_tensor(g.normal(size=(n_img, d)), dtype=torch.float32)
    w1 = g.normal(size=(d, 16)) / np.sqrt(d)
    w2 = g.normal(size=(16, n_out)) / 4
    h = np.tanh(x.numpy() @ w1)
    ys = {}
    for s in range(n_people):
        dev = g.normal(size=(16, rank)) @ g.normal(size=(rank, n_out)) * 0.25
        ys[f"p{s}"] = torch.as_tensor(h @ (w2 + dev) + 0.05 * g.normal(size=(n_img, n_out)), dtype=torch.float32)
    return x, ys


def corr(p, y):
    p, y = p - p.mean(0), y - y.mean(0)
    return float(np.mean((p * y).sum(0) / np.sqrt((p ** 2).sum(0) * (y ** 2).sum(0))))


def test_head_starts_near_shared_and_copies_donors():
    torch.manual_seed(0)
    head = su.SuLoRAHead(8, 5, ["a", "b"], rank=2)
    z = torch.randn(64, 8)
    # SuLoRA initialises B near zero, so an adapter starts as a small change to the shared read-out
    gap = (head(z, "a") - head(z, None)).abs().mean()
    assert gap < 0.1 * head(z, None).abs().mean()
    head.add_subject("c", like="a")
    assert torch.allclose(head(z, "c"), head(z, "a"))


def test_new_person_adapter_beats_shared_only():
    g = np.random.default_rng(0)
    torch.manual_seed(0)
    x, ys = people(g)
    donors = {s: y for s, y in ys.items() if s != "p5"}
    model = su.SuLoRAEncoder(x.shape[1], 12, list(donors), rank=4, seed=0)
    su.train_donors(model, x, donors, np.arange(500), np.arange(500, 600), epochs=40, lr=3e-3, batch=64, log=lambda m: None)
    k = np.arange(150)
    test = np.arange(400, 600)
    y_t = ys["p5"][test].numpy()
    shared = su.adapt_new_person(model, x[k], ys["p5"][k], x[test], "shared")
    new = su.adapt_new_person(model, x[k], ys["p5"][k], x[test], "new", steps=200)
    best = su.adapt_new_person(model, x[k], ys["p5"][k], x[test], "best_donor", steps=200)
    assert corr(new, y_t) > corr(shared, y_t) and corr(best, y_t) > corr(shared, y_t)
    assert all(p.requires_grad is False for p in model.parameters())  # the shared model was frozen, not changed
