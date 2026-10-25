"""Can a person's channel-mixing map identify them across recording sessions? THINGS-EEG2, 09h's maps.

Run from the repository root (numpy, matplotlib; no EEG data):

    python scripts/fingerprint_maps.py

Reads `results/09h_eeg2_h4b_sessions/09h_maps.npz`: per person, four single-repetition maps (E1, E2 from each
image's earlier session; L1, L2 from its later one), one 17x17 map per 100 ms window. A probe map from one session is
matched to all ten people's reference maps from the other session by cosine similarity of the off-diagonal part;
correct if the best match is the same person (chance 10%). Writes `results/figures/fingerprint.png` and `.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
EARLY, LATE = ("E1", "E2"), ("L1", "L2")


def offdiag(maps: np.ndarray, windows: list[int]) -> np.ndarray:
    """(people, windows, ch, ch) -> (people, features): off-diagonal entries of the chosen windows, concatenated."""
    off = ~np.eye(maps.shape[-1], dtype=bool)
    return np.concatenate([maps[:, w][:, off] for w in windows], axis=1)


def accuracy(probe: np.ndarray, ref: np.ndarray, labels: np.ndarray | None = None) -> float:
    p = probe / np.linalg.norm(probe, axis=1, keepdims=True)
    r = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    truth = np.arange(len(p)) if labels is None else labels
    return float(np.mean(np.argmax(p @ r.T, axis=1) == truth))


def cross_session(maps: np.ndarray, names: list[str], windows: list[int], labels=None) -> float:
    idx = {n: i for i, n in enumerate(names)}
    accs = [accuracy(offdiag(maps[:, idx[a]], windows), offdiag(maps[:, idx[b]], windows), labels)
            for a in EARLY for b in LATE] + [accuracy(offdiag(maps[:, idx[b]], windows), offdiag(maps[:, idx[a]], windows), labels)
                                             for a in EARLY for b in LATE]
    return float(np.mean(accs))


def main() -> None:
    z = np.load(ROOT / "results/09h_eeg2_h4b_sessions/09h_maps.npz")
    maps, names = z["maps"].astype(np.float64), [str(n) for n in z["names"]]
    n_people, n_windows = maps.shape[0], maps.shape[2]
    pooled = [2, 3, 4]  # 0-300 ms (window w covers -200 + 100 w to -100 + 100 w ms)
    acc = cross_session(maps, names, pooled)
    idx = {n: i for i, n in enumerate(names)}
    same = np.mean([accuracy(offdiag(maps[:, idx[a]], pooled), offdiag(maps[:, idx[b]], pooled))
                    for a, b in (("E1", "E2"), ("E2", "E1"), ("L1", "L2"), ("L2", "L1"))])
    rng = np.random.default_rng(13)
    null = np.array([cross_session(maps, names, pooled, rng.permutation(n_people)) for _ in range(10000)])
    per_window = [cross_session(maps, names, [w]) for w in range(n_windows)]
    out = {"cross_session_0_300ms": acc, "same_session_0_300ms": float(same), "chance": 1 / n_people,
           "p_permutation": float((np.sum(null >= acc) + 1) / (len(null) + 1)), "null_mean": float(null.mean()),
           "per_window": {f"{-200 + 100 * w} to {-100 + 100 * w} ms": round(a, 3) for w, a in enumerate(per_window)},
           "people": n_people, "note": "post hoc, development maps (09h)"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fingerprint.json").write_text(json.dumps(out, indent=1))
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    centres = np.arange(-150, 800, 100)
    ax.axvspan(100, 200, color="#EFEFEF", zorder=0)
    ax.plot(centres, per_window, "o-", color="#C8102E", ms=3.5, lw=1.3, label="map from the other session")
    ax.axhline(1 / n_people, color="#7A7A7A", ls=(0, (3, 2)), lw=1, label="chance")
    ax.set(xlabel="window centre (ms after image onset)", ylabel="people identified", ylim=(0, 1.05),
           title=f"THINGS-EEG2: identifying a person by their map across sessions (0–300 ms pooled: {acc:.0%})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "fingerprint.png", dpi=200, bbox_inches="tight")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
