"""How much does choosing the best of several options flatter it? Choose on half the people, score on
the other half, over every split of the 10 THINGS-EEG2 people. Reads the per-person scores of 18h (feature sets) and
18j (stacks); writes results/selection_check/summary.json.

    python scripts/selection_check.py
"""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "results"
OUT = ROOT / "results" / "selection_check"


def table(path, method, k, column="variant"):
    """people x options matrix of centred top-1, averaged over draws."""
    f = pd.read_csv(path)
    f = f[(f.method == method) & (f.k.astype(str) == str(k))]
    return f.groupby(["person", column])["top1"].mean().unstack()


def check(t: pd.DataFrame, fixed: str) -> dict:
    people = list(t.index)
    chosen, held_gain, optimism = [], [], []
    # each unordered partition once (126 for 10 people), then both directions: 252 ordered splits
    for half in combinations(range(len(people)), len(people) // 2):
        if 0 not in half:
            continue
        a = [people[i] for i in half]
        b = [p for p in people if p not in a]
        for pick, other in ((a, b), (b, a)):
            best = t.loc[pick].mean().idxmax()
            chosen.append(best)
            held_gain.append(t.loc[other, best].mean() - t.loc[other, fixed].mean())
            optimism.append((t.loc[pick, best].mean() - t.loc[other, best].mean())
                            - (t.loc[pick, fixed].mean() - t.loc[other, fixed].mean()))
    held_gain, optimism = np.array(held_gain), np.array(optimism)
    full_best = t.mean().idxmax()
    return {"options": {c: float(t[c].mean()) for c in t.columns}, "full_sample_best": full_best,
            "full_sample_gain_over_fixed": float(t[full_best].mean() - t[fixed].mean()),
            "chosen_share": {c: chosen.count(c) / len(chosen) for c in t.columns},
            "held_out_gain_over_fixed": {"mean": float(held_gain.mean()), "p2.5": float(np.percentile(held_gain, 2.5)),
                                         "p97.5": float(np.percentile(held_gain, 97.5))},
            "optimism": {"mean": float(optimism.mean()), "p2.5": float(np.percentile(optimism, 2.5)),
                         "p97.5": float(np.percentile(optimism, 97.5))},
            "splits": len(chosen)}


if __name__ == "__main__":
    h = K / "18h_eeg2_decoding_features" / "18h_scores.csv"
    j = K / "18j_eeg2_decoding_system" / "18j_scores.csv"
    summary = {"note": "centred top-1 of 200, per person averaged over draws; percentiles over splits",
               "18h_borrowing_k0": check(table(h, "avg", 0), "alexnet"),
               "18h_map_k250": check(table(h, "h4b", 250), "alexnet"),
               "18h_own_full": check(table(h, "own_full", "all"), "alexnet")}
    for k in (250, 500, 1000):
        summary[f"18j_stack_k{k}"] = check(table(j, "stack", k), "alexnet")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    for name, r in summary.items():
        if isinstance(r, dict):
            print(f"{name}: best {r['full_sample_best']} (+{r['full_sample_gain_over_fixed']:.1f}); chosen "
                  f"{ {c: round(v, 2) for c, v in r['chosen_share'].items()} }; held-out gain {r['held_out_gain_over_fixed']['mean']:.1f} "
                  f"[{r['held_out_gain_over_fixed']['p2.5']:.1f}, {r['held_out_gain_over_fixed']['p97.5']:.1f}]; optimism {r['optimism']['mean']:.2f}")
