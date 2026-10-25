"""Shared loaders for the paper's figures (read by `plot_paper_v2.py`), and a first set of summary figures
drawn from the result files only: no EEG data, no new fits. The figures it writes itself are not used in the paper.

Run from the repository root (needs numpy, matplotlib):

    python scripts/plot_paper_figures.py
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results"
OUT = ROOT / "results" / "figures"
T95 = {9: 1.833, 21: 1.721}  # one-sided 95% t quantiles for 10 and 22 people (development bounds only)
LABELS = {"avg_sources": "others' average", "gu_window": "Gu ensemble", "linear_converter": "converter",
          "own_ridge": "own ridge", "own_ridge_oc": "own ridge + map", "generic": "generic map"}


def load(path: str):
    return json.loads((RUNS / path).read_text())


def rows(path: str):
    with open(RUNS / path) as f:
        return list(csv.DictReader(f))


def per_person_from_decision(decision: dict) -> tuple[list[str], np.ndarray]:
    people = sorted(decision["d"])
    return people, np.array([decision["d"][p] for p in people])


def h8_differences() -> dict[str, np.ndarray]:
    """08m: borrowed minus own-data, per person, condition 'k' (whitening from the k labelled trials only)."""
    m = defaultdict(list)
    for r in rows("08m_eeg1_h8_confirmation/08m_scores.csv"):
        if r["condition"] == "k":
            m[(r["person"], r["method"], r["k"])].append(float(r["r_100_200ms"]))
    people = sorted({p for p, _, _ in m})
    score = lambda p, meth: mean(mean(m[(p, meth, k)]) for k in ("250", "500"))
    out = {}
    for a, b in [("avg_sources", "own_ridge"), ("avg_sources", "own_ridge_oc"), ("gu_window", "own_ridge"),
                 ("gu_window", "own_ridge_oc")]:
        out[f"{a} - {b}"] = np.array([score(p, a) - score(p, b) for p in people])
    return out


def alignment_differences(path: str) -> dict[str, np.ndarray]:
    """09i/09j: H4b minus others' average and minus Procrustes per window, per person, mean over k = 250/500."""
    m = defaultdict(list)
    for r in rows(path):
        m[(r["person"], r["method"], r["k"])].append(float(r["r_100_200ms"]))
    people = sorted({p for p, _, _ in m})
    score = lambda p, meth, k: mean(m[(p, meth, "all" if meth == "avg" else k)])
    out = {}
    for name, other in [("avg", "avg"), ("procrustes", "procrustes_w100")]:
        out[name] = np.array([mean(score(p, "h4b", k) - score(p, other, k) for k in ("250", "500")) for p in people])
    return out


def dev_lower(x: np.ndarray) -> float:
    return float(x.mean() - T95[len(x) - 1] * x.std(ddof=1) / math.sqrt(len(x)))


def strip(ax, y: int, values: np.ndarray, lower: float, color: str, rng: np.random.Generator):
    jitter = rng.uniform(-0.18, 0.18, len(values))
    ax.scatter(values, y + jitter, s=12, color=color, alpha=0.55, linewidths=0)
    ax.plot([lower, values.mean()], [y, y], color="black", lw=2.2, solid_capstyle="butt")
    ax.plot(values.mean(), y, "D", color="black", ms=5)


def figure2(summary: dict) -> None:
    rng = np.random.default_rng(13)
    core = load("08e_eeg1_confirmation/08e_decision.json")
    h8 = load("08m_eeg1_h8_confirmation/08m_decision.json")
    h4b = load("08n_eeg1_h4b_confirmation/08n_decision.json")
    delta = core["delta"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=False)
    panels = []
    _, d = per_person_from_decision(core)
    comps = list(core["per_comparator"])
    panels.append(("A  Shared core (08e):\nrejected", [(f"core − {LABELS[c]}", d[:, i], core["per_comparator"][c]["lower"])
                                                      for i, c in enumerate(comps)]))
    diffs = h8_differences()
    names = {"avg_sources - own_ridge": "average − own ridge", "avg_sources - own_ridge_oc": "average − own ridge + map",
             "gu_window - own_ridge": "Gu − own ridge", "gu_window - own_ridge_oc": "Gu − own ridge + map"}
    panels.append(("B  Borrowing vs own data\n(H8, 08m): supported",
                   [(names[c], diffs[c], h8["conditions"]["k"]["contrasts"][c]["lower"]) for c in names]))
    _, d = per_person_from_decision(h4b)
    comps = list(h4b["per_comparator"])
    panels.append(("C  Personal map\n(H4b, 08n): supported",
                   [(f"H4b − {LABELS[c]}", d[:, i], h4b["per_comparator"][c]["lower"]) for i, c in enumerate(comps)]))
    colors = plt.cm.tab10.colors
    for ax, (title, items) in zip(axes, panels):
        for j, (label, values, lower) in enumerate(items):
            strip(ax, j, values, lower, colors[j % 10], rng)
        ax.axvline(0, color="grey", lw=0.8)
        ax.axvline(delta, color="crimson", lw=1, ls="--")
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([lab for lab, _, _ in items], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("difference in r, 100–200 ms (per person)")
        summary["fig2"][title] = {lab: {"mean": float(v.mean()), "lower": float(lo), "positive": int((v > 0).sum()),
                                        "n": len(v)} for lab, v, lo in items}
    axes[0].text(delta, len(core['per_comparator']) - 0.6, f" δ = {delta:.4f}", color="crimson", fontsize=8)
    fig.suptitle("THINGS-EEG1, 24 reserved people, prespecified: dots = people, bar = mean to its corrected "
                 "lower bound", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "paper_fig2_eeg1_confirmations.png", dpi=170)
    plt.close(fig)


def figure3(summary: dict) -> None:
    dec = load("09l_alljoined_confirmation/09l_decision.json")["primary"]
    people = sorted(dec["d"], key=int)
    d = np.array([dec["d"][p] for p in people])
    order = np.argsort(d[:, 2])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(people))
    ax.bar(x - 0.2, d[order, 2], 0.4, color="tab:blue", label="C3: H4b − Procrustes per window (supported)")
    ax.bar(x + 0.2, d[order, 0], 0.4, color="tab:orange", label="C1: H4b − others' average (inconclusive)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axhline(dec["delta"], color="crimson", ls="--", lw=1, label=f"δ = {dec['delta']:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"sub-{int(people[i]):02d}" for i in order], rotation=60, fontsize=8)
    ax.set_ylabel("difference in r, 100–200 ms")
    c1, c3 = dec["claims"]["C1"], dec["claims"]["C3"]
    ax.set_title(f"Alljoined-1.6M, 16 fresh people, consumer headset (P-6)\nC3 mean {c3['mean']:+.3f} "
                 f"[lower {c3['lower']:+.3f}], {c3['positive_targets']}/16; C1 mean {c1['mean']:+.3f} "
                 f"[lower {c1['lower']:+.3f}], {c1['positive_targets']}/16", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "paper_fig3_alljoined_procrustes.png", dpi=170)
    plt.close(fig)
    summary["fig3"] = {c: {k: dec["claims"][c][k] for k in ("mean", "lower", "positive_targets", "verdict")}
                       for c in ("C1", "C3")}


def figure6(summary: dict) -> None:
    rng = np.random.default_rng(14)
    eeg1_h4b = load("08n_eeg1_h4b_confirmation/08n_decision.json")
    eeg2_h4b = load("09a_eeg2_h4b_replication/09a_decision.json")
    allj = load("09l_alljoined_confirmation/09l_decision.json")["primary"]
    eeg1_al = alignment_differences("09i_eeg1_alignment_controls/09i_scores.csv")
    eeg2_al = alignment_differences("09j_eeg2_alignment_controls/09j_scores.csv")
    col = lambda dec, c: per_person_from_decision(dec)[1][:, list(dec["per_comparator"]).index(c)]
    a1 = np.array([allj["d"][p] for p in sorted(allj["d"], key=int)])
    panels = [
        ("H4b − others' average", [
            ("THINGS-EEG1 (24, confirmation)", col(eeg1_h4b, "avg_sources"), eeg1_h4b["per_comparator"]["avg_sources"]["lower"], eeg1_h4b["delta"]),
            ("THINGS-EEG2 (10, replication)", col(eeg2_h4b, "avg_sources"), eeg2_h4b["per_comparator"]["avg_sources"]["lower"], eeg2_h4b["delta"]),
            ("Alljoined (16, confirmation)", a1[:, 0], allj["claims"]["C1"]["lower"], allj["delta"])]),
        ("H4b − Procrustes per window", [
            ("THINGS-EEG1 (22 sources, development)", eeg1_al["procrustes"], dev_lower(eeg1_al["procrustes"]), None),
            ("THINGS-EEG2 (10, development)", eeg2_al["procrustes"], dev_lower(eeg2_al["procrustes"]), None),
            ("Alljoined (16, confirmation)", a1[:, 2], allj["claims"]["C3"]["lower"], allj["delta"])]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharex=True)
    colors = ["tab:green", "tab:purple", "tab:orange"]
    for ax, (title, items) in zip(axes, panels):
        for j, (label, values, lower, delta) in enumerate(items):
            strip(ax, j, values, lower, colors[j], rng)
            if delta is not None:
                ax.plot([delta, delta], [j - 0.35, j + 0.35], color="crimson", lw=1.4)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([lab for lab, *_ in items], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("difference in r, 100–200 ms (per person; k = 250/500)")
        summary["fig6"][title] = {lab: {"mean": float(v.mean()), "lower": float(lo), "positive": int((v > 0).sum()),
                                        "n": len(v), "delta": dl} for lab, v, lo, dl in items}
    fig.suptitle("Three datasets: dots = people; bar = mean to its lower bound (confirmations: protocol bound; "
                 "development: one-sided 95%); red tick = that dataset's margin δ", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "paper_fig6_three_datasets.png", dpi=170)
    plt.close(fig)


# one colour per method in every figure
COLOR = {"avg": "#7f7f7f", "h4b": "#d62728", "gu_window": "#1f77b4", "converter": "#9467bd", "own_ridge": "#2ca02c",
         "procrustes_w100": "#ff7f0e", "procrustes_epoch": "#e7ba52", "srm_d8": "#8c564b"}
NAME = {"avg": "others' average", "h4b": "personal map (H4b)", "gu_window": "Gu ensemble", "converter": "converter",
        "own_ridge": "own ridge", "procrustes_w100": "Procrustes per window", "procrustes_epoch": "Procrustes, epoch",
        "srm_d8": "SRM (d = 8)"}
CENTRES = np.arange(-150, 800, 100)  # 100 ms windows from -200 ms


def _means(path: str, value: str = "r_100_200ms", keys=("method", "k")):
    m = defaultdict(lambda: defaultdict(list))
    for r in rows(path):
        m[tuple(r[k] for k in keys)][r["person"]].append(float(r[value]))
    return {key: np.array([mean(v) for _, v in sorted(per.items())]) for key, per in m.items()}


def figure4(summary: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, (label, path) in enumerate([("THINGS-EEG1 (22 sources)", "09i_eeg1_alignment_controls/09i_scores.csv"),
                                         ("THINGS-EEG2 (10)", "09j_eeg2_alignment_controls/09j_scores.csv")]):
        m = _means(path)
        ax = axes[row, 0]
        ks = [250, 500, 1000]
        for meth in ("h4b", "procrustes_w100", "srm_d8", "procrustes_epoch"):
            ax.plot(ks, [m[(meth, str(k))].mean() for k in ks], "o-", color=COLOR[meth], label=NAME[meth])
        ax.axhline(m[("avg", "all")].mean(), color=COLOR["avg"], ls="--", label=NAME["avg"])
        ax.set(xscale="log", xticks=ks, xlabel="labelled images k", ylabel="r, 100-200 ms", title=label)
        ax.set_xticklabels(ks)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(fontsize=7)
        ax = axes[row, 1]
        avg_w = np.array([_means(path, f"r_w{w}")[("avg", "all")].mean() for w in range(10)])
        for meth in ("h4b", "procrustes_w100", "srm_d8"):
            gain = np.array([_means(path, f"r_w{w}")[(meth, "500")].mean() for w in range(10)]) - avg_w
            ax.plot(CENTRES, gain, "o-", color=COLOR[meth], label=NAME[meth])
            summary["fig4"][f"{label} {meth} per-window gain k=500"] = [round(float(g), 4) for g in gain]
        ax.axhline(0, color="grey", lw=0.8)
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.set(xlabel="window centre (ms)", ylabel="gain over others' average", title=f"{label}: per window, k = 500")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "paper_fig4_alignment.png", dpi=170)
    plt.close(fig)


def figure_s1(summary: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2))
    for row, (label, path) in enumerate([("THINGS-EEG1 (22 sources)", "09b_eeg1_h4b_ablation/09b_scores.csv"),
                                         ("THINGS-EEG2 (10)", "09c_eeg2_h4b_ablation/09c_scores.csv")]):
        m = _means(path, keys=("method", "k", "variant"))
        ks = [50, 100, 250, 500, 1000, 2000]
        ax = axes[row, 0]
        for meth, var in (("h4b", "w100r17"), ("gu_window", "-"), ("converter", "-"), ("own_ridge", "-")):
            ax.plot(ks, [m[(meth, str(k), var)].mean() for k in ks], "o-", color=COLOR[meth], label=NAME[meth])
        ax.axhline(m[("avg", "all", "-")].mean(), color=COLOR["avg"], ls="--", label=NAME["avg"])
        ax.set(xscale="log", xticks=ks, xlabel="labelled images k", ylabel="r, 100-200 ms", title=label)
        ax.set_xticklabels(ks)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.legend(fontsize=7)
        avg = m[("avg", "all", "-")]
        gain = lambda var: np.mean([m[("h4b", k, var)] - avg for k in ("250", "500")], 0)
        ax = axes[row, 1]
        vs = [("w50r17", "50 ms"), ("w100r17", "100 ms"), ("w200r17", "200 ms"), ("wepochr17", "epoch")]
        g = [gain(v) for v, _ in vs]
        ax.bar(range(len(vs)), [x.mean() for x in g], color=COLOR["h4b"], alpha=0.75)
        for i, x in enumerate(g):
            ax.plot(np.full(len(x), i) + np.linspace(-0.15, 0.15, len(x)), x, "k.", ms=3, alpha=0.6)
        ax.set(xticks=range(len(vs)), ylabel="gain over others' average (k = 250/500)", title="one map per ...")
        ax.set_xticklabels([n for _, n in vs])
        ax = axes[row, 2]
        rs = [("w100r1", "1"), ("w100r2", "2"), ("w100r4", "4"), ("w100r8", "8"), ("w100r17", "17 (full)")]
        g = [gain(v) for v, _ in rs]
        ax.bar(range(len(rs)), [x.mean() for x in g], color=COLOR["h4b"], alpha=0.75)
        for i, x in enumerate(g):
            ax.plot(np.full(len(x), i) + np.linspace(-0.15, 0.15, len(x)), x, "k.", ms=3, alpha=0.6)
        ax.set(xticks=range(len(rs)), xlabel="map rank", title="spatial components per window")
        ax.set_xticklabels([n for _, n in rs])
        summary["figS1"][label] = {"window": {n: round(float(gain(v).mean()), 4) for v, n in vs},
                                   "rank": {n: round(float(gain(v).mean()), 4) for v, n in rs}}
    fig.tight_layout()
    fig.savefig(OUT / "paper_figS1_ablation.png", dpi=170)
    plt.close(fig)


def figure_s2(summary: dict) -> None:
    sim = rows("09h_eeg2_h4b_sessions/09h_similarity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, part, title in zip(axes, ("full", "offdiag"), ("full map", "channel mixing only (off-diagonal)")):
        for col, colour, name in (("same_session", "#1f77b4", "same person, same session"),
                                  ("cross_session", "#d62728", "same person, other session"),
                                  ("between_people", "#7f7f7f", "different people")):
            v = np.array([[float(r[col]) for r in sim if r["part"] == part and int(r["window"]) == w] for w in range(10)])
            ax.errorbar(CENTRES, v.mean(1), yerr=v.std(1, ddof=1) / np.sqrt(v.shape[1]), fmt="o-", color=colour, label=name, ms=4, capsize=2)
            summary["figS2"][f"{part} {col}"] = [round(float(x), 3) for x in v.mean(1)]
        ax.axvline(0, color="grey", lw=0.5, ls=":")
        ax.set(xlabel="window centre (ms)", title=title)
    axes[0].set_ylabel("map similarity (cosine)")
    axes[0].legend(fontsize=7)
    fig.suptitle("THINGS-EEG2, 10 people: maps from single repetitions, by recording session", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "paper_figS2_sessions.png", dpi=170)
    plt.close(fig)


def figure_s3(summary: dict) -> None:
    m = defaultdict(list)
    for r in rows("09m_eeg2_modern_features/09m_scores.csv"):
        m[(r["features"], r["person"], r["method"], r["k"])].append(float(r["r_100_200ms"]))
    feats = ["alexnet", "dinov2", "dinov3", "siglip2"]
    labels = ["AlexNet (2012)", "DINOv2 (2023)", "DINOv3 (2025)", "SigLIP 2 (2025)"]
    people = sorted({k[1] for k in m})
    cols = [("own_full", "all", "own full-data", "#bcbd22"), ("avg", "all", NAME["avg"], COLOR["avg"]),
            ("own_ridge", "500", "own ridge, k = 500", COLOR["own_ridge"]), ("h4b", "500", "personal map, k = 500", COLOR["h4b"])]
    fig, ax = plt.subplots(figsize=(10, 4))
    for j, (meth, k, name, colour) in enumerate(cols):
        for i, f in enumerate(feats):
            v = np.array([mean(m[(f, p, meth, k)]) for p in people])
            x = i + (j - 1.5) * 0.2
            ax.bar(x, v.mean(), 0.19, color=colour, label=name if i == 0 else None)
            ax.plot(np.full(len(v), x), v, "k.", ms=2.5, alpha=0.5)
            summary["figS3"][f"{f} {meth} {k}"] = round(float(v.mean()), 4)
    ax.set(xticks=range(4), ylabel="r, 100-200 ms", title="THINGS-EEG2, 10 people: image features through the same kernel pipeline")
    ax.set_xticklabels(labels)
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "paper_figS3_modern.png", dpi=170)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"fig2": {}, "fig3": {}, "fig6": {}, "fig4": {}, "figS1": {}, "figS2": {}, "figS3": {}}
    figure2(summary)
    figure3(summary)
    figure6(summary)
    figure4(summary)
    figure_s1(summary)
    figure_s2(summary)
    figure_s3(summary)
    (OUT / "paper_figures_summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("paper_fig*"))))


if __name__ == "__main__":
    main()
