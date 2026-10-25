"""Paper figures in the style of `plot_paper_v2.py`. Run from the repository root:

    python scripts/plot_paper_v3.py

Writes PDF and PNG to `results/figures/v3/`, among them:
  fig2_borrow   Figure 2: borrowing on THINGS-EEG1 (08e, 08m)
  fig_nod       Figure S1: the three NOD claims, one dot per person, with bounds and margins (11d)
  fig_curves    Figure S3: EEG per-person curves (16a, 17a, 18j)
  fig_decoding  Figure S6: EEG decoding (18l, 18m, 18n)
  v3_figures_summary.json   the numbers drawn
The other figures it writes are not used in the paper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_paper_v2 as v2  # noqa: E402  (style, colours, save)

OUT = ROOT / "results" / "figures" / "v3"
RUNS = ROOT / "results"
NAME = {"avg_sources": "others' average", "gu_window": "weighted ensemble (Gu)", "linear_converter": "converter",
        "own_ridge": "own model", "own_ridge_oc": "own model + output map", "generic": "others' mean map"}
VERDICT = {"supported": ("#C8102E", True), "inconclusive": ("#7A7A7A", False), "rejected": ("#2E5EAA", True)}


def load(path):
    return json.loads((RUNS / path).read_text())


def row_verdict(lower, upper, margin):
    return "supported" if lower > margin else ("rejected" if upper < margin else "inconclusive")


def claims():
    """Groups of (label, mean, lower, upper, margin, positive, n, unit) with the family verdict."""
    groups = []
    e = load("08e_eeg1_confirmation/08e_decision.json")
    groups.append(("THINGS-EEG1, 24 new: shared network (P-1)", e["verdict"],
                   [(f"shared network − {NAME[c]}", v["mean"], v["lower"], v["upper"], e["delta"], v["positive_targets"], 24, "r")
                    for c, v in e["per_comparator"].items()]))
    m = load("08m_eeg1_h8_confirmation/08m_decision.json")
    groups.append(("THINGS-EEG1, 24 new: borrowing vs own data", m["primary_verdict"],
                   [(f"{NAME[a.strip()]} − {NAME[b.strip()]}", v["mean"], v["lower"], v["upper"], 0.0035, v["positive"], 24, "r")
                    for (a, b), v in ((tuple(k.split(" - ")), v) for k, v in m["conditions"]["k"]["contrasts"].items())]))
    n = load("08n_eeg1_h4b_confirmation/08n_decision.json")
    groups.append(("THINGS-EEG1, 24 new: personal map", n["verdict"],
                   [(f"map − {NAME[c]}", v["mean"], v["lower"], v["upper"], n["delta"], v["positive_targets"], 24, "r")
                    for c, v in n["per_comparator"].items()]))
    a = load("09a_eeg2_h4b_replication/09a_decision.json")
    groups.append(("THINGS-EEG2, 10: map replication", a["verdict"],
                   [(f"map − {NAME[c]}", v["mean"], v["lower"], v["upper"], a["delta"], v["positive_targets"], 10, "r")
                    for c, v in a["per_comparator"].items()]))
    l = load("09l_alljoined_confirmation/09l_decision.json")["primary"]
    label = {"C1": "map − others' average", "C2": "weighted ensemble + map − weighted ensemble", "C3": "map − Procrustes"}
    groups.append(("Alljoined, consumer EEG, 16 new (P-6)", "per claim",
                   [(label[c], v["mean"], v["lower"], v["upper"], l["delta"], v["positive_targets"], 16, "r")
                    for c, v in l["claims"].items()]))
    s = json.loads((RUNS / "10c_nsd_confirmation/10c_summary.json").read_text())["claims"]
    groups.append(("NSD fMRI, 8 new (P-7)", "per claim", [
        ("low-rank map − others' average (encoding)", *(s["C1_lowrank_map_minus_avg_enc_r"][k] for k in ("mean", "lower", "upper", "margin", "positive")), 8, "r"),
        ("map + own model − others' average (decoding)", *(s["C2_stack_minus_avg_top1"][k] for k in ("mean", "lower", "upper", "margin", "positive")), 8, "pts"),
        ("map + own model − own model (decoding)", *(s["C3_stack_minus_own_ridge_top1"][k] for k in ("mean", "lower", "upper", "margin", "positive")), 8, "pts")]))
    return groups


def fig3_forest(summary):
    groups = claims()
    rows, y, ticks = [], 0, []
    fig, ax = plt.subplots(figsize=(v2.W, 7.6))
    fig.subplots_adjust(left=0.37, right=0.755, top=0.97, bottom=0.1)
    for title, family, items in groups:
        ax.text(-0.02, y, title + (f" — {family}" if family != "per claim" else ""), transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.4, fontweight="bold")
        y -= 1
        for label, mean, lower, upper, margin, positive, n, unit in items:
            verdict = row_verdict(lower, upper, margin)
            colour, filled = VERDICT[verdict]
            lo, mi, hi = lower / margin, mean / margin, upper / margin
            ax.plot([lo, hi], [y, y], color=colour, lw=1.4, solid_capstyle="butt", zorder=2)
            ax.plot(mi, y, "D", ms=4.6, mec=colour, mfc=colour if filled else "white", mew=1.1, zorder=3)
            ax.text(-0.02, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.1)
            fmt = "{:+.3f}" if unit == "r" else "{:+.1f}"
            ax.text(1.02, y, f"{fmt.format(mean)} [{fmt.format(lower)}, {fmt.format(upper)}]{'' if unit == 'r' else ' pts'}"
                             f"  {positive}/{n}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.6,
                    color="#333")
            rows.append({"group": title, "claim": label, "mean": mean, "lower": lower, "upper": upper, "margin": margin,
                         "verdict": verdict, "positive": positive, "n": n, "unit": unit})
            y -= 1
        y -= 0.6
    ax.axvline(0, color="black", lw=0.7)
    ax.axvline(1, color="#C8102E", lw=0.8, ls=(0, (3, 2)))
    ax.set_xscale("symlog", linthresh=2, linscale=1.2)
    ax.set_xticks([-5, 0, 1, 2, 5, 10, 30])
    ax.set_xticklabels(["−5", "0", "1", "2", "5", "10", "30"])
    ax.set_xlim(-7, 40)
    ax.set_ylim(y + 0.4, 1.0)
    ax.set_yticks([])
    ax.spines[["left"]].set_visible(False)
    ax.set_xlabel("effect ÷ the protocol's margin", fontsize=7)
    for verdict, (colour, filled) in VERDICT.items():
        ax.plot([], [], "D", ms=4.6, mec=colour, mfc=colour if filled else "white", mew=1.1, label=verdict)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=6.3, frameon=False)
    ax.text(1.02, 1.0, "difference [bounds]  improved", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=5.8, fontweight="bold")
    summary["fig3_forest"] = rows
    v2.fit_width(fig)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig3_forest.{ext}", dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


def _per_person_eeg1():
    """EEG1 confirmation people. P-2 (08m, condition k: noise normalisation from the labelled trials): own ridge and
    others' average, mean over k = 250/500 and draws. P-1 (08e, its own preprocessing): others' average and the shared
    network, mean over k, draws and seeds, as scored (both from the same runs, not one run's level plus another's
    difference)."""
    import csv
    from collections import defaultdict
    acc = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(RUNS / "08m_eeg1_h8_confirmation/08m_scores.csv")):
        if r["condition"] == "k" and r["method"] in ("avg_sources", "own_ridge"):
            acc[r["person"]][r["method"]].append(float(r["r_100_200ms"]))
    p1 = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(RUNS / "08e_eeg1_confirmation/08e_scores.csv")):
        if r["method"] in ("avg_sources", "prior_both"):
            p1[r["person"]][r["method"]].append(float(r["r_100_200ms"]))
    core_d = {p: v[0] for p, v in load("08e_eeg1_confirmation/08e_decision.json")["d"].items()}
    people = sorted(acc)
    own = np.array([np.mean(acc[p]["own_ridge"]) for p in people])
    avg = np.array([np.mean(acc[p]["avg_sources"]) for p in people])
    avg_p1 = np.array([np.mean(p1[p]["avg_sources"]) for p in people])
    core = np.array([np.mean(p1[p]["prior_both"]) for p in people])
    if not np.allclose(core - avg_p1, [core_d[p] for p in people]):
        raise ValueError("08e levels do not reproduce its decision differences")
    return people, own, avg, avg_p1, core


def _slope(ax, left, right, labels, colour_up, colour_down):
    for a, b in zip(left, right):
        ax.plot([0, 1], [a, b], color=colour_up if b > a else colour_down, lw=0.8, alpha=0.75)
        ax.scatter([0, 1], [a, b], s=7, color=colour_up if b > a else colour_down, zorder=3, linewidths=0)
    ax.plot([0, 1], [left.mean(), right.mean()], color="black", lw=2.2, solid_capstyle="round", zorder=4)
    ax.set_xticks([0, 1], labels)
    ax.set_xlim(-0.25, 1.25)


def fig2_borrow(summary):
    """Borrow first (THINGS-EEG1, the 24 people of the prespecified tests). A: each person's own model at 250-500 images against
    others' average (H8, supported). B: others' average against the shared network (P-1, rejected)."""
    people, own, avg, avg_p1, core = _per_person_eeg1()
    fig, ax = plt.subplots(1, 2, figsize=(v2.W, 2.7), sharey=True)
    # neutral person lines; slope shows direction
    _slope(ax[0], own, avg, ["own model\n(250–500 images)", "borrowing"], "#8C8C8C", "#8C8C8C")
    _slope(ax[1], avg_p1, core, ["borrowing", "shared\nnetwork"], "#8C8C8C", "#8C8C8C")
    up = int((avg > own).sum()); down = int((core < avg_p1).sum())
    ax[0].set_title("Own model vs borrowing", loc="left")
    ax[0].text(0.02, 0.02, f"supported · {up}/24 people higher", transform=ax[0].transAxes, fontsize=6.8, color="#222222")
    ax[1].set_title("Borrowing vs shared network (first test)", loc="left")
    ax[1].text(0.02, 0.02, f"rejected · {down}/24 people lower", transform=ax[1].transAxes, fontsize=6.8, color="#222222")
    ax[0].set_ylabel("encoding r (100–200 ms)")
    v2.letter(ax[0], "A"); v2.letter(ax[1], "B")
    fig.tight_layout(w_pad=2)
    summary["fig2_borrow"] = {"own": float(own.mean()), "avg": float(avg.mean()), "avg_p1": float(avg_p1.mean()), "core": float(core.mean()),
                              "avg_above_own": up, "core_below_avg": down}
    _save(fig, "fig2_borrow")


def _nsd_person_diffs():
    import csv
    from collections import defaultdict
    v = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(RUNS / "10c_nsd_confirmation/10c_scores.csv")):
        key = (r["method"], r["k"], r["centred"])
        v[r["subject"]][key].append((float(r["enc_r"]), float(r["top1"])))
    subs = sorted(v)
    m = lambda s, method, ks, centred, i: np.mean([x[i] for k in ks for x in v[s][(method, k, centred)]])
    enc = np.array([m(s, "lowrank_map", ("250", "500"), "False", 0) - m(s, "avg", ("0",), "False", 0) for s in subs])
    dec = np.array([m(s, "stack", ("250", "500"), "False", 1) - m(s, "avg", ("0",), "True", 1) for s in subs])
    return enc, dec


def fig3_map(summary):
    """The personal map. A: map minus others' average for every population tested before scoring, one dot per person, with the
    mean, its Bonferroni bounds and the margin. B: the map against every alternative tested before scoring: cells
    coloured by verdict, effect divided by margin and people improved inside."""
    n = load("08n_eeg1_h4b_confirmation/08n_decision.json")
    a = load("09a_eeg2_h4b_replication/09a_decision.json")
    l = load("09l_alljoined_confirmation/09l_decision.json")["primary"]
    c = json.loads((RUNS / "10c_nsd_confirmation/10c_summary.json").read_text())["claims"]
    enc, dec = _nsd_person_diffs()
    strips = [("THINGS-EEG1\n24 new people", np.array([v[0] for v in n["d"].values()]), n["per_comparator"]["avg_sources"], n["delta"], "Δ r"),
              ("THINGS-EEG2\n10 people", np.array([v[0] for v in a["d"].values()]), a["per_comparator"]["avg_sources"], a["delta"], "Δ r"),
              ("consumer EEG\n16 new people", np.array([v[0] for v in l["d"].values()]), l["claims"]["C1"], l["delta"], "Δ r"),
              ("fMRI encoding\n8 new people", enc, c["C1_lowrank_map_minus_avg_enc_r"], c["C1_lowrank_map_minus_avg_enc_r"]["margin"], "Δ r"),
              ("fMRI decoding\n8 new people", dec, c["C2_stack_minus_avg_top1"], c["C2_stack_minus_avg_top1"]["margin"], "Δ top-1 (points)")]
    fig = plt.figure(figsize=(v2.W, 5.3))
    top = fig.add_gridspec(1, 5, left=0.07, right=0.99, top=0.9, bottom=0.6, wspace=0.75)
    rng = np.random.default_rng(3)
    out = {}
    for i, (name, vals, bounds, margin, unit) in enumerate(strips):
        ax = fig.add_subplot(top[0, i])
        verdict = row_verdict(bounds["lower"], bounds["upper"], margin)
        colour = VERDICT[verdict][0]
        ax.axhline(0, color="black", lw=0.6)
        ax.axhline(margin, color="#C8102E", lw=0.8, ls=(0, (3, 2)))
        ax.scatter(rng.uniform(-0.18, 0.18, len(vals)), vals, s=9, color=colour, alpha=0.55, linewidths=0, zorder=2)
        ax.plot([0.38, 0.38], [bounds["lower"], bounds["upper"]], color=colour, lw=2, solid_capstyle="butt")
        ax.plot(0.38, bounds["mean"], "D", ms=4.5, mec=colour, mfc=colour if VERDICT[verdict][1] else "white", mew=1.1, zorder=3)
        ax.set_xlim(-0.4, 0.6)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(name, fontsize=6.8, loc="center", fontweight="normal")
        ax.text(0.5, -0.08, verdict, transform=ax.transAxes, ha="center", va="top", fontsize=6.6, color=colour, fontweight="bold")
        ax.set_ylabel(unit, fontsize=6.6, labelpad=1)
        ax.tick_params(labelsize=6)
        out[name.split("\n")[0]] = {"mean": float(bounds["mean"]), "verdict": verdict}
        if i == 0:
            v2.letter(ax, "A", x=-0.55, y=1.12)
    fig.text(0.5, 0.965, "Personal map minus others' average, one dot per person (dashed: the protocol's margin)", ha="center", fontsize=8, fontweight="bold")
    # B: the verdict grid
    rows = [("others' average", "avg_sources"), ("weighted ensemble (Gu)", "gu_window"), ("converter", "linear_converter"),
            ("own model", "own_ridge"), ("own model + output map", "own_ridge_oc"), ("others' mean map", "generic"), ("Procrustes", None)]
    cols = ["THINGS-EEG1", "THINGS-EEG2", "consumer EEG", "fMRI"]
    cells = {}
    for r, key in rows:
        if key:
            for col, dec_ in (("THINGS-EEG1", n), ("THINGS-EEG2", a)):
                v = dec_["per_comparator"][key]
                cells[(r, col)] = (v["mean"], v["lower"], v["upper"], dec_["delta"], v["positive_targets"], len(dec_["d"]))
    cells[("others' average", "consumer EEG")] = (l["claims"]["C1"]["mean"], l["claims"]["C1"]["lower"], l["claims"]["C1"]["upper"], l["delta"], l["claims"]["C1"]["positive_targets"], 16)
    cells[("Procrustes", "consumer EEG")] = (l["claims"]["C3"]["mean"], l["claims"]["C3"]["lower"], l["claims"]["C3"]["upper"], l["delta"], l["claims"]["C3"]["positive_targets"], 16)
    cc = c["C1_lowrank_map_minus_avg_enc_r"]
    cells[("others' average", "fMRI")] = (cc["mean"], cc["lower"], cc["upper"], cc["margin"], cc["positive"], 8)
    ax = fig.add_axes([0.3, 0.04, 0.66, 0.42])
    for j, col in enumerate(cols):
        ax.text(j + 0.5, len(rows) + 0.15, col, ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    for i, (r, _) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.text(-0.08, y + 0.5, r, ha="right", va="center", fontsize=6.8)
        for j, col in enumerate(cols):
            if (r, col) not in cells:
                ax.text(j + 0.5, y + 0.5, "–", ha="center", va="center", fontsize=7, color="#BBBBBB")
                continue
            mean, lo, hi, margin, pos, nn = cells[(r, col)]
            verdict = row_verdict(lo, hi, margin)
            face = {"supported": "#C8102E", "inconclusive": "#D9D9D9", "rejected": "#2E5EAA"}[verdict]
            ax.add_patch(plt.Rectangle((j + 0.04, y + 0.06), 0.92, 0.88, color=face, lw=0, alpha=0.9 if verdict != "inconclusive" else 1))
            txt = "white" if verdict != "inconclusive" else "#333333"
            ax.text(j + 0.5, y + 0.5, f"{mean / margin:.1f}×   {pos}/{nn}", ha="center", va="center", fontsize=6.4, color=txt)
    ax.set_xlim(-0.02, len(cols) + 0.02)
    ax.set_ylim(-0.05, len(rows) + 0.7)
    ax.axis("off")
    ax.text(-2.9, len(rows) + 0.6, "B", fontsize=9, fontweight="bold", va="bottom")
    ax.text(-2.75, len(rows) + 0.6, "  Personal map against every alternative", fontsize=8, fontweight="bold", va="bottom")
    ax.text(len(cols), len(rows) + 0.6, "cell: effect ÷ margin   people improved   (– not in that protocol)", ha="right",
            va="bottom", fontsize=5.8, color="#555555")
    for k_, (face, lab) in enumerate((("#C8102E", "supported"), ("#D9D9D9", "inconclusive"))):
        ax.add_patch(plt.Rectangle((-2.75, len(rows) - 1.0 - 0.7 * k_), 0.18, 0.45, color=face, lw=0, clip_on=False))
        ax.text(-2.5, len(rows) - 0.78 - 0.7 * k_, lab, fontsize=6, va="center")
    summary["fig3_map"] = {"strips": out, "cells": {f"{r}|{col}": round(v[0] / v[3], 2) for (r, col), v in cells.items()}}
    _save(fig, "fig3_map")


def _nod_person_diffs():
    """P-8's per-person differences: mean over k = 250, 500 and draws, vertices at the 0.05 rule, uncentred."""
    import csv
    from collections import defaultdict
    v = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(RUNS / "11d_nod_confirmation/11d_scores.csv")):
        if r["threshold"] == "0.05" and r["centred"] == "False":
            v[r["subject"]][r["method"]].append((float(r["enc_r"]), float(r["top1"])))
    subs = sorted(v)
    m = lambda s, method, i: np.mean([x[i] for x in v[s][method]])
    e1 = np.array([m(s, "avg", 0) - m(s, "own_ridge", 0) for s in subs])
    e2 = np.array([m(s, "mixed", 1) - m(s, "own_ridge", 1) for s in subs])
    e3 = np.array([m(s, "lowrank_map", 0) - m(s, "avg", 0) for s in subs])
    return e1, e2, e3


def fig_nod(summary):
    """P-8 on NOD's 21 new single-session people: the three frozen claims as strips, one dot per person, with the mean,
    the Bonferroni bounds the verdict used and the margin (dashed). Verdict colours as Figure 3."""
    fam = load("11d_nod_confirmation/11d_summary.json")["family"]
    e1, e2, e3 = _nod_person_diffs()
    strips = [("borrowing − own model\n(encoding)", e1, fam["E1_avg_minus_own_ridge_enc_r"], "Δ r", "predicted: supported"),
              ("combined system − own model\n(identification)", e2, fam["E2_mixed_minus_own_ridge_top1"], "Δ top-1 (points)", "predicted: supported"),
              ("personal map − borrowing\n(encoding)", e3, fam["E3_lowrank_map_minus_avg_enc_r"], "Δ r", "predicted: rejected")]
    fig, axes = plt.subplots(1, 3, figsize=(v2.W, 2.25))
    rng = np.random.default_rng(3)
    out = {}
    for ax, (name, vals, b, unit, pred) in zip(axes, strips):
        if abs(vals.mean() - b["mean"]) > 1e-6:
            raise ValueError(f"{name}: per-person mean {vals.mean()} differs from the decision file {b['mean']}")
        verdict = b["verdict"]
        colour = VERDICT[verdict][0]
        ax.axhline(0, color="black", lw=0.6)
        ax.axhline(b["margin"], color="#C8102E", lw=0.8, ls=(0, (3, 2)))
        ax.scatter(rng.uniform(-0.18, 0.18, len(vals)), vals, s=10, color=colour, alpha=0.55, linewidths=0, zorder=2)
        ax.plot([0.38, 0.38], [b["lower"], b["upper"]], color=colour, lw=2, solid_capstyle="butt")
        ax.plot(0.38, b["mean"], "D", ms=4.5, mec=colour, mfc=colour if VERDICT[verdict][1] else "white", mew=1.1, zorder=3)
        ax.set_xlim(-0.4, 0.6)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(name, fontsize=7, loc="center", fontweight="normal")
        ax.text(0.5, -0.06, f"{verdict} · {int((vals > 0).sum())}/{len(vals)} above 0", transform=ax.transAxes, ha="center",
                va="top", fontsize=6.6, color=colour, fontweight="bold")
        ax.text(0.5, -0.17, pred, transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color="#555555")
        ax.set_ylabel(unit, fontsize=6.6, labelpad=1)
        ax.tick_params(labelsize=6)
        out[name.split("\n")[0]] = {"mean": round(float(b["mean"]), 4), "lower": round(float(b["lower"]), 4),
                                     "upper": round(float(b["upper"]), 4), "verdict": verdict, "positive": int((vals > 0).sum())}
    fig.tight_layout(w_pad=2.5, rect=(0, 0.05, 1, 1))
    summary["fig_nod"] = out
    _save(fig, "fig_nod", 0.86 * v2.W)


def _person_means(path, value, keep, by):
    """Mean of `value` per person and per `by` over rows passing `keep` (draws averaged)."""
    import csv
    from collections import defaultdict
    acc = defaultdict(list)
    for r in csv.DictReader(open(RUNS / path)):
        if keep(r):
            acc[(r["person"], by(r))].append(float(r[value]))
    return {key: float(np.mean(v)) for key, v in acc.items()}


def _gain_lines(ax, x, table, base, people, colour, label):
    """One thin line per person of table - base (others' average), and the mean as a thick line."""
    ys = np.array([[table[(p, k)] - base[p] for k in x] for p in people])
    for y in ys:
        ax.plot(range(len(x)), y, color=colour, lw=0.6, alpha=0.35, zorder=1)
    ax.plot(range(len(x)), np.nanmean(ys, 0), color=colour, lw=2.2, marker="o", ms=3.2, zorder=3, label=label)
    return ys


def _zero(ax, text):
    ax.axhline(0, color=v2.C["avg"], lw=1.4, zorder=2)
    ax.annotate(text, (0.99, 0), xycoords=ax.get_yaxis_transform(), xytext=(0, -2), textcoords="offset points", ha="right", va="top", fontsize=6.2, color=v2.C["avg"])


def fig_curves(summary):
    """Per-person curves: each person's gain over others' average (the zero line), one thin line per person and
    a thick mean. A: encoding against labelled images, THINGS-EEG1 development people (16a, random choice of images;
    own ridge was run to 250 only). B: the map's gain over time in 100 ms windows at 500 images (17a). C: decoding
    top-1 among 200 against labelled images, THINGS-EEG2 (18j, AlexNet; the map with the own model)."""
    import csv
    from collections import defaultdict
    fig, ax = plt.subplots(1, 3, figsize=(v2.W, 2.5), gridspec_kw={"width_ratios": [1, 1.2, 1], "wspace": 0.5})
    out = {}
    # A
    ks = ["50", "100", "250", "500"]
    t = _person_means("16a_eeg1_few_labels/16a_scores.csv", "r_100_200ms",
                      lambda r: r["policy"] in ("random", "-") and r["method"] in ("own_ridge", "h4b", "avg"), lambda r: (r["method"], r["k"]))
    people = sorted({p for p, _ in t})
    base = {p: t[(p, ("avg", "all"))] for p in people}
    a_own = _gain_lines(ax[0], ks, {(p, k): t.get((p, ("own_ridge", k)), np.nan) for p in people for k in ks}, base, people, v2.C["own"], "own model")
    a_map = _gain_lines(ax[0], ks, {(p, k): t[(p, ("h4b", k))] for p in people for k in ks}, base, people, v2.C["map"], "personal map")
    _zero(ax[0], "")
    ax[0].set_xticks(range(len(ks)), ks)
    ax[0].set_xlabel("labelled images")
    ax[0].set_ylabel("gain over borrowing (r)")
    ax[0].legend(loc="lower right", fontsize=6.3, frameon=False, handlelength=1.4)
    ax[0].set_title("EEG1 encoding")
    out["A"] = {"people": len(people), "own_gain": np.nanmean(a_own, 0).round(4).tolist(), "map_gain": a_map.mean(0).round(4).tolist(),
                "map_above_avg": [int((a_map[:, i] > 0).sum()) for i in range(len(ks))]}
    # B
    acc = defaultdict(list)
    for r in csv.DictReader(open(RUNS / "17a_eeg1_latency/17a_scores.csv")):
        if (r["method"] == "avg" and r["k"] == "all") or (r["method"] == "h4b" and r["k"] == "500"):
            acc[(r["person"], r["method"])].append([float(r[f"r_w{w}"]) for w in range(10)])
    pe = sorted({p for p, _ in acc})
    centres = -150 + 100 * np.arange(10)
    gain = np.array([np.mean(acc[(p, "h4b")], 0) - np.mean(acc[(p, "avg")], 0) for p in pe])
    for y in gain:
        ax[1].plot(centres, y, color=v2.C["map"], lw=0.6, alpha=0.35)
    ax[1].plot(centres, gain.mean(0), color=v2.C["map"], lw=2.2, marker="o", ms=3.2)
    _zero(ax[1], "")
    ax[1].axvspan(100, 200, color="#F2F2F2", zorder=0)
    ax[1].set_xticks([-100, 100, 300, 500, 700])
    ax[1].set_xlabel("ms after image onset")
    ax[1].set_title("EEG1 by time")
    ax[1].set_ylabel("gain over borrowing (r)")
    out["B"] = {"people": len(pe), "map_gain": gain.mean(0).round(4).tolist(), "positive_100_200": int((gain[:, 3] > 0).sum())}
    # C
    kc = ["250", "500", "1000"]
    t = _person_means("18j_eeg2_decoding_system/18j_scores.csv", "top1",
                      lambda r: r["variant"] == "alexnet" and r["method"] in ("avg", "own_ridge", "stack"), lambda r: (r["method"], r["k"]))
    pc = sorted({p for p, _ in t})
    base = {p: t[(p, ("avg", "0"))] for p in pc}
    c_own = _gain_lines(ax[2], kc, {(p, k): t[(p, ("own_ridge", k))] for p in pc for k in kc}, base, pc, v2.C["own"], "own model")
    c_st = _gain_lines(ax[2], kc, {(p, k): t[(p, ("stack", k))] for p in pc for k in kc}, base, pc, v2.C["map"], "map + own model")
    _zero(ax[2], "")
    ax[2].set_xticks(range(len(kc)), kc)
    ax[2].set_xlabel("labelled images")
    ax[2].set_ylabel("gain in top-1 among 200 (points)")
    ax[2].legend(loc="upper left", fontsize=6.3, frameon=False, handlelength=1.4)
    ax[2].set_title("EEG2 decoding")
    out["C"] = {"people": len(pc), "avg_top1": float(np.mean(list(base.values()))), "own_gain": c_own.mean(0).round(2).tolist(),
                "stack_gain": c_st.mean(0).round(2).tolist()}
    for a_, s_ in zip(ax, "ABC"):
        v2.letter(a_, s_, x=-0.14)
    summary["fig_curves"] = out
    _save(fig, "fig_curves")


def fig_decoding(summary):
    """Decoding a new person (THINGS-EEG2, 10 people, development). A: top-1 among 200
    against labelled images, raw score with the person's labelled mean as intercept (inductive): map + own model on
    AlexNet and with the DINOv2 map; borrowing (raw, k = 0) and the own model on all images as references. B: the raw-
    score fix at 500 images, per person: with the donors' mean response, with the person's labelled mean, and centred.
    C: identification in 100 ms windows (centred score): borrowing and map + own at 500. D: leakage controls at 500
    (centred): real, labels shuffled, test pairing shuffled; borrowing with and without the person's own encoder."""
    import csv
    from collections import defaultdict
    fig, axs = plt.subplots(2, 2, figsize=(v2.W, 4.9), gridspec_kw={"wspace": 0.3, "hspace": 0.55})
    ax = axs.ravel()
    out = {}
    # A
    t = _person_means("18n_eeg2_decoding_stack_mean/18n_scores.csv", "top1",
                      lambda r: r["centred"] == "False" and (r["how"] in ("own_mean", "-")) and r["method"] in ("stack", "avg", "own_full"),
                      lambda r: (r["method"], r["variant"], r["k"]))
    people = sorted({p for p, _ in t})
    ks = ["250", "500", "1000"]
    for variant, colour, ls, mk, lab in (("alexnet", v2.C["map"], "-", "o", "map + own model (AlexNet)"),
                                         ("mixed", "#5A0A16", (0, (4, 1.5)), "s", "same, map on DINOv2")):
        ys = np.array([[t[(p, ("stack", variant, k))] for k in ks] for p in people])
        for y in ys:
            ax[0].plot(range(3), y, color=colour, lw=0.6, alpha=0.3, ls=ls)
        ax[0].plot(range(3), ys.mean(0), color=colour, lw=2, ls=ls, marker=mk, ms=3.4, label=lab)
        out[f"A_{variant}"] = ys.mean(0).round(1).tolist()
    borrow = np.mean([t[(p, ("avg", "alexnet", "0"))] for p in people])
    own_full = np.mean([t[(p, ("own_full", "alexnet", "all"))] for p in people])
    ax[0].axhline(borrow, color=v2.C["avg"], lw=1.4)
    ax[0].axhline(own_full, color=v2.C["own"], lw=1.4, ls=(0, (4, 2)))
    ax[0].annotate("borrowing, no labels", (-0.15, borrow), xytext=(0, 2), textcoords="offset points", fontsize=5.8, color=v2.C["avg"], ha="left", va="bottom")
    ax[0].annotate("own model, all images", (2.1, own_full), xytext=(0, 2), textcoords="offset points", fontsize=5.8, color=v2.C["own"], ha="right", va="bottom")
    ax[0].set_xticks(range(3), ks)
    ax[0].set_xlim(-0.2, 2.2)
    ax[0].set_ylim(0, 75)
    ax[0].set_xlabel("labelled images")
    ax[0].set_ylabel("top-1 among 200 (%)")
    ax[0].set_title("A  more labels, better decoding (raw)", fontsize=7.5)
    ax[0].legend(loc="lower right", bbox_to_anchor=(1, 0.2), fontsize=5.8, handlelength=1.2, borderaxespad=0.1)
    out["A_borrow_raw"], out["A_own_full_raw"] = round(float(borrow), 1), round(float(own_full), 1)
    # B
    t = _person_means("18n_eeg2_decoding_stack_mean/18n_scores.csv", "top1",
                      lambda r: r["method"] == "stack" and r["variant"] == "alexnet" and r["k"] == "500" and r["how"] in ("as_09e", "own_mean"),
                      lambda r: (r["how"], r["centred"]))
    cols = [("as_09e", "False"), ("own_mean", "False"), ("own_mean", "True")]
    ys = np.array([[t[(p, c)] for c in cols] for p in people])
    for y in ys:
        ax[1].plot(range(3), y, color=v2.C["map"], lw=0.6, alpha=0.35, marker="o", ms=1.8)
    ax[1].plot(range(3), ys.mean(0), color="black", lw=2, marker="o", ms=3)
    ax[1].set_xticks(range(3), ["donors'\nmean", "person's mean\nfrom labels", "centred on\ntest batch"], fontsize=6)
    ax[1].set_xlim(-0.3, 2.3)
    ax[1].set_ylim(0, 75)
    ax[1].set_ylabel("top-1 among 200 (%)")
    ax[1].set_title("B  effect of the prediction intercept, map + own at 500", fontsize=7.5)
    out["B"] = ys.mean(0).round(1).tolist()
    # C
    acc = defaultdict(list)
    for r in csv.DictReader(open(RUNS / "18l_eeg2_decoding_time/18l_scores.csv")):
        if r["window_s"] != "post" and ((r["method"] == "avg") or (r["method"] == "stack" and r["k"] == "500")):
            acc[(r["person"], r["method"], float(r["window_s"]))].append(float(r["top1"]))
    starts = sorted({w for _, _, w in acc})
    centres = (np.array(starts) + 0.05) * 1000
    for method, colour, lab in (("avg", v2.C["avg"], "borrowing"), ("stack", v2.C["map"], "map + own (500)")):
        ys = np.array([[np.mean(acc[(p, method, w)]) for w in starts] for p in people])
        for y in ys:
            ax[2].plot(centres, y, color=colour, lw=0.5, alpha=0.3)
        ax[2].plot(centres, ys.mean(0), color=colour, lw=2, label=lab)
        out[f"C_{method}_peak"] = {"centre_ms": float(centres[ys.mean(0).argmax()]), "top1": round(float(ys.mean(0).max()), 1)}
    ax[2].axvline(0, color="black", lw=0.5, ls=":")
    ax[2].axhline(0.5, color="black", lw=0.5)
    ax[2].set_xlabel("window centre (ms)")
    ax[2].set_ylabel("top-1 (%), 100 ms window")
    ax[2].set_title("C  decoding over time (centred)", fontsize=7.5)
    ax[2].legend(loc="upper right", fontsize=5.6, handlelength=1.2, borderaxespad=0.1)
    # D
    acc = defaultdict(list)
    for r in csv.DictReader(open(RUNS / "18m_eeg2_decoding_controls/18m_scores.csv")):
        if r["centred"] != "True":
            continue
        cond = r["condition"].rsplit("_", 1)[0] if r["condition"][-1].isdigit() else r["condition"]
        if r["method"] == "stack" and r["k"] == "500" and cond in ("real", "labels_shuffled", "test_shuffled"):
            acc[(r["person"], cond)].append(float(r["top1"]))
        if r["method"] in ("avg", "avg_with_self") and cond in ("real", "positive_control"):
            acc[(r["person"], r["method"])].append(float(r["top1"]))
    groups = [("real", "map + own\n(500)", v2.C["map"]), ("labels_shuffled", "labels\nshuffled", "#B0B0B0"), ("test_shuffled", "test pairs\nshuffled", "#B0B0B0"),
              ("avg", "borrow-\ning", v2.C["avg"]), ("avg_with_self", "leak:\n+ own", "#555555")]
    rng = np.random.default_rng(5)
    for i, (key, lab, colour) in enumerate(groups):
        vals = np.array([np.mean(acc[(p, key)]) for p in people if (p, key) in acc])
        ax[3].scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=8, color=colour, alpha=0.7, linewidths=0)
        ax[3].plot([i - 0.28, i + 0.28], [vals.mean()] * 2, color="black", lw=1.6)
        out[f"D_{key}"] = round(float(vals.mean()), 1)
    ax[3].axvline(2.5, color="#DDDDDD", lw=0.6)
    ax[3].set_xticks(range(len(groups)), [g[1] for g in groups], fontsize=6)
    ax[3].set_ylabel("top-1 among 200 (%)")
    ax[3].set_ylim(-2, 75)
    ax[3].set_title("D  leakage controls (centred)", fontsize=7.5)
    summary["fig_decoding"] = out
    _save(fig, "fig_decoding")


def _save(fig, name, target=v2.W):
    v2.fit_width(fig, target)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    fig3_forest(summary)
    fig2_borrow(summary)
    fig3_map(summary)
    fig_nod(summary)
    fig_curves(summary)
    fig_decoding(summary)
    (OUT / "v3_figures_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "fig3_forest"}, indent=1))


if __name__ == "__main__":
    main()
