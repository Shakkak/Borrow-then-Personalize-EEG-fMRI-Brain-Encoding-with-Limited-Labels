"""Paper figures in the style of `plot_paper_v3.py`. Run from the repository root:

    python scripts/plot_paper_v5.py

Writes PDF and PNG to `results/figures/v5/`:
  fig_map_strips   the personal map minus others' average in every population tested before scoring, one dot per person:
                   THINGS-EEG1 (P-3), THINGS-EEG2 (P-4), consumer EEG (P-6 C1), NSD 7 T (P-7 C1) and NOD 3 T single
                   trials (P-8 E3, predicted to be rejected). Mean, Bonferroni bounds as each protocol used them, margin.
  fig_nsd_budget   NSD budget curve to the whole recording (10k), identification among 200 (centred): one
                   thin line per person for the map on ridge borrowing and the own ridge model, means thick, borrowing
                   dashed; right, map − own model and combination − own model per person. Stops if 10k's means change.
  fig_label_reps   THINGS-EEG2 label repetitions (09t): each map's gain over borrowing with labels averaged over
                   1, 2 or 4 repetitions, one line per person
  fig_examples     NSD decoding examples (10g): the seen image and each system's top guesses, photographs fetched from
                   COCO's servers (needs internet once; cached in COCO_CACHE). Offline: --skip-examples, or it is skipped
                   with a message when COCO cannot be reached.
  v5_figures_summary.json   the numbers drawn
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_paper_v2 as v2  # noqa: E402  (style, colours)
import plot_paper_v3 as v3  # noqa: E402  (loaders, verdict colours, per-person differences)

OUT = ROOT / "results" / "figures" / "v5"


def _bounds(x, n_claims):
    # one-sided bounds at 1 - 0.05 / n_claims, as the protocols (P-6 to P-8) define them
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(1 - 0.05 / n_claims, len(x) - 1)
    return {"mean": float(x.mean()), "lower": float(x.mean() - t * se), "upper": float(x.mean() + t * se)}


def fig_map_strips(summary):
    n = v3.load("08n_eeg1_h4b_confirmation/08n_decision.json")
    a = v3.load("09a_eeg2_h4b_replication/09a_decision.json")
    l = v3.load("09l_alljoined_confirmation/09l_decision.json")["primary"]
    c = json.loads((v3.RUNS / "10c_nsd_confirmation/10c_summary.json").read_text())["claims"]
    enc, _ = v3._nsd_person_diffs()
    _, _, e3 = v3._nod_person_diffs()
    nod = _bounds(e3, 3)
    strips = [("THINGS-EEG1\nlab EEG, 24 new", np.array([v[0] for v in n["d"].values()]), n["per_comparator"]["avg_sources"], n["delta"]),
              ("THINGS-EEG2\nlab EEG, 10", np.array([v[0] for v in a["d"].values()]), a["per_comparator"]["avg_sources"], a["delta"]),
              ("Alljoined\nconsumer EEG, 16 new", np.array([v[0] for v in l["d"].values()]), l["claims"]["C1"], l["delta"]),
              ("NSD\n7 T fMRI, 8 new", enc, c["C1_lowrank_map_minus_avg_enc_r"], c["C1_lowrank_map_minus_avg_enc_r"]["margin"]),
              ("NOD\n3 T single trials, 21 new", e3, nod, 0.011)]
    if abs(nod["mean"] - 0.0008) > 0.0005:
        raise ValueError(f"NOD E3 mean {nod['mean']:.4f} differs from 11d's +0.0008")
    fig, axes = plt.subplots(1, len(strips), figsize=(v2.W, 2.55), gridspec_kw={"wspace": 0.85})
    rng = np.random.default_rng(3)
    out = {}
    for i, (ax, (name, vals, bounds, margin)) in enumerate(zip(axes, strips)):
        verdict = v3.row_verdict(bounds["lower"], bounds["upper"], margin)
        colour = v2.C["map"]  # colour names the method; the verdict is written
        ax.axhline(0, color="black", lw=0.6)
        ax.axhline(margin, color="#C8102E", lw=0.8, ls=(0, (3, 2)))
        ax.scatter(rng.uniform(-0.18, 0.18, len(vals)), vals, s=9, color=colour, alpha=0.55, linewidths=0, zorder=2)
        ax.plot([0.38, 0.38], [bounds["lower"], bounds["upper"]], color=colour, lw=2, solid_capstyle="butt")
        ax.plot(0.38, bounds["mean"], "D", ms=4.5, mec=colour, mfc=colour if v3.VERDICT[verdict][1] else "white", mew=1.1, zorder=3)
        ax.set_xlim(-0.4, 0.6)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(name, fontsize=6.8)
        label = verdict + (" (predicted)" if name.startswith("NOD") else "")
        if name.startswith("THINGS-EEG2"):  # this contrast clears the margin; P-4's family needed all six and did not
            label = "clears margin*"
        ax.text(0.5, -0.06, label, transform=ax.transAxes, ha="center", va="top", fontsize=6.6, color="#222222", fontweight="bold")
        ax.text(0.5, -0.16, f"{int((vals > 0).sum())}/{len(vals)} above 0", transform=ax.transAxes, ha="center", va="top", fontsize=6.4, color="#555555")
        if i == 0:
            ax.set_ylabel("map − borrowing (r)", fontsize=7)
        ax.tick_params(labelsize=6)
        out[name.split("\n")[0]] = {"mean": round(float(bounds["mean"]), 4), "lower": round(float(bounds["lower"]), 4),
                                    "upper": round(float(bounds["upper"]), 4), "margin": round(float(margin), 4),
                                    "positive": int((vals > 0).sum()), "n": len(vals), "verdict": verdict, "label": label}
    summary["fig_map_strips"] = out
    fig.subplots_adjust(left=0.07, right=0.99, top=0.83, bottom=0.24)
    v2.fit_width(fig)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_map_strips.{ext}", dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


def fig_nsd_budget(summary):
    import csv
    rows = list(csv.DictReader((v3.RUNS / "10k_nsd_budget_curve/results/metrics/10k_scores.csv").open()))
    people = sorted({r["subject"] for r in rows})
    ks = ["500", "1000", "2000", "4000", "all"]
    get = {(r["subject"], r["method"], r["donors"], r["k"]): float(r["top1_centred"]) for r in rows}
    series = {name: np.array([[get[(s, m, d, k)] for k in ks] for s in people])
              for name, (m, d) in {"map": ("map", "ridge"), "own": ("own_ridge", "-"), "comb": ("combination", "ridge")}.items()}
    borrow = np.array([get[(s, "borrowing", "ridge", "0")] for s in people])
    if abs(series["map"][:, -1].mean() - 72.56) > 0.05 or abs(series["own"][:, -1].mean() - 70.72) > 0.05:
        raise ValueError("10k means differ from the reported values (map 72.56, own ridge 70.72 on the whole pool)")
    x = np.arange(len(ks))
    labels = ["500", "1,000", "2,000", "4,000", "whole\nrecording"]
    fig, ax = plt.subplots(1, 2, figsize=(v2.W, 2.45), gridspec_kw={"wspace": 0.32})
    for name, colour, text in (("own", v2.C["own"], "own model"), ("map", v2.C["map"], "map on borrowing")):
        for y in series[name]:
            ax[0].plot(x, y, color=colour, lw=0.6, alpha=0.35)
        ax[0].plot(x, series[name].mean(0), color=colour, lw=2.2, marker="o", ms=3.2, label=text)
    ax[0].axhline(borrow.mean(), color=v2.C["avg"], lw=1.0, ls=(0, (3, 2)), label="borrowing (no labels)")
    ax[0].set_ylabel("identification among 200 (%)")
    ax[0].set_title("A  Accuracy by labelled images")
    ax[0].legend(loc="lower right")
    gap = series["map"] - series["own"]
    for y in gap:
        ax[1].plot(x, y, color=v2.C["map"], lw=0.6, alpha=0.35)
    ax[1].plot(x, gap.mean(0), color=v2.C["map"], lw=2.2, marker="o", ms=3.2, label="map − own model")
    ax[1].plot(x, (series["comb"] - series["own"]).mean(0), color=v2.C["own_map"], lw=1.6, marker="s", ms=2.8,
               label="map + own model − own model")
    ax[1].axhline(0, color="black", lw=0.6)
    ax[1].set_ylabel("difference (points)")
    ax[1].set_title("B  What the map adds over the own model")
    ax[1].legend(loc="upper right")
    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels(labels)
        a.set_xlabel("labelled images of the new person")
    for i in range(len(ks)):
        ax[1].text(x[i], -2.6, f"{int((gap[:, i] > 0).sum())}/{len(people)}", ha="center", va="top", fontsize=6.2, color="#555555")
    ax[1].set_ylim(-4.5, None)
    summary["fig_nsd_budget"] = {"k": ks, "map": series["map"].mean(0).round(2).tolist(), "own": series["own"].mean(0).round(2).tolist(),
                                 "combination": series["comb"].mean(0).round(2).tolist(), "borrowing": round(float(borrow.mean()), 2),
                                 "map_minus_own_positive": [int((gap[:, i] > 0).sum()) for i in range(len(ks))]}
    v2.fit_width(fig)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_nsd_budget.{ext}", dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


def fig_label_reps(summary):
    """THINGS-EEG2 (09t, development): each labelled response averaged over 1, 2 or 4 of its
    repetitions, everything else fixed. Gain of each map over borrowing (post-onset encoding r), one thin line per
    person at 500 labelled images, mean thick; dashed, the mean at 250. A: low-rank map. B: the 17 x 17 window map."""
    import csv
    rows = list(csv.DictReader(open(v3.RUNS / "09t_eeg2_label_reps/09t_people.csv")))
    reps = [1, 2, 4]
    fig, ax = plt.subplots(1, 2, figsize=(v2.W, 2.6), sharey=True)
    out = {}
    for a, (method, title) in zip(ax, (("lowrank", "A  low-rank map"), ("h4b", "B  17 × 17 map per window"))):
        for k, style in ((500, "-"), (250, (0, (3, 2)))):
            g = np.array([[float(r[f"{method}_k{k}_n{n}_r_post"]) - float(r["avg_r_post"]) for n in reps] for r in rows])
            if k == 500:
                for y in g:
                    a.plot(range(3), y, color=v2.C["map"], lw=0.6, alpha=0.35, marker="o", ms=1.8)
            a.plot(range(3), g.mean(0), color="black", lw=2 if k == 500 else 1.2, ls=style, marker="o", ms=3,
                   label=f"mean, {k} images")
            out[f"{method}_k{k}"] = {"mean": g.mean(0).round(4).tolist(), "four_above_one": int((g[:, 2] > g[:, 0]).sum())}
        a.axhline(0, color="black", lw=0.6)
        a.set_xticks(range(3), ["1", "2", "4"])
        a.set_xlim(-0.2, 2.2)
        a.set_xlabel("repetitions averaged per labelled image")
        a.set_title(title, loc="left", fontsize=7.5)
    ax[0].set_ylabel("gain over borrowing (r, 0–790 ms)")
    ax[0].legend(loc="upper left", fontsize=6, frameon=False)
    fig.tight_layout(w_pad=1.5)
    summary["fig_label_reps"] = out
    v2.fit_width(fig)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_label_reps.{ext}", dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


REUSABLE = ("No known copyright restrictions", "United States Government Work", "Attribution License")  # licences that allow reuse


def fig_examples(summary, rows=3, top=3):
    """Decoding examples, readable at page width (10g). The first `rows` of 10g's six
    randomly drawn seen images (seed 370, CC BY or public domain), each system's top `top` guesses among 200. Photos
    come from COCO's servers (cached outside the repository), never from NSD's stimulus files; a guess whose licence
    does not allow reuse is a grey tile with the start of its COCO caption. Also writes the credits of every photo shown."""
    import io
    import os
    import textwrap
    import urllib.request
    from PIL import Image, ImageOps
    src = v3.RUNS / "10g_nsd_licensed_examples/results/metrics/10g_examples.json"
    if not src.exists():
        print("fig_examples: 10g's examples are missing, skipped")
        return
    ex = json.loads(src.read_text())
    cache = Path(os.environ.get("COCO_CACHE", Path.home() / ".cache" / "xsub_coco"))
    cache.mkdir(parents=True, exist_ok=True)

    def photo(m):
        f = cache / f"{m['coco_id']}.jpg"
        if not f.exists():
            with urllib.request.urlopen(m["coco_url"], timeout=60) as r:
                f.write_bytes(r.read())
        return ImageOps.fit(Image.open(io.BytesIO(f.read_bytes())).convert("RGB"), (256, 256))

    systems = [("avg", "borrowing, AlexNet"), ("stack", "map + own model, AlexNet"), ("mixed", "map + own model, DINOv2")]
    keys = list(ex)[:rows]
    ncol = 1 + top * len(systems)
    fig = plt.figure(figsize=(v2.W, 0.78 * rows + 0.42))
    widths = [1, 0.25] + sum([[1] * top + [0.25] for _ in systems], [])[:-1]
    grid = fig.add_gridspec(rows, len(widths), width_ratios=widths, left=0.005, right=0.995, top=1 - 0.36 / (0.78 * rows + 0.42),
                            bottom=0.01, wspace=0.06, hspace=0.08)
    col = [0] + [2 + s * (top + 1) + j for s in range(len(systems)) for j in range(top)]
    credits, hits = {}, {m: [] for m, _ in systems}
    for r, key in enumerate(keys):
        seen = ex[key]["seen"]
        cells = [(seen, None)] + [(g, m) for m, _ in systems for g in ex[key][m][:top]]
        for c, (meta, m) in zip(col, cells):
            ax = fig.add_subplot(grid[r, c])
            ax.set_xticks([]); ax.set_yticks([])
            if meta["licence"] in REUSABLE:
                ax.imshow(photo(meta))
                credits[meta["coco_id"]] = {k: meta[k] for k in ("licence", "flickr_url", "coco_url")}
            else:  # a grey square the size of a photo, with the start of the COCO caption
                ax.imshow(np.full((256, 256, 3), 236, np.uint8))
                w = meta["caption"].rstrip(" .").split()  # fewer words at 6 pt
                words = " ".join(w[:5]) + ("…" if len(w) > 5 else "")
                ax.text(0.5, 0.5, "\n".join(textwrap.wrap(words, 10)[:4]), ha="center", va="center",
                        fontsize=6, color="#333333", transform=ax.transAxes, linespacing=1.05)
            hit = m is not None and meta["coco_id"] == seen["coco_id"]
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color("#C8102E" if hit else ("black" if m is None else "#CCCCCC"))
                sp.set_linewidth(2.2 if hit else (1.2 if m is None else 0.5))
        for m, _ in systems:
            hits[m].append(next((j + 1 for j, g in enumerate(ex[key][m][:top]) if g["coco_id"] == seen["coco_id"]), 0))
    yt = 1 - 0.1 / (0.78 * rows + 0.42)
    fig.text(0.005 + 0.5 * 1 / sum(widths) * 0.99, yt, "seen", ha="center", va="top", fontsize=7.2, fontweight="bold")
    for s, (_, lab) in enumerate(systems):
        x0 = sum(widths[:2 + s * (top + 1)]) / sum(widths)
        x1 = sum(widths[:2 + s * (top + 1) + top]) / sum(widths)
        fig.text(0.005 + 0.99 * (x0 + x1) / 2, yt, lab, ha="center", va="top", fontsize=7.2, fontweight="bold")
    v2.fit_width(fig)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_examples.{ext}", dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    summary["fig_examples"] = {"rows": keys, "rank_of_seen": hits, "credits": {str(k): v for k, v in sorted(credits.items())}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    fig_map_strips(summary)
    fig_nsd_budget(summary)
    fig_label_reps(summary)
    if "--skip-examples" in sys.argv:
        print("fig_examples skipped (--skip-examples)")
    else:
        try:
            fig_examples(summary)
        except OSError as e:  # urllib's URLError is an OSError: no internet and no cached photograph
            print(f"fig_examples skipped: COCO photographs could not be fetched ({e}); rerun online or pass --skip-examples")
    (OUT / "v5_figures_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
