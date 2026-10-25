"""Paper figures in one style, each drawn at the page's own width (6.3 in), with plain labels; the style,
colours and save helpers are shared by the other plot scripts.

Run from the repository root (numpy, scipy, matplotlib; no EEG data, no new fits):

    python scripts/plot_paper_v2.py

Reads the result files in `results/` (with the loaders of `plot_paper_figures.py`). Writes PDF and PNG to
`results/figures/v2/`, among them:
  fig4_alignment       Figure S4: the map against functional alignment, by budget and per 100 ms window (09i, 09j)
  fig5_map             Figure S2: what the map is: every THINGS-EEG2 person's map pattern from two disjoint image sets,
                       and within- vs between-person similarity of the channel mixing over time (09b, 09c, 09h)
  v2_figures_summary.json   the numbers drawn
The other figures it writes are not used in the paper.
"""
from __future__ import annotations

import re

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import plot_paper_figures as v1  # noqa: E402  (loaders only)
from cross_subject_encoding.data.montage import POSITIONS  # noqa: E402
from cross_subject_encoding.data.things_eeg1 import CHANNELS as EEG1_CHANNELS  # noqa: E402
from cross_subject_encoding.data.things_eeg2 import CHANNELS as EEG2_CHANNELS  # noqa: E402
from cross_subject_encoding.eval import h4b_analysis as ha  # noqa: E402

OUT = ROOT / "results" / "figures" / "v2"
W = 6.3  # text width in inches
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "legend.frameon": False,
    "legend.fontsize": 6.8, "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "pdf.fonttype": 42,
    "axes.titlepad": 4})
# one colour per method, in every figure
C = {"map": "#C8102E", "avg": "#7A7A7A", "gu": "#1F6F8B", "own": "#D99A2B", "own_map": "#A86B12",
     "converter": "#5B8C5A", "generic": "#B0B0B0", "procrustes": "#2E5EAA", "procrustes_epoch": "#9DB4D9",
     "srm": "#7B4F9E", "core": "#4A4A4A"}
LABEL = {"avg_sources": "others' average", "gu_window": "weighted ensemble (Gu)", "linear_converter": "converter",
         "own_ridge": "own model", "own_ridge_oc": "own model + output map", "generic": "others' mean map"}
COMP_COLOUR = {"avg_sources": C["avg"], "gu_window": C["gu"], "linear_converter": C["converter"],
               "own_ridge": C["own"], "own_ridge_oc": C["own_map"], "generic": C["generic"]}
DATASET = {"EEG1": "#3B7D4F", "EEG2": "#6A4C93", "ALL": "#D97B29"}
RESPONSE = (100, 200)  # the scored window, shaded where time is on an axis


def fit_width(fig, target=W, tries=6):
    """Resize the figure until its saved (tight) width is `target` inches. The paper prints every figure at the width
    it is drawn for, so no figure is scaled and every label keeps its point size."""
    import io
    for _ in range(tries):  # measured on the saved PDF itself: its text metrics differ slightly from the screen's
        buf = io.BytesIO()
        fig.savefig(buf, format="pdf", metadata={"CreationDate": None})
        box = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+[\d.]+\s+([\d.]+)", buf.getvalue())
        width = (float(box.group(2)) - float(box.group(1))) / 72
        if abs(width - target) < 0.005:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + target - width, h)
    return fig


def save(fig, name, target=W):
    fit_width(fig, target)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


def letter(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", ha="right", va="bottom")


def row(ax, y, values, lower, colour, rng, filled=True, size=9):
    ax.scatter(values, y + rng.uniform(-0.2, 0.2, len(values)), s=size, color=colour, alpha=0.65, linewidths=0, zorder=2)
    ax.plot([lower, values.mean()], [y, y], color="black", lw=1.6, solid_capstyle="butt", zorder=3)
    ax.plot(values.mean(), y, "D", ms=4.2, zorder=4, mec="black", mew=0.9, mfc="black" if filled else "white")


def tidy_x(ax, values, pad=0.006):
    lo = min(0.0, min(float(np.min(v)) for v in values)) - pad
    hi = max(float(np.max(v)) for v in values) + pad
    ax.set_xlim(lo, hi)


def fig2(summary):
    """One axis, three blocks of rows, shared x: what was decided on the 24 reserved THINGS-EEG1 people."""
    rng = np.random.default_rng(13)
    core = v1.load("08e_eeg1_confirmation/08e_decision.json")
    h8 = v1.load("08m_eeg1_h8_confirmation/08m_decision.json")
    h4b = v1.load("08n_eeg1_h4b_confirmation/08n_decision.json")
    _, d_core = v1.per_person_from_decision(core)
    _, d_map = v1.per_person_from_decision(h4b)
    diffs = v1.h8_differences()
    blocks = [
        ("A  Shared network minus …", "rejected",
         [(LABEL[c], d_core[:, i], core["per_comparator"][c]["lower"], COMP_COLOUR[c]) for i, c in enumerate(core["per_comparator"])]),
        ("B  Borrowing minus own data", "supported",
         [("others' average − own model", diffs["avg_sources - own_ridge"], h8["conditions"]["k"]["contrasts"]["avg_sources - own_ridge"]["lower"], C["avg"]),
          ("others' average − own model + map", diffs["avg_sources - own_ridge_oc"], h8["conditions"]["k"]["contrasts"]["avg_sources - own_ridge_oc"]["lower"], C["avg"]),
          ("weighted ensemble − own model", diffs["gu_window - own_ridge"], h8["conditions"]["k"]["contrasts"]["gu_window - own_ridge"]["lower"], C["gu"]),
          ("weighted ensemble − own model + map", diffs["gu_window - own_ridge_oc"], h8["conditions"]["k"]["contrasts"]["gu_window - own_ridge_oc"]["lower"], C["gu"])]),
        ("C  Personal map minus …", "supported",
         [(LABEL[c], d_map[:, i], h4b["per_comparator"][c]["lower"], COMP_COLOUR[c]) for i, c in enumerate(h4b["per_comparator"])]),
    ]
    n_rows = sum(len(b[2]) for b in blocks) + len(blocks)
    fig, ax = plt.subplots(figsize=(W, 0.215 * n_rows + 0.5))
    y, ticks, labels = 0, [], []
    for title, verdict, items in blocks:
        ax.text(-0.012, y - 0.05, title, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5,
                fontweight="bold")
        ax.text(1.0, y - 0.05, verdict, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7,
                style="italic", color="#B00020" if verdict == "rejected" else "#1B6E3A")
        y += 1
        for label, values, lower, colour in items:
            row(ax, y, values, lower, colour, rng)
            ticks.append(y)
            labels.append(label)
            summary.setdefault("fig2", {})[f"{title} | {label}"] = {"mean": float(values.mean()), "lower": float(lower),
                                                                    "positive": int((values > 0).sum()), "n": len(values)}
            y += 1
    ax.axvline(0, color="black", lw=0.6)
    ax.axvline(core["delta"], color="#B00020", lw=0.8, ls=(0, (3, 2)))
    ax.text(core["delta"], -0.9, " margin", color="#B00020", fontsize=6.5, va="bottom")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y - 0.4, -1.0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("difference in correlation per person (100–200 ms; mean of 250 and 500 labelled images)")
    save(fig, "fig2_confirmations")


def fig3(summary):
    rng = np.random.default_rng(14)
    eeg1 = v1.load("08n_eeg1_h4b_confirmation/08n_decision.json")
    eeg2 = v1.load("09a_eeg2_h4b_replication/09a_decision.json")
    allj = v1.load("09l_alljoined_confirmation/09l_decision.json")["primary"]
    al1 = v1.alignment_differences("09i_eeg1_alignment_controls/09i_scores.csv")
    al2 = v1.alignment_differences("09j_eeg2_alignment_controls/09j_scores.csv")
    col = lambda dec, c: v1.per_person_from_decision(dec)[1][:, list(dec["per_comparator"]).index(c)]
    a = np.array([allj["d"][p] for p in sorted(allj["d"], key=int)])
    panels = [("A  Personal map − others' average", [
                  ("THINGS-EEG1", col(eeg1, "avg_sources"), eeg1["per_comparator"]["avg_sources"]["lower"], eeg1["delta"], True, "EEG1"),
                  ("THINGS-EEG2", col(eeg2, "avg_sources"), eeg2["per_comparator"]["avg_sources"]["lower"], eeg2["delta"], True, "EEG2"),
                  ("Alljoined\n(consumer headset)", a[:, 0], allj["claims"]["C1"]["lower"], allj["delta"], True, "ALL")]),
              ("B  Personal map − Procrustes per window", [
                  ("THINGS-EEG1", al1["procrustes"], v1.dev_lower(al1["procrustes"]), None, False, "EEG1"),
                  ("THINGS-EEG2", al2["procrustes"], v1.dev_lower(al2["procrustes"]), None, False, "EEG2"),
                  ("Alljoined\n(consumer headset)", a[:, 2], allj["claims"]["C3"]["lower"], allj["delta"], True, "ALL")])]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.15), sharey=True)
    for ax, (title, items) in zip(axes, panels):
        for j, (label, values, lower, delta, confirm, ds) in enumerate(items):
            row(ax, j, values, lower, DATASET[ds], rng, filled=confirm, size=11)
            if delta is not None:
                ax.plot([delta, delta], [j - 0.32, j + 0.32], color="#B00020", lw=1.1)
            summary.setdefault("fig3", {})[f"{title} | {label}"] = {"mean": float(values.mean()), "lower": float(lower),
                                                                    "positive": int((values > 0).sum()), "n": len(values)}
        tidy_x(ax, [v for _, v, *_ in items])
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(title)
        ax.set_yticks(range(3))
        ax.set_yticklabels([lab for lab, *_ in items])
        ax.set_ylim(2.5, -0.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("difference in correlation per person")
    handles = [plt.Line2D([], [], marker="D", ls="", mfc="black", mec="black", ms=4, label="prespecified test"),
               plt.Line2D([], [], marker="D", ls="", mfc="white", mec="black", ms=4, label="development"),
               plt.Line2D([], [], color="#B00020", lw=1.1, label="margin")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.1))
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig3_datasets")


def fig4(summary):
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.0), gridspec_kw={"width_ratios": [1, 1.35]})
    # each method has its own marker and line style as well as colour
    methods = (("h4b", "personal map", C["map"]), ("srm_d8", "shared response model (8)", C["srm"]),
               ("procrustes_w100", "Procrustes per window", C["procrustes"]),
               ("procrustes_epoch", "Procrustes, whole epoch", C["procrustes_epoch"]))
    style = {"h4b": ("o", "-"), "srm_d8": ("s", (0, (4, 1.5))), "procrustes_w100": ("^", (0, (1, 1))),
             "procrustes_epoch": ("v", (0, (5, 1.5, 1, 1.5)))}
    for r, (name, path) in enumerate([("THINGS-EEG1, 22 development people", "09i_eeg1_alignment_controls/09i_scores.csv"),
                                      ("THINGS-EEG2, 10 people", "09j_eeg2_alignment_controls/09j_scores.csv")]):
        m = v1._means(path)
        ax = axes[r, 0]
        ks = [250, 500, 1000]
        for meth, lab, colour in methods:
            ax.plot(range(3), [m[(meth, str(k))].mean() for k in ks], marker=style[meth][0], ls=style[meth][1], color=colour, ms=3.6, lw=1.3, label=lab)
        ax.axhline(m[("avg", "all")].mean(), color=C["avg"], ls=(0, (3, 2)), lw=1, label="borrowing")
        ax.set_xticks(range(3))
        ax.set_xticklabels(ks)
        ax.set_ylabel("correlation, 100–200 ms")
        ax.set_title(f"{'AB'[r * 0] if False else ''}{name}")
        letter(ax, "AC"[r])
        ax = axes[r, 1]
        avg_w = np.array([v1._means(path, f"r_w{w}")[("avg", "all")].mean() for w in range(10)])
        ax.axvspan(*RESPONSE, color="#EFEFEF", zorder=0)
        for meth, lab, colour in methods[:3]:
            gain = np.array([v1._means(path, f"r_w{w}")[(meth, "500")].mean() for w in range(10)]) - avg_w
            ax.plot(v1.CENTRES, gain, marker=style[meth][0], ls=style[meth][1], color=colour, ms=3.6, lw=1.3)
            summary.setdefault("fig4", {})[f"{name} {meth}"] = [round(float(g), 4) for g in gain]
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("gain over borrowing (r)")
        ax.set_title(f"{name.split(',')[0]}: gain by window (500 images)")
        letter(ax, "BD"[r])
    for ax in axes[1]:
        ax.set_xlabel("labelled images" if ax is axes[1, 0] else "window centre (ms after image onset)")
    axes[0, 1].text(150, axes[0, 1].get_ylim()[0], "scored\nwindow", ha="center", va="bottom", fontsize=6.3, color="#666")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(h_pad=1.2, w_pad=1.6, rect=(0, 0.1, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig4_alignment")


def _topo(ax, xy, tri, values):
    vmax = np.max(np.abs(values)) or 1.0
    ax.tricontourf(tri, values, levels=np.linspace(-vmax, vmax, 13), cmap="RdBu_r")
    ax.add_patch(Circle((0, 0), 1, fill=False, lw=0.5, color="#333"))
    ax.plot(xy[:, 0], xy[:, 1], ".", color="#222", ms=0.9)
    ax.set(xlim=(-0.86, 0.86), ylim=(-1.03, -0.33), aspect="equal")
    ax.axis("off")


def _offdiag_similarity(path):
    z = np.load(path)
    a, b = z["half_a"], z["half_b"]
    off = ~np.eye(a.shape[-1], dtype=bool)
    cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    n, n_w = a.shape[:2]
    within = np.array([[cos(a[p, w][off], b[p, w][off]) for p in range(n)] for w in range(n_w)])
    between = np.array([[np.mean([cos(a[p, w][off], a[q, w][off]) for q in range(n) if q != p]) for p in range(n)]
                        for w in range(n_w)])
    return within, between


def fig5(summary):
    z = np.load(ROOT / "results/09c_eeg2_h4b_ablation/09c_maps_k1000.npz")
    people = [str(p) for p in z["people"]]
    xy = np.array([POSITIONS[c] for c in EEG2_CHANNELS])
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    window = 3  # 100-200 ms
    fig = plt.figure(figsize=(W, 3.3))
    top = fig.add_gridspec(2, len(people), left=0.1, right=0.99, top=0.89, bottom=0.62, hspace=-0.35, wspace=0.04)
    for p, person in enumerate(people):
        ca = ha.leading_component(z["half_a"][p, window])
        cb = ha.leading_component(z["half_b"][p, window], reference=ca["to"])
        for r, comp in enumerate((ca, cb)):
            ax = fig.add_subplot(top[r, p])
            _topo(ax, xy, tri, comp["to"])
            if r == 0:
                ax.set_title(f"person {p + 1}", fontsize=6.5, fontweight="normal", loc="center", pad=1)
            if p == 0:
                ax.text(-0.08, 0.5, ("images 1–1,000", "images 1,001–2,000")[r], transform=ax.transAxes, ha="right",
                        va="center", fontsize=6.3)
    # each map is scaled to its own largest absolute value
    cax = fig.add_axes([0.02, 0.585, 0.10, 0.012])
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap="RdBu_r"), cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cbar.ax.tick_params(labelsize=5.5, length=1.5)
    cbar.ax.set_title("pattern ÷ its largest absolute value", fontsize=5.8, pad=2)
    fig.text(0.01, 0.94, "A", fontsize=9, fontweight="bold")
    fig.text(0.04, 0.94, "Each person's map, fitted twice on disjoint images (THINGS-EEG2, 100–200 ms, leading pattern)",
             fontsize=8, fontweight="bold")
    bottom = fig.add_gridspec(1, 2, left=0.09, right=0.99, top=0.5, bottom=0.12, wspace=0.28)
    for i, (name, path) in enumerate([("THINGS-EEG1, 22 development people", "09b_eeg1_h4b_ablation/09b_maps_k1000.npz"),
                                      ("THINGS-EEG2, 10 people", "09c_eeg2_h4b_ablation/09c_maps_k1000.npz")]):
        within, between = _offdiag_similarity(ROOT / "results" / path)
        ax = fig.add_subplot(bottom[0, i])
        ax.axvspan(*RESPONSE, color="#EFEFEF", zorder=0)
        for vals, lab, colour in ((within, "same person, other images", C["map"]), (between, "different people", C["avg"])):
            m, se = vals.mean(1), vals.std(1, ddof=1) / np.sqrt(vals.shape[1])
            ax.fill_between(v1.CENTRES, m - se, m + se, color=colour, alpha=0.18, lw=0)
            ax.plot(v1.CENTRES, m, "o-", color=colour, ms=3, lw=1.2, label=lab)
            summary.setdefault("fig5", {})[f"{name} {lab}"] = [round(float(x), 3) for x in m]
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(0, color="#999", lw=0.5, ls=":")
        ax.set(xlabel="window centre (ms after image onset)", title=name)
        if i == 0:
            ax.set_ylabel("similarity of channel mixing")
            ax.legend(loc="upper right")
        letter(ax, "BC"[i], x=-0.05)
    save(fig, "fig5_map")


def figs1(summary):
    dec = v1.load("09l_alljoined_confirmation/09l_decision.json")["primary"]
    people = sorted(dec["d"], key=int)
    d = np.array([dec["d"][p] for p in people])
    order = np.argsort(d[:, 2])
    fig, ax = plt.subplots(figsize=(W, 2.2))
    x = np.arange(len(people))
    ax.bar(x - 0.2, d[order, 2], 0.38, color=C["procrustes"], label="personal map − Procrustes per window (supported)")
    ax.bar(x + 0.2, d[order, 0], 0.38, color=C["avg"], label="personal map − others' average (inconclusive)")
    ax.axhline(0, color="black", lw=0.6)
    ax.axhline(dec["delta"], color="#B00020", lw=0.8, ls=(0, (3, 2)), label="margin")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(people[i])) for i in order])
    ax.set(xlabel="Alljoined participant", ylabel="difference in correlation")
    ax.legend(loc="upper left")
    save(fig, "figS1_alljoined")


def figs2(summary):
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.0), gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for r, (name, path) in enumerate([("THINGS-EEG1, 22 development people", "09b_eeg1_h4b_ablation/09b_scores.csv"),
                                      ("THINGS-EEG2, 10 people", "09c_eeg2_h4b_ablation/09c_scores.csv")]):
        m = v1._means(path, keys=("method", "k", "variant"))
        ks = [50, 100, 250, 500, 1000, 2000]
        ax = axes[r, 0]
        for meth, var, lab, colour in (("h4b", "w100r17", "personal map", C["map"]), ("gu_window", "-", "weighted ensemble", C["gu"]),
                                       ("converter", "-", "converter", C["converter"]), ("own_ridge", "-", "own model", C["own"])):
            ax.plot(range(len(ks)), [m[(meth, str(k), var)].mean() for k in ks], "o-", color=colour, ms=3, lw=1.2, label=lab)
        ax.axhline(m[("avg", "all", "-")].mean(), color=C["avg"], ls=(0, (3, 2)), lw=1, label="others' average")
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"{k:,}" for k in ks])
        ax.set(ylabel="correlation, 100–200 ms", title=name)
        avg = m[("avg", "all", "-")]
        gain = lambda var: np.mean([m[("h4b", k, var)] - avg for k in ("250", "500")], 0)
        for c, (variants, title) in enumerate(((["w50r17", "w100r17", "w200r17", "wepochr17"], "one map per …"),
                                               (["w100r1", "w100r2", "w100r4", "w100r8", "w100r17"], "components per window"))):
            ax = axes[r, c + 1]
            g = [gain(v) for v in variants]
            ax.bar(range(len(g)), [x.mean() for x in g], color=C["map"], alpha=0.8, width=0.65)
            for i, x in enumerate(g):
                ax.plot(np.full(len(x), i) + np.linspace(-0.18, 0.18, len(x)), x, ".", color="#333", ms=2.2, alpha=0.6)
            ax.axhline(0, color="black", lw=0.6)
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels(["50", "100", "200", "all"] if c == 0 else ["1", "2", "4", "8", "17"])
            ax.set_title(title)
            if c == 0:
                ax.set_ylabel("gain over others' average")
    axes[1, 0].set_xlabel("labelled images")
    axes[1, 1].set_xlabel("window length (ms)")
    axes[1, 2].set_xlabel("rank")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(h_pad=1.0, w_pad=1.0)
    save(fig, "figS2_ablation")


def figs3(summary):
    sim = v1.rows("09h_eeg2_h4b_sessions/09h_similarity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2), sharey=True)
    for ax, part, title in zip(axes, ("full", "offdiag"), ("Whole map", "Channel mixing only")):
        ax.axvspan(*RESPONSE, color="#EFEFEF", zorder=0)
        for column, colour, lab in (("same_session", C["procrustes"], "same person, same session"),
                                    ("cross_session", C["map"], "same person, other session"),
                                    ("between_people", C["avg"], "different people")):
            v = np.array([[float(r[column]) for r in sim if r["part"] == part and int(r["window"]) == w] for w in range(10)])
            m, se = v.mean(1), v.std(1, ddof=1) / np.sqrt(v.shape[1])
            ax.fill_between(v1.CENTRES, m - se, m + se, color=colour, alpha=0.18, lw=0)
            ax.plot(v1.CENTRES, m, "o-", color=colour, ms=3, lw=1.2, label=lab)
        ax.set(xlabel="window centre (ms after image onset)", title=title)
    axes[0].set_ylabel("map similarity (cosine)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "figS3_sessions")


def figs4(summary):
    m = defaultdict(list)
    for r in v1.rows("09o_eeg1_source_count/09o_scores.csv"):
        m[(r["person"], r["method"], int(r["n_sources"]), r["k"])].append(float(r["r_100_200ms"]))
    people = sorted({k[0] for k in m})
    ns = [2, 5, 10, 15, 21]
    avg = {n: np.array([mean(m[(p, "avg", n, "all")]) for p in people]) for n in ns}
    mp = {n: np.array([mean(mean(m[(p, "h4b", n, k)]) for k in ("250", "500")) for p in people]) for n in ns}
    ex = np.array([mean(mean(m[(p, "exchanged", 21, k)]) for k in ("250", "500")) for p in people])
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.15), gridspec_kw={"width_ratios": [1.2, 1.2, 0.8]})
    ax = axes[0]
    ax.plot(range(5), [avg[n].mean() for n in ns], "o-", color=C["avg"], ms=3, lw=1.2, label="others' average")
    ax.plot(range(5), [mp[n].mean() for n in ns], "o-", color=C["map"], ms=3, lw=1.2, label="personal map")
    ax.set(xticks=range(5), xticklabels=ns, xlabel="people averaged", ylabel="correlation, 100–200 ms",
           title="Population strength")
    ax.legend(loc="lower right")
    ax = axes[1]
    for p in range(len(people)):
        ax.plot(range(5), [mp[n][p] - avg[n][p] for n in ns], color="#BBB", lw=0.6)
    ax.plot(range(5), [(mp[n] - avg[n]).mean() for n in ns], "o-", color=C["map"], ms=3, lw=1.5)
    ax.axhline(0, color="black", lw=0.6)
    ax.set(xticks=range(5), xticklabels=ns, xlabel="people averaged", ylabel="personal gain", title="Gain per person")
    ax = axes[2]
    rng = np.random.default_rng(3)
    for j, (vals, lab, colour) in enumerate(((mp[21] - avg[21], "own map", C["map"]), (ex - avg[21], "another\nperson's map", C["avg"]))):
        ax.scatter(j + rng.uniform(-0.15, 0.15, len(vals)), vals, s=8, color=colour, alpha=0.7, linewidths=0)
        ax.plot([j - 0.25, j + 0.25], [vals.mean()] * 2, color="black", lw=1.4)
    ax.axhline(0, color="black", lw=0.6)
    ax.set(xticks=[0, 1], xticklabels=["own map", "another\nperson's map"], ylabel="minus others' average",
           title="Exchanged maps", xlim=(-0.6, 1.6))
    summary["figS4"] = {"gain_by_n": {n: round(float((mp[n] - avg[n]).mean()), 4) for n in ns},
                        "exchanged_minus_avg": round(float((ex - avg[21]).mean()), 4)}
    fig.tight_layout(w_pad=1.2)
    save(fig, "figS4_population")


def figs5(summary):
    m = defaultdict(list)
    for r in v1.rows("09m_eeg2_modern_features/09m_scores.csv"):
        m[(r["features"], r["person"], r["method"], r["k"])].append(float(r["r_100_200ms"]))
    feats = [("alexnet", "AlexNet\n(2012)"), ("dinov2", "DINOv2\n(2023)"), ("dinov3", "DINOv3\n(2025)"), ("siglip2", "SigLIP 2\n(2025)")]
    people = sorted({k[1] for k in m})
    cols = [("own_full", "all", "own model, all images", "#C9C9C9"), ("avg", "all", "others' average", C["avg"]),
            ("own_ridge", "500", "own model, 500 images", C["own"]), ("h4b", "500", "personal map, 500 images", C["map"])]
    fig, ax = plt.subplots(figsize=(W, 2.2))
    for j, (meth, k, lab, colour) in enumerate(cols):
        for i, (f, _) in enumerate(feats):
            v = np.array([mean(m[(f, p, meth, k)]) for p in people])
            x = i + (j - 1.5) * 0.2
            ax.bar(x, v.mean(), 0.18, color=colour, label=lab if i == 0 else None)
            ax.plot(np.full(len(v), x), v, ".", color="#333", ms=1.8, alpha=0.5)
    ax.set(xticks=range(4), ylabel="correlation, 100–200 ms", ylim=(0, 0.65))
    ax.set_xticklabels([lab for _, lab in feats])
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, "figS5_features")


def _map_embedding():
    """The fingerprint maps (09h): per person four single-repetition maps (two per session), off-diagonal mixing at 0-300 ms."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from fingerprint_maps import offdiag
    z = np.load(ROOT / "results/09h_eeg2_h4b_sessions/09h_maps.npz")
    maps, names = z["maps"].astype(np.float64), [str(n) for n in z["names"]]
    n_people = maps.shape[0]
    order = [names.index(n) for n in ("E1", "E2", "L1", "L2")]
    x = np.stack([offdiag(maps[:, i], [2, 3, 4]) for i in order], 1)  # (people, 4 maps, features)
    return x / np.linalg.norm(x, axis=2, keepdims=True), n_people


def figs6(summary):
    """The map as a fingerprint and not a time shift.
    A: t-SNE of 40 maps (10 THINGS-EEG2 people x 2 maps from each of 2 sessions); B: their cosine similarity matrix;
    C: identification across sessions per window; D: per person, similarity to their own other-session maps against
    everyone else's; E: the map's gain without and after a per-person time shift (17a/17b)."""
    from scipy.spatial import ConvexHull
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    xn, n_people = _map_embedding()
    flat = xn.reshape(-1, xn.shape[-1])
    person = np.repeat(np.arange(n_people), 4)
    session = np.tile([0, 0, 1, 1], n_people)  # earlier, earlier, later, later
    sim = flat @ flat.T
    emb = TSNE(2, perplexity=8, metric="cosine", init="pca", random_state=0).fit_transform(flat)
    sil = float(silhouette_score(flat, person, metric="cosine"))
    pal = plt.cm.tab10(np.arange(n_people))
    fp = json.loads((ROOT / "results/figures/fingerprint.json").read_text())

    fig = plt.figure(figsize=(W, 5.0))
    top = fig.add_gridspec(1, 2, left=0.05, right=0.99, top=0.95, bottom=0.55, wspace=0.12, width_ratios=[1.15, 1])
    a = fig.add_subplot(top[0, 0])
    for p in range(n_people):
        pts = emb[person == p]
        hull = pts[ConvexHull(pts).vertices] if len(pts) >= 3 else pts
        centre = pts.mean(0)
        hull = centre + (hull - centre) * 1.35  # a little room around the points
        a.fill(hull[:, 0], hull[:, 1], color=pal[p], alpha=0.14, lw=0)
        a.plot(np.r_[hull[:, 0], hull[:1, 0]], np.r_[hull[:, 1], hull[:1, 1]], color=pal[p], lw=0.6, alpha=0.6)
        for sess, marker in ((0, "o"), (1, "^")):
            q = pts[session[person == p] == sess]
            a.scatter(q[:, 0], q[:, 1], s=20, marker=marker, color=pal[p], edgecolor="white", linewidth=0.5, zorder=3)
        a.text(*(centre + np.array([0, (pts[:, 1].max() - centre[1]) + 2.2])), f"P{p + 1}", ha="center", va="bottom",
               fontsize=6.5, color=pal[p], fontweight="bold")
    a.scatter([], [], s=16, marker="o", color="#555", label="earlier session")
    a.scatter([], [], s=16, marker="^", color="#555", label="later session")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=6.3, handletextpad=0.2)
    a.margins(x=0.06, y=0.14)
    a.set(xticks=[], yticks=[], title=f"40 personal maps (t-SNE); silhouette by person {sil:.2f}")
    a.spines[["left", "bottom"]].set_visible(False)
    letter(a, "A", x=-0.01)

    b = fig.add_subplot(top[0, 1])
    order = np.lexsort((session, person))
    im = b.imshow(sim[np.ix_(order, order)], cmap="magma", vmin=np.percentile(sim[~np.eye(len(sim), dtype=bool)], 2), vmax=1)
    for k in range(1, n_people):
        b.axhline(4 * k - 0.5, color="white", lw=0.5)
        b.axvline(4 * k - 0.5, color="white", lw=0.5)
    ticks = np.arange(1.5, 4 * n_people, 4)
    b.set_xticks(ticks, [f"P{p + 1}" for p in range(n_people)], fontsize=5.8)
    b.set_yticks(ticks, [f"P{p + 1}" for p in range(n_people)], fontsize=5.8)
    b.tick_params(length=0)
    b.set_title("Similarity between maps (cosine)")
    cb = fig.colorbar(im, ax=b, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    letter(b, "B", x=-0.06)

    bottom = fig.add_gridspec(1, 3, left=0.08, right=0.99, top=0.4, bottom=0.08, wspace=0.42, width_ratios=[1.25, 1, 1])
    c = fig.add_subplot(bottom[0, 0])
    c.axvspan(*RESPONSE, color="#EFEFEF", zorder=0)
    c.plot(v1.CENTRES, list(fp["per_window"].values()), "o-", color=C["map"], ms=3, lw=1.3)
    c.axhline(fp["chance"], color=C["avg"], ls=(0, (3, 2)), lw=1)
    c.text(760, fp["chance"] + 0.03, "chance", ha="right", fontsize=6.3, color=C["avg"])
    c.set(xlabel="window centre (ms)", ylabel="people identified", ylim=(0, 1.05), title="Identified across sessions")
    letter(c, "C")

    d = fig.add_subplot(bottom[0, 1])
    own_other = np.array([sim[np.ix_((person == p) & (session == 0), (person == p) & (session == 1))].mean() for p in range(n_people)])
    others = np.array([sim[np.ix_(person == p, person != p)].mean() for p in range(n_people)])
    for p in range(n_people):
        d.plot([0, 1], [others[p], own_other[p]], color=pal[p], lw=0.9, alpha=0.8)
        d.scatter([0, 1], [others[p], own_other[p]], color=pal[p], s=12, zorder=3)
    d.set(xticks=[0, 1], xlim=(-0.35, 1.35), ylabel="mean similarity", title="Own maps vs others'")
    d.set_xticklabels(["other\npeople", "same person,\nother session"], fontsize=6.3)
    letter(d, "D")

    e = fig.add_subplot(bottom[0, 2])
    rows = []
    for name, path in (("EEG1", "17a_eeg1_latency/17a_summary.json"), ("EEG2", "17b_eeg2_latency/17b_summary.json")):
        s = v1.load(path)
        rows.append((name, s["map_minus_avg"], s["map_after_shift"]))
    x = np.arange(len(rows))
    for off, key, colour, lab in ((-0.18, 1, C["map"], "map"), (0.18, 2, "#E58A99", "after a per-person\ntime shift")):
        means = [r[key]["mean"] for r in rows]
        err = [[r[key]["mean"] - r[key]["lower95"] for r in rows], [r[key]["upper95"] - r[key]["mean"] for r in rows]]
        e.bar(x + off, means, 0.34, color=colour, label=lab, yerr=err, error_kw={"lw": 0.8, "capsize": 2})
    e.set(xticks=x, ylabel="gain over others' average", title="Not a time shift")
    e.set_xticklabels([r[0] for r in rows])
    e.legend(loc="upper left", fontsize=5.8)
    letter(e, "E")
    summary["figS6"] = {"fingerprint_0_300ms": fp["cross_session_0_300ms"], "silhouette_by_person": sil,
                        "own_other_session_similarity": float(own_other.mean()), "other_people_similarity": float(others.mean()),
                        "persons_own_above_others": int((own_other > others).sum()),
                        "latency": {r[0]: {"map": r[1]["mean"], "after_shift": r[2]["mean"]} for r in rows}}
    save(fig, "figS6_fingerprint_latency")


def figs7(summary):
    """How big should the map be (09r): accuracy against the map's free weights, one colour per labelled budget,
    one line style per map family. Per-sensor scaling is drawn for decoding only (a per-sensor rescaling cannot change a
    per-sensor correlation, so its encoding gain is zero at best)."""
    import csv
    rows = list(csv.DictReader(open(ROOT / "results/09r_eeg2_map_size/09r_scores.csv")))
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[(r["method"], r["k"])][r["person"]].append((float(r["r_100_200ms"]), float(r["top1_raw_labelled_mean"])))
    per_person = {key: {p: np.mean(v, 0) for p, v in d.items()} for key, d in by.items()}
    people = sorted(per_person[("avg", "0")])
    avg = np.array([per_person[("avg", "0")][p] for p in people])
    params = {r["method"]: int(r["params"]) for r in rows}
    fams = {"per-window maps (50–200 ms, whole response)": (["full_wepoch", "full_w200", "full_w100", "full_w50"], "-", "o"),
            "17 × 17 map cut to rank 1–8": (["rank1_w100", "rank2_w100", "rank4_w100", "rank8_w100"], (0, (2, 1.5)), "s"),
            "any sensor, any time (low-rank, K = 25–400)": (["lowrank25", "lowrank50", "lowrank100", "lowrank200", "lowrank400"], "-", "D")}
    ks = ["50", "100", "250", "500", "1000"]
    cols = dict(zip(ks, plt.cm.viridis(np.linspace(0.05, 0.85, len(ks)))))
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.7))
    out = {}
    for i, (a, title) in enumerate(zip(ax, ("Encoding gain over others' average", "Decoding a new person (raw score)"))):
        for k in ks:
            for fam, (methods, ls, marker) in fams.items():
                x = [params[m] for m in methods]
                vals = [np.mean([per_person[(m, k)][p][i] for p in people]) - (avg[:, i].mean() if i == 0 else 0) for m in methods]
                a.plot(x, vals, ls=ls, marker=marker, ms=2.6, lw=1.0, color=cols[k], alpha=0.95)
                out.setdefault(f"k{k}", {}).update({m: round(float(v), 4) for m, v in zip(methods, vals)})
            if i == 1:
                a.plot(params["diag_w100"], np.mean([per_person[("diag_w100", k)][p][1] for p in people]), marker="x", ms=4, color=cols[k], ls="none")
        a.set(xscale="log", xlabel="free weights in the personal map", title=title)
        if i == 0:
            a.axhline(0, color="black", lw=0.6)
            a.set_ylabel("gain in r (100–200 ms)")
        else:
            a.axhline(avg[:, 1].mean(), color=C["avg"], ls=(0, (3, 2)), lw=1)
            a.text(1.2e2, avg[:, 1].mean() + 0.6, "others' average", fontsize=6.3, color=C["avg"])
            a.set_ylabel("top-1 of 200 (%)")
        a.axvline(params["full_w100"], color="#BBBBBB", lw=0.6, ls=":")
        letter(a, "AB"[i])
    handles = [plt.Line2D([], [], color=cols[k], lw=2, label=f"{k} images") for k in ks]
    handles += [plt.Line2D([], [], color="#444", ls=ls, marker=mk, ms=3, lw=1, label=fam) for fam, (_, ls, mk) in fams.items()]
    handles += [plt.Line2D([], [], color="#444", marker="x", ls="none", label="per-sensor scaling only")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.2, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(w_pad=2)
    summary["figS7"] = out
    save(fig, "figS7_map_size")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for f in (fig2, fig3, fig4, fig5, figs1, figs2, figs3, figs4, figs5, figs6, figs7):
        f(summary)
    (OUT / "v2_figures_summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))


if __name__ == "__main__":
    main()
