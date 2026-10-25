"""The paper's two fMRI figures, in the style of `plot_paper_v2.py`. Needs nilearn and the public fsaverage
template mesh, so it runs in the plotting image (Docker), not the CPU-test image:

    docker build -t xsub-nilearn:local -f environment/Dockerfile.cortex .
    docker run --rm -v "$PWD":/repo -v <nilearn_data>:/root/nilearn_data -w /repo xsub-nilearn:local \
        python scripts/plot_paper_v3_fmri.py

Both figures need 10f's per-vertex and per-image files, which are derived from NSD and not redistributed under its
terms: run notebook 10f with your own NSD access and put its results/metrics folder at the path below.

Inputs: 10f (`10f_vertex.npz`, `10f_images.npz`: per-vertex encoding r and per-image hit rates, the 8 NSD people of
P-7, k = 250/500, draws 13-15) and 10e (`10e_scores.csv`, `10e_roi.csv`). Development, descriptive.
Writes to `results/figures/v3/`:
  fig_fmri_cortex    Figure 5: group cortical maps on the inflated fsaverage surface: borrowing's accuracy, the map's
                     gain, and the share of the remaining headroom (noise ceiling minus borrowing) that the map recovers
  fig_fmri_regions   Figure 4: the gain against headroom per vertex; per named region against borrowing's accuracy
                     (AlexNet and DINOv2); feature sets per person; per-image identification (borrowing against the
                     mixed system)
  fmri_figures_summary.json   the numbers drawn
Surface panels are rendered one at a time into `results/figures/v3/tiles/` and cropped to the drawn cortex, which
keeps memory low.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import plot_paper_v2 as v2  # noqa: E402  (style, colours)

OUT = ROOT / "results" / "figures" / "v3"
TILES = OUT / "tiles"
RUNS = ROOT / "results"
F10 = RUNS / "10f_nsd_rich_figures/results/metrics"
E10 = RUNS / "10e_nsd_feature_regions/results/metrics"
VIEWS = [("left", "lateral"), ("left", "medial"), ("left", "ventral"), ("right", "ventral"), ("right", "medial"), ("right", "lateral")]
CLASS = {"prf-visualrois": ("early visual", "#4A4A4A"), "floc-faces": ("faces", "#C8102E"), "floc-bodies": ("bodies", "#D99A2B"),
         "floc-places": ("places", "#2E5EAA"), "floc-words": ("words", "#5B8C5A")}
MIN_HEADROOM = 0.05  # share recovered is shown only where the ceiling leaves at least this much r above borrowing


def vertex_maps():
    v = np.load(F10 / "10f_vertex.npz")
    r = {k[2:]: v[k].astype(np.float64) for k in v.files if k.startswith("r_")}
    ceiling = np.sqrt(np.clip(v["noise_ceiling"].astype(np.float64), 0, None) / 100)  # r for 3-repeat averages
    gain = r["lowrank_map"] - r["avg"]
    headroom = ceiling - r["avg"]
    return v, r, ceiling, gain, headroom


def render_tiles(name, values, cmap, vmin, vmax, lh_idx, rh_idx):
    cached = [TILES / f"{name}_{hemi}_{view}.png" for hemi, view in VIEWS]
    if os.environ.get("REUSE_TILES") and all(p.exists() for p in cached):  # layout-only redraws
        return cached
    from nilearn import datasets, plotting
    fs = datasets.fetch_surf_fsaverage("fsaverage")
    full = {"left": np.full(163842, np.nan), "right": np.full(163842, np.nan)}
    full["left"][lh_idx] = values[:len(lh_idx)]
    full["right"][rh_idx] = values[len(lh_idx):]
    TILES.mkdir(parents=True, exist_ok=True)
    paths = []
    for hemi, view in VIEWS:
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(projection="3d")
        plotting.plot_surf_stat_map(fs[f"infl_{hemi}"], full[hemi], hemi=hemi, view=view, bg_map=fs[f"sulc_{hemi}"], axes=ax,
                                    colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax, threshold=1e-9, bg_on_data=False, darkness=0.5)
        path = TILES / f"{name}_{hemi}_{view}.png"
        fig.savefig(path, dpi=220, transparent=False, facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths


def crop(img):
    """Crop a rendered tile to its non-white pixels."""
    mask = (img[..., :3] < 0.97).any(-1)
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def fig_cortex(summary):
    """Group cortical maps: borrowing's accuracy, the map's gain, the share of headroom recovered (6 views each)."""
    v, r, ceiling, gain, headroom = vertex_maps()
    g_avg, g_gain, g_head = r["avg"].mean(0), gain.mean(0), headroom.mean(0)
    share = np.where(g_head > MIN_HEADROOM, g_gain / np.where(g_head > MIN_HEADROOM, g_head, 1), np.nan)
    # the gain row is signed: negative mean gains show in blue instead of being clipped
    rows = [("borrow", g_avg, "viridis", 0, 0.5, "borrowing\n(encoding r)"),
            ("gain_signed", g_gain, "RdBu_r", -0.15, 0.15, "map minus\nborrowing (r)"),
            ("share", np.where(np.isnan(share), 0, np.clip(share, 0, None)), "Purples", 0, 0.3, "share of headroom\nrecovered")]
    tiles = {name: render_tiles(name, vals, cm, lo, hi, v["fsaverage_lh"], v["fsaverage_rh"]) for name, vals, cm, lo, hi, _ in rows}
    fig = plt.figure(figsize=(v2.W, 2.9))
    grid = fig.add_gridspec(3, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.06], left=0.13, right=0.94, top=0.93, bottom=0.01,
                            wspace=0.04, hspace=0.16)
    for i, (name, vals, cm, lo, hi, label) in enumerate(rows):
        for j, path in enumerate(tiles[name]):
            ax = fig.add_subplot(grid[i, j])
            ax.imshow(crop(mpimg.imread(path)))
            ax.axis("off")
            if i == 0:  # view names on one line above the row, not on each tile's own (cropped) height
                cell = grid[0, j].get_position(fig)
                fig.text((cell.x0 + cell.x1) / 2, cell.y1 + 0.012, f"{VIEWS[j][0][0].upper()}H {VIEWS[j][1]}",
                         ha="center", va="bottom", fontsize=6.5)
        cax = fig.add_subplot(grid[i, 6])
        fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(lo, hi), cmap=cm), cax=cax, ticks=[lo, (lo + hi) / 2, hi], format="%g")
        cax.tick_params(labelsize=5.5, length=1.5)
        bb = cax.get_position()
        fig.text(0.125, (bb.y0 + bb.y1) / 2, label, ha="right", va="center", fontsize=6.6)
    ok = ~np.isnan(share)
    summary["fig_fmri_cortex"] = {"vertices": int(len(g_gain)), "mean_gain": float(g_gain.mean()), "mean_borrow": float(g_avg.mean()),
                                  "corr_gain_borrow": float(np.corrcoef(g_avg, g_gain)[0, 1]),
                                  "corr_gain_ceiling": float(np.corrcoef(ceiling.mean(0), g_gain)[0, 1]),
                                  "share_median": float(np.median(share[ok])), "share_vertices": int(ok.sum()),
                                  "gain_per_person": gain.mean(1).round(4).tolist()}
    v2.fit_width(fig, v2.W)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_fmri_cortex.{ext}", dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


def fig_regions(summary):
    roi = pd.read_csv(E10 / "10e_roi.csv")
    scores = pd.read_csv(E10 / "10e_scores.csv")
    fig, axs = plt.subplots(2, 2, figsize=(v2.W, 5.2), gridspec_kw={"wspace": 0.32, "hspace": 0.42})
    out = {}
    # A: gain against headroom per vertex (group means)
    _, _, _, gain, headroom = vertex_maps()
    g_gain, g_head = gain.mean(0), headroom.mean(0)
    ax = axs[0, 0]
    ax.hexbin(g_head, g_gain, gridsize=45, bins="log", cmap="Greys", mincnt=1, linewidths=0)
    fit = np.polyfit(g_head, g_gain, 1)
    xs = np.linspace(np.percentile(g_head, 0.5), np.percentile(g_head, 99.5), 10)
    ax.plot(xs, np.polyval(fit, xs), color=v2.C["map"], lw=1.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("headroom: noise ceiling minus borrowing (r)")
    ax.set_ylabel("map gain (r)")
    rho = float(np.corrcoef(g_head, g_gain)[0, 1])
    per = [float(np.corrcoef(headroom[p], gain[p])[0, 1]) for p in range(len(gain))]
    ax.text(0.03, 0.97, f"r = {rho:.2f} over {len(g_gain):,} vertices\neach person: {min(per):.2f} to {max(per):.2f}",
            transform=ax.transAxes, va="top", fontsize=6.2)
    ax.set_title("A  the gain follows the headroom", fontsize=7.5)
    out["vertex"] = {"corr_gain_headroom": rho, "per_person": per, "slope": float(fit[0]), "intercept": float(fit[1])}
    # B: regions, borrowing r against map gain, AlexNet (filled) to DINOv2 (open), regions seen in at least 6 people
    ax = axs[0, 1]
    m = roi[roi.method.isin(["avg", "lowrank_map"])].groupby(["roi_class", "roi", "feature", "method", "subject"]).r.mean().unstack("method")
    m["gain"] = m.lowrank_map - m.avg
    t = m.groupby(["roi_class", "roi", "feature"]).agg(avg=("avg", "mean"), gain=("gain", "mean"), n=("gain", "size")).reset_index()
    t = t[t.n >= 6]
    regions, labels = {}, []
    for (cls, name), g in t.groupby(["roi_class", "roi"]):
        a = g[g.feature == "alexnet"].iloc[0]
        d = g[g.feature == "dinov2"].iloc[0]
        colour = CLASS[cls][1]
        ax.plot([a.avg, d.avg], [a.gain, d.gain], color=colour, lw=0.6, alpha=0.6)
        ax.scatter(a.avg, a.gain, s=12, color=colour, zorder=3, linewidths=0)
        ax.scatter(d.avg, d.gain, s=12, facecolor="white", edgecolor=colour, zorder=3, linewidths=0.9)
        labels.append([name, float(d.avg), float(d.gain), float(d.gain), colour])
        regions[name] = {"class": CLASS[cls][0], "avg_alexnet": round(float(a.avg), 3), "gain_alexnet": round(float(a.gain), 3),
                         "avg_dinov2": round(float(d.avg), 3), "gain_dinov2": round(float(d.gain), 3), "people": int(a.n)}
    # push labels of nearby points apart vertically, with a leader line when moved
    gap = 0.0075
    for _ in range(200):
        moved = False
        labels.sort(key=lambda L: L[3])
        for i in range(len(labels) - 1):
            a, b = labels[i], labels[i + 1]
            if abs(a[1] - b[1]) < 0.045 and b[3] - a[3] < gap:
                shift = (gap - (b[3] - a[3])) / 2 + 1e-5
                a[3] -= shift
                b[3] += shift
                moved = True
        if not moved:
            break
    for name, x, y, ly, colour in labels:
        ax.annotate(name, (x, y), xytext=(x + 0.006, ly), textcoords="data", fontsize=5.6, color=colour, va="center",
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.4, alpha=0.6) if abs(ly - y) > 0.002 else None)
    for cls, (lab, colour) in CLASS.items():
        ax.scatter([], [], s=12, color=colour, label=lab)
    ax.legend(loc="upper right", fontsize=5.6, handletextpad=0.2, labelspacing=0.25, borderaxespad=0.1)
    ax.set_xlabel("borrowing in the region (r)")
    ax.set_ylabel("map gain (r)")
    ax.set_title("B  named regions: AlexNet (filled), DINOv2 (open)", fontsize=7.5)
    ax.set_ylim(0, 0.17)
    ax.set_xlim(0.13, 0.47)
    out["regions"] = regions
    # C: feature sets, top-1 per person
    ax = axs[1, 0]
    feats = ["alexnet", "dinov2", "siglip2"]
    lab = {"alexnet": "AlexNet", "dinov2": "DINOv2", "siglip2": "SigLIP 2"}
    # the own model at the same budget: the whole-recording reference is the budget figure's
    series = [("avg", "0", v2.C["avg"], "borrowing"), ("own_ridge", "500", v2.C["own"], "own model, 500 images"),
              ("lowrank_map", "500", v2.C["map"], "map, 500 images")]
    fb = {}
    for method, k, colour, name in series:
        sel = scores[(scores.method == method) & (scores.k.astype(str) == k)]
        per_ = sel.groupby(["subject", "feature"]).top1.mean().unstack("feature")[feats]
        for _, row in per_.iterrows():
            ax.plot(range(3), row.values, color=colour, lw=0.6, alpha=0.35)
        ax.plot(range(3), per_.mean().values, color=colour, lw=2, marker="o", ms=3, label=name)
        fb[name] = per_.mean().round(1).to_dict()
    ax.set_xticks(range(3), [lab[f] for f in feats])
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylabel("top-1 among 200 (%)")
    ax.set_title("C  image features, 8 people", fontsize=7.5)
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.2, borderaxespad=0.1)
    ax.set_ylim(10, 100)
    out["features_top1"] = fb
    # D: per image hit rate across people and runs
    ax = axs[1, 1]
    im = np.load(F10 / "10f_images.npz")
    gal = im["in_gallery"]
    x, y = im["hit_avg"][:, gal].mean(0) * 100, im["hit_mixed"][:, gal].mean(0) * 100
    hb = ax.hexbin(x, y, gridsize=18, cmap="Reds", mincnt=1, linewidths=0.2, edgecolors="white", extent=(0, 100, 0, 100))
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("test images per cell", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    ax.plot([0, 100], [0, 100], color="black", lw=0.6, ls=(0, (3, 2)))
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel("borrowing (AlexNet): identified (%)")
    ax.set_ylabel("map + own model, DINOv2 (%)")
    above, below = int((y > x).sum()), int((y < x).sum())
    ax.text(0.03, 0.97, f"{above} of {len(x)} images better, {below} worse, {len(x) - above - below} tied", transform=ax.transAxes, va="top", fontsize=6.2)
    ax.set_title("D  each test image, across people", fontsize=7.5)
    out.update({"images": len(x), "images_better": above, "images_worse": below, "never_identified_avg": int((x == 0).sum()),
                "never_identified_mixed": int((y == 0).sum()), "mean_avg": float(x.mean()), "mean_mixed": float(y.mean())})
    summary["fig_fmri_regions"] = out
    v2.fit_width(fig, 0.86 * v2.W)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_fmri_regions.{ext}", dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)


def main():
    missing = [p.name for p in (F10 / "10f_vertex.npz", F10 / "10f_images.npz") if not p.exists()]
    if missing:
        raise SystemExit(f"missing {', '.join(missing)} in {F10.relative_to(ROOT)}: these come from notebook 10f, which reads "
                         "NSD and is not redistributed (NSD terms); run 10f with your own NSD access first")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"source": "10e, 10f (development)"}
    fig_regions(summary)
    fig_cortex(summary)
    (OUT / "fmri_figures_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "fig_fmri_regions"}, indent=1))
    print(json.dumps({k: v for k, v in summary["fig_fmri_regions"].items() if k not in ("regions",)}, indent=1)[:1500])


if __name__ == "__main__":
    main()
