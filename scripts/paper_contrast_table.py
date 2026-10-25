"""Every prespecified contrast of P-1 to P-8 as LaTeX table rows, read from the decision files (Table S2).
Run from the repository root; prints the rows.

    python scripts/paper_contrast_table.py
"""
import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "results"
NAME = {"avg_sources": "borrowing", "gu_window": "Gu's ensemble", "linear_converter": "linear converter",
        "own_ridge": "own model", "own_ridge_oc": "own model + output map", "generic": "averaged correction"}


def load(rel):
    return json.loads((RUNS / rel).read_text())


def row(claim, x, margin, n, positive, prec=4):
    f = lambda v: f"{v:+.{prec}f}".replace("+-", "-").replace("-", "$-$")
    return f"\\quad {claim} & {f(x['mean'])} & {f(x['lower'])} & {f(x['upper'])} & {margin:.{prec}f} & {positive}/{n} \\\\"


def head(text):
    return f"\\multicolumn{{6}}{{@{{}}l}}{{{text}}} \\\\"


def main():
    """Rows grouped by test, named as in the paper's Table 2 (PROVENANCE.md maps the names to the protocol files)."""
    out = []
    eeg = {"P-1": ("08e_eeg1_confirmation/08e_decision.json", "shared network", r"\textbf{THINGS-EEG1}, first test: the shared network"),
           "P-1 follow-up": ("08h_eeg1_latent_correction/08h_decision.json", "borrowing + correction", r"\textbf{THINGS-EEG1}, follow-up 1: the network correction"),
           "P-3": ("08n_eeg1_h4b_confirmation/08n_decision.json", "map", r"\textbf{THINGS-EEG1}, follow-up 3: the map"),
           "P-4": ("09a_eeg2_h4b_replication/09a_decision.json", "map", r"\textbf{THINGS-EEG2}: the map")}
    for proto in ("P-1", "P-1 follow-up", "P-2", "P-3", "P-4"):
        if proto == "P-2":
            out.append(head(r"\textbf{THINGS-EEG1}, follow-up 2: borrowing against the own model"))
            h8 = load("08m_eeg1_h8_confirmation/08m_decision.json")["conditions"]["k"]
            for k, v in h8["contrasts"].items():
                a, b = k.split(" - ")
                out.append(row(f"{NAME[a]} $-$ {NAME[b]}", v, 0.0035478, 24, v["positive"]))
            continue
        rel, cand, title = eeg[proto]
        d = load(rel)
        out.append(head(title))
        for k, v in d["per_comparator"].items():
            out.append(row(f"{cand} $-$ {NAME.get(k, k)}", v, d["delta"], d["n_targets"], v["positive_targets"]))
    aj = load("09l_alljoined_confirmation/09l_decision.json")["primary"]
    out.append(head(r"\textbf{Alljoined-1.6M}: the map"))
    for c, lab in (("C1", "map $-$ borrowing"), ("C2", "Gu + map $-$ Gu"), ("C3", "map $-$ Procrustes-style control")):
        v = aj["per_comparator"][c]
        out.append(row(lab, v, aj["delta"], aj["n_targets"], v["positive_targets"]))
    nsd = load("10c_nsd_confirmation/10c_summary.json")["claims"]
    out.append(head(r"\textbf{NSD}: the map and the combination"))
    for k, lab, prec in (("C1_lowrank_map_minus_avg_enc_r", "map $-$ borrowing", 4),
                         ("C2_stack_minus_avg_top1", "combination $-$ borrowing (centred), top-1", 2),
                         ("C3_stack_minus_own_ridge_top1", "combination $-$ own model, top-1", 2)):
        v = nsd[k]
        out.append(row(lab, v, v["margin"], v["n"], v["positive"], prec))
    nod = load("11d_nod_confirmation/11d_summary.json")["family"]
    out.append(head(r"\textbf{NOD}: three predictions"))
    for k, lab, prec in (("E1_avg_minus_own_ridge_enc_r", "borrowing $-$ own model", 4),
                         ("E2_mixed_minus_own_ridge_top1", "combination (DINOv2) $-$ own model, top-1", 2),
                         ("E3_lowrank_map_minus_avg_enc_r", "map $-$ borrowing", 4)):
        v = nod[k]
        out.append(row(lab, v, v["margin"], v["n"], v["positive"], prec))
    print("\n".join(out))


if __name__ == "__main__":
    main()
