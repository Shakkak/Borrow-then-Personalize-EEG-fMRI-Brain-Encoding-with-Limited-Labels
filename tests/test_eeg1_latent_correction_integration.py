"""Execute the generated 08h confirmation function on tiny CPU-only synthetic releases."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from cross_subject_encoding.data import things_eeg2
from cross_subject_encoding.data.splits import nested_subsets
from cross_subject_encoding.eval import eeg1_decision
from cross_subject_encoding.eval.paper_correlation import paper_correlation
from cross_subject_encoding.models import paper_linearizing
from cross_subject_encoding.models.baselines import best_source, gu_by_group, window_groups
from cross_subject_encoding.models.ridge import RidgePerOutput
from cross_subject_encoding.models.shared_core import MultiSubjectEncoder, latent, train_shared_images
from cross_subject_encoding.utils.eeg1_pilot import PilotClock, sha256_file, source_file, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["O1", "Oz"]
TIMES = np.round(-0.2 + np.arange(50) / 100, 3)


def save(folder, person, part, data):
    np.save(folder / f"{person}_preprocessed_eeg_{part}.npy",
            {"preprocessed_eeg_data": np.asarray(data, dtype=np.float32), "ch_names": CHANNELS, "times": TIMES})


@pytest.fixture
def world(tmp_path):
    torch.set_num_threads(1)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    eligible = [f"sub-{n:02d}" for n in range(1, 51) if n not in (1, 6, 18, 23)]
    manifest = {"protocol": "N188-source-pilot-draft-v1", "approval_reference": "synthetic fixture only",
                "assignment_approved": True, "channel_mapping_checked": True, "pilot_spending_approved": True,
                "primary_sources": eligible[:22], "confirmation_targets": eligible[22:], "pseudo_targets": eligible[:4]}
    g = np.random.default_rng(13)
    train, test = g.normal(size=(48, 6)), g.normal(size=(8, 6))
    weight = g.normal(size=(6, 100))
    for i, person in enumerate(eligible):
        save(inputs, person, "training", (train @ weight + 0.5 * g.normal(size=(48, 100)) + i / 100).reshape(48, 1, 2, 50))
        if person in manifest["confirmation_targets"]:
            val = (test @ weight)[:, None, :] + 0.5 * g.normal(size=(8, 12, 100))
            save(inputs, person, "test", val.reshape(8, 12, 2, 50))
    art = {n: inputs / n for n in ("eeg1_image_metadata.npy", "eeg1_paper_kpca1000.npz", "08b_readiness.json", "eeg1_k_subsets.json")}
    meta = {"train_img_concepts": ["toy"] * 48, "train_img_files": [f"t{i}.jpg" for i in range(48)],
            "test_img_concepts": ["toy"] * 8, "test_img_files": [f"v{i}.jpg" for i in range(8)]}
    np.save(art["eeg1_image_metadata.npy"], meta)
    order = ["toy/" + f for f in meta["train_img_files"] + meta["test_img_files"]]
    np.savez(art["eeg1_paper_kpca1000.npz"], train=train, test=test, image=np.array(order), kpca_fit_member=np.arange(48) < 24)
    art["08b_readiness.json"].write_text("{}")
    art["eeg1_k_subsets.json"].write_text(json.dumps(nested_subsets(48, [8, 12], manifest["confirmation_targets"], [13, 14, 15])))
    frozen = {"delta": 0.0035, "feature_sha256": sha256_file(art["eeg1_paper_kpca1000.npz"]),
              "k_subsets_sha256": sha256_file(art["eeg1_k_subsets.json"]), "budgets": [8, 12], "draws": [13, 14, 15],
              "core_seeds": [13, 14], "epochs": 2, "window_s": [0.1, 0.2], "alpha": 0.05 / 3, "min_targets": 20, "candidate": "avg_latent"}
    notebook = json.loads((ROOT / "notebooks/08h_eeg1_latent_correction.ipynb").read_text())
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if c["cell_type"] == "code" and "def run_confirmation():" in "".join(c["source"]))
    return SimpleNamespace(root=tmp_path, inputs=inputs, art=art, manifest=manifest, frozen=frozen, source=source)


def run(world, label):
    out = world.root / label
    out.mkdir()
    ns = {"Path": Path, "time": time, "json": json, "np": np, "pd": pd, "torch": torch, "resource": __import__("resource"),
          "C": SimpleNamespace(METRICS_DIR=out, find_artifact=lambda n: world.art[n], EEG_WINDOW_MS=100, RIDGE_FOLDS=3),
          "report_path": out / "08h_status.json", "report": {"status": "disabled", "completed_targets": [], "failed_targets": {}},
          "FROZEN": world.frozen, "APPROVED_MANIFEST": world.manifest, "ALPHAS": [0.01, 1.0, 100.0], "DEVICE": "cpu",
          "INPUT_ROOT": world.inputs, "N_TRAIN": 48, "N_TEST": 8, "N_FEATURES": 6, "N_FIT": 24, "CHANNELS": CHANNELS,
          "N_CHANNELS": 2, "N_TIMES": 50, "N_REPS": 12, "MIN_HOST_BYTES": 0, "SESSION_START": time.monotonic(),
          "RUN_CONFIRMATION": False, "preflight": Mock(return_value={}), "PilotClock": PilotClock,
          "sha256_file": sha256_file, "source_file": source_file, "validate_manifest": validate_manifest,
          "e2": things_eeg2, "dec": eeg1_decision, "paper_correlation": paper_correlation, "pl": paper_linearizing,
          "best_source": best_source, "gu_by_group": gu_by_group, "window_groups": window_groups,
          "RidgePerOutput": RidgePerOutput, "MultiSubjectEncoder": MultiSubjectEncoder, "latent": latent,
          "train_shared_images": train_shared_images}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(world.source, "generated08h", "exec"), ns)
        ns["run_confirmation"]()
    return ns, out


def test_full_run_writes_scores_and_a_decision(world):
    ns, out = run(world, "full")
    report = ns["report"]
    assert report["status"] == "complete_confirmation", report.get("reason")
    assert len(report["completed_targets"]) == 24 and not report["failed_targets"]
    rows = pd.read_csv(out / "08h_scores.csv")
    assert len(rows) == 24 * (1 + 3 * 2 * (4 + 2))
    decision = json.loads((out / "08h_decision.json").read_text())
    assert decision["n_targets"] == 24 and decision["verdict"] in ("supported", "rejected", "inconclusive")
    assert set(decision["per_seed_mean_d"]) == {"13", "14"}
    assert report["eeg_files_opened"] == 22 + 24 * 2


def test_failed_target_is_excluded_not_imputed(world):
    (world.inputs / f"{world.manifest['confirmation_targets'][0]}_preprocessed_eeg_test.npy").unlink()
    ns, out = run(world, "missing")
    assert world.manifest["confirmation_targets"][0] in ns["report"]["failed_targets"]
    decision = json.loads((out / "08h_decision.json").read_text())
    assert decision["n_targets"] == 23 and decision["n_excluded"] == 1


def test_changed_k_subsets_stop_before_any_eeg_read(world):
    world.art["eeg1_k_subsets.json"].write_text("{}")
    with pytest.raises(ValueError, match="k-subsets differ"):
        run(world, "bad_subsets")


def test_candidate_starts_from_the_average_not_the_core(world):
    ns, out = run(world, "label")
    rows = pd.read_csv(out / "08h_scores.csv")
    assert "avg_latent" in set(rows["method"]) and "prior_both" not in set(rows["method"])
    decision = json.loads((out / "08h_decision.json").read_text())
    assert abs(decision["alpha"] - 0.05 / 3) < 1e-12
