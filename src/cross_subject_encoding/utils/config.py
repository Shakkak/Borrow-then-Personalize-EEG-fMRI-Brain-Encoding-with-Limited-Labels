"""Shared paths and constants for the cross-subject brain encoding project.

Resolves paths differently on Kaggle (data under /kaggle/input) and elsewhere
(paths under the repository, or PROJECT_ROOT). Every number a notebook
reports depends on a constant here, so change it here, never in a notebook.
"""
from __future__ import annotations

import os
from pathlib import Path

IS_KAGGLE = os.path.exists("/kaggle")
KAGGLE_INPUT = Path("/kaggle/input")


def _kaggle_input(name: str) -> Path:
    """An attached Kaggle input by dataset name: /kaggle/input/<name>, or one or two levels deeper
    (newer Kaggle mounts use /kaggle/input/datasets/<owner>/<name>)."""
    for p in (KAGGLE_INPUT / name, *KAGGLE_INPUT.glob(f"*/{name}"), *KAGGLE_INPUT.glob(f"*/*/{name}")):
        if p.is_dir():
            return p
    return KAGGLE_INPUT / name


if IS_KAGGLE:
    PROJECT_ROOT = Path("/kaggle/working")
    EEG_RAW = _kaggle_input("cross-subject-brain-encoding-eeg")
    NSD_RAW = None  # the NSD notebooks read their attached inputs on Kaggle
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
else:
    # repository root: an editable install keeps this file inside the checkout (src/<pkg>/utils/config.py);
    # PROJECT_ROOT can also be set in the environment, e.g. inside the Docker image
    PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
    EEG_RAW = PROJECT_ROOT / "data" / "raw" / "things_eeg2"
    NSD_RAW = PROJECT_ROOT / "data" / "raw" / "nsd"  # Algonauts 2023 release, mounted by docker run
    DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# The Algonauts 2023 challenge data (NSD), read only by notebook 10a. The Drive folder link is a personal grant
# for accepting the NSD and Algonauts terms and is not distributed.
ALGONAUTS_DRIVE_URL = ""  # not distributed: notebook 10a reads the ALGONAUTS_DRIVE_URL environment variable or Kaggle Secret (request the data at algonauts.csail.mit.edu; NSD terms apply)
NSD_AWS = "https://natural-scenes-dataset.s3.amazonaws.com"  # NSD's public bucket (noise-ceiling files, no account)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

for _d in (DATA_PROCESSED, FIGURES_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def find_artifact(name: str) -> Path:
    """Locate a file written by an earlier notebook.

    Looks in this run's own output dirs first, then (on Kaggle) in every
    attached input, since each notebook runs as its own Kaggle session and
    reads earlier notebooks' outputs as attached datasets.
    """
    for d in (DATA_PROCESSED, METRICS_DIR, FIGURES_DIR, PROJECT_ROOT):
        if (d / name).exists():
            return d / name
    for d in (DATA_PROCESSED, METRICS_DIR):  # checkpoints sit in subfolders here; the export flattens them on Kaggle
        hits = sorted(d.rglob(name)) if d.exists() else []
        if hits:
            return hits[0]
    if KAGGLE_INPUT.exists():
        hits = sorted(KAGGLE_INPUT.rglob(name))
        if len(hits) > 1:  # two attached inputs carry the file, e.g. an old and a new run: say which is used
            print(f"find_artifact: {len(hits)} copies of {name}; using {hits[0]}; others: {[str(h) for h in hits[1:]]}")
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"{name} not found locally or under {KAGGLE_INPUT}; attach the output of the notebook that writes it"
    )


# --- Seeds ------------------------------------------------------------------
SEED = 13  # primary seed
SEEDS = [13, 14, 15]  # k-subset draws and initialisations

# Label budgets of the THINGS-EEG2 analyses and their family-wise error rate
H1_KS = [50, 100, 250, 500, 1000]
H1_ALPHA = 0.05  # family-wise; each budget uses H1_ALPHA / len(H1_KS)

# --- THINGS-EEG2 --------------------------------------------------------------
EEG_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]
EEG_N_TRAIN_IMAGES = 16540  # published count; notebook 01 asserts it
EEG_N_TEST_IMAGES = 200
EEG_N_CHANNELS = 17  # preprocessed release, O, PO and P channels
EEG_N_TIMEPOINTS = 100  # -200 to 800 ms at 100 Hz in the preprocessed release
EEG_WINDOW_MS = 100  # reporting window for per-time scores
# Leave-one-subject-out: each subject is held out once, the other 9 are the donors.
EEG_FOLDS = [{"held_out": s, "source": [t for t in EEG_SUBJECTS if t != s]} for s in EEG_SUBJECTS]
# OSF download ids, read from the OSF API. Each subject zip (about 1.07 GB) holds
# sub-XX/preprocessed_eeg_training.npy and sub-XX/preprocessed_eeg_test.npy: pickled dicts with
# "preprocessed_eeg_data" (float64), "ch_names" and "times".
EEG_OSF_SUBJECT = {"sub-01": "nb8wr", "sub-02": "u43p5", "sub-03": "fy4n6", "sub-04": "py8jz", "sub-05": "2cn7v",
                   "sub-06": "bd83f", "sub-07": "mw8hr", "sub-08": "n9bzm", "sub-09": "fv6jz", "sub-10": "8fyu9"}
EEG_OSF_IMAGE_METADATA = "qkgtf"  # image_metadata.npy, 656,579 bytes: the image order shared by every subject
EEG_OSF_TRAINING_IMAGES = "3v527"  # training_images.zip, 655 MB
EEG_OSF_TEST_IMAGES = "znu7b"  # test_images.zip, 8 MB
NC_SPLITS = 100  # random half splits of the test repetitions for the noise ceiling (seed 13)
ORDER_CHECK_WINDOW_S = (0.08, 0.20)  # where image-specific responses are strongest (peaks 110 to 130 ms)
EEG_NOISE_CEILING = "eeg_noise_ceiling.npz"
EEG_K_SUBSETS = "eeg_k_subsets.json"
EEG_MANIFEST = "eeg_manifest.csv"

# --- Label budget -------------------------------------------------------------
K_GRID = [0, 50, 100, 250, 500, 1000, 2000, None]  # None = every training image of the held-out subject
EEG_TEST_REPS_SCORED = [80, 4]  # ceiling and a realistic session
PAPER_TEST_REPS = 40  # the THINGS-EEG2 paper scores against a 40-repetition test average

# --- Image features and models ------------------------------------------------
RIDGE_FOLDS = 5  # cross-validation folds for each output's penalty, training images only
CORE_WIDTH = 512
LATENT_DIM = 256
