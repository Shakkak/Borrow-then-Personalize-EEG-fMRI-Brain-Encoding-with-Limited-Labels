"""THINGS-EEG2 preprocessed release: locating, downloading and loading subjects.

Layout checked. Each subject zip on OSF holds

    sub-XX/preprocessed_eeg_training.npy   (16540, 4, 17, 100) float64, about 0.9 GB
    sub-XX/preprocessed_eeg_test.npy       (200, 80, 17, 100) float64, about 0.2 GB

each a pickled dict with "preprocessed_eeg_data", "ch_names" and "times"
(-0.2 to 0.79 s). The files carry no image ids. The image order is the one
in the shared image_metadata.npy (train_img_files, test_img_files, ...), so
"same order for every subject" is checked statistically by
`order_agreement`, not by matching ids.

Downloads run on Kaggle or any machine with enough disk.
"""
from __future__ import annotations

import io
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from cross_subject_encoding.utils.config import (
    EEG_N_CHANNELS,
    EEG_N_TEST_IMAGES,
    EEG_N_TIMEPOINTS,
    EEG_N_TRAIN_IMAGES,
    EEG_OSF_IMAGE_METADATA,
    EEG_OSF_SUBJECT,
)

OSF = "https://osf.io/download/{}/"
PARTS = {"training": (EEG_N_TRAIN_IMAGES, 4), "test": (EEG_N_TEST_IMAGES, 80)}


WAITS_S = (30, 60, 120, 240, 300)  # delays between download attempts


def fetch(url: str, path: Path | None = None, waits=WAITS_S, log=print) -> bytes | Path:
    """Download url (to path, or into memory), retrying after an HTTP or network error.

    OSF answers with a redirect to a short-lived storage link, and it sometimes refuses a request
    (403 or 429) when many large files are fetched in a row. Each attempt asks OSF again, so it gets a fresh link. A zip that comes
    back incomplete counts as a failed attempt.
    """
    req = lambda: urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (cross-subject-brain-encoding)"})
    for attempt in range(len(waits) + 1):
        try:
            with urllib.request.urlopen(req(), timeout=600) as r:
                if path is None:
                    return r.read()
                with open(path, "wb") as f:
                    shutil.copyfileobj(r, f, length=1 << 22)
            if str(path).endswith(".zip") and not zipfile.is_zipfile(path):
                raise OSError("incomplete zip")
            return Path(path)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if attempt == len(waits):
                raise
            log(f"download failed ({e}); retrying in {waits[attempt]} s ({attempt + 1} of {len(waits)})")
            time.sleep(waits[attempt])


# channel order of the release files, read from all 10 people by 09g (manifest); NOT THINGS-EEG1's order
CHANNELS = ["Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2"]
FLAT = "things-eeg2_{subject}_preprocessed_eeg_{part}.npy"  # the 09g dataset's flat, unambiguous names
SHUFFLE_SEED = 20200220  # the authors' preprocessing seed (eeg_encoding, 02_eeg_preprocessing/preprocessing.py)


def subject_file(root: Path, subject: str, part: str) -> Path | None:
    """The .npy for one subject and part anywhere under root (an attached dataset may nest folders): the
    release layout `sub-XX/preprocessed_eeg_<part>.npy` or the 09g dataset's flat `FLAT` name."""
    hits = sorted(Path(root).rglob(f"{subject}/preprocessed_eeg_{part}.npy")) + sorted(Path(root).rglob(FLAT.format(subject=subject, part=part)))
    return hits[0] if hits else None


def attached(root: Path) -> bool:
    """True if root holds THINGS-EEG2 test files in either layout (an attached dataset, or a download)."""
    root = Path(root)
    return root.is_dir() and (any(root.rglob("preprocessed_eeg_test.npy")) or any(root.rglob("things-eeg2_*_preprocessed_eeg_test.npy")))


def write_float32(src: Path, dst: Path) -> dict:
    """Copy one release file as float32 in the same pickled-dict format (every notebook loads float32 anyway,
    via `load_part`). Returns its shape and the largest absolute change the cast made."""
    d = np.load(src, allow_pickle=True).item()
    x64 = np.asarray(d["preprocessed_eeg_data"])
    x32 = x64.astype(np.float32)
    err = float(np.max(np.abs(x32.astype(np.float64) - x64)))
    np.save(dst, {"preprocessed_eeg_data": x32, "ch_names": d["ch_names"], "times": d["times"]}, allow_pickle=True)
    return {"shape": list(x32.shape), "source_dtype": str(x64.dtype), "max_cast_error": err}


def session_labels(part: str, seed: int = SHUFFLE_SEED) -> np.ndarray:
    """Session label of each repetition in the released order, reconstructed from the authors' `save_prepr`: sessions are concatenated in order, then the repetition axis is shuffled with
    `sklearn.utils.shuffle(np.arange(n), random_state=seed)`, the same permutation for every image.
    test: 80 repetitions, 20 from each of sessions 0-3. training: 4 repetitions, 2 from each of the image's two
    sessions, labelled 0 (earlier) and 1 (later). Must be verified on data (`session_agreement`) before use."""
    from sklearn.utils import shuffle
    n = PARTS[part][1]
    original = np.arange(n) // (20 if part == "test" else 2)
    return original[shuffle(np.arange(n), random_state=seed)]


def rep_correlations(data: np.ndarray) -> np.ndarray:
    """(images, reps, ...) -> (reps, reps) correlation of whole repetitions across images and outputs, each
    output centred over images first: how alike two repetitions' image-by-image responses are."""
    x = np.asarray(data, np.float64).reshape(data.shape[0], data.shape[1], -1)
    x = x - x.mean(0, keepdims=True)
    z = x.transpose(1, 0, 2).reshape(x.shape[1], -1)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return z @ z.T


def session_agreement(corr: np.ndarray, labels: np.ndarray, n_perm: int = 1000, seed: int = 0) -> dict:
    """Mean correlation of same-session minus different-session repetition pairs, and a one-sided p from
    relabelling (group sizes kept). With 4 repetitions in two pairs there are only 3 partitions, so `rank`
    (1 = the labelled partition gives the largest difference) is the useful statistic there."""
    off = ~np.eye(len(labels), dtype=bool)
    def diff(lab):
        same = (lab[:, None] == lab[None, :]) & off
        return float(corr[same].mean() - corr[~same & off].mean())
    observed = diff(labels)
    rng = np.random.default_rng(seed)
    null = np.array([diff(rng.permutation(labels)) for _ in range(n_perm)])
    return {"difference": observed, "p": float((1 + np.sum(null >= observed)) / (1 + n_perm)),
            "rank": int(1 + np.sum(np.unique(np.round(null, 12)) > round(observed, 12)))}


def download_subject(subject: str, dest: Path, parts=("training", "test")) -> Path:
    """Fetch one subject zip from OSF into dest, extract the requested parts, delete the zip."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / f"{subject}.zip"
    fetch(OSF.format(EEG_OSF_SUBJECT[subject]), zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for part in parts:
            z.extract(f"{subject}/preprocessed_eeg_{part}.npy", dest)
    zip_path.unlink()
    return dest / subject


def load_part(path: Path, dtype=np.float32) -> dict:
    """Load one preprocessed file as {"data": array, "ch_names": list, "times": array}.

    float32 halves the memory (about 0.45 GB per training file) and is ample for EEG amplitudes.
    """
    d = np.load(path, allow_pickle=True).item()
    return {"data": np.asarray(d["preprocessed_eeg_data"], dtype=dtype),
            "ch_names": [str(c) for c in d["ch_names"]],
            "times": np.asarray(d["times"], dtype=np.float64)}


def check_part(loaded: dict, part: str) -> list[str]:
    """Problems with one loaded file; an empty list means it matches the published shapes."""
    n_img, n_rep = PARTS[part]
    want = (n_img, n_rep, EEG_N_CHANNELS, EEG_N_TIMEPOINTS)
    problems = []
    if loaded["data"].shape != want:
        problems.append(f"{part}: shape {loaded['data'].shape}, expected {want}")
    if len(loaded["ch_names"]) != EEG_N_CHANNELS:
        problems.append(f"{part}: {len(loaded['ch_names'])} channel names, expected {EEG_N_CHANNELS}")
    if not np.isfinite(loaded["data"]).all():
        problems.append(f"{part}: contains NaN or inf")
    return problems


def load_image_metadata(root: Path | None = None) -> dict:
    """image_metadata.npy from an attached copy, or straight from OSF (656 KB)."""
    if root is not None:
        hits = sorted(Path(root).rglob("image_metadata.npy"))
        if hits:
            return np.load(hits[0], allow_pickle=True).item()
    raw = fetch(OSF.format(EEG_OSF_IMAGE_METADATA))
    return np.load(io.BytesIO(raw), allow_pickle=True).item()


def image_profile(test_data: np.ndarray, times: np.ndarray, window_s: tuple[float, float]) -> np.ndarray:
    """One number per test image and channel: the repetition-averaged response in the window, (200, channels)."""
    sel = (times >= window_s[0]) & (times <= window_s[1])
    return test_data.mean(axis=1)[:, :, sel].mean(axis=2)


def order_agreement(profiles: dict[str, np.ndarray], n_perm: int = 200, seed: int = 0) -> list[dict]:
    """For every subject pair, how strongly their per-image responses agree, against shuffled image orders.

    If two subjects stored the test images in the same order, their image-by-channel profiles correlate
    clearly above zero. A different order would give a correlation near the shuffled null.
    """
    rng = np.random.default_rng(seed)
    subs = sorted(profiles)
    z = {s: (p - p.mean(0)) / (p.std(0) + 1e-12) for s, p in profiles.items()}  # standardize per channel
    rows = []
    for i, a in enumerate(subs):
        for b in subs[i + 1:]:
            r = float(np.corrcoef(z[a].ravel(), z[b].ravel())[0, 1])
            null = [float(np.corrcoef(z[a].ravel(), z[b][rng.permutation(len(z[b]))].ravel())[0, 1])
                    for _ in range(n_perm)]
            rows.append({"a": a, "b": b, "r": r, "null_mean": float(np.mean(null)),
                         "null_p99": float(np.quantile(null, 0.99)),
                         "p_perm": float((np.sum(np.array(null) >= r) + 1) / (n_perm + 1))})
    return rows
