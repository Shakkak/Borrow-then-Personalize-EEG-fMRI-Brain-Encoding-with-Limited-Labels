"""THINGS-EEG1 (Grootswagers et al. 2022, OpenNeuro ds003825) in THINGS-EEG2's release format.

The THINGS-EEG1 analyses run the THINGS-EEG2 pipeline unchanged, so this
module turns each THINGS-EEG1 person into the arrays that pipeline reads:

- training: (22,248 images, 1 trial, 17 channels, 100 samples), one single trial per image, in one fixed
  image order shared by everyone (sorted by THINGS concept number, then image number);
- test: (200 validation images, 12 repetitions, 17, 100), ordered by the validation number (0 to 199).

Steps, as close to THINGS-EEG2's release as the data allow (Gifford et al. 2022's preprocessing code):
the authors' continuous files (0.1 to 100 Hz, average reference, 250 Hz; `derivatives/eeglab`), the 17
occipital and parietal channels THINGS-EEG2 keeps, epochs from -200 to 800 ms around every stimulus
event ("E  1", in the order of the events table, as the authors' `preprocessing_things.m` pairs them),
baseline correction on -200 to 0 ms, 250 Hz down to 100 Hz by polyphase resampling, then multivariate
noise normalization with the channel covariance estimated over time within each training epoch (shrunk,
averaged), applied to training and test alike. People 1 and 6 have no validation images; they are training sources only.

Smoke-tested on synthetic data (tests/test_things_eeg1.py), not run on the real files.
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np

S3 = "https://s3.amazonaws.com/openneuro.org/ds003825/"
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 51)]
NO_VALIDATION = ["sub-01", "sub-06"]  # the paper: people 1 and 6 have no validation data
HELD_OUT = [s for s in SUBJECTS if s not in NO_VALIDATION]  # the 48 people with a validation part
N_TRAIN, N_TEST, N_TEST_REPS = 22248, 200, 12
CHANNELS = ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"]
STIM_EVENT = "E  1"
SFREQ_IN, SFREQ_OUT = 250, 100
TMIN, TMAX = -0.2, 0.8  # THINGS-EEG2's window; 100 samples at 100 Hz
PAD = 0.1  # seconds cut on each side after resampling, so filter edges stay outside the window
THINGS_ZIP_OSF = "rdxy2"  # images_THINGS.zip on osf.io/jum2f (5.0 GB; password in password_images.txt)
THINGS_ZIP_PASSWORD = "things4all"  # published in osf.io/jum2f's password_images.txt, for research use only
TEST_IMAGES_OSF = "e485y"  # test_images.csv on osf.io/hd6zk: the 200 validation images


def fetch(url: str, dest: Path | None = None, tries: int = 4) -> bytes | Path:
    """GET with retries (OpenNeuro's bucket is public); to `dest` when given, else the bytes."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=600) as r:
                if dest is None:
                    return r.read()
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    while chunk := r.read(1 << 22):
                        f.write(chunk)
                tmp.rename(dest)
                return dest
        except Exception as e:  # noqa: BLE001 (network errors of every kind are retried)
            if i == tries - 1:
                raise RuntimeError(f"download failed after {tries} tries: {url}: {e}") from e
            time.sleep(10 * (i + 1))


def download_subject(s: str, work: Path) -> dict[str, Path]:
    """The person's continuous EEGLAB files (.set and .fdt, about 250 MB) and events table."""
    files = {"set": f"derivatives/eeglab/{s}_task-rsvp_continuous.set",
             "fdt": f"derivatives/eeglab/{s}_task-rsvp_continuous.fdt",
             "events": f"{s}/eeg/{s}_task-rsvp_events.tsv"}
    out = {}
    for k, rel in files.items():
        dest = work / s / Path(rel).name
        out[k] = dest if dest.exists() else fetch(S3 + rel, dest)
    return out


def read_continuous(set_path: Path) -> tuple[np.ndarray, list[str], float, np.ndarray]:
    """(data channels x samples in volts, channel names, sampling rate, stimulus onsets in samples), via MNE."""
    import mne

    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="error")
    onsets = np.array([round(a["onset"] * raw.info["sfreq"]) for a in raw.annotations if a["description"].strip() == STIM_EVENT.strip()])
    return raw.get_data(), list(raw.ch_names), float(raw.info["sfreq"]), onsets


def read_events(path: Path):
    import pandas as pd

    ev = pd.read_csv(path, sep="\t")
    if "isteststim" not in ev:  # people without validation data have no validation columns
        ev["isteststim"], ev["teststimnumber"] = 0, -1
    return ev


def image_key(ev) -> np.ndarray:
    """THINGS file name of every training presentation, e.g. 'carousel/carousel_11s.jpg'."""
    return np.array([p.replace("\\", "/").split("stimuli/", 1)[-1] for p in ev["stim"]])


def training_order(ev) -> list[str]:
    """The fixed training image order every person is stored in: by THINGS concept number, then image number."""
    tr = ev[ev.isteststim != 1]
    keys = image_key(tr)
    order = np.lexsort((tr["stimnumber"].to_numpy(), tr["objectnumber"].to_numpy()))
    return list(keys[order])


def epoch(data: np.ndarray, onsets: np.ndarray, sfreq: float, tmin: float = TMIN, tmax: float = TMAX, pad: float = PAD) -> np.ndarray:
    """(events, channels, samples at SFREQ_OUT) from continuous data: cut with padding, baseline, resample, crop."""
    from scipy.signal import resample_poly

    a, b = int(round((tmin - pad) * sfreq)), int(round((tmax + pad) * sfreq))
    if (onsets + a < 0).any() or (onsets + b > data.shape[1]).any():
        raise ValueError("an epoch runs past the recording's edges")
    idx = onsets[:, None] + np.arange(a, b)[None, :]
    ep = data[:, idx].transpose(1, 0, 2)  # (events, channels, samples)
    times = np.arange(a, b) / sfreq
    base = (times >= tmin) & (times < 0)
    ep = ep - ep[:, :, base].mean(-1, keepdims=True)
    g = np.gcd(int(SFREQ_OUT), int(sfreq))
    ep = resample_poly(ep, int(SFREQ_OUT) // g, int(sfreq) // g, axis=-1)
    start = int(round(pad * SFREQ_OUT))
    n = int(round((tmax - tmin) * SFREQ_OUT))
    return ep[:, :, start:start + n].astype(np.float32)


def mvnn_whitener(epochs: np.ndarray) -> np.ndarray:
    """Inverse square root of the channel covariance over each epoch's time points, shrunk (Ledoit-Wolf) and
    averaged over epochs: the 'time' mode of THINGS-EEG2's preprocessing code. Its release used covariances
    across repetitions of an image instead, which single trials do not allow (a decision)."""
    from scipy.linalg import fractional_matrix_power
    from sklearn.covariance import LedoitWolf

    sigma = np.mean([LedoitWolf().fit(e.T).covariance_ for e in epochs], 0)
    return np.real(fractional_matrix_power(sigma, -0.5)).astype(np.float32)


def build_person(data: np.ndarray, ch_names: list[str], sfreq: float, onsets: np.ndarray, ev, train_keys: list[str],
                 whiten: bool = True, scale: float = 1.0) -> dict:
    """The person's arrays in THINGS-EEG2's format, plus what was checked on the way.

    whiten=False skips MVNN (H8 needs the unwhitened epochs, P-2); scale multiplies the epochs first
    (1e3 stores volts as millivolts: float16 keeps its relative precision and artifacts up to 65 V fit;
    microvolts can overflow it)."""
    missing = [c for c in CHANNELS if c not in ch_names]
    if missing:
        raise ValueError(f"channels missing: {missing}")
    if len(onsets) < len(ev):
        raise ValueError(f"{len(onsets)} stimulus events in the EEG for {len(ev)} rows of the events table")
    report = {"events_eeg": int(len(onsets)), "events_table": int(len(ev))}
    onsets = onsets[:len(ev)]  # the authors pair the i-th 'E  1' with the i-th row
    ep = epoch(data[[ch_names.index(c) for c in CHANNELS]], onsets, sfreq) * scale
    is_test = (ev.isteststim == 1).to_numpy()
    keys = image_key(ev)
    pos = {k: i for i, k in enumerate(train_keys)}
    tr_rows = np.where(~is_test)[0]
    if len(tr_rows) != len(train_keys) or set(keys[tr_rows]) != set(train_keys):
        raise ValueError("this person's training images differ from the shared training list")
    train = np.empty((len(train_keys), 1, len(CHANNELS), ep.shape[-1]), np.float32)
    train[[pos[k] for k in keys[tr_rows]], 0] = ep[tr_rows]
    w = mvnn_whitener(train[:, 0]) if whiten else np.eye(len(CHANNELS), dtype=np.float32)
    train = np.einsum("ij,nrjt->nrit", w, train)
    test = None
    if is_test.any():
        te_rows = np.where(is_test)[0]
        num = ev.teststimnumber.to_numpy()[te_rows].astype(int)
        counts = np.bincount(num, minlength=N_TEST)
        report["test_reps_min"], report["test_reps_max"] = int(counts.min()), int(counts.max())
        reps = int(counts.min())
        test = np.empty((N_TEST, reps, len(CHANNELS), ep.shape[-1]), np.float32)
        for i in range(N_TEST):
            rows = te_rows[num == i][:reps]
            test[i] = np.einsum("ij,rjt->rit", w, ep[rows])
    times = np.round(TMIN + np.arange(train.shape[-1]) / SFREQ_OUT, 3)
    report["whitened"] = bool(whiten)
    return {"train": train, "test": test, "times": times, "report": report}


def save_release(out: Path, s: str, person: dict, dtype=np.float16) -> list[Path]:
    """`<out>/<s>_preprocessed_eeg_{training,test}.npy`, dicts in THINGS-EEG2's layout, stored as float16.

    Flat, person-prefixed names, because the Kaggle export flattens folders by file name."""
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for part in ("training", "test"):
        arr = person["train" if part == "training" else "test"]
        if arr is None:
            continue
        p = out / f"{s}_preprocessed_eeg_{part}.npy"
        stored = arr.astype(dtype)
        if not np.isfinite(stored).all():  # float16 overflows above 65504 (e.g. microvolts)
            raise ValueError(f"{s}/{part}: values overflow {np.dtype(dtype).name}; rescale before saving")
        np.save(p, {"preprocessed_eeg_data": stored, "ch_names": np.array(CHANNELS), "times": person["times"]}, allow_pickle=True)
        paths.append(p)
    return paths


def person_file(root: Path, s: str, part: str) -> Path | None:
    """One person's converted file anywhere under root (an attached dataset may nest folders)."""
    hits = sorted(Path(root).rglob(f"{s}_preprocessed_eeg_{part}.npy"))
    return hits[0] if hits else None


def validation_images(ev) -> list[str]:
    """THINGS file names of the 200 validation images, by validation number."""
    te = ev[ev.isteststim == 1]
    keys, num = image_key(te), te.teststimnumber.to_numpy().astype(int)
    out = [None] * N_TEST
    for k, i in zip(keys, num):
        out[i] = k
    return out


def extract_things_images(zip_path: Path, wanted: list[str], dest: Path, password: str = THINGS_ZIP_PASSWORD, batch: int = 200) -> dict[str, Path]:
    """Extract only the wanted THINGS images ('concept/file.jpg') from images_THINGS.zip with `unzip -P`
    (the zip is password protected; Python's zipfile decrypts slowly). Returns key -> extracted path."""
    import subprocess
    import zipfile

    with zipfile.ZipFile(zip_path) as z:
        members = {n.rsplit("/", 1)[-1]: n for n in z.namelist() if n.lower().endswith(".jpg")}  # THINGS file names are unique
    base = lambda k: k.rsplit("/", 1)[-1]
    missing = [k for k in wanted if base(k) not in members]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images not in the zip, e.g. {missing[:3]}")
    names = [members[base(k)] for k in wanted]
    for i in range(0, len(names), batch):
        r = subprocess.run(["unzip", "-o", "-q", "-P", password, str(zip_path), *names[i:i + batch], "-d", str(dest)], capture_output=True, text=True)
        if r.returncode not in (0, 1):
            raise RuntimeError(f"unzip failed ({r.returncode}): {r.stderr[:300]}")
    return {k: dest / members[base(k)] for k in wanted}
