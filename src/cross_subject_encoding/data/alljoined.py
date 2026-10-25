"""Alljoined-1.6M (Xu et al. 2025, arXiv:2508.18571; Hugging Face Alljoined/Alljoined-1.6M, gated, non-commercial)
converted to the THINGS-EEG2 layout.

The release's preprocessed files (`preprocessed_eeg/sub-XX/preprocessed_eeg_{training,test}_flat.npy`) hold one row per
kept trial, (trials, 32 channels, 250 samples at 250 Hz, 0 to 1 s after onset), whitened per session (MVNN), with
`ch_names` and `times`, written with `pickle.dump` despite the `.npy` name (`load_flat`). Rows follow the per-subject metadata table (`experiment_metadata_categories.parquet`): rows of
the partition (`stim_train` / `stim_test`) that are not `dropped`, session by session in table order (the authors'
`save_data` concatenates sessions). `image_path` ends in a five-digit image number. Here: our 17 occipito-parietal
channels in THINGS-EEG2's order, 100 Hz, and an (images, repetitions, channels, time) array in THINGS-EEG2's image
order, missing repetitions as NaN, with each repetition's session.
"""
from __future__ import annotations

import re

import numpy as np

from cross_subject_encoding.data.things_eeg2 import CHANNELS

FLAT = "alljoined_{subject}_preprocessed_eeg_{part}.npy"
PARTITION = {"training": "stim_train", "test": "stim_test"}
N_IMAGES = {"training": 16540, "test": 200}
MAX_REPS = {"training": 4, "test": 80}


class MappingError(ValueError):
    """The metadata does not prove which THINGS-EEG2 image each Alljoined image is."""


# Time labels measured wrong in the release: sub-08's epochs are labelled -0.2 to 1.0 s, but its responses
# line up with everyone else's only when every label is moved 0.2 s later (a 250 Hz, 0-1 s epoch like the others).
# The converted dataset keeps the release's labels; add this to them before any use of time.
TIME_FIX_S = {"sub-08": 0.2}


def true_times(subject: str, times: np.ndarray) -> np.ndarray:
    """The converted file's time labels corrected by TIME_FIX_S (unchanged for everyone else)."""
    return np.round(np.asarray(times, float) + TIME_FIX_S.get(subject, 0.0), 4)


def image_number(paths) -> np.ndarray:
    """The five-digit image number at the end of each `image_path` (the authors' `str[-9:-4]`)."""
    return np.array([int(re.search(r"(\d{5})\.\w+$", str(p)).group(1)) for p in paths])


def path_column(meta, min_share: float = 0.99):
    """The column whose values are image paths ending in a five-digit number (the release calls it `image_path` in the
    per-subject tables; the stimulus table may name it differently), or None."""
    pattern = re.compile(r"\d{5}\.\w+$")
    for c in meta.columns:
        values = meta[c]
        if values.map(lambda v: isinstance(v, str)).all() and values.map(lambda v: bool(pattern.search(v))).mean() >= min_share:
            return c
    return None


def _concept(name: str) -> str:
    """'00001_aardvark' or 'aardvark_01b.jpg' or 'aardvark' -> 'aardvark'."""
    name = str(name).rsplit("/", 1)[-1].lower()
    name = re.sub(r"\.\w+$", "", name)
    name = re.sub(r"^\d+_", "", name)
    return re.sub(r"_\d+[a-z]?$", "", name)


def map_images(meta, eeg2_meta: dict, min_agree: float = 0.99) -> tuple[dict, dict]:
    """Map each Alljoined image number to (part, index in THINGS-EEG2 order). Tries, in order: a column whose values
    are THINGS-EEG2 file names (exact match); otherwise the order assumption (training numbers 0-16539 in THINGS-EEG2
    training order, test numbers 16540-16739 in test order) checked against a column of concept names. Raises
    MappingError if neither is supported by at least `min_agree` of the images. Returns (mapping, evidence)."""
    if "partition" in meta.columns:  # only stimulus images (oddball or catch rows have no THINGS-EEG2 counterpart)
        meta = meta[meta["partition"].isin(PARTITION.values())]
    path_col = path_column(meta)
    if path_col is None:
        raise MappingError(f"no column holds image paths ending in a five-digit number; columns: {list(meta.columns)}; "
                           f"first rows: {meta.head(3).astype(str).to_dict(orient='records')}")
    numbers = image_number(meta[path_col])
    files = {"training": [str(f) for f in eeg2_meta["train_img_files"]], "test": [str(f) for f in eeg2_meta["test_img_files"]]}
    concepts = {"training": [_concept(c) for c in eeg2_meta["train_img_concepts"]], "test": [_concept(c) for c in eeg2_meta["test_img_concepts"]]}
    base = lambda s: re.sub(r"\.\w+$", "", str(s).rsplit("/", 1)[-1].lower())
    by_file = {base(f): (part, i) for part in files for i, f in enumerate(files[part])}
    # text columns other than the path: at least 99% text (a few rows may be empty); only filled rows are compared
    is_text = lambda col: meta[col].map(lambda v: isinstance(v, str))
    columns = [c for c in meta.columns if c != path_col and is_text(c).mean() >= min_agree]
    for c in columns:  # 1. exact file names
        hits = [by_file.get(base(v)) if isinstance(v, str) else None for v in meta[c]]
        share = np.mean([h is not None for h in hits])
        if share >= min_agree:
            mapping = {int(n): h for n, h in zip(numbers, hits) if h is not None}
            return mapping, {"method": f"file names in column '{c}'", "agree": float(share)}
    order = {n: ("training", n) if n < N_IMAGES["training"] else ("test", n - N_IMAGES["training"]) for n in np.unique(numbers)}
    if "partition" in meta.columns:  # the partition must agree with the number ranges
        clash = np.mean([order[n][0] != part for n, part in zip(numbers, meta["partition"].map({v: k for k, v in PARTITION.items()}))])
        if clash > 0:
            raise MappingError(f"{clash:.1%} of rows have a partition that contradicts their image number")
    for c in columns:  # 2. order assumption, checked on concept names
        agree = [concepts[order[n][0]][order[n][1]] == _concept(v) for n, v in zip(numbers, meta[c])
                 if isinstance(v, str) and order[n][1] < N_IMAGES[order[n][0]]]
        if agree and np.mean(agree) >= min_agree:
            return order, {"method": f"number order, concepts in column '{c}' agree", "agree": float(np.mean(agree)), "rows_compared": len(agree)}
    raise MappingError(f"no column proves the image mapping; string columns: {columns}; "
                       f"first rows: {meta.head(3).to_dict(orient='records')}")


def kept_rows(meta, part: str):
    """The metadata rows that correspond, in order, to the flat file's trials."""
    rows = meta[(meta["partition"] == PARTITION[part]) & ~meta["dropped"].astype(bool)]
    return rows.sort_values("session", kind="stable")


def load_flat(path) -> dict:
    """A release file: the authors' `save_data` pickles a dict (data, configs, ch_names, times) into a `.npy` name;
    a real NumPy file is accepted too."""
    import pickle
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError:
        return np.load(path, allow_pickle=True).item()


def select_channels(data: np.ndarray, ch_names, wanted=CHANNELS) -> np.ndarray:
    """(trials, channels, time) -> the wanted channels, by name, in the wanted order."""
    names = [str(c) for c in ch_names]
    missing = [c for c in wanted if c not in names]
    if missing:
        raise ValueError(f"channels missing from the release: {missing}")
    return data[:, [names.index(c) for c in wanted]]


def to_100hz(data: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """250 Hz -> 100 Hz along the last axis (polyphase, up 2 down 5, as THINGS-EEG1's conversion)."""
    from scipy.signal import resample_poly
    out = resample_poly(np.asarray(data, np.float64), 2, 5, axis=-1)
    return out.astype(np.float32), np.round(times[0] + np.arange(out.shape[-1]) / 100, 3)


def organize(data: np.ndarray, rows, mapping: dict, part: str) -> tuple[np.ndarray, np.ndarray]:
    """Flat trials (trials, channels, time) with their metadata rows -> (images, repetitions, channels, time) in
    THINGS-EEG2 order, NaN where a repetition is missing, and the session of each repetition (-1 if missing). The
    repetition axis is MAX_REPS[part] (the paper's 4 or 80) or longer if some image was shown more often
    (some test images have more than 80): no trial is dropped."""
    if len(rows) != len(data):
        raise ValueError(f"{part}: {len(data)} trials in the file, {len(rows)} kept metadata rows")
    places = []
    for number in image_number(rows[path_column(rows) or "image_path"]):
        where, idx = mapping[int(number)]
        if where != part:
            raise ValueError(f"image {number} is a {where} image but sits in the {part} file")
        places.append(idx)
    places = np.asarray(places)
    reps = max(MAX_REPS[part], int(np.bincount(places, minlength=N_IMAGES[part]).max()))
    out = np.full((N_IMAGES[part], reps) + data.shape[1:], np.nan, np.float32)
    sessions = np.full((N_IMAGES[part], reps), -1, np.int16)
    count = np.zeros(N_IMAGES[part], int)
    for trial, (idx, session) in enumerate(zip(places, rows["session"])):
        out[idx, count[idx]], sessions[idx, count[idx]] = data[trial], int(session)
        count[idx] += 1
    return out, sessions
