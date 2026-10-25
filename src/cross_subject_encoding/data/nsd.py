"""NSD through the Algonauts 2023 challenge release.

Layout of one subject's unzipped challenge folder, as the release README describes it (checked by the conversion
notebook 10a, which fails loudly on anything else):

    subjNN/training_split/training_fmri/{lh,rh}_training_fmri.npy   (images, vertices), z-scored per session,
                                                                     repeats averaged
    subjNN/training_split/training_images/train-0001_nsd-00013.png  (index from 1; NSD id from 0)
    subjNN/test_split/test_images/test-0001_nsd-NNNNN.png           (no fMRI released)
    subjNN/roi_masks/*.npy                                          (ROI indices, fsaverage maps)

Files are found by name anywhere under the subject folder, so an extra top-level folder in a zip does not matter.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

NAME = re.compile(r"^(train|test)-(\d+)_nsd-(\d+)\.png$")


def image_ids(folder: Path) -> tuple[np.ndarray, list[Path]]:
    """NSD ids (0-based) of the PNGs in a split folder, ordered by the split index; checks the index runs 1..n."""
    rows = []
    for p in Path(folder).glob("*.png"):
        m = NAME.match(p.name)
        if m is None:
            raise ValueError(f"unexpected image name {p.name}")
        rows.append((int(m.group(2)), int(m.group(3)), p))
    rows.sort()
    if [r[0] for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError(f"{folder}: split index is not 1..{len(rows)}")
    return np.array([r[1] for r in rows], dtype=np.int64), [r[2] for r in rows]


def find_one(root: Path, pattern: str) -> Path:
    """The single file under root matching a glob pattern; zero or several matches raise."""
    hits = sorted(Path(root).rglob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern} under {root}: {len(hits)} matches")
    return hits[0]


def find_dir(root: Path, name: str) -> Path:
    hits = sorted(p for p in Path(root).rglob(name) if p.is_dir())
    if len(hits) != 1:
        raise FileNotFoundError(f"folder {name} under {root}: {len(hits)} matches")
    return hits[0]


def shared_ids(ids_by_subject: dict[str, np.ndarray]) -> np.ndarray:
    """NSD ids present in every subject's array, sorted."""
    sets = [set(map(int, v)) for v in ids_by_subject.values()]
    return np.array(sorted(set.intersection(*sets)), dtype=np.int64)


def noise_ceiling(ncsnr: np.ndarray, n_trials: int = 3) -> np.ndarray:
    """NSD's noise ceiling (percent of variance) for responses averaged over n_trials, from the noise-ceiling SNR
    (Allen et al. 2022): 100 * ncsnr^2 / (ncsnr^2 + 1 / n_trials)."""
    s2 = np.asarray(ncsnr, np.float64) ** 2
    return 100 * s2 / (s2 + 1.0 / n_trials)


def to_float16(x: np.ndarray) -> tuple[np.ndarray, float]:
    """float16 copy of z-scored responses and the largest absolute rounding error (reported, not hidden)."""
    x = np.asarray(x, np.float32)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite fMRI values")
    if np.abs(x).max() >= 60000:
        raise ValueError("values too large for float16")
    h = x.astype(np.float16)
    return h, float(np.abs(h.astype(np.float32) - x).max())
