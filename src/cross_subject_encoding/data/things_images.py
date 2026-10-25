"""The THINGS-EEG2 stimulus images and their order.

The two zips on OSF hold `training_images/<concept>/<file>.jpg` (16,540
files, 655 MB) and `test_images/<concept>/<file>.jpg` (200 files, 8 MB),
checked by reading their file lists. The zips are not
in presentation order, so the order always comes from `image_metadata.npy`:
image i of a split is `<concept[i]>/<file[i]>`, the same i that indexes the
EEG arrays.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from cross_subject_encoding.data.things_eeg2 import OSF, fetch
from cross_subject_encoding.utils.config import EEG_OSF_TEST_IMAGES, EEG_OSF_TRAINING_IMAGES

ZIPS = {"training": ("training_images", EEG_OSF_TRAINING_IMAGES), "test": ("test_images", EEG_OSF_TEST_IMAGES)}
META_KEYS = {"training": ("train_img_concepts", "train_img_files"), "test": ("test_img_concepts", "test_img_files")}


def find_image_root(root: Path, split: str) -> Path | None:
    """The `training_images` or `test_images` folder anywhere under root (an attached dataset may nest it)."""
    folder = ZIPS[split][0]
    hits = sorted(p for p in Path(root).rglob(folder) if p.is_dir())
    return hits[0] if hits else None


def download_images(dest: Path, splits=("training", "test")) -> dict[str, Path]:
    """Fetch and extract the image zips from OSF into dest, deleting each zip after extraction."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in splits:
        folder, oid = ZIPS[split]
        if (dest / folder).is_dir():
            out[split] = dest / folder
            continue
        zip_path = dest / f"{folder}.zip"
        fetch(OSF.format(oid), zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
        zip_path.unlink()
        out[split] = dest / folder
    return out


def image_paths(image_root: Path, meta: dict, split: str) -> list[Path]:
    """Paths in metadata order; raises if any image is missing, since the order is what links images to EEG."""
    concepts, files = (meta[k] for k in META_KEYS[split])
    paths = [Path(image_root) / c / f for c, f in zip(concepts, files)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {split} images missing, e.g. {missing[:3]}")
    return paths
