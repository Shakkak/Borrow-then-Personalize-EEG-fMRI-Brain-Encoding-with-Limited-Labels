from pathlib import Path

import numpy as np
import pytest

from cross_subject_encoding.data import nsd


def make_split(folder: Path, kind: str, ids):
    folder.mkdir(parents=True)
    for i, nid in enumerate(ids, 1):
        (folder / f"{kind}-{i:04d}_nsd-{nid:05d}.png").write_bytes(b"")


def test_image_ids_order_by_split_index(tmp_path):
    make_split(tmp_path / "imgs", "train", [13, 5, 72999, 0])
    ids, paths = nsd.image_ids(tmp_path / "imgs")
    assert ids.tolist() == [13, 5, 72999, 0]
    assert paths[0].name == "train-0001_nsd-00013.png"


def test_image_ids_rejects_gaps_and_odd_names(tmp_path):
    make_split(tmp_path / "a", "train", [1, 2])
    (tmp_path / "a" / "train-0001_nsd-00001.png").unlink()
    with pytest.raises(ValueError):
        nsd.image_ids(tmp_path / "a")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "image.png").write_bytes(b"")
    with pytest.raises(ValueError):
        nsd.image_ids(tmp_path / "b")


def test_find_one_and_find_dir(tmp_path):
    f = tmp_path / "extra" / "subj01" / "training_split" / "training_fmri" / "lh_training_fmri.npy"
    f.parent.mkdir(parents=True)
    np.save(f, np.zeros((2, 3)))
    assert nsd.find_one(tmp_path, "lh_training_fmri.npy") == f
    assert nsd.find_dir(tmp_path, "training_fmri") == f.parent
    with pytest.raises(FileNotFoundError):
        nsd.find_one(tmp_path, "rh_training_fmri.npy")


def test_shared_ids_and_noise_ceiling():
    got = nsd.shared_ids({"a": np.array([3, 1, 2]), "b": np.array([2, 3, 9]), "c": np.array([3, 2])})
    assert got.tolist() == [2, 3]
    nc = nsd.noise_ceiling(np.array([0.0, 1.0, 1e6]), n_trials=3)
    assert nc[0] == 0 and abs(nc[1] - 75.0) < 1e-9 and nc[2] > 99.99


def test_to_float16_reports_error_and_rejects_bad_values():
    x = np.random.default_rng(0).normal(size=(50, 40)).astype(np.float32)
    h, err = nsd.to_float16(x)
    assert h.dtype == np.float16 and 0 < err < 5e-3
    with pytest.raises(ValueError):
        nsd.to_float16(np.array([np.nan], np.float32))
