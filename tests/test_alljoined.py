"""Alljoined-1.6M conversion helpers: image mapping (proved or refused), trial order, channels, resampling."""
import numpy as np
import pandas as pd
import pytest

from cross_subject_encoding.data import alljoined as aj


def eeg2_meta(n_train=6, n_test=2):
    return {"train_img_files": [f"c{i // 2}_{i:02d}b.jpg" for i in range(n_train)],
            "train_img_concepts": [f"{i // 2 + 1:05d}_c{i // 2}" for i in range(n_train)],
            "test_img_files": [f"t{i}_01b.jpg" for i in range(n_test)],
            "test_img_concepts": [f"{i + 1:05d}_t{i}" for i in range(n_test)]}


def test_image_number_and_concepts():
    assert aj.image_number(["stimuli/00042.jpg", "x/16739.png"]).tolist() == [42, 16739]
    assert aj._concept("00001_aardvark") == aj._concept("aardvark_01b.jpg") == "aardvark"


def test_mapping_by_file_names_and_refusal(monkeypatch):
    monkeypatch.setattr(aj, "N_IMAGES", {"training": 6, "test": 2})
    m = eeg2_meta()
    meta = pd.DataFrame({"image_path": [f"img/{n:05d}.jpg" for n in (3, 0, 7)],
                         "original": ["c1_03b.jpg", "c0_00b.jpg", "t1_01b.jpg"]})
    mapping, ev = aj.map_images(meta, m)
    assert mapping[3] == ("training", 3) and mapping[7] == ("test", 1) and "file names" in ev["method"]
    by_concept = pd.DataFrame({"image_path": [f"img/{n:05d}.jpg" for n in (3, 0, 7)], "category": ["c1", "c0", "t1"]})
    mapping, ev = aj.map_images(by_concept, m)
    assert mapping[7] == ("test", 1) and "order" in ev["method"]
    wrong = pd.DataFrame({"image_path": [f"img/{n:05d}.jpg" for n in (3, 0, 7)], "category": ["c2", "c2", "t0"]})
    with pytest.raises(aj.MappingError):
        aj.map_images(wrong, m)


def test_organize_puts_trials_under_their_image_and_marks_sessions(monkeypatch):
    monkeypatch.setattr(aj, "N_IMAGES", {"training": 3, "test": 1})
    monkeypatch.setattr(aj, "MAX_REPS", {"training": 2, "test": 2})
    meta = pd.DataFrame({"image_path": ["00002.jpg", "00000.jpg", "00002.jpg", "00001.jpg", "00003.jpg"],
                         "partition": ["stim_train", "stim_train", "stim_train", "stim_train", "stim_test"],
                         "session": [2, 1, 1, 2, 1], "dropped": [False, False, False, True, False]})
    rows = aj.kept_rows(meta, "training")
    assert rows["image_path"].tolist() == ["00000.jpg", "00002.jpg", "00002.jpg"]  # session 1 first, dropped removed
    data = np.arange(3, dtype=float)[:, None, None] * np.ones((3, 1, 2))
    mapping = {0: ("training", 0), 1: ("training", 1), 2: ("training", 2), 3: ("test", 0)}
    out, sessions = aj.organize(data, rows, mapping, "training")
    assert out[0, 0, 0, 0] == 0 and out[2, 0, 0, 0] == 1 and out[2, 1, 0, 0] == 2
    assert np.isnan(out[1]).all() and sessions[2].tolist() == [1, 2] and sessions[1].tolist() == [-1, -1]
    with pytest.raises(ValueError, match="kept metadata rows"):
        aj.organize(data[:2], rows, mapping, "training")


def test_channels_by_name_and_resampling():
    names = ["Fp1"] + list(reversed(aj.CHANNELS))
    data = np.arange(len(names), dtype=float)[None, :, None] * np.ones((2, 1, 250))
    sel = aj.select_channels(data, names)
    assert [names[int(v)] for v in sel[0, :, 0]] == aj.CHANNELS
    with pytest.raises(ValueError, match="missing"):
        aj.select_channels(data, names[:5])
    x, t = aj.to_100hz(np.sin(np.linspace(0, 4, 250))[None, None], np.arange(250) / 250)
    assert x.shape == (1, 1, 100) and t[0] == 0 and t[-1] == 0.99


def test_load_flat_reads_the_authors_pickle_and_numpy(tmp_path):
    import pickle
    d = {"preprocessed_eeg_data": np.zeros((2, 3, 4)), "ch_names": ["a", "b", "c"], "times": np.arange(4) / 250}
    with open(tmp_path / "p.npy", "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(tmp_path / "n.npy", d, allow_pickle=True)
    for name in ("p.npy", "n.npy"):
        assert aj.load_flat(tmp_path / name)["preprocessed_eeg_data"].shape == (2, 3, 4)


def test_mapping_finds_the_path_column_by_content_and_refuses_without_one(monkeypatch):
    monkeypatch.setattr(aj, "N_IMAGES", {"training": 6, "test": 2})
    m = eeg2_meta()
    meta = pd.DataFrame({"stimulus": [f"stim/{n:05d}.jpg" for n in (3, 0, 7)], "category": ["c1", "c0", "t1"]})
    mapping, ev = aj.map_images(meta, m)
    assert mapping[3] == ("training", 3) and "category" in ev["method"]
    with pytest.raises(aj.MappingError, match="five-digit"):
        aj.map_images(pd.DataFrame({"category": ["c1"], "idx": [3]}), m)


def test_mapping_tolerates_rows_without_a_category_and_checks_partitions(monkeypatch):
    """The real stimulus table has 9 extra rows with no category; they must not hide the column."""
    monkeypatch.setattr(aj, "N_IMAGES", {"training": 6, "test": 2})
    m = eeg2_meta()
    rows = [{"partition": "stim_train", "fname": f"{n:05d}.jpg", "category_name": f"c{n // 2}"} for n in range(6)]
    rows += [{"partition": "stim_test", "fname": f"{6 + i:05d}.jpg", "category_name": f"t{i}"} for i in range(2)]
    rows += [{"partition": "oddball", "fname": "00008.jpg", "category_name": None}]
    mapping, ev = aj.map_images(pd.DataFrame(rows), m)
    assert mapping[7] == ("test", 1) and "category_name" in ev["method"] and 8 not in mapping
    rows[0]["partition"] = "stim_test"
    with pytest.raises(aj.MappingError, match="contradicts"):
        aj.map_images(pd.DataFrame(rows), m)


def test_organize_keeps_extra_repetitions(monkeypatch):
    """A test image can have more repetitions than the paper's 80; the array grows instead of failing."""
    monkeypatch.setattr(aj, "N_IMAGES", {"training": 2, "test": 1})
    monkeypatch.setattr(aj, "MAX_REPS", {"training": 2, "test": 2})
    rows = pd.DataFrame({"image_path": ["00002.jpg"] * 3, "partition": ["stim_test"] * 3, "session": [1, 1, 2], "dropped": [False] * 3})
    out, sessions = aj.organize(np.ones((3, 1, 2)), aj.kept_rows(rows, "test"), {2: ("test", 0)}, "test")
    assert out.shape == (1, 3, 1, 2) and sessions[0].tolist() == [1, 1, 2] and not np.isnan(out).any()


def test_sub08_time_labels_are_moved_200_ms_later():
    """sub-08's release labels sit 0.2 s early; everyone else's are right."""
    import numpy as np
    labels = np.round(-0.199 + np.arange(100) / 100, 3)
    assert np.allclose(aj.true_times("sub-08", labels)[[0, -1]], [0.001, 0.991])
    assert np.allclose(aj.true_times("sub-01", labels), labels)
