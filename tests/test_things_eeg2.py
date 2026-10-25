"""THINGS-EEG2 helpers for the 09g dataset: layouts, float32 copies, session labels."""
import numpy as np
import pytest
from sklearn.utils import shuffle

from cross_subject_encoding.data import things_eeg2 as te


def release_dict(data):
    return {"preprocessed_eeg_data": data, "ch_names": [f"c{i}" for i in range(data.shape[2])], "times": np.arange(data.shape[3]) / 100}


def test_subject_file_finds_release_and_flat_layouts(tmp_path):
    (tmp_path / "a" / "sub-01").mkdir(parents=True)
    np.save(tmp_path / "a" / "sub-01" / "preprocessed_eeg_test.npy", release_dict(np.zeros((2, 3, 4, 5))), allow_pickle=True)
    np.save(tmp_path / te.FLAT.format(subject="sub-02", part="test"), release_dict(np.zeros((2, 3, 4, 5))), allow_pickle=True)
    assert te.subject_file(tmp_path, "sub-01", "test").parent.name == "sub-01"
    assert te.subject_file(tmp_path, "sub-02", "test").name == "things-eeg2_sub-02_preprocessed_eeg_test.npy"
    assert te.subject_file(tmp_path, "sub-03", "test") is None
    assert te.attached(tmp_path) and not te.attached(tmp_path / "missing")


def test_float32_copy_loads_exactly_as_the_original(tmp_path):
    x = np.random.default_rng(0).standard_normal((3, 4, 17, 100))
    np.save(tmp_path / "src.npy", release_dict(x), allow_pickle=True)
    info = te.write_float32(tmp_path / "src.npy", tmp_path / "dst.npy")
    a, b = te.load_part(tmp_path / "src.npy"), te.load_part(tmp_path / "dst.npy")
    assert np.array_equal(a["data"], b["data"]) and a["ch_names"] == b["ch_names"] and np.array_equal(a["times"], b["times"])
    assert info["shape"] == [3, 4, 17, 100] and info["max_cast_error"] < 1e-6


def test_session_labels_match_the_authors_merge_and_shuffle():
    """Mimic save_prepr: test sessions appended along the repetition axis (20 each), training repetitions in
    session order per image (2 + 2), then one shuffle of the repetition axis with the authors' seed."""
    test_sessions = [np.full((5, 20, 1, 1), s, float) for s in range(4)]
    merged = np.concatenate(test_sessions, 1)
    merged = merged[:, shuffle(np.arange(80), random_state=te.SHUFFLE_SEED)]
    np.testing.assert_array_equal(merged[0, :, 0, 0], te.session_labels("test"))
    train = np.concatenate([np.zeros((5, 2, 1, 1)), np.ones((5, 2, 1, 1))], 1)
    train = train[:, shuffle(np.arange(4), random_state=te.SHUFFLE_SEED)]
    np.testing.assert_array_equal(train[0, :, 0, 0], te.session_labels("training"))
    assert np.bincount(te.session_labels("test")).tolist() == [20] * 4


def test_session_agreement_detects_shared_session_noise_and_not_random_labels():
    rng = np.random.default_rng(1)
    labels = te.session_labels("test")
    signal = rng.standard_normal((60, 1, 30))
    session_noise = rng.standard_normal((4, 60, 30))
    data = signal + 0.8 * session_noise[labels].transpose(1, 0, 2) + rng.standard_normal((60, 80, 30))
    corr = te.rep_correlations(data)
    good = te.session_agreement(corr, labels, n_perm=200)
    bad = te.session_agreement(corr, rng.permutation(labels), n_perm=200)
    assert good["difference"] > 0.05 and good["p"] < 0.01 and good["rank"] == 1
    assert bad["p"] > 0.01


def test_export_folder_refuses_subfolders(tmp_path):
    from cross_subject_encoding.utils.kaggle_export import export_folder
    (tmp_path / "nested").mkdir()
    with pytest.raises(ValueError, match="subfolders"):
        export_folder("x", "XSUB test outputs", tmp_path)


def test_dataset_titles_outside_kaggle_limits_fail_before_upload(tmp_path):
    from cross_subject_encoding.utils.kaggle_export import check_title, export_folder
    check_title("XSUB Alljoined in THINGS-EEG2 layout")
    for bad in ("short", "x" * 51):
        with pytest.raises(ValueError, match="6-50"):
            check_title(bad)
    with pytest.raises(ValueError, match="6-50"):
        export_folder("x", "XSUB Alljoined-1.6M in THINGS-EEG2 layout (private, non-commercial)", tmp_path)
