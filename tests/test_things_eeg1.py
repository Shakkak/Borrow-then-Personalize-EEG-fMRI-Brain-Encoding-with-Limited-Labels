"""THINGS-EEG1 in THINGS-EEG2's format, on a planted continuous recording (no MNE, no download).

A recording at 250 Hz with 64 named channels, stimuli every 100 ms as in the 10 Hz streams, and a planted
response per image: a bump at 110 ms on the occipital channels whose size is the image's own number. The
conversion must put every image in the shared training order, keep 17 channels and 100 samples, put the
bump at 110 ms, recover each image's size, and order the validation images by their number.
"""
import numpy as np
import pandas as pd
import pytest

from cross_subject_encoding.data import things_eeg1 as e1

CH64 = e1.CHANNELS + [f"X{i}" for i in range(64 - len(e1.CHANNELS))]
SF = 250.0


def planted(n_train=40, test_reps=3, seed=0):
    rng = np.random.default_rng(seed)
    train_keys = [f"c{c}/c{c}_{i:02d}s.jpg" for c in range(n_train // 4) for i in range(1, 5)]
    val_keys = [f"v{i}/v{i}_13s.jpg" for i in range(e1.N_TEST)]
    size = {k: rng.uniform(0.5, 2.0) for k in train_keys + val_keys}
    rows = []
    for k in rng.permutation(train_keys):  # each person sees the training images in their own order
        c, f = k.split("/")
        rows.append({"stim": "stimuli\\" + k.replace("/", "\\"), "objectnumber": int(c[1:]), "stimnumber": int(f.split("_")[1][:2]) - 1,
                     "isteststim": 0, "teststimnumber": -1})
    for r in range(test_reps):
        for i in rng.permutation(e1.N_TEST):
            rows.append({"stim": "stimuli\\" + val_keys[i].replace("/", "\\"), "objectnumber": -1, "stimnumber": -1, "isteststim": 1, "teststimnumber": int(i)})
    ev = pd.DataFrame(rows)
    step = int(0.1 * SF)
    onsets = int(1.0 * SF) + step * np.arange(len(ev))
    n = onsets[-1] + int(2.0 * SF)
    t = np.arange(-0.3, 0.9, 1 / SF)
    bump = np.exp(-((t - 0.11) / 0.02) ** 2)
    data = 1e-7 * rng.normal(size=(64, n))
    occ = [CH64.index(c) for c in ("O1", "Oz", "O2")]
    for o, k in zip(onsets, image_keys(ev)):
        a = int(round(-0.3 * SF))
        data[np.ix_(occ, o + a + np.arange(len(t)))] += 1e-6 * size[k] * bump
    return data, onsets, ev, size


def image_keys(ev):
    return e1.image_key(ev)


def test_epochs_put_each_image_at_110_ms_with_its_own_size():
    data, onsets, ev, size = planted()
    ep = e1.epoch(data[[CH64.index(c) for c in e1.CHANNELS]], onsets, SF)
    assert ep.shape == (len(ev), 17, 100)
    sizes = np.array([size[k] for k in e1.image_key(ev)])
    # at 10 Hz the mean response repeats every 100 ms (each image's bump overlaps its neighbours'), so the
    # check is where the response follows each image's own size
    r_t = [np.corrcoef(ep[:, 1, t], sizes)[0, 1] for t in range(100)]
    assert int(np.argmax(r_t)) - 20 == 11 and max(r_t) > 0.95, (int(np.argmax(r_t)), max(r_t))


def test_conversion_stores_whitened_epochs_in_the_shared_order():
    data, onsets, ev, size = planted()
    keys = e1.training_order(ev)
    p = e1.build_person(data, CH64, SF, onsets, ev, keys)
    assert p["train"].shape == (40, 1, 17, 100) and p["test"].shape == (200, 3, 17, 100)
    assert np.allclose(p["times"][[0, -1]], [-0.2, 0.79])
    ep = e1.epoch(data[[CH64.index(c) for c in e1.CHANNELS]], onsets, SF)
    ik = e1.image_key(ev)
    tr = {k: i for i, k in enumerate(ik) if ev.isteststim.iloc[i] != 1}
    train = ep[[tr[k] for k in keys]]
    w = e1.mvnn_whitener(train)
    assert np.allclose(p["train"][:, 0], np.einsum("ij,njt->nit", w, train), atol=1e-6)
    val = e1.validation_images(ev)
    first = [np.where((ik == val[i]) & (ev.isteststim == 1).to_numpy())[0][0] for i in range(e1.N_TEST)]
    assert np.allclose(p["test"][:, 0], np.einsum("ij,njt->nit", w, ep[first]), atol=1e-6)
    assert keys == sorted(keys, key=lambda k: (int(k.split("/")[0][1:]), k))


def test_unwhitened_microvolt_release_rewhitens_to_the_standard_one():
    data, onsets, ev, size = planted()
    keys = e1.training_order(ev)
    standard = e1.build_person(data, CH64, SF, onsets, ev, keys)
    raw = e1.build_person(data, CH64, SF, onsets, ev, keys, whiten=False, scale=1e3)
    assert raw["report"]["whitened"] is False and standard["report"]["whitened"] is True
    w = e1.mvnn_whitener(raw["train"][:, 0])  # the whitener is scale-free: microvolts give the same result
    assert np.allclose(np.einsum("ij,njt->nit", w, raw["train"][:, 0]), standard["train"][:, 0], atol=1e-4)
    assert np.allclose(np.einsum("ij,nrjt->nrit", w, raw["test"]), standard["test"], atol=1e-4)
    assert np.abs(raw["train"]).max() < 65504  # millivolts: safe in float16


def test_save_refuses_float16_overflow(tmp_path):
    person = {"train": np.full((2, 1, 17, 100), 1e5, np.float32), "test": None, "times": np.arange(100) / 100}
    with pytest.raises(ValueError, match="overflow"):
        e1.save_release(tmp_path, "sub-99", person)


def test_people_see_their_own_order_but_are_stored_in_the_shared_one():
    data_a, on_a, ev_a, size = planted(seed=0)
    data_b, on_b, ev_b, _ = planted(seed=0)
    ev_b = ev_b.copy()
    keys = e1.training_order(ev_a)
    assert keys == e1.training_order(ev_b)
    pa = e1.build_person(data_a, CH64, SF, on_a, ev_a, keys)
    assert pa["report"]["test_reps_min"] == 3


def test_missing_channel_extra_events_and_wrong_images():
    data, onsets, ev, _ = planted()
    keys = e1.training_order(ev)
    with pytest.raises(ValueError, match="channels missing"):
        e1.build_person(data, [c.replace("Oz", "OZ") for c in CH64], SF, onsets, ev, keys)
    p = e1.build_person(data, CH64, SF, np.r_[onsets, onsets[-1] + 25], ev, keys)  # a trailing extra event is dropped
    assert p["report"]["events_eeg"] == len(ev) + 1
    with pytest.raises(ValueError, match="differ from the shared"):
        e1.build_person(data, CH64, SF, onsets, ev, keys[:-1] + ["zz/zz_01s.jpg"])
    with pytest.raises(ValueError, match="edges"):
        e1.epoch(data, np.array([5]), SF)


def test_whitener_on_correlated_noise_evens_out_the_channels():
    rng = np.random.default_rng(1)
    mix = rng.normal(size=(17, 17))
    ep = np.einsum("ij,njt->nit", mix, rng.normal(size=(300, 17, 100))).astype(np.float32)
    w = e1.mvnn_whitener(ep)
    white = np.einsum("ij,njt->nit", w, ep)
    spread = lambda x: (lambda ev: ev.max() / ev.min())(np.linalg.eigvalsh(np.mean([np.cov(e) for e in x], 0)))
    before, after = spread(ep), spread(white)
    assert after < before / 10, (before, after)  # shrinkage over 100-sample epochs keeps it well above 1


def test_saved_release_reads_back_as_float32_in_eeg2_layout(tmp_path):
    data, onsets, ev, _ = planted()
    p = e1.build_person(data, CH64, SF, onsets, ev, e1.training_order(ev))
    paths = e1.save_release(tmp_path, "sub-02", p)
    d = np.load(paths[0], allow_pickle=True).item()
    assert set(d) == {"preprocessed_eeg_data", "ch_names", "times"} and d["preprocessed_eeg_data"].shape == (40, 1, 17, 100)
    assert list(d["ch_names"]) == e1.CHANNELS
