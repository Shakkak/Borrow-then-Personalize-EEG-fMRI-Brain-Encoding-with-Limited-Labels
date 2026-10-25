"""The reproduction of the THINGS-EEG2 paper's pipeline against sklearn and against a literal copy of its code."""
import numpy as np
import pytest


def test_precomputed_kernel_pca_matches_sklearn_poly():
    from sklearn.decomposition import KernelPCA
    from sklearn.preprocessing import StandardScaler

    from cross_subject_encoding.models.paper_linearizing import gram, kernel_pca, poly_kernel, standardize_stats

    rng = np.random.default_rng(0)
    x = (rng.gamma(2.0, 1.0, size=(140, 300)) * (rng.random(300) > 0.1)).astype(np.float16)  # ReLU-like, some dead units
    train = np.arange(140) < 120
    ref = StandardScaler().fit(x[train].astype(np.float32))
    kp = KernelPCA(n_components=40, kernel="poly", degree=4, random_state=1).fit(ref.transform(x[train].astype(np.float32)))
    want_tr, want_te = kp.transform(ref.transform(x[train].astype(np.float32))), kp.transform(ref.transform(x[~train].astype(np.float32)))
    mean, scale = standardize_stats(x, train, chunk=64)
    assert np.allclose(mean, ref.mean_, atol=1e-5) and np.allclose(scale, ref.scale_, rtol=1e-4)
    g = gram(x, mean, scale, train, "cpu", chunk=64)
    k = poly_kernel(g, x.shape[1])
    got_tr, got_te, _ = kernel_pca(k[train], k[~train], 40, 1)
    sign = np.sign((got_tr * want_tr).sum(0))
    assert np.allclose(got_tr * sign, want_tr, rtol=1e-3, atol=1e-3 * np.abs(want_tr).max())
    assert np.allclose(got_te * sign, want_te, rtol=1e-3, atol=1e-3 * np.abs(want_te).max())


def test_paper_correlation_matches_a_literal_copy_of_its_loop():
    from scipy.stats import pearsonr as corr
    from sklearn.utils import resample

    from cross_subject_encoding.eval.paper_correlation import paper_correlation

    rng = np.random.default_rng(2)
    bio = rng.normal(size=(30, 8, 3, 5)); synt = rng.normal(size=(30, 3, 5))
    got = paper_correlation({"all_layers": synt}, bio, n_iter=7, seed=20200220)
    # correlation.py, as released, with n_iter = 7
    np.random.seed(20200220)
    c = np.zeros((7, 3, 5)); lo = np.zeros((7, 3, 5)); up = np.zeros((7, 3, 5))
    avg_all = np.mean(bio, 1)
    for i in range(7):
        idx = resample(np.arange(0, bio.shape[1]), replace=False, n_samples=int(bio.shape[1] / 2))
        h1 = np.mean(np.delete(bio, idx, 1), 1); h2 = np.mean(bio[:, idx, :, :], 1)
        for t in range(5):
            for ch in range(3):
                c[i, ch, t] = corr(synt[:, ch, t], h1[:, ch, t])[0]
                lo[i, ch, t] = corr(h2[:, ch, t], h1[:, ch, t])[0]
                up[i, ch, t] = corr(avg_all[:, ch, t], h1[:, ch, t])[0]
    assert np.allclose(got["correlation"]["all_layers"], c.mean(0))
    assert np.allclose(got["noise_ceiling_low"], lo.mean(0)) and np.allclose(got["noise_ceiling_up"], up.mean(0))


def test_ols_and_alexnet_shapes():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from cross_subject_encoding.models.paper_linearizing import N_FEATURES, PaperAlexNet, ols_paper

    rng = np.random.default_rng(3)
    x = rng.normal(size=(200, 5)); w = rng.normal(size=(5, 4)); y = 2 + x @ w
    assert np.allclose(ols_paper(x, y, x[:10]), y[:10], atol=1e-3)
    out = PaperAlexNet(pretrained=False).eval()(torch.zeros(2, 3, 224, 224))
    assert sum(o.reshape(2, -1).shape[1] for o in out) == N_FEATURES == 205672


def test_paper_stats_matches_a_literal_copy_of_correlation_stats():
    from sklearn.utils import resample

    from cross_subject_encoding.eval.paper_correlation import paper_stats

    rng = np.random.default_rng(4)
    per_person = rng.normal(0.3, 0.1, size=(10, 3, 6))
    got = paper_stats(per_person, n_iter=500)
    np.random.seed(20200220)  # correlation_stats.py, as released, with n_iter = 500
    lo, hi = np.zeros(6), np.zeros(6)
    for t in range(6):
        dist = np.zeros(500)
        for i in range(500):
            dist[i] = np.mean(resample(np.mean(per_person[:, :, t], 1)))
        lo[t], hi[t] = np.percentile(dist, 2.5), np.percentile(dist, 97.5)
    assert np.allclose(got["ci_lower"], lo) and np.allclose(got["ci_upper"], hi)
    assert np.allclose(got["curve"], per_person.mean(1).mean(0))


def test_disk_backed_extraction_matches_memory_and_preserves_source_copies(tmp_path, monkeypatch):
    import torch
    from PIL import Image

    from cross_subject_encoding.models import paper_linearizing as paper

    class TinyFeatures(torch.nn.Module):
        def __init__(self, pretrained=True):
            super().__init__()

        def forward(self, values):
            return [values.mean(dim=(2, 3))]

    def transform():
        return lambda value: torch.from_numpy(np.asarray(value).copy()).permute(2, 0, 1).float() / 255

    monkeypatch.setattr(paper, "PaperAlexNet", TinyFeatures)
    monkeypatch.setattr(paper, "paper_transform", transform)
    monkeypatch.setattr(paper, "N_FEATURES", 3)
    original, source_copy = tmp_path / "original.jpg", tmp_path / "source_copy.jpg"
    pixels = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)
    Image.fromarray(pixels).save(original)
    source_copy.write_bytes(original.read_bytes())
    requested = [original, original, source_copy, original]
    unique, inverse = paper.unique_image_paths(requested)
    assert unique == [original, source_copy]
    assert inverse.tolist() == [0, 0, 1, 0]
    expected, _ = paper.extract(requested, "cpu", batch=2, workers=0, pretrained=False)
    actual, _ = paper.extract(unique, "cpu", batch=2, workers=0, pretrained=False,
                              output_path=tmp_path / "features.npy")
    assert isinstance(actual, np.memmap)
    assert np.array_equal(actual[inverse], expected)
    assert np.array_equal(np.load(tmp_path / "features.npy", mmap_mode="r"), actual)


def test_batched_kernel_pipeline_matches_existing_fit_and_repeated_queries(tmp_path):
    from cross_subject_encoding.models import paper_linearizing as paper

    rng = np.random.default_rng(41)
    features = rng.gamma(2.0, 1.0, (38, 31)).astype(np.float16)
    features[:, :2] = 0
    stored = np.lib.format.open_memmap(tmp_path / "features.npy", mode="w+", dtype=np.float16,
                                       shape=features.shape)
    stored[:] = features
    fit_rows = np.arange(30)
    requested_rows = np.array([35, 30, 35, 33, 31, 37])
    train_mask = np.arange(len(features)) < len(fit_rows)
    expected_mean, expected_scale = paper.standardize_stats(features, train_mask, chunk=7)
    mean, scale = paper.standardize_stats_batched(stored, fit_rows, row_chunk=4, feature_chunk=7)
    assert np.allclose(mean, expected_mean, atol=1e-12)
    assert np.allclose(scale, expected_scale, atol=1e-12)
    expected_kernel = paper.poly_kernel(paper.gram(features, mean, scale, train_mask, "cpu", chunk=7), 31)
    actual_kernel = paper.training_kernel_batched(stored, mean, scale, fit_rows, "cpu", tmp_path / "kernel.npy",
                                                  row_chunk=4, feature_chunk=7)
    assert isinstance(actual_kernel, np.memmap)
    assert np.array_equal(actual_kernel, actual_kernel.T)
    assert np.allclose(actual_kernel, expected_kernel[fit_rows], rtol=1e-6, atol=1e-6)
    batches = list(paper.polynomial_kernel_batches(stored, mean, scale, fit_rows, requested_rows, "cpu", 4, 7))
    assert all(len(block) <= 4 for _, block in batches)
    assert np.allclose(np.concatenate([block for _, block in batches]), expected_kernel[requested_rows],
                       rtol=1e-6, atol=1e-6)
    expected_train, expected_test, _ = paper.kernel_pca(expected_kernel[fit_rows], expected_kernel[requested_rows], 10)
    kernel_before_fit = actual_kernel.copy()
    actual_train, actual_test, _ = paper.kernel_pca_batched(actual_kernel, iter(batches), len(requested_rows), 10,
                                                          row_chunk=4)
    assert np.array_equal(actual_kernel, kernel_before_fit)
    signs = np.sign((actual_train * expected_train).sum(0))
    assert np.allclose(actual_train * signs, expected_train, rtol=1e-4, atol=1e-4)
    assert np.allclose(actual_test * signs, expected_test, rtol=1e-4, atol=1e-4)
    assert np.array_equal(actual_test[0], actual_test[2])


def test_batched_kernel_pca_rejects_missing_query_rows():
    from cross_subject_encoding.models import paper_linearizing as paper

    kernel = np.eye(5) + 1
    with pytest.raises(ValueError, match="cover all"):
        paper.kernel_pca_batched(kernel, iter([]), 2, 2, row_chunk=2)
