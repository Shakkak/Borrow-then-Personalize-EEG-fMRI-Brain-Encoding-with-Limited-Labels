import numpy as np

from cross_subject_encoding.features import streamed
from cross_subject_encoding.models import paper_linearizing as pl


def test_streamed_kpca_matches_the_in_memory_pipeline():
    g = np.random.default_rng(0)
    x = (g.normal(size=(300, 50)) * g.uniform(0.5, 3, 50)).astype(np.float16)
    fit = np.sort(g.choice(300, 200, replace=False))
    train = np.zeros(300, bool)
    train[fit] = True
    mean, scale = pl.standardize_stats(x, train)
    k = pl.poly_kernel(pl.gram(x, mean, scale, train, "cpu"), x.shape[1])
    ref_train, ref_rest, _ = pl.kernel_pca(k[train], k[~train], 20)
    got, info = streamed.streamed_kpca(lambda rows: x[rows], 300, fit, "cpu", n_components=20, chunk=64)
    # the streamed version stores standardized features in float16, so allow for that rounding; signs of components
    # are fixed by the same KernelPCA fit on the same kernel, so compare directly
    assert info == {"raw_features": 50, "fit_images": 200, "other_images": 100}
    scale_ref = np.abs(ref_train).max()
    assert np.abs(got[train] - ref_train).max() < 2e-2 * scale_ref
    assert np.abs(got[~train] - ref_rest).max() < 2e-2 * scale_ref
