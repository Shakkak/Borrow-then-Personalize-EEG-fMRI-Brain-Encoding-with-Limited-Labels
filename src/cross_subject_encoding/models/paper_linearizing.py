"""The THINGS-EEG2 paper's linearizing encoder, reproduced from its released code.

Source: github.com/gifale95/eeg_encoding, read in full:
`03_dnn_feature_maps_extraction/extract_feature_maps_alexnet.py` and
`feature_maps_pca.py`, `04_synthesizing_eeg_data/linearizing_encoding_utils.py`
and `ols.py`. What it does, step by step:

1. torchvision AlexNet with ImageNet weights, eval mode; images resized to
   224 x 224 (no crop), ImageNet mean and std; the full outputs of
   maxpool1, maxpool2, ReLU3, ReLU4, maxpool5, ReLU6, ReLU7 and fc8,
   flattened and appended (205,672 values per image).
2. `StandardScaler` fitted on the training images, then
   `KernelPCA(kernel='poly', degree=4)` fitted on them (gamma = 1 / number
   of features, coef0 = 1, sklearn's defaults), keeping the first 1,000
   components.
3. Ordinary least squares with an intercept, one per output, solved in
   float32 through the normal equations (`OLS_pytorch`). Within a person
   on their training EEG averaged over its 4 repetitions; across people on
   the average of the other nine people's averaged training EEG.

Two things differ, both for memory and both checked: the activations are
held in float16 (the notebook measures what that does to the kernel), and
the polynomial kernel is computed in blocks and handed to sklearn's
KernelPCA as a precomputed kernel. That is the same kernel and the same
eigen-solver sklearn would use (tested against `KernelPCA(kernel='poly')`
in tests/test_paper_linearizing.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# the paper's names for torchvision AlexNet's modules, in order (extract_feature_maps_alexnet.py)
CONV_LAYERS = ["conv1", "ReLU1", "maxpool1", "conv2", "ReLU2", "maxpool2", "conv3", "ReLU3", "conv4", "ReLU4",
               "conv5", "ReLU5", "maxpool5"]
FC_LAYERS = ["Dropout6", "fc6", "ReLU6", "Dropout7", "fc7", "ReLU7", "fc8"]
PAPER_LAYERS = ["maxpool1", "maxpool2", "ReLU3", "ReLU4", "maxpool5", "ReLU6", "ReLU7", "fc8"]
LAYER_SIZES = {"maxpool1": 64 * 27 * 27, "maxpool2": 192 * 13 * 13, "ReLU3": 384 * 13 * 13, "ReLU4": 256 * 13 * 13,
               "maxpool5": 256 * 6 * 6, "ReLU6": 4096, "ReLU7": 4096, "fc8": 1000}
N_FEATURES = sum(LAYER_SIZES.values())  # 205,672
PAPER_SEED = 20200220  # the seed in every script of the paper's code
N_COMPONENTS = 1000


class PaperAlexNet(nn.Module):
    """The paper's AlexNet wrapper: the same modules, the same layers kept, in the same order."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        weights = models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.alexnet(weights=weights)  # `pretrained=True` in the paper's torchvision is these weights
        self.alex_feats, self.alex_classifier = net.features, net.classifier
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        out = []
        for name, layer in self.alex_feats._modules.items():
            x = layer(x)
            if CONV_LAYERS[int(name)] in PAPER_LAYERS:
                out.append(x)
        x = self.avgpool(x).view(x.size(0), -1)
        for name, layer in self.alex_classifier._modules.items():
            x = layer(x)
            if FC_LAYERS[int(name)] in PAPER_LAYERS:
                out.append(x)
        return out


def paper_transform():
    from torchvision import transforms as trn

    return trn.Compose([trn.Resize((224, 224)), trn.ToTensor(),
                        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class _Images(torch.utils.data.Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB"))


@torch.no_grad()
def extract(paths, device: str, batch: int = 64, workers: int = 2, pretrained: bool = True, keep_float32_rows: int = 0,
            progress=None, output_path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Appended feature maps, (n_images, 205672) float16, in the order of `paths`.

    The first `keep_float32_rows` rows are also returned in float32, to measure the effect of float16 storage.
    With output_path, feature maps are written to a disk-backed NPY array instead of a full RAM array.
    """
    torch.manual_seed(PAPER_SEED)
    model = PaperAlexNet(pretrained).to(device).eval()
    loader = torch.utils.data.DataLoader(_Images(paths, paper_transform()), batch_size=batch, num_workers=workers)
    if output_path is None:
        out = np.empty((len(paths), N_FEATURES), dtype=np.float16)
    else:
        out = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=(len(paths), N_FEATURES))
    exact = np.empty((min(keep_float32_rows, len(paths)), N_FEATURES), dtype=np.float32) if keep_float32_rows else None
    i = 0
    for b, x in enumerate(loader):
        feats = torch.cat([f.reshape(f.shape[0], -1) for f in model(x.to(device))], 1).float().cpu().numpy()
        out[i:i + len(feats)] = feats
        if exact is not None and i < len(exact):
            n = min(len(exact) - i, len(feats))
            exact[i:i + n] = feats[:n]
        i += len(feats)
        if progress and b % 20 == 0:
            progress(i, len(paths))
    assert i == len(paths)
    if isinstance(out, np.memmap):
        out.flush()
    return out, exact


def unique_image_paths(paths):
    """Stable path deduplication and a map back to requested order; distinct source copies stay distinct."""
    unique, positions, inverse = [], {}, []
    for path in paths:
        key = Path(path)
        if key not in positions:
            positions[key] = len(unique)
            unique.append(path)
        inverse.append(positions[key])
    return unique, np.asarray(inverse, dtype=np.intp)


def standardize_stats_batched(features, fit_rows, row_chunk: int = 256, feature_chunk: int = 4096):
    """Population moments over fit rows using bounded float64 blocks and parallel variance updates."""
    if row_chunk < 1 or feature_chunk < 1 or len(fit_rows) == 0:
        raise ValueError("Positive chunk sizes and nonempty fit rows are required")
    fit_rows = np.asarray(fit_rows, dtype=np.intp)
    mean, scale = np.empty(features.shape[1]), np.empty(features.shape[1])
    for start in range(0, features.shape[1], feature_chunk):
        columns = np.arange(start, min(start + feature_chunk, features.shape[1]))
        block_mean, moment = np.zeros(len(columns)), np.zeros(len(columns))
        count = 0
        for row_start in range(0, len(fit_rows), row_chunk):
            rows = fit_rows[row_start:row_start + row_chunk]
            values = features[np.ix_(rows, columns)].astype(np.float64)
            current_mean = values.mean(0)
            values -= current_mean
            current_moment = np.einsum("ij,ij->j", values, values)
            total = count + len(rows)
            difference = current_mean - block_mean
            moment += current_moment + difference ** 2 * count * len(rows) / total
            block_mean += difference * len(rows) / total
            count = total
        deviation = np.sqrt(moment / count)
        mean[columns], scale[columns] = block_mean, np.where(deviation == 0, 1.0, deviation)
    return mean, scale


def polynomial_kernel_batches(features, mean, scale, fit_rows, query_rows, device: str,
                              row_chunk: int = 128, feature_chunk: int = 4096):
    """Yield (row offset, float64 kernel block), with only a query tile on the accelerator."""
    if row_chunk < 1 or feature_chunk < 1 or len(fit_rows) == 0:
        raise ValueError("Positive chunk sizes and nonempty fit rows are required")
    fit_rows, query_rows = np.asarray(fit_rows, dtype=np.intp), np.asarray(query_rows, dtype=np.intp)
    for row_start in range(0, len(query_rows), row_chunk):
        rows = query_rows[row_start:row_start + row_chunk]
        accumulator = torch.zeros((len(rows), len(fit_rows)), dtype=torch.float64, device=device)
        for start in range(0, features.shape[1], feature_chunk):
            columns = np.arange(start, min(start + feature_chunk, features.shape[1]))
            offset = torch.as_tensor(mean[columns], dtype=torch.float32, device=device)
            divisor = torch.as_tensor(scale[columns], dtype=torch.float32, device=device)
            train_block = torch.as_tensor(features[np.ix_(fit_rows, columns)].astype(np.float32), device=device)
            query_block = torch.as_tensor(features[np.ix_(rows, columns)].astype(np.float32), device=device)
            train_block.sub_(offset).div_(divisor)
            query_block.sub_(offset).div_(divisor)
            accumulator += (query_block @ train_block.T).double()
            del train_block, query_block
        block = accumulator.cpu().numpy()
        block /= features.shape[1]
        block += 1.0
        np.power(block, 4, out=block)
        yield row_start, block
        del block, accumulator


def training_kernel_batched(features, mean, scale, fit_rows, device: str, output_path,
                            row_chunk: int = 128, feature_chunk: int = 4096, progress=None):
    """Write the dense fit kernel to disk without allocating a full GPU or host Gram matrix."""
    kernel = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float64,
                                      shape=(len(fit_rows), len(fit_rows)))
    for start, block in polynomial_kernel_batches(features, mean, scale, fit_rows, fit_rows, device,
                                                  row_chunk, feature_chunk):
        kernel[start:start + len(block)] = block
        if progress:
            progress(start + len(block), len(fit_rows))
    for start in range(0, len(fit_rows), row_chunk):
        for column_start in range(start, len(fit_rows), row_chunk):
            rows = slice(start, start + row_chunk)
            columns = slice(column_start, column_start + row_chunk)
            symmetric = (kernel[rows, columns] + kernel[columns, rows].T) / 2
            kernel[rows, columns] = symmetric
            kernel[columns, rows] = symmetric.T
    kernel.flush()
    return kernel


def kernel_pca_batched(k_train, test_batches, n_test: int, n_components: int = N_COMPONENTS,
                       seed: int = PAPER_SEED, row_chunk: int = 128):
    """Keep the original dense KPCA fit, but transform training and query kernels a tile at a time."""
    from sklearn.decomposition import KernelPCA

    if row_chunk < 1:
        raise ValueError("A positive row chunk is required")
    model = KernelPCA(n_components=n_components, kernel="precomputed", random_state=seed).fit(k_train)
    train = np.empty((len(k_train), n_components))
    test = np.empty((n_test, n_components))
    for start in range(0, len(k_train), row_chunk):
        train[start:start + row_chunk] = model.transform(k_train[start:start + row_chunk])
    count = 0
    for start, block in test_batches:
        if start != count or count + len(block) > n_test:
            raise ValueError("Query kernel batches must cover the requested rows in order")
        test[start:start + len(block)] = model.transform(block)
        count += len(block)
    if count != n_test:
        raise ValueError("Query kernel batches did not cover all requested rows")
    return train, test, model


def standardize_stats(x: np.ndarray, train: np.ndarray, chunk: int = 16384) -> tuple[np.ndarray, np.ndarray]:
    """StandardScaler's mean and scale over the training rows (population std; a zero std becomes 1, as in sklearn)."""
    mean = np.empty(x.shape[1]); scale = np.empty(x.shape[1])
    rows = np.where(train)[0]
    for a in range(0, x.shape[1], chunk):
        block = x[rows, a:a + chunk].astype(np.float64)
        mean[a:a + chunk] = block.mean(0)
        sd = block.std(0)
        scale[a:a + chunk] = np.where(sd == 0, 1.0, sd)
    return mean, scale


def gram(x: np.ndarray, mean: np.ndarray, scale: np.ndarray, train: np.ndarray, device: str,
         chunk: int = 8192) -> np.ndarray:
    """Z_all @ Z_train.T for the standardized features Z, (n_all, n_train) float64.

    Accumulated over blocks of features: each block's product in float32, summed in float64.
    """
    tr = np.where(train)[0]
    acc = torch.zeros((len(x), len(tr)), dtype=torch.float64, device=device)
    for a in range(0, x.shape[1], chunk):
        z = (torch.from_numpy(x[:, a:a + chunk].astype(np.float32)).to(device)
             - torch.from_numpy(mean[a:a + chunk]).float().to(device)) / torch.from_numpy(scale[a:a + chunk]).float().to(device)
        acc += (z @ z[tr].T).double()
        del z
    return acc.cpu().numpy()


def poly_kernel(g: np.ndarray, n_features: int, degree: int = 4, coef0: float = 1.0) -> np.ndarray:
    """sklearn's polynomial kernel from a Gram matrix: (gamma <x, y> + coef0) ** degree, gamma = 1 / n_features."""
    return (g / n_features + coef0) ** degree


def kernel_pca(k_train: np.ndarray, k_test: np.ndarray, n_components: int = N_COMPONENTS, seed: int = PAPER_SEED):
    """sklearn KernelPCA on a precomputed kernel: the same centring and eigen-solver as kernel='poly' would use."""
    from sklearn.decomposition import KernelPCA

    kp = KernelPCA(n_components=n_components, kernel="precomputed", random_state=seed).fit(k_train)
    return kp.transform(k_train), kp.transform(k_test), kp  # fit, then transform, as feature_maps_pca.py does


def ols_paper(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, device: str = "cpu") -> np.ndarray:
    """The paper's OLS_pytorch: intercept column, float32 normal equations, solved for every output at once."""
    ones = lambda a: np.concatenate([np.ones((len(a), 1)), a], 1)
    xt = torch.from_numpy(ones(x_train)).float().to(device)
    y = torch.from_numpy(np.asarray(y_train)).float().to(device)
    beta = torch.linalg.solve(xt.T @ xt, xt.T @ y)
    return (torch.from_numpy(ones(x_test)).float().to(device) @ beta).cpu().numpy()
