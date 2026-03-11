"""
Graph construction from PySPI output.

Converts SPI MxMxK tensors + node features into PyG Data objects.
Adjacency prior and sparsification happen inside the model forward pass
(weights w are learnable), so this module builds the *dense* edge data
and lets the model handle the rest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from .features import node_features


def load_spi_tensor(dataset_dir: Path, spi_names: list[str]) -> np.ndarray:
    """
    Load SPI matrices from spi_mpis.npz and stack into (M, M, K) tensor.

    Args:
        dataset_dir: Path containing spi_mpis.npz.
        spi_names: Ordered list of SPI names to include.

    Returns:
        (M, M, K) float array. NaN/inf values are replaced with 0.
    """
    npz_path = dataset_dir / "spi_mpis.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")

    with np.load(npz_path) as npz:
        mats = []
        for name in spi_names:
            if name not in npz:
                raise KeyError(f"SPI '{name}' not in {npz_path}")
            mat = np.asarray(npz[name], dtype=np.float64)
            mats.append(mat)

    tensor = np.stack(mats, axis=-1)  # (M, M, K)
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return tensor


def build_graph(
    spi_tensor: np.ndarray,
    mts: np.ndarray,
    label: int,
    *,
    gt_motif_edges: list[tuple[int, int]] | None = None,
) -> Data:
    """
    Build a PyG Data object from SPI tensor and raw MTS.

    The graph is initially fully connected (excluding self-loops).
    Sparsification happens in the model forward pass.

    Args:
        spi_tensor: (M, M, K) SPI feature tensor (already scaled).
        mts: (T, M) raw time series for node feature extraction.
        label: Integer class label.
        gt_motif_edges: Ground-truth motif edges for explanation evaluation.

    Returns:
        PyG Data with:
            x: (M, F_n) node features
            spi_tensor: (M, M, K) dense SPI tensor (for on-the-fly graph build)
            y: scalar label
            num_nodes: M
            gt_edge_mask: (M, M) bool if gt_motif_edges provided
    """
    M, _, K = spi_tensor.shape

    # Node features
    x = node_features(mts)  # (M, F_n)

    data = Data(
        x=torch.from_numpy(x),
        spi_tensor=torch.from_numpy(spi_tensor.astype(np.float32)),
        y=torch.tensor(label, dtype=torch.long),
        num_nodes=M,
    )

    # Ground-truth motif edge mask (for synthetic explanation eval)
    if gt_motif_edges is not None:
        mask = np.zeros((M, M), dtype=bool)
        for src, dst in gt_motif_edges:
            mask[src, dst] = True
        data.gt_edge_mask = torch.from_numpy(mask)

    return data


class SPIScaler:
    """
    Per-SPI-dimension robust scaling fitted on training data only.

    Stores median and IQR per SPI dimension (K values).
    Transforms (M, M, K) tensors element-wise per dimension.
    """

    def __init__(self):
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None
        self.is_fitted: bool = False

    def fit(self, tensors: list[np.ndarray]) -> "SPIScaler":
        """
        Fit on a list of (M, M, K) tensors.

        Pools all off-diagonal values per SPI dimension.
        """
        K = tensors[0].shape[2]
        all_vals = [[] for _ in range(K)]

        for t in tensors:
            M = t.shape[0]
            mask = ~np.eye(M, dtype=bool)
            for k in range(K):
                vals = t[:, :, k][mask]
                finite = vals[np.isfinite(vals)]
                all_vals[k].append(finite)

        self.median_ = np.zeros(K)
        self.iqr_ = np.ones(K)

        for k in range(K):
            pooled = np.concatenate(all_vals[k])
            if pooled.size > 0:
                q25, q50, q75 = np.percentile(pooled, [25, 50, 75])
                self.median_[k] = q50
                iqr = q75 - q25
                self.iqr_[k] = iqr if iqr > 1e-12 else 1.0

        self.is_fitted = True
        return self

    def transform(self, tensor: np.ndarray) -> np.ndarray:
        """Scale a single (M, M, K) tensor using fitted statistics."""
        if not self.is_fitted:
            raise RuntimeError("SPIScaler not fitted")
        return (tensor - self.median_) / self.iqr_

    def fit_transform(self, tensors: list[np.ndarray]) -> list[np.ndarray]:
        self.fit(tensors)
        return [self.transform(t) for t in tensors]


def filter_spi_dimensions(
    spi_names: list[str],
    tensors: list[np.ndarray],
    *,
    max_missing_rate: float = 0.05,
    min_variance: float = 1e-8,
) -> tuple[list[str], list[int]]:
    """
    Decide which SPI dimensions to keep based on training data.

    Drop if:
        - missing rate (NaN/inf) > max_missing_rate across all samples
        - variance < min_variance across all off-diagonal entries

    Returns:
        (retained_names, retained_indices)
    """
    K = len(spi_names)
    retained_names: list[str] = []
    retained_indices: list[int] = []

    for k in range(K):
        # Collect all off-diagonal values for this SPI
        all_vals = []
        n_total = 0
        n_missing = 0
        for t in tensors:
            M = t.shape[0]
            mask = ~np.eye(M, dtype=bool)
            vals = t[:, :, k][mask]
            n_total += vals.size
            n_missing += (~np.isfinite(vals)).sum()
            all_vals.append(vals[np.isfinite(vals)])

        # Missing rate check
        missing_rate = n_missing / max(n_total, 1)
        if missing_rate > max_missing_rate:
            print(f"[SPI-FILTER] Dropping '{spi_names[k]}': missing rate {missing_rate:.1%}")
            continue

        # Variance check
        pooled = np.concatenate(all_vals) if all_vals else np.array([])
        if pooled.size < 2:
            print(f"[SPI-FILTER] Dropping '{spi_names[k]}': insufficient data")
            continue
        var = pooled.var()
        if var < min_variance:
            print(f"[SPI-FILTER] Dropping '{spi_names[k]}': variance {var:.2e} < {min_variance:.2e}")
            continue

        retained_names.append(spi_names[k])
        retained_indices.append(k)

    print(f"[SPI-FILTER] Retained {len(retained_names)}/{K} SPI dimensions")
    return retained_names, retained_indices
