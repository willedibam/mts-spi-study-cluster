"""Dyad-correspondence interventions on normalized SPI edge clouds."""
import numpy as np


def dyad_rows(m):
    positions = np.full((m, m), -1, dtype=int)
    positions[~np.eye(m, dtype=bool)] = np.arange(m * (m - 1))
    upper = np.triu_indices(m, 1)
    return positions[upper], positions[upper[::-1]]


def permute_edge_columns(values, m, rng, *, shared=False):
    if values.shape[0] != m * (m - 1):
        raise ValueError("expected all ordered off-diagonal rows, without padding")
    upper, lower = dyad_rows(m)
    moved = values.copy()
    common = None
    for col in range(values.shape[1]):
        if common is None or not shared:
            common = rng.permutation(len(upper)), rng.integers(2, size=len(upper)).astype(bool)
        permutation, flip = common
        a, b = values[upper[permutation], col], values[lower[permutation], col]
        moved[upper, col] = np.where(flip, b, a)
        moved[lower, col] = np.where(flip, a, b)
    return moved


def canonical_dyads(values, m):
    upper, lower = dyad_rows(m)
    pairs = np.stack([np.minimum(values[upper], values[lower]),
                      np.maximum(values[upper], values[lower])], axis=-1)
    order = np.lexsort((pairs[:, :, 1], pairs[:, :, 0]), axis=0)
    return np.take_along_axis(pairs, order[:, :, None], axis=0)
