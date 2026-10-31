"""Whole-participant source folds shared by statistical and raw baselines."""
import numpy as np


def source_folds(groups, seed=261031):
    people = np.random.default_rng(seed).permutation(sorted(set(groups)))
    return [(np.flatnonzero(~np.isin(groups, block)), np.flatnonzero(np.isin(groups, block)))
            for block in np.array_split(people, 3)]


def group_brier(y, p, groups):
    values = []
    for group in sorted(set(groups)):
        ix = groups == group
        if set(y[ix]) != {0, 1}:
            raise ValueError('participant lacks one state')
        values.append(np.mean([np.mean((p[ix & (y == label)] - label) ** 2) for label in (0, 1)]))
    return float(np.mean(values))


def group_weights(groups, y):
    weights = np.empty(len(y))
    for group in set(groups):
        for label in (0, 1):
            ix = (groups == group) & (y == label)
            if not ix.any():
                raise ValueError('participant lacks one state')
            weights[ix] = 1 / ix.sum()
    return weights * len(y) / weights.sum()
