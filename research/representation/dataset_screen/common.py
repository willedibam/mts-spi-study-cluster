"""Grouped multiclass scoring shared by all discovery comparators."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def folds(y, groups):
    unique = np.unique(groups)
    labels = [np.unique(y[groups == g]) for g in unique]
    if all(len(v) == 1 for v in labels):
        # Configuration labels are constant within each structural test state.
        # Stratify whole states, never repetitions of a state.
        cv = StratifiedKFold(3, shuffle=True, random_state=261031)
        result = [(np.flatnonzero(np.isin(groups, unique[a])), np.flatnonzero(np.isin(groups, unique[b])))
                  for a,b in cv.split(unique, np.array(labels).reshape(-1))]
    else:
        result = list(StratifiedGroupKFold(3, shuffle=True, random_state=261031).split(y, y, groups))
    classes = set(y)
    for train, valid in result:
        if set(groups[train]) & set(groups[valid]) or set(y[train]) != classes or set(y[valid]) != classes:
            raise ValueError('source folds cannot support all classes without group leakage')
    return result


def weights(y, groups):
    result = np.zeros(len(y), dtype=float)
    for label in np.unique(y):
        people = np.unique(groups[y == label])
        for group in people:
            ix = (y == label) & (groups == group)
            result[ix] = 1 / (len(people) * ix.sum())
    return result * len(y) / result.sum()


def probabilities(values):
    p = np.clip(np.asarray(values, dtype=float), 0, 1)
    if p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError('invalid class scores')
    total = p.sum(1, keepdims=True)
    return np.divide(p, total, out=np.full_like(p, 1 / p.shape[1]), where=total > 0)


def metrics(y, p, groups):
    y, p = np.asarray(y), np.asarray(p)
    if p.shape[0] != len(y) or not np.isfinite(p).all() or not np.allclose(p.sum(1), 1):
        raise ValueError('misaligned/nonfinite probabilities')
    w = weights(y, groups)
    onehot = np.eye(p.shape[1])[y]
    return dict(brier=float(np.average(.5 * np.sum((p - onehot) ** 2, axis=1), weights=w)),
                balanced_accuracy=float(np.average(np.argmax(p, axis=1) == y, weights=w)),
                log_loss=float(np.average(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)), weights=w)))
