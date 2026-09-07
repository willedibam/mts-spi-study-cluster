"""Bounded marginal/readout controls for the exploratory representation screen."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from src.representation_screen import classifier, fit_view

QUANTILES = np.arange(1, 20) / 20
RICH_NAMES = ("mean", "std", "skewness", "kurtosis", *(f"q{int(q * 100):02d}" for q in QUANTILES))


def rich_marginals(mpis, order):
    """23 descriptors/SPI; population moments, Pearson (not excess) kurtosis.

    Entirely finite ordered off-diagonal vectors are required. Constant vectors
    retain means/scales/quantiles but have undefined standardized moments.
    Rescaling before moment calculation avoids overflow without changing shape.
    """
    m = len(mpis[order[0]])
    mask = ~np.eye(m, dtype=bool)
    values = np.asarray([mpis[name][mask] for name in order], dtype=float)
    result = np.full((len(order), len(RICH_NAMES)), np.nan)
    valid = np.isfinite(values).all(axis=1)
    edges = values[valid]
    if len(edges):
        block = np.full((len(edges), len(RICH_NAMES)), np.nan)
        block[:, 0:2] = np.stack([edges.mean(axis=1), edges.std(axis=1)], axis=1)
        block[:, 4:] = np.quantile(edges, QUANTILES, axis=1).T
        scale = np.max(np.abs(edges), axis=1)
        scaled = edges / np.where(scale > 0, scale, 1)[:, None]
        centered = scaled - scaled.mean(axis=1, keepdims=True)
        std = centered.std(axis=1)
        varying = std > 0
        unit = centered[varying] / std[varying, None]
        block[varying, 2] = np.mean(unit**3, axis=1)
        block[varying, 3] = np.mean(unit**4, axis=1)
        result[valid] = block
    return result.ravel()


def fit_head(scores, labels, c, head, config):
    if head == "logistic":
        return classifier(scores, labels, c, config)
    if head == "rbf":
        # Fixed scale rule; only C is tuned, with the same five-value grid.
        return SVC(C=c, kernel="rbf", gamma="scale", tol=config["tolerance"],
                   cache_size=512).fit(scores, labels)
    raise ValueError(head)


def select_head(bank, view, train, labels, preprocessing, config, seed, head):
    candidates = sorted(config["C_grid"])
    scores = {c: [] for c in candidates}
    folds = []
    for fit, val in StratifiedKFold(config["inner_folds"], shuffle=True, random_state=seed).split(train, labels[train]):
        transform, x = fit_view(bank, view, train[fit], preprocessing)
        y = transform.transform(bank, train[val])
        for c in candidates:
            model = fit_head(x, labels[train[fit]], c, head, config)
            scores[c].append(float(balanced_accuracy_score(labels[train[val]], model.predict(y))))
        folds.append({"fit": train[fit].tolist(), "validation": train[val].tolist()})
    means = {c: np.mean(s) for c, s in scores.items()}
    best = max(means.values())
    chosen = next(c for c in candidates if means[c] >= best - 1e-12)
    return chosen, {"scores": scores, "folds": folds}
