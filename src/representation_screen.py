"""Training-only transforms and grouped evaluation for the p90 descriptor screen."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class BlockTransform:
    name: str
    keep: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    clip_limit: np.ndarray | None = None

    def transform(self, values: np.ndarray) -> np.ndarray:
        selected = np.asarray(values[:, self.keep], dtype=np.float64)
        filled = np.where(np.isfinite(selected), selected, self.median)
        centered = filled - self.mean
        if self.clip_limit is not None:
            centered = np.clip(centered, -self.clip_limit, self.clip_limit)
        return centered / self.scale


def fit_block(name: str, values: np.ndarray, config: dict) -> BlockTransform:
    values = np.asarray(values, dtype=np.float64)
    keep = np.flatnonzero(np.mean(np.isfinite(values), axis=0) >= config["minimum_valid_fraction"])
    selected = values[:, keep]
    median = np.nanmedian(selected, axis=0)
    filled = np.where(np.isfinite(selected), selected, median)
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    varying = np.isfinite(std) & (std > config["variance_threshold"])
    keep, median, mean, std = keep[varying], median[varying], mean[varying], std[varying]
    standardize = name in ("m", "g", "u", "validity") or config["z_scaling"] == "standard"
    scale = std if standardize else np.ones(len(keep))
    clip_sd = config.get("clip_standard_deviations")
    if clip_sd is not None and clip_sd <= 0:
        raise ValueError("clipping threshold must be positive")
    clip_limit = None if clip_sd is None else clip_sd * std
    if len(keep):
        # Unit total training variance per block, including centre-only z.
        if clip_limit is None:
            balance = np.sqrt(np.sum((std / scale) ** 2))
        else:
            centered = filled[:, varying] - mean
            clipped = np.clip(centered, -clip_limit, clip_limit) / scale
            balance = np.sqrt(np.sum(np.var(clipped, axis=0)))
        scale = scale * balance
    return BlockTransform(name, keep, median, mean, scale, clip_limit)


@dataclass
class ViewTransform:
    blocks: list[BlockTransform]
    pca: PCA | None

    def transform(self, bank: dict, indices: np.ndarray) -> np.ndarray:
        output = []
        for start in range(0, len(indices), 128):
            rows = indices[start:start + 128]
            parts = [b.transform(bank[b.name][rows]) for b in self.blocks]
            matrix = np.concatenate(parts, axis=1)
            if not matrix.shape[1]:
                output.append(np.zeros((len(rows), 1)))
            elif self.pca is None:
                output.append(matrix)
            else:
                output.append(self.pca.transform(matrix))
        return np.concatenate(output)


def fit_view(bank: dict, view: str, train: np.ndarray, config: dict) -> tuple[ViewTransform, np.ndarray]:
    blocks = [fit_block(name, bank[name][train], config) for name in view.split("+")]
    matrix = np.concatenate([b.transform(bank[b.name][train]) for b in blocks], axis=1)
    if matrix.shape[1] == 0:
        return ViewTransform(blocks, None), np.zeros((len(train), 1))
    dimensions = min(config["pca_dimensions"], len(train) - 1, matrix.shape[1])
    pca = PCA(n_components=dimensions, svd_solver=config["pca_solver"],
              random_state=config["pca_random_state"])
    scores = pca.fit_transform(matrix)
    return ViewTransform(blocks, pca), scores


def classifier(scores: np.ndarray, labels: np.ndarray, c: float, config: dict):
    if np.all(np.std(scores, axis=0) == 0):
        return DummyClassifier(strategy="prior").fit(scores, labels)
    return LogisticRegression(C=c, solver="lbfgs", max_iter=config["max_iter"],
                              tol=config["tolerance"]).fit(scores, labels)


def select_c(bank: dict, view: str, train: np.ndarray, labels: np.ndarray,
             preprocessing: dict, config: dict, seed: int) -> tuple[float, dict]:
    candidates = sorted(config["C_grid"])
    scores = {c: [] for c in candidates}
    cv = StratifiedKFold(n_splits=config["inner_folds"], shuffle=True, random_state=seed)
    for fit, validation in cv.split(train, labels[train]):
        transformer, fit_scores = fit_view(bank, view, train[fit], preprocessing)
        validation_scores = transformer.transform(bank, train[validation])
        for c in candidates:
            model = classifier(fit_scores, labels[train[fit]], c, config)
            scores[c].append(float(balanced_accuracy_score(labels[train[validation]], model.predict(validation_scores))))
    average = {c: float(np.mean(values)) for c, values in scores.items()}
    best = max(average.values())
    chosen = next(c for c in candidates if average[c] >= best - 1e-12)
    return chosen, {str(c): {"fold_scores": scores[c], "mean": average[c]} for c in candidates}


def training_subsets(labels: np.ndarray, pool: np.ndarray, budgets: list[int], seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = {label: rng.permutation(pool[labels[pool] == label]) for label in np.unique(labels[pool])}
    if any(len(rows) < max(budgets) for rows in shuffled.values()):
        raise ValueError("insufficient training examples")
    return {n: np.concatenate([rows[:n] for rows in shuffled.values()]) for n in budgets}


def evaluation_cells(m: np.ndarray, t: np.ndarray, source: dict) -> dict[str, np.ndarray]:
    different_m, different_t = m != source["M"], t != source["T"]
    return {"same_cell": ~different_m & ~different_t,
            "M_only_changed": different_m & ~different_t,
            "T_only_changed": ~different_m & different_t,
            "both_M_and_T_changed": different_m & different_t,
            "any_changed": different_m | different_t}


def bootstrap_group_means(values: np.ndarray, group_labels: np.ndarray, repetitions: int, seed: int) -> np.ndarray:
    """Values end with group axis. Resample paired groups within each class.

    Call once with stacked representations/differences to reuse identical draws.
    Each group already averages its cells and fitted training subsets. These
    intervals are conditional on fitted models, not independent training corpora.
    """
    rng = np.random.default_rng(seed)
    result = np.zeros(values.shape[:-1] + (repetitions,))
    classes = np.unique(group_labels)
    for label in classes:
        members = np.flatnonzero(group_labels == label)
        indices = rng.choice(members, size=(repetitions, len(members)), replace=True)
        result += values[..., indices].mean(axis=-1) / len(classes)
    return result
