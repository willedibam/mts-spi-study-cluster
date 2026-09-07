"""Raw reference models for the discrete-map local-sensitivity pilot."""
from dataclasses import dataclass

import numpy as np


def energy_share(own, cross, observed_channels, population=None):
    """Optional dyad/node inclusion-ratio correction, not an unbiased q estimate.

    For actual Jacobian entries and uniform sensor subsets, off-diagonal and
    diagonal energy have different inclusion probabilities. Their corrected
    ratio is still biased; fitted observed-map coefficients need not equal the
    restricted full-state Jacobian, especially with hidden variables.
    """
    factor = 1.0 if population is None else (population - 1) / (observed_channels - 1)
    denominator = own + factor * cross
    return float(factor * cross / denominator) if denominator > 0 else float("nan")


@dataclass
class FittedMap:
    linear: np.ndarray
    nonlinear: np.ndarray
    intercept: np.ndarray

    def predict(self, x):
        return x @ self.linear.T + np.tanh(x) @ self.nonlinear.T + self.intercept

    def jacobian(self, x):
        return self.linear[None] + self.nonlinear[None] * (1 - np.tanh(x)**2)[:, None, :]

    def energies(self, x):
        derivative = 1 - np.tanh(x)**2
        first, second = derivative.mean(axis=0), np.square(derivative).mean(axis=0)
        energy = self.linear**2 + 2 * self.linear * self.nonlinear * first[None] + self.nonlinear**2 * second[None]
        # Clamp only roundoff-scale negative individual energies.
        energy = np.maximum(energy, 0)
        own = np.trace(energy)
        return float(own), float(energy.sum() - own)


def fit_map(raw, nonlinear=False, ridge_fraction=1e-3):
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 2 or len(raw) < 3 or raw.shape[1] < 2 or not np.isfinite(raw).all():
        raise ValueError("expected finite T by M observations")
    if ridge_fraction <= 0:
        raise ValueError("ridge fraction must be positive")
    x, y = raw[:-1], raw[1:]
    design = np.concatenate([x, np.tanh(x)], axis=1) if nonlinear else x
    xmean, ymean = design.mean(axis=0), y.mean(axis=0)
    centered = design - xmean
    gram = centered.T @ centered
    penalty = ridge_fraction * np.trace(gram) / len(gram)
    if penalty <= 0:
        raise ValueError("constant input")
    coefficients = np.linalg.solve(gram + penalty * np.eye(len(gram)), centered.T @ (y - ymean))
    m = raw.shape[1]
    return FittedMap(coefficients[:m].T, coefficients[m:].T if nonlinear else np.zeros((m, m)),
                     ymean - xmean @ coefficients)


def own_memory(raw):
    """Mean separate-channel AR(1) slope; contains no cross-channel products."""
    x, y = raw[:-1].copy(), raw[1:].copy()
    x -= x.mean(axis=0); y -= y.mean(axis=0)
    return float(np.mean(np.sum(x * y, axis=0) / np.sum(x * x, axis=0)))
