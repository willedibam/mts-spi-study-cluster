"""HCP source grouping and participant metrics, independent of neural libraries."""
import numpy as np


def _ids(values, n):
    values = np.asarray(values, dtype=str)
    if values.shape != (n,) or any(v.strip().lower() in {'', 'none', 'nan'} for v in values):
        raise ValueError('Each record requires a nonmissing identifier')
    return values


def participant_state_weights(y, participants):
    """Mean-one weights: equal total weight per participant, then per class."""
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected a nonempty binary target vector with both classes')
    participants = _ids(participants, len(y))
    weights = np.zeros(len(y), dtype=float)
    for person in np.unique(participants):
        for label in (0, 1):
            mask = (participants == person) & (y == label)
            if not mask.any():
                raise ValueError(f'Participant {person} lacks class {label}')
            weights[mask] = 1 / mask.sum()
    return weights * (len(y) / weights.sum())


def participant_balanced_brier(y, probability, participants):
    y, probability = np.asarray(y), np.asarray(probability)
    if (probability.shape != y.shape or not np.isfinite(probability).all()
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError('Expected aligned finite probabilities in [0,1]')
    weights = participant_state_weights(y, participants)
    return float(np.average((probability - y)**2, weights=weights))


def family_folds(participants, families, n_splits, seed):
    """Deterministic whole-family folds, independent of labels and waveforms."""
    participants = _ids(participants, len(participants))
    families = _ids(families, len(participants))
    for person in np.unique(participants):
        if len(np.unique(families[participants == person])) != 1:
            raise ValueError(f'Participant {person} appears in multiple families')
    unique = np.unique(families)
    if not 2 <= n_splits <= len(unique):
        raise ValueError('Need at least one family per validation fold and two folds')
    shuffled = np.random.default_rng(seed).permutation(unique)
    folds = []
    for held_families in np.array_split(shuffled, n_splits):
        held = np.isin(families, held_families)
        folds.append((np.flatnonzero(~held), np.flatnonzero(held)))
    return folds


