"""Participant-level HCP outcomes with paired, whole-family uncertainty."""
import numpy as np
from sklearn.metrics import roc_auc_score

from src.hcp_grouping import _ids, participant_state_weights


METRICS = ('balanced_accuracy', 'auroc', 'balanced_brier')


def participant_scores(y, probabilities, participants, families):
    """All columns are predictions on the same ordered target observations."""
    y = np.asarray(y)
    participant_state_weights(y, participants)
    people, groups = _ids(participants, len(y)), _ids(families, len(y))
    probability = np.asarray(probabilities, dtype=float)
    if probability.ndim != 2 or probability.shape[0] != len(y):
        raise ValueError('Expected observations by models probability matrix')
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Probabilities must be finite and in [0, 1]')
    unique = np.unique(people)
    values = np.empty((len(unique), probability.shape[1], len(METRICS)))
    family = []
    for i, person in enumerate(unique):
        selected = people == person
        identifiers = np.unique(groups[selected])
        if len(identifiers) != 1:
            raise ValueError('A participant cannot belong to multiple families')
        family.append(identifiers[0])
        target, p = y[selected], probability[selected]
        predicted = p >= .5
        values[i, :, 0] = .5*((~predicted[target == 0]).mean(0) + predicted[target == 1].mean(0))
        values[i, :, 1] = [roc_auc_score(target, p[:, j]) for j in range(p.shape[1])]
        values[i, :, 2] = .5*((p[target == 0]**2).mean(0) + ((p[target == 1]-1)**2).mean(0))
    return unique, np.array(family), values


def summarize(y, probabilities, participants, families, model_names, *, seed, replicates=2000):
    """Paired family-cluster percentile intervals; family IDs stay out of output."""
    names = list(model_names)
    people, family, values = participant_scores(y, probabilities, participants, families)
    if len(names) != values.shape[1] or len(set(names)) != len(names) or not names:
        raise ValueError('Expected a unique name for every prediction column')
    groups, inverse = np.unique(family, return_inverse=True)
    if len(groups) < 2 or replicates < 100:
        raise ValueError('Uncertainty requires at least two families and 100 replicates')
    # Resample entire families together for every method and metric. Larger
    # families retain all their participants; the estimand is a participant mean.
    rng = np.random.default_rng(seed)
    sampled = rng.integers(len(groups), size=(replicates, len(groups)))
    counts = np.array([np.bincount(row, minlength=len(groups)) for row in sampled])
    weights = counts[:, inverse]
    bootstrap = np.einsum('ri,ijk->rjk', weights, values)/weights.sum(1)[:, None, None]
    estimate = values.mean(0)
    lower, upper = np.quantile(bootstrap, [.025, .975], axis=0)
    models = {name:{metric:dict(mean=float(estimate[j,k]), lower=float(lower[j,k]), upper=float(upper[j,k]))
                         for k, metric in enumerate(METRICS)} for j,name in enumerate(names)}
    pairs = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            interval = np.quantile(bootstrap[:, i]-bootstrap[:, j], [.025, .975], axis=0)
            pairs.append(dict(first=names[i], second=names[j], difference='first minus second',
                              metrics={metric:dict(mean=float(estimate[i,k]-estimate[j,k]),
                                                   lower=float(interval[0,k]), upper=float(interval[1,k]))
                                       for k,metric in enumerate(METRICS)}))
    return dict(observations=len(y), participants=len(people), families=len(groups),
                models=models, paired_differences=pairs,
                uncertainty=dict(method='paired whole-family percentile bootstrap', replicates=replicates,
                                 seed=seed, confidence=.95, estimand='mean participant score',
                                 limitation='Conditional on the fitted models and source cohort; few target families yield imprecise intervals; not simultaneous multiple-comparison intervals.'))


def evaluate_with_schedule_slice(y, probabilities, participants, families, blocks, model_names,
                                 *, discordant_blocks, seed, replicates=2000):
    """Reuse predictions for the primary comparison and prespecified schedule slice."""
    y, participants, families, blocks = map(np.asarray, (y, participants, families, blocks))
    if blocks.shape != y.shape:
        raise ValueError('Expected one block position per observation')
    probabilities = np.asarray(probabilities)
    main = summarize(y, probabilities, participants, families, model_names, seed=seed, replicates=replicates)
    position_mask = np.isin(blocks, discordant_blocks)
    eligible = [p for p in np.unique(participants) if set(y[position_mask & (participants == p)]) == {0, 1}]
    keep = position_mask & np.isin(participants, eligible)
    counts = dict(selected_positions=list(discordant_blocks),
                  observations_at_positions=int(position_mask.sum()), retained_observations=int(keep.sum()),
                  excluded_observations_for_class_coverage=int((position_mask & ~keep).sum()),
                  excluded_participants=int(len(np.unique(participants))-len(eligible)))
    if len(np.unique(families[keep])) < 2:
        diagnostic = dict(status='insufficient_family_or_class_coverage', **counts)
    else:
        diagnostic = dict(status='evaluated', **counts,
                          result=summarize(y[keep], probabilities[keep], participants[keep], families[keep],
                                           model_names, seed=seed, replicates=replicates))
    return dict(primary=main, discordant_positions=diagnostic,
                interpretation='The schedule slice tests the simplest shared-position template, not all confounding or causal specificity.')
