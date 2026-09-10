"""Fixed-threshold, date-then-animal summaries; no target calibration."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.neurotycho_learning import balanced_brier


def summarize_run(y, probability, animal, archive, m, t):
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('invalid prediction probabilities')
    dates, animals = [], []
    metrics = ['balanced_accuracy', 'auroc', 'balanced_brier']
    for name in sorted(set(archive)):
        ix = archive == name
        assert len(set(animal[ix])) == 1
        dates.append(dict(archive=str(name), animal=str(animal[ix][0]), M=int(m), T=int(t), n=int(ix.sum()),
            balanced_accuracy=float(balanced_accuracy_score(y[ix], probability[ix] >= .5)),
            auroc=float(roc_auc_score(y[ix], probability[ix])), balanced_brier=balanced_brier(y[ix], probability[ix])))
    for name in sorted(set(animal)):
        rows = [row for row in dates if row['animal'] == name]
        animals.append(dict(animal=str(name), dates=len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in metrics}))
    return dict(M=int(m), T=int(t), dates=dates, animals=animals,
                mean={k: float(np.mean([r[k] for r in animals])) for k in metrics})
