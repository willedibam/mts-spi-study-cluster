"""Do z and the core add over per-channel baselines? Same design as sufficiency.py (run on Gadi).

S = Pearson+strength+channel spectra; Q = Pearson+strength+channel value shape (per-channel skewness,
kurtosis and amplitude-envelope CV, as mean and SD over channels). Reports AUROC of S, S+core, S+z,
S+core+z, Q, Q+z and paired bootstrap intervals for core over S, z over S, z over S+core and z over Q.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import signal, stats

from research.representation.character_axes.evaluate_cohorts import CONTRASTS, paired_bootstrap, summary
from research.representation.character_axes.sufficiency import blocks, predict


def value_shape(x):
    x = np.asarray(x, dtype=np.float64)                       # (n, T, M)
    envelope = np.abs(signal.hilbert(x, axis=1))
    per = [stats.skew(x, axis=1), stats.kurtosis(x, axis=1), envelope.std(1) / envelope.mean(1)]
    return np.column_stack([f(v, 1) for v in per for f in (np.mean, np.std)])


SETS = dict(S=['pearson', 'strength', 'spectral'], S_core=['pearson', 'strength', 'spectral', 'core'],
            S_z=['pearson', 'strength', 'spectral', 'z'], S_core_z=['pearson', 'strength', 'spectral', 'core', 'z'],
            Q=['pearson', 'strength', 'shape'], Q_z=['pearson', 'strength', 'shape', 'z'])
PAIRS = dict(core_over_S=('S_core', 'S'), z_over_S=('S_z', 'S'), z_over_S_core=('S_core_z', 'S_core'), z_over_Q=('Q_z', 'Q'))


def main(cohort, root, output):
    bank = np.load(root / 'bank.npz')
    X, *_ = blocks(bank)
    X['spectral'] = np.asarray(bank['spectral'])
    X['shape'] = value_shape(bank['x'])
    y_all, groups_all = bank['y'], bank['group'].astype(str)
    if cohort == 'neurotycho':
        groups_all = np.array([f'{g}-{r}' for g, r in zip(groups_all, bank['role'].astype(str))])
    report = dict(cohort=cohort, contrasts={})
    for label, positive, negative in CONTRASTS[cohort]:
        ix = np.isin(y_all, [positive, negative])
        y, groups = (y_all[ix] == positive).astype(int), groups_all[ix]
        Xc = {k: v[ix] for k, v in X.items()}
        pred, scores = {}, {}
        for name, names in SETS.items():
            keys, unit_label, pred[name] = predict(cohort, Xc, y, groups, names, bank, ix)
            scores[name] = summary(unit_label, pred[name])['auroc']
        diffs = {k: dict(difference=scores[a] - scores[b], interval=paired_bootstrap(unit_label, pred[a], pred[b], keys))
                 for k, (a, b) in PAIRS.items()}
        report['contrasts'][label] = dict(auroc=scores, differences=diffs)
        print(cohort, label, {k: round(v, 3) for k, v in scores.items()},
              {k: (round(v['difference'], 3), [round(t, 3) for t in v['interval']]) for k, v in diffs.items()}, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, default=float) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('cohort', choices=sorted(CONTRASTS))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.cohort, a.root, a.output)
