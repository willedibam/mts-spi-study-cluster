"""MiniRocket on the same standardized cohort windows, folds and unit aggregation.

The transform is refitted on each outer training fold; ridge alpha is chosen on
inner grouped folds of those training features (the transform itself is
label-free). Scores are ridge decision values; unit scores average windows.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from research.representation.character_axes.evaluate_cohorts import CONTRASTS, summary, unit_labels
from research.representation.dataset_screen.common import folds

ALPHAS = [1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5]


def main(cohort, root, output, repeats):
    from aeon.transformations.collection.convolution_based import MiniRocket
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = manifest['rows']
    with np.load(root / 'views.npz', allow_pickle=False) as views:
        x = np.stack([views[r['row_id']] for r in rows]).astype(np.float32).transpose(0, 2, 1)
    y_all = np.array([r['target'] for r in rows]); groups_all = np.array([r['group'] for r in rows])
    report = {}
    for label, positive, negative in CONTRASTS[cohort]:
        ix = np.flatnonzero(np.isin(y_all, [positive, negative]))
        y, groups = (y_all[ix] == positive).astype(int), groups_all[ix]
        units = unit_labels(y, groups); keys = np.unique(units)
        scores = np.zeros((repeats, len(keys)))
        for r in range(repeats):
            s = np.zeros(len(y))
            for train, test in StratifiedGroupKFold(5, shuffle=True, random_state=261102 + r).split(y, y, groups):
                transform = MiniRocket(n_kernels=10000, max_dilations_per_kernel=32, n_jobs=4, random_state=11)
                f_train = transform.fit_transform(x[ix[train]]).astype(np.float64)
                f_test = transform.transform(x[ix[test]]).astype(np.float64)
                scaler = StandardScaler().fit(f_train)
                f_train, f_test = scaler.transform(f_train), scaler.transform(f_test)
                best = None
                for alpha in ALPHAS:
                    p = np.zeros(len(train))
                    for a, b in folds(y[train], groups[train]):
                        p[b] = Ridge(alpha=alpha).fit(f_train[a], y[train][a]).predict(f_train[b])
                    loss = np.mean((np.clip(p, 0, 1) - y[train]) ** 2)
                    if best is None or loss < best[0]: best = (loss, alpha)
                s[test] = Ridge(alpha=best[1]).fit(f_train, y[train]).predict(f_test)
                print(cohort, label, 'repeat', r, 'alpha', best[1], flush=True)
            scores[r] = [s[units == k].mean() for k in keys]
        label_unit = np.array([y[units == k][0] for k in keys])
        p = np.clip(scores.mean(0), 0, 1)
        report[label] = dict(minirocket=summary(label_unit, p), auroc_from_scores=float(roc_auc_score(label_unit, scores.mean(0))),
                             keys=keys.tolist(), scores=scores.mean(0).tolist())
        print(cohort, label, report[label]['minirocket'], flush=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=sorted(CONTRASTS))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    main(args.cohort, args.root, args.output, args.repeats)
