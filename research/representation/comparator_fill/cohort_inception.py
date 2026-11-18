"""InceptionTime on the character-axes cohort windows, under the cohort split.

One participant-grouped five-fold split per contrast (the first repeat of the cohort protocol). Inside each
training fold the discovery-screen recipe is reused unchanged (dataset_screen.raw.neural): learning rate and epoch
count chosen on three grouped folds of the training rows, then a five-seed ensemble. Unit probability = mean window
probability. Member weights are deleted after scoring; they are cheap to refit.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mtsspi.provenance import file_sha256
from research.representation.comparator_fill.cohort import CONTRASTS, aggregate, splits, summary, windows
from research.representation.dataset_screen import raw


def main(cohort, root, output, only):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(device)
    bank = np.load(root / 'bank.npz')
    x_all = windows(root, bank).astype(np.float32)
    y_all, groups_all = bank['y'], bank['group'].astype(str)
    report_path = output / 'inceptiontime.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for label, positive, negative in CONTRASTS[cohort]:
        if label in report or (only and label not in only):
            continue
        ix = np.flatnonzero(np.isin(y_all, [positive, negative]))
        x, y, groups = x_all[ix], (y_all[ix] == positive).astype(int), groups_all[ix]
        window_p = np.full((1, len(y)), np.nan)
        folds_report = []
        for k, (_, train, test) in enumerate(splits(cohort, y, groups, None, None, 1)):
            out = output / label / f'fold{k}'
            out.mkdir(parents=True, exist_ok=True)
            identity = dict(cohort=cohort, contrast=label, fold=k, training_rows=train.tolist(), device=device,
                            bank_sha256=file_sha256(root / 'bank.npz'), runner_sha256=file_sha256(Path(__file__)))
            fitted = raw.neural(root, 'inceptiontime', x[train], y[train], groups[train], identity, out)
            members = []
            for member in fitted['members']:
                saved = torch.load(out / member['path'], weights_only=True)
                model = raw.make_network('inceptiontime', saved['spec'])
                model.load_state_dict(saved['state_dict'])
                members.append(raw.network_prediction(model, (torch.as_tensor(x[test]) - saved['mean']) / saved['scale'])[:, 1])
                (out / member['path']).unlink()
            window_p[0, test] = np.mean(members, axis=0)
            folds_report.append(dict(lr=fitted['selected']['lr'], epochs=fitted['final_epochs'],
                                     source_validation_brier=fitted['source_validation_brier']))
            print(cohort, label, 'fold', k, folds_report[-1], flush=True)
        keys, unit_label, unit_p = aggregate(cohort, y, groups, None, window_p, 1)
        report[label] = dict(scores=dict(inceptiontime=summary(unit_label, unit_p)), units=dict(inceptiontime=unit_p.tolist()),
                             unit_keys=keys.tolist(), unit_labels=unit_label.tolist(), folds=folds_report, repeats=1)
        report_path.write_text(json.dumps(report, indent=1) + '\n')
        print(cohort, label, report[label]['scores'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=['ad_ftd', 'lemon', 'propofol'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', nargs='*')
    a = parser.parse_args()
    main(a.cohort, a.root, a.output, a.only)
