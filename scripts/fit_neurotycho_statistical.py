"""Fit one frozen source-only statistical comparator, retaining grouped CV evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import yaml

from src.neurotycho_statistical import METHODS, fit_grouped, predict_fitted


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bank(paths, spectral):
    spectral_report = json.loads(spectral.with_suffix('.json').read_text())
    assert spectral_report['phase'] == 'source' and sha(spectral) == spectral_report['sha256']
    pieces = []
    keys = ['m', 'g', 'z', 'validity', 'record_id', 'animal', 'archive', 'y', 'M', 'T']
    for path in paths:
        report = json.loads(path.with_suffix('.json').read_text())
        assert sha(path) == report['sha256'] and 'KTMD' in report['phase']
        with np.load(path) as values:
            pieces.append({key: values[key] for key in keys})
    bank = {key: np.concatenate([piece[key] for piece in pieces]) for key in keys}
    assert len(bank['y']) == 704
    identity = list(zip(bank['record_id'], bank['M'], bank['T'], strict=True))
    assert len(set(identity)) == 704
    with np.load(spectral) as values:
        lookup = {(r, m, t): i for i, (r, m, t) in enumerate(zip(values['record_id'], values['M'], values['T'], strict=True))}
        indices = [lookup[key] for key in identity]
        for key in ['y', 'animal', 'archive']:
            np.testing.assert_array_equal(bank[key], values[key][indices])
        bank['spectral'] = values['spectral'][indices]
    return bank


def main(args):
    config = yaml.safe_load(args.config.read_text())
    assert args.animal in config['target_animals']
    stem = args.output / f'{args.method}-{args.animal}'
    identity = dict(config_sha256=sha(args.config), animal=args.animal, method=args.method,
        inputs={str(p): sha(p) for p in [*args.banks, args.spectral]},
        modules={str(p): sha(p) for p in [Path(__file__), Path('src/neurotycho_statistical.py'),
            Path('src/representation_screen.py'), Path('src/interaction_share_learning.py')]})
    if stem.with_suffix('.json').exists():
        report = json.loads(stem.with_suffix('.json').read_text())
        assert report['identity'] == identity and sha(stem.with_suffix('.joblib')) == report['model_sha256']
        print('Reused', stem); return
    bank = load_bank(args.banks, args.spectral)
    keep = (bank['animal'] != args.animal) & (bank['M'] == 16) & (bank['T'] == 2000)
    bank = {key: values[keep] for key, values in bank.items()}
    assert args.animal not in bank['animal']
    started = time.perf_counter()
    fitted, report = fit_grouped(bank, args.method, config)
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted, stem.with_suffix('.joblib'))
    probability = predict_fitted(fitted, bank, np.arange(len(bank['y'])))
    replay = predict_fitted(joblib.load(stem.with_suffix('.joblib')), bank, np.arange(len(bank['y'])))
    np.testing.assert_array_equal(replay, probability)
    np.savez_compressed(stem.with_suffix('.npz'), training_ids=bank['record_id'], y=bank['y'], probability=probability)
    report.update(identity=identity, model_sha256=sha(stem.with_suffix('.joblib')),
        predictions_sha256=sha(stem.with_suffix('.npz')), training_windows=len(bank['y']),
        target_data_used=False, seconds=time.perf_counter()-started, saved_model_replay_max_difference=0.)
    stem.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(args.method, args.animal, 'selected', report['selected'], 'source CV Brier',
          report['selected_source_validation_brier'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/analysis/neurotycho-transfer-260910.yaml'))
    parser.add_argument('--banks', type=Path, nargs='+', required=True)
    parser.add_argument('--spectral', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--animal', required=True)
    parser.add_argument('--method', choices=list(METHODS), required=True)
    main(parser.parse_args())
