"""Retrospective spectrum-only comparator on an existing interaction-share corpus.

Uses the phase experiment's unchanged 165-feature map and PLS grid. Does not alter
the original corpus, protocol, feature bank or fit artifacts.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sklearn
import yaml

from src.interaction_share_learning import fit_statistical, select_statistical
from src.phase_surrogates import pooled_autospectrum
from src.representation_screen import training_subsets
from src.representation_state_data import file_hash, load_state_data, observed_view, source_pool_for_seed
from src.run_external_corpus import _atomic_json, _atomic_savez


def run(config, data, output):
    protocol = yaml.safe_load(config.read_text())
    manifest, masters = load_state_data(data, config)
    rows = manifest['rows']
    spectra = np.stack([pooled_autospectrum(observed_view(masters[r['master_index']], r['M'], r['T']))
                        for r in rows])
    output.mkdir(parents=True, exist_ok=True)
    bank_path = output / 'autospectra.npz'
    _atomic_savez(bank_path, dict(X_u=spectra, row_id=np.asarray([r['row_id'] for r in rows])))
    identity = dict(protocol_sha256=file_hash(config), manifest_sha256=file_hash(data / 'manifest.json'),
                    feature_bank_sha256=file_hash(bank_path), numpy=np.__version__, sklearn=sklearn.__version__,
                    code_sha256={str(p): file_hash(p) for p in [Path(__file__), Path('src/phase_surrogates.py'),
                                  Path('src/interaction_share_learning.py'), Path('src/representation_screen.py')]})
    target = np.asarray([r['target'] for r in rows])
    strata = np.asarray([r['coupling_index'] for r in rows])
    evaluation = np.asarray([i for i, r in enumerate(rows) if r['role'] == 'evaluation'])
    bank = {'u': spectra}
    methods = protocol['methods']
    for family in protocol['generator']['families']:
        pool = np.asarray([i for i, r in enumerate(rows) if r['family'] == family and r['role'] == 'training_pool'])
        for seed in methods['subset_seeds']:
            cohort = source_pool_for_seed(rows, pool, protocol, seed)
            for n, train in training_subsets(strata, cohort, protocol['sampling']['labelled_training_masters_per_coupling'], seed).items():
                stem = output / family / f'autospectrum-pls-n{n}-s{seed}'
                ident = dict(identity, method='autospectrum-pls', source_family=family, n_per_coupling=n, seed=seed)
                if stem.with_suffix('.json').exists():
                    old = json.loads(stem.with_suffix('.json').read_text())
                    assert old['identity'] == ident and old['predictions_sha256'] == file_hash(stem.with_suffix('.npz'))
                    continue
                chosen, details = select_statistical(bank, 'u', train, target, strata, methods, seed, 'pls')
                transform, model = fit_statistical(bank, 'u', train, target, methods['preprocessing'], 'pls', *chosen)
                prediction = np.clip(model.predict(transform.transform(bank, evaluation)).reshape(-1), 0, 1)
                _atomic_savez(stem.with_suffix('.npz'), dict(prediction=prediction, target=target[evaluation],
                    train_indices=train, evaluation_indices=evaluation, row_id=np.asarray([rows[i]['row_id'] for i in evaluation])))
                _atomic_json(stem.with_suffix('.json'), dict(identity=ident, details=dict(details, chosen=chosen),
                    labels_total=len(train), predictions_sha256=file_hash(stem.with_suffix('.npz')),
                    status='retrospective_control_motivated_by_phase_experiment'))
    print(f'Completed autospectrum PLS: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['config', 'data', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.data, args.output)
