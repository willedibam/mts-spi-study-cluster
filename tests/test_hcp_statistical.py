from pathlib import Path

import numpy as np
import pytest
import yaml

from src.hcp_statistical import fit_source, predict


@pytest.mark.parametrize('method', ['m-pca', 'z-pca', 'mz-pca', 'm-pls', 'z-pls', 'spectra', 'minirocket'])
def test_source_grouping_and_prediction_on_random_inputs(method):
    config = yaml.safe_load(Path('configs/analysis/hcp-statistical-candidates-260915.yaml').read_text())
    config['pca_caps'] = [2]
    config['ridge_alpha_grid'] = [1.]
    config['pls_components'] = [1]
    config['linear_classification']['logistic_C'] = [1.]
    config['minirocket']['minirocket_kernels'] = 84
    config['selection']['folds'] = 2
    rng = np.random.default_rng(715)
    bank = dict(y=np.tile([0, 1], 8), participant=np.repeat(['a', 'b', 'c', 'd'], 4),
                family=np.repeat(['twins', 'twins', 'f2', 'f3'], 4),
                record_id=np.array([f'r{i}' for i in range(16)]),
                x=rng.normal(size=(16, 64, 3)).astype('float32'),
                z=rng.normal(size=(16, 21)), m=rng.normal(size=(16, 46)),
                spectra=rng.uniform(size=(16, 18)))
    fitted, report = fit_source(bank, method, config, 151, threads=1)
    probability = predict(fitted, bank)
    assert probability.shape == (16,) and np.isfinite(probability).all()
    assert np.all((probability >= 0) & (probability <= 1))
    assert not report['target_data_used'] and report['family_count'] == 3
    visited = []
    choice = report['candidates'][0]
    for fold in choice['folds']:
        train = [int(i[1:]) for i in fold['training_ids']]
        valid = [int(i[1:]) for i in fold['validation_ids']]
        assert not set(bank['family'][train]) & set(bank['family'][valid])
        visited.extend(valid)
    assert sorted(visited) == list(range(16))
    oof = np.asarray(choice['oof_probability'])
    expected = np.mean([np.mean((oof[bank['participant'] == p]-bank['y'][bank['participant'] == p])**2)
                        for p in np.unique(bank['participant'])])
    assert choice['mean_participant_brier'] == pytest.approx(expected)
    # Prediction must use the frozen source fit, not labels in the evaluation bank.
    changed = {**bank, 'y': 1-bank['y']}
    np.testing.assert_array_equal(predict(fitted, changed), probability)
