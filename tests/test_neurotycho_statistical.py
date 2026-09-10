from pathlib import Path

import numpy as np
import pytest
import yaml

from src.neurotycho_statistical import fit_grouped, predict_fitted


def fixture_bank():
    rng = np.random.default_rng(12)
    y = np.tile([0, 1], 18)
    return dict(y=y, animal=np.repeat(['a', 'b', 'c'], 12), archive=np.repeat(['da', 'db', 'dc'], 12),
                record_id=np.array([str(i) for i in range(36)]), M=np.full(36, 16), T=np.full(36, 2000),
                z=np.column_stack([y, y + rng.normal(0, .1, 36), np.full(36, np.nan), np.ones(36)]))


def config():
    result = yaml.safe_load(Path('configs/analysis/neurotycho-transfer-260910.yaml').read_text())
    result.update(pca_caps=[1, 2], ridge_alpha_grid=[.01, 1.], pls_components=[1, 2])
    return result


@pytest.mark.parametrize('method', ['z-pca', 'z-pls'])
def test_grouped_selection_predictions_and_source_only_transforms(method):
    bank = fixture_bank()
    fitted, report = fit_grouped(bank, method, config())
    for candidate in report['candidates']:
        scores = []
        for fold in candidate['folds']:
            train, valid = set(fold['training_ids']), set(fold['validation_ids'])
            assert train.isdisjoint(valid) and len(train) == 24 and len(valid) == 12
            assert set(bank['animal'][[int(i) for i in valid]]) == {fold['animal']}
            y, p = np.array(fold['y']), np.array(fold['probability'])
            score = np.mean([np.mean((p[y == k] - k)**2) for k in [0, 1]])
            assert abs(score - fold['balanced_brier']) < 1e-12
            scores.append(score)
        assert abs(np.mean(scores) - candidate['mean_brier']) < 1e-12
    assert report['selected_source_validation_brier'] == min(c['mean_brier'] for c in report['candidates'])
    assert report['selected_source_validation_brier'] < .05
    before = fitted['transform'].blocks[0].mean.copy()
    unseen = {**bank, 'z': bank['z'] + 10000}
    assert np.isfinite(predict_fitted(fitted, unseen, np.arange(36))).all()
    np.testing.assert_array_equal(before, fitted['transform'].blocks[0].mean)
    assert len(fitted['transform'].blocks[0].keep) == 2


def test_reject_reduced_views_and_wrong_animal_count():
    bank = fixture_bank(); bank['M'][0] = 8
    with pytest.raises(ValueError, match='full source'):
        fit_grouped(bank, 'z-pca', config())
    bank = fixture_bank(); bank['animal'][0] = 'd'
    with pytest.raises(ValueError, match='three source'):
        fit_grouped(bank, 'z-pca', config())
