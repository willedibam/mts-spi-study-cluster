import numpy as np
import json
import yaml
from pathlib import Path

from research.representation.empirical_transfer.build_bank import raw_features
from research.representation.empirical_transfer.fit_source import source_folds, fit_fmri, group_brier


def test_grouped_folds_never_split_a_participant():
    groups = np.repeat([f'p{i}' for i in range(9)], 4)
    validation = []
    for fit, val in source_folds(groups):
        assert set(groups[fit]).isdisjoint(groups[val])
        validation.extend(val)
    np.testing.assert_array_equal(sorted(validation), np.arange(len(groups)))


def test_group_metrics_do_not_count_duplicate_windows_as_people():
    y = np.array([0, 1, 0, 1])
    p = np.array([.1, .9, .5, .5])
    groups = np.array(['a', 'a', 'b', 'b'])
    ix = np.r_[np.tile([0, 1], 20), [2, 3]]
    assert group_brier(y, p, groups) == group_brier(y[ix], p[ix], groups[ix])


def test_small_source_only_pca_and_kernel_fits():
    rng = np.random.default_rng(1)
    groups = np.repeat([f'p{i}' for i in range(9)], 2)
    y = np.tile([0, 1], 9)
    bank = dict(m=rng.normal(size=(18, 5)) + y[:, None], y=y, group=groups,
                row_id=np.array([f'r{i}' for i in range(18)]))
    config = dict(pca_caps=[2], ridge_alpha_grid=[1], preprocessing=dict(
        minimum_valid_fraction=.95, variance_threshold=1e-8, z_scaling='center',
        clip_standard_deviations=5, pca_solver='randomized', pca_random_state=1729))
    for method in ['m-pca', 'm-rbf']:
        model, report = fit_fmri(bank, method, config)
        assert np.isfinite(report['selected_source_validation_brier'])
        assert len(report['candidates'][0]['folds']) == 3


def test_raw_connectivity_retains_aligned_edges():
    x = np.random.default_rng(8).normal(size=(128, 3))
    x[:, 2] = x[:, 0]
    p, s, pooled = raw_features(x)
    assert p.shape == (3,) and np.isclose(p[1], 1)
    assert s.shape == (96,) and pooled.shape == (160,)


def test_raw_loader_does_not_read_confirmation_arrays(tmp_path):
    from research.representation.empirical_transfer.fit_raw import inputs
    from mtsspi.provenance import file_sha256
    np.savez(tmp_path / 'views.npz', source=np.ones((128, 17)))
    manifest = dict(archive_sha256=file_sha256(tmp_path / 'views.npz'), rows=[
        dict(row_id='source', role='source', group='p1', target=0),
        dict(row_id='intentionally_absent', role='confirmation_primary', group='p2', target=1)])
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    x, y, groups, rows = inputs(tmp_path)
    assert x.shape == (1, 128, 17) and groups.tolist() == ['p1']


def test_existing_neural_architectures_accept_seventeen_networks():
    import torch
    from research.representation.shared.representation_state_neural import make_encoder
    from research.representation.learned_baselines.inceptiontime_baseline import InceptionNetwork
    torch.set_num_threads(1)
    root = Path('research/representation')
    a = yaml.safe_load((root / 'neurotycho/configs/analysis/neurotycho-transfer-260910.yaml').read_text())['raw_encoder']
    b = yaml.safe_load((root / 'learned_baselines/configs/analysis/inceptiontime-followup-260911.yaml').read_text())['architecture']
    for model in [make_encoder(a), InceptionNetwork({**b, 'input_channels': 17})]:
        model.eval()
        with torch.no_grad():
            result = model(torch.randn(2, 128, 17))
        assert result.shape == (2,) and torch.isfinite(result).all()


def test_resume_requires_matching_inputs_and_training_code(tmp_path):
    import shutil
    import pytest
    from research.representation.empirical_transfer.fit_raw import resume_candidates
    from mtsspi.provenance import file_sha256
    old_root = tmp_path / 'snapshot'
    runner = old_root / 'research/representation/empirical_transfer/fit_raw.py'
    runner.parent.mkdir(parents=True)
    runner.write_text('previous orchestration revision')
    paths = ['research/representation/empirical_transfer/splits.py',
             'research/representation/shared/neurotycho_learning.py',
             'research/representation/shared/representation_state_neural.py',
             'research/representation/learned_baselines/inceptiontime_baseline.py']
    for name in paths:
        (old_root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, old_root / name)
    old = dict(runner_sha256=file_sha256(runner), manifest_sha256='original-inputs')
    current = dict(old, runner_sha256='new-orchestration')
    progress = tmp_path / 'progress.json'
    progress.write_text(json.dumps(dict(identity=old, completed_candidates=[],
        current=dict(lr=.001, decay=0., folds=[dict(fold=0, best_epoch=20)]))))
    assert resume_candidates(progress, current, old_root)[(.001, 0., 0)]['best_epoch'] == 20
    with pytest.raises(ValueError, match='source inputs'):
        resume_candidates(progress, {**current, 'manifest_sha256': 'different'}, old_root)
    (old_root / paths[0]).write_text('different training implementation')
    with pytest.raises(ValueError, match='training implementation'):
        resume_candidates(progress, current, old_root)
