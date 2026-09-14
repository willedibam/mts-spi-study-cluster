import numpy as np
from scripts.finite_regime_pipeline import apply_coordinate, fit_coordinate, export_arrays, physics_identities, verify_physics_identity
import json
import pytest


def test_coordinate_ignores_evaluation_features_and_has_frozen_scale():
    rng = np.random.default_rng(1)
    signal = np.linspace(-1,1,80)
    z = signal[:,None] * rng.normal(size=(1,30)) + rng.normal(scale=.01,size=(80,30))
    dev = np.arange(80) < 60
    seeds = np.arange(80)%4
    model, gate = fit_coordinate(z,dev,seeds)
    changed = z.copy()
    changed[~dev] *= 1e6
    other, _ = fit_coordinate(changed,dev,seeds)
    for key in model:
        np.testing.assert_array_equal(model[key],other[key])
    q, missing = apply_coordinate(z,model)
    assert gate['passes'] and not missing.any()
    np.testing.assert_allclose(q[dev].std(),1,atol=1e-12)


def test_export_has_named_full_state_arrays(tmp_path):
    rows = [dict(row_id=f'row-{i}',corpus_index=i+1,M=6,T=1000,
        system='test',view='full-state',role='development') for i in range(4)]
    arrays = {row['row_id']:np.arange(6000).reshape(6,1000)+i for i,row in enumerate(rows)}
    out = tmp_path/'corpus'
    export_arrays(out,arrays,rows,dict(system='test'))
    manifest = json.loads((out/'manifest.json').read_text())
    assert len(manifest['rows']) == 4
    with np.load(out/'observations.npz') as a:
        assert a['__axis_order__'].tolist() == ['process','observation']
        assert a['row-0'].shape == (6,1000)


def test_physics_gate_rejects_changed_master(tmp_path):
    np.savez(tmp_path/'case-0000.npz', X=np.ones((3,10)))
    gate = dict(master_identities=physics_identities(tmp_path))
    verify_physics_identity(tmp_path,gate)
    np.savez(tmp_path/'case-0000.npz', X=np.zeros((3,10)))
    with pytest.raises(ValueError,match='identities differ'):
        verify_physics_identity(tmp_path,gate)


def test_confirmation_export_keeps_all_evaluation_roles(tmp_path):
    rows = [dict(row_id=f'row-{i}', corpus_index=i+1, M=3, T=10,
        system='test', view='full-state', role='evaluation') for i in range(4)]
    arrays = {row['row_id']:np.ones((3,10))*i for i,row in enumerate(rows)}
    out = tmp_path/'confirmation'
    export_arrays(out,arrays,rows,dict(system='test'))
    manifest = json.loads((out/'manifest.json').read_text())
    assert all(row['role'] == 'evaluation' for row in manifest['rows'])
    assert (out/'smoke-indices.txt').read_text().splitlines() == ['1','4']
    assert (out/'node-indices.txt').read_text().splitlines() == ['2','3']
