import numpy as np
from scipy.stats import multivariate_t
from scripts.scout_tail_alignment import entropy,population,recording


def test_population_means_match_but_joint_structure_and_tail_quantity_change():
    rows=[population(j) for j in range(5)]
    for key in ['mean_r','mean_tau','mean_MI']:
        np.testing.assert_allclose([r[key] for r in rows],rows[0][key],rtol=0,atol=1e-14)
    assert np.all(np.diff([r['Q_tail'] for r in rows])>0)
    assert rows[-1]['z_r_MI']-rows[0]['z_r_MI']>.15


def test_entropy_against_scipy_and_recording_replay():
    for nu in [3.,30.]:
        for d in [1,2]:
            np.testing.assert_allclose(entropy(d,nu),multivariate_t(shape=np.eye(d),df=nu).entropy(),atol=1e-12)
    x=recording(2,8)
    assert x.shape==(16,1000) and np.isfinite(x).all()
    np.testing.assert_array_equal(x,recording(2,8))
    np.testing.assert_allclose(x.std(1),1,atol=1e-12)
def test_p90_readout_uses_observed_not_population_pearson(tmp_path,monkeypatch):
    import json
    import pandas as pd
    from scripts.analyze_tail_alignment import analyze
    from scripts import plot_large_m_pair_sampling
    monkeypatch.setattr(plot_large_m_pair_sampling,'figure_style',lambda:None)
    rows=[];raw={};rng=np.random.default_rng(42)
    for control in range(3):
        for seed in range(8):
            key=f'c{control}-s{seed}';raw[key]=rng.normal(size=(4,30))
            rows.append(dict(row_id=key,role='development' if seed<4 else 'evaluation',
                control=control/2,seed=seed,Q_tail=.01+.002*control,mean_r=.01))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(rows=rows)))
    np.savez(tmp_path/'observations.npz',**raw)
    np.savez(tmp_path/'features.npz',row_id=[r['row_id'] for r in rows],
        mean=rng.normal(size=(24,4)),z=rng.normal(size=(24,6)),
        distribution=rng.normal(size=(24,12)),spi_order=['a','b','c','d'])
    analyze(tmp_path,tmp_path)
    scores=pd.read_csv(tmp_path/'scores.csv')
    expected=[np.corrcoef(raw[key])[~np.eye(4,dtype=bool)].mean() for key in scores.row_id]
    np.testing.assert_allclose(scores.empirical_mean_r,expected,atol=1e-14)
    assert scores.empirical_mean_r.std()>.001
    assert pd.read_csv(tmp_path/'metrics.csv').n.nunique()==1
