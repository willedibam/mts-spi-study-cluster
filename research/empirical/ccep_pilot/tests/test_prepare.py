import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy.stats import pearsonr

from research.empirical.ccep_pilot.prepare import clean_windows, feature_vector, lag_matrix


def test_windows_are_disjoint_and_filter_margins_avoid_annotations():
    events = pd.DataFrame([dict(onset=60., duration=10., trial_type='artefact'),
                           dict(onset=600., duration=20., trial_type='seizure')])
    windows = clean_windows(events, 550, 32)
    assert len(windows) == 3
    assert all(b-a >= 32 for a,b in zip(windows, windows[1:]))
    for start in windows:
        assert start-4 >= 5 and start+36 <= 300
        assert start+36 <= 59 or start-4 >= 71


def test_lag_uses_physical_time_and_both_directions():
    x = np.random.default_rng(94).normal(size=(3, 4096))
    x[1] += np.roll(x[0], 16)
    expected = (pearsonr(x[0,16:],x[1,:-16]).statistic + pearsonr(x[1,16:],x[0,:-16]).statistic)/2
    assert_allclose(lag_matrix(x, 1/128, 2048)[0,1], expected, atol=1e-14)


def test_collapsed_profile_is_undefined_without_imputation():
    z = feature_vector({'constant':np.ones(8), 'ascending':np.arange(8.), 'descending':-np.arange(8.)})
    assert np.isnan(z[:2]).all()
    assert_allclose(z[2], -1.)


def test_revised_archive_pairs_reference_with_first_retained_window(tmp_path, monkeypatch):
    import json
    from research.empirical.ccep_pilot.prepare import build
    subject='sub-ccepAgeUMCU73'
    (tmp_path/'scope').mkdir();(tmp_path/'prepared').mkdir()
    report=[dict(subject=subject,status='audited',starts=[10.,50.,90.],channels=['C25','C26'])]
    (tmp_path/'scope/participants.json').write_text(json.dumps(report))
    pd.DataFrame([dict(subject=subject,window=w,qc_pass=True) for w in range(3)]).to_csv(tmp_path/'scope/quality.csv',index=False)
    rng=np.random.default_rng(17)
    bank={f'w{w}_fs128_bw45':rng.normal(size=(2,4096)) for w in range(3)}
    np.savez(tmp_path/'prepared'/f'{subject}.npz',**bank)
    calls=[]
    class Native:
        def get_data(self,picks,start,stop):
            calls.append((start,stop));return rng.normal(size=(2,stop-start))
    monkeypatch.setattr('mne.io.read_raw_brainvision',lambda *a,**k:Native())
    build(tmp_path,128,45,32,True,subject,[0],'-reviewed')
    folder=tmp_path/'corpora'/f'{subject}-fs128-t4096-reviewed'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert [r['window'] for r in manifest['rows']]==[1,2,1]
    assert calls==[(46*2048,86*2048)]
    assert manifest['excluded_windows']==[0]
    with np.load(folder/'observations.npz') as saved:
        original=bank['w1_fs128_bw45']
        assert_allclose(saved[manifest['rows'][0]['name']],(original-original.mean(axis=1,keepdims=True))/original.std(axis=1,keepdims=True))
