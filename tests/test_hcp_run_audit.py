import numpy as np
import pandas as pd
import pytest

from src.hcp_run_audit import align_eprime, ica_projection


def test_missing_retained_trials_do_not_shift_labels_or_times():
    task = np.zeros((8, 9))
    task[:, 1] = [1, 1, 2, 2, 4, 4, 5, 5]
    task[:, 3] = [1, 1, 1, 1, 2, 2, 2, 2]
    task[:, 4] = [1, 1, 1, 1, 2, 2, 2, 2]
    task[:, 8] = [1, 2, 1, 2, 1, 2, 1, 2]
    onsets = np.array([1000, 3500, 15000, 17500, 40000, 42500, 54000, 56500])
    task[:, 6] = 1+2*onsets+5000
    frame = pd.DataFrame({'Stim.OnsetTime': onsets, 'BlockNumber': [1, 1, 2, 2, 1, 1, 2, 2],
                          'StimType': ['Face']*4+['Tools']*4,
                          'BlockType': ['0-Back']*4+['2-Back']*4,
                          'Fix15sec.OnsetTime': [np.nan]*8})
    frame.loc[8] = [np.nan, np.nan, None, None, 20000]
    frame.loc[9] = [np.nan, np.nan, None, None, 100]  # Leading rest does not renumber task blocks.
    retained = task[[0, 3, 4, 6, 7]]
    result = align_eprime(retained, frame, 2000)
    assert result['retained_events'] == 5 and result['eprime_events'] == 8
    assert result['clock_slope'] == pytest.approx(1)
    assert result['offset_seconds'] == pytest.approx(2.5)
    assert result['max_affine_timing_residual_seconds'] < 1e-12
    frame.loc[7, 'BlockType'] = '0-Back'
    with pytest.raises(ValueError, match='Memory label mismatch'):
        align_eprime(retained, frame, 2000)


def test_ica_projection_uses_run_annotation_and_handles_empty_exclusion():
    mixing, _ = np.linalg.qr(np.random.default_rng(41).normal(size=(5, 3)))
    comp = dict(topolabel=[f'A{i}' for i in range(5)], topo=mixing, unmixing=mixing.T)
    _, projection, excluded, error = ica_projection(comp, 'vs.ecg_eog_ic = [1 3];')
    np.testing.assert_array_equal(excluded, [0, 2])
    np.testing.assert_allclose(mixing.T[excluded]@projection, 0, atol=1e-12)
    np.testing.assert_allclose(projection@mixing[:, 1], mixing[:, 1], atol=1e-12)
    assert error < 1e-12
    _, identity, empty, _ = ica_projection(comp, 'vs.ecg_eog_ic = [];')
    np.testing.assert_array_equal(identity, np.eye(5))
    assert empty.size == 0
    with pytest.raises(ValueError, match='Invalid or duplicate'):
        ica_projection(comp, 'vs.ecg_eog_ic = [4];')
