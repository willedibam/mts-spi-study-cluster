import json

import numpy as np

from scripts.analyze_stuart_landau_temporal_scout import summarize


def archive(tmp_path, amplitude):
    meta = dict(N=2, gamma=.8, seed=1, burn=200., dt=.02,
                sample_dt=.1, samples=len(amplitude), elapsed_seconds=0., carrier=2.)
    times = meta['burn'] + (np.arange(len(amplitude)) + 1) * meta['sample_dt']
    z = amplitude * np.exp(2j * times)
    path = tmp_path / 'fixture.npz'
    np.savez(path, Z=z, observed=np.column_stack([z, z]), metadata_json=json.dumps(meta))
    return path


def test_rotating_carrier_does_not_create_collective_amplitude_fluctuation(tmp_path):
    row = summarize(archive(tmp_path, np.ones(1000)))
    assert row['R_std'] < 1e-12
    assert np.isnan(row['R_spectral_entropy'])
    assert row['lab_real_channel_sd_median'] > .5
    assert row['rotating_real_channel_sd_median'] < 1e-12


def test_collective_amplitude_period_is_recovered_without_chaos_label(tmp_path):
    times = (np.arange(1000) + 1) * .1
    amplitude = 1 + .2 * np.sin(2 * np.pi * .05 * times)
    row = summarize(archive(tmp_path, amplitude))
    np.testing.assert_allclose(row['R_mean'], 1.)
    np.testing.assert_allclose(row['R_std'], .2 / np.sqrt(2))
    np.testing.assert_allclose(row['R_peak_frequency'], .05)
    assert 0 <= row['R_spectral_entropy'] <= 1
