import numpy as np
import pytest

from src.neurotycho_pilot import preprocess, spectral_features, window_specs


def test_windows_respect_one_based_labels_and_margins():
    condition = dict(ConditionLabel=['AwakeEyesClosed-Start', 'AwakeEyesClosed-End'],
                     ConditionIndex=[100001, 500001], ConditionTime=[100., 500.])
    specs = window_specs(condition)
    assert len(specs) == 16
    assert specs[0]['start'] == 130000
    assert specs[-1]['start'] == 462000
    assert all(s['context_start'] >= 100000 and s['context_stop'] <= 500000 for s in specs)
    condition['ConditionTime'][0] += .001
    with pytest.raises(ValueError, match='timestamp'):
        window_specs(condition)


def test_bipolar_reference_cancellation_and_frequency_retention():
    t = np.arange(28000) / 1000
    raw = np.array([np.sin(2*np.pi*8*t), .2*np.sin(2*np.pi*4*t),
                    np.sin(2*np.pi*70*t), .2*np.sin(2*np.pi*6*t)])
    pairs = [[0, 1], [2, 3]]
    x, q = preprocess(raw, pairs)
    referenced, _ = preprocess(raw + 2*np.cos(2*np.pi*3*t), pairs)
    assert q['accepted'] and x.shape == (2, 2000) and x.dtype == np.float64
    np.testing.assert_allclose(x, referenced, atol=1e-7)
    f = np.fft.rfftfreq(2000, 1/250)
    peaks = f[np.abs(np.fft.rfft(x, axis=1)).argmax(1)]
    np.testing.assert_allclose(peaks, [8, 70])
    assert spectral_features(x).shape == (70,)
    assert np.isfinite(spectral_features(x)).all()
    raw[0, 11000:12000] = 0
    rejected, quality = preprocess(raw, pairs)
    assert rejected is None and not quality['accepted']
