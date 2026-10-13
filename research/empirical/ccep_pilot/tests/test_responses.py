import numpy as np
from numpy.testing import assert_allclose
from research.empirical.ccep_pilot.responses import latency_ms,site_key


def test_published_sample_numbers_are_one_based():
    time=np.arange(1,10241)/2048-2
    values=latency_ms(np.array([4096,4156,np.nan,0,10241]),time)
    assert_allclose(values[:2],[0,29.296875])
    assert np.isnan(values[2:]).all()


def test_polarity_uses_the_same_stimulation_contact_set():
    assert site_key('FC01-FC02')==site_key('FC02-FC01')
