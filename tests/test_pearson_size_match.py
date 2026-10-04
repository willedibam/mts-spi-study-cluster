import numpy as np
import pytest
from scripts.build_pearson_size_match import joint_match


@pytest.mark.parametrize('m,t', [(8, 100), (32, 1000)])
def test_two_summary_match_with_finite_unit_variance_channels(m, t):
    rng = np.random.default_rng(712)
    signal = rng.normal(size=(m, t)) + rng.normal(size=(1, t))
    components = dict(signal=signal, first=rng.normal(size=(m, t)),
                      second=rng.normal(size=(m, t)), kind='attenuation')
    a = .15
    sign_sum = 2*round(np.sqrt(2*m)/2)
    b = a*(sign_sum**2-m)/(m*(m-1))
    x, parameters = joint_match(components, a, b, instance=0)
    covariance = np.cov(x, bias=True)
    offdiagonal = covariance[~np.eye(m, dtype=bool)]
    assert np.isfinite(x).all() and x.shape == (m, t)
    np.testing.assert_allclose(x.mean(1), 0, atol=1e-12)
    np.testing.assert_allclose(np.diag(covariance), 1, atol=1e-12)
    np.testing.assert_allclose(offdiagonal.mean(), b, atol=1e-9)
    np.testing.assert_allclose(abs(offdiagonal).mean(), a, atol=1e-9)
    assert parameters['parameter'] > 0
