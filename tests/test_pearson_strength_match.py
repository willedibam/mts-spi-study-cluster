import numpy as np
from scripts.build_pearson_strength_match import normalize,attenuate,match_polarity,absolute_mean


def test_raw_noise_calibration_and_polarity_match_both_summaries():
    rng=np.random.default_rng(45)
    x=rng.normal(size=(16,1000))+.7*rng.normal(size=(1,1000))
    x=normalize(x);y,sigma=attenuate(x,rng.normal(size=x.shape),.07)
    assert sigma>0 and abs(absolute_mean(y)-.07)<1e-10
    final,sign=match_polarity(y,.005)
    np.testing.assert_allclose(final,normalize(y)*np.array(sign)[:,None])
    np.testing.assert_allclose(final.var(axis=1),1,atol=1e-12)
    mask=~np.eye(16,dtype=bool)
    np.testing.assert_allclose(abs(np.corrcoef(y)),abs(np.corrcoef(final)),atol=1e-12)
    assert abs(np.cov(final,bias=True)[mask].mean()-.005)<5e-5
    # Sensor polarity does not change each channel's autocorrelation.
    np.testing.assert_allclose((final[:,1:]*final[:,:-1]).mean(axis=1),
                              (normalize(y)[:,1:]*normalize(y)[:,:-1]).mean(axis=1))
