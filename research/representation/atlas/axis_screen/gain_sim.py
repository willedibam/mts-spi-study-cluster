"""Synthetic check (post hoc, result.md "Preprocessing sensitivity"): does amplitude (gain) modulation move the MinCovDet
axis z(pcorr, pcorr_mcd) beyond IAAFT, on z-scored data (marginal-final IAAFT) and normal-scored data (spectrum-final)?

30 channels, T 1000, smoothed white noise mixed by a banded covariance; multiplied by exp(s * slow Gaussian) gain, either
one gain shared by all channels (an elliptical scale mixture) or an independent gain per channel. 4 seeds per case.

  python -m research.representation.atlas.axis_screen.gain_sim
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mtsspi.surrogates import multivariate_iaaft
from research.representation.atlas.axis_screen.mechanism import metrics
from research.representation.atlas.axis_screen.stats import zscore
from research.representation.atlas.axis_screen.run import _ns

T, M = 1000, 30


def gen(kind, s, rng):
    L = np.linalg.cholesky(.4 + .6 * np.eye(M) + .2 * np.diag(np.ones(M - 1), 1) + .2 * np.diag(np.ones(M - 1), -1))
    z = gaussian_filter1d(rng.standard_normal((T, M)), 2, axis=0) @ L.T
    if kind == 'none':
        return z
    shape = T if kind == 'shared' else (T, M)
    g = np.exp(s * zscore(gaussian_filter1d(rng.standard_normal(shape), 20, axis=0)))
    return z * (g[:, None] if kind == 'shared' else g)


if __name__ == '__main__':
    zp = lambda x: metrics(x)[0][7]
    for kind, s in (('none', 0), ('shared', .5), ('shared', .8), ('own', .5), ('own', .8)):
        R = np.array([[zp(zscore(x)), zp(multivariate_iaaft(zscore(x), rng, final='marginal')[0]),
                       zp(_ns(x)), zp(multivariate_iaaft(_ns(x), rng, final='spectrum')[0])]
                      for rng in map(np.random.default_rng, range(4)) for x in [gen(kind, s, rng)]]).mean(0)
        print(f'{kind:6s} s={s}: z-scored real {R[0]:.3f} IAAFT {R[1]:.3f} net {R[0] - R[1]:+.3f} | '
              f'normal-scored real {R[2]:.3f} IAAFT {R[3]:.3f} net {R[2] - R[3]:+.3f}')
