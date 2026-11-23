"""Per-edge cross-cumulant decomposition of the MI pair's non-Gaussian excess, on a normal-scored (T, M) window.

For each pair the two channels are standardised and whitened by their correlation (w = L^-1 [x_i, x_j], Sigma = L L').
Weak non-Gaussianity gives negentropy J ~ (1/12) sum kappa3(w)^2 + (1/48) sum kappa4(w)^2 over all index tuples (Edgeworth /
Gram-Charlier; Comon 1994). With normal-scored marginals I - G = J(joint) - J(x_i) - J(x_j) ~ J(joint), so
  J3  third-order term: sign-asymmetric joint structure (curved coupling, regimes that shift the mean); zero for any
      mixture of zero-mean Gaussians.
  J4  fourth-order term: symmetric co-modulation of magnitudes (gain or coupling mixtures, decoupling episodes, heavy joint tails).
Also the envelope readout: p = corr(|a_i|^2, |a_j|^2) of the analytic signals against its Gaussian value |gamma|^2 (Isserlis for
circular complex Gaussians), gamma the normalised complex correlation of the analytic signals.
"""
import numpy as np
from scipy.signal import hilbert

M3 = (1, 3, 3, 1)            # multiplicities of kappa_111, 112, 122, 222
M4 = (1, 4, 6, 4, 1)         # of kappa_1111, 1112, 1122, 1222, 2222


def edge_cum(x):
    """Return dict of per-edge arrays (i < j): r, G, J3, J4, env (= p - |gamma|^2), p, g2."""
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape; iu, ju = np.triu_indices(M, 1)
    r = (x.T @ x / T)[iu, ju]; G = -.5 * np.log(1 - r ** 2)
    a, b = x[:, iu], x[:, ju]                                   # (T, E)
    w1 = a; w2 = (b - r * a) / np.sqrt(1 - r ** 2)              # Cholesky whitening, unit covariance
    m = lambda p, q: np.mean(w1 ** p * w2 ** q, 0)
    k3 = [m(3, 0), m(2, 1), m(1, 2), m(0, 3)]
    m11, m20, m02 = m(1, 1), m(2, 0), m(0, 2)                   # ~0, 1, 1 by construction (sample-exact up to rounding)
    k4 = [m(4, 0) - 3 * m20 ** 2, m(3, 1) - 3 * m20 * m11, m(2, 2) - m20 * m02 - 2 * m11 ** 2,
          m(1, 3) - 3 * m02 * m11, m(0, 4) - 3 * m02 ** 2]
    J3 = sum(c * k ** 2 for c, k in zip(M3, k3)) / 12; J4 = sum(c * k ** 2 for c, k in zip(M4, k4)) / 48
    h = hilbert(x, axis=0); e = np.abs(h) ** 2; e = (e - e.mean(0)) / e.std(0)
    p = (e.T @ e / T)[iu, ju]
    hc = h / np.sqrt(np.mean(np.abs(h) ** 2, 0)); g2 = np.abs((hc.conj().T @ hc / T)[iu, ju]) ** 2
    return dict(r=r, G=G, J3=J3, J4=J4, env=p - g2, p=p, g2=g2)
