"""Scalar-gain test, synthetic calibration only (result.md, "Scalar-gain test: calibrated, not run").

H0: x(t) = a(t) y(t) + e(t), y a stationary carrier with covariance S, e an independent floor with covariance N, and
a(t) one gain for every channel. Then local covariance is a^2 S + N, so its fluctuations over time are rank one.
Three within-window statistics of that implication, on z-scored synthetic windows (T 1000, M 126, 100 Hz-like):

  strata  rank-one share of the deviations of K = 4 amplitude-strata covariances (strata by the smoothed
          across-channel power), real against the remodulation surrogate a_hat * IAAFT(x / a_hat), which is a scalar
          gain by construction with the window's own gain path
  block   cross-fitted rank-one share of 0.5-s block covariances on the top 8 principal components (each block's
          covariance from its two halves; noise cancels in the cross-product), real against the same surrogate
  env     cross-fitted second eigenvalue of the 0.5-s block powers of the top 8 principal components, less its value
          with blocks misaligned (noise reference), over the first eigenvalue

Generators: stationary; shared slow, burst and correlated-floor gain (H0); two sources with independent gains,
independent gain per channel, a global mode that appears with amplitude, and covariance switching at constant power.
Values are pooled over 10 windows per case (ratio of sums for block and env).

  python -m research.representation.atlas.axis_screen.gain [workers]
"""
import sys, zlib
import numpy as np
from multiprocessing import Pool
from scipy.ndimage import gaussian_filter1d

T, M, SIGMA, K, L, Q, NREP = 1000, 126, 25, 4, 50, 8, 10


def zs(x):
    return (x - x.mean(0)) / x.std(0)


def gain_path(x):
    return np.sqrt(gaussian_filter1d((x ** 2).mean(1), SIGMA, mode='nearest'))


def remod(x, rng):
    from mtsspi.surrogates import multivariate_iaaft
    a = gain_path(x)[:, None]
    return zs(a * multivariate_iaaft(x / a, rng, final='spectrum')[0])


def strata(x):
    a = gain_path(x); iu = np.triu_indices(M, 1)
    lab = np.empty(T, int); lab[np.argsort(a)] = np.arange(T) * K // T
    C = np.array([np.cov(x[lab == j].T)[iu] for j in range(K)])
    s = np.linalg.svd(C - C.mean(0), compute_uv=False) ** 2
    return s[0] / s.sum()


def _pcs(x):
    return x @ np.linalg.eigh(np.cov(x.T))[1][:, -Q:]


def _eig(G):
    return np.linalg.eigvalsh((G + G.T) / 2)[::-1]


def block(x):
    y = _pcs(x); iu = np.triu_indices(Q); nb, h = T // L, L // 2
    H = [np.array([(y[b * L + o:b * L + o + h].T @ y[b * L + o:b * L + o + h] / h)[iu] for b in range(nb)]) for o in (0, h)]
    H = [A - A.mean(0) for A in H]
    l = _eig(H[0] @ H[1].T)
    return l[0], l.sum()


def env(x, rng):
    y = _pcs(x); y /= y.std(0); nb, h = T // L, L // 2
    A, B = [np.array([(y[b * L + o:b * L + o + h] ** 2).mean(0) for b in range(nb)]) for o in (0, h)]
    A, B = A - A.mean(0), B - B.mean(0)
    lr = _eig(A.T @ B / nb)
    lp = np.mean([_eig(A.T @ B[rng.permutation(nb)] / nb)[1] for _ in range(20)])
    return lr[1] - lp, lr[0]


def _carrier(rng, W):
    z = gaussian_filter1d(rng.standard_normal((T, W.shape[1])), 2, axis=0) @ W.T
    return zs(z + .5 * gaussian_filter1d(rng.standard_normal((T, M)), 2, axis=0))


def _slow(rng, s, m=None):
    return np.exp(s * zs(gaussian_filter1d(rng.standard_normal(T if m is None else (T, m)), 30, axis=0)))


def _burst(rng, ratio, dwell=80):
    on, t, out = rng.random() < .5, 0, np.ones(T)
    while t < T:
        d = int(rng.exponential(dwell)) + 1
        out[t:t + d] = ratio if on else 1.
        t += d; on = not on
    return gaussian_filter1d(out, 3)


def _floor(rng, level, corr):
    e = gaussian_filter1d(rng.standard_normal((T, M)), 1, axis=0)
    if corr:
        e = e + gaussian_filter1d(rng.standard_normal((T, 3)), 1, axis=0) @ rng.standard_normal((3, M)) * .8
    return level * zs(e)


def generate(kind, rng, s):
    W = rng.standard_normal((M, 6)) / np.sqrt(6)
    y = _carrier(rng, W)
    if kind == 'stationary':
        return y + _floor(rng, .5, True)
    if kind == 'shared_slow':
        return _slow(rng, s)[:, None] * y + _floor(rng, .5, False)
    if kind == 'shared_slow_corrfloor':
        return _slow(rng, s)[:, None] * y + _floor(rng, .7, True)
    if kind == 'shared_burst':
        return _burst(rng, np.exp(2 * s))[:, None] * y + _floor(rng, .5, True)
    if kind == 'two_source':
        W2 = rng.standard_normal((M, 6)) / np.sqrt(6)
        return _slow(rng, s)[:, None] * y + _slow(rng, s)[:, None] * _carrier(rng, W2) + _floor(rng, .5, True)
    if kind == 'per_channel':
        return _slow(rng, s, m=M) * y + _floor(rng, .5, True)
    if kind == 'linked_mode':
        a = _slow(rng, s)
        u = gaussian_filter1d(rng.standard_normal(T), 2) * (a - a.min()) / (a.max() - a.min())
        return a[:, None] * y + 1.5 * zs(u[:, None] * np.ones(M)) * u.std() + _floor(rng, .5, True)
    if kind == 'switching':
        W2 = rng.standard_normal((M, 6)) / np.sqrt(6)
        return np.where((_burst(rng, 2.) > 1.5)[:, None], y, _carrier(rng, W2)) + _floor(rng, .5, True)
    raise ValueError(kind)


CASES = [('stationary', 0), ('shared_slow', .3), ('shared_slow', .6), ('shared_slow_corrfloor', .3),
         ('shared_burst', .3), ('shared_burst', .6), ('two_source', .3), ('two_source', .6), ('per_channel', .3),
         ('per_channel', .6), ('linked_mode', .3), ('linked_mode', .6), ('switching', 0)]


def one(args):
    kind, s, rep = args
    rng = np.random.default_rng([rep, zlib.crc32(kind.encode()), int(100 * s)])
    x = zs(generate(kind, rng, s)); u = remod(x, rng)
    return (strata(x), strata(u)) + block(x) + block(u) + env(x, rng)


if __name__ == '__main__':
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        print(f"{'generator':24s} {'s':>3s} | strata real  null   net | block real  null    net | env  (λ2-ref)/λ1 ± se")
        for kind, s in CASES:
            v = np.array(p.map(one, [(kind, s, r) for r in range(NREP)]))
            br, bn = v[:, 2].sum() / v[:, 3].sum(), v[:, 4].sum() / v[:, 5].sum()
            ev = v[:, 6].sum() / v[:, 7].sum(); se = np.std(v[:, 6] / v[:, 7]) / np.sqrt(NREP)
            print(f'{kind:24s} {s:3.1f} | {v[:, 0].mean():10.3f} {v[:, 1].mean():.3f} {v[:, 0].mean() - v[:, 1].mean():+.3f} | '
                  f'{br:10.3f} {bn:.3f} {br - bn:+.3f} | {ev:+.3f} ± {se:.3f}', flush=True)
