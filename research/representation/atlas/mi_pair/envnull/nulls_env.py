"""Gain-over-floor null for the MI pair: a model-based (typical-realisation) surrogate whose H0 is time-varying gain.

H0 (GOF): x(t) = D_g(t) s(t) + n(t), applied to the raw, standardised (T, M) window x, where
  s  stationary linear-Gaussian, arbitrary spatial covariance C (the coupled signal, private parts allowed),
  g  one slow positive gain per channel (shared, per-channel, or both),
  n  stationary Gaussian floor that the gain does not scale; covariance F = diag(phi) (floor='diag') or diag(phi) + lam lam^T
     (floor='cm': one common-mode factor, e.g. a reference or volume-conducted component).
So the local covariance is Sigma(t) = D_g(t) C D_g(t) + F: a channel's coupling may only rise with its own gain (more signal over
the same floor). Local decoupling (private power rising while shared coupling falls) and static nonlinear coupling are outside H0.

Fit: local covariances K(t) = Gaussian-smoothed x_i x_j (sigma samples) on a grid of step sigma/2; alternating least squares
for C, F, g. Gains fitted on a stationary Gaussian process fluctuate from estimation noise alone; that noise would inject a
spurious scale mixture into the surrogate. So log g is split into a shared mode (channel mean) and per-channel deviations, and
each is shrunk towards its mean by kappa = max(0, 1 - noise var / total var) (robust MAD variances), the noise variance the
mean over fits of the same model to three stationary joint phase-randomised copies of x (empirical-Bayes deconvolution of the gain).
C is then moment-matched so the surrogate's time-averaged covariance equals x's (refit_CF).
Generation: Z, Z' independent joint phase randomisations of x (x's time-averaged cross-spectra, common random phases), spatially
whitened; s = Z C^1/2, n = Z' F^1/2; y = g(t) * s + n, then normal-scored as the real window is.
The spatial covariance C is then adjusted (MATCH_ITERS fixed-point steps, same Z, Z' and gains) until the normal-scored surrogate's
correlation matrix matches the normal-scored window's, so the Gaussian-MI profile is (nearly) the real one, as under N0.
arm='decor' (descriptive only): each channel's fitted gain trajectory is circularly shifted by its own random offset, which
breaks gain co-modulation across channels and keeps each channel's own gain dynamics.

Not a constrained realisation (Theiler and Prichard 1996): nothing guarantees size except the synthetic calibration (calib.py).
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, rankdata

MATCH_ITERS = 20
ns = lambda x: norm.ppf((np.apply_along_axis(rankdata, 0, x) - .5) / len(x))


def psd_sqrt(A, floor=0.):
    w, V = np.linalg.eigh((A + A.T) / 2); return (V * np.sqrt(np.maximum(w, floor))) @ V.T


def local_cov(x, sigma):
    T, M = x.shape; step = max(1, sigma // 2); tg = np.arange(step // 2, T, step)
    P = gaussian_filter1d(x[:, :, None] * x[:, None, :], sigma, axis=0, mode='reflect')
    return tg, P[tg]


def fit(x, sigma, floor, iters=40):
    """Fit K(t_b) ~ D_b C D_b + F. Returns grid, g (B, M), C, F."""
    tg, K = local_cov(x, sigma); B, M, _ = K.shape
    g = np.sqrt(np.maximum(np.einsum('bii->bi', K), 1e-6)); C = (K / (g[:, :, None] * g[:, None, :])).mean(0); F = np.zeros((M, M))
    for _ in range(iters):
        A = K - F
        for _ in range(3):  # per-block gain update, fixed point of the least-squares normal equations
            num = np.einsum('bij,bj,ij->bi', A, g, C); den = np.einsum('bj,ij->bi', g ** 2, C ** 2)
            g = np.sqrt(g * np.maximum(num / np.maximum(den, 1e-12), .05 * g.mean(0)))  # damped; floored at 5% of the channel's mean gain
        s = np.sqrt((g ** 2).mean(0)); g = g / s
        GG = g[:, :, None] * g[:, None, :]
        C = np.einsum('bij,bij->ij', GG, K - F) / np.einsum('bij->ij', GG ** 2)
        R = (K - GG * C).mean(0)
        if floor == 'diag':
            F = np.diag(np.maximum(np.diag(R), 0))
        else:  # diag + rank-1 common mode: principal factor iteration on R
            lam = np.sqrt(max(np.linalg.eigvalsh(R)[-1], 0)) * np.linalg.eigh(R)[1][:, -1]
            for _ in range(10):
                Rd = R.copy(); np.fill_diagonal(Rd, lam ** 2); w, V = np.linalg.eigh(Rd); lam = np.sqrt(max(w[-1], 0)) * V[:, -1]
            F = np.outer(lam, lam); F[np.diag_indices(M)] = np.maximum(np.diag(R), lam ** 2)
    return tg, g, C, F


def phase_rand(x, rng):
    T = len(x); X = np.fft.rfft(x, axis=0); th = rng.uniform(0, 2 * np.pi, len(X)); th[0] = 0
    if T % 2 == 0: th[-1] = 0
    return np.fft.irfft(X * np.exp(1j * th)[:, None], n=T, axis=0)


rvar = lambda a, ax=0: (1.4826 * np.median(np.abs(a - np.median(a, ax, keepdims=True)), ax)) ** 2  # robust (MAD) variance


def shrink(g, g0s):
    """Shrink log-gain modes of g towards their means; noise variance = mean over the stationary copies' fits g0s."""
    L = np.log(g); m = L.mean(1, keepdims=True); dev = L - m
    n_sh = np.mean([rvar(np.log(q).mean(1)) for q in g0s]); n_ch = np.mean([rvar(np.log(q) - np.log(q).mean(1, keepdims=True)) for q in g0s], 0)
    k_sh = max(0., 1 - n_sh / max(rvar(m[:, 0]), 1e-12)); k_ch = np.maximum(0., 1 - n_ch / np.maximum(rvar(dev), 1e-12))
    Ls = m.mean() + k_sh * (m - m.mean()) + dev.mean(0) + k_ch * (dev - dev.mean(0))
    gs = np.exp(Ls); return gs / np.sqrt((gs ** 2).mean(0)), k_sh, k_ch


def refit_CF(x, sigma, floor, g, tg):
    """F for fixed gains (after shrinkage); then C moment-matched so that the surrogate's time-averaged covariance equals x's:
    C_ij = (cov_ij - F_ij) / mean_t g_i g_j, projected to PSD."""
    _, K = local_cov(x, sigma); M = x.shape[1]; GG = g[:, :, None] * g[:, None, :]; F = np.zeros((M, M))
    for _ in range(20):
        C = np.einsum('bij,bij->ij', GG, K - F) / np.einsum('bij->ij', GG ** 2); R = (K - GG * C).mean(0)
        if floor == 'diag': F = np.diag(np.maximum(np.diag(R), 0))
        else:
            Rd = R.copy(); lam = np.zeros(M)
            for _ in range(10):
                np.fill_diagonal(Rd, lam ** 2); w, V = np.linalg.eigh(Rd); lam = np.sqrt(max(w[-1], 0)) * V[:, -1]
            F = np.outer(lam, lam); F[np.diag_indices(M)] = np.maximum(np.diag(R), lam ** 2)
    F[np.diag_indices(M)] = np.minimum(np.diag(F), .95)  # the floor never takes a whole channel
    C = (np.cov(x.T, bias=True) - F) / GG.mean(0); w, V = np.linalg.eigh((C + C.T) / 2)
    return (V * np.maximum(w, 0)) @ V.T, F


def gof(x, rng, sigma, floor='cm', arm='keep', diag=None):
    """One GOF surrogate of the raw window x (T, M); returns it normal-scored. diag: optional dict to receive fit details."""
    x = (x - x.mean(0)) / x.std(0); T, M = x.shape
    tg, g, C, F = fit(x, sigma, floor)
    g0s = [fit(phase_rand(x, rng), sigma, floor)[1] for _ in range(3)]
    g, k_sh, k_ch = shrink(g, g0s); C, F = refit_CF(x, sigma, floor, g, tg)
    gt = np.column_stack([np.interp(np.arange(T), tg, g[:, i]) for i in range(M)])
    if arm == 'decor': gt = np.column_stack([np.roll(gt[:, i], rng.integers(T)) for i in range(M)])
    W = np.linalg.inv(psd_sqrt(np.cov(x.T), 1e-8)); Zs, Zn = phase_rand(x, rng) @ W, phase_rand(x, rng) @ W
    n = Zn @ psd_sqrt(F); Rx = np.corrcoef(ns(x).T); off = ~np.eye(M, dtype=bool); GGm = (gt[:, :, None] * gt[:, None, :]).mean(0)
    for _ in range(MATCH_ITERS):  # match the normal-scored correlation matrix (hence the Gaussian-MI profile), as N0 does exactly
        y = ns(gt * (Zs @ psd_sqrt(C)) + n); E = np.where(off, np.corrcoef(y.T) - Rx, 0)
        C = C - E / GGm; w, V = np.linalg.eigh((C + C.T) / 2); C = (V * np.maximum(w, 0)) @ V.T
    y = ns(gt * (Zs @ psd_sqrt(C)) + n)
    if diag is not None: diag.update(k_sh=k_sh, k_ch=k_ch, g=g, tg=tg, C=C, F=F, corr_err=float(np.abs(np.corrcoef(y.T) - Rx)[off].max()))
    return y
