"""Generators for the character-axis validation study (2 November 2026).

Four families, each returning ``(x, truth)`` with ``x`` of shape (T, M), channels z-scored:

- ``edge_drivers`` (family A): every coupled edge e = (j, k) of a random d-regular graph has its own
  narrowband driver D_e. The leader reads D_e; the follower reads it delayed by
  delta_e = phi_e / (2 pi f_e) (fixed phase, not fixed delay), with a slow random warp W_e * omega_e(t),
  through g(u) = cos(theta_e) u + sin(theta_e) (u^2 - 1)/sqrt(2). Strength w_e, phase phi_e, band f_e,
  form theta_e and warp W_e are set per edge and independently. Nuisances: strength spread, private
  and measurement noise, driver bandwidth (autocorrelation), instantaneous spatial mixing, marginal sinh map.
- ``var_network`` (family B): VAR with self-AR and lag-tau coupling on a d-regular graph; each edge has
  total coupling b split by asymmetry alpha_e between its two directions.
- ``ggm_network`` (family C): innovations with a correlation matrix that mixes a direct (sparse
  precision) structure and an observed-hub structure (spoke-spoke edges are marginal only), filtered by
  a common AR(1) so the spatial structure holds at every lag.
- ``kuramoto_tree`` (family K, nonlinear): Kuramoto-Sakaguchi oscillators on a random tree with
  antisymmetric phase lags alpha_e (each edge locks near alpha_e); x = cos(theta) plus noise.

All randomness for the network (graph, weights, per-edge values) comes from ``net_seed``; the
time-series noise comes from ``noise_seed``, so two windows of one recording share their network.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy import signal

OVERSAMPLE = 8
PAD = 300


def _z(x):
    x = x - x.mean(0)
    return x / x.std(0)


def _lognormal(rng, n, cv):
    if cv <= 0:
        return np.ones(n)
    s2 = np.log1p(cv ** 2)
    return rng.lognormal(-s2 / 2, np.sqrt(s2), n)


def _balanced_signs(rng, n):
    s = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    return rng.permutation(s)


def per_edge(rng, n, centre, spread):
    """Values centre +/- spread/2 on a balanced random half of the edges (spread 0: all at centre)."""
    return centre + _balanced_signs(rng, n) * spread / 2


def _ar2(rng, n, f, r):
    """Unit-variance narrowband AR(2) with pole radius r at frequency f (cycles per sample)."""
    a1, a2 = 2 * r * np.cos(2 * np.pi * f), -r * r
    x = signal.lfilter([1.0], [1.0, -a1, -a2], rng.standard_normal(n + 500))[500:]
    return (x - x.mean()) / x.std()


def _smooth(rng, n, scale):
    """Unit-SD smooth Gaussian path with correlation time ~scale samples."""
    k = np.exp(-0.5 * (np.arange(-4 * scale, 4 * scale + 1) / scale) ** 2)
    w = np.convolve(rng.standard_normal(n + len(k)), k, mode='same')[len(k) // 2: len(k) // 2 + n]
    return (w - w.mean()) / w.std()


def _mixing_matrix(rng, M, mu, length=0.3):
    pos = rng.uniform(0, 1, (M, 2))
    d2 = ((pos[:, None] - pos[None]) ** 2).sum(-1)
    K = np.exp(-d2 / (2 * length ** 2))
    K /= K.sum(1, keepdims=True)
    return (1 - mu) * np.eye(M) + mu * K


def _finish(x, rng_mix, mix, beta):
    x = _z(x)
    if mix > 0:
        x = x @ _mixing_matrix(rng_mix, x.shape[1], mix).T
        x = _z(x)
    if beta > 0:
        x = _z(np.sinh(beta * x) / beta)
    return x


def edge_drivers(*, net_seed, noise_seed, M=16, T=2000, degree=3, cv_s=0.5,
                 phi=None, f=None, theta=None, warp=None, warp_scale=100,
                 r=0.9, private=0.5, noise=0.25, mix=0.0, beta=0.0):
    """Family A. Per-edge arrays (length E = M*degree/2) or scalars for phi, f, theta, warp."""
    net = np.random.default_rng(net_seed)
    g = nx.random_regular_graph(degree, M, seed=int(net.integers(2 ** 31)))
    edges = np.array(sorted(tuple(sorted(e)) for e in g.edges()))
    E = len(edges)
    vec = lambda v, d: np.full(E, d if v is None else v, float) if np.ndim(v) == 0 else np.asarray(v, float)
    phi, f, theta, warp = vec(phi, np.pi / 8), vec(f, 0.1), vec(theta, 0.0), vec(warp, 0.0)
    w = _lognormal(net, E, cv_s)
    lead = net.integers(0, 2, E)            # which end of the edge leads
    mix_seed = int(net.integers(2 ** 31))
    rng = np.random.default_rng(noise_seed)
    n = T + 2 * PAD
    fine_t = np.arange(n * OVERSAMPLE) / OVERSAMPLE
    x = np.zeros((T, M))
    for e, (a, b) in enumerate(edges):
        D = _ar2(rng, n * OVERSAMPLE, f[e] / OVERSAMPLE, r ** (1 / OVERSAMPLE))
        t = np.arange(PAD, PAD + T, dtype=float)
        lead_sig = np.interp(t, fine_t, D)
        delay = phi[e] / (2 * np.pi * f[e])
        shift = delay + (warp[e] * _smooth(rng, T, warp_scale) if warp[e] > 0 else 0.0)
        u = np.interp(t - shift, fine_t, D)
        u = (u - u.mean()) / u.std()
        foll = np.cos(theta[e]) * u + np.sin(theta[e]) * (u ** 2 - 1) / np.sqrt(2)
        j, k = (a, b) if lead[e] == 0 else (b, a)
        x[:, j] += np.sqrt(w[e]) * lead_sig
        x[:, k] += np.sqrt(w[e]) * foll
    for j in range(M):
        x[:, j] += np.sqrt(private) * _ar2(rng, T, 0.1, r)
    x = _z(x) + np.sqrt(noise) * rng.standard_normal((T, M))
    x = _finish(x, np.random.default_rng(mix_seed), mix, beta)
    truth = dict(edges=edges, w=w, phi=phi, f=f, theta=theta, warp=warp, lead=lead)
    return x, truth


def var_network(*, net_seed, noise_seed, M=16, T=2000, degree=3, b=0.3, alpha=0.0, a=0.6, tau=2,
                noise=0.25, cv_s=0.3):
    """Family B. alpha (scalar or per-edge, in [0, 1]) splits coupling b_e as b_e(1 +/- alpha)/2."""
    net = np.random.default_rng(net_seed)
    g = nx.random_regular_graph(degree, M, seed=int(net.integers(2 ** 31)))
    edges = np.array(sorted(tuple(sorted(e)) for e in g.edges()))
    E = len(edges)
    alpha = np.full(E, alpha, float) if np.ndim(alpha) == 0 else np.asarray(alpha, float)
    bw = b * _lognormal(net, E, cv_s)
    lead = net.integers(0, 2, E)
    B = np.zeros((M, M))                    # B[target, source]
    for e, (i, j) in enumerate(edges):
        s, t = (i, j) if lead[e] == 0 else (j, i)
        B[t, s] = bw[e] * (1 + alpha[e]) / 2
        B[s, t] = bw[e] * (1 - alpha[e]) / 2
    # companion stability check; shrink coupling if needed
    for _ in range(50):
        C = np.zeros((M * tau, M * tau))
        C[:M, :M] = a * np.eye(M)
        C[:M, M * (tau - 1):] += B
        C[M:, :-M] = np.eye(M * (tau - 1))
        if np.max(np.abs(np.linalg.eigvals(C))) < 0.95:
            break
        B *= 0.9
    rng = np.random.default_rng(noise_seed)
    n = T + 500
    e_ = rng.standard_normal((n, M))
    x = np.zeros((n, M))
    for t in range(tau, n):
        x[t] = a * x[t - 1] + B @ x[t - tau] + e_[t]
    x = _z(x[500:]) + np.sqrt(noise) * rng.standard_normal((T, M))
    return _z(x), dict(edges=edges, B=B, alpha=alpha, lead=lead)


def _corr_from_precision(Theta):
    S = np.linalg.inv(Theta)
    d = np.sqrt(np.diag(S))
    return S / np.outer(d, d)


def ggm_network(*, net_seed, noise_seed, M=16, T=2000, lam=0.0, a=0.6, noise=0.25, degree=3, hubs=3,
                rho=0.22, rho_hub=0.45):
    """Family C. lam mixes direct (lam=0) and hub-mediated (lam=1) correlation structure."""
    net = np.random.default_rng(net_seed)
    g = nx.random_regular_graph(degree, M, seed=int(net.integers(2 ** 31)))
    A = nx.to_numpy_array(g)
    S_dir = _corr_from_precision(np.eye(M) - rho * A)
    order = net.permutation(M)
    H = np.zeros((M, M))
    hub_ids = order[:hubs]
    for k, j in enumerate(order[hubs:]):
        h = hub_ids[k % hubs]
        H[h, j] = H[j, h] = 1
    Hn = H / np.sqrt(H.sum(0).max())
    off = ~np.eye(M, dtype=bool)
    target = np.abs(S_dir[off]).mean()
    lo, hi = 0.0, 0.999                     # match mean |r| in precision space (stays PD)
    for _ in range(40):
        mid = (lo + hi) / 2
        if np.abs(_corr_from_precision(np.eye(M) - mid * Hn)[off]).mean() < target:
            lo = mid
        else:
            hi = mid
    rho_hub = (lo + hi) / 2
    S_hub = _corr_from_precision(np.eye(M) - rho_hub * Hn)
    S = (1 - lam) * S_dir + lam * S_hub
    L = np.linalg.cholesky(S)
    rng = np.random.default_rng(noise_seed)
    n = T + 500
    e = rng.standard_normal((n, M)) @ L.T
    x = signal.lfilter([1.0], [1.0, -a], e, axis=0)[500:]
    x = _z(x) + np.sqrt(noise) * rng.standard_normal((T, M))
    return _z(x), dict(S=S, S_dir=S_dir, S_hub=S_hub, lam=lam)


def kuramoto_tree(*, net_seed, noise_seed, M=16, T=2000, f0=0.1, alpha=np.pi / 8, K=0.08, cv_s=0.3,
                  sigma_w=0.01, phase_noise=0.02, noise=0.25, substeps=10):
    """Family K. alpha (scalar or per-edge) is the antisymmetric Sakaguchi lag on each tree edge."""
    net = np.random.default_rng(net_seed)
    tree = nx.random_labeled_tree(M, seed=int(net.integers(2 ** 31)))
    edges = np.array(sorted(tuple(sorted(e)) for e in tree.edges()))
    E = len(edges)
    alpha = np.full(E, alpha, float) if np.ndim(alpha) == 0 else np.asarray(alpha, float)
    Kw = K * _lognormal(net, E, cv_s)
    lead = net.integers(0, 2, E)
    omega = 2 * np.pi * f0 + sigma_w * net.standard_normal(M)
    Kmat = np.zeros((M, M)); Amat = np.zeros((M, M))
    for e, (i, j) in enumerate(edges):
        s, t = (i, j) if lead[e] == 0 else (j, i)
        Kmat[s, t] = Kmat[t, s] = Kw[e]
        Amat[t, s] = alpha[e]               # target t locks to theta_s - theta_t near -alpha? see below
        Amat[s, t] = -alpha[e]
    rng = np.random.default_rng(noise_seed)
    dt = 1.0 / substeps
    th = rng.uniform(0, 2 * np.pi, M)
    n = T + 500
    out = np.zeros((n, M))
    for t in range(n):
        for _ in range(substeps):
            diff = th[None, :] - th[:, None] - Amat   # [i, j] = theta_j - theta_i - alpha_ij
            th = th + dt * (omega + (Kmat * np.sin(diff)).sum(1)) + np.sqrt(dt) * phase_noise * rng.standard_normal(M)
        out[t] = np.cos(th)
    x = _z(out[500:]) + np.sqrt(noise) * rng.standard_normal((T, M))
    return _z(x), dict(edges=edges, alpha=alpha, K=Kw, lead=lead)


def ft_surrogate(x, rng):
    """Multivariate Fourier surrogate: one random phase per frequency shared by all channels.

    Preserves every channel's periodogram and the sample cross-periodogram; randomises the rest.
    """
    T = x.shape[0]
    X = np.fft.rfft(x, axis=0)
    ph = rng.uniform(0, 2 * np.pi, X.shape[0])
    ph[0] = 0
    if T % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(X * np.exp(1j * ph)[:, None], n=T, axis=0)
