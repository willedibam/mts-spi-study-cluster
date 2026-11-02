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


def _edge_signal(rng, T, f, r, phi, warp, warp_scale, theta):
    """One edge's driver as its leader reads it and as its follower reads it (delayed, warped, through g)."""
    n = T + 2 * PAD
    fine_t = np.arange(n * OVERSAMPLE) / OVERSAMPLE
    D = _ar2(rng, n * OVERSAMPLE, f / OVERSAMPLE, r ** (1 / OVERSAMPLE))
    t = np.arange(PAD, PAD + T, dtype=float)
    lead_sig = np.interp(t, fine_t, D)
    delay = phi / (2 * np.pi * f)
    shift = delay + (warp * _smooth(rng, T, warp_scale) if warp > 0 else 0.0)
    u = np.interp(t - shift, fine_t, D)
    u = (u - u.mean()) / u.std()
    return lead_sig, np.cos(theta) * u + np.sin(theta) * (u ** 2 - 1) / np.sqrt(2)


def edge_drivers(*, net_seed, noise_seed, M=16, T=2000, degree=3, cv_s=0.5,
                 phi=None, f=None, theta=None, warp=None, warp_scale=100,
                 r=0.9, private=0.5, noise=0.25, mix=0.0, beta=0.0, alpha=None,
                 f_private=0.1, burst=None, burst_scale=25):
    """Family A. Per-edge arrays (length E = M*degree/2) or scalars for phi, f, theta, warp.

    alpha (stage 2, direction): each edge also gets an independent driver read the other way, and
    alpha_e splits the edge's variance (1 +/- alpha_e)/2 between the two directions (1: one-way, as
    with alpha=None; 0: symmetric). None leaves the generator exactly as in stage 1.

    Stage 3. f_private: frequency of each channel's private AR(2) component (0.1 before stage 3).
    burst (per-edge log-SD sigma_e, or None): both ends of edge e read it through one shared envelope
    a_e(t) proportional to exp(sigma_e v_e(t)), v_e a unit-SD smooth Gaussian path (correlation time
    ~burst_scale samples), scaled so mean(a_e^2) = 1 in the window. Coupling then concentrates in joint
    high-amplitude episodes at unchanged edge variance. Envelopes use their own random stream, so every
    other draw matches the run without bursts, and sigma_e = 0 reproduces it exactly.
    """
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
    if burst is not None:
        burst, brng = vec(burst, 0.0), np.random.default_rng([noise_seed, 3])
    x = np.zeros((T, M))
    for e, (a, b) in enumerate(edges):
        lead_sig, foll = _edge_signal(rng, T, f[e], r, phi[e], warp[e], warp_scale, theta[e])
        if burst is not None:
            env = np.exp(burst[e] * _smooth(brng, T, burst_scale))
            env /= np.sqrt(np.mean(env ** 2))
            lead_sig, foll = env * lead_sig, env * foll
        j, k = (a, b) if lead[e] == 0 else (b, a)
        if alpha is None:
            x[:, j] += np.sqrt(w[e]) * lead_sig
            x[:, k] += np.sqrt(w[e]) * foll
        else:
            back_lead, back_foll = _edge_signal(rng, T, f[e], r, phi[e], warp[e], warp_scale, theta[e])
            p = (1 + alpha[e]) / 2
            x[:, j] += np.sqrt(w[e] * p) * lead_sig + np.sqrt(w[e] * (1 - p)) * back_foll
            x[:, k] += np.sqrt(w[e] * p) * foll + np.sqrt(w[e] * (1 - p)) * back_lead
    for j in range(M):
        x[:, j] += np.sqrt(private) * _ar2(rng, T, f_private, r)
    x = _z(x) + np.sqrt(noise) * rng.standard_normal((T, M))
    x = _finish(x, np.random.default_rng(mix_seed), mix, beta)
    truth = dict(edges=edges, w=w, phi=phi, f=f, theta=theta, warp=warp, lead=lead)
    if alpha is not None:
        truth['alpha'] = np.asarray(alpha, float)
    if burst is not None:
        truth['burst'] = burst
    if f_private != 0.1:
        truth['f_private'] = float(f_private)
    return x, truth


def _mean_abs_r(y):
    c = np.corrcoef(y.T)
    return float(np.abs(c[~np.eye(len(c), dtype=bool)]).mean())


def hub_drivers(*, net_seed, noise_seed, lam=0.0, hubs=2, M=16, T=2000, degree=3, cv_s=0.5, phi=None,
                r=0.9, private=0.5, noise=0.25, mix=0.0, beta=0.0):
    """Family A, stage 2 (mediation). lam mixes family A's direct edges (lam=0: the edge_drivers graph,
    weights and phases at f=0.1) with observed hubs (lam=1) that each broadcast one driver to their spokes,
    so spoke-spoke dependence is mediated by an observed channel. A gain on the hub structure, found by
    bisection, holds the mean |r| of the source signal (before measurement noise, mixing and the marginal
    map) at its lam=0 value, as family C holds mean |r| fixed.
    """
    net = np.random.default_rng(net_seed)
    g = nx.random_regular_graph(degree, M, seed=int(net.integers(2 ** 31)))
    edges = np.array(sorted(tuple(sorted(e)) for e in g.edges()))
    E = len(edges)
    phi = np.full(E, np.pi / 8 if phi is None else phi, float) if np.ndim(phi) == 0 else np.asarray(phi, float)
    w = _lognormal(net, E, cv_s)
    lead = net.integers(0, 2, E)
    mix_seed = int(net.integers(2 ** 31))
    order = net.permutation(M)
    hub_ids, spokes = order[:hubs], order[hubs:]
    S = len(spokes)
    w_hub = _lognormal(net, S, cv_s)
    phi_hub = np.pi / 8 + _balanced_signs(net, S) * np.pi / 16
    rng = np.random.default_rng(noise_seed)
    direct, hub = np.zeros((T, M)), np.zeros((T, M))
    for e, (a, b) in enumerate(edges):
        lead_sig, foll = _edge_signal(rng, T, 0.1, r, phi[e], 0.0, 100, 0.0)
        j, k = (a, b) if lead[e] == 0 else (b, a)
        direct[:, j] += np.sqrt(w[e]) * lead_sig
        direct[:, k] += np.sqrt(w[e]) * foll
    n = T + 2 * PAD
    fine_t = np.arange(n * OVERSAMPLE) / OVERSAMPLE
    t = np.arange(PAD, PAD + T, dtype=float)
    for q, h_id in enumerate(hub_ids):
        D = _ar2(rng, n * OVERSAMPLE, 0.1 / OVERSAMPLE, r ** (1 / OVERSAMPLE))
        mine = [k for k in range(S) if k % hubs == q]
        hub[:, h_id] += np.sqrt(w_hub[mine].mean()) * np.interp(t, fine_t, D)
        for k in mine:
            u = np.interp(t - phi_hub[k] / (2 * np.pi * 0.1), fine_t, D)
            hub[:, spokes[k]] += np.sqrt(w_hub[k]) * (u - u.mean()) / u.std()
    priv = np.stack([np.sqrt(private) * _ar2(rng, T, 0.1, r) for _ in range(M)], 1)
    meas = rng.standard_normal((T, M))
    source = lambda gain: np.sqrt(1 - lam) * direct + np.sqrt(lam * gain) * hub + priv
    target = _mean_abs_r(direct + priv)
    gain = 1.0
    if lam > 0:
        lo, hi = 0.0, 1.0
        while _mean_abs_r(source(hi)) < target and hi < 1e4:
            hi *= 2
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _mean_abs_r(source(mid)) < target else (lo, mid)
        gain = (lo + hi) / 2
    src = source(gain)
    x = _z(src) + np.sqrt(noise) * meas
    x = _finish(x, np.random.default_rng(mix_seed), mix, beta)
    return x, dict(edges=edges, w=w, phi=phi, lead=lead, hubs=hub_ids, spokes=spokes, w_hub=w_hub,
                   phi_hub=phi_hub, lam=lam, gain=gain, target=target, achieved=_mean_abs_r(src))


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
                  sigma_w=0.01, phase_noise=0.02, noise=0.25, substeps=10, asym=None):
    """Family K. alpha (scalar or per-edge) is the antisymmetric Sakaguchi lag on each tree edge.

    asym (stage 2, direction; per-edge in [0, 1]): the leader pulls the follower with K_e(1 + asym_e)
    and is pulled back with K_e(1 - asym_e). None (or 0) is the symmetric stage-1 family.
    """
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
        if asym is not None:                # row = the pulled oscillator, so [t, s] is s pulling t
            a_e = float(np.broadcast_to(asym, (E,))[e])
            Kmat[t, s], Kmat[s, t] = Kw[e] * (1 + a_e), Kw[e] * (1 - a_e)
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
