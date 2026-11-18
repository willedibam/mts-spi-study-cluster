"""Physics-only scout: delayed -> anticipated synchronization (DS -> AS) as a sign-of-lag transition.

Mechanism (Dalla Porta et al. 2019, Front. Syst. Neurosci. 13:41, Sec. 3.2 and Fig. 4): in a
unidirectionally coupled sender -> receiver system the sender-receiver lag decreases smoothly
through zero (DS -> AS) as the receiver's free-running frequency rises past the sender's. This
scout uses the minimal reduction of that mechanism, not the published neural-mass model: eight
independent modules of noisy Stuart-Landau oscillators S_k -> R_k (M = N = 16), receiver
detuning d = omega_R - omega_S as the control, Re Z observed with small sensor noise.

Q is the module-mean phase lag arg<Z_S conj(Z_R)> on a disjoint future window (> 0: receiver lags,
DS; < 0: receiver leads, AS). Probes are cheap numpy stand-ins, not pyspi: zero-lag Pearson
(symmetric), lagged Pearson corr(x_i(t), x_j(t+k)) for k = 2, 4 (directed) and bivariate linear
Granger i -> j (directed). Readouts follow the benchmark contract on fit seeds only: mean |r| over
ordered edges, standardized PC1 of per-probe ordered-edge means, centered PC1 of z over ordered edges.

Run: python -m research.inference.candidate_search.scout_anticipated_sync
"""
import json
from pathlib import Path
import numpy as np
from scipy.signal import hilbert
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/order-parameter-inference/anticipated-sync-scout-261118'
MODULES, T, STEP, DT = 8, 1000, .5, .01
BURN, FUTURE = 200., 500.
OMEGA_RANGE, K, NOISE, SENSOR = (.8, 1.2), .4, .05, .1
DETUNINGS = np.round(np.linspace(-.3, .3, 13), 3)
SEEDS, FIT_SEEDS, SEED = 8, 4, 261118
GC_ORDER = 4
PROBES = ('pearson0', 'lagsym2', 'plv', 'phase', 'granger', 'tlmi1')  # sym, sym, sym, antisym, directed, directed


def simulate(detunings=None, seeds=None, modules=MODULES, seed=None, dt=None, observed_q=False):
    """Vectorised Euler-Maruyama over (control, seed, module). Returns x (C,S,M,T), Q (C,S), lock (C,S), channel order
    (C,S,M), and with observed_q also Q on the observed window (stationarity check)."""
    detunings = DETUNINGS if detunings is None else detunings
    seeds = SEEDS if seeds is None else seeds
    seed = SEED if seed is None else seed
    DT = globals()['DT'] if dt is None else dt
    rng = np.random.default_rng(seed)
    shape = (len(detunings), seeds, modules)
    grid = np.linspace(*OMEGA_RANGE, modules)                   # distinct module rhythms, permuted per record
    ws = np.array([[rng.permutation(grid) for _ in range(seeds)] for _ in detunings])
    wr = ws + np.asarray(detunings)[:, None, None]              # receiver free-running frequency
    zs = np.exp(2j * np.pi * rng.uniform(size=shape)); zr = np.exp(2j * np.pi * rng.uniform(size=shape))
    every, nb, nr = round(STEP / DT), round(BURN / DT), round(T * STEP / DT)
    nf = round(FUTURE / DT)
    x = np.empty((T,) + shape[:2] + (2 * modules,)); acc = np.zeros(shape, complex); obs = np.zeros(shape, complex); i = 0
    amp = NOISE * np.sqrt(DT)
    for k in range(nb + nr + nf):
        ns = amp * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
        nr_ = amp * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
        dzs = (1 + 1j * ws - abs(zs) ** 2) * zs
        dzr = (1 + 1j * wr - abs(zr) ** 2) * zr + K * (zs - zr)
        zs, zr = zs + DT * dzs + ns, zr + DT * dzr + nr_
        if nb <= k < nb + nr and (k - nb + 1) % every == 0:
            x[i] = np.concatenate([zs.real, zr.real], -1); i += 1
        if nb <= k < nb + nr:
            obs += zs * np.conj(zr) / (abs(zs) * abs(zr))
        elif k >= nb + nr:
            acc += zs * np.conj(zr) / (abs(zs) * abs(zr))
    acc /= nf
    q = np.angle(acc).mean(-1)                                   # module-mean signed lag (rad)
    lock = abs(acc).min(-1)                                      # weakest module's phase-locking index
    x = np.moveaxis(x, 0, 2)
    x = x + SENSOR * x.std(2, keepdims=True) * np.random.default_rng(seed + 1).normal(size=x.shape)
    order = np.array([[rng.permutation(2 * modules) for _ in range(seeds)] for _ in detunings])  # channel order carries no role
    return (x, q, lock, order, np.angle(obs).mean(-1)) if observed_q else (x, q, lock, order)


def granger(x, p=GC_ORDER):
    """Bivariate linear Granger log-variance ratio for every ordered pair i -> j."""
    t, m = x.shape
    lagged = np.stack([x[p - l - 1:t - l - 1] for l in range(p)], 1)  # (t-p, p, m)
    g = np.zeros((m, m))
    for j in range(m):
        y = x[p:, j]; own = np.c_[np.ones(t - p), lagged[:, :, j]]
        r0 = y - own @ np.linalg.lstsq(own, y, rcond=None)[0]
        for i in range(m):
            if i == j: continue
            full = np.c_[own, lagged[:, :, i]]
            r1 = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
            g[i, j] = np.log(r0.var() / r1.var())
    return g


def probes(x):
    """Matrices v[i, j]; directed ones read i -> j. pyspi-like: lagged correlation is symmetrised,
    analytic-signal phase is antisymmetric (> 0: i leads j), Granger and lag-1 Gaussian TLMI are directed."""
    x = (x - x.mean(0)) / x.std(0); n = len(x)
    h = hilbert(x, axis=0); h = h / abs(h)
    cross = h.T @ h.conj() / n
    lag1 = x[:-1].T @ x[1:] / (n - 1); lag2 = x[:-2].T @ x[2:] / (n - 2)
    return {'pearson0': np.corrcoef(x.T), 'lagsym2': (lag2 + lag2.T) / 2, 'plv': abs(cross),
            'phase': np.angle(cross), 'granger': granger(x), 'tlmi1': -.5 * np.log(1 - lag1 ** 2)}


def features(mats, order):
    m = len(mats['pearson0']); off = ~np.eye(m, dtype=bool)
    v = np.array([mats[p][off] for p in PROBES])
    iu = np.triu_indices(len(PROBES), 1)
    pos = np.argsort(order)  # pos[c]: displayed index of physical channel c; senders are 0..MODULES-1
    anti = {p: np.mean([mats[p][pos[s], pos[s + MODULES]] - mats[p][pos[s + MODULES], pos[s]] for s in range(MODULES)]) for p in PROBES}
    return dict(mean_abs_r=abs(mats['pearson0'][off]).mean(), means=v.mean(1), z=np.corrcoef(v)[iu], anti=anti)


def pc1(train, test, standardize):
    mu, sd = train.mean(0), (train.std(0) if standardize else np.ones(train.shape[1]))
    sd[sd == 0] = 1
    w = np.linalg.svd((train - mu) / sd, full_matrices=False)[2][0]
    return (test - mu) / sd @ w


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, q, lock, order = simulate()
    C, S = q.shape
    feats = [[features(probes(x[c, s][:, order[c, s]]), order[c, s]) for s in range(S)] for c in range(C)]
    fit = np.zeros((C, S), bool); fit[:, :FIT_SEEDS] = True; held = ~fit
    flat = lambda key: np.array([[f[key] for f in row] for row in feats])
    means, z, mabs = flat('means'), flat('z'), flat('mean_abs_r')
    scores = {'mean_abs_r': mabs, 'mean_probe_pc1': pc1(means[fit], means.reshape(C * S, -1), True).reshape(C, S),
              'z_pc1_centered': pc1(z[fit], z.reshape(C * S, -1), False).reshape(C, S)}
    res = {'controls': np.asarray(DETUNINGS).tolist(), 'Q_mean': q.mean(1).tolist(), 'Q_sd': q.std(1).tolist(),
           'lock_min': float(lock.min()), 'held_abs_spearman_Q': {}, 'held_abs_spearman_absQ': {},
           'z_names': [f'{PROBES[i]}|{PROBES[j]}' for i, j in zip(*np.triu_indices(len(PROBES), 1))],
           'z_control_means': z.mean(1).round(3).tolist(), 'probe_means_control': means.mean(1).round(4).tolist(),
           'module_antisymmetry_control': {p: np.mean([[f['anti'][p] for f in row] for row in feats], 1).round(4).tolist() for p in PROBES}}
    for name, s in scores.items():
        res['held_abs_spearman_Q'][name] = round(abs(spearmanr(s[held], q[held])[0]), 3)
        res['held_abs_spearman_absQ'][name] = round(abs(spearmanr(s[held], abs(q[held]))[0]), 3)
    for i, n in enumerate(res['z_names']):
        res['held_abs_spearman_Q']['z:' + n] = round(abs(spearmanr(z[..., i][held], q[held])[0]), 3)
    zc = z[fit] - z[fit].mean(0); ev = np.linalg.svd(zc, compute_uv=False) ** 2
    res['z_evr'] = (ev / ev.sum()).round(3).tolist()[:4]
    res['z_pc1_loadings'] = dict(zip(res['z_names'], np.linalg.svd(zc, full_matrices=False)[2][0].round(3).tolist()))
    for i, n in enumerate(PROBES):
        res['held_abs_spearman_Q']['mean:' + n] = round(abs(spearmanr(means[..., i][held], q[held])[0]), 3)
    (OUT / 'scout.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
