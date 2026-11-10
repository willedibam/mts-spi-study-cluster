"""N_MI under four levels of measurement noise, on every system it has been tried on (exploratory, not pre-registered).

    python -m research.estimators.nmi_noise [--out results/estimators/nmi-noise] [--reps 20] [--workers 8]
    python -m research.estimators.nmi_noise --out <dir> --run-only     # cluster: rows and summary only
    python -m research.estimators.nmi_noise --out <dir> --figure-only  # locally, from <dir>/summary.csv

Each system's own y is standardised and Gaussian measurement noise of SD sigma is added: y_obs = y / sd(y) + sigma * e,
sigma in {1/4, 1/2, 1, 2} (signal-to-noise variance ratio 16, 4, 1, 1/4). Static systems are defined without noise, so
sigma is all of it; dynamical systems keep their own dynamics and process noise.

Reference lines:
- long run: the same pipeline at T = 2^16 (3 seeds, no shifts), at each sigma;
- low-noise-limit: as sigma grows, I -> Var m(x) / 2 sigma^2 and I_Gauss -> Cov(x, y)^2 / (2 Var x sigma^2), so
  N_MI -> 1 - Corr(x, m(x))^2 = 1 - r^2 / eta^2, with m(x) = E[y | x] (small-SNR expansion of the mutual information,
  e.g. the I-MMSE relation of Guo, Shamai and Verdu 2005; cited from memory). Drawn for static systems.
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
import os
from pathlib import Path

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mtsspi.figures import save, style
from research.estimators.estimators import nonlinearity_mi
from research.estimators.systems import make

HERE = Path(__file__).parent
SIGMAS = (.25, .5, 1., 2.)
T, LONG_T, LONG_SEEDS = 2000, 2 ** 16, 3
he2 = lambda x: (x ** 2 - 1) / np.sqrt(2)
g = lambda rng, n: rng.standard_normal(n)


# ---- static systems: (source, y given x). m(x) = E[y | x] for the low-noise limit ----------------------------------

def gauss(rng, n): return g(rng, n)
def lognormal(rng, n): return np.exp(g(rng, n))
def uniform(rng, n): return rng.uniform(-1, 1, n)

CENTRES = np.array([[-1., 2.], [0., -1.], [2., 1.]])   # the three clusters, sd 0.15


def clusters(rng, n):
    k = rng.integers(0, 3, n)
    return CENTRES[k, 0] + .15 * g(rng, n), CENTRES[k, 1] + .15 * g(rng, n)


def cluster_mean(x):
    w = np.exp(-(x[:, None] - CENTRES[:, 0]) ** 2 / (2 * .15 ** 2))
    return w @ CENTRES[:, 1] / w.sum(1)


def circle(rng, n):
    t = rng.uniform(0, 2 * np.pi, n)
    return np.cos(t) + .1 * g(rng, n), np.sin(t)


def outliers(rng, x):
    y = x.copy()
    i = rng.random(len(x)) < .01
    y[i] += 15 * g(rng, i.sum())
    return y


# name, group, source, y(rng, x), m(x) or None (then from the joint draw)
STATIC = [
    ('linear', 'linear', gauss, lambda r, x: x, lambda x: x),
    ('linear, lognormal source', 'linear', lognormal, lambda r, x: x, lambda x: x),
    ('linear, uniform source', 'linear', uniform, lambda r, x: x, lambda x: x),
    ('linear, 1% gross outliers', 'linear', gauss, outliers, lambda x: x),
    ('linear, noise SD $\\propto |x|$', 'linear', gauss, lambda r, x: x + .7 * np.abs(x) * g(r, len(x)), lambda x: x),
    ('$x + 0.3x^2$', 'bent', gauss, lambda r, x: x + .3 * x ** 2, lambda x: x + .3 * x ** 2),
    ('repo linearity, $\\theta = 0.5$', 'bent', gauss, lambda r, x: np.sqrt(.5) * (x + he2(x)), lambda x: x + he2(x)),
    ('$\\tanh 2x$', 'bent', gauss, lambda r, x: np.tanh(2 * x), lambda x: np.tanh(2 * x)),
    ('$x^3$', 'bent', gauss, lambda r, x: x ** 3, lambda x: x ** 3),
    ('$e^x$', 'bent', gauss, lambda r, x: np.exp(x), np.exp),
    ('sign $x$', 'bent', gauss, lambda r, x: np.sign(x), np.sign),
    ('$\\sin 2x$', 'bent', gauss, lambda r, x: np.sin(2 * x), lambda x: np.sin(2 * x)),
    ('$\\sin 5x$', 'no linear part', gauss, lambda r, x: np.sin(5 * x), lambda x: np.sin(5 * x)),
    ('$|x|$', 'no linear part', gauss, lambda r, x: np.abs(x), np.abs),
    ('$x^2$', 'no linear part', gauss, lambda r, x: x ** 2, lambda x: x ** 2),
    ('$x \\cdot \\epsilon$ (variance only)', 'no linear part', gauss, lambda r, x: x * g(r, len(x)), lambda x: 0 * x),
    ('X shape, $\\pm x$', 'no linear part', gauss, lambda r, x: r.choice([-1, 1], len(x)) * x, lambda x: 0 * x),
    ('circle', 'no linear part', None, circle, None),
    ('three clusters', 'no linear part', None, clusters, cluster_mean),
]


# ---- dynamical systems: (x, y) of length n ---------------------------------------------------------------------------

def ar_pair(rng, n, ax, ay, c, f=lambda u: u, sd=1.):
    n0 = n + 500
    ex, ey = g(rng, n0), g(rng, n0)
    x, y = np.zeros(n0), np.zeros(n0)
    for t in range(1, n0):
        x[t] = ax * x[t - 1] + ex[t]
        y[t] = ay * y[t - 1] + c * f(x[t - 1]) + sd * ey[t]
    return x[500:], y[500:]


def volatility(rng, n):
    x = g(rng, n)
    return x, np.r_[0, np.sqrt(.2 + .8 * x[:-1] ** 2) * g(rng, n - 1)]


def logistic(rng, n, c=.3):
    f = lambda u: 3.9 * u * (1 - u)
    x, y = np.empty(n + 500), np.empty(n + 500)
    x[0], y[0] = rng.uniform(.2, .8, 2)
    for t in range(1, n + 500):
        x[t] = f(x[t - 1]); y[t] = (1 - c) * f(y[t - 1]) + c * f(x[t - 1])
    return x[500:], y[500:]


def henon(rng, n, c=.4):
    x, y = np.zeros(n + 1000), np.zeros(n + 1000)
    x[:2], y[:2] = rng.uniform(0, .5, 2), rng.uniform(0, .5, 2)
    for t in range(1, n + 999):
        x[t + 1] = 1.4 - x[t] ** 2 + .3 * x[t - 1]
        y[t + 1] = 1.4 - (c * x[t] * y[t] + (1 - c) * y[t] ** 2) + .3 * y[t - 1]
    return x[1000:], y[1000:]


def lorenz(rng, n, sub=5, dt=.01):
    f = lambda s: np.array([10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]])
    s, out = np.array([1., 1., 20.]) + g(rng, 3), np.empty((n, 3))
    for i in range(-2000, n * sub):
        k1 = f(s); k2 = f(s + dt / 2 * k1); k3 = f(s + dt / 2 * k2); k4 = f(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if i >= 0 and i % sub == 0:
            out[i // sub] = s
    return out


def kuramoto(rng, n, dw=.2, K=1., sig=.15, sub=4, dt=.05):
    th, w, out = rng.uniform(0, 2 * np.pi, 2), np.array([1 + dw, 1 - dw]), np.empty((n, 2))
    for i in range(-2000, n * sub):
        th = th + dt * (w + K * np.sin(th[::-1] - th)) + sig * np.sqrt(dt) * g(rng, 2)
        if i >= 0 and i % sub == 0:
            out[i // sub] = np.sin(th)
    return out[:, 0], out[:, 1]


def regime(rng, n):
    x, y = g(rng, n), g(rng, n)
    y[:n // 2] = x[:n // 2] + .75 * g(rng, n // 2)
    return x, y


def signflip(rng, n):
    x = g(rng, n)
    return x, np.where(np.arange(n) < n // 2, 1, -1) * x + .5 * g(rng, n)


DYNAMIC = [
    ('VAR(1)', 'time series', lambda r, n: ar_pair(r, n, .5, .5, .5)),
    ('VAR(1), slow', 'time series', lambda r, n: ar_pair(r, n, .95, .9, .3)),
    ('AR, rectified drive', 'time series', lambda r, n: ar_pair(r, n, .5, .5, 1., lambda u: np.maximum(u, 0), .5)),
    ('AR, quadratic drive', 'time series', lambda r, n: ar_pair(r, n, .5, .5, .5, lambda u: u ** 2)),
    ('volatility drive', 'time series', volatility),
    ('logistic maps, $c = 0.3$', 'time series', logistic),
    ('Hénon maps, $c = 0.4$', 'time series', henon),
    ('Lorenz $x$–$y$', 'time series', lambda r, n: tuple(lorenz(r, n)[:, :2].T)),
    ('Lorenz $x$–$z$', 'time series', lambda r, n: tuple(lorenz(r, n)[:, [0, 2]].T)),
    ('Kuramoto pair', 'oscillators, regimes', kuramoto),
    ('repo lag, $\\theta = 0.5$', 'oscillators, regimes', lambda r, n: make('lag', .5, n, r)),
    ('repo on/off, $\\theta = 0.6$', 'oscillators, regimes', lambda r, n: make('intermittency', .6, n, r)),
    ('linear half the time', 'oscillators, regimes', regime),
    ('sign flips halfway', 'oscillators, regimes', signflip),
]
SYSTEMS = {s[0]: s for s in STATIC} | {d[0]: d for d in DYNAMIC}
GROUPS = {'linear': '#7F7F7F', 'bent': '#0072B2', 'no linear part': '#D55E00', 'time series': '#009E73',
          'oscillators, regimes': '#CC79A7'}


def draw(name, rng, n):
    s = SYSTEMS[name]
    if len(s) == 3:
        return s[2](rng, n)
    _, _, source, f, _ = s
    if source is None:
        return f(rng, n)
    x = source(rng, n)
    return x, f(rng, x)


def noisy(name, sigma, rng, n):
    x, y = (np.asarray(v, float) for v in draw(name, rng, n))
    return x, (y - y.mean()) / y.std() + sigma * g(rng, n)


def limit(name, n=10 ** 6):
    """1 - Corr(x, m(x))^2, the large-noise limit of N_MI (static systems)."""
    s = SYSTEMS[name]
    if len(s) == 3:
        return np.nan
    rng = np.random.default_rng(0)
    x, y = draw(name, rng, n)
    m = s[4](x) if s[4] is not None else None
    if m is None:                                     # circle: E[y | x] = 0 by symmetry
        return 1.
    if np.std(m) < 1e-12 * np.std(y):                 # no conditional mean: r = 0 at every noise level
        return 1.
    return float(1 - np.corrcoef(x, m)[0, 1] ** 2)


def _one(task):
    name, sigma, rep = task
    rng = np.random.default_rng([rep, list(SYSTEMS).index(name), SIGMAS.index(sigma), 7])
    e = nonlinearity_mi(*noisy(name, sigma, rng, T), rng)
    return dict(system=name, sigma=sigma, rep=rep, **e)


def _long(task):
    name, sigma = task
    v = []
    for seed in range(LONG_SEEDS):
        rng = np.random.default_rng([seed, list(SYSTEMS).index(name), SIGMAS.index(sigma), 8])
        v.append(nonlinearity_mi(*noisy(name, sigma, rng, LONG_T), rng, shifts=0, blocks=0)['nonlinearity_mi'])
    return dict(system=name, sigma=sigma, long_run=float(np.mean(v)))


def run(out, reps, workers):
    out.mkdir(parents=True, exist_ok=True)
    cells = [(s, sg) for s in SYSTEMS for sg in SIGMAS]
    with Pool(workers) as pool:
        long = pd.DataFrame(pool.map(_long, cells, chunksize=1))
        rows = pd.DataFrame(pool.map(_one, [c + (r,) for c in cells for r in range(reps)], chunksize=2))
    rows.to_csv(out / 'rows.csv', index=False)
    g_ = rows.assign(se2=rows.nonlinearity_mi_se ** 2).groupby(['system', 'sigma'])
    summ = pd.DataFrame(dict(n=g_.size(), defined=g_.nonlinearity_mi_defined.mean(), mean=g_.nonlinearity_mi.mean(),
                             sd=g_.nonlinearity_mi.std(), se=np.sqrt(g_.se2.mean()), mi=g_.mi.mean())).reset_index()
    summ = summ.merge(long, on=['system', 'sigma'])
    summ['limit'] = summ.system.map({s: limit(s) for s in SYSTEMS})
    summ['group'] = summ.system.map({s: SYSTEMS[s][1] for s in SYSTEMS})
    summ['order'] = summ.system.map({s: i for i, s in enumerate(SYSTEMS)})
    summ = summ.sort_values(['order', 'sigma']).drop(columns='order')
    summ.to_csv(out / 'summary.csv', index=False)
    return summ


def report(summ, out):
    (HERE / 'tables' / 'nmi').mkdir(parents=True, exist_ok=True)
    summ.round(4).to_csv(HERE / 'tables' / 'nmi' / 'noise.csv', index=False)
    figure(summ, out)


def figure(summ, out):
    style()
    names = list(dict.fromkeys(summ.system))
    cols = 6
    rows_ = -(-len(names) // cols)
    fig, axes = plt.subplots(rows_, cols, figsize=(11, 1.75 * rows_), sharex=True, sharey=True, constrained_layout=True)
    xs = np.log2(SIGMAS)
    for ax, name in zip(axes.flat, names):
        c = summ[summ.system == name]
        col = GROUPS[c.group.iloc[0]]
        ok = c.defined >= .8
        ax.axhline(0, color='0.85', lw=.6, zorder=0)
        ax.axhline(1, color='0.85', lw=.6, zorder=0)
        if np.isfinite(c.limit.iloc[0]):
            ax.axhline(c.limit.iloc[0], color=col, lw=.9, ls=':', zorder=1)
        ax.plot(xs, c.long_run, color='0.35', lw=1., zorder=2)
        ax.fill_between(xs, c['mean'] - c.sd, c['mean'] + c.sd, color=col, alpha=.18, lw=0, zorder=1)
        ax.plot(xs, c['mean'], color=col, lw=1.4, zorder=3)
        ax.scatter(xs[ok], c['mean'][ok], s=11, color=col, zorder=4)
        ax.scatter(xs[~ok], c['mean'][~ok], s=11, facecolor='white', edgecolor=col, zorder=4)
        for xx, (_, r) in zip(xs, c.iterrows()):
            ax.text(xx, -.78, f'{r.mi:.2f}', ha='center', va='bottom', fontsize=5.5, color='0.45')
        ax.set_title(name, fontsize=7.5, color=col, pad=3)
    for ax in axes.flat[len(names):]:
        ax.axis('off')
    for ax in axes[:, 0]:
        ax.set_ylabel('$N_{\\mathrm{MI}}$', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('noise SD $\\sigma$', fontsize=8)
    ax0 = axes.flat[0]
    ax0.set_ylim(-.8, 1.2); ax0.set_yticks([-.5, 0, .5, 1]); ax0.set_xticks(xs, ['1/4', '1/2', '1', '2'])
    for ax in axes.flat:
        ax.tick_params(labelsize=6.5, length=2)
    fig.suptitle('$N_{\\mathrm{MI}}$ under added measurement noise (T = 2,000; 20 recordings per point). Line and dots: mean '
                 '(open: defined in < 80%); band: SD across recordings;\ngrey line: long run (T = $2^{16}$); dotted: '
                 'large-noise limit $1 - r^2/\\eta^2$; small numbers: mean $I$ (nats)', fontsize=7.5, x=.01, ha='left')
    save(fig, out, 'nmi_noise')
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=Path('results/estimators/nmi-noise'))
    ap.add_argument('--reps', type=int, default=20)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--figure-only', action='store_true', help='table and figure from <out>/summary.csv')
    ap.add_argument('--run-only', action='store_true', help='no table or figure, so a cluster checkout stays clean')
    a = ap.parse_args()
    summ = pd.read_csv(a.out / 'summary.csv') if a.figure_only else run(a.out, a.reps, a.workers)
    if not a.run_only:
        report(summ, a.out)
