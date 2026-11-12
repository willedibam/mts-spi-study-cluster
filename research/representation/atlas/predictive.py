"""Non-Gaussian predictive structure beyond a spectrum-matched null (pyspi-free; findings.md, "Linear against
model-free prediction").

Readout per window: the median over ordered channel pairs of TE_kNN - TE_Gauss (history 1, lag 1, nats). For
Gaussian data the two agree, since Gaussian TE is GC / 2 (Barnett, Barrett & Seth 2009). TE_kNN is the
Frenzel-Pompe / KSG conditional-MI estimator (k = 4, max norm) with pyspi's AUTO Theiler window
(int(2 <acf_i, acf_j>)). Its bias depends on the spectrum, so the readout is compared with the same readout on
multivariate IAAFT surrogates of the same window, which keep auto- and cross-spectra (and marginals) and hence
the bias. Real minus surrogate is the non-Gaussian part; its sign is read with the synthetic controls below.

controls: synthetic 16-channel systems with known answers (linear fast, linear slow, quadratic coupling).
bank: a channel subset of every window of an atlas bank, real and surrogate, then within-unit contrasts.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.special import digamma
from scipy.stats import norm

from mtsspi.surrogates import multivariate_iaaft
from research.representation.character_axes.episodic import sleep_tertiles

K = 4


def acf(x):
    x = (x - x.mean()) / x.std()
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    a = np.fft.irfft(f * f.conj(), n=2 * n)[:n]
    return a / a[0]


def theiler(A, i, j, k=K):
    """pyspi AUTO window for pair (i, j) from precomputed ACFs, reduced only if too few neighbours remain."""
    w = max(int(2 * np.dot(A[i], A[j])), 0)
    n = A.shape[1] - 1
    return min(w, (n - k - 2) // 2)


def ksg_te(src, tgt_dist, w, k=K):
    """TE src -> target given the target's |future|, |past| distance matrices (n, n) and the source series."""
    dz, dy = tgt_dist
    x = src[:-1]
    dx = np.abs(x[:, None] - x[None, :])
    n = len(x)
    band = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) <= w
    dzy = np.maximum(dz, dy)
    joint = np.where(band, np.inf, np.maximum(dzy, dx))
    eps = np.partition(joint, k - 1, axis=1)[:, k - 1][:, None]
    count = lambda d: np.sum(np.where(band, False, d < eps), axis=1)
    return float(digamma(k) + np.mean(digamma(count(dy) + 1) - digamma(count(dzy) + 1) - digamma(count(np.maximum(dx, dy)) + 1)))


def gauss_te(src, tgt):
    """Gaussian TE (nats): half the log ratio of residual variances without and with the source's past."""
    z, y, x = tgt[1:], tgt[:-1], src[:-1]
    r1 = z - np.polyval(np.polyfit(y, z, 1), y)
    X = np.column_stack([np.ones_like(y), y, x])
    r2 = z - X @ np.linalg.lstsq(X, z, rcond=None)[0]
    return float(0.5 * np.log(np.sum(r1 ** 2) / np.sum(r2 ** 2)))


def normal_scores(X):
    r = np.argsort(np.argsort(X, axis=0), axis=0)
    return norm.ppf((r + 0.5) / len(X))


def window_readout(X, rng, k=K, marked=()):
    """(T, M) window -> medians over ordered pairs of TE_kNN - TE_Gauss, TE_kNN, TE_Gauss, and TE_kNN - TE_Gauss on
    normal scores; with ``marked`` (source, target) pairs, also the median of the difference over those pairs."""
    X = (X - X.mean(0)) / X.std(0)
    X = X + 1e-8 * rng.normal(size=X.shape)  # break ties, as JIDT does
    M = X.shape[1]
    A = np.array([acf(X[:, m]) for m in range(M)])
    S = normal_scores(X)
    kn, ga, gs, mark = [], [], [], []
    for j in range(M):
        z, y = X[1:, j], X[:-1, j]
        dist = (np.abs(z[:, None] - z[None, :]), np.abs(y[:, None] - y[None, :]))
        for i in range(M):
            if i == j:
                continue
            kn.append(ksg_te(X[:, i], dist, theiler(A, i, j, k), k))
            ga.append(gauss_te(X[:, i], X[:, j]))
            gs.append(gauss_te(S[:, i], S[:, j]))
            mark.append((i, j) in marked)
    kn, ga, gs, mark = map(np.array, (kn, ga, gs, mark))
    out = dict(diff=float(np.median(kn - ga)), te_knn=float(np.median(kn)), te_gauss=float(np.median(ga)),
               diff_scores=float(np.median(kn - gs)))
    if mark.any():
        out['diff_marked'] = float(np.median((kn - ga)[mark]))
    return out


def readout_with_null(X, seed, draws=3, final='spectrum', marked=()):
    """Real readout, the mean surrogate readout over draws, and real minus surrogate for every field."""
    rng = np.random.default_rng(seed)
    real = window_readout(X, rng, marked=marked)
    surr = [window_readout(multivariate_iaaft(X, rng, final=final)[0], rng, marked=marked) for _ in range(draws)]
    null = {key: float(np.mean([s[key] for s in surr])) for key in real}
    return dict(real=real, null=null, residual={key: real[key] - null[key] for key in real})


# ---------------------------------------------------------------- synthetic controls
def simulate(kind, T, M=16, rng=None, burn=500):
    """AR(1) channels: the first half are independent Gaussian drivers; channel h + m is driven by driver m through a
    linear term or a quadratic one (zero mean and uncorrelated with the driver, so invisible to GC)."""
    rng = rng or np.random.default_rng()
    a = {'linear_fast': .5, 'linear_slow': .95, 'quadratic_fast': .5, 'quadratic_slow': .95}[kind]
    h = M // 2
    c = .6 * np.sqrt(1 - a ** 2)  # coupling on the scale of the innovations
    x = np.zeros((T + burn, M))
    e = rng.normal(size=(T + burn, M)) * np.sqrt(1 - a ** 2)
    for t in range(1, T + burn):
        d = x[t - 1, :h]
        term = d if kind.startswith('linear') else (d ** 2 - 1) / np.sqrt(2)
        x[t, :h] = a * x[t - 1, :h] + e[t, :h]
        x[t, h:2 * h] = a * x[t - 1, h:2 * h] + c * term + e[t, h:2 * h]
    return x[burn:]


def coupled(M=16):
    return tuple((m, M // 2 + m) for m in range(M // 2))


def controls(output, T_values=(200, 1000), windows=12, jobs=8):
    kinds = ('linear_fast', 'linear_slow', 'quadratic_fast', 'quadratic_slow')
    tasks = [(kind, T, w) for kind in kinds for T in T_values for w in range(windows)]
    run = lambda kind, T, w: readout_with_null(simulate(kind, T, rng=np.random.default_rng(1000 * w + T)), seed=w,
                                               marked=coupled())
    out = Parallel(n_jobs=jobs)(delayed(run)(*t) for t in tasks)
    report = {}
    for (kind, T, _), r in zip(tasks, out):
        e = report.setdefault(f'{kind}:T{T}', {})
        for part in ('real', 'null', 'residual'):
            for key, v in r[part].items():
                e.setdefault(f'{part}:{key}', []).append(v)
    for e in report.values():
        for key, v in e.items():
            v = np.array(v)
            e[key] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), positive=int((v > 0).sum()), n=len(v))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


# ---------------------------------------------------------------- atlas banks
def bank(bank_dir, output, reference, channels=16, draws=3, jobs=8):
    manifest = json.loads((bank_dir / 'manifest.json').read_text())
    rows = manifest['rows']
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in x for x in rows) else [x['state'] for x in rows])
    with np.load(bank_dir / 'windows.npz') as archive:
        windows = [np.asarray(archive[r['id']], dtype=np.float64) for r in rows]
    pick = lambda X: X[:, np.unique(np.linspace(0, X.shape[1] - 1, channels).round().astype(int))]
    out = Parallel(n_jobs=jobs)(delayed(readout_with_null)(pick(X), seed=k, draws=draws) for k, X in enumerate(windows))
    unit = np.array([r['unit'] for r in rows])
    agent = {r['unit']: r.get('agent', 'all') for r in rows}
    animal = {r['unit']: r.get('animal') or r['unit'] for r in rows}
    fields = [f'{p}:{k}' for p in ('real', 'null', 'residual') for k in out[0]['real']]
    Y = np.array([[o[f.split(':')[0]][f.split(':')[1]] for f in fields] for o in out])
    means = {}
    for u, s in sorted(set(zip(unit, state))):
        means[(u, s)] = Y[(unit == u) & (state == s)].mean(0)
    report = dict(bank=str(bank_dir), reference=reference, channels=channels, draws=draws, states={}, contrasts={})
    for (u, s), v in means.items():
        report['states'].setdefault(f'{agent[u]}:{s}', []).append(v.tolist())
        if s == reference or (u, reference) not in means:
            continue
        report['contrasts'].setdefault(f'{agent[u]}:{s}', []).append((animal[u], (v - means[(u, reference)]).tolist()))
    report['states'] = {k: dict(zip(fields, np.mean(v, 0).tolist())) for k, v in report['states'].items()}
    for k, v in report['contrasts'].items():
        D = np.array([d for _, d in v])
        by = {}
        for a_, d in v:
            by.setdefault(a_, []).append(d)
        am = np.array([np.mean(x, 0) for x in by.values()])
        report['contrasts'][k] = {f: dict(mean=float(D[:, i].mean()), units=len(D), positive=int((D[:, i] > 0).sum()),
                                          animals=len(am), animals_positive=int((am[:, i] > 0).sum()))
                                  for i, f in enumerate(fields)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('controls')
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--jobs', type=int, default=8)
    b = sub.add_parser('bank')
    b.add_argument('bank', type=Path)
    b.add_argument('--output', type=Path, required=True)
    b.add_argument('--reference', default='awake_ec')
    b.add_argument('--draws', type=int, default=3)
    b.add_argument('--jobs', type=int, default=8)
    args = p.parse_args()
    if args.command == 'controls':
        controls(args.output, jobs=args.jobs)
    else:
        bank(args.bank, args.output, args.reference, draws=args.draws, jobs=args.jobs)
