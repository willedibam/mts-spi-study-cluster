"""Is dependence under anaesthesia carried by brief joint events? pyspi-free deciding analyses (11 November 2026).

Reads raw windows of a real bank and its multivariate-IAAFT surrogate banks (same row order) and writes one
JSON of per-window metrics with unit-level contrasts. Reading rules: docs/interpreting-spi-pairs.md.

- mcd (NeuroTycho tail link): MinCovDet per window. Flags = samples outside the reweighted support. Reports
  the flag fraction; temporal clustering (lag-1 phi of the flag series); spatial focality of flagged samples
  (participation ratio of per-channel contributions to the Mahalanobis distance, divided by M: 1 = spread
  over all channels, 1/M = one channel); common-mode share (|channel mean| of flagged over unflagged samples);
  and whether flagged samples carry a distinct network: similarity of Ledoit-Wolf partial correlations on
  flagged against unflagged samples, minus the same similarity for the flag mask circularly shifted (same
  size and clustering, 20 shifts). Negative = flagged samples carry a distinct network.
- burst (propofol linear-rank residual): z(Pearson, Spearman) across pairs, split through Pearson on
  normal scores: z(r, r_ns) reads marginal shape, z(r_ns, rho) reads copula departure. Each also with the
  top q% of joint-amplitude samples (sum of squared standardised channels) removed.
- events (cross-species index): per edge, the share of sum |x_i x_j| carried by its top 5% samples, and the
  lag-1 phi of those samples' indicator; window means over edges.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf, MinCovDet

QS = (0.01, 0.02, 0.05)


def standardise(x):
    return (x - x.mean(0)) / x.std(0)


def phi(mask):
    a, b = mask[:-1].astype(float), mask[1:].astype(float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def partial(x):
    p = LedoitWolf().fit(x).precision_
    d = np.sqrt(np.diag(p))
    return (-p / np.outer(d, d))[np.triu_indices(x.shape[1], 1)]


def mcd_metrics(x, seed=261111):
    x = standardise(x)
    T, M = x.shape
    mcd = MinCovDet(random_state=seed).fit(x)
    flag = ~mcd.support_
    if flag.sum() < M + 2 or (~flag).sum() < M + 2:
        return dict(fraction=float(flag.mean()))
    d = x - mcd.location_
    contrib = np.abs(d * (d @ mcd.precision_))
    pr = (contrib.sum(1) ** 2 / (contrib ** 2).sum(1)) / M
    common = np.abs(x.mean(1))
    sim = float(np.corrcoef(partial(x[flag]), partial(x[~flag]))[0, 1])
    rng = np.random.default_rng(seed)
    null = []
    for s in rng.integers(T // 10, T - T // 10, 20):
        m = np.roll(flag, s)
        null.append(np.corrcoef(partial(x[m]), partial(x[~m]))[0, 1])
    return dict(fraction=float(flag.mean()), clustering=phi(flag), focality=float(pr[flag].mean()),
                focality_unflagged=float(pr[~flag].mean()),
                common_mode_ratio=float(common[flag].mean() / common[~flag].mean()),
                network_similarity=sim, network_similarity_null=float(np.mean(null)),
                distinct_network=sim - float(np.mean(null)))


def edges(c):
    return c[np.triu_indices(c.shape[0], 1)]


def burst_metrics(x):
    from scipy.special import ndtri  # imported here: joblib workers cannot unpickle the ufunc from __main__
    x = standardise(x)
    T = len(x)
    amp = (x ** 2).sum(1)
    out = {}
    for q in (0.0,) + QS:
        keep = amp <= np.quantile(amp, 1 - q) if q else np.ones(T, bool)
        w = x[keep]
        ranks = np.apply_along_axis(rankdata, 0, w)
        r = edges(np.corrcoef(w.T))
        r_ns = edges(np.corrcoef(ndtri((ranks - 0.5) / len(w)).T))
        rho = edges(np.corrcoef(ranks.T))
        tag = f'q{int(q * 100)}'
        out.update({f'{tag}_linear_rank': float(np.corrcoef(r, rho)[0, 1]),
                    f'{tag}_marginal': float(np.corrcoef(r, r_ns)[0, 1]),
                    f'{tag}_copula': float(np.corrcoef(r_ns, rho)[0, 1])})
    return out


def event_metrics(x, top=0.05):
    x = standardise(x)
    i, j = np.triu_indices(x.shape[1], 1)
    prod = np.abs(x[:, i] * x[:, j])
    cut = np.quantile(prod, 1 - top, axis=0)
    hit = prod >= cut
    share = (prod * hit).sum(0) / prod.sum(0)
    clus = [phi(hit[:, k]) for k in range(hit.shape[1])]
    return dict(event_share=float(share.mean()), event_clustering=float(np.nanmean(clus)))


def window(x, analyses):
    out = {}
    if 'mcd' in analyses:
        out.update(mcd_metrics(x))
    if 'burst' in analyses:
        out.update(burst_metrics(x))
    if 'events' in analyses:
        out.update(event_metrics(x))
    return out


def run(bank_path, analyses, jobs):
    b = np.load(bank_path, allow_pickle=False)
    x = np.asarray(b['x'], dtype=np.float64)
    rows = Parallel(n_jobs=jobs)(delayed(window)(x[k], analyses) for k in range(len(x)))
    return dict(row_id=b['row_id'].astype(str).tolist(), y=b['y'].tolist(), group=b['group'].astype(str).tolist(), rows=rows)


def units(dataset, row_ids, groups):
    """Unit per window: NeuroTycho = agent x animal (6 units); propofol = participant."""
    if dataset == 'neurotycho':
        return [f"{r.split('--')[0][8:12]}-{g}" for r, g in zip(row_ids, groups)]
    return list(groups)


def contrasts(dataset, banks):
    """Unit-level state contrasts of every metric, for the real bank, each surrogate and real minus surrogate."""
    real = banks['real']
    u = np.array(units(dataset, real['row_id'], real['group']))
    y = np.array(real['y'])
    pairs = [(1, 0)] if dataset == 'neurotycho' else [(2, 0), (1, 0), (2, 3)]
    keys = sorted({k for r in real['rows'] for k in r})
    out = {}
    for name, bank in banks.items():
        vals = {k: np.array([r.get(k, np.nan) for r in bank['rows']], dtype=float) for k in keys}
        if name != 'real':
            vals = {k: np.array([r.get(k, np.nan) for r in real['rows']], dtype=float) - v for k, v in vals.items()}
            name = f'real_minus_{name}'
        out[name] = {}
        for a, c in pairs:
            entry = {}
            for k, v in vals.items():
                per_unit = []
                for unit in np.unique(u):
                    sa, sc = v[(u == unit) & (y == a)], v[(u == unit) & (y == c)]
                    if len(sa) and len(sc):
                        per_unit.append(np.nanmean(sa) - np.nanmean(sc))
                per_unit = np.array(per_unit)
                entry[k] = dict(mean_difference=float(np.nanmean(per_unit)), units=int(np.sum(np.isfinite(per_unit))),
                                units_positive=int(np.sum(per_unit > 0)),
                                d=float(np.nanmean(per_unit) / np.nanstd(per_unit, ddof=1)) if len(per_unit) > 2 else None,
                                state_means={str(s): float(np.nanmean(v[y == s])) for s in (a, c)})
            out[name][f'{a}_vs_{c}'] = entry
    return out


def main(dataset, real, surrogates, output, analyses, jobs):
    banks = {'real': run(real, analyses, jobs)}
    for s in surrogates:
        banks[Path(s).parent.name] = run(s, analyses, jobs)
        if banks[Path(s).parent.name]['y'] != banks['real']['y']:
            raise ValueError(f'{s}: row order differs from the real bank')
    report = dict(dataset=dataset, analyses=analyses, real=str(real), surrogates=[str(s) for s in surrogates],
                  contrasts=contrasts(dataset, banks), windows={k: v['rows'] for k, v in banks.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', choices=['neurotycho', 'propofol'])
    p.add_argument('--real', type=Path, required=True)
    p.add_argument('--surrogates', type=Path, nargs='*', default=[])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--analyses', nargs='+', default=['mcd', 'burst', 'events'])
    p.add_argument('--jobs', type=int, default=8)
    a = p.parse_args()
    main(a.dataset, a.real, a.surrogates, a.output, a.analyses, a.jobs)
