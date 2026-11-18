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
- character (what the flags are): share of flagged samples dominated (> 50% of |contribution|) by one
  channel; which channel dominates (topography); the largest single-channel share of flags in the window (a
  bad-contact signature); mean flagged run length (ms); amplitude ratio flagged/unflagged; and the envelope
  of the dominant channel at flagged samples relative to its unflagged mean, in 0.5-4 Hz and 20-45 Hz.
- atlas mode: the same mcd and events readouts on an atlas window bank (prepare.py layout), on fixed 16-channel
  subsets, with both IAAFT variants generated per window; contrasts are within unit, each state against the
  unit's reference state, grouped by agent. Sleep windows are also split into within-archive delta-fraction
  tertiles (sleep_deep, sleep_light).
- index mode: the mcd and events readouts on a bank.npz with both IAAFT variants generated per window.
- gain control (--gain, atlas and index modes): each channel divided by its sliding 1-s RMS before surrogates
  and scoring, so within-window gain modulation cannot produce flags.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt
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


def runs(mask):
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def envelope(x, fs, band):
    sos = butter(4, [band[0], min(band[1], fs / 2 - 1)], btype='band', fs=fs, output='sos')
    return np.abs(hilbert(sosfiltfilt(sos, x, axis=0), axis=0))


def character_metrics(x, fs, seed=261111):
    x = standardise(x)
    M = x.shape[1]
    mcd = MinCovDet(random_state=seed).fit(x)
    flag = ~mcd.support_
    if flag.sum() < 5 or (~flag).sum() < 5:
        return {}
    d = x - mcd.location_
    contrib = np.abs(d * (d @ mcd.precision_))
    share = contrib / contrib.sum(1, keepdims=True)
    top = share.argmax(1)
    top_share = np.bincount(top[flag], minlength=M) / flag.sum()
    t = np.flatnonzero(flag)
    out = dict(dominated=float((share[flag].max(1) > .5).mean()), max_channel_share=float(top_share.max()),
               run_ms=float(runs(flag).mean() * 1000 / fs),
               amplitude_ratio=float(np.abs(x[flag]).max(1).mean() / np.abs(x[~flag]).max(1).mean()),
               top_share=top_share.tolist())
    for name, band in (('low', (0.5, 4.0)), ('high', (20.0, 45.0))):
        env = envelope(x, fs, band)
        base = np.array([env[~flag, k].mean() for k in range(M)])
        out[f'{name}_band_ratio'] = float(np.mean(env[t, top[t]] / base[top[t]]))
    return out


def window(x, analyses, fs=None):
    out = {}
    if 'mcd' in analyses:
        out.update(mcd_metrics(x))
    if 'burst' in analyses:
        out.update(burst_metrics(x))
    if 'events' in analyses:
        out.update(event_metrics(x))
    if 'character' in analyses:
        out.update(character_metrics(x, fs))
    return out


def run(bank_path, analyses, jobs, fs=None):
    b = np.load(bank_path, allow_pickle=False)
    x = np.asarray(b['x'], dtype=np.float64)
    rows = Parallel(n_jobs=jobs)(delayed(window)(x[k], analyses, fs) for k in range(len(x)))
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
    keys = sorted({k for r in real['rows'] for k, v in r.items() if not isinstance(v, list)})
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


def topography(banks):
    """Mean share of flagged samples dominated by each channel, per state and bank."""
    out = {}
    for name, bank in banks.items():
        y = np.array(bank['y'])
        vecs = [r.get('top_share') for r in bank['rows']]
        if not any(v is not None for v in vecs):
            continue
        out[name] = {str(s): np.nanmean([v for v, ys in zip(vecs, y) if v is not None and ys == s], axis=0).tolist()
                     for s in np.unique(y)}
    return out


def main(dataset, real, surrogates, output, analyses, jobs, fs=None):
    banks = {'real': run(real, analyses, jobs, fs)}
    for s in surrogates:
        banks[Path(s).parent.name] = run(s, analyses, jobs, fs)
        if banks[Path(s).parent.name]['y'] != banks['real']['y']:
            raise ValueError(f'{s}: row order differs from the real bank')
    report = dict(dataset=dataset, analyses=analyses, real=str(real), surrogates=[str(s) for s in surrogates],
                  contrasts=contrasts(dataset, banks), topography=topography(banks),
                  windows={k: v['rows'] for k, v in banks.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


SUBSET = 16
VARIANTS = ('marginal', 'spectrum')


def subsets(M):
    """Two fixed, interleaved 16-channel subsets (all channels if M <= 16)."""
    if M <= SUBSET:
        return dict(all=np.arange(M))
    step = (M - 1) / (SUBSET - 1)
    a = np.round(np.arange(SUBSET) * step).astype(int)
    return dict(a=a, b=np.minimum(a + int(step // 2), M - 1))


def gain_normalise(x, fs, seconds=1.0):
    """Each channel divided by its centred sliding RMS over `seconds`."""
    power = uniform_filter1d(x ** 2, size=max(3, int(round(fs * seconds))), axis=0, mode='reflect')
    return x / np.sqrt(power + 1e-12 * power.mean())


def versions_metrics(w, seed):
    """mcd and events readouts on a window and on both IAAFT surrogates of it: {version: metrics}."""
    from mtsspi.surrogates import multivariate_iaaft
    versions = {'real': w}
    for v in VARIANTS:
        versions[v] = multivariate_iaaft(w, np.random.default_rng(seed), final=v)[0]
    return {name: dict(mcd_metrics(arr), **event_metrics(arr)) for name, arr in versions.items()}


def atlas_window(x, seed, fs=None):
    out = {}
    if fs:
        x = gain_normalise(x, fs)
    for tag, idx in subsets(x.shape[1]).items():
        for name, metrics in versions_metrics(x[:, idx], [seed, len(tag)]).items():
            out.update({f'{name}|{tag}|{k}': v for k, v in metrics.items()})
    return out


def index_window(x, seed, fs=None):
    return versions_metrics(gain_normalise(x, fs) if fs else x, seed)


def index(dataset, bank_path, output, jobs, gain_fs=None):
    """index mode: real and per-window surrogate readouts of a bank.npz, contrasted as in main."""
    b = np.load(bank_path, allow_pickle=False)
    x = np.asarray(b['x'], dtype=np.float64)
    values = Parallel(n_jobs=jobs)(delayed(index_window)(x[k], k, gain_fs) for k in range(len(x)))
    meta = dict(row_id=b['row_id'].astype(str).tolist(), y=b['y'].tolist(), group=b['group'].astype(str).tolist())
    banks = {name: dict(meta, rows=[v[name] for v in values]) for name in ('real',) + VARIANTS}
    report = dict(dataset=dataset, bank=str(bank_path), gain_normalised=bool(gain_fs),
                  contrasts=contrasts(dataset, banks), windows={k: v['rows'] for k, v in banks.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


def sleep_tertiles(rows):
    """Relabel sleep windows by within-archive delta-fraction tertile: sleep_deep (top), sleep_light (bottom)."""
    states = []
    for r in rows:
        if r['state'] != 'sleep':
            states.append(r['state'])
            continue
        own = np.array([q['delta_fraction'] for q in rows if q['unit'] == r['unit'] and q['state'] == 'sleep'])
        lo, hi = np.quantile(own, [1 / 3, 2 / 3])
        states.append('sleep_deep' if r['delta_fraction'] >= hi else 'sleep_light' if r['delta_fraction'] <= lo else 'sleep_mid')
    return states


def atlas(bank, output, reference, jobs, gain=False):
    manifest = json.loads((bank / 'manifest.json').read_text())
    rows = manifest['rows']
    with np.load(bank / 'windows.npz') as archive:
        values = Parallel(n_jobs=jobs)(delayed(atlas_window)(np.asarray(archive[r['id']], dtype=np.float64), k,
                                                             manifest['fs'] if gain else None)
                                       for k, r in enumerate(rows))
    agent = [r.get('agent', manifest['dataset']) for r in rows]
    unit = np.array([r['unit'] for r in rows])
    state = np.array(sleep_tertiles(rows) if any('delta_fraction' in r for r in rows) else [r['state'] for r in rows])
    keys = sorted({k.split('|', 1)[1] for v in values for k in v})
    report = dict(bank=str(bank), reference=reference, gain_normalised=gain, subsets={t: i.tolist() for t, i in subsets(rows[0]['M']).items()},
                  contrasts={}, windows=[dict(id=r['id'], **v) for r, v in zip(rows, values)])
    for name in ('real',) + tuple(f'real_minus_{v}' for v in VARIANTS):
        table = {}
        for key in keys:
            if name == 'real':
                val = np.array([v.get(f'real|{key}', np.nan) for v in values], dtype=float)
            else:
                val = np.array([v.get(f'real|{key}', np.nan) - v.get(f"{name.split('_')[-1]}|{key}", np.nan) for v in values], dtype=float)
            for a in sorted(set(agent)):
                for s in sorted(set(state) - {reference}):
                    diffs = []
                    for u in np.unique(unit[np.array(agent) == a]):
                        sa, sr = val[(unit == u) & (state == s)], val[(unit == u) & (state == reference)]
                        if len(sa) and len(sr):
                            diffs.append(np.nanmean(sa) - np.nanmean(sr))
                    if diffs:
                        diffs = np.array(diffs)
                        table.setdefault(f'{a}:{s}', {})[key] = dict(
                            mean_difference=float(np.nanmean(diffs)), units=len(diffs), units_positive=int(np.sum(diffs > 0)),
                            d=float(np.nanmean(diffs) / np.nanstd(diffs, ddof=1)) if len(diffs) > 2 else None)
        report['contrasts'][name] = table
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', choices=['neurotycho', 'propofol', 'atlas'])
    p.add_argument('--mode', choices=['banks', 'index'], default='banks',
                   help='banks: stored surrogate banks (main); index: surrogates generated per window')
    p.add_argument('--gain', action='store_true', help='atlas and index modes: sliding 1-s RMS normalisation')
    p.add_argument('--real', type=Path, required=True, help='bank.npz, or an atlas bank directory in atlas mode')
    p.add_argument('--surrogates', type=Path, nargs='*', default=[])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--analyses', nargs='+', default=['mcd', 'burst', 'events'])
    p.add_argument('--jobs', type=int, default=8)
    p.add_argument('--fs', type=float, help='sampling rate, for the character analysis')
    p.add_argument('--reference', default='awake_ec', help='atlas mode: the reference state')
    a = p.parse_args()
    if a.dataset == 'atlas':
        atlas(a.real, a.output, a.reference, a.jobs, a.gain)
    elif a.mode == 'index':
        index(a.dataset, a.real, a.output, a.jobs, a.fs if a.gain else None)
    else:
        main(a.dataset, a.real, a.surrogates, a.output, a.analyses, a.jobs, a.fs)
