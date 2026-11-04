"""Choose window length T and channel set from the nested sweep (blind; no WAIS).

Each arm is a bank of z (SPI-SPI) vectors: 4 nested windows for each subject in the subset. Per arm:
- **Reliability:** single-window ICC(1) of every z feature across subjects, then the Spearman-Brown reliability of a subject mean over the K windows a fixed clean-data budget allows (K = budget / window length).
- **Fingerprinting:** the fraction of windows whose nearest other window, by correlation of standardised z, comes from the same subject.
- **Convergence:** the correlation of each window's z with the same window's T = 6000 z (canonical arms only, since the windows are nested).
- **Cost:** pyspi core-seconds per window, from the run metadata when recorded.
Writes <sweep>/analysis/sweep.json and figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mtsspi.figures import save, style

BUDGETS = (60, 120, 240)


def load(root):
    with np.load(root / 'bank.npz', allow_pickle=False) as b:
        return dict(z=b['z'].astype(np.float64), subject=b['group'], row=b['row_id'], T=int(b['T'][0]), M=int(b['M'][0]))


def icc1(x, subject):
    """One-way random-effects ICC(1) per column of x (windows, features), balanced subjects only."""
    subs, counts = np.unique(subject, return_counts=True)
    k = counts.min()
    keep = np.concatenate([np.flatnonzero(subject == s)[:k] for s in subs])
    g = x[keep].reshape(len(subs), k, -1)
    msb = k * np.nanvar(np.nanmean(g, 1), 0, ddof=1)
    msw = np.nanmean(np.nanvar(g, 1, ddof=1), 0)
    return (msb - msw) / (msb + (k - 1) * msw)


def fingerprint(x, subject):
    s = (x - np.nanmean(x, 0)) / np.nanstd(x, 0)
    s = np.nan_to_num(s)
    r = np.corrcoef(s); np.fill_diagonal(r, -np.inf)
    return float(np.mean(subject[r.argmax(1)] == subject))


def cost(root):
    """Median pyspi wall seconds per window from meta.json, when the runner recorded it."""
    vals = []
    for meta in (root / 'features').rglob('meta.json'):
        m = json.loads(meta.read_text())
        for key in ('elapsed_seconds', 'compute_seconds', 'wall_seconds'):
            v = m.get(key) or m.get('pyspi', {}).get(key) or m.get('timing', {}).get(key)
            if v:
                vals.append(float(v)); break
    return float(np.median(vals)) if vals else None


def main(sweep):
    arms = {p.name: load(p) for p in sorted(sweep.iterdir()) if (p / 'bank.npz').exists()}
    names = list(arms)
    ok = np.ones(arms[names[0]]['z'].shape[1], bool)
    for a in arms.values():
        ok &= np.isfinite(a['z']).mean(0) >= .9
    out, reference = {}, arms.get('canonical-T6000')
    for name, a in arms.items():
        x = a['z'][:, ok]
        icc = icc1(x, a['subject'])
        rec = dict(T=a['T'], M=a['M'], windows=len(x), subjects=len(np.unique(a['subject'])), features=int(ok.sum()),
                   icc_median=float(np.nanmedian(icc)), icc_over_half=float(np.nanmean(icc > .5)),
                   fingerprint=fingerprint(x, a['subject']), seconds_per_window=cost(sweep / name))
        for B in BUDGETS:
            K = max(1, B * 100 // a['T'])
            r = K * icc / (1 + (K - 1) * icc)
            rec[f'mean_reliability_{B}s'] = float(np.nanmedian(r)); rec[f'windows_{B}s'] = int(K)
        if reference is not None and a['M'] == reference['M']:
            pos = {r: k for k, r in enumerate(reference['row'])}
            pairs = [(k, pos[r]) for k, r in enumerate(a['row']) if r in pos]
            y = reference['z'][:, ok]
            rec['convergence_to_T6000'] = float(np.median([np.corrcoef(np.nan_to_num(x[i]), np.nan_to_num(y[j]))[0, 1] for i, j in pairs]))
        out[name] = rec; rec['_icc'] = icc
    if 'common62-T2000' in arms and 'canonical-T2000' in arms:
        a, b = arms['canonical-T2000'], arms['common62-T2000']
        pos = {r: k for k, r in enumerate(b['row'])}
        pairs = [(k, pos[r]) for k, r in enumerate(a['row']) if r in pos]
        out['m19_vs_m62_same_window_r'] = float(np.median([np.corrcoef(np.nan_to_num(a['z'][i, ok]), np.nan_to_num(b['z'][j, ok]))[0, 1] for i, j in pairs]))
    figure(out, sweep / 'analysis')
    for v in out.values():
        if isinstance(v, dict):
            v.pop('_icc', None)
    (sweep / 'analysis').mkdir(exist_ok=True)
    (sweep / 'analysis' / 'sweep.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def figure(out, path):
    style()
    arms = {k: v for k, v in out.items() if isinstance(v, dict)}
    can = sorted((v['T'], k) for k, v in arms.items() if v['M'] == 19)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    a = ax[0]
    labels = [f"M={arms[k]['M']}\nT={arms[k]['T']}" for _, k in can] + [f"M={v['M']}\nT={v['T']}" for k, v in arms.items() if v['M'] != 19]
    data = [arms[k]['_icc'] for _, k in can] + [v['_icc'] for v in arms.values() if v['M'] != 19]
    a.violinplot([d[np.isfinite(d)] for d in data], showmedians=True)
    a.set_xticks(range(1, len(labels) + 1), labels)
    a.axhline(0, color='.7', lw=.8)
    a.set(ylabel='single-window ICC(1) per z feature', title='Test–retest reliability of z')
    a = ax[1]
    T = [t for t, _ in can]
    for B, c in zip(BUDGETS, plt.cm.viridis(np.linspace(.15, .8, len(BUDGETS)))):
        a.plot(T, [arms[k][f'mean_reliability_{B}s'] for _, k in can], 'o-', color=c, label=f'{B} s of clean data')
    a.plot(T, [arms[k]['icc_median'] for _, k in can], 'o--', color='k', label='one window')
    a.set(xlabel='window length T (samples at 100 Hz)', ylabel='median reliability of the subject mean', title='Fixed data budget: fewer long or more short windows')
    a.legend()
    a = ax[2]
    a.plot(T, [arms[k]['fingerprint'] for _, k in can], 'o-', color='k', label='window-to-subject identification')
    if all('convergence_to_T6000' in arms[k] for _, k in can):
        a.plot(T, [arms[k]['convergence_to_T6000'] for _, k in can], 's-', color='#C44E52', label='r(z_T, z_6000), same window')
    a.set(xlabel='window length T', ylim=(0, 1.02), title='Identification and convergence')
    a.legend(loc='lower right')
    save(fig, path, 'sweep'); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    main(parser.parse_args().sweep)
