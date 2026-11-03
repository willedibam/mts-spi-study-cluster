"""Are the NeuroTycho state effects of the top z pairs and core readouts beyond auto- and cross-spectra?

Multivariate IAAFT keeps every channel's spectrum and every cross-spectrum (one surrogate per window), so a state
effect that reappears in the surrogate is linear-spectral; the per-window residual real − surrogate is what is
beyond it. Spectra-adjusted ANCOVA could not decide this (states do not overlap spectrally); this intervention
does not extrapolate. Per surrogate variant: paired unit effects (d, units agreeing) of real, surrogate and
residual for the core readouts and the figure-data top pairs, and the residual d of every z coordinate (top
listed). Run on Gadi (jobs/surrogate_pairs.pbs).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mtsspi.character_axes import core_scores, edge_profiles
from research.representation.character_axes.evaluate_cohorts import effect


def load(root):
    bank = np.load(root / 'bank.npz')
    spi = [str(s) for s in bank['spi_order']]
    core, names = core_scores(edge_profiles(bank['aligned'], int(bank['M'][0]), len(spi)), spi)
    keys = np.array([f'{g}-{r}-{w}' for g, r, w in zip(bank['group'].astype(str), bank['role'].astype(str), bank['y'])])
    return bank, core, names, keys


def paired_d(values, y, units):
    """Paired d over units (positive minus negative class means), and the fraction of units agreeing in sign."""
    common = sorted(set(units[y == 1]) & set(units[y == 0]))
    diff = np.stack([np.nanmean(values[(units == u) & (y == 1)], 0) - np.nanmean(values[(units == u) & (y == 0)], 0) for u in common])
    with np.errstate(invalid='ignore', divide='ignore'):
        d = np.nanmean(diff, 0) / np.nanstd(diff, 0, ddof=1)
    return d, np.mean(np.sign(diff) == np.sign(np.nanmean(diff, 0)), 0)


def main(real_root, surrogate_roots, figure_data, output, n_top):
    bank, core, core_names, keys = load(real_root)
    D = np.load(figure_data)
    top = [int(k) for k in D['top'][:n_top]]
    names_z = [str(n) for n in D['names_z']]
    ix = np.isin(bank['y'], [0, 1])
    y = (bank['y'][ix] == 1).astype(int)
    groups = np.array([f'{g}-{r}' for g, r in zip(bank['group'].astype(str), bank['role'].astype(str))])[ix]
    z_real = np.asarray(bank['z'], dtype=np.float64)[ix]
    real = np.column_stack([core[ix], z_real[:, top]])
    names = list(core_names) + [names_z[k] for k in top]
    report = dict(names=names, real=effect(real, y, groups), variants={})
    d_real, _ = paired_d(z_real, y, groups)
    for root in surrogate_roots:
        sbank, score, _, skeys = load(root)
        assert np.array_equal(skeys, keys), f'{root}: windows not aligned with the real bank'
        z_s = np.asarray(sbank['z'], dtype=np.float64)[ix]
        surrogate = np.column_stack([score[ix], z_s[:, top]])
        d_res, agree = paired_d(z_real - z_s, y, groups)
        ok = np.isfinite(d_res) & (agree == 1)
        best = np.argsort(-np.abs(np.where(ok, d_res, 0)))[:25]
        report['variants'][root.name] = dict(
            surrogate=effect(surrogate, y, groups), residual=effect(real - surrogate, y, groups),
            mean_abs_z_gap=float(np.nanmean(np.abs(z_real - z_s))),
            residual_all=dict(d_quantiles={q: float(np.nanquantile(np.abs(d_res), q)) for q in (.5, .9, .99)},
                              unanimous_abs_d_gt_2=int(np.sum(ok & (np.abs(d_res) > 2))),
                              top=[dict(name=names_z[k], d_residual=float(d_res[k]), d_real=float(d_real[k])) for k in best]))
        f = lambda e: f"{e['standardized']:+6.2f} {e['positive']}/{e['units']}" if e['standardized'] is not None else '   n/a'
        print(root.name, 'unanimous |d|>2 residual coordinates:', report['variants'][root.name]['residual_all']['unanimous_abs_d_gt_2'], flush=True)
        for k, n in enumerate(names):
            print(f'  {n[:60]:60s} real', f(report['real'][k]), 'surr', f(report['variants'][root.name]['surrogate'][k]),
                  'resid', f(report['variants'][root.name]['residual'][k]), flush=True)
        for t in report['variants'][root.name]['residual_all']['top'][:10]:
            print(f"  top residual {t['name'][:70]:70s} {t['d_residual']:+.2f} (real {t['d_real']:+.2f})", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, default=float) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real', type=Path, required=True)
    p.add_argument('--surrogates', type=Path, nargs='+', required=True)
    p.add_argument('--figure-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n-top', type=int, default=12)
    a = p.parse_args()
    main(a.real, a.surrogates, a.figure_data, a.output, a.n_top)
