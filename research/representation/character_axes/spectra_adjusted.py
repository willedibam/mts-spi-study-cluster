"""Do the NeuroTycho state effects survive adjustment for the channels' own spectra?

Spectra separate anaesthesia from awake at AUROC ~1, so prediction cannot show that the core readouts or the top
z pairs carry anything beyond them. Instead, per readout, compare the paired state effect before and after
ANCOVA-style adjustment (sufficiency.adjust: slopes pooled within class) for two spectral covariate sets:
'summary' (channel-mean and across-channel SD of the 32-bin log spectrum) and 'pca' (first 20 principal
components of the full per-channel spectrum). Because the classes barely overlap in spectral space, the adjustment
extrapolates within-class slopes; the report gives the class separation of each covariate set so the reader can
judge that. Run on Gadi (jobs/spectra_adjusted.pbs).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mtsspi.character_axes import core_scores, edge_profiles
from research.representation.character_axes.evaluate_cohorts import effect
from research.representation.character_axes.sufficiency import adjust

BINS = 32


def spectral_covariates(bank, n):
    M = int(bank['M'][0])
    s = np.asarray(bank['spectral'], dtype=np.float64)
    per = s.reshape(n, BINS, M)
    summary = np.column_stack([per.mean(2), per.std(2)])
    centred = s - s.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return dict(summary=summary, pca=centred @ vt[:20].T)


def separation(Z, y):
    """Fraction of windows whose covariates lie inside the other class's range on every column (overlap)."""
    lo = [Z[y == c].min(0) for c in (0, 1)]
    hi = [Z[y == c].max(0) for c in (0, 1)]
    inside = [np.all((Z[y == c] >= lo[1 - c]) & (Z[y == c] <= hi[1 - c]), 1).mean() for c in (0, 1)]
    return float(np.mean(inside))


def main(root, figure_data, output, n_top):
    bank = np.load(root / 'bank.npz')
    spi = [str(s) for s in bank['spi_order']]
    y_all = bank['y']
    ix = np.isin(y_all, [0, 1])
    y = (y_all[ix] == 1).astype(int)
    groups = np.array([f'{g}-{r}' for g, r in zip(bank['group'].astype(str), bank['role'].astype(str))])[ix]
    core, core_names = core_scores(edge_profiles(bank['aligned'], int(bank['M'][0]), len(spi)), spi)
    D = np.load(figure_data)
    top = [(int(k), str(D['names_z'][k])) for k in D['top'][:n_top]]
    z = np.asarray(bank['z'], dtype=np.float64)[ix][:, [k for k, _ in top]]
    values = np.column_stack([core[ix], z])
    names = list(core_names) + [name for _, name in top]
    covariates = {k: v[ix] for k, v in spectral_covariates(bank, len(y_all)).items()}
    raw = effect(values, y, groups)
    report = dict(names=names, raw=raw, adjusted={}, overlap={k: separation(v, y) for k, v in covariates.items()})
    for label, Z in covariates.items():
        report['adjusted'][label] = effect(adjust(values, Z, y), y, groups)
    f = lambda e: f"{e['standardized']:+6.2f} {e['positive']}/{e['units']}" if e['standardized'] is not None else '   n/a'
    print('overlap', report['overlap'])
    for k, n in enumerate(names):
        print(f'{n[:70]:70s}', f(raw[k]), *(f(report['adjusted'][c][k]) for c in covariates), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1, default=float) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--figure-data', type=Path, required=True, help='figure_data.py output for neurotycho (top z pairs)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--n-top', type=int, default=12)
    a = p.parse_args()
    main(a.root, a.figure_data, a.output, a.n_top)
