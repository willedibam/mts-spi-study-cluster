"""Heatmap of the full SPI-SPI feature matrix of a proof bank (recordings x SPI pairs).

python -m research.proof.p90.feature_matrix [--bank ...] [--spis 207] [--out results/proof/figure-261126]
"""
import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import leaves_list, linkage

from mtsspi.figures import SANS, SANS_PREAMBLE
from research.proof.p90.proof_figure import load_config
from research.proof.p90.refresh_proof_figures import load_bank

# Display order: the proof-figure classes, then the three not shown there, next to their family.
ORDER = ['gaussian-noise', 'cauchy-noise', 'var-self-0.7_total-0.2', 'var-self-0.2_total-0.7', 'var-self-0.2_total-0.2',
         'kuramoto_omega-fast', 'kuramoto_omega-slow', 'sti-i', 'fdstc', 'defect-turbulence', 'frozen-chaos', 'wave-1d']
EXTRA = {'var-self-0.2_total-0.2': ('VAR', 'weak', '#B9A0D6'), 'defect-turbulence': ('CML', 'defect turb.', '#5E7F2A'),
         'frozen-chaos': ('CML', 'frozen chaos', '#B7CF7E')}


def column_order(X, rank=40, seed=0):
    """Ward leaf order of the SPI-pair columns, on their (imputed, standardised) profiles reduced to ``rank`` dims."""
    Z = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    u, _, _ = np.linalg.svd(Z, full_matrices=False)
    return leaves_list(linkage(Z.T @ u[:, :rank], method='ward'))  # column scores on the top components


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--bank', default='data/proof/refresh-261126/features.npz')
    p.add_argument('--spis', type=int, default=207)
    p.add_argument('--out', default='results/proof/figure-261126')
    a = p.parse_args(argv)
    mpl.rcParams.update({'text.usetex': bool(shutil.which('latex') and shutil.which('dvipng')),
                         'text.latex.preamble': SANS_PREAMBLE, **SANS, 'font.size': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7, 'figure.facecolor': 'white',
                         'xtick.direction': 'out', 'ytick.direction': 'out'})
    bank = load_bank(a.bank, expected_spis=a.spis, contract='unified_ordered_v3')
    info = {c['name']: (c['family'], c['label'], c['color']) for c in load_config()['classes']} | EXTRA
    M, T = bank['M'].astype(int), bank['T'].astype(int)
    rows = np.lexsort((bank['instance'], T, M, [ORDER.index(y) for y in bank['y']]))
    X = bank['X'][rows]
    cols = column_order(X)
    y, M, T = bank['y'][rows], M[rows], T[rows]

    cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#BDBDBD')
    fig = plt.figure(figsize=(10, 6.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[.12, .06, .06, 12, .12], wspace=.03)
    ax = fig.add_subplot(gs[3])
    im = ax.imshow(np.ma.masked_invalid(X[:, cols]), aspect='auto', cmap=cmap, vmin=-1, vmax=1,
                   interpolation='antialiased', rasterized=True)
    ax.set_yticks([]); ax.set_xticks([0, X.shape[1] - 1], ['1', f'{X.shape[1]:,}'])
    ax.set_xlabel(f'SPI pair (Ward order on profiles across recordings; {X.shape[1]:,} pairs of {a.spis} SPIs)')
    for s in ax.spines.values(): s.set_visible(False)

    # Row annotations: class strip with labels, then M and T (sequential greys).
    strips = [(np.array([ORDER.index(v) for v in y]), ListedColormap([info[c][2] for c in ORDER]), 0, len(ORDER) - 1),
              (np.log2(M), 'Greys', 2, 5.5), (np.log2(T / 250), 'Greys', 0, 3.5)]
    for k, (v, cm, lo, hi) in enumerate(strips):
        s = fig.add_subplot(gs[k])
        s.imshow(v[:, None], aspect='auto', cmap=cm, vmin=lo, vmax=hi, interpolation='nearest')
        s.set_xticks([0], [['class', '$M$', '$T$'][k]], rotation=90); s.tick_params(length=0)
        for sp in s.spines.values(): sp.set_visible(False)
        s.set_yticks([])
    edges = np.flatnonzero(np.r_[True, y[1:] != y[:-1], True])
    first = fig.axes[1]
    first.set_yticks((edges[:-1] + edges[1:] - 1) / 2,
                     [f"{info[c][0]} ({info[c][1]})" for c in y[edges[:-1]]])
    for e in edges[1:-1]:
        ax.axhline(e - .5, color='white', lw=.6)
    cax = fig.add_subplot(gs[4])
    fig.colorbar(im, cax=cax, ticks=[-1, 0, 1]).set_label(r'$r$ between two SPIs across ordered channel pairs')
    cax.tick_params(length=2)
    fig.text(.5, -.01, f'{len(X):,} recordings: 12 classes $\\times$ $M\\in\\{{8,16,32\\}}$ $\\times$ '
             f'$T\\in\\{{500,1000,2000\\}}$ $\\times$ 10 instances (rows sorted by $M$, then $T$). '
             f'Grey: not finite ({100*(~np.isfinite(X)).mean():.1f}\\% of entries).', ha='center', fontsize=7)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [('png', 600), ('pdf', 600)]:
        fig.savefig(out / f'feature-matrix.{ext}', dpi=dpi, bbox_inches='tight')
    print(out / 'feature-matrix.png')


if __name__ == '__main__':
    main()
