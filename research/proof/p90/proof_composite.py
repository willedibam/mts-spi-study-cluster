"""Two embedding views above three illustrative, training-selected coordinates."""
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from mtsspi.figures import raincloud, save, style
from research.proof.p90.refresh_proof_figures import CML, display_name, rank_pair


DEFAULT_PAIRS = [('sti-i', 'defect-turbulence'), ('kuramoto_omega-fast', 'fdstc'),
                 ('kuramoto_omega-fast', 'kuramoto_omega-slow')]
TICK_NAMES = {'sti-i':'Intermittency', 'defect-turbulence':'Defect\nturbulence',
              'fdstc':'Fully developed\nturbulence',
              'kuramoto_omega-fast':'Kuramoto: fast\nfrequencies',
              'kuramoto_omega-slow':'Kuramoto: slow\nfrequencies'}


def marker_areas(m):
    """Matplotlib s is marker area in points squared: encode sqrt(M) explicitly."""
    return 12 * np.sqrt(np.asarray(m, dtype=float) / 8)


def short_name(name):
    if name.startswith('var-self-'):
        self_weight, total = name.removeprefix('var-self-').split('_total-')
        return f'VAR (self {self_weight}, total {total})'
    return display_name(name)


def select_examples(bank, output, *, pairs=DEFAULT_PAIRS, feature_pairs=None):
    if len(pairs) != 3:
        raise ValueError('The composite requires exactly three class contrasts.')
    feature_pairs = [None]*3 if feature_pairs is None else feature_pairs
    if len(feature_pairs) != 3:
        raise ValueError('Provide one optional SPI pair per class contrast.')
    rows = []
    for (a, b), fixed in zip(pairs, feature_pairs):
        names = bank['pairs'].astype(str)
        if fixed is not None:
            columns = np.flatnonzero(np.all(names == np.asarray(fixed), axis=1))
            if len(columns) != 1: raise ValueError(f'Expected one coordinate for {fixed}.')
            family = 'fixed SPI pair'
        elif a.startswith('kuramoto_') and b.startswith('kuramoto_'):
            # A focused temporal interpretation: agreement between Pearson profiles
            # at different lags. Freeze this family before examining display rows.
            eligible = np.all(np.char.startswith(names, 'corr_pearson_tau-') &
                              ~np.char.endswith(names, '-sq'), axis=1)
            columns = np.flatnonzero(eligible)
            if not len(columns): raise ValueError('No unsquared Pearson lag-pair coordinates.')
            family = 'unsquared Pearson lag pairs'
        else:
            columns = np.arange(len(names))
            family = 'all finite training coordinates'
        focused = dict(bank, X=bank['X'][:, columns], pairs=bank['pairs'][columns])
        row = rank_pair(focused, a, b)
        row['feature_index'] = int(columns[row['feature_index']])
        row['candidate_family'] = family
        rows.append(row)
    result = pd.DataFrame(rows)
    Path(output).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(output) / 'composite-selected-features.csv', index=False)
    return result


def plot_composite(bank, coords, selected, method, output):
    if method not in {'pca', 'tsne'} or len(selected) != 3:
        raise ValueError('Use PCA or t-SNE with exactly three selected contrasts.')
    style()
    # Inch-based geometry: 3.25-inch embeddings, 2.20-inch rainclouds, with
    # the lower row extending 0.65 inches beyond each upper side.
    fig = plt.figure(figsize=(9, 7.38))
    upper = [fig.add_axes([left/9, 3.88/7.38, 3.25/9, 3.25/7.38]) for left in [.95, 4.80]]
    lower = [fig.add_axes([left/9, 1/7.38, 2.2/9, 2.2/7.38]) for left in [.30, 3.40, 6.50]]
    for ax, panel, title in zip(upper, ['inter', 'cml'], ['Inter-class', 'Within CML']):
        indices = coords[panel+'_indices']; labels = bank['y'][indices]
        xy = coords[panel+'_'+method]; sizes = marker_areas(bank['M'][indices])
        for cls in sorted(set(labels)):
            use = labels == cls
            ax.scatter(xy[use, 0], xy[use, 1], s=sizes[use], color=bank['colors'][cls],
                       alpha=.45, linewidths=0, rasterized=True)
        ax.set(title=title); ax.set_box_aspect(1)
        ax.spines[['top', 'right']].set_visible(True)
        if method == 'pca':
            variance = coords[panel+'_variance']
            ax.set(xlabel=f'PC1 ({variance[0]:.1%})', ylabel=f'PC2 ({variance[1]:.1%})')
            ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.tick_params(labelsize=8)
        else:
            ax.set(xlabel='t-SNE 1', ylabel='t-SNE 2', xticks=[], yticks=[])
    size_handles = [Line2D([], [], marker='o', ls='', color='.45', markersize=np.sqrt(marker_areas(m)),
                           markeredgewidth=0, label=f'M={m}') for m in [8, 16, 32]]
    upper[1].legend(handles=size_handles, loc='upper left', fontsize=7,
                    handletextpad=.4, borderaxespad=.6, labelspacing=.35)
    for ax, (_, row) in zip(lower, selected.iterrows()):
        col = int(row.feature_index)
        for pos, cls in enumerate([row.class_a, row.class_b]):
            mask = (bank['y']==cls) & (bank['instance']>=6)
            values = bank['X'][mask, col]
            raincloud(ax, values, pos, bank['colors'][cls], np.random.default_rng(260924+col+pos),
                      width=.65, box=True, point_sizes=marker_areas(bank['M'][mask]))
        values = bank['X'][(bank['instance']>=6) & np.isin(bank['y'], [row.class_a, row.class_b]), col]
        values = values[np.isfinite(values)]; pad = max(np.ptp(values)*.08, .0001)
        ax.set(ylim=(values.min()-pad, values.max()+pad),
               xlim=(-.6, 1.6), xticks=[0, 1],
               xticklabels=[TICK_NAMES.get(c, fill(short_name(c), 19)) for c in [row.class_a, row.class_b]],
               ylabel=r'$z_{ab}$', title=f'{row.spi_a}\n× {row.spi_b}')
        ax.title.set_fontsize(7); ax.tick_params(axis='x', labelsize=7, length=2)
        ax.yaxis.set_major_locator(MaxNLocator(2)); ax.tick_params(axis='y', labelsize=7, length=2)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        ax.spines[['top', 'right']].set_visible(True); ax.set_box_aspect(1)
    order = [c for c in sorted(set(bank['y'])) if c not in CML] + CML
    handles = [Line2D([], [], marker='o', ls='', color=bank['colors'][c], markersize=4,
                      markeredgewidth=0, label=short_name(c)) for c in order]
    fig.legend(handles=handles, loc='lower center', ncols=4, fontsize=6.5,
               bbox_to_anchor=(.5, .005), columnspacing=1.3, handletextpad=.4)
    output = Path(output); name = f'composite-{method}'
    save(fig, output, name); fig.savefig(output / f'{name}.pdf')
    return fig
