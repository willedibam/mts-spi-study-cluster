"""Proof figure: one t-SNE of the displayed classes above fixed class-contrast rainclouds.

Settings, classes and contrasts live in ``configs/analysis/proof-figure-261029.yaml``.
Colour = class (families share a hue), marker area proportional to M, marker shape = T.
"""
from pathlib import Path
import json
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import yaml

from mtsspi.figures import raincloud
from research.cases.composite_style import panel_title, style
from research.proof.p90.refresh_proof_figures import ROOT, preprocess, rank_pair

CONFIG = ROOT/'research/proof/p90/configs/analysis/proof-figure-261029.yaml'
T_MARKERS = {500: '^', 1000: 's', 2000: 'o'}
# Matplotlib sizes a marker's bounding box; these factors roughly equalise drawn area.
SHAPE_AREA = {'^': 1.35, 's': .8, 'o': 1.}


def load_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text())


def marker_area(m, marker='o'):
    """Marker area in pt^2 at final size, proportional to M (8, 16, 32 -> 6, 12, 24 for circles)."""
    return .75*np.asarray(m, dtype=float)*SHAPE_AREA[marker]


def fit_embedding(bank, output, config):
    """Fit preprocessing, PCA50 and t-SNE on the displayed classes only; save and audit."""
    import openTSNE
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    names = [c['name'] for c in config['classes']]
    mask = np.isin(bank['y'], names); indices = np.flatnonzero(mask)
    Y, keep = preprocess(bank['X'][mask])
    scores = PCA(n_components=min(50, *Y.shape), svd_solver='randomized',
                 random_state=260924).fit_transform(Y)
    params = {k: v for k, v in config['tsne'].items() if k != 'library'}
    xy = np.asarray(openTSNE.TSNE(n_jobs=1, **params).fit(scores))
    # Is the class structure present before t-SNE? 5NN on PCA50 with each M/T cell held
    # out. Preprocessing/PCA see every cell, so this is not a fully held-out test.
    y, M, T = bank['y'][mask], bank['M'][mask], bank['T'][mask]
    predicted = np.empty_like(y)
    for m, t in sorted(set(zip(M, T))):
        test = (M == m) & (T == t)
        predicted[test] = KNeighborsClassifier(5).fit(scores[~test], y[~test]).predict(scores[test])
    audit = {'datasets': int(mask.sum()), 'classes': names, 'features': int(keep.sum()),
             'tsne': config['tsne'], 'pca50_knn5_cell_held_out_accuracy': float((predicted == y).mean())}
    np.savez_compressed(output/'proof-figure-tsne.npz', indices=indices, tsne=xy,
                        feature_indices=np.flatnonzero(keep))
    (output/'proof-figure-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    return {'indices': indices, 'tsne': xy, 'audit': audit}


def select_contrasts(bank, output, config):
    """Score each fixed SPI pair: training AUC (instances 0-5), display AUC (6-9)."""
    rows = []
    names = bank['pairs'].astype(str)
    for contrast in config['contrasts']:
        a, b = contrast['spis']
        column = np.flatnonzero(((names[:, 0] == a) & (names[:, 1] == b)) |
                                ((names[:, 0] == b) & (names[:, 1] == a)))
        if len(column) != 1:
            raise ValueError(f'Expected one coordinate for {a} × {b}.')
        row = rank_pair(dict(bank, X=bank['X'][:, column], pairs=names[column]), *contrast['classes'])
        row.update(feature_index=int(column[0]), label=contrast['label'])
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(output)/'proof-figure-contrasts.csv', index=False)
    return frame


def legend_label(c):
    return f"{c['family']} ({c['label']})"


def tick_label(c):
    """The legend label, family on its own line."""
    return c['family'] + '\n' + fill(f"({c['label']})", 13, break_long_words=False)


def _scatter_by_t(ax, x, y, M, T, color, style_kw):
    for t, marker in T_MARKERS.items():
        use = np.flatnonzero(T == t)
        use = use[np.argsort(-M[use], kind='stable')]  # Small markers drawn on top.
        ax.scatter(x[use], y[use], s=marker_area(M[use], marker), marker=marker, color=color,
                   rasterized=True, **style_kw)


def _encoding_legend(ax):
    """Inset key: area for M, shape for T."""
    grey = '.45'
    handles = [Line2D([], [], ls='', marker='o', color=grey, markeredgewidth=0,
                      markersize=np.sqrt(marker_area(m)), label=f'$M={m}$') for m in [8, 16, 32]]
    handles += [Line2D([], [], ls='', marker=mk, color=grey, markeredgewidth=0,
                       markersize=np.sqrt(marker_area(16, mk)), label=f'$T={t}$') for t, mk in T_MARKERS.items()]
    ax.legend(handles=handles, loc='best', ncols=2, fontsize=7, handletextpad=.3, columnspacing=.8,
              labelspacing=.3, borderaxespad=.4, handlelength=1.)


def _kde_island(ax, points, grid, color, islands):
    """Tinted region and thin contour where the class KDE exceeds the density at which
    ``mass`` of its own points lie above (a sample-based highest-density region)."""
    kde = gaussian_kde(points.T)
    level = np.quantile(kde(points.T), 1 - islands['mass'])
    density = kde(np.vstack([g.ravel() for g in grid])).reshape(grid[0].shape)
    ax.contourf(*grid, density, [level, np.inf], colors=[color], alpha=islands['fill_alpha'], zorder=0)
    ax.contour(*grid, density, [level], colors=[color], linewidths=islands['line_width'], zorder=0)


def _save(fig, output, name, dpi=600):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(output/f'{name}.{ext}', dpi=dpi, bbox_inches='tight', pad_inches=.03)


def plot_proof_figure(bank, embedding, contrasts, config, output):
    """Full page width (183 mm). a: square t-SNE with side margins left free for example MTS;
    b: class-contrast rainclouds. Font sizes follow the case composites at final size."""
    style()
    W, H = 7.2, 6.1
    fig = plt.figure(figsize=(W, H))
    inch = lambda left, bottom, w, h: fig.add_axes([left/W, bottom/H, w/W, h/H])
    classes = {c['name']: c for c in config['classes']}
    marker_kw = {'alpha': config['markers']['alpha'], 'edgecolors': config['markers']['edgecolor'],
                 'linewidths': config['markers']['linewidth']}
    side = 3.0
    ax = inch((W-side)/2, 3.0, side, side)
    indices = embedding['indices']; xy = embedding['tsne']
    y, M, T = bank['y'][indices], bank['M'][indices].astype(int), bank['T'][indices].astype(int)
    centre, half = (xy.max(0)+xy.min(0))/2, (xy.max(0)-xy.min(0)).max()/2*1.08
    lo, hi = centre-1.3*half, centre+1.3*half  # Grid beyond the view, so contours are never clipped.
    grid = np.meshgrid(np.linspace(lo[0], hi[0], 400), np.linspace(lo[1], hi[1], 400))
    for c in config['classes']:
        use = y == c['name']
        _kde_island(ax, xy[use], grid, c['color'], config['islands'])
        _scatter_by_t(ax, xy[use, 0], xy[use, 1], M[use], T[use], c['color'], marker_kw)
    ax.set(xlabel='t-SNE 1', ylabel='t-SNE 2', xticks=[], yticks=[],
           xlim=(centre[0]-half, centre[0]+half), ylim=(centre[1]-half, centre[1]+half))
    ax.set_box_aspect(1)
    _encoding_legend(ax)
    size, gap, left, bottom = 1.35, .45, .38, .95
    for k, (_, row) in enumerate(contrasts.iterrows()):
        rax = inch(left + k*(size+gap), bottom, size, size)
        col = int(row.feature_index)
        for pos, cls in enumerate([row.class_a, row.class_b]):
            mask = (bank['y'] == cls) & (bank['instance'] >= 6)
            markers = np.array([T_MARKERS[v] for v in bank['T'][mask].astype(int)])
            areas = np.array([marker_area(m, mk) for m, mk in zip(bank['M'][mask].astype(int), markers)])
            raincloud(rax, bank['X'][mask, col], pos, classes[cls]['color'], np.random.default_rng(260924+col+pos),
                      width=.65, box=True, point_sizes=areas, point_markers=markers, point_kw=marker_kw)
        values = bank['X'][(bank['instance'] >= 6) & np.isin(bank['y'], [row.class_a, row.class_b]), col]
        values = values[np.isfinite(values)]; pad = max(np.ptp(values)*.08, .0001)
        rax.set(ylim=(values.min()-pad, values.max()+pad), xlim=(-.5, 1.5), xticks=[0, 1],
                xticklabels=[tick_label(classes[c]) for c in [row.class_a, row.class_b]])
        rax.set_title(row.label, pad=4, fontsize=7)
        rax.tick_params(axis='x', labelsize=6.5)
        if k == 0: rax.set_ylabel(r'$z_{ab}$')
        rax.yaxis.set_major_locator(MaxNLocator(3))
        rax.ticklabel_format(axis='y', style='plain', useOffset=False)
        rax.set_box_aspect(1)
    # Four columns, filled column-wise: the single-member family joins the first pair.
    family = [c['family'] for c in config['classes']]
    single = [c for c in config['classes'] if family.count(c['family']) == 1]
    paired = [c for c in config['classes'] if family.count(c['family']) > 1]
    order = paired[:2] + single + paired[2:]
    fig.legend(handles=[Line2D([], [], ls='', marker='o', markersize=4, markeredgewidth=0, color=c['color'],
                               label=legend_label(c)) for c in order],
               loc='lower center', bbox_to_anchor=(.5, 0), ncols=4, fontsize=7,
               columnspacing=1.6, handletextpad=.3, labelspacing=.35)
    panel_title(fig, .05, .15, 'a', '')
    panel_title(fig, .05, H-bottom-size-.3, 'b', '')
    _save(fig, output, 'proof-figure')
    return fig
