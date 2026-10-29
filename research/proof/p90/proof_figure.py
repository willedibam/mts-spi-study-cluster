"""Proof figure. a: one t-SNE of the displayed classes, ringed by one silo per generator family
with carpets of the recordings behind ringed points; b: fixed class-contrast rainclouds.

Settings, classes, contrasts and panel-a layout live in ``configs/analysis/proof-figure-261029.yaml``.
Colour = class (a family's variants take neighbouring hues), marker area proportional to M,
marker shape = T.
"""
from pathlib import Path
import json
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import yaml

from mtsspi.figures import raincloud
from research.cases.composite_style import INK, carpet, panel_title, style
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


def select_examples(bank, embedding, config, output=None, sizes=None, all_instances=False):
    """Per class and M/T cell (default: the diagonal in ``panel_a.sizes``): the display instance
    (6-9) nearest its cell's t-SNE medoid, regenerated from the frozen generation config.
    ``series`` is the whole recording (channels x T); ``carpet`` its first T/5 samples.
    ``all_instances`` returns every instance (0-9) of each cell instead of the medoid."""
    from mtsspi.generators.harness import generate_from_spec
    from mtsspi.mapping import DatasetMapping, ExperimentConfig
    a = config['panel_a']
    specs = DatasetMapping(ExperimentConfig.from_file(ROOT/a['generation'])).specs
    indices, xy = embedding['indices'], embedding['tsne']
    rows = []
    for c in config['classes']:
        for m, t in sizes or a['sizes']:
            cell = np.flatnonzero((bank['y'][indices] == c['name']) & (bank['M'][indices] == m) &
                                  (bank['T'][indices] == t) & (all_instances | (bank['instance'][indices] >= 6)))
            if all_instances:
                chosen = cell[np.argsort(bank['instance'][indices[cell]].astype(int))]
            else:
                chosen = [cell[np.argmin(np.linalg.norm(xy[cell, None] - xy[None, cell], axis=-1).sum(1))]]
            for k in chosen:
                instance = int(bank['instance'][indices[k]])
                spec = next(sp for sp in specs if sp.mts_class == c['name'] and sp.M == m and sp.T == t
                            and sp.instance == instance)
                x = np.asarray(generate_from_spec(spec)[0], dtype=float)
                x = x if x.shape == (t, m) else x.T
                rows.append({'name': c['name'], 'M': m, 'T': t, 'instance': instance, 'rng_seed': spec.rng_seed,
                             'embedding_row': int(k), 'series': x.T, 'carpet': x[:int(t*a['window_fraction'])].T})
    if output is not None:
        pd.DataFrame([{k: v for k, v in r.items() if k not in ('carpet', 'series')} for r in rows]).to_csv(
            Path(output)/'proof-figure-examples.csv', index=False)
    return rows


def legend_label(c):
    return f"{c['family']} ({c['label']})"


def tick_label(c):
    """The legend label, family on its own line."""
    return c['family'] + '\n' + fill(f"({c['label']})", 13, break_long_words=False)


def _scatter_by_t(ax, x, y, M, T, color, style_kw):
    for t, marker in T_MARKERS.items():
        use = np.flatnonzero(T == t)
        use = use[np.argsort(-M[use], kind='stable')]  # Small markers drawn on top.
        ax.scatter(x[use], y[use], s=marker_area(M[use], marker), marker=marker, color=color, **style_kw)


def _encoding_legend(ax, loc='best'):
    """Inset key: area for M, shape for T."""
    grey = '.45'
    handles = [Line2D([], [], ls='', marker='o', color=grey, markeredgewidth=0,
                      markersize=np.sqrt(marker_area(m)), label=f'$M={m}$') for m in [8, 16, 32]]
    handles += [Line2D([], [], ls='', marker=mk, color=grey, markeredgewidth=0,
                       markersize=np.sqrt(marker_area(16, mk)), label=f'$T={t}$') for t, mk in T_MARKERS.items()]
    ax.legend(handles=handles, loc=loc, ncols=2, fontsize=7, handletextpad=.3, columnspacing=.8,
              labelspacing=.3, borderaxespad=.4, handlelength=1.)


def _island(ax, points, grid, color, islands):
    """Tinted region and thin contour where the class KDE exceeds the density at which
    ``mass`` of its own points lie above (a sample-based highest-density region).
    Returns the contour vertices, which the silo leaders end on."""
    kde = gaussian_kde(points.T)
    level = np.quantile(kde(points.T), 1 - islands['mass'])
    density = kde(np.vstack([g.ravel() for g in grid])).reshape(grid[0].shape)
    ax.contourf(*grid, density, [level, np.inf], colors=[color], alpha=islands['fill_alpha'], zorder=0)
    lines = ax.contour(*grid, density, [level], colors=[color], linewidths=islands['line_width'], zorder=0)
    return lines.get_paths()[0].vertices


def _crisp(ax):
    """Store a carpet's samples unresampled (one image pixel per sample and channel) and mark them
    for nearest-neighbour scaling, so blocks stay sharp at any zoom in vector output."""
    for image in ax.images:
        image.set_interpolation('none'); image.set_rasterized(False)


def _save(fig, output, name, dpi=600):
    """PDF and PNG on the fixed canvas; the SVG is converted from the PDF. Matplotlib's own usetex
    SVG typesets text at 100 pt (cmr17 letterforms) then shrinks it, so text comes out narrower than
    the space TeX laid out at its real size and centred titles shift left."""
    import shutil, subprocess, warnings
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(output/f'{name}.{ext}', dpi=dpi)  # Fixed canvas: exactly the designed width.
    if shutil.which('pdftocairo'):
        subprocess.run(['pdftocairo', '-svg', output/f'{name}.pdf', output/f'{name}.svg'], check=True)
        svg = output/f'{name}.svg'  # The only images are carpets: keep their blocks sharp when scaled.
        svg.write_text(svg.read_text().replace('<image ', '<image style="image-rendering:pixelated" '))
    else:
        warnings.warn('pdftocairo not found: matplotlib SVG text is narrower than the PDF/PNG.')
        fig.savefig(output/f'{name}.svg', dpi=dpi)


# --------------------------------------------------------------------------- panel a geometry
# Inches. Carpets share one ruler: height per channel and width per sample.
SILO_W, PAD, HEAD, NAME, DIVIDE, COL_GAP = 1.95, .07, .2, .13, .09, .1
CH_IN = .0095


def _silo_heights(config):
    tallest = max(m for m, _ in config['panel_a']['sizes'])*CH_IN
    block = NAME + tallest
    count = {f: sum(c['family'] == f for c in config['classes']) for c in config['classes'] for f in [c['family']]}
    return {f: 2*PAD + HEAD + n*block + (n-1)*DIVIDE for f, n in count.items()}, block


def _column_height(names, heights):
    return sum(heights[f] for f in names) + COL_GAP*(len(names)-1)


def _members(config, family, order=None):
    members = [c for c in config['classes'] if c['family'] == family]
    if order and family in order:
        members = sorted(members, key=lambda c: order[family].index(c['name']))
    return members


def _geometry(config, columns, legend, order=None, silos=True):
    """Figure inches for a given silo arrangement (and optional variant order within silos).
    Without silos the embedding sits just above panel b, leaving the space above free."""
    heights, block = _silo_heights(config)
    W = 183/25.4  # Nature double-column width.
    side, gap, left, right = 2.8, .3, .4, .09
    size = (W - left - right - 3*gap)/4
    Ha = max(side + .3, *(_column_height(columns[k], heights) for k in columns))
    bottom = (.42 if legend else 0.) + .55
    ya = bottom + size + .6
    H = ya + Ha + .25  # Top margin keeps the panel letter clear of the silos.
    g = dict(W=W, H=H, Ha=Ha, ya=ya, side=side, size=size, gap=gap, left=left, bottom=bottom, block=block,
             ex0=(W-side)/2, ey0=ya + (Ha-side)/2 + .1 if silos else bottom + size + .5)
    rects, anchors = {}, {}
    for key, x in [('left', 0.), ('right', W - SILO_W)]:
        top = ya + (Ha + _column_height(columns[key], heights))/2
        for f in columns[key]:
            h = heights[f]; top -= h
            rects[f] = (x, top, SILO_W, h)
            for i, c in enumerate(_members(config, f, order)):
                yc = top + h - PAD - HEAD - i*(block + DIVIDE) - block/2
                anchors[c['name']] = (x + SILO_W if key == 'left' else x, yc)
            top -= COL_GAP
    return dict(g, rects=rects, anchors=anchors)


def _orient(xy, angle, reflect):
    q = xy - xy.mean(0)
    if reflect: q = q*[-1, 1]
    c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    return q @ np.array([[c, s], [-s, c]])


def _frame(q, g):
    """Oriented t-SNE coordinates -> figure inches (square limits, 8% margin)."""
    centre, half = (q.max(0)+q.min(0))/2, (q.max(0)-q.min(0)).max()/2*1.08
    return lambda p: np.array([g['ex0'], g['ey0']]) + (np.asarray(p) - (centre - half))/(2*half)*g['side']


def _crosses(p, q, r, s):
    d = lambda a, b, c: (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    return d(p, q, r)*d(p, q, s) < 0 and d(r, s, p)*d(r, s, q) < 0


def _leader_cost(anchors, cen, rad, pts, owner):
    """Leader length to the island edge, plus penalties for leader crossings and for other
    classes' points lying under a leader (within 0.06 in)."""
    v = cen - anchors; L = np.linalg.norm(v, axis=1)
    end = cen - v/L[:, None]*rad[:, None]
    cost = (L - rad).sum()
    for i in range(len(anchors)):
        for j in range(i+1, len(anchors)):
            cost += 5*_crosses(anchors[i], end[i], anchors[j], end[j])
        w = end[i] - anchors[i]; other = owner != i
        t = np.clip((pts[other] - anchors[i]) @ w / (w @ w), 0, 1)
        near = np.linalg.norm(anchors[i] + t[:, None]*w - pts[other], axis=1) < .06
        cost += .5*near.sum()
    return cost


def choose_layout(xy, labels, config, *, step=6, legend=False, keep=60):
    """Search silo sides/orders and the rotation/reflection of the (arbitrary) t-SNE axes for
    short leaders that neither cross nor run over other islands, with a free corner for the
    inset key; then the variant order inside each silo. Freeze the result in the YAML."""
    from itertools import combinations, permutations, product
    families = list(dict.fromkeys(c['family'] for c in config['classes']))
    names = [c['name'] for c in config['classes']]
    owner = np.array([names.index(n) for n in labels])

    def cost(columns, order, angle, reflect):
        g = _geometry(config, columns, legend, order)
        anchors = np.array([g['anchors'][n] for n in names])
        q = _orient(xy, angle, reflect); pts = _frame(q, g)(q)
        cen = np.array([pts[owner == i].mean(0) for i in range(len(names))])
        rad = np.array([np.sqrt(((pts[owner == i] - cen[i])**2).sum(1).mean())*1.6 for i in range(len(names))])
        return _leader_cost(anchors, cen, rad, pts, owner) + _key_corner(pts, g)[1]

    scored = []
    for k in (2, 3):
        for chosen in combinations(families, k):
            rest = [f for f in families if f not in chosen]
            for lo, ro in product(permutations(chosen), permutations(rest)):
                columns = {'left': list(lo), 'right': list(ro)}
                for reflect in (False, True):
                    for angle in range(0, 360, step):
                        scored.append((cost(columns, None, angle, reflect), columns, angle, reflect))
    scored.sort(key=lambda r: r[0])
    pairs = [f for f in families if sum(c['family'] == f for c in config['classes']) == 2]
    best = None
    for _, columns, angle, reflect in scored[:keep]:
        for flips in product((False, True), repeat=len(pairs)):
            order = {f: [c['name'] for c in _members(config, f)][::-1 if flip else 1] for f, flip in zip(pairs, flips)}
            c = cost(columns, order, angle, reflect)
            if best is None or c < best[0]: best = (c, columns, order, angle, reflect)
    c, columns, order, angle, reflect = best
    return {'columns': columns, 'variant_order': order, 'rotation_deg': angle, 'reflect_x': reflect, 'cost': float(c)}


KEY_W, KEY_H = 1.35, .55


def _corner_boxes(g):
    x0, y0, s = g['ex0'], g['ey0'], g['side']
    return {'upper left': (x0, y0+s-KEY_H), 'upper right': (x0+s-KEY_W, y0+s-KEY_H),
            'lower left': (x0, y0), 'lower right': (x0+s-KEY_W, y0)}


def _key_corner(points_in, g):
    """Corner of the embedding with the fewest points under the inset key, and that count."""
    count = {k: int(((points_in[:, 0] > b[0]) & (points_in[:, 0] < b[0]+KEY_W) &
                     (points_in[:, 1] > b[1]) & (points_in[:, 1] < b[1]+KEY_H)).sum())
             for k, b in _corner_boxes(g).items()}
    best = min(count, key=count.get)
    return best, count[best]


def _glyph(fig, x, y, w, h, kind, W, H, color):
    """Coupling topology of the generator, drawn small: six nodes."""
    ax = fig.add_axes([x/W, y/H, w/W, h/H]); ax.set_axis_off(); ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    n = 6
    if kind == 'chain':
        pts = np.column_stack([np.linspace(-1, 1, n), .25*np.sin(np.linspace(0, np.pi, n))])
        edges = [(i, i+1) for i in range(n-1)]
    else:
        ang = np.pi/2 + 2*np.pi*np.arange(n)/n; pts = np.column_stack([np.cos(ang), np.sin(ang)])
        edges = {'none': [], 'ring': [(i, (i+1) % n) for i in range(n)],
                 'complete': [(i, j) for i in range(n) for j in range(i+1, n)]}[kind]
    for i, j in edges:
        ax.plot(*pts[[i, j]].T, color=color, lw=.45, solid_capstyle='round', zorder=1)
    for p in pts:
        ax.add_patch(Circle(p, .2, fc='white', ec=color, lw=.5, zorder=2))


def _silo(fig, rect, family, members, examples, block, W, H, kind):
    x, y, w, h = rect
    from matplotlib.colors import to_rgb
    hue = min((c['color'] for c in members), key=lambda h: sum(to_rgb(h)))  # The darker variant names the family.
    fig.add_artist(FancyBboxPatch((x/W, y/H), w/W, h/H, transform=fig.transFigure,
                                  boxstyle='round,pad=0,rounding_size=0.012', fc='none', ec=hue, lw=.5))
    fig.text((x+PAD)/W, (y+h-PAD-.13)/H, family, color=hue, fontsize=8, va='baseline')
    _glyph(fig, x+w-PAD-.3, y+h-PAD-.19, .3, .19, kind, W, H, hue)
    for i, c in enumerate(members):
        top = y + h - PAD - HEAD - i*(block + DIVIDE)
        if i:  # Subtle internal split between variants.
            yd = top + DIVIDE/2
            fig.add_artist(Line2D([(x+PAD)/W, (x+w-PAD)/W], [yd/H, yd/H], transform=fig.transFigure,
                                  color=hue, lw=.4, ls=(0, (1, 1.5)), alpha=.7))
        fig.text((x+PAD)/W, (top-.1)/H, c['label'], color=c['color'], fontsize=7, va='baseline')
        cells = [e for e in examples if e['name'] == c['name']]
        width = w - 2*PAD; gaps = .05*(len(cells)-1)
        per_sample = (width - gaps)/sum(e['carpet'].shape[1] for e in cells)
        cx = x + PAD
        for e in cells:
            cw, ch = e['carpet'].shape[1]*per_sample, e['carpet'].shape[0]*CH_IN
            cax = fig.add_axes([cx/W, (top-block)/H, cw/W, ch/H])
            carpet(cax, e['carpet']); _crisp(cax)
            for spine in cax.spines.values(): spine.set_linewidth(.4)
            cx += cw + .05


def plot_proof_figure(bank, embedding, contrasts, examples, config, output, *, legend=True, silos=True,
                      rings=True, title_tt=False, sans=False, name='proof-figure'):
    """Full page width (183 mm). a: square t-SNE with family silos; b: class-contrast rainclouds.
    Font sizes follow the case composites at final size. ``sans`` sets text in Helvetica (TeX
    ``helvet``), keeping math, and tick numbers as text, in Computer Modern math only where $...$."""
    style()
    if sans: plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica']})
    a = config['panel_a']
    order = a.get('variant_order')
    g = _geometry(config, a['columns'], legend, order, silos)
    W, H, side, size, gap, left, bottom = (g[k] for k in ['W', 'H', 'side', 'size', 'gap', 'left', 'bottom'])
    classes = {c['name']: c for c in config['classes']}
    fig = plt.figure(figsize=(W, H))
    inch = lambda l, b, w, h: fig.add_axes([l/W, b/H, w/W, h/H])
    ax = inch(g['ex0'], g['ey0'], side, side)
    marker_kw = {'alpha': config['markers']['alpha'], 'edgecolors': config['markers']['edgecolor'],
                 'linewidths': config['markers']['linewidth']}
    indices = embedding['indices']
    y, M, T = bank['y'][indices], bank['M'][indices].astype(int), bank['T'][indices].astype(int)
    xy = _orient(embedding['tsne'], a['rotation_deg'], a['reflect_x'])
    to_in = _frame(xy, g)
    centre, half = (xy.max(0)+xy.min(0))/2, (xy.max(0)-xy.min(0)).max()/2*1.08
    lo, hi = centre-1.3*half, centre+1.3*half  # Grid beyond the view, so contours are never clipped.
    grid = np.meshgrid(np.linspace(lo[0], hi[0], 400), np.linspace(lo[1], hi[1], 400))
    contours = {}
    for c in config['classes']:
        use = y == c['name']
        contours[c['name']] = _island(ax, xy[use], grid, c['color'], config['islands'])
        _scatter_by_t(ax, xy[use, 0], xy[use, 1], M[use], T[use], c['color'], marker_kw)
    rows = np.array([e['embedding_row'] for e in examples])
    if rings: ax.scatter(xy[rows, 0], xy[rows, 1], s=marker_area(M[rows])*2.4, facecolors='none', edgecolors=INK,
               linewidths=.45, zorder=4)
    # Axes are rotated (arbitrary in t-SNE anyway), so one plain label replaces numbered axes.
    ax.set(xticks=[], yticks=[], xlim=(centre[0]-half, centre[0]+half), ylim=(centre[1]-half, centre[1]+half))
    ax.set_xlabel('t-SNE', labelpad=3)
    ax.set_box_aspect(1)
    _encoding_legend(ax, _key_corner(np.vstack([to_in(xy)] + [to_in(v) for v in contours.values()]), g)[0])
    for f, rect in (g['rects'] if silos else {}).items():
        members = _members(config, f, order)
        _silo(fig, rect, f, members, examples, g['block'], W, H, a['topology'][f])
        for c in members:  # Leader: variant anchor -> nearest point on its island contour.
            p0 = np.array(g['anchors'][c['name']]); pts = to_in(contours[c['name']])
            p1 = pts[np.argmin(np.linalg.norm(pts - p0, axis=1))]
            fig.add_artist(Line2D([p0[0]/W, p1[0]/W], [p0[1]/H, p1[1]/H], transform=fig.transFigure,
                                  color=c['color'], lw=.5, alpha=.8))
    for k, (_, row) in enumerate(contrasts.iterrows()):
        rax = inch(left + k*(size+gap), bottom, size, size)
        col = int(row.feature_index)
        for pos, cls in enumerate([row.class_a, row.class_b]):
            mask = (bank['y'] == cls) & (bank['instance'] >= 6)
            markers = np.array([T_MARKERS[v] for v in bank['T'][mask].astype(int)])
            areas = np.array([marker_area(m, mk) for m, mk in zip(bank['M'][mask].astype(int), markers)])
            raincloud(rax, bank['X'][mask, col], pos, classes[cls]['color'], np.random.default_rng(260924+col+pos),
                      width=.65, box=True, point_sizes=areas, point_markers=markers, point_kw=marker_kw,
                      outline=True, line_scale=.7)
        values = bank['X'][(bank['instance'] >= 6) & np.isin(bank['y'], [row.class_a, row.class_b]), col]
        values = values[np.isfinite(values)]; pad = max(np.ptp(values)*.08, .0001)
        rax.set(ylim=(values.min()-pad, values.max()+pad), xlim=(-.5, 1.5), xticks=[0, 1],
                xticklabels=[tick_label(classes[c]) for c in [row.class_a, row.class_b]])
        rax.set_title(rf'\texttt{{{row.label}}}' if title_tt else row.label, pad=4, fontsize=6.5 if title_tt else 7)
        rax.tick_params(axis='x', labelsize=6.5)
        if k == 0: rax.set_ylabel(r'$z_{ab}$')
        rax.yaxis.set_major_locator(MaxNLocator(3))
        rax.ticklabel_format(axis='y', style='plain', useOffset=False)
        if sans:  # Tick numbers are text, not math: Helvetica digits and minus.
            from matplotlib.ticker import FuncFormatter
            rax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.1f}'.replace('-', r'\textminus ')))
        rax.set_box_aspect(1)
    if legend:  # Four columns, filled column-wise: the single-member family joins the first pair.
        family = [c['family'] for c in config['classes']]
        single = [c for c in config['classes'] if family.count(c['family']) == 1]
        paired = [c for c in config['classes'] if family.count(c['family']) > 1]
        order = paired[:2] + single + paired[2:]
        fig.legend(handles=[Line2D([], [], ls='', marker='o', markersize=4, markeredgewidth=0, color=c['color'],
                                   label=legend_label(c)) for c in order],
                   loc='lower center', bbox_to_anchor=(.5, 0), ncols=4, fontsize=7,
                   columnspacing=1.6, handletextpad=.3, labelspacing=.35)
    panel_title(fig, .02, .15, 'a', '')
    panel_title(fig, .02, H-bottom-size-.3, 'b', '')
    _save(fig, output, name)
    return fig


# --------------------------------------------------------------------------- draw.io exports
def _generation(config):
    """Generator parameters per class, from the frozen generation config."""
    gen = yaml.safe_load((ROOT/config['panel_a']['generation']).read_text())
    return {c['name']: c['base_params'] for c in gen['mts_classes']}


def carpet_size(m, t, h0=.15, w0=.315):
    """Square-root scaling (inches): M=32 is twice as tall as M=8, T=2000 twice as wide as T=500."""
    return w0*np.sqrt(t/500), h0*np.sqrt(m/8)


def _carpet_views(name, m, t, cfg):
    """(suffix, stride, samples) per exported view of one recording, per ``panel_a.carpets``."""
    w, _ = carpet_size(m, t, cfg['h0'], cfg['w0']); n = int(round(w*cfg['per_in']))
    stride = cfg.get('stride', {}).get(name, 1)
    views = [(f'_stride{stride}' if stride > 1 else '', stride, n)]
    if name in cfg.get('whole', []):
        views.append(('_whole', int(np.ceil(t/n)), n))
    return views


def export_carpets(rows, config, output, *, sheet=True):
    """One SVG per recording and view at final size (all M channels; see ``panel_a.carpets``),
    a folder per class with a contact sheet, and ``index.csv`` (with ``embedding_row``)."""
    style()
    output = Path(output); cfg = config['panel_a']['carpets']; index = []
    pad = .4/72/2  # Half the frame line, so the frame is not clipped at the canvas edge.
    for i, c in enumerate(config['classes']):
        folder = output/f"{i+1:02d}-{c['name']}"; folder.mkdir(parents=True, exist_ok=True)
        for r in (r for r in rows if r['name'] == c['name']):
            w, h = carpet_size(r['M'], r['T'], cfg['h0'], cfg['w0'])
            for suffix, stride, n in _carpet_views(c['name'], r['M'], r['T'], cfg):
                x = r['series'][:, ::stride][:, :n]
                stem = f"{c['name']}_M{r['M']}_T{r['T']}_i{r['instance']}{suffix}"
                fig = plt.figure(figsize=(w + 2*pad, h + 2*pad))
                ax = fig.add_axes([pad/(w+2*pad), pad/(h+2*pad), w/(w+2*pad), h/(h+2*pad)])
                carpet(ax, x); _crisp(ax)
                for spine in ax.spines.values(): spine.set_linewidth(.4)
                fig.savefig(folder/f'{stem}.svg', dpi=600); plt.close(fig)
                index.append({'file': f'{folder.name}/{stem}.svg', 'name': c['name'], 'M': r['M'], 'T': r['T'],
                              'instance': r['instance'], 'view': suffix.lstrip('_') or 'clip', 'stride': stride,
                              'samples_shown': x.shape[1], 'samples_spanned': x.shape[1]*stride,
                              'width_in': round(w, 4), 'height_in': round(h, 4), 'embedding_row': r['embedding_row']})
        if sheet: _contact_sheet([r for r in rows if r['name'] == c['name']], c, cfg, folder)
    index = pd.DataFrame(index); index.to_csv(output/'index.csv', index=False)
    return index


def _contact_sheet(rows, c, cfg, folder):
    """Every exported view of one class at true size: rows = instance (and view), columns = M/T cell."""
    cells = sorted({(r['M'], r['T']) for r in rows}, key=lambda mt: (mt[1], mt[0]))
    lines = sorted({(r['instance'], v[0]) for r in rows for v in _carpet_views(c['name'], r['M'], r['T'], cfg)})
    big = carpet_size(32, 2000, cfg['h0'], cfg['w0']); colw, rowh, lab = big[0] + .12, big[1] + .12, .9
    W, H = lab + len(cells)*colw, .7 + len(lines)*rowh
    fig = plt.figure(figsize=(W, H)); series = {(r['M'], r['T'], r['instance']): r['series'] for r in rows}
    fig.text(.05/W, 1 - .2/H, legend_label(c), fontsize=8, color=c['color'])
    for j, (m, t) in enumerate(cells):
        fig.text((lab + j*colw)/W, 1 - .5/H, f'$M={m}$\n$T={t}$', fontsize=6.5)
    for i, (inst, suffix) in enumerate(lines):
        y = H - .7 - (i+1)*rowh + .06
        fig.text(.05/W, (y + .1)/H, f'$i={inst}$' + (' whole' if suffix == '_whole' else ''), fontsize=6.5)
        for j, (m, t) in enumerate(cells):
            stride, n = next((s, n) for sf, s, n in _carpet_views(c['name'], m, t, cfg) if sf == suffix)
            w, h = carpet_size(m, t, cfg['h0'], cfg['w0'])
            carpet(fig.add_axes([(lab + j*colw)/W, y/H, w/W, h/H]), series[m, t, inst][:, ::stride][:, :n])
    fig.savefig(folder/'contact-sheet.png', dpi=250); plt.close(fig)


def _weighted_glyph(ax, kind, self_w, edge_w, color, repulsive=False):
    """Coupling graph with line width proportional to weight: self-loops for the weight on a
    node's own past, ring edges for the weight on each neighbour (near-zero edges dotted).
    All-to-all coupling is drawn through its mean field (a hub), dashed when repulsive."""
    n, r, loop = 6, .2, .22
    ang = np.pi/2 + 2*np.pi*np.arange(n)/n; pts = np.column_stack([np.cos(ang), np.sin(ang)])
    lw = lambda w: .25 + 1.1*w
    tiny = edge_w is not None and edge_w < .01
    edge_kw = dict(color=color, solid_capstyle='round', zorder=1,
                   lw=.4 if edge_w is None or tiny else lw(edge_w),
                   ls=(0, (1, 1.2)) if tiny else (0, (2.2, 1.4)) if repulsive else '-')
    if kind == 'ring':
        for i in range(n): ax.plot(*pts[[i, (i+1) % n]].T, **edge_kw)
    elif kind == 'complete':
        for q in pts: ax.plot(*np.array([[0, 0], q]).T, **edge_kw)
        ax.add_patch(Circle((0, 0), .1, fc=color, ec='none', zorder=2))
    for q in pts:
        if self_w:
            ax.add_patch(Circle(q*(1 + r + loop*.75), loop, fc='none', ec=color, lw=lw(self_w), zorder=1))
        ax.add_patch(Circle(q, r, fc='white', ec=color, lw=.5, zorder=2))


def export_glyphs(config, output):
    """SVG glyphs at final size: the current topology glyph per family, and candidate
    weighted-coupling glyphs per class built from the generator parameters."""
    style()
    output = Path(output); params = _generation(config); a = config['panel_a']
    from matplotlib.colors import to_rgb
    hue = {f: min((c['color'] for c in config['classes'] if c['family'] == f), key=lambda h: sum(to_rgb(h)))
           for f in a['topology']}
    (output/'topology').mkdir(parents=True, exist_ok=True); (output/'weighted').mkdir(parents=True, exist_ok=True)
    for f, kind in a['topology'].items():
        for name, color in [(f, hue[f]), (f'ink-{kind}', INK)]:
            fig = plt.figure(figsize=(.3, .19)); _glyph(fig, 0, 0, .3, .19, kind, .3, .19, color)
            fig.savefig(output/'topology'/f"{name.replace(' ', '-').lower()}.svg"); plt.close(fig)
    for c in config['classes']:
        p, kind = params[c['name']], a['topology'][c['family']]
        if c['family'] == 'VAR': self_w, edge_w = p['phi'], p['coupling']
        elif c['family'] == 'CML': self_w, edge_w = 1 - p['eps'], p['eps']/2
        else: self_w, edge_w = None, None
        fig = plt.figure(figsize=(.45, .45)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
        ax.set(xlim=(-1.75, 1.75), ylim=(-1.75, 1.75)); ax.set_aspect('equal')
        _weighted_glyph(ax, kind, self_w, edge_w, c['color'], repulsive=p.get('K', 0) < 0)
        fig.savefig(output/'weighted'/f"{c['name']}.svg"); plt.close(fig)
