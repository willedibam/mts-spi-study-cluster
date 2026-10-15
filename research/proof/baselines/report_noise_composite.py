"""Cached noise-only embeddings, strength colours and training-selected z examples."""
import json
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from mtsspi.figures import raincloud, save, style
from research.proof.baselines.strength_factorial import ROOT, DATA, OUT, CLASSES, sha
from research.proof.baselines.report_strength_factorial import NOTEBOOK, COLORS, DISPLAY
from research.proof.baselines.analyze_strength_factorial import split

TAG = 'noise-composite-261016'
# Chosen as within-CML, shared-input tail-law and cross-family contrasts before
# inspecting their feature rankings. No selection on evaluation observations.
CONTRASTS = [('CML-intermittency', 'CML-defect'),
             ('Gaussian-shared', 'Cauchy-shared'), ('VAR', 'Wave')]


def candidate_pairs(names, contrast):
    """Diverse scientific families, fixed before inspecting revised rankings."""
    if contrast == 0:
        left = {'spearmanr', 'kendalltau', 'dcorr', 'dtw'}
        right = {'mi_kraskov_NN-4'}
        family = 'Rank/distance versus KSG mutual information'
    elif contrast == 1:
        left = {'cov_EmpiricalCovariance'}
        right = {'spearmanr', 'kendalltau'}
        family = 'Pearson versus rank correlation'
    else:
        left = {n for n in names if n.startswith('corr_pearson_tau-') and not n.endswith('-sq')}
        right = {n for n in names if n.startswith(('cohmag_multitaper_mean_', 'plv_multitaper_mean_'))}
        family = 'Lagged Pearson versus spectral coherence/phase locking'
    return np.array([(a in left and b in right) or (b in left and a in right)
                     for a, b in names[np.array(np.triu_indices(len(names), 1)).T]]), family


def spi_label(name):
    known = {'cov_EmpiricalCovariance': 'Pearson (zero lag)',
             'spearmanr': 'Spearman', 'kendalltau': 'Kendall', 'dcorr': 'Distance correlation',
             'dtw': 'Dynamic time warping', 'mi_kraskov_NN-4': 'Mutual information (KSG, k=4)'}
    if name in known: return known[name]
    if name.startswith('corr_pearson_tau-'): return 'Pearson (lag '+name.split('tau-')[1]+')'
    for prefix, label in [('cohmag_', 'Coherence magnitude'), ('plv_', 'Phase locking value')]:
        if name.startswith(prefix):
            lo, hi = name.split('fmin-')[1].split('_fmax-')
            return label+' (mean)\n'+f'f = {lo.replace("-", ".")}–{hi.replace("-", ".")} cycles/sample'
    return name


def select_features(rows, values, names):
    train = rows.arm.eq('fixed_sensor') & rows.instance.lt(20)
    pairs = np.array([(names[i], names[j]) for i, j in zip(*np.triu_indices(len(names), 1))])
    selected = []
    for contrast, (ca, cb) in enumerate(CONTRASTS):
        a = values[train & rows.label.eq(ca)]
        b = values[train & rows.label.eq(cb)]
        eligible, family = candidate_pairs(names, contrast)
        valid = eligible & np.isfinite(a).all(0) & np.isfinite(b).all(0)
        # Require three distinct coordinates, in the declared contrast order.
        valid[[r['feature_index'] for r in selected]] = False
        columns = np.flatnonzero(valid)
        ranks = rankdata(np.vstack([a[:, valid], b[:, valid]]), axis=0)
        auc = (ranks[len(a):].sum(0) - len(b)*(len(b)+1)/2)/(len(a)*len(b))
        winner = int(np.argmax(abs(auc-.5)))
        col = int(columns[winner])
        selected.append(dict(class_a=ca, class_b=cb, feature_index=col,
                             spi_a=pairs[col, 0], spi_b=pairs[col, 1],
                             training_auc=max(auc[winner], 1-auc[winner]),
                             orientation=1 if auc[winner] >= .5 else -1,
                             candidates=int(valid.sum()), candidate_family=family))
    return pd.DataFrame(selected)


def main():
    source = OUT/'embedding-coordinates.npz'
    coordinate_hash = sha(source)
    rows = pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows'])
    with np.load(source) as a:
        np.testing.assert_array_equal(a['row_id'], rows.row_id)
        xy = {key: a[key] for key in ['z_PCA', 'm_PCA', 'z_UMAP', 'm_UMAP']}
    with np.load(OUT/'features.npz') as a:
        np.testing.assert_array_equal(a['row_id'], rows.row_id)
        z, names = a['z'], a['spi_order']
    assert sha(OUT/'features.npz') == json.loads((OUT/'extraction-audit.json').read_text())['features_sha256']
    selected = select_features(rows, z, names)
    selected.to_csv(OUT/'noise-composite-selection.csv', index=False)
    use = rows.arm.eq('fixed_sensor') & rows.instance.ge(20) & rows.label.isin(CLASSES)
    assert use.sum() == 140
    display_values = []
    for row in selected.itertuples():
        samples = []
        for cls in [row.class_a, row.class_b]:
            mask = use & rows.label.eq(cls)
            v = z[mask, row.feature_index]
            samples.append(v[np.isfinite(v)])
            for index, value in zip(rows.index[mask], v):
                display_values.append(dict(row_id=rows.loc[index, 'row_id'], label=cls,
                                           feature_index=row.feature_index, value=float(value)))
        a, b = samples
        ranks = rankdata(np.r_[a, b])
        auc = (ranks[len(a):].sum()-len(b)*(len(b)+1)/2)/(len(a)*len(b))
        selected.loc[row.Index, 'display_auc'] = auc if row.orientation == 1 else 1-auc
        selected.loc[row.Index, 'display_n_a'] = len(a)
        selected.loc[row.Index, 'display_n_b'] = len(b)
    selected.to_csv(OUT/'noise-composite-selection.csv', index=False)
    pd.DataFrame(display_values).to_csv(OUT/'noise-composite-feature-values.csv', index=False)
    train, _, _ = split(rows)
    prep = pd.read_csv(OUT/'preprocessing.csv').set_index('representation')
    variance = {}
    with np.load(OUT/'coordinates.npz') as a:
        for rep in ['z', 'm']:
            np.testing.assert_array_equal(a[rep][:, :2], xy[rep+'_PCA'])
            v = a[rep][train].var(0)
            variance[rep] = v[:2]/v.sum()*prep.loc[rep, 'training_variance_retained']
    # Descriptive associations: keep family-specific results to avoid mistaking
    # between-family differences for within-family strength dependence.
    associations = []
    for rep in ['z', 'm']:
        for pc in range(2):
            for label in ['pooled', *CLASSES]:
                mask = use if label == 'pooled' else use & rows.label.eq(label)
                rho = spearmanr(xy[rep+'_PCA'][mask, pc], rows.loc[mask, 'mean_abs_r']).statistic
                associations.append(dict(representation=rep, axis=pc+1, label=label, spearman_rho=rho))
    pd.DataFrame(associations).to_csv(OUT/'noise-composite-axis-associations.csv', index=False)
    style()
    for method in ['PCA', 'UMAP']:
        fig = plt.figure(figsize=(9.2, 10.2))
        for col, rep in enumerate(['z', 'm']):
            v = xy[rep+'_'+method]
            # Keep the source noise-focus limits, including its clean comparator.
            frame = rows.label.isin(CLASSES) & rows.instance.ge(20) & rows.arm.isin(['fixed_clean', 'fixed_sensor'])
            lo, hi = v[frame].min(0), v[frame].max(0)
            pad = .08*(hi-lo)
            for row, bottom in enumerate([7.05, 3.95]):
                ax = fig.add_axes([(1.25+3.35*col)/9.2, bottom/10.2, 2.65/9.2, 2.65/10.2])
                for cls in CLASSES:
                    mask = use & rows.label.eq(cls)
                    kwargs = dict(c=COLORS[cls]) if row == 0 else dict(c=rows.loc[mask, 'mean_abs_r'], cmap='viridis', vmin=0, vmax=1)
                    artist = ax.scatter(*v[mask].T, s=22, alpha=.85, linewidths=0, **kwargs)
                ax.set(xlim=(lo[0]-pad[0], hi[0]+pad[0]), ylim=(lo[1]-pad[1], hi[1]+pad[1]))
                ax.set_box_aspect(1); ax.spines[['top', 'right']].set_visible(True)
                title = r'SPI–SPI $z$' if rep == 'z' else r'Mean-MPI $m$'
                ax.set_title(title if row == 0 else r'Mean $|r|$ colouring', fontsize=9)
                if method == 'PCA':
                    ax.set(xlabel=f'PC1 ({variance[rep][0]:.1%})', ylabel=f'PC2 ({variance[rep][1]:.1%})')
                    ax.xaxis.set_major_locator(MaxNLocator(3)); ax.yaxis.set_major_locator(MaxNLocator(3))
                else:
                    ax.set(xlabel='UMAP 1', ylabel='UMAP 2', xticks=[], yticks=[])
                if row == 0:
                    ax.set_xlabel(''); ax.tick_params(axis='x', labelbottom=False)
        cax = fig.add_axes([7.65/9.2, 3.95/10.2, .12/9.2, 2.65/10.2])
        fig.colorbar(artist, cax=cax, label=r'Observed mean $|r|$', ticks=[0, .5, 1])
        for k, row in enumerate(selected.itertuples()):
            ax = fig.add_axes([(.50+3.05*k)/9.2, .85/10.2, 2.2/9.2, 2.2/10.2])
            for pos, cls in enumerate([row.class_a, row.class_b]):
                mask = use & rows.label.eq(cls)
                raincloud(ax, z[mask, row.feature_index], pos, COLORS[cls],
                          np.random.default_rng(261016+row.feature_index+pos), width=.65, box=True, point_sizes=12)
            vals = z[use & rows.label.isin([row.class_a, row.class_b]), row.feature_index]
            vals = vals[np.isfinite(vals)]; pad = max(np.ptp(vals)*.08, .002)
            ax.set(ylim=(max(-1.005, vals.min()-pad), min(1.005, vals.max()+pad)), xlim=(-.6, 1.6),
                   xticks=[0, 1], xticklabels=[fill(DISPLAY[c], 18) for c in [row.class_a, row.class_b]],
                   ylabel=r'$z_{ab}$', title=f'{spi_label(row.spi_a)}\n× {spi_label(row.spi_b)}')
            ax.title.set_fontsize(7); ax.tick_params(labelsize=7)
            ax.yaxis.set_major_locator(MaxNLocator(3)); ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.spines[['top', 'right']].set_visible(True); ax.set_box_aspect(1)
        handles = [Line2D([], [], marker='o', ls='', color=COLORS[c], markersize=4, markeredgewidth=0, label=DISPLAY[c]) for c in CLASSES]
        fig.legend(handles=handles, loc='lower center', ncols=4, fontsize=7, bbox_to_anchor=(.5, .005))
        fig.suptitle('Varying sensor noise · fixed physical coupling · M=16, T=1000', y=.997)
        stem = 'noise-composite-'+method.lower()
        save(fig, OUT/'figures', stem); fig.savefig(OUT/'figures'/f'{stem}.pdf'); plt.close(fig)
    assert sha(source) == coordinate_hash
    audit = dict(source='strength-factorial-261015', arm='fixed_sensor', embedding_sha256=coordinate_hash,
                 features_sha256=sha(OUT/'features.npz'), source_sha256=sha(__file__), embedding_refit=False,
                 classes=list(CLASSES), evaluation_instances=list(range(20, 40)), points=140,
                 selection='Three declared class contrasts; maximal oriented AUC within declared diverse candidate families, finite in noisy training instances 0–19; catalogue-order ties; exclude coordinates selected for preceding contrasts to obtain three distinct features. No held selection.',
                 contrast_pairs=CONTRASTS, post_result_presentation=True, strength_colour_range=[0, 1],
                 axis_limits='Identical across colouring rows; reuse noise-focus clean/noisy limits.',
                 PCA_training_explained_variance={k: v.tolist() for k, v in variance.items()})
    (OUT/'noise-composite-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(selected.to_string(index=False))
    print(pd.DataFrame(associations).query("label=='pooled'").to_string(index=False))


def append():
    original = NOTEBOOK.read_bytes(); nb = nbformat.reads(original.decode(), 4)
    md = nbformat.v4.new_markdown_cell; code = nbformat.v4.new_code_cell
    cells = [md(r'''### Composite: varying sensor noise, fixed latent dynamics

These PCA/UMAP composites reuse the **fixed_sensor** arm of the October 15 factorial, not the later stronger-noise experiment: all seven classes, M=16, T=1000, 20 evaluation recordings/class. This includes nominal CML intermittency and **defect turbulence**. Starting from sample-standardized latent channels, add independent Gaussian samples with one common SD per recording, $\eta_d=\sqrt{1/\alpha_d-1}$, where $\alpha_d\sim U[.25,.95]$ (thus $\eta\approx .23$–$1.73$). Noise strength varies between recordings, with the same draw across classes at a given instance; physical coupling stays fixed. Channels are standardized again before p90. For finite-variance population channels this gives nominal correlation attenuation $\alpha_d$; the identity does not apply to Cauchy population moments.

Columns compare z and raw mean-MPI (training feature scaling/clipping, no within-recording normalization). The second row changes only colour to observed mean absolute Pearson correlation. Coordinates and limits are reused without fitting; PCA variance percentages refer to training data, and neither PC is defined to be a strength axis. Here mean-MPI PC1 strongly tracks observed mean absolute correlation (pooled Spearman $\rho=.962$, within-class $.904$–$.995$), versus $.350$ pooled for z PC1. The colour gradients describe association, not a uniquely identified physical coordinate; z still has within-class strength associations. Mean-MPI retains class information despite less robust clean-reference distances.

The lower row shows three illustrative SPI–SPI coordinates: within CML, Gaussian versus Cauchy shared input, and VAR versus Wave. Candidate families are rank/distance versus KSG mutual information, Pearson versus rank correlation, and lagged Pearson versus spectral coherence/phase locking, respectively. Maximal oriented AUC within each family selects a coordinate using noisy **training instances 0–19 only** (catalogue order breaks ties; preceding selections are excluded to obtain three distinct features); rainclouds show instances 20–39, with boxes denoting median/IQR and 1.5-IQR whiskers. Undefined values are omitted and counts are reported below. Candidate coordinates can tie at training AUC 1; these are illustrative examples, not uniquely best probes. Class contrasts were chosen for illustration, so this remains a retrospective presentation, not a new confirmation or mechanistic identification. Each feature panel has its own vertical scale.'''),
        code("from pathlib import Path\nfrom IPython.display import display,Image\nimport pandas as pd\nNC_ROOT=Path.cwd().resolve()\nwhile not (NC_ROOT/'pyproject.toml').exists() and NC_ROOT!=NC_ROOT.parent:NC_ROOT=NC_ROOT.parent\nNC_OUT=NC_ROOT/'results/proof/strength-factorial-261015'\ndisplay(Image(filename=str(NC_OUT/'figures/noise-composite-pca.png')))\ndisplay(Image(filename=str(NC_OUT/'figures/noise-composite-umap.png')))\ndisplay(pd.read_csv(NC_OUT/'noise-composite-selection.csv')[['class_a','class_b','spi_a','spi_b','training_auc','display_auc','display_n_a','display_n_b']].round(3))")]
    for c in cells: c.metadata['tags'] = [TAG]
    extra = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    NotebookClient(extra, timeout=120, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    assert NOTEBOOK.read_bytes() == original, 'Notebook edited during execution; retry against saved version'
    preserved = [c for c in nb.cells if TAG not in c.metadata.get('tags', [])]
    nb.cells = preserved + extra.cells; nbformat.write(nb, NOTEBOOK)
    assert nbformat.read(NOTEBOOK, 4).cells[:len(preserved)] == preserved
    print('Preserved', len(preserved), 'cells; appended', len(cells))


if __name__ == '__main__':
    main(); append()
