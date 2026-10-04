"""Figures and lean notebook for the frozen six-family strength control."""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import nbformat as nbf
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from scripts.build_pearson_strength_match import DATA, OUT, ROOT, RUN
from scripts.analyze_pearson_strength_match import rows
from scripts.refresh_case_figures import style, save
from scripts.spi_baseline_exploration import project_features

LABELS = {'VAR-0.7': 'VAR', 'Wave': 'Wave', 'CML-1.895': 'CML',
          'Kuramoto-fast': 'Kuramoto', 'Gaussian-correlated': 'Correlated Gaussian',
          'Cauchy-correlated': 'Correlated Cauchy',
          'Gaussian-independent': 'Independent Gaussian', 'Cauchy-independent': 'Independent Cauchy'}
COLORS = dict(zip(LABELS, ['#0072B2', '#E69F00', '#D55E00', '#555555', '#009E73', '#CC79A7', '#888888', '#BBBBBB']))


def strength(stage):
    frame = rows(stage)
    test = frame.development_part.eq('validation') if stage == 'development' else frame.role.eq('evaluation')
    frame = frame.loc[test]
    diagnostics = []
    with np.load(DATA/'observations.npz') as archive:
        for r in frame.itertuples():
            x = archive[r.row_id]
            diagnostics.append(dict(label=LABELS[r.label], block=r.block,
                mean_channel_lag1=float(np.mean(x[:, 1:]*x[:, :-1])),
                mean_channel_fourth_moment=float(np.mean(x**4))))
    (OUT/stage).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(OUT/stage/'observation-diagnostics.csv', index=False)
    style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), layout='constrained')
    jitter = np.random.default_rng(261007).uniform(-.12, .12, frame.block.nunique())
    for i, (label, text) in enumerate(LABELS.items()):
        group = frame.loc[frame.label.eq(label)].sort_values('block')
        for ax, key in zip(axes, ['mean_covariance', 'mean_abs_Pearson']):
            ax.scatter(group[key], i + jitter, s=20, color=COLORS[label], alpha=.8, linewidths=.3, edgecolors='white')
    for ax in axes:
        ax.set_yticks(range(len(LABELS)), LABELS.values())
        ax.invert_yaxis()
        ax.axhline(5.5, color='.8', lw=.7)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0].set(xlabel=r'Mean covariance $b$', title='Signed strength (unit channel variance)')
    axes[1].set(xlabel=r'Mean absolute Pearson $a$', title='Cancellation check')
    axes[0].axvspan(.003, .007, color='.8', alpha=.25, zorder=-1)
    axes[1].axvspan(.065, .075, color='.8', alpha=.25, zorder=-1)
    return save(fig, OUT/stage/'figures', 'matched-strength')


def embeddings(stage):
    from umap import UMAP
    frame = rows(stage)
    frame = frame.loc[frame.panel.eq('matched')]
    test = frame.development_part.eq('validation') if stage == 'development' else frame.role.eq('evaluation')
    labels = frame.loc[test, 'label'].to_numpy()
    style()
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 8.2), layout='constrained')
    coordinates = {}
    with np.load(OUT/stage/'projections.npz') as p:
        for col, (key, title) in enumerate([('mean', r'Per-SPI means $m$'), ('distribution', 'Per-SPI distributions'), ('z_complete', r'SPI–SPI $z$')]):
            d, h = p[key+'_train'], p[key+'_test']
            for row, kind in enumerate(['PCA', 'UMAP']):
                mapper = PCA(n_components=2) if kind == 'PCA' else UMAP(n_neighbors=30, min_dist=.1, random_state=261003, transform_seed=261003, n_jobs=1)
                mapper.fit(d)
                xy = mapper.transform(h)
                coordinates[key+'_'+kind] = xy
                ax = axes[row, col]
                for label in list(LABELS)[:6]:
                    keep = labels == label
                    ax.scatter(*xy[keep].T, s=24, alpha=.8, color=COLORS[label], edgecolors='white', linewidths=.3, label=LABELS[label])
                ax.set(title=title+' — 6 classes', xlabel=kind+' 1', ylabel=kind+' 2')
                ax.set_box_aspect(1)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                if kind == 'UMAP':
                    ax.set(xticks=[], yticks=[])
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncol=3)
    np.savez_compressed(OUT/stage/'embedding-coordinates.npz', **coordinates, row_id=frame.loc[test, 'row_id'].to_numpy())
    return save(fig, OUT/stage/'figures', 'baseline-hierarchy')


def controls(stage):
    frame = rows(stage)
    train_role = frame.development_part.eq('train') if stage == 'development' else frame.role.eq('development')
    train = (train_role & frame.panel.eq('matched')).to_numpy()
    test = (frame.development_part.eq('validation') if stage == 'development' else frame.role.eq('evaluation')).to_numpy()
    labels = frame.loc[test, 'label'].to_numpy()
    style()
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.7), layout='constrained')
    with np.load(OUT/stage/'features.npz') as archive:
        np.testing.assert_array_equal(archive['row_id'], frame.row_id)
        for ax, key, title in zip(axes, ['mean', 'z'], [r'Per-SPI means $m$', r'SPI–SPI $z$']):
            _, d, h = project_features(archive[key][train], archive[key][test], dimensions=20,
                                      standard=key == 'mean', valid=1. if key == 'z' else .95)
            xy = PCA(n_components=2).fit(d).transform(h)
            for label in LABELS:
                control = label.endswith('independent')
                marker = ('D' if label.startswith('Gaussian') else '^') if control else 'o'
                ax.scatter(*xy[labels == label].T, s=35 if control else 18, marker=marker,
                           alpha=1 if control else .35, color=COLORS[label],
                           edgecolors='black' if control else 'none', linewidths=.5, label=LABELS[label])
            ax.set(title=title+' — independent controls', xlabel='PCA 1', ylabel='PCA 2')
            ax.set_box_aspect(1)
            for spine in ax.spines.values():
                spine.set_visible(True)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncol=4)
    return save(fig, OUT/stage/'figures', 'independent-controls')


def notebook(stage):
    result = OUT/stage
    metrics = pd.read_csv(result/'metrics.csv').set_index('method')
    score = metrics.BA
    analysis = json.loads((result/'analysis.json').read_text())
    evidence = 'Development validation' if stage == 'development' else 'Fresh held evaluation'
    summary = f'{evidence}: covariance mean **{score.b:.1%}**, all MPI means **{score["mean"]:.1%}**, distribution summaries **{score.distribution:.1%}**, and training-complete SPI–SPI **{score.z_complete:.1%}**; chance is **16.7%**. The strongest tested mean-vector readout scores **{score[["mean", "mean_full", "mean_RBF", "mean_trees"]].max():.1%}**. These comparisons do not imply that information is absent from the MPI marginals.'
    cells = [('md', '# Matching a scalar dependence strength across six MTS families\n\n'+summary),
    ('md', r'''The simplest baseline is the off-diagonal mean of the empirical covariance MPI,

$$b(X)=\frac{1}{M(M-1)}\sum_{i\ne j}\widehat{\operatorname{Cov}}(x_i,x_j).$$

Every channel is centered and scaled to unit empirical variance (divisor $T$), so this also equals mean signed Pearson correlation. Because positive and negative correlations can cancel, we additionally match $a(X)=\operatorname{mean}_{i\ne j}|r_{ij}|$. A scalar has only one informative dimension; its display is a classwise strip plot, not a two-dimensional PCA/UMAP. The comparison then progresses to all 289 MPI means $m$, seven summaries per MPI (mean, SD, 10/25/50/75/90 percentiles), and signed Pearson SPI–SPI $z$.

The six matched classes are VAR, Wave, CML, Kuramoto, shared-input Gaussian and shared-input Cauchy. Independent Gaussian and Cauchy recordings are separate zero-population-dependence controls, excluded from the six-class classifier. All recordings use $M=16$, $T=1000$: a practical common size inherited from the cached proof bank, yielding 240 ordered channel pairs (120 unique undirected pairs), not 240 independent observations. This size was not optimized to maximize separation.'''),
    ('code', f'''from pathlib import Path
import sys, json
ROOT=Path.cwd().resolve()
while not (ROOT/'pyproject.toml').exists() and ROOT!=ROOT.parent: ROOT=ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import pandas as pd
from IPython.display import display, Image
RUN={RUN!r}; STAGE={stage!r}
RESULT=ROOT/'results/representation'/RUN/STAGE
metrics=pd.read_csv(RESULT/'metrics.csv')
display(metrics.round(3))'''),
    ('md', r'''## What is normalized, and why?

Each block draws a signed target $b_j\sim U(.003,.007)$ and an absolute target $a_j\sim U(.065,.075)$, shared by all six classes. Targets vary across recordings while their distribution is identical across classes. Their low positive ranges were selected for feasibility across the cached development families, not from full-catalogue separation outcomes. They are engineered experimental settings, not universal physical constants.

For the four coupled dynamical families, add independent Gaussian observation noise and tune its scale to $a_j$, then standardize each channel. For the noise variants, tune the shared-input mixture $(1-g)\epsilon_i+g\ell_iF$, with heterogeneous positive loadings, to the same target. The innovations and common input are Gaussian or Cauchy according to class. Enumerating channel polarities $s_i\in\{-1,+1\}$ then selects the closest signed target. Polarity changes preserve absolute pairwise correlations and each channel's autocorrelation; additive observation noise does not preserve every aspect of the original dynamics.

This is **data-conditioned observation-space normalization of two explicit summaries**, not equality of universal physical or causal coupling. The noise classes acquire dependence through shared inputs, without direct channel-to-channel coupling. Cauchy variables have undefined population variance; their covariance matching here is strictly a finite-recording operation. No MPI entries are adjusted after computation. The matched scalar baseline is expected to fail by construction; the empirical question is whether useful distinctions survive in the other representations.'''),
    ('code', "display(Image(filename=str(RESULT/'figures/matched-strength.png')))\ndisplay(pd.read_csv(RESULT/'observation-diagnostics.csv').groupby('label')[['mean_channel_lag1','mean_channel_fourth_moment']].median().round(3))"),
    ('md', 'The table gives medians across evaluation recordings of two raw-data diagnostics: mean channel lag-one autocovariance and mean standardized fourth moment. These describe remaining temporal and channel-distribution differences; they are not fitted alternatives or population moments for Cauchy data. Substantial residual differences would limit attribution of class separation specifically to cross-channel dependence character. Both signed and absolute strength targets can match while these properties differ.'),
    ('md', r'''## Frozen comparison and fixed embeddings

Development blocks 0–15 train, blocks 16–23 validate; blocks 24–47 are held until the development gate. There are six matched recordings and two independent controls per block. Coupled development records reuse previously examined raw inputs and are explicitly exploratory; held coupled inputs use fresh generator seeds. Shared target/noise blocks stay together across splits. Final fitting uses all development blocks, then evaluates the untouched held blocks.

Feature selection, imputation, scaling and PCA use training recordings only. Means/distributions are standardized, clipped at five SD and reduced to 20 PCs. The primary $z$ readout centers without rescaling, retaining only features finite in every training recording, followed by 20 PCs. All use logistic regression with $C=1$. Full-vector linear, RBF and tree mean controls, standardized $z$, estimator-validity-only features and independently edge-shuffled $z$ are retained. The shuffle preserves each MPI's marginal values but destroys their alignment; it is a representation diagnostic and need not describe a realizable MTS.

PCA/UMAP are fit on the training projections and transform evaluation points; UMAP fixes 30 neighbors, minimum distance .1 and seed 261003. Every evaluation point is shown. Classifier results use the full retained representation, not the plotted two coordinates. Conditional 95% intervals resample paired blocks 5,000 times while keeping the fitted models fixed, so they omit training and experimental-design uncertainty.'''),
    ('code', "display(Image(filename=str(RESULT/'figures/baseline-hierarchy.png')))\ndisplay(pd.read_csv(RESULT/'paired.csv').round(3))"),
    ('md', 'Independent controls are projected into the same training-fitted mean and SPI–SPI spaces below; they do not influence those fits. Larger outlined diamonds/triangles denote independent Gaussian/Cauchy recordings, and faded circles show the six matched classes. These are a diagnostic view, not an additional eight-class accuracy claim. Positions in two PCs alone cannot determine which aspect of the data drives separation.'),
    ('code', "display(Image(filename=str(RESULT/'figures/independent-controls.png')))"),
    ('md', r'''## Interpretation and limits

Failure of $b$ with successful $z$ establishes information beyond average signed covariance under this normalization. Matching absolute correlation makes this stronger than a cancellation-only demonstration. It does not show that full SPI means or distributions are blind to character: their average detector responses can themselves distinguish temporal structure, non-Gaussianity and other properties. Strong $m$ must be credited, rather than treated as a failed baseline to conceal.

SPI–SPI measures how notions of dependence vary together across channel pairs. It is invariant to positive affine transformations of each SPI profile, not to arbitrary changes in generator coupling or measurement noise. Separation here may reflect temporal structure, channel distributions, topology, estimator behavior and their combinations; it is not an identified axis of pure nonlinearity or causal interaction. Independent Gaussian/Cauchy controls illustrate why distinguishing MTS families alone does not prove different cross-channel coupling. Validity and shuffled-edge controls constrain, but do not eliminate, these interpretations.

Earlier negative evidence remains in the all-proof native-gain, band-swap and factor-parity notebooks. Native gain matching did not make the full mean vector uninformative; the Gaussian factor construction gave only partial full-catalogue gains. This experiment addresses the simpler scalar-strength claim and must not be presented as resolving that stronger target.'''),
    ('code', "analysis=json.loads((RESULT/'analysis.json').read_text())\ndisplay(pd.DataFrame(analysis['diagnostics']).T)\nprint('Frozen numerical criteria:',analysis['development_gate'])"),
    ('md', f'Protocol and source bindings: `configs/analysis/{RUN}.yaml` and `configs/analysis/{RUN}-bindings.yaml`. Raw inputs and MPI provenance are recorded in the manifest and execution receipt; audited source hashes are in this stage’s `sources.json`. Figures and evaluation are reproduced from saved features without recomputing SPIs. Stage: **{stage}**. Frozen numerical criteria all passed: **{analysis["development_gate"]["all_pass"]}**.')]
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(s) if k == 'md' else nbf.v4.new_code_cell(s) for k, s in cells])
    nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    path = ROOT/'notebooks/embeddings/spi_pearson_strength_matched_261004.ipynb'
    nbf.write(nb, path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['development', 'final'], default='development')
    parser.add_argument('--strength-only', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        strength(args.stage)
        if not args.strength_only:
            embeddings(args.stage)
            controls(args.stage)
            print(notebook(args.stage))
