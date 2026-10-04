"""Lean exploratory report across all nine requested observation sizes."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA
from umap import UMAP
import nbformat as nbf
from threadpoolctl import threadpool_limits
from scripts.build_pearson_size_match import ROOT, DATA, OUT, RUN
from scripts.analyze_pearson_size_match import rows
from scripts.report_pearson_strength_match import LABELS, COLORS
from scripts.spi_baseline_exploration import project_features
from scripts.refresh_case_figures import style, save

MARKERS = {8: 'o', 16: 's', 32: '^'}
SIZES = {100: 15, 500: 25, 1000: 38}


def strength():
    frame = rows()
    style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8), layout='constrained')
    jitter = np.random.default_rng(261011).uniform(-.16, .16, 45)
    for i, label in enumerate(LABELS):
        f = frame[frame.label.eq(label)]
        for ax, key in zip(axes, ['mean_covariance', 'mean_abs_Pearson']):
            ax.scatter(f[key], i+jitter, color=COLORS[label], s=14, alpha=.65, linewidths=0)
    for ax in axes:
        ax.set_yticks(range(8), LABELS.values())
        ax.invert_yaxis()
        ax.axhline(5.5, color='.8', lw=.7)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0].set(title='Signed covariance — all nine sizes', xlabel=r'Mean covariance $b$')
    axes[1].set(title='Absolute covariance — all nine sizes', xlabel=r'Mean absolute Pearson $a$')
    return save(fig, OUT/'figures', 'matched-strength')


def embedding():
    frame = rows().query("panel == 'matched'").reset_index(drop=True)
    style()
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 8.7), layout='constrained')
    coordinates = {}
    with np.load(OUT/'display-projections.npz') as p:
        np.testing.assert_array_equal(p['row_id'], frame.row_id)
        for col, (key, title) in enumerate([('mean', r'Per-SPI means $m$'), ('distribution', 'Per-SPI distributions'), ('z_complete', r'SPI–SPI $z$')]):
            for row, kind in enumerate(['PCA', 'UMAP']):
                mapper = PCA(n_components=2) if kind == 'PCA' else UMAP(n_neighbors=30, min_dist=.1, random_state=261003, n_jobs=1)
                xy = mapper.fit_transform(p[key])
                coordinates[key+'_'+kind] = xy
                ax = axes[row, col]
                for label in list(LABELS)[:6]:
                    for m in MARKERS:
                        keep = frame.label.eq(label) & frame.M.eq(m)
                        ax.scatter(*xy[keep].T, s=frame.loc[keep, 'T'].map(SIZES), marker=MARKERS[m],
                            color=COLORS[label], alpha=.7, edgecolors='white', linewidths=.25)
                ax.set(title=title+' — 6 classes', xlabel=kind+' 1', ylabel=kind+' 2')
                ax.set_box_aspect(1)
                for spine in ax.spines.values():
                    spine.set_visible(True)
                if kind == 'UMAP':
                    ax.set(xticks=[], yticks=[])
    handles = [Line2D([], [], marker='o', ls='', color=COLORS[k], label=v) for k, v in list(LABELS.items())[:6]]
    handles += [Line2D([], [], marker=v, ls='', color='.4', label=f'M = {k}') for k, v in MARKERS.items()]
    fig.legend(handles=handles, loc='outside lower center', ncol=3)
    np.savez_compressed(OUT/'embedding-coordinates.npz', **coordinates, row_id=frame.row_id.to_numpy())
    return save(fig, OUT/'figures', 'baseline-hierarchy')


def controls():
    frame = rows()
    fit = frame.panel.eq('matched').to_numpy()
    style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8), layout='constrained')
    with np.load(OUT/'features.npz') as a:
        for ax, key, title in zip(axes, ['mean', 'z'], [r'Per-SPI means $m$', r'SPI–SPI $z$']):
            _, d, h = project_features(a[key][fit], a[key], dimensions=20, standard=key == 'mean', valid=1. if key == 'z' else .95)
            xy = PCA(n_components=2).fit(d).transform(h)
            for label in LABELS:
                independent = label.endswith('independent')
                marker = ('x' if label.startswith('Gaussian') else '+') if independent else 'o'
                keep = frame.label.eq(label)
                ax.scatter(*xy[keep].T, marker=marker, color='black' if independent else COLORS[label],
                    s=25 if independent else 15, alpha=.8 if independent else .25, linewidths=.7 if independent else 0, label=LABELS[label])
            ax.set(title=title+' — independent controls', xlabel='PCA 1', ylabel='PCA 2')
            ax.set_box_aspect(1)
            for spine in ax.spines.values():
                spine.set_visible(True)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncol=4)
    return save(fig, OUT/'figures', 'independent-controls')


def notebook():
    score = pd.read_csv(OUT/'metrics.csv').set_index('method').BA
    summary = f'Parent-grouped class accuracy: mean covariance **{score.b:.1%}**, MPI means **{score["mean"]:.1%}**, distributions **{score.distribution:.1%}**, and training-complete SPI–SPI **{score.z_complete:.1%}**; chance is **16.7%**. The strongest full-mean comparison scores **{score[["mean", "mean_full", "mean_RBF", "mean_trees"]].max():.1%}**.'
    cells = [('md', '# Strength-matched MTS across nine observation sizes\n\n'+summary),
    ('md', r'''This is an exploratory extension to $M\in\{8,16,32\}$ and $T\in\{100,500,1000\}$, with five instances of each class in every cell. Six matched classes give 270 observations: VAR, Wave, CML, Kuramoto, correlated Gaussian and correlated Cauchy. Independent Gaussian/Cauchy add 90 separate controls. No class, point or embedding seed is selected for favorable separation.

**Five parents, not 45 independent recordings per class.** Each class has five independent 32-channel, 1,000-observation parents. Centered contiguous channel windows and initial time prefixes give the nine nested views. Validation leaves out an entire parent-instance block, including all nine sizes and all classes. This prevents overlapping observations from appearing in both training and evaluation. Five independent blocks support an exploratory comparison, not precise per-cell or population-performance claims.

The preceding single-size held experiment obtained 16.7% covariance-mean accuracy and 100% for MPI means, distributions and SPI–SPI. Thus matching scalar strength did **not** remove information from all MPI means. This extension tests observation-size robustness; it is not assumed to solve the stronger marginal-matching problem.'''),
    ('code', f'''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image
ROOT=Path.cwd().resolve()
while not (ROOT/'pyproject.toml').exists() and ROOT!=ROOT.parent: ROOT=ROOT.parent
RUN={RUN!r}; OUT=ROOT/'results/representation'/RUN
metrics=pd.read_csv(OUT/'metrics.csv')
display(metrics.round(3))'''),
    ('md', r'''## Generator and matching rule

Native parents use a 32-node VAR with self coefficient .7 and .1 per ring neighbor; a periodic wave with speed $10\sqrt{.2/.08}$ and fixed time step .00125; an observed 32-channel window of a 100-site CML with $\alpha=1.895$, $\epsilon=.109375$; and a 32-oscillator Kuramoto model with $K=-50$, step .00625 and mean intrinsic frequency 3, observing sine of phase. The CML and Kuramoto settings are medians of the preceding development calibrations, not selected using this experiment's SPI outcomes. Observation-size changes therefore do not change the parent dynamics. Noise variants use heterogeneous shared-input loadings; independent controls preserve the paired independent innovations.

All channels are centered and standardized to unit empirical variance. Both $b=\operatorname{mean}_{i\ne j}\widehat{\operatorname{Cov}}(x_i,x_j)$ and $a=\operatorname{mean}_{i\ne j}|r_{ij}|$ are matched within each size/parent block. At $T=100$, chance sample correlations are larger, so enforcing the old absolute target .07 through independent observation noise is not generally feasible. Instead the common absolute target is the midpoint between the largest sampled innovation-correlation floor and the smallest coupled-signal absolute correlation in that block. It uses covariance only, never other SPI outcomes.

The signed target is $b=a(S_M^2-M)/[M(M-1)]$, where $S_M=2\operatorname{round}(\sqrt{2M}/2)$. This follows from an equicorrelated reference after channel polarity changes with sign sum $S_M$: it gives a motivated feasible sign–magnitude ratio, rather than an arbitrary tiny signed mean. Targets vary across observation sizes, but their **pooled distributions are identical across the balanced classes**.

Rotate two fixed independent innovation arrays as $E_\theta=\cos\theta E_1+\sin\theta E_2$. For the four dynamical families, tune Gaussian measurement-noise amplitude to the absolute target; for Gaussian/Cauchy variants, tune common-input mixing using innovations of the corresponding family. A bounded search over channel polarities and a continuous root in $\theta$ then match signed mean while retaining the absolute target. No covariance whitening or post-computation MPI adjustment is applied. The noise amplitude and orientation are data-conditioned, so this is an engineered observation-space control, not universal physical/causal coupling equality. Cauchy covariance statements are finite-sample only.

A simpler signed-mean-only feasibility construction was rejected before any SPI job because it matched mainly through positive–negative cancellation while absolute correlations remained strongly different. Its inputs remain preserved as `pearson-size-control-261004`.'''),
    ('code', "display(Image(filename=str(OUT/'figures/matched-strength.png')))\ntargets=pd.read_csv(OUT/'targets.csv')\ndisplay(targets.groupby(['M','T'])[['target_absolute','target_signed']].agg(['min','max']).round(4))"),
    ('md', r'''## Baseline hierarchy and fixed geometry

The hierarchy is mean covariance $b$, the 289-derived vector of MPI means $m$, seven marginal summaries per MPI (mean, SD, 10/25/50/75/90 percentiles), and signed Pearson SPI–SPI $z$ over aligned ordered off-diagonal entries. Means/distributions use training-only SD scaling, clipping at five SD and PCA20. Primary $z$ uses training-complete features, centering and PCA20, with no per-feature rescaling. Logistic regression fixes $C=1$. Full-mean linear/RBF/tree classifiers, standardized $z$, validity-only and independently edge-shuffled $z$ remain visible as controls. Shuffling preserves MPI marginal values while disrupting their alignment; this need not represent a realizable MTS.

Every validation fold refits all feature selection and preprocessing using only the other four parent blocks. Pooled and per-cell class accuracies below are predictions for excluded parents. Parent minimum/maximum are descriptive variability, not confidence intervals. Only five parents contribute to each cell.

**The plots are descriptive all-data fits**, separately from validation: PCA20 then PCA2 or UMAP with 30 neighbors, minimum distance .1 and seed 261003, using all 270 matched observations. Colors denote class, marker shapes denote $M$, and marker size increases with $T$. No classifier score is computed from this all-data fit. Overlap or clustering in two dimensions is not a test of information absence or superiority.'''),
    ('code', "display(Image(filename=str(OUT/'figures/baseline-hierarchy.png')))\nf=pd.read_csv(OUT/'cell-metrics.csv')\ndisplay(f[f.method.isin(['b','mean','distribution','z_complete'])].pivot(index=['M','T'],columns='method',values='BA').round(3))"),
    ('md', 'Independent noise controls are projected into the same matched-data PCA spaces below, without influencing the fit. Crosses and plus signs denote independent Gaussian and Cauchy controls. These controls help reveal geometry associated with channel distributions or estimator behavior even without population cross-channel dependence; they are excluded from the six-class accuracy calculation. If a true MPI is constant across channel pairs, its population meta-correlation is undefined. Finite-sample fluctuations can nevertheless vary together across estimators and produce structured numerical z. Consequently, a noise-family signature alone is not evidence of a corresponding population interaction mechanism.'),
    ('code', "display(Image(filename=str(OUT/'figures/independent-controls.png')))\ndisplay(pd.read_csv(OUT/'diagnostics.csv').groupby('method')[['features','test_missing_fraction']].agg(['min','max']).round(4))"),
    ('md', r'''## What this can establish

Weak $b$ with successful $z$ demonstrates information beyond matched scalar covariance summaries. The scalar baseline is deliberately deprived of class information: its near-chance result validates the control, rather than independently establishing general superiority of SPI–SPI. It does not prove that all marginal statistics are blind to character or that $z$ uniquely accesses it. Any strong MPI-mean result must be credited. Scalar matching also leaves temporal persistence, channel distributions, topology and estimator effects available; a class separation does not identify which one drives the geometry. In particular, the Gaussian and Cauchy shared-input constructions are both linear mixtures: distinguishing them does not establish nonlinear coupling. The method is invariant to positive affine changes of individual SPI profiles, not arbitrary generator-strength changes or observational-size changes.

Size-dependent targets do not give class information in this balanced design, but may contribute to size structure in the embedding. The five parent realizations, engineered normalization and exploratory model comparison limit generalization. The single-size confirmation and earlier failed all-proof, band-swap and Gaussian-factor experiments remain intact.'''),
    ('md', f'Frozen protocol and source bindings: `configs/analysis/{RUN}.yaml` and `-bindings.yaml`; run provenance: `results/representation/{RUN}/execution.json`. The raw archive, parent archive, ordered SPI catalogue and serial estimator RNG are audited before interpretation. All 289 SPIs are attempted; train-fold validity selection determines the numerical features actually used.')]
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(s) if k == 'md' else nbf.v4.new_code_cell(s) for k, s in cells])
    nb.metadata.kernelspec = dict(display_name='Python 3', language='python', name='python3')
    path = ROOT/'notebooks/embeddings/spi_pearson_size_matched_261004.ipynb'
    nbf.write(nb, path)
    return path


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        strength()
        embedding()
        controls()
        print(notebook())
