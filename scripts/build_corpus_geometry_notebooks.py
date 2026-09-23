"""Build the proof and Zenodo legacy-versus-optimized geometry notebooks."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "embeddings"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def base_notebook(title: str):
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    notebook.cells.append(markdown(f"# {title}"))
    return notebook


def proof_notebook():
    notebook = base_notebook("Proof $p90$: historical versus fitted SPI--SPI geometry")
    notebook.cells.extend(
        [
            markdown(
                r"""
This notebook compares the historical preprocessing with three explicit fitted
alternatives on the current 14-class proof bank. Development contains 1,260
rows and confirmation contains 2,520 independent rows. The primary representation
remains the historical symmetrized $\binom{289}{2}=41{,}616$ feature block so this
comparison is compatible with `proof_p90_260824.ipynb`.

The historical recipe is transductive: it estimates feature means and variances
using development and confirmation together, then drops incomplete rows. The new
recipes fit every statistic on development only.
"""
            ),
            code(
                """
from pathlib import Path
import sys
ROOT = Path.cwd().resolve()
while not (ROOT / 'pyproject.toml').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from umap import UMAP

from src.corpus_visualization import plot_embedding

sns.set_theme(context='notebook', style='ticks')
RESULTS = ROOT / 'results' / 'corpus_geometry'
comparison = pd.read_csv(RESULTS / 'geometry-comparison.csv')
proof_rows = comparison[(comparison.study == 'proof_p90') & comparison.recall_at_1.notna()].copy()
proof_rows[['recipe', 'retained_reference_rows', 'retained_query_rows',
            'retained_features', 'pca_cumulative_variance',
            'mean_average_precision', 'recall_at_1', 'recall_at_5']]
"""
            ),
            markdown(
                r"""
The centre-only covariance representation is retained as primary: it has the
highest independent-confirmation mean average precision and recall. Standardizing
each feature does not recover hidden low-variance signal here; it slightly reduces
retrieval. Robust scaling is materially worse. The historical recipe also removes
45 development and 80 confirmation rows, so it cannot represent the complete bank.
"""
            ),
            code(
                """
recipes = {}
for recipe in ('legacy', 'center', 'standard', 'robust'):
    with np.load(RESULTS / 'proof_p90' / f'{recipe}.npz', allow_pickle=True) as archive:
        recipes[recipe] = {key: archive[key] for key in archive.files}

fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
for ax, recipe in zip(axes, recipes):
    payload = recipes[recipe]
    labels = payload['query_class'].astype(str)
    coordinates = payload['query_pca'][:, :2]
    sns.scatterplot(x=coordinates[:, 0], y=coordinates[:, 1], hue=labels,
                    s=7, linewidth=0, alpha=.65, legend=False, ax=ax)
    ax.set(title=recipe, xlabel='PC1', ylabel='PC2')
fig.suptitle('Scaling sensitivity; known classes shown only for proof validation', y=1.04)
plt.show()
"""
            ),
            markdown(
                r"""
## PCA, UMAP and t-SNE

PCA and UMAP below use the frozen development-fitted centre-only representation.
t-SNE has no stable out-of-sample transform, so its confirmation map is explicitly
transductive and illustrative. Neither UMAP nor t-SNE is used to fit clusters.
"""
            ),
            code(
                """
center = recipes['center']
coordinates_path = ROOT / 'results' / 'cross_mt_transfer_260824' / 'confirmation-coordinates.npz'
with np.load(coordinates_path, allow_pickle=True) as archive:
    umap_coordinates = archive['confirmation_umap_sym']
tsne_coordinates = TSNE(n_components=2, perplexity=50, init='pca',
                         learning_rate='auto', max_iter=1500,
                         random_state=260824).fit_transform(center['query_pca'])
views = [('PCA', center['query_pca'][:, :2]), ('UMAP', umap_coordinates), ('t-SNE', tsne_coordinates)]
labels = center['query_class'].astype(str)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
for ax, (name, coordinates) in zip(axes, views):
    sns.scatterplot(x=coordinates[:, 0], y=coordinates[:, 1], hue=labels,
                    s=7, linewidth=0, alpha=.68, legend=False, ax=ax)
    ax.set(title=name, xlabel=f'{name} 1', ylabel=f'{name} 2')
fig.suptitle('Independent confirmation rows', y=1.04)
plt.show()
"""
            ),
            markdown(
                r"""
## Historical UMAP recipe

The original proof used $n_{\mathrm{neighbors}}=9$ and
$\mathrm{min\_dist}=0.75$. Here it is applied to the legacy PCA50 coordinates
rather than all retained features; the exact all-feature reproduction remains
available through `scripts/legacy_compat/proof_of_principle.py`. The PCA step makes
the notebook tractable and is the only deliberate deviation.
"""
            ),
            code(
                """
legacy = recipes['legacy']
legacy_umap = UMAP(n_neighbors=9, min_dist=.75, random_state=42, n_jobs=1).fit_transform(legacy['query_pca'])
fig, ax = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
sns.scatterplot(x=legacy_umap[:, 0], y=legacy_umap[:, 1],
                hue=legacy['query_class'].astype(str), s=8, linewidth=0,
                alpha=.7, legend=False, ax=ax)
ax.set(title='Historical preprocessing + old UMAP settings', xlabel='UMAP 1', ylabel='UMAP 2')
plt.show()
"""
            ),
            markdown("## Live point-to-MTS browser"),
            code(
                """
from src.corpus_visualization import interactive_embedding_browser

CONFIRMATION_FEATURES = ROOT / 'data' / 'proof_p90_260824' / 'features' / 'confirmation.npz'
RAW_ROOT = ROOT / 'data' / 'proof_p90_260824' / 'raw' / 'confirmation'
REMOTE_ROOT = Path('/g/data/ql44/we2614/mts-spi-study/representation/cross-mt/data/confirmation/cross_mt_confirmation_260824')
with np.load(CONFIRMATION_FEATURES, allow_pickle=True) as archive:
    remote_paths = archive['dataset_paths'].astype(str)
path_by_row = dict(zip(center['query_names'].astype(str), remote_paths))

def launch_proof_browser():
    def load_row(name):
        relative = Path(path_by_row[name]).relative_to(REMOTE_ROOT)
        return np.load(RAW_ROOT / relative / 'timeseries.npy')
    return interactive_embedding_browser(
        umap_coordinates,
        center['query_names'],
        load_row,
        title='Click a confirmation row to inspect its raw MTS',
        scaling='zscore',
    )

print('Run launch_proof_browser() in a live kernel to activate point-to-MTS selection.')
"""
            ),
        ]
    )
    return notebook


def global_notebook():
    notebook = base_notebook("SPI--SPI geometry of 1,053 heterogeneous MTS datasets")
    notebook.cells.extend(
        [
            markdown(
                r"""
Every dataset is shown neutrally. Primary cluster models are fitted in PCA space.
The final exploration section also reproduces the historical 2-D clustering as a
descriptive comparison. Grey density contours guide inspection; interpretation
and dynamical naming are post hoc.
"""
            ),
            code(
                """
from pathlib import Path
import sys
ROOT = Path.cwd().resolve()
while not (ROOT / 'pyproject.toml').exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.atlas_analysis import cluster_medoids
from src.corpus_geometry import nearest_neighbour_table
from src.corpus_visualization import (
    interactive_embedding_browser,
    plot_embedding,
    plot_mts_heatmap,
)

from scripts.explore_zenodo_geometry import paper_style, map_axis
paper_style()  # docs/benchmark-figure-style.md: serif, compact, unboxed, no grid
GEOMETRY = ROOT / 'results' / 'corpus_geometry'
ATLAS = ROOT / 'results' / 'zenodo_7118947' / 'atlas-seed1729'
comparison = pd.read_csv(GEOMETRY / 'geometry-comparison.csv')
comparison[(comparison.study == 'zenodo_1053') & comparison.retained_features.notna()][
    ['recipe', 'retained_reference_rows', 'retained_features', 'pca_cumulative_variance']
]
"""
            ),
            markdown(
                r"""
The historical rule excludes 30 complete datasets solely because their feature
validity falls below its row threshold. The fitted recipes retain all 1,053.
Scaling changes the geometry. Centre-only remains a working primary view, with
standardized PCA retained as a sensitivity analysis; proof-data retrieval is not
evidence that the same choice is universally optimal on this heterogeneous corpus.
"""
            ),
            code(
                """
with np.load(ATLAS / 'atlas-results.npz', allow_pickle=True) as archive:
    atlas = {key: archive[key] for key in archive.files}
summary = json.loads(atlas['summary_json'].item())
cluster_summary = pd.DataFrame([
    {'method': 'K-means', 'selected': f"k={int(atlas['kmeans_clusters'])}",
     'validated': True,
     'stability': summary['primary_kmeans']['subsample_stability']},
    {'method': 'GMM', 'selected': f"k={int(atlas['gmm_components'])}",
     'validated': bool(atlas['gmm_validated']),
     'stability': summary['diagnostic_gmm']['subsample_stability']},
    {'method': 'HDBSCAN', 'selected': f"k={len(set(atlas['hdbscan_labels'])) - (int(-1 in atlas['hdbscan_labels']))}",
     'validated': bool(atlas['hdbscan_validated']),
     'stability': summary['primary_hdbscan_subsample_stability']},
])
cluster_summary
"""
            ),
            markdown(
                "The stable K-means partition is a useful resolution of this representation, not a claim that the corpus has exactly eight natural dynamical classes. The fitted GMM and HDBSCAN views fail the prespecified stability threshold and remain diagnostics."
            ),
            code(
                """
views = [
    ('PCA', atlas['pca_scores'][:, :2]),
    ('UMAP', atlas['umap']),
    ('t-SNE', atlas['tsne']),
]
fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.25), constrained_layout=True)
for ax, (name, coordinates) in zip(axes, views):
    map_axis(ax, coordinates, atlas['kmeans_labels'], title=name, axes_name=name)
fig.suptitle('All 1,053 datasets; projected K-means density contours in grey', y=1.04)
plt.show()
"""
            ),
            markdown(
                r"""
## Historical distance-matrix recipe (not the old t-SNE preprocessing)

The adapter for `old/compute_distance_matrix.py` standardizes features, filters
at 90% feature / 80% row validity, fills missing standardized values with zero,
and uses Euclidean distance. Its PCA view contains 1,023 datasets. It omits the
old catalogue-specific exclusion lists and is not a full historical reproduction.
The actual Figure 4--5 t-SNE preprocessing is reproduced in the final subsection.
"""
            ),
            code(
                """
with np.load(GEOMETRY / 'zenodo_1053' / 'legacy.npz', allow_pickle=True) as archive:
    legacy = {key: archive[key] for key in archive.files}
fig, ax = plot_embedding(legacy['query_pca'][:, :2], title='Historical preprocessing',
                         xlabel='PC1', ylabel='PC2')
plt.show()
"""
            ),
            markdown("## GMM diagnostic view"),
            code(
                """
from matplotlib.patches import Ellipse

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.15), constrained_layout=True)
coordinates = atlas['pca_scores'][:, :2]
axes[0].scatter(coordinates[:, 0], coordinates[:, 1], s=7, c='.15', alpha=.6, linewidth=0)
for mean, covariance in zip(atlas['gmm_means'], atlas['gmm_covariances']):
    cov2 = covariance[:2, :2]
    eigval, eigvec = np.linalg.eigh(cov2)
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    angle = np.degrees(np.arctan2(eigvec[1, 0], eigvec[0, 0]))
    axes[0].add_patch(Ellipse(mean[:2], *(2 * np.sqrt(5.991 * np.maximum(eigval, 0))),
                              angle=angle, fc='.55', ec='.25', alpha=.10, lw=.7))
axes[0].set(title='Projected 95% component ellipses', xlabel='PC1', ylabel='PC2')

grid = pd.read_csv(ATLAS / 'cluster-model-grid.csv')
gmm = grid[grid.method == 'gmm'].copy()
for (dimension, covariance), group in gmm.groupby(['dimensions', 'covariance_type']):
    if dimension == int(atlas['gmm_dimension']):
        group = group.sort_values('clusters')
        axes[1].plot(group.clusters, group.bic - group.bic.min(), marker='o', ms=2.5,
                     lw=.9, label=covariance)
axes[1].set(title=f"BIC at PCA{int(atlas['gmm_dimension'])}", xlabel='Components', ylabel=r'$\\Delta$BIC')
axes[1].legend(frameon=False, fontsize=7)

probability = atlas['gmm_probability']
entropy = -np.sum(probability * np.log(probability + 1e-15), axis=1) / np.log(probability.shape[1])
axes[2].hist(entropy, bins=25, color='.25', edgecolor='white', lw=.3)
axes[2].set(title='Assignment uncertainty', xlabel='Normalized posterior entropy', ylabel='Datasets')
plt.show()
"""
            ),
            markdown(
                "The Gaussian mixture is useful for showing soft assignments and covariance geometry. Its low resampling stability means the 13-component solution should not be promoted to a corpus taxonomy."
            ),
            markdown("## Raw-MTS heatmap scaling"),
            code(
                """
DATABASE = ROOT / 'data' / 'zenodo_7118947' / 'database.npz'
examples = [
    'noise_cauchy_M5_T100',
    'wave-2D_M-25_T-1000',
    'mousefMRI_S-0_R-23-24',
    'epidemic_incidence_C40-54',
]
methods = [('legacy', 'Old cross-process robust'),
           ('zscore', 'Per-process z-score'),
           ('robust', 'Per-process robust')]
with np.load(DATABASE, allow_pickle=False) as database:
    fig, axes = plt.subplots(len(examples), len(methods), figsize=(11, 8), constrained_layout=True)
    for row, dataset in enumerate(examples):
        for col, (method, label) in enumerate(methods):
            plot_mts_heatmap(database[dataset], method=method, ax=axes[row, col],
                             title=(label if row == 0 else None))
            if col == 0:
                axes[row, col].set_ylabel(dataset + '\\nProcess', fontsize=7)
plt.show()
"""
            ),
            markdown(
                r"""
The adopted display is per-process robust scaling: subtract the temporal median
and divide by $\mathrm{IQR}/1.349$, then clip symmetrically at the within-MTS 99th
percentile of absolute scaled values. The alternatives above show what changes.
This display transform does not modify the SPI feature bank. PNG serves previews;
SVG with an embedded raster keeps the MTS panels small and usable in figure editors.
"""
            ),
            markdown("## Feature matrix and nearest-neighbour retrieval"),
            code(
                """
from IPython.display import Markdown, display
display(Markdown('[Open the full $1053\\times41616$ feature matrix]('
                 '../../results/zenodo_7118947/atlas-seed1729/figures/atlas-full-feature-matrix.png)'))

medoids = cluster_medoids(atlas['pca_scores'][:, :int(atlas['kmeans_dimension'])], atlas['kmeans_labels'])
rows = []
for cluster, index in sorted(medoids.items())[:4]:
    focal = str(atlas['dataset'][index])
    for rank, (dataset, distance) in enumerate(nearest_neighbour_table(
            atlas['pca_scores'][:, :int(atlas['kmeans_dimension'])], atlas['dataset'], focal, k=5), 1):
        rows.append({'cluster': cluster, 'focal': focal, 'rank': rank,
                     'neighbour': dataset, 'distance': distance})
pd.DataFrame(rows)
"""
            ),
            markdown(
                "Nearest neighbours are direct retrievals in the fitted PCA geometry. The focal datasets above are algorithmic medoids, not manually selected exemplars."
            ),
            markdown("## Live point-to-MTS browser"),
            code(
                """
# Call launch_browser() in a live kernel. It is deliberately not executed here:
# serializing a Plotly FigureWidget adds about 5 MB of widget state and callbacks
# are inactive in static PDF/HTML exports.
def launch_browser():
    database = np.load(DATABASE, allow_pickle=False)
    return interactive_embedding_browser(
        atlas['umap'],
        atlas['dataset'],
        lambda name: database[name],
        title='Click a dataset to inspect its raw MTS',
        scaling='robust',
    )

print('Run launch_browser() in a live kernel to activate point-to-MTS selection.')
"""
            ),
        ]
    )
    notebook.cells.extend(exploration_cells())
    return notebook


def exploration_cells():
    return [
        markdown(r"""
## Figures 4--5: historical method and visual sensitivity

The draft's Figure 4 is a corpus map with selected examples; Figure 5 collects
members of selected groups. Its original SPI similarity is Spearman-based.
Here every experiment uses our existing Pearson bank, $X\in\mathbb{R}^{1053\times41616}$,
with 289 SPIs and one correlation of aligned ordered off-diagonal entries per SPI pair.
Undefined features remain NaN in the canonical NPZ; no new pyspi computation is needed.

**Historical downstream recipe:** remove duplicate feature columns, retain rows
with at least 20% observed features, then columns with at least 80% observed rows;
fill raw NaNs with zero; direct t-SNE ($p=10$, seed 42, PCA initialization,
automatic learning rate, 1,000 iterations); contrib HDBSCAN with minimum cluster
size 5 and minimum samples 5. Defaults come from the available `CorrelationFrame`
implementation; the original execution environment/feature bank is unavailable.
The control panels isolate adding PCA50 and changing preprocessing.

**Current alternatives:** 95%-valid varying features, median imputation and
centering, covariance PCA at $d\in\{20,50,100\}$, followed by t-SNE at
$p\in\{10,30,50\}$ or UMAP at $n\in\{15,30,60\}$ (Euclidean, minimum distance 0.1).
Additional panels vary standardization, minimum distance and random seed.
PCA50, t-SNE $p=30$ and UMAP $n=30$ are moderate starting points for inspection,
not established optima. Truncation and small neighbourhoods can hide or split structure.

K-means ($k=8,20,40,60$), diagonal GMM ($k=8,20,40$, PCA20), and HDBSCAN EOM/leaf
partitions explore broad and fine resolutions in PCA space. Separate map partitions
repeat HDBSCAN in 2-D. Grey KDE mass contours (50%/80%) are visual summaries, not
confidence regions. All points, including noise, remain visible. Cluster names are yours
to assign after inspection; a visually attractive separation is not validation.

**Observed:** the historical adaptation yields 45 clusters and 138 noise rows.
PCA50 + t-SNE $p=30$ is a useful first inspection view: full-feature neighbour
recall is 0.492 versus 0.494 for PCA100, so this metric gives little reason to
prefer the larger input. For UMAP, $n=15$ preserves more local neighbours than
$n=30$ or $60$ here; PCA100/$n=15$ reaches 0.425. For a finer PCA-space partition,
K-means PCA20/$k=20$ has conditional resampling ARI 0.832; PCA20/$k=40$ falls to
0.720. These are exploratory comparisons, not a validation of 20 natural classes.

[t-SNE guidance](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
supports PCA pre-reduction; [UMAP guidance](https://umap-learn.readthedocs.io/en/latest/clustering.html)
warns that 2-D embeddings can introduce false separations. Cluster quality is assessed in
the fitted space. BIC compares only the GMMs fitted to the same PCA20 representation.
"""),
        code("""
from IPython.display import Image, Markdown, display
EXPLORE = ROOT / 'results/zenodo_7118947/visual-exploration'
display(Markdown(f'[Open offline cluster inspection and SVG/PNG exports]({(EXPLORE / "index.html").as_uri()})'))
display(pd.read_csv(EXPLORE / 'pca-sensitivity.csv').round(3))
display(Image(filename=str(EXPLORE / 'figures/historical-controls.png')))
"""),
        code("""
display(Image(filename=str(EXPLORE / 'figures/tsne-sensitivity.png')))
display(Image(filename=str(EXPLORE / 'figures/umap-sensitivity.png')))
"""),
        code("""
display(Image(filename=str(EXPLORE / 'figures/scaling-and-spacing.png')))
display(Image(filename=str(EXPLORE / 'figures/cluster-resolutions.png')))
metrics = pd.read_csv(EXPLORE / 'cluster-metrics.csv')
selected = ['map-historical', 'map-tsne-pca50-p30', 'kmeans-pca20-k20',
            'kmeans-pca20-k40', 'kmeans-pca50-k20', 'kmeans-pca50-k60',
            'gmm-pca20-k20', 'hdbscan-pca50-leaf-m10']
display(metrics[metrics.partition.isin(selected)][
    ['partition', 'space', 'clusters', 'coverage', 'median_size', 'subsample_ARI']].round(3))
"""),
        markdown(r"""
Full numeric outputs: `exploration.npz` (coordinates, PCA scores, partitions),
`memberships.csv` (every dataset in every partition), `medoids.csv` (actual members
minimizing total within-cluster distance), and metric CSVs. The offline gallery allows
point/cluster selection, name filtering, raw-MTS inspection and CSV export of manually
labelled subsets. Heatmap filenames map to dataset names in `heatmap-index.csv`.

The canonical feature bank is local at
`data/zenodo_7118947/features/pearson-unified-v3-seed1729.npz` (`X`, validity mask,
SPI-pair schema, metadata/provenance); raw MTS are in `data/zenodo_7118947/database.npz`.
The recorded Gadi mirror is `/g/data/ql44/we2614/mts-spi-study/zenodo/7118947/`.
The raw matrix occupies 167 MiB as float32; local analysis suffices.
Run `.venv/bin/python -m scripts.explore_zenodo_geometry` to reproduce, or append
`--figures-only` to render cached results. The script requires `hdbscan==0.8.44`.
"""),
        markdown(r"""
### Higher resolution and reviewed groups

Added $k=80,120,160,200$ on PCA20/50/100, preserving existing embeddings and
partition IDs. The historical map also has HDBSCAN leaf/minimum-size-5/10 variants.
The old code's active method was HDBSCAN size 5; its optional K-medoids branch
fixed $k=200$ in the 2-D map. Our high-$k$ comparison uses K-means in PCA space.

For browsing, start at **PCA100/$k=80$**, then compare PCA50/$k=80$ and $120$.
At $k=80$, median cluster size is 10 and approximately 4% of rows lie in groups
smaller than five; at $k=200$, median size is 4 and roughly a quarter lie in such
small groups. Higher $k$ therefore adds fragmentation as well as useful specificity.
PCA dimension also matters: PCA100/$k=80$ isolates all ten walking recordings,
whereas some higher-$k$ PCA20 partitions still mix walking with other activities.

The shortlist shows **all members** of each selected group. Homogeneous/heterogeneous
describe whether the explicitly listed classes are the same/different; real/synthetic/mixed
describe origin separately. Neither axis establishes dynamical coherence. Titles describe
source composition, not inferred mechanisms. Group-level Jaccard matches across five
80% subsamples supplement global ARI; PCA remains fixed. A complete-feature
sensitivity uses the 9,313 varying features observed in every MTS, requiring no
imputation. The mixed financial/Brownian/fMRI group is explicitly provisional.
The report also shows measured dynamics and nearest excluded records. An exact
prefix check finds the excluded M16/T100 wave record inside longer included wave
records: that boundary reflects observation/estimation effects, not a different
underlying system. Nested windows require grouped rather than independent-row
resampling before making stronger stability claims.
"""),
        code("""
display(Markdown(f'[Open reviewed groups and full member sheets]({(EXPLORE / "shortlist.html").as_uri()})'))
high_k = pd.read_csv(EXPLORE / 'high-k-metrics.csv')
display(high_k[high_k.dimensions.eq(100)][
    ['partition', 'median_size', 'rows_in_under5_fraction', 'subsample_ARI']].round(3))
display(Image(filename=str(EXPLORE / 'figures/high-k-resolutions.png')))
shortlist = pd.read_csv(EXPLORE / 'shortlist.csv')
display(shortlist[['title', 'composition', 'origin', 'partition', 'cluster', 'size', 'jaccard',
                   'full_silhouette', 'complete_silhouette']].round(3))
"""),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = {
        OUT / "proof_p90_geometry_comparison.ipynb": proof_notebook(),
        OUT / "zenodo_1053_geometry.ipynb": global_notebook(),
    }
    for path, notebook in targets.items():
        nbf.write(notebook, path)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
