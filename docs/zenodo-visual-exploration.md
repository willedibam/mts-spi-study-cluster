# Zenodo corpus: Figure 4--5 inspection

## Explorer v2

Open `results/zenodo_7118947/visual-exploration-v2/index.html`. Four views separate feature-space analysis, historical reproduction, PCA-only axes, and UMAP curiosity. GMM is omitted. V1, its shortlist, coordinates and memberships remain unchanged; v2 references its existing robust per-process heatmap assets, so keep both directories together.

```bash
.venv/bin/python -m scripts.explore_zenodo_v2
# Rebuild the page and static figures without refitting:
.venv/bin/python -m scripts.explore_zenodo_v2 --render-only
```

Configuration: `configs/analysis/zenodo-visual-exploration-v2.yaml`. Default: PCA100/t-SNE p30 displayed with PCA100 K-means k80. These are inspection settings, not an inferred optimal taxonomy. Primary preprocessing is unchanged. Exact, unwhitened SVD supplies PCA50/100/200 and full-rank scores; the latter's Euclidean distances are explicitly checked against the full retained-feature matrix before using them for full-space K-means. This accelerates computation without truncating the corpus geometry. V2 recomputes this PCA rather than silently mixing it with v1's randomized PCA.

The sweep includes full-input and PCA50/100/200 t-SNE p10/30/50, one PCA100 p100 view, two additional PCA100/p30 seeds, PC1–2/1–3/2–3, and UMAP on PCA50/100 with neighbours 15/30/60 plus PCA100/n30 min_dist 0/0.5. Primary t-SNE uses PCA initialization, automatic learning rate, early exaggeration 12 and 1,500 iterations; fit times exclude diagnostics and preceding PCA. K-means k40/80/120/160 runs in each primary space with n_init=20. HDBSCAN EOM/leaf uses min_samples=5 and min_cluster_size=5/10/20 on PCA50/100. K-means seed ARI holds all rows and preprocessing fixed: it is initialization sensitivity, **not** independent-recording resampling stability.

Historical p10 coordinates are copied unchanged from the verified v1 source bank; recomputed EOM HDBSCAN labels must match v1 exactly. Historical p30/p50 are explicitly variations, not the original implementation. Map EOM/leaf comparisons use min_cluster_size=min_samples=5. The old optional K-medoids k200 branch is a small NumPy compatibility implementation of the documented Euclidean/heuristic/alternate defaults (strict cost improvement, at most 300 iterations), because the released sklearn-extra binary is incompatible with NumPy 2. This is not the historical binary; neither it nor the main historical adaptation recreates the unavailable original Spearman feature bank/environment.

Changing an embedding does not refit a partition. Feature-space memberships remain available across all views. Map partitions default to the displayed map; an explicit comparison checkbox exposes other maps' partitions and displays a mismatch warning. Noise is retained. Click a point or shaded group, inspect its members/medoid, filter names, paginate, select arbitrary subsets, and export CSV with coordinates, partition, label and asset links. Selection is session-only. View links preserve controls, not selections. Heatmap PNG/SVG links open separately; current map SVG/PNG exports preserve the visible selection and zoom. Static clean map exports and CSV metrics sit alongside the page.

“Group shading” selects None, KDE, or Convex hulls. The embedded `scripts/zenodo_kde.js` computes independent Gaussian KDEs in the displayed 2D coordinates for nonnegative cluster labels only, using Scott bandwidth ×1.2 and a 48×48 grid, with interpolated approximate 80%/50% probability-mass contours. Fewer than three or rank-deficient points have no KDE. Density tails are clipped to the plot frame; all memberships, coordinates and axis limits stay fixed. Contours are descriptive, not confidence regions or additional clustering. Unassigned/excluded labels never contribute a pooled KDE or hull. Clicking an unassigned point shows its MTS and switches to the unassigned group, just like other point clicks; this does not create a KDE for that group. Contours are clickable and are included in current-map SVG/PNG downloads. The default shading is None.

`exploration.npz` stores row-aligned coordinates/partitions, full-rank PCA scores, the first 200 component vectors, fitted filtering/imputation/centring arrays, explained variance, and the full retained-feature Euclidean distance matrix. It loads with `allow_pickle=False`; the raw feature bank remains separate and unchanged. `manifest.json` records source hash, settings and dependency versions. Group silhouettes and representative medoids use the same full retained-feature geometry; HDBSCAN silhouette summaries cover assigned rows only.

V2 plots use square axes with equal coordinate scaling and all four spines, including SVG/PNG exports, as requested. This is a deliberate exception to the benchmark style's usual unboxed axes; the remaining typography and formatting are retained.

Dot area increases with process count M on one fixed corpus-wide scale: SVG radius is `sqrt(4 + 12*(M-Mmin)/(Mmax-Mmin))`, spanning 2–4 units for this corpus's M=5–29. Static plots use the same relative areas, and both formats include a size key. Selection changes colour/outline, not the M-encoded radius. Coordinates and memberships are unchanged.

The optional “Real / synthetic shading” control defaults on. Archive tags plus `configs/corpora/zenodo-7118947-label-corrections.yaml` give 566 dark-gray real (`#333333`) and 487 lighter-gray synthetic (`#969696`) markers with thin white outlines: the correction adds `real` to 10 `hcp_tfMRI_*` and 8 `hcp_rsfMRI_*` recordings. Original archive metadata is retained unchanged; `dataset-metadata.csv` exports original and corrected tags. Other missing tags remain unlabelled rather than guessed. Source origin appears in tooltips, inspection cards and selection CSV; blue selection overrides origin fill. Current-map SVG/PNG exports preserve this setting; the separate “Neutral map SVG / PNG” links remain neutral, also with white outlines. Origin is a display annotation only, never a clustering input.

Important historical display difference: `old/plot_clusters.py:114` removes `cluster == -1` before both KDE shading and scatter plotting; `old/plot_dataspace.py` likewise uses `plot_nas=False`, implemented in `old/utils.py:537`. A figure where every visible point has a cluster therefore does not establish 100% assignment. V2 keeps unassigned points visible, unlike that display filter. Historical EOM assigns 915/1,053 (45 groups, 138 unassigned); leaf assigns 687 (61 groups, 366 unassigned). The old gray shades were assigned randomly per cluster, not by real/synthetic origin. Its KDEs are smoothed density guides, whereas v2's optional hulls are convex envelopes; neither creates cluster memberships.

## Original explorer

Run locally from the repository root:

```bash
uv pip install --python .venv/bin/python --no-deps hdbscan==0.8.44
.venv/bin/python -m scripts.explore_zenodo_geometry
```

Configuration: `configs/analysis/zenodo-visual-exploration.yaml`. Rendering alone: append `--figures-only`. No pyspi rerun or cluster access is required.

Open `results/zenodo_7118947/visual-exploration/index.html` directly in a browser. Choose an embedding, partition and cluster; click points to inspect the raw MTS; select any subset, enter a post-hoc label, and download the selected rows as CSV. Selections are session-only until downloaded. PNG/SVG links provide Figure 5 assets. All 1,053 rows have heatmaps, including density-clustering noise. No dynamical labels are assigned automatically. The notebook's last subsection presents the comparisons.

## Data and interpretation

- Canonical bank: `data/zenodo_7118947/features/pearson-unified-v3-seed1729.npz`. `X` is a raw `1053 x 41616` float32 matrix, with NaNs, validity mask, SPI-pair identifiers and provenance. This is 167 MiB expanded, 118 MiB compressed. Each feature is Pearson correlation between aligned ordered off-diagonal MPI entries for two of the 289 p90 SPIs. Symmetric and directed SPIs share one bank.
- Raw MTS: `data/zenodo_7118947/database.npz`, one named `M x T` array per dataset.
- Recorded Gadi mirror: `/g/data/ql44/we2614/mts-spi-data/zenodo_7118947/`.
- Current preprocessing: features >=95% finite, median imputation, discard standard deviations <1e-8, mean centering. This retains 21,788 features. The atlas stores PCA100 scores locally; these supply PCA20/50/100 comparisons. Standardization is a separate sensitivity, with its own fitted PCA.
- Historical downstream preprocessing: available `CorrelationFrame` defaults remove duplicate features, filter rows at 20% finite, then features at 80%, and fill missing values with raw zero. No z-score or preliminary PCA. On this bank it retains 1,053 rows and 28,066 features. t-SNE uses perplexity 10, seed 42, PCA initialization, automatic learning rate and 1,000 iterations; contrib HDBSCAN uses minimum cluster size/minimum samples 5 and EOM selection. This is distinct from the old distance-matrix/feature-selection scripts.
- The draft uses Spearman-based MPI similarities, a different SPI catalogue and an unavailable historical environment. Matching downstream settings on our Pearson bank is a method adaptation, not an exact reconstruction of its figures.
- Clustering in PCA space and clustering the plotted 2-D map answer different questions. Map partitions are descriptive; t-SNE/UMAP may introduce separations. K-means/GMM resampling ARI is conditional on the frozen PCA preprocessing, not an end-to-end uncertainty estimate. No unique cluster count is asserted.
- Observed historical adaptation: 45 clusters, 138 noise rows. Start inspection with PCA50/t-SNE perplexity 30 (full-primary-feature 15-NN recall 0.492 versus 0.494 for PCA100), and compare UMAP PCA100/neighbours 15 (recall 0.425). PCA20 K-means k=20/40 has conditional subsample ARI 0.832/0.720; finer partitions are less stable. These are exploratory operating points, not a final taxonomy.
- KDE shading encloses approximately 50%/80% of each cluster's projected density; it is not a confidence interval or a generative-model boundary.

## Outputs

All outputs are under `results/zenodo_7118947/visual-exploration/`:

- `exploration.npz`: named embeddings, row indices, PCA scores and partition labels; load with `allow_pickle=False`. `manifest.json`: settings, versions and provenance.
- `memberships.csv`: every row's cluster in each partition (`-1` noise; `-2` excluded).
- `medoids.csv`: actual observed member minimizing within-cluster Euclidean distance in the clustering space. `cluster-metrics.csv`: sizes, coverage, silhouette, conditional resampling ARI, and GMM BIC (comparable within PCA20 only).
- `embedding-metrics.csv`: trustworthiness and neighbour recall relative to each input, plus recall against the common full primary feature geometry. Compare the common reference when judging different PCA truncations.
- `figures/`: editable SVG maps and PNG notebook previews; formatting follows `docs/benchmark-figure-style.md` with neutral points and grey contours.
- `heatmaps/`: one PNG and one single-raster SVG per dataset; the row-to-name map is `heatmap-index.csv`. Temporal median/(IQR/1.349) scaling per process, then symmetric clipping at the MTS-wide 99th percentile of absolute values, as in the notebook. These limits differ across MTS; colours show within-series structure, not absolute amplitude comparisons. All original samples are retained; gallery widths are equal.

Guidance: [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [UMAP parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html), [UMAP clustering caveats](https://umap-learn.readthedocs.io/en/latest/clustering.html).

## Higher resolution and reviewed groups (2026-09-10)

After the initial run, extend the cached results and render the shortlist:

```bash
.venv/bin/python -m scripts.refine_zenodo_clusters
.venv/bin/python -m scripts.render_zenodo_shortlist
```

This adds K-means k=80/120/160/200 on PCA20/50/100 and four 2-D HDBSCAN leaf variants, for 40 partitions in the same gallery. Existing coordinates and partition IDs are preserved. Rerunning the original full exploration replaces the cache, so run the extension again afterwards. `--figures-only` preserves the extended cache.

Start browsing with PCA100/k80 (median 10 members; about 4% of rows in groups <5). At k200, the median is four and about a quarter of rows are in groups <5; it helps some local subsets but is too fragmented as a default. Group-specific stability is more informative here than selecting k solely by global ARI. Five 80% subsamples are matched by maximum Jaccard to each original group; PCA is held fixed. The source tags were inspected after fitting, never used as clustering inputs.

`high-k-metrics.csv`, `small-cluster-audit.csv`, `small-cluster-members.csv` and `full-feature-medoids.csv` provide resolution, per-group diagnostics, source metadata, and medoids in the full retained feature space. HDBSCAN silhouettes use assigned rows only and cannot be compared directly with full-coverage K-means scores. `high-k-resamples.npz` preserves the replicate assignments.

`shortlist.html` contains seven complete member sheets, editable SVGs and direct gallery links. `configs/analysis/zenodo-cluster-shortlist.json` defines the reviewed choices and provisional descriptions. Foreign-exchange, walking, wave-2D and LEGION are the clearer source-family examples; wave-1D/hysteresis and articulation/EigenWorms are cross-source candidates. The financial/Brownian/HCP group is intentionally marked weaker. Raw heatmaps were visually inspected; differing M,T and per-MTS colour limits must not be mistaken for physical time/amplitude equivalence.

Wave-2D and wave/hysteresis have only 79% and 63% observed entries within retained features. Their mean silhouette remains positive (0.60 and 0.16) on the 9,313 varying features observed for every row, so imputation is not the sole evidence for grouping. This sensitivity does not establish a mechanism or rule out all estimator artefacts.

The old active implementation was direct t-SNE p10 followed by EOM HDBSCAN size5; its optional K-medoids branch used k200 in 2D. Our extension uses PCA-space K-means, not that optional branch. The EOM/leaf figure isolates cluster extraction on the same maps; [leaf selection](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html) can provide finer groups without specifying k.

Shortlist classification has two explicit axes: `composition` (homogeneous means one listed class; heterogeneous means several) and `origin` (real/synthetic/mixed). The `classes` list states the granularity used; parameter or M,T changes do not automatically define a different class. These categories are not quality scores. Each group's expandable inspection table now includes its nearest excluded MTS and transparent sample-scale dynamics (`shortlist-dynamics.csv`): median lag-1 correlation, median channel linear-trend R2, and shared-PC1 fractions before/after detrending. No physical-frequency or mechanistic equivalence is inferred. In particular, the financial/Brownian/fMRI group shares strong trends, but fMRI lag-1 correlation is .80 versus .99-1.00 for its other members; EigenWorms and articulatory recordings differ substantially in trend contribution. These qualify the earlier visual impressions instead of automatically confirming the groupings.

Exact raw-array check: `wave-1D_M-16_T-100` (nearest excluded row for the wave/hysteresis group) is a prefix of the included T500/T1000 versions. That particular cluster boundary cannot indicate a change in underlying system. The report now lists exact prefix relationships within groups and to their nearest outsiders. Nested windows are not independent replicates; future stability claims should group resampling by underlying recording/simulation, not count each window as independent. This check is stronger evidence about this boundary than its 2-D shape.
