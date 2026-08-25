# Zenodo MTS corpus map

Active workstream as of 2026-08-26. This is exploratory representation analysis;
proximity is SPI interaction-profile similarity, not established mechanistic identity.

## Verified source

- Zenodo 7118947 v1.1 contains 1,053 named `M x T` arrays. The verified local,
  pickle-free archive is `data/zenodo_7118947/database.npz`, SHA-256
  `928ce41f...b77ba84`; metadata is in the adjacent `manifest.json`.
- Observed ranges are `M=5..29`, `T=30..3000`; 1,052 arrays are float64 and one is
  int64. There are no non-finite values or constant channels. `sim1` and `sim21`
  are exact duplicates.
- Labels comprise 548 real, 487 synthetic and 18 HCP fMRI rows without a
  real/synthetic tag. Source tags are multi-label and must not be treated as one
  mutually exclusive ground truth.

## Implemented boundary

- `src/run_external_corpus.py` and `configs/corpora/zenodo-7118947-p90.yaml` provide
  a dedicated YAML API. Each task reads one member directly from the shared NPZ,
  transposes to `T x M`, preserves source data and asks pyspi to z-score each process.
- The config requires an explicit RNG seed and the runner pins/restores NumPy/Python
  global state per dataset. Feature reconstruction accepts both legacy class/dataset
  and direct experiment/dataset layouts, but never mixes experiment banks.
- Output is one atomic `spi_mpis.npz` plus `meta.json` per dataset; completion checks
  bind source/config/runner/compute/pyspi identities and validate MPI shape/catalogue.
- Gadi uses one process/core/dataset through `nci-parallel`; no PBS array and no copied
  per-dataset input files. Source and outputs belong on gdata via a dataset symlink.
- Local/Gadi contract tests and full source validation pass. Gadi smoke job
  `177419496` (9 heterogeneous datasets) and stratified scout `177420118` (48,
  non-overlapping) both audited complete. Measured runtime was 34--1,593 s and
  peak memory was 42.4 GB across 48 tasks.
- Production is partitioned by the transparent measured proxy `M^2*T`: fast
  `<100k` (807 datasets, 288 cores/6 nodes), medium `100k..<400k` (203,
  96 cores/2 nodes), slow `>=400k` (43, 48 cores/1 node). Corresponding index
  files are under `configs/corpora/`; `--skip-existing` validates/reuses 57 pilots.
- Seeded production jobs `177429006/007/010` exited 0 in 6:30/12:05/27:41 with
  807/807, 203/203 and 43/43 outputs; all 1,053 audit clean. Maximum allocation
  was 432 cores on 9 nodes (427 useful workers); observed aggregate memory was
  about 344 GiB against 1.69 TiB requested. Reconstruction job `177430243` exited
  0 in 33 s at 5.7 GiB; the final 119-MB artifact is exactly `1053 x 41,616`,
  records seed 1729 and has SHA-256 `dc28dfd3...21500634`.

## Analysis boundary

- Primary reconstruction is Pearson `unified_ordered_v3`: correlate complete aligned
  ordered off-diagonal MPI entries for every SPI pair, producing one raw
  `1053 x 41,616` matrix for 289 SPIs plus validity mask/schema/provenance. This
  preserves aligned direction for directed--directed comparisons. Reverse-edge
  comparison and per-SPI self-reciprocity belong only in an optional v2 sensitivity,
  not the primary `K choose 2` atlas.
- The 57-pilot real-data smoke artifact has the exact `57 x 41,616` shape. Its raw
  row-validity range is `.478--.952`; 16,419 features are valid in at least 95% of
  pilot rows and 25,451 in at least 90%. Re-estimate all missingness gates on the
  complete corpus; preserve NaNs and compare 90/95/100% feature-validity sensitivity.
- Full-corpus raw row validity is `.464--.966` (median `.841`); 24,956/23,782/10,878
  features meet 90/95/100% validity before variance gating. Missing SPIs are retained
  as NaN provenance and are not interpreted as failed MTS datasets.
- The unseeded duplicate control exposed 12 RNG-dependent SPIs. Seeded `sim1`/`sim21`
  and a repeated ill-conditioned wave dataset are bit-identical across all 289 MPIs.
  Relative to the provisional artifact, seeded 50-PC distance Spearman is `.9999`
  and 15-NN overlap `.972`; residual changes in 23 spectral SPIs occur on only 1--5
  ill-conditioned wave rows and are reproducible under the seeded runner.
- Final atlas job `177430329` exited 0 in 14:15 (1:29:49 CPU, 2.22 GiB peak) and
  retained 21,788 varying features at the 95% gate. Stable K-means resolutions are
  PCA10 `k=8` (subsample ARI `.955`) and PCA80 `k=2` (`.971`); the configured
  near-tie/parsimony rule selects the former. GMM PCA10/full `k=13` (`.587`) and
  HDBSCAN `k=2` (`.511`) fail the `.70` stability gate. There is no unique inferred
  cluster count; GMM/HDBSCAN are diagnostic views.
- Fit clustering in a preprocessed PCA/meta-feature space, never in t-SNE coordinates.
  UMAP/t-SNE are visualizations. Compare stable GMM, HDBSCAN and graph/consensus
  solutions using resampling stability and method-appropriate criteria, not appearance.
- `src/run_atlas_analysis.py` is configured by
  `configs/analysis/zenodo-7118947-atlas.yaml`. It compares 90/95/100% validity
  gates; covariance PCA at 10/20/40/80 dimensions; GMM `k=1..15` with
  diagonal/tied/full covariance where estimable; K-means and HDBSCAN; and seeded
  UMAP/t-SNE grids. BIC selects GMM form only within a fixed PCA dimension;
  resampling stability selects/validates dimension and partitions. Pilot smoke
  passes; its GMM solution is correctly flagged unvalidated at ARI `.415`.
- `src/run_catch22_corpus.py` implements the 94-corpus precedent's 22-per-channel,
  min/Q1/mean/Q3/max aggregation as a 110-feature control. All 1,053 local datasets
  completed with finite values and no channel errors. Against the final atlas,
  pairwise-distance Spearman was `.385` and 15-NN overlap `.320` versus `.014` random;
  the spaces are related but non-equivalent, and Catch22 had slightly greater source-tag
  neighbourhood homogeneity, so the current evidence is not a superiority result.
- Quantify `M,T`, estimator-validity and broad-tag leakage; compare simple raw-series
  baselines. Treat tag enrichment as post-hoc characterization with multiplicity control.
- Do not claim novelty or mechanistic discovery without a literature audit and external
  validation. The broad atlas claim is already false: [Navarro et al. (PMLR
  2023)](https://proceedings.mlr.press/v224/navarro23a.html) projected 94
  heterogeneous MTS datasets in aggregated Catch22 meta-feature space; only the
  SPI interaction-profile construction/scale is a plausible narrower novelty.
