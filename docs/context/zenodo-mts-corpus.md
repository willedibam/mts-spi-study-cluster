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
- Production jobs `177423444/475/512` exited 0 in 6:51/12:01/23:06 with
  807/807, 203/203 and 43/43 outputs. A combined audit validated all 1,053 with
  no failures. Unified reconstruction job `177427052` exited 0 in 43 s, peaking
  at 10 GB; its artifact is 119 MB and exactly `1053 x 41,616`. The old
  48-core/192-GB reconstruction default was reduced to 8 cores/32 GB.

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
- The first complete atlas retained 21,788 varying features at the 95% gate. Its
  only strongly stable partition was a coarse PCA80 K-means split (`k=2`, subsample
  ARI `.971`); finer GMM (`k=10`, ARI `.614`) and HDBSCAN (`k=7`, ARI `.620`) views
  were not validated, and method agreement was low. Treat these results as provisional:
  the exact `sim1`/`sim21` source duplicate exposed 12/289 RNG-dependent SPIs. The
  runner now pins and records corpus-wide seed 1729, restores caller RNG state, and
  writes to the seed-labelled experiment directory; the duplicate smoke test is
  bit-identical across all 289 MPIs. Rerun before final interpretation.
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
  completed with finite values and no channel errors. Against the provisional atlas,
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
