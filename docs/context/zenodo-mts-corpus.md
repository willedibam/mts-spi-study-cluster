# Zenodo MTS corpus map

Active workstream as of 2026-08-25. This is exploratory representation analysis;
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
- Local contract tests and full source validation pass. No Gadi job from this workstream
  had been submitted when this entry was written.

## Analysis boundary

- Primary reconstruction is Pearson `direction_preserving_v2`: retain raw `X_sym`
  (`1053 x 41,616` for 289 SPIs) and `X_dir` (`1053 x 32,079` for 111 directed SPIs),
  with masks/schema/provenance. Direction cannot be represented faithfully by only a
  single `K choose 2` upper triangle.
- Fit clustering in a preprocessed PCA/meta-feature space, never in t-SNE coordinates.
  UMAP/t-SNE are visualizations. Compare stable GMM, HDBSCAN and graph/consensus
  solutions using resampling stability and method-appropriate criteria, not appearance.
- Quantify `M,T`, estimator-validity and broad-tag leakage; compare simple raw-series
  baselines. Treat tag enrichment as post-hoc characterization with multiplicity control.
- Do not claim novelty or mechanistic discovery without a literature audit and external
  validation.
