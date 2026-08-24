# Cross-M,T transfer

Active workstream as of 2026-08-24. Facts are marked **verified**; unexecuted design choices are marked **frozen proposal** until the development manifest is committed. This study tests quantitative transfer, not mathematical invariance.

## Verified starting state

- The current pyspi-v3 proof bank has 900 rows: ten classes, `M={8,16,32}`, `T={500,1000,2000}`, instances `0:9`; feature artifact `/scratch/ql44/we2614/spi-spi-direction-v2/proof.npz` has 289 SPIs/111 directed, `41,616` symmetric and `32,079` directional features, schema `803abbe5…`, and complete pyspi `3.0.0.r7` provenance.
- Goal 1 compact JSONs were copied to the ignored local path `results/spi_spi_direction_v2/`; proof/CML-panel/CML-order/Kuramoto SHA-256 are `94cb7eba…`, `c690e6f1…`, `c243b94d…`, `08ccfe81…`. Large feature banks remain on Gadi.
- The existing six-class CML panel cannot be pooled as development evidence: it contains 297 SPIs from `configs/pyspi-v2/benchmarked90_amortized_config.yaml`, lacks config/version provenance, and has a different schema. Its four non-overlapping classes must be recomputed under current p90; the two overlapping classes (`defect-turbulence`, `sti-i`) already exist in the proof bank.
- The four additions are `frozen-chaos`, `brownian-defect`, `fdstc`, and `chaotic-traveling-wave`, yielding 14 classes. Existing dirty notebooks are out of scope and must not be overwritten.

## Generation boundary

- Development additions: `configs/generate/embeddings/cross-mt-cml-development-260824.yaml`, instances `0:9`, 360 rows under gdata. Confirmation: `cross-mt-confirmation-260824.yaml`, instances `10:29`, 2,520 rows under a separate gdata root. Overlapping generator parameters exactly match the original proof config.
- Explicit instance lists preserve the untouched split. Gadi farms are selected by one `(M,T)` cell, use one core/dataset, and emit only `timeseries.npy`, `spi_mpis.npz`, `meta.json` and logs. Confirmation generation must not begin until the common schema and analysis manifest pass development checks and are committed.

## Frozen proposal before confirmation

- Primary: `z_sym`. Prespecified sensitivities: `z_dir` and development-total-variance-balanced `[z_sym,z_dir]`.
- Each leave-one-`(M,T)`-cell-out fold fits validity filtering, median imputation, centring, PCA and multinomial logistic regression only on instances `0:9` from the other eight cells; evaluation uses instances `10:29` from the held cell.
- Complementary tests: cross-cell class retrieval; per-class prediction of `M`, `T` and their joint cell (size leakage); additive class-versus-cell geometry in one development-fitted shared PCA space. Shared PCA/UMAP figures are illustrative, not inferential.
- Dimension-compatible raw-data baselines will pool channelwise temporal/marginal summaries and pairwise-dependence distributions; their scaling/models use development rows only.
- Report per-cell results, stratified bootstrap intervals and held-label permutation nulls. No confirmation labels may tune schema, validity gates, PCA dimension, model hyperparameters, metrics, nulls or plotting choices.

## Open until frozen

- Exact preprocessing thresholds, baseline descriptors, model parameters, bootstrap/permutation counts and geometry/retrieval definitions require implementation tests and a content-addressed development manifest.
- Confirmation runtime/memory caps require a two-row smoke and representative one-cell timing gate. No confirmation PBS job has been submitted.
