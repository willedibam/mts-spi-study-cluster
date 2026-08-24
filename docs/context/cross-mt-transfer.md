# Cross-M,T transfer

Active workstream as of 2026-08-24. Facts are marked **verified**. The Pearson development protocol is frozen; this study tests quantitative transfer, not mathematical invariance.

## Verified starting state

- The current pyspi-v3 proof bank has 900 rows: ten classes, `M={8,16,32}`, `T={500,1000,2000}`, instances `0:9`; feature artifact `/scratch/ql44/we2614/spi-spi-direction-v2/proof.npz` has 289 SPIs/111 directed, `41,616` symmetric and `32,079` directional features, schema `803abbe5…`, complete pyspi `3.0.0.r7` provenance, and Pearson SPI-edge aggregation.
- Goal 2 uses Pearson SPI-edge aggregation, matching the existing proof artifact and the established proof construction. A completed Spearman development freeze from job `177252878` is non-authoritative and must not release confirmation; it is retained only as an audit artifact.
- Goal 1 compact JSONs were copied to the ignored local path `results/spi_spi_direction_v2/`; proof/CML-panel/CML-order/Kuramoto SHA-256 are `94cb7eba…`, `c690e6f1…`, `c243b94d…`, `08ccfe81…`. Large feature banks remain on Gadi.
- The existing six-class CML panel cannot be pooled as development evidence: it contains 297 SPIs from `configs/pyspi-v2/benchmarked90_amortized_config.yaml`, lacks config/version provenance, and has a different schema. Its four non-overlapping classes must be recomputed under current p90; the two overlapping classes (`defect-turbulence`, `sti-i`) already exist in the proof bank.
- The four additions are `frozen-chaos`, `brownian-defect`, `fdstc`, and `chaotic-traveling-wave`, yielding 14 classes. Existing dirty notebooks are out of scope and must not be overwritten.

## Generation boundary

- Development additions: `configs/generate/embeddings/cross-mt-cml-development-260824.yaml`, instances `0:9`, 360 rows under gdata. Confirmation: `cross-mt-confirmation-260824.yaml`, instances `10:29`, 2,520 rows under a separate gdata root. Overlapping generator parameters exactly match the original proof config.
- Explicit instance lists preserve the untouched split. Gadi farms are selected by one `(M,T)` cell, use one core/dataset, and emit only `timeseries.npy`, `spi_mpis.npz`, `meta.json` and logs. Confirmation generation must not begin until the common schema and analysis manifest pass development checks and are committed.
- **Verified:** one-row smoke `177231878` completed in 126.5 s with 289 SPIs, exact config/version provenance and no table/heatmap. Development farms `177242178/191/194/198/203/216/227/234/238` covered the nine homogeneous 40-row cells and all exited 0. Initial freeze job `177247167` correctly rejected a Pearson/Spearman mismatch; Spearman freeze `177252878` completed but was superseded before confirmation. Authoritative Pearson freeze `177255809` exited 0.

## Implemented protocol before confirmation

- `configs/analysis/cross-mt-transfer-260824.yaml` freezes the proposal. Primary: `z_sym`; prespecified sensitivities: `z_dir` and development-total-variance-balanced `[z_sym,z_dir]`.
- Each leave-one-`(M,T)`-cell-out fold fits validity filtering, median imputation, centring, PCA and multinomial logistic regression only on instances `0:9` from the other eight cells; evaluation uses instances `10:29` from the held cell.
- Complementary tests: cross-cell retrieval and a stricter gallery where both `M` and `T` differ; per-class prediction of `M`, `T` and joint cell (size leakage); additive class-versus-cell geometry in one development-fitted shared PCA space. Shared PCA/UMAP figures are illustrative, not inferential.
- Three fixed-dimensional raw-data baselines pool channelwise temporal/marginal summaries (50 features), pairwise-dependence distributions (32), or both (82); their scaling/models use development rows only.
- The implementation fixes PCA50, multinomial logistic regression (`C=1`), 1,000 stratified bootstrap replicates and 1,000 held-label permutations. `scripts/freeze_cross_mt_protocol.py` serializes every development-fitted fold/shared model; `analyze_cross_mt_transfer.py` consumes that bundle without refitting. No confirmation label may tune any choice.

## Frozen development state

- **Verified:** the 1,260-row grid is exact and unique across 14 classes, nine `(M,T)` cells and instances `0:9`. Both source artifacts have Pearson metric, schema `803abbe5…`, 289 SPIs/111 directed, 41,616 symmetric and 32,079 directional features, and complete current-v3 provenance.
- The content-addressed manifest is tracked at `results/cross_mt_transfer_260824/development-manifest.json`, SHA-256 `775d0fc37a817cee4e332ee9a468d0dd5b8050f8e5ea8bacbd10c604f6f7f978`. Its status is `development_frozen_confirmation_unseen`; protocol, model-bundle and baseline-cache hashes were independently verified.
- Development filtering retains 21,280–22,567 symmetric, 12,608–13,900 directional and 33,888–36,467 balanced-augmented features across held-cell folds. PCA50 explains 88.8–90.7%, 89.4–91.2% and 88.6–90.5% respectively.
- The updated six-class current-v3 CML development check uses instances `0:5` for fitting and `6:9` for evaluation: balanced accuracy is 0.981 for `z_sym`, 0.981 for `z_dir`, and 0.968 for balanced augmentation. These are development diagnostics, not confirmation evidence.

## Remaining gate

- **Verified:** the audited manifest was committed in `8fcb086`. Two-dataset smoke `177256967` (instances `10:11`, `M=8,T=500`) exited 0 with exact current-v3 provenance and no tables/heatmaps.
- The nine 280-row homogeneous confirmation farms were `177257654/658/662/663/667/671/688/694/699`; each used one core per selected dataset. The first eight exited 0. Job `177257699` completed 248/280 rows for `M=32,T=2000` before exiting 1 at 02:10:06; its final validator reported the exact 32 missing indices, while completed artifacts remained intact.
- Recovery `177270330` uses 48 one-core workers, skips the 248 complete rows and allows four hours for the 32 slow rows. Replacement Pearson reconstruction `177270338` depends on recovery; frozen analysis `177270341` depends on reconstruction. The obsolete held jobs `177257703/704` were deleted. None of these active/pending jobs is transfer evidence.
