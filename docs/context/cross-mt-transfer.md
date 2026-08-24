# Cross-M,T transfer

Active workstream as of 2026-08-24. Facts are marked **verified**; the implemented protocol remains a prespecification until its development manifest is committed. This study tests quantitative transfer, not mathematical invariance.

## Verified starting state

- The current pyspi-v3 proof bank has 900 rows: ten classes, `M={8,16,32}`, `T={500,1000,2000}`, instances `0:9`; feature artifact `/scratch/ql44/we2614/spi-spi-direction-v2/proof.npz` has 289 SPIs/111 directed, `41,616` symmetric and `32,079` directional features, schema `803abbe5…`, complete pyspi `3.0.0.r7` provenance, and Pearson SPI-edge aggregation.
- Goal 2 prespecifies Spearman SPI-edge aggregation. The same 900 current-v3 MPI banks must therefore be re-aggregated—not recomputed with pyspi—into `/scratch/ql44/we2614/spi-spi-cross-mt-260824/proof-development-spearman.npz`. The original Pearson artifact remains unchanged and cannot be pooled with Spearman features.
- Goal 1 compact JSONs were copied to the ignored local path `results/spi_spi_direction_v2/`; proof/CML-panel/CML-order/Kuramoto SHA-256 are `94cb7eba…`, `c690e6f1…`, `c243b94d…`, `08ccfe81…`. Large feature banks remain on Gadi.
- The existing six-class CML panel cannot be pooled as development evidence: it contains 297 SPIs from `configs/pyspi-v2/benchmarked90_amortized_config.yaml`, lacks config/version provenance, and has a different schema. Its four non-overlapping classes must be recomputed under current p90; the two overlapping classes (`defect-turbulence`, `sti-i`) already exist in the proof bank.
- The four additions are `frozen-chaos`, `brownian-defect`, `fdstc`, and `chaotic-traveling-wave`, yielding 14 classes. Existing dirty notebooks are out of scope and must not be overwritten.

## Generation boundary

- Development additions: `configs/generate/embeddings/cross-mt-cml-development-260824.yaml`, instances `0:9`, 360 rows under gdata. Confirmation: `cross-mt-confirmation-260824.yaml`, instances `10:29`, 2,520 rows under a separate gdata root. Overlapping generator parameters exactly match the original proof config.
- Explicit instance lists preserve the untouched split. Gadi farms are selected by one `(M,T)` cell, use one core/dataset, and emit only `timeseries.npy`, `spi_mpis.npz`, `meta.json` and logs. Confirmation generation must not begin until the common schema and analysis manifest pass development checks and are committed.
- **Verified:** one-row smoke `177231878` completed in 126.5 s with 289 SPIs, exact config/version provenance and no table/heatmap. Development farms `177242178/191/194/198/203/216/227/234/238` covered the nine homogeneous 40-row cells and all exited 0. Initial freeze job `177247167` correctly rejected the Pearson/Spearman artifact mismatch before fitting; confirmation remains unsubmitted.

## Implemented protocol before confirmation

- `configs/analysis/cross-mt-transfer-260824.yaml` freezes the proposal. Primary: `z_sym`; prespecified sensitivities: `z_dir` and development-total-variance-balanced `[z_sym,z_dir]`.
- Each leave-one-`(M,T)`-cell-out fold fits validity filtering, median imputation, centring, PCA and multinomial logistic regression only on instances `0:9` from the other eight cells; evaluation uses instances `10:29` from the held cell.
- Complementary tests: cross-cell retrieval and a stricter gallery where both `M` and `T` differ; per-class prediction of `M`, `T` and joint cell (size leakage); additive class-versus-cell geometry in one development-fitted shared PCA space. Shared PCA/UMAP figures are illustrative, not inferential.
- Three fixed-dimensional raw-data baselines pool channelwise temporal/marginal summaries (50 features), pairwise-dependence distributions (32), or both (82); their scaling/models use development rows only.
- The implementation fixes PCA50, multinomial logistic regression (`C=1`), 1,000 stratified bootstrap replicates and 1,000 held-label permutations. `scripts/freeze_cross_mt_protocol.py` serializes every development-fitted fold/shared model; `analyze_cross_mt_transfer.py` consumes that bundle without refitting. No confirmation label may tune any choice.

## Open until frozen

- Development feature reconstruction must verify the proof schema exactly, after which the content-addressed manifest/model bundle must be copied locally, audited and committed.
- Confirmation runtime/memory caps still require representative timing from these development cells. No confirmation PBS job has been submitted.
