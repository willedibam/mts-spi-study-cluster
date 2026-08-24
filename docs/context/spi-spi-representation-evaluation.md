# SPI–SPI representation evaluation

Authoritative handoff as of 2026-08-24. Repository: `~/Code/mts-spi-study-cluster`, branch `refactor-lagged-warping`, inspected HEAD `8e74dac4c93cd5fc1a31f86ab9e11402156901dc`. This separates verified facts from accepted design decisions and untested proposals.

## Verified: current feature construction

- `src/process_features.py::_edge_vectors` symmetrizes every undirected MPI and, by default, every directed MPI as `(A + A.T)/2`, then retains only the strict upper triangle. `--split-directed` is false by default.
- With `split_directed=True`, a directed MPI becomes fixed-label vectors `A[upper]` and `A.T[upper]`. This retains both orientations for a fixed channel order but is not invariant to arbitrary channel relabelling.
- `build_spi_spi_features` correlates MPI edge vectors (Pearson or Spearman), then flattens the strict upper triangle over SPI pairs. MPI diagonals and SPI self-pairs are excluded.
- The Pearson implementation correctly centres and normalizes vectors; Spearman uses average ranks for ties. Existing tests check fixed upper/lower ordering, not channel-permutation invariance.
- Default `nonfinite_policy="zero"` silently maps undefined SPI–SPI correlations to zero. This changes their meaning and can destroy any correlation-Gram interpretation/positive-semidefiniteness.
- The CLI applies corpus-wide `std >= 1e-8` filtering after constructing all rows. The resulting schema depends on the analysed corpus and uses all observations rather than a training-only fit.
- Cache identity includes data path, limit/subset, split flag and metric, but not the variance threshold, feature-contract version, pyspi configuration/computation version, normalization, source-data hash or repository commit. Existing files may therefore be stale-compatible rather than semantically identical.
- Loading validates SPI order and directed flags across datasets, but not the above computation provenance. `src/compute.py` treats `directed`, `asymmetric` and `antisymmetric` SPIs as directed; default symmetrization makes a purely antisymmetric MPI zero.
- PySPI v3 changed numerical behaviour and transposed six directed spectral SPIs (`src/run_experiments.py`), so source/target convention and computation version are material provenance.

## Verified: archived schemas and evidence

- `features/data-embeddings-multi_p90_260701_pearson.npz`: 900 rows, 40,184 retained features; 297 source SPIs, 91 flagged directed, legacy unsplit construction.
- `features/data-embeddings-cml-embedding_pearson.npz`: 540 rows, 40,174 features; the same 297/91 catalogue and legacy construction. Its pair set is a subset of the multi-system set; the latter has ten additional covariance–covariance pairs. Both retained schemas reference 284 unique SPI names. Corpus filtering/failures explain at least part of the reduction, but the exact decomposition was not established.
- `features/data-embeddings-cml_param_sweep_260508_pearson.npz`: 820 rows, 39,567 features; 295 source SPIs, 89 directed, legacy unsplit construction.
- Quadratic CML evidence (`notebooks/embeddings/cml-order-parameter-inference.ipynb`; `docs/context/cml-order-parameter-inference.md`): at epsilon 0.3, alpha 1.60–2.00, `M=20,T=1000`, 20 seeds, PCA10 followed by Isomap1 produced latent `q`. Against the selected cropped spectral observable `Q_sel` (operational, not canonical), held internal confirmation gave `|rho_S(q,Q_sel)|=0.911` and within-alpha `0.611`; simple temporal entropy/recurrence were about as strong. This supports unsupervised dynamical/order-coordinate recovery, not superiority or a canonical CML order parameter.
- Kuramoto evidence (`notebooks/embeddings/kuramoto-order-parameter-confirmation.ipynb`; `docs/context/order-parameter-benchmarks.md`): canonical full-system mean phase coherence was hidden from the representation. For 1,536 `M=20,T=1000,N=256` observations, 164 stable non-phase SPIs gave 13,366 legacy features, 12,737 after SD filtering, and PC1 explained 36.1%. Overall `rho_S(PC1,R)` was -0.924 (Gaussian) and -0.933 (logistic); within-control correlations were -0.523/-0.618. A diffusion coordinate agreed with PC1 (`|rho_S|=0.981`). Simple mean absolute correlation was stronger, so this is capability evidence, not benchmark dominance.
- Kuramoto generation commit recorded by the terminal contract is `33a4284b11044c89cacb71ce24f09d973313ce23`; compact outputs are under `data/order_parameter/kuramoto_final_confirmation_contract/`. Recorded hashes include representation model `325bcf1e...`, readout `8729092b...`, and pyspi config `bc4bafa1...`. No reliable PBS scheduler job identifiers were found in the compact contracts.
- Proof notebook (`notebooks/embeddings/proof_p90_260712.ipynb`): independent rows span `M={8,16,32}` and `T={500,1000,2000}`. PCA50 retained 88.6%/93.3% before separately fitted UMAPs; one graph warned of disconnection. The plots are useful exploratory class geometry, but there is no held-out multiclass or cross-size invariance result. The univariate AUC section selects orientation using test labels and its permutation null does not match the reported minimum-cell statistic.
- These successful results used the symmetrized legacy subspace. Directed information was not necessary for them; adding it is an extension and must not retroactively redefine their representation.

## Implemented v2 contract (local worktree, not yet frozen)

- `src/spi_spi_contract.py` defines `direction_preserving_v2`; `src/process_features.py` retains explicit `legacy_symmetrized_v1`. `z_sym[a,b]=corr(upper_offdiag((A_a+A_a.T)/2), ...)` is exactly equal to the legacy NaN-valued calculation for Pearson, Spearman and valid MI cases.
- Ordered vectors are `v_a=(A_a[i,j]:i!=j)` in C row-major order. `z_dir` contains parallel correlations for every pair involving a directed SPI, reverse correlations for directed--directed pairs, and self-reciprocity for each directed SPI. Undirected--undirected duplicates are omitted. For `K` SPIs of which `D` are directed, `dim(z_sym)=K(K-1)/2` and `dim(z_dir)=DK`.
- A common channel permutation leaves both blocks unchanged. Transposing one directed MPI swaps its parallel/reverse relation with another directed MPI; transposing all directed MPIs leaves the representation unchanged. Pure antisymmetry is invalid in `z_sym` but retained in `z_dir` with reciprocity `-1`.
- Diagonals are excluded; undefined correlations remain NaN with row validity masks and per-SPI reasons. V2 stores `X_sym` and `X_dir` once; `z_aug` is their concatenation. It does not apply corpus variance filtering.
- `src/spi_spi_analysis.py` fits imputation, validity/variance selection and centring on development rows only. Optional block balancing gives each block unit total development variance without feature-wise whitening. Deterministic PC1 sign uses its largest absolute loading, never a target.
- Tests: `tests/test_spi_spi_contract.py`, `tests/test_feature_artifact_contract.py`, and `tests/test_spi_spi_analysis.py`; full local suite passed 56 tests on 2026-08-24. Directional motifs remain correctness probes only.

## Implemented validity, schema and provenance decisions

- Canonical v2 construction retains NaN; the historical zero policy exists only inside explicit legacy reproduction. Artifacts include a complete named schema (`block`, `relation`, `spi_a`, `spi_b`), its SHA-256, block masks/reasons, and no corpus filter. Legacy archives remain untouched.
- When pooling current proof archives, the immediate exact intersection is 40,174 features. The preferred final route is to rebuild from intact MPIs with one catalogue/contract and no corpus-specific feature deletion.
- Cache identity now includes contract, metric, invalid policy, direction/reciprocity mode, ordered SPI/direction hashes, content hashes of every metadata/MPI input, pyspi configuration/version/normalization, builder-source hashes and repository state. Loading revalidates identity, schema hash, dimensions and masks. Older metadata is marked provenance-incomplete rather than silently upgraded. Downstream fitted-model hashes remain to be added to result contracts.

## Ordered workstreams

1. **Goal 1 — representation correctness:** core implementation and local tests now pass; production freezing still requires review of directed-family semantics and a clean commit/Gadi smoke. Small motifs remain diagnostics, not paper datasets.
2. **Goal 2 — quantitative representation evaluation:** `scripts/analyze_direction_preserving_sensitivity.py` provides exploratory sym/dir/block-balanced comparisons for proof, CML and Kuramoto. Broader held-cell cross-`(M,T)` transfer, retrieval, baselines and real-data evaluation remain deferred until these sensitivities justify them.

## Verified reconstruction inventory (Gadi, 2026-08-24)

- Intact MPIs: proof `data/embeddings/proof_benchmarked90_260603` (900, but `M={5,10,20}`), CML regime corpus `data/embeddings/cml-embedding` (540), and Kuramoto terminal bank `data/order_parameter/kuramoto_final_confirmation` (1,536); its two development banks are also present.
- Missing locally and on Scratch/gdata: exact proof-notebook `data/embeddings/multi_p90_260701` (`M={8,16,32}`) and CML-order `data/embeddings/cml_param_sweep_260508`. Their configs remain. Recomputing them with current pyspi v3 is a new sensitivity, not reconstruction of the old 297/295-SPI numerical archive.
- `jobs/gadi/run_feature_reconstruction.pbs` performs 48-way dataset reconstruction from intact MPIs; `jobs/gadi/run_directional_sensitivity.pbs` runs the numeric comparisons. Pyspi generation, if needed, uses the existing dataset-level `nci-parallel` farm and no heatmaps/CSV.

## Deferred proposals

- Cross-`M,T` transfer is not part of Goal 1 and is not required to retain the current proof-of-concept. Distinguish fixed output schema, channel-permutation invariance, and robustness to observation budget.
- Nested-master views are a paired diagnostic, not a replacement for independent realizations: derive contiguous spatial crops/fixed nested random oscillator subsets and temporal prefixes from each large master while retaining many independent masters. Avoid this when changing `M` changes the physical system, topology, boundary conditions or sensor semantics.
- Existing independent fixed-control CML/Kuramoto realizations correctly sample `p(X|lambda)`; they do not represent temporal parameter evolution. A ramped-control/hysteresis study is separate.
- Structured learning on the SPI correlation/graph object, matrix geometry, spectral summaries, and self-supervised agreement between nested views remain hypotheses. Establish validity and cross-size value with vector baselines first.

## Unresolved questions

- What is the verified source-to-target convention and estimator validity for every directed SPI family?
- Should self-reciprocity `r_a` be primary or a sensitivity block, and how should sym/directional blocks or SPI families be balanced?
- How much do directional additions improve held-out tasks, and how stable are they at small `M` given dependent dyads and singular Gram matrices?
- Are all required archived MPI matrices intact locally/on Gadi, and can Goal 2 avoid rerunning pyspi?
- What common catalogue survives estimator failures without inducing corpus-specific schemas? Should proof panels share one fitted transform or be explicitly presented as separate local embeddings?
- Which cross-size baselines, transfer gates and real-world variable-channel dataset would make a genuine advantage claim rather than only a visualization?
- Does structured matrix/graph learning add value over frozen vector, PCA and sparse baselines? No evidence yet establishes that it does.
