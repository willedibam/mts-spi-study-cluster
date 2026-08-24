# SPI–SPI representation evaluation

Authoritative handoff as of 2026-08-24. Repository: `~/Code/mts-spi-study-cluster`, branch `refactor-lagged-warping`; feature/analysis implementation baseline `fbbb9036de4ae72f9f4d50e08692e27982a13cad`, with handoff/cluster-operation baseline `37b1c3f56cc5173f35b4df2422e324b4e67981d5`. Facts below were checked against code, artifacts or PBS state; proposals are labelled.

## Verified legacy behaviour and defects

- `src/process_features.py` still defaults to `legacy_symmetrized_v1` so old commands remain reproducible. For each MPI it uses the strict upper triangle of `(A+A.T)/2`; `build_spi_spi_features` correlates these edge vectors and flattens SPI-pair upper triangles. MPI diagonals and SPI self-pairs are excluded.
- Consequently, direction is discarded by default and a purely antisymmetric MPI becomes zero. Legacy `--split-directed` retains fixed-label upper/lower vectors, but is not invariant to arbitrary channel relabelling.
- Legacy `nonfinite_policy="zero"` conflates undefined correlation with zero association and can break the correlation-Gram interpretation. Its corpus-wide `std >= 1e-8` filter makes the schema depend on all analysed rows, including test rows.
- Legacy cache identity omitted variance threshold, contract version, pyspi configuration/version/normalization, source hashes and repository revision. Old caches are therefore evidence for their archived calculation, not automatically interchangeable with current computations.
- Directed flags include pyspi `directed`, `asymmetric` and `antisymmetric` SPIs. PySPI v3 also transposed six directed spectral SPIs, so estimator version and source/target convention are material provenance.

## Frozen v2 feature contract — implemented and tested

- Core commit `77ba664` introduced `src/spi_spi_contract.py`; final analysis alignment is `fbbb903`. New analyses must explicitly request `direction_preserving_v2`.
- For SPI MPI `A_a`, `z_sym[a,b] = corr(u_a,u_b)`, where `u_a` is the strict upper triangle of `(A_a+A_a.T)/2`. This exactly reproduces the legacy NaN-valued subspace for Pearson/Spearman and valid histogram-MI cases.
- Let `v_a=(A_a[i,j]: i != j)` in C row-major order. `z_dir` contains: parallel `corr(v_a,v_b)` for every SPI pair involving at least one directed SPI; reverse `corr(v_a,v_b^T)` for each directed–directed pair; and self-reciprocity `corr(v_a,v_a^T)` for every directed SPI. Undirected–undirected duplicates are omitted.
- With `K` SPIs and `D` directed SPIs, `dim(z_sym)=K(K-1)/2` and `dim(z_dir)=DK`. Both are invariant to a common channel permutation. Transposing one directed MPI swaps its directed–directed parallel/reverse relations; pure antisymmetry remains visible through reciprocity `-1`.
- The canonical artifact stores `X_sym` and `X_dir` once. `z_aug=[z_sym,z_dir]`; block balancing is an optional downstream fit, not part of feature identity. It scales each block to unit total development variance without feature whitening.
- Undefined values remain NaN with validity masks and per-SPI reasons. V2 performs no corpus variance filtering. Downstream imputation, validity/variance selection, centring and any balancing are fit on development rows only (`src/spi_spi_analysis.py`).
- V2 schema records block, relation and SPI names plus SHA-256. Cache identity includes contract/metric/options, ordered SPI/direction hashes, input content hashes, pyspi config/version/normalization, builder hashes and repository state; loading revalidates identity, schema, dimensions and masks. Older metadata is explicitly provenance-incomplete.
- Tests in `tests/test_spi_spi_contract.py`, `tests/test_feature_artifact_contract.py` and `tests/test_spi_spi_analysis.py` cover exact symmetric reproduction, permutation/transpose laws, antisymmetry, undirected equivalence, diagonal exclusion, NaN handling, schema/cache provenance, development-only transforms and block balancing. `python -m pytest -q` passed 58/58 on 2026-08-24.
- Gadi smoke `177203315` exited 0: two rows, 164 SPIs/29 directed, `13,366` symmetric and `4,756` directional features, zero invalid values and complete provenance. Directional motifs are correctness probes only, not scientific datasets.

## Verified existing evidence

- Archived proof matrices: `features/data-embeddings-multi_p90_260701_pearson.npz` has 900 rows/40,184 retained legacy features; `features/data-embeddings-cml-embedding_pearson.npz` has 540/40,174. Both arose after corpus filtering and are not a single frozen schema. `notebooks/embeddings/proof_p90_260712.ipynb` shows exploratory PCA/UMAP class geometry across `M={8,16,32}`, `T={500,1000,2000}`, but no held-out cross-size invariance result; its AUC orientation and null require redesign before inferential use.
- CML order study (`notebooks/embeddings/cml-order-parameter-inference.ipynb`, `docs/context/cml-order-parameter-inference.md`): epsilon `0.3`, alpha `1.60:0.01:2.00`, `M=20,T=1000`, 20 seeds. Legacy PCA10→Isomap1 latent `q` tracked selected cropped spectral observable `Q_sel` with held-confirmation `|rho_S|=.911`, within-alpha `.611`; temporal entropy/recurrence were about as strong. `Q_sel` is operational, not a canonical order parameter, so this supports unsupervised dynamical-coordinate recovery, not superiority or a canonical CML claim.
- Kuramoto (`notebooks/embeddings/kuramoto-order-parameter-confirmation.ipynb`, `docs/context/order-parameter-benchmarks.md`): hidden canonical full-system mean coherence `R`; `N=256`, observed `M=20,T=1000`. The frozen 164 non-phase-SPI symmetric PC1 gave Gaussian/logistic overall `|rho_S|=.924/.933` and within-control `.523/.618`. Mean absolute input correlation was stronger, so this is capability rather than dominance.
- V2 Kuramoto reconstruction jobs `177203650/651/652` and CML-panel reconstruction `177203721` exited 0. Sensitivity job `177204263` exactly reproduced frozen symmetric Kuramoto counts/variance and correlations. Direction-only was weaker (`.675/.803` overall; `.403/.508` within-control); block-balanced augmentation was also weaker (`.862/.907`; `.480/.576`). Direction is informative but supplied no incremental benefit here.
- Existing six-class CML-panel sensitivity `177204897` exited 0: held-instance balanced accuracy was symmetric `1.000`, directional `.995`, augmented-balanced `.991`. This is an easy held-instance class task, not cross-`M,T` transfer; it likewise gives no direction-improvement evidence.
- Current-pyspi CML-order regeneration/reconstruction/analysis `177206563/177207598/177207603` exited 0 with 820 rows, 289 SPIs/111 directed, `41,616` symmetric and `32,079` directional features. Held-confirmation Isomap--`Q_sel` was symmetric `.945` (within-alpha `.707`), directional `.940` (`.679`) and augmented-balanced `.942` (`.694`); temporal entropy was `.937`. Direction remains independently informative but gives no gain. Feature/result SHA-256 are `621ca881…`/`c243b94d…`.
- Gadi compact results: `results/spi_spi_direction_v2/{kuramoto,cml_embedding}.json`; reconstructed features: `/scratch/ql44/we2614/spi-spi-direction-v2/*.npz`. Kuramoto generation provenance includes commit `33a4284b11044c89cacb71ce24f09d973313ce23` and compact contracts under `data/order_parameter/kuramoto_final_confirmation_contract/`.

## Goal separation and current state

1. **Goal 1 — representation correctness, first.** Preserve exact `z_sym`; add invariant direction without duplication; freeze invalid/schema/cache/provenance semantics; test mathematical laws. Core code, tests, commit and Gadi smoke are complete. A per-family audit of directed SPI source/target semantics remains before interpreting individual directional features scientifically.
2. **Goal 2 — quantitative representation evaluation, only after Goal 1.** Compare frozen `z_sym`, `z_dir` and development-fitted augmented variants on held-out tasks. Kuramoto, the existing CML panel and regenerated CML-order sensitivity are complete and do not favour augmentation. Exact proof/CML-order MPI archives were absent, so current-pyspi runs are sensitivities, not numerical reconstruction of the old archives.

PBS snapshot: proof main/bridge farms `177205470/177205481` produced 710/720 and 180/180 rows (890/900 total); the main farm exited 1 only because indices `441:450` (`kuramoto_omega-slow,M=32,T=2000`) exceeded its 90-minute per-task cap. Exact-range recovery `177218375` is running with a 150-minute cap; feature/analysis `177218376/177218377` are dependency-held. A duplicate recovery chain caused by an SSH-client timeout was cancelled before output completion. No 30-instance extension was submitted. All active inputs are gdata-backed MPI/time-series artifacts; independently removed calc tables, heatmaps, TUH and sleep-onset paths are not dependencies. Legacy proof-v2 banks are archived at `/g/data/ql44/we2614/mts-spi-archives/legacy-proof-v2/`.

## Deferred proposals — not established results

- Cross-`M,T` transfer should test the claimed common-schema advantage only after Goal 2 sensitivities: train at selected observation budgets and test held-out budgets against dimension-compatible summary, kernel and learned baselines. Current UMAP co-location is not such a test.
- Nested-master views are a paired robustness design, not a replacement for independent realizations: derive nested spatial subsets/crops and time prefixes from many independent maximal realizations. Do not use them where changing `M` changes the physical system, topology, boundary conditions or sensor semantics. Fixed-control ensembles are not temporal control-parameter evolution.
- Structured learning on the SPI correlation/graph object, correlation-matrix geometry, spectral summaries and self-supervised agreement across nested views is future work. No evidence yet shows it beats the frozen vector/PCA baseline.

## Unresolved questions

- What source→target convention and estimator-validity conditions hold for every directed SPI family under the pinned pyspi version?
- Should reciprocity remain inside primary `z_dir` or be a separately reported sensitivity? How should blocks or SPI families be regularized without post-hoc target tuning?
- Do regenerated proof/CML-order results confirm any incremental directional value? If not, direction should be retained for correctness, not marketed as a performance gain.
- Which exact common SPI catalogue survives estimator failures without inducing corpus-specific schemas, and which fitted transform should be shared across proof panels?
- Does direction remain stable at small `M`, where ordered dyads are dependent and MPI vectors can be low-rank/singular?
- Which cross-size task, baselines and real variable-channel dataset would support a genuine advantage claim? Does structured matrix/graph learning add anything beyond frozen vector, PCA and sparse baselines?
