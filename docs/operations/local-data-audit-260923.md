# Local data review, 23 September 2026

Scope: local `features/` (~1.0 GiB), `data/` (~20 GiB), and `analysis/` (~792 KiB). No data banks were deleted. Checked remaining notebook source cells, code references, NPZ metadata, matching-size NPZ SHA-256 hashes and the partial/full July feature matrices. Absence of a literal reference is not proof that an artifact is unused; paths can be assembled dynamically.

## Retained notebook dependencies

`notebooks/embeddings/proof_p90_260712.ipynb` directly loads `features/data-embeddings-cml-embedding_pearson.npz` (540 rows, 297 SPIs) and `features/data-embeddings-multi_p90_260701_pearson.npz` (900 rows, 297 SPIs). The CML source paths name `data/embeddings/cml-embedding`, corresponding to Gadi's `mts-spi-study/legacy-non-p90/data/cml-embedding`. This establishes recorded source lineage, not a new raw-byte comparison with Gadi. The cached features suffice for the notebook; raw/MPIs are needed to rebuild them. Neither bank is the current 289-SPI p90 bank despite the July filename. Keep both local matrices. The Gadi legacy-CML deletion recommendation is conditional on no longer needing historical feature recomputation.

`gp_regression_cml_260716.ipynb` still uses `features/data-embeddings-cml_param_sweep_260508_pearson.npz` (~102 MB). The Gadi raw bank was deliberately deleted earlier; retain this local feature bank. The current proof data in `data/proof_p90_260824` are still used by the September 21 baseline scripts even though the user deleted the August proof notebook.

## Candidates and limits

| Artifact | Finding / recommendation |
| --- | --- |
| `features/data-embeddings-proof_benchmarked90_260701_pearson.npz` (~82 MB) | Strong redundancy: all 644 rows have matching class/M/T/instance IDs and exactly equal features (including NaNs) in the retained 900-row July multi bank; feature-pair order is identical. No remaining literal consumer found. Candidate to delete, retained pending a data-removal decision. |
| `features/data-embeddings-proof_benchmarked{80,95}_260603_pearson.npz` (~220 MB total) | Older catalogue comparisons; no remaining literal notebook consumer found. Distinct catalogues, not exact duplicates. Retain unless those comparisons are retired. |
| `features/data-embeddings-proof_benchmarked90_260603_pearson.npz` (~115 MB) | Historical presentation/storyboard references remain; keep unless those uses are retired. |
| `features/data-embeddings-proof_benchmarked90_260603_pearson_final_S.txt.npz` (~0.6 MB) | Separate 20-SPI subset, not interchangeable with the full bank; purpose/continued use unclear. |
| `features/sleep_onset_m{38,83}_pearson.npz` (~40 MB total) | Separate observation sizes; no evidence of duplication. Continued study value needs the user's decision. |
| `data/neurotycho_spi_source_scout_260910` and `...-metadata-diagnostic` (~38 MiB each) | Older float32 scout/diagnostic iterations. Corrected f64 banks exist, but keep diagnostic provenance; no new byte-equivalence claim. |
| `data/260726_r1_nonlinear` + `260726_r1_demo` (~913 MiB) | Local July EEML copies; full cohorts have verified Gadi archives. Candidates if offline use is no longer required; archive identity was not freshly compared to these local partial copies. |
| `data/order_parameter/finite_regime_260915/tasep` (~103 MiB) | Remote TASEP was retired, but local benchmark/pair-sampling results were explicitly preserved. Do not infer permission to remove these local inputs from the remote cleanup. |
| `data/dtw_euclidean` (~1.0 GiB) and `data/r_rho_mi` (~1.1 GiB) | Remaining case/presentation notebooks use these families. Age alone is not a deletion reason. |
| `analysis/feature_cache/old_260318` (20 tracked NPZs, <1 MB) | March-generated caches. No direct reference to the archived subfolder found; two retained case notebooks reference the old parent cache paths instead. Those paths are stale. Verify/regenerate before redirecting them to old caches. Negligible space benefit from deletion. |

The NPZ duplicate scan found eight identical-content groups above 100 kB, with about 49.8 MB of potentially duplicated payload. The largest is the 31.8 MB CML2D L6 feature copy under `analysis/` and `transfer-analysis/`; another is a 14.0 MB TASEP N64 feature copy. Most remaining duplicates are deliberately reused frozen models. Two differently named Zenodo records also share an identical MPI file; distinct dataset identities must not be collapsed on MPI equality alone. No top-level `features/` file was byte-identical to another scanned bank. Raw `.npy`/`.mat` files were not exhaustively deduplicated.

## Minimal repository hygiene

Preserve the user's notebook removals and moves. Ignore generated `tmp/` review images and new `analysis/feature_cache/` outputs; existing tracked historical cache files remain untouched. Keep a short root README pointing to context, cluster setup and the local data boundary. Update documentation links to moved notebooks. Correct the blank row in the migration inventory and reject blank paths in its optional checker.

Do not reorganise local `data/` wholesale: retained notebooks and provenance use its current paths. Use one natural location for each new bank, and explicit references to shared inputs. Historical notebook generators remain useful even when their generated notebooks were deleted; do not remove them by filename association. `scripts/audit_spi_baseline_exploration.py` still contains a historical unchanged-notebook assertion referring to the deleted August notebook and older notebook versions; it needs an explicit new preservation baseline before being used for a fresh whole-study audit. Do not silently weaken that provenance check.
