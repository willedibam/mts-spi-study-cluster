# CML2D independent confirmation

Frozen before generating the new outcomes. User approved independent replication
and finer controls; available compute is not a scientific constraint. This is
not a new generator, model search, longitudinal benchmark or local-patch rescue.

## Fixed design

- The same synchronous logistic square CML, g=.2, L256/N65536; 200k burn,
  first2k observation steps, then a disjoint one-million-step Q reference.
  These are finite-run quantities, not an equilibrium critical-point estimate.
- 32 new seeds260911101–132; all are confirmation, none train the coordinate.
  The nine original r controls plus their eight arithmetic midpoints make17
  controls/544 independent seed-control masters. Controls within a seed are
  paired; inference resamples the32 seed clusters, not individual M,T views.
- Primary: dispersed M32/T1000 on all544 masters. Secondary: M16/T500,
  M16/T1000 and M32/T500 on the nine original controls, 864 additional views.
  The same primary M32/T1000 old-grid rows complete the paired2x2 comparison.
  No new M64, T100, contiguous, alternate-L or alternate-model extraction.
- Dynamic and sensor RNG streams and nesting unchanged. No site/window search.
  More seeds improve uncertainty/replication; the interleaved controls improve
  sampling resolution along r. Neither is a larger-L critical extrapolation.

## Frozen transform and eligibility

Use the completed pilot `primary-analysis/model.npz` without refitting any
feature mask, imputation, centring, PC direction, score scale or display sign.
Model SHA256: fddda6abf3b1ddbd30f9aff84158dee5436716f1eb1c01704516f753b22995b2.
Geometry SHA256: bc1777579703c232a060f536aa3969791fdb1f07ffcde1a77a9a1e3251c02357.
Pilot summary SHA256: 78e1c9ebf83f45c2007839a562b76d2335630d22333a87583f680ac7484ad9d8.
All289 p90 SPIs are attempted; unified_ordered_v3 Pearson construction is unchanged.

Per-row selected-feature missingness<=.05, as in the pilot. Report primary and
secondary arms separately; each must exclude<=10% of its rows and retain at
least24 of32 records per control/M/T cell. The stronger coverage requirement
is specified prospectively for this larger replication bank and is sealed by
the scorer before physical targets enter its score table. The pilot's original
two-per-cell gate remains unchanged in its historical outputs. No outcome-
dependent threshold relaxation or model rescue.

## Endpoints and interpretation

Primary: frozen-q vs Q_reference Spearman, seed-cluster bootstrap95% interval
(2000 resamples); control-mean association; whole-grid and midpoint-only results;
physical and q curves with seed uncertainty; steepest sampled-slope interval.
Report same-window and within-control diagnostics as secondary, not mandatory
success endpoints. Include mean-absolute-correlation and sampled-Q baselines;
superiority is not required. No binary correlation threshold is imposed; use
effect size and uncertainty rather than redefining success after seeing results.

Secondary: per-M,T associations and paired q differences/agreement relative to
the primary in the original frozen units. Do not recenter or rescale by arm.

Audit Q_blocks and first/second-half differences at every r. The common horizon
is retained to test the original finite-run estimand. If near-critical drift
persists, report it; do not replace the targets post hoc to optimise correlation.
A subsequent physics-only longer-horizon audit is justified only to resolve a
material reference instability, and is separately labelled.

Feature attribution is a post-hoc description of the existing pilot's fixed
linear map, not feature selection for this confirmation. Three headline snapshots
use the first pilot held seed26091115 at r3.84/3.86212/3.89 and its first100
consecutive steps (fullT1000 exports also retained); no appearance-based selection.

## Execution

Immutable isolated source branch/worktree; preserve unrelated local changes.
Physics masters on Scratch; confirmation corpora/results directly on project
gdata to avoid Scratch inode pressure. Parallelise datasets by homogeneous M,T.
Stage dependencies are finite PBS jobs, not a recurring reminder. Verify model,
input, execution and source identities; report failures without silent replacement.
No new recurring automation. Record jobs/status in the result ledger and context.
