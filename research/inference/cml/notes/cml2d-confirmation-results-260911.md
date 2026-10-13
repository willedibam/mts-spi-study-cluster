# CML2D: independent confirmation

## Conclusion

Positive independent confirmation of **unsupervised order-coordinate inference and tracking across the control sweep**, using sparse dispersed recordings. The pilot's feature mask, imputation, centre, PCA direction, scale and display sign were frozen before the new data; no fitting or model selection used the confirmation outcomes. This is not numerical calibration of Q, discovery of its formula, a new thermodynamic critical-point estimate, or longitudinal fluctuation tracking.

Primary:32 fresh seed clusters at17 controls,544 physical simulations; L256/N65536, M32/T1000. The grid includes nine old controls and eight midpoints. Each simulation uses200k burn,2k maximal observations and a subsequent disjoint million-step physical reference. All observations retain every time step. [Prespecified design and gates](cml2d-confirmation-protocol-260911.md).

## Primary results

| Evaluation set | Records | Spearman q vs future Q | Seed-bootstrap95% interval | Control-mean rho |
|---|---:|---:|---|---:|
| All17 controls | 544 | .883018 | [.867676,.898475] | 1.000 |
| Eight new midpoints only | 256 | .879546 | [.853120,.900824] | 1.000 |
| Nine original controls | 288 | .882634 | [.860110,.902983] | 1.000 |

The q and Q curves both have their steepest sampled derivative over **r=3.86212–3.86306**. This is agreement on the measured finite-run curve, not an independent estimate of the exact infinite-lattice critical boundary. Matched-window association is.901597. Within-control future-Q association is.078464, with interval[-.035996,.169479]: the strong evidence remains across controls, not recovery of future run-to-run fluctuations at fixed r.

The raw mean-absolute-correlation baseline is stronger (.966366), as is the sampled-Q baseline (.894216). Superiority was not a success criterion.

## One frozen coordinate across M,T

All rows below use the same32 seeds and original nine controls. The primary M32/T1000 rows are reused, not recomputed; the other three shapes add864 views.

| M | T | q vs Q rho | 95% interval | Control-mean rho | Paired rho with primary q | Mean q shift | Median absolute q difference |
|---:|---:|---:|---|---:|---:|---:|---:|
| 16 | 500 | .837301 | [.809202,.863967] | .983333 | .867378 | -.124687 | .199256 |
| 16 | 1000 | .836885 | [.807868,.863328] | .983333 | .911371 | -.023116 | .131431 |
| 32 | 500 | .893753 | [.875962,.911751] | 1.000 | .928732 | -.121040 | .145845 |
| 32 | 1000 | .882634 | [.860110,.902983] | 1.000 | 1.000 | 0 | 0 |

Shifts/differences are in the original development-score SD units: no arm-wise normalisation or sign fitting. The order trend transfers, but the coordinates are not exactly invariant. Longer T does not monotonically improve q–Q association here; the M16/T1000 view does have closer paired agreement with the primary coordinate than M16/T500. These are observed comparisons, not separately tested claims of superiority. M16/T500's steepest interval moves to 3.864–3.866, one original-grid interval above Q; the other shapes agree with Q over3.86212–3.864. Exact localisation is not invariant to observation size.

Both primary and secondary quality gates pass: **1408/1408 records retained, zero selected-feature missingness**,32 records in every planned cell versus the required minimum24. All289 p90 SPIs were attempted; the frozen pilot mask retains19444 pairs spanning253 SPIs. Individual estimator failures outside the selected mask are not being described as zero computational warnings/errors.

## Integrity, reference precision and presentation

- Independent postprocessing verifies1408 input-member identities,864 exact nested prefixes/shared targets, unchanged model arrays and execution core, and all q values (maximum reconstruction error2.22e-16). Eight predetermined MPI archives independently reproduce their selected z entries. Local checks verify both raw-archive hashes, all864 prefixes, both copied model bundles and all six overall/within-control bootstrap intervals to1e-12.
- The physical mean-Q curve decreases across all17 controls, but near-critical temporal variability persists: at r3.86306, maximum half-mean difference .07912 and maximum descriptive block SE.02313. Targets/horizons were not changed after seeing results; use the stated finite-run interpretation.
- The comparison notebook keeps pilot and confirmation sections distinct and uses confirmation in its summary table. M sets colour; T sets opacity.
- Three unchanged pilot MTS examples have exact aligned100-step global-mean traces above them and their respective full-lattice heatmaps below them in one3x3 figure. White circles mark the32 recorded sites; all lattices share the raw[0,1] colour scale and use saved final fields, explicitly later than the MTS. No simulation or numerical result changed for this layout update. PNG/SVG and hashes/timing: `notebooks/inference/figures/cml2d-period-doubling/`.

## Provenance and durable outputs

Local compact results/raw inputs: `data/order_parameter/cml2d_confirmation_260911/`. Full corpora, MPIs, features, reports and `integrity-audit.json`: `/g/data/ql44/we2614/mts-spi-data/order_parameter/cml2d_confirmation_260911/`. Physics archive (tar comparison passed; originals preserved): `/g/data/ql44/we2614/mts-spi-archives/cml2d_confirmation_260911_physics.tar`, SHA256 `51bb6c94934290426ce9e506557d8af2c606842a8159288cf803aa1da6fae707`. Primary/sensitivity raw hashes are in their manifests and the [execution ledger](cml2d-results-260911.md), which also preserves both launcher failures and recovery details. No failed case was silently discarded.

Runtime/sourcee9cf5cc; auditbe3ca29; archival794a354 on the isolated `codex/cml2d-period-doubling` branch. Main and unrelated edits were not pushed. Final p90 jobs178774140–143; reports178774144/145; integrity178776139; physics archive178776864. No recurring automation or new scientific arm. All stages completed successfully; total confirmation charge including launcher repairs and archival1179.91SU. Notebook46 cells executes without errors and all four new/updated visual panels were inspected. The archive/integrity jobs passed.
