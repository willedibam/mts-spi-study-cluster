# Faster-dynamics transfer: verified findings

The prospective transfer test is favourable for SPI–SPI, including PCA/ridge.
Source-trained models retain useful co-organization prediction when the carrier,
phase-diffusion and envelope timescales change. Learned SPI pooling matches their
high accuracy with 20–40 labels; explicit correlations remain stronger with 10 labels
under the tested training budget. This is a scoped within-generator result.

The [protocol](oscillatory-pooling-transfer.md) was frozen before new-regime
recordings and pooling fits. All models use the original regime's three disjoint
source cohorts,10/20/40label budgets, preprocessing, selected hyperparameters and
epochs. No target labels, target calibration, pretraining or augmentation enter
fitting. The faster regime has200 independent test masters/400 paired observation
views, separate from both source data and the32master raw feasibility scout.
N32 and the binary latent group organization remain fixed; this is not unseen
physical-system size, sensor mixing, topology or generator-family transfer.

## Primary: changed dynamics, original observation

Balanced accuracy, M16/T1000, mean over the three source cohorts:

| Representation/readout | 10 labels | 20 labels | 40 labels |
|---|---:|---:|---:|
| z + PCA/ridge | 98.5% | 100.0% | 100.0% |
| z + PLS | 100.0% | 100.0% | 100.0% |
| Learned SPI pooling | 63.2% | 100.0% | 100.0% |
| Rich individual-SPI summaries | 78.0% | 67.8% | 80.0% |
| Normalized individual-SPI shapes | 67.7% | 74.7% | 85.3% |
| Rich summaries + z | 98.0% | 99.5% | 100.0% |
| Direct phase/envelope agreement | 99.8% | 99.8% | 100.0% |
| Raw marginal/covariance/cumulant/window summaries | 60.7% | 90.7% | 78.2% |

Both original raw neural encoders remain near chance; full metrics include them.
The stronger learned comparator uses SPI inputs, so these results do not establish
superiority over all neural learning or pretrained raw-data models.

## Secondary: dynamics plus sensor/duration reduction

Balanced accuracy, M8/T500:

| Representation/readout | 10 labels | 20 labels | 40 labels |
|---|---:|---:|---:|
| z + PCA/ridge | 89.7% | 98.8% | 99.0% |
| z + PLS | 93.8% | 99.0% | 99.2% |
| Learned SPI pooling | 61.7% | 97.8% | 99.5% |
| Rich individual-SPI summaries | 71.3% | 77.0% | 79.3% |
| Normalized individual-SPI shapes | 78.5% | 83.2% | 86.7% |
| Rich summaries + z | 89.5% | 97.7% | 99.0% |
| Direct phase/envelope agreement | 98.5% | 98.7% | 99.8% |
| Raw marginal/covariance/cumulant/window summaries | 58.0% | 78.3% | 71.7% |

At 40 labels, reduced zPLS−richm is+19.83 percentage points, conditional pointwise95%CI
[16.0,23.5]; richm+z−richm +19.67[16.0,23.33]. PCA−learned pooling at 10 labels is
+28.0[25.0,31.0], but at 40 labels -.50[-1.83,.50]. These intervals resample independent
test masters within class, conditional on the fitted models; no multiplicity
adjustment or population-wide training-cohort guarantee. They do not establish
formal equivalence or a precise sample-efficiency factor.

![Matched source and transfer learning curves](../results/oscillatory_coorganization_transfer_260909/learning-curves.png)

## What this does and does not establish

The fixed transfer succeeded: the representation does not require the exact
original timescale distribution for this target. This is not evidence that the
shift is uniformly more difficult. The direct specialist also improves, and
faster dynamics may yield more effectively independent observations per record.
That is a plausible explanation, not measured causal attribution. Do not describe
the result as robustness under increasingly severe stress.

A material distinction is ranking versus decision-score transfer. At 40 labels and
M16/T1000, richm AUROC=.99977 yet BA=.800; normalized-shape AUROC=.9968 yet BA=.8533.
For richm, all aligned recordings are classified correctly, while27%,28%,65% of
crossed recordings in the three cohort models exceed the fixed.5 threshold.
Their mean scores are.378/.420/.569. PCA classifies both classes correctly there.
Thus much of this primary gap concerns stable source-trained decision scores,
not complete absence of discriminative information in marginal summaries.
Target-domain calibration might reduce it; that was neither allowed nor tested.
At reduced observation, richm AUROC=.9509 versus zPCA.99983/zPLS.99990, so a ranking
gap remains as well. Brier scores at 40 labels are.1392/.0241/.0222, respectively.

The10label pooling caveat persists: large cohort variability, and two selected
source validation folds reached the600 epoch ceiling. No new fitting was done on
the faster regime. The small-budget advantage is conditional on both label and
training constraints; it is not proof that more optimization could not help.
Pooling also receives validity masks and normalized higher-order distribution
information beyond z; its successful predictions do not identify its learned
mechanism. The existing z dyad-shuffle result remains the direct mechanism test.

PCA remains a credible candidate for a simple representation pipeline, while PLS
is the stronger10label readout in these tests. PCA axes ignore labels, but cap and
ridge selection use labels. The predeclared cap is truncated by training sample count and available feature count; cap choices are not estimates of intrinsic dimension. User preference for
PCA remains provisional; both readouts and the original primary analysis remain.

The strongest defensible claim is that cross-statistic relationships provide a
useful low-label representation and more reliably transferable decision scores
than the tested individual-statistic summaries in this co-organization setting.
The moments-generator negative result delineates a different regime where
retaining magnitudes or learning instantaneous raw features is more useful.
Neither result implies information inaccessible in principle from raw data.

## Verification and reproduction

All 400 MPI outputs and source hashes pass the cluster audit. Both downloaded banks
pass SHA verification; eight independent raw/MPI replays match feature values
within numerical precision and normalized edges exactly. All 99 model references
and prediction files are checked:72 statistical reconstructions reproduce original
source-regime predictions exactly;27 neural checkpoints are frozen. Independent
neural replays use saved raw views or re-extracted SPI edges for eight target
records per checkpoint; maximum discrepancy2.39e-7. See
`frozen-model-verification.json` and `completion-verification.json`.

Data/results root: `oscillatory_coorganization_transfer_260909`.
FeaturebankSHA `fb6e606d7e0a16745e71c17ff03ca86a850500d85f0435bb844ef0d9ba8dfc05`;
manifestSHA `6fbcba90a7be9678aca3db5c072ad34179d06016cbcfc6888b8726bbb8bcff1f`.
242–282 SPIs valid per recording. Per-record extraction times sum to 23.79 hours (one worker per recording); dataset
parallelism made it operationally affordable. Production178543332/427 Exit0
(7:09/1:26), analysis178543622 Exit0(3:40). Two data-mover attempts failed; disabling
SSH multiplexing coincided with a successful verified download, not a proven
root-cause diagnosis. No extraction was repeated because of transfer failures.

Reproduce with `scripts/evaluate_oscillatory_transfer.py`,
`report_oscillatory_pilot.py --source-data data/oscillatory_coorganization_pilot_260909`,
`check_oscillatory_transfer.py`, and `plot_oscillatory_transfer.py`.
Authoritative tables: `results/oscillatory_coorganization_transfer_260909/verified-report/`;
all 99predictions and source references: `predictions/`. Source comparison tables
remain in the original pilot's `pca-pooling-report/`; no historical output replaced.

Stop this approved phase: no additional same-generator extraction, shift grid,
neural tuning or PCA sweep is justified by the remaining uncertainty. A next
study should add independent scientific scope or resolve a specifically identified
limitation, rather than repeat these largely saturated conditions. No journal
acceptance, clinical usefulness or standalone-paper decision is established here.
