# Learned aggregation and held-out dynamics

Declared after the completed co-organization pilot, before new model fitting or
new-regime generation. This is a focused follow-up, not an independent
confirmation of an architecture selected before the original pilot. User approved
one learned-pooling comparison followed by one frozen dynamics-transfer test.

## Question and interpretation

Does explicit cross-statistic correlation expose co-organization with fewer
labels than learning an aggregation of the same standardized SPI measurements?
Does that representation transfer to faster dynamics with unchanged group
organization? Both are within-generator questions. No causal, clinical,
foundation-model or cross-system claim follows automatically.

For recording r and ordered link e=(i,j), let v_e contain the289SPI values.
Standardize each SPI over all M(M-1) links using its recording mean and population
SD. A nonfinite or constant SPI is invalid as in unified_ordered_v3; set its
normalized values to0 and supply a validity indicator at every link. Both edge
directions are retained, SPI identities are fixed coordinates, and no node
identity, incidence, simulator variable, label or test-set fitted transform enters.
For valid SPIs, z_ab is exactly the mean product of standardized columns a,b.
The learned comparator receives the inputs to this operation, not z itself.

Use one Deep-Sets-style network: per-edge Linear(578,32), GELU, Linear(32,32),
GELU; concatenate mean and populationSD of these32features across edges; then
Linear(64,32), GELU, Linear(32,1). Pooling is invariant to common edge permutation,
with parameter count independent of M. Normalization deliberately matches z's
location/positive-scale invariance. This comparator does not test whether
retaining magnitudes would help, or reproduce a published raw-time-series model.
Validity indicators retain missingness information; existing validity controls stay.
All289SPI slots remain; unlike z's training feature filter this does not restrict
the network to95%-valid SPI pairs. Report this access/preprocessing distinction.

No topology is available to either method. Learned nonlinear aggregation and its
optimization differ from the PLS/PCA readouts; a result cannot isolate Pearson
compression from every downstream model-class effect. Explicit quadratic
per-edge functions followed by mean pooling can reproduce z, so missing raw
information cannot explain a failure of learned pooling in principle.

## Matched fitting, existing data

Use the unchanged pilot320masters/520views and all three disjoint source cohorts,
10/20/40total labels, same two-fold stratified splits. AdamW/MSE, learning rates
1e-4/1e-3 and decay1e-4/1e-2, batch16,600epochs, patience20, minimum30. Stop on
unclipped validationMAE; select by clipped validationMAE as in existing neural
comparisons; final refit uses rounded median selected fold best epoch. No
pretraining, observation augmentation, test-label tuning or architecture sweep.

Check exact z recovery from standardized edges, order invariance, finite
missing-value handling, gradient flow, small synthetic correlation-task learning,
checkpoint/prediction hashes and CPU replay. Synthetic checks validate executable
capacity/optimization before actual pilot evaluation; no real test outcomes select
architecture changes. Preserve every fit and report BA/AUROC/Brier/MAE at both
observation sizes, paired with original pilot methods. No repeated epoch doubling.

## One prospective faster regime

Keep N32, group partitions, class balance, jitter, amplitude scale, sensor gain,
noise and observation views unchanged. Change only three timescale parameters:

| Parameter | Original training regime | Faster held-out regime |
|---|---|---|
| Carrier frequency, Hz (Fs100) | U[6,12] | U[13,16] |
| Phase innovation SD per sample | U[.10,.18] | U[.20,.26] |
| Envelope AR coefficient | U[.94,.98] | U[.86,.92] |

These disjoint ranges define a joint faster-dynamics intervention. The carrier
stays within the existing3–20Hz control passband; no specialist retuning. More
rapid phase diffusion and envelope mixing shorten correlation times without
changing the latent membership target. Observed strengths/estimation quality can
still change; this is not an invariance theorem or a measured physiological range.
This first experiment does not isolate the three timescale changes individually.

First generate32independent raw scout masters (16/class), seed260909307, no p90.
Gate: direct observed agreement PearsonAUROC>=.90 at both M16/T1000 and M8/T500,
with finite controls. Preserve a failed gate; do not search new shifts to obtain a
win. If it passes, independently generate200test masters (100/class), seed260909311,
400views. The scout is excluded from fitting and final tests. No new training
records or labels. Retain original source-fitted preprocessing, chosen settings
and epochs for every method, including direct specialist and learned pooling.

Primary: faster-regime BA at original M16/T1000, separating dynamics transfer from
the already studied size shift. Secondary: combined dynamics+M/T shift, AUROC,
Brier and original-regime scores. Keep zPLS/PCA, richm, shape, m+z, direct agreement,
raw combined moments, both original raw neural encoders and new learned pooling;
no additional model family. Test labels are used for reporting only.

A gate pass authorizes p90 smoke,48record node test and a bounded400view farm on
Gadi using measured resource limits. Freeze config/code before generation and
before fitting; verify data/feature provenance and source-only refits. No larger
same-generator replication or further shift grid automatically follows.

## Sources

Deep Sets: https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
Set Transformer: https://proceedings.mlr.press/v97/lee19d.html
The architecture above is a small shared-MLP pooling comparator, not a reproduction
of Set Transformer. SPI-relationship precedent and claim boundaries are in the
workstream context and original pilot findings.

## Execution note before learned-pooling results

Use GadiCPU training (three disjoint cohort processes, two threads each) while
transferring the77MBedge bank for local replay. This avoids making fitting depend
on a slow download. Architecture, source splits, grid, stopping and seed remain
unchanged; no local duplicate fits. Device/runtime are recorded per fit; previous
raw encoders used MPS, so cross-device bitwise training identity is not claimed.
