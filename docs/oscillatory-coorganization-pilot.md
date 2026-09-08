# Fresh pilot: transfer of phase/envelope co-organization

Frozen after the96-view exploratory scout, before generating this pilot.
The [scout protocol](oscillatory-coorganization-scout.md) explains motivation and
limitations. All36scout fits replay locally with identical selections and maximum
prediction discrepancy1.25e-14. P90 captures the intended organization at the
original size; shifted performance and marginal-versus-z differences are less
stable. This justifies independent replication of those effects, not a larger
claim based on the same12test blocks.

Keep exactly the same generator, nuisance distributions, four8-channel groups,
and binary aligned/crossed conditions. The target concerns functional
co-organization under common drives, not direct causal edges or consciousness.
No new family, spectral condition, sensor geometry or noise grid is added.

## Sampling

- Masterseed260909211. Every recording now has independent nuisance and innovation
  draws, including across conditions. No paired-condition simulations in this
  pilot. Class-balanced acquisition is assumed and reported.
- Three disjoint source cohorts, each20recordings/class (40labels). Nested5/10/20
  per class give10/20/40label curves. Cohort seeds11/23/47 fix ordering and inner
  splits. Total120training records.
- Independent100evaluation recordings/class,200total. N32 remains fixed; training
  observesM16/T1000. Evaluation observes this and nestedM8/T500. Use the finalT
  samples of each master, consistent with the existing neural/state runner;
  the earlier scout used the first500, an explicit view-definition difference
  for these stationary signals, not reused data.
- Total320masters/520views. Pretraining and sensor/duration augmentation: none.
  Two-fold stratified tuning stays inside each label budget. Independent training
  cohorts and evaluation masters are never mixed. Count recordings, not windows,
  as labelled examples; bootstrap paired observation views by master.

## Fixed comparisons

All statistical preprocessing/readouts use the earlier training-only scaling,
validity filtering, clipping and block balancing. PLS components1/2/4. z also gets
PCA/ridge with caps1/2/4/8/16/32 and ridge.01/.1/1/10/100, as before.

- z, rich SPI marginals m, normalized shapes, m+z, shapes+z.
- Graph summaries and validity flags as diagnostics.
- Raw autospectrum; marginal moments, covariance, cumulants and window summaries
  concatenated; individual phase/envelope summaries concatenated; their direct
  Pearson/Spearman agreement. Keep the direct task-informed competitor even if
  it wins. No supplied latent groups/drivers.
- The corrected94,017parameter aligned-channel and6,993parameter temporal-pair
  encoders, unchanged from the covariance study, trained from raw data. Same
  twolearningrates×twoweightdecays, unclipped validation stopping,600maximum
  epochs, source-selected refit epochs. These are small controlled neural
  comparators, not a claim against all neural/foundation methods.

Primary utility metric: balanced accuracy at the fixed.5 score threshold after
joint M/T reduction. Also report AUROC (ranking), Brier score, and MAE for both
sizes. Fit/select the existing regression readouts using clipped validation MAE;
do not choose a test threshold or calibrator. PLS/neural outputs are bounded
scores without an established calibration guarantee. AUROC success with poor
threshold accuracy must be described as preserved ranking, not successful
automatic state decisions.

Primary contrasts: zPLS−richm and zPLS−normalizedshapes; assess m+z−m and
shape+z−shape as separate complementarity questions. Compare neural/direct raw
controls regardless of ranking. Report all3cohorts, allbudgets and both sizes;
conditional master-bootstrap intervals are pointwise, not corrected for all
comparisons. No best-cohort or best-budget claim. For this exploratory-to-fresh
pilot transition, consistent useful performance and effect magnitude matter
more than a single nominal significance threshold.

Stop scaling this experiment if differences disappear, are confined to ranking
without usable decisions, or merely reproduce specialist performance without
an informative representation result. A positive result warrants a narrowly
stated contribution and a separately justified next step; it does not establish
clinical value or independently ensure a top-venue paper.

## Supplementary frozen-encoder control

Declared after the trained-neural results, before full pilotSPI results: both
trained raw encoders remain near chance. Compare frozen random aligned-encoder
features with matched PCA/ridge and PLS heads, using the same labels/cohorts and
no pretraining. This tests whether end-to-end fitting harms already useful
features. It is supplementary, not part of the original frozen comparison set.
Existing runner`run_representation_state_random_control.py`; no newdata/p90.
