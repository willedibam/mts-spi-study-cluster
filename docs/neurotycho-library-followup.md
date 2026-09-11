# Conventional feature baselines: declared follow-up

User authorized 11 September 2026 local time, after inspection of the completed
PF pilot. This is a follow-up comparison, not a new untouched confirmation set.
The original [protocol](neurotycho-transfer-pilot.md), predictions and
[findings](neurotycho-transfer-findings.md) remain unchanged.

Question: how competitive is Pearson z against three established feature
representations on the existing **full-size** animal/agent/period transfer task?
No expectation of beating them is built into the experiment.

Reuse all 352 matched source windows and 128 PF windows, each 16 bipolar channels
by 2,000 samples. Exclude each target animal's KTMD data before fitting anything.
Use the same three source-animal validation folds, select mean balanced Brier,
retain fixed threshold .5 and report date→animal BA/AUC/Brier. No target calibration,
new recordings, SPI extraction, reduced-view adaptation, neural fits or enriched
exposure extension. These methods receive the matched window budget; the already
completed enriched neural result remains a stronger-exposure reference.

## Fixed representations and classifiers

- **catch22:** pycatch22 0.5.0, its 22 standard features per channel, no additional
  mean/SD coordinates (not catch24). Extract from filtered float64 channels;
  retain the package's internal normalization conventions.
- **tsfresh:** 0.21.2 `EfficientFCParameters`, all resulting coordinates per
  filtered float64 channel, including amplitude-related features. This is the
  documented efficient catalogue, not the smaller Minimal set. No supervised
  tsfresh relevance filter or data-dependent catalogue restriction.
- **MiniRocket:** aeon 1.5.0 multivariate transform, requested 10,000 features
  (implementation rounds to a multiple of 84), maximum 32 dilations/kernel,
  seeds 11/23/47. Each channel is standardized within its own window and converted
  to float32, matching the raw neural input convention. Learn transform biases
  separately on each inner training fold, then on the final source training set.
  Transform fitting must not see validation or PF observations.

For catch22 and tsfresh, source validation chooses between concatenating the
16 channel-feature vectors and pooling each feature across channels using
mean, SD, 25th percentile, median and 75th percentile. Pooling ignores nonfinite
values, retaining NaN when all channels are invalid. This choice addresses the
lack of guaranteed anatomical channel correspondence; pooling omits explicit
cross-channel interaction features. Channel-concatenated and MiniRocket results
do not establish permutation invariance or variable-channel support.

All three use median imputation and standard scaling fitted only to inner
training observations, followed by L2 logistic regression. Retain coordinates
finite in at least 95% of inner training windows and with post-imputation SD
above 1e-8. C is selected from .01/.1/1/10, identical to the original spectral
baseline grid; max_iter=2000, solver=lbfgs. Equal date/state loss weights match the
existing protocol. Ties prefer pooled features, then smaller C. No PCA is forced
on these baselines. These are explicitly **feature-transform + logistic** pipelines,
not claims about every classifier available in each library or the exact default
MiniRocket/RidgeClassifier pipeline.

Save all candidate fold predictions, selected settings, source IDs, extraction
times, runtime versions, input/config/code/model hashes and final source replay.
Freeze all ten source models (two catch22, two tsfresh, six MiniRocket) before
evaluating any of their PF predictions. Check data alignment, independently
recompute scores and reload saved models. If a library fails or optimization
does not converge, diagnose before scoring; do not report an implementation
failure as a z advantage. Do not extend the grid based on PF outcomes.

## References and execution

[MiniRocket documentation](https://www.aeon-toolkit.org/en/stable/api_reference/auto_generated/aeon.transformations.collection.convolution_based.MiniRocket.html),
[tsfresh efficient settings](https://tsfresh.readthedocs.io/en/stable/text/feature_extraction_settings.html),
[pycatch22](https://github.com/DynamicsAndNeuralSystems/pycatch22).
Settings are in `configs/analysis/neurotycho-library-followup-260911.yaml`.
Use a separate `.venv-neuro-baselines` overlay; do not change the shared pyspi
environment or unrelated workstream dependencies. Benchmark source extraction
first and choose local or scheduled cluster computation based on measured cost.
