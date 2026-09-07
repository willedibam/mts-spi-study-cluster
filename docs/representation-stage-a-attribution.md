# Stage A: marginal richness and nonlinear readout

Completed exploratory analysis, 2026-09-07. This follows the negative Stage B
result for synchronization-state prediction under observation changes; it does
not replace that result. The question is whether the positive synthetic-class
screen depended on limited marginal summaries or a linear readout.

## Design and verification

- Reuse all 2,660 hash-verified Stage A MPI archives, 289 p90 SPIs, and the same
  source/evaluation split. No new pyspi computation or additional labels.
- Cross four feature views (m, richer m, z, richer m+z) with logistic regression
  and an RBF SVC. Richer m uses 23 features/SPI: mean, standard deviation,
  skewness, Pearson kurtosis and 19 quantiles from .05 to .95.
- Preserve training-only missingness/variance gates, clipping, block balancing,
  PCA32, nested label subsets and two-fold C selection. Both heads use the same
  five C candidates; RBF fixes gamma='scale'. This is one bounded nonlinear
  comparison, not the best possible learner for either representation.
- All historical evaluation records were already inspected. The new protocol
  was specified before the attribution outcomes, but this is not confirmation.
- Original m/logistic and z/logistic predictions replay exactly at every budget,
  seed and evaluation row. The original seven summaries are preserved within
  richer m. Ten relevant tests pass, including nonlinear selection isolation and
  moment/quantile semantics. Extraction took 36.5 s and the 120 fits took 29.5 s.

## Results

Primary metric: balanced accuracy when both M and T differ from source. These
M changes alter the physical system size, unlike the fixed-population Stage B
sensor-subset experiment.

| Representation / head | 2/class | 4/class | 8/class |
|---|---:|---:|---:|
| Original m / logistic | .7914 | .8318 | .8546 |
| Original m / RBF | .7691 | .8387 | .8595 |
| Richer m / logistic | .8020 | .8443 | .8684 |
| Richer m / RBF | .7698 | .8468 | .8627 |
| z / logistic | .8905 | .9123 | .9168 |
| z / RBF | .8829 | .8963 | .9059 |
| Richer m+z / logistic | .8612 | .8791 | .8848 |
| Richer m+z / RBF | .8684 | .8875 | .8839 |

At eight labels/class, z exceeds richer m by .04839 with logistic regression
(paired conditional 95% interval .03929–.05804), and by .04321 with RBF
(.03571–.05125). Adding z to richer m improves the logistic result by .01643
(.00786–.02607). All three budgets show positive paired gaps for those
comparisons. Intervals condition on fitted models and have no multiplicity
adjustment. Concatenation remains weaker than z alone under this pipeline.

This supports utility beyond these richer descriptors and this nonlinear head.
It does not establish inaccessible information, the specific source of the gain,
optimality of Pearson pooling, or cross-system transfer of a common state.

## Where the gain occurs and what follows

At eight labels/class, two classes contribute 76.4% of the **net** aggregate
z/logistic minus richer-m/logistic gain:

- `var-phi-0.2_cpl-0.8`: .9225 versus .6300.
- `var-phi-0.95_cpl-0.4`: .7250 versus .5000.

The generator at the recorded source commit fbbb903, `src/generators/linear.py`,
is a Gaussian linear VAR with zero MA coefficients for these configurations.
Its transition matrix is rescaled to spectral radius .98, so class-name phi and
coupling values are not literal final coefficients. This concentration argues
against explaining the aggregate gain as nonlinear interaction discovery.
Brownian-defect is worse with z (.9175 versus .9650); several classes saturate.

The next targeted controls are independent SPI-wise dyad permutations (retain
per-SPI distributions, reciprocity and validity while disrupting alignment), and
direct regularized linear-dynamics descriptors on the raw data. A shared dyad
permutation is an invariance check, not a new predictor. Perturbed MPI objects
need not be realizable time series, so this is a representation-level mechanism
test. Do not pick a new biological/engineering benchmark merely because the
current representation happens to score well on it. A later shared-property
task needs a reason for cross-statistic relationships to matter and fresh
generator-family/real-data validation.

## Reproduction

Run `scripts/run_representation_attribution.py` with
`configs/analysis/representation-stage-a-attribution-260907.yaml` and a new
output directory. The rich-bank cache verifies its manifest/module/content
identity; completed fit directories are not overwritten. Results, class-wise
scores, paired intervals, predictions, source hashes and exact replay checks:
`results/representation_stage_a_260907/attribution/`.

The Stage B final report is
`results/representation_stage_b_260907/final/report/`; its 90 local statistical
fits reproduce Gadi to 1.73e-14 with identical selected regularization.
