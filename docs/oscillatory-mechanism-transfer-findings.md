# Direct phase-mechanism transfer: verified findings

The frozen representation transfers to directly coupled phases, but the result
qualifies the earlier low-label and robustness claims. At the original observation
size, z PCA/ridge, z PLS and learned SPI pooling all classify every target record
correctly with 40 source labels. With reduced observation, learned pooling has
better fixed-threshold accuracy than z, while their class rankings remain very
similar. Individual-SPI summaries retain a substantial disadvantage in ranking
as well as decision accuracy.

The [protocol](oscillatory-mechanism-scout.md) replaces shared phase drivers with
direct noisy Kuramoto interactions; amplitude and observation laws are retained.
The group-average diffusion is analytically matched to the original model, and
the strong-coupling limit approaches common motion. This is a controlled change
in one generating mechanism, not transfer between arbitrary unrelated systems.
Independent long references, step-size checks and a coupling-removal intervention
passed before confirmation. No result here establishes a clinical application.

## Frozen comparison

Fresh 200 target masters (100/class), each observed at M16/T1000 and M8/T500.
All 99 source-fit references, predictions and neural weights were bound by a hash
manifest before target generation. Models retain the three original disjoint
source cohorts, nested 10/20/40 total-label budgets, preprocessing and selected
settings. No target training, calibration or model selection enters the primary
comparison. The two views share masters and are not independent test cohorts.

Balanced accuracy at the fixed .5 threshold, mean over three source cohorts:

| Representation/readout | M16: 10 labels | 20 | 40 | M8: 10 labels | 20 | 40 |
|---|---:|---:|---:|---:|---:|---:|
| z + PCA/ridge | 79.2% | 99.5% | 100.0% | 66.7% | 83.8% | 89.0% |
| z + PLS | 88.2% | 99.5% | 100.0% | 73.7% | 84.3% | 88.8% |
| Learned SPI pooling | 67.2% | 99.2% | 100.0% | 65.7% | 85.2% | 93.8% |
| Individual-SPI summaries | 57.0% | 64.3% | 71.8% | 54.8% | 60.8% | 67.5% |
| Normalized individual-SPI shapes | 62.0% | 62.0% | 71.8% | 51.3% | 51.0% | 52.0% |
| Summaries + z | 73.2% | 90.7% | 96.2% | 64.0% | 73.7% | 81.8% |
| Shapes + z | 74.5% | 88.0% | 97.7% | 59.8% | 65.3% | 76.3% |
| Direct phase/envelope agreement | 100.0% | 100.0% | 100.0% | 94.3% | 94.2% | 94.3% |
| Raw marginal/covariance/cumulant/window summaries | 57.3% | 72.0% | 72.8% | 54.3% | 66.2% | 64.8% |
| Raw aligned-channel encoder | 50.7% | 48.3% | 50.8% | 50.7% | 49.3% | 50.0% |
| Raw pair encoder | 49.0% | 50.2% | 49.5% | 50.2% | 51.8% | 52.8% |

At 40 labels, zPLS−summaries is +28.17 percentage points at M16, conditional
pointwise 95% CI [24.16,32.33], and +21.33 at M8 [17.17,25.83]. At reduced
observation, PCA−pooling is −4.83 points [−7.50,−2.33]. These intervals resample
independent target masters within class and condition on the three fitted source
cohorts; they do not quantify variation over a population of training cohorts.

At 10 labels, PCA−pooling is +12.0 points [8.5,15.5] at M16, but only +1.0
[−2.17,4.33] at M8. PLS−pooling at M8 is +8.0 [4.5,11.67]. Thus the PCA small-data
advantage is not consistent across this combined shift. The earlier pooling
optimization caveat persists: two selected source validation folds at 10 labels
reached the fixed 600-epoch ceiling, and results vary substantially by cohort.
No claim of an optimization-independent advantage or universal label crossover.

![Source and direct-mechanism learning curves](../results/oscillatory_mechanism_transfer_260910/learning-curves.png)

## Ranking survives better than the decision threshold

At M8/T500 with 40 source labels:

| Representation | Frozen BA | AUROC | Brier |
|---|---:|---:|---:|
| z + PCA/ridge | 89.0% | .99833 | .08466 |
| z + PLS | 88.8% | .99777 | .08425 |
| Learned SPI pooling | 93.8% | .99813 | .07211 |
| Individual-SPI summaries | 67.5% | .80043 | .21953 |
| Normalized shapes | 52.0% | .78647 | .35553 |
| Direct agreement | 94.3% | .99347 | .04865 |

The three PCA models make zero false-positive errors but miss 25%,20%,21% of
aligned records. Their mean aligned scores are .617,.653,.645; mean crossed
scores .036,.053,.058. This is evidence of decision-score transfer failure despite
retained ordering. It is not evidence that Pearson compression erased the target
information on these records. Which physical/observation change produces this
score shift has not been isolated: M, T and phase mechanism change together in
the reduced comparison.

The [post-hoc diagnostic](oscillatory-transfer-score-diagnostic.md), declared at
`6544f01` and implemented at `a8d8c91`, applies one label-blind rule to saved scores:
label the highest-scoring half of each target batch as aligned. It uses the
unlabelled target batch **and known 50% prevalence**, additional information not
allowed in the primary source-only comparison. Ties are handled through expected
random assignments, so class order in the archive cannot help the rule. This is
not probability calibration or an assumption that application prevalence is known.

At reduced observation/40 labels, expected BA becomes PCA 98.33%, PLS 97.33%,
pooling 97.67%, summaries 73.33%, shapes 71.04%, and direct agreement 95.67%.
The pooling-versus-PCA fixed-threshold advantage largely disappears under this
different information budget, whereas the individual-summary deficit remains.
These are descriptive post-hoc comparisons, not a new superiority or equivalence
test. All budgets and comparators are retained in `score-diagnostic/summary.csv`;
the original 198 per-fit/view metrics and all prediction files remain unchanged.

## Scientific interpretation and stopping decision

The positive claim is transferable discrimination of this interaction property
using cross-statistic representations, with less successful extraction by the
tested individual-statistic summary readouts. Explicit z and learned aggregation
both work; learning the aggregation can improve transferred decision scores at
the larger source budget. The learned model also has normalized distribution and
validity information beyond z, so its advantage cannot be uniquely attributed to
aggregation flexibility, recovered topology or information lost by Pearson.

PCA remains a credible simple pipeline, but PLS is stronger at 10 labels and
learned pooling is stronger for fixed-threshold reduced-observation accuracy at
40. Neither changing the preferred method after seeing this target nor adding
another PCA sweep is justified. The specialist remains strong, especially with
few labels. The two small raw encoders do not stand for all neural methods or
pretrained time-series encoders.

Together with the covariance-modulation negative and correspondence intervention,
this supports a selective statistical inductive bias. It does not establish
information inaccessible from raw data, a general invariance to sampling or
measurement changes, cross-task reuse, or automatic suitability for a particular
venue. The current constructed property still needs an independent scientific
use case before making application-impact claims.

Stop the current phase. The main original-size comparison is saturated, the
combined-shift ranking/threshold distinction is now measured, and no implementation
failure calls for rerunning extraction or training. The renewed 12-hour capacity
enabled the focused diagnostic; it does not justify another simulator, larger
replicate or calibration grid. A future application or adaptation study should
begin with its own concrete target and resource contract.

## Verification and reproduction

All 400 MPI source/hash audits pass. Both bank hashes pass; eight independent
raw/MPI/feature replays match, including exact normalized edges. All 99 frozen
source references are checked; 72 original source predictions replay with zero
difference. All 27 neural checkpoints replay on eight independent raw/MPI inputs
each, maximum difference 4.18e−7. Source/target catalogue, feature-module hashes
and pyspi versions agree. The learning-curve figure was rendered and inspected.
The score diagnostic passes tie/permutation tests and reproduces all original
BA/AUROC metrics. See `completion-verification.json` in the results root.

Data/results root: `oscillatory_mechanism_transfer_260910`. Authoritative tables:
`verified-report/`; independent diagnostic: `score-diagnostic/`. Reproduce using
`check_oscillatory_pilot_features`, `evaluate_oscillatory_transfer` with the new
mechanism-transfer YAML and original source data/root, `report_oscillatory_pilot`
with `--source-data`, `check_oscillatory_transfer`, and `plot_oscillatory_transfer`.

Feature bank SHA `e81dffb2b330c85a370853b54d163e0aa9aa60985ea09cc9229917fbb3d2d9fa`;
normalized-edge bank SHA `8edee0c12d0a3d6e4365135b36ae92765a7f3a50f3f16b15450067af0c41e000`.
263–282 SPIs are valid per recording; the existing undefined group-delay SPI
remains invalid consistently with the source study. Summed per-record pyspi
time is 24.659 hours; dataset parallelism yielded production walltimes 7:13 and
1:53 plus 3:03 bank construction. The bank download finished without recomputation.
