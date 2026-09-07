# Prospective interaction-share confirmation

Protocol: `configs/analysis/interaction-share-confirmation-260908.yaml`.
This design is fixed before generating fresh data. The pilot and its post-result
controls remain separate evidence in [findings](interaction-share-findings.md).

## Question and decision

Do aligned relationships among dependence measures improve prediction of future
interaction share over individual-SPI distribution summaries, with 10–40 labelled
systems and a joint change of observation coverage and map family?

The pilot supports this narrow hypothesis after matched normalization, a bounded
nonlinear readout and a dyad-alignment control. It does not establish superiority
over a calibrated dynamical reference. The latter remains in this confirmation.

A larger exact copy would mainly reduce conditional test uncertainty. This run
instead addresses two concrete limitations: five discrete nominal parameter
settings and overlapping training subsets. It is a continuous-parameter extension,
not an identical-distribution replication or a new generator search.

## Fixed experiment

- Same 32-node ring maps, linear/tanh coupling, independent gain U(.25,.8), noise
  SD .5, 300 burn-in steps, 1,000 input and 1,000 future steps. The target remains
  the future off-diagonal squared drift-Jacobian energy divided by total energy.
  State units and discrete time step are fixed. This is a local-sensitivity probe,
  not a universal order parameter or a test of causal identifiability.
- Draw nominal shares continuously in five equal-width bins covering [.1,.9].
  Each family has 200 training and 200 evaluation masters. Within each family,
  five disjoint training cohorts each have eight independent masters per bin.
  The 10/20/40-label budgets are nested only within a cohort. Sampling is balanced
  over known simulation strata, not a claim about unstratified label acquisition.
- Total: 800 independent masters; 1,200 observed records. Training uses M16/T1000.
  Each evaluation master supplies paired M16/T1000 and nested M8/T500 views.
  All labels used for tuning count against the budget. No target-family training,
  pretraining, augmentation or use of future samples as inputs.
- Two transfer directions, with source/source, source/shift, other/source and
  other/shift cells retained. Primary score is equally weighted joint-shift MAE.
  The families are closely related; call this specified map-family transfer,
  not broad transfer to unseen dynamical families.

## Frozen comparisons

Primary: z/PLS versus standardized marginal shapes/PLS; shapes+z/PLS versus
shapes/PLS. Secondary: z/PLS versus z/PCA-ridge, frozen random encoder/PCA-ridge,
and calibrated linear/linear+tanh references. Rich unnormalized m/PLS, own-memory
and source median remain reference points. No new end-to-end encoder tuning is
needed for this question; its pilot result remains reported, with its limits.

All statistical methods use the same 289-SPI p90 catalogue, train-only validity
filtering, imputation, clipping and block weighting. PCA caps and regularization,
PLS components and raw-reference calibration grids are unchanged. The random
encoder architecture and seeds are unchanged. Both off-diagonal MPI triangles
enter z. Shapes retain skewness, kurtosis and 19 standardized edge quantiles.

Report every budget, direction and cohort, including reversals. Paired evaluation
bootstrap intervals are conditional on the fitted models, pointwise and not
adjusted for multiple comparisons. Five independent training cohorts expose
training variability but do not justify precise unconditional population CIs.
Do not judge success by whichever single budget or direction looks best. A useful
confirmation should show an appreciable marginal-comparator advantage across
budgets and directions, with cohort results revealing whether it is fragile.
The random-encoder comparison is secondary, particularly the pilot's 10-label
advantage; failure to beat it at 40 labels is not concealed.

## Execution and stopping boundary

The pilot's 520-record alignment control completed successfully on Gadi
(`178391492`, code `4932773`); shuffling removes prediction while preserving edge
multisets and validity. The continuous/cohort implementation tests pass. These
satisfy the launch gate. Generate with fresh master seed 26090883 and bind source
archive, manifest, protocol and code hashes before fitting.

Use a fresh two-record p90 smoke test. The completed representative 48-record
pilot node gate already covers the same physics range and M/T shapes (maximum
361 seconds, 58 GiB aggregate). Reuse that sizing evidence: separate source and
shift farms, one core/record, 4 GB/core, 1,200-second record cap. Expected total
record computation is roughly 80 hours, with wall time reduced by dataset-level
parallelism. Check live allocation and inodes before submitting.

Audit all records before bank construction and inference. Verify data hashes,
all nested views, disjoint cohorts and exact simulator replays. Once complete,
interpret against the pilot without comparing raw MAEs across different target
distributions. Do not expand the family/topology/sensor grid based on rankings.
Even a positive result supports a controlled representation finding, not by
itself a compelling general-purpose method or a high-impact publication claim.

