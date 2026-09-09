# Learned SPI pooling: correspondence attribution

Declared 2026-09-10 before any null-pooling outcomes. The user approved the
application audit and this attribution check. This is a post-result mechanism
analysis of the original co-organization pilot, not a new confirmation dataset.

Question: does the competent learned SPI pooling model benefit from cross-SPI
dyad correspondence beyond each SPI's complete normalized edge distribution and
record-level validity? The existing z intervention does not answer this for the
learned model, whose inputs retain additional distributional information.

Reuse the original 520-view normalized edge bank and 200 evaluation masters.
Use only the 40-label budget, where the intact learner is competent, and all
three original source cohorts (11,23,47). Preserve the architecture, learning-rate
and decay grid, two source-only folds, stopping rule, 600-epoch ceiling and
training initialization. Reuse the three intact 40-label fits; do not refit them.
The runner's `--budgets 20` means 20 labels per class, 40 total.

For each record and SPI, independently permute unordered dyads, optionally
swapping the two directions together, using the existing `permute_dyads`
convention. Apply this directly to normalized edges; retain SPI identities,
validity flags, edge counts and padding. This preserves the full per-SPI edge
multiset and the reciprocal-pair multiset up to orientation, but destroys
cross-SPI correspondence. Seeds 503,509,521 are fixed before computation,
combined with the existing corpus row index. Shared permutations must preserve
the pooling prediction up to floating-point tolerance. These are representation
interventions; they need not correspond to realizable raw time series.

Refit and tune on each perturbed source bank, then evaluate on its perturbed
held-out records. Nine new fits: three permutations × three source cohorts.
Use the same source and evaluation indices as intact fits; no target fitting or
threshold adjustment. This avoids interpreting a test-only distribution shift
as proof that correspondence is required for learning.

Verify all 520 per-SPI edge multisets, reciprocal pairs, validity and padding;
check equivalence with the MPI-based permutation on small diagnostic matrices;
verify common-permutation model invariance; independently replay eight held-out
inputs per checkpoint. Report BA, AUROC and Brier for both M16/T1000 and M8/T500,
with every cohort and permutation visible. The nulls reuse the same masters and
are not nine independent test datasets. Do not select the weakest null or hide
variation between permutations.

A consistent loss supports incremental use of cross-SPI correspondence by this
learner under this task/training regime. Retained performance weakens that
attribution and points to marginal shapes or validity as sufficient alternatives
for this learner; it does not prove correspondence can never help. Neither
outcome proves a unique learned mechanism or inaccessible information.

Stop after the nine fits and verification. No new pyspi extraction, generator,
larger label grid, architecture sweep or additional optimization ceilings.
Run on local CPU unless resource contention justifies Gadi; this is a bounded
reuse analysis, and changing device does not change the scientific comparison.

## Verified result

Complete on local CPU. Protocol frozen at `12a2d8e`, implementation `7effbea`.
The three null banks preserve all 520 records' per-SPI edge distributions,
reciprocal dyads, validity and padding. All nine null fits have exactly the same
training and validation indices as their intact 40-label counterparts.

Balanced accuracy, means of the three source cohorts:

| Input | M16/T1000 | M8/T500 |
|---|---:|---:|
| Intact correspondence | 100.0% | 98.8% |
| Independent correspondence, seed 503 | 51.0% | 49.2% |
| Independent correspondence, seed 509 | 51.7% | 51.2% |
| Independent correspondence, seed 521 | 49.0% | 50.3% |

Reduced-view AUROCs are .5129,.4963,.4850 for the three nulls, versus .9997 intact.
Thus the collapse is not just a moved classification threshold. Under the tested
architecture, training budget and task, the learned model benefits substantially
from cross-SPI correspondence beyond preserved marginal distributions and validity.
This does not prove those preserved inputs contain no task information: existing
individual-SPI summary readouts already demonstrate otherwise. Nor does it prove
that Pearson correlations are the unique useful relationships learned by pooling.

All nine checkpoints replay on eight held-out inputs each (maximum error6.56e−7).
Shared permutations preserve the three intact models' predictions within1.20e−7
on eight inputs/model. No selected null fold reaches the600epoch cap. Two tests
independently verify matrix/edge permutation equivalence and common-permutation
model invariance. No original fit or pyspi extraction was repeated.

Results: `results/oscillatory_pooling_attribution_260910/`, including
`completion-verification.json`, `shared-permutation-and-split-verification.json`,
and `report/summary.csv`. All permutation/cohort results are retained. Stop this
attribution check; application validation remains a separate question.
