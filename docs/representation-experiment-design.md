# A focused experiment design for SPI–SPI representations

**Question:** when do relationships among dependence measures improve inference
of a system's dynamics, beyond practical alternatives, under changing observation?

The paper needs three experiments with distinct purposes—not a growing collection
of unrelated benchmarks. Learning over z is a separate claim from using fixed z.

| Purpose | Experiment | What it establishes |
|---|---|---|
| Utility | Existing broad class screen, plus the synchronization-state pilot | Where z helps and where a simpler observable is sufficient. Retain both results. |
| Explanation | The focused VAR mechanism check below | Whether edge alignment matters and whether direct dynamics estimates explain the task. |
| Transfer | The local-sensitivity pilot: family shift x observation shift | Whether one shared dynamical property can be inferred with fewer labels under both shifts. |

## 1. The completed mechanism experiment

**Question:** does the same-channel-pair correspondence across SPIs explain the
VAR gain, and is z needed once linear dynamics are estimated directly?

All three existing VAR classes enter a new three-way diagnostic. Every method
is refitted on identical subsets from the original screen: source M16/T1000,
2/4/8 labels per class including validation, five subset seeds, 60 independent
evaluation class-instance groups. Physical size and duration vary in evaluation.
These accuracies are not directly comparable with the earlier 14-class task.

| Representation | Purpose | Shifted accuracy at 8/class |
|---|---|---:|
| Richer SPI marginals | Same catalogue; distribution shape without alignment | .4750 |
| Intact z | Representation of interest | .6892 |
| Disrupted z | Preserve each SPI's values, reciprocity and validity; break correspondence between SPIs | .3492 |
| Two direct VAR summaries | Estimate self-memory and total cross-channel contribution from raw recordings | 1.0000 |
| SPI validity | Detect predictive estimator-failure patterns | .3592 |

Chance is 1/3. Three independent null runs are averaged as technical repetitions,
not additional systems or a prediction ensemble. A shared dyad permutation is an
invariance check. All methods use the same logistic/PCA procedure; the nonlinear
head question was already tested separately. No new pyspi computation was needed.

**Conclusion:** alignment matters, but direct linear estimation solves this task
more effectively. The correctly specified VAR model has a justified advantage in
this diagnostic; this does not establish a universal alternative to z. Likewise,
the alignment result does not separate algebraic SPI aliases from distinct physical
mechanisms. It does not justify a large representation-learning campaign yet.

Details, paired intervals and verification are in
`results/representation_stage_a_260907/var-mechanism/`;
protocol `configs/analysis/representation-var-mechanism-260907.yaml`.

![Mechanism check and direct VAR descriptors](../results/representation_stage_a_260907/var-mechanism/mechanism-summary.png)

The right panel shows all 570 recordings, including the source training pool;
it is a descriptive view, not an additional validation result. Regime names
refer to configured parameters; estimated coefficients reflect the generator's
spectral-radius rescaling.

## 2. The smallest sensible transfer design

The [local-sensitivity pilot](interaction-share-pilot.md) is the active proposal.
Predict the future share of squared drift-Jacobian sensitivity attributable to
other channels. Keep physical N=32 fixed; cross two factors:

| | Source observation M16/T1000 | Reduced observation M8/T500 |
|---|---|---|
| Source family, fresh masters | Within-family prediction | Observation shift |
| Held-out family, fresh masters | Interaction-law shift | Joint shift |

The two families use linear or tanh cross-channel interactions. Train in each
direction, without choosing the more favourable one. Independently varying overall
contraction removes the initial fixed-sum generator's deterministic memory
shortcut. This correction was made from raw diagnostics before any z results.

Compare rich SPI marginals, z, m+z, pooled raw summaries, raw linear/nonlinear
models and the existing temporal/channel encoder. Include simple memory and
source-median controls. All learning curves count independent labelled masters,
including tuning; views never cross master-level splits. Keep noise, time step,
units and sensor mixing fixed. The pilot specifies the exact target, generator,
raw audit and remaining comparator calibration.

The primary contrast is joint-shift MAE at matched label budgets, with both
single-shift cells retained for interpretation. This is property inference;
generic change detection is deferred. No new application, broad family diversity,
or real annotation-cost saving follows from this controlled synthetic test.

## 3. Decision now

Preserve the positive class result, the negative synchronization result, and the
VAR explanation. Purposeful exploration of regimes where an inductive bias helps
is legitimate research. A correctly specified specialist beating z does not rule
out general-purpose utility when the model family is unknown. Hypotheses should
precede new outcomes, and any promising result needs fresh verification matched
to its scope; neither universal superiority nor an ever-growing benchmark is required.

The next bounded candidate is now specified in the
[local-sensitivity pilot](interaction-share-pilot.md). It estimates one common
property across linear and tanh interaction laws at fixed physical N, with one
observation shift. This restores property inference as the primary objective;
generic change detection remains deferred. The raw reference audit supports
a nontrivial test after removing the initial memory shortcut. No new pyspi farm
or z result exists for that candidate.

Geometry belongs in this same test: a useful distance or learned map should
separate true regime changes from observation changes. A new matrix metric or
attractive embedding is not a separate scientific contribution by itself.

Relevant precedent: [Cliff et al.](https://arxiv.org/abs/2201.11941) establish the
utility of diverse dependence measures, so their relationships need an incremental
argument. [VAR model documentation](https://www.statsmodels.org/stable/vector_ar.html)
provides the standard linear-dynamics reference. Dyad invariance follows from the
Pearson edge-vector construction and is checked in repository tests.
