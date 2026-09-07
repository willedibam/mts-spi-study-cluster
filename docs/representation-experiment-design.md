# A focused experiment design for SPI–SPI representations

**Question:** when do relationships among dependence measures improve inference
of a system's dynamics, beyond practical alternatives, under changing observation?

The paper needs three experiments with distinct purposes—not a growing collection
of unrelated benchmarks. Learning over z is a separate claim from using fixed z.

| Purpose | Experiment | What it establishes |
|---|---|---|
| Utility | Existing broad class screen, plus the synchronization-state pilot | Where z helps and where a simpler observable is sufficient. Retain both results. |
| Explanation | The focused VAR mechanism check below | Whether edge alignment matters and whether direct dynamics estimates explain the task. |
| Transfer | A prospective four-cell state-change experiment, if justified | Whether real dynamical changes remain distinguishable from observation changes on a held-out family. |

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

This is a **conditional proposal**, not a new benchmark already launched.
Its motivation is concrete: avoid calling a change in available sensors a change
in the underlying system. Cross two factors independently:

| | Observation unchanged | Fewer sensors / shorter recording |
|---|---|---|
| Dynamics unchanged | Reference stability | False alarms from observation changes |
| Dynamics changed | Sensitivity to a real regime change | Transfer under observation change |

The positive label comes from a specified intervention on the coupling rule, not
thresholding an SPI. Keep physical population size fixed. Start with one source
shape and one observation shift; recompute SPIs from the observed channels. Do
not mix sensor subsampling with resimulating a different physical system.

Linear VAR is the model-based anchor. A stochastic nonlinear autoregression is
a defensible second family **only to test transfer beyond the linear explanation**:
it can retain discrete time, channel count and observation law. Train on one
family and test the same change/no-change target on the other, then reverse the
direction. Two families would be a pilot, not evidence of universality. Additional
oscillators or dataset names are unnecessary unless they test a distinct failure.

Compare a small set fixed before evaluation: m, z, m+z, direct dynamics/change
estimates, and a competent raw encoder. Report false-alarm rate and detection
quality in all four cells, with learning curves over independent labelled systems.
Split masters before forming any windows, views or pairs. Keep noise, sampling
frequency and sensor mixing fixed initially; add a nuisance only when a use case
requires it. Exposure from pretraining or augmentation must be matched and recorded.

A coupling change may alter strength without altering normalized SPI relationships;
that is part of the test, not a reason to discard a losing case. Fresh realizations
are essential after exploratory choices. No real application is established here;
a real-data claim still needs independent labels and meaningful coverage variation.

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
the generic change-detection table above remains an alternative application.
Only raw target feasibility has been checked. No new pyspi farm or learned
representation result exists for that candidate.

Geometry belongs in this same test: a useful distance or learned map should
separate true regime changes from observation changes. A new matrix metric or
attractive embedding is not a separate scientific contribution by itself.

Relevant precedent: [Cliff et al.](https://arxiv.org/abs/2201.11941) establish the
utility of diverse dependence measures, so their relationships need an incremental
argument. [VAR model documentation](https://www.statsmodels.org/stable/vector_ar.html)
provides the standard linear-dynamics reference. Dyad invariance follows from the
Pearson edge-vector construction and is checked in repository tests.
