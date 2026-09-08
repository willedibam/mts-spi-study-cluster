# Focused proposal: cross-channel local sensitivity

**Historical design:** the pilot and confirmation are complete; see [current findings](interaction-share-findings.md). The active follow-up is the [raw phase-control experiment](interaction-share-phase-control.md). Historical status statements below describe the original planning stage.

Status: target and raw reference audit complete. The fixed-sum prototype had a
memory shortcut; the revised independent-gain generator is adopted for a focused
exploratory comparison. Fresh data and raw-control curves are complete; neural fitting is running and
p90 production is being prepared. No z comparison or confirmation result yet.
See [execution and fresh results](interaction-share-execution.md).

## Question and scope

Can a representation of cross-statistic relationships estimate a shared dynamical
property with few labelled systems when the local interaction law and available
channels change? The primary claim is label efficiency under a specified shift,
not information unavailable from raw observations or universal superiority.

Use one property: the share of local deterministic response sensitivity associated
with other processes, rather than the process itself. This keeps the original
property-inference question central; generic change detection is deferred. It
does not claim recovery of the full latent dynamical state.

For a discrete-time conditional-mean map f and J(x)=df/dx, define

\[
q=\frac{\mathbb E_t\|J(x_t)-\operatorname{diag}J(x_t)\|_F^2}
        {\mathbb E_t\|J(x_t)\|_F^2}.
\]

Expectations are averages over a future window disjoint from the observed input.
This is a precise, dimensionless **local-sensitivity share**, not a universal
coupling order parameter. It depends on state units and the discrete-time map;
we do not vary sensor scaling, mixing or sampling interval in this pilot.

Why care about the underlying quantity? Estimating changing interaction strengths
from observations is an established scientific problem, for example in
[empirical dynamic modelling](https://pmc.ncbi.nlm.nih.gov/articles/PMC4721089/).
That motivates studying cross-process response, but does not make this exact
ratio an established ecological observable. [Miki et al.](https://arxiv.org/abs/2411.09030)
warn specifically about interpreting Jacobian diagonals as self-interaction and
mixing discrete-time sensitivities with continuous-time interaction strengths.
Here the target is explicitly the derivative of the discrete conditional-mean
map, including persistence; no biological self-regulation claim is made.

## Two related model families, one controlled contrast

\[
x_{t+1}=a x_t+b W g(x_t)+0.5\epsilon_t,
\qquad\epsilon_t\sim N(0,I).
\]

- Physical N=32; W is a symmetric ring with two neighbors of weight 1/2 and no
  self-edges. Channel names/topology are unavailable to predictors.
- Linear anchor: g(x)=x. Nonlinear contrast: g(x)=tanh(x).
- Specify nominal linear sensitivity share r in {.1,.3,.5,.7,.9}. Independently
  draw one contraction nuisance gamma ~ Uniform(.25,.8) per master. Set
  h=sqrt(2r)/(sqrt(1-r)+sqrt(2r)), a=gamma(1-h), b=gamma h.
  Then linear q=r exactly, while tanh q is measured from the true Jacobian.
  The same r does not imply exactly the same q across families.
- This replaces a in {.10,.25,.40,.55,.70}, b=.80-a: fixing the sum makes linear
  q a deterministic function of self-memory a. The raw diagnostic below exposed
  that shortcut before any z result. Independent gain removes that deterministic
  restriction, not every possible marginal clue; retain marginal/raw controls.
- Both maps are globally Lipschitz with bound gamma<=.8 in Euclidean norm. Burn 300
  steps; observe 1,000 samples and evaluate q over the following 1,000 states.
- These are related model families, not a claim of broad cross-system diversity.
  Nonlinearity changes state-dependent derivatives while keeping units, time,
  topology, population and noise law fixed.

## Proposed bounded evaluation

Use fresh master seeds, disjoint from the raw feasibility sample. Per family and
r setting: 12 source-training masters and 20 evaluation masters. Each recording
uses the same prediction endpoint for all observation views.

| | Source observation M16/T1000 | Reduced observation M8/T500 |
|---|---|---|
| Source family, fresh masters | Within-family prediction | Observation shift |
| Held-out family, fresh masters | Interaction-law shift | Joint shift (primary) |

Train on linear and test tanh, then reverse the direction. Do not select the more
favourable direction. Label budgets are 2/4/8 per r setting: 10/20/40 independent
masters including inner validation. All windows from a master stay in one split.
No unlabelled pretraining or augmentation in the initial comparison; this keeps
the first result interpretable. A later paired-view learning claim would need its
own matched-exposure experiment.

Primary report: MAE learning curves for joint shift in both directions and their
prespecified equal-weight mean. The other cells diagnose which shift matters.
Intervals group by master. Counting three observation windows as three independent
systems is prohibited. Labels are cheap here; this pilot cannot establish real
annotation-cost savings by itself.

Use a small, strong comparator set: rich catalogue-matched marginals, z, their
concatenation, pooled raw summaries, direct linear and nonlinear system estimates,
and the existing raw temporal/channel encoder. Include the encoder in this first
comparison, rather than waiting for a favourable z result. Raw references now use
ridge one-step models with linear or linear-plus-tanh coordinate bases and analytic
Jacobians. Predictors receive neither the true family nor a,b,W. The corrected
energy ratio additionally assumes known physical N=32; disclose this advantage.
Before the matched-label benchmark, give both raw models the same bounded
regularization selection within the training budget; the fixed-ridge feasibility
audit is not an optimized nonlinear comparator. Include source-median and
single-channel memory controls. Do not interpret weak performance of this specific
nonlinear estimator as a failure of nonlinear system identification generally.

For statistical views, compare the existing train-selected PCA/ridge procedure
with one supervised-compression alternative, PLS with 1/2/4 components selected
within the same label budget. Apply this alternative to marginals as well as z.
This tests whether target-relevant low-variance directions were lost by PCA;
PLS itself is established, not a claimed new learning algorithm. Do not add an
architecture search unless these comparisons identify a specific need.

Before pyspi, validate raw model-based estimates at full and reduced coverage.
An oracle full-state Jacobian is a target check, never a competing predictor.
If reduced observations contain too little usable target information, or all
methods solve the task trivially, revise the question based on that diagnosis
before spending on a benchmark. Do not revise controls after seeing z scores.

## Original raw feasibility result (fixed-sum prototype)

80 realizations were checked: 2 families × 5 a values × 8 independent seeds.
Linear q spans .0101–.9608; tanh q spans .00594–.94358. Past/future mean-q MAE
is 0 and .000387 respectively. States remained finite; maximum absolute sampled
state was 3.54 or less. Finite-difference tests verify both Jacobians and the
known linear ratio; channel relabelling preserves q.

This establishes well-defined, stable, varying labels. It does **not** establish
observability under subsampling, a representation advantage, or an important
real-world application. Results: `results/interaction_share_feasibility_260907.json`;
script: `scripts/check_interaction_share_feasibility.py`. Existing generators
were not modified; the two small maps are isolated in this prototype script.

## Raw reference audit and generator decision — 2026-09-07

Each parameterization has 80 masters (2 families x 5 settings x 8 replicates).
Nested random sensor views are M32/T1000 (diagnostic), M16/T1000 (source), and
M8/T500 (shift), all with the same endpoint and disjoint future-label window.
Twenty labelled source masters fit each one-dimensional monotone calibrator;
20 fresh masters per destination family evaluate it. These are exploratory
diagnostics, not the future 10/20/40-label learning curves or confidence claims.

Under fixed sum, calibrated own-channel AR(1) memory attains within-family,
source-observation MAE .0017/.0022 (linear/tanh). That is too easy a marginal
shortcut for the intended interaction-representation test. With independent gain,
the same errors become .1491/.1330. The target remains stable: linear q spans
.1–.9; tanh q spans .0670–.8655, with past/future MAE approximately 0/.00049.

Revised-generator joint shift, training at M16/T1000 and evaluating M8/T500:

| 20-label calibrated predictor | Linear to tanh MAE | Tanh to linear MAE | Equal-weight mean |
|---|---:|---:|---:|
| Source-label median | .2550 | .2582 | .2566 |
| Own-channel memory | .1457 | .1875 | .1666 |
| Linear model sensitivity | .1253 | .1060 | .1156 |
| Linear+tanh model sensitivity | .1763 | .1152 | .1458 |

This supports an informative, nontrivial pilot, not an advantage for z or proof
that cross-channel information is necessary. Rich univariate summaries might
still do well. Do not compare raw scores between parameterizations as a controlled
method improvement: parameter distributions and realization seeds changed.

Two limitations matter. First, subsampling loses direct edges: even the true
restricted Jacobian underestimates full sensitivity. Adjusting the cross/own
energy ratio by (N-1)/(M-1) corrects their relative inclusion probabilities under
uniform sensor sampling, but does not make the ratio unbiased. Second, squared
estimated coefficients contain estimation noise. Full-observation direct MAEs
are .1279/.1463 for linear and .2492/.2871 for linear+tanh models; the latter's
extra flexibility hurts this plug-in estimator. Hidden-variable bias cannot
explain full-observation errors. Partial-observation fitted maps need not equal
restricted true Jacobians, so the sampling correction alone is insufficient.

**Decision:** use the independent-gain generator for one fresh exploratory p90
comparison with the existing neural encoder and matched controls. No additional
model family or observation grid is justified now. Keep the negative
synchronization result and this generator revision visible. If z or m+z helps,
run an alignment null and fresh verification; if it does not, do not redefine q
to improve the ranking. Scientific importance of this exact target remains an
open issue beyond its usefulness as a controlled mechanism probe.

Reproduce from the repository root (each output directory must be new):

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m scripts.check_interaction_share_references --parameterization fixed_sum --output results/interaction_share_references_260907/fixed-sum-final
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m scripts.check_interaction_share_references --parameterization independent_gain --output results/interaction_share_references_260907/independent-gain-final
.venv/bin/python -m pytest -q tests/test_interaction_share.py tests/test_interaction_share_reference.py
```

Both directories contain per-master predictions, summaries and code hashes in
`results.json`, plus a readable `report.md`. Six tests pass; the fixed-sum targets
exactly reproduce all 80 original feasibility labels, all 240 views per run are
nested as specified, and adding the median control leaves existing predictions
unchanged (`results/interaction_share_references_260907/verification.json`).
