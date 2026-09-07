# Focused proposal: cross-channel local sensitivity

Status: target defined and raw-only feasibility checked; no pyspi extraction,
representation fitting or confirmation run has been launched for this proposal.

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
- a in {.10,.25,.40,.55,.70}; b=.80-a. This fixes an upper contraction bound
  while changing relative self/cross-channel sensitivity. It is not a search
  over systems selected for favourable z performance.
- Both maps are globally Lipschitz with bound .8 in Euclidean norm. Burn 300
  steps; observe 1,000 samples and evaluate q over the following 1,000 states.
- These are related model families, not a claim of broad cross-system diversity.
  Nonlinearity changes state-dependent derivatives while keeping units, time,
  topology, population and noise law fixed.

## Proposed bounded evaluation

Use fresh master seeds, disjoint from the raw feasibility sample. Per family and
a value: 12 source-training masters and 20 evaluation masters. Each recording
uses the same prediction endpoint for all observation views.

| | Source observation M16/T1000 | Reduced observation M8/T500 |
|---|---|---|
| Source family, fresh masters | Within-family prediction | Observation shift |
| Held-out family, fresh masters | Interaction-law shift | Joint shift (primary) |

Train on linear and test tanh, then reverse the direction. Do not select the more
favourable direction. Label budgets are 2/4/8 per a setting: 10/20/40 independent
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
and the existing raw temporal/channel encoder. The nonlinear reference must be
included; making a VAR model fail on tanh alone would be unpersuasive. First
implement a ridge one-step model with linear and tanh coordinate bases and its
analytic Jacobian; assess the bias from hidden channels explicitly.

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

## Raw feasibility result

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
