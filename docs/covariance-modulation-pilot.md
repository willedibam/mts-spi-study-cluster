# Covariance modulation at fixed population second-order statistics

Prospectively specified exploratory pilot, 2026-09-09. Independent of the older
illustrative notebooks. Question: does SPI correspondence make a change in joint
dependence easier to learn when individual-channel behaviour and average
connectivity do not identify the target at the population level?

## Motivation and exact scope

Average connectivity can miss changes in joint dependence. Distinguishing such
changes from ordinary covariance-driven fluctuations is an existing methodological
problem, for example in dynamic functional connectivity. It is not sufficient to
reject a Gaussian null and conclude temporal organization or nonstationarity.
[Novelli and Razi 2022](https://www.nature.com/articles/s41467-022-29775-7) demonstrate
how static Gaussian models reproduce many edge-centric observations;
[Miller et al. 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6135983/) distinguish
Gaussianity, stationarity and second-order null assumptions. This pilot is a
controlled statistical experiment, not a model validated against brain data.

## Generator and target

N=32 channels, two equally sized latent groups, randomly relabelled per record.
At each time, g,h,j and all channel noises are independent standard Gaussians.
S(t) is a balanced stationary two-state Markov chain independent of these factors.

    H_A(t) = h(t)
    H_B(t) = alpha*S(t)*h(t) + sqrt(1-alpha^2)*j(t)
    X_i(t) = sqrt(c)*g(t) + sqrt(d_i)*H_group(i)(t)
             + sqrt(1-c-d_i)*epsilon_i(t)

The target alpha is continuous in [.05,.95]. Independently draw c in [.05,.15],
base d in [.25,.60] and channel multipliers in [.8,1.2]. These ranges ensure
positive innovation variance. No true group, c, d, S or alpha enters a predictor.

Each channel has the *entire iid N(0,1) time-series law* for every alpha. Within
group covariance is c+sqrt(d_i*d_j), between-group unconditional covariance is c,
and all nonzero cross-lag covariances are zero. Thus population marginal laws,
covariance and cross-spectra do not depend on alpha. Conditional between-group
covariance is c+alpha*S(t)*sqrt(d_i*d_j). Its fourth cross-cumulant at one time is
2*alpha^2*d_i*d_j, which supplies an explicit strong raw-data reference.

The unconditional process is stationary, including the persistent-state case.
Do not call this a test of nonstationarity or causal mechanism identification.
Finite-record covariances and pooled marginal descriptors can still be informative
through their sampling distributions. The matching is a population statement,
not exact equality of every realized covariance or every pooled statistic.

State flip probability .05 gives persistent episodes (mean duration20 samples).
Probability .5 is the iid-state control, with identical instantaneous joint law.
Success common to both can concern non-Gaussian joint dependence without implying
recovery of state timing. Cross-control transfer is not broad generator transfer.

## Fixed pilot design

- Config: `configs/analysis/covariance-modulation-260909.yaml`; master seed26090961.
- 440 independent masters:120 training+100 evaluation per state process;640views.
- Three disjoint40-label training cohorts per process, five alpha strata.
- Nested10/20/40-label budgets, including two-fold tuning; no pretraining.
- Train M16/T1000; evaluate fresh masters at M16/T1000 and paired M8/T500.
- Primary: same-process reduced-observation MAE, averaged across both processes.
  Full-observation and cross-process results are mandatory secondary diagnostics.
- Full p90 catalogue,289 SPIs, ordered off-diagonal z, training-only preprocessing.
- Compare rich per-SPI marginals, normalized distribution shapes, z/PCA, z/PLS,
  shape+z/PLS, graph summaries, validity and the source median.
- Raw references: marginal moments/memory;165-feature autospectra; covariance
  distribution; fourth-cumulant distributions;25/100-sample windowed covariance
  variance; their combined features; an observed-covariance cluster-based moment
  estimate calibrated using source labels. True groups/parameters are unavailable.
- Neural: existing aligned-channel encoder and a smaller relational pair encoder
  operating on aligned raw channel pairs before pooling. The latter is inspired
  by [relation networks](https://arxiv.org/abs/1706.01427), not a claimed reproduction
  of a published time-series benchmark. Both are trained and evaluated on the
  same independent origins; describe them as tested architectures, not all neural
  methods. Random-feature controls can reuse their untrained architecture.

## Gates and follow-up decisions

Before p90: check positive conditional covariance, population matching, the
fourth-moment formula, exact replay, channel-permutation invariance and raw-target
estimability. Freeze generator/protocol before generating the pilot. Run a2-record
smoke and representative48-record node gate before the production farm.

A useful primary result needs improved prediction over the matched SPI summaries,
with learning curves and cohort consistency; neural comparison alone is not
sufficient. Compare strong raw references immediately. If z adds no useful
advantage beyond these controls, increasing sample count merely for precision is
not justified. If useful correspondence and practical complementarity are both
credible, freeze a fresh confirmation with more independent cohorts, rather than
changing the task to maximize the gap. If results differ across persistent/iid
states, investigate temporal ordering using paired common time permutations;
this follow-up is conditional, not an automatic extra grid.

P90 is affordable. Scientific informativeness, rather than available12-hour time
or maximum parallelism, determines whether another run is justified. Keep all
exploratory results, and leave one-versus-two-paper structure undecided.
