# Does origin-state inference require cross-channel phase relationships?

This is a new mechanism experiment, not another replication or a search for a
generator that favours z. Protocol: `configs/analysis/interaction-share-phase-260908.yaml`.
It is fixed before generating or evaluating surrogate recordings.

The preceding MPI-dyad shuffle established that correspondence among SPI outputs
matters. It did not establish whether useful information requires cross-channel
temporal relationships in the raw recordings. Interaction strength can also alter
each channel's spectrum, permitting prediction from individual-channel dynamics.

## Three matched conditions

Use the existing continuous-parameter origin systems, observed at M16/T1000.
Keep their original future interaction-share labels as **origin-system labels**;
do not assign those labels to a hypothetical surrogate's physical Jacobian.

1. Intact recordings, reusing their verified p90 feature bank.
2. Common phase: multiply each Fourier frequency by the same random unit phase
   across all channels. This preserves the complete cross-periodogram and each
   autospectrum exactly, hence circular second-order correlations.
3. Independent phase: use independent random unit phases per frequency/channel.
   This preserves each autospectrum, while disrupting relative channel phases.

DC is unchanged; the Nyquist multiplier is a real random sign for even lengths.
Shared Fourier phases are established multivariate-surrogate methodology
([Prichard and Theiler, 1994](https://arxiv.org/abs/comp-gas/9405002)). The need to
state exactly what is preserved is central to surrogate analysis
([Schreiber and Schmitz](https://arxiv.org/abs/chao-dyn/9909037)).

Important limits: individual time-domain histograms are not preserved exactly.
Common and independent arms have the same marginal surrogate law for a channel
conditional on its spectrum, rather than identical realized histograms. Independent
phases do not imply complete channel independence: spectral amplitudes can remain
dependent, and cross-periodogram magnitudes are unchanged. Full-record circular
correlations are not identical to every windowed estimator or nonperiodic VAR fit.
Differences between common phase and intact data cannot alone diagnose nonlinearity.

## Fixed sample and models

One label budget: 20, including two-fold tuning. Five disjoint cohorts per source
family use the previously defined 20-label subsets. Retain all 200 evaluation
origins per family. Thus each arm has 200 training and 400 evaluation recordings,
with the same origin IDs, targets and cohort memberships. One surrogate draw per
origin/arm; origins are paired across arms and never counted as extra independent
systems. All 600 intact feature rows are reused; 1,200 surrogate p90 computations
are new. No M/T sweep or new dynamical family is introduced.

Refit each model within each arm. The main contrast is independent-minus-common
MAE on same-family evaluation, averaged equally over the two families. Retain
both family-transfer directions as secondary outcomes. This avoids interpreting
mere deployment shift of an intact-trained model as information loss.

Models: z/PLS, normalized per-SPI distribution shapes/PLS, a calibrated linear
reference, a source median, pooled autospectra/PLS, the existing trained temporal
CNN/channel-attention encoder, and its frozen-random/PCA-ridge control. All grids,
label counting, preprocessing and neural optimization settings are retained.
No new architecture or hyperparameter search is introduced.

The spectral comparator pools log band powers from 32 fixed frequency bands
using channel mean, SD and 10th/50th/90th quantiles, plus the same summaries of
channel means (165 features). It uses no cross-channel products. Recomputed
features must match across phase arms numerically; identical copies are stored
after verification. Its fitted predictions should therefore match as well.

## Interpretation and execution gates

- If common-phase prediction is retained but independent-phase prediction loses
  accuracy, relative channel phases supply useful information for that learner.
- If prediction remains strong after independent phases, or the spectral baseline
  performs comparably, the original task contains usable autospectral information.
  That weakens an interpretation requiring cross-channel interaction information;
  it does not imply the origin systems lack coupling.
- If both surrogate arms degrade similarly, do not attribute the loss specifically
  to cross-channel phase relationships. Histogram/higher-order/boundary changes
  and estimator response remain possible explanations.
- Report each family and cohort, all predefined methods, and conditional paired
  origin-level intervals. This is a controlled surrogate comparison, not an exact
  per-record surrogate significance test or fresh confirmation of the old score.

Verify Fourier contracts and shared label/cohort identities, then run two-record
smokes and representative 48-record node gates for both surrogate arms. Size
production from those measurements. Fit the neural models locally while Gadi
computes p90. The user authorizes runs up to 12 hours when useful; available time
does not dictate corpus size. Preserve the earlier positive and negative results.

## Bounded follow-up declared after the raw-control results

Before surrogate SPI results, the completed spectrum-only control achieved
same-family MAE .0796 and family-transfer MAE .0826 at fixed M16/T1000. The latter
must not be compared directly with earlier joint M/T-shift scores. Consequently,
apply this exact 165-feature map and existing PLS1/2/4 grid to both already-built
interaction-share corpora, including their original M8/T500 evaluations and
10/20/40-label cohorts. This checks whether an omitted simple comparator changes
the earlier observation-transfer interpretation. It needs no new simulation,
pyspi work, architecture or feature search. Keep it explicitly retrospective,
separate from the frozen primary comparisons; preserve all earlier results.
Runner: `scripts/run_interaction_share_autospectrum.py`.
