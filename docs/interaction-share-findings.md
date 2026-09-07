# Interaction-share representation findings

Status: fresh pilot, normalization/readout controls complete; alignment control
being prepared. These are exploratory findings, not a confirmed general-purpose
or neural-superiority result. [Setup and reproduction](interaction-share-execution.md).

## What has survived the controls

Joint-shift MAE, equal weight to linear→tanh and tanh→linear, M16/T1000→M8/T500:

| Representation / readout | 10 labels | 20 labels | 40 labels |
|---|---:|---:|---:|
| Rich raw marginals / PLS | .2255 | .2187 | .2647 |
| Standardized marginal shapes / PLS | .2117 | .2141 | .2065 |
| z / PCA-ridge | .2063 | .1659 | .1621 |
| z / PLS | .1769 | .1641 | .1600 |
| Standardized shapes + z / PLS | .1893 | .1674 | .1639 |
| Calibrated linear dynamics | .1485 | .1457 | .1379 |
| Calibrated linear+tanh dynamics | .1483 | .1451 | .1394 |
| Trained temporal/channel encoder | .2370 | .2213 | .2048 |
| Frozen random encoder / PCA-ridge | .2234 | .1814 | .1616 |
| Source median | .2561 | .2560 | .2560 |

The standardized-shape control retains each SPI's skewness, kurtosis and 19
quantiles after within-record centering/scaling of its edge distribution.
It removes mean and positive scale, matching Pearson z's affine invariance,
while retaining no cross-SPI edge correspondence. Constants/invalid SPIs remain
undefined. Across-record preprocessing and tuning still use source training only.
This is a post-result control, not a replacement for the original baseline.

At 40 labels, z/PLS improves over shape/PLS by .04650 MAE (conditional paired95%
interval −.06231 to −.03070). Shape+z/PLS improves over shape/PLS by .04259
(−.05347 to −.03214). Both transfer directions favour z over shapes. Normalization
therefore reduces part of the original marginal gap, but does not remove it.
This is evidence beyond these finite marginal descriptors, not proof of information
unrecoverable by every possible marginal-based learner.

PLS improves z at 10 labels by .02943 relative to PCA/ridge (−.03389 to −.02484).
At 20/40 labels the smaller differences include zero. This supports testing
target-aware compression; it does not prove that PCA's discarded directions
were intrinsically low dimensional or pure noise.

## Alternatives and limits that matter

- A correctly specified/approximately adequate dynamical reference remains better:
  z/PLS minus the linear reference is +.02841/+.01847/+.02200 across budgets;
  all three conditional intervals exclude zero. The specialist result does not
  invalidate a generic descriptor, but the pilot does not establish best-available
  label efficiency for this task.
- The trained encoder is not the strongest generic raw comparator. Frozen random
  features match z at 40 labels; z/PLS gains .04651 at 10 labels (−.06307 to
  −.02885), while the 20/40-label intervals include zero. This is a comparison of
  pipelines, not proof that learning an encoder damages its representation.
  Random convolutional features are established competitive time-series tools
  ([ROCKET](https://arxiv.org/abs/1910.13051)); our control uses the existing
  attention encoder, not ROCKET itself.
- A bounded nonlinear-head check does not close the marginal gap. With fixed
  training-only RBF bandwidth scaling, training-mean target centering and the
  same PCA-cap/regularization grid, m/shape/z MAEs at 40 labels are
  .2484/.2118/.1751. PLS remains better for z. This is one kernel rule, not an
  exhaustive search over nonlinear marginal learners.
- Concatenating unnormalized marginals with z hurts relative to z alone (.2218
  versus .1600 with PLS at 40 labels). Replacing them with standardized shapes
  largely removes that loss (.1639). This is consistent with unreliable magnitude
  cues under observation shift; it is not yet a causal attribution of the entire
  gap to a particular SPI or estimator.

## Why the nonlinear model did not clearly win

An oracle diagnostic on all 320 masters replaces the true time-varying Jacobian
by its temporal mean. For tanh, the target changes by only .00958 on average,
maximum .01591; variability of the Jacobian accounts for 2.41% of total squared
Jacobian energy on average (maximum 5.32%). Linear cases agree to roundoff.
Thus the *chosen averaged scalar target* is quite close to a constant-Jacobian
surrogate. This helps explain why a nonlinear reference is not decisively better;
it does not establish equality with the observed-data best linear predictor,
especially under hidden channels. The surrogate is not a competitor.

State-dependent Jacobian estimation has substantive scientific precedent
([Deyle et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4721089/)), but that does
not make our precise normalized ratio an established ecological observable.
This remains a controlled local-sensitivity probe in fixed units/discrete time.
Do not make the generator more nonlinear merely to force a linear baseline to lose.

## Next evidence and scale decision

Before enlarging the dataset, test cross-SPI dyad correspondence directly using
three independent per-SPI dyad permutations, preserving edge multisets,
reciprocity and validity. A shared permutation must leave z unchanged. Average
null errors as technical replications, not predictions as an ensemble or extra
independent systems. The perturbed MPI collections need not be realizable by
raw time series; this is a representation-level mechanism control.

A useful subsequent confirmation would use fresh source/evaluation populations
and a frozen, narrowed comparator set. Simply adding more versions of these two
related generators would not establish broader dynamics, applied importance or
novelty. Continuous parameter sampling would specifically test property learning
beyond the current five nominal-share settings; it would be a distinct prospective
extension, not an identical-distribution replication. Decide after the mechanism
control, without rewriting the target to improve the method's ranking.

The strongest current claim is a scoped benefit of SPI relationships over tested
individual-SPI distributions under limited labels and observation/family shifts.
It survives a matched affine-invariance control and a bounded nonlinear readout.
No unique-information claim, broad neural superiority, universal state recovery,
or high-venue prediction follows. [Cliff et al.](https://www.nature.com/articles/s43588-023-00519-x)
already studied empirical SPI relationships; novelty must lie in the record-level
representation's useful behaviour and the insight explaining it.

Evidence: `results/interaction_share_260908/{report,shape-report,followup-report}/`,
`linearization-diagnostic.json`, `neural-completion.json` and
`neural-cpu-verification.json`. Intervals condition on the fitted models, resample
independent evaluation masters within nominal-share strata and are unadjusted
for multiple comparisons. Five training subsets share a source pool; they are
not five independent training populations.
