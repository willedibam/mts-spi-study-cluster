# Interaction-share representation findings

Status: the pilot, continuous-parameter confirmation and new raw phase-control
experiment are complete. The phase mechanism is supported; an omitted spectral
comparator materially limits the representation-utility claim.
[Setup and reproduction](interaction-share-execution.md).

## Raw phase-control result: dependence sensitivity, not superior prediction

The new [phase experiment](interaction-share-phase-control.md) held M16/T1000 and
20 labels fixed. It reused 600 intact feature rows, computed 1,200 new surrogate
p90 records, and fitted every method separately in each arm. The 600 origins,
400 evaluation origins and five disjoint training cohorts per family are paired
across conditions. Labels refer to original systems, not surrogate Jacobians.
All seven predefined methods are complete: 210 fitted models, including 30 new
trained neural encoders and 30 frozen-random readouts.

Primary same-family MAE, averaged equally over the two families:

| Representation / readout | Intact | Common phase | Independent phase |
|---|---:|---:|---:|
| z / PLS | .1029 | .1033 | .1788 |
| Standardized SPI distribution shapes / PLS | .1451 | .1424 | .1967 |
| Calibrated linear reference | .0884 | .0883 | .1679 |
| Pooled autospectra / PLS | .0796 | .0796 | .0796 |
| Trained temporal/channel-attention encoder | .1869 | .1876 | .1861 |
| Frozen random encoder / PCA-ridge | .1403 | .1375 | .1413 |
| Source median | .1975 | .1975 | .1975 |

For z, independent-minus-common MAE is **+.07546**, conditional paired 95%
interval[+.06813,+.08287]. Common-minus-intact is +.00046[-.00133,+.00229]:
no resolved loss at this precision, not a formal equivalence claim. Cross-family
z MAEs are .1081/.1072/.1877; the independent-minus-common difference is
+.08046[+.07248,+.08821]. Independent phases worsen z in all ten family/cohort
comparisons for both evaluation scopes. These cohort counts share evaluation
origins and are descriptive, not independent Bernoulli trials.

**Positive mechanism evidence:** the tested z/PLS pipeline benefits from relative
cross-channel phases beyond the preserved channelwise spectra. The common-phase
control and within-arm refitting distinguish this from simply deploying a model
on a perturbed input distribution. This extends the earlier MPI-alignment control
to realizable raw recordings. It does not establish unique causal organization,
nonlinear information, or information unavailable to raw-data learners.
The effect is not unique to z: the linear reference and SPI distribution summaries
also worsen under independent phases, in all ten cohort comparisons.

**Negative utility evidence for this target:** intact z is worse than autospectra
by .02328[.01746,.02921] within family and .02548[.01942,.03181] across families.
The target can be inferred more efficiently here from information the surrogates
preserve. The trained encoder is weak in every arm; beating it is not evidence
of superiority to competent raw-data representations generally. Its small phase
contrasts should not be promoted as important improvements.

The Fourier controls preserve autospectra exactly; common phases preserve the
full cross-periodogram. Neither preserves exact time-domain histograms, and
independent phases do not imply complete channel independence. Their per-channel
surrogate laws conditional on spectra are matched. This is a pipeline-level
predictive effect, not a mutual-information measurement or an exact per-record
surrogate significance test. Mean finite-z fractions are .8990/.8992/.8844;
feature availability is part of the tested pipeline. CIs condition on fitted
models and resample paired evaluation origins; they are pointwise and unadjusted.

All 1,200 new MPI audits passed. Extraction used 20fd905/pyspi65317c9; final
cluster analysis used6095c47. Both production jobs exited0 (7:42/6:55 on576cores);
analysis jobs178437214/215 exited0 (4:21/4:09). Eight local statistical replays
preserve selected settings and all candidate scores; max prediction difference
4.55e-15. All30 neural checkpoints replay on10 evaluation records each with
maxCPU-MPSdifference4.47e-7;0/60 selected validation fits reached200epochs.
Protocol/data/feature schemas and hashes passed. The analysis-directory slug
mismatch was caught and corrected before analysis submission; no result was lost.

Evidence: `results/interaction_share_phase_260908/report/` (paired tables,
family/cohort results, `phase-controls.png` and SVG), `bank-verification.json`,
`neural-verification.json`, per-arm `local-gadi-replay.json`, and `cluster-state.json`.

**Decision:** keep this as a bounded mechanism-and-limitation section in the
existing paper. Do not expand this interaction-share task again merely for a
ranking or precision gain. The distinct phase experiment was worth running:
it supported the mechanism and exposed a material comparator omission. A strong
standalone representation-learning claim still needs a consequential target
where dependence organization adds useful information beyond competent temporal
and spectral controls. This experiment does not establish such an application;
it supplies no reason to change this generator solely to favour z.

## Current update: a strong omitted autospectral comparator

The phase-control study introduced a fixed spectrum-only map: 32 log band powers,
pooled over channels with five summaries, plus channel-mean summaries (165 features).
It has no cross-channel products and uses the existing PLS1/2/4 source-only grid.
At M16/T1000 with 20 labels, MAE is .0796 within family and .0826 across families.
Its inputs and predictions are identical across intact, common-phase and
independent-phase recordings. Thus this origin-system property can be inferred
well without relative cross-channel phases; coupling can leave a signature in
individual-channel spectra.

A bounded follow-up applied the identical map to both existing corpora, preserving
their original training cohorts, label budgets and joint M/T/family shifts.
This comparison is **retrospective**, motivated by the new raw-control result;
it is not a new prospective confirmation or a new simulator run.

| Dataset / method | 10 labels | 20 labels | 40 labels |
|---|---:|---:|---:|
| Pilot: autospectra/PLS | .1192 | .1141 | .1069 |
| Pilot: z/PLS | .1769 | .1641 | .1600 |
| Pilot: linear reference | .1485 | .1457 | .1379 |
| Continuous: autospectra/PLS | .1142 | .1138 | .1075 |
| Continuous: z/PLS | .1723 | .1519 | .1511 |
| Continuous: linear reference | .1454 | .1433 | .1427 |

The continuous-data 40-label autospectra-minus-z difference is -.04364,
conditional paired 95% interval[-.05511,-.03324]; versus linear it is -.03524
[-.04429,-.02603]. These are the original joint-shift evaluations, not the easier
fixed-M/T phase-study scores. All original report means were reproduced exactly.
The missing dedicated spectral baseline was a material comparator omission,
although it does not invalidate the original z-versus-SPI-distribution result.
A broad label-efficiency or interaction-information necessity claim is weaker,
and the linear reference is no longer the best tested standalone method.

Evidence: both datasets' `autospectrum-control/` and `autospectrum-report/`;
`scripts/run_interaction_share_autospectrum.py`. All sampling and tuning remain
source-only. The completed [phase-control protocol](interaction-share-phase-control.md)
asks what information z uses, with no generator change to obtain a favourable ranking.
The following sections preserve the earlier completed comparisons and their scope.

The spectral result has a direct explanation in the linear generator. With
`B = a I + b W` and the symmetric nearest-neighbour ring, mode decay factors are
`lambda_j = a + b cos(2 pi j/N)`. Up to the chosen spectral-density normalization,
each channel has autospectrum

\[
S_{ii}(\omega)=\frac{\sigma^2}{N}\sum_{j=0}^{N-1}
\frac{1}{1-2\lambda_j\cos\omega+\lambda_j^2}.
\]

Thus b changes individual-channel spectral shape, while the target is
`q = (b²/2)/(a²+b²/2)`. This is a derivation for the stationary linear ring with
independent equal-variance innovations, not a proof that the finite spectral
summary uniquely identifies q or an exact formula for the tanh system. It
explains why removing relative phases need not remove target information.
This information is a legitimate consequence of coupling, not label leakage.

## Prospective confirmation and present decision

The fresh run uses 800 independent masters, continuous nominal interaction shares
and five disjoint training cohorts per family, with nested 10/20/40-label budgets.
Physical N32, linear/tanh maps, M16/T1000 training and M8/T500 joint shift are
unchanged. All 1,200 MPI records passed their audits. This is a prospective
extension to a changed parameter distribution, not an identical replication;
absolute MAEs should not be compared across studies as a learning improvement.

| Representation / readout | 10 labels | 20 labels | 40 labels |
|---|---:|---:|---:|
| Rich raw SPI marginals / PLS | .2208 | .2451 | .2453 |
| Standardized marginal shapes / PLS | .1885 | .1869 | .1853 |
| z / PCA-ridge | .1769 | .1555 | .1525 |
| z / PLS | .1723 | .1519 | .1511 |
| Standardized shapes + z / PLS | .1816 | .1602 | .1569 |
| Frozen random encoder / PCA-ridge | .2170 | .1881 | .1771 |
| Frozen random encoder / PLS (supplementary) | .2176 | .1916 | .1845 |
| Calibrated linear dynamics | .1454 | .1433 | .1427 |
| Calibrated linear+tanh dynamics | .1466 | .1444 | .1436 |
| Channel-wise memory | .1978 | .1828 | .1819 |
| Source median | .2084 | .2047 | .2057 |

**Primary hypothesis supported within this setup.** z/PLS versus shapes/PLS
improves MAE by .01619/.03498/.03419. At 40 labels this is an 18.5% relative
reduction; the conditional paired 95% interval for the difference is
[-.04377,-.02443]. Adding z to shapes improves by .00692/.02671/.02840,
with all three conditional intervals below zero. Both comparisons favour z
in all 10 source-family/cohort cases at 20/40 labels; at 10 labels the counts are
8/10 for z and 7/10 for concatenation. The smallest-budget effect is less stable.
These are descriptive cohort counts on shared evaluation populations, not
independent Bernoulli trials. Intervals remain conditional on fitted models.

**Raw-reference comparison remains mixed.** z beats both random-feature readouts
on average at all budgets with paired intervals below zero; versus random/PCA
it wins 9/10,9/10,7/10 cohort cases. It does not beat the linear reference overall.
At 40 labels z-minus-linear is +.00840, interval[-.00263,.01912]; the 20-label
interval also spans zero, while linear is clearly better at 10. These are not
equivalence tests. Direction matters: at 40 labels z/linear MAEs are
.1417/.1553 for linear-to-tanh and .1605/.1301 for tanh-to-linear. The former
apparent z advantage is unresolved (difference interval[-.02977,.00268]);
the reverse-direction linear advantage is resolved. Do not promote the favourable
direction alone or call this a demonstrated win over dynamical estimation.

**The large PLS gain did not replicate in magnitude.** PLS improves z over PCA
by only .00459/.00362/.00136 here. The first two conditional intervals exclude
zero, but the 40-label interval does not. PCA is a strong compression baseline;
the evidence does not support a blanket account of PCA retaining destructive
noise or of supervised compression being essential. Both changed parameter
sampling and new training populations could affect the difference; the experiment
does not isolate which caused it.

The strongest supported statement is: **aligned relationships between SPIs provide
predictive structure for interaction-share inference under partial observation
that these individual-SPI distribution descriptors expose less efficiently.**
The pilot's matched-normalization, bounded-RBF and dyad-alignment controls support
that interpretation. They do not establish inaccessible raw information, causal
mechanism identification, or that all possible marginal-based learners fail.
The target mostly describes a stationary system's interaction balance, not
within-record state tracking; fixed state units and time step remain essential.

**Stop this experiment here.** The larger run answered two real uncertainty
questions, so it was justified. Another expansion of the same grid would mainly
add precision. There is useful evidence for the existing paper, but a standalone
high-impact ML/Nature Computational Science claim is not established. The missing
piece is scientific importance or a more general explanatory result, not simply
more systems or a better leaderboard score. A consequential scientific application
or a principled result about when these relationships help could change that
assessment; neither should be assumed from this controlled probe. Earlier negative
Kuramoto results remain part of the evidence.

No result-invalidating implementation or leakage defect was identified. Twenty
relevant tests pass; all data/feature hashes and cohort separation checks pass.
Ten local replays spanning all five statistical pipelines and both source families
reproduce Gadi's selected settings, all candidate scores and predictions
(maximum prediction difference 8.22e-15). The meaningful comparator correction
was to give random features the same PLS option; its supplementary status is
explicit, and all 20 re-extracted feature banks are exactly unchanged.

Evidence: `results/interaction_share_confirmation_260908/report/`,
`cohort-summary.json`, `direction-reference-contrasts.json`, `local-gadi-replay.json`
and `learning-curves.{png,svg}`. Shaded plot intervals resample evaluation masters;
right-panel points show independently sampled training cohorts. The trained
end-to-end encoder was tested in the pilot below, not rerun in this confirmation.

## Retrospective complementarity check after confirmation

The next bounded question was whether z adds to a stronger dynamical estimator,
even when it performs worse alone. No additional data or models were fitted.
We averaged predictions 50/50 on exactly the same labelled cohorts, preserving
all budgets, directions and observation cells. Both datasets had already been
evaluated, so this is retrospective, including results on the dataset originally
generated for confirmation. Neither weights nor budgets were optimized.

Results on the continuous-parameter dataset:

| Predictor or equal-weight prediction average | 10 labels | 20 labels | 40 labels |
|---|---:|---:|---:|
| Linear reference alone | .1454 | .1433 | .1427 |
| Linear + z/PLS | .1333 | .1242 | .1232 |
| Linear + normalized SPI distributions/PLS | .1331 | .1345 | .1277 |
| Linear + random encoder/PCA-ridge | .1565 | .1352 | .1228 |
| Linear + source median | .1553 | .1537 | .1535 |
| Linear + nonlinear reference | .1459 | .1435 | .1430 |

The linear+z average improves over each component at every budget in both
datasets, with conditional paired intervals excluding zero. At 40 labels in the
continuous dataset, its difference from linear alone is -.01954
[-.02603,-.01322], with improvement in nine of ten source-family/cohort cases.
The equal-weight source-median control does not explain this gain; it does not
rule out every alternative shrinkage or calibration strategy.

**Complementarity is not consistently specific to z.** In the original pilot,
linear+z versus linear+normalized-distributions differs by only
-.00112/-.00094/-.00245 across budgets, with all intervals including zero. In the
continuous dataset, the differences are +.00021/-.01035/-.00450; only the
20-label interval excludes zero ([-.01499,-.00570]). Linear+random matches
linear+z at 40 labels in both datasets. Thus standalone representation rankings
do not determine which representation adds the most to a dynamical reference.
The narrower conclusion is practical complementarity of statistical descriptors;
a consistent z-specific fusion advantage is not established.

This does not justify a third large run of the same generators. Selecting the
one favourable budget for another enlarged test would require a stronger
scientific rationale. Preserve the positive standalone z finding and this limit
on its incremental value. No new physics, hyperparameter search or cluster jobs
were added in this follow-up.

Reproduce with `scripts/check_interaction_share_complementarity.py` (execution
commit `6b3fe2d`) and the same study's `--config`, `--data`, `--results` and a new
`--output` directory. Results are under each study's `complementarity-controls/`;
the initial control set remains under `complementarity/`. The audit checks source
hashes, labels, row ordering and identical training cohorts. Its six standalone
component curves reproduce the existing report across all eight evaluation cells
and the primary aggregate (54 checks per study; maximum difference 1.11e-16).

## Pilot and mechanism controls

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
| Frozen random encoder / PLS (supplementary) | .2080 | .1889 | .1586 |
| Source median | .2561 | .2560 | .2560 |

The standardized-shape control retains each SPI's skewness, kurtosis and 19
quantiles after within-record centering/scaling of its edge distribution.
It removes mean and positive scale, matching Pearson z's affine invariance,
while retaining no cross-SPI edge correspondence. Constants/invalid SPIs remain
undefined. Across-record preprocessing and tuning still use source training only.
This is a post-result control, not a replacement for the original baseline.

At 40 labels, z/PLS improves over shape/PLS by .04650 MAE (conditional paired 95%
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
  A supplementary matched-PLS check gives random-feature MAEs
  .2080/.1889/.1586; z/PLS improves at 10/20 labels with conditional intervals
  excluding zero, but not at 40. With matched PCA/ridge, z's clear advantage is
  at 10 labels. Exact equality of all re-extracted frozen feature banks isolates
  the readout change. Do not select the worse random readout to inflate a gain.
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

## Pilot mechanism evidence and original scale decision

Three independent per-SPI dyad permutations preserve edge multisets, reciprocity
and validity across all 520 records. A shared permutation preserves z to 3.73e-9.
Independent permutations raise z/PLS MAE to .2535 at all three budgets, near the
.256 median reference, versus aligned .1769/.1641/.1600. This supports the
predictive role of cross-SPI correspondence beyond marginal distributions.
It does not prove causal organization recovery: the perturbed MPI collections
need not be realizable by raw time series. Null errors are averaged over technical
seeds, not predictions as an ensemble or extra independent systems. Gadi job
178391492 exited successfully at code 4932773.

This justified the completed [prospective confirmation](interaction-share-confirmation.md)
with fresh continuous parameter values and disjoint training cohorts, keeping
the same target, map families, observation cells and frozen comparator set.
It addresses discrete settings and shared-pool uncertainty rather than broadening
the benchmark to obtain a favourable result. It is a distinct continuous-parameter
extension, not an identical-distribution replication.

The strongest current claim is a scoped benefit of SPI relationships over tested
individual-SPI distributions under limited labels and observation/family shifts.
It survives a matched affine-invariance control and a bounded nonlinear readout.
No unique-information claim, broad neural superiority, universal state recovery,
or high-venue prediction follows. [Cliff et al.](https://www.nature.com/articles/s43588-023-00519-x)
already studied empirical SPI relationships; novelty must lie in the record-level
representation's useful behaviour and the insight explaining it.

Evidence: `results/interaction_share_260908/{report,shape-report,followup-report,complete-report}/`,
`linearization-diagnostic.json`, `neural-completion.json` and
`neural-cpu-verification.json`. Intervals condition on the fitted models, resample
independent evaluation masters within nominal-share strata and are unadjusted
for multiple comparisons. Five training subsets share a source pool; they are
not five independent training populations.

## Related-work boundary, checked 2026-09-08

The contribution must be more specific than combining statistical features.
[Bryant et al. (2024)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012692)
already compare local dynamical features, per-SPI connectivity and combinations
for neuropsychiatric classification. Their spatially indexed representations are
different from our per-record, spatially pooled cross-SPI representation.
[Wang et al. (2025)](https://www.nature.com/articles/s42003-025-09165-7)
correlate univariate feature profiles *between brain regions*: another feature
similarity construction, with a different unit of comparison from SPIs across
dyads. Neither paper validates our proposed transfer claim.

[Nguyen et al. (2025)](https://journals.aps.org/prresearch/abstract/10.1103/qnx2-yp4c)
([accessible preprint](https://arxiv.org/abs/2404.05929)) study dependence mediated
by temporal features over long timescales. Their feature-based inference gains
under limited/noisy data illustrate a meaningful constrained-efficiency question;
the construction and target differ from z. These are precedents for scientifically
useful representations, not evidence that our present controlled target has
equivalent practical significance. This targeted check is not an exhaustive
novelty search. No biological experiment is being added on the basis of it.
