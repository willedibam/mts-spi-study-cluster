# Representation/transfer programme: complete evidence map

Checked against the saved study reports on 11 September 2026. This concerns the
whole representation/transfer programme, including the pre-NeuroTycho studies.
The corpus/embedding/order-parameter workstreams are separate evidence and are
not implicitly included as completed support here.

## The progression and what it changed

1. **Broad 14-class screen → attribution.** z/logistic achieved91.68% versus86.84%
   for rich SPI marginals at8labels/class under M/T changes. However, two Gaussian
   VAR classes accounted for76.4% of the net gain. A three-VAR-class diagnostic
   gave intactz68.9%, richm47.5%, disruptedz34.9%, and direct two-feature VAR
   estimates100%. This established use of correspondence, but not a general
   nonlinear-learning advantage. Physical size changes here differ from later
   fixed-population sensor-subset changes.
   [Evidence](representation-stage-a-attribution.md).
2. **Future Kuramoto coherence.** FixedN32,192masters,16/32/64labels and six M/T
   views. z did not beat simpler marginal/phase-based controls; a PCA-cap sweep
   did not rescue the intended advantage. The future label was highly persistent
   (hidden past/future coherence MAE.00587), so this was not a difficult forecasting
   success. Raw neural/frozen-feature controls were also inferior to the simple
   physical observables. [Evidence](representation-state-pilot.md).
3. **Interaction-share inference across linear/tanh systems.** The fresh
   continuous-parameter confirmation used800independentmasters and10/20/40label
   budgets. At40labels under the joint family/observation shift, zPLS MAE.1511
   versus normalized SPI distributions.1853; PCA.1525. However, the subsequently
   added autospectral control achieved.1075. Realizable Fourier phase controls
   showed z used relative cross-channel phase, while autospectra predicted the
   target better without that information. The omitted spectral comparator was
   a material correction; dependence sensitivity did not establish necessity.
   [Evidence](interaction-share-findings.md).
4. **Covariance-modulation negative.** A stationary Gaussian-mixture construction
   held individual-channel laws and population covariance/cross-spectra fixed
   while changing conditional dependence.440masters/640views. At40labels reduced
   zPCA MAE.2258 and zPLS.2375 were near the median baseline.2265; corrected
   pointwise neural.1652 and the calibrated moment reference.1261 were better.
   A reproduced clipping/stopping issue was repaired, then the pointwise model
   received one bounded extension: the negative z finding survived competent
   comparison. An exact R-versus-fourth-cumulant example shows that scaling a
   target-bearing MPI can leave its Pearson agreement unchanged; this is not a
   proof about all289SPIs. [Evidence](covariance-modulation-findings.md).
5. **Co-organization: the central positive experiment.** The target is whether
   phase-coupling and amplitude-envelope groupings coincide or cross. It is defined
   from the generating organization, not from z. The fresh frozen pilot used
   320independentrecordings: three disjoint40labelsourcepools and200testmasters,
   with10/20/40TOTALlabelbudgets and M16T1000→M8T500. Single-channel process laws
   are matched between classes; all finite-sample spectra and SPI distributions
   are not identical. Results below establish a useful scoped advantage.
   [Evidence](oscillatory-coorganization-findings.md).
6. **Mechanism and nonlinear-access controls.** Independent per-SPI dyad shuffles
   preserve each statistic's complete edge multiset, reciprocal structure and
   validity while disrupting correspondence; matched retraining drives z toward
   chance. Common permutations preserve z. Fixed marginal PCA/RBF readouts do
   not close the gap; removing envelope SPIs weakens z, and a36coordinate PEC×PLV
   panel retains substantial signal. These are post-result attribution tests,
   not independently confirmed new predictors or realizable waveform surrogates.
   [Evidence](oscillatory-coorganization-mechanism.md).
7. **Learned aggregation and faster-dynamics transfer.** A competent Deep-Sets-style
   model sees the normalized SPI edge vectors that precede Pearson compression,
   plus validity. It catches up at20–40labels and can outperform z after shifts.
   Separate200-master faster-regime tests preserve source models and vary carrier,
   phase-diffusion and envelope timescales. This is prospective transfer within
   the generator family; faster dynamics are not established to be harder.
   [Evidence](oscillatory-transfer-findings.md).
8. **Transfer to a different phase-generating mechanism.** New200-master test:
   direct Kuramoto coupling replaces shared phase drivers. Functional-property,
   coupling-removal and numerical checks precede the frozen test; amplitude laws
   and matched group diffusion limit its independence. All99source model references
   stay frozen. Performance and ranking retain a large advantage over SPI
   distributions. [Evidence](oscillatory-mechanism-transfer-findings.md).
9. **Separate attribution for the learned model.** At40labels, intact pooling
   achieves98.8% reduced BA; three independent-correspondence nulls achieve49.2%,
   51.2%,50.3%, with AUROCnear.5. Refitting the null source models avoids mistaking
   test-only distribution shift for mechanistic evidence. This supports use of
   correspondence by both fixed and learned aggregation, not Pearson optimality
   or absence of all signal from marginal information.
   [Evidence](oscillatory-pooling-attribution.md).
10. **External pilot and conventional-library follow-up.** NeuroTycho demonstrates
    useful state transfer and z wins over several declared baselines; the enriched
    raw encoder remains best. Most methods rank states nearly perfectly. These
    tests add external feasibility and decision-score evidence; they do not
    retrospectively reduce the synthetic results to threshold effects.
    [Pilot](neurotycho-transfer-findings.md),
    [follow-up](neurotycho-library-followup-findings.md).

## Quantitative core before real data

Original co-organization test, reduced observations, mean over three source cohorts:

| Method | 10 total labels | 20 | 40 |
|---|---:|---:|---:|
| z + PCA/ridge |83.0%|96.0%|98.5%|
| z + PLS |88.8%|96.0%|98.0%|
| Rich SPI marginals |64.3%|74.0%|80.8%|
| Normalized SPI shapes |65.3%|70.0%|73.7%|
| Learned SPI pooling |65.8%|96.5%|98.8%|
| Direct phase/envelope specialist |96.3%|96.3%|98.0%|

At40labels, reduced zPLS AUROC.9986 versus richm.9215: a ranking gap, not just
threshold transfer. The zPLS–richm BA gap is17.17points, conditional pointwise95%CI
[13.50,21.17]. The small raw temporal/channel and pair encoders remain near chance;
their failure alone is not the reason this is a positive representation result.

Frozen-model tests at40labels, reduced observation:

| Target regime | zPCA BA | zPLS BA | Pooling BA | Rich marginals BA |
|---|---:|---:|---:|---:|
| Original regime |98.5%|98.0%|98.8%|80.8%|
| Faster dynamics |99.0%|99.2%|99.5%|79.3%|
| Direct phase coupling |89.0%|88.8%|93.8%|67.5%|

The direct-mechanism reduced AUROCs are zPCA.9983, zPLS.9978, pooling.9981,
richm.8004. At full size/40labels, z andpooling score100%BA, richm71.8%.
Thus there is substantive discrimination evidence beyond the real-data threshold
story. A post-hoc known50%prevalence/top-half diagnostic raises reduced PCA BA
to98.3%; it uses extra target-batch/prior information and is not the primary result.

## Baseline coverage must not be overstated

The synthetic programme includes rich SPI distributions, normalized shapes,
fusions, selected graph summaries in some stages, validity controls, bounded
nonlinear readouts, raw moments/spectra/dynamical estimates, fixed random encoders,
small trained raw encoders, task-informed specialists and learned SPI pooling.
They were not all run on every dataset; reports specify exact coverage.

**In this supervised transfer programme, MiniRocket, tsfresh and catch22 were
only run on NeuroTycho.** The wider heterogeneous-corpus work separately includes
an exploratory catch22 representation comparison across its mixed real/synthetic
collection. That is not a supervised co-organization or mechanism-transfer test.
The enriched neural control also belongs to NeuroTycho. None of these results
can be presented as comparisons on the central synthetic co-organization target.
A completed InceptionTime comparison now covers full M=16 synthetic targets; no broad pretrained/foundation-model comparison is established.

Two selected10label pooling folds hit the600epoch ceiling and source-cohort
variability is large. This limits an optimization-independent sample-efficiency
claim. At20–40labels pooling is competitive or stronger; its success strengthens
catalogue-based relational representation, but limits a Pearson-specific claim.
The direct specialist is better at10labels and competitive thereafter. It uses
observed data, not latent answers, and must remain visible.

## Combined assessment and efficient next decision

There is already a coherent, substantive pre-real-data result: explicit
cross-statistic relationships make a selected interaction-organization target
accessible to simple predictors with few labels, survive specified changes in
observation and mechanism, and are supported by correspondence interventions.
The negative tasks delineate selectivity rather than making the positive task
invalid. The largest advantages are not the few-point real-data wins.

The remaining generality question is how much this depends on a constructed
phase/envelope target and the tested learners. Fresh cohorts and frozen targets
reduce finite-sample overfitting but do not remove research-level selection of a
favourable task after earlier failures. Independent empirical phase/amplitude
patterns provide scientific motivation ([Siems and Siegel2020](https://pubmed.ncbi.nlm.nih.gov/31935522/)),
not validation of this synthetic binary target as a consequential application.

A strong synthetic methods contribution does **not** logically require a real-data
win. The earlier assessment overweighted the application as the only route to a
stronger paper. The completed bounded convolutional comparison tests the central learning claim more directly than another loosely related application. Full-size source/target cells permit
an unmodified comparator; changing M requires a declared compatible adaptation.
The user has now authorized InceptionTime on both this synthetic core and
NeuroTycho: [bounded follow-up](inceptiontime-followup.md). It reuses all existing
data and source splits, evaluates full M=16 only, and reports five-member
ensembles alongside members. The comparison is now independently verified: all 55 source models, 174 metric rows and 200 sampled CPU replays pass. At 40 labels, full-size InceptionTime ensemble BA is 50.33% in each synthetic regime versus 100% for z-PCA/PLS; ensemble AUROCs remain near chance. On matched NeuroTycho data its ensemble reaches 95.31% BA, exceeding z-PCA 92.97% and z-PLS 94.53%. Four selected 10-label source folds hit the epoch limit; none do at 20/40 labels. See the linked follow-up for individual members, source generalization, hashes and limitations. Conventional feature libraries remain untested on synthetics.

Current status: strong scoped empirical/mechanistic evidence for a statistical
representation prior; conditional low-label and transfer advantages; incomplete
case for broad neural-learning superiority or major application impact. A focused
methods paper has a defensible backbone. A separate high-impact learning paper
must articulate an independent general lesson beyond the existence of a task
aligned with the chosen descriptor; venue and paper-split decisions remain open.

The [whole-programme manuscript blueprint](spi-spi-manuscript-blueprint.md)
connects this learning programme to the construction, cross-dimension proof,
heterogeneous corpus and physical order-coordinate studies while retaining
their separate evidential roles. It includes a provisional learning-module
abstract and figure plan; it does not decide the eventual paper split.
