# SPI–SPI: current scientific claim and significance

Updated 11 September 2026 after the verified NeuroTycho pilot and conventional
library follow-up. This replaces the earlier prospective application plan.

The [complete experimental evidence map](spi-representation-evidence-map.md)
reconstructs the earlier screens, failures, synthetic confirmations, baselines
and transfer tests. The synthetic programme is the primary mechanistic evidence;
the real-data pilot is a separate external check.

## Central claim

**Relationships among dependence measures provide a useful fixed representation
of multivariate dynamics. In controlled systems, this representation captures
interaction co-organization that the tested individual-statistic summaries expose
less successfully, and supports source-only transfer under specified dynamical
and observation changes. A small ECoG pilot establishes competitive state-transfer
performance, with clear limits to superiority and mechanism attribution.**

This is a scoped statistical-representation and domain-generalization claim.
Applying source-trained predictors without target fitting has established domain
[generalization precedent](https://proceedings.mlr.press/v28/muandet13.html);
calling it transfer learning is not itself a contribution.

## End-to-end content

Each multivariate recording produces 289 p90 MPI matrices. Both directions of all
off-diagonal links enter aligned edge profiles; their pairwise Pearson correlations
produce 41,616 named z coordinates. The feature map is fixed. PCA learns a
projection, PLS learns a supervised projection, and the readout learns prediction.
The alternative learned SPI-pooling model learns how to aggregate edge-wise
statistical vectors. No experiment establishes cross-task reuse of one learned
embedding or information unavailable in principle from the raw recordings.

The valid SPI correlation matrix is a Gram matrix of normalized edge profiles:
its entries compare angles between how statistics vary across links. This is a
useful geometry and an interpretable statistical prior, not a novel matrix-
geometry learning algorithm. The construction removes individual-SPI offsets and
positive scales as well as edge incidence/spatial identity. Its common coordinate
system handles changing M,T syntactically; empirical robustness to those changes
requires testing and is not guaranteed by the construction.

The motivation is to describe *how different dependence modes coincide across
links*, rather than requiring a fixed spatial coordinate system. This can matter
when the organization of dependence is informative and its absolute scale or
measurement layout varies. Neither nuisance invariance in general nor unique
identification of a causal interaction mechanism has been established.

## What each component contributes

| Evidence | Supported contribution | Boundary |
|---|---|---|
| Controlled phase/envelope co-organization | Cross-statistic correspondence can be predictive beyond the tested individual-SPI distributions; disrupting correspondence while retaining those distributions destroys z/pooling performance. | Constructed functional property; a direct phase/envelope specialist is strong; does not establish brain mechanism. |
| Fresh cohorts and 10/20/40-label curves | Simple predictors exploit this representation with few independent source realizations in specified settings. | Advantages vary with readout, source cohort and shift; no general sample-complexity or clinical label-efficiency factor. |
| Direct-coupling mechanism transfer | The same functional property remains recognizable when shared phase drivers are replaced by direct Kuramoto coupling. | Amplitude/observation laws are retained and diffusion is matched; not arbitrary cross-system generalization. |
| Covariance-modulation negative | Dependence-sensitive input statistics do not ensure that their normalized agreement retains the target; a two-statistic identity illustrates magnitude loss. | Not a proof that the entire catalogue is uninformative or that all empirical errors have one cause. |
| NeuroTycho transfer | z supports awake/anaesthetized discrimination across fitted-animal exclusion and agent/period change; the reduced view retains useful performance. | Two target animals/four dates; physiology versus acquisition/confounding not isolated; enriched raw encoder wins. |
| MiniRocket/tsfresh/catch22 follow-up | z has higher full-size fixed-threshold BA than these specified feature+logistic pipelines. | Added after PF inspection; ranking is nearly perfect for all; small gains over MiniRocket, no whole-library superiority. |

References: [co-organization](oscillatory-coorganization-findings.md),
[correspondence attribution](oscillatory-pooling-attribution.md),
[mechanism transfer](oscillatory-mechanism-transfer-findings.md),
[covariance negative](covariance-modulation-findings.md),
[NeuroTycho](neurotycho-transfer-findings.md),
[library follow-up](neurotycho-library-followup-findings.md).

## AUROC and what the real-data result means

AUROC is the fraction of positive-negative score pairs correctly ordered, with
half credit for ties. Our primary AUROC is computed per date and averaged over
animals/dates. It need not imply a single successful threshold or comparable
score levels across recordings and animal-specific fitted models.

For z-PCA on George's 31 July PF date, awake scores span .3505–.6965 and
anaesthetized scores .9174–1: AUROC1 with five false positives at threshold.5.
Across all dates, z-PCA also has pooled AUROC1; its overall largest awake score
is .6965 and smallest anaesthetized score .7363. Thus perfect ranking is real
on these sampled observations. Its nine errors arise from applying the frozen
threshold, not from overlapping class-score rankings. A threshold chosen from
these target labels would violate the source-only comparison.

Other models do show aggregation effects: catch22 has date-wise AUROC1 but pooled
AUROC.9392. Pooling different animal-specific fitted models is a score-comparability
diagnostic, not a replacement primary endpoint. See the recorded
`results/neurotycho_library_followup_260911/report/auc-diagnostic.json`.

The likely reason for widespread ceiling discrimination is the relatively coarse
contrast between sustained awake and anaesthetized conditions, with substantial
spectral/waveform differences. The successful spectral control supports sufficiency
of that kind of signal here; it does not isolate a causal explanation or rule out
acquisition/time-within-session confounds. These are not near-transition labels,
continuous consciousness measurements, or 128 independent subjects. The original
[recording study](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0080845)
provides the task/label background; our checks validate the implemented split and
numeric results, not every possible biological or recording confound.

The z advantage over MiniRocket is 3.125 BA points for PCA and 4.6875 for PLS,
equivalent to four and six fewer errors in 128 balanced windows. All these feature
pipelines score perfectly on Chibi, so gains concern George. The enriched neural
control remains strongest and conventional spectra already rank perfectly.
These are useful transfer results but do not establish additional inaccessible
state information or unique dependence-based physiological utility.

## Significance assessment

**Scientific interest:** the strongest contribution is the selective statistical
prior: agreement can preserve functional co-organization across some changes,
while discarding magnitudes can remove other targets. Positive interventions,
mechanism transfer and the negative case make this more than a feature leaderboard.

**Methodological maturity:** there is a credible focused methods-paper foundation,
with functioning software, explicit information budgets, multiple controls and
verified results. Related work matters: [Cliff et al. 2023](https://www.nature.com/articles/s43588-023-00519-x)
already studies relationships among diverse SPIs. Novelty must concern the
record-level representation and demonstrated capability, not merely calculating
relationships among statistics or proposing a shared feature space.

**Practical materiality:** useful in the tested cases, but a consequential real-world
advantage is not yet established. Better fixed-threshold transfer can matter when
new labels/calibration are unavailable; its demonstrated magnitude here is modest,
concentrated in one of two target animals, and does not beat the stronger-exposure
neural control. The conventional alternatives are inexpensive. Cost superiority,
clinical utility and general subject-level label savings remain unproven.

**High-impact ambition:** focused work can be important without universal dominance
or new neural machinery. The present evidence supports a defensible contribution;
it does not yet establish a broad new representation-learning capability with
clear scientific consequences. Venue names should not substitute for assessing
that capability. I would not infer ICLR/ICML/NeurIPS/Nature-level impact from these
accuracy gains alone, nor conclude those venues are impossible on the basis of
a rigid checklist.

## Direction

Synthesize the existing mechanistic and transfer evidence before adding more
benchmarks. A strong synthetic methods result does not require a real-data win.
One specific gap is that MiniRocket/tsfresh/catch22 and the enriched raw encoder
were tested only on NeuroTycho, not the central synthetic co-organization task.
The completed convolutional comparison on the existing synthetic core directly tests the learning claim alongside the qualified application evidence. On 2026-09-11 the user authorized [InceptionTime on both existing
synthetic and NeuroTycho tasks](inceptiontime-followup.md); the comparison is complete and independently verified. InceptionTime stays near chance on the synthetic targets at 10–40 labels, while its matched-data NeuroTycho ensemble achieves 95.31% BA, slightly exceeding z-PCA/PLS. This strengthens the scoped synthetic prior result and narrows the real-data neural claim; four selected 10-label folds hit the epoch ceiling, and this is not an optimization-independent impossibility result.
The missing link for a stronger application claim is an independently
motivated use where interaction co-organization matters and z provides a material
benefit beyond credible simpler/neural controls, ideally evaluated on fresh
recordings or a genuinely separate cohort. That is a scientific question, not
permission to search targets until z wins. Alternatively, a general, testable
characterization of when this compression helps could strengthen the methods
contribution; a theorem is not mandatory.

The construction/embeddings/corpus/order-parameter work belongs to linked
workstreams and must be assessed on its own verified evidence. Those components
may support a broader methodological paper, but cannot be silently counted as
proof of the representation-learning claim. One versus two papers remains open;
a separate learning paper needs an independent substantive contribution rather
than repackaging the same experiments under a different label.

## Workshop assessment, 2026-09-11

Judgment: the current evidence supports a credible submission to a relevant
NeurIPS/ICML/ICLR workshop on time series, scientific learning or representation
transfer. The substantive content is a mechanism-motivated descriptor, controlled
correspondence intervention, source-frozen transfer, informative negative tasks
and a qualified external example. It need not beat every neural architecture to
be worth presenting. Acceptance depends on the actual workshop and execution;
this is not an assessment of main-track acceptance or a claim of established
major scientific impact. The completed InceptionTime study fills that comparator gap at full M=16; it does not close reduced-channel or synthetic conventional-library coverage.

The [ICLR workshop remit](https://iclr.cc/Conferences/2026/CallForWorkshops)
expressly includes discussion of work in progress; individual workshops set
their own paper policies. JMLR and the Nature journals are not interchangeable
workshop destinations. [PMLR](https://proceedings.mlr.press/) is the proceedings
series formerly called JMLR Workshop and Conference Proceedings; it is distinct
from a JMLR research article. Nature Computational Science lists journal
[content types](https://www.nature.com/natcomputsci/content), not a generic
workshop-paper track. Do not treat a Brief Communication as a lower-evidence
workshop equivalent. Choosing a focused workshop submission need not decide
whether the eventual full methodological work is one paper or two.
