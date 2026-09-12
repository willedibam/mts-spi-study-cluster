# SPI–SPI research: manuscript blueprint and evidence boundaries

Working synthesis, 2026-09-11. This preserves the overall programme while making
the representation/transfer contribution independently assessable. It does not
choose one paper versus two, declare a venue, or authorize new experiments.
Numbers below come from the linked completed reports; InceptionTime is now independently verified; its synthetic failure and stronger real-data ensemble narrow the claim in different ways.

## Central question and motivation

Can a common statistical description of interactions support comparison and
state inference across multivariate recordings whose observation dimensions
and recording conditions differ?

This question has three distinct levels: constructing comparable coordinates,
showing that those coordinates organize scientifically relevant variation,
and showing that a specified learner can exploit them efficiently. A fixed
output dimension establishes only the first. Useful geometry, successful
physical-state association and a supervised learning advantage need their own
evidence. The practical motivation is to compare recordings without requiring
one-to-one channel correspondence and to infer properties of their interaction
organization. Sensor relabelling is not equivalent to arbitrary sensor mixing
or loss of informative coverage.

For the learning component, the narrower question is:

> When do relationships among dependence measures make interaction organization
> easier to infer and transfer than individual-statistic summaries or raw-data
> learners under specified label and observation constraints?

The central positive example concerns correspondence between **phase-coupling
and amplitude-envelope-coupling patterns**. Do not shorten this to
"phase–amplitude coupling", which names a different physiological phenomenon.
Its biological motivation does not establish a validated clinical target.

## Full programme: distinct evidential roles

| Component | Role in the overall argument | What the present evidence supports | Boundary that must remain visible |
|---|---|---|---|
| Construction and illustrative cases | Explain the object being represented | Ordered edge profiles become a fixed statistical coordinate system; existing r/rho/MI and distance/DTW cases supply intuition | Illustrations are not benchmark confirmation; do not use their existence to choose targets |
| Controlled class embeddings and cross-M/T study | Test comparability and retrieval | Frozen 14-class, 2,520-record confirmation: primary held-cell BA .9913 and cross-M/T retrieval mAP .8177 | Residual observation-size information remains; physical-size changes differ from fixed-population sensor subsampling; later geometry selection is post-confirmation exploration |
| 1,053-record heterogeneous corpus | Demonstrate exploratory breadth | Complete seeded p90 representation, reproducibility/validity audits, stable partitions at specified resolutions | Similarity is not shared mechanism; tags are not mutually exclusive truths; duplicate/nested records complicate independence; historical visualization recipes differ from the primary analysis |
| Unsupervised physical order coordinates | Test scientific state organization without state-label fitting | Several system-specific coordinates track known order trends; independent CML2D confirmation is a particularly well-supported example | Across-control tracking is not microscopic fluctuation prediction, numerical calibration, formula discovery or a new critical-point estimate; simple observables can be better |
| Supervised co-organization and transfer | Test access to a specified interaction property | Few-label benefits over tested SPI summaries, source-frozen dynamics/mechanism transfer and correspondence interventions | Research-level target selection remains; specialist and learned-pooling successes limit exclusivity; InceptionTime is near chance at full size under the tested label budgets; four 10-label folds hit the epoch ceiling |
| NeuroTycho | External feasibility and transfer | Source-animal-excluded, cross-agent state prediction; z adds value over several declared pipelines | Two target animals, drug/year confounding, ranking saturation and stronger enriched neural performance prevent broad application superiority |

Evidence routes: [cross-M/T](context/cross-mt-transfer.md),
[corpus](context/zenodo-mts-corpus.md),
[physical benchmarks](context/order-parameter-benchmarks.md),
[CML2D independent confirmation](research/order-parameter-benchmarks/cml2d-confirmation-results-260911.md),
[complete learning evidence map](spi-representation-evidence-map.md).
These workstreams retain their original protocols, schemas and owners. This
synthesis does not silently recompute or relabel their results.

Coverage clarification: the corpus study already has an exploratory catch22
comparison on its mixed real/synthetic collection. Statements that the three
conventional libraries were tested "only on NeuroTycho" refer to the supervised
transfer programme, not the entire repository. Native MiniRocket/ridge has now been verified on the central full-size synthetic tasks and remains near chance; tsfresh/catch22 have not been tested on that core.

In particular, distinguish the historical one-dimensional quadratic-CML
epsilon=.3 archive and its operational spectral-order coordinate from the
independently confirmed two-dimensional CML benchmark. They are different
systems and different physical claims. The latter's 544-master confirmation
gives rho=.8830 with future finite-run Q, but within-control association is
weak and raw mean-absolute correlation is stronger. Its failed contiguous
sensor-layout pilot is useful evidence about coverage, not a nuisance to omit.

## Construction and geometric interpretation

For K=289 p90 statistics, let A_k be an M by M MPI. Form a_k by vectorizing all
M(M-1) ordered off-diagonal entries in the same order for every statistic.
Both i→j and j→i are included. For a common finite edge set and nonconstant
profiles, define b_k=(a_k-mean(a_k))/||a_k-mean(a_k)||. Then

    R_kl = b_k^T b_l;  R = B^T B;  z = offdiag_upper(R).

Thus z has K(K-1)/2=41,616 coordinates. Each coordinate describes agreement
across the same links, not an interaction magnitude or the identity of a link.
With a common finite edge set, R is positive semidefinite and has rank at most
min(K,M(M-1)-1). Arbitrary pairwise missing-value handling need not preserve
that Gram property. The actual validity policy must accompany the definition.

Common edge permutations leave z unchanged, including permutations that are
not induced by channel relabelling. Positive per-SPI affine changes also leave
it unchanged. These facts explain both useful invariance and loss of means,
scales and edge incidence. Keeping both directions does not make z a unique
identifier of causal direction or network topology.

The per-record rank bound does **not** establish a low-rank matrix across all
recordings or prove that PCA will improve prediction. PCA's utility is tested,
not inferred from the number of coordinates. Likewise, calling Pearson a good
bias–variance tradeoff requires empirical comparison; coarseness alone does
not establish that claim. No new SPD-manifold method is proposed here.

The full MPIs contain everything used by z. Rich m consists of 23 separate
summaries per SPI, not full MPIs. Failure of a fitted predictor using m does
not prove formal statistical insufficiency of m; intervention evidence and
readout comparisons support more specific empirical claims.

## Representation/transfer module: draft narrative

Provisional title: **Cross-statistic correspondence for transferable inference
of interaction organization**.

Provisional abstract, to incorporate the completed InceptionTime and MiniRocket comparisons:

Multivariate recordings can differ in sensor coverage and duration, making
direct comparison of their interaction patterns difficult. We investigate a
record-level representation formed from agreement between statistical measures
of dependence across aligned channel pairs. Using a catalogue of 289 measures,
the representation maps each recording to 41,616 cross-statistic coordinates.
In controlled oscillatory systems, this representation supports inference of
whether phase-coupling and amplitude-envelope-coupling groupings coincide.
With 40 labelled source realizations, a PCA-based predictor achieves 98.5%
balanced accuracy after observation reduction, compared with 80.8% for rich
individual-statistic summaries. Frozen predictors retain useful discrimination
under changes in dynamical timescales and the phase-generating mechanism.
Correspondence-disrupting interventions preserve individual SPI edge
distributions while reducing prediction toward chance, supporting the
importance of joint statistical organization. Learned aggregation over SPI
edge profiles is competitive, while a separate covariance-modulation task
illustrates information that Pearson agreement can fail to expose usefully.
An external macaque ECoG pilot demonstrates state transfer but does not establish
superiority over the strongest neural comparator. Together, the results support
cross-statistic correspondence as a selective inductive bias for statistical
state inference, with benefits contingent on the target, observations and
learning constraints.

This is a working abstract, not a submission-ready claim of neural superiority.
Incorporate the completed published-architecture and fixed-feature comparisons before finalizing it.
State clearly that extraction is fixed, PCA is learned without labels, PLS and
readouts use labels, and learned SPI pooling learns a different aggregation.

Suggested argument order:

1. Define the scientific property and practical observation shift before the
   descriptor. Explain why comparing separate dependence distributions can
   miss correspondence. Cite relevant prior SPI-comparison and phase/envelope
   literature from the [study design](spi-representation-learning-study.md);
   record-wise correlations themselves are not the novelty claim.
2. Define z, its geometry and its deliberate information losses. Use one clear
   illustration; existing case studies can supply supporting intuition.
3. Show the principal learning curves and the correspondence intervention.
   Display task-informed specialists and learned SPI pooling alongside z.
4. Test frozen-source transfer, separating ranking from a fixed decision
   threshold. Preserve the extra-information status of prevalence adjustment.
5. Present the covariance-modulation failure as a boundary of the proposed
   representation, not a neural-baseline failure. Explain the corrected
   comparison and put earlier broad-screen/spectral diagnostics in context.
6. Present NeuroTycho as a qualified external test, then discuss what remains
   unknown about generality and scientific utility.

## Compact figure and table plan

| Item | Question answered | Evidence and essential control |
|---|---|---|
| Figure 1: representation and correspondence | What changes in z while separate SPI distributions stay fixed? | Mathematical illustration plus already-completed independent-dyad permutation intervention; identify feature-space intervention versus realizable raw-data surrogate |
| Figure 2: labels and transfer | At what label budgets does the representation help, and what transfers? | Original/faster/direct learning curves, matched dimensions, cohort variability; PCA, PLS, rich m, learned pooling, specialist, raw encoders and verified full-size InceptionTime |
| Figure 3: limits of compression | Why is the prior selective? | Covariance-modulation failure and the established scalar-invariance example; avoid extrapolating a two-statistic example to the whole catalogue |
| Figure 4: external state transfer | Is there useful behaviour on real recordings? | NeuroTycho per-animal/date results, fixed-threshold BA alongside AUROC/Brier; matched/enriched neural exposure and library pipelines explicit |
| Main methods table | What information and training did each comparator receive? | Input, invariance, learned stage, source label/recording budget, validation groups, pretraining, target access and extraction/training cost |

For a broader methods paper, insert the embedding/corpus and physical-coordinate
results as their own capability sections; do not squeeze them into Figure 2 as
additional supervised benchmark wins. For a separate learning paper, keep a
short method origin and cite those capabilities rather than duplicate their
principal results. The eventual split should follow distinct contributions and
clarity after synthesis, not a predetermined venue ambition.

### Provisional assessment of a two-paper structure

The user asks for an early assessment, explicitly not a decision. Structurally,
two connected papers are a plausible cleaner destination if the learning work
matures into an independent contribution. Both are representation research;
the useful boundary is **comparative geometry and scientific discovery** versus
**statistical learning behaviour and transfer**, not method versus representation.

Paper A would establish a common interaction-based description of recordings:
construction/intuition, quantitative cross-size comparison, heterogeneous-corpus
organization and physical order-coordinate inference. Its conclusion must survive
removing supervised co-organization/NeuroTycho results. Corpus cluster appearance
alone is insufficient; quantitative validation and physical benchmarks give the
method scientific content. Preserve simple-baseline wins and layout failures.

Paper B would explain and test when cross-statistic correspondence helps infer
a common state with limited source labels and changing observations/mechanisms.
It needs a self-contained construction and one short mechanistic prelude, then
learning curves, correspondence interventions, learned aggregation, negative
controls and transfer/application evidence. It should not repeat the entire
atlas, original case studies or physical-benchmark programme. A useful test is
whether its central conclusion remains interesting to a reader already familiar
with Paper A. Adding readouts and benchmark tables alone may not pass that test;
the correspondence mechanism and conditional transfer lesson are the candidate
independent contribution. A new neural architecture is not mandatory.

Both papers need distinct positioning relative to
[Cliff et al. 2023](https://www.nature.com/articles/s43588-023-00519-x), which already
assembled the SPI library and analyzed the same 1,053-record corpus. Paper A's
new contribution cannot be the existence of that library/corpus; Paper B's
cannot simply be that diverse dependence measures can aid classification.
The methodological and learning stories may ultimately strengthen one integrated
paper if neither is independently persuasive, or if most decisive evidence must
be duplicated to support both. Working with separable modules preserves that
choice. Do not freeze a split around the outcome of one InceptionTime comparison.

## Immediate work and decision boundaries

InceptionTime is complete: 55 source fits, 174 independently checked metric rows and 200 CPU replays. Full-size synthetic ensemble BA stays near 50% while z remains strong; the matched NeuroTycho ensemble reaches 95.31%, slightly above z-PCA/PLS. Preserve the four selected 10-label epoch ceilings, individual-member variability and distinction between ensemble and seed-mean metrics. See [verified findings](inceptiontime-followup.md). No further fits, pyspi extraction or generator work are warranted by this result.

After this comparison, decide whether a conventional MiniRocket/ridge synthetic
control is necessary to close the fixed-feature baseline gap. Choose on the
claim it tests, not on whether its result might favour z. This blueprint does
not launch that follow-up or expand into a library grid.

If z retains a benefit, quantify the budget/shift and uncertainty. If InceptionTime
matches or exceeds it, narrow the neural-comparison claim while retaining the
mechanistic and SPI-summary comparisons. If training is unresolved, report the
comparison as unresolved. Do not change a generator to recover a preferred
ranking or describe full-size results as evidence for reduced-channel superiority.

For a larger contribution, the unresolved question is generality with useful
consequences: an independently motivated new application, or a prospectively
testable account of which targets benefit from correspondence. Neither a larger
saturated sample nor a more elaborate encoder automatically answers that.
There is enough material to draft now; importance and venue readiness remain
judgments to substantiate, not qualities guaranteed by preserving scope.

## Scope reaffirmation and execution policy, 2026-09-12

HCP working-memory MEG is one candidate external-validity experiment within the representation/transfer component. It does not replace the construction/geometry, cross-dimension embeddings, heterogeneous corpus, physical order-coordinate studies, controlled co-organization learning, correspondence interventions, mechanism/observation transfer, negative tasks or existing NeuroTycho evidence. Those distinct evidential roles remain the programme. HCP feasibility or failure must not determine the entire paper narrative, and success would not automatically justify turning the work into a neuroscience application paper. The possible two-paper structure remains conditional on independent contributions.

Keep methodological synthesis progressing while external data are audited. For this research stream, download and process large raw external datasets directly on Gadi; retain small manifests, audits, figures and compact results locally. The initial local HCP waveform attempt failed and has been replaced with a direct S3-to-Gadi copyq transfer. This change concerns data logistics, not authorization to enlarge the scientific benchmark grid or rerun completed comparisons.
