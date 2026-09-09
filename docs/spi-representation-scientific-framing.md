# SPI–SPI: the scientific claim and next evidence gap

Updated 2026-09-10. Scientific decision brief; the subsequent
[direct phase-mechanism protocol](oscillatory-mechanism-scout.md) now specifies the
mechanism experiment. It is now complete, including a bounded post-hoc score
diagnostic; see the [mechanism-transfer findings](oscillatory-mechanism-transfer-findings.md).
The earlier pooling and faster-regime experiments are also complete; see their
[verified findings](oscillatory-transfer-findings.md).

## Question and reason to care

**Can relationships among dependence measures provide a transferable representation
of interaction organization across changes in dynamics and observation?**

The practical objective is to recognize comparable organization in recordings
whose waveforms, timescales and observed channels differ, using a shared statistical
description. Label efficiency measures how readily that description can be used;
it is not the whole motivation. Fewer than 20 independent labelled units is a
credible resource constraint in some applications, not an established universal
regime or a reason to disregard useful within-unit windows and unlabelled data.

Here, “organization” means how different kinds of dependence coincide across
links. It does not mean reconstructing spatial topology or identifying causal
coupling. The candidate hypothesis is that some properties of this coincidence
transfer more reliably than individual dependence magnitudes. There is no general
guarantee: z deliberately removes information that some tasks require.

Zero-target-label application of a source-trained predictor is a domain
generalization question; target adaptation is a different experiment. This
distinction has established precedent in [Muandet et al., ICML 2013](https://proceedings.mlr.press/v28/muandet13.html).
The contribution must be the particular statistical representation, mechanism and
useful behaviour, not relabelling the problem as transfer learning.

## Evidence and its limits

| Proposition | Current evidence | Missing evidence |
|---|---|---|
| Cross-statistic correspondence can carry useful task information. | Co-organization prediction; independent per-SPI dyad shuffling destroys z's useful predictions while preserving each SPI's edge multiset. | This does not establish exclusive access to that information or application usefulness. |
| Fixed agreement features can reduce the burden of learning. | Several 10-label comparisons favour z/readouts; this advantage is not uniform for PCA under the combined mechanism/observation shift. Pooling is competitive at 20–40 and sometimes better. | General label-efficiency factors, optimization-independent advantages and comparison with pretrained encoders. |
| The representation transfers beyond its exact training distribution. | Source-frozen discrimination survives prospective timescale and controlled phase-mechanism changes; combined observation reduction exposes score-transfer limits. | Broader mechanisms, independent application utility and cross-task reuse; the faster regime is not demonstrably harder. |
| Transfer gains involve more than small-data training. | Differences from individual-SPI summaries persist at 40 labels; the controlled mechanism test also reveals ranking gaps. | The earlier faster-regime gap was partly threshold transfer, and z itself develops a score-transfer deficit under the combined mechanism/observation shift. These effects must be distinguished. |
| Compression has a useful, selective bias. | Co-organization positive and covariance-modulation negative; the latter has an exact two-statistic magnitude-loss example. | A general characterization of when the full catalogue succeeds or fails. |

The present claim is therefore a **useful statistical representation of a specific
interaction property, with scoped transfer across timescales and phase-generating
mechanism, and a label-efficiency advantage that varies with readout and shift**.
This is more substantial than dimensional
compatibility, but not yet a general reusable representation of dynamical systems.

The construction of z is fixed. PCA learns axes; its dimension and readout are
selected using source labels in the current pipeline. Learned SPI pooling learns
a task-dependent embedding. None of these facts establishes transfer to new tasks.
For valid columns, the SPI correlation matrix is a Gram matrix of standardized
edge profiles. This gives an interpretable geometric construction, not by itself
a novel geometry method or evidence that manifold machinery would help.

## Completed checkpoint: one change in mechanism, the same functional property

The design below records the rationale for the now-completed checkpoint. The
linked findings give its outcomes and limits; it is not an outstanding run request.

Prioritize a cross-mechanism test over more training sizes, timescale shifts or
PCA choices. Keep the current source fits frozen. A candidate target model would
produce phase coordination through direct oscillator interactions rather than
the current shared phase drivers, with a separately specified amplitude process.
This is a candidate model class, not a claim that another simulator automatically
constitutes an independent application.

Before any new p90 extraction, resolve one crucial issue: **the target must refer
to the same functional property in both mechanisms**. Identical versus crossed
coupling graphs do not automatically imply identical versus crossed observed
phase/envelope dependence networks. Check that correspondence using independent
long reference trajectories, separate from the finite recordings used for
prediction. Define the property without SPI–SPI or a fitted classifier; use the
direct phase/envelope observable as a reference competitor, never as evidence of
z's special ability. Check stationarity, numerical resolution, observable
variation and whether the proposed partitions actually generate the intended
functional organization. Do not select parameters using z performance.

The model equation, parameter ranges, reference criterion, scout size and acceptance
rule must be written and frozen before that feasibility scout. A scout can reject
an unobservable or physically incoherent target; it cannot validate a learning
claim. If the shared-property premise fails, repair the model for a documented
scientific reason or stop this extension, rather than search for a favourable z
result. Any repair creates a new exploratory protocol and requires fresh
confirmation data.

If the premise passes, freeze one independent target test, with M16/T1000 primary
and the existing M8/T500 view secondary. Reuse all existing source model fits and
10/20/40 budgets: z PCA/ridge, z PLS, individual-SPI summaries, normalized shapes,
fusion, learned SPI pooling, raw controls and raw encoders. Retain the direct
specialist. No target fitting, feature selection or threshold adjustment. Report
balanced accuracy, AUROC and Brier separately to distinguish discriminability
from decision-score transfer. Existing test-cohort bootstrap intervals remain
conditional on the trained source cohorts.

Success would extend the claim to a second mechanistic realization of one
functional property. Failure with a successful direct reference would identify
a transfer limit; failure of the reference/property check would make that target
unsuitable for the proposed inference. Neither outcome requires changing the
completed positive or negative studies. An application and cross-task reuse remain
distinct possible later contributions, not additional axes of this experiment.

## Paper significance

A focused, well-explained result can be significant without universal superiority
or a new neural architecture. A stronger paper would show why the representation
works, where it loses information, and why the transferable property matters
beyond its construction. The present positive/negative pair and intervention
already support that explanation. A second synthetic mechanism would strengthen
scope, but cannot by itself establish real-world impact or guarantee a particular
venue. Do not split or combine papers merely to match fashionable terminology.

## Recommended direction after the completed tests

The next contribution should be transferable inference of an independently
defined scientific state using statistically grounded representations. Label
efficiency is supporting evidence. The controlled direct-coupling test strengthens
the mechanism argument, but matching group diffusion and retaining amplitude laws
limits its independence; it should not carry a broad cross-system claim.

Treat fixed z and learned SPI aggregation as competing realizations of this
approach. The learned-pooling advantage is not evidence against the broader
representation idea. The [completed attribution check](oscillatory-pooling-attribution.md)
now supports a role for cross-SPI correspondence in the learned model: preserving
each SPI's edge distribution, reciprocal pairing and validity while disrupting
correspondence reduces accuracy to near chance after matched retraining at40labels.
That supports the shared statistical prior under this task/training regime; it
does not prove that Pearson compression is optimal or identify the unique learned
function. No architecture sweep is warranted by this check.

The [completed NeuroTycho audit](neurotycho-application-audit.md) finds30anesthesia/sleep
archives, all accessible, with actual condition annotations inspected in six
animal/agent cells. This supersedes the earlier indexed-page count of31. Only
Chibi and George have propofol, ketamine-alone and medetomidine-alone data;
KTMD spans allfour animals. The KTMD/propofol comparison also changes recording
year. This supports a qualified pilot, not broad subject generalization. Sources:
[task documentation](https://wiki.neurotycho.org/Anesthesia_and_Sleep_Task_Details)
and [live catalogue](https://neurotycho.org/data/detail.json).

The original study defined anaesthetic unresponsiveness with behavioural testing,
with slow waves as additional confirmation; its sleep labels instead used spatial
slow-wave synchrony. Thus sleep is unsuitable as an independent validation of an
interaction marker without separate label evidence. Propofol covered only two
animals in that study. Do not infer that the released archive supports an arbitrary
cross-animal/cross-agent split or equate unresponsiveness with subjective
consciousness. Audit actual annotations before defining the target.
[Yanagawa et al. 2013](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0080845).

If this candidate passes, choose one primary transfer question, subject/session
grouping, baseline family and untouched evaluation partition. Include spectra,
appropriate established state markers, individual-SPI summaries, z with the
existing simple heads, learned pooling and one competent published raw-data
baseline. Permit legitimate within-unit training windows and report pretraining
exposure; scarcity concerns independent units, not an artificial prohibition on
using their recordings. Sensor coverage is a secondary robustness test unless
the application audit establishes it as the primary need. Variable-channel
learning already exists: [BIOT, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html)
is precedent, not an automatically suitable ECoG comparator without adaptation
and exposure checks.

Related work also limits novelty: [Santoro et al. 2026](https://www.nature.com/articles/s41467-026-75959-w)
compares information-theoretic/topological higher-order metrics on HCP fMRI,
their relationships and functional utility. That is a different construction from
record-level SPI–SPI features, but mapping relationships among interaction metrics
is not an untouched idea. Our claim must concern a specific transferable capability
and its mechanism; a meta-statistic is not automatically a direct estimator of
genuine higher-order dependence.

A convincing outcome would be reproducible transfer or practical complementarity
beyond strong spectral/domain controls and neural baselines, with attribution to
interaction relationships. Reject an application if the labels or splits cannot
support that claim, or if simpler features solve the relevant task equally well
without another demonstrated practical benefit. Real data are not logically
necessary for a strong methods paper, but they are the most direct missing evidence
for the application-impact ambition here. If that evidence does not materialize,
the existing learning studies remain a scoped part of the main method paper;
a separate learning paper should earn an independent contribution. The subsequent approved pooling attribution and annotation-only application audit
are complete; no application waveform extraction or full benchmark is running.
