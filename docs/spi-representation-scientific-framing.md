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
