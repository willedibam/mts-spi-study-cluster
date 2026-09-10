# NeuroTycho transfer pilot: verified findings

Completed 10 September 2026 UTC. Frozen [protocol](neurotycho-transfer-pilot.md),
[model guide](neurotycho-models-technical-guide.md), and
[configuration](../configs/analysis/neurotycho-transfer-260910.yaml).

**Outcome: useful transfer by Pearson z, but no advantage over the strongest
neural control and no evidence that this application requires cross-statistic
information beyond conventional spectra.** The result supports z as a viable
representation in this pilot. It does not establish general neural superiority,
label efficiency, or clinical utility.

## What was tested

The task was awake-eyes-closed versus sustained anaesthetized discrimination.
Training used 11 KTMD recording dates from four macaques, with each target
animal excluded from its own fitted model. Thus each final model used three
source animals: nine dates/288 matched windows when excluding Chibi, or eight
dates/256 windows when excluding George. The enriched raw control used all
eligible nonoverlapping windows from those same labelled intervals: 2,837 or
2,916 windows, respectively, plus source channel/duration augmentation.

Testing used four propofol dates, two each from Chibi and George, with 32 balanced
windows per date. Every model was selected solely using grouped source-animal
validation. The decision threshold remained 0.5. The primary observation was
16 bipolar channels × 2,000 samples (8 s at 250 Hz); the secondary view was the
first eight channels and final 1,000 samples (4 s). Both inherit 28 s of filtering
context. This is a joint sensor/analysis-window reduction, not a strict short
raw-recording-access experiment.

All 32 prespecified final fits completed: 14 statistical and 18 neural models.
All 704 source and 256 target SPI views used the p90 catalogue of 289 SPIs.
Neither target statistics nor labels entered model fitting, PCA, imputation,
hyperparameter selection or calibration. Neural results below average the
metrics of three seeds, not predictions from an ensemble.

## Results

Balanced accuracy averages dates within each animal and then the two animals.
The full-size comparison is primary; the reduced view is secondary. These are
descriptive results from two animals, not population estimates with narrow
confidence intervals.

| Model | Full BA | Reduced BA | Full AUROC | Reduced AUROC | Full Brier | Reduced Brier |
|---|---:|---:|---:|---:|---:|---:|
| SPI marginals + PCA | .8359 | .6641 | 1.0000 | .7891 | .0743 | .3151 |
| Marginals + graph summaries + PCA | .8516 | .5938 | 1.0000 | .8516 | .0751 | .3128 |
| Pearson z + PCA | .9297 | .9062 | 1.0000 | .9971 | .0616 | .0755 |
| Pearson z + PLS | .9453 | .9141 | 1.0000 | .9971 | .0597 | .0654 |
| Marginals + z + PCA | .9531 | .8203 | 1.0000 | 1.0000 | .0473 | .0965 |
| Validity mask + PCA | .6250 | .6719 | .8428 | .8525 | .2282 | .2136 |
| Conventional spectrum + logistic | .8672 | .9531 | 1.0000 | 1.0000 | .0936 | .0398 |
| Matched raw encoder | .8307 | .8724 | .9980 | 1.0000 | .1670 | .1209 |
| Enriched raw encoder | .9635 | .9740 | 1.0000 | 1.0000 | .0284 | .0163 |
| Learned SPI pooling | .8307 | .7812 | .9993 | .9880 | .1342 | .1734 |

![Transfer performance by animal](../results/neurotycho_target_pilot_260910/evaluation/transfer-by-animal.png)

### What is favourable

Pearson z performs useful state discrimination after the animal/agent/period
shift. PCA alone is sufficient for good performance; PLS adds only 1.56 percentage
points at full size and 0.78 points after reduction. These observations do not
justify a new PCA sweep or a claim that supervised dimension reduction is essential.

Compared with the tested marginal-summary/PCA model, z-PCA improves BA by 9.38
points at full size and 24.22 points after reduction. Adding the selected graph
summaries does not close the gap. Under reduction, z also preserves ranking much
better than those SPI summary models. That is evidence for the practical utility
of this representation/readout combination relative to these specified controls.
It is not a proof that all useful information is absent from SPI marginals.

The full-size m+z result is the best statistical BA/Brier result, which is
consistent with useful complementarity between agreement and marginal summaries.
The combination performs worse than z alone after reduction, despite AUROC 1.0.
Thus concatenation is not a uniformly robust improvement.

### What prevents a stronger claim

The enriched raw encoder is best on both sizes in mean BA and Brier. Its full-size
BA across seeds is .9609/.9766/.9531; reduced BA is .9531/.9922/.9766. It uses more
windows and augmentation together, so this result does not isolate which change
helps. It is nevertheless a necessary practical control: those extra windows
already belong to labelled intervals and require no additional labelled animals
or dates. The matched-window comparison cannot establish a label-cost advantage
when this available exposure is ignored.

The matched encoder's seed dependence is substantial: full BA .7500/.7500/.9922,
reduced BA .8438/.7734/1.0000. z beats its mean, not every trained instance. Selecting
the best seed after seeing PF would be invalid, but hiding that seed would also
misrepresent the neural model's capability. The main failure occurs for George;
Chibi is generally easy for the leading models.

Conventional spectral features, selected without PF data, achieve AUROC 1.0 at
both sizes and exceed z's reduced BA/Brier. These are ordinary within-channel
power-spectrum summaries, **not spectral SPI or cross-SPI coordinates**. Therefore
this dataset does not establish that interactions are necessary for the target,
or that z exposes state information beyond spectra. z's full-size BA advantage
over spectrum concerns the transferred score/threshold, since both rank perfectly.

Learned SPI pooling does not improve on Pearson z here. It ranks states well but
makes more errors at the frozen threshold, predominantly false positives. This
does not establish that richer pooling destroys information, that Pearson is
universally optimal, or that a different pooling architecture would necessarily
help. The tested learned aggregation simply did not deliver better transfer.

The validity mask has nontrivial signal. It performs substantially below z, which
rules out equivalence to this mask/PCA control, but does not exclude all influence
of estimator missingness in the full models. No real-data attribution experiment
isolates that contribution or establishes a physiological mechanism.

## Animal and score behaviour

| Model | Full Chibi | Full George | Reduced Chibi | Reduced George |
|---|---:|---:|---:|---:|
| Marginals + PCA | 1.0000 | .6719 | .8281 | .5000 |
| z + PCA | 1.0000 | .8594 | .9844 | .8281 |
| z + PLS | 1.0000 | .8906 | 1.0000 | .8281 |
| Marginals + z + PCA | 1.0000 | .9062 | 1.0000 | .6406 |
| Spectrum | 1.0000 | .7344 | 1.0000 | .9062 |
| Matched raw | 1.0000 | .6615 | .9948 | .7500 |
| Enriched raw | .9896 | .9375 | .9688 | .9792 |
| Learned SPI pooling | .9479 | .7135 | .8229 | .7396 |

Full-size z-PCA makes nine false positives and no false negatives across the 128
windows; z-PLS makes seven and zero. Reduced z-PCA makes eleven false positives
and one false negative; z-PLS eleven and zero. Near-perfect ranking alongside
these errors directly demonstrates imperfect transfer of the fixed operating
point. It does not identify whether drug, period, animal physiology, observation
conditions or their combination caused the score shift. No post-hoc threshold
repair or target-batch calibration is included.

Per-date and per-seed results, including error counts, remain in the
[independent verification](../results/neurotycho_target_pilot_260910/evaluation/independent-verification.json)
and original [report](../results/neurotycho_target_pilot_260910/evaluation/report.json).

## Numerical and procedural verification

Final evaluation job `178711206.gadi-pbs` completed successfully in 3:28. It checked
target waveform/spectral/bank provenance, all model hashes, target edge/Gram
agreement (maximum discrepancy 1.08e-7), and complete source-prediction CPU replays
for all 18 neural models before writing its report.

The first evaluation stopped because two enriched CUDA models exceeded the
initial 2e-5 CPU replay tolerance. The source-only diagnostic job `178710981`
replayed every source prediction from all six enriched checkpoints exactly on
CUDA. CPU agreed with double precision within 1.32e-7 on the 32 largest-disagreement
source observations per model; the maximum CPU/CUDA discrepancy was 6.59e-5 and
no source classifications changed. TF32 was already disabled. This supports a
backend numerical difference; the particular kernel was not isolated.
Commit `2bdf292` permits 1e-4 only for checkpoints matching that source-only audit,
while retaining 2e-5 for other models. No model, fit, threshold or target selection
changed. The original failed job and diagnostic remain available.

An independent local checker matched all 4,096 prediction rows to the original
128 target-window labels, verified complete unique model/seed/view coverage and
all 32 source-report hashes, and recomputed date/animal aggregates. BA and AUROC
agree exactly; Brier differs by at most 2.55e-8 because the checker uses float64
instead of the original neural float32 arithmetic. All six enriched source CV
splits, selected settings and saved-artifact hashes also pass; source Brier
recomputation differs by at most 1.68e-8.

Reproduce these checks and the figure with
`scripts/check_neurotycho_transfer_results.py` and
`scripts/plot_neurotycho_transfer_results.py`. The original prediction SHA256 is
recorded in the report, whose SHA256 is
`d7d697f9c716ee0fe4bcbe7ca9f1aa1ad9e33725f2195a5fe08699cfc1daa402`.
The six enriched fits consumed 4.49 summed GPU-job training hours, including all
source validation candidates, excluding queue time. PF p90 production completed
in 34:11 for M16 and 4:48 for M8 using dataset parallelism. These are different
hardware/work components, not an end-to-end compute-efficiency comparison.

## Research implication and stopping decision

This is a positive external feasibility result and a negative result for a
stronger claim of z outperforming the strongest tested learner. Together with
the controlled co-organization studies, it supports the more defensible account:
**relationships among dependence statistics can furnish useful transferable
descriptors, with benefits and failure modes determined by the task and learner.**
The synthetic correspondence interventions support a mechanism in those systems;
the present recordings show application feasibility but do not establish the
same mechanism in the brain.

No low-subject learning curve, cross-task representation reuse, state-onset
inference or clinical generalization was tested. Only two target animals exist;
both appeared in exploratory source development, and agent and calendar period
are confounded. The enriched control also shows why counting windows as expensive
independent labels would be misleading for this application.

The frozen pilot is complete. No larger repeat, new drug, threshold adjustment,
PCA grid or neural architecture search is justified merely to reverse the ranking.
The immediate useful step is scientific synthesis: retain the controlled positive
and negative mechanisms, and use this pilot as qualified external evidence.
A standalone high-impact representation-learning claim remains unestablished;
this result alone should not be used to promise a venue or force a separate paper.
Any subsequent experiment should resolve a new, externally motivated uncertainty,
not reselect a task because z lost to an available comparator.
