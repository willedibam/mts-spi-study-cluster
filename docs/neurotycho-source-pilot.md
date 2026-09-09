# NeuroTycho source pilot: waveform and baseline gate

Declared 2026-09-10, before any waveform classification. The source-only scout
uses the four KTMD archives already sampled for annotation audit: Su20110527,
Kin220110513, George20110112, Chibi20110622. Propofol waveforms remain unopened.
This is development evidence, not an independent confirmation cohort.

## Question and progression

Can a consistent, physically defensible observation of these recordings support
awake-eyes-closed versus sustained anaesthetized classification across animals?
First verify raw data, referencing, timestamps and a strong spectral baseline.
If usable, finish the eleven KTMD source dates before fixing the full transfer
comparison. Four propofol dates in two animals are the prospective evaluation;
exclude each target animal entirely from its source training. Agent and year are
confounded. No claim about subjective consciousness, clinical deployment or a
large independent-subject learning curve is supported here.

## Observation, fixed before outcomes

- Provider numerical electrode coordinates are taken from the four Toru Yanagawa
  spatial-map archives, whose dates/animals match this experiment. They are image
  coordinates, not geodesic brain distances or registered anatomical regions.
- Construct nonoverlapping short electrode pairs greedily in ascending Euclidean
  distance, capped at1.5 times median nearest-neighbour distance. Select16pairs
  by farthest-point coverage of their midpoints, starting with the leftmost.
  Electrode indices break ties. The first8 form the reduced observation.
  Save exact pairs, coordinates and map hashes before waveform retrieval.
- Pair signal is lower-numbered minus higher-numbered electrode; do not change
  orientation using outcomes. Each observation uses32raw electrodes for16bipolar
  channels (or16raw for8); it does not borrow a common-average reference from
  unobserved electrodes. This is not an exact reconstruction of the original
  study's manually chosen bipolar montage. Coordinate proximity is approximate.
- Only `AwakeEyesClosed` and `Anesthetized` intervals are eligible. Confirm the
  one-based ConditionIndex convention against ECoGTime. Exclude30seconds at both
  interval boundaries. Choose16nonoverlapping8-second windows per state, evenly
  spread across the remaining interval; all source dates use the same rule.
- Filter each window with10seconds of context on each side, within the labelled
  interval: subtract each pair, linear detrend, fourth-order Butterworth
  0.5–100Hz bandpass and50Hz notch(Q30), zero-phase; resample1000→250Hz with
  anti-alias filtering, then discard context. M16/T2000 is the primary observation.
  M8/final4seconds(T1000) is a predeclared later joint observation shift.
  Do not filter across acquisition boundaries or concatenate discontinuous data.
  **Access qualification:** these are analysis-window lengths after preprocessing;
  the filter uses28seconds of raw context for the8-second output, and the nested
  4-second view inherits that context. This does not test deployment with only
  4or8seconds of raw recording available. All comparators share that exposure.
- Save pre-normalization band powers and quality summaries. The SPI/raw neural
  input will be standardized per channel within its observed window. A spectral
  baseline can retain absolute power; show this advantage explicitly rather than
  stripping a useful clinical/domain signal to favour z.

## Quality gate and first baseline

Fail a window on nonfinite inputs, a constant raw electrode/pair, or a continuous
one-second run of identical raw samples (flat/clipped trace requiring inspection).
Do not reject short repeated values caused by ADC quantization.
Report robust amplitude, crest factor, line-power fraction and raw finite/flat
counts, without rejecting large slow waves just because they occur in anaesthesia.
Reject an archive from this fixed-size scout if either state has fewer than12
valid windows; do not replace channels based on label-specific performance.
Inspect waveforms and spectra before declaring the processing gate passed.

Spectral features: per-channel log absolute and relative power in0.5–4,4–8,8–13,
13–30,30–45,55–100Hz, spectral entropy and95percent spectral-edge frequency.
Use Welch spectra with2-second Hann segments and50percent overlap.
Pool channel means, standard deviations and25/50/75percent quantiles. Include
a regularized logistic readout with C in{.01,.1,1,10}; inner leave-one-source-animal
out, train-only imputation/scaling, select mean animal-level balanced accuracy.
Outer leave-one-animal-out provides an exploratory diagnostic. Report each animal,
balanced accuracy, AUROC and Brier, with no window-level confidence interval
pretending four animals are128independent test subjects. Inner folds must never
mix windows of the same animal across train and validation.

Pipeline continuation depends on waveform usability, not z beating this baseline.
A strong spectral baseline is necessary information, not a reason to weaken it.
After this gate, specify and freeze the matched SPI/PCA/PLS/learned-pooling/raw-neural
comparison before accessing propofol outcomes. Do not fit synthetic state labels
to these recordings, repeatedly tune on the two target animals, or expand to
additional drugs simply because one comparison disappoints.

## Evidence and execution

Original [recording and analysis methods](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0080845)
support bipolar referencing and broad frequency coverage. The
[provider's numerical maps](https://neurotycho.org/spatial-map-ecog-array-task)
identify electrode positions. [Access/label audit](neurotycho-application-audit.md)
records the limitations and source archives.

`scripts/stage_neurotycho_source.py` uses bounded ZIP-member reads (two concurrent
downloads), CRC/size checks, SHA256 provenance and cache reuse. Inputs live in
`data/neurotycho_source_260910`; metadata in `results/neurotycho_audit_260910`.
The scout plan is written before reading waveform bytes. Full source extension,
if warranted, reuses existing member files and selects all eleven KTMD dates;
the staging script cannot select propofol waveforms.

## Neural comparator audit (selection still pending)

[BIOT](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html)
is relevant published variable-channel precedent. Its
[released implementation](https://github.com/ycq091044/BIOT/blob/main/model/biot.py)
uses **STFT magnitudes**, channel embeddings and temporal positions before joint
attention/pooling. Consequently it is a credible spectral biosignal comparator,
but cannot alone test access to all phase-dependent interactions. A comparison
claim about dependence learning also needs a raw, temporally aligned encoder.
The released pretrained models use named human scalp-EEG bipolar channels at200Hz;
these do not directly identify this macaque ECoG montage. Do not map arbitrary
ECoG indices to human electrodes and call that a fair unmodified pretrained test.
Specify any channel-embedding/sampling adaptation and pretraining exposure before
outcomes. No pretrained weights downloaded or neural architecture selected yet.

## Initial execution result

Su20110527 is staged with all66selected signal/time members verified. All32windows
(16perstate) pass the numerical gate; timestamp spacing/origin and waveform lengths
agree. Longest constant raw run is5samples in each state; median raw49–51Hz power
fraction is.00205awake/.00591anaesthetized. Maximum crest factors8.60/12.52 are
reported, not silently excluded. The fixed midpoint waveform/PSD inspection in
`results/neurotycho_source_pilot_260910/su-initial-qc.png` looks plausible, including
the intended50Hz notch. This does not establish exhaustive artifact freedom or
classification utility. `execution.json` records processes and deadline.

The four-source scout subsequently completed:128/128windows accepted, and the
fixed midpoint waveform/spectral figure was inspected for all four animals.
All referenced raw/annotation/time hashes pass; four independent raw-window and
spectral recomputations are exact, as are all four selected classifier predictions.
Results are exploratory, from one date per animal:

| Held-out animal | Balanced accuracy | AUROC | Brier |
|---|---:|---:|---:|
| Chibi |1.000|1.000|.00231|
| George |.9375|1.000|.02120|
| Kin2 |1.000|1.000|.00010|
| Su |.500|.91797|.39003|

Mean balanced accuracy.859375 and mean AUROC.979492. Every Su prediction is below
.5; the result separates ranking from source-threshold transfer. It does not
identify the cause (animal, date, montage, physiology or their combination), and
does not justify tuning a threshold on Su test labels. Original source findings
remain saved. The declared all-eleven-KTMD extension is now justified by usable
signals and the need for recording-date replication, **not by a z advantage**;
reuse the four staged archives and extract the seven remaining dates. Continue
with the same preprocessing and readout grid. No propofol waveform has been used.

One bounded, **post-hoc source diagnostic** is declared before its results: remove
the six absolute-power coordinates from each pooled spectral block, leaving40
relative-power/entropy/spectral-edge features. Repeat the same four grouped folds
and C grid, saving `relative-spectral-*` separately. Motivation is the mismatch
between strong ranking and absolute scores in Su, together with uncalibrated
amplitude units and per-record normalization in the future SPI/neural inputs.
This checks a simple gain-insensitive baseline, not a guarantee of resolving the
failure or an isolated causal attribution to gain. Do not change the threshold
using held-out Su labels. Keep the original70-feature results. Any future choice
between these spectral views must use source-only inner validation before PF.

This diagnostic is complete: relative-only balanced accuracy is1/.90625/1/.5
for Chibi/George/Kin2/Su, mean.85156; Su AUROC falls to.83203. Removing absolute
power does **not** resolve this source failure. Keep both declared results;
no further spectral grid or Su-threshold search is warranted.

## Source p90 resource gate

While the remaining source dates download, export the existing128source windows
and their128reduced views once for a p90 resource/validity gate. A two-task smoke
uses the first George awake full view and first anaesthetized reduced view (indices
1and34). It measures runtime/memory and checks the289-SPI output contract, not
classification. Use `neurotycho-source-scout-260910.yaml`, then a representative
node gate before further extraction. Source-only extraction may proceed without
PF labels or fitting a transfer classifier. Keep exact row/array/catalogue hashes.
The seven-date extension must exclude previously exported record IDs and verify
their unchanged hashes; reuse completed source MPI outputs rather than rerun them.
