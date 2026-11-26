# Human replication of the MI-pair anaesthesia effect: PRIOS ECoG and ds004541 scalp EEG

Frozen 19 November 2026, before any sample of either dataset was read. Only public metadata has been inspected: BIDS sidecars, channel tables and event tables (summaries in `research/representation/dataset_candidates/candidates.md`, Part C). Code: [`../human/`](../human/). Changes after freezing go under "Deviations", dated, with the reason.

## Claim under test

In macaque ECoG, z(Gaussian MI, kNN MI) beyond a window-matched joint IAAFT surrogate falls under anaesthesia, after normal-scoring each channel (`../../findings.md`, "Gaussian against model-free MI"). Prediction for both datasets: the anaesthetised-minus-awake change in the residual is negative.

## Shared statistic (identical for both datasets)

- Channels: good channels only; common average reference over them, then drop the last (rank). Primary: the 32 good channels evenly spaced in channel order (`edges.sub32`), fixed per subject before windows are read; if a subject has fewer than 33 good channels, all of them.
- Per window: normal-score each channel; Gaussian MI −½ log(1 − r²) and KSG MI (k = 4, AUTO Theiler window) for every pair, as `edges.edge_mi`; z = their Pearson correlation across pairs.
- Surrogate: joint multivariate IAAFT over the same channels, spectrum-final, 3 draws per window (seeds fixed by subject and window index). Residual = z_real − mean of the 3 surrogate z values.
- Per subject: Δ = mean residual (anaesthetised) − mean residual (awake).
- Null scale: within-awake split by time (first half of awake windows minus second half), SD across subjects.
- Secondary, descriptive: the same with the recording's own reference (no CAR); burst index (median MAD/SD over channels) and delta fraction per window, and Δ at matched burst index (bins of .05, subjects with ≥ 3 windows per state in a shared bin).

## Decision rule (each dataset separately)

- REPLICATES: mean Δ < 0, one-sided t-test on subject Δ p < .05, and at most one subject with Δ > 0 (PRIOS: ≥ 6/7 negative; ds004541: ≥ 6/7 or 7/8).
- OPPOSITE: mean Δ > 0 and one-sided t (greater) p < .05.
- NOT DETECTED: anything else. Report mean Δ in null SDs and its 95% interval; a null is interpreted against the macaque size (2-s arm about −.02; 10-s arm about −.013), not as absence.
- A subject is included if it has ≥ 10 windows in each state (PRIOS: ≥ 20).

## PRIOS (OpenNeuro ds004370, CC0): within-patient ECoG, awake ward against propofol in theatre

- Subjects: PRIOS01–06, 09. Awake = all SPESclin runs; propofol = the SPESprop run. Channels: type ECOG with status `included` in every run of that patient.
- Windows between stimulation pulses (median interval 5 s). For each gap between consecutive pulses (`electrical_stimulation` onsets) of at least 4.0 s, take the 2.25-s epoch ending 0.25 s before the next pulse; it therefore starts ≥ 1.5 s after the previous pulse. Each epoch is processed alone with the atlas short-arm chain (`prepare.short_epoch`: detrend, 1-Hz high-pass, 40-Hz low-pass, resample to 100 Hz, keep the central 2 s; T = 200).
- Exclusions: an epoch overlapping any `artefact` (any electrodes), `seizure` or `burst suppression` annotation, extended by 2 s on each side; atlas artefact rule (`window_ok`, robust scale from the patient's awake epochs after CAR; flat run ≥ 1 s at source rate); atlas bad-channel screen over all eligible epochs.
- At most 100 epochs per state, spread evenly over passing candidates (`prepare.spread`).
- Pre-stated confounds: awake and propofol on different days and settings (ward, theatre); epilepsy patients; mastoid reference; residual evoked responses (CCEPs), which propofol changes. Pre-stated check (descriptive): Δ using only epochs that start ≥ 3.0 s after the previous pulse; an effect carried by evoked tails should shrink there.
- Power: the macaque 2-s arm gave about −.02 against an eyes-open SD of .009 (16 channels). Seven patients give one-sided t power of about .8 only if subject Δ SD is below about |mean|; this is not known in advance.

## ds004541 (OpenNeuro, CC0): scalp EEG, awake before induction against loss of consciousness

- Subjects: those with both `loc` and `roc` events (sub-02, 03, 04, 07, 08, 09, 10; sub-11 lacks them and is excluded). Agent not stated in the metadata.
- Channels: EEG channels with status other than `bad` in channels.tsv, then the atlas bad-channel screen.
- Continuous chain (atlas long arm, `prepare.continuous`): notch 50 Hz, 0.5-Hz high-pass, 40-Hz low-pass, resample to 100 Hz; 10-s non-overlapping windows (T = 1000), per-window linear detrend.
- Awake: from 10 s after recording start to 10 s before the `start` event. Unconscious: from 60 s after `loc` to 60 s before `roc`.
- Exclusions: any window within 15 s of a `verbal/*`, `motor`, `tetanic/*` or `other` event; atlas artefact rule with the subject's awake robust scale.
- At most 30 windows per state, spread evenly.
- Pre-stated confounds: scalp volume conduction and ceiling (Chennu scalp z ≈ .98, no effect); unknown agent; the `end` event's meaning is not documented and is ignored; reference not stated.

## Deviations

**Deviation 1 (19 November 2026, after the frozen run; PRIOS only): stimulated electrodes left out per epoch.** The frozen PRIOS primary could not be computed: 3 of 1052 windows gave NaN surrogate z (PRIOS01, PRIOS06), because a channel pair was identical (|r| = 1.0000) and IAAFT reproduced it exactly. The identical pair is the electrode pair stimulated by the preceding pulse (checked against `electrical_stimulation_site` in 23 sampled PRIOS01 epochs: every |r| > .999 pair was the stimulation site); it carries the stimulator's common signal between pulses, in 81/200 PRIOS01 windows and 43/100 PRIOS06 windows, and its near-infinite Gaussian MI is a leverage point in every such window, not only the NaN ones. Fix: for each epoch the electrodes of the previous and the next pulse's site are left out of the CAR, the artefact check and robust scale (NaN-aware), the burst and delta indices, and the analysed channels (the subject's fixed 32 minus those present). Without stimulated electrodes the code is identical to the frozen path, so ds004541 is unaffected and not rerun. Everything else is as frozen; the frozen run's outputs are kept (`results/.../human/prios-frozen/`). Because the defect was found by looking at the data, the PRIOS verdict after this deviation is reported as post-deviation.

**PRIOS sensitivity set (19 November, after the post-deviation result was seen; declared before computing; exploratory, every row reported whatever it shows; nothing else will be tried).** Both target exclusions found below.
- S-A, annotation scope: an `artefact` annotation that names electrodes (`electrodes_involved_onset/offset` not `all`) leaves those electrodes out of each overlapping epoch, as Deviation 1 does; `all` artefacts, `seizure` and `burst suppression` still exclude the epoch. Expected to recover PRIOS02 (its 1831-s annotation is T18 alone).
- S-B, own-state artefact scale: each state's robust scale instead of awake's, as the Claar sensitivity [V]. Here propofol is not burst-like (median burst index .95–.98), unlike the isoflurane case that motivated the awake scale. Expected to recover PRIOS04 and PRIOS06.
- S-AB, both. Inclusion, statistic and decision rule as frozen.
- S-full, the maximum-power design, declared before computing (user decision, 19 November: the effect is expected, so test it with all the data rather than the comparability-driven subset): S-AB rules, plus every good channel instead of the evenly spaced 32 (CAR over all good channels minus left-out electrodes, last dropped; as the macaque all-channel edge test), every passing epoch instead of at most 100 per state, and 10 joint-IAAFT draws instead of 3. Designated the headline sensitivity row. Decision rule and inclusion as frozen.
- Reported for every row: the frozen split-half null SD; its robust version (1.4826 × MAD across subjects, because one subject's awake drift sets the plain SD with n = 5); d_z = mean Δ / SD(Δ); and Δ split into real-z and surrogate-z changes. No between-day null exists (one awake day per patient), so the day and setting confound is not addressed by any of these.

Not changed, reported as found (each would be a post hoc choice): the awake-scale artefact rule rejects most or all anaesthetised windows where amplitude rises (PRIOS06 propofol median |CAR| 88 against awake 27 µV; PRIOS04 18/97 pass; ds004541 sub-03 and sub-10 0 pass), so retained anaesthetised windows are the least awake-unlike ones; PRIOS02's propofol run is covered by two `artefact` annotations (399 s and 1831 s), so the "any electrodes" rule leaves no epoch; ds004541's own `channels.tsv` marks 35–52 of 58 EEG channels `bad`, leaving 6–23 per subject.

**Sensitivity results (19 November; all declared rows; `results/.../human/nulls.txt`).** S-A 5/6 negative, −.0081, p = .005; S-B 6/6, −.0096, p = .007; S-AB 7/7, −.0090, p = .003; S-full (headline) 6/7, −.0083 [−.0163, −.0002], p = .023: REPLICATES in every row. Robust null −2.0 to −2.7 SD (frozen null −0.7); d_z −1.53 to −0.95. Gadi: S-A, S-B, S-AB about 11 SU each; S-full 75 SU (48 cpus, 47 min).

**Post-hoc secondary (kshift), 26 November 2026: declared before any kshift value was computed; PRIOS only.** Added after every result above; it changes no frozen or post-deviation verdict. Code [`../kshift/kz.py`](../kshift/kz.py), [`../kshift/kscore.py`](../kshift/kscore.py); result [`../kshift-result.md`](../kshift-result.md).
- Estimator: kshift = k4 minus the mean k4 of the same pair with x_j circularly shifted by T/4, T/2 and 3T/4, with the same AUTO Theiler window and ψ(N) (`ksgbias.py` `edge_all`). It is applied to the real window and to every joint-IAAFT draw (same seeds), and z = corr over pairs of (Gaussian MI, kshift). Optional third column: k2te (2τ₁/ₑ window, pool ψ), a candidate only. k = 4.
- Rows: the post-deviation primary (`prios/`) and S-full, the headline (`prios-sfull/`). S-A, S-B and S-AB are not rescored. ds004541 is not rescored: it is uninformative at ceiling.
- Estimand: each row's Δ (propofol − awake per subject), recomputed with kshift. Reported with R = Δ(kshift)/Δ(k4) against the same run's k4, subjects negative, one-sided t, sign agreement with k4, the declared null scales (split-half SD, its robust version, d_z, via `nulls.py` unchanged), and Δ split into real-z and surrogate-z changes.
- Reading: SURVIVES if kshift meets the frozen REPLICATES rule (mean < 0, one-sided p < .05, at most one subject > 0) and R ≥ .5. LOST if R < .25 or no more than half the subjects are negative. PARTIAL otherwise.
- Gate: the k4 column must reproduce the stored z. The stored z were made on Gadi. On macOS, 4 windows reproduced the surrogate z to 8e-16 but the real z only to 1.8e-4, because normal-scored windows have exact distance ties that different hardware breaks differently. The gate is reported real and surrogate separately.
- Caveat, stated in advance: at T = 200 the shifts are 0.5, 1 and 1.5 s. The non-stationary-null study found structure on about 1-s scales, so a shared slow component would be partly subtracted, which pushes R down, as the robustness prereg states for the NeuroTycho 2-s arm.
