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

(none yet)
