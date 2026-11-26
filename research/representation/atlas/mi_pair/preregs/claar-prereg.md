# Frozen prediction: z(Gaussian MI, kNN MI) beyond IAAFT under isoflurane, Claar et al. 2023 mouse EEG (DANDI 000458)

Frozen 15 November 2026, before any EEG sample was read. Only public metadata (epochs, trial tables, electrode tables, chunk layout) has been inspected (`scratchpad/nwbscan.py`, `windowplan.py`). Changes after this point are listed under "Deviations" with reasons.

## Why this dataset

Independent of NeuroTycho on every axis that matters for the claim: species (mouse), agent (isoflurane, untested in macaque), recording (30-contact skull EEG, left-cerebellum reference), lab. Many animals: 18 sessions from 17 mice have ≥ 10 stimulus-free 10-s windows in both states, which removes the two-animals-per-drug limit.

## Data and windows

- EEG only (`acquisition/ElectricalSeriesEEG`, 2500 Hz, int16), channels with `is_data_valid` (17–30 per session). Neuropixels data are not read.
- The protocol delivers cortical electrical (and in some sessions visual) stimuli every ~4 s in both states, so only stimulus-free stretches are used:
  - awake: EEG start + 30 s to 5 s before the session's first stimulus (pre-stimulation baseline, ~250–420 s);
  - isoflurane: inside the `isoflurane_anesthesia` epoch, from its start + 60 s, at least 5 s after and 1 s before any stimulus.
- 10-s non-overlapping windows, at most 30 per state, evenly spread over the eligible stretches. Sessions with < 10 windows in either state are excluded (fixed list: 543393, 543394, 521887, 521885, 521886, 546655 out; 18 sessions in). Mouse 569073 has two sessions: the first (20210415) is the primary unit; the second is reported separately.
- Artefact rule (atlas): drop a window if any channel has > 5% of samples beyond 6 robust SD, a flat run ≥ 1 s, or non-finite values.
- Preprocessing: zero-phase FIR anti-alias and decimation 2500 → 100 Hz (5 × 5); common average reference over valid channels (primary, matching the NeuroTycho pipeline); per-window linear detrend.

## Statistic

- Per window: normal-score each channel (rank → Gaussian quantile); Gaussian MI = −½ log(1 − r²); KSG MI with k = 4 and the AUTO Theiler window min(int(2·acf_i·acf_j), T/10); z = Pearson correlation of the two across all channel pairs.
- Surrogate: joint multivariate IAAFT over all valid channels, spectrum-final, 3 draws per window; residual = z_real − mean z_surrogate.
- Session effect Δ = mean isoflurane residual − mean awake residual.
- Null scale: within-awake split by time (first half minus second half of awake windows), SD across sessions.

## Prediction and decision rule

- Primary: Δ < 0. One-sided sign test over the 17 mice (primary sessions): replication if ≥ 13 of 17 are negative (p ≈ .025) and the mean Δ is beyond 2 null SD.
- Falsified (for this dataset) if ≤ 9 of 17 are negative or the mean is within 2 null SD. 10–12 of 17 is inconclusive.
- Prior [S]: uncertain. Isoflurane at surgical plane in mice often produces burst suppression, i.e. globally shared high-amplitude bursts, which in the calibration is the shared-gain type and could raise rather than lower z; the fly isoflurane set went opposite to NeuroTycho on other pairs. A negative result is informative, not a failure of the analysis.

## Secondary (reported whatever the primary outcome)

- Same Δ on raw (not normal-scored) channels; on the as-recorded reference.
- Stationary awake only: the half of awake windows with the lowest mean running speed (cm/s, `running_speed`), against isoflurane.
- Excess kNN MI on strong pairs and the amplitude–coupling slope beyond IAAFT (the macaque mechanism readouts), exploratory.

## Deviations

### Pre-data clarifications (15 November, still before any EEG sample was read)

Only layout was inspected (chunk shape, dtype, timestamps regularity, running-speed path). These settle points the text above left open; the decision rule is unchanged.

1. Preprocessing follows the atlas chain exactly (`research/representation/atlas/prepare.py`), because the macaque windows the claim rests on were made with it and the stated intent above is parity; the FIR-decimation and per-window-detrend wording is replaced. Per contiguous stimulus-free stretch at source rate: notch 60 Hz (Q 30), high-pass 0.5 Hz (Butterworth 2) and low-pass 40 Hz (Butterworth 8), both zero-phase, `resample_poly` to 100 Hz; common average over good channels, then the last good channel is dropped. No per-window detrend.
2. Stretches read: awake from EEG start to 1 s before the first stimulus; isoflurane from 2 s after a stimulus to 0.5 s before the next (or the epoch end). Windows lie only in the eligible ranges above (≥ 3 s from any stretch edge), so filter edge transients and stimulus artefacts stay out.
3. Selection as in the atlas: tile each eligible range with non-overlapping 10-s candidates, apply the artefact rule to every candidate, keep at most 30 passing candidates spread evenly (`spread`). Rejected windows are therefore replaced. A session with < 10 passing windows in either state is excluded and reported.
4. Artefact rule = atlas `window_ok` and `bad_channels`:
   - Channels: `is_data_valid` and the atlas screen (SD < 1% or > 5× median SD over the eligible data).
   - Robust centre and scale: per channel (median, 1.4826·MAD) over all of that state's eligible 100-Hz average-referenced data.
   - A window fails if any channel has > 5% of samples beyond 6 robust SD, any non-finite value, or any sample inside a source-rate run of ≥ 2500 identical values (1 s) in any good channel.
   - Burst suppression is judged against the isoflurane state's own spread; no further handling. Rejection counts are reported by state.
5. Statistic: `zgk` and `ns` imported unchanged from the macaque script (`zgk_ns.py`). All good channels, normal-scored; 3 joint-IAAFT draws per window, with real z computed once.
6. Null: per session, mean residual of the later half of selected awake windows minus the earlier half; SD across the 17 primary sessions. Mouse 569073's primary session is 20210415.
7. Blind order: all 18 sessions are extracted and QC'd (amplitude, spectrum, channel count, rejection by state, timing check) before z or Δ is computed on any of them.

### Deviation 1 (15 November): post-QC, pre-z. Artefact scale taken from the awake state

**What had been seen.** QC on the design session 569070 only: amplitude, rejection counts and spectra; no z, residual or Δ on any session.
- Isoflurane there is burst suppression. The isoflurane robust SD (MAD) is about 10 µV, set by suppression, while the window SD is 9–31 µV with bursts. So clarification 4's rule rejected 38 of 45 isoflurane windows (awake: 0 of 26).
- Pass/fail tracked burst content: passes were 9–21 µV, fails mostly 23–31 µV. The rule was selecting isoflurane windows by the very feature the prior names as a possible sign-reverser.

**Change.** `window_ok` is unchanged, but centre and scale for both states now come from the session's awake eligible data, where the EEG is continuous and the MAD is a valid spread.
- An artefact is now an excursion beyond what that animal's waking EEG does.
- On 569070 this keeps 41 of 45 isoflurane windows and all 26 awake ones. It drops the 249 µV window and three windows in which one channel reaches 12–21× its own median window SD (passes ≤ 1.6×), i.e. channel-specific artefacts.
- A second candidate rule (any channel's window SD > 5× its median window SD, atlas `NOISY_SD`) rejected exactly the same 4 windows.
- This rule is frozen and applied to the other 17 sessions without further change. Sessions left with < 10 windows are excluded and reported, not re-amended.

**Also fixed.**
- Code to text: eligible ends are capped at 3 s before each stretch end, which clarification 2 requires. Isoflurane stretches end 0.5 s before the next stimulus while eligibility ended 1 s before it.
- Sign-test threshold for the surviving n: the smallest k with one-sided binomial p ≤ .025 (n = 17: 13; n = 16: 13; n = 15: 12).

**Sensitivity.** The clarification-4 rule (state's own scale) is rerun as a labelled sensitivity analysis on sessions keeping ≥ 10 windows under it.

**Burst index, frozen before z.** Per window: the median over channels of 1.4826·MAD / SD of the average-referenced window (≈ 1 for continuous Gaussian-like EEG, small under burst suppression; on 569070, awake .93–1.02 and isoflurane .05–.87). Secondaries:
- the within-isoflurane per-session Spearman correlation of window residual with burst index (sign test across sessions);
- across mice, Spearman of Δ with the session's median isoflurane burst index;
- Δ using only isoflurane windows with burst index ≥ .85, for sessions with ≥ 5 such windows.

## Results (15 November)

Data and code: `claar/` here (selected windows `claar_sel/`, QC `claar_sel_qc.txt`, scripts `code/`). Transfer 1.8 GB. All 18 sessions passed QC under deviation 1: 11–30 isoflurane and 18–30 awake windows each, timing error ≤ 4.1 ms, no channels lost beyond `is_data_valid`. Isoflurane EEG is burst-like in every mouse: median burst index .41–.75 against an awake .93–1.02.

**Primary (as frozen): replicates.** Normal-scored z residual, Δ = isoflurane − awake:
- 16 of 17 mice negative (needed ≥ 13; one-sided p = .0001).
- Mean Δ −0.0191 [−0.0260, −0.0132] (session bootstrap), = −7.3 null SD (null SD .0026).
- The second 569073 session: −0.0134.
- Stationary-awake secondary (lowest-speed half of awake windows): −7.4 SD, 16 of 17.

**Burst secondaries (frozen before z) locate the effect in burst suppression:**
- Within isoflurane, Spearman(residual, burst index) has median +.75 and is positive in 17 of 17 mice: the burstier the window, the lower the residual.
- Across mice, Spearman(Δ, median isoflurane burst index) = +.52 (p = .034).
- Using only continuous-like isoflurane windows (burst index ≥ .85), 7 mice qualify: 4 negative, mean Δ −0.0004 = −0.1 null SD.

**Exploratory, not preregistered** (`claar_bimatch.py`). Per-mouse fit of isoflurane residual on burst index, evaluated at that mouse's median awake burst index:
- Δ = +0.0040 [−0.0001, +0.0080] = +1.5 null SD; 5 of 17 negative.
- Slope positive in 17 of 17.
- Interpolated, not extrapolated: isoflurane burst index reaches the awake value within each mouse.

**Macaque comparison** (`nt_burst.py`, existing NeuroTycho normal-scored results):
- Drug windows are not burst-like: median burst index .91–.97, with 95–100% of windows ≥ .85.
- Restricting to burst index ≥ .85 leaves Δ unchanged: KT −0.0159 (4 of 4), KTMD −0.0151 (11 of 11), MD −0.0048 (4 of 4), PF −0.0116 (4 of 4); sleep −0.0008.
- Within-drug Spearman(residual, burst index) is weaker but also positive (median +.12 to +.48).

**Checks on the exploratory layer** (`claar_check1.py`, `nt_burst.py`, `synth_burst.py`):
- **Mouse matched windows are sparse.** Isoflurane windows inside the awake burst-index range number 0–9 per mouse, and 0 in 8 of 17 mice. Their amplitude is mixed: some near-isoelectric (8–14 µV against an awake 15–27 µV), some at awake amplitude (571619, 586468, 569062). So "matched" rests on few, partly suppression-dominated windows.
- **Mouse continuous-like subset.** The 7-mouse subset (burst index ≥ .85) has mean Δ −0.0004 [−0.0032, +0.0028]. That excludes macaque-sized effects (−0.012 to −0.016) for those windows, but the windows are not a well-populated continuous anaesthetised regime.
- **Primary without the design session** 569070: 15 of 16 negative, −7.2 null SD.
- **Macaque, same per-unit regression** (drug residual on burst index, evaluated at the awake index). Slopes are similar to the mouse (median +.03 to +.10 against +.055). The drug effect mostly survives matching: KT −0.0159 → −0.0128 (4 of 4), KTMD −0.0155 → −0.0132 (11 of 11), PF −0.0117 → −0.0121 (4 of 4), MD −0.0062 → −0.0027 (4 of 4); sleep −0.0010 → −0.0001.
- **Synthetic.** A linear Gaussian network (16 channels, 6 AR(2) sources), average-referenced and normal-scored, against joint IAAFT:
  - with no gain, residual +0.0002 ± 0.0019;
  - with a shared on/off gain (bursts over an independent floor), residual −0.068 ± 0.007.
  So shared gain lowers network z; the prior above, which said it might raise z, was wrong. (16 November: this shows gain is sufficient to lower z, not that real burst suppression is shared gain; see `amplitude-prereg.md`.)

**Reading** [I]:
- **Frozen verdict: replicates.** It stands as computed.
- **Same sensitivity in both species.** The residual responds the same way to within-window amplitude non-stationarity (burst index) in mouse and macaque.
- **Mouse, isoflurane epoch.** The isoflurane-epoch drop is fully accounted for by burst suppression, which a shared-gain mechanism is sufficient to produce. At matched burst index no isoflurane effect remains, though the matched windows are sparse and partly isoelectric.
- **Macaque.** The drug effect is mostly a shift beyond burst index.
- **What this means for the claim (superseded by Addenda A and B).** The isoflurane epoch alone could neither support nor refute the macaque effect beyond burst suppression, because it has almost no continuous anaesthetised EEG. The induction epoch supplies it, and there the drop is present (Addendum A) and is not drift (Addendum B).
- **Burst-driven copula non-Gaussianity is still non-Gaussian dependence.** The statistic is behaving as designed. The open question is whether the macaque effect is a subtler form of the same amplitude-linked loosening (the hub result was also amplitude-linked) or something else.
- **Consequence for any state contrast.** Report an amplitude-shape covariate (burst index) and the contrast at matched value alongside it.
- **Stationary-awake secondary.** Close to vacuous: median awake running speed is 0 in most mice.
- **Follow-up.** Done: Addenda A and B.
- **Raw secondary not run.** It is about 30× slower, since IAAFT converges slowly on raw burst data; est. 9 h.

Pending: as-recorded-reference secondary and the clarification-4 sensitivity (own-state scale; 16 sessions keep ≥ 10 windows); see the end of this file.

## Addendum A (15 November): induction epoch, frozen before any induction sample was read

**Why.** The isoflurane-epoch test is dominated by burst suppression; at matched burst index almost no continuous anaesthetised EEG was left. The `isoflurane_induction` epoch (81–538 s per session, stimulus-free in all 18; scan.json) passes from waking through continuous slow-wave EEG towards burst suppression. It is the one place in this dataset where a macaque-type effect beyond bursts can be tested.

**Data and statistic.**
- Windows: from induction start + 30 s up to anesthesia-epoch start + 60 s. This is disjoint from the primary isoflurane windows, which start at anesthesia start + 60 s.
- Extraction, preprocessing and statistic are identical to the primary (deviation 1 artefact rule with awake scale, average reference, normal-scored z, 3 joint-IAAFT draws). All candidates are kept (cap 60).
- The awake comparison is each session's primary awake windows, already computed.

**Target windows, "continuous anaesthetised".** Both conditions must hold:
- burst index ≥ the session's awake 5th percentile;
- delta fraction ≥ the session's awake 95th percentile.

The delta condition shows the window is no longer waking EEG; the burst condition shows it is not burst-like.

**Estimand.** Per session, Δ_ind = mean residual of target windows − mean awake residual. Units are the 17 primary mice, counting only those with ≥ 3 target windows.

**Decision.**
- Macaque-type effect present: ≥ k negative, where k is the smallest count with one-sided binomial p ≤ .025, and mean Δ_ind below −2 null SD (same null SD .0026).
- Absent: the mean is within 2 null SD and the session-bootstrap 95% CI excludes −0.010 (the lower end of macaque KT/KTMD/PF at matched burst index, about −0.012). Otherwise inconclusive.
- If fewer than 8 mice qualify, the result is reported as underpowered and no verdict is given.

**Secondary.**
- The induction windows' residual against burst index and delta fraction, within session (Spearman, sign test).
- Δ for all induction windows regardless of target status.

**Prior [S].** Uncertain. Isoflurane is a GABAergic/volatile agent like propofol, and the macaque PF effect survived burst matching (−0.012). But induction is non-stationary, early windows may contain movement, and the mouse isoflurane slope against burst index already accounts for the epoch effect.

### Addendum A result (15 November)

**PRESENT by the frozen rule [V].** 9 of 17 mice had ≥ 3 target windows (406 induction windows in 18 sessions, 8–37 per session). All 9 were negative (need ≥ 8; one-sided p = .0020). Mean Δ_ind −0.0063 [−0.0095, −0.0034] = −2.4 null SD. Per mouse: 551399 −.0059, 569062 −.0008, 569070 −.0011, 569072 −.0078, 571619 −.0155, 575102 −.0070, 586468 −.0113, 590479 −.0009, 599017 −.0060.

**Secondaries [V].** All induction windows regardless of target: mean −0.0074, 17/17 negative. Within induction, residual against burst index: median Spearman +.63 (17/17 positive). Against delta fraction: −.56 (17/17 negative).

**Checks, post hoc (claar_ind_check.py) [V].**
- Target windows are burst-matched to awake: mean BI difference −.018 to +.036. The adjustment using each mouse's own isoflurane residual~BI slope moves the mean to −0.0065 [−0.0100, −0.0032].
- Amplitude matches awake (median rms 11.6–29.1 µV against 14.5–27.2). Running is low (median 0–7 cm/s).
- Against stationary awake windows: −0.0068 [−0.0099, −0.0038], 9/9.
- Leave-one-mouse-out means: −0.0051 to −0.0069.

**Open threat: time.** Induction starts 25–66 min after the awake windows, and the null SD is from a within-5-min awake split. In the macaque, recovery against awake drifted −.001 to −.004, so drift of that size would take up a large share of −0.0063 [I]. In the 9 mice, Δ is larger at longer gaps: mean −.0084 at gaps ≥ 52 min against −.0045 at ≤ 37 min. That is consistent with drift, but n is too small to separate it from session protocol [I]. Addendum B tests it.

**Reading [I].** The mouse does show the drop in continuous, burst-matched, slow-wave-dominated anaesthetised EEG, about a third of the isoflurane-epoch size (−0.0191) and similar to macaque MD and PF at matched BI. Within induction, the residual still falls with lower burst index and higher delta even near BI ≈ 1, so the signature scales with slow-wave depth, not only with the presence of burst suppression. The burst index measures amplitude dispersion, so matching on it does not exclude slow-wave UP/DOWN gain modulation, the leading candidate mechanism. A nonlinear coupling function is not distinguished from it.

## Addendum B (15 November): late-awake drift control, frozen before any late-awake sample was read

**Data.** Stimulus-free gaps inside the awake stimulation block, before induction: trial stop + 5 s to next trial start − 1 s, eligible ≥ 10 s. Scan metadata shows these at about 1750–2500 s in 13 of 17 primary mice (absent in 551399, 569068, 569069, 569070). Same extraction, awake artefact scale, good channels, statistic. All candidates kept (cap 60).

**Estimand.** Δ_late = mean late-awake residual − mean primary (early) awake residual, per mouse with ≥ 3 late windows.

**Decision.**
- Drift explains the induction effect if mean Δ_late ≤ −.0031, which is half of Δ_ind. It is negligible if mean Δ_late > −.0031 and its CI lower end > −.0063. Otherwise it is inconclusive.

**Secondary.** Δ_ind against late awake (induction target − late awake) in mice with both, sign count and mean.

## Secondary results (15 November)

- **As-recorded reference, no common-average re-reference [V].** 16/17 negative, mean Δ −0.0147 [−0.0211, −0.0097] = −7.2 null SD (its own split-half null SD .0020). The primary does not depend on the re-reference.
- **Own-state artefact scale (clarification 4), the pre-deviation rule [V].** 16 sessions qualify, 15 primary units (551399 and 569070 drop out). 14/15 negative, mean Δ −0.0164 [−0.0227, −0.0108]. In null SD that is −15.6, but this selection's own split-half null SD is only .0011, so the absolute Δ is the comparable figure. Deviation 1 did not create the result. Iso windows with BI ≥ .85: 9 sessions, 7 negative, −0.0016.
- **Transfer.** Primary 1.8 GB, induction 0.55 GB, late awake 0.30 GB; 2.65 GB in all.

### Addendum B result (15 November)

**DRIFT NEGLIGIBLE by the frozen rule [V].** 13 mice with 4–60 late-awake windows (235 in all), taken from stimulus-free gaps inside the awake stimulation block, starting 29–45 min after recording start. Their delta fraction and burst index match early awake (δ medians .20–.50 against .24–.48; BI .91–1.01). Δ_late (late − early awake): 6/13 negative, mean −0.0000 [−0.0006, +0.0004], across-mouse SD .0010. The script printed the explains-threshold as −.0032, half of the unrounded Δ_ind, against −.0031 in the text; the result is far from either.

**Secondary [V].** Induction target − late awake in the 7 mice with both: 6/7 negative (p = .0625), mean −0.0065 [−0.0108, −0.0024]. The exception is 569062, at +.0020 with 4 targets; its Δ_ind against early awake was −.0008.

**Reading [I].** The awake residual is stable from early awake to 29–45 min. Over that span neither time drift nor stimulation history explains the induction drop. Four of the nine induction mice start induction 52–66 min after their awake windows, beyond the covered span; for them drift is bounded only by extrapolation. Inference rests on the sign test and bootstrap CI. The split-half null SD and the drift spread average different numbers of windows (Δ_late 4–60, Δ_ind 3–11 targets), so their ratio is not a calibration.

**δ-filter check, post hoc (claar_delta_check.py) [V].** Awake windows (early plus late) that meet the same target criteria, against the other awake windows: 6 mice with ≥ 3 such windows, 5/6 negative, mean −0.0008, about an eighth of Δ_ind. Within early awake, residual against δ: median Spearman −.21 (13/17 negative). Within induction, residual against time into induction: median −.56, 17/17 negative.

**Net reading [I].** Selecting high-δ windows does not by itself produce the drop. High-δ awake windows show a small drop of the same sign, while induction shows about 8×. The drop deepens monotonically through induction. So the mouse shows an anaesthetised-state drop in continuous, burst- and amplitude-matched EEG, consistent with the macaque effect.

This does not contradict the isoflurane-epoch match at awake burst index (+0.0040). Those matched windows were sparse (none in 8 mice) and partly low-amplitude isoelectric suppression (8–14 µV). The induction targets are amplitude-matched continuous EEG.

Remaining caveats: n = 9 (7 for the late-awake secondary). Induction is a transition state. The high-δ awake windows only just pass the threshold, so δ level is not fully matched. Gain modulation (slow-wave UP/DOWN) versus a nonlinear coupling function is unresolved.

## Post-hoc secondary (kshift), 26 November 2026: declared before any kshift value was computed

Added after every result above; it changes no frozen primary, addendum or verdict. Code [`../kshift/kz.py`](../kshift/kz.py) and [`../kshift/kscore.py`](../kshift/kscore.py); result [`../kshift-result.md`](../kshift-result.md).

- **Estimator.** kshift = k4 minus the mean k4 of the same pair with x_j circularly shifted by T/4, T/2 and 3T/4 (2.5, 5, 7.5 s), with the same AUTO Theiler window and ψ(N) (`ksgbias.py` `edge_all`). It is applied to the real window and to each of the 3 joint-IAAFT draws (same seeds), and z = corr over pairs of (Gaussian MI, kshift). Optional third column: k2te (window 2τ₁/ₑ, pool ψ), a candidate, not a primary. k = 4 throughout.
- **Estimands.** The primary Δ (isoflurane − awake, 17 mice), Addendum A's Δ_ind (9 mice; target selection uses burst index and δ only, so it is unchanged) and Addendum B's Δ_late, each recomputed with kshift. Reported with R = Δ(kshift)/Δ(k4) against the same run's k4, mice negative, sign agreement with k4, each estimator's own split-half null SD, and Δ split into real-z and surrogate-z changes, in the format of `robustness-result.md`. The frozen scorers (`claar_sum.py`, `claar_ind_sum.py`, `claar_late_sum.py`) are rerun unchanged on each estimator; `claar_ind_sum.py` and `claar_late_sum.py` hard-code the k4 null SD .0026, so own-SD values come from `kscore.py`.
- **Reading.** SURVIVES: the frozen rule's sign and size conditions hold for kshift (primary ≥ 13/17 and beyond −2 own null SD; Δ_ind ≥ 8/9 and below −2 own null SD) and R ≥ .5. LOST: R < .25, or no more than half the mice negative. PARTIAL: anything else. Addendum B: drift explains if kshift Δ_late ≤ kshift Δ_ind / 2, as frozen.
- **Gate.** The k4 column must reproduce the stored z (verified exactly on 4 primary, 2 induction and 2 late windows on the machine that made them). On Gadi, real-window k4 may differ at about 1e-4: normal-scored windows share one quantile set, so distance ties are exact, and different hardware can break them differently. Surrogate windows have no ties. Real and surrogate gaps are reported separately.
- **Caveat, stated in advance.** Isoflurane EEG is burst suppression. Bursts shared across channels and lasting longer than 2.5 s survive part of the shift, so kshift may subtract real shared-gain dependence as well as estimator bias. A low R would then not separate bias removal from removal of slow shared dependence. The per-window mean subtracted offset is saved, real and surrogate, so that this can be read without a rerun.
