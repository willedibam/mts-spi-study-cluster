# Prereg: does the MI pair stay at wake in human NREM sleep? (MNI Open iEEG Atlas)

Frozen 27 November 2026, before any N2 or N3 sample was read. Read before freezing: metadata of every recording (`events.tsv`, `channels.tsv`, `electrodes.tsv`, `participants.tsv`) and the wake recordings only (baseline numbers below). Protocol: [`transfer/PROTOCOL.md`](../../../transfer/PROTOCOL.md) Tier 1, with the deviations stated here. Freeze record: [`transfer/mni-ieeg-atlas/freeze.md`](../../../transfer/mni-ieeg-atlas/freeze.md). Code: [`../sleep/`](../sleep/) (`fetch.py`, `extract.py`, `score.py`, `baseline.py`, `analyse.py`), frozen with this file. Changes after freezing go under "Deviations", dated, with the reason.

## Question and prediction

- The anaesthesia claim is specificity: Δz (z(Gaussian MI, KSG MI), real minus joint IAAFT) falls under four anaesthetics, while natural sleep moves it little. The sleep half rests on 2 macaques (7 sessions, −0.24 eyes-open SD against drugs −3.3; `findings.md`, "Not sleep depth").
- Prediction from the macaque: human N3 moves Δz by at most about .1–.25 of the anaesthesia effect, with the same (negative) sign if it moves at all.
- If N3 moves Δz by more than a quarter of the human anaesthesia effect, the "drug-specific, not arousal" claim fails in humans, and the result says so.

## Data

- MNI Open iEEG Atlas, processed 200 Hz clips, as the NEMAR BIDS derivative nm000362 v1.0.2 (CC-BY-NC-4.0, DOI 10.82901/nemar.nm000362), fetched from NEMAR's public S3 remote with SHA-256 checks (`fetch.py`). The origin portal (mni-open-ieegatlas.research.mcgill.ca) now asks for an account with research-only terms; this study uses only the openly licensed mirror and makes no account.
- Bipolar derivations between adjacent contacts of one electrode, judged normal by the source authors (outside the seizure-onset zone and lesions, no interictal discharges, grey matter), sEEG and subdural strips/grids, three centres (codes G, M, N).
- States: wake = 60 s of resting wakefulness with eyes closed (1-min set); N2 and N3 = up to 10 min each from the first sleep cycle (10-min set), scored by the source. N1 does not exist in the release; REM is not used (not part of the question).
- Each recording concatenates artifact-free segments with 2-s zero buffers; segments were not contiguous in the original recording.
- Unit: patient. One recording per state per patient, so session = patient.

## Windows and channels (window rule from metadata and wake data)

- Field band: sleep iEEG is read from slow oscillations (< 1 Hz) to spindles (11–16 Hz) and beta (AASM scoring; von Ellenrieder et al. 2020 report 0.3–80 Hz bands). The atlas long-arm chain keeps 0.5–40 Hz at 100 Hz, and is used unchanged: per segment, never across a buffer, high-pass 0.5 Hz (order 2), low-pass 40 Hz (order 8), zero-phase with even padding of up to 2 s, polyphase 200 → 100 Hz.
- L_max = the longest artifact-free segment (metadata): wake up to 60 s, N2/N3 up to 600 s. T = 1000 fits; non-overlapping 10-s windows tiled from each segment's start, remainder dropped; each window linearly detrended. Short window (T = 200): not used (see Deviations).
- No amplitude artefact rule: the source authors removed artefacts visually, and an awake-scale amplitude rule rejects deep-sleep windows (the PRIOS lesson). Windows must be finite.
- Channels: the derivations present in all three recordings (wake, N2 10-min, N3 10-min), in wake order. If more than 32, 32 are drawn by name with seed 20261127 + patient number (`extract.channel_set`), before any window is read. No re-referencing (already bipolar).
- Inclusion: ≥ 8 channels and ≥ 4 wake windows (metadata: 65 of 91 patients with all three recordings); a contrast needs ≥ 10 target windows (≥ 4 for the 1-min rows).

## Statistic (as the anchor)

- Per window: normal-score each channel; Gaussian MI −½ log(1 − r²) and KSG MI (k = 4, AUTO Theiler window) for every pair (`edges.edge_mi`); z = Pearson r of the two profiles (raw z, the PRIOS unit). Surrogate: 3 spectrum-final joint IAAFT draws of the normal-scored window, seeded by patient, state and window index. Net = z_real − mean of the 3 surrogate z.
- Patient Δ = mean net (target) − mean net (wake). Primary target N3 (10-min); secondary N2 (10-min).
- Null SD: wake split by start time into two halves of equal window count (middle window dropped when odd), second half − first half, scaled per patient by √((1/n_b + 1/n_t)/(2/n_half)) to that patient's contrast counts; SD over patients. Every effect is reported raw and in null SD.

## Decision (primary, N3; N2 by the same rule, secondary)

Margins, fixed here from external results, not from these data:
- SD margin ±.675 null SD = .25 × the PRIOS propofol S-full effect (−2.7 robust null SD; `human-prereg.md`). The .25 ratio is the margin of `mens-prereg.md` (branch `claude/mi-alt-stats-dec35d`).
- Raw margin ±.0021 = .25 × the PRIOS S-full raw Δ (−.0083).

Verdict, from the t-based 90% CI of the patient-mean Δ (TOST at α .05 each side), in this order:
1. **UNINFORMATIVE** if wake real z ≥ .98 (ceiling), wake reliability of either profile < .3 (floor), or the CI half-width in null SD exceeds .675.
2. **MOVES** if the CI lies wholly beyond +.675 or −.675 null SD: N3 shifts Δz by more than a quarter of the human anaesthesia effect, and the specificity claim fails in humans.
3. **NEAR WAKE** if the CI lies inside both ±.675 null SD and ±.0021 raw. **NEAR WAKE IN NULL-SD UNITS ONLY** if inside the SD margin but not the raw one.
4. **INCONCLUSIVE** otherwise.

"Sleep equals wake" is not the claim; NEAR WAKE means "smaller than a quarter of the human anaesthesia effect".

Also reported for the "sleep moves Δz" direction, not part of the verdict: the protocol gates (two-sided Monte Carlo sign-flip over patients, 200,000 flips; |Δ| ≥ 3 null SD; split-half ≤ .25 |Δ|; covariate-adjusted Δ on mean |r|, SD |r|, participation ratio and burst index; reliability-disattenuated Δ). Gate 4 is not applied: there is no nuisance contrast in these data.

## Reported rows (every row printed whatever it shows; nothing else will be tried)

- Ratios of N3 Δ to PRIOS (−2.7 SD, −.0083 raw) and to the macaque 10-s drug session mean (−3.29 eyes-open SD, −.0133 raw). Units differ across datasets; ratios are descriptive.
- The PRIOS null construction (unscaled split-half, first minus second, 1.4826 MAD), so the unit of the −2.7 is matched.
- Δ of real z and of surrogate z separately; Fisher-z version.
- Pairs of derivations sharing a contact (same electrode, adjacent first contacts) excluded.
- Patients with ≥ 16 channels; per centre.
- Processing-matched rows: N3 and N2 from the 1-min set, which had the same adaptive power-line step as wake (the 10-min set did not; the 40-Hz low-pass also removes 50/60 Hz).
- Conventional metrics, reported and not criteria: each metric of `bench.py` at 3b0f0259 (copied verbatim into `sleep/bench_metrics.py`: slow-wave share, relative alpha, beta-ratio proxy, SEF95, spectral entropy, aperiodic slope, LZc, permutation entropy, mean |r|, top eigenvalue, delta and alpha coherence and wPLI, alpha Kuramoto order, wSMI), plus burst-suppression ratio, mean G, mean K, mean K − G and delta fraction: target − wake in that metric's own split-half null SD, on the same windows and channels (bipolar, no normal-scoring).
- Wake real and surrogate z level, reliability, τ, T/(2τ).

## Baseline numbers (wake only, computed before this freeze; `sleep/baseline.py`)

Local run of `score.py run ... wake:1min` and `baseline.py` on the 65 included patients (wake only; outputs kept outside the repository, regenerable).

- Patients 65; channels 8–32 (median 19); wake windows 4 (5 patients), 5 (31), 6 (29).
- Simultaneity: the largest aligned |r| of channel 0 with the others has median .314, against .061 with channel 0 shifted 3 s; aligned exceeds shifted in 63/65 patients. Channels within a patient are simultaneous.
- τ median 5.3 samples (IQR 4.0–6.9); T/(2τ) ≈ 94.
- Reliability (mean pairwise profile r between wake windows): G .915, K .761; no patient below .3 for either. Both above the floor, so the primary is testable.
- Wake real z .903 (median .925; 5 patients ≥ .98); surrogate z .903. Not at ceiling.
- Null SD (protocol scaling to N3 counts) .0067 raw (N2 the same); PRIOS construction .0069. Fisher: .0385 (PRIOS construction .0674).
- Forecast 90% half-width of the patient-mean Δ under no change: .0014 raw = .21 null SD, inside both margins (.675 SD; .0021 raw = .31 null SD). The raw margin is the binding one.

## Pre-stated limits

- Wake is a daytime eyes-closed standardised recording; N2/N3 are from the first night cycle. Day, time and setting differ between baseline and target, and no timestamps exist. The split-half null cannot see this (as in PRIOS).
- Wake is at most 60 s (4–6 windows), so the null and each patient's Δ are dominated by wake noise. Precision is checked by the half-width rule above.
- No drug arm in these patients, so there is no within-dataset positive control: a NEAR WAKE result carries no assay-sensitivity evidence beyond the ceiling, reliability and precision checks. The margins come from other datasets with different montages (PRIOS: 2-s CAR ECoG; macaque: 10-s ECoG), so raw-unit comparisons are approximate.
- Epilepsy patients, normal-appearing channels only; sparse sEEG sampling with weak cross-pair coupling (wake median |r| about .04), so z is carried by a few strong, mostly same-electrode pairs.

## Deviations from Tier 1 (declared at freeze)

- Only the primary readout `mi_g|ksg` is computed. The secondaries (`r2|mcd2`, `l1|linf`, `mcd2|ksg`) and reported rows (`pcorr|pcorr_mcd`, `r2|ksg`, `dcorr|ksg`) are out of scope: the question is the sleep half of the MI-pair specificity claim, the secondaries have no sleep claim to test, and the protocol expects them to be untestable outside anaesthesia ECoG. Hence no short window and no Holm set.
- Gate 1 for n ≥ 6 uses a Monte Carlo sign-flip (200,000 flips, seed 0) instead of the exact test, which is infeasible at n ≈ 65.
- The verdict is an equivalence rule (above), because the question is whether sleep stays at wake. The protocol gates are reported for the opposite direction.

## Cost

Gadi `normal`, one job of 48 CPUs, 96 GB. Measured locally 31 ms per KSG pair at T = 1000; 1.64 M pair-windows × 4 passes, plus about 10% for the 1-min rows: about 62 CPU-h, so about 1.3 h on 48 CPUs, expected about 130 SU; worst case 10 h walltime, 960 SU.

## Deviations

None yet.
