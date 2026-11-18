# Frozen prediction: z(Gaussian MI, kNN MI) beyond IAAFT under ketamine, human stereo-EEG (Kauvar, Richman, Liu et al. 2025, DANDI 001347)

Frozen 15 November 2026, before any voltage sample was read. Only metadata had been read: the asset list, plus for one file (SD185 puffInfusion) the session description, electrode table and array layout. The macaque and mouse results are in `review-nature-level.md` and `claar-prereg.md`.

## Why this dataset

It is the only open human intracranial recording with a drug contrast that we have found. There are 7 epilepsy patients with stereo-EEG (about 116 contacts, 200 Hz), each with a ketamine infusion session and pre-drug sessions. Each file holds about 10 min before the eyepuffs start and 10 min after ("Voltage traces of multichannel SEEG around eyepuffs, 10min pre and post puff start").

The dose is not in the files. Patients reported their experience under ketamine (press coverage of the paper), so the dose is sub-anaesthetic and dissociative, not loss of consciousness [I, unverified]. This tests a conscious, dissociated ketamine state in humans. It is not anaesthesia.

## Data and windows

- **Subjects:** the 7 `sub-SD*` subjects with a `puffInfusion` session: SD168, SD172, SD181, SD185, SD186, SD211, SD245.
- **Sessions:**
  - Reference: `puffPreinfusion` if present, else `puffBaseline`.
  - Drug: `puffInfusion`.
  - Secondary: `puffPostinfusion`, `puffControl` and `puffBaseline` (where `puffPreinfusion` is the reference). `puffPostinfusionfast` is not used.
- **Segment:** samples from 0 to 590 s of each file, the pre-puff period by the dataset description. If a DC trigger channel shows an event earlier than 595 s, the segment ends 5 s before the first event.
- **Channels:**
  - Contacts labelled `OUT OF BRAIN` are excluded.
  - Bipolar derivations are made from adjacent contacts on the same shaft (consecutive label numbers in one `group_name`), using the subject's channels common to all sessions used.
  - Atlas `continuous` chain: notch 60 Hz (Q 30), high-pass 0.5 Hz, low-pass 40 Hz, zero-phase, resample to 100 Hz.
  - Bad channels: atlas `bad_channels` on the subject's concatenated used sessions.
  - Then 32 channels spread evenly over the good bipolar list in shaft order, or all of them if there are fewer than 32.
- **Windows:**
  - 10-s non-overlapping windows over 5–585 s.
  - Artefact rule: atlas `window_ok` with the robust centre and scale of the reference session (as in Claar Deviation 1).
  - At most 30 windows per session, spread evenly.
  - A subject needs ≥ 10 windows in both the reference and drug sessions.

## Statistic

As in Claar, `claar_z.py` variant `asis`: normal-scored, no further re-reference (bipolar already removes the common reference), `zgk` on the 32 channels, and 3 joint multivariate IAAFT draws (spectrum-final). Residual = real − mean of the surrogates.

## Prediction and decision rule

- **Estimand:** per subject, Δ = mean residual (infusion) − mean residual (reference).
- **Null scale:** reference-session split half (later half − earlier half of its windows by time), SD across subjects.
- **Prediction:** Δ < 0, the same sign as the macaque (ketamine 4/4 sessions, both animals) and the mouse.
- **Decision (n = 7):**
  - **PRESENT:** ≥ 6 of 7 negative, mean Δ beyond −2 null SD, and the subject-bootstrap 95% CI upper end < 0. One-sided sign p for 6 of 7 is .0625 and for 7 of 7 is .0078, so the CI condition is part of the rule.
  - **ABSENT:** |mean Δ| within 2 null SD and the CI lower end > −0.010.
  - **Otherwise INCONCLUSIVE.** If fewer than 6 subjects qualify, the result is UNDERPOWERED and no verdict is given.

## Secondary (reported whatever the primary outcome)

- Post-infusion − reference (reversal), and control − reference.
- Baseline − preinfusion in the three subjects with both: a descriptive, between-session null.
- Burst index and delta fraction by session, and the within-infusion Spearman of residual against burst index.
- Window rejection rate by session, as a proxy for interictal discharges.
- Absolute 1−z², real and surrogate, by session.

## Prior [S]

The macaque ketamine effect was at an anaesthetic dose. If the signature is slow-wave gain modulation, as the mouse induction result suggests, a sub-anaesthetic dose without slow waves predicts ABSENT. A drug-driven change in the form of coupling would allow PRESENT. I lean towards ABSENT or INCONCLUSIVE; both remain informative.

**Known threats:**
- Interictal epileptiform discharges are themselves non-Gaussian bursts, and their rate may change with ketamine.
- Infusion timing within the session is unknown.
- n = 7.
- Sessions may be on different days. Positive results need the between-session null (baseline − preinfusion) to be small.

## Pre-data clarifications (before any LFP sample was read for analysis)

1. **Trigger channel.** The puff trigger is the DC channel most active after 600 s relative to before 590 s. Its first event is the first sample beyond half its post-600 s 99.9th-percentile excursion. Only DC traces had been read when this was set. A generic threshold had caught an unrelated pre-600 s excursion on DC01. On SD168 baseline, the trigger was DC03, with its first event at exactly 600.0 s. The DC read is 0–630 s.
2. **Transfer budget.** Only the reference, infusion and post-infusion sessions are streamed (21 sessions, about 3.5 GB), within the approved 2–4 GB. The control and baseline-vs-preinfusion secondaries are not run.
3. **Electrode count varies by subject.** SD168 has 168 contacts, giving 154 bipolar derivations. The 32-channel spread is taken over the subject's common good bipolar list.

## Result (15 November 2026) [V]

**Primary: ABSENT.** n = 7, 3 of 7 negative (one-sided sign p = .77), mean Δ_inf +0.0000, subject-bootstrap 95% CI [−0.0053, +0.0051], null SD 0.0055 (0.0 SD). Output `kauvar/kv_result.txt`, windows `kauvar/kv_z.npz`, code `kauvar/code/`.

| Subject | Δ_inf | Δ_post | split half |
|---|---|---|---|
| SD168 | +.0099 | +.0178 | −.0039 |
| SD172 | +.0048 | +.0116 | +.0064 |
| SD181 | −.0058 | −.0042 | −.0019 |
| SD185 | −.0113 | −.0013 | +.0075 |
| SD186 | −.0053 | — | −.0018 |
| SD211 | +.0021 | +.0051 | +.0029 |
| SD245 | +.0057 | +.0100 | −.0075 |

Secondaries:
- Post-infusion − reference: 2 of 6 negative, mean +0.0065.
- Burst index unchanged in every subject (reference/infusion 0.93–0.97 / 0.92–0.98); delta fraction moves both ways (−0.08 to +0.07). There is no slow-wave or burst state change under infusion.
- Within-infusion Spearman(residual, BI): median +0.12, 4/7 positive.
- Absolute 1−z² falls under infusion in SD172 (0.905 → 0.863) and SD245 (0.811 → 0.708), equally in real and surrogate, so it is linear-spectral, not the non-Gaussian residual.
- 30/30 windows in every used session.

Sensitivity [I]: the CI lower end (−0.0053) excludes the macaque ketamine effect (CAR −0.016, bipolar −0.032), so a macaque-sized effect would have been seen, assuming the units are comparable across deep bipolar sEEG and surface ECoG.

Between-session noise: Δ_post is positive in 4 of 6 subjects (mean +0.0065, three values ≥ +0.010), larger than the within-session split-half scale (SD 0.0055). Between-session variability therefore exceeds the null used here, and the 1−z² drops in SD172 and SD245 are not by themselves evidence of drug.

Drug exposure in the analysed segment [V]:
- The paper's methods (Kauvar et al. 2025, PMC12286656) state that the iEEG cohort received target-controlled ketamine infusion, directed by an anaesthesiologist. The first stage targeted 0.13 µg/ml blood level and the second 0.26 µg/ml. The two bolus participants were excluded from neural analysis.
- The eyepuff assay ran preceding infusion, at about 30 min after infusion start, and at 110 min.
- The files hold 10 min before puff onset, so the 0–590 s segment is about 20–30 min into a maintained infusion. The 7 subjects are those meeting the paper's behavioural inclusion criterion (a ketamine effect on late eye closure).
- Post hoc check, not preregistered: log10 relative band power, infusion − reference. Low gamma (30–40 Hz) is +0.10 to +0.26 in 7/7 subjects, against reference split halves within ±0.07. Alpha is −0.01 to −0.32 in 7/7. This is the expected ketamine spectral shift, so the drug was acting.
- Anaesthetic ketamine blood levels are several-fold higher than 0.26 µg/ml [I, general pharmacology, not checked here].

Reading [I]: in human deep sEEG, at sub-anaesthetic, dissociative ketamine that measurably changed the spectrum and the behaviour, the signature did not change. There was no burst-index change either. This shows the signature is not a generic marker of ketamine acting on the brain. It is not a contradiction of the macaque and mouse anaesthesia results, but it cannot attribute the absence to dose: dose, species, recording depth, montage and patient population all differ at once. There is still no human evidence for the anaesthesia effect; that needs a human anaesthetic-dose recording.

Remaining threats: a sites/montage difference (deep bipolar versus cortical surface), interictal activity, and n = 7.

Deviations and failures:
- SD186 puffPostinfusion failed in extraction (`padlen` error in the filter, most likely a very short segment from an early trigger detection). It was not investigated; it affects only the post-infusion secondary (n = 6).
- Transfer: about 2.8 GB of LFP plus the DC trigger reads.
- Compute: 600 windows, 113 min local (7 cores, numpy KSG + IAAFT; no pyspi).
