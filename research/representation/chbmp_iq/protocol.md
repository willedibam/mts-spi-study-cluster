# CHBMP IQ: pre-registered analysis plan (draft, 5 November 2026)

Status: **draft, not frozen**. It freezes once the window length is fixed from the sweep (§2). The commit hash of this file and of [analysis.py](analysis.py) is recorded in the run root, and only then is WAIS joined (`--unblind --frozen-commit`). Candidate-screen rationale: [candidates.md](../dataset_candidates/candidates.md#a1-chbmp--iq-in-adults-with-full-scale-wais-iii).

## 1. Question

Does the character of coupling differ with IQ in resting adults, at matched channel spectra, alpha frequency and coupling strength? Character here means form, timing, phase lag, band or conditioning.
- **Primary readout:** the features themselves. If many important SPI–SPI features contrast the same kind of assumption, e.g. zero-lag against lagged dependence, that names the character difference. Without bespoke construction, that question could not be asked.
- **Prior:** one specific prediction comes from [Thatcher, North & Biver 2005](https://doi.org/10.1016/j.clinph.2005.04.026): higher IQ goes with shorter frontal and longer posterior phase delays (abstract-level claim, adapted here, not a replication of their measures).
- **Not the goal:** classification performance.

## 2. Data, cleaning and the window contract

**Data.** CHBMP ([Valdes-Sosa et al. 2021](https://doi.org/10.1038/s41597-021-00829-7)), Synapse syn28129311 (BIDS, 279 participants).
- IQ: WAIS-III FSIQ (Cuban norms), `WAIS_III.csv` (syn22334581, columns Code, FSIQ, VIQ, PIQ and subtests).
- Age, sex and education come from `Demographic_data.csv`; `participants.tsv` holds sex only.
- Age is 18–68, but 206 of 282 are 20–40 and 13 are over 50, so CHBMP is not an age cohort.

**Recording facts** (blind scoping, [scope.py](scope.py); verified on the first recordings, cohort figures in the run root):
- 200 Hz, common reference, hardware 0.5–50 Hz with a 60 Hz notch.
- Two montages: 62 EEG channels, or 120 EEG plus 2 EOG and ECG.
- Protocol: about 10 min eyes closed, then alternating eyes open/closed, hyperventilation, recovery and photic stimulation.
- **No artefact annotations.** Events mark states only; some recordings carry `discontinuity` markers.
- **Labelling errors:**
  - The EDF types ECG/EOG as EEG; channels.tsv is correct.
  - In the 62-channel montage, "AF7"/"AF8" are not scalp EEG: on sub-CBM00002 they have ~110 µV SD, correlate 0.90 with each other and 0.02 with every scalp neighbour. The common set is therefore 60 channels.

**Window contract** ([prepare.py](prepare.py)):
- **State:** the first eyes-closed block only, split at discontinuities. The later short eyes-closed epochs follow eye opening and differ in alpha reactivity.
- **Sampling rate, 100 Hz.**
  - Measured power above 45 Hz is about 10⁻⁴ of the in-band level, so anti-aliased resampling to 100 Hz (Nyquist 50 Hz) after the 0.5–45 Hz band-pass loses nothing.
  - 50 Hz would discard 25–45 Hz.
  - pyspi's bands are fixed fractions of fs, so at 100 Hz low = 0–25 Hz and high = 25–50 Hz (mostly beta2 and muscle in eyes-closed EEG). One lag step is 10 ms, and lags 1–20 cover 10–200 ms.
  - 100 Hz also matches the LEMON arm (§7).
- **Channels:** the 19 canonical 10–20 channels, unless the sweep shows the 60-channel set changes z.
  - The case for 19: comparability with LEMON, a 10.6× lower pyspi cost, and dense common-reference scalp channels being highly redundant (volume conduction).
- **Artefacts:** 1 s blocks with peak-to-peak > 150 µV on any selected channel are rejected, and windows lie inside clean runs.
- **Bad channels:** a recording is excluded if a selected channel is bad (median 1 s SD over 3× or under 1/5 of the recording's median channel, flat, or isolated). There is no interpolation.
- **Ocular contamination:** slow frontal eye movements below 150 µV remain. They are handled as a sensitivity analysis (H1 without Fp1/Fp2), not by ICA.
- **Window length T (pending the sweep).**
  - Design: nested windows at T = 1000, 2000, 4000 and 6000 on 12 subjects (6 per montage), 4 windows each, plus 60 channels at T = 2000 ([sweep.py](sweep.py)).
  - Criterion: the median reliability of the subject-mean z at a fixed clean-data budget, then cost. T is the shortest length within 0.05 of the best. T > 5000 is used only if it wins on that criterion.
  - The count follows: as many windows as about 120 s of clean data allows, capped by the 10th-percentile recording's clean runs.

**Exclusions**, decided before WAIS is joined: no eyes-closed block, a bad selected channel, or fewer than the required clean windows. Then only finite FSIQ.

**Power:** N = 150 detects |ρ| ≥ 0.23 at 80% (two-sided α = .05). A null is reported as such.

## 3. Analyses

Associations are partial Spearman correlations with FSIQ, on subject means over windows. Covariates: age, sex, montage, individual alpha frequency and aperiodic slope. Tests are two-sided permutations (10,000 permutations).

- **H1 (primary, confirmatory).**
  - Per window, compute the phase-lagged share icoh/√cohmag (broad band) in frontal pairs {Fp1, Fp2, F7, F3, Fz, F4, F8} and posterior pairs {P7, P3, Pz, P4, P8, O1, O2}.
  - Readout: D = posterior − frontal share, residualised on the regional coherence magnitudes.
  - Prediction: ρ(D, FSIQ) > 0.
- **H2 (secondary).** The CORE readouts against FSIQ, Holm-corrected.
- **E1 (feature-level discovery; the main interpretive analysis).**
  - Test every z feature against FSIQ, with the family-wise error rate from the max-|ρ| permutation null.
  - Group the top 100 by what separates their two SPIs:
    - a named minimal-pair axis, when the pair is one;
    - a within-family parameter (lag, band, history, aggregation);
    - otherwise the estimator-family pair.
  - Test each group's count against the same permutations (enrichment).
  - Groups summarise; they do not veto. A coherent set of strong features is reported on its own merit, whether or not an axis agrees.
- **E2 (fine-level interrogation of the top features).**
  - Edge-profile scatter (SPI a against SPI b over channel pairs) for the low and high FSIQ tertiles.
  - Leave-one-channel-out stability of the association.
  - Correlation with alpha frequency and aperiodic slope.
  - Any E1/H1 positive is then re-run on IAAFT surrogates. It is called non-spectral only if the surrogate association is absent.
- **H3 (secondary, expected null).** Ridge prediction of FSIQ (5 × 10-fold CV): covariates; + band power; + Pearson; + z (PCA ≤ 50 within folds).

**Weighting.** A positive that survives E2 (sign-stable on leave-one-channel-out, not explained by spectra, surviving surrogates) is weighted heavily even if modest in size. Small negatives in secondary analyses are not over-read.

## 4. Sensitivities and controls

- Education as an added covariate. It correlates with IQ, so it absorbs real IQ variance and is reported beside the primary result, not in place of it.
- H1 without Fp1/Fp2 (ocular contamination).
- Split-half reliability of D and of each core readout, reported before unblinding.
- Alpha frequency and aperiodic slope come from the first eyes-closed block of the scope; the slope is a log-log fit at 2–40 Hz excluding 7–14 Hz.
- Vigilance: the alpha/theta course over the block is shown in the scope. Windows are spread over the whole block.

## 5. Order of operations

1. Download (done; 2-subject smoke first).
2. Run the blind scope and sweep.
3. Fix T, the count and the channel set in §2.
4. Prepare the cohort.
5. Compute pyspi r10 through the `character_axes.launch` chain (measurement, sized farm, bank).
6. Freeze: commit this file, record the hash.
7. Unblind once and run `analysis.py`.
8. Run IAAFT surrogates for positives.

Deviations are reported as deviations.

## 6. Decisions fixed from scoping

- State: the first `eyes_closed` → `eyes_open` interval, split at `discontinuity` markers.
- No artefact labels exist, so the amplitude rule in §2 is used.
- Both montages carry all 19 canonical channels, each on its own common reference.
- `derivatives/` is not used.

## 7. Optional replication arm

LEMON young (existing r10 corpus at 100 Hz, T = 2000, N = 91 under the 19-channel rule), with the same H1 and E1 against LPS-2 and WST. It is labelled underpowered in advance. If CHBMP fixes T ≠ 2000, LEMON is re-prepared at that T for this arm.
