# CHBMP IQ: pre-registered analysis plan (frozen, 5 November 2026)

Status: **frozen** on 5 November 2026, after the window length was fixed from the sweep (§2) and the cohort bank was built, before WAIS was joined. It was re-frozen once, the same day, because the first unblinded run stopped while parsing `WAIS_III.csv` (38 blank trailing rows) before any association was computed; the fix only drops rows without a Code. The commit hash of this file and of [analysis.py](analysis.py) is recorded in the run root, and only then is WAIS joined (`--unblind --frozen-commit`). Candidate-screen rationale: [candidates.md](../dataset_candidates/candidates.md#a1-chbmp--iq-in-adults-with-full-scale-wais-iii).

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

**Recording facts** (blind cohort scope of all 253 recordings, [scope.py](scope.py); records and figures in `<run root>/scope/`):
- 200 Hz, common reference, hardware 0.5–50 Hz with a 60 Hz notch.
- 253 of 279 participants have an EEG recording (26 have none on Synapse; listed in the download receipt).
- Two cap families, with varying EEG channel counts: 58–62 channels (174 recordings) or 115–123 (79), the latter with EOG and ECG. All 253 carry the 19 canonical channels. The montage covariate is the cap family, not the channel count.
- Protocol: about 10 min eyes closed, then alternating eyes open/closed, hyperventilation, recovery and photic stimulation.
- **No artefact annotations.** Events mark states only; some recordings carry `discontinuity` markers.
- **Labelling errors:**
  - The EDF types ECG/EOG as EEG; channels.tsv is correct.
  - "AF7"/"AF8" are bad in 73 of 174 recordings on the 62-channel cap. On sub-CBM00002 they have ~110 µV SD, correlate 0.90 with each other and 0.02 with every scalp neighbour, which suggests a non-scalp input in those slots. They are dropped, so the common set is 60 channels.
  - "DC1" is typed EEG in channels.tsv but is a DC input; it is never selected.
  - sub-CBM00203's EDF labels channels '-A12' (linked ears) while every other recording says '-REF' and its sidecar says 'common'. Its reference cannot be verified, so it is excluded.

**Window contract** ([prepare.py](prepare.py)):
- **State:** the first eyes-closed block only, split at discontinuities. The later short eyes-closed epochs follow eye opening and differ in alpha reactivity.
- **Sampling rate, 100 Hz.**
  - Measured power above 45 Hz is about 10⁻⁴ of the in-band level, so anti-aliased resampling to 100 Hz (Nyquist 50 Hz) after the 0.5–45 Hz band-pass loses nothing.
  - 50 Hz would discard 25–45 Hz.
  - pyspi's bands are fixed fractions of fs, so at 100 Hz low = 0–25 Hz and high = 25–50 Hz (mostly beta2 and muscle in eyes-closed EEG). One lag step is 10 ms, and lags 1–20 cover 10–200 ms.
  - 100 Hz also matches the LEMON arm (§7).
- **Channels:** the 19 canonical 10–20 channels. The 60-channel T = 2000 arm was planned as a reported sensitivity, not a gate. It was not farmed: one 60-channel recording peaks at about 3.9 GB, above a normal-node core share, so it needs multi-core tasks. It is deferred.
  - The case for 19: comparability with LEMON, a 10.6× lower pyspi cost, and dense common-reference scalp channels being highly redundant (volume conduction).
  - The 27 recordings on 58–60-channel caps carry only 58 of the 60-channel set, so 60 channels cohort-wide would exclude them.
- **Artefacts:** 1 s blocks with peak-to-peak > 150 µV on any selected channel are rejected, and windows lie inside clean runs.
- **Bad channels:** a recording is excluded if a selected channel is bad (median 1 s SD over 3× or under 1/5 of the recording's median channel, flat, or isolated). There is no interpolation.
- **Ocular contamination:** slow frontal eye movements below 150 µV remain. They are handled as a sensitivity analysis (H1 without Fp1/Fp2), not by ICA.
- **Window length T = 1000 (10 s), up to 24 windows (240 s), at least 12.**
  - Design: nested windows at T = 1000, 2000, 4000 and 6000 on 11 subjects (one of the 12 had a bad Fp1), 4 windows each, plus 60 channels at T = 2000 ([sweep.py](sweep.py)).
  - Criterion: the median reliability of the subject-mean z at a fixed clean-data budget, then cost. T is the shortest length within 0.05 of the best. T > 5000 is used only if it wins on that criterion.
  - Result (T = 1000 and 2000; 24,058 features finite in ≥ 90% of windows in both arms):
    - Single-window ICC(1) rises only from 0.50 to 0.57 when T doubles, so at a fixed budget shorter windows win: subject-mean reliability is 0.93 against 0.89 at 120 s, and 0.97 against 0.95 at 240 s.
    - T = 2000 is better in 28% of features, and by more than 0.05 in 13.5%. Every estimator family favours T = 1000 in its median; envelope correlation least reliably, at 0.78 against 0.61.
    - On the same nested window, z at T = 1000 and at T = 2000 correlate at a median of 0.97, so the shorter window reads the same SPI–SPI structure.
    - Cost per second of data is 55 core-s at T = 1000 and 96 core-s at T = 2000.
    - For T = 4000 to win at 120 s, its single-window ICC would have to exceed about 0.82; the observed rate of increase makes that implausible. The decision was made before the T = 4000 and 6000 arms finished; they are reported as a convergence check, not used to choose T. T = 6000 exceeded the 4 h pilot walltime on one recording and was dropped; T = 4000 is the only convergence arm.
    - Short windows fail more often on spectral VAR measures: DTF/PDC/dcoh are valid in 89% of windows against 95%, and nonparametric spectral GC in 73% against 86%. Failures are scattered across subjects. Subject means use the finite windows, so with 12–24 windows these SPIs survive; the extra cost of shorter windows is in them.
    - Not tested: T < 1000. At 5 s the 0.5 Hz lower edge spans under 3 cycles, and the validity losses above grow.
  - Count: compute at T = 1000 is cheap (about 1.6 KSU for the cohort), so up to 24 windows (240 s of ~600 s clean) are used rather than the 120 s first planned. A recording needs at least 12, so that the budget removes only 2 recordings rather than 10.

**Cohort scope results** (all 253 recordings, canonical channels):
- Every recording has a first eyes-closed block (median 604 s; 10th percentile 363 s, because a group of 62-channel-cap recordings have about 6 min).
- Clean seconds at 150 µV: median 600, 10th percentile 359.
- 30 recordings (12%) have a bad canonical channel: Fp1 14, Fp2 8, C3 6, O1 5, T7 4, others ≤ 2. With sub-CBM00203, 222 recordings remain.
- Of those, 218 hold six disjoint clean 20 s windows, 209 six 40 s windows, and 177 six 60 s windows.
- IAF median 10.0 Hz (5–95%: 8.75–12 Hz). The occipital alpha/theta ratio shows no systematic drift over the block, so windows spread over the whole block do not mix vigilance states by position.
- Power at 45–60 Hz is 3–6 orders below 10 Hz.

**Exclusions**, decided before WAIS is joined: no eyes-closed block, a bad selected channel, a non-cohort reference label, or fewer than the required clean windows. Then only finite FSIQ.

**Power:** 215 recordings pass the blind exclusions, but FSIQ is blank for 80 of them in `WAIS_III.csv` (115 of its 282 coded rows have no WAIS scores at all). The analysable N = 135 detects |ρ| ≥ 0.25 at 80% (two-sided α = .05, 5 covariates); the planning figure of N ≈ 200 assumed complete FSIQ. Checked blind: missingness is unrelated to age (median 31 against 31.5), cap family (missing in 39% on the 62-channel cap against 34% on the 120-channel cap) and education, and somewhat more common in women (43% against 35% in men). A null is reported as such.

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
  - Only a named axis pair or a single-parameter within-family difference isolates one assumption. Other pairs are grouped by estimator-family pair and marked compound.
  - Pairs whose two SPIs are both functions of the cross-spectral matrix are flagged linear-spectral: they cannot read non-Gaussian form.
  - Each group carries its known construction nuisance (fixed lag: frequency; linear–rank: marginal shape; band organisation: within-band location).
  - E1 adds coupling strength (mean and SD of |Pearson|) to the covariates.
  - Each top feature is also re-tested with the first five principal components of channel band power added. Channel spectra absorbed z's predictive gain in every earlier cohort.
- **E2 (fine-level interrogation of the top features).**
  - Edge-profile scatter (SPI a against SPI b over channel pairs) for the low and high FSIQ tertiles.
  - Leave-one-channel-out stability of the association.
  - Correlation with alpha frequency and aperiodic slope.
  - Any E1/H1 positive is then re-run on IAAFT surrogates. It is called non-spectral only if the surrogate association is absent. IAAFT preserves covariance, precision and rank statistics but not coherence, phase or TE SPIs, so the real − surrogate residual is read only for the time-domain family. The uncorrected Kraskov estimators are autocorrelation-biased; read the DCE variants.
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
