# CHBMP IQ: pre-registered analysis plan (draft, 5 November 2026)

Status: **draft, not frozen**. It freezes when §6's open decisions are filled from the blind inspection and this file's commit hash is recorded in the run root. WAIS is joined only after that. Candidate-screen rationale: [candidates.md](../dataset_candidates/candidates.md#a1-chbmp--iq-in-adults-with-full-scale-wais-iii).

## 1. Question

At matched channel spectra, alpha frequency and coupling strength, do higher-IQ adults organise lag differently across channel pairs? The specific prediction comes from [Thatcher, North & Biver 2005](https://doi.org/10.1016/j.clinph.2005.04.026): shorter frontal and longer posterior phase delays with higher IQ (abstract-level claim, adapted here, not a replication of their measures).

## 2. Data and sample

- **Data:** CHBMP ([Valdes-Sosa et al. 2021](https://doi.org/10.1038/s41597-021-00829-7)), Synapse syn28129311 (`ds_bids_cbm_loris_24_11_21`). EEG state A is 10 min eyes closed.
- **IQ:** WAIS-III full-scale IQ (FSIQ, Cuban-standardised), from `WAIS_III.csv` (syn22334581).
- **Contract:** that of the LEMON/AD-FTD/propofol cohorts ([loader](prepare.py)): 19 canonical 10–20 channels, own reference, 0.5–45 Hz, 100 Hz, six 20-s windows (T = 2000, M = 19) inside clean state-A data, no interpolation.
- **Exclusions,** decided before WAIS is joined:
  - a missing canonical channel;
  - fewer than six clean windows.
  - Then only those with a finite FSIQ enter. Expected N ≈ 142–167 (descriptor Table 6).
- **Power:** N = 150 detects |r| ≥ 0.23 at 80% power, two-sided α = .05. Smaller effects are not detectable, and a null is reported as such, not as absence.

## 3. Hypotheses

**H1 (primary).** Per window, compute the phase-lagged share |Im C|/|C| (the `lag_sin_phi` construction in `mtsspi.character_axes`, broad band) for each within-region pair.
- Frontal pairs are among {Fp1, Fp2, F7, F3, Fz, F4, F8}; posterior pairs among {P7, P3, Pz, P4, P8, O1, O2}.
- Readout: D = mean posterior share − mean frontal share, averaged over each participant's six windows.
- Strength match: D is residualised on the participants' mean frontal and posterior |C|. LEMON showed a phase-lagged-share effect can be lower coherence in disguise.
- Prediction: partial Spearman ρ(D, FSIQ) > 0.
- Covariates: age, sex, montage (64/120), individual alpha frequency, aperiodic exponent.
- Test: two-sided permutation, 10,000 permutations, α = .05.

**H2 (secondary).** The nine `CORE` readouts (organisation, share, spread and level) against FSIQ, with the same covariates and test. Holm correction across the nine.

**H3 (predictive, expected null).** Ridge prediction of FSIQ, with repeated grouped 10-fold CV (5 repeats) and all scaling and projection fitted within folds. Feature blocks:
- (a) absolute band power, 19 × 5;
- (b) Pearson profile + strength summaries;
- (c) a + b;
- (d) c + core;
- (e) c + z (PCA, at most 50 components).

Report R² gain of d and e over c, with paired bootstrap 95% CI. Prior cohorts predict no gain over channel spectra.

## 4. Covariates and controls

- **Individual alpha frequency:** the peak, 7–13 Hz, of the mean O1/O2/Pz Welch spectrum over the six windows.
- **Aperiodic exponent:** the slope of a log-log linear fit to the same spectrum over 2–40 Hz, excluding 7–14 Hz. No spectral-parameterisation package is in the environment, so this is a simple estimator, stated as such.
- **Surrogates:** any H1/H2 result with p < .05 is re-run on both IAAFT variants (marginal-final and spectrum-final, 1,000 iterations, `character_axes/prepare.py surrogates`). It is called non-spectral only if the surrogate effect is absent.
- **Reliability:** split-half reliability (windows 0–2 vs 3–5, Spearman–Brown) of D and of each core readout, reported before unblinding. A readout with reliability < 0.5 is declared uninformative in advance.

## 5. Order of operations

1. Download EEG and cognitive tables (`jobs/download.pbs`, copyq). A 2-subject smoke run comes first (`LIMIT=2`).
2. Blind inspection (`jobs/prepare.pbs` with `INSPECT`): sampling rates, references, channel names, event labels.
3. Fill §6 and commit.
4. Prepare the corpus, then run quality control (window counts, reliability).
5. Compute pyspi with the same p90/r10 chain as LEMON (`research.representation.character_axes.launch --root <run root> --stage pilot`).
6. Write the analysis script, commit it, and record this file's and the script's commit hash in the run root.
7. Join WAIS and run the analysis once. Deviations are reported as deviations.

## 6. Open decisions (fill from inspection, before WAIS is joined)

- Event column and the labels marking state A and technician-marked artefacts.
- Whether the 120-channel recordings carry all 19 canonical labels, and each montage's reference.
- Whether to use `derivatives/`; the default is no.

## 7. Optional replication arm

LEMON young (existing r10 corpus, N = 91 under the 19-channel rule). Same H1 readout against LPS-2 and WST separately. It is labelled underpowered in advance; a positive needs CHBMP agreement to be reported as a finding.
