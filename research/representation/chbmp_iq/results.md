# CHBMP IQ: results (5 November 2026)

Plan: [protocol.md](protocol.md), frozen at `ede7430` before WAIS was joined. Run root: `/scratch/ql44/we2614/mts-spi-study/representation/empirical/chbmp-iq-261105/` (Gadi). pyspi env `mts-spi-v3-631de27`; 215 subjects, 5,109 windows of 10 s at 100 Hz, 19 channels.

## Answer

SPI–SPI coupling character in resting eyes-closed EEG shows no detectable association with full-scale IQ at N = 135. This is a pre-registered null, and the controls show it is not explained by noisy or individually uninformative features.

- **E1, all 31,415 finite z features:** 0 survive the max-|ρ| family-wise error rate. The largest |ρ| is 0.25 against a 95% null maximum of 0.36. No contrast group is enriched in the top 100; the best is linear × other information-theoretic, 20 against 6.1 expected, p = .07 uncorrected.
- **H1, posterior minus frontal phase-lagged share:** ρ = −0.02, p = .86. With education added, ρ = −0.07; without Fp1/Fp2, ρ = −0.01.
- **H2, core readouts:** all Holm p = 1.
- **H3, ridge prediction:** cross-validated R² < 0 for covariates, + spectra, + Pearson and + z.
- **E2, leading features:** they are stable under leave-one-channel-out, but none approaches significance, so they are not interpreted. One of the four shown (`bary_dtw_mean` × `wpli_max`) has wPLI pinned near 1, a saturation case.
- **IAAFT surrogates:** not run, because there is no positive to test.

## Deviations and corrections

- **Re-freeze:** the protocol was frozen at `98af2f7` and then re-frozen at `ede7430`, before any association was computed. The first unblinded run stopped while parsing 38 blank trailing rows in `WAIS_III.csv`.
- **N is 135, not ≈ 200:** FSIQ is blank for 80 of the 215 prepared subjects. That gives 80% power for |ρ| ≥ 0.25 per test, and about 0.36 is needed after correction over 31k features. Missingness, checked blind, is unrelated to age, cap or education and somewhat higher in women.
- **H2 `location_frequency`:** the frozen run reports p = 1e-4 with ρ = NaN. That entry is invalid: an undefined ρ was given p = 1/(perms + 1). Fixed in `cde4fd8`; the corrected rerun (`analysis-postfix-cde4fd8/`) is otherwise identical. The readout is undefined for 2 subjects (AR(2) pole fit; 474 of 5,109 windows). On complete cases, ρ = −0.10, p = .25.
- **Sweep convergence (protocol §2):**
  - T = 6000 timed out and the 60-channel arm needs multi-core tasks, so neither was used.
  - T = 4000 confirms T = 1000. Single-window ICC is 0.68 (above 0.82 was needed). Subject-mean reliability at 120 s is 0.87, against 0.93 at T = 1000.
  - On the same window, r(z at T = 1000, z at T = 4000) = 0.95.
  - Cost per second of data: 55, 96 and 180 core-s at T = 1000, 2000 and 4000.

## Post-hoc controls (not pre-registered; [controls.py](controls.py), `controls/controls.json`)

- **Reliability:** z is reliable and highly individual.
  - Odd against even windows, about 2 min of data each: median split-half reliability of the subject mean is 0.97, and 97% of features have half-to-half r > 0.5.
  - The odd-window profile identifies its own even-window half for 215 of 215 subjects.
  - Caveat: both halves come from one session, interleaved 10 s apart, so session- and cap-specific factors count as identity. Cross-session test–retest is unknown.
- **Age as a positive control (N = 215):**
  - With sex and cap family adjusted, 34 features survive FWER (max |ρ| 0.33, null 0.29).
  - Adding IAF, aperiodic slope and coupling strength, 235 survive (max 0.36).
  - The top features contrast VAR-based directed spectral measures (dcoh, gpdcoh, 0–25 Hz) with undirected coupling (coherence magnitude, PLV, PPC, shape and distance measures). Directed spectral × shape/distance pairs are 14 of the top 100 against 3.1 expected (enrichment p = .04 uncorrected).
- **Age after five spectral principal components are added:**
  - 0 features survive, and the maximum is at threshold (0.285 against 0.288, FWER p = .06).
  - The same contrast still leads: directed coherence against coherence, PLV and PPC, ρ > 0.
  - Within-family spectral estimator pairs are enriched: 7 against 0.5 expected, p = .008 uncorrected over 8 groups.
  - Reading: most of the age signal in z is shared with channel spectra. The remainder sits at the edge of detection.

## Interpretation

- **The IQ null is informative within its power.** z is reliable at the subject level (0.97) and carries detectable individual differences (age). Any IQ–character association is therefore weaker than |ρ| ≈ 0.25 per feature, and none is concentrated in a family that enrichment could detect.
  - This fits brain-wide association work showing that true brain–cognition correlations are small (|r| ≲ 0.1–0.2) and that N in the hundreds is underpowered for feature-wise discovery (Marek et al. 2022, *Nature* 603:654). The EEG–IQ coherence literature is mixed and largely unreplicated (literature status: recalled, not re-checked here).
- **The age result is the candidate positive.** It is not a pre-registered finding, and the framing matters.
  - What holds: with age, the across-edge profile of directed VAR coherence aligns more with that of undirected coherence and phase locking (ρ ≈ 0.3).
  - Inferred reading: in older adults, directed (Granger-type) spectral coupling becomes less specific relative to symmetric zero-lag coupling. That is consistent with more volume-conduction-like or common-drive structure, and with transfer entropy agreeing more with 30 ms lagged correlation with age (ρ = 0.35 before spectral adjustment; not in the top 15 after).
  - It is not established as coupling character beyond the spectrum. Adjusting for spectral PCs may over-adjust, because channel spectra partly reflect coupling, but the conservative reading is that the effect is borderline once spectra are held fixed.
- **Next test:** a replication in LEMON's young-against-old contrast (protocol §7 arm), with spectral PCs adjusted, before this is claimed.

## LEMON age replication: pre-specified (5 November, before any LEMON feature-level age statistic was computed)

- **Data:** the existing `lemon-age-r10-261102` bank (132 people, 41 aged 55–80 and 91 aged 20–35; 6 windows of 20 s at 100 Hz, 19 channels; same pyspi computation and feature schema). It is not re-prepared at T = 1000, because z correlates 0.95–0.97 between T = 1000 and longer windows on CHBMP.
  - LEMON is ICA-cleaned and average-referenced, while CHBMP uses each cap's own reference. That is a real difference for coupling measures, so a failure could be reference-specific.
  - LEMON age was used before (axes, prediction, spectral sufficiency), but not at this feature level.
- **Targets:** fixed by CHBMP alone in [lemon_age.py](lemon_age.py). The LEMON design is sex, IAF, aperiodic slope and strength, then + 5 spectral PCs of band power.
  - **T1:** the CHBMP `age_full` FWER features (235). Sign concordance in LEMON, one-sided against LEMON label permutations. For the spectra design, the targets are the CHBMP spectra-design top 100.
  - **T2:** the correlation between the two cohorts' age ρ over all common features.
  - **T3:** the directed × undirected spectral composite (dcoh/gpdcoh against cohmag/PLV/PPC), predicted ρ > 0.
- **Decision:**
  - Replicates if T1 (base design) p < .05.
  - Replicates beyond spectra only if T1 and T3 both have p < .05 (one-sided) in the spectra design.
  - Anything else is reported as not replicated.

## Reproduce

```bash
qsub -v RUN=<run root>,CORPUS=cohort-canonical-T1000,FROZEN=ede7430c8deccc5d29d3f9a62941da603880ec43 research/representation/chbmp_iq/jobs/analysis.pbs
python -m research.representation.chbmp_iq.controls --root <run root>/cohort-canonical-T1000 --scope <run root>/scope --wais <run root>/raw/cognitive/WAIS_III.csv --out <run root>/cohort-canonical-T1000/controls
python -m research.representation.chbmp_iq.sweep --sweep <run root>/sweep
```

The frozen analysis must run at `ede7430`; the corrected one runs at `cde4fd8` or later. Each script takes 1–3 min on 4 cores with about 13 GB of memory. Figures: `analysis-postfix-cde4fd8/iq-{features,top-features,h1-embedding}.png` and `sweep/analysis/sweep.png`.
