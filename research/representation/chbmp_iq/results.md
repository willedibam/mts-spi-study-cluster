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

## LEMON age replication: result (`lemon-age/lemon-age.{json,png}`)

The result is **not replicated**: the pre-specified composite goes the other way.

| | base design | + 5 spectral PCs |
|---|---|---|
| T1 sign concordance (targets, same-sign share) | −0.050, p = .79 (235, 22%) | −0.112, p = .98 (100, 14%) |
| T2 profile r over 30,384 features | −0.05, p = .59 | −0.20, p = .84 |
| T3 directed × undirected composite (216 features) | ρ = −0.16, two-sided p = .06 | ρ = −0.23, two-sided p = .008 |

- **LEMON has its own age signal, mostly spectral.**
  - 127 features pass FWER in the base design (max |ρ| 0.47 against 0.37); 1 survives with spectral PCs.
  - The leaders are DTW-barycentre estimator pairs and robust covariance × 20 ms lagged correlation. The latter has the same sign in CHBMP (ρ = −0.19 there, −0.43 here).
  - That agreement is not pre-specified and is hypothesis-generating only.
- **Reading:**
  - The CHBMP directed-against-undirected coherence contrast is not a cohort-general age marker.
  - In both cohorts z carries age information, but different features carry it, and most of it is shared with channel spectra.
  - No coupling-character age claim survives.
- **Differences that could explain the reversal (untested):**
  - CHBMP is on each cap's own reference, LEMON on an ICA-cleaned average reference. A common reference adds a shared zero-lag component that both coherence and VAR coherence pick up.
  - CHBMP has continuous age, LEMON two extreme age groups.
  - The feature itself sits near ceiling: cohmag against dcoh has z ≈ 0.91 with a between-subject SD of about 0.02. Its subject variation is therefore a small departure from an almost identical edge ordering.
- **Window length:** at 20 s × 6, LEMON T is adequate but not lossless (about 26% of each person's ≈ 470 s eyes-closed recording). The cost is reliability, not bias.
  - Median feature reliability of the 6-window mean is 0.85 (IQR 0.63–0.93); for the cohmag × dcoh feature it is 0.89.
  - Using all ≈ 23 windows would give about 0.96 (Spearman–Brown). That raises observable correlations by roughly 6%, more for the least reliable features.
  - A re-preparation at T = 1000 over all clean data would be the lossless choice, but it cannot turn the observed sign reversal (ρ = −0.23) into a replication: unreliability shrinks effects toward 0.
  - The 71 people excluded for a missing canonical channel (mostly T7/T8, then Cz) are the larger loss. Interpolating them would add artificial zero-lag coupling, so it was not done.
  - The figure `lemon-show/lemon-data.png` shows windows → SPI matrices → one z feature.

## Extreme FSIQ groups (post hoc; [extremes.py](extremes.py), `extremes/extremes.{json,png}`)

Top against bottom FSIQ fractions of the E1 sample were compared by Cohen's d on covariate-adjusted ranks. Nulls come from label splits within the extreme set (exact at 5%).

| fraction (n per group; FSIQ) | z max \|d\| (null 95%) | z RMS-d p | spectra RMS-d p | Pearson RMS-d p |
|---|---|---|---|---|
| 5% (7; 71–79 against 123–135) | 4.07 (4.42) | .64 | .78 | .23 |
| 10% (14; 71–83 against 119–135) | 1.31 (2.16) | .96 | .69 | .01 |
| 25% (34; 71–93 against 112–135) | 0.74 (1.16) | .98 | .35 | .15 |

- **No stark difference** in z, channel spectra or Pearson at any fraction.
  - The leading 5% features have |d| ≈ 4. That looks large, but 7 against 7 over 31k features gives |d| ≈ 4.4 by chance.
  - The single p = .01 (Pearson, 10%) is one of 12 tests and is absent at 5% and 25%.
- **The groups are imbalanced, as expected:**
  - education is higher in the high group;
  - at 5%, all 7 high-IQ subjects are on the 62-channel cap, against 4 of 7 low.
  - Adjustment covers cap but not education, as in E1.
- Extreme-group designs inflate d relative to the population effect (Preacher et al. 2005, *Psychol Methods* 10:178; recalled). They add nothing that the continuous E1 test lacked.

## Reproduce

```bash
qsub -v RUN=<run root>,CORPUS=cohort-canonical-T1000,FROZEN=ede7430c8deccc5d29d3f9a62941da603880ec43 research/representation/chbmp_iq/jobs/analysis.pbs
python -m research.representation.chbmp_iq.controls --root <run root>/cohort-canonical-T1000 --scope <run root>/scope --wais <run root>/raw/cognitive/WAIS_III.csv --out <run root>/cohort-canonical-T1000/controls
python -m research.representation.chbmp_iq.sweep --sweep <run root>/sweep
```

The frozen analysis must run at `ede7430`; the corrected one runs at `cde4fd8` or later. Each script takes 1–3 min on 4 cores with about 13 GB of memory. Figures: `analysis-postfix-cde4fd8/iq-{features,top-features,h1-embedding}.png` and `sweep/analysis/sweep.png`.

```bash
python -m research.representation.chbmp_iq.lemon_age --chbmp-root <run root>/cohort-canonical-T1000 --scope <run root>/scope --lemon-root <empirical>/character-axes-261102/lemon --out <run root>/lemon-age
python -m research.representation.chbmp_iq.extremes --root <run root>/cohort-canonical-T1000 --scope <run root>/scope --wais <run root>/raw/cognitive/WAIS_III.csv --out <run root>/cohort-canonical-T1000/extremes
```

Each takes 8–25 min on 4 cores with under 10 GB of memory. The LEMON data figure comes from an ad hoc viewer, `<run root>/operations/adhoc/show_lemon.py`.
