# Named character axes: protocol (fixed 2 November 2026, before cohort features exist)

## Why axes, and what they can read

A single pairwise statistic reports coupling strength under the assumptions it fixes, so it cannot place a coupling on those assumptions: Pearson correlation alone cannot say whether a dependence is linear. Two statistics that differ in exactly one assumption can. Across the channel pairs of one recording, their edge-profile correlation stays high when that property is constant or tracks shared strength, and falls when it varies between pairs independently of strength. Each axis in [`src/mtsspi/character_axes.py`](../../../src/mtsspi/character_axes.py) is a set of such minimal pairs from the p90 catalogue; its **organisation** score is their mean profile correlation.

Organisation reads heterogeneity of a property across pairs, not its level: equal slowing of every connection barely moves it. Where both statistics share units, a **level** score (median over ordered entries of their contrast) reads typical magnitude instead. Organisation stays inside the z contract; levels are a bridge toward the future per-edge character estimators that replace the correlation aggregator.

| Axis | From | To | Minimal pairs (examples) | Main confound |
|---|---|---|---|---|
| form_monotone | linear | monotone nonlinear | Pearson vs Spearman/Kendall, zero and lagged | heavy-tailed marginals |
| form_information | Gaussian | non-Gaussian dependence | Gaussian vs Kraskov/kernel MI, TLMI; Gaussian GC vs nonparametric TE; r² vs dCor | estimator bias at short T |
| timing_fixed_lag | instantaneous | lagged at fixed delays | r(0) vs r(τ), MI vs TLMI | dominant frequency, cos(2πfτ) |
| timing_best_shift | zero shift | best rigid shift | r(0) vs max cross-correlation; Euclidean vs min-shift Euclidean | |
| timing_warping | rigid shift | elastic (variable lag) | min-shift Euclidean vs DTW variants; barycentre maxima | |
| phase_lag | zero-phase | phase-lagged | |coherency| vs imaginary coherence; PLV vs PLI/wPLI; PPC vs debiased squared PLI; PEC vs orthogonalised PEC | volume conduction |
| predictive | contemporaneous | Granger-type | MI vs GC/TE; coherence vs spectral GC | |
| history | short | long history | GC k=1 vs optimised history; TE k=1 vs 2; spectral GC order 1 vs 20 | |
| mediation | marginal | direct | covariance vs precision; DTF vs direct DTF; directed vs partial directed coherence | montage rank, fixed M |
| band | low band | high band | each spectral SPI at 0–fs/4 vs fs/4–fs/2 | |
| spectral_peak | broadband | peaked | mean vs maximum over frequency | |
| phase_amplitude | phase | envelope | PLV/PPC vs envelope correlation | waveform shape |
| tail | bulk | outlier-robust | empirical vs minimum-covariance-determinant covariance; Euclidean vs Chebyshev/cityblock | |

Considered and not built, by the criterion that one statistic alone cannot read the property:

- Symmetric ↔ directed: one directed statistic already reads asymmetry from A against Aᵀ, so this is a single-statistic property; contemporaneous ↔ predictive is the paired version that one statistic cannot read.
- Positive ↔ mixed-sign coupling: a single signed statistic reads sign.
- Amplitude ↔ shape similarity (Euclidean distance versus correlation): an identity on standardized inputs; it needs unstandardized inputs to exist.
- Stationary ↔ intermittent: p90 statistics are whole-window, so the tail axis is only a proxy; a real axis needs time-resolved statistics.
- Within- ↔ cross-frequency and pairwise ↔ higher-order: no minimal pair exists in p90 (no phase–amplitude or synergy estimator with a matched counterpart).

Controls: an identity set (statistics that are fixed functions of each other on standardized inputs; |score| ≈ 1 sets the numerical floor) and an estimator set (shrinkage, DCE, barycentre optimiser variants), which flags estimator artefacts rather than character.

Catalogue facts found while building the axes: 16 statistics (frequency-averaged PLI and wPLI, phase, PSI, group delay) are signed and antisymmetric, so their stored z with any of the 181 symmetric statistics is exactly zero; axes use |A| for them. Barycentre means are structurally ~0 on standardized inputs and are excluded.

## Surrogate control

[`src/mtsspi/surrogates.py`](../../../src/mtsspi/surrogates.py): one multivariate IAAFT surrogate per window (common phase rotation per frequency, exact marginals). It preserves cross-spectra, so linear lag, coherence and spectral organisation survive; non-Gaussian, phase–amplitude and intermittent structure do not. For every effect we report real, surrogate and real − surrogate. An effect present in surrogates is linear-spectral character (still not strength alone); an effect only in real − surrogate is non-Gaussian character.

## Cohorts and fixed predictions

Common window contract: the 19 standard 10–20 channels, native reference, 0.5–45 Hz, 100 Hz, per-channel standardization, windows within contiguous data, no interpolation (participants lacking a canonical channel are excluded and counted). Independent unit: participant.

- **AD/FTD/control** ([ds004504](https://openneuro.org/datasets/ds004504), 88 people, derivatives, eight 20 s windows). Predictions for AD versus control: spectral slowing (positive control); lower non-Gaussian dependence level (Jeong 2004; Stam 2005); lower phase-lag level, i.e. less lagged coupling, mainly reported in alpha (Stam et al. 2007). FTD contrasts are exploratory. Confound: ASR/ICA cleaning may differ by group; raw recordings are the sensitivity check.
- **LEMON age** (preprocessed eyes-closed, aged 20–35 versus 55–80, six 20 s windows). Predictions for old versus young: alpha slowing (positive control); lower timing_best_shift organisation and higher best-shift level (longer, more variable delays); higher phase-lag level. Fixed-lag readings are frequency-confounded and are interpreted only alongside best-shift and spectra.
- **Propofol** (Chennu et al. 2016, 20 people, four levels, ten 10 s epochs per level). Predictions for moderate versus baseline: frontal alpha increase (positive control); reduced frontal→parietal directed asymmetry (Lee et al. 2013), tested at edge level with phase-slope index and transfer entropy on frontal→parietal edges because z discards channel identity; recovery closer to baseline than moderate.
- **NeuroTycho surrogates**: the ketamine-medetomidine → propofol sign replication found on 2 November (more non-Gaussian dependence, less phase-lagged coupling, weaker tail agreement under anaesthesia) is tested against IAAFT.

## Evaluation

Five-fold participant-grouped, class-stratified cross-validation repeated five times; participant-level predictions average window probabilities. Feature blocks (absolute band power, relative spectra, Pearson, z, axes, axes+levels) are imputed, scaled and PCA-capped at 32 inside each training fold and fused by concatenation into L2 logistic regression with C selected by inner grouped CV. Primary contrasts: axes(+levels) versus Pearson; Pearson+axes versus Pearson; power+axes versus power. Uncertainty: participant bootstrap of the paired out-of-fold predictions. Interpretation uses participant-level axis means with the surrogate decomposition, not model weights.
