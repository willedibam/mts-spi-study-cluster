# Axis screen: SPI–SPI character axes under anaesthesia beyond the MI pair (prereg)

Frozen 19 November 2026, before any output of this screen exists. Code: [`stats.py`](stats.py) (statistics, validated against pyspi), [`run.py`](run.py) (compute), [`score.py`](score.py) (rules below; written after this freeze and committed before any output is read), [`jobs/axis.pbs`](jobs/axis.pbs). Reading rules: [`docs/interpreting-spi-pairs.md`](../../../../docs/interpreting-spi-pairs.md). Seen before freezing: the atlas findings, the movers, matched and PRIOS screens and the MI-pair results, which name candidates (MI pair, Chebyshev against average distances, robust against classical covariance, `ppc|dswpli`, best lag, frequency locus). The rule below is the same for every pair, so no candidate is favoured by it.

## Why a new screen

- Earlier screens were at 2 s, where 187/272 SPIs are unreliable in PRIOS and lag and phase-lag SPIs fail reliability in the macaque. The MI anchor's strongest evidence is at 10 s. A full-pyspi 10-s IAAFT bank costs about 15 SU per window per bank (measured on stage A), about 39 kSU for real and surrogate, against 21.7 kSU left in ql44 this quarter. So the screen uses pyspi-free re-implementations of 33 atlas SPIs (528 pairs), with 3 IAAFT draws and 3 Gaussian nulls per window, all channels and all windows.
- Validation (`stats.py validate`, two raw 10-s windows, KTMD awake and PF drug, M = 126): every statistic's profile correlates ≥ .9992 with pyspi's stored MPI (exact for 25 of 33; multitaper coherence magnitude and high-band phase statistics differ by band-edge handling), and per-window z over all pairs is within .005 of pyspi's (median difference 0).

## Data

- Discovery: NeuroTycho 10-s arm, `neurotycho-long` (1,260 windows; stages A and B; 30 archives; M 126–127; 100 Hz).
- Secondary, same rule: the 2-s arm `neurotycho-short` (1,260 windows), to show which axes exist at the PRIOS window length.
- Prediction test: PRIOS S-full (`prios-ns`, 3,911 2-s epochs, 7 patients).

## Statistics and versions

- 33 statistics (`stats.KEYS`): Pearson r and r², Gaussian MI from r, partial correlation; Spearman and Kendall (signed and squared); MinCovDet covariance (signed, squared) and partial correlation; L1, L2, L∞ distances; distance correlation; envelope correlation (linear, log); multitaper coherence magnitude (0–25, 25–50, 0–50 Hz), imaginary coherence (0–25, 0–50), PLV, PPC and debiased squared wPLI (each 0–50, 0–25, 25–50); best-lag cross-correlation (signed and squared, pyspi threshold). On a 32-channel subset (as the anchor): the same plus KSG MI (k 4, DCE AUTO).
- Each window is normal-scored per channel. Versions: real; 3 spectrum-final joint IAAFT draws (seed `[extract.seed(id), 10, b]`); stationary, zero-phase and bias Gaussian nulls built from the group's (archive × state; sleep by within-archive delta tertile) normal-scored windows, as `discovery.null_bank`.
- Per window and version: z for every pair, z of double-centred profiles (c + a_i + a_j removed), node R². Per archive × state: between-window reliability (mean pairwise profile correlation) and z of window-mean profiles.

## Tiers and nets (fixed by the statistics, before data)

- **Higher-order statistics** (not fixed by the sample cross-spectrum): Spearman, Kendall, MinCovDet (3), L1, L∞, distance correlation, envelope correlations, KSG MI. **Linear** (Gaussian-parametric or spectral): r, r², Gaussian MI, partial correlation, L2, all multitaper statistics, cross-correlation.
- **F1, beyond linear:** pairs with at least one higher-order statistic. Net per window = real − mean of the 3 IAAFT draws.
- **F2, linear lag beyond floor:** linear pairs with at least one lag-sensitive member (imaginary coherence, dswpli, best-lag cross-correlation). Net = real − zero-phase.
- **F3, linear residual:** all linear pairs. Net = real − stationary (window-to-window or non-Gaussian structure the state's mean cross-spectrum does not give); bias = bias − stationary, reported, and named rows read net of bias.
- Linear pairs are not tested in F1: spectrum-final IAAFT reproduces their sample cross-spectra, so their net is degenerate (reported as a check, expected ≈ 0).

## Contrasts and units

- Session Δ = mean net (drug windows) − mean net (awake eyes-closed), within archive. Drug sessions: KT, KTMD, MD, PF (23 sessions, 4 animals). Animal mean = mean of its session Δ.
- Usable pair (blind to contrast): finite in ≥ 95% of windows of every drug-session state used; mean awake-eyes-closed real |z| < .98; both statistics' awake-eyes-closed reliability ≥ .3 (session mean). Pairs failing only reliability are reported descriptively.

## Screen (per family)

- Two-sided one-sample t over the 23 session Δ; BH q ≤ .05 within the family; every animal mean with the sign of the overall mean.

## Axis gates (a screened pair is an axis only if all hold)

1. Animal level: two-sided t(3) on the 4 animal means, p < .05 (|t| > 3.18), all 4 the same sign.
2. Size: |Δ| ≥ 3 null SD, null SD = SD over sessions of the within-awake split-half Δ (awake eyes-closed, last 5 minus first 5 windows by start time) divided by √2 (5-against-5 to 10-against-10).
3. Within-awake split-half: |mean split-half Δ| ≤ .25 |Δ|.
4. Eyes-open: |mean eyes-open Δ| ≤ .33 |Δ| (eyes open is a real change; it must be clearly smaller than the drug's).
5. Covariates: within-session OLS of window net on state plus covariates, on awake-eyes-closed and drug windows. F1: strength (mean |r|, SD |r|, covariance participation ratio) and burst index (median over channels of 1.4826·MAD/SD); the surrogate already matches each window's spectrum. F2, F3: strength plus spectral entropy, peak frequency and delta fraction. The adjusted Δ keeps the sign, is ≥ .5 |Δ|, and all 4 animal means share the sign.
6. Reliability: Δ of the disattenuated net (each version's per-window z divided by √(rel_A rel_B) of that archive × state and version) keeps the sign in all 4 animals.

## Reported for every axis and the top screened pairs (not gates)

Per agent; sleep (sleep − awake eyes-closed, 7 sessions, 2 animals; and the deep tertile) as a ratio of the drug Δ; recovery (recovery − awake eyes-closed) as a ratio; raw real Δ and surrogate Δ separately; double-centred Δ and node R²; mean-profile z Δ (linear pairs: a fall in per-window z that the mean-profile z does not show is noise); reliability change of each statistic; stage A against stage B signs; window-level correlation of the axis net with the MI-pair net (32 channels) within drug sessions; the 2-s arm under the same rule; minimal-pair class (`assumptions.classify`) and reading. For non-minimal axes: every chain A–B–C of minimal pairs within the statistic set, with Δ of each link's z and of the partial z(A, C | B) per window.

The MI anchor (Gaussian MI against KSG MI, 32 channels, F1) is scored under the same rules as a calibration of strictness; its result does not change the rules.

## Reading tier

- Beyond linear: an F1 axis whose pair is minimal and passes the gates.
- Linear: an F2 or F3 axis; it reads lag, its variability, or frequency locus per the guide.
- Literal: any non-minimal axis, read through its links.

## Prediction test on PRIOS (frozen now)

- Prediction set: every macaque axis (10-s gates passed) that is usable in PRIOS (rules above with PRIOS awake as reference and ≥ 20 windows per patient and state), each with its macaque sign. PRIOS patient Δ = mean net (anaes) − mean net (awake), the same nets. PRIOS windows differ in channels within a patient (stimulated electrodes left out per epoch), so its Gaussian nulls are built on the group's common channels and linear-tier PRIOS tests use those channels for real and null.
- **A (primary):** mean over the prediction set of sign-aligned patient-mean Δ, exact one-sided sign-flip over the 7 patients (128 patterns).
- **B:** per axis one-sided t in the macaque direction, Holm over the set.
- Control: PRIOS within-awake split-half (second half against first half in time), same statistic.
- If no macaque axis passes, A and B are not run; the top screened pairs' PRIOS Δ are reported descriptively.

## Pre-stated limits

- Four animals; session t treats sessions as replicates, so the screen is session-level and the axis gate adds the animal-level t.
- Drug and calendar are confounded within archives; eyes-open, split-half, sleep and recovery are the specificity checks.
- Pyspi-free statistics are validated on two windows; the statistic set is a subset (33) of the atlas's 272, chosen for readable minimal pairs.
- 2-s PRIOS against 10-s macaque: an axis absent in the macaque 2-s arm is not expected to transfer.

## Compute

Two PBS jobs, 48 cpus and 190 GB each: 10-s arm about 64 cpu-hours expected (walltime 8 h), 2-s arm about 15. Expected about 0.3 kSU together; worst case (both at walltime) about 1.2 kSU. PRIOS after the macaque score, about 0.1 kSU.

## Addendum, 20 November, after the macaque score and before any PRIOS compute

Macaque outcome that fixes the PRIOS set (from `score-long.json`, 10-s arm): 20 F1 axes, no F2 or F3 axis. Prediction set and signs, unchanged by this addendum: `r|mcd` −, `rho2|mcd2` −, `tau|mcd` −, `r2|mcd2` −, `mcd2|dcorr` −, `mcd2|xc2max` −, `mcd|l1` +, `tau2|mcd2` −, `pcorr|pcorr_mcd` −, `mcd2|ksg` −, `rho2|ksg` −, `r2|ksg` −, `dcorr|icoh_bb` +, `dcorr|ksg` −, `mi_g|ksg` −, `xc2max|ksg` −, `tau2|icoh_bb` +, `tau2|pcorr` −, `l1|icoh_bb` −, `pcorr|l1` +. Tests A and B and the split-half control are run on this set as frozen above.

- **PRIOS nulls (deviation).** The common-channel plan is infeasible: PRIOS leaves out stimulated electrodes per epoch, so the channels common to a whole patient × state group number 0–5 (28 in one group). Since no linear-tier axis passed, PRIOS needs only the per-window IAAFT net: versions are real and the 3 IAAFT draws on each window's own channels; KSG uses all of a window's channels. Reliability (usability rule) is the mean pairwise profile correlation over the channels both windows keep, within tasks of 40 consecutive windows (by run and onset). `run.py` macaque mode reproduces its earlier output exactly after this change (checked on a test bank).
- **Post hoc finding, before PRIOS: spectrum-final IAAFT is not degenerate for multitaper and best-lag statistics.** The prereg assumed it reproduces every linear statistic. It reproduces the sample covariance exactly (r, r², Gaussian MI, partial correlation, L2), but DPSS-tapered spectra mix neighbouring Fourier coefficients whose common phases IAAFT randomises, and zero-padded lagged correlation is not fixed by the circular periodogram. Linear–linear pairs among those statistics move by up to .08 under real − IAAFT with the drug (median .011). The guide already says IAAFT reproduces spectral and lag families poorly. So an F1 axis with a multitaper or xcorr member is not read as beyond-linear. **Secondary set A-valid** drops them (`mcd2|xc2max`, `xc2max|ksg`, `dcorr|icoh_bb`, `tau2|icoh_bb`, `l1|icoh_bb`): 15 axes, same test A.
- **Secondary 2-s prediction.** The 2-s arm (same rules) gives one new beyond-linear axis with a valid null that the 10-s arm does not: `l1|linf` (Chebyshev against L1), sign −. Tested in PRIOS by a one-sided t over the 7 patient Δ, alone.
- **Stated expectation.** The MinCovDet axes are absent in the macaque 2-s arm (M 126, T 200, where MCD's support of about 163 of 200 samples leaves it near-classical); PRIOS has M 34–68 at T 200, so MCD is less degenerate there, but transfer is not expected by the pre-stated limit. The MI-pair family was tested in PRIOS before (matched screen); it is in the set as frozen.
- Compute: one PBS job, 48 cpus, 3,911 windows × 4 versions; expected about 0.05 kSU, worst case (4-h walltime) 0.4 kSU.
