# Named character axes: findings

Protocol and fixed predictions: [protocol.md](protocol.md). Code: [`character_axes.py`](../../../src/mtsspi/character_axes.py), [`surrogates.py`](../../../src/mtsspi/surrogates.py), [`evaluate_banks.py`](evaluate_banks.py), [`evaluate_cohorts.py`](evaluate_cohorts.py). Gadi root `/g/data/ql44/we2614/mts-spi-study/representation/empirical/character-axes-261102/`; frozen sources `operations/sources/character-axes-{c944ac6,3f224ad,ae7faed}`.

## Catalogue facts (2 November)

- 16 p90 statistics (frequency-averaged PLI and wPLI, phase, PSI, group delay) are exactly antisymmetric. Their stored z with the 180 symmetric statistics is identically zero: 2,880 of 39,903 coordinates (7.2%), verified to |z| ≤ 3e-15 on LANL with its own schema. Axes use |A|.
- Barycentre means are structurally ~1e-17 on standardized inputs, so their profiles are invalid everywhere.
- Inputs are standardized, so Euclidean distance is a fixed function of Pearson; identity controls sit at |r| ≥ 0.97 (Siena 0.92, plausibly its rank-14 bipolar montage).

## Existing banks (source-fit, held-out scored)

- NeuroTycho, ketamine–medetomidine sources → propofol targets, target animal excluded: spectra AUROC 1.0/BA .984; 13 axes .978/.914; axes+levels .983/.938; z 1.0/.836; Pearson .428/.57. Ten readouts share sign in all six animal–drug units.
- CHB-MIT → Siena: axes do not transfer (Siena axis-pair BA .60), like z. fMRI rest/movie: Pearson BA 1.0, axes .81. GRABMyo and HMC: per-channel features win.

## Surrogate decomposition

Multivariate IAAFT, one surrogate per window, marginal-final, 200 iterations. Effects are unit-level class contrasts; d is standardized over units.

- **LANL:** most contact effects reappear in surrogates (mediation held-out d +10.8 real versus +5.8 surrogate; predictive, phase–amplitude, best-shift similar), so they are linear-spectral. A smaller replicated non-Gaussian part: phase-lag level, spectral-peak level, predictive organisation.
- **CHB-MIT and Siena:** seizure effects are mostly linear-spectral (warping −1.01 real versus −0.99 surrogate). Three small residuals share sign in both cohorts: phase-lag +.47/+.36, predictive +.43/+.50, tail −.53/−.68.
- **NeuroTycho:** timing (fixed lag +.124 real versus +.127 surrogate; warping +.167 versus +.168), best-shift and spectral-peak levels are entirely linear-spectral: slow-wave reorganisation. Non-Gaussian dependence level rises +.205 nats real versus +.112 surrogate, residual +.094 (6/6 units, d 2.8).

**Corrections (2 November).** (1) Residual axes alone predicted held-out anaesthesia at AUROC .999, but IAAFT spectral mismatch is twice as large under anaesthesia (8.1% versus 4.1%) and alone predicts state at AUROC .976; not claimed. (2) The non-Gaussian dependence level is estimator-dependent and **withdrawn**. All three surrogate variants (marginal-final 200 and 1000 iterations, spectrum-final) agree to within .05 SD, so surrogate mismatch is not the driver; the estimator is: with Kraskov MI it rises (+.205 real, +.094 residual, 6/6), but with the autocorrelation-corrected Kraskov estimator (`mi_kraskov_NN-4_DCE-AUTO`, Theiler window) it falls (−.061 real, −.032 residual, 0/6; reproduced here after the [axis review](axes-review.md) found it). Slow waves raise autocorrelation, which inflates uncorrected Kraskov MI; IAAFT preserves spectra and cannot cancel it.

## Independent axis review

[axes-review.md](axes-review.md), a separate session's audit against pyspi source and eight banks. Main points: (a) organisation z is attenuated by edge-profile unreliability, which itself changes with state (NeuroTycho reliability rises under anaesthesia), so raw organisation effects shrink sharply once disattenuated (predictive d 3.36→0.75, history 2.79→0.53, phase locking 2.15→0.13; fixed lag and warping survive at 1.3); (b) 9 of 13 axes violate the one-assumption rule once direction, exponent, sign, frequency resolution and estimator bias are required to match, and some pairs are exact duplicates (`dtf` ≡ `pdcoh` bivariately, MCD ≡ EllipticEnvelope); (c) the identity control's nonlinear pairs move with strength range (d −1.2), so it is a strength-range yardstick, not a floor; (d) axes explain 16–40% of z variance, less than random pair groupings, as expected for targeted contrasts. Proposed core: form (Gaussian vs DCE-Kraskov MI/TLMI/TE), lag (|coherency| vs imaginary coherence), band (low vs high coherence), warp (Euclidean vs constrained DTW), plus conditional mediation (Ledoit–Wolf r² vs partial r²), with reliability and saturation gates and names fixed to one constituent set. Its selection used label-free reliability and saturation on these cohorts; its AUROC table looked at outcomes and must not drive choices. Its verdict: the defensible character is currently linear-spectral (lag, band, warp, mediation); no estimator-robust non-Gaussian demonstration exists yet.

## Cohorts

- AD/FTD (88 people, 704 windows; five-fold participant CV × 5): AD versus control AUROC power+Pearson .888, spectra .874, Pearson .835, z .837, MiniRocket .813, 13 axes .641. Pearson+axes+levels over Pearson +.045 [−.010, +.103]; power+axes over power ≈ 0 in all contrasts. AD versus FTD is near chance for every method (MiniRocket .43, spectra .47, axes+levels .60 with an interval spanning zero). Fixed predictions: spectral slowing confirmed (theta/alpha d 1.13); lower non-Gaussian level failed in the opposite direction (d +.71, uncorrected Kraskov, so now suspect for the same autocorrelation reason); phase-lag level null.
- LEMON (132 of 203 people, 41 old versus 91 young; 70 excluded for a missing canonical channel, 34% young versus 37% old): power+Pearson AUROC .946, MiniRocket .945, power .932, spectra .910, Pearson+axes+levels .902, z .874, Pearson .859, 13 axes .738. **Pearson+axes+levels over Pearson +.043 [+.011, +.077]**, the first interval excluding zero, but power+axes over power is −.004. Fixed delay predictions fail: best-shift organisation d −.16 and level +.09. The apparent rise in phase-lagged share (bounded |Im C|/|C| d +.68, all bands) vanishes at matched coherence (participant ANCOVA age t .6; zero difference in every |C| decile): it follows from lower coherence in older people, not separate timing change. Strength must be matched before any character reading.
- Propofol (20 people × 4 levels, 800 windows; within-participant CV): moderate versus baseline AUROC power .995, spectra .992, MiniRocket .975, z .930, axes+levels .922, Pearson .747; axes+levels over Pearson +.175 [+.052, +.307], Pearson+axes+levels over Pearson +.073 [+.008, +.157], power+axes over power −.003. Mild versus baseline is weaker (Pearson+axes+levels over Pearson +.052 [−.003, +.115]). The frontal→parietal prediction is not supported: baseline asymmetry is ≈0 (PSI .002, GC ≈0), moderate PSI change d −.17, GC shifts opposite to prediction (d +.68, 16/20). Recovery differs from moderate in PSI (d −.76, p .002) while plasma propofol remains elevated; exploratory. Lee et al. (2013) studied unconsciousness with dPLI, not responsive sedation.
- Propofol surrogates (marginal- and spectrum-final, 1000 iterations; identity floor .98): moderate-versus-baseline phase-lag organisation (+.66 real, +.87/+.53 surrogate), fixed-lag timing (−.46/−.48) and spectral-peak/band levels are linear-spectral; mild-sedation phase-lag (+.92) and warping (−.81) likewise. Residuals robust to both variants are modest and sit in axes the review flagged: linear–rank agreement −.83/−.82 (16/20 lower), tail −.69/−.54, uncorrected Kraskov level +.57/+.56. The band-level residual flips with variant (+.88/−.03), so it is mismatch.
- **Pattern across LEMON and propofol:** axes and z add reliably over correlation (strength under one lens) but not over per-channel amplitude or spectra. What z captures beyond correlation is largely mirrored in channel spectra. Both cohort surrogate variants are running.

## Open

- Whether to adopt the review's core (and DCE, reliability gating, strength-range control) as the reported axes; the 13 stay as the extended set if so.
- InceptionTime is not yet in the cohort comparison; MiniRocket is.
- Corpora moved to scratch after use: `/scratch/ql44/we2614/mts-spi-study/representation/empirical/character-axes-261102/{ad-ftd,surr-chbmit,surr-lanl,surr-neurotycho,surr-siena}` (checksum-verified moves, gdata quota).
