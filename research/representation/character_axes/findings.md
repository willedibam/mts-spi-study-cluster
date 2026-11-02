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

**Corrections (2 November).** (1) Residual axes alone predicted held-out anaesthesia at AUROC .999, but IAAFT spectral mismatch is twice as large under anaesthesia (8.1% versus 4.1%) and alone predicts state at AUROC .976; not claimed. (2) The non-Gaussian dependence level is estimator-dependent and **withdrawn**: with Kraskov MI it rises (+.205 real, +.094 residual, 6/6), but with the autocorrelation-corrected Kraskov estimator (`mi_kraskov_NN-4_DCE-AUTO`, Theiler window) it falls (−.061 real, −.032 residual, 0/6; reproduced here after the [axis review](axes-review.md) found it). Slow waves raise autocorrelation, which inflates uncorrected Kraskov MI; IAAFT preserves spectra and cannot cancel it.

## Independent axis review

[axes-review.md](axes-review.md), a separate session's audit against pyspi source and eight banks. Main points: (a) organisation z is attenuated by edge-profile unreliability, which itself changes with state (NeuroTycho reliability rises under anaesthesia), so raw organisation effects shrink sharply once disattenuated (predictive d 3.36→0.75, history 2.79→0.53, phase locking 2.15→0.13; fixed lag and warping survive at 1.3); (b) 9 of 13 axes violate the one-assumption rule once direction, exponent, sign, frequency resolution and estimator bias are required to match, and some pairs are exact duplicates (`dtf` ≡ `pdcoh` bivariately, MCD ≡ EllipticEnvelope); (c) the identity control's nonlinear pairs move with strength range (d −1.2), so it is a strength-range yardstick, not a floor; (d) axes explain 16–40% of z variance, less than random pair groupings, as expected for targeted contrasts. Proposed core: form (Gaussian vs DCE-Kraskov MI/TLMI/TE), lag (|coherency| vs imaginary coherence), band (low vs high coherence), warp (Euclidean vs constrained DTW), plus conditional mediation (Ledoit–Wolf r² vs partial r²), with reliability and saturation gates and names fixed to one constituent set. Its selection used label-free reliability and saturation on these cohorts; its AUROC table looked at outcomes and must not drive choices. Its verdict: the defensible character is currently linear-spectral (lag, band, warp, mediation); no estimator-robust non-Gaussian demonstration exists yet.

## Open

- AD/FTD (88 people, 704 windows, no exclusions), LEMON (132 of 203 people; 70 excluded for a missing canonical channel, 34% young versus 37% old) and propofol (20 people × 4 levels, 800 windows) farms are running.
- Cohort surrogates need the cohort banks; both IAAFT variants should be used.
- InceptionTime is not yet in the cohort comparison; MiniRocket is.
