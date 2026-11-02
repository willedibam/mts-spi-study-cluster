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

**Correction to the residual result.** Residual axes alone predicted held-out anaesthesia at AUROC .999/BA .992, but IAAFT spectral mismatch is twice as large under anaesthesia (8.1% versus 4.1%) and alone predicts state at AUROC .976; the residual MI level tracks it (ρ .83 pooled, .37–.42 within state). After regressing mismatch out within each unit (fitted on awake windows, so extrapolated), two effects survive in 6/6 units: non-Gaussian dependence level (d 2.37) and phase-lag organisation (d 1.86, carried by phase-locking rather than coherency statistics, since cross-spectra are preserved). Tail, predictive, mediation, spectral-peak and phase-lag-level residuals do not survive. Spectrum-final and 1000-iteration marginal-final surrogates (submitted 2 November) bracket the null; claims wait for both.

## Open

- AD/FTD (88 people, 704 windows, no exclusions), LEMON (132 of 203 people; 70 excluded for a missing canonical channel, 34% young versus 37% old) and propofol (20 people × 4 levels, 800 windows) farms are running.
- Cohort surrogates need the cohort banks; both IAAFT variants should be used.
- InceptionTime is not yet in the cohort comparison; MiniRocket is.
