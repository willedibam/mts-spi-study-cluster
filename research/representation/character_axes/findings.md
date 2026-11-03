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
- AD/FTD surrogates (both variants, 1000 iterations; identity floor .96; IAAFT mismatch does not differ by group, p ≥ .19): AD's lower linear–rank agreement (form_monotone −.69, p .011) reappears in surrogates (−.63/−.65), so it is linear-spectral, as are the band, spectral-peak and fixed-lag effects. The uncorrected Kraskov level rise (+.71, residual +.62/+.61) vanishes with DCE-Kraskov (+.21, p .37; residual p .42): the non-Gaussian prediction is null with the corrected estimator. One residual survives both variants: tail level (Chebyshev/Euclidean, +.69 real, −.16/−.36 surrogate, residual +.75/+.75, p .009/.008; FTD +.14), i.e. larger peak cross-channel differences than linear structure with the same marginals predicts. It is one of 26 readouts (not significant after Bonferroni), in a review-flagged axis, and transients surviving ASR cleaning are the obvious confound; exploratory. The best-shift level is degenerate here (its median over edges is 0 in most windows because cross-correlation peaks at lag 0), so its −.26 is not interpretable.
- LEMON (132 of 203 people, 41 old versus 91 young; 70 excluded for a missing canonical channel, 34% young versus 37% old): power+Pearson AUROC .946, MiniRocket .945, power .932, spectra .910, Pearson+axes+levels .902, z .874, Pearson .859, 13 axes .738. **Pearson+axes+levels over Pearson +.043 [+.011, +.077]**, the first interval excluding zero, but power+axes over power is −.004. Fixed delay predictions fail: best-shift organisation d −.16 and level +.09. The apparent rise in phase-lagged share (bounded |Im C|/|C| d +.68, all bands) vanishes at matched coherence (participant ANCOVA age t .6; zero difference in every |C| decile): it follows from lower coherence in older people, not separate timing change. Strength must be matched before any character reading.
- LEMON surrogates (both variants, 1000 iterations; identity floor .92): every age effect is linear-spectral. Fixed-lag (−.52, surrogate −.52/−.52), history (+.41, +.43/+.39), band (−.68, −.50/−.66), spectral-peak (−.59), |sin φ| share (+.68, +.67/+.69) and even the uncorrected Kraskov level (−.72, −.73/−.66) reappear in surrogates. Residuals present only under marginal-final (tail +.55, band and spectral-peak levels) vanish under spectrum-final, i.e. mismatch, which differs slightly by age (spectrum-final marginal error .122 versus .129, p .004). The one residual robust to both is form_information organisation (−.40/−.50, p .029/.006), driven by a single pair, Gaussian GC versus kernel TE, whose profiles are barely correlated even in real data (mean z .08), so it is an unreliable readout, not a non-Gaussian finding. The remaining form pairs are saturated (z ≈ 1.00).
- Propofol (20 people × 4 levels, 800 windows; within-participant CV): moderate versus baseline AUROC power .995, spectra .992, MiniRocket .975, z .930, axes+levels .922, Pearson .747; axes+levels over Pearson +.175 [+.052, +.307], Pearson+axes+levels over Pearson +.073 [+.008, +.157], power+axes over power −.003. Mild versus baseline is weaker (Pearson+axes+levels over Pearson +.052 [−.003, +.115]). The frontal→parietal prediction is not supported: baseline asymmetry is ≈0 (PSI .002, GC ≈0), moderate PSI change d −.17, GC shifts opposite to prediction (d +.68, 16/20). Recovery differs from moderate in PSI (d −.76, p .002) while plasma propofol remains elevated; exploratory. Lee et al. (2013) studied unconsciousness with dPLI, not responsive sedation.
- Propofol surrogates (marginal- and spectrum-final, 1000 iterations; identity floor .98): moderate-versus-baseline phase-lag organisation (+.66 real, +.87/+.53 surrogate), fixed-lag timing (−.46/−.48) and spectral-peak/band levels are linear-spectral; mild-sedation phase-lag (+.92) and warping (−.81) likewise. Residuals robust to both variants are modest and sit in axes the review flagged: linear–rank agreement −.83/−.82 (16/20 lower), tail −.69/−.54, uncorrected Kraskov level +.57/+.56 (with DCE-Kraskov +.17/+.07, so estimator-driven as on NeuroTycho). The band-level residual flips with variant (+.88/−.03), so it is mismatch.
- **Pattern across LEMON and propofol:** axes and z add reliably over correlation (strength under one lens) but not over per-channel amplitude or spectra. What z captures beyond correlation is largely mirrored in channel spectra. In every cohort, the uncorrected Kraskov non-Gaussian level shifts and the DCE-corrected one does not. Across the four cohorts with both surrogate variants (NeuroTycho, propofol, AD/FTD, LEMON), every replicated state effect on the named axes is linear-spectral. The earlier marginal-final-only residuals (LANL; CHB-MIT and Siena) are untested against spectrum-final, and in LEMON and propofol such residuals vanished under it. The residuals robust to both variants (propofol rank agreement and tail, AD tail level, LEMON GC versus kernel TE) are isolated, modest and sit in review-flagged or unreliable readouts. The defensible character is therefore linear-spectral (lag, band, spectral shape, conditioning), and a non-Gaussian claim would need a reliable, DCE-corrected form readout validated by construction.

## Construction validation (2 November, Gadi)

Record: [validation/README.md](validation/README.md) (Results) and `validation/tables/`; pre-registered criteria, Gadi job 182947411 at 8d239c6, 4,716 windows, no pyspi errors, analysed by the review session.

- **Core by construction** (selectivity ratio SR ≥ 2, sign-stable): band (ppc L|H organisation SR 2.06, high-share level 3.74) and mediation (cov² versus prec², SR 3.25/2.88). Lag is a sub-axis: its sign holds across three estimator families, but SR is 0.66–1.60, band level moves coherency lag pairs more than lag dispersion does, and it is insensitive in the Kuramoto family.
- **Stage 2 (second families, 3 November; README "Stage 2 results"):**
  - Direction (spectral GC A versus Aᵀ, SR 3.12 in VAR) is dropped because it reverses in directed Kuramoto. In family A it is selective (SR 2.28 after the sign correction; the earlier 0.96 was an artefact).
  - Mediation stays core only for strong instantaneous mediation. With lagged, partial mediation, lag level, noise and mixing swamp it.
  - Band spread (SD of each edge's high-frequency coherence share over the strongest quarter of edges) passes as core (SR 5.3) and is level-aware: band level moves it under 0.1 times as much as band dispersion. One generator.
  - Lag spread is not level-aware.
- **Stage 3 (within-band location and joint bursts; README "Stage 3 results"):**
  - No new readout passes as core.
  - The AR(2) pole frequency of each strong edge's lag-0 to lag-3 correlations is calibrated (it recovers a 0.10 cycle-per-sample shift to within 0.003) and selective (SR 5.9). But channel spectra move it 0.36–0.53 times as much as coupling frequency does, so it is a sub-axis, reported beside the channel peak-frequency shift.
  - Band organisation (ppc L|H) also reads within-band location (0.77 of its own response), so it is now a sub-axis; band high share and band spread do not (≤ 0.07). Fixed-lag readouts track location level strongly (post hoc).
  - Every raw linear–rank and tail readout is dominated by marginal shape (β moves them 1.2–6.6 times as much as joint bursts), and IAAFT keeps that effect. Only real − IAAFT zero-lag Pearson|Spearman organisation reads joint bursts (d −3.2/−2.5, β ≤ 0.24). Propofol's real − IAAFT linear–rank decrease therefore fits more joint-burst coupling (inferred); its raw readout cannot support that.
  - Design flaw found: stage 1's band, form and warp dispersion knobs reused the phase-dispersion sign vector, so those dispersions were perfectly correlated with phase across edges. The correction (1,596 windows, signs balanced within each phase group; README "Correction results") leaves CORE unchanged:
    - Band spread: SR 5.41. High share: SR 3.74. No lag verdict changes.
    - ppc L|H stays a sub-axis (SR 1.33 with the location knobs). coh L|H reaches SR 2.03 in stage-1 terms but was not scored against location.
    - Form gc|te_ksgDCE becomes insensitive, since its response came from the phase alignment.
- **Form and warp organisation fail at cohort-like coupling.** The only non-Gaussian reader is the form level median(te − gc) (surrogates keep 0.14 of its response, SR 0.78). Post hoc, at strong coupling, r²|dcorr and mi_g|mi_kernel respond to form dispersion and are non-Gaussian on surrogates (tlmi and eucl|dtwSC do not after the sign correction, so warp has no non-Gaussian timing reading). The cohort form nulls therefore reflect an insensitive instrument, not an absence of non-Gaussian change.
- **Organisation is not level-free**, contrary to the protocol: lag level moves lag organisation about twice as much as lag dispersion, in the opposite direction, and band level moves band organisation 0.8 times as much. The NeuroTycho rise in lag organisation therefore fits higher lag level as well as more uniform lags. Levels are cleaner (band share 0.03; lag level −0.64).
- **No reliability or strength correction is usable** for any passing pair: noise lowers population SPI values rather than adding sampling error. Noise, strength spread and mixing move lag and band organisation about as much as their own knob and stay unbounded.
- **Cohort bound.** Sign claims surviving both reliability floors: NeuroTycho anaesthesia lag organisation (coh|icoh 0–0.25, d 1.61 → 1.57, 0.85 at the stricter floor); NeuroTycho mediation (−2.1 → −0.7, two-thirds of the raw effect a reliability drop); HMC N2 → REM lag organisation (0.83 → 0.67, weak: SR 0.66 and spectra change). Nothing stands in CHB-MIT, AD/FTD, LEMON or propofol.
- **Consequence:** z-axes support existence and sign claims under a nuisance bound, not magnitude; lag claims also need band and lag level excluded; non-Gaussian claims rest on one level measure.

## Core sufficiency (3 November, Gadi)

The [protocol's](protocol.md) two checks, run for the stage-1 core (77408cd) and the revised core (b56d908). Records are at `/g/data/ql44/we2614/mts-spi-study/representation/empirical/character-axes-261102/sufficiency/{,core-v2/}<cohort>.json`. Both cores give the same verdicts; the numbers below are for the revised core, as AUROC with paired bootstrap intervals.

- **z's increment over Pearson+strength:**
  - NeuroTycho: z adds +0.35 (.60 → .95), and the core accounts for it: P+core reaches .97, and z adds +0.03 [0.00, 0.03] on top (six units).
  - Propofol, moderate versus baseline: z adds +0.18 [+0.09, +0.30], and the core does not account for it: the core adds +0.03, and z adds +0.16 [+0.07, +0.25] on top.
  - LEMON: z's +0.06 [−0.00, +0.12] is not eligible, and the core adds nothing.
  - AD/FTD: z adds nothing.
- **Extended readouts.** No contrast is fully absorbed. The recurring survivors are:
  - fixed-lag timing: LEMON, FTD, mild propofol;
  - linear–rank agreement (form_monotone): AD, propofol;
  - phase-lag organisation: propofol;
  - spectral peak: NeuroTycho, and the propofol level.

  NeuroTycho's tail and best-shift level grow after adjustment.
- **Reading (inferred):**
  - The core is sufficient where a state reorganises coherence across bands (deep anaesthesia), not where the shift lies within a band. The band construct splits at fs/4 (25 Hz in the EEG cohorts), so alpha slowing sits entirely inside the low band and reaches z only through fixed-lag and peak readouts, which measure frequency in sample units.
  - Propofol's lower linear–rank agreement is also its residual that survived both surrogate variants (−.83/−.82), so it is the one candidate non-Gaussian effect left.
  - Spectral location within a band and linear–rank agreement each need a construction test before joining the core.
- **Stage-3 core** (474e443; adjustment fixed in e345720 so that a window missing a readout is imputed rather than dropped, and an unestimable adjustment is not counted as absorbed; records in `sufficiency/core-v3b/`):
  - LEMON: the core now adds +0.043 [+0.012, +0.079] over Pearson+strength, and z adds +0.016 [−0.03, +0.06] on top of it. Fixed-lag timing is absorbed.
  - NeuroTycho: unchanged.
  - Propofol, moderate versus baseline: still unaccounted for (z adds +0.13 [+0.06, +0.21] on top of the core). Linear–rank agreement and phase-lag organisation survive.
  - AD/FTD: location frequency is missing in 97% of windows (no complex AR(2) pole on the strongest edges), so it cannot be read there.
- **Location frequency mirrors channel spectra in the EEG cohorts** (`location-check/`, `location_check.py`):
  - Across windows it correlates 0.75–0.84 with the channels' 4–20 Hz spectral centroid.
  - After adjusting for centroid and peak, its state effects vanish: LEMON +0.87 → +0.39 (p .06); propofol moderate +0.60 → −0.17, mild +0.70 → −0.08.
  - So LEMON's core gain is channel-spectral information, not coupling character.
  - NeuroTycho falls relative to its channels (−1.86, six units), but its 0.04–0.2 cycle-per-sample window is not 4–20 Hz at its sampling rate, so this is not interpreted.

## Open

- Decision (user, 2 November): report a validated core and keep the 13 as an extended set; the core should be a sufficient interpretable readout, which the extended set tests. After stage 3 the core is band high share, band spread and mediation (one generator). Sub-axes are band organisation, lag (organisation and level), form (te − gc level) and location frequency; direction and warp are dropped. The core is not yet sufficient (above). Open: what carries propofol sedation's z increment (channel marginal shape, joint-burst coupling via real − IAAFT, or phase-lag organisation), and a noise- and strength-robust correction.
- Level readouts that are a median over edges can be degenerate: the best-shift level is 0 in most AD windows because cross-correlation peaks at lag 0.
- InceptionTime is not yet in the cohort comparison; MiniRocket is.
- Corpora moved to scratch after use: `/scratch/ql44/we2614/mts-spi-study/representation/empirical/character-axes-261102/{ad-ftd,lemon,propofol,surr-*}`; every real and surrogate run now lives there (checksum-verified moves, gdata quota). Scratch is purged after 100 days without access; the banks are cheap to regenerate.
