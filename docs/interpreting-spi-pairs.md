# Interpreting SPI–SPI features: reading rules

Read this before describing any z feature, axis or family-block contrast in a note, figure, result or manuscript. Written from the user's guidance (11 November 2026), the repository's validation record and a Fable 5.2 review. It exists so that strong results are neither under-read ("pattern of A decouples from pattern of B") nor over-read ("the couplings became more nonlinear"). Where each claim of the paper stands, and the validation status of each character axis, are in the [premise and character context](context/premise-and-character-axes.md#reading-status-11-november), not here.

Prediction supports the paper's argument; it is not the argument. A result that only ties or loses on a benchmark can still carry the paper if it gives a reading of dependence character that the comparators cannot word.

## The reading rule

For a feature z(A, B), the correlation across channel pairs of statistic A's and statistic B's edge profiles:

1. **Say what each statistic measures and how**: estimator, transformation, sign handling (signed, absolute, squared), lag, band, conditioning set and normalisation.
2. **Name the assumptions in which A and B differ.**
3. **List every consequence of that difference, not only the intended one.** The [axes protocol](../research/representation/character_axes/protocol.md) gives each axis's main confound.
4. **Give the character reading only when** A and B are one assumption apart and the other consequences are controlled (below). Otherwise state the reading literally. A literal statement is often where an insight starts, not a failure.
5. **For a non-minimal pair, list the candidate links.** A chain of minimal pairs from A to C is not a decomposition. Correlation is not transitive, so the links bound z(A, C) but do not sum to it, and the chain is not unique: rank-transforming and conditioning do not commute.
   - Report each link's own z and the partial z(A, C | B) in the same contrast.
   - Read the link that carries the effect.
   - Example: squared robust precision against squared rank correlation differs by robustness, conditioning, rank-transforming and sign loss. Each is a candidate link.
6. **Write both readings**: the character reading, if earned, and the literal one, so a reader can see what was assumed.
7. **Selection.** Features chosen as the top of about 41,000 coordinates are post-selection: read them only after they hold on held-out units.

### Worked example: Pearson against Spearman

Spearman is Pearson on ranks, and it is invariant to any monotone transform of each channel. So a static monotone nonlinearity applied to one channel and a change in that channel's marginal shape are the same thing to this pair. Both are channel-level properties, not properties of the coupling. What remains at the pair level is departure of the joint dependence (the copula) from Gaussian: a nonlinear coupling function, joint bursts, tail dependence. Two separations need no pyspi:

- **A normal-score intermediate.** Compare Pearson on the raw channels, Pearson on channels transformed to normal scores, and Spearman.
  - Pearson against normal-score Pearson reads marginal shape.
  - Normal-score Pearson against Spearman reads copula departure.
- **IAAFT surrogates.** These keep each channel's marginal distribution, so real minus IAAFT removes the marginal part. That is why validation found the real − IAAFT linear–rank readout tracks joint bursts, while the raw readout tracks marginal shape.

"Monotone nonlinearity of the coupling" is earned only for the copula part, and only after a calibration shows the readout responds to a nonlinear coupling function rather than to bursts.

### Worked example: Gaussian against flexible MI (clarified 17 November)

For a continuous pair with population normal-score marginals, let G = −½ log(1 − ρ²), with ρ measured after that transform. Then I − G = D_KL(p || q_ρ) ≥ 0, where q_ρ is the bivariate Gaussian with the same covariance. This follows from Gaussian maximum entropy with the marginals and covariance fixed; see [Ince et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC5324576/). Thus the Gaussian-versus-flexible MI pairing has a principled non-Gaussian-dependence interpretation, not merely an unnamed disagreement. The identity concerns population quantities: finite-sample kNN estimates can fall below G, so the observed comparison still needs matched estimator/null calibration.

Under the operational definition below, a validated residual after normal-scoring and window-matched IAAFT is evidence of nonlinear dependence. Gain mixtures and linear mixtures of non-Gaussian sources can generate such dependence; identifying its generating mechanism is a further question, not a prerequisite for this statistical reading. Exact surrogate preservation depends on the final step: spectrum-final IAAFT keeps cross-spectra exactly and marginals approximately, so diagnostics/calibration remain part of the evidence.

For edge profiles I = G + E, z = corr(G, G + E). A fall in z concerns how Gaussian-predicted dependence accounts for the organisation of MI across edges; it is not a monotone measure of mean E, its variance, or the number of nonlinear pairs. For example, adding the same excess to every edge leaves z unchanged. A change in the spread of G also changes z's sensitivity. Report the null-corrected organisation result separately from any level claim; do not require a nonlinear coupling-function mechanism before calling the former nonlinear dependence under the stated definition.

Scientific target, clarified 17 November: characterising the observed dependence is a substantive result; a curved physical coupling function is not a required endpoint. A curved conditional mean can be estimated and validated predictively, but is coordinate-dependent and is not by itself a causal response. Recovering a generating coupling law requires an identifiable dynamical/measurement model or informative interventions; observational identifiability is possible under additional assumptions, not impossible in principle ([Peters et al.](https://jmlr.org/papers/v15/peters14a.html)). Mechanism discrimination is optional unless the stated biological claim depends on it. Prioritise independently reproducible, well-calibrated dependence-character effects over identifying curvature for its own sake; retaining gain as a possible generator does not weaken the operational non-Gaussian-dependence claim when that gain actually creates the departure.

### Worked example: robust against classical precision (tail link)

MinCovDet keeps the half of the time samples with the smallest Mahalanobis distance. Mahalanobis distance penalises departure from the dominant covariance structure, not amplitude along it. Large common-mode samples (a global slow wave) lie along the leading mode and are kept. The discarded samples are those where channels disagree with the common pattern: spatially focal or pattern-atypical transients [I].

The defensible reading of a robust–classical divergence is time-heterogeneous dependence: "a minority of samples carries a conditional-dependence pattern different from the bulk". "Episodic coupling" is earned only if the down-weighted samples are temporally clustered, involve several channels and carry a distinct network. Check this directly from MinCovDet's support mask on the raw windows.

Two corrections from the axis screen ([result](../research/representation/atlas/axis_screen/result.md), 22 November):
- Common-mode samples are kept only when the amplitude lies along one mode. A shared gain inflates every direction, so high-amplitude samples are trimmed too, and shared gain is a sufficient generator of the axis's sign.
- MinCovDet also trims on marginal tails. On z-scored windows the robust–classical contrast is not separable from the channels' tails placed in time by the spectrum: it fails against marginal-final IAAFT, and spectrum-final IAAFT, whose draws lose the tails, gives a spurious effect. Read it only on normal-scored windows, where the macaque drug effect is a rise in the trimmed fraction beyond the surrogate. Chebyshev distance behaves the same way.

### Gaussian-determined partners (pair catalogue, 20 November)

- **Static statistics.** On standardised channels, the Gaussian expectation of any time-exchangeable bivariate statistic is a fixed function of r. This covers rank correlations, MI estimators, distance correlation, IDS, L1, L2 and Canberra distances, and barycentre moments. Any pair from this set is therefore a principled non-minimal pair beyond IAAFT, read through the feature the more flexible statistic weights. Maxima (Chebyshev, barycentre max) are the exception: their expectation also depends on T and on the spectrum of x ∓ y. The relations are checked numerically in [`identities.json`](../research/representation/atlas/pair_catalogue/identities.json): Spearman (6/π) arcsin(ρ/2), Kendall (2/π) arcsin ρ, Székely's dCor, Chatterjee ξ.
- **Envelope correlation.** Its Gaussian partner is |γ|² = r² + corr(x, H[y])² (H the Hilbert transform), not r², so z(pec, r²) mixes in lag. pec against |γ|² beyond IAAFT reads amplitude co-modulation beyond linear coupling. Likewise PLV = (π/4)|γ| ₂F₁(½, ½; 2; |γ|²) per frequency.
- **Ceiling rule for mapped pairs.** When the Gaussian-determined partner is mapped through its exact relation, z₀ is 1 minus estimator noise by construction. Judge such pairs by the null SD of the net, not by z₀ > .98.
- **Chebyshev against L1** (no class reading in `assumptions.READINGS`): how far each pair's largest disagreement departs from its typical disagreement, i.e. the tail of x − y. Read on normal-scored windows only (above).

## What z can and cannot say

- **Sensitive to, not specific to, heterogeneity across pairs.** z responds when a property varies across pairs independently of shared strength. A uniform change barely registers, but level still moves z in three ways:
  - **Saturation:** estimators that are nonlinear in strength.
  - **Range restriction:** strength's spread across pairs shrinks relative to estimator noise.
  - **Reliability:** edge-profile reliability changes.
  
  Level questions ("all couplings became more nonlinear", "delays lengthened") need level scores or the per-edge estimators in [`research/estimators/`](../research/estimators/README.md) (N, ℓ, V).
- **Strength bleeds in.** z is invariant to a positive affine change of each profile, but not to how strength is spread across pairs. Under Gaussianity, coupling measures are functions of coherency (Nolte et al. 2020), so a global mode can move agreement mechanically.
- **Conditioning sets matter.** Precision and partial-correlation SPIs condition on the other channels in the window. Values from a different M, montage or reference are different quantities, so channel subsets of conditional SPIs must be re-extracted. Edge subsets, such as the strength-matched control, can be taken from stored MPIs for every SPI.
- **Exact zeros.** Seven per cent of a stored z vector is identically zero: antisymmetric × symmetric statistics. PLI and wPLI are stored antisymmetric, so `plv|pli` and `plv|wpli` are not lag pairs; use `ppc|dswpli` (both debiased) or `cohmag|icoh` (`cohmag` is |C|², `icoh` is |Im C|).
- **Precedence pairs are not minimal.** Same-time MI against GC/TE differs in lag, in conditioning on the target's own past, and in direction. Read through the links: `mi_gaussian|tlmi_gaussian` (lag only, mostly reads autocorrelation at one sample), `tlmi_gaussian|gc_gaussian` (conditioning on the target's past only), and `cohmag|sgc` (Geweke 1982: total linear dependence is the sum of the two directed parts and an instantaneous part, and −log(1−|C|²) is its spectral density). GC and TE are equivalent for Gaussian variables (Barnett, Barrett & Seth 2009), so `gc_gaussian|te_kraskov` reads non-Gaussian predictive structure, plus TE estimator noise, which is large at T ≤ 1000. The `DCE-AUTO` Theiler window fixes kNN autocorrelation bias; it does not remove the conceptual dependence of GC/TE on autocorrelation, which is what conditioning on the target's past is.
- **Ceiling pairs.** Pairs with z₀ near 1, such as Spearman against Kendall or Pearson against Spearman, have tiny null SDs. A change of −0.01 can be −10 SD, so report the z change, not only its SD multiple.
- **Priorities.** Nonlinearity and timing (lag and its variability) are the most interpretable and actionable readings; finer characters risk being too narrow to matter.
- **"Nonlinear dependence" (terminology, user decision 17 November).**
  - *Definition:* dependence beyond the linear-Gaussian expectation, whatever produces it. This is what rejecting a linear-Gaussian surrogate null shows in the surrogate-data tradition (Theiler et al. 1992). It is a property of the joint distribution, not of the generating equations.
  - *Three routes produce it:*
    - a nonlinear coupling function;
    - linear coupling with a time-varying gain or state (a mixture of Gaussians);
    - linear mixing of non-Gaussian sources (ICA's "linear non-Gaussian" case).
  - *Conditional mean:* only a pure scale mixture (one shared gain on an unchanged correlation pattern) keeps it linear. With state-dependent correlation or non-Gaussian sources it is generally nonlinear, even when the generating equations are linear.
  - *Segment surrogates locate a timescale; they do not reclassify route 2 as linear.* Any process is close to linear-Gaussian on short enough stretches, and a surrogate fitted per short segment absorbs everything (a static quadratic keeps .14 of its residual at 50-sample segments). The classic non-stationarity false positive is drift slower than the analysis window, tested by segments long relative to the drift. Absorption only at segments much shorter than the window says the departure is a mixture over that timescale, which is route 2. Same-sample MI pools the window's samples, so it cannot tell drift from recurrent switching; that distinction concerns the generator, not the statistic.
  - *Odd against even cross-cumulants describe the shape of the departure* ([cumulant test](../research/representation/atlas/mi_pair/preregs/cumulants-prereg.md), 23 November).
    - Third-order cross-cumulants of a whitened pair vanish for any mixture of zero-mean Gaussians. Curved coupling or mean-shifting regimes produce them.
    - So the J3 share of the Edgeworth negentropy separates a mixture of magnitudes from a sign-asymmetric shape: calibrated .03–.13 against .30–.38.
    - It does not separate gain from coupling strength.
    - The Edgeworth size overshoots kNN MI for heavy-tailed mixtures (up to 6.6×), so read the share, not the size.
  - *Paper wording:* state the definition once. Reserve "nonlinear coupling" for a mechanism shown by a calibrated readout.
  - *Two directions:* "more or less nonlinear" requires a declared level measure, such as population mean excess MI. A fall in z(Gaussian, model-free) concerns organisation relative to the Gaussian profile; it does not by itself identify the amount or spread of excess MI (worked example above).
- **Wording tiers.** Confirmatory statements use only axes that pass validation. Everything else is exploratory, e.g. "departure from linear-stationary-Gaussian dependence in families F".

## Controls that turn a z difference into a character statement

- **Strength.** Regress on mean |r| and the covariance participation ratio; recompute z on edges strength-matched across conditions.
- **Spectra and per-SPI levels.** Check whether the channel spectra or the level picture (mean-SPI) already say it.
- **Reliability.** Disattenuate by within-condition split-half reliability, per SPI as well as overall. Rising reliability raises agreement, so the bias has a direction.
- **Surrogate levels.** Use them to classify a statement, not to measure contributions.
  - Floor: univariate surrogates.
  - Linear strength and character: multivariate surrogates minus the floor.
  - Departure from linear-stationary-Gaussian: real minus multivariate surrogates.
  - Use common-phase Fourier surrogates for spectral, phase and lag families, which IAAFT reproduces poorly or not at all.
- **Null effect size.** A contrast that should not differ, such as eyes open against eyes closed.

## Mistakes already made, kept so they are not repeated

- A non-Gaussian MI "level rise" from the uncorrected Kraskov estimator, read as character. Withdrawn.
- Linear pairs whose member is a multitaper or best-lag statistic assumed degenerate under spectrum-final IAAFT. DPSS tapers and zero-padded lagged correlation are not fixed by the periodogram, so real − IAAFT moves by up to .08 with the drug (axis screen, 22 November).
- Tail-sensitive statistics (MinCovDet, Chebyshev) on z-scored windows read against spectrum-final IAAFT. The draws lose the channel tails, giving spurious effects of −16 SD or of reversed sign.
- A between-animal, extrapolated delta-matched regression described as "sleep at matched delta" (manuscript draft, 20 November). Within animal, sleep and drug windows barely overlap in delta fraction.
- Absorption by 1-s segment surrogates read as "non-stationarity rather than nonlinear dependence" (19 November). A mixture within the window is route 2 of the definition; only drift slower than the window is the classic false positive.
- A raw linear–rank fall read as nonlinearity when it tracks marginal shape.
- Surrogate-reproduced effects called "beyond spectral".
- Classifier weights averaged within families and read as a pattern. Use contrasts of family-block means.
- Concatenated "+" models read as additivity tests. Use nested stacking ([`fusion.py`](../research/representation/comparator_fill/fusion.py)).
- A lag-pair rise read as "more lag" when a zero-phase null with the state's spectrum reproduced most of it: |Im C|, best-lag correlation and other unsigned lag statistics have an estimation floor that rises as spectral concentration lowers the effective sample size. Test lag pairs against zero-phase and stationary nulls (`discovery.py lag-null`); debiased statistics (`ppc`, `dswpli`) had almost no floor in the NeuroTycho bank, but a residual beyond the stationary null is not lag-specific: window-to-window strength change raises it too, so check across-window phase dispersion.
- A precision finding at one montage assumed to replicate at another (16 bipolar against 127 average-referenced channels).
- "Pattern of A decouples from pattern of B" left as the conclusion, without the candidate links and their readings.
