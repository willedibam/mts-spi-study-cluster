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

### Worked example: robust against classical precision (tail link)

MinCovDet keeps the half of the time samples with the smallest Mahalanobis distance. Mahalanobis distance penalises departure from the dominant covariance structure, not amplitude along it. Large common-mode samples (a global slow wave) lie along the leading mode and are kept. The discarded samples are those where channels disagree with the common pattern: spatially focal or pattern-atypical transients [I].

The defensible reading of a robust–classical divergence is time-heterogeneous dependence: "a minority of samples carries a conditional-dependence pattern different from the bulk". "Episodic coupling" is earned only if the down-weighted samples are temporally clustered, involve several channels and carry a distinct network. Check this directly from MinCovDet's support mask on the raw windows.

## What z can and cannot say

- **Sensitive to, not specific to, heterogeneity across pairs.** z responds when a property varies across pairs independently of shared strength. A uniform change barely registers, but level still moves z in three ways:
  - **Saturation:** estimators that are nonlinear in strength.
  - **Range restriction:** strength's spread across pairs shrinks relative to estimator noise.
  - **Reliability:** edge-profile reliability changes.
  
  Level questions ("all couplings became more nonlinear", "delays lengthened") need level scores or the per-edge estimators in [`research/estimators/`](../research/estimators/README.md) (N, ℓ, V).
- **Strength bleeds in.** z is invariant to a positive affine change of each profile, but not to how strength is spread across pairs. Under Gaussianity, coupling measures are functions of coherency (Nolte et al. 2020), so a global mode can move agreement mechanically.
- **Conditioning sets matter.** Precision and partial-correlation SPIs condition on the other channels in the window. Values from a different M, montage or reference are different quantities, so channel subsets of conditional SPIs must be re-extracted. Edge subsets, such as the strength-matched control, can be taken from stored MPIs for every SPI.
- **Exact zeros.** Seven per cent of a stored z vector is identically zero: antisymmetric × symmetric statistics.
- **Priorities.** Nonlinearity and timing (lag and its variability) are the most interpretable and actionable readings; finer characters risk being too narrow to matter.
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
- A raw linear–rank fall read as nonlinearity when it tracks marginal shape.
- Surrogate-reproduced effects called "beyond spectral".
- Classifier weights averaged within families and read as a pattern. Use contrasts of family-block means.
- Concatenated "+" models read as additivity tests. Use nested stacking ([`fusion.py`](../research/representation/comparator_fill/fusion.py)).
- A precision finding at one montage assumed to replicate at another (16 bipolar against 127 average-referenced channels).
- "Pattern of A decouples from pattern of B" left as the conclusion, without the candidate links and their readings.
