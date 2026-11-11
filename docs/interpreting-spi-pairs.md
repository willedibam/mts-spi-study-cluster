# Interpreting SPI–SPI features: reading rules (11 November 2026)

Required reading before any z feature, axis or family-block contrast is described in a note, figure, result or manuscript. Written from the user's guidance (11 November) and the repository's validation record; it exists so that strong results are neither under-read ("pattern of A decouples from pattern of B") nor over-read ("couplings became more nonlinear") again.

## What the paper argues, and where each link stands

The contribution is the character of dependence as a dimension that strength-only representations cannot word ([introduction scaffold](../drafts/introduction_draft.md)), and z as one simple, interpretable way to quantify it that draws on the statistical literature. The chain, each link with its evidence:

1. Character is an important, under-asked question: argument and precedent ([premise context](context/premise-and-character-axes.md)).
2. The construction makes sense: designed case studies, e.g. [Pearson–Spearman–MI](../research/cases/r_rho_mi/notebooks/r_rho_mi_260924.ipynb).
3. z encodes useful information about systems: proof-of-concept embeddings.
4. It works unsupervised and general-purpose: phase-transition and order-parameter studies.
5. It has real-world use: [real-world overview](../research/representation/representation_state/notes/real-world-results-overview.md), Zenodo corpus.
6. Bonus: recordings of any M and T share one feature space.

Prediction supports the argument; it is not the argument. A result that only ties or loses on a benchmark can still carry the paper if it yields a reading of character the comparators cannot word.

## The reading rule

For a feature z(A, B), the correlation across channel pairs of statistic A's and statistic B's edge profiles:

1. **Say what each statistic measures and how**: its estimator, transformation, lag, band, conditioning set and normalisation.
2. **Name the assumptions in which A and B differ.** Spearman differs from Pearson in one step, rank-transforming each channel.
3. **List every consequence of that difference, not only the intended one.** Rank-transforming removes monotone nonlinearity, but it also removes the influence of heavy tails, outliers and marginal shape. A fall in z(Pearson, Spearman) is therefore consistent with nonlinearity varying across pairs or with marginal shape varying across pairs. The [axes protocol](../research/representation/character_axes/protocol.md) lists the main confound of each axis.
4. **Give the character reading only when** A and B are one assumption apart and the alternative consequences are controlled (below). Otherwise state the reading literally. A literal statement is often where the insight starts, not a failure.
5. **Decompose a non-minimal pair into a chain of minimal pairs.** For example, robust precision against Spearman = tail (classical against MinCovDet precision) + mediation (precision against covariance) + form (covariance against Spearman). Then read each link separately.
6. **Write both readings**: the character reading (if earned) and the literal one, so a reader can see what was assumed.

## What z can and cannot say

- **z reads organisation, not level.** A low z(Pearson, Spearman) says that whatever separates the two statistics varies across pairs independently of shared strength. It does not say the couplings are more nonlinear. A uniform change across all pairs barely moves z.
  - Validation found level moves organisation as much as dispersion for lag and band. Even an organisation change can therefore partly reflect a level change.
  - Level questions ("all couplings became more nonlinear", "delays lengthened") need level scores or the dedicated per-edge estimators ([`research/estimators/`](../research/estimators/README.md): N, ℓ, V).
- **Strength bleeds in.** z is invariant to positive affine changes of each profile, but not to how strength is distributed across pairs. Under Gaussianity, coupling measures are functions of coherency (Nolte et al. 2020), so a global mode can move agreement mechanically (saturation).
- **Conditioning sets matter.** Precision and partial-correlation SPIs condition on the other channels in the window. Values from different M, montage or reference are different quantities. Subsets must be re-extracted, not cut from stored MPIs.
- **Exact zeros.** Seven per cent of a stored z vector is identically zero: antisymmetric × symmetric statistics.

## Priority characters and their validation status

Nonlinearity and timing (lag and its variability) are the most interpretable and actionable readings. Anything finer risks being too narrow to matter.

| Character | Inside z (axes) | Dedicated estimator | Status at cohort-like coupling [R] |
|---|---|---|---|
| Monotone nonlinearity | form_monotone (Pearson against rank) | N (directional, cost-corrected) | Axis fails: raw linear–rank pairs read marginal shape. Its real − IAAFT version reads joint bursts. N passes on synthetic families. |
| Non-Gaussian dependence | form_information | — | Kraskov MI artefact withdrawn (autocorrelation bias). Use the Theiler-window estimator. |
| Lag | timing_fixed_lag, best_shift | ℓ (lag share) | Lag is a sub-axis; band/frequency confound (cos 2πfτ). ℓ passes and transfers. |
| Lag variability / intermittency | warping, tail (proxy) | V | Warp fails. Tail is only a whole-window proxy. V is accurate but imprecise (sampling floor). |
| Conditional against marginal (mediation) | mediation | — | Core, but montage and M dependent. |
| Band organisation | band | — | Core. Band level moves organisation. |

**Wording tiers.** Confirmatory statements use only axes that pass validation. Everything else is worded as exploratory, e.g. "departure from linear-stationary-Gaussian dependence in families F". "More nonlinear" stays unavailable until a strength- and SNR-matched calibration shows the readout responds to nonlinearity, not to bursts or drift.

## Controls that turn a z difference into a character statement

- **Strength:** regress on mean |r| and covariance participation ratio, and recompute z on edges strength-matched across conditions.
- **Spectra and per-SPI levels:** does the level picture (mean-SPI) or the channel spectra already say it?
- **Reliability:** z attenuates with edge-profile reliability. Disattenuate with within-condition split-half reliability, and note the direction of the bias: rising reliability raises agreement.
- **Surrogate levels**, used to classify a statement, not to measure contributions:
  - Floor: univariate surrogates.
  - Linear strength and character: multivariate Fourier or IAAFT minus the floor.
  - Departure from linear-stationary-Gaussian: real minus multivariate.
  - Choice of surrogate: IAAFT reproduces coherence and spectral Granger poorly, and phase SPIs not at all, so use common-phase Fourier surrogates for spectral, phase and lag families.
- **Null effect size:** a contrast that should not differ (e.g. eyes open against eyes closed).

## Mistakes already made, kept so they are not repeated

- A non-Gaussian MI "level rise" from the uncorrected Kraskov estimator, read as character (withdrawn).
- A raw linear–rank fall read as nonlinearity when it tracks marginal shape.
- Surrogate-reproduced effects called "beyond spectral".
- Classifier weights averaged within families and read as a pattern (use contrasts of family-block means).
- Concatenated "+" models read as additivity tests (use nested stacking, [`fusion.py`](../research/representation/comparator_fill/fusion.py)).
- A precision finding at one montage assumed to replicate at another (16 bipolar against 127 average-referenced channels).
- "Pattern of A decouples from pattern of B" left as the conclusion, without the decomposition into minimal pairs and their readings.
