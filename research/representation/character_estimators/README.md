# Character estimators: five per-edge axes of dependence character

Question: can the character of a pairwise dependence, as opposed to its strength, be measured directly, edge by edge, on interpretable axes on [0, 1], and does each estimator respond to its named property and to nothing else in systems where the truth is known by construction?

These estimators use no SPI–SPI features and no pyspi statistic as a building block. They are derived from one principle and use numpy/scipy only. They are a separate line from the [character axes inside z](../character_axes/protocol.md), whose construction failures ([validation](../character_axes/validation/README.md)) motivate several design choices below. Context: [`docs/context/premise-and-character-axes.md`](../../../docs/context/premise-and-character-axes.md).

## Principle

An axis is the share of a pair's dependence that needs one named property: 1 − (dependence a model without the property reproduces) / (total dependence). Every axis must:

1. **Be defined on the process first**, with the estimator second.
2. **Be strength-free:** rescaling the coupling or the private noise leaves the population value unchanged.
3. **Have named endpoints:** 0 and 1 are reached by constructible systems.
4. **Be undefined without dependence:** an axis is reported only where the dependence it is a share of exceeds a circular-shift independence null (39 shifts, p ≤ .05). Its defined fraction is reported, never imputed.
5. **Have a closed-form truth** in its test system, so the estimator is checked for calibration, not only direction.

Both channels are first mapped to normal scores, so every main axis is invariant to a static monotone distortion of either channel (sensor or gain nonlinearity). `nonlinearity_raw` is the one exception: it is the same estimator on the raw scale, and it reads a monotone bend such as y = exp(x) as nonlinear.

## Axes

| Axis (0 ↔ 1) | Population definition | Estimator |
|---|---|---|
| nonlinearity N (linear ↔ nonlinear), directional | 1 − Var(best linear predictor of y from x) / Var(best predictor of y from x). The reverse, y from x swapped, is reported as `nonlinearity_yx` (v3) | Cross-validated R² (5 contiguous folds) of a linear window (±3 lags around the strongest first- or second-order lag within ±10) against a second-order Volterra model with cubic diagonal. Each R² is corrected for its model's out-of-sample cost, measured on the circular shifts (v3) |
| lag share ℓ (instantaneous ↔ quadrature) | Σ_f Im(C)² / Σ_f \|C\|², C the coherency: the share of linear coupling that zero-lag mixing cannot produce. Equals sin²φ for a constant phase φ. | Multitaper cross-spectra of W-sample segments, with squares estimated from distinct segments only |
| lag variability V (fixed ↔ variable lag) | 1 − (Σ_f \|E c\|² / Σ_f E\|c\|²) / ((E P)² / E P²), where c is a segment's cross-spectrum, after removing the fixed delay and dividing each quarter by its own power, and P = Σ_f \|c\|² its coupling power. Equals 1 − E cos(phase drift between segments) when the gain is constant or on/off. A continuously varying gain makes it read low. Warpedness is a variable lag, so it is included here. | Per-segment quarters and distinct segments, so every product multiplies time-disjoint estimates |
| intermittency S (steady ↔ intermittent) | 1 − (E P)² / E P². Equals 1 − q for coupling present in a fraction q of segments. Common amplitude modulation of both channels changes power, not coupling, and does not count. | As for V |
| directionality D (symmetric ↔ one-way), *auxiliary* | \|(G_xy − G_yx) − (G̃_xy − G̃_yx)\| / (G_xy + G_yx + G̃_xy + G̃_yx). G is linear Granger causality (order 4) minus its null mean, and ~ marks the time-reversed pair. Asymmetry that time reversal does not flip does not count. That includes unequal noise and instantaneous coupling, for which D is undefined. | OLS |

- **Estimates are not clipped.** Population values lie in [0, 1], but a per-edge estimate can fall slightly outside it by noise; clipping would bias every average upward. `delay` (samples, the peak of the cross-correlation within ±20) is reported as an auxiliary coordinate.
- **Directionality is auxiliary.** It is a normalised form of time-reversed Granger causality, which already exists, so it opens no new question. It is computed and scored, but it is outside the core set.
- **Segment length W = 100 samples.** This is the timescale V and S refer to: lag drift or coupling change slower than W and faster than the recording. Lag jitter faster than W is equivalent to a low-pass filter and reads as weaker coupling. That is a property of the data, not an estimator failure.
- **Spectral axes run on whitened channels.** ℓ, V and S are computed on channels whitened by a zero-phase filter, which leaves coherency unchanged.
- **Prior art for parts of these definitions (cited, not claimed as new):**
  - Gaussian-surrogate MI shares for nonlinearity ([Hlinka et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC4139498));
  - Pearson's correlation ratio, for the predictor-variance form;
  - instantaneous/lagged decompositions ([Pascual-Marqui 2007](https://arxiv.org/abs/0711.1455v1));
  - time-reversed Granger causality (Haufe et al. 2013; Winkler et al. 2016, unverified here).

## Test systems (`systems.py`)

x is a unit-variance Gaussian AR(1) process (coefficient 0.8) and η is an independent copy. Each knob θ is mapped so that the system's own axis equals θ analytically; directionality is the exception, and its truth is the long-run estimate. Every other property is held fixed.

| System | Construction | Held fixed |
|---|---|---|
| linearity | y = √(1−θ)·x + √θ·He₂(x) + σ·η | predictable variance; He₂ is uncorrelated with x at every lag |
| lag | y = cos φ·x + sin φ·H[x] + σ·η, sin²φ = θ | coherence magnitude |
| lag variability | as lag, with φ = π/4 + s·δ_k per segment, 1 − e^(−s²) = θ | local coupling, ℓ = 0.5 |
| intermittency | y = a_k·x + noise, coupling on in a fraction 1 − θ of segments | Pearson r = 0.4 and var(y) |
| direction | VAR, own AR(1) 0.5, lag-2 cross coupling 0.6 split (1 ± θ)/2 | total coupling |
| independent | x, η independent | — |

**Nuisances**, each at θ = 0.2 and 0.6 (0.25 and 0.75 for direction):
- **strength:** σ = 0.5, 2, 3; Pearson r = 0.2, 0.3, 0.5 for intermittency; c = 0.2, 0.4, 0.8 for direction;
- **length:** T = 1000, 4000, 8000;
- **spectrum:** AR(1) 0.5, AR(1) 0.95, narrowband AR(2) at 0.1 cycles/sample (not for direction);
- **monotone distortion:** y → sinh(βy)/β with β = 1, 3;
- **instantaneous mixing:** (x, y) → (x + μy, y + μx) with μ = 0.2, 0.5;
- **noise on x only:** white noise of SD 0.5 or 1;
- **fixed delay** of y behind x by 5 or 10 samples (added in v2);
- **common amplitude modulation** of both channels by exp(0.3 z_k), z_k ~ N(0, 1) per segment (added in v2).

The independent pairs also include a very slow narrowband spectrum, AR(2) with pole radius 0.99 at 0.02 cycles/sample (added in v2).

**Replicates:**
- 40 recordings per level of each main sweep at T = 2000;
- 20 per nuisance cell;
- 100 per null spectrum.

**Truth:**
- *By construction* where it holds: base, strength, T, spectrum, and β for the main axes (normal scores remove β exactly). Undefined (NaN) where the system has no dependence of that kind, such as Granger dependence for instantaneous coupling.
- *Otherwise the long-run value:* the mean of 3 recordings at T = 2¹⁷, each floored with 19 shifts, and NaN unless at least two are defined.

## Pre-registered criteria

Fixed before the scored run. Tolerances are on the [0, 1] scale, at T = 2000 and base settings unless stated. A cell counts only where the axis is defined in at least 80% of recordings.

- **C1, definition consistency.** Where both exist, the long-run value equals the construction value within 0.03 at every main-sweep level.
- **C2, accuracy on the own axis.** At every level: defined in ≥ 80% of recordings, |mean − truth| ≤ 0.10 and SD ≤ 0.15. Across levels, Spearman correlation between mean and truth ≥ 0.9.
- **C3, selectivity.** In the other systems' main sweeps, |mean − truth| ≤ 0.10 wherever the truth is defined.
- **C4, strength-free.** |mean − truth| ≤ 0.10 for the own axis at every strength level. The defined fraction is reported.
- **C5, null.** On independent pairs, and wherever the truth is undefined, the axis is defined in ≤ 10% of recordings.

**Verdict.** An axis that passes C1–C5 is *validated by construction in one generator family*. Any failure is reported with the condition that caused it. Other nuisances (T, spectrum, β for raw N, mixing, noise on x only) are reported as bias maps, not verdicts, together with whether the nuisance moved the truth itself (the definition responds) or only the estimate.

**Then:**
- **Adversarial pass.** An adversarial agent tries to break each axis: a system where the axis moves without its property changing, or stays put while it changes.
- **Fresh-seed re-score.** Breaks become new nuisance cells, and any estimator change is re-scored on fresh seeds (`--offset`).
- **Second generator family** (coupled nonlinear oscillators), sign checks only, is a later step.

**Design history (before pre-registration).** The estimators were iterated on smoke tests of these five systems, at a handful of recordings each, before this protocol was fixed. The changes were:
- unbiased products moved from same-sample tapers to time-disjoint quarters and segments, because non-Gaussian coupling biased the former;
- per-segment power normalisation was dropped, because a random divisor biases fourth moments;
- nonlinearity uses the better-predicted direction, after per-segment scaling, because pooling both directions mixed in the inverse regression;
- truths go through the same independence floor.

The smoke tests are not part of the scored run.

## Running it

No pyspi. A few minutes locally with reduced replicates; the full scored run took 52 CPU-hours, so it runs on Gadi (`jobs/validate.pbs`) and is analysed locally.

```bash
OMP_NUM_THREADS=1 python -m research.representation.character_estimators.validate run --out results/representation/character-estimators/<tag> --workers 9
python -m research.representation.character_estimators.validate analyse --out results/representation/character-estimators/<tag>
```

`analyse` writes `summary.csv`, `verdicts_own.csv` and `verdicts_axis.csv` to `tables/` (tracked), and the figures to `<out>/figures` (regenerated, not tracked).

## v2: changes after the adversarial pass (5 November 2026)

v1 (`f8dcc82`) was not scored: its run was stopped once the adversarial report showed that it could not pass. v2 is scored on fresh seeds (`--offset 1`) against the same criteria, with the new cells above. Breaks were triaged by one rule: fix a break if it is population-level, occurs in a regime real data plausibly has, and changes how a reading is interpreted; otherwise state it as a limitation. There is one adversarial round only.

| Adversarial finding | Decision |
|---|---|
| A fixed delay of 5 or more samples sends V and S to noise: a 25-sample quarter cannot resolve coupling at that lag | **Fixed.** The fixed delay is removed before the quarter analysis, because a fixed delay is not a change over time. |
| Common amplitude modulation of both channels (e.g. alpha bursts) reads as intermittency | **Fixed.** Each quarter is divided by its own power, so S measures change in coupling, not in co-power. |
| Per-segment scaling biased nonlinearity low for heavy-tailed coupling (x³: 0.25 against 0.40) | **Fixed by removing it.** The cost is that on/off linear coupling now leaks into N (about 0.17 at a 40% off-fraction), through the inverse regression; that is a limitation below. |
| Clipping at 0 biases averages of noisy per-edge V and S upward | **Fixed.** Estimates are unclipped. |
| The circular-shift floor is liberal for very slow signals, because the wrap adds a jump | **Fixed on the spectral path** (end points joined before whitening and shifting); scored on the very slow null. |
| Instantaneous nonlinear coupling of a time-irreversible signal gives ℓ and D > 0 | **Limitation.** ℓ and D are second-order readings; read ℓ as lag only where N is near 0. |
| With a frequency-varying phase (a delay), ℓ depends on how coherence is spread over frequency, including noise | **Limitation.** ℓ is a phase share, not a delay; `delay` is reported separately, and any broadband delay reads about 0.5. |
| Main N is partial and noise-dependent for monotone coupling applied before the noise (x³, exp) | **Limitation**, by design: normal scores remove monotone distortion after the noise; `nonlinearity_raw` reads such coupling (x³: 0.40). |
| V reads frequency drift as lag drift | **Limitation:** V is drift of the cross-spectral profile, which includes frequency; it is reduced by the delay removal. |
| D reads a common driver with unequal delays as one-way | **Limitation:** D is temporal precedence, not causation (auxiliary anyway). |

**Envelope coupling (value ↔ envelope) is deferred.** Its simple Gaussian-reference form, 1 − Σ|C|² / Σ ρ_power, reads a bursting source mixed linearly into both channels as envelope coupling. That is exactly the volume-conduction case it was meant to be robust to, and robust versions (leakage-corrected envelope correlation) are a research problem of their own.

## v3: changes after scoring (5 November 2026)

These follow the v2 scoring (Status below), so v3 is post-scoring and is labelled that way. It is rescored on fresh seeds (`--offset 2`).

- **N is corrected for out-of-sample cost.** Under independence, cross-validated R² is −k, where k is the model's cost: measured as 0.004 for the linear model and 0.03–0.05 for the full model at T = 2,000, about twice the textbook k/(n − k − 1) for slow or narrowband signals. In general, R²_cv ≈ 1 − (1 − R²)(1 + k), so R² = 1 − (1 − R²_cv) / (1 − R²_null). R²_null is the mean over the 39 circular shifts the floor already computes, so it carries the pair's own autocorrelation and marginals. On the same v2 recordings, the failing cells move from −0.4 to −3.1 to within about 0.1 of 0. Own-axis values are unchanged; at weak coupling they are better (0.01 → 0.23 against 0.20).
- **N is directional.** "The better-predicted direction" chooses whichever channel is easier to predict: with narrowband x and white private noise in y, x is predicted from y better than y from x, whatever the coupling. Linearity of a regression is not symmetric outside the Gaussian case. On/off linear coupling makes E[x | y] nonlinear, because large |y| implies the coupling was on; this was the v2 intermittency "leak". So N is x → y, and y → x is reported beside it. Edges are ordered anyway. The direction choice also biased N upward at weak coupling, because it took the larger of two noisy fits.
- **Intermittency truth for N:** forward raw N is 0 (E[y | x] = q·a·x); main N uses the long run, because normal scores bend a scale mixture slightly. The long-run N truths are recomputed under the new definition. All other long-run truths are reused from `7a3ace0`, where nothing they depend on changed (`validate run --truth … --refresh nonlinearity nonlinearity_raw`).

## Transfer family (`transfer.py`; pre-registered before its run, 5 November 2026)

The first family shares its construction with the estimators:
- its nonlinearity (He₂) lies inside the Volterra basis;
- its lag is the Hilbert transform that defines ℓ;
- its phase jumps and on/off switches are aligned to the estimators' W-sample segments;
- x and the private noise have the same spectrum.

Passing it shows that the estimators compute what they claim, not that they transfer. The transfer family removes each of these. All its systems use x = narrowband AR(2) (carrier 0.1 cycles per sample, the `narrow` spec) and white private noise η, except `direction_nl`.

| System | Construction | Truth of its own axis |
|---|---|---|
| rectifier | y = √(1−θ)·x + √θ·h(x) + σ·η, with h = standardised \|x\|. \|x\| is even, so it is uncorrelated with x at every lag. It lies outside the Volterra basis (88% of its variance is quadratic), and for an oscillation it carries the envelope and the second harmonic: cross-frequency coupling | raw N = θ exactly |
| delay | y = x(t − d) + σ·η, a conventional delay (fractional, exact for band-limited x), with sin²(2π·0.1·d) = θ | ℓ = ∫ \|C\|² sin²(2πfd) df / ∫ \|C\|² df, with \|C(f)\|² = S_x/(S_x + σ²) from the known spectrum |
| warp | y(t) = x(t − s·z(t)) + σ·η: a smooth time-varying delay (warping), with z an AR(1) of correlation time 2W, interpolated by cubic spline, and 1 − e^{−(2π·0.1·s)²} = θ | V = 1 − ∫ \|C\|² e^{−(2πfs)²} df / ∫ \|C\|² df: the limit of independent delays between segments, which a finite correlation time and drift within quarters move. The long run measures that gap (C1) |
| telegraph | y = a(t)·x + √(1 − r²/q)·η, with a(t) a two-state Markov chain (mean coupled dwell 2W, off-fraction θ) whose switches ignore the W grid, and r = 0.4 | S = θ in the long-dwell limit |
| direction_nl | as `direction`, with the cross terms through tanh(2·): nonlinear directed dynamics | long run |

Off-axis values are 0 by construction where the table in `transfer.py` says so: linear systems have N = 0; ℓ = 0 without a delay; V = S = 0 without time variation; D is undefined for instantaneous coupling. Otherwise they come from the long run (T = 2¹⁷, 3 seeds). D co-moves with ℓ in the delay system, which is correct: a delay is precedence.

The criteria are C1–C5 unchanged. Base levels have 40 recordings, strength cells (σ ∈ {0.5, 2}; r ∈ {0.3, 0.5}; c ∈ {0.3, 0.9}) have 20, and independent narrowband/white pairs have 100.

**Expected failure, stated before the run:** in a smoke run at T = 2¹⁵, V read 0.40 against 0.52 on warp, and V went negative (−0.2) on telegraph, because a switch inside a quarter makes the gain vary continuously, which reads V low (see the V definition).

## Status: v2 scored (5 November 2026)

v2 (`7a3ace0`) scored on fresh seeds: 204 conditions, 4,980 recordings, on Gadi (job 183193511: 77 min on 48 cores, 52 CPU-hours, 123 SU; `jobs/validate.pbs`). Tables are in `tables/`. `rows.csv` and `truth.csv` are kept untracked in `results/representation/character-estimators/7a3ace0/` and on Gadi scratch; `validate analyse` regenerates everything else from them.

Against the pre-registered criteria, only the lag share passes everything. "Systematic" below means the median over recordings is also off by more than 0.10, so the bias is not driven by a few outliers.

| Axis | C1 long run | C2 own-axis accuracy | C3 selectivity | C4 strength | C5 null |
|---|---|---|---|---|---|
| ℓ | pass | pass (bias ≤ 0.015, SD ≤ 0.03) | pass (worst 0.05) | pass | pass |
| N | — | pass (bias ≤ 0.03) | **fail**: 40 of 152 off-axis cells systematic | fail: −0.19 at r ≈ 0.3, θ = 0.2 | pass |
| N raw | pass | pass | **fail**: 38 of 142 systematic | fail: −0.19, same cell | pass |
| V | pass | fail: one recording at −3.2 (θ = 0.8) gives SD 0.66; median 0.80 | fail: 8 of 134 systematic | pass | pass |
| S | pass | fail: defined in 78% at θ = 0.8 (needs 80%); bias ≤ 0.04 | fail: 2 of 139 systematic | pass on bias; defined in 45% at r = 0.2 | pass |
| D (aux.) | — | pass | fail: 17 of 90 systematic, up to +0.33 | pass | defined in 23% where its truth is undefined |

For independent pairs, every axis is defined in at most 8% of recordings, for all five spectra (nominal 5%).

**Diagnosis.**

- **N is unstable whenever the windowed models explain little of y.** N = 1 − R²_lin / R²_full divides two small cross-validated R²s. The full model has 35 more terms than the linear one, so out of sample it loses roughly 35 / 1,600 ≈ 0.02 of R², and more with heavy-tailed product regressors. When R²_full is about 0.05 — weak coupling, or lag and phase-drift coupling that a ±3-sample window cannot represent — the ratio goes strongly negative: median −0.3 to −0.5, with a tail to −50. The arithmetic matches the sign and size, but this cause has not been isolated by an experiment. The candidate fix is the one D already uses: subtract the expected out-of-sample cost of the extra terms, and gate definedness on that corrected gain.
- **Main N reads on/off linear coupling as nonlinear (+0.50 at an 80% off-fraction); raw N does not (median about 0).** The inferred mechanism is that the normal-score transform of a heavy-tailed scale mixture is itself a nonlinear map. So the main/raw disagreement is informative, not noise.
- **V and S are accurate on their own axes; they fail at the extremes.** The failures are one outlier, definedness at an 80% off-fraction, and slow (AR 0.95) signals with mostly nonlinear coupling, where the coherent part is tiny: S reads +0.62 and V reads −0.67 there.
- **D is biased by measurement noise on the driver and by weak coupling.** This is a known weakness of Granger-type precedence, and the time-reversal correction does not remove it here.

The validation systems share one generator family with the definitions. Passing them shows construction validity; it does not show transfer. A second family, such as coupled nonlinear oscillators, is the open test.
