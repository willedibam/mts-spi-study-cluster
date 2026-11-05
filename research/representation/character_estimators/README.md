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
| nonlinearity N (linear ↔ nonlinear) | 1 − Var(best linear predictor) / Var(best predictor), predicting one channel from a window of the other, in the better-predicted direction | Cross-validated R² (5 contiguous folds) of a linear window (±3 lags around the strongest first- or second-order lag within ±10) against a second-order Volterra model with cubic diagonal. |
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

Local; no pyspi.

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

## Status

v2 pre-registered; scoring on Gadi.
