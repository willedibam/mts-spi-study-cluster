# Does time-varying gain explain the MI-pair drug fall? A gain-over-floor null on the macaque 10-s windows

Part 1 (construction, calibration rule and real-data rule) frozen 24 November 2026 at the commit that adds it. At that point the null's fit had been debugged on synthetic windows using fit diagnostics only: gain recovery, shrinkage factors, and the match of correlation matrices. No z or MI value of any GOF, DECOR or N1 surrogate had been computed. Code: [`../envnull/nulls_env.py`](../envnull/nulls_env.py) (null), [`../envnull/envcalib.py`](../envnull/envcalib.py) (calibration).

## Why

- **Claim at stake.** z(Gaussian MI, KSG MI) falls beyond joint IAAFT under every drug (4/4 animals). Under the [operational reading](../../../../../docs/interpreting-spi-pairs.md#worked-example-gaussian-against-flexible-mi-clarified-17-november), that is dependence beyond the linear-Gaussian expectation, whatever produces it. This test does not change that claim.
- **What this test decides.** The wording of the mechanism.
  - The [segment null](nonstat-prereg.md) found that about 57% of the fall is a mixture over about 1-s stretches, but its calibration could not separate gain from coupling.
  - Post hoc, the residual tracks local (per-channel) envelope non-stationarity ([matched-result](../../movers/matched-result.md)).
  - A reviewer's reading is "strength varying in time". If a time-varying gain on each channel's coupled signal, over a fixed floor, reproduces the fall, the reading is gain. If not, the change is in coupling.
- **The discriminating constraint is sign.**
  - Under gain over a fixed floor, a channel's coupling can only rise with its own gain, because more signal sits over the same floor.
  - Local decoupling (private power up, coupling down) and a static nonlinear coupling function violate this.
  - A null that lets correlation depend on amplitude freely absorbs both kinds. The segment null at short L does this.

## Constructions considered

- **N1, envelope remodulation** (`nonstat/nulls.n1`): IAAFT of envelope-normalised carriers, with the real envelopes multiplied back.
  - H0: a stationary carrier correlation under each channel's own gain, with the gain also scaling the floor.
  - Dropped on 19 November: it gave +.039 to +.125 on shared gain over an independent floor, i.e. it injects dependence when the floor is not gained.
  - Kept here as a baseline at σ = 25 and 100, never as a candidate.
- **Wavelet- or envelope-pinning surrogates** (pinned-wavelet IAAFT and IAAWT, Keylock; local mean and variance kept per scale): each channel's local energy is fixed, and the carriers are re-randomised.
  - The H0 is the same as N1's (carrier correlation unconditioned on the gain), so it carries the same floor failure.
  - Pinned coefficients also keep cross-channel phase relations at the pinned scales, i.e. part of the coupling.
  - Not used.
- **Truncated-Fourier surrogates** (Nakamura, Small and Hirata 2006; amplitude-adjusted variant, Lucio et al. 2012, PRE 85, 056202): they keep the low-frequency phases of the signal itself (trends), not the envelope of a carrier.
  - Applied jointly, they keep slow-band coupling.
  - Not an envelope null.
- **Segment IAAFT** (N2): already run, and non-separating (above).
- **Chosen: gain-over-floor (GOF), a model-based null.**
  - H0: x(t) = D_g(t) s(t) + n(t), where:
    - s is stationary linear-Gaussian with any spatial covariance (private parts allowed);
    - g is one slow positive gain per channel (shared, per-channel or both);
    - n is a stationary floor the gain does not scale, either diagonal or diagonal plus one common-mode factor (`floor = diag | cm`).
  - Local covariance: Σ(t) = D_g C D_g + F.
  - Fit (`nulls_env.fit`), on the raw standardised window: Gaussian-smoothed local covariances (σ samples), then alternating least squares for C, F and g.
  - Gain noise is removed by empirical-Bayes shrinkage of the shared and per-channel log-gain modes. The noise variance comes from fitting three stationary phase-randomised copies of the window.
  - C is moment-matched, and then adjusted until the normal-scored surrogate's correlation matrix matches the normal-scored window's (20 steps), so the Gaussian-MI profile is nearly the real one, as under N0.
  - Generation: y = g(t) · (Z C^½) + Z′ F^½, with Z and Z′ spatially whitened joint phase randomisations of the window; then normal-scored.
  - It is a typical realisation, not a constrained one (Theiler and Prichard 1996, Physica D 94, 221). Nothing guarantees its size except the synthetic calibration below. Its cross-spectra are not exact, so its estimator bias can differ from N0's. The in-H0 generators landing at zero is the guard against that bias entering drug − awake.
- **DECOR arm (descriptive only, never a gate candidate):** the same fit, with each channel's gain trajectory circularly shifted by its own random offset. It keeps each channel's own gain dynamics and breaks gain co-modulation across channels. By construction it should not absorb shared-gain generators. On real data it asks whether per-channel burstiness alone, without co-modulation, reproduces the fall.

## H0 membership of the generators (checked on paper before any compute)

| Generator (code) | Structure | Inside GOF H0? |
|---|---|---|
| `gain_homog` (reorder) | shared slow gain on a latent network; white floor .3, not gained | yes (diag floor) |
| `gain_hetero` (reorder) | the same, floor heterogeneous .05–2 | yes (diag floor) |
| `gain_cmfloor` (nonstat) | shared slow gain on the signal; floor with a common-mode share of .5, not gained | yes with `cm`; no with `diag` |
| `burst`, `burst_slow` (nonstat) | shared on/off gain (0.8-s, 3-s dwell) on signal and its private floor; white floor .15 | yes |
| `gain_local` (nonstat) | each channel's whole signal under its own slow gain | yes (floor 0) |
| `decouple` (nonstat) | local episodes: shared part ×.3, private burst ×2.5 | no (two gains in one channel, opposite direction) |
| `quadratic` (nonstat), `pair_nonmono` (reorder) | static curved coupling | no |
| `switch` (nonstat) | loading matrix switches | no; reported only |
| `quad_weak` (nonstat) | weak curved coupling | no; reported only (too close to noise to grade) |
| `linear`, `linear_hetspec` (new: heterogeneous AR(2) spectra) | stationary linear | yes (gain constant) |
| `burst_quad`, `burst_decouple` (new) | `burst` gain on top of `quadratic` or `decouple` | mixed; additivity check, reported |

## Calibration rule (synthetic; fixed before any calibration z exists)

Pipeline as the anchor: 32 channels, T = 1000, real window normal-scored, `edges.edge_mi`, one draw per null. Residual = z(real) − z(null). Retention R_v = mean residual under null v ÷ mean residual under N0 (joint IAAFT), per generator.

- **Development** (`envcalib.py dev 48`): seeds 50000 + 1000·k + s, 48 windows per generator. Candidates are the four GOF variants σ ∈ {15, 40} × floor ∈ {diag, cm}. Baselines are N1 at σ 25 and 100. Each candidate also gets its DECOR arm.
- **A candidate passes** only if all of these hold:
  1. `linear` and `linear_hetspec`: |mean residual| ≤ .002.
  2. Each of the six gain generators: −.3 ≤ R ≤ .3. The bound is two-sided: overshoot fails, as N1's did. If a gain generator's N0 residual has t > −3 (no fall to remove), |mean residual| ≤ .003 instead.
  3. Each coupling generator: R ≥ .6. If its N0 t > −3, it is not gradable and is dropped from the gate (reported). At least `quadratic` and `decouple` must be gradable, or the calibration fails.
- **Selection:** among passing candidates, the one with the largest minimum coupling R. Ties within .05 go to the larger σ, then to `diag` (fewer parameters).
- **Confirmation** (`envcalib.py confirm 96`): fresh seeds 90000 + …, 96 windows. Run in parallel with development for all four variants, but only the variant development selects is read (`gate.py --only`); the others' confirmation numbers are not looked at before the verdict. The selected variant must pass the same gate on point estimates. Report a 95% bootstrap interval (over windows, ratio of means) for every R.
  - If it fails, there is no second pick: the verdict is "cannot be separated".
- **If no candidate passes development, or confirmation fails:** stop. The null cannot separate gain from coupling at this resolution. This is written into findings and not run on real data.
- **Reported but never gating:** N1, DECOR, `switch`, `quad_weak` and the two mixtures.
  - Additivity: a mixture's GOF residual should be close to the coupling part's own N0 residual.
  - Without additivity, a real-data retention is read as "a fall the gain model does not produce", not as a fraction of mechanism.

**Amendment, 24 November, before any calibration output existed.** The first freeze (`28883b11` on main) set 16 development and 24 confirmation windows, run locally. That local run was stopped after 3 minutes, before it had written anything (output is written only at the end). The calibration moved to Gadi (`jobs/envcalib.pbs`, six 48-cpu jobs sharded over generators), with 48 and 96 windows for tighter retention intervals. The thresholds, generators, candidates and selection are unchanged.

## Real-data rule (applies only if a variant passes confirmation)

- **Windows:** exactly as the [segment null](nonstat-prereg.md): NeuroTycho 10-s stages A and B, 1,260 windows, 32-channel subsets (rowmode `first`, except stage-B sleep `row`).
  - Real z on the normal-scored window. N0 is the anchor's own draw, which is the reproduction gate: real and N0 z within 1e-4 of the stored `zns`, or stop.
  - GOF is fitted on the raw window. 3 GOF draws per window (residual = real − mean over draws), plus 1 DECOR draw. Seeds [row index, 10 + d].
- **Statistic:**
  - Session Δ = mean residual (drug) − mean residual (awake eyes-closed).
  - Animal means pool the drug sessions (KT, KTMD, MD, PF), as `montage/animal_level.py`.
  - Retained fraction R = pooled drug Δ under GOF ÷ pooled drug Δ under N0, by animal and by agent.
- **Verdict (primary null = the confirmed variant):**
  - **COUPLING SURVIVES:** R ≥ .6, every animal mean < 0, and one-sided t(3) on animal means p < .05.
  - **GAIN EXPLAINS:** R ≤ .3, and the animal-level GOF Δ is not significant (p ≥ .05, or fewer than 4/4 negative).
  - **PARTIAL:** anything else. Report R with its interval and read it against the calibration table (no verdict on mechanism).
- **Controls (reported):** the eyes-open − eyes-closed contrast under GOF (SD and mean, same pooling) and sleep − awake under GOF. Gain-explains is only readable if the eyes-open contrast is at control size.
- **Descriptive:** DECOR retention, i.e. how much per-channel burstiness alone reproduces.

## Pre-stated limits

- The null is model-based, so its validity is exactly the calibration's coverage. Gain shapes outside the six generators, such as gain on private parts or gain faster than ~0.2 s, are not covered.
- A gain that does not scale the floor changes correlation over time, so it is a change in coupling strength in the correlation sense. "Gain" here means specifically a change in the signal's amplitude over a fixed floor.
- One surrogate family; 32-channel subsets; four animals; 10-s arm only.

## Compute (if run)

- Gadi, 48 cpus: about 1,260 windows × 6 MI evaluations at ~10 s, plus fits, ≈ 22 cpu-hours, about 0.5–1 h wall.
- Expected ≈ 0.1 kSU, worst case 0.3 kSU.
