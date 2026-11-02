# Character axes: validation by construction

Question: does each candidate character axis respond to the generative property it is named for, with a stable sign, and how much does it respond to strength, noise and the other properties? The axes ([`src/mtsspi/character_axes.py`](../../../../src/mtsspi/character_axes.py), [protocol](../protocol.md), [review](../axes-review.md)) had never been scored on constructed data. Design and criteria below were fixed on 2 November 2026 before any scored run; only coherence magnitudes and Kuramoto phase locking were piloted, to place the baseline between NeuroTycho (median cohmag 0.21) and LEMON (0.37).

## What a pass licenses

z-axes are proxies. A named claim needs a known, stable sign of response, measured nuisance sensitivities, and an observed effect larger than the measured nuisances could produce in that dataset. Claim tiers: existence (z carries character beyond strength), sign with a nuisance bound (e.g. lag organisation rose), magnitude (lag changed more than band, or across datasets). Magnitude is out of scope for z-axes. Toy result behind this: for a = s, b = s·c with s, c independent across edges, z = √(rel_a·rel_b)·[1 + CV_c²(1 + CV_s⁻²)]^(−1/2), so strength spread and reliability move z with character fixed.

## Candidate pairs

| Construct | Pairs (estimator family) | Organisation predicted to | Level |
|---|---|---|---|
| form | mi_g\|mi_ksgDCE, tlmi_g\|tlmi_ksgDCE, gc\|te_ksgDCE (KSG); r²\|dcorr (distance); mi_g\|mi_kernel (kernel) | fall with form dispersion | median(mi_ksgDCE − mi_g), rises with form level |
| lag | cohmag\|icoh at 0–0.5, 0–0.25, 0.25–0.5 (coherency); plv\|\|wpli\|, ppc\|dswpli (phase locking); pec\|pec_orth (envelope) | fall with lag dispersion | median icoh/√cohmag, rises with lag level |
| band | cohmag L\|H (coherency); ppc L\|H (phase locking); sgc L\|H (spectral Granger) | fall with band dispersion | median high share cohmag_H/(cohmag_L + cohmag_H), rises with band level |
| mediation | cov²\|prec² Ledoit–Wolf and empirical | fall with mediated share | — |
| direction | corr(A, Aᵀ) over ordered entries for gc, te_ksgDCE, sgc 0–0.5; tlmi_g\|gc (matched lag) | corr(A, Aᵀ) falls with asymmetry | median \|A − Aᵀ\|/(A + Aᵀ) for gc, rises with asymmetry |
| warp | eucl\|dtw-SC, dtw-SC\|dtw (alignment) | fall with warp dispersion | median dtw-SC/eucl, falls with warp level |
| controls | mi_ksg\|mi_ksgDCE (estimator); r²\|mi_g (strength range); r\|ρ (monotone, for the marginal-map nuisance) | — | — |

The 35 statistics are written to `spis.yaml` by `build_subset.py` from pyspi `benchmarked_p90.yaml`, so names and settings match the cohort banks.

## Design

Generators are in `generators.py`; M = 16, T = 2000 and per-channel z-scoring match the cohort windows (bands in cycles per sample, so sampling-relative). Every recording has two windows that share their network (graph, weights, per-edge values) and differ only in noise, so cross-window reliability is computed as in the cohorts. Network and noise seeds depend only on baseline, replicate and window, not on the knob level: every level of a knob reuses the same draws, and responses are paired within replicate.

- **Family A, edge drivers:** each coupled edge of a random 3-regular graph (24 of 120 edges) has its own narrowband AR(2) driver. The follower reads it at a fixed phase φ_e (delay φ_e/2πf_e), through a slow random warp W_e, and through g(u) = cos θ_e·u + sin θ_e·(u² − 1)/√2. Strength w_e, phase, band f_e, form θ_e and warp W_e are set per edge independently.
- **Family B, VAR:** lag-2 coupling on a 3-regular graph; each edge's coupling is split (1 ± α_e)/2 between its two directions.
- **Family C, GGM:** AR(1)-filtered innovations whose correlation mixes a direct structure (sparse precision) with an observed-hub structure (spoke–spoke edges marginal only) at matched mean |r|.
- **Family K, Kuramoto–Sakaguchi (nonlinear):** oscillators on a random tree with antisymmetric lags α_e; x = cos θ plus noise.

Knobs (five levels each). Level knobs set one value on every coupled edge; dispersion knobs set centre ± Δ/2 on a balanced random half of the edges:

| Family | Knob | Levels |
|---|---|---|
| A | lag level, φ | 0, π/8, π/4, 3π/8, π/2 |
| A | lag dispersion, φ = π/4 ± Δ/2 | Δ = 0 … π/2 |
| A | band level, f | 0.06, 0.15, 0.25, 0.35, 0.44 |
| A | band dispersion, f = 0.25 ± Δ/2 | Δ = 0 … 0.32 |
| A | form level, θ | 0 … π/3 |
| A | form dispersion, θ = π/6 ± Δ/2 | Δ = 0 … π/3 |
| A | warp level, W (SD of delay, samples) | 0 … 3 |
| A | warp dispersion, W = 1.5 ± Δ/2 | Δ = 0 … 3 |
| B | direction level, α | 0 … 1 |
| B | direction dispersion, α = 0.5 ± Δ/2 | Δ = 0 … 1 |
| C | mediated share λ (two observed hubs, mean \|r\| matched at 0.30) | 0 … 1 |
| K | lag level α; lag dispersion π/8 ± Δ/2; band level f₀ | 0 … π/4; Δ = 0 … π/4; 0.06 … 0.44 |

Baseline B0: strength CV 0.5, driver pole radius r 0.75, φ = π/8 ± π/16, f = 0.1, θ = 0, W = 0, private 0.2, measurement noise 0.05, no mixing, Gaussian marginals. B1 differs in strength CV 1.0 and r 0.9 (longer autocorrelation).

Family A nuisance sweeps (all axes scored, construct knobs at baseline):
- strength CV: 0.1 … 1.5
- measurement noise: 0 … 1
- T: 500 … 4000
- r: 0.6 … 0.95
- marginal sinh map β: 0 … 2
- instantaneous spatial mixing μ: 0 … 0.5 (the volume-conduction pattern, which also makes strong edges zero-phase)
- M: 8 … 24

Sign grid: each Family A construct knob at its zero and maximum level, crossed with each nuisance at its minimum and maximum.

FT surrogates (one random phase per frequency shared across channels, which preserves the sample cross-periodogram) are computed for the form and warp sweeps, and for lag dispersion as a positive control.

Replicates: 6 recordings × 2 windows per main-sweep condition, 4 × 2 per grid condition. Uncertainty: 95% t-intervals over replicate-level paired differences.

## Pre-registered criteria

Δ is the score at a knob's maximum minus at its zero level.

1. **Sign stability (hard).** A pair fails if its Δ for its own knob has a 95% interval entirely on the wrong side of zero in either baseline's main sweep, at any nuisance extreme of the sign grid, or in a second family where the construct exists. An interval that includes zero is recorded as insensitive at that setting, not as a reversal.
2. **Sensitivity.** The predicted sign, with the interval excluding zero, in the main sweep at both baselines. Otherwise the pair is insensitive and dropped.
3. **Convergence.** Within a construct, pairs from different estimator families that pass 1–2 agree in sign. The axis is multi-method if two or more families pass.
4. **Selectivity.** SR = \|Δ(own knob)\| / max \|Δ(other construct knob)\| over each knob's full design range (dispersion knob for organisation, level knob for level). Core axis: passes 1–2 with SR ≥ 2. Sub-axis: passes 1–2 with SR < 2. Drop: fails 1 or 2. Nuisance sensitivities are reported as Δ over the realistic range divided by \|Δ(own knob)\|. They do not set the verdict; they feed the real-data bound.
5. **Surrogate test.** For warp pairs: if the surrogate keeps ≥ 0.8 of the real response with the same sign, warp is second-order and redundant with lag and band; if ≤ 0.5, it is a non-Gaussian timing sub-axis of form; otherwise undetermined. Form pairs must keep ≤ 0.5 of their response on surrogates to count as reading non-Gaussianity. Lag pairs should keep ≥ 0.8 (method check).
6. **Correction check.** Organisation is corrected first for reliability: z_cross/√(rel_a·rel_b), with cross-window z and reliabilities. It is then corrected for strength spread by inverting the toy model (multiplicative, or additive σ_c = σ_s√(z⁻² − 1)) with the strength member's reliability-corrected spread. A correction is usable for a pair if, across the strength-CV and noise sweeps, it cuts the nuisance response by at least half while keeping at least 75% of the own-knob response.
7. **Real-data bound.** For each cohort contrast and passing pair, report the paired or between-group d on:
   - raw organisation;
   - reliability-corrected organisation;
   - strength-corrected organisation, where criterion 6 passes;
   - an indicative autocorrelation-explained change: the simulated gradient from the r sweep times the measured change in channel lag-1 autocorrelation.

   A named sign claim stands if the corrected d keeps the sign with \|d\| ≥ 0.5 and the indicative autocorrelation change is smaller than the observed change. Mixing cannot be measured in the cohorts and is reported as a caveat.

## Files

- `build_subset.py`: writes `spis.yaml`.
- `generators.py`: families A, B, C and K, plus the FT surrogate.
- `run.py`: builds the condition table and runs pyspi in parallel, resumable, into `results/representation/character-axes-validation/` (git-ignored).
- `analyse.py`: scores, criteria, tables and figure.

## Status (2 November 2026, 18:30)

The run moved to Gadi (user decision, 2 November): `qsub -v VALIDATION_OUT=<dir>/raw research/representation/character_axes/validation/jobs/farm.pbs` re-runs every main and post hoc window there into a fresh directory, so the corpus comes from one platform. The stopped local run (3,758 windows on an Apple M5) is not used. The runner is resumable: finished windows are skipped. dcorr is computed only where the r²|dcorr pair is evaluated (main sweeps, form-knob grid cells, form surrogates), via `spis_nodcorr.yaml`. When the run finishes: `python -m research.representation.character_axes.validation.analyse --refresh` writes `tables/` and `figures/selectivity.*` and runs the cohort bound (`bound.py`, banks from `--banks`).

Post hoc addition: the pre-registered levels are medians over all ordered entries. Family A has 24 coupled edges of 120, so those medians mostly read uncoupled edges, as they would in sparse ECoG. Each level is therefore also reported over the strongest quartile of edges, labelled post hoc.

Pilot facts:
- Mediation with one observed hub is non-monotone: uniform mediation is invisible to organisation. With two hubs, the population corr(r², pcor²) falls 0.91 → 0.67 at matched mean |r|.
- Kuramoto edges lock at the target lag (0.79 vs 0.79 rad).

## Results

Pending.
