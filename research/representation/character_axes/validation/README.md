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
- `run.py`: builds the condition table and runs pyspi in parallel, resumable, into `results/representation/character-axes-validation/raw` (git-ignored) or `$VALIDATION_OUT`.
- `analyse.py`: scores, criteria 1–6, `tables/` and `figures/selectivity.*`; reads the same `$VALIDATION_OUT`.
- `bound.py`: criterion 7, the cohort bound (`tables/bound.csv`).
- `figure.py`: the selectivity figure.
- `jobs/farm.pbs`, `jobs/analyse.pbs`: the Gadi farm and the analysis job.

## Status (2 November 2026, complete)

Run on Gadi (user decision, 2 November), so the corpus comes from one platform; the stopped local run (3,758 windows on an Apple M5) is not used. Farm: `qsub -v VALIDATION_OUT=<dir>/raw research/representation/character_axes/validation/jobs/farm.pbs` (resumable; dcorr is computed only where the r²|dcorr pair is evaluated, via `spis_nodcorr.yaml`). Analysis: `qsub -v VALIDATION_OUT=<dir>/raw research/representation/character_axes/validation/jobs/analyse.pbs` writes `<dir>/analysis/{tables,figures}` and runs the cohort bound against the six cohort banks on scratch. The tables are copied into `tables/` here and committed; the figure is regenerated, not tracked.

Post hoc addition: the pre-registered levels are medians over all ordered entries. Family A has 24 coupled edges of 120, so those medians mostly read uncoupled edges, as they would in sparse ECoG. Each level is therefore also reported over the strongest quartile of edges, labelled post hoc.

Pilot facts:
- Mediation with one observed hub is non-monotone: uniform mediation is invisible to organisation. With two hubs, the population corr(r², pcor²) falls 0.91 → 0.67 at matched mean |r|.
- Kuramoto edges lock at the target lag (0.79 vs 0.79 rad).

## Results (2 November 2026)

Corpus: Gadi farm job 182947411 at commit 8d239c6, pyspi 3.0.0. All 4,236 main and 480 post hoc windows completed. No window has a pyspi error or an all-NaN statistic, and dcorr is present exactly where it is routed. Raw windows are in `/g/data/ql44/we2614/mts-spi-study/representation/validation/character-axes-8d239c6/raw`. Tables in `tables/` come from analysis job 182952105 at f1bc779. Δ is the response from a knob's zero level to its maximum. SR is the selectivity ratio of criterion 4.

### Verdicts (criteria 1–4)

Organisation pairs, scored on their dispersion knob (direction on its level knob):

| Construct | Passing pairs (verdict, SR) | Dropped | Notes |
|---|---|---|---|
| form | gc\|te_ksgDCE (sub-axis, 0.18) | mi_g\|mi_ksgDCE reverses (main B0, r 0.95); tlmi_g\|tlmi_ksgDCE, r²\|dcorr and mi_g\|mi_kernel are insensitive at B0 or B1 | The one passing pair moves 5.6 times more with lag level than with form dispersion, and its surrogates keep 1.8–2.0 times the response (below), so it reads linear-spectral structure, not form. |
| lag | coh\|icoh 0–0.25 (sub-axis, 1.60); pec\|pec_orth (1.29); plv\|wpli (0.88); ppc\|dswpli (0.80); coh\|icoh 0–0.5 (0.66) | coh\|icoh 0.25–0.5 is insensitive at B1 | Three estimator families agree in sign, so lag is multi-method. The worst other knob is band level (coherency, ppc) or warp level (plv, pec). In family K every lag pair is insensitive (interval includes zero, no reversal). Most are also insensitive at β = 2 or noise 1. |
| band | ppc L\|H (core, 2.06); coh L\|H (sub-axis, 1.96) | sgc L\|H reverses at both baselines and in 9 of 12 grid cells | ppc and coh sit on the SR = 2 boundary and are not distinguishable. sgc's reversal is consistent: its organisation rises with band dispersion. |
| mediation | Ledoit–Wolf (core, 3.25); empirical (core, 2.88) | — | One generator (family C) and one estimator family. SR divides family C's own response by family A knob responses. |
| direction | sgc A\|Aᵀ (core, 3.12); gc A\|Aᵀ (sub-axis, 1.43) | te A\|Aᵀ is insensitive at B0 | One generator (family B). Every nuisance ratio for sgc is at most 0.09 in magnitude. |
| warp | — | eucl\|dtwSC reverses at noise 0; dtwSC\|dtw is insensitive | — |

Levels, scored on their level knob:
- **form:** median(te − gc) is a sub-axis (SR 0.78; band level moves it more). Its surrogates keep only 0.14 of the response, so it is the one pre-registered measure that reads non-Gaussian form. median(ksgDCE − mi_g) is insensitive.
- **lag:** icoh/√cohmag is a sub-axis (1.59).
- **band:** high share is core (3.74).
- **direction:** median gc asymmetry is a sub-axis (1.42).
- **warp:** dtwSC/eucl is insensitive.
- **Post hoc top-quartile versions:**
  - lag (1.31), band (1.99), direction (1.66) and warp (0.19) are sub-axes. The warp version's surrogates keep 0.99 of its response, so it reads linear timing.
  - form ksgDCE − mi_g is insensitive, and te − gc reverses at β = 2.

Response to the construct's other knob, divided by |own Δ| (`other own knob/own` in `verdicts.csv`; not part of SR):
- **Lag organisation responds to lag level about twice as much as to lag dispersion, in the opposite direction.** The ratio is +1.6 to +2.1 for the coherency, ppc\|dswpli and pec pairs, and +0.7 for plv\|wpli. Raising every edge's phase lag raises lag organisation, and spreading the lags lowers it. A rise in lag organisation is therefore not evidence of more uniform lags unless the lag level is checked. The lag level icoh/√cohmag is less confounded: its ratio to lag dispersion is −0.64.
- **Band organisation** moves 0.8 times as much with band level as with band dispersion, also in the opposite direction. The band high-share level is unaffected by band dispersion (0.03).
- **Form organisation** falls 1.4 to 38 times more with form level than with form dispersion, so these pairs read form level, if anything. At B0 the tables do not separate how much of that is spectral.
- **Direction organisation** (sgc A\|Aᵀ) barely moves with direction dispersion (−0.17).

Nuisance sensitivity is Δ over the sweep divided by |own Δ| at B0. Some nuisances move the lag and band organisation pairs about as much as their own knob:
- **lag pairs:** mixing (−0.4 to −1.4) and the marginal map β (0.9 to 1.3 in magnitude). Measurement noise also does (−0.2 to −1.1) for every lag pair except pec.
- **band pairs:** noise (−1.3), driver pole radius r (−0.8) and β (0.7 to 0.8).

Mediation's largest ratio is mixing (−0.6).

### Surrogates (criterion 5)

- **Lag pairs:** they keep 0.94–1.12 of their B0 response on FT surrogates. The method check passes, because lag is linear-spectral.
- **Form organisation:** gc\|te_ksgDCE keeps 1.98 (B0) and 1.77 (B1), so its response is spectral. The other form pairs fail criteria 1–2, so the test does not apply to them. The form level median(te − gc) keeps 0.14 at both baselines.
- **Warp:** both organisation pairs are dropped at B0 and B1, so criterion 5 is moot there. Post hoc results at strong coupling are below.

### Corrections (criterion 6)

No correction is usable for any passing pair.
- **Reliability correction does not remove noise.** For coh\|icoh 0–0.5 the noise ratio rises 0.48 → 1.47, and for ppc\|dswpli 0.98 → 1.77. In these generators, measurement noise lowers the population SPI values rather than adding sampling error, so it is not a reliability effect.
- **Strength corrections fail one way or the other.** Most lose the own response: coh\|icoh 0–0.5 goes from d 2.9 to 0.6, plv\|wpli from 8.2 to 1.0, and the band pairs from 7.6 to 0.7–1.1. The rest keep the response but do not halve the noise and strength ratios (ppc\|dswpli, pec\|pec_orth).

Corrections pass only for dropped pairs (r²\|dcorr, mi_g\|mi_kernel, sgc L\|H). Strength spread and noise therefore stay unbounded in criterion 7, and mixing cannot be measured.

### Post hoc: strong coupling

Baseline PHS: degree 2, private 0.05, no measurement noise; one baseline, not pre-registered.
- **Form dispersion:** three form organisation pairs respond with the predicted sign, and their surrogates keep at most 0.27 of the response:
  - tlmi_g\|tlmi_ksgDCE: Δ −0.046, d −2.1, kept 0.11;
  - r²\|dcorr: Δ −0.0035, d −1.3, kept 0.02;
  - mi_g\|mi_kernel: Δ −0.016, d −1.5, kept −0.27.
- **Form level:** both form levels respond (d 1.9 to 6.2 over the pre-registered and top-quartile versions), with surrogates keeping at most 0.29.
- **Form organisation against form level:** three pairs respond strongly (mi_g\|mi_ksgDCE, tlmi_g\|tlmi_ksgDCE and mi_g\|mi_kernel; d −6.7 to −12.9), but their surrogates respond 1.7 to 3.3 times as much. The u² term moves power to twice the driver frequency, and those pairs read that spectral change. r²\|dcorr also responds (d −10.0), but its surrogate keeps 0.13, so it reads the non-Gaussian part.
- **Warp:** eucl\|dtwSC responds to warp dispersion (Δ −0.064, d −2.0) while its surrogate moves the other way (kept −1.77). By criterion 5, that is non-Gaussian timing, a sub-axis of form.

So form and warp organisation are measurable when coupling is strong, but not at the cohort-like coupling of B0 and B1.

### Cohort bound (criterion 7)

Contrasts:
- NeuroTycho, awake → anaesthesia: 15 sessions, paired.
- CHB-MIT, seizure-free → ictal: 22 people, paired.
- HMC, N2 → REM: 39 people, paired.
- AD/FTD, control → AD: 29 and 36 people, between groups.
- LEMON, young → old: 91 and 41 people.
- Propofol, baseline → moderate: 20 people, paired.

The table lists every claim with raw |d| ≥ 0.5. "Floor" is the lower clip on the reliabilities in the disattenuation: 0.05 as coded, with 0.001 for sensitivity.

| Cohort | Pair | d raw | d corrected (floor 0.05 / 0.001) | Indicative AC Δ / observed Δ | Stands |
|---|---|---|---|---|---|
| NeuroTycho | lag coh\|icoh 0–0.25 | 1.61 | 1.57 / 0.85 | −0.05 / 0.14 | yes, both floors |
| NeuroTycho | mediation LW | −2.10 | −0.69 / −0.69 | −0.01 / −0.16 | yes, both floors |
| NeuroTycho | mediation empirical | −2.15 | −0.67 / −0.67 | −0.01 / −0.16 | yes, both floors |
| HMC | lag coh\|icoh 0–0.5 | 0.83 | 0.67 / 0.67 | −0.02 / 0.11 | yes, both floors |
| NeuroTycho | lag ppc\|dswpli | 2.31 | 1.23 / 0.46 | −0.07 / 0.17 | only at 0.05 |
| NeuroTycho | band ppc L\|H | 3.14 | 3.04 / 0.34 | −0.29 / 0.39 | only at 0.05 |
| NeuroTycho | band coh L\|H | 2.67 | 1.52 / 0.67 | −0.31 / 0.30 | no (AC magnitude rule) |
| CHB-MIT | lag coh\|icoh 0–0.25, plv\|wpli; band ppc L\|H | 0.65, 0.57, −0.52 | 0.42, 0.17, −0.35 | — | no |
| HMC | lag coh\|icoh 0–0.25 | 0.55 | 0.43 | — | no |
| AD/FTD | form gc\|te_ksgDCE | −0.65 | −0.12 | — | no |
| LEMON | lag coh\|icoh 0–0.5 | 0.62 | 0.40 | 0.23 / 0.01 | no |
| Propofol | lag plv\|wpli, ppc\|dswpli | 0.88, 0.58 | 0.55, 0.57 | 0.13 / 0.05, 0.08 / 0.04 | no (AC magnitude rule) |

Reading:
- **Reliability.** Two-thirds of NeuroTycho's raw mediation effect is reliability: both members' reliabilities fall (d −1.05 and −0.99).
- **Opposite-sign AC predictions.** The NeuroTycho band coh L\|H claim fails only because the pre-registered rule compares magnitudes. The indicative autocorrelation change has the opposite sign, so autocorrelation works against the observed rise. The same holds for the other NeuroTycho lag and band rows. For mediation the prediction has the same sign as the observed change but is a tenth of its size.
- **What "stands" covers.** The bound covers reliability and autocorrelation only. Noise, strength spread and mixing are unbounded (criterion 6), and so are changes in other constructs. Every lag pair has SR < 2; the HMC pair has 0.66, and N2 → REM changes spectra. A standing claim means the sign survives the measured nuisances, not that the named construct moved. The NeuroTycho and HMC rises in lag organisation fit higher lag level as well as more uniform lags.

### Implications for the axes

- **Supported by construction:**
  - band: ppc L\|H and coh L\|H organisation, and the high-share level;
  - mediation: cov²\|prec², from one generator;
  - direction: sgc A\|Aᵀ, from one generator and insensitive to every nuisance.
- **Lag** is multi-method and sign-stable in family A, but not selective (SR 0.66–1.60) and insensitive in family K. Lag level moves lag organisation about twice as much as lag dispersion. A lag organisation claim needs band level and lag level ruled out; the lag level is the cleaner lag reading.
- **Form organisation** fails at cohort-like coupling. Its one passing pair reads spectra. The form level median(te − gc) is the only non-Gaussian reader (sub-axis).
- **Warp** fails as an organisation axis. Its post hoc level reads linear timing.
- No strength correction works, so z-axes support existence and sign claims under a nuisance bound. Magnitude stays out of scope.
