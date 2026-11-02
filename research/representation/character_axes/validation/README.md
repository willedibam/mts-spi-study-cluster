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

- `build_subset.py`: writes `spis.yaml` (stages 1–2) and, with `--stage3`, `spis_stage3.yaml`.
- `generators.py`: families A, B, C and K, plus the FT surrogate; family D and joint bursts are options of `edge_drivers`.
- `run.py`: builds the condition table and runs pyspi in parallel, resumable, into `results/representation/character-axes-validation/raw` (git-ignored) or `$VALIDATION_OUT`.
- `analyse.py`: scores, criteria 1–6, `tables/` and `figures/selectivity.*`; reads the same `$VALIDATION_OUT`.
- `bound.py`: criterion 7, the cohort bound (`tables/bound.csv`).
- `figure.py`: the selectivity figure.
- `stage3.py`: stage 3 scoring and rules (`tables/stage3_*.csv`).
- `jobs/farm.pbs`, `jobs/analyse.pbs`, `jobs/analyse3.pbs`: the Gadi farm and the analysis jobs.

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

## Stage 2: second generator family and level-aware readouts

Pre-registered on 3 November 2026, before any stage-2 window was generated. Two questions:
- **Second family.** Mediation and direction passed as core in one generator each (families C and B). Criterion 1 asks for a second family where the construct exists. Their SR also divided an own response in C or B by responses to family A knobs. Putting both constructs in family A makes the SR within one family.
- **Level-aware readout.** Can a readout of lag or band dispersion avoid the level confound found in stage 1?

### Generators (`generators.py`)

- **Direction in family A** (`edge_drivers(alpha=…)`):
  - Each coupled edge also carries an independent driver read the other way. α splits the edge's variance (1 + α)/2 forward and (1 − α)/2 backward.
  - α = 1 is the stage-1 family A, and α = 0 is symmetric.
  - Graph, weights and phases come from the same network seeds as stage 1.
  - As in family B, α also moves phase lag, because the two directions' imaginary coherency cancels at α = 0. In linear systems, direction and lag are coupled. That is why SR against family A's lag knobs is the test.
- **Mediation in family A** (`hub_drivers(lam=…)`):
  - λ = 0 is family A's direct edges: the stage-1 graph, weights and phases, at f = 0.1.
  - At λ = 1, two observed hubs, drawn from the network seed, each broadcast one driver to their 7 spokes. Each spoke reads its hub's driver at phase π/8 ± π/16.
  - Intermediate λ mixes the two sources with weights √(1 − λ) and √(λ·gain).
  - The gain is found by bisection so the mean |r| of the source signal stays at its λ = 0 value, as family C holds mean |r|.
  - Unlike C, hubs carry private noise and spokes read them with a delay, so conditioning at lag 0 removes mediated dependence only partly.
- **Direction in family K** (`kuramoto_tree(asym=…)`):
  - The leader pulls the follower with K_e(1 + α) and is pulled back with K_e(1 − α).
  - α = 0 is the stage-1 family K.
  - K's B1 has strength CV 1.0 and phase noise 0.01 (longer coherence), mirroring A's B1, which changes strength spread and autocorrelation.

### Design and criteria

- **Knobs:** α ∈ {0, 0.25, 0.5, 0.75, 1} in families A and K; λ ∈ {0, 0.25, 0.5, 0.75, 1} in family A.
- **Baselines:** B0 and B1 as in stage 1, with 6 recordings × 2 windows per condition.
- **Sign grid:** each family A knob at its zero and maximum, crossed with noise {0, 1}, mixing 0.5 and strength CV {0.1, 1.5}, with 4 × 2 windows per cell.
- **Size:** 520 windows; dcorr is not computed for any of them.
- **Predicted signs:**
  - direction organisation corr(A, Aᵀ) falls with α;
  - direction level (median gc asymmetry) rises with α;
  - mediation organisation falls with λ.

Rules for every mediation and direction measure, with Δ from a knob's zero level to its maximum as before:
1. **Sign stability (hard, criterion 1).** A measure fails if its own-knob Δ has a 95% interval entirely on the wrong side at either baseline in family A or K, or in any family A grid cell.
2. **Replication.** The predicted sign, with the interval excluding zero, at both baselines in family A.
3. **Selectivity in A.** SR_A is |Δ own (A, B0)| divided by the largest |Δ| among family A's lag, band, form and warp knobs (stage 1) and the other stage-2 family A knob.
4. **Verdict:**
   - **core, two generators:** stage-1 core, replicated in A, and SR_A ≥ 2;
   - **sub-axis, two generators:** replicated in A, but stage-1 sub-axis or SR_A < 2;
   - **⟨stage-1 verdict⟩, one generator:** not reversed, but insensitive in A at either baseline;
   - **drop (sign reversal in a second family):** fails rule 1.

   Family K counts for rule 1 only; its sensitivity is reported. Risk stated in advance: in strongly locked oscillators the follower is close to a delayed copy of the leader, so spectral Granger asymmetry may be weak or distorted. A reversal in K would still drop the measure, as criterion 1 says.
5. **Nuisance sensitivity in the new networks (reported).** The change in the measure at knob level 0 between each nuisance's grid extremes, divided by |own Δ at B0|. Mixing is grid 0.5 minus B0. Grid cells have their own networks, so these are unpaired.

For all stage-1 measures, responses to the stage-2 knobs are reported, together with an SR that includes them. Stage-1 verdicts are not changed.

### Level-aware readouts

**Organisation conditioned on level is not pre-specified, because stage 1 already shows that a linear adjustment fails.** The lag level itself responds to lag dispersion: −0.64 of its own response for the median version, −0.33 for the top-quartile version. Take the stage-1 B0 endpoint ratio of Δ lag organisation to Δ lag level on the lag-level sweep as the slope, and subtract slope × level from lag organisation (coh\|icoh 0–0.25):
- with the median level, the lag-dispersion response becomes −0.47 + 0.64 = +0.17, a reversal;
- with the top-quartile level, it becomes −0.47 + 0.33 = −0.14, about 30% of the own response, below the 75% that criterion 6 asks a correction to keep.

A z between two profiles mixes level and dispersion through the toy's reliability and CV terms, so the level has to be removed per edge, before aggregation.

**Pre-specified alternative: spread.** The spread is the SD of a per-edge contrast over the strongest quartile of entries (by broadband cohmag):
- lag: icoh/√cohmag (0–0.5);
- band: cohmag_H/(cohmag_L + cohmag_H).

Both are predicted to rise with their construct's dispersion knob. They are scored on the stage-1 corpus with criteria 1–4 (family K lag dispersion is lag's second family), plus one more rule:
- **Level-aware:** |Δ level knob| ≤ 0.5·|Δ dispersion knob| at both B0 and B1.

These are per-edge dispersion estimators, not z organisation.

### Pilot

Run after the pre-registration commit (42bd982) and before scoring, on generator output only, with no pyspi:
- **Stage 1 unchanged:** stage-1 generator calls are bit-identical to f1bc779 for 8 conditions across families A, B, C and K.
- **Matched mean |r|:** mediation holds the source mean |r| at its λ = 0 value (0.079 at B0, 0.086 at B1).
- **Mediation exists, partly:** at λ = 1, same-hub spoke pairs have marginal |r| 0.135 (B0) and partial |r| 0.076. Conditioning on the hub removes only about 45% of the mediated dependence.
- **Direction exists:** in family A, the lag-2 cross-correlation asymmetry between leader and follower is 0.145 at α = 1 and 0.009 at α = 0.
- **Kuramoto stays coupled:** directed family K keeps tree-edge |r| at 0.59–0.74.

### Running it

- **Farm:** `qsub -l ncpus=48,mem=190GB,walltime=00:45:00 -v STAGE=2,VALIDATION_OUT=<dir2>/raw research/representation/character_axes/validation/jobs/farm.pbs`, with `<dir2>` = `/g/data/ql44/we2614/mts-spi-study/representation/validation/character-axes-stage2-<commit>`.
- **Analysis:** `jobs/analyse.pbs` with `STAGE2_OUT=<dir2>/raw` scores both corpora and writes `tables/second_family.csv`.

### Stage 2 results (3 November 2026)

Corpus: Gadi farm job 182978008 at 42bd982, 520 windows, 0 pyspi errors, 21.7 SU. Analysis job 182978377 at 2e82bad scored both corpora together. Stage-1 verdicts and SRs are unchanged. Second-family rows are in `tables/second_family.csv`.

| Measure | Family A, B0 / B1: Δ (d) | SR in A | Family K, B0 / B1: Δ (d) | Verdict |
|---|---|---|---|---|
| mediation LW | −0.021 (−1.2) / +0.012, insensitive | 0.28 (lag level) | — | core, one generator |
| mediation empirical | −0.018, insensitive / +0.021, insensitive | 0.22 | — | core, one generator |
| direction sgc A\|Aᵀ | −0.258 (−3.2) / −0.250 (−2.8) | 0.96 (band dispersion) | **+0.578 (4.7) / +0.506 (2.5), reversed** | drop (sign reversal in K) |
| direction gc A\|Aᵀ | **+0.141 / +0.088, reversed** | 0.22 | **+0.603 / +0.443, reversed** | drop |
| direction median gc asymmetry | insensitive; reversed at noise 0 and strength CV 0.1 | 0.11 | insensitive | drop |
| direction gc asymmetry, top quartile (post hoc) | +0.091 (0.7), insensitive / +0.050 (2.0) | 0.20 | +0.254 (2.3) / insensitive | sub-axis, one generator |

- **Direction is not a validated axis.** The sgc readout keeps its sign in family A, the linear-delay family where direction and phase lag go together. But it reverses in directed Kuramoto, so pre-registered criterion 1 drops it. Even without family K it would not be core: band dispersion moves it as much as direction does (SR_A 0.96), and noise moves it 1.2 times as much. Gaussian GC reverses in both families.
  - How the readout fails in Kuramoto has not been examined; inferred, it reads the locked phase relation rather than the coupling asymmetry.
  - This is consistent with work showing that coupling direction between synchronised oscillators needs phase-dynamics methods (Rosenblum and Pikovsky 2001), and that Granger direction can invert with unequal noise (Vinck et al. 2015).
- **Mediation keeps its stage-1 core label only from family C,** strong and instantaneous Gaussian-graphical mediation at mean |r| 0.30. In family A the mediation is lagged and partial (conditioning on the hub removes about 45%), at mean |r| 0.08.
  - There the readout is sensitive only at B0 (d −1.2), and lag level moves it 3.5 times as much.
  - Its nuisance ratios are 3.8 for noise, −7.7 for mixing 0.5 and 1.6 for strength CV, so it does not read mediation selectively.
  - Stage 1's SR of 3.25 compared C's response with family A knobs. In a single family, the readout is not selective.
- **Stage-2 knobs and stage-1 measures.** The family A direction knob moves the lag measures, because α changes phase lag. Including it, the SR of lag coh\|icoh 0–0.25 falls from 1.60 to 0.89 and the lag level's from 1.59 to 1.27. The core band measures are unchanged (ppc L\|H 2.06, high share 3.74).

Level-aware spreads:
- **Band spread** (SD of the per-edge high share over the strongest quartile of entries) passes as **core** with SR 5.28 (5.23 with the stage-2 knobs) and own d 5.9. It is **level-aware**: band level moves it −0.09 (B0) and +0.02 (B1) times as much as band dispersion. Its sign holds at every grid cell; it is insensitive only at β = 2. Family K has no band-dispersion knob, so this is one generator.
- **Lag spread** is a sub-axis (SR 0.98; band level moves it as much). It is **not level-aware**: lag level moves it −1.1 (B0) and −1.2 (B1) times as much as lag dispersion. It is insensitive in family K, at β = 2 and at mixing 0.5. FT surrogates keep 1.00 of its response: linear-spectral, as expected.

What holds after stage 2:
- **Band** is the one construct with level-aware readouts that pass as core: the high-share level (two generators) and band spread (one generator, level-aware). ppc L\|H organisation passes too, but band level moves it 0.8 times as much.
- **Mediation** is core only for strong instantaneous mediation, in one generator.
- **Direction** is dropped.
- **Lag** stays a sub-axis with no level-aware readout.
- **Form** has only the te − gc level, a sub-axis.

## Stage 3: spectral location within a band and joint-burst coupling

Pre-registered on 3 November 2026, before any stage-3 window was generated. The core sufficiency checks ([findings](../findings.md), "Core sufficiency") leave recurring survivors after adjusting for the core and strength: readouts that measure frequency in sample units (fixed-lag timing, spectral peak, best shift) and linear–rank agreement (form_monotone, whose propofol residual survives IAAFT). Under the protocol a survivor joins the core only after passing construction. Two questions:
- **Location.** Does any p90 readout read the frequency at which channels couple, inside the low band, selectively against lag, phase and driver bandwidth r, and from the coupling rather than from channel spectra? The band construct splits at fs/4, so alpha slowing stays inside the low band.
- **Bursts.** Does linear–rank agreement read a non-Gaussian form of coupling, and do IAAFT surrogates separate it from heavy-tailed marginals (β)?

### Generators

- **Location, family A:** the stage-3 knobs set each coupled edge's driver frequency f inside the low band. Level: f ∈ {0.06, 0.085, 0.11, 0.135, 0.16} on every edge. Dispersion: f = 0.11 ± Δ/2, Δ ∈ {0, 0.025, 0.05, 0.075, 0.10}. Private components stay at 0.1.
- **Location, family K:** f₀ over the same five values, as the second family for the level (a Kuramoto edge couples at its oscillators' frequency).
- **Family D, coupling against channel spectra:** family A with private power 6 per channel, against about 3 from its edges, so a private AR(2) at f_p dominates the channel spectra while coupling sits at f_c. Knobs: f_c over the five values with f_p = 0.11, and f_p over them with f_c = 0.11. B0 and B1 as in family A.
- **Joint bursts, family A** (`edge_drivers(burst=…)`): both ends of an edge read its driver through one shared envelope a_e(t) ∝ exp(σ_e v_e(t)), where v_e is a smooth Gaussian path with a correlation time of about 25 samples. a_e is scaled to mean a² = 1 in each window, so edge variance and Pearson coupling are unchanged in expectation while coupling concentrates in joint high-amplitude episodes. Level: σ ∈ {0, 0.375, 0.75, 1.125, 1.5}. Dispersion: σ = 0.75 ± Δ/2, Δ up to 1.5. Envelopes use their own random stream, so σ = 0 reproduces family A exactly. There is no second family.
- **Form θ** is the stage-1 knob.
- The stage-3 dispersion knobs draw their ± signs independently of the baseline phase dispersion. Stage 1's band, form and warp dispersion knobs reused the phase-dispersion sign vector, so in stage 1 those dispersions were perfectly correlated with phase across edges. That is noted here and not changed.

### Corpus

A separate corpus with its own pyspi subset, `spis_stage3.yaml` (93 statistics, `build_subset.py --stage3`): every constituent of the timing_fixed_lag, timing_best_shift, spectral_peak, form_monotone and tail axes and of their levels, as `character_axes.py` defines them, plus the core's constituents and the top-quartile ranking. The statistics differ from stages 1–2, so every knob used for selectivity is re-run here with the same seeds, and therefore the same time series:
- family A at B0: the eight stage-1 construct knobs, direction α and mediation λ (stage 2), and the nuisances strength CV, noise, r, β and mixing (T and M are not repeated);
- family A at B0 and B1: the four stage-3 knobs; families K and D at B0 and B1;
- FT and IAAFT surrogates (`mtsspi.surrogates.multivariate_iaaft`, exact marginals): burst level and dispersion at B0 and B1, form level and β at B0;
- sign grid: each stage-3 knob at its zero and maximum, crossed with r {0.6, 0.95}, noise {0, 1}, mixing 0.5, strength CV {0.1, 1.5} and β 2.

Replicates are 6 recordings × 2 windows per main condition and 4 × 2 per grid cell: 2,972 windows. Each window records its pyspi version.

### Readouts

"Top quartile" is the median over the entries ranked in the top quartile by broadband cohmag, as `band_spread` ranks them. "All entries" is the cohort definition (`level_scores`). Axis organisation is `axis_scores` over each axis's full pair list, as in the cohorts. Learned sign means the sign of the family A response at B0 when its interval excludes zero; B1, the grid and the second families must then agree, so these measures have one fewer independent check.

| Measure | Definition | Own knob | Sign |
|---|---|---|---|
| lag-1 ratio | top-quartile median of corr_pearson_tau-1 / cov; ≈ ρ(1)·cos 2πf for narrowband coupling, whatever its phase | location level | − |
| AR(2) frequency f̂ | per entry, fit s_k = a₁s_(k−1) + a₂s_(k−2) at k = 2, 3 to the lag-0 to lag-3 correlations (cov, corr_pearson_tau-1 to 3); f̂ = arccos(a₁/2√−a₂)/2π, NaN without a complex pole pair; top-quartile median | location level | + |
| AR(2) frequency spread | SD (ddof 0) of f̂ over the top quartile | location dispersion | + |
| spectral-peak level | cohmag max/mean, broadband; top quartile and all entries | location level | learned |
| best-shift level | xcorr² max − cov²; top quartile and all entries | location level | − (a fixed phase is a delay of φ/2πf samples, which shrinks as f rises) |
| fixed-lag, spectral-peak and best-shift axes | organisation | location dispersion | learned |
| linear–rank level | \|ρ_S\| − \|r\|; top quartile and all entries | burst level | − (Pearson weights covariance by amplitude², ranks do not) |
| linear–rank axis; r\|ρ_S; r\|τ_K | organisation (form_monotone axis; Pearson against Spearman and Kendall) | burst dispersion | − |
| tail level; tail axis | Chebyshev/Euclidean distance; organisation (robust against empirical covariance, Chebyshev and cityblock against Euclidean) | burst level; burst dispersion | learned |

The tail readouts were not requested; they are included because the tail axis was built for rare-event coupling. For every burst measure the residual (real minus IAAFT, per recording) is also scored, as the cohort residual is computed. The core readouts (CORE at b56d908) are scored for cross-talk with the new knobs.

### Rules

Δ, intervals and d are as before.
1. **Sign stability (hard).** No interval may lie entirely on the wrong side in family A at B0 or B1, at any grid cell or, for location levels, in family K and in family D's f_c sweep at B0 and B1.
2. **Sensitivity.** The sign, with the interval excluding zero, in family A at B0 and B1.
3. **Selectivity.** SR3 = \|Δ own (A, B0)\| / max \|Δ\| over every other construct knob of family A in this corpus (lag, form θ, warp, direction α, mediation λ, the other stage-3 construct, and band for bursts) and the construct's named nuisance: r for location, β for bursts. Location leaves out the band knobs, which move the same variable over a wider range; their responses are reported.
4. **Location also needs:**
   - **reads coupling frequency** (levels): in family D, the f_c response has the sign with its interval excluding zero at B0 and B1, and \|Δ f_p\| ≤ 0.5·\|Δ f_c\| at both;
   - **incremental over the core:** \|d own\| exceeds every core band readout's \|d\| (high share, band spread, ppc L\|H) on the same knob, at B0 and B1.
5. **Bursts also need non-Gaussianity:** IAAFT keeps ≤ 0.5 of the own response at B0 and B1. Reported: FT kept; for β, IAAFT kept (method check) and FT kept. The residual is reported as selective if its own response is ok at B0 and B1 and β and θ each move it at most half as much.
6. **Dispersion readouts** (organisation and spread) need level-awareness for core, as stage 2 required of spreads: \|Δ level knob\| ≤ 0.5·\|Δ dispersion knob\| at B0 and B1.

Verdict: drop if rule 1 or 2 fails; core if SR3 ≥ 2 and every applicable requirement of rules 4–6 passes; sub-axis otherwise, listing what failed. Nuisance ratios for strength CV, noise, r, β and mixing are reported. Mediation λ is family A's, where stage 2 found the mediation readout weak.

### Predictions stated in advance

From first principles, not from data:
- **lag-1 ratio:** r moves it about as much as location. Over r 0.6 → 0.95 at f = 0.1, r·cos 2πf goes 0.49 → 0.77; over f 0.06 → 0.16 at r = 0.75 it goes 0.70 → 0.40. So SR3 is near 1.
- **f̂** is built not to depend on r. pyspi's lagged correlation is symmetrised, ½(c(τ) + c(−τ)), which for narrowband coupling at phase φ is proportional to cos φ·cos 2πfτ. It vanishes for quadrature coupling, so lag level (φ → π/2) is expected to destabilise both lag-correlation readouts.
- **Ratios of lagged correlations** cancel the channel normalisation, so they should follow f_c and not f_p in family D. Coherence-based readouts (spectral peak, band) are normalised per frequency by the channel spectra and should respond to f_p as well.
- **spectral-peak level** reads bandwidth (r) more than location.
- **best-shift level** reads delay in samples, so lag level should move it more than location (SR3 < 1).
- **fixed-lag axis:** it averages signed pairs whose profile correlation starts positive at τ = 1, 2 and negative at τ = 3–5, because cos 2πfτ changes sign at f = 0.11. Dispersion moves those pairs in opposite directions, so no sign is derivable.
- **linear–rank level:** β moves it opposite to bursts, because Pearson falls under heavy-tailed marginals and ranks are invariant. IAAFT should reproduce β's effect (keep ≥ 0.8) but not the bursts'. To first order, θ moves it only through strength, because rank and Pearson correlation both scale with cos θ.
- **core band high share:** it may respond to within-band location through AR(2) spectral leakage across fs/4, so it is not assumed that the core misses within-band location.

A location or burst readout whose verdict is core will be proposed for CORE with its exact constituents. For bursts, the residual is the version comparable with the cohort's real − IAAFT residual.

### Running it

- **Farm:** `qsub -l ncpus=192,mem=760GB,walltime=01:30:00 -v STAGE=3,VALIDATION_OUT=<dir3>/raw research/representation/character_axes/validation/jobs/farm.pbs`, with `<dir3>` = `/g/data/ql44/we2614/mts-spi-study/representation/validation/character-axes-stage3-<commit>`.
- **Analysis:** `qsub -v VALIDATION_OUT=<dir3>/raw .../jobs/analyse3.pbs` writes `<dir3>/analysis/tables/stage3_{verdicts,responses,core}.csv`, which are copied to `tables/`.
