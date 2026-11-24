# Transfer protocol: frozen character-axis results per new dataset

Frozen 24 November 2026, before any new dataset was chosen or computed. Two rewrites, both before any dataset was frozen:
- **24 November:** split into a data-agnostic core and an anaesthesia annex. The first version (`18d235d7`) carried macaque ECoG choices into every field.
- **25 November:** the core is organised around axis cards. The previous version (`372c94fe`) tested only the MI anchor, which made it a protocol for one pair rather than for the interpretable axes of the SPI–SPI space.

Sources:
- **Reading rules:** [`docs/interpreting-spi-pairs.md`](../../../docs/interpreting-spi-pairs.md).
- **Readings, tiers, nulls and current status of each pair:** [pair catalogue](../atlas/pair_catalogue/catalogue.md), with [`catalogue.json`](../atlas/pair_catalogue/catalogue.json) as the authority.
- **Gates and units:** the [axis screen prereg](../atlas/axis_screen/prereg.md) and its [result](../atlas/axis_screen/result.md).
- **Planted generators and nuisance ratios:** the [construction validation](../character_axes/validation/README.md).
- **The 13 named axes of 2 November** ([`character_axes/protocol.md`](../character_axes/protocol.md)) are not the unit here. Several of them failed construction (warp, direction, form organisation). Cards are built from catalogue entries instead.

Every dataset goes through the core. An anaesthesia or other neural dataset also runs the annex. A change to this file after a dataset's freeze record is committed applies only to later datasets, and is dated here with its reason.

## Purpose and scope

- **Purpose.** The paper adds results per field. Choosing pairs, windows, nulls and contrasts after seeing each dataset is a forking-paths problem. This file fixes those choices, or the rules that make them, before any contrast is computed.
- **Unit of analysis: an axis card.** A card is one catalogue entry: a minimal pair (one assumption apart), or a principled pair whose Gaussian relation is exact. The card fixes everything that differs between axes. The core fixes what is shared.
- **Scope.** Multichannel time series with M ≥ 10 channels (45 edges), continuous or tie-corrected values, and a contrast between states that metadata defines. Data or cards that fail the preconditions are reported as out of scope, not forced through.

## What a card fixes

1. **Pair as computed.** Exact pyspi identifiers, or the pyspi-free implementation. For a mapped pair, the map; judge it by the null SD, not by z₀ > .98 (guide, ceiling rule).
2. **The assumption difference, and every consequence of it** (guide, rule 3). This includes the level confound: for lag and band, level moves organisation as much as dispersion does.
3. **Tier, which sets the null and the readout** (catalogue "Tier and null"):
   - **Higher-order.** Both statistics have a Gaussian expectation fixed by second-order structure.
     - Readout: the net, real − mean of the null draws.
     - Default null: 3 joint spectrum-final IAAFT draws.
     - The claim is "beyond linear".
   - **Linear-spectral.** IAAFT is degenerate here (net ≈ 0 by construction).
     - Readout: the raw value, with the zero-phase and stationary Gaussian nulls sharing the state's cross-spectrum ([`discovery.py`](../atlas/discovery.py) `spectral_null` and `lag_null`).
     - The claim is linear-spectral character.
   - **Linear-static.** Readout: the raw value with strength controls. Null: stationary Gaussian with the state's covariance.
4. **Window preparation.**
   - Normal-scored or z-scored, per card.
   - The MinCovDet and maximum-based pairs are read only on normal-scored windows.
   - Canberra's gain invariance holds only on z-scored windows.
5. **Applicability checks on baseline data.** Examples: a defined narrowband oscillation for envelope and phase pairs; T/M for precision; fixed M and montage rank across states for mediation.
6. **Controls, run with the card's exact readout at the dataset's T and M** (synthetic data built from the dataset, below):
   - a negative arm, for higher-order cards;
   - a positive plant, a generator that moves the property the card is named for.
7. **Status and wording**, from the catalogue and the construction results.
8. **Companions that must be reported beside it:** level readouts, link pairs, and channel spectra.

## Eligibility and wording

- **Confirmatory testing** requires the card to be named in the dataset's freeze record before any target-state window is processed. It also requires the card to pass its applicability checks, the reliability floor and, for higher-order cards, the negative arm. A readout can be replicated without a positive plant. The plant governs what may be said, not whether a test may be run.
- **Multiplicity.** Holm over the record's confirmatory cards.
- **Wording**, by the strongest evidence the card has:
  - **Character reading:** the card passed the frozen axis-screen gates in some field, and its positive plant passes in this dataset.
  - **Construction-supported reading:** the construction validation passed for this exact readout, and the plant passes in this dataset. The nuisance ratios it measured are stated beside the result.
  - **Literal reading only:** in every other case, including when no plant is built yet or the plant fails here.
- **Null results.** A null is "absent" only if the plant passes. Otherwise it is "undetected at this T and M".
- **On current evidence few cards reach a character reading.**
  - The MI anchor.
  - MinCovDet at long T and Chebyshev at short T, both selected on macaque data and readable only on normal-scored windows.
  - Every other card starts with literal wording.
  - The linear-tier cards also have nuisances the construction found unbounded (noise, strength spread, mixing).

## Axis cards

The pairs are checked against upstream `benchmarked_p90` at 736ee5c (161 SPIs). Where a member is not in that set, its derivation is given. Plants marked *built* exist in the repo; *to build* means none exists yet.

| Card | Pair as computed | Tier, prep | Applies when | Positive plant | Status (source) |
|---|---|---|---|---|---|
| `mi-ksg` (anchor) | `mi_gaussian` · `mi_kraskov_NN-4_DCE-AUTO`; pyspi-free in [`stats.py`](../atlas/axis_screen/stats.py) | higher-order, normal-scored | always (after the tie rule) | shared slow gain, *built* ([`gain_sim.py`](../atlas/axis_screen/gain_sim.py)) | axis in macaque; replicated in PRIOS and mouse |
| `mcd` | `cov_EmpiricalCovariance` · `cov_MinCovDet`, both squared (r2\|mcd2); the `pcorr_*` pair is reported | higher-order, normal-scored only | T/M reported, no cut-off | network switch in high-amplitude epochs, *built* ([`decoupling_generator.py`](../character_axes/decoupling_generator.py)); shared gain also produces the sign | axis at 10 s in macaque; absent at 2 s |
| `chebyshev` | `pdist_cityblock` · `pdist_chebyshev` (l1\|linf) | higher-order, normal-scored only; fixed sample count | the maximum depends on T and the spectrum, so all states use the same T | intermittent large divergence of a subset of pairs, *to build* | axis at 2 s in macaque; fails at 10 s |
| `rank-linear` | Pearson on normal scores, mapped by (6/π) arcsin(r/2), against `spearmanr` | higher-order, normal-scored | always | joint bursts, *built* (`edge_drivers(burst=…)` in [`generators.py`](../character_axes/validation/generators.py)) | unmapped pair failed at ceiling in macaque; mapped form untested; real − IAAFT r\|ρ reads joint bursts by construction (sub-axis; selectivity against lag and band untested) |
| `xi` | `mi_gaussian` · `xi_tau-0` | higher-order, normal-scored | T long enough for xi's noise (10 s in macaque) | non-monotone coupling at matched r, *to build* | post hoc candidate at 10 s |
| `canberra` | `pdist_canberra` against its analytic Gaussian value f_G(r) | higher-order, z-scored; Gaussian FT surrogate | no channel with a floor or offset near zero | correlation that changes with amplitude, *built* as a catalogue synthetic (principle 4) | beyond IAAFT on normal-scored windows only; the z-scored form, which carries the reading, is untested |
| `envelope-gamma` | `pec` against \|γ\|² from the analytic signal (pyspi-free, not yet written) | higher-order, z-scored | a narrowband oscillation in a declared band | shared envelope modulation, *to build* | untested |
| `gc-te` | level median(`te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1` − `gc_gaussian_k-1_kt-1_l-1_lt-1`) | higher-order level, z-scored | T long enough for the TE reliability floor | form level, *built* (family A) | z unreliable; the level is a construction sub-axis (IAAFT keeps .14 of its response; band level moves it more) |
| `mediation` | r² against partial r², from `cov_EmpiricalCovariance` and `pcorr_EmpiricalCovariance` squared | linear-static, z-scored | fixed M and montage across states; T/M reported | GGM with a mediated share, *built* (family C) | construction core only for strong instantaneous mediation; raw-only in macaque. The new p90 has no `prec-sq_*`, and pcorr² differs from prec² by diagonal scaling, so this readout is not the construction readout |
| `band` | high share and spread from `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25` and `…fmin-0-25_fmax-0-5`; z between them reported | linear-spectral, z-scored | fs and the signal band meaningful | band dispersion, *built* (family A) | high share and spread are construction core (SR 3.7, 5.3, one generator); the z failed in macaque (not drug-specific) |
| `lag` | z(`cohmag_…fmin-0_fmax-0-25`, `icoh_…fmin-0_fmax-0-25`) and the level icoh/√cohmag | linear-spectral, z-scored | a declared band with oscillatory coupling; the field's analogue of volume conduction declared | lag dispersion and lag level, *built* (family A) | construction sub-axis: lag level moves organisation about twice as much as dispersion, in the opposite direction; failed in macaque (floor, covariates) |

Any other catalogue entry becomes a card by filling the eight fields above and dating the addition here. Until then it is Tier 2 only.

## Shared preconditions (checked on baseline-state data, before the freeze)

1. **Ties.**
   - If any channel's distinct-value fraction in baseline windows is below .99, add uniform dither within each channel's quantisation step. The seed is fixed in the record.
   - Controls are run on synthetic data quantised the same way.
2. **Deterministic cycles.** Declare any known cycle (daily, weekly, seasonal, tidal). Either the longest window is at most a tenth of the cycle period, or the cycle is removed first: subtract the mean and divide by the SD by cycle phase, both from baseline data.
3. **Within-window stationarity.**
   - Per window, take the log ratio of the second half's variance to the first half's, averaged over channels.
   - Compare it with the same quantity in stationary linear draws of that window.
   - If more than .10 of baseline windows exceed the draws' 95th percentile, no card's null matches the data unless a different null is frozen. The .10 is twice the nominal rate, a choice, not derived.
4. **Controls on synthetic data built from the dataset.**
   - The base is multivariate phase-randomised draws of baseline windows, rank-remapped to each channel's marginals and quantised as the data are.
   - **Negative arm (higher-order cards).** The net on these linear draws must be within 1 null SD of 0. If it fails, the card is untestable in this dataset.
   - **Positive arm.** Each card's plant is added to the base at a size fixed in the record. The anchor uses a shared gain exp(g), SD .5, correlation time 10τ, on half the channels; size and timescale are choices, not derived. The readout must move beyond 3 null SD in the plant's predicted direction. If it does not, the card keeps a literal reading only.
   - **Alternative nulls.** A field may freeze a null other than the tier default if it keeps the same linear structure (cross-spectra, marginals, any declared cycle), for example a cycle-preserving or block surrogate. The record states why, and the null must pass the negative arm.

## Windows

The inputs are metadata and baseline-state data only. Outputs are logged in the record before any target-state window is processed.

- **Sampling.** Resample, with anti-aliasing, to the lowest rate that keeps the band the field treats as signal (cited). Record fs and the median integrated autocorrelation time τ in samples.
- **Stationarity limit.** L_max is the longest window that fits within a state epoch as metadata defines it, and within the cycle rule.
- **Ladder, per card.** T is 250, 500, 1000, 2000, 4000 … up to L_max. Each card uses the shortest T at which both its statistics pass the reliability floor. If none passes, the card is untestable.
- **Fixed T.** A card whose catalogue status is tied to one window length (`mcd` long, `chebyshev` short) is tested at that T only if it passes the floor there, and is otherwise untestable.
- **Counts.** At least 10 baseline windows per session.
- **Reporting.** T, T/(2τ) and T/M per card.
- **Channels.**
  - kNN, MinCovDet and other costly statistics use all channels if M ≤ 32. Otherwise they use 32 channels drawn by a seed fixed in the record, from channels that pass quality control, before any state label is read.
  - Mediation uses all channels and the same set in every state.

## Reliability floor

- Reliability is the mean pairwise edge-profile correlation between baseline windows within a session, averaged over sessions.
- Both statistics of a card must reach .3 at its T. Below that, the card is untestable: its Δ is printed descriptively and enters no test. No exceptions; a PRIOS row tested at .25 is the precedent this rules out.
- Usability also requires finite values in ≥ 95% of windows of every state used, and baseline real |z| < .98 for unmapped pairs.

## Contrasts and units

- **Unit.** The independent replicate is declared in the record (animal, patient, site, city, station). Sessions nest within units.
- **States.** Baseline and target states come from metadata or an external index fixed in the record. They are never defined from SPI output.
- **Effect.** Session Δ = mean readout over target windows − mean over baseline windows, within a recording. The readout is the net or the raw value, per tier. A unit's Δ is the mean of its sessions' Δ.
- **Null SD.**
  - Split each session's baseline windows by start time into two halves of equal count, and take the second half minus the first.
  - Scale each session's value by √((1/n_b + 1/n_t)/(2/n_half)), where n_half is the count per half, and n_b and n_t are the baseline and target counts. This assumes independent windows.
  - The null SD is the SD of these scaled values over sessions.
  - Report every effect as raw Δ and as Δ / null SD.
- **Nuisance contrast.** Declare a real but irrelevant change where the data have one. If none, say so; gate 4 is then not applied, and the result is labelled as lacking a specificity check.

## Gates (per card; axis screen, generalised to n units)

1. **Unit test.**
   - n ≥ 6: exact sign-flip test over unit Δ, p < .05, two-sided unless the record predicts a sign with a reason.
   - n = 3–5: t over unit means, p < .05, with every unit the same sign.
   - n < 3: descriptive only.
   - If the smallest attainable p is above the threshold, the card is underpowered by design, not failed.
   - Holm applies across the confirmatory cards.
2. **Size.** |Δ| ≥ 3 null SD.
3. **Split-half.** |mean split-half Δ| ≤ .25 |Δ|.
4. **Nuisance.** |nuisance Δ| ≤ .33 |Δ|, where declared.
5. **Covariates.**
   - Within-recording OLS of the window readout on state plus covariates.
   - The adjusted Δ keeps its sign and is ≥ .5 |Δ|.
   - Units agree in sign: all of them when n ≤ 5; a one-sided sign-flip test at p < .05 when n ≥ 6.
   - **Every card:** strength (mean |r|, SD |r|, covariance participation ratio) and marginal shape (median over channels of 1.4826·MAD/SD before normal-scoring).
   - **Linear-spectral and linear-static cards also:** channel spectral summaries (spectral entropy, peak frequency, low-band fraction). If under .2 of either state's windows fall inside the other state's range on these summaries, the adjustment is unidentifiable (as in NeuroTycho, where 4% did). The result is then worded "not separable from channel spectra". The .2 is a choice, not derived.
   - **Lag and band cards:** the card's level readout is reported beside it, and a lag claim also reports band high share.
6. **Reliability.** The disattenuated Δ (z divided by √(rel_A · rel_B) per session, state and version) keeps its sign, with the unit rule of gate 5.

**Reported, not gated:**
- real Δ and null Δ separately, since a net can move through its null;
- double-centred and mean-profile Δ;
- spectral covariates for higher-order cards;
- Δ per subgroup.

## Edge-level report (optional)

- Only when a calibrated drop test ([`mi_pair/droptest.py`](../atlas/mi_pair/droptest.py)) says a card's change is localised.
- Its synthetic calibration (concentrated, broad and skewed-broad plants) is first repeated at the dataset's M and T.
- Decision rule:
  - localised if f < .8 and below every random draw;
  - broad if f ≥ .9 or inside the random range;
  - otherwise unresolved.
- Without "localised", no edge or node is named.

## Wording

- Per the guide, at the card's wording tier.
- A higher-order pass reads: across pairs, the organisation of [the card's property] departs further from (or nearer to) the Gaussian prediction than the null process with the same linear structure gives.
- A linear-tier pass reads as linear-spectral or linear-static character, with the spectral-separability label from gate 5.
- No card's pass is a level change of its property, unless it is a level readout. No pass is a mechanism.
- If a cycle was removed, the result is conditional on that removal.

## Per-dataset freeze record

Commit `research/representation/transfer/<dataset>/freeze.md` before computing any target-state output. It states:

- source, version and licence; unit and count; states with definitions and sources; nuisance contrast or none;
- confirmatory cards, each with its applicability checks, T, T/(2τ), T/M, null, plant size, both control results, baseline reliability, and predicted sign with reason (or two-sided);
- descriptive cards;
- channels, quality control and the subset seed;
- fs, τ, cycles and their handling, L_max; tie handling and dither seed; the stationarity check;
- the adapter commit (no generalised driver exists yet; each dataset uses a study-local adapter around `stats.py`, `run.py`, `score.py` and `mechanism.score`, plus the `discovery.py` nulls for linear-tier cards);
- the pyspi pin and computation version, if pyspi output is used. Where a new-p90 corpus gives a card's statistics, its per-window z must agree with the pyspi-free value within .005, or the run stops;
- expected and worst-case compute.

## Annex A: anaesthesia and other neural recordings

The macaque, mouse and human anaesthesia results came from 100 Hz ECoG. These defaults hold there and are not assumed elsewhere.

- **Confirmatory cards and windows:**
  - `mi-ksg` at T = 1000 (10 s), predicted negative (drug − awake), one-sided.
  - `mcd` at T = 1000 (r2|mcd2), with the sign from the macaque axis-screen result.
  - `chebyshev` at T = 200 (2 s), with the sign from the macaque axis-screen result.
  - These were selected on macaque data. They are confirmatory only in a new anaesthesia dataset.
  - The ladder still applies if a card fails the floor at its T.
- **Non-minimal link, reported:** `mcd2|ksg` (MinCovDet squared covariance against KSG MI, on the KSG subset), read through `mcd` and `mi-ksg`.
- **Reported, not tested:**
  - `pcorr|pcorr_mcd`, the macaque's largest row: robust precision needs T well above M, and it failed the PRIOS floor.
  - `r2|ksg` and `dcorr|ksg`, which sit on the anchor's own axis.
- **Window ratios.** No T/M cut-off applies. MinCovDet passed at T/M 7.9, and PRIOS kept `mcd` usable at 2.9–5.9.
- **Derived members.** The squared covariance members are taken by squaring the signed `cov_*` matrices. This is inferred and unchecked against pyspi's `squared` option.
- **Nuisance.** Eyes open against eyes closed, or the nearest real non-drug change. Sleep moves the anchor the same way at about a tenth of the size.
- **Envelope null.** `findings.md` calls a local-envelope-preserving null the decisive test of the MI pair, and a sibling session is building it. It is reported, along with `env_E` from [`gaussref.py`](../atlas/axis_screen/gaussref.py), until its calibration is committed. Making it a gate is a dated change here.
- **Open.** The macaque T/(2τ) at 10 s is to be measured and entered here before the first anaesthesia dataset is frozen.

## Tier 2 (exploratory): deferred

**What it will need:**
- a screen over all usable catalogue entries of the new `benchmarked_p90`;
- selection on one half of units or windows and confirmation on the other;
- the identical pipeline on a true no-change control (a within-baseline split-half) to estimate FDP = control discoveries ÷ discoveries, with no claims above a threshold frozen with Tier 2;
- a real nuisance contrast reported separately.

**Why it is deferred:**
- The macaque full-pyspi screen ([`movers/matched-result.md`](../atlas/movers/matched-result.md)) estimated FDP at .47 against its frozen .2 and made no claims. But that control was eyes-open, a real change: 19 of its 26 hits were a disjoint lag change, and a within-awake split-half control found 0. Which control defines FDP must be fixed first.
- The datasets are not chosen.
- It needs the refactor below.

**Refactor to propose with Tier 2 (not done):** one gate module, parameterised by unit, state names, n and card tier, replacing the macaque-specific logic in `axis_screen/score.py`, `mechanism.score` and `movers/matched.py`. The macaque runs would serve as regression tests.
