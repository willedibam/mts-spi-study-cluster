# Transfer protocol: one character-axis result per new dataset

Frozen 24 November 2026, before any new dataset is chosen or computed. Rewritten the same day, before any dataset was frozen, into a data-agnostic core and an anaesthesia annex. The first version (`18d235d7`) carried macaque ECoG choices into every field.

Reading rules: [`docs/interpreting-spi-pairs.md`](../../../docs/interpreting-spi-pairs.md). The gates, nulls and units come from the [axis screen prereg](../atlas/axis_screen/prereg.md) and [result](../atlas/axis_screen/result.md). The named-axis construction in [`character_axes/protocol.md`](../character_axes/protocol.md) is a different design and is not replaced.

Every dataset goes through the core. An anaesthesia or other neural dataset also runs the annex. A change to this file after a dataset's freeze record is committed applies only to later datasets, and is dated here with its reason.

## Purpose and scope

- **Purpose.** The paper adds one result per field. Without a fixed procedure, choosing the axis, window, null and contrast after seeing each dataset is a forking-paths problem. The core fixes those choices, or the rule that makes them, before any contrast is computed.
- **Scope.** Multichannel time series with M ≥ 10 channels (45 edges), continuous or tie-corrected values, and a contrast between states that metadata defines. Data that fail the preconditions below are reported as out of scope, not forced through.

## The axis: the MI anchor only

- **Readout.** z(`mi_gaussian`, `mi_kraskov_NN-4_DCE-AUTO`): Gaussian MI from r against KSG MI (k 4, DCE AUTO Theiler window), computed on normal-scored windows.
- **Net per window.** real − mean of the null draws (below).
- **Why this axis and no other.** It was not chosen by searching the SPI–SPI space. No search of that space under these gates has made a claim; the full-pyspi screen's FDP was .47 (Tier 2, below). It is used because its reading does not depend on the data. For normal-scored marginals, flexible MI minus Gaussian MI is the KL departure from the Gaussian with the same covariance (guide, worked example). It is also the only axis that held in macaque, mouse and human data.
- **Expectation.** Whether it moves in a non-neural contrast is unknown. A null result reported with its numbers is a valid outcome.
- **Channels.** If M ≤ 32, all channels. Otherwise 32 channels drawn with a seed fixed in the freeze record, from channels passing quality control, before any state label is read.
- **Implementation.** [`axis_screen/stats.py`](../atlas/axis_screen/stats.py), pyspi-free and validated against pyspi (per-window z within .005). It is used because surrogate banks need the statistic hundreds of times per window.
  - Both statistics are in upstream pyspi's re-ranked `benchmarked_p90` (161 SPIs, #211, pinned at 736ee5c).
  - So a new-p90 corpus of the dataset gives the real-data anchor directly. Where one exists, its anchor z must agree with `stats.py` within .005 per window, or the run stops.
- **Generalised driver.** None exists yet. Until one does, a dataset runs through a study-local adapter around `stats.py`, `run.py`, `score.py` and `mechanism.score`, frozen in its record.

## Preconditions (checked on baseline-state data, before the freeze)

1. **Ties.**
   - If any channel's distinct-value fraction in baseline windows is below .99, as with counts or coarse quantisation: add uniform dither within each channel's quantisation step before normal-scoring, with a seed fixed in the record.
   - Then the null calibration (5) must pass on data quantised the same way. KSG MI and normal-scoring are biased by ties without this.
2. **Deterministic cycles.**
   - Declare any known cycle, such as daily, weekly, seasonal or tidal.
   - Either the long window is at most a tenth of the cycle period, or the cycle is removed before windowing: subtract each channel's mean and divide by its SD by cycle phase, both estimated from baseline-state data.
3. **Within-window stationarity.**
   - Per window, take the log ratio of the second half's variance to the first half's, averaged over channels.
   - Compare it with the same ratio in that window's own null draws.
   - If more than .10 of baseline windows exceed the null's 95th percentile (twice the nominal rate; a choice, not derived), the null does not match the data. The readout is untestable unless a different null is frozen (4).
4. **Null.**
   - Default: 3 spectrum-final joint IAAFT draws over the channel set, with seeds fixed by window.
   - A field may freeze another null that keeps the same linear structure: cross-spectra, marginals, and any declared cycle. Examples are a cycle-preserving or block surrogate. The record states the reason.
   - Whatever the choice, it must pass (5).
5. **Null calibration (on synthetic data built from the dataset).**
   - **Linear arm.** Take multivariate phase-randomised draws of baseline windows, rank-remapped to each channel's marginals and quantised as the data are, then run the full pipeline. Their net must be within 1 null SD of 0.
   - **Non-Gaussian arm.** Multiply half the channels of those draws by one shared slow gain, exp(g(t)), with g Gaussian, SD .5 and correlation time 10τ. Size and timescale are choices, not derived. The net must fall beyond 3 null SD, since a shared gain lowers the anchor's z (`mi_pair/edges-prereg.md`). This checks that the pipeline can detect a known departure at the dataset's T and M.
   - Fail either arm and the anchor is untestable in that dataset.

## Windows

The inputs are metadata and baseline-state data only. The outputs are logged in the record before any target-state window is processed.

- **Sampling.** Resample, with anti-aliasing, to the lowest rate that keeps the band the field treats as signal, cited in the record. Record fs and the median integrated autocorrelation time τ (in samples).
- **Stationarity limit.** L_max is the longest window that fits within a state epoch as metadata defines it, and within the cycle rule (precondition 2).
- **Ladder.** Take T from 250, 500, 1000, 2000, 4000 and so on, up to L_max. Choose the shortest T at which both statistics pass the reliability floor. Shorter windows give more windows and less drift. If none passes, the anchor is untestable.
- **Window counts.** At least 10 baseline windows per session, so each split half has 5 or more.
- **Reporting.** Report T, T/(2τ) and T/M.

## Reliability floor

- Reliability of a statistic is the mean pairwise edge-profile correlation between baseline-state windows within a session, averaged over sessions.
- Both statistics must reach .3 at the chosen T. Below that, the readout is **untestable**: its Δ is printed descriptively and enters no test. No exceptions. A PRIOS row tested at .25 is the precedent this rules out.
- The other usability rules follow the prereg: finite in ≥ 95% of windows of every state used, and baseline real |z| < .98. For mapped near-ceiling pairs, judge by the null SD instead (guide, ceiling rule).

## Contrasts and units

- **Unit.** The independent replicate is declared in the record: animal, patient, site, city or station. Sessions or recordings nest within units.
- **States.** Baseline and target states are defined from metadata or an external index fixed in the record, such as a clinical label, a geomagnetic index threshold or an incident log. They are never defined from SPI output.
- **Effect.** Session Δ = mean net (target windows) − mean net (baseline windows), within a recording. Unit Δ is the mean of its session Δ.
- **Null SD (the SD unit).**
  - Split each session's baseline windows by start time into two halves of equal count; take second half − first half.
  - Scale each session's split-half Δ by √((1/n_b + 1/n_t) / (2/n_half)) to its contrast's window counts. Here n_half is the count per half, and n_b and n_t are the baseline and target counts. This assumes independent windows; for 5-against-5 and 10-against-10 it gives the prereg's 1/√2.
  - Null SD = SD over sessions of the scaled split-half Δ.
  - Report every effect as **raw Δz** and as **Δz / null SD**.
- **Nuisance contrast.** Where the data have a real but irrelevant change (for example weekday against weekend, if that is not the question), declare it in the record. If there is none, say so: gate 4 is not applied, and the result is labelled as lacking a specificity check.

## Gates (axis screen, generalised to n units)

1. **Unit test.**
   - n ≥ 6: exact sign-flip test over unit Δ, p < .05, two-sided unless the record predicts a sign with a stated reason.
   - n = 3–5: t over unit means, p < .05, with every unit the same sign.
   - n < 3: a descriptive case study, with no test.
   - If the smallest attainable p is above the threshold, the readout is underpowered by design, not failed.
2. **Size.** |Δ| ≥ 3 null SD.
3. **Split-half.** |mean split-half Δ| ≤ .25 |Δ|.
4. **Nuisance.** |nuisance Δ| ≤ .33 |Δ|, where declared.
5. **Covariates.**
   - Within-recording OLS of window net on state, strength (mean |r|, SD |r|, covariance participation ratio) and marginal shape (median over channels of 1.4826·MAD/SD on the window before normal-scoring).
   - The adjusted Δ keeps the sign and is ≥ .5 |Δ|.
   - Units agree in sign: all of them when n ≤ 5; a one-sided sign-flip test at p < .05 when n ≥ 6.
6. **Reliability.** The disattenuated Δ (z divided by √(rel_A · rel_B) per session × state and version) keeps the sign, with the unit rule of gate 5.

**Reported, not gated:**
- real Δ and null Δ separately, since a net can move through its null;
- double-centred Δ and mean-profile Δ;
- spectral covariates (spectral entropy, peak frequency, low-band fraction);
- Δ per subgroup where the data have subgroups.

## Edge-level report (optional)

- Only when a calibrated drop test ([`mi_pair/droptest.py`](../atlas/mi_pair/droptest.py)) says the change is localised.
- Selection: on even windows, measuring on odd and the reverse, against leverage-matched random drops.
- First repeat its synthetic calibration (concentrated, broad and skewed-broad plants) at the dataset's M and T.
- Decision rule:
  - localised if f < .8 and below every random draw;
  - broad if f ≥ .9 or inside the random range;
  - otherwise unresolved.
- Without "localised", no edge or node is named.

## Wording

- Per the guide.
- A pass reads: across pairs, the organisation of dependence departs further from (or nearer to) the Gaussian prediction than the null process with the same linear structure gives.
- It is not "more nonlinear coupling", not a level change, and not a mechanism.
- If the cycle rule removed a cycle, say that the result is conditional on that removal.

## Per-dataset freeze record

Commit `research/representation/transfer/<dataset>/freeze.md` before computing any target-state output. It states:

- source, version and licence;
- unit and count;
- states, with their definitions and sources;
- nuisance contrast, or none;
- channels, quality control and the KSG subset seed;
- fs, τ, cycles and their handling, L_max, the chosen T, T/(2τ) and T/M;
- tie handling and dither seed;
- the null, with the stationarity check and both calibration arms;
- baseline reliability of both statistics;
- predicted sign and reason, or two-sided;
- the adapter commit;
- pyspi pin and computation version, if any pyspi output is used;
- expected and worst-case compute.

## Annex A: anaesthesia and other neural recordings

The macaque, mouse and human anaesthesia results came from 100 Hz ECoG. These additions hold there and are not assumed elsewhere.

- **Signs.** Drug − awake is negative for the anchor (one-sided allowed). Sleep moves it the same way at about a tenth of the size.
- **Windows.** Long T = 1000 (10 s at 100 Hz), where the anchor's evidence is strongest. Short T = 200 (2 s) for `l1|linf` only. The ladder above still decides if 1000 fails the floor.
- **Nuisance.** Eyes open against eyes closed, or the nearest real non-drug change.
- **Secondary axes** (Holm over those testable; selected on macaque data, so confirmatory only for a new anaesthesia dataset):

  | Readout | Statistics | Window | Channels |
  |---|---|---|---|
  | `r2\|mcd2` | classical against MinCovDet covariance, squared | long | all |
  | `l1\|linf` | L1 against Chebyshev distance | short | all |
  | `mcd2\|ksg` | MinCovDet squared covariance against KSG MI (non-minimal; read through `r2\|mcd2` and `r2\|ksg`) | long | KSG subset |

  - Reported, not tested: `pcorr|pcorr_mcd`, the macaque's largest row. Robust precision needs T well above M, and it failed the PRIOS floor.
  - Reported, not tested: `r2|ksg` and `dcorr|ksg`, which are the anchor's own axis.
  - No T/M cut-off applies. MinCovDet passed at T/M 7.9 and was near-classical at 1.6, and PRIOS kept `r2|mcd2` and `mcd2|ksg` usable at 2.9–5.9.
  - The new p90 has no squared covariance. The squared members are taken by squaring the signed `cov_*` entries; this is inferred and unchecked against pyspi's `squared` option.
- **Reading.** MinCovDet reads as time-heterogeneous dependence at the copula level, on normal-scored windows only. `l1|linf` has only the literal reading: the tail of x − y relative to its typical size.
- **Expectation.** Both secondaries depend on window length and T/M, and both are copula-level only. They will often be untestable even in new anaesthesia data.
- **Envelope null.** `findings.md` calls a local-envelope-preserving null the decisive test of the MI pair, and a sibling session is building it. It is reported, along with `env_E` from [`gaussref.py`](../atlas/axis_screen/gaussref.py), until its calibration is committed. Making it a gate is a dated change here.
- **Open.** The macaque T/(2τ) at 10 s is to be measured and entered here before the first anaesthesia dataset is frozen.

## Tier 2 (exploratory): deferred

**What it will need:**
- a dataset-agnostic screen over all usable minimal pairs of the new `benchmarked_p90`;
- selection on one half of units or windows and confirmation on the other;
- the identical pipeline on a true no-change control (within-baseline split-half) to estimate FDP = control discoveries ÷ discoveries, with no claims above a threshold frozen with Tier 2;
- a real nuisance contrast reported separately.

**Why it is deferred:**
- The macaque full-pyspi screen ([`movers/matched-result.md`](../atlas/movers/matched-result.md)) estimated FDP at .47 against its frozen .2 and made no claims. That control was eyes-open, a real change (19 of its 26 hits were a disjoint lag change), and a within-awake split-half control found 0. Which control defines FDP must be fixed first.
- The datasets are not chosen.
- It needs the refactor below.

**Refactor to propose with Tier 2 (not done):** one gate module, parameterised by unit, state names and n, replacing the macaque-specific logic in `axis_screen/score.py`, `mechanism.score` and `movers/matched.py`, with the macaque runs as regression tests.
