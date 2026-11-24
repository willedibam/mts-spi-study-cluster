# Transfer protocol: one character-axis result per new dataset

Frozen 24 November 2026, before any new dataset is chosen or computed. Every dataset added to the paper beyond the anaesthesia anchor goes through Tier 1 unchanged. Tier 2 is deferred (bottom). Reading rules: [`docs/interpreting-spi-pairs.md`](../../../docs/interpreting-spi-pairs.md). Source of the axes, gates and nulls: the [axis screen prereg](../atlas/axis_screen/prereg.md) and [result](../atlas/axis_screen/result.md). The named-axis construction in [`character_axes/protocol.md`](../character_axes/protocol.md) is a different design (axes built from minimal pairs before cohorts) and is not replaced here.

Changes to this file after a dataset's freeze record is committed apply only to later datasets. Each change is dated here with its reason.

## Expectation, stated before any data

- The MI anchor z(`mi_gaussian`, `mi_kraskov`) is the only axis expected to transfer. It is the only screen axis that holds without normal-scoring and at both window lengths. Its record: macaque (every gate), mouse induction (9/9 mice) and human PRIOS (6/7 patients, p .023, after a stated deviation) (`atlas/findings.md`; `axis_screen/result.md`, status).
- The other frozen axes come from anaesthesia ECoG and depend on window length and on T/M. The MinCovDet axis was absent at 2 s. In PRIOS, `pcorr|pcorr_mcd` was below the reliability floor and `r2|mcd2` was near zero (4/7). `l1|linf` was below the PRIOS reliability floor. Outside anaesthesia ECoG they will probably be reported as untestable, which is a valid outcome and not a failure of the protocol.
- No sign is known outside anaesthesia. The macaque signs (drug − awake: all negative) are predictions only for anaesthesia datasets.

## Tier 1: the frozen axis set

All axes are F1 (beyond linear): net per window = real − mean of 3 spectrum-final joint IAAFT draws, on normal-scored windows. The reference implementation is [`axis_screen/stats.py`](../atlas/axis_screen/stats.py) (pyspi-free, validated against pyspi to profile r ≥ .9992). No generalised driver exists yet. Until the deferred skeleton is written, a dataset runs through a study-local adapter around `stats.py`, `run.py`, `score.py` and `mechanism.score`, with the adapter frozen in that dataset's record.

| Role | Readout | Statistics | Window | Applicable when |
|---|---|---|---|---|
| **Primary** | `mi_g\|ksg` | Gaussian MI from r against KSG MI (k 4, DCE AUTO Theiler window) | long | always (reliability floor permitting) |
| Secondary | `r2\|mcd2` | classical against MinCovDet covariance, squared | long | T ≥ 7.5 M |
| Secondary | `l1\|linf` | L1 against Chebyshev distance | short | always (reliability floor permitting) |
| Secondary | `mcd2\|ksg` | MinCovDet squared covariance against KSG MI (non-minimal) | long | T ≥ 7.5 M |
| Reported | `pcorr\|pcorr_mcd`, `r2\|ksg`, `dcorr\|ksg` | MinCovDet sibling; MI siblings (the same axis as the primary) | long | as above |

- **Multiplicity.** The primary is tested alone at α .05. The secondaries are tested with Holm over those testable in the dataset. Reported rows are not tested.
- **`mcd2|ksg`** is read only through its links (`r2|mcd2` and `r2|ksg`), with the partial z reported, never as a character of its own.
- **T/M rule (inferred from two datasets):** MinCovDet passed with T/M 7.9 (macaque 10 s) and was degenerate or unreliable at T/M 1.6–5.9 (macaque 2 s, PRIOS). The 7.5 cut-off is set from those points, not derived.
- **KSG channels.** If M ≤ 32, use all channels. Otherwise use 32 channels drawn with a seed fixed in the freeze record, from channels passing quality control, before any state label is read. MinCovDet and `l1|linf` use all channels. IAAFT is joint over every channel a readout uses.
- **Envelope null slot.** `findings.md` calls a local-envelope-preserving null the decisive test of the MI pair, and a sibling session is building it. Until its calibration is committed, it is a reported control and not a gate. Making it a gate is a dated change to this file.

## Windows: chosen from baseline data and metadata only

The window rule is fixed here so that "per dataset" is not a free choice. Its inputs are metadata and the baseline state's data only. Its outputs are logged in the freeze record before any target-state window is processed.

1. **Sampling.** Resample, with anti-aliasing, to the lowest rate that keeps the band the dataset's field treats as its signal (declared with a citation in the freeze record), then record fs. Record the median integrated autocorrelation time τ (in samples) over channels in baseline windows.
2. **Stationarity limit.** L_max is the longest window that fits within a state epoch as the dataset defines it (e.g. a sleep stage, a quiet interval, a drug plateau). It is taken from metadata, not data.
3. **Long window.** Start at T = 1000 samples, the anchor's strongest evidence (macaque 10 s at 100 Hz). Assess reliability. If either statistic of the primary is below the floor, double T and try again, up to L_max. If L_max < 1000, start at L_max.
4. **Short window.** T = 200 samples, for `l1|linf` only.
5. **Assumption, unverified:** macaque and PRIOS share T = 200 at 2 s. The protocol carries the window length across fields in samples, scaled by nothing else. Whether `l1|linf` instead scales with physical time or with τ is unknown, and the freeze record states τ so this can be checked later.
6. **Effective sample size.** Report T / (2τ) for each window length. The macaque reference value is to be measured on the NeuroTycho 10-s arm and entered here before the first dataset is frozen (open).

## Reliability floor

- Reliability of a statistic = the mean pairwise edge-profile correlation between windows of the baseline state within one unit-session (as in the prereg), averaged over unit-sessions.
- **Floor: .3 for both statistics of a readout, at the chosen window length, in the baseline state, before any contrast is computed.**
- A readout below the floor is reported as **untestable**: its Δ is printed descriptively and enters no test or Holm set. No exceptions. The PRIOS `l1|linf` row, tested at reliability .25, is the precedent this rules out.
- The other usability rules are as in the prereg: finite in ≥ 95% of windows of every state used, and baseline real |z| < .98. For mapped near-ceiling pairs, judge by the null SD instead (guide, ceiling rule).

## Contrasts and units

- **Unit.** The independent unit is declared in the freeze record: animal, patient, site or station. Recordings and sessions nest within units.
- **Target contrast.** Session Δ = mean window net in the target state − mean window net in the baseline state, within a recording. Examples: anaesthesia − wake, storm − quiet. Unit Δ = mean of its session Δ.
- **No-change baseline (the SD unit).** Split each baseline-state recording by start time into two halves of equal window count: second half − first half. Null SD = SD over units of that split-half Δ, scaled to the window counts of the target contrast by √((1/n_b + 1/n_t) / (2/n_half)). Here n_half is the window count of each half, and n_b and n_t are the baseline and target counts. The scaling assumes independent windows. For the prereg's 5-against-5 split and 10-against-10 contrast it gives the prereg's 1/√2. Every effect is reported as **raw Δz** and as **Δz / null SD**.
- **Nuisance contrast.** A real but irrelevant state change within the same data (the eyes-open analogue), declared in the freeze record. If the dataset has none, say so. Gate 4 is then not applied, and the result is labelled as lacking a specificity check.

## Gates (as in the axis screen, generalised to n units)

A testable readout is a result only if every gate holds:

1. **Unit-level test.**
   - n ≥ 6: exact sign-flip test over unit Δ, p < .05. It is two-sided unless the freeze record predicts a sign, which it may do only for anaesthesia datasets.
   - n = 3–5: t over unit means, p < .05, with every unit the same sign.
   - n < 3: descriptive only, with no test.
   - The primary tests at α .05; secondaries use Holm.
2. **Size.** |Δ| ≥ 3 null SD.
3. **Split-half.** |mean split-half Δ| ≤ .25 |Δ|.
4. **Nuisance.** |nuisance Δ| ≤ .33 |Δ| (where declared).
5. **Covariates.** Within-recording OLS of window net on state plus strength (mean |r|, SD |r|, covariance participation ratio) and burst index (median over channels of 1.4826·MAD/SD on the pre-normal-scoring window). The adjusted Δ keeps the sign and is ≥ .5 |Δ|. Its unit Δ must also hold up: every unit the same sign when n ≤ 5; a one-sided sign-flip test in Δ's direction at p < .05 when n ≥ 6. Spectral covariates (spectral entropy, peak frequency, low-band fraction) are reported for F1 and not gated, since IAAFT matches each window's spectrum.
6. **Reliability.** The disattenuated Δ (z divided by √(rel_A · rel_B) per unit × state and version) keeps the sign, with the same unit rule as gate 5.

Also reported, not gated: raw real Δ and surrogate Δ separately (the PRIOS lesson: a net can move through the surrogate); double-centred Δ; mean-profile Δ; `env_E` net Δ and its window-level r with the anchor net (from [`gaussref.py`](../atlas/axis_screen/gaussref.py)); and per-subgroup Δ where the data have subgroups, such as agents.

## Edge-level report (optional)

- Only when a calibrated drop test says the change is localised. Use [`mi_pair/droptest.py`](../atlas/mi_pair/droptest.py): select on even windows, measure on odd and the reverse, compare with leverage-matched random drops.
- First repeat the synthetic calibration (concentrated, broad and skewed-broad plants) at the dataset's M and T.
- Decision rule, as frozen for the MI pair:
  - localised if f < .8 and below every random draw;
  - broad if f ≥ .9 or inside the random range;
  - otherwise unresolved.
- Without "localised", no edge or node is named. The macaque MI fall was broad (f 1.27).

## Wording

- Per the guide.
- A primary pass reads: the across-pair organisation of dependence departs further from (or nearer to) the Gaussian prediction than a linear process with the same spectra and covariance gives. It is not "more nonlinear coupling" and not a level change.
- MinCovDet reads as time-heterogeneous dependence at the copula level, on normal-scored windows only.
- `l1|linf` has only the literal reading: the tail of x − y relative to its typical size.
- A result that passes is confirmatory in that dataset. One that fails is reported with its numbers.

## Per-dataset freeze record

Commit `research/representation/transfer/<dataset>/freeze.md` before computing any target-state output. It states:

- source, version and licence;
- units and their count;
- the baseline, target and nuisance states, and how their epochs are defined;
- channels and quality control;
- fs, τ, L_max, the long and short T, and T/(2τ);
- the KSG subset seed;
- reliability per statistic in the baseline state (computed before the freeze) and the testable readouts that follow;
- predicted signs, if anaesthesia;
- the adapter's commit;
- pyspi pin and computation version, if any pyspi output is used;
- expected and worst-case compute.

## Tier 2 (exploratory): deferred

**What it will need:**
- a dataset-agnostic screen driver over all usable minimal pairs;
- selection on one half of units or windows (even/odd) and confirmation on the other;
- the identical pipeline run on a true no-change control (within-baseline split-half) to estimate the false-discovery proportion (FDP = control discoveries ÷ discoveries), with no claims if it exceeds a threshold frozen with Tier 2;
- a real nuisance contrast reported separately.

**Why it is deferred:**
- The macaque full-pyspi screen ([`movers/matched-result.md`](../atlas/movers/matched-result.md)) gave an FDP estimate of .47 against its frozen .2 and made no claims.
  - That control was eyes-open, a real change: 19 of its 26 hits were a disjoint lag change.
  - A within-awake split-half control found 0.
  - So .47 overstates the false discoveries, but which control defines FDP must be fixed before Tier 2 runs.
- The datasets are not chosen yet.
- The generalised gate and null code is a refactor (below), not yet proposed in detail.

**Refactor to propose with Tier 2 (not done):** move the gate logic of `axis_screen/score.py`, `mechanism.score` and `movers/matched.py` into one module parameterised by unit, state names and n, with the macaque runs as regression tests. Today those scripts hard-code the macaque manifest, the state names and t(3).
