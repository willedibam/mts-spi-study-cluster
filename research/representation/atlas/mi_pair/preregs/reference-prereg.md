# Reference axis: does the macaque drug fall in z(G, K) hold under nearest-neighbour bipolar derivations?

Frozen 25 November 2026 at the commit that adds it, before any bipolar value from the frozen montage below was computed. Code: [`../reference/pairs.py`](../reference/pairs.py) (pairs, geometry only), [`../reference/bank.py`](../reference/bank.py) (gates, bipolar banks), [`../robust.py`](../robust.py) (unchanged; `run --shift`), [`../reference/score.py`](../reference/score.py), [`../jobs/reference.pbs`](../jobs/reference.pbs). Pairs: [`../reference/pairs.json`](../reference/pairs.json). The result goes to [`../robustness-result.md`](../robustness-result.md) as a "reference" axis.

## Why

- Every 10-s, 100 Hz configuration with a Theiler window SURVIVES on the atlas reference (common average over good channels, last good channel dropped): anchor AUTO, k = 4, ψ(T) pooled drug Δz −.0133, t(3) −6.11 ([`../robustness-result.md`](../robustness-result.md)). The reference was held fixed; its Limits line names this.
- The joint IAAFT surrogate keeps every pair's cross-spectrum, so zero-lag linear mixing (which a reference is) enters real and surrogate alike. A reference change can still alter the non-Gaussian part of the dependence and the edge profile z is taken over. Drugs change global power, which the common average spreads into every channel. So it is a live question whether the common average contributes to the fall.
- This is a sensitivity row for one axis, not a re-selection. The direction of change is not predicted.

## Real-data looks already taken (disclosed)

- **The unfrozen montage check** ([`../montage/zbip.py`](../montage/zbip.py), [`../montage/zbip_result.txt`](../montage/zbip_result.txt), 17–19 November). It used index-neighbour differences (ch s+1 − ch s at the 32 `sub32` sites of the CAR windows, ignoring the electrode map and the column shift after bad channels) with zgk ≈ AUTO k4 ψ(T) and one IAAFT draw. Results:
  - eyes-open SD: CAR .0020, bipolar .0082;
  - drug Δz, bipolar (CAR): KT −.0316 (−.0159) 4/4 sessions; KTMD −.0325 (−.0155) 11/11; MD −.0123 (−.0062) 3/4; PF −.0345 (−.0117) 4/4; SLP −.0040 (−.0010) 5/7;
  - per-session drug Δ, CAR against bipolar: r = +.80.
- So R ≥ .5 against −.0133 is expected a priori. The criteria that carry this verdict are 4/4 animals, the t(3) and every agent < 0, and the effect in own eyes-open SD. MD is the likely failure.
- No value of the frozen nearest-neighbour montage has been computed. The electrode maps were read for geometry only.
- **The map gate was run before the freeze.** It used awake eyes-closed CAR windows only; no bipolar value and no drug or sleep window was involved.
  - Spearman ρ(map distance, r): Chibi −.48, George −.59, Kin2 −.62, Su −.63.
  - Selected pairs' mean r: +.62, +.83, +.63, +.57, at the 98th, 97th, 95th and 96th percentile of all electrode pairs. PASS.
  - So the index mapping holds.
  - Two Chibi pairs are not correlated: 39–51 (r −.05) and 37–50 (+.04). They are probably across an array edge or sulcus that the 2-D image does not show. They stay: the rule is fixed and is not changed by a look at the data.

## Montage (frozen in `pairs.json`)

- **Source.** Provider maps `results/representation/neurotycho/neurotycho_audit_260910/montage/{Chibi,George,Kin2,Su}.zip` (`*Map.mat`, X and Y of 128 electrodes in image pixels; SHA-256 in `pairs.json`). The 2011 maps are used for the 2012 recordings of the same animal (assumed: the arrays are chronic).
- **Mapping.** Map electrode n (1-based) is `ECoG_ch{n}.mat`. This is assumed: `SampleCode.m` only plots X and Y in order. It is checked by the map gate below.
- **Rule.** The rule of `stage_neurotycho_source.montage()` and [`neurotycho-source-pilot.md`](../../../neurotycho/notes/neurotycho-source-pilot.md), one set per animal:
  - threshold 1.5 × the median nearest-neighbour distance over all 128 map electrodes;
  - candidates are pairs of available electrodes within the threshold, taken in ascending distance;
  - greedy non-overlapping pairs;
  - 32 pairs by farthest-point coverage of the midpoints, starting from the leftmost;
  - signal = lower-numbered minus higher-numbered electrode (immaterial here: normal-scoring and both MIs are sign-invariant).
- **Available electrodes.** Excluded: 128 (dropped after the common average in every unit) and the union of the animal's bad channels (George 34 and 53, Kin2 10, Su 38, Chibi none). One set per animal keeps the montage fixed across that animal's sessions.
- **Counts.** Candidate pairs and threshold (pixels): Chibi 54 (61.9), George 57 (58.6), Kin2 47 (50.4), Su 57 (61.5). M = 32 (496 edges, as the anchor) uses 64 of the 128 electrodes.

## Windows

- **Derivation.** The bipolar windows are the differences of the CAR bank columns of each pair's electrodes, at the same rows, ids, `start_s`, order and stage A/B split (1,260 windows), with no re-screening.
- **Identity.** The common average is subtracted from every channel, and `prepare.continuous` (notch, 0.5 Hz high-pass, 40 Hz low-pass, 1000 → 100 Hz) is linear and per-channel. So this equals the bipolar derivation re-cut from raw, for any two kept electrodes, to float rounding.
- **Deviation from the brief.** The brief asked for a re-cut from raw. The derivation gives identical values and removes a separate pass over 30 archives. The re-cut is run as a gate instead.

## Gates (drug-blind; nothing is scored unless both pass)

- **Map gate** (`bank.py mapcheck`):
  - on awake eyes-closed CAR windows only, per animal, the window-mean correlation of every pair of electrodes kept in all of that animal's units;
  - PASS if, in every animal, the selected pairs' mean r exceeds the 90th percentile of r over all electrode pairs;
  - Spearman ρ(map distance, r) is reported but not gated, because the common average flattens distance decay.
  - A wrong index mapping would put the selected pairs near the all-pair mean.
- **Identity gate** (`bank.py gate`):
  - every window of George 20110112 KTMD (bad channel 53, column shift) and Chibi 20120730 PF (a 2012 recording) is re-cut from the raw archive with `prepare.continuous` on each pair electrode, at k = round(start_s · 100), 1000 samples;
  - PASS if max |derived − re-cut| / SD < 1e-6.

## Procedure

`robust.py run <bipolar bank> <out> <agent> first 48 --shift`, with the calls and order of [`../jobs/robust10.pbs`](../jobs/robust10.pbs) (`row` for stage-B SLP). The rest is unchanged:
- the 32 bipolar channels are all used (`sub32(32)` is the identity);
- normal-scoring;
- one spectrum-final joint IAAFT draw per window, seeded by the row index within the call, so the seeds and rows are those of the CAR `rb10` run;
- AUTO Theiler window min(int(2⟨a_i, a_j⟩), T//10);
- KSG count terms at k = 4 and 10 under w0, AUTO, 2τ₁/ₑ and 4×AUTO, and kshift.
There is no reproduction gate against stored z: no bipolar value is stored.

## Statistic

As `robust.py score` and [`../montage/animal_level.py`](../montage/animal_level.py):
- per window, residual = z real − z surrogate;
- per session, Δ = drug (or sleep) − awake eyes-closed;
- four animal means over drug sessions;
- one-sided one-sample t(3).
Reported per configuration:
- pooled drug Δz, raw and in the bipolar eyes-open session SD;
- per-agent Δz with sessions down;
- sleep Δz;
- R.
**R** = bipolar pooled ÷ CAR pooled. For the verdict this is against the anchor −.0133. Per configuration it is against the matching CAR value: AUTO k10 −.0126 (from `rb10`), kshift −.0148 (ksgbias).
Descriptive: the Pearson r of per-session drug Δ, bipolar against CAR, on the same rows.

## Reading (stated before output)

**Verdict row: AUTO, k = 4, ψ(T).** The rule is that of the ksgbias and robustness preregs:
- **SURVIVES** if all hold:
  - 4/4 animal means < 0;
  - one-sided p < .05;
  - every drug agent's Δz (KT, KTMD, MD, PF) < 0;
  - R ≥ .5.
- **LOST** if R < .25 or ≤ 2/4 animal means < 0.
- **PARTIAL** otherwise.
- AUTO k10 and kshift are scored by the same rule as secondaries. w0, 2τ₁/ₑ and 4×AUTO are descriptive. SLP is descriptive.

**What each outcome means.**
- **SURVIVES.** The fall does not need the common average: it is present in local differences of adjacent electrodes, with the same sign in every animal and agent.
  - It does not show that the fall is local. A field gradient from a distributed source survives differencing.
  - It does not settle volume conduction in general.
- **PARTIAL.** The sign holds overall, but one animal, one agent or the t(3) fails. The CAR result stands for the CAR signal, with the reference named as a condition. The failing criterion is reported. The likely case is MD (from the prior check).
- **LOST.** The fall is not present in nearest-neighbour bipolar derivations. Two readings are not separated:
  - (a) the common average creates or carries it, for example a drug-changed global component spread into every channel;
  - (b) the dependence is real but spatially broad, and a bipolar derivation (a spatial high-pass) removes it.
  - Either way, the paper states the fall as a property of the common-average signal.
- **Comparability.** Values from a different reference are different quantities ([`docs/interpreting-spi-pairs.md`](../../../../../docs/interpreting-spi-pairs.md)). z is taken over a different set of 496 edges, with different strengths and spectra. So R is a ratio of different quantities: its size is not an attenuation, and only its sign and the consistency criteria carry the verdict.

**Prediction (mine, before output).** SURVIVES:
- R 1.5–3 raw;
- own eyes-open SD about −3 to −5, smaller than the anchor's −6.7, because the bipolar null spread is larger;
- t(3) weaker than −6.1;
- MD the weakest agent.

## Limits

- One montage (32 of 47–57 candidate pairs per animal), one surrogate draw per window, four animals.
- Distances are image pixels of a 2-D map, not cortical distances. A pair close in the image could cross a sulcus or array edge.
- Mapping and map-date assumptions as above; the map gate checks the first.
- The montage and the reference change together with the edge set. This is not a matched-edge comparison.

## Compute

Gadi, pushed snapshot `operations/sources/mi-ref-<commit>`, one job of 48 cores.
- Gates: minutes.
- Banks: minutes.
- Sweep with `--shift`: about 1.75 × the 1.6-h `rb10` job.
- Expected 0.27 kSU (about 2.8 h); worst 0.58 kSU (6 h walltime).
