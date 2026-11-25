# Window duration at 100 Hz: the macaque MI-pair drug fall on 50-s windows and their own 10-s cuts

Frozen 25 November 2026 at the commit that adds it, before any 50-s 100 Hz window is built or scored. Exploratory addendum to [`lowrate-prereg.md`](lowrate-prereg.md), requested by the robustness session (`robustness-result.md`: "100 Hz windows longer than 10 s are untested"). It cannot change the 10-s anchor verdict. Code: [`../lowrate/long100.py`](../lowrate/long100.py) (build, calibration and frozen pair readings), scoring by the unchanged `lowrate/score.py`, tables by `lowrate/summarise.py`.

## Why

- The low-rate long arms (D20L, S20L: 50 s at 20 Hz, T = 1000) change rate and duration together. At fixed 20 Hz, 10 → 50 s keeps .63–.68 of the fall, but that leg also changes N (T = 200 → 1000), where the estimator's stationary drug − awake offset swings sign (robustness calibration: −.011 at T = 200, +.006 at T = 1000).
- At 100 Hz only 2-s and 10-s windows exist, and 2-s own bank and 2-s cuts differed about 1.5×, so window bank is itself a factor.
- Here the 50-s windows are the long arms' own (same rows, sessions and starts), at 100 Hz, and their 10-s cuts are the same samples. Duration is the only change in the primary pair, and both sides have T ≥ 1000.

## Arms (100 Hz, the anchor's estimator and null)

| Arm | Band (Hz) | Window | T | Windows |
|---|---|---|---|---|
| W50 | 0.5–40 | 50 s | 5000 | the 1,221 D20L/S20L windows (stage A 489, B 732) |
| W50L8 | 0.5–8 (low-pass as L20, D20L) | 50 s | 5000 | same |
| C10 | 0.5–40 | five contiguous 10-s cuts of each W50 window | 1000 | 6,105 |
| C10L8 | 0.5–8 | cuts of W50L8 | 1000 | 6,105 |

Selection is the long arms' (as `prepare` selects 10-s windows, at 50-s length; `NT_CAPS` 10 per slot, 30 for sleep; 2–30 per unit × state, median 10; counts per slot reported). Gate A: every W50 window subsampled by 5 equals its S20L window, and every W50L8 window its D20L window, or nothing is written. Normal-scoring, one spectrum-final joint IAAFT draw per window (seeded by its row index within the per-agent call, so each cut has its own draw), 32-channel subsets, AUTO Theiler window with cap T // 10 (500 at T = 5000), k = 4, ψ(T): `edges.py` recompute, unchanged.

## Statistic and readings (stated before output)

Per arm, as `lowrate-prereg.md`: normal-scored residual (primary), raw reported; per-session Δ = state − awake eyes-closed; own eyes-open SD; pooled drug Δ, four animal means, one-sided t(3); per-agent Δ; eyes-open and sleep controls (flag at 2 own SD); K < 0 fractions; Theiler cap hits.

Pair readings (`long100.py read`), R = pooled drug Δ(first) / Δ(second) over animal means, paired animal-level first − second t(3): **neutral** if .75 ≤ R ≤ 1.33; **smaller in the first** if R < .75 and t(3) ≥ +2.35; **larger in the first** if R > 1.33 and t(3) ≤ −2.35; otherwise **inconclusive**.

1. **W50 / C10 (primary): duration at 100 Hz, 0.5–40 Hz, same samples.** Because pair 1 also changes N (T 5000 against 1000), and the estimator's stationary drug − awake offset moves with T, the call is made on the calibration as well (below): a "smaller" or "larger" call is attributed to duration only if (Δ(W50) − Δ(C10)) − (off₅₀₀₀ − off₁₀₀₀) has the same sign and exceeds 2 SE of the offset difference; otherwise it reads "not distinguished from the estimator's N response". Offset-adjusted R = (Δ(W50) − off₅₀₀₀) / (Δ(C10) − off₁₀₀₀) is reported beside R.
2. W50L8 / C10L8: duration at 100 Hz, 0.5–8 Hz.
3. D20L / W50L8: rate and N at fixed band (0.5–8 Hz) and identical 50-s windows (20 Hz, T = 1000 against 100 Hz, T = 5000).
4. S20L / W50: plain subsampling against 100 Hz on identical 50-s windows (aliasing included).

**Calibration (drug-blind, run here, `long100.py calib`).** The anchor estimator only (AUTO k4 ψ(T)). The robustness session's `block` generator (stationary linear coupling; MI 0 between blocks) for the offset, n = 200 per spectrum per T, and `gain_shared` (one shared gain switch) for detectability, n = 40; 16 channels phase-randomised from the periodograms of normal-scored W50 stage-A windows (awake eyes-closed of KTMD units, KTMD drug, PF drug); T = 1000 is the first 1000 samples of the same 50-s source and seed. Offset = mean(KTMD, PF) − awake of z real − surrogate on `block`, ± SE. W50L8 is not calibrated (0.5–8 Hz spectra), so pairs 2 and 3 are read without an offset adjustment and say so.

Secondary:
- **C10 as a 10-s, 100 Hz replication** on a mostly new set of windows: retains (t(3) ≤ −2.35, pooled ≤ −2 own SD, controls not flagged), lost (within ±1 own SD or t(3) > −1), otherwise partial. C10 against the anchor and C10L8 against L20 are descriptive (different window banks).
- **Within-session fade.** The robustness session found drug Δ fading about +.0012 per 10 min (t(3) 2.49, p .09). A 50-s window averages over 5× the time of a 10-s one but over the same session span as its cuts, so it does not bias pair 1.

**Interpretation, fixed in advance.** The neutral band starts at .75 and the 20 Hz duration leg kept .63–.68, so an R of about .75–.9 does not exclude an effect of that size; it is reported as "neutral by the rule, compatible with a 20 Hz-sized duration effect". If pair 1 is neutral with R ≥ .9, duration at 100 Hz does not shrink the fall, and the 20 Hz long-arm loss is rate (or N at low rate), not duration. If pair 1 is smaller in W50, duration contributes at the analysis rate, and 10 s is not a neutral choice: the fall is a property of 10-s windows and below. Pair 3 then splits the remaining 20 Hz loss into rate/N at fixed duration.

**My prediction.** Pair 1: R .7–1.0 (between neutral and the 20 Hz .63–.68), most likely neutral by the rule. Pair 3: R .3–.5, smaller in D20L. C10: retains.

## Pre-stated limits

- KSG at T = 5000 sees five times as many neighbours as at T = 1000; z is not T-invariant (the 2-s arms), so pair 1 measures window duration and N together at fixed rate, with the T ≥ 1000 regime as the only protection.
- IAAFT is a stationary null; a 50-s window is less stationary than a 10-s one (the long arms' K < 0 misfit), which can enter W50 through the null, not the signal.
- C10's 10-s cuts were screened by the artefact rule at 50 s, not at 10 s, so a cut can hold a short artefact that the 10-s rule would have rejected (descriptive comparisons with the anchor carry this).
- One surrogate draw per window; 32-channel subsets; four animals (t(3)). The 50-s windows are the slots' non-overlapping 50-s spans, so PF and short awake slots contribute fewer windows (counts reported).

## Compute

Gadi, pushed snapshot `operations/sources/mi-lowrate-<commit>`. Build (30 archives, 30 cpus, 190 GB: twice the band build's sample volume): expected 0.02 kSU, worst 0.1. Calibration (720 synthetic windows at T = 5000, about 300 cpu-s each, plus T = 1000): expected 0.13 kSU, worst 0.3. The estimator is brute-force O(T²): one KSG pair costs 1.27 s at T = 5000 against 0.036 s at T = 1000 (35×, measured locally), about 2,500 cpu-s per 50-s window. Scoring: W50 and W50L8 about 1.7 kSU each, C10 and C10L8 about 0.25 kSU each (one arm per job); one timing call first (stage-A PF, all 48 workers), after which the worst case is restated. Expected 4.1 kSU, worst 6.8 kSU. `score.py` writes only at the end of a call, so walltimes are set at three times the timed rate.

## Addenda A and B (frozen 26 November 2026, before any further output)

Requested after the supervisor's question whether the per-pair departure is persistent or a one-window transient, and whether it survives aggregation over longer stretches. They cannot change the readings above.

**What existed when they were frozen.** The four banks (build job 185197383, gate A 0.0); the drug-blind calibration (job 185200307; read: offset(5000) − offset(1000) +.0008 ± .0015); W50 stage-A PF from the timing call (job 185200308, 80 windows in 3,808 s on 48 cores; `scores/W50/zns_A_PF.npz`, sha256 `9c91abd0…e210`). The session that froze these addenda printed only that file's array shapes, not its values; whether another session read its values was not verified.

**Restated cost (from the timing call).** One 48-window round of W50 takes about 32 min, so a W50 window costs about 1,900 core-s. Remaining W50 calls 27 rounds, W50L8 29, C10 and C10L8 about 2 h each, Addendum A about 6 h: expected 3.8 kSU, worst 5.7 kSU. Walltimes are three times the timed rate.

**Addendum A: the 50-s windows against a 10-s segment null** ([`../lowrate/long100_seg.py`](../lowrate/long100_seg.py), [`../lowrate/jobs/long100_seg.pbs`](../lowrate/jobs/long100_seg.pbs)).
- Why: a 50-s window can owe excess over a stationary IAAFT to slow change between its 10-s pieces (gain or linear-coupling drift), which is non-stationarity, not coupling form. Over a whole drug period this would grow.
- Null: nonstat's N2 at L = 1000 (joint IAAFT, spectrum-final, per 10-s piece, concatenated) on each normal-scored W50 window; seed `default_rng([row, 1000])`; rows and subsets as `score.py`. The real KSG is not recomputed; z_real comes from the W50 score, matched by id, subset labels and the Gaussian-MI vector (gate: max difference < 1e-5).
- Size check (run first, drug-blind): `long100.py`'s generators, sources and seeds at T = 5000 (100 `block` and 40 `gain_shared` per spectrum), scored against N0 (must reproduce `calib_long100.npz`'s dz to 1e-8) and N2. N2 passes if its drug-blind offset mean(KTMD, PF) − AW of z_real − z_N2 on `block` is within 2 SE of 0 or within ±.001. If it fails, Addendum A is reported as descriptive only.
- Reading: R_seg = pooled drug Δ(real − N2) / pooled drug Δ(real − N0) over animal means, W50, normal-scored. R_seg is read only if W50's own N0 drug Δ is ≤ −2 own eyes-open SD with 4/4 animals negative; otherwise only the per-animal differences are reported. **Not drift**: R_seg ≥ .75. **Mostly drift**: R_seg ≤ .5. Otherwise **partial**.
- Prediction: not drift (R_seg about .8–1.0), since 5-s and 2-s segments kept all of the 10-s change.

**Addendum B: is a pair's departure persistent or one-off?** ([`../lowrate/long100_edges.py`](../lowrate/long100_edges.py); no new compute: the W50 and C10 score files' per-edge G and K).
- P1, the test: `reorder.py` v2's cross-window repeat of the pair-level residual (isotonic fit of K on G, then an additive node fit), real minus surrogate, session Δ = drug − awake eyes-closed. At 50 s over the W50 windows of a session; at 10 s over C10 cuts from different 50-s windows only. A persistent pair property repeats more as estimator noise falls; one-off episodes do not.
- **Persistent** at the pair level: r50 − r10 drug Δ > 0 in ≥ 3/4 animals, and pooled r50 drug Δ > 0 with one-sided p < .05. **One-off**: pooled r50 − r10 ≤ 0 and r50 drug Δ not significantly > 0. Otherwise **mixed**. Per agent reported.
- Descriptive: the repeat across the five contiguous cuts of one 50-s window (inflated by shared slow state); P2, per edge the number of cuts (of 5) above the cut's own surrogate band, against each cut's own surrogate, as the ratio to independence; P(above the band at 50 s | that number).
- Scope: persistence only. Like v2, it cannot separate gain from coupling (v2's gate failed on a heterogeneous fixed channel floor, +.037), and the node fit removes additive node effects only.
- Synthetic check of the reader (local, planted): a persistent pair pattern gives r50 − r10 +.30 and PERSISTENT; episodic, pair-random departures in 20% of pairs per cut give r50 −.02 and ONE-OFF.
- Prediction: persistent under ketamine (v2: +.149), one-off or mixed under propofol (v2: +.008).

**Jobs.** Frozen arms from the existing snapshot `mi-lowrate-ec3d7c29` with `score_calls.pbs` (W50 stage-A PF is not rescored); Addenda A and B from a snapshot of the commit that adds this section. Job IDs and the read commands are added below when submitted.

**Submitted 26 November 2026** (queued; logs in `/scratch/ql44/we2614/mts-spi-study/logs/mi-long100/`).
- Frozen arms, snapshot `mi-lowrate-ec3d7c29`, `score_calls.pbs`: W50 185222173–75, 185222177; W50L8 185222178, 185222180, 185222182–83; C10 185222184; C10L8 185222185.
- Addendum A, snapshot `mi-long100-4535cf13`, `long100_seg.pbs`: 185222186 (size check first, written to `calib/calib_long100_seg.txt`).
- Read, with `G=/g/data/ql44/we2614/mts-spi-study/representation/atlas-mi-pair-lowrate-261125/scores` (checksums: `SHA256SUMS.<job>` per arm):
  - tables: `lowrate/summarise.py $G /g/data/ql44/we2614/mts-spi-study/representation/atlas-mi-pair-261117/zns W50 W50L8 C10 C10L8`;
  - frozen pairs: `lowrate/long100.py read $G`;
  - Addendum A: `lowrate/long100_seg.py read $G` (after checking the size line in `calib_long100_seg.txt`);
  - Addendum B: `lowrate/long100_edges.py $G`.
  - `long100.py read` prints R, t and the rule's call without the calibration offset: the primary call is attributed to duration only after applying off₅₀₀₀ − off₁₀₀₀ (+.0008 ± .0015, `calib/calib_long100.txt`) by hand as the rule above states, with the offset-adjusted R reported beside R.
