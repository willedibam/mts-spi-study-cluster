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
