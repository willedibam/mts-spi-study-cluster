# Does the MI-pair drug change depend on the sampling rate? The macaque z(G, K) at 20 and 10 Hz

Frozen 25 November 2026 at the commit that adds it, before any low-rate score exists. The banks were built first (yields below are counts of windows, not outcomes). Exploratory robustness check: it does not replace the 100 Hz anchor. Code: [`../lowrate/`](../lowrate/) (`build.py`, `score.py`, `summarise.py`, `jobs/`).

## Why

- The anchor is at 100 Hz on the 0.5–40 Hz signal. The question was whether a non-Gaussian excursion sampled 5–10 times at 100 Hz registers differently at 10–20 Hz.
- G and K are zero-lag statistics of the cloud (x_i(t), x_j(t)); sample order enters only through the Theiler window. Plain subsampling keeps the instantaneous joint distribution and cuts N. Anti-aliased decimation is a band restriction: 20 Hz keeps 0.5–8 Hz, 10 Hz keeps 0.5–4 Hz. So the real question is whether the fall lives in the slow band.
- Direction is open. Linear filtering pushes a linear process with non-Gaussian innovations towards Gaussian when the filter's largest coefficient is small (Mallows 1967, J. Appl. Prob. 4:313; result as summarised in the abstract, full text not read). Anaesthetic slow waves are themselves slow, synchronous and long-silent (Chauvette et al. 2011, J. Neurosci. 31:14998, ketamine–xylazine against sleep in cat), and could carry non-Gaussian dependence into the slow band. The earlier tests point to a ~1-s mixture of magnitudes (segment and cumulant tests); if that is gain on a faster carrier, a low-pass removes the carrier and the fall should weaken. No prior is adopted.

## Arms

r = 20 or 10 Hz; q = 100 / r. LP_r is `prepare.lowpass` (order-8 zero-phase Butterworth) at 0.8 × r/2 = 8 or 4 Hz, on the continuous 100 Hz signal rebuilt from the 1 kHz archives with `prepare.continuous` (no window edge effects); CAR per window after it, as `prepare`. Subsampling offset 0.

| Arm | Signal | Window | T | Theiler cap |
|---|---|---|---|---|
| D20, D10 | anti-aliased: LP_r, every q-th sample | the bank's 10-s windows | 200, 100 | 20, 10 |
| S20, S10 | plain: every q-th sample, no filter | the same windows | 200, 100 | 20, 10 |
| L20, L10 | LP_r kept at 100 Hz (consistency) | the same windows | 1000 | 100 |
| D20L, D10L | anti-aliased, sample-matched | 50 s, 100 s | 1000 | 100 |
| S20L, S10L | plain, sample-matched | the same long windows | 1000 | 100 |

- Long windows are selected as `prepare` selects 10-s ones (same labelled spans and 5-s edge trim, state-own artefact scale at 100 Hz, flat-run mask, non-overlapping, spread evenly, caps 10 per state and 30 for sleep), at q × 10 s. Unit and stage assignment are the bank's.
- The 2-s arm is dropped: 40 or 20 samples per window.
- **Estimator and null unchanged:** `edges.py` recompute's calls (32-channel subsets by the same `first`/`row` rule, normal-scoring, one spectrum-final joint IAAFT draw per window seeded by the row index within each per-agent call), made through `lowrate/score.py`, which adds Theiler diagnostics only. The surrogate is drawn on each low-rate window; no 100 Hz surrogate is decimated. A sibling session is studying KSG negative bias and the Theiler window; nothing here changes the estimator.
- **Gates (both before any arm is scored):** A, the rebuilt 100 Hz windows equal the stage banks' windows (max |Δ| / SD ≤ 1e-9); B, `score.py` on the original stage-A PF bank reproduces the stored `zns_A_PF` V within 1e-6. Either failing stops the run.

## Statistic and reading (stated before output)

Per arm, as `review/sumall.py` and `montage/animal_level.py`: residual = z_real − z_surr per window, normal-scored primary (raw reported); per session Δ = state − awake eyes-closed; the arm's own eyes-open session SD; per agent mean Δ, in own SD, sessions down; four animal means over drug sessions, one-sided t(3); same animals' eyes-open.

- **Plain subsampling is expected to attenuate.** Same estimand, but at N = 100–200 KSG noise attenuates real and surrogate z alike and the Theiler window and its cap change. So no low-rate arm is read against 100 Hz as a band effect.
- **Band is read only as D against S at the same N and duration** (D20 vs S20, D10 vs S10, D20L vs S20L, D10L vs S10L). Retention B = pooled drug Δ(D) ÷ pooled drug Δ(S), with the paired animal-level difference D − S (t(3)):
  - band-neutral: .75 ≤ B ≤ 1.33;
  - stronger in the slow band: B > 1.33 and paired t(3) ≤ −2.35;
  - weaker in the slow band (the fall needs frequencies above the cutoff): B < .75 and paired t(3) ≥ +2.35;
  - otherwise inconclusive; report B and t.
- **Consistency:** L_r against D_r is the same signal at T = 1000 against T = 200/100 with a larger Theiler cap; a difference there is N and estimator, not band.
- **Controls:** an arm whose eyes-open per-animal mean or pooled sleep Δ exceeds 2 of its own eyes-open SD in magnitude is flagged, and its drug Δ is not read as drug-specific. A larger drug Δ with a wider control spread is not a stronger result: the own-SD column is reported beside raw Δ.
- **Diagnostics per arm:** fraction of K < 0 (real and surrogate, by state) and fraction of edges whose AUTO Theiler window reaches the cap T // 10, with the median window.
- **Figure (d) extension** to mouse and human is triggered only by a "stronger" or "weaker" band reading at either rate.
- All arms are reported, not the best one.

## Pre-stated limits

- One surrogate draw per window; 32-channel subsets; four animals (t(3)).
- The long arms trade N for nonstationarity over 50–100 s, which IAAFT does not model; both lengths are reported.
- LP_r has finite roll-off: at 1.25 × cutoff (10 and 5 Hz) the zero-phase gain is −31 dB, so a little power just above the new Nyquist aliases.
- Delta fraction is a 0.5–4 Hz share; D10 keeps essentially only that band, so states differing in delta power differ in spectrum more at 10 Hz. IAAFT matches each window's spectrum, so this enters the residual only through the estimator's spectral sensitivity (the 2-s stationary-null finding: spectrum alone gives 60–65% of the raw fall).

## Yields

Build job 185156131 (snapshot `mi-lowrate-18d7c108`): gate A worst max |rebuilt − bank| / SD = 0.0. D, S and L at 10 s reuse the bank's windows (stage A 500, stage B 760). The long arms (D = S by construction) yield, windows per state (awake_ec / awake_eo / drug / recovery_ec / sleep):

| Arm | Stage A | Stage B |
|---|---|---|
| 10 s (all six) | 120 / 120 / 100 / 100 / 60 | 180 / 180 / 130 / 120 / 150 |
| D20L, S20L (50 s) | 113 / 118 / 98 / 100 / 60 (489) | 167 / 172 / 130 / 113 / 150 (732) |
| D10L, S10L (100 s) | 94 / 90 / 74 / 79 / 56 (393) | 122 / 115 / 104 / 94 / 150 (585) |

Every animal keeps drug windows in every arm (Kin2 and Su lose none; the losses are Chibi and George, at most 11 of 40 drug windows at 100 s). Sleep is Chibi and George only, as at 100 Hz.

## Compute

Gadi, pushed snapshot `operations/sources/mi-lowrate-<commit>`. Build (30 archives, 30 cpus): expected 0.05 kSU, worst 0.15. Scoring: the T = 1000 arms cost about the anchor's 60 SU each, the T = 100–200 arms a few SU; expected 0.4 kSU, worst 0.8 kSU in all.
