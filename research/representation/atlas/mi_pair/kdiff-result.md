# Result: mean difference kurtosis as a level readout beside Δz, and anaesthesia against sleep (25 November 2026)

- **Rule:** [preregs/kdiff-prereg.md](preregs/kdiff-prereg.md), frozen at f42dce5 before any output of this readout.
- **Code:** [`cumulants/kdiff.py`](cumulants/kdiff.py) and [`jobs/kdiff.pbs`](jobs/kdiff.pbs).
- **Gadi job:** 185182245, run from the snapshot `mi-kdiff-f42dce5`. It ran for 1 min 42 s on 48 CPUs (about 3 SU) and exited 0.
- **Outputs:**
  - [`kd-score.txt`](../../../../results/representation/atlas-mi-pair-261117/cumulants/kd-score.txt), `kd-score.json`, `kd-windows.csv.gz` and the `kd_*.npz` files are in `results/representation/atlas-mi-pair-261117/cumulants/`.
  - The `kd_*.npz` files are checked against `SHA256SUMS-kdiff`.

## Verdict: a companion level readout, not a replacement for panel (c)

**Gates.**
- The net `Kdiff` reproduction passes for all 10 files, with a maximum gap of 1.7e-9.
- The glyph gate printed 9.90 against the frozen 2.37. The frozen reference was wrong; the data are not at fault (verified):
  - The rendered `mi-figure*.pdf` files print "kurtosis 9.8".
  - The pair is the same: electrodes 74 and 86 in both the bank and the figure.
  - The gap from 9.78 to 9.90 is only the plotting-position convention of the normal scores: rank/(n+1) in the figure against (rank − .5)/n in the bank. Kurtosis at r = .91 is that sensitive to the extreme ranks.
  - Neither convention, nor edge 407 of the 127-channel window, reproduces 2.366. Its source is unknown.

The table below is in panel (c)'s units. "Drug" is the mean over 4 animals; "sleep" is Chibi and George only. Retention is the drug coefficient given delta divided by the coefficient without it.

| Readout | Drug, t(3), p | Animals | Sleep | Drug − sleep, window AUC (Chibi / George) | Retention given delta |
|---|---|---|---|---|---|
| ZGK (Δz) | −3.27, −6.11, .009 | −4.69 −2.16 −2.85 −3.38 | −0.11 −0.41 | −4.58 .93 / −1.75 .82 | .56 |
| **KD (primary)** | +1.19, +2.71, .073 | +2.11 +1.42 −0.00 +1.24 | +0.81 +0.64 | +1.29 .61 / +0.78 .62 | .80 |
| KD_real | +0.44, +0.87, .45 | +1.43 +0.49 −0.97 +0.82 | +0.73 +0.64 | .46 / .48 | — |
| KD_surr | −2.65, −6.37, .008 | −2.49 −3.29 −3.27 −1.53 | −0.36 −0.06 | .81 / .86 | .51 |
| KD_tight | +1.89, +4.74, .018 | +2.29 +2.44 +0.71 +2.11 | +0.83 +0.61 | .63 / .70 | .77 |
| MARDIA | +2.11, +6.96, .006 | +2.94 +2.16 +1.57 +1.76 | +0.76 +0.61 | .78 / .72 | .56 |
| KD_raw | +0.32, +0.53, .63 | −0.09 −0.04 −0.69 +2.11 | +0.39 +0.54 | .37 / .42 | — |

**Decision rules (as frozen).**
- `KD` does not separate anaesthesia from sleep better than Δz: its AUCs are .61 and .62, against .93 and .82.
- It is less confounded by slow-wave share: it keeps .80 of its drug coefficient given delta, against Δz's .56. But the drug effect itself is not significant (t(3) +2.71, with Kin2 at 0.00).
- So it is a companion, not a replacement.
- MARDIA is the strongest level readout: 4/4 animals, t(3) +6.96. Its AUCs are below Δz's in both animals, and its retention equals Δz's.

## What carries KD's drug effect: the null moves, the data barely do

- `KD_real` barely moves under drug (+0.44, t(3) +0.87). `KD_surr` falls (−2.65, t(3) −6.37, 4/4 animals). The net rise is therefore mostly the IAAFT surrogate's difference kurtosis turning negative. That surrogate is a Gaussian-like process with the window's spectrum.
- The likely cause is the slow-wave spectrum (inferred):
  - Within drug windows, `KD_surr` tracks the delta fraction (Spearman −.53), and half its drug coefficient goes when delta is adjusted for.
  - Slow, near-periodic differences are platykurtic: a sinusoid's excess kurtosis is −1.5.
- The honest wording is "under drug the electrode differences are heavier-tailed than their spectrum predicts", not "more kurtotic".
- Δz does not decompose the same way. Here z_real and z_surr are both in Δz's eyes-open SD units, from the stored `zns` V[:, 1]:

  | Part | Drug | Animals | t(3) | Sleep |
  |---|---|---|---|---|
  | z_real | −4.95 | −0.84 −6.79 −7.33 −4.85 | −3.37 | +0.96 −0.82 |
  | z_surr | −1.68 | +3.85 −4.63 −4.48 −1.47 | −0.85 | — |

  Both parts move, and they move heterogeneously, Chibi's surrogate in the opposite direction. Only their difference is consistent in all 4 animals. So the real data carry more of Δz's drug effect than of KD's. Δz's consistency still comes from the contrast with the null, not from either part alone.

## Reading

- All three routes raise mean difference kurtosis: a nonlinear coupling function, a time-varying gain or coupling mixture, and linear mixing of bursty sources. It is not evidence of nonlinear coupling.
- Normal-scoring is required. On raw signals the readout is null (`KD_raw`), because marginal burstiness swamps it.
- Window-level correlation with ZGK is −.44 for KD and −.53 for MARDIA. They are related but not redundant.
- Medetomidine sits at or below sleep on KD in George (−0.55) and only just above it in Chibi. The drug-against-sleep contrast is carried by KT, KTMD and PF.
- Sleep exists in 2 animals, so every drug-against-sleep statement here is descriptive. The binding constraint for an "anaesthesia differs from both wake and sleep" paper is sleep data, not the choice of statistic.
- Recommendation: keep Δz as panel (c). If a level readout is shown, show MARDIA or KD beside Δz with its real and surrogate parts. A level-only panel would show mostly the null's spectral shift.
