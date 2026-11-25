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
| KD_real (own scale) | +0.44, +0.87, .45 | +1.43 +0.49 −0.97 +0.82 | +0.73 +0.64 | .46 / .48 | — |
| KD_surr (own scale) | −2.65, −6.37, .008 | −2.49 −3.29 −3.27 −1.53 | −0.36 −0.06 | .81 / .86 | .51 |
| KD_tight | +1.89, +4.74, .018 | +2.29 +2.44 +0.71 +2.11 | +0.83 +0.61 | .63 / .70 | .77 |
| MARDIA | +2.11, +6.96, .006 | +2.94 +2.16 +1.57 +1.76 | +0.76 +0.61 | .78 / .72 | .56 |
| KD_raw | +0.32, +0.53, .63 | −0.09 −0.04 −0.69 +2.11 | +0.39 +0.54 | .37 / .42 | — |

**Decision rules (as frozen).**
- `KD` does not separate anaesthesia from sleep better than Δz: its AUCs are .61 and .62, against .93 and .82.
- It is less confounded by slow-wave share: it keeps .80 of its drug coefficient given delta, against Δz's .56. But the drug effect itself is not significant (t(3) +2.71, with Kin2 at 0.00).
- So it is a companion, not a replacement.
- MARDIA is the strongest level readout: 4/4 animals, t(3) +6.96. Its AUCs are below Δz's in both animals, and its retention equals Δz's.

## What carries the level readouts' drug effect: mostly the null falling

Parts are on the net readout's own eyes-open SD scale, so real − surrogate = net. Recomputed from `kd-windows.csv.gz` and the `kd_*.npz` files. The table rows `KD_real` and `KD_surr` above are each on their own scale and are not comparable.

| Part | Drug | Animals | t(3) | Sleep | Raw k04 drug − EC | Spearman with delta (drug windows) |
|---|---|---|---|---|---|---|
| KD net | +1.19 | +2.11 +1.42 −0.00 +1.24 | +2.71 | +0.81 +0.64 | +.102 | — |
| KD real | +0.43 | +1.39 +0.47 −0.94 +0.80 | +0.87 | +0.71 +0.62 | +.046 | −.31 |
| KD surrogate | −0.76 | −0.71 −0.94 −0.94 −0.44 | −6.37 | −0.10 −0.02 | −.056 | −.60 |
| MARDIA net | +2.11 | +2.94 +2.16 +1.57 +1.76 | +6.96 | +0.76 +0.61 | +.205 | — |
| MARDIA real | +0.54 | +1.45 +0.36 −0.43 +0.79 | +1.37 | +0.52 +0.51 | +.061 | −.30 |
| MARDIA surrogate | −1.57 | −1.49 −1.81 −2.01 −0.97 | −6.97 | −0.23 −0.10 | −.143 | −.71 |

- About 64% of KD's drug rise, and 74% of MARDIA's, is the surrogate's kurtosis falling. The surrogate fall is consistent in 4/4 animals. The real part rises in 3/4 animals (Kin2 falls) and is not significant.
- The surrogate's mean k04 is already negative in eyes-closed wake (−.042) and falls further with delta. Two candidate mechanisms remain; neither is tested:
  - **(a) Finite-sample bias.** Sample kurtosis is biased negative when the effective sample size is small, and slow-wave windows have fewer effective samples. This bias hits the real data equally. On this reading the net is the bias-corrected quantity, and the flat real part is a genuine rise masked by bias. That is the case for subtracting a spectrum-matched null.
  - **(b) A spectral or phase-structure effect** specific to the IAAFT construction, which would not apply to the real data in the same way.
  - Telling (a) from (b) needs the k04 of Gaussian processes at each window's spectrum, or a bias correction by effective n. Neither has been run.
- Safe wording: "under drug the electrode differences are heavier-tailed than a Gaussian with the same spectrum". Do not write "more kurtotic" in absolute terms.
- Δz split the same way, with z_real and z_surr both in Δz's eyes-open SD units, from the stored `zns` V[:, 1]:

  | Part | Drug | Animals | t(3) | Sleep |
  |---|---|---|---|---|
  | z_real | −4.95 | −0.84 −6.79 −7.33 −4.85 | −3.37 | +0.96 −0.82 |
  | z_surr | −1.68 | +3.85 −4.63 −4.48 −1.47 | −0.85 | — |

  - For Δz the real part carries most of the effect and falls in 4/4 animals.
  - The surrogate part is inconsistent: Chibi's rises by +3.85.
  - The net is the tightest of the three (t(3) −6.11 against −3.37 for z_real), because subtracting the null removes animal-specific spectral shifts.
  - So Δz's drug effect sits mainly in the data, while the kurtosis readouts' sits mainly in the null.

## Reading

- All three routes raise mean difference kurtosis: a nonlinear coupling function, a time-varying gain or coupling mixture, and linear mixing of bursty sources. It is not evidence of nonlinear coupling.
- Normal-scoring is required. On raw signals the readout is null (`KD_raw`), because marginal burstiness swamps it.
- Window-level correlation with ZGK is −.44 for KD and −.53 for MARDIA. They are related but not redundant.
- Medetomidine sits at or below sleep on KD in George (−0.55) and only just above it in Chibi. The drug-against-sleep contrast is carried by KT, KTMD and PF.
- Sleep exists in 2 animals, so every drug-against-sleep statement here is descriptive. The binding constraint for an "anaesthesia differs from both wake and sleep" paper is sleep data, not the choice of statistic.
- Recommendation: keep Δz as panel (c). If a level readout is shown, show MARDIA or KD beside Δz with its real and surrogate parts. A level-only panel would plot a quantity that is about two-thirds the null's shift, so its reading depends on the untested mechanism above.

## Follow-ups (post hoc, 25 November)

- **Level readout in MI units.** The mean over pairs of (K − G), real minus surrogate, is identical to the draft figure's `lvl` (mean K real − surrogate), because G is the same in both. In panel (c)'s units:
  - Drug: −1.47, with animals +0.36, −2.70, −2.20 and −1.36. t(3) −2.20, p .115.
  - Sleep: −0.09 and −0.53. AUC .65 and .72.
  - Drug-against-sleep coefficient: −0.82, which becomes +1.76 given delta.
  - It also carries KSG's real-only bias offset ([ksg-bias-result.md](ksg-bias-result.md)). It is not a usable readout.
- **Component breakdown of MARDIA's drug effect, net, in MARDIA's eyes-open SDs.** Values are the drug mean, then the animals in the order Chibi, George, Kin2, Su.

  | Component | Drug | Chibi | George | Kin2 | Su |
  |---|---|---|---|---|---|
  | k04 | +0.94 | +1.66 | +1.12 | −0.00 | +0.98 |
  | 2·k22 | +0.72 | +0.69 | +0.45 | +1.70 | +0.04 |
  | k40 | +0.45 | +0.59 | +0.59 | −0.13 | +0.74 |

  - Sleep 2·k22 is −0.15 and −0.21.
  - Sign of k22 (verified on synthetic pairs with normal-scored marginals): k22 = (ka + kb − 2 cum(a,a,b,b)) / (4(1 − r²)). It is negative under a shared gain or a coupling-strength mixture and positive when the two electrodes' amplitudes alternate. Positive k22 is not co-modulation.
- **Mechanism, local check on 4 figure windows** (details in [preregs/kens-prereg.md](preregs/kens-prereg.md)):
  - The surrogate's window-mean k04 matches the random-phase expectation computed from the window's own Fourier coefficients, so candidate (a) is supported in this form.
  - Part of single-draw net k04 is a marginal leak. The spectrum-final IAAFT leaves platykurtic marginals, which enter k04 as (ka + kb) / (4(1 − r)²).
  - Against the same draws normal-scored (marginals matched), net k04 falls in the 3 drug windows: +.179 → +.131, +.628 → +.604 and +.011 → −.027.
  - Net k22 falls more: +.033 → −.017 and +.089 → +.051.
  - The ensemble run tests both on all windows: `kens.py` at 10e6b9e, Gadi job 185197676, with B = 100 draws in three null arms.
