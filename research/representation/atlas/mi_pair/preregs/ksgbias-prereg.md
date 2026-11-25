# Does the MI-pair drug change survive a bias-corrected KSG? Circular-shift offset on the macaque 10-s windows

Frozen 25 November 2026 at the commit that adds it, after the synthetic calibration and before any corrected KSG value of a real window was looked at. Code: [`../ksgbias.py`](../ksgbias.py) (estimators, scoring), [`../ksgbias_calib.py`](../ksgbias_calib.py) (calibration), [`../jobs/ksgbias.pbs`](../jobs/ksgbias.pbs). Calibration output: [`calib.txt`](../../../../../results/representation/atlas-mi-pair-261117/ksgbias/calib.txt).

## Why

- Panel (b) of the MI figure shows many edges with negative KSG MI (K). True MI is ≥ 0, and with Gaussian marginals, MI ≥ G = −½log(1 − r²) (the GCMI bound). So, at normal-scored weak edges, K < 0 and K − G < 0 beyond sampling error are estimator error.
- Real K at weak edges (G < .01) sits .010–.016 nats below its IAAFT surrogate's K under KTMD and PF. IAAFT matches G and the AUTO Theiler window w, so this part of the bias is real-only and could carry some of the drug Δz.
- The question: does the drug Δz of z(G, K) survive an estimator whose bias at independence is removed per edge, under that edge's own serial structure?

## Calibration (done; synthetic only)

- **Generators.** Generators per `ksgbias_calib.py`: 16 channels with the spectra of real stage-A channels (PF drug, KTMD drug, KTMD awake eyes-closed), four blocks of four, T = 1000 and T = 200 (PF), n = 40 per cell. Between-block edges have MI 0.
- **Estimators.**
  - k4: as `edges.py`, KSG k = 4 with the AUTO Theiler window and ψ(N).
  - kpool: as k4, with the pool-averaged ψ.
  - k2te: kpool with w = 2 max τ(1/e).
  - kshift: k4 minus the mean k4 of the same pair with x_j circularly shifted by T/4, T/2 and 3T/4.
- **Results at T = 1000.**
  - **Real-only bias at MI-0 edges (real − surrogate K).** It comes from per-channel gain switching.
    - Under gain_ind and burst_ind: k4 −.017 to −.035, kpool the same, k2te −.018 to −.038.
    - kshift: within ±.002 in every cell.
  - **Size.** For stationary linear coupling (block), z real − surrogate is within about 1.5 SE of 0 for every estimator.
  - **Sensitivity.** Shared gain (genuine non-Gaussian dependence on every edge) is kept by kshift. Its z real − surrogate is −.029 / −.039 (KTMD / PF), against k4's −.033 / −.034. Real − surrogate K at MI-0 edges is +.024 / +.028 for kshift, against +.012 / +.011 for k4.
- **Choice.** kshift is primary. It is the only candidate that removes the real-only bias. kpool and k2te shift the level by a w-dependent offset that the surrogate shares, so they cannot change z real − surrogate except through edge-to-edge variation in w. They are reported as secondaries.
- **Calibration prediction.** kshift ≈ k4 on z under every generator, including independent gain alone, which gives −.03 to −.06 under both. So the real-only bias at MI-0 edges carries little of z, and the drug Δz is expected to be retained at about .85–1.1.

## Real-data looks already taken

All were taken with k4 only.
- Weak-edge K real vs surrogate (above).
- Dropping edges with G < .01 retains 90% of pooled drug Δz (t(3) −7.0).
- Clipping K below the surrogate band retains 78% (t(3) −5.45).
- The local reproduction test ran `edge_all` (every estimator column) on real stage-A windows. Only k4 and w were compared to `edges.py`. No corrected column was read or scored.

## Procedure

- Windows, 32-channel subsets (`first` for stage A and stage B drug agents, `row` for stage B SLP), normal-scoring and the single spectrum-final joint IAAFT draw with seed = row index are exactly as `edges.py recompute`.
- Reproduction gate: k4 z (real and surrogate) must equal the stored `zns_[AB]_*.npz` z within 1e-4, or stop.
- Statistic and pooling are as [`montage/animal_level.py`](../montage/animal_level.py):
  - per window, residual = z real − z surrogate;
  - per session, Δ = drug (or sleep) − awake eyes-closed;
  - animal means over drug sessions;
  - one-sided one-sample t on the four animal means.
- Retention R = pooled drug Δz(kshift) ÷ pooled drug Δz(k4).

## Reading (stated before output)

- **SURVIVES.** All of these must hold under kshift:
  - 4/4 animal means < 0;
  - one-sided p < .05;
  - the drug Δz of each of the four drug agents (KT, KTMD, MD, PF) < 0;
  - R ≥ .5.
- **BIAS-CARRIED.** R < .25, or ≤ 2/4 animal means < 0.
- **PARTIAL.** Otherwise.
- SLP (sleep) and the secondaries (kpool, k2te) are descriptive only.
- **Panel (b) expectation.** kshift centres MI-0 K at 0, but its per-edge SD at T = 1000 is about .045–.05. So the fraction of weak edges with K < 0 should stay about .4–.6 under kshift. kpool should make the level more negative than k4 (about −.04), because ψ(N) partly masked the window-induced negative bias. Negative per-edge values in panel (b) are therefore expected under any unbiased estimator at this T; only their mean offset is bias.

## Limits

- **kshift's assumption.** The shifted pairs are independent only if no dependence survives a shift of ≥ T/4 (2.5 s). Calibration gains decorrelate in about 1 s, so slower shared structure in real data was not tested. A slow shared component would be partly subtracted, which pushes R down.
- **Nonstationarity.** Independent per-channel gain alone produces a z fall beyond IAAFT that kshift keeps. That is the territory of the segment and envelope nulls ([nonstat](nonstat-prereg.md), [envelope](envelope-prereg.md)), not of this test.
- **The 2-s arm (T = 200).** It is not rescored here. In calibration, stationary block coupling gives z real − surrogate ≈ −.022 ± .007 for every estimator. That covers one source set at n = 40, the cause is unknown, and whether it cancels in drug − awake is untested. 17–22% of windows have w at the T/10 cap.

## Compute

Gadi, 48 cores, 96 GB, one job. Expected about 0.14 kSU (about 1.5 h), worst case 0.38 kSU (4 h walltime).
