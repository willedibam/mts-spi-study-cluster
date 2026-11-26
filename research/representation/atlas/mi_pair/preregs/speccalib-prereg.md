# Own-spectrum calibration: is the residual fall in PRIOS and monkey B beyond what each dataset's state spectra give a linear-Gaussian process?

Frozen 26 November 2026, before any calibration value exists. Post hoc in motivation: both observed effects are known (PRIOS primary −.0083; monkey B `pa` −.0043, `paall` −.0033). Code: [`calib_own.py`](../calib_own.py).

## Why

The residual (real z − mean z of three joint IAAFT draws) is read as non-Gaussian dependence because the surrogate keeps the spectra, cross-spectra and marginals. That reading needs the residual's drug − awake change to be about zero for a process with no non-Gaussian dependence. NeuroTycho's calibration ([`robustness-result.md`](../robustness-result.md)) shows it is not: a stationary, linearly coupled Gaussian process with NeuroTycho's drug spectra shifts it by about −.011 at T = 200 and +.006 at T = 1000. PRIOS (T = 200, −.0083) has never been calibrated at its own spectra, and neither has monkey B (T = 1000). Both effects are surrogate-driven (the real z rises and the surrogate rises more), which is the case where a mismatch between the estimator's behaviour on real and IAAFT data would matter most.

## Generator

- Sources: real windows of each state after the scored run's reference and channel set, normal-scored.
  - PRIOS primary (`human/prios`, post-deviation): CAR over good, non-stimulated channels, last dropped, the evenly spaced 32 (as `human/score.py`, variant car); the 5 included subjects; awake and propofol; T = 200.
  - Monkey B: `W32pa` (per-array CAR, the fixed 32 channels of `pa`); W1 and D; T = 1000. The 190-channel run is not calibrated (cost); its observed Δ is compared descriptively.
- Per synthetic window: the source's channels in random order, each phase-randomised independently (own periodogram, no cross-dependence; `ksgbias_calib.phase_rand`).
  - Then `block`: stationary linear coupling within blocks of four, as in `ksgbias_calib.gen` (equal-correlation block, strength U(.2, .9), diagonal jitter U(0, .3)), on as many full blocks as the channel count allows.
  - `gain_shared` (sensitivity only): the same, times one shared two-state gain (×3, mean dwell T/10).
- Then normal-scored, three spectrum-final joint IAAFT draws, z by the scored statistic (KSG k = 4, AUTO Theiler window; `claar/zgk_ns.zgk`, identical output to `edges.edge_mi`). Residual = real − mean of the three surrogate z.
- Counts: PRIOS 80 `block` and 20 `gain_shared` windows per subject and state, sources spread evenly over that subject-state's windows. Monkey B 150 `block` and 40 `gain_shared` per state. Seeds fixed in the code.
- Gate before scoring: `zgk` on the stored real windows reproduces the stored real z of two PRIOS windows and two monkey B `pa` windows (to 1e-6, or the declared platform-tie gap of about 1e-4 for PRIOS, which was computed on Gadi).

## Rule

- Offset O = mean `block` residual (drug) − mean `block` residual (awake). For PRIOS it is computed per subject and averaged over subjects, as the observed Δ is. SE from the per-window spread.
- **Primary (the frozen 2-s rule of `robust-prereg.md`, applied at each dataset's own spectra):** the observed Δ is DISTINGUISHED from the spectrum response if Δ_obs − O < −2 SE(O); otherwise NOT DISTINGUISHED. If O ≥ 0, the observed fall is not spectrum-made in the drug direction, and the test is conservative.
- **Secondary:**
  - PRIOS: a one-sided t on subject (Δ_s − O_s), n = 5.
  - Net Δ_obs − O for both datasets.
  - Monkey B `paall` against the 32-channel O (descriptive; the channel count differs).
- Sensitivity: detectability = (mean `block` − mean `gain_shared`) / SD(`block`) per state. If it is below .5 in both states, the calibration cannot show sensitivity to known non-Gaussian dependence; the offset is still reported.
- **Limits, stated now:**
  - The block structure is not the real coupling structure.
  - `gain_shared` is one kind of non-Gaussian dependence.
  - Neither frozen verdict (PRIOS REPLICATES, monkey B UNINFORMATIVE) changes. The outcome sets how the effects are worded.
