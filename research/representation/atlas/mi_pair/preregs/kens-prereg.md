# Surrogate-ensemble fourth-order readouts: what moves the null, and a fairer level readout beside Δz

Post hoc and exploratory, frozen 25 November 2026 at the commit that adds it, before any ensemble output exists. Code: [`../cumulants/kens.py`](../cumulants/kens.py); job [`../jobs/kens.pbs`](../jobs/kens.pbs). It follows [`../kdiff-result.md`](../kdiff-result.md).

## Why

- In kdiff, 64% of mean difference kurtosis's drug rise (`KD`), and 74% of Mardia's, was the single IAAFT draw's kurtosis falling. Two mechanisms were left untested: finite-window bias at slow-wave effective n, or something specific to IAAFT.
- One draw gives no per-pair null spread, so effects cannot be standardised, and averaging over all 496 pairs may dilute an effect concentrated in a subset.

## Already seen (so this is not blind)

These come from a local check on 4 figure windows (`figwin/windows.npz`: 1 eyes-open window, 3 drug windows), with 10 to 40 draws each.

- **Mechanism.** The random-phase expectation of k04 (window mean), computed from each window's own Fourier coefficients, matched the phase-randomised and IAAFT ensemble means.
  - Eyes-open: −.017 predicted, −.008 phase-randomised, −.027 IAAFT.
  - Drug: −.113, −.124 and −.118; −.051, −.056 and −.068; −.130, −.142 and −.094.
  - Per-pair agreement was modest at these draw counts (r .15 to .87).
- **Marginal leak.** The spectrum-final IAAFT leaves platykurtic marginals: −.07 to −.12 in drug windows, against −.03 for the normal-scored data. That enters k04 as (ka + kb) / (4(1 − r)²).
  - In the 3 drug windows, this marginal term was +.075, +.075 and +.301 of single-draw net k04 per pair. The totals were +.179, +.628 and +.011.
- **k22 sign.** Synthetic check with normal-scored marginals: k22 is negative under a shared gain or a coupling-strength mixture, and positive when the two electrodes' amplitudes alternate. kdiff-result's reading of positive k22 as "co-modulation" was backwards.

## Readouts (per window, all in panel (c)'s units)

Each window is expressed as its value minus its session's eyes-closed mean, in SDs of the eyes-open windows.

- **Primary: `KX`.** For each pair, take the cross-only k04, i.e. k04 with the marginal kurtosis terms removed. Standardise it against the IAAFT ensemble: (real − ensemble mean) / ensemble SD, with B = 100. `KX` is the mean of this over the 496 pairs.
- **Secondary (descriptive).**
  - `MX`: the same construction for cross-only Mardia (k40 + 2k22 + k04).
  - The cross-only components k40x, k22x and k04x, each as standardised and as raw net against the ensemble mean.
  - L-kurtosis net (real − ensemble mean) along the difference (`tau_v`) and the sum (`tau_u`).
  - Tail fraction: the share of pairs whose real value exceeds all but 2 of the 100 IAAFT draws.
  - `KX` within the tightest |r| tercile.
  - `KD_ens`: the KD construction against the ensemble mean instead of draw 0.
  - `KX` against the phase-randomised ensemble.
- **Mechanism.** Run three checks on the window-mean k04 and k22 of each null (IAAFT mean, phase-randomised mean, random-phase prediction):
  - **(M1)** Compare their drug − eyes-closed changes per animal, and their window-level r.
  - **(M2)** Share of single-draw `KD`'s drug change carried by the marginal term (real minus draw 0).
  - **(M3)** Whether the standardised readouts still rise once the null's own spectral fall is part of the reference.

## Tests

1. **Gates.**
   - Draw 0's window-mean net k04 reproduces `kd_*.npz` `KD` within 1e-5.
   - The identities in `kens.py` match `kdiff.k4` (checked locally, within 1e-13).
2. **Mechanism.**
   - **Spectral concentration:** the random-phase prediction reproduces the IAAFT mean's drug change in window-mean k04 within ±25% pooled, with window r ≥ .9. Under this reading the null's fall is the finite-window kurtosis of a random-phase signal with that spectrum.
   - **IAAFT-specific:** the IAAFT mean departs from the prediction by more than 25%.
3. **Drug effect.** The animal means of drug − eyes-closed, with a one-sample t test (3 df), as for the anchor.
4. **Anaesthesia against sleep** (Chibi and George only; descriptive). Report the drug − sleep difference per animal and the window AUC. Report the drug coefficient with and without the delta fraction, with an animal intercept.
5. **Decision.**
   - `KX` replaces `KD` as the level readout if its drug t(3) is at least `KD`'s +2.71 and it is positive in 4/4 animals.
   - It improves on Δz for anaesthesia against sleep only if its AUC exceeds Δz's (.93 and .82) in both animals.
   - If `KX` is not positive in 4/4 animals, the level readouts' drug effect is attributed to the null's spectral and marginal shifts, not to the data.

## Reading

- A standardised cross-only rise means the pair's joint fourth cumulants exceed what spectrum-matched linear Gaussian dynamics give at that spectrum and window length.
- It is still produced by any of the three routes: a nonlinear coupling function, a gain or coupling mixture, or linear mixing of non-Gaussian sources. It is not evidence of nonlinear coupling.
- k22x > 0 means the sum and difference amplitudes rise together beyond the null, which with fixed marginals means the electrodes' amplitudes alternate. k22x < 0 means a shared gain or a coupling-strength mixture.
- Sleep is 2 animals, so all anaesthesia-against-sleep statements remain descriptive.

## Amendment (25 November 2026, before any output; the job at aacb335 never started because scratch was over its inode quota)

- **Cross-only terms dropped.** The difference kurtosis is bounded. As r → 1, a spectrum-final surrogate's marginal and cross terms nearly cancel, so either term alone inverts the marginal mismatch.
  - Local check: in drug:05, cross-only net was −.290 against −.024 for the marginal-matched null.
  - The rows `k40x`, `k22x`, `k04x` and `Mx` are removed.
- **New null arm `im`.** These are the same IAAFT draws, normal-scored, so their marginals equal the real window's exactly and the (ka + kb) terms cancel in the net. Their r differs from the real window's by at most .033 in the local check.
- **Primary is now `KZ`.** For each pair, take plain k04 (real) minus the `im` ensemble mean, divided by the `im` ensemble SD. `KZ` is the mean of this over the 496 pairs.
  - `MX` becomes `MZ`, the same construction for Mardia.
  - All secondary readouts use `im` as the reference. The spectrum-final arm (`ia`) and the phase-randomised arm (`ft`) are diagnostics.
- **M2 is redefined.** The marginal share of single-draw `KD` is now draw 0's net against `ia` minus its net against `im`.
- **Decision 5 applies to `KZ`.**
- Local check of net k04 against `ia` and against `im`, in 4 figure windows with 10 draws each: +.066 / +.060, +.179 / +.131, +.628 / +.604, +.011 / −.027.
  - Net k22 drops under `im` in every drug window: +.033 → −.017, −.022 → −.048, +.089 → +.051.
