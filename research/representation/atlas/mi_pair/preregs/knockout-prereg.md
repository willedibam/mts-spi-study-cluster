# Which band's structure does the macaque MI-pair drug fall need? Band-knockout surrogates at fixed spectrum, and a drug-blind offset for the band-sweep arms

**DRAFT, not frozen.** Drafted 26 November 2026. Freezes at the commit that adds the code and the Part A calibration gates, before any real-data knockout value is computed. Exploratory addendum to [`bands-prereg.md`](bands-prereg.md); it cannot change the anchor or band-sweep verdicts.

**Known when drafted.** Every band-sweep value in [`../lowrate-result.md`](../lowrate-result.md) had been read, including the per-agent pattern below. That choice of 8–15 Hz for generator 3 and the Part B citability rule were made knowing those values. No knockout value, synthetic or real, had been computed.

## Why

- The band sweep ([`../lowrate-result.md`](../lowrate-result.md), addendum) filtered the signal. Filtering changes four things together: the band, the spectrum the estimator sees, the number of independent samples (about 2 × bandwidth × 10 s) and, by Mallows' result (Mallows 1967, J. Appl. Prob. 4:313; abstract only), how Gaussian the signal is. So a band that loses the fall may only have lost power, and a band that keeps it may owe part of its size to the estimator's response to its spectrum. The fast arms (HP8, HP15, BP8-15: R about 1.5, half the shift matched by the surrogate) are uncited for that reason.
- A knockout keeps the window whole and destroys the nonlinear structure in one band only: the Fourier phases in that band are randomised (one random phase per frequency, shared across channels, as the joint IAAFT does), and every other band keeps its real phases. Power spectrum, cross-spectrum, sample count, duration and rate stay fixed. Only "is band B's structure, and its phase relation to the other bands, needed?" changes.
- Part B gives the filter arms the drug-blind offset the band sweep lacked, so their magnitudes can be cited or retracted.
- Prior expectations (from the band sweep, not adopted as predictions): ketamine's fall sits below about 8 Hz; propofol and ketamine–medetomidine grow above 4 Hz, most in 8–15 Hz.

## Part A: knockout

**Data and estimator.** The anchor's 10-s, 100 Hz banks (stage A and B, 1,260 windows), the `edges.py recompute` calls (32-channel subsets by the same `first`/`row` rule), normal-scored, AUTO Theiler window with cap T // 10, k = 4, ψ(T). Unchanged.

**Bands.** FFT bins of the 1000-sample window (0.1 Hz resolution), partitioned: B1 0.1–2, B2 2–4, B3 4–8, B4 8–15, B5 15–25, B6 25–50 Hz (lower edge inclusive). The DC and Nyquist bins are left alone. The partition is exact, so knocking out every band is the full joint phase-randomised (FT) surrogate.

**Variants per window.** v = normal-scored window. One phase vector φ per window, seed `default_rng([row, 2000])`, the same φ for every variant of that window (so variants are paired):
- real: v (z must reproduce the stored anchor `zns` V[:, 1, 0] to 1e-6: gate);
- knock-B (six): bins in B multiplied by e^{iφ(f)}, all channels alike;
- keep-B (six): every bin except B's multiplied;
- knock-all: every bin multiplied.
Each variant is normal-scored again before `edge_mi`. That restores Gaussian marginals and moves the spectrum slightly, alike for every variant. 14 KSG evaluations per window (13 new).

**Readouts.** Per window, z for each variant. Per session, Δ = drug − awake eyes-closed. Pooled over four animal means:
- D = Δ(z_real − z_all): the knockout version of the anchor's fall. The anchor used a spectrum-final IAAFT; D against the anchor's Δ is reported (descriptive).
- Loss ℓ_B = Δ(z_real − z_knock-B) / D: the share of the fall removed by destroying band B's structure.
- Sufficiency s_B = Δ(z_keep-B − z_all) / D: the share kept when only B is real.
- Power share π_B per state (mean over channels of B's share of the normal-scored window's power), reported beside each ℓ_B.

**Calibration (drug-blind, run first, frozen before real data).** `long100.py`'s `gen` and sources at T = 1000 (16 channels phase-randomised from the periodograms of normal-scored anchor windows: awake eyes-closed of KTMD units, KTMD drug, PF drug), all 14 variants:
1. `block` (stationary linear), n = 100 per spectrum. False-positive check: every ℓ_B numerator (z_real − z_knock-B) within 2 SE of 0 or within ±.001.
2. `gain_shared` (one broadband gain switch, as now), n = 40. Gives the reference loss ℓ^gain_B, built as the real readout is: numerator and denominator are each (`gain_shared` on the drug spectrum) − (`block` on the awake spectrum), with the drug spectrum taken as the mean of KTMD and PF. A shared broadband gain is expected to make ℓ_B track π_B, because destroying a high-power band's phase leaves the modulated part a small share of the variance. That is the confound this generator measures.
3. `gain_B4` (new): the same Markov gain switch applied to the 8–15 Hz component only (FFT split, gain on B4, summed back), n = 40.
4. `gain_slowB4` (new, descriptive): the 8–15 Hz component's gain follows the sign of a shared 0.1–2 Hz driver (cross-frequency coupling), n = 40.

**Identifiability gates.** The band-specific reading is licensed only if (i) generator 3 singles out B4: ℓ_B4 − ℓ^gain_B4 above a margin, and every other band's ℓ_B within a tolerance of its ℓ^gain_B, for every spectrum; and (ii) generator 1 passes. The margins below (.25, .15, .10) are provisional. Before freezing they are reset in SE units from a drug-blind synthetic pilot (bootstrap SE of each ratio over generator draws), which is legitimate because no real-data value enters. If (i) fails, Part A is descriptive: the ℓ_B and s_B tables with π_B, no locus claim. Calibration gates of this kind have failed before (reorder v1 and v2, the envelope null, the scalar-gain test), so failure is a live outcome.

**Reading on real data (if the gates pass).** D must be ≤ −2 own eyes-open SD with 4/4 animals negative, or nothing beyond the table is read. Then for each band, against the broadband-gain reference for the drug spectra (mean of KTMD and PF):
- **Needed beyond shared gain:** ℓ_B − ℓ^gain_B ≥ .25 and the animal-level Δ(z_real − z_knock-B) has t(3) ≤ −2.35 (4/4 negative).
- **Not needed beyond shared gain:** ℓ_B − ℓ^gain_B ≤ .10.
- **Sufficient:** s_B ≥ .5 with t(3) ≤ −2.35.
- Otherwise in between; report the numbers.
- Overall: **single-band locus** if exactly one band is needed beyond shared gain and sufficient; **compatible with shared broadband gain** if no band is needed beyond it; **cross-band** if two or more bands are each needed and none is sufficient. Per agent: the same table, descriptive (KT n = 4 sessions, PF 4, MD 4, KTMD 11).
- Controls: eyes-open and sleep Δ for every variant, flagged at 2 own SD as before.

**What a knockout removes.** Randomising band B's phases destroys (a) non-Gaussian joint structure inside B, including shared amplitude bursts in B, and (b) phase relations between B and the other bands, including cross-frequency coupling and the harmonics of non-sinusoidal waveforms. It cannot tell (a) from (b); generator 4 shows how cross-band structure appears. It keeps each band's power and cross-spectrum, so it does not test linear coupling.

## Part B: drug-blind offset for the band-sweep arms

For each of the twelve band-sweep arms (LP2, LP4 = L10, LP8 = L20, LP15, LP25, HP2, HP4, HP8, HP15, BP2-4, BP4-8, BP8-15; the 0.5–8 and 0.5–4 Hz spectra were never calibrated, as `long100-prereg.md` notes for W50L8), `long100.py`'s calibration at T = 1000 on that arm's filtered windows: `block` n = 100 and `gain_shared` n = 40 per spectrum (awake eyes-closed of KTMD units, KTMD drug, PF drug), against the arm's spectrum-final IAAFT, as in the anchor calibration. Offset = mean(KTMD, PF) − AW of z_real − z_surr on `block`, ± SE; detectability on `gain_shared`.

**Reading.** Offset-adjusted R_adj = (arm Δ − offset) / (anchor Δ − anchor offset). An arm's magnitude is **citable** if the offset is within 2 SE of 0 or |offset| ≤ .25 |arm Δ|; otherwise it is reported as R_adj with "offset-dependent". The arm Δs are known, so this rule is set with them in view; it applies the same threshold to every arm, and no arm is chosen as best.

## Deferred, and why

- **Theiler window fixed in seconds (0.25, 0.5, 1, 2 s), same in every state.** Partly covered: the robustness study's 4×AUTO is pair-adaptive (about 0.2 s awake, .68 s under the drugs) and survives, but its size flips with the ψ convention (R .72 under ψ(T), 1.19 with pool ψ), so a fixed window in both states is untested. It tests whether the estimator needs temporal neighbours within the ~1-s stretches, the proposed reason for the rate dependence. `lowrate-result.md` assigns it to the KSG-bias workstream; cost about 0.25 kSU. Deferred to that workstream, not dropped.
- **40 Hz with a 15 Hz low-pass, 25-s windows (T = 1000).** It keeps the 8–15 Hz content at a lower rate, but it changes rate and duration together, like the 20 Hz long arms. Until the long100 duration cell (W50 against C10) returns, its result could not be attributed to rate. Run it after long100, using that cell's duration effect to read it.

## Pre-stated limits

- One phase draw per window; 32-channel subsets; four animals (t(3)).
- FFT phase randomisation treats the window as periodic; end mismatch enters every variant except real, alike across bands except that it weighs most on the low bands.
- Re-normal-scoring after the knockout makes the spectra approximately, not exactly, equal across variants.
- The gain generators are synthetic; real drug gain may mix bands in ways none of them captures, so "not needed beyond shared gain" is relative to this generator.
- MI is not additive: losses need not sum to 1, and both halves of a split can each be needed.

## Compute (estimated, not timed)

Measured basis: a T = 1000, 32-channel KSG evaluation costs about 20 core-s (band-sweep scoring, about 60 SU per arm of 4 evaluations × 1,260 windows). Part A real data 14 × 1,260 evaluations: about 0.2 kSU. Part A calibration (3 spectra × 220 generators × 14 evaluations, 16 channels, about 4× cheaper): about 0.03 kSU. Part B (12 arms × 3 spectra × 140 × 2 evaluations, 16 channels): about 0.05 kSU. Expected 0.3 kSU, worst 0.6 kSU.
