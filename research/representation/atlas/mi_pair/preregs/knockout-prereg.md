# Where the macaque MI-pair drug fall lives, and how the estimator responds to known structure: a synthetic response map, band knockouts and band-arm offsets

**DRAFT, not frozen.** Drafted 26 November 2026, revised the same day (slimmed, synthetic map added). Code: [`../knockout.py`](../knockout.py) and `band_phase_randomized` in `src/mtsspi/surrogates.py`. Part 0 is synthetic and drug-blind and runs first on this code. The real-data readings for Parts A and B are written into this file and `knockout.py`, and frozen, after Part 0 returns and before any real-data knockout value or band-arm offset is computed. Exploratory addendum to [`bands-prereg.md`](bands-prereg.md); it cannot change the anchor or band-sweep verdicts.

**Known when drafted.** Every band-sweep value in [`../lowrate-result.md`](../lowrate-result.md) had been read. The choice of 8–15 Hz for the band-limited generator and the Part B citability rule were made knowing them. No knockout value, synthetic or real, had been computed.

## Why

- The band sweep filtered the signal, which changes the band, the spectrum the estimator sees, the number of independent samples and how Gaussian the signal is (Mallows 1967, J. Appl. Prob. 4:313; abstract only) all at once. The low-rate arms change rate and duration together. So neither says where the fall lives or why an arm grows or shrinks; the fast arms (R about 1.5) are uncited for that reason.
- Two tools address this. A knockout randomises the Fourier phases in one band only (one phase per frequency, shared across channels), keeping power spectrum, cross-spectrum, sample count, duration and rate fixed. And a synthetic response map scores signals with known structure in every cell (rate, duration, band, Theiler window, knockout), so how the estimator responds to each manipulation is seen where the answer is known.
- The map is the reusable part. It characterises the estimator, not the macaque data: other datasets need only new source spectra, and other statistics built from the same per-edge Gaussian and KSG MI can be read from the stored arrays without recomputation. A statistic built from other estimates needs a new scorer and a rerun.

## Part 0: synthetic response map (drug-blind)

**Generators.** 16 channels, phase-randomised from the periodograms of normal-scored W50 stage-A windows (awake eyes-closed of KTMD units, KTMD drug, PF drug; 32-channel subset as `edges.py`), interpolated onto a 100-s grid at 100 Hz, so every scored window is cut from inside a longer signal and has non-circular ends, as real windows do. Four 4-channel blocks with stationary linear coupling (`long100.gen`'s mixing). Then:
- `block`: nothing more (false-positive reference), n = 100 per spectrum;
- `gain_shared`: ×3 broadband gain switch, Markov, mean dwell 1 s, n = 40;
- `gain_B4`: the same switch on the 8–15 Hz component only, n = 40;
- `xfreq`: 8–15 Hz amplitude ×3 while the channel-mean 0.1–2 Hz component is positive (cross-frequency coupling), n = 40.

**Cells.** Each realisation, normal-scored, against its own spectrum-final joint IAAFT, z(G, K) real − surrogate: 100 Hz 10 s (anchor-like) and 50 s; 8 Hz low-pass then 20 Hz, 10 s (T = 200) and 50 s (T = 1000); 8 Hz low-pass, 8 Hz high-pass and 8–15 Hz band-pass at 100 Hz, 10 s (order-8 zero-phase Butterworth on the whole signal, then cut); Theiler window fixed at 0.25 s and 1 s in every state, 100 Hz 10 s. Plus the 14 knockout variants of Part A on the anchor-like cut.

**Readouts (descriptive).** Per cell and generator, the drug − awake contrast built as the real readout is: the generator on the drug spectra (mean of KTMD and PF) minus `block` on the awake spectrum, and its ratio to the anchor-like cell. Knockout loss ℓ_B and sufficiency s_B per generator (definitions in Part A), with bootstrap SEs over draws.

**What Part 0 licenses in Part A.**
- *False positives and ends.* Each `block` knockout numerator (z_real − z_knock-B, per spectrum) must be within 2 SE of 0 or ±.001. A band that fails gets no real-data reading; the low bands are the ones at risk, since end mismatch introduced by the knockout weighs most there.
- *Denominator.* Ratios are read only where |D| ≥ 3 bootstrap SE; otherwise only the numerators are reported (the same rule applies to real data).
- *Identifiability.* `gain_B4`'s excess loss over `gain_shared`'s, ℓ_B − ℓ^gain_B, must be larger in B4 than the largest |excess| of any other band, with the bootstrap 2.5 % point of that difference above 0. A "within 2 SE" criterion is not used, because it passes on noise. If this fails, Part A is reported as tables only, with no locus claim. Gates of this kind have failed before here (reorder v1 and v2, the envelope null, the scalar-gain test).
- Everything else in Part 0 is descriptive: it shows how a known gain switch moves with rate, duration, band-limiting and a fixed Theiler window, which is what the deferred real-data arms (below) would have to be read against.

## Part A: knockouts on the anchor windows

**Data and estimator.** The anchor's 10-s, 100 Hz banks (stage A and B, 1,260 windows), the `edges.py recompute` calls and subsets, normal-scored, AUTO Theiler window capped at T // 10, k = 4, ψ(T). Unchanged.

**Bands.** FFT bins of the 1000-sample window: B1 0.1–2, B2 2–4, B3 4–8, B4 8–15, B5 15–25, B6 25–50 Hz (lower edge inclusive), an exact partition of every bin except DC and Nyquist, so knocking out all bands is the full joint phase-randomised (FT) surrogate (unit-tested).

**Variants per window.** One phase vector per window, `default_rng([row, 2000])`, shared by its variants (paired): real; knock-B (six, B's bins rotated); keep-B (six, every bin except B's rotated); knock-all. Each is normal-scored again before `edge_mi`. Stored: z and every edge's G and K per variant. Gate: z_real reproduces the stored anchor `zns` V[:, 1, 0] to 1e-6.

**Readouts.** Per session Δ = drug − awake eyes-closed, pooled over four animal means: D = Δ(z_real − z_all); loss ℓ_B = Δ(z_real − z_knock-B) / D; sufficiency s_B = Δ(z_keep-B − z_all) / D; each band's power share. Each ℓ_B is read beside the `gain_shared` reference ℓ^gain_B from Part 0, because a shared broadband gain makes loss track power share. Reported with animal-level t(3) and the Part 0 bootstrap SE. D must be ≤ −2 own eyes-open SD with 4/4 animals negative before anything beyond the table is read. The decision wording is fixed at the freeze; no locus taxonomy is set in advance.

**What a knockout removes.** Within-band non-Gaussian structure (including shared amplitude bursts) and the band's phase relations to other bands (including cross-frequency coupling and the harmonics of non-sinusoidal waveforms). It cannot tell the two apart; `xfreq` shows how the second looks. It keeps power and cross-spectrum, so it does not test linear coupling.

## Part B: drug-blind offset for the band-sweep arms

For each of the twelve arms (LP2, LP4 = L10, LP8 = L20, LP15, LP25, HP2, HP4, HP8, HP15, BP2-4, BP4-8, BP8-15; L10 and L20 were never calibrated), `long100.calib_one` at T = 1000 on that arm's filtered windows: `block` n = 100 and `gain_shared` n = 40 per spectrum. Offset = mean(KTMD, PF) − AW of z_real − z_surr on `block`, ± SE. R_adj = (arm Δ − offset) / (anchor Δ − anchor offset). An arm's magnitude is citable if the offset is within 2 SE of 0 or |offset| ≤ .25 |arm Δ|, otherwise "offset-dependent". The arm Δs are known, so this rule is set with them in view; it applies the same threshold to every arm and picks no best arm.

## Real-data arms still deferred

- **Theiler window fixed in seconds.** The 4×AUTO result covers it partly (pair-adaptive, survives; size flips with the ψ convention). Part 0 shows on synthetic data whether a fixed window changes the estimator's response to a gain switch; the real-data sweep stays with the KSG-bias workstream (about 0.25 kSU).
- **40 Hz with a 15 Hz low-pass on 25-s windows.** It changes rate and duration together; it is read after the long100 duration cell returns, and against Part 0's rate × duration cells.

## Limits

- One phase draw per window; 32-channel subsets on real data, 16 channels synthetic; four animals (t(3)).
- Re-normal-scoring after a knockout makes the spectra approximately, not exactly, equal across variants.
- The generators are synthetic; real drug changes may mix bands in ways none of them captures, so every "relative to shared gain" statement is relative to `gain_shared`.
- MI is not additive: losses need not sum to 1.

## Compute

Timed locally (one core, this code): a 16-channel KSG evaluation takes 3.9 s at T = 1000 and 119 s at T = 5000; 32 channels at T = 1000, 19.5 s (the band sweep on Gadi implied about 20 s). Part 0: 660 realisations × about 350 core-s (the 50-s cell is two thirds of it), about 64 core-hours, 0.13 kSU expected, 0.3 worst. Part A: 13 × 1,260 × 20 core-s, about 0.18 kSU. Part B: about 0.02 kSU. All three: about 0.33 kSU expected, 0.7 worst.
