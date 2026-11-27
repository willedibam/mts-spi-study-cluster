# Prereg: does Δz fall with slow-wave depth inside human NREM sleep, independent of the stage label? (MNI Open iEEG Atlas)

Frozen 27 November 2026, before any window-level z of this test was computed. Code: [`sleep/depth.py`](../sleep/depth.py), job [`jobs/sleep_depth.pbs`](../jobs/sleep_depth.pbs). Data: the frozen windows of the [sleep test](sleep-prereg.md) (job 185372579, `SHA256SUMS` checked before running), no new scoring.

## Why this test, and what it cannot settle

- The frozen sleep test moved: human N3 lowers Δz by −.0140 raw (−2.07 null SD), N2 by −.0086 ([result](../human-sleep-result.md)). Post hoc, a line fitted to the 30 macaque sessions, Δ = +.0011 − .0411 × (rise in delta fraction over the unit's own wake), predicts both human means to about 25%, and the macaque sleep sessions were shallow (rise +.11, against drugs +.26 to +.37, human N2 +.30, N3 +.47). Inferred reading: Δz tracks slow-wave depth, not drug identity.
- That reading rests on group means across species. If it is right, Δz should also fall with depth *within* sleep, window by window, and at equal depth the N2 and N3 labels should not differ. If instead Δz is set by something that differs between stages but not with depth, the window slope is flat and an N3 offset remains.
- Not settled here: whether a drug and natural sleep give the same Δz at equal depth in the same subjects. No dataset in hand has both (MNI has no drug arm, PRIOS no sleep). The macaque within-animal matching has been done on seen data (`findings.md`, "Not sleep depth": thin overlap; George's matched drug windows equal sleep, Chibi's ketamine does not) and is not repeated. A positive result here is necessary, not sufficient, for the depth reading.

## Already seen (not tests)

- Per-patient N2 and N3 Δ against wake, real and surrogate z means, state medians of delta fraction and mean |r|, the between-patient correlation of N3 Δ with the delta rise (−.01), the macaque line, and the N3 − N2 difference of patient means (−.0054 raw).
- Covariates only, read for this prereg (`C` and `rows`; no `Z`): within-state SD of window delta fraction, median over patients, N2 .086 (IQR .069–.104), N3 .057 (.037–.082); within-state r(delta, mean |r|) median +.57 (N2) and +.61 (N3); 52 of 65 patients have ≥ 5 N2 and ≥ 5 N3 windows inside the delta range both states reach; median N3 − N2 gap in window delta .162.

## Statistic

- Window value: net = z_real − mean of the 3 joint-IAAFT z, as frozen. Delta fraction: `prepare.delta_fraction` of the window (column 3 of `C`).
- Per patient, over its 10-min N2 and N3 windows: OLS of net on [1, N3 indicator, delta fraction]. β = slope on delta within state (z per unit delta fraction); γ = N3 − N2 offset at equal delta (raw z).
- Inclusion: ≥ 10 windows in each of N2 and N3 (64 patients expected; sub-098 has 3 N3 windows). γ is used only for patients with ≥ 5 windows of each state inside the common delta range.
- Patient is the unit: mean, t 90% CI, two-sided Monte Carlo sign-flip p (as frozen, 200,000 draws).

## Margins and verdicts

- β margin: .25 × .0411 = .0103 per unit delta fraction, a quarter of the macaque line's slope (the same quarter rule as the frozen test). γ margin: .0021 raw, the frozen raw margin.
- Order: UNINFORMATIVE if the 90% CI half-width exceeds the margin; GRADED (β) / OFFSET (γ) if the CI lies beyond −margin (REVERSED / opposite OFFSET beyond +margin); FLAT / NO OFFSET if the CI lies inside ±margin; otherwise INCONCLUSIVE.
- Stop rule: if the negative control's β is GRADED or REVERSED, the pipeline responds to depth by itself, and nothing below is interpreted.

## Predictions and what each outcome means

- Depth account: β < 0, GRADED; γ NO OFFSET; β × the delta gap reproduces the N3 − N2 difference (−.0411 × .162 ≈ −.0067 against −.0054 seen). The within-patient β need not equal the macaque between-session slope: window delta carries measurement noise and short-timescale fluctuation, which attenuate β, so only its sign and the margin decide.
- Readings, fixed now:
  - **A. Depth supported within sleep:** β GRADED, γ NO OFFSET, control FLAT. The cross-species depth reading survives a within-subject test it could have failed. Strength depends on the adjusted row (below).
  - **B. Stage, not depth:** β FLAT, γ OFFSET (N3 below N2). The macaque line's fit to human means is a coincidence of group means; the depth reading fails, and the human N3 effect needs a stage-specific account.
  - **C. Both:** β GRADED and γ OFFSET. Depth carries part; something tied to the N3 label carries the rest. Attenuation of β also biases γ toward an offset, so C is weak evidence against A unless the common-range γ (below) agrees.
  - **D. Neither resolved:** β FLAT and γ NO OFFSET, or INCONCLUSIVE rows. The N3 − N2 difference is not explained by the window model; no claim.
  - **E. Coupling strength rather than depth:** A or C on the primary, but the adjusted β FLAT. Depth and coupling strength are not separable here (within-state r ≈ +.6); the reading becomes "slow-wave depth or the coupling strength that comes with it".
  - **F. Reversed:** β REVERSED. Against the depth reading.

## Secondary rows (reported; verdicts by the same rule, not decisive)

- Adjusted: add mean |r|, SD |r|, participation ratio (`C[:, :3]`) and burst index, the frozen g5 covariates.
- Negative control: surrogate draw 1 as "real" against the mean of draws 2–3 (stop rule above).
- Route: the same model on real z alone and on mean surrogate z alone. If β comes from the surrogate rising rather than real z falling, a depth effect is the estimator's response to the window spectrum, not dependence beyond it; reported with the decomposition of the frozen test (N3: real −.0146, surrogate −.0006).
- Within N2 only and within N3 only (OLS of net on [1, delta]).
- Common range: the model refitted on windows inside the delta range both states reach (patients with ≥ 5 of each).
- Fisher z: reported by sign and p only (the margins are in raw units).
- Descriptive: β / −.0411; median per-patient delta gap; β × gap against the observed N3 − N2 difference.

## Limits stated before the data

- Within-state delta may also track artefact, spindle density or time in the night; none is modelled.
- 10-s windows; delta fraction is a spectral summary, not a slow-wave count.
- The same 64 patients as the frozen test, so this is a within-dataset check, not a replication.

## Compute

- One PBS job, 1 CPU, normal queue: expected about 0.0002 kSU (under 10 min), worst case 0.001 kSU (30-min walltime).
