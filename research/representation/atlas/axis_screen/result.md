# Axis screen: result

Rules: [prereg.md](prereg.md) (frozen 19 November; addendum 20 November before PRIOS compute). Code: [run.py](run.py), [score.py](score.py), [prios_score.py](prios_score.py), [posthoc.py](posthoc.py). Outputs: `results/representation/atlas-axis-screen-261119/` (`score-long.json`, `score-short.json`, `score-prios.json`, `posthoc-*.json`; the screen npz files are on Gadi scratch, `atlas/axis_screen/`). Reading rules: [docs/interpreting-spi-pairs.md](../../../../docs/interpreting-spi-pairs.md). Everything below is confirmatory under the frozen rules unless marked **post hoc**.

## What was run

33 pyspi-free SPIs (validated against pyspi on two 10-s windows: profile r ≥ .9992, per-window z within .005), 528 pairs plus KSG MI against each on the anchor's 32-channel subset; every window of the 10-s arm (primary) and the 2-s arm (secondary), all channels; per window 3 spectrum-final joint IAAFT draws and stationary, zero-phase and bias Gaussian nulls. Families: F1 (a higher-order member; real − IAAFT), F2 (linear lag pair; real − zero-phase), F3 (linear; real − stationary). Session Δ = drug − awake eyes-closed (23 sessions, 4 animals); screen BH q ≤ .05 within family with every animal mean the same sign; axis gates: animal t(3) p < .05 with 4/4 same sign, |Δ| ≥ 3 split-half null SD, split-half ≤ .25 Δ, eyes-open ≤ .33 Δ, covariate-adjusted ≥ .5 Δ with 4/4 sign, disattenuated 4/4 sign. Compute: 10-s arm 184 SU, 2-s arm 26 SU (plus 1 SU failed start: the 2-s manifest had no delta fraction), PRIOS 21 SU; about 0.23 kSU in all.

Calibration: the MI anchor (Gaussian against KSG MI, 32 channels, F1) passes every gate in both arms (10 s: Δ −.013, t(3) −4.4, 4.4 null SD; 2 s: Δ −.019, t(3) −5.2, 3.6 null SD), so the rules are about as strict as the anchor's evidence and not stricter.

Screen counts (usable / screened / axes): 10 s F1 292/113/20, F2 61/22/0, F3 144/70/0; 2 s F1 213/79/10, F2 25/3/0, F3 88/37/1.

## Ranked axes (10-s arm unless stated)

Ratios are of the drug Δ. Sleep is SLP sleep − awake (7 sessions, 2 animals; deep = top delta tertile). Recovery is recovery − awake (ratio near 0 = full reversal). rMI = within-drug-session window-level r of the axis net with the MI-pair net.

| # | Axis (class) | Δz | Animals, t(3) | Null SD × | Split-half / eyes-open | Covariate-adj. / disattenuated | Sleep (deep) / recovery | Tier | Main confound |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `pcorr|pcorr_mcd`: classical against MinCovDet partial correlation (Empirical vs MinCovDet) | −.079 | 4/4, −10.7 | 11.4 | .04 / .05 | .87 / −.156 (4/4) | .05 (.13) / .02 | beyond linear | MCD noise rises with spectral concentration (IAAFT moves −.050 of the real −.129); precision at M 126, T 1000 |
| 2 | `r2|mcd2`: classical against MinCovDet covariance, squared (Empirical vs MinCovDet) | −.056 | 4/4, −7.5 | 6.2 | .07 / −.07 | .92 / −.042 (4/4) | .14 (.25) / .06 | beyond linear | as 1; IAAFT −.031 of real −.086 |
| 3 | `mi_g|ksg`: the MI anchor (Gaussian vs KSG) | −.013 | 4/4, −4.4 | 4.4 | −.03 / −.03 | .96 / −.015 (4/4) | .06 (.13) / .22 | beyond linear | ceiling-near (z0 .96); spectral concentration (KSG) |
| 4 | `l1|linf` (2-s arm only): Chebyshev against L1 distance (chebyshev vs cityblock) | −.049 | 4/4, −10.3 | 4.7 | .07 / −.06 | .96 / −.039 (4/4) | .10 (.18) / −.01 | beyond linear | window-length dependent (fails at 10 s); one-sample maximum is noisy |

PRIOS per row (one-sided, 7 patients; details below): 1 below the PRIOS reliability floor, descriptively −.033, 6/7, p .003; 2 −.004, 4/7, p .15 (sibling `tau2|mcd2` −.011, 6/7, p .014); 3 −.008, 6/7, p .017; 4 −.025, 7/7, p .005 (`linf` reliability .25). None survives Holm alone; the frozen aggregate test A passes (p = .023).

Same-axis rows (they pass all gates but add no new character):

- **MinCovDet family** (rows 1–2): `r|mcd` (−.025, ceiling z0 .98), `rho2|mcd2` (−.051), `tau|mcd` (−.021), `tau2|mcd2` (−.052) (MinCovDet against rank estimators: the same robust-against-classical contrast with ranks in place of Pearson; rank and Pearson agree at z0 ≈ 1), `mcd|l1` (+.020; L1 on z-scored data tracks r), and the non-minimal `mcd2|dcorr` (−.051) and `mcd2|xc2max` (−.051), each carried by its `mcd2|r2` link (−.056) with the other link near 0 (`r2|dcorr` −.002, `r2|xc2max` −.003). `mcd2|ksg` (−.061) is the sum of the MCD and MI axes (rMI .21).
- **MI family** (row 3): `r2|ksg`, `rho2|ksg`, `dcorr|ksg`, `xc2max|ksg` (Δ −.010 to −.013, rMI .50–.84). r² and Gaussian MI are monotone in |r|, so these are the anchor.
- **Minor, non-minimal:** `tau2|pcorr` (−.005) and `pcorr|l1` (+.005): tiny, eyes-open .27–.28 of Δ (near the limit), and the mediation contrast (marginal against partial) mixed with an estimator contrast; no reading earned.

Detail for the MinCovDet axis (rows 1–2):

- Per agent (row 2): KT −.062, KTMD −.045, MD −.053, PF −.083; row 1: −.048, −.091, −.035, −.120. Every agent, including ketamine.
- Distinct from the MI pair: rMI .21–.28; node-additive structure removed (double-centred Δ equals Δ: −.043, −.079).
- Mean-profile Δ is smaller than the per-window Δ (−.033 against −.056; −.024 against −.079): much of the effect is window-specific (which samples MCD discards differs between windows) rather than one stable edge pattern. That fits "a minority of samples carries a different dependence pattern" and does not by itself say "episodic coupling" (the guide's stronger reading needs the support mask; not done).
- **Post hoc**, reliability: MinCovDet edge-profile reliability falls beyond IAAFT under the drug (net −.025 to −.031 for `mcd`, `mcd2`, `pcorr_mcd`; 4/4 animals, t(3) −6 to −12), the only statistics where it does. The disattenuated Δ keeps its sign in 4/4 animals, so the axis is not only a reliability change.
- 2-s arm: absent (`r2|mcd2` not screened; `rho2|mcd2` −.003). At T 200 and M 126 MinCovDet keeps about 163 of 200 samples and is close to classical, so this is a power limit of the estimator, not evidence against the axis.
- Reading (guide's worked example): time-heterogeneous dependence, a minority of time samples carrying a conditional-dependence pattern that departs from the bulk, more so (less agreement between robust and classical) under every drug, beyond what the IAAFT surrogate with the same spectra and covariance gives. Sleep moves it .05–.14 as far.

Detail for `l1|linf` (row 4, 2-s arm): real Δ −.048 against IAAFT +.001, so the change is entirely beyond linear. At 10 s it is screened (Δ −.044, t(3) −6.1) but fails size (1.7 null SD), eyes-open (.35 of Δ) and covariates, and sleep moves it .62 as far: the 10-s maximum over 1,000 samples is dominated by a few transients. The class has no entry in `assumptions.READINGS`; by the two neighbouring distance classes the reading is "how far each pair's largest disagreement departs from its typical disagreement" (the tail of x − y, on normal-scored channels so not marginal shape). Rank 4 because it holds only at one window length.

## MinCovDet mechanism check (post hoc)

Criteria stated and committed before the run. Code: [mechanism.py](mechanism.py); 10-s arm, every window; real against the screen's 3 IAAFT draws (same seeds), net = real − mean IAAFT; animal-level t(3) on the 4 animal means, as in the gates. sklearn's MinCovDet covariance is the classical covariance of its reweighted support, so the whole classical–robust difference is carried by the samples outside it. The guide earns "episodic coupling" only if those samples are temporally clustered, involve several channels and carry a distinct network. Each property is measured on the top 5% of samples by robust distance (50 per window, the same count in real and IAAFT, so estimation noise matches):

- **Clustered:** Fano factor of top-5% counts in 0.5-s bins (about 1 for independent samples; IAAFT carries the clustering that the spectrum alone gives). Holds if the drug-window net is > 0 in 4/4 animals with t(3) p < .05.
- **Several channels:** node participation ratio of the outlying network d = corr(top 5%) − corr(rest). Holds if it is ≥ 5 channels in drug windows (not a one- or two-channel artefact); the number of channels beyond |x| = 2 at those samples is reported.
- **Distinct network:** r_net, the correlation of d's two parts (top-5% against rest edge profiles), has a drug-window net < 0 in 4/4 animals with t(3) p < .05 (more distinct than selection by distance gives in IAAFT), and the network recurs: the between-window correlation of d within a session has a net > 0 in 4/4 animals.
- **Link to the axis:** the clustering and distinctness nets also move with the drug (drug − awake eyes-closed: Fano net up, r_net net down, 4/4 animals); the window-level correlation of the axis net with each metric net within drug sessions is reported.

Outcomes: all hold, "episodic coupling" is earned for the drug change; the drug-state levels hold but the change does not, the outlying samples are episodic in both states and the axis is not explained by more episodes; clustering fails, the reading stays "time-heterogeneous dependence".

## What failed

- **Best lag against zero lag** (`r|xcmax`, F2): Δ −.103, t(3) −7.1, 4.6 null SD, split-half .01, but the spectral-covariate-adjusted Δ reverses sign (−.59 of Δ) and eyes-open is .30: a spectral-shape effect, not lag. `r2|xc2max` (F2 −.016) fails covariates the same way; F3 (beyond stationary) it is ≈ 0.
- **Propofol-only high-band lagged phase** (`plv_hi|dswpli_hi`): F2 Δ +.105, 7.1 null SD, but `dswpli_hi` reliability is .03 (disattenuated Δ −3.0, meaningless) and the covariate-adjusted Δ is −.06 of Δ. Per agent KT +.035, KTMD +.116, MD +.073, PF +.175: not propofol-only at the screen level, and not an axis.
- **`ppc|dswpli` beyond the stationary null** (F3): Δ +.075, t(3) 4.4, 3.9 null SD; fails covariates (adjusted .01 of Δ) and reliability (`dswpli` .11). KTMD +.094 and PF +.129 against KT +.012 and MD +.033, as the earlier candidate said, but the spectral covariates absorb it. F2: fails animal t and eyes-open (.50).
- **Frequency locus** (`coh_lo|coh_hi`, band low against high): Δ −.013, t(3) −1.6, fails every gate; `r2|coh_lo` likewise. The 2-s `coh_hi|coh_bb` (F3 +.036, passes) is broad against high band ("mostly band power" in the class table) with MD in the opposite direction (−.008); not read.
- **Edge-profile reliability falling under every drug** (**post hoc**): real reliability falls under every agent for almost every statistic (10 s: r −.087; KT smallest, −.008), but the IAAFT surrogate's reliability falls by the same amount for every linear and rank statistic (net 0 to −.008). It is what the drug's spectra and covariance give (lower effective sample size), not a character change. Only MinCovDet falls beyond IAAFT (above). Envelope correlation falls less than its surrogate (net +.05).
- **Imaginary coherence and coherence/PLV/PPC lag pairs** (F2, F3; e.g. `coh_hi|icoh_bb` +.13 to +.15): large and 4/4 at the screen, but the spectral covariates reverse them (adjusted −.6 to −.9 of Δ).
- **Chebyshev against L2** (`l2|linf`, 10 s): Δ −.034, 1.5 null SD, eyes-open .42, sleep .66; fails. Against r² and MI (`r2|linf`, `linf|mi_g`, `linf|ksg`): not significant at the animal level.
- **Ceiling and estimator pairs:** `r|rho` (−.0007), `rho|tau` (−.0002, 7 null SD but z0 1.00 and reliability fails), `r2|dcorr` (−.002, animal t fails), `l1|l2` (−.002, z0 1.00): raw changes too small to read.
- **Envelope linear against log** (`pec|pec_log`): fails; `pec|coh_bb` fails.
- **F1 axes invalidated post hoc by the null** (pass every frozen gate, not read): `dcorr|icoh_bb`, `tau2|icoh_bb`, `l1|icoh_bb` (10 s) and `mcd|xcmax`, `rho|xcmax`, `tau|xcmax`, `l1|xcmax`, `linf|xcmax`, `xc2max|ksg` (2 s). **Post hoc:** spectrum-final IAAFT does not reproduce multitaper or best-lag statistics (DPSS tapers mix neighbouring Fourier coefficients whose common phase IAAFT randomises; zero-padded lagged correlation is not fixed by the circular periodogram). Linear–linear pairs among them move by up to .08 under real − IAAFT with the drug (median .011; r, r², Gaussian MI, partial correlation and L2 against each other: 0). The 2-s xcorr rows are carried by `r|xcmax`'s IAAFT net (−.037), with `r|mcd` ≈ 0. The prereg's statement that linear pairs are degenerate under IAAFT was wrong for these statistics; the guide already says IAAFT reproduces spectral and lag families poorly.

## PRIOS test (frozen; human propofol ECoG, 2-s epochs, 7 patients)

Run as frozen in the prereg and the 20 November addendum (real and 3 IAAFT draws per epoch on its own channels; 21 SU). Of the 20 macaque axes, 14 are usable in PRIOS; 6 fail the PRIOS awake reliability floor of .3 (`pcorr|pcorr_mcd` .29/.20, `tau2|pcorr`, `pcorr|l1`, and the three `icoh_bb` pairs, `icoh_bb` .03).

- **A (primary): passes.** Mean sign-aligned patient Δ over the 14 axes +.0079, exact sign-flip p = .023 (3 of 128 patterns); 6/7 patients positive (PRIOS04 −.002). Awake split-half control: +.0024, p = .12, about 30% of the effect, concentrated in PRIOS03 (control .013 against effect .016).
- **A-valid (secondary, addendum):** 12 axes, +.0071, p = .031; control +.0023, p = .15.
- **B (per axis, Holm over 14):** none survives. Best: `mcd2|ksg` −.016, 7/7, one-sided p .004, Holm .053; `tau2|mcd2` −.011, 6/7, p .014; `mi_g|ksg` −.008, 6/7, p .017; `rho2|ksg`, `r2|ksg`, `dcorr|ksg` p .015–.022; `r2|mcd2` −.004, 4/7, p .15; `r|mcd`, `mcd|l1` ≈ 0.
- **`l1|linf` (secondary 2-s prediction, addendum):** Δ −.025, 7/7 patients, t 3.67, one-sided p = .005, 4.3 split-half SD. Caveat: `linf`'s PRIOS awake reliability is .25, under the usability floor that the main set uses (the addendum did not restate the floor for this test).
- **Descriptive, unusable:** `pcorr|pcorr_mcd` Δ −.033, 6/7 (PRIOS01 +.001), t 4.2, 11 split-half SD, below the reliability floor (precision at T 200 is noisy).
- **How the net moves in PRIOS:** real z barely changes or rises under propofol while the IAAFT z rises more (e.g. `mi_g|ksg` real +.022, IAAFT +.030). In the macaque at 10 s the real z falls and the surrogate moves little. Both are a fall relative to the linear expectation, but in PRIOS the contrast is carried by what the surrogate predicts from the propofol spectrum, so the human result is a weaker form of the claim.
- Size: PRIOS Δ are a fifth to a tenth of the macaque 10-s Δ, as the macaque 2-s arm predicts for the MinCovDet rows (2-s macaque `rho2|mcd2` −.003). Transfer of the set is supported in aggregate; no single axis is confirmed in humans at the corrected level.

## Freeze list (macaque predictions for further human tests)

Signs are drug − awake. For any new human dataset (or longer PRIOS epochs, if they can be had between stimulations), freeze these before computing, each with the same F1 net (real − mean of 3 spectrum-final IAAFT draws on normal-scored windows) and the usability floor:

1. `pcorr|pcorr_mcd` (−) and `r2|mcd2` (−): robust against classical (MinCovDet). Needs T well above M (10-s windows at M ≈ 126; at T 200 only with M ≲ 60, and even then near the reliability floor).
2. `mi_g|ksg` (−): the MI anchor, with its siblings `r2|ksg`, `dcorr|ksg` as the same axis.
3. `l1|linf` (−) at 2-s windows only: Chebyshev against L1.
4. `mcd2|ksg` (−) as the combined MCD + MI readout (strongest single PRIOS row, 7/7).

Not to freeze: anything with a multitaper or best-lag member under an IAAFT null; lag, phase-lag and frequency-locus pairs (all fail the spectral covariates); reliability falls (reproduced by IAAFT).

## Limits

- Four animals; session-level screen with an animal-level gate. Drug and calendar are confounded within archives; eyes-open, split-half, sleep and recovery are the specificity checks.
- 33 of the atlas's 272 SPIs, chosen for readable minimal pairs; pyspi-free implementations validated on two windows.
- The MinCovDet mechanism check is a separate post hoc section above.
- The prereg's IAAFT degeneracy assumption failed for multitaper and xcorr statistics (above); the frozen gates were applied unchanged, and the affected axes are set aside by a post hoc rule stated before PRIOS.
