# Result: macaque full-pyspi screen against per-window joint IAAFT, and its test on PRIOS (19 November 2026)

- **Rule:** [matched-prereg.md](matched-prereg.md). Readings per [`docs/interpreting-spi-pairs.md`](../../../../docs/interpreting-spi-pairs.md) and `assumptions.READINGS`.
- **Data:** 1,260 NeuroTycho 2-s windows; primary KT, KTMD and PF, 19 sessions in 4 animals; 272 SPIs. Outputs in [`results/representation/atlas-matched-screen-261119/`](../../../../results/representation/atlas-matched-screen-261119/).
- **Gadi:**
  - bank 184621750;
  - farms 184621752 and 184621754 (432 cpus, 2:20 and 2:17);
  - post 184626419 (10 min).
  - About 4.0 kSU in all, against 4.2 expected.

## Frozen outcome: no claims

- **Primary screen:** 164 usable minimal pairs, 55 discoveries (BH q ≤ .05, every animal agreeing in sign). The eyes-open negative control gives 26, so the false-discovery estimate is .47 > .2. **By the rule, the macaque screen makes no claims, gives no prediction set, and PRIOS tests A and B are not run.**
- **PRIOS test C (concordance):**
  - minimal pairs: r = .55 over 67 coordinates, sign-flip p = .22, 69% same sign;
  - all coordinates: r = .35 over 2,858, p = .30.
- **MI anchor** (Gaussian against KSG MI): −.0278, 19/19.
- **All usable coordinates:** 2,519 discoveries of 6,679, control 678.

## Post hoc (everything below was decided after the output was read)

**The control is a real eyes-open change on a different set of SPIs, not a proxy for false drug discoveries** ([`matched_posthoc.py`](matched_posthoc.py), `posthoc/controls.txt`).
- 19 of the 26 control discoveries are rank-correlation lag pairs (lag 0 or 1 against lags 2–5, +.010 to +.019). Drugs leave those pairs unmoved (q ≈ 1).
- Control and drug discoveries share one pair.
- A within-state control finds 0 of 164. It takes awake eyes-closed windows, second half against first half in time, which is the PRIOS control construction.
  - Its 5 against 5 windows make it less powerful than the 10 against 10 drug test.
- The prereg called the eyes-open control conservative; it did not anticipate a disjoint real effect counting as false discoveries. That was a design error in the rule, not something learned from the drug contrast.

**PRIOS tests A and B, run post hoc with the 55 as the prediction set** (`posthoc/prios-cross-posthoc.json`).
- 21 of the 55 are usable in PRIOS.
- **A:** sign-aligned mean +.0087, 81% in the macaque direction, sign-flip p = .0078, the smallest attainable with 7 patients (every patient agrees).
- **B (Holm over 21):**
  - Gaussian against kernel conditional entropy p = .004 (7/7);
  - Gaussian against Kozachenko causally conditioned entropy p = .009 (7/7);
  - the MI anchor is 7/7 but Holm p = .21.
- The chance-level control result and the eyes-open composition make the 55 a plausible prediction set, but A is not the frozen test.

**Breadth and a common driver** ([`matched_drivers.py`](matched_drivers.py), `posthoc/drivers.txt`; one surrogate draw, exploratory).
- **Breadth:** with 5 drug windows against the first awake half (power-matched to the split-half control), 2,218 of 6,679 usable coordinates move beyond the surrogate; the split-half control moves 0. The departure from the linear-Gaussian-stationary null under these drugs is pervasive, not specific to MI.
- **Local amplitude non-stationarity:** the envelope CV over 8 blocks of 25 samples is .246 in real awake windows against .218 in their surrogates, and .359 against .302 under drug. So the excess beyond the surrogate doubles (.028 → .057). The excess shared co-fluctuation of channel envelopes barely moves (.086 → .096).
- **Within drug windows,** after removing each session's mean:
  - the MI and Chebyshev nets covary (r = +.55; awake +.27);
  - both track the window's envelope-CV excess (MI r = −.51, Chebyshev −.43; awake −.16, −.33).
- **Adjusting the drug contrast** by the within-state slope removes 18% of MI (−.0278 → −.0228, 19/19, animal t −4.5), 27% of Chebyshev (−.0666 → −.0483, 19/19, t −13.5) and 15% of CCE.
  - The index is noisy, so regression dilution makes this an under-correction: it bounds the gain explanation, it does not exclude it.
- **Reading:** the residuals travel with local, not shared, amplitude non-stationarity. That is the false-positive class the surrogate-data literature names, and it fits the propofol amplitude-linked decoupling candidate. A crude linear adjustment does not remove the drug effect. A null that preserves each window's local envelopes is the decisive test.

## The macaque discoveries, by family

Δ is the net change (drug − awake); "real/surr" splits it into the real and surrogate changes; dc is the double-centred (pair-level) Δ.

| Family | Δ net | Sessions | t(3) | dc | Eyes-open | Real / surr | Usable in PRIOS |
|---|---|---|---|---|---|---|---|
| Gaussian against model-free MI (KSG, KSG-DCE, kernel; and lag-1 MI) | −.028 to −.048 | 19/19 | −4.2 to −6.3 | −.027 to −.038 | ≈ 0 | real-side (MI-DCE −.050 / −.022) | MI yes, lag-1 MI no |
| Chebyshev against L1, L2, cosine, Canberra, Bray–Curtis | −.037 to −.067 | 18–19/19 | −8.8 to −11.5 | −.030 to −.051 | ≈ 0 | real-side (Canberra −.075 / −.009) | no |
| Gaussian against Kozachenko causally conditioned entropy | −.052 | 19/19 | −3.6 | +.010 | −.003 | surrogate-side (−.014 / +.038) | yes |
| Best lag against zero lag correlation | −.017, −.037 | 19/19, 18/19 | −11.1, −8.6 | −.016, −.037 | ≈ 0 | both fall (−.219 / −.182) | yes |
| Low against high band coherence and PPC | −.030, −.034 | 18–19/19 | −7.0, −8.5 | −.008, −.009 | −.008, −.009 | surrogate-side | no |
| DTW barycentre and DTW options, elastic against rigid | +.06 to −.04 | 15–19/19 | up to +20 | mixed | up to +.018 | surrogate-side | no |
| Small estimator diagnostics (dcorr bias, Bray–Curtis, lagged Pearson, Johansen) | |·| ≤ .006 | 13–19/19 | | | | | partly |

Specificity contrasts against awake eyes-closed:
- **MI family:** MD −.014 to −.025, sleep −.002 to −.008, recovery −.002 to −.008.
- **Chebyshev:** MD −.011 to −.023, sleep −.002 to −.005, recovery ≈ 0.

**Readings.**
- **The MI family is the anchor result again under this construction, at pair level.** dc ≈ net and node R² ≈ .14. This is unlike the PRIOS kernel entropy rows, which carried no pair-level change.
  - It reads as nonlinear dependence in the operational sense: dependence beyond the linear-Gaussian expectation, whatever produces it. After normal-scoring, I − G is the departure from the covariance-matched Gaussian, and a residual beyond a linear-Gaussian-stationary surrogate is evidence of nonlinearity in the surrogate-data sense (Theiler et al. 1992; Schreiber and Schmitz 2000).
  - The drug reorganises which pairs carry that departure; z does not measure its mean amount.
  - That literature names non-stationarity as the main false positive. Here the relevant case is gain or amplitude change within a 2-s window, which the per-window surrogate does not remove.
  - A curved coupling function is neither claimed nor needed.
- **Chebyshev is the strongest pair-level family** (largest t(3), drug-specific, real-side).
  - Literal: after normal-scoring, which pairs have the largest single-sample divergence is predicted less well by which pairs differ most on average, beyond a process with each window's exact spectra and cross-spectra.
  - Character candidate (not validated): intermittent large divergence of a pair becomes pair-specific. That fits the propofol amplitude-linked decoupling candidate in [findings](../findings.md).
  - Confounds: the one-sample maximum is noisy; transients.
  - It cannot be tested in PRIOS (unreliable at 2 s there).
- **Causally conditioned entropy is a channel-level result in both species.**
  - pyspi `cce_n-5` is the sum over i = 1..5 of H(Y_t | the target's past i − 1 samples, the source's past i samples).
  - Its profiles are 70–93% node-additive, so the edge z is mostly agreement, across target channels, between linear-Gaussian and model-free estimates of short-horizon predictive uncertainty.
  - With normal-scored marginals, the fall beyond IAAFT reads as anaesthesia reorganising which channels' dynamics depart from the linear-Gaussian expectation (univariate nonlinearity in the surrogate-data sense, with the same non-stationarity caveat).
  - Decomposition: surrogate-side here (the surrogate's z rises), real-side in PRIOS. The equal net size (−.052 here, −.063 in the old macaque screen and in PRIOS) is the predicted quantity, but its parts differ between species.
  - A per-channel statistic could state this without pairs, so it is not evidence for what z offers beyond comparators.
- **Best lag against zero lag:** a small net residual of two large, nearly equal changes in the real data and the surrogate, so it is near the limit of one surrogate draw. Not read.
- **Band locus and DTW** move in eyes-open or are surrogate-driven, and DTW warp failed construction validation. Not read.

## What this establishes

- Under one construction (normal-scored, per-window joint IAAFT), two pair-level, drug-specific families stand out in the macaque: Gaussian against model-free MI and Chebyshev against average distances.
- The frozen rule withholds claims because its control was mis-specified. Post hoc, the macaque discoveries predict the PRIOS direction (81%, every patient). Of the testable pairs, only the information-estimator family passes individually.
- Chebyshev needs a human dataset where it is reliable: longer windows or more channels.
