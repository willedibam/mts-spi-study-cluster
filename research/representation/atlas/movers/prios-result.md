# Result: PRIOS full-pyspi screen beyond per-window joint IAAFT (19 November 2026)

- **Rule:** [prios-prereg.md](prios-prereg.md), with Deviation 1 (reliability aligns windows whose channel sets differ) and Deviation 2 (BH and Holm count an undefined p as 1; found after a faulty summary was seen, see the prereg). Readings per [`docs/interpreting-spi-pairs.md`](../../../../docs/interpreting-spi-pairs.md).
- **Data:** 7 patients, 3,911 S-full windows (2,897 awake, 1,014 propofol), 272 SPIs; outputs in [`results/representation/atlas-prios-screen-261119/`](../../../../results/representation/atlas-prios-screen-261119/).
- **Gadi:** farms 184616787 and 184616788 (384 cpus, 1:18 and 1:23; 2,058 SU), failed post 184616789 (9 SU), post 184625303 (22 SU), rescore 184626418 (< 1 SU): about 2.09 kSU against 2–2.6 expected.

## Coverage

- 34,453 of 36,856 coordinates finite; 3,408 usable after the reliability filter; **123 of 704 minimal pairs usable**.
- 187 of 272 SPIs have awake reliability below .3 (the macaque had far more reliable SPIs). Untestable here: 9 of the 11 macaque minimal movers, including Chebyshev and kernel lagged MI against TE, the two families that survived the macaque per-window IAAFT check.
- 20 usable minimal pairs have Δ identically 0 (both SPIs fixed by the covariance, which the surrogate keeps exactly).

## Primary: usable minimal pairs (BH over 123; awake split-half control)

**3 discoveries, 0 control discoveries (estimated false-discovery proportion 0).** All three are Gaussian against model-free information estimators, and all fall in 7/7 patients.

| Pair | Δ net [95%] | q | Δ real / surrogate | Double-centred Δ | Awake node R² | Macaque net (Gaussian bank) |
|---|---|---|---|---|---|---|
| Gaussian against kernel joint entropy | −.0102 [−.0134, −.0069] | .023 | +.016 / +.026 | +.000 | −.07, .05 | −.047 |
| Gaussian against kernel conditional entropy | −.0093 [−.0125, −.0061] | .023 | +.016 / +.026 | −.001 | .08, .17 | −.038 |
| Gaussian against Kozachenko causally conditioned entropy | −.0631 [−.0883, −.0379] | .035 | −.066 / −.003 | +.016 | .93, .80 | −.063 |

The next pairs have q ≥ .15.

**Readings.**
- **Joint and conditional entropy are the MI pair again, with a kernel estimator.** By construction, after normal-scoring every channel has the same set of values, so marginal entropies are equal across channels under any estimator, and H(X,Y) and H(Y|X) profiles are a constant minus the MI profile. These two rows therefore re-estimate "Gaussian against model-free MI agree less than the surrogate predicts" with a box-kernel estimator. They are one finding, not two, and not independent of the anchor. Unlike the KSG anchor (double-centred −.012), their double-centred change is ≈ 0. So by this estimator the change sits in the channel-effect component of the profiles, even though channel effects explain little of either profile. Confounds as listed: kernel bandwidth, autocorrelation bias.
- **Causally conditioned entropy is channel-level.** Both profiles are 80–93% node-additive, and the double-centred change has the opposite sign. It reads which channels have non-Gaussian dynamics given their own past, not pair dependence. The change is in the real data, not the surrogate. It is the PRIOS version of the macaque node-level reading (macaque node R² .76, .69), at the same size.

## Macaque predictions (one-sided, macaque direction)

- **Minimal movers (2 of 11 usable):** Gaussian against Kozachenko causally conditioned entropy, −.063, 7/7, Holm p = .0009 (predicted −; macaque −.063). Joint entropy against kernel MI, −.003, 3/7, Holm p = .64 (predicted +; not replicated).
- **All 225 movers (46 usable):** signed mean +.011, 83% in the macaque direction, exact sign-flip p = .016 (the smallest attainable is .0078).
- **Concordance with the macaque drug net:** minimal pairs r = .42 (77 coordinates), sign-flip p = .125; all coordinates r = .32 (2,883), p = .086. The nets use different nulls (Gaussian bank against per-window IAAFT).

## Anchor and secondary

- **MI anchor** (Gaussian against KSG MI): −.0132 [−.0239, −.0026], **7/7 negative**, p = .023 two-sided; real +.019, surrogate +.032 (one draw per window; the MI-pair S-full analysis with 10 draws gave −.0083, 6/7).
- **All usable coordinates:** 8 discoveries of 3,408, 0 control; 7 involve kernel joint entropy (against covariance-squared estimators and linear-model fits), the same Gaussian-against-kernel signal.
- **Minimal pairs without the reliability filter:** 40 of 595, 0 control. Mostly directed spectral measures (DTF, PDC, dcoh, gPDC: mean against max, broad or low against high band) and lag-pair correlations (adjacent against 10-sample lag), all rising in 7/7. Also: Gaussian against kernel cross-map entropy (−.013, 7/7), unconstrained against Sakoe–Chiba DTW (−.029, 7/7), covariance against MinCovDet partial correlation (−.011, 7/7; macaque sign). These SPIs are unreliable at 2 s in this data, so these are descriptive: a changing profile of an unreliable SPI can reflect a change in its noise.

## What this establishes

- Across species and against a per-window spectrum- and cross-spectrum-matched null, the Gaussian against model-free information family is the robust change under propofol: the KSG anchor, a kernel re-estimate of it, and a channel-level conditional-entropy counterpart predicted from the macaque at the same size. Its interpretation remains that of the [MI pair](../findings.md).
- No new pair-level character beyond that family passes in humans. Most macaque candidates could not be tested, because only 85 SPIs are reliable on these 2-s human windows.
- Limits: 7 patients with epilepsy; awake ward days against propofol in theatre on a later day; one surrogate draw per window; Deviation 2 was found after a faulty summary was seen.
- The matched macaque screen ([matched-result.md](matched-result.md)) made no frozen claims: its eyes-open control was a disjoint real change. Post hoc, its 21 PRIOS-usable discoveries agree in direction in every patient (p = .0078).
