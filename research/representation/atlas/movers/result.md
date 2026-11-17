# Result: whole-atlas movers beyond the spectral null, and their edge carriers (critic 1, 17 November 2026)

- **Rule:** [prereg.md](prereg.md), with Revision 1. The code and scorer were frozen and checksum-verified before scoring (`code.sha256`).
- **Gadi:** phase 1 job 184254543 (seconds) and phase 2 job 184254544 (21 SU, 13 min), both from snapshot `atlas-movers-b3127101`.
- **Outputs:**
  - gdata `/g/data/ql44/we2614/mts-spi-study/representation/atlas-movers-261117/`;
  - scores and figure in `results/representation/atlas-movers-261117/`.
- **Arm:** NeuroTycho 2-s windows. Primary agents KT, KTMD and PF (19 sessions, 4 animals). Every value is net of the pyspi stationary-Gaussian and bias null banks.

![movers](../../../../results/representation/atlas-movers-261117/movers-figure.png)

## How many features move, and is it real?

- **Usable coordinates:** 6,957. The filters require finite values in every window of every bank, both SPIs with reliability ≥ .3, and |z₀| < .98.
- **Movers:** 495 selected on even windows; **225 confirmed** on odd windows (105 rising).
- **Negative control:** the identical pipeline on eyes-open minus eyes-closed confirms 25. The estimated false-discovery proportion is **.11**, under the .2 stop.
- **The movers collapse onto a few SPIs.** Of the 225:
  - Kozachenko conditional entropy appears in 38;
  - partial correlation in 33 (Empirical) and 33 (MinCovDet);
  - high-band (25–50 Hz) coherence in 30;
  - Chebyshev distance in 24;
  - kernel information-theoretic SPIs, cointegration, barycentre and LCSS follow.
- **Only 11 movers are minimal pairs** (one assumption apart). The other 214 are compound.

## Agents and sleep (all 225 movers; change ÷ the mover's mean drug change)

| | KT | KTMD | PF | MD | Sleep |
|---|---|---|---|---|---|
| Median ratio | .6 | 1.3 | 1.3 | .42 | −.20 |

- **Medetomidine** shares the drug sign in 74% of movers.
- **Sleep moves the opposite way in 80%.**
  - Movers were selected on the drugs only, so sleep's sign was free to fall either way.
  - The features are correlated, so 80% is not 225 independent votes.
  - Sleep's sizes are small: the median signed change in the movers' direction is −.011.

## Broad against localised (calibrated drop test, 55 decomposed features)

| Verdict | Edges | Nodes |
|---|---|---|
| BROAD | 32 | 36 |
| PARTIAL | 13 | 14 |
| LOCALISED | 10 | 5 |

- The test removes the top-1% carrying edges, or the edges of the top 6 nodes, and recomputes the held-out net Δz. The result is compared with leverage-matched random drops.
- **Different movers use different edges.** The mean correlation between features' edge maps is .06.

## The minimal pairs that move (readings per `docs/interpreting-spi-pairs.md`)

| Pair (class) | Net change: KT, KTMD, PF / MD / sleep | Carriage (edges / nodes) | Reading |
|---|---|---|---|
| Kernel lagged MI against kernel TE (conditioning on the target's past) | +.07, +.14, +.12 / +.04 / −.05 | broad / broad | Conditioning on the target's own past changes the pair ranking less. Precedence structure becomes less tied to self-history, beyond a stationary Gaussian process with the same autocorrelation. The kernel estimators have no autocorrelation correction. |
| Chebyshev against L1, L2, Canberra and cosine distance (4 members converge) | −.04 to −.10 / −.03 / +.02 | broad / broad | Which pairs differ at their single most extreme sample stops matching which pairs differ on average. Extreme-sample (tail) dissimilarity becomes pair-specific. **Confound:** the null is Gaussian, so heavier channel tails (marginal shape) pass. Earlier calibration found that raw distance facets read marginal shape. |
| Gaussian against Kozachenko causally conditioned entropy | −.03, −.05, −.07 / −.03 / +.01 | broad / broad | A second Gaussian-against-model-free estimator pair, beside the MI pair, falling in the same direction. It converges with the headline. Kozachenko has no Theiler correction. |
| Covariance against partial correlation (mediation; Empirical and MinCovDet) | −.01/−.01, −.09/−.08, −.08/−.07 / −.06/−.05 / .00 | partial / **localised** | The pairs strongly coupled overall become less the pairs coupled directly given the other channels: more of the coupling is shared through other channels. Mediation is a construction-validated core axis, montage- and M-dependent. It is node-localised (dropping the top-6 nodes' edges keeps .74–.75 against random ~1), and the two estimators pick the same nodes (George 80, 67, 81, 60, 68; Chibi 49, 58, 26, 52, 54, 105). The precision-estimation floor is in the null, so this is beyond it. |
| Low against high band (`cohmag`, `plv`) | +.08, +.05, +.07 / .00 / −.04 | broad, partial / partial | Low- and high-band coupling rankings converge. Earlier null-decompose found eyes-open carrying 60–75% of this, so it is weakly drug-specific. |
| Joint entropy against MI (kernel; marginal entropies added) | +.05, +.05, +.03 / .00 / −.01 | broad / broad | Literal only. It reads channel-entropy spread. |

The MI anchor at 2 s has a net change of −.024 and is BROAD on both levels, as at 10 s.

## The localised movers

- **Regularised precision² against broadband coherence, PLV or PPC** (Shrunk, LedoitWolf, OAS): edge drop keeps .59–.74, against random .96–1.06; nodes BROAD.
  - The carrier edges recur, but eyes-open maps share them too (Jaccard .22–.24 against eyes-open .16–.17). They are structural, high-leverage edges where regularisation acts, not drug-specific edges.
  - Mostly Chibi and Kin2 (per animal .19–.40 and .43–.72; George .76–.93). Read as estimator-localised.
- **Partial correlation against Kozachenko or kernel SI** (largest net changes, −.14 to −.20; KTMD −.21 to −.24): nodes localised.
  - Top nodes lie in the high electrode labels in every animal: George 73–78 and 120; Kin2 73, 93, 105–117; Su 91–94, 104–120; Chibi 74–102.
  - Label numbering is per array and there are no coordinates here, so a shared anatomical locus is **unverified**. NeuroTycho electrode maps would decide it.

## Reading

- **Main.** Beyond the spectrum, anaesthesia moves ~225 dependence-agreement features (false-discovery estimate .11). They concentrate on a handful of SPI families:
  - direct (partial) dependence;
  - non-Gaussian, model-free information;
  - high-band phase coupling;
  - extreme-sample distance.
- **Carriage.** Most changes are carried broadly across the electrode pairs (32 of 55 BROAD by drop test). Different features use different edges.
- **Agents.** The signature is graded KTMD ≈ PF > KT > MD. Natural sleep moves the opposite way on most of it.
- **Localised exceptions.**
  - Mediation and partial-correlation pairs are node-localised, with the same nodes for two estimators.
  - The precision × phase family is edge-localised on structural edges shared with eyes-open.
- **System-wide reading.** The anaesthetic change is a system-wide reorganisation of which pairs carry which kind of dependence, not a few hotspots. The exception is a mediated-coupling change that sits on a few nodes per animal.

## Limits

- **The null.** The stationary Gaussian null (one draw per window) is looser than IAAFT. Rank, distance, cointegration and `xcorr_mean` movers may reflect marginal shape or slow drift.
- **Data.** 2-s arm only; 4 animals; 2 animals each for KT and PF.
- **Anatomy.** No anatomy without electrode maps.
- **Selection.** Movers are post-selection over ~7,000 features. The eyes-open estimate bounds the false-discovery proportion, but individual movers can still be false.
