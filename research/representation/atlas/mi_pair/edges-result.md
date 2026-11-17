# Result: MI-pair recompute and edge-level concentration (critic 1, 17 November 2026)

- **Rule:** [edges-prereg.md](edges-prereg.md), with Deviation 1.
- **Code:** `edges.py`, plus the frozen scorer `edges_score.py`, checksum-verified before scoring.
- **Gadi jobs:**
  - recompute 184230163: 60 SU;
  - edge arm 184242002: 894 SU;
  - first edge array 184230192: failed on save, about 850 SU lost.
- **Outputs:**
  - gdata `/g/data/ql44/we2614/mts-spi-study/representation/atlas-mi-pair-261117/{zns,full}/`, with checksums;
  - scores in `results/representation/atlas-mi-pair-261117/`.

## Part R: the lost arrays are recomputed and match

Same seeds and invocations. Every value matches the reported numbers to the printed digit (`zns/sumall.txt`):

| | KT | KTMD | MD | PF | Sleep |
|---|---|---|---|---|---|
| Normal-scored change | −.0159 | −.0155 | −.0062 | −.0117 | −.0010 |
| Sessions down | 4/4 | 11/11 | 4/4 | 4/4 | 7/7 |

The eyes-open SD is .0020, so the gate passes. The arrays now also carry each edge's Gaussian and kNN MI.

## Part E: all channels, edges labelled by electrode

Primary agents KT, KTMD and PF; 19 sessions; 4 animals.

| Decision | Frozen verdict | Numbers |
|---|---|---|
| **E1. Concentration** of held-out excess kNN MI (beyond 3-draw joint IAAFT) at matched wake strength | **CONCENTRATED**, at the margin | Top 1% of edges carries a median of .054 of the positive held-out excess (Chibi .059, Su .057, Kin2 .051, George .050). Eyes-open: .011–.018. Uniform: .01. 18/19 sessions exceed their eyes-open. |
| **E2. Same edges across sessions** within an animal | **NOT REPRODUCIBLE** (rule: drug ρ − eyes-open ρ ≥ .10 in every animal) | Drug ρ against eyes-open ρ: Su .19/.04, Kin2 .16/.02, George .13/.03, Chibi .11/.02. George (.094) and Chibi (.090) miss the margin. |
| **E3. Do the edges driving the z fall gain?** | **mixed**: neither prestated outcome | Fraction of the bottom-1% Δp edges with ẽ above the median: .46–.59. ρ(Δp, ẽ) is −.08 to −.01. |

**Descriptive.**
- *Positive tail top-1% share (C), median by agent:*
  - KT .053, KTMD .053, PF .061;
  - MD .026, against eyes-open .014;
  - sleep .010, against .015, so at control.
- *Losses are not concentrated:* the negative-tail share is .014–.015, at the eyes-open level, except KT at .025.
- *Recurrence (E2b):* top-1% edges recur between sessions within an animal (Jaccard .12–.31, against eyes-open .004–.034). Sessions here mix agents within George and Chibi.
- *Label distance:* the top-1% edges sit at a median of 12–14 apart in electrode label order, against 38 for all edges. Label order is only a proxy for adjacency.
- *No node or hub claim:* the top-5 node share is about .20–.22 under the drugs and .20 in sleep, with no eyes-open comparison.

## Post hoc checks (`edges_posthoc.py`; written after scoring)

- **Strength.** Matching was on wake strength only, and the drugs change coupling strength.
  - Binning on wake × drug-state Gaussian MI (10 × 10) leaves C at .042–.054 against eyes-open .010–.017, and within-animal ρ at .09–.17 against .02–.04. So the concentration is not mainly excess following pairs that strengthen.
  - The top-1% edges still pile into the strongest wake decile (34–58% of them). ẽ tracks Δg only weakly (ρ .11–.19).
- **Surrogate averaging.** Using the three-draw mean surrogate profile biases z_res with no consistent sign. With one draw's noise restored (k_s + N(0, ksd²)), the all-channel z_res, drug − wake, is:
  - KT −.016 (4/4), KTMD −.014 (11/11), PF −.011 (4/4);
  - MD −.005 (4/4), sleep −.001 (7/7);
  - eyes-open session SD .0011.

  So the z fall holds against the stricter all-channel joint null. This is one noise realisation and descriptive.

## Reading

- **What holds.** Under KT, KTMD and PF, the model-free dependence that pairs gain beyond the linear-Gaussian null is modestly concentrated:
  - about 5% of it sits in the top 1% of edges, 4–5× eyes-open;
  - it is mostly among the most strongly coupled, short-label-distance pairs;
  - these edges recur within an animal across sessions and agents more than eyes-open maps do;
  - losses are diffuse; medetomidine is weaker, and sleep is at control.
- **What does not hold.**
  - "Massively concentrated": about 95% of the held-out excess lies outside the top 1%.
  - A reproducible locus by the frozen bar.
  - That the z fall is carried by the excess-gaining edges. The edgewise contributions to Δz are unrelated to ẽ, so the z fall is spread across many edges, not localised.
- **The one-animal hub result does not generalise under this measure.** Its "hub edges fail to gain" is not the pattern across animals.
- **Claim this supports:** "Anaesthetics, not sleep, make model-free dependence beyond the linear-Gaussian expectation more uneven across pairs. The gain is enriched among strongly coupled pairs that recur within an animal across sessions, while the agreement change itself is broad."
