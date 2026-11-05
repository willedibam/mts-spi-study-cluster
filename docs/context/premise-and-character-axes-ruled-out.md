# Premise and character axes: ruled out

Approaches tried and not adopted, with the evidence. Linked from [premise-and-character-axes.md](premise-and-character-axes.md).

## Character estimators

**Envelope axis, moment-identity form (prototype, 6 November 2026).** For one shared waveform per frequency plus independent circular noise, cov(|X|², |Y|²) = |E X²Y*²| − |E XY*|², for any source non-Gaussianity and any zero-lag mixing. E = 1 − Σ_f(|M4| − |M2|²)/Σ_f cov is then 0 under volume conduction of a bursting source; the Gaussian reference ([Nolte et al. 2020](https://www.frontiersin.org/articles/10.3389/fnins.2020.577574)) reads 0.87 there. The identity was checked numerically; I found no prior statement of it in one search.
  - *Holds at T = 2^15* (quarter taper coefficients, normal scores, delay removed): volume conduction is about 0, normal scores do not break it, a constant lagged coupling gives 0, and a shared-amplitude/independent-phase knob is tracked (0.71–0.77 at θ = 0.8).
  - *Fails on selectivity:* time-varying and nonlinear waveform coupling violate the identity and read negative. Linearity θ = 0.6 gives −0.45, and the transfer rectifier gives −0.2 to −0.4. On/off gives −θ uncorrected; a correction from the local coherencies (E|c|²/|Ec|², E|c|²/|Ec²|) fixes on/off (−0.05) but not drift (−0.14), warp (−0.25) or telegraph (−0.35). Whitening smears segment phases, so E c² decays more slowly than the model assumes (0.24 measured against 0.16).
  - *Fails on precision at T = 2000:* SD is 0.15–0.19 in family 1 and 1–6 in the transfer family (narrowband, heavy-tailed fourth moments). The debiased modulus shifts nulls by +0.1 to +0.25; the plain one is biased low.
  - Separating envelope coupling from time-varying or nonlinear waveform coupling seems to need source-model assumptions (Gaussian carrier, slow amplitude); this is inferred, not proven. The prototype is not in the repository; the recipe is the formula above on `_tapered(..., 4)` coefficients.
