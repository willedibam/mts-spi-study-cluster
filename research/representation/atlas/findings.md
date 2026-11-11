# Atlas step 1: discovery findings (12 November 2026)

Exploratory. Reading rules: [`docs/interpreting-spi-pairs.md`](../../../docs/interpreting-spi-pairs.md). Design and costs: [design.md](design.md). Code: [`discovery.py`](discovery.py) (`analyse`, `pairs`, `screen`, `reliability`); outputs in `results/representation/atlas-discovery-261112/`.

## Data and contrasts

- **NeuroTycho macaque ECoG:** 126 channels, 100 Hz, pyspi r13 with 272 SPIs.
  - 2-s arm: 1,260 windows.
  - 10-s arm (stage A): 500 windows.
  - Units are sessions, and animals are fewer. In the 2-s arm:

    | Condition | Sessions | Animals |
    |---|---|---|
    | Ketamine (KT) | 4 | 2 |
    | Ketamine–medetomidine (KTMD) | 11 | 4 |
    | Medetomidine (MD) | 4 | 2 |
    | Propofol (PF) | 4 | 2 |
    | Sleep | 7 | 2 |

- **Fly isoflurane:** 13 flies. There is no eyes-open state, so fly has no null scale.
- **Effects:** within-unit differences against eyes-closed wake, raw and adjusted.
  - *Strength-adjusted:* within-unit OLS on mean |r|, SD |r| and the covariance participation ratio.
  - *Spectrally adjusted:* additionally on the channel-mean peak frequency and spectral entropy.
- **Scale:** the between-unit SD of the eyes-open effect.
- **No surrogates in this bank.** Statements about these z effects therefore cannot say "beyond linear". Fixed-lag SPIs (sample units, 10 ms per sample) read dominant frequency.

## Lag: the coupling becomes less zero-lag, and the strongest pairs carry the lag (all four drugs, not sleep)

- **Main readout:** z(`cohmag` low band, `icoh` low band), where `cohmag` = |C|² and `icoh` = |Im C| = |C||sin φ|.
  - It is negative in wake (z₀ = −0.11): the most coherent pairs are the most zero-phase, as expected from volume conduction and short-range coupling.
  - It rises under every drug, in both arms and in every session after strength adjustment. 2-s arm, strength-adjusted:
    - ketamine +.19 (4/4)
    - ketamine–medetomidine +.18 (11/11)
    - medetomidine +.08 (4/4)
    - propofol +.22 (4/4)
  - Mixed-stage sleep: +.00. Deep-sleep tertile in the 10-s arm: −.01. The eyes-open change has the opposite sign (−.01 to −.05).
  - It survives spectral adjustment at 2–4 null SDs per agent. It partly reverses at recovery.
- **Three estimator families agree in sign** (but see the reliability section: these partners are noisy, so this support is weak), each set against its own zero-lag-insensitive partner:
  - phase consistency: `ppc` against `dswpli`, both bias-corrected, so the coherence noise floor is excluded;
  - amplitude envelopes: `pec` against `pec_orth` (leakage-orthogonalised);
  - zero-lag against best-lag correlation: `cov-sq` against `xcorr-sq_max`, which falls by 4.5 to 12 null SDs.
  - The best-lag/zero-lag split is also the top family of the unsupervised screen: `xcorr-sq_max` parts from every zero-lag statistic, in every session of every agent.
- **Level moves too.** The mean lagged share rises under every drug except ketamine:
  - `icoh/cohmag`: +.03 to +.07;
  - `dswpli/ppc`: +.09 to +.34;
  - `pec_orth/pec`: +.13 to +.19.
  - Sleep stays within ±.04.
  - So the statement is not "lags became more heterogeneous". It is: more of the coupling is non-zero-lag, and the non-zero-lag part now sits on the strongly coupled pairs rather than the weak ones.
- **Frequency works against the result.** Peak frequency falls 1–2 Hz under the drugs, which shrinks the phase of a fixed delay (φ = 2πfτ), so this confound predicts a fall, not the observed rise.
- **Not resolved:**
  - delay against phase (a lower carrier with longer delays is not separated);
  - the average reference;
  - whether the cause is physiological (propagating slow activity is one candidate [S]).
- **Fly isoflurane goes the opposite way** (inferred):
  - broadband `cohmag|icoh` −.04, with 11 of 13 flies negative;
  - `cov-sq|xcorr-sq_max` +.05, with 11 of 13 flies positive.
  - So the reading is regime-specific, not a generic marker of anaesthesia. The fly montage and probe geometry differ.

## Gaussian against model-free MI: fewer pairs follow the Gaussian prediction (all four drugs, not sleep)

- **Full bank** (pyspi, 126 channels): z(`mi_gaussian`, `mi_kraskov_NN-4_DCE-AUTO`) has z₀ = .91 and null SD .010.
  - Spectrally adjusted, it falls under every drug in every session (0 positive):
    - ketamine −2.3 SD
    - ketamine–medetomidine −12.0 SD
    - medetomidine −4.1 SD
    - propofol −5.9 SD
  - The 10-s arm agrees. Mixed-stage sleep is ≈ 0, and the deep tertile is −0.6 SD.
- **Two other model-free estimators agree:**
  - kernel MI: −2.2 to −7.0 SD;
  - distance correlation against r²: −4.5 to −9.7 SD; ketamine weak.
  - So no single estimator's quirk explains it.
- **Level goes the "wrong" way.** Mean kNN-to-Gaussian MI falls from .78 to about .2 under ketamine–medetomidine, and to about zero in the 10-s arm.
  - Genuinely non-Gaussian dependence would push kNN MI up, not down.
  - Candidate causes:
    - the DCE Theiler window grows as drugged signals slow, which shrinks the effective sample;
    - Pearson-based Gaussian MI is inflated by large-amplitude joint epochs.
  - The first would lower z through reliability, not character; see the reliability check below.
- **The pyspi-free version is the anchor** (independent review, 16-channel subsets of the 2-s arm). It is real minus IAAFT, so it matches autocorrelation and marginal distributions.
  - It falls under every drug.
  - It survives normal scores and top-5% trimming.
  - Gain normalisation cuts it to a third.
  - Conditional-mean curvature rises.
- **Working name:** "nonlinear (non-Gaussian) dependence". Which form it takes is undecided: curved coupling, or amplitude-state-dependent coupling. The decisive test is on the 10-s arm (design.md).

## Edge-profile reliability (job 183670442, `*-reliability.json`)

Reliability here is the mean correlation of an SPI's edge profile between windows of one unit and state. It mixes estimator noise with real window-to-window change of the coupling pattern. Disattenuating by it overcorrects: in wake the disattenuated z of the MI pair exceeds 1. So it bounds, but does not decide.

- **Profiles become less reproducible between windows under every drug, not in sleep.** This is a result in its own right, not a z feature.
  - 2-s arm, ketamine–medetomidine and propofol: Gaussian MI .82 → about .60, kNN MI .69 → about .46, distance correlation −.26, `xcorr-sq_max` −.33. Sleep changes by −.03 to −.06.
  - The 10-s arm agrees, at about half the size.
  - Reading: the coupling pattern changes more from one 2-s window to the next under anaesthesia, i.e. temporal non-stationarity of dependence, which is the lag-variability and intermittency priority.
  - Confound: drug epochs may drift in depth, while wake is steady. Not yet controlled; compare adjacent windows against distant ones.
- **MI pair (Gaussian against kNN): unresolved on the full bank.**
  - Gaussian MI, a function of r with no Theiler window, loses reliability exactly as much as kNN MI. So a kNN-specific DCE artefact is not supported.
  - But the raw z fall (−.03 to −.08) is far smaller than independent noise at these reliabilities would produce (about .91 → .64). So most of the reliability drop is shared, real non-stationarity, and the z fall cannot be attributed to noise or to character from this bank.
  - The IAAFT version, whose surrogates share window length and autocorrelation, stays the anchor. The kNN-to-Gaussian level collapse is still unexplained.
- **Lag pair holds.** In the 10-s arm (`icoh` reliability .46) z(`cohmag`, `icoh`) rises in every animal under every drug, raw and disattenuated (+.17 to +.24), and the deep tertile is −.04.
- **Demoted:**
  - Partners with near-zero between-window reliability: `dswpli` (.03), `pec_orth` (.03), `te_kraskov_DCE` (.03–.07), `icoh` in 2-s windows (.18). The `ppc|dswpli`, `pec|pec_orth` and TE `k-1|k-2` effects mostly carry within-window structure, and their disattenuated values swing in sign. They support the lag reading only weakly; the TE memory pair is not interpretable.
  - The best-lag/zero-lag `xcorr` fall turns positive when disattenuated, and `xcorr` loses the most reliability (−.33). Read it as stability, not lag, until a within-window split-half check exists.

## Other named pairs (42 pre-listed pairs; `*-pairs.json`)

- **Consistent under all four drugs in both arms, sleep ≈ 0:**
  - **Interaction memory:** TE with one against two samples of target history (10 against 20 ms), `te_kraskov_DCE k-1|k-2`. It rises by +3.7 to +5.0 SD spectrally adjusted and is flat in sleep. Read literally: one more sample of target history changes the pair ranking less. It is sensitive to autocorrelation structure, so a character reading is not earned.
  - **Rank against Pearson:**
    - `cov|spearmanr`, `cov-sq|spearmanr-sq`, `corr_pearson_tau-1|corr_spearman_tau-1` and `pdist_euclidean|cityblock` fall by −0.01 to −0.02.
    - These are ceiling pairs (z₀ ≈ .99): large in SD, tiny in z.
    - Without surrogates they read marginal shape first; earlier work found the raw linear–rank readout tracks marginal shape.
- **Consistent across drugs, but deep sleep also moves:**
  - **Phase against amplitude coupling** (`plv|pec`): −4 to −9 SD in both arms; 10-s deep tertile −3.8 SD.
  - **Bulk against atypical samples** (`prec-sq` LedoitWolf against MinCovDet; `cov-sq` Empirical against MinCovDet): −4 to −14 SD in the 10-s arm only, where the deep tertile is −2.4 to −3.9 SD. This matches the macaque MinCovDet flag result.
  - **Rigid against elastic alignment** (`dtw|dtw_sakoe-chiba`): −5 to −14 SD in the 10-s arm only, deep tertile −3.4. The 2-s windows are too short for warping to matter. The warp axis failed construction validation.
  - All 10-s arm entries rest on two units per agent, and raw effects only.
- **Inconsistent or null:** band locus, broadband against peaked, predictive against contemporaneous, nonlinear against linear prediction, `sgc` order, `gc` against AIS, and macaque mediation.
- **Fly mediation:** `cov-sq|prec-sq` +.105 (13/13), and the LedoitWolf version +.076 (12/13). This is fly-specific.
- **Structural zeros, dropped:** `plv|pli` and `plv|wpli`. PLI and wPLI are stored antisymmetric, so their z with a symmetric SPI is exactly 0.
- **dcoh/pdcoh:** not testable here, because the spectral-directed SPIs are not finite in every 2-s window.

## Screen over every z coordinate (`*-screen*.json`)

- **2-s arm:** 1,264 of 19,503 usable coordinates pass. To pass, a coordinate needs the same sign in at least 75% of each agent's sessions, raw and adjusted, and a mean beyond 3 eyes-open null SDs. Node-additive SPIs are excluded.
- **Top families:**
  - `xcorr-sq_max` against any zero-lag statistic (lag, above);
  - non-DCE against DCE Kraskov. This is an estimator diagnostic: the autocorrelation bias grows as drugged signals slow, which supports the reliability concern above.
- **Two ranking problems:**
  - Near-identity ceiling pairs rank high, because their null SD is tiny: Spearman against Kendall, and in the 10-s arm `gc_gaussian` against `sgc_parametric`. Rank by effect relative to 1 − z₀, or drop z₀ > .98.
  - Fly has no eyes-open null, so its "strength" divides by about 1e-6 and its ranking is meaningless. Its top hits are fixed-lag-in-samples pairs, which read frequency.

## Fixed on the way

`delta_fraction` exists only in the 10-s arm. Its all-NaN column emptied every adjusted 2-s contrast and the screen's adjusted effects, and in the 10-s arm it adjusted away the drug state itself. Adjustment now uses strength covariates in every arm, plus a separate spectral adjustment.

## Beyond "all drugs": single-condition, opposite-sign and all-but-one rankings (`*-explore.json`)

`discovery.py explore` treats each condition separately. A condition counts as consistent when at least 75% of its sessions agree in sign, raw and adjusted, and every animal's mean agrees. Ceiling coordinates (|z₀| ≥ .98) and node-additive SPIs are excluded, and only one coordinate per SPI-family pair is reported. These rankings are post-selection over about 17,500 coordinates; read them only where several coordinates or both window lengths converge.

- **Propofol only: lagged high-band phase coupling appears as a pattern.**
  - In 25–50 Hz, debiased lag-insensitive phase measures (`dswpli`, `dspli`, `icoh`) begin to track high-band phase locking.
  - z(`plv` hi, `dswpli` hi): +.13 under propofol in both animals (2-s +.10/+.17; 10-s +.13/+.12). The other drugs are at most +.03, sleep +.005, and recovery largely reverses it (+.01 to +.05).
  - The level moves too: the `dswpli/ppc` share in the high band rises +.33 (2-s) and +.30 (10-s) under propofol, in both animals, against at most +.07 for the other drugs.
  - Many coordinates on `dswpli`/`dspli`/`icoh` in the high band rank propofol-specific in both arms, at +6 to +9 null SD.
  - Reading: under propofol, but not ketamine, medetomidine or sleep, a reproducible non-zero-lag 25–50 Hz phase coupling emerges on the pairs with strong high-band locking. Physiology is not established [S].
- **All but medetomidine: robust against classical dependence** (10-s arm). For example, z(`cov-sq_MinCovDet`, `mi_kraskov_DCE`) falls by about 9.5 SD under ketamine, ketamine–medetomidine and propofol, and by +0.1 SD under medetomidine. This is the same exception as the pyspi-free MinCovDet flag fraction (borderline under medetomidine), so two independent methods agree.
- **Weaker or uninterpretable:**
  - Ketamine-only and medetomidine-only hits are scattered across unrelated families.
  - Opposite-sign pairs mostly involve noisy partners: `dswpli` broad against low band (ketamine against ketamine–medetomidine), non-DCE TE, and `psi`.
  - Deep-sleep opposites rest on non-DCE `te_kraskov`.
- **Fly:** nothing beyond d ≈ 1.7 among 13 flies.
