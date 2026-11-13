# Atlas step 1: discovery findings (12 November 2026)

Exploratory. Reading rules: [`docs/interpreting-spi-pairs.md`](../../../docs/interpreting-spi-pairs.md). Design and costs: [design.md](design.md). Code: [`discovery.py`](discovery.py) (`analyse`, `pairs`, `screen`, `reliability`); outputs in `results/representation/atlas-discovery-261112/`.

## Data and contrasts

- **NeuroTycho macaque ECoG:** 126 channels, 100 Hz, pyspi r13 with 272 SPIs.
  - 2-s arm: 1,260 windows.
  - 10-s arm: 1,260 windows, stages A (500) and B (760) pooled. Both stages share one pyspi identity (config, `extract.py`, computation 3.0.0.r13, 272 SPIs in one order); job 183816046, `neurotycho-long-*.json`. Stage-A-only outputs stay as `neurotycho-long-stage-a-*.json`.
  - Units are sessions, and animals are fewer. Both arms now have:

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
- **Pooling the 10-s arm (13 November).** Raw and null-decomposed means of the drug contrasts mostly replicate stage A, but not everywhere: spectrally adjusted effects of some pairs change sign or collapse (`ppc|dswpli` propofol +.056 → +.008; `dtw` ketamine–medetomidine +.050 → −.037; `prec-sq` LedoitWolf against MinCovDet propofol −.078 → +.129), and effects carried by single agents in stage A can vanish in stage B (derived as (n_P·P − n_A·A)/n_B). Stage A had two units per agent, and its eyes-open null SD was too small for some pairs (MI pair .0036 → .0059; `prec-sq` LedoitWolf against MinCovDet .019 → .064), so its SD multiples overstated those effects. **10-s numbers in this file are stage A unless marked pooled**; the pooled outputs `local`, `stability`, `screen`, `screen-dc`, `report` and the reliability contrasts are not yet read.
- **No surrogates in this bank.** Statements about these z effects therefore cannot say "beyond linear". Fixed-lag SPIs (sample units, 10 ms per sample) read dominant frequency.

## Lag: the raw rise is mostly an estimation floor; a debiased non-stationary residual survives (KTMD, PF), not sleep

- **Bank readout:** z(`cohmag` low band, `icoh` low band), where `cohmag` = |C|² and `icoh` = |Im C|. z₀ = −0.11 in wake. 2-s arm, strength-adjusted: ketamine +.19 (4/4), ketamine–medetomidine +.18 (11/11), medetomidine +.08 (4/4), propofol +.22 (4/4); sleep +.00. Fly isoflurane goes the opposite way (−.04, 11/13 negative).
- **Lag null** (`discovery.py lag-null`, `*-lag-null.json`; pyspi-free, 32 evenly spaced channels, 3 draws per null, job 183766068). For each unit and state, the real windows are set against two stationary Gaussian nulls with the same T and window count:
  - *zero-phase:* the state's mean cross-spectrum with every phase set to zero. No lag at all, so whatever moves here is the **floor**: estimation noise in |Im C| and kin, which grows as spectral concentration lowers the effective sample size.
  - *stationary:* the state's mean cross-spectrum itself. Stationary − zero-phase is the **mean lag** (lags fixed across windows); real − stationary is the **residual** (anything that changes from window to window, or is non-Gaussian).
  - Construction bias (a null re-drawn from a null, one draw) is within ±.02 at 2 s and down to −.044 at 10 s; for phase dispersion it is −.006 to −.019 under the drugs, the size of its residual, so that row is read net of bias. The pyspi-free real deltas reproduce the bank's raw deltas (`cohmag|icoh` KTMD +.146 against +.143; `ppc|dswpli` +.092 against +.090; `pec|pec_orth` +.131 against +.144; `cov-sq|xcorr-sq_max` −.150 against −.142).
- **2-s arm, Δ against own eyes-closed wake** (sessions positive / total, animals positive / total; eyes-open contrasts of the lag-pair residuals are within ±.02, of `pec|pec_orth` up to +.04):

  | Pair | Part | KT | KTMD | MD | PF | Sleep |
  |---|---|---|---|---|---|---|
  | `cohmag\|icoh` | real | +.027 | +.146 | +.127 | +.232 | +.020 |
  | | floor | −.074 | +.090 | +.066 | +.126 | −.005 |
  | | mean lag | +.088 (3/4, 2/2) | −.007 | +.053 (4/4, 2/2) | +.030 | +.025 |
  | | residual | +.013 | +.063 (11/11, 4/4) | +.009 | +.075 (4/4, 2/2) | +.001 |
  | `ppc\|dswpli` (both debiased) | real | +.033 | +.092 | +.026 | +.198 | +.016 |
  | | floor | −.019 | +.013 | +.016 | +.039 | −.003 |
  | | residual | +.019 | **+.090 (11/11, 4/4)** | +.019 | **+.131 (4/4, 2/2)** | +.021 (5/7) |
  | `pec\|pec_orth` | floor | −.015 | +.070 | +.073 | +.101 | +.021 |
  | | residual | +.059 | +.041 | +.002 | +.050 | +.031 (7/7) |
  | `cov-sq\|xcorr-sq_max` | real | −.070 | −.150 | −.104 | −.149 | −.021 |
  | | floor | −.038 | −.129 | −.057 | −.115 | −.023 |
  | | residual | −.016 | −.004 | +.006 | −.017 | −.010 |
  | Phase dispersion | residual | −.007 | +.016 (11/11, 4/4) | +.007 | +.025 (4/4, 2/2) | −.018 (1/7) |
  | | bias | +.005 | −.009 | −.019 | −.006 | +.010 |
  | Magnitude dispersion | residual | −.001 | −.002 | −.000 | −.007 | −.006 |

  Phase dispersion is 1 − |Σ_w C_w| / Σ_w |C_w| per edge and frequency (zero when only magnitudes vary); magnitude dispersion is SD_w |C_w| / mean_w |C_w|; both in 0–25 Hz.
- **10-s arm (pooled) agrees:** `ppc|dswpli` residual KTMD +.064 (10/11 sessions, 4/4 animals; bias −.007), PF +.126 (4/4, 2/2; bias +.010), KT +.022, MD +.007, deep tertile +.018; `cohmag|icoh` residual KTMD +.032 (3/4 animals), PF +.093 (2/2), MD +.024 (2/2); phase-dispersion residual KTMD +.011 (bias −.011), PF +.012 (bias −.008), deep tertile +.007 (bias −.002); magnitude-dispersion residual ≈ 0 everywhere; `cov2|xcorr2_max` residual ≈ 0.
  - Stage B alone (derived): `ppc|dswpli` net of bias KTMD +.058, PF +.143, so both reproduce; KT −.000 and MD −.022, so the single agents do not. Propofol phase dispersion net of bias is −.002 in stage B (stage A +.009).
  - Not drug-specific at 10 s: the `cohmag|icoh` residual net of bias is +.038 in the deep tertile (7/7 sessions), about KTMD's +.043, so the "not sleep" statement holds for `ppc|dswpli` (deep +.003 net) only. Eyes-open `ppc|dswpli` net of bias is −.019 (KTMD) and −.022 (PF), at the ±.02 bound.
- **pyspi null banks, all units, 2-s arm** (`discovery.py null-bank` and `null-decompose`, `jobs/null_banks.pbs`; every unit, 1,260 windows per kind, the real bank's config, `extract.py` and pyspi 0c798b4; `$A/discovery/neurotycho-short-null-decompose.json`). Parts as in the lag null, per session against its own eyes-closed wake, over all 272 SPIs. Net = residual + bias: the stationary null is itself drawn from an estimated spectrum, and this is how the table above reads "net of bias".
  - **Provenance:** build 183853102; farms zero_phase 183853103, stationary 183853104 and bias 183853105, about 4.8 h each and 5.57 kSU together, with complete audits. Reduce 183853106 failed its audit because `PYSPI_SOURCE` was unset: the identity holds the pyspi path, so every shard read "stale identity". It was rerun as 183872304, and the stage now requires the source. Decompose 183872574 ran from snapshot `atlas-acc7cd44`. The one-unit cost check (job 183843903) gave 42 core-minutes per window, as for real windows.
  - **Checks:** restricted to the cost-check unit (Chibi KTMD 20110621), the reader reproduces it exactly: `cov-sq|xcorr-sq_max` floor −.174, `ppc|dswpli` residual +.182, residual RMS .049, bias RMS .027. All-unit residuals match the pyspi-free lag null (in brackets): `ppc|dswpli` KTMD +.091 (+.090), PF +.124 (+.131); `cohmag|icoh` KTMD +.057 (+.063), PF +.063 (+.075); `pec|pec_orth` KT +.050 (+.059), KTMD +.043 (+.041), PF +.058 (+.050). The stationary null reproduces the real fall in covariance participation ratio (KTMD −2.80 against −3.28, KT −3.17 against −3.32); the zero-phase null does not (KTMD −0.40).
  - **Construction:** the zero-phase null clips about 14% of the |S| eigenvalue mass and raises mean |r| (one unit: wake .152 → .265). Floor and mean lag are read for coherence-type pairs only; time-domain pairs are read through the stationary residual.
  - **Bias is mostly noise at the condition mean, systematic on some pairs:** over coordinates, the condition-mean bias RMS is .013–.023 under the drugs against a residual RMS of .029–.045, and the two correlate at −.03 to .33. Under KTMD (11 sessions), the unit-level t RMS is 1.21 for bias and 3.82 for the residual (eyes-open 1.47 and 1.67); with 4 sessions the t RMS is dominated by near-zero SDs and is not read. Some pairs carry a consistent bias: the MI pair under MD (+.006) and PF (+.011, 4/4 sessions each), and band locus in KTMD eyes-open (+.016, 11/11). So named pairs are read as net. The one-unit "bias equals the residual" on the uncorrected Kraskov MI pair (−.018 against −.024) was that unit's draw; all-unit bias there is +.000 to +.011.
  - **Most of every drug's z change is what its mean cross-spectrum gives a stationary Gaussian process.** Over the usable coordinates (36,046; 34,716 under KT), the residual keeps 11–19% of the drug contrast's sum of squares (KT .12, KTMD .12, MD .11, PF .19; net .14–.27). Sleep's contrast is small (RMS .027) and keeps .43. One null draw per window, 2-s arm only.
  - **Broad change beyond a stationary Gaussian process: clear under KTMD, about 3× control under PF, not in sleep.** Coordinates whose |net| exceeds a threshold shared by every condition, with every session's residual of one sign, drug against the same sessions' eyes-open (|net| > .04; > .06): KTMD 3,958 against 119 (1,777 against 21; 77% positive; recovery 484); PF 8,084 against 2,841 (4,833 against 1,628; 68% positive against 46%); KT 4,092 against 2,034 (2,153 against 705); MD 3,372 against 2,079 (1,613 against 968); sleep 1,070 against 1,040 (358 against 648). With 4 sessions, "one sign" passes 1 in 8 by chance per coordinate, which is why the 4-session eyes-open counts are large; only KTMD (11 sessions) is clean. An earlier version thresholded each condition at 3× its own bias RMS, which made drug and eyes-open counts incomparable (job 183873527 replaced it).
  - **Where, by SPI (net RMS over each SPI's pairs, exploratory):** KT and PF lead with signed mean cross-correlation `xcorr_mean` (PF .26, KT .15, against condition RMS .05 and .04). KTMD leads with Johansen cointegration statistics, `si_kernel` and `pec_orth` (.06–.07). MD, sleep and eyes-open lead with fixed-lag correlations (`corr_*_tau-10/20`; `tau-4/5` in eyes-open), so those are generic state-change entries. Cointegration and mean cross-correlation are sensitive to slow within-window drift, which a stationary null lacks [I]; not yet read.
  - **Named pairs** (2 s; residual with sessions and animals of that sign; KTMD eyes-open as control):

    | Pair | Part | KT | KTMD | MD | PF | Sleep | KTMD EO |
    |---|---|---|---|---|---|---|---|
    | MI pair, `mi_gaussian\|mi_kraskov_DCE` | real | −.012 | −.076 | −.029 | −.075 | −.002 | +.003 |
    | | residual | −.023 (4/4, 2/2) | −.028 (11/11, 4/4) | −.009 (4/4, 2/2) | −.030 (4/4, 2/2) | +.001 | +.003 |
    | | net | −.022 | −.025 | −.004 | −.021 | +.002 | +.006 |
    | MI pair, uncorrected Kraskov | residual | −.033 (4/4) | −.022 (10/11) | −.014 (4/4) | −.030 (4/4) | −.001 | +.007 |
    | r² against uniform 0–25 Hz coherence, `cov-sq\|cohmag` | real | −.096 | −.157 | −.141 | −.146 | −.013 | +.005 |
    | | residual | −.025 (4/4) | −.017 (9/11, 4/4) | −.026 (4/4) | −.026 (4/4) | −.014 (7/7) | −.004 |
    | Band locus, `cohmag` 0–25 against 25–50 Hz | real | +.071 | +.096 | −.027 | +.085 | −.010 | +.052 |
    | | residual | +.070 (4/4) | +.061 (11/11) | −.041 (4/4 down) | +.073 (4/4) | −.058 (7/7 down) | +.032 (11/11) |
    | `cohmag` 0–25 Hz against `pec_log` | residual | −.030 | +.052 (11/11, 4/4) | −.020 | −.000 | +.004 | −.005 |

  - **Readings:**
    - *MI pair (net):* a stationary Gaussian process with the drug's spectrum gives 60–65% of the raw 2-s fall under KTMD and PF (−.048 of −.076, −.045 of −.075), which is the kNN estimator's response to spectral concentration. What is left, net of bias, is −.022 (KT, 3/4 sessions), −.025 (KTMD, 10/11) and −.021 (PF, 4/4); MD −.004; sleep, eyes-open and recovery within ±.006 except KT recovery −.014. That is the same sign and order as the 16-channel IAAFT residual at 2 s (−.018 to −.024), on all 126 channels, so the full bank's large spectrally adjusted SDs overstate the part beyond a stationary Gaussian process, which is 30–40% of the raw fall. This null is looser than IAAFT (state-mean spectrum, Gaussian marginals, no window matching), so it does not replace the anchor.
    - *r² against uniform low-band coherence:* the stationary null gives 75–90% of the fall. Net of bias, the remainder is −.011 to −.025 under the drugs and −.013 in sleep, while eyes-open nets reach −.010 to −.018 (KTMD, MD, sleep sessions). So beyond the spectrum it is at control size, and the link stays linear-spectral, as the weighting null found.
    - *Band locus (net):* the beyond-spectrum rise is not specific to the drug state. Eyes-open carries 60–75% of it (KT +.050 against drug +.080, KTMD +.048 against +.064, PF +.041 against +.066; MD −.004), recovery does not reverse it (KT +.103, KTMD +.087, PF +.046), and it is negative under medetomidine (−.032, 4/4) and in sleep (−.045, 7/7). Literal reading: low- and high-band coherence rankings converge beyond the spectrum whenever the state leaves eyes-closed wake, except under medetomidine and sleep. The medetomidine-with-sleep grouping recurs.
    - *`cohmag|pec_log`:* a KTMD-only residual (+.052, every session and animal; recovery +.015). Single agent; literal only.
- **Calibration** (`test_lag_null_calibration`; synthetic, 10 channels, 40 windows, five seeds): across-window phase dispersion beyond the stationary null rises only for window-varying lag (+.10 to +.12; zero-lag within-window bursts and window-to-window strength change ≤ +.025); magnitude dispersion falls with varying lag (−.06 to −.10) and rises with strength change (+.02 to +.03). The `ppc|dswpli` residual is not lag-specific: strength change raises it as well (+.01 to +.05), and it is noisy at this size.
- **Reading (literal):** under ketamine–medetomidine and propofol, debiased lagged phase coupling (`dswpli`) tracks phase consistency (`ppc`) within a window more than a stationary Gaussian process with the state's spectrum allows (net of bias: KTMD +.08, PF +.14; both arms). The debiased pair has almost no floor here, so this is not estimation noise. It reverses at recovery (KTMD residual −.002, PF +.012) and is absent in natural sleep.
- **Reading (character, inferred, by condition):**
  - *Propofol:* lag variability is the leading candidate. Phase dispersion rises beyond the null net of bias at 2 s (+.019, 4/4) while magnitude dispersion does not rise (−.007), which in the calibration only varying lag does. Two animals. At 10 s it is not reproduced: stage A +.009, stage B −.002, pooled +.004. So the candidate rests on the 2-s arm alone.
  - *Ketamine–medetomidine:* the residual is not across windows (phase dispersion net of bias +.007 at 2 s, ≈ 0 at 10 s; magnitude flat), so it is within-window structure or non-Gaussian, form unidentified.
  - *Ketamine and medetomidine:* a stable mean lag (`cohmag|icoh` +.088 and +.053; 2 s every animal; 10 s +.188 and +.081, ketamine 1/2 animals), little residual. Two animals each.
- **Withdrawn:**
  - the level claim ("more of the coupling is non-zero-lag": the `icoh/cohmag` share rises as much in the zero-phase null);
  - "the strongest pairs carry the lag" as a static statement;
  - the best-lag against zero-lag `xcorr-sq_max` fall, and with it the screen's top family: its floor carries 77–87% and its residual is ≈ 0;
  - `pec|pec_orth` as lag support: mostly floor, and its residual also appears in sleep and eyes open.
- **Not resolved:** within-window lag switching against non-Gaussian stationary structure under ketamine–medetomidine (decider: whether the sign of Im C persists between the halves of a 10-s window); the average reference; delay against phase as peak frequency falls; physiology (variable propagation direction of slow activity is one candidate [S]).

## Gaussian against model-free MI: fewer pairs follow the Gaussian prediction (all four drugs, not sleep; survives IAAFT)

- **Full bank** (pyspi, 126 channels): z(`mi_gaussian`, `mi_kraskov_NN-4_DCE-AUTO`) has z₀ = .91 and null SD .010.
  - Spectrally adjusted, it falls under every drug in every session (0 positive):
    - ketamine −2.3 SD
    - ketamine–medetomidine −12.0 SD
    - medetomidine −4.1 SD
    - propofol −5.9 SD
  - 10-s arm (pooled), raw: ketamine −.014 (0/4), ketamine–medetomidine −.032 (0/11), propofol −.032 (0/4), medetomidine −.003 (1/4); deep tertile −.001. Spectral adjustment keeps ketamine and ketamine–medetomidine (−.030, −.022) and removes propofol (+.001) and medetomidine. So medetomidine does not reproduce at 10 s here, and the propofol fall is shared with the spectral covariates (the IAAFT version below keeps it).
- **Full-bank null decomposition (2 s, lag section):** a stationary Gaussian null with each state's spectrum gives 60–65% of the raw fall under KTMD and PF. Net of bias, the rest is −.021 to −.025 under ketamine, ketamine–medetomidine and propofol, −.004 under medetomidine and ≈ 0 in sleep, eyes-open and recovery: same sign and order as the 16-channel IAAFT anchor below, which it does not replace (looser null).
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
  - The two model-free estimators disagree on level: kernel MI (no Theiler window) rises under every drug (2-s .30–.34 → .40–.42) while DCE-kNN MI falls. So level is estimator-dependent and is not read; only the pattern (z) is.
- **The pyspi-free version is the anchor** (independent review, real minus window-matched IAAFT, so it matches spectra, autocorrelation and marginal distributions, and with them the effective sample size; scripts in that session's scratchpad, not in the repo).
  - 2-s arm, 16 channels: falls under every drug in 22 of 23 sessions (−.018 to −.024); sleep −.005; eyes-open null SD .009. It survives normal scores, top-5% trimming and strength adjustment.
  - 10-s arm (16 channels), in eyes-open null SDs:
    - stage A: ketamine −6.0, ketamine–medetomidine −4.7 (3/4), medetomidine −2.7, propofol −7.9; sleep +0.1, deep tertile −0.5.
    - stage B (18 new archives): ketamine −5.1 (2/2 down), ketamine–medetomidine −5.1 (7/7), medetomidine −1.0 (2/2), propofol −4.0 (2/2); sleep −0.6 (5/5, inside the null), deep tertile −0.8.
    - Stages A and B together: 22 of 23 drug sessions down. "Drugs, not deep sleep" holds at both window lengths; sleep has the same sign at about a tenth of the size.
  - Where it sits (2-s, edges chosen on even windows and measured on odd):
    - Sparse: the top 1% of edges by excess kNN MI over IAAFT carry it (propofol +.025 against wake +.006, 4/4; ketamine–medetomidine +.043 against +.003, 8/11; sleep 0).
    - Pair-level, not channel-additive: an a_i + a_j fit explains 6–14% of the edge map, as it does for eyes-open maps.
    - Propofol hubs reproduce within an animal: George r = .69 (.67 after partialling node strength and its drug change; eyes-open same-animal null ≈ 0, max .24), Chibi .40 (.20 partialled). Across all sessions and animals edge maps do not reproduce (r = .05).
    - Anatomy (neurotycho.org electrode maps): hub edges join adjacent electrodes (~5 mm). George: a dorsolateral patch near the central sulcus (electrodes 3, 10–12, 21); Chibi: medial frontal wall (52, 53, 58, 59), plus lateral temporal and frontal. No shared locus across animals.
    - **Hubs mark where the ranking breaks, not where the non-Gaussian dependence is** (review hub test, verified there; normal-scored, 126-channel surrogates, hubs chosen on one George propofol session and tested on the other, both directions, against 60 edges matched on |r| ≈ .93). Most tightly coupled neighbour pairs gain model-free dependence beyond the surrogate under propofol; the hub edges gain less than matched controls (−3.8 [−5.4, −2.4] and −1.6 [−2.6, −0.6] surrogate SD). The z fall is a divergence among equally strong pairs: a reproducible patch (George electrodes 2–4, 10–13, 21) does not gain, so its kNN-MI rank falls behind its Gaussian-MI rank.
  - Surrogate choice: IAAFT built jointly over 126 channels stays closer to the data than a 32-channel IAAFT and roughly halves the 2-s residual on the same windows and edges; George's ketamine sessions change sign at 2 s between the two but are negative at 10 s under both. The 126-channel joint IAAFT is the conservative reference.
  - Provenance: `review-nature-level.md` in the session notes folder (outside the repo); scripts and arrays are in the review session's scratchpad. Not yet in the repo.
- **Level, beyond IAAFT:** kNN-minus-Gaussian MI falls beyond the surrogates under ketamine–medetomidine (−5.8 SD), propofol (−6.2) and medetomidine (−2.6), and rises under ketamine (+3.8); stage B agrees in sign (propofol −6.5, ketamine–medetomidine −2.9, ketamine +6.4). So the level fall is not only a DCE artefact: with spectra, marginals and r matched, model-free MI sits below the linear-Gaussian expectation. Since non-Gaussian dependence cannot lower true MI below the Gaussian value for Gaussian marginals, the level is not read as character; the z fall reads as a reshuffling of which pairs carry non-Gaussian dependence.
- **Form: curved coupling is not supported.** A calibrated 10-s curvature readout (synthetic: static curvature positive, gain modulation negative, coupling switching ≈ 0) shows no positive curvature beyond IAAFT (−1.3 to −1.7 SD). Within-window coupling switching beyond IAAFT is large under every drug (propofol +20 SD) but also in deep sleep (+4.1), and the window-level z residual tracks none of switching, curvature or envelope co-modulation.
- **Copula level, every agent:** normal-scoring each channel leaves the residual intact, so it is dependence, not a change in channel marginals. 10-s arm, 32-channel joint IAAFT, drug minus eyes-closed wake, normal-scored (raw):
  - ketamine −.016 (−.016), 4/4 sessions down;
  - ketamine–medetomidine −.016 (−.017), 11/11;
  - medetomidine −.006 (−.005), 4/4;
  - propofol −.012 (−.012), 4/4;
  - sleep −.001 (−.002), 2/2 stage-A units only, because the stage-B sleep run failed on a channel-count mismatch (126 against 127 channels).
  
  Eyes-open stays within ±.0054. So 23/23 drug sessions are down, medetomidine the smallest. Computed here from the review session's arrays (`zns_*.npz`, `zgk_ns.py`, scratchpad, not in the repo); the calibration that normal-scoring removes marginal artefacts but keeps genuine non-Gaussian coupling types is in the review notes.
- **Form, propofol, inferred: amplitude-linked decoupling.** On the matched control edges beyond the surrogate: excess MI +3 to +4 SD (wake ≈ +1), block-to-block coupling variability +2, heavy-tailed residuals +12 to +14; curvature −0.7 and joint co-kurtosis ≈ 0 (no curved coupling function, no shared bursty driver); allowing ±50 ms lag recovers none of it. The amplitude–coupling readout is −6.5 to −7.5: high-power blocks are more coupled in wake (+.55 raw) and less so under propofol (+.25). Reading: tight neighbours loosen during local, non-shared high-amplitude episodes. Inferred, not directly shown; one animal, held out across its two sessions. The earlier residual-kurtosis evidence came from raw data and is retracted.
- **Human propofol EEG (Chennu, 19 channels): no effect, underpowered, not a failed replication.** Every level is within ±0.2 SD. Scalp z is at ceiling (.983), so a macaque-sized relative drop is under one null SD; the scalp cannot resolve ~5 mm neighbour pairs; and it is sedation, not surgical anaesthesia. A real test needs intracranial or high-density EEG under anaesthesia.
- **Wording:** "anaesthetics reversibly create non-Gaussian dependence beyond a linear process with the same spectra and marginals, sparse and pair-level, much weaker in sleep". Not "curved coupling" or "more nonlinear". For propofol the leading form is amplitude-linked decoupling of tight neighbours (inferred).

## Linear against model-free prediction: the TE − GC level tracks the spectrum here (12 November)

- **Readout:** the core level median over edges of `te_kraskov_DCE` − `gc_gaussian` (k = 1). pyspi's `gc_gaussian` is the same TE class with the Gaussian estimator, i.e. Gaussian TE in nats (read from pyspi source), so for Gaussian data the difference is about zero; GC ≡ 2 TE for Gaussian variables (Barnett, Barrett & Seth 2009). In construction validation it was the only non-Gaussian reader that worked (character_axes findings), and on the r10 16-channel NeuroTycho bank its real − IAAFT residual was robust in all three variants (d −3.0, 0/6). The z version (`gc|te`) was insensitive in construction and TE's between-window reliability is .02–.24 here, so z is not the readout.
- **Atlas, raw:** falls under every drug in every session and animal, both arms (2-s: ketamine −.015, ketamine–medetomidine −.027, medetomidine −.020, propofol −.021 nats; 10-s −.010 to −.027); eyes-open +.002; sleep −.003 (deep −.005); fly +.001.
- **But it is spectral concentration plus estimator bias, not yet character:**
  - The drop comes from DCE-kNN TE becoming more negative (−.010 to −.027); GC barely moves (+.000 to +.007). A negative kNN TE is bias.
  - Across all non-reference sessions the drop tracks the fall in channel spectral entropy: r = +.90 (2-s, n = 30) and +.85 (10-s, n = 16), not peak frequency (−.01, −.20). Fly, whose spectral entropy does not fall, shows nothing; sleep, which falls little, shows little.
  - Within-unit spectral adjustment halves it (2-s −.010 to −.016, still 0/4–0/11 except ketamine 1/4), and linear adjustment need not remove a nonlinear bias.
- **Against a spectrum-matched null** (`predictive.py`, job 183723401, `predictive-controls.json`, `*-predictive.json`; pyspi-free, 16 evenly spaced channels, KSG TE with pyspi's AUTO Theiler window against Gaussian TE, three spectrum-final multivariate IAAFT draws per window):
  - **Controls (12 windows each):**
    - linear systems: residual ≈ 0 (fast ±.001);
    - slow linear system: the raw bias is large (−.057 at T = 200, −.034 at T = 1000) and the null reproduces it, leaving −.008 at T = 200 (0/12, so the null is imperfect for slow short windows) and −.001 at T = 1000;
    - quadratic coupling (invisible to GC): on the coupled pairs the residual is **positive**, +.09 to +.13 for fast systems and +.06 for slow ones at T = 1000, but only +.007 at T = 200. A median over all pairs barely moves when few pairs are coupled.
  - **Atlas, residual change against wake (nats):**

    | Arm | Ketamine | Ketamine–medetomidine | Medetomidine | Propofol | Sleep | Eyes open |
    |---|---|---|---|---|---|---|
    | 2-s | −.002 (1/4) | −.006 (0/11, 0/4 animals) | −.004 (0/4) | −.006 (0/4) | +.000 | ≈ 0 |
    | 10-s (pooled) | −.000 (1/4) | −.007 (0/11, 0/4 animals) | −.002 (1/4) | −.008 (0/4) | deep −.001 | ≈ 0 |

    The null reproduces 65–95% of the raw fall (least under propofol and ketamine–medetomidine); normal-score Gaussian TE gives the same residual. Fly: 0.
  - **Reading.** A residual survives under ketamine–medetomidine and propofol, strongest in 10-s windows, where the linear control's artefact is −.001. In 2-s windows it is no larger than the slow-linear control's −.008, so it is not distinguished there. Its sign is opposite to what nonlinear coupling gives in the controls: model-free TE is *below* what a linear process with the same spectrum predicts. Gaussian TE does not differ between real and surrogate, so the kNN side carries it. The likeliest reading is that the target's own non-Gaussian dynamics (for example, sharp or asymmetric slow waves) explain away more of the source's apparent prediction once conditioning is model-free [I]; the deciding check is kNN against Gaussian active information storage, real − IAAFT. It matches the r10 16-channel result (d −3.0, 0/6) and the raw `gc|te` grouping (ketamine–medetomidine, propofol and deep sleep against the single agents).
- **Deciding check: kNN against Gaussian AIS (job 183737970, same outputs; history 1, same targets, distances and Theiler rule as TE; decision rule fixed in `predictive.py` before the run, commit c86ca33). The old TE fields reproduce exactly.**
  - **Verdict: the rule is not met, so the explaining-away reading is not supported, and the TE − GC residual stays a small 10-s effect of unestablished cause.**
    - The threshold-AR control (bounded nonlinear own dynamics, linear coupling) gives the negative TE residual only in its slow version (T = 1000: slow −.035, 0/12; fast +.006, 9/12).
    - Its AIS residual is negative, not positive (fast −.023, slow −.033, 1/12 each).
    - The AIS residual also rises under ketamine and medetomidine (10-s +.013, +.020), where the TE residual is ≈ 0.
  - **Separate exploratory observation, node-level, not a z feature: model-free self-prediction beyond the spectrum-matched null rises under every drug.**
    - Residual AIS_kNN − AIS_Gauss, change against wake:
      - 10-s (pooled): propofol +.062 (4/4), ketamine–medetomidine +.025 (9/11, 4/4 animals), medetomidine +.020 (4/4), ketamine +.021 (3/4).
      - 2-s: propofol +.039 (4/4); others +.008 to +.009.
      - Deep sleep +.007 (2/2); fly −.002. Recovery reverses it.
      - The normal-score variant agrees.
    - Per target, the channels with the largest AIS residual have the most negative incoming-TE residual. Residual association: ketamine–medetomidine −.44 (0/4) and propofol −.41 (0/2), against +.11 and −.01 in the linear controls. It is not specific: ketamine recovery shows −.22 without an AIS rise.
  - **Null quality is imperfect, and how this bears on the result is inferred [I]:**
    - The surrogates under-reproduce lag-1 linear persistence. The `residual:ais_gauss` is +.07 to +.19 in the slow controls and +.02 to +.05 under the drugs, against ≤ .016 in wake. The cause is unknown; it barely shrinks from T = 200 to T = 1000, so it is not mainly end-point mismatch.
    - In the linear-slow control this mismatch pushes the readout down (−.015), and the kNN residual is smaller than the Gaussian one (.056 against .069).
    - Under propofol (10-s) it is .119 against .018, and under ketamine–medetomidine .059 against .028. So the rise sits in the model-free estimate, against the artefact's sign [I].
  - **Alternatives not separated:**
    - nonlinear (for example, stereotyped or time-irreversible) waveform dynamics;
    - within-window state switching, which IAAFT nulls also reject.
  - Propofol rests on 4 sessions in 2 animals, so it is "consistent", not decided. No follow-up is planned.

## Channel-level z and further candidate pairs (12 November)

### Channel-level z

[`local.py`](local.py), job 183765960, `*-local.json`. Raw tier: stored pyspi, no surrogates.

**Method.** For each pair, node-wise z_i is the correlation of A and B over the edges incident to channel i.
- Even and odd windows are contrasted with wake separately.
- The k = 5 channels with the largest even-half change are scored on the odd half, minus the odd half's all-channel mean. This removes the global shift.
- The null is the same pipeline run on the eyes-open contrast.
- The decision rule was fixed in the module before the run.
- Pairs with a partner whose wake reliability is below .3 are skipped (20 of 42 in the 2-s arm).
- Stage A of the 10-s arm had one session per animal per agent, so its within-animal agreement was undefined. The pooled run (`neurotycho-long-local.json`) has the 2-s session counts and is not yet read.

**Results.**
- **Local structure under the drugs is real and reproducible.** For most reliable pairs, the change maps are stable under every drug and not in sleep:
  - split-half .5–.8, against .2–.5 for eyes-open;
  - within-animal agreement across sessions .6–.87, against .2–.45 for eyes-open and sleep.
  - Several pairs have large local changes while the global z barely moves. This is what the global z hides.
- **But nearly every such map tracks a nuisance map**, so under the rule only the literal reading stands:
  - Most track the change in A's spread over incident edges (|r| .5–.9). This is the range restriction of the toy formula; the eyes-open maps track it too (.4–.8). It holds for the MI pair (.6–.8), `plv|pec`, `cov-sq|xcorr-sq_max`, `lcss` and the band-locus pairs.
  - A few track the channel's change in spectral entropy instead.
- **Mediation (`cov-sq|prec-sq`, Empirical) is the clearest local case.**
  - Global Δz is about 0 (−.013 to +.014).
  - Held-out lower-tail excess +.22 to +.31 in every session of every drug (4/4, 11/11, 4/4, 4/4). Split-half .75–.82; within-animal agreement .79–.86 against .30 for eyes-open; sleep 1/7. It does not track A's spread.
  - But it tracks the channel spectral-entropy change: r −.42 under ketamine, −.58 to −.80 under the others.
  - The regularised LedoitWolf version fails the held-out test, although its maps also agree within animal (.60–.74 against .24 for eyes-open).
  - The likeliest reading is that unregularised precision (T = 200, M = 125) degrades most at the spectrally concentrated channels [I]: an estimation effect, localised.
- **Under the rule nothing passes cleanly.** The two exceptions, mediation under ketamine and `plv|pec` under medetomidine, sit at |r| ≈ .42 against the .5 cutoff. The spread covariate does not discriminate on its own: it is .6–.7 on the eyes-open maps too, it flags the known-positive MI pair, and it covers only A. So these results show that raw node-wise z is tied to local spread, not that the local changes lack character. The residualised rerun is below.
- **Residualised rerun** (job 183766906; exploratory, because the readout was changed after the first results). Each half's change map is regressed on the changes in the spread of A, the spread of B and the channel spectral entropy, then held out as before.
  - **Mediation (`cov-sq|prec-sq`, Empirical) keeps a local signature.** Ketamine +.17 (4/4), ketamine–medetomidine +.19 (10/11), propofol +.12 (4/4), medetomidine +.12 (2/4). Global Δz stays about 0.
    - The LedoitWolf version does not keep it (5/11 under ketamine–medetomidine). So it is a change in which channels' unregularised partial dependence departs from their marginal dependence.
    - A precision estimate conditioned on 124 channels from 200 samples may well be what moves [I].
  - **The control pairs also pass under single agents, at tiny size.** Saturation-only: ketamine and medetomidine +.013 to +.015 (4/4). Identity: ketamine +.004 (4/4). So passing the residualised test is not enough on its own: the size has to clear the controls.
    - The low-band `cohmag|plv` passes (ketamine–medetomidine 10/11, propofol 4/4) at .011–.016, which is control size. Not read.
  - **Elsewhere:** scattered single-agent passes (MI kernel, `je`, kNN-MI against lagged kNN-MI, `xcorr`, sign structure under ketamine). No other pair passes under three or more agents.
- **Scope.** Independent review ran the MI pair's fine-grained analysis beyond IAAFT (notes, `review-nature-level.md`). It reports:
  - propofol concentration that holds out;
  - edge maps that do not reproduce across sessions;
  - ketamine–medetomidine diffuse.

### Further candidate pairs (now in `PAIRS`; NeuroTycho `*-pairs.json` regenerated, fly still has the 42-pair file)

2-s arm, raw (spectral) Δz.

- **Sign structure, z(r, r²) and z(ρ, ρ²).** Wake z = .49; eyes-open SD .037.
  - Falls under every drug, every session: ketamine −.20, ketamine–medetomidine −.17 (0/11), medetomidine −.13, propofol −.15. Spectral adjustment still gives −.07 to −.22 (0–1 sessions positive).
  - Sleep −.003. The 10-s arm is weaker: ketamine −.20, the others −.05 to −.09 with mixed sessions, deep sleep +.01.
  - Mean r is near 0: .03 in wake and about .00 under ketamine–medetomidine, medetomidine and propofol. The common-average reference pushes it toward 0 [I]. About half the edges are negative.
  - So z(r, r²) reads whether the strongly coupled pairs are in-phase (wake) or include strong anti-correlations (drugs). It is a property of the across-pair distribution of r alone, so it does not need SPI–SPI.
  - **Sign-structure null** (`discovery.py sign-null`, job 183766906, `*-sign-null.json`): z of each unit-state's window-mean r profile (noise averaged out), and stationary Gaussian windows with the state's mean cross-spectrum.
    - Under ketamine–medetomidine, medetomidine and propofol, the per-window fall is noise. The null reproduces it (−.169 against −.165 under ketamine–medetomidine), and the window-mean profile does not fall (−.005, −.030, +.039).
    - So it is fewer effective samples making r noisier, with no anticorrelation structure.
    - Ketamine is different: its window-mean profile falls as much as the per-window z (2-s −.193, 1/4 positive; 10-s −.208, 1/4). Under ketamine, strongly anticorrelated pairs are part of the mean coupling pattern.
  - **Unresolved for ketamine:** whether the reference produces it. A stronger, spatially graded common mode becomes a dipole of positive and negative correlations after the common-average reference. Deciding test: the same z on a reference-free montage (bipolar or Laplacian) from the raw recordings.
- **Saturation control, z(r², Gaussian MI).** These are a deterministic monotone function of each other, so any change is pure strength-distribution mechanics.
  - Under every drug it moves by −.010 (0/4, 0/11, 0/4, 1/4 positive).
  - That is the same size as the raw full-bank change of the rank pairs (`cov|spearmanr` −.003 to −.011) and of `cov-sq|dcorr` (−.006 to −.016). Changes of about .01 at the ceiling therefore happen mechanically, and those raw changes are not distinguishable from saturation without a surrogate. Independent review's beyond-IAAFT form ladder is the version to read.
- **Amplitude-weighted against amplitude-free synchrony, high band (`cohmag|plv`, 25–50 Hz).**
  - Ketamine +.06 (4/4), ketamine–medetomidine +.04 (8/11), propofol +.03 (4/4).
  - Medetomidine −.055 (0/4): the medetomidine exception again, as for band locus and MinCovDet.
  - Broad and low bands are at ceiling (z₀ .92–.96). The 10-s arm raw disagrees.
- **Burst against proportional amplitude coupling, orthogonalised (`pec_orth|pec_orth_log`).**
  - +.04 to +.05 under every drug, every session (11/11 under ketamine–medetomidine); sleep +.009; eyes-open SD .009.
  - But `pec_orth` is unreliable between windows (.03), and a log transform also reads envelope marginal shape. Not read.
- **Direct against marginal with partial correlation.** `pcorr` Empirical falls under ketamine–medetomidine, medetomidine and propofol (−.06 to −.10, 0 positive); ketamine −.014 (1/4). But it is node-additive (R² .63–.78). The LedoitWolf version is mixed. Not read.
- **No signal:** `psi|gc` (phase lead against predictive direction), and `xcorr` max against mean (null SD .07).

## Assumption classes over the whole atlas (13 November)

[`assumptions.py`](assumptions.py), job 183770335, `*-assumptions.json`; class table `assumption-classes.json` and reading tiers committed (740c677) before the run. Each valid SPI is parsed into attributes (statistic, estimator, sign handling, lag, band, aggregation, options); pairs differing in exactly one attribute are grouped by that difference: 584 minimal pairs in 100 classes, 25 with a curated character reading. Edge-profile reliability now covers all 222 valid SPIs (`*-reliability-all.json`). Raw tier, stored pyspi.

- **Forward (classes):** no character class converges across its members under the drugs at 2 s; the one convergent class is the Kraskov autocorrelation correction (estimator diagnostic).
  - **Ceiling classes, against the saturation control per condition** (2-s raw Δz, KT / KTMD / MD / PF): saturation control −.011 / −.010 / −.010 / −.004; squared rank against squared linear −.015 / −.023 / −.007 / −.018; L1 against L2 distance −.010 / −.017 / −.005 / −.013; distance correlation against r² −.009 / −.016 / −.006 / −.012. So under KTMD they are 1.6–2.3× the control and under PF 3–4.5×, KT about 1×, MD below it: the MI pair's condition order (KTMD −.076, PF −.075, MD −.029, KT −.012) across four statistics of non-Gaussian form [I]. At 10 s only PF clears the control (−.005 against −.002). One control pair sets the bar and these classes have 1–2 usable members each, so this is a hint that agrees with independent review's beyond-IAAFT form ladder, not a separate result.
  - **Non-Gaussian dependence does not converge across statistics.** Of the Gaussian against bias-corrected kNN members, only MI moves (KTMD −.076, PF −.075, MD −.029, KT −.012); lagged MI does not (−.025 to +.052) and the TE partner is unreliable (.03). Gaussian against kernel members fall more (2-s `je`, `ce` −.03 to −.12; 10-s `si` −.15 to −.46, deep sleep −.10), but the kernel estimators have no autocorrelation correction. So the MI reading stays specific to same-sample MI [I].
  - **Sign handling** (37 members) does not converge; only the covariance members move, already read as sampling noise except under ketamine (10-s −.22).
  - **Robust against classical** converges at 10 s only: `cov-sq` Empirical against MinCovDet −.085 (KT), −.119 (KTMD), −.075 (MD), −.149 (PF) in stage A, pooled −.076, −.111, −.054, −.148; graded in sleep (pooled deep −.040, mid −.014). Spectral adjustment (pooled) keeps KT and KTMD (−.073, −.087) and removes MD (−.020) and PF (+.056). At 2 s, ≈ 0. Matches the earlier explore result; the MinCovDet flag analysis found flag clustering spectral.
  - Lag classes (zero against lagged, adjacent lags, MI against lagged MI) rise about 2 SD under every drug, not sleep, and shrink after spectral adjustment: slower signals.
- **Reverse (top coordinates):** 1,323 of 22,406 usable 2-s coordinates are consistent under all four drugs (raw and spectral, 75% of sessions, every animal); at 10 s, 36. At 2 s nearly every top coordinate links through one class, **best lag against zero lag** (`xcorr_max` against r): signed −.14 to −.25 under every drug (every session), sleep −.019; 10-s −.10 to −.19, deep sleep −.03; reliability ≥ .68.
  - **At 2 s it is mostly a spectral estimation floor; at 10 s it is mostly stable mean lag.** Lag null (`discovery.py lag-null`, job 183766068, `*-lag-null.json`; squared pair `cov2|xcorr2_max`; 32 channels, 3 draws):
    - 2 s: the zero-phase null (state spectrum, no lag) reproduces most of the fall (KTMD −.129 of −.150, PF −.115 of −.149, MD −.057 of −.104, KT −.038 of −.070). The mean-lag term (stationary minus zero-phase null) is within the null-of-null bias (≤ .014) except under MD (−.053, 0/4). Residual |Δ| ≤ .017.
    - 10 s (pooled): under KT and MD the floor is small (−.003, −.004 of −.033, −.046) and the mean-lag term carries most of it (−.030, −.031). Under KTMD floor and mean lag are about equal (−.034, −.032) and under PF the floor exceeds it (−.038, −.017). No unit is positive under any drug; deep sleep −.006; eyes-open ≤ +.004; bias ≤ .005; residual ≈ 0.
    - **Character reading (candidate, linear tier):** under every drug, coupling at a stable non-zero lag reorders which pairs are strongest at their best lag relative to zero lag. It lives in the state's mean cross-spectrum (phases), so it is a linear-stationary property, not non-Gaussian. Two units per agent, 3 draws, 32 channels: a candidate. The signed pair was not nulled; it adds sign flips of anticorrelated pairs at half-period lags [I].
- **Reading.** Forward, no class converges across statistics at 2 s except an estimator diagnostic; the ceiling form classes follow the MI pair's condition order at 1.6–4.5× the saturation control under KTMD and PF (hint). Reverse, the top class (best lag against zero lag) is floor at 2 s and stable mean lag at 10 s, a linear timing candidate. Time-heterogeneous dependence (robust against classical) converges at 10 s, graded with sleep depth. The second reverse pass converges on frequency weighting of coupling (r² against uniform low-band coherence): every drug, not sleep, half structure in the state's mean cross-spectrum (below).

- **Reverse rankings, second pass** (job 183771638; near-identical SPIs collapsed at |z₀| ≥ .99, `xcorr` and uncorrected Kraskov dropped, both partners reliable ≥ .3; 2,200 usable 2-s coordinates, 3,385 at 10 s). A condition counts when raw and spectral-adjusted effects agree in ≥ 75% of sessions and every animal, beyond 3 SD.
  - 2 s: 100 coordinates under all four drugs, 121 under all but one, 197 drugs-not-sleep, none single-agent, none drugs-and-deep-sleep. 10 s (pooled, 3,550 usable): 11, 35, 15; single-agent only ketamine (7). Stage A had 1, 23, 7, ketamine 6 and propofol 1.
  - **Pooled, 8 of the 11 10-s all-drug coordinates are on the 2-s link:** 0–25 Hz `cohmag`/`plv` against kernel MI, `cov-sq`, `je`, `ids` or `pec_log` (e.g. `cohmag` low against `pec_log` −.14 to −.21 under every drug, deep sleep −.04, not consistent). The other three (`di_gaussian|mi_kernel`, `si_gaussian|mi_kernel`, `tlmi_kernel|pec`) involve no coherence.
  - **The 2-s rankings converge on one link: zero-lag time-domain coupling against uniform low-band coherence.** `cohmag` or `plv` mean over 0–25 Hz against r², ρ², squared MinCovDet covariance, `dcorr`, `ids` or kNN MI: −.09 to −.18 under every drug (every session), sleep −.011 to −.019; reliability ≥ .65. The two differ in power weighting across frequency and in lag sensitivity (r² uses the real part).
    - **Weighting null** (`assumptions.py weighting-null`, job 183772042, `*-weighting-null.json`; pyspi-free, 64 channels, 2 draws). Chain r² → phase-free power-weighted coherence (lag link) → uniform 0–25 Hz coherence (weighting link), per window and on window-mean profiles (noise averaged out), against zero-phase and stationary nulls with each state's mean cross-spectrum.
    - The chain reproduces the atlas effect: z(r², low-band coherence) −.10 (KT), −.18 (KTMD), −.16 (MD), −.18 (PF), every session; sleep −.016; 10-s pooled −.10 to −.15 (every session), deep sleep −.033.
    - **No lag:** the zero-phase null reproduces all of it (−.09 to −.22). The lag link is small and also reproduced by the null.
    - **About half is per-window noise, half is structure:** the window-mean profile z still falls −.07 to −.11 under every drug (every unit, both arms), against +.012 in 2-s sleep and −.007 in 10-s deep sleep. The weighting link carries it (mean profile −.06 to −.12). The stationary null's mean profile matches the real one, so it is a property of the state's mean cross-spectrum (linear-spectral tier). Residual noise in the mean profile is not fully excluded at 20–60 windows per state [I].
    - **Character reading (linear-spectral candidate):** under every anaesthetic, and not in sleep, the pairs coupled at the dominant (high-power, slow) frequencies stop being the pairs coupled across 0–25 Hz: the frequency locus of coupling becomes pair-specific. It needs pairs whose coherence spectra differ in shape, not only a channel-power change, and deep sleep (also slow) shows little of it. Related to band locus (above), which excludes medetomidine; this one does not.
    - Literal reading: the across-pair agreement between power-weighted zero-lag r² and frequency-uniform low-band coherence falls, through the frequency weighting, not through lag.
    - **pyspi null banks (lag section):** the stationary null gives 75–90% of the `cov-sq|cohmag` fall; net of bias the remainder (−.011 to −.025) is at the eyes-open size and also in sleep, so beyond the spectrum it is not drug-specific.
  - Also recurring: Gaussian `je`/`si`/`di` combinations (fixed functions of linear-Gaussian structure and autocorrelation) and `canberra`/`braycurtis` distances (sample-weighted metrics without a clean reading on signed data). Literal only.
  - 10 s: `lcss` against `plv`/`cohmag` (KT, KTMD, PF −.09 to −.14; MD −.04); ketamine-only hits on `bary_euclidean_max` and `braycurtis` (−.16 to +.31), scattered.

## Edge-profile reliability (job 183670442, `*-reliability.json`)

Reliability here is the mean correlation of an SPI's edge profile between windows of one unit and state. It mixes estimator noise with real window-to-window change of the coupling pattern. Disattenuating by it overcorrects: in wake the disattenuated z of the MI pair exceeds 1. So it bounds, but does not decide.

- **Profiles become less reproducible between windows under every drug, not in sleep.** This is a result in its own right, not a z feature.
  - 2-s arm, ketamine–medetomidine and propofol: Gaussian MI .82 → about .60, kNN MI .69 → about .46, distance correlation −.26, `xcorr-sq_max` −.33. Sleep changes by −.03 to −.06.
  - The 10-s arm agrees, at about half the size.
  - **Resolved: it is the spectrum, not the coupling** (`*-spectral-null.json`, `discovery.py spectral-null`, pyspi-free). For each unit and state, the between-window agreement of r and r² profiles is set against that of stationary Gaussian windows with the state's mean cross-spectrum (same T and count, 10 draws). The null is biased slightly high (its common pattern is a mean of about ten windows), so excess contrasts within about ±.03 are undecided.
    - 2-s arm, r² agreement drop against wake, real against null: ketamine −.125/−.090, ketamine–medetomidine −.261/−.240, medetomidine −.176/−.150, propofol −.277/−.227. So 75–90% of the drop is what a stationary process with the drug spectrum gives. Excess drops are −.02 to −.05 (largest under propofol).
    - 10-s arm (pooled): reproduced for ketamine–medetomidine (−.118/−.106) and propofol (−.122/−.114); ketamine's null drop exceeds the real one; medetomidine is undecided (−.068/−.051, excess −.017, 1/4).
    - Natural sleep is the exception: 2-s −.052/−.020; 10-s deep −.069/−.028, mid −.055/−.013. Its drop is mostly not spectral. The plainest explanation is stage mixing inside one state label (the 2-s sleep state is unstaged, and the 10-s tertiles are coarse); real window-to-window change of the coupling pattern and non-Gaussian amplitude structure are the alternatives, which the null does not separate.
    - So the earlier reading, "temporal non-stationarity of dependence under anaesthesia", is withdrawn. Disattenuation by between-window reliability is better justified under the drugs than first thought, since the drop is mostly estimator noise.
  - **Drift checked** (`*-stability.json`, job 183697156): agreement compared with wake within matched separations (0–150, 150–400, over 400 s). Drug windows are spaced about 120 s apart and wake about 60 s, so unmatched comparisons are biased.
    - In 2-s windows the drop is already full at the shortest separation and barely grows with distance: Gaussian MI, propofol −.21 / −.23 / −.24; ketamine–medetomidine −.20 / −.21 / −.23.
    - So it is neither slow drift in depth nor a spacing artefact. In 10-s windows it is about half the size and grows a little with separation (some drift there).
  - **Fewer effective samples (confirmed by the spectral null above).**
    - Channel-mean spectral entropy falls far more under the drugs (−.15 to −.22) than in sleep (−.04), although peak frequency falls similarly.
    - The ordering of the entropy fall (ketamine–medetomidine > propofol > medetomidine > ketamine) roughly matches the ordering of the reliability fall.
    - A spectrally concentrated 2-s window carries fewer independent samples, so its profiles are noisier. That is estimator noise, not coupling fluctuation.
    - Decisive and pyspi-free: compare the between-window agreement of r profiles on real windows with that on multivariate phase-randomised surrogates of the same windows. If the surrogates lose as much agreement, it is the spectrum.
- **MI pair (Gaussian against kNN): unresolved on the full bank.**
  - Gaussian MI, a function of r with no Theiler window, loses reliability exactly as much as kNN MI. So a kNN-specific DCE artefact is not supported.
  - But the raw z fall (−.03 to −.08) is far smaller than independent noise at these reliabilities would produce (about .91 → .64). So most of the reliability drop is shared, real non-stationarity, and the z fall cannot be attributed to noise or to character from this bank.
  - The IAAFT version, whose surrogates share window length and autocorrelation, stays the anchor. The kNN-to-Gaussian level collapse is still unexplained.
- **Lag pair:** in the 10-s arm (`icoh` reliability .46) z(`cohmag`, `icoh`) rises in every animal under every drug raw, and disattenuated by +.17 to +.24; but disattenuation does not touch the estimation floor, which the lag section shows carries most of the rise. The low between-window reliability of `dswpli` (.03) is compatible with window-to-window lag change, but does not show it.
- **Demoted:**
  - Partners with near-zero between-window reliability: `dswpli` (.03), `pec_orth` (.03), `te_kraskov_DCE` (.03–.07), `icoh` in 2-s windows (.18). The `pec|pec_orth` and TE `k-1|k-2` effects mostly carry within-window structure, and their disattenuated values swing in sign; the TE memory pair is not interpretable. `ppc|dswpli` is the exception: its rise survives the lag null (lag section).
  - The best-lag/zero-lag `xcorr` fall: an estimation floor (lag null), not lag.

## Other named pairs (42 pre-listed pairs; `*-pairs.json`)

- **Consistent under all four drugs in both arms, sleep ≈ 0:**
  - **Interaction memory:** TE with one against two samples of target history (10 against 20 ms), `te_kraskov_DCE k-1|k-2`. It rises by +3.7 to +5.0 SD spectrally adjusted and is flat in sleep. Read literally: one more sample of target history changes the pair ranking less. It is sensitive to autocorrelation structure, so a character reading is not earned.
  - **Rank against Pearson:**
    - `cov|spearmanr`, `cov-sq|spearmanr-sq`, `corr_pearson_tau-1|corr_spearman_tau-1` and `pdist_euclidean|cityblock` fall by −0.01 to −0.02.
    - These are ceiling pairs (z₀ ≈ .99): large in SD, tiny in z.
    - Without surrogates they read marginal shape first; earlier work found the raw linear–rank readout tracks marginal shape.
- **Consistent across drugs, but deep sleep also moves:**
  - **Phase against amplitude coupling** (`plv|pec`): −4 to −9 SD in both arms; 10-s pooled deep tertile −2.6 SD.
  - **Bulk against atypical samples** (10-s arm only, pooled): `cov-sq` Empirical against MinCovDet −.05 to −.15 (−2.9 to −7.9 SD), deep tertile −.04 (−2.1); `prec-sq` LedoitWolf against MinCovDet −.09 to −.31 but only −1.4 to −4.8 SD, because its pooled null SD is .064, not stage A's .019. This matches the macaque MinCovDet flag result.
  - **Rigid against elastic alignment** (`dtw|dtw_sakoe-chiba`): −.05 to −.17 (−3.7 to −13.2 SD) in the 10-s arm only, deep tertile −3.1; spectral adjustment removes it (|Δz| ≤ .04, mixed sign). The 2-s windows are too short for warping to matter. The warp axis failed construction validation.
  - 10-s entries are raw effects on 4/11/4/4 sessions (2/4/2/2 animals).
- **Subsets of conditions** (relaxed: consistent sessions and animals within a condition, not across all four drugs; descriptive):
  - **Band locus** (`cohmag` and `plv`, 0–25 against 25–50 Hz), a 2-s result (10-s raw ≈ 0): rises under ketamine, ketamine–medetomidine and propofol (2-s raw and spectral, 4/4, 10/11–11/11, 4/4), not under medetomidine (−.03 to +.00) or sleep. Fly rises too (11/13, 10/13). The pairs strongly coupled at low frequency become more the same pairs as those coupled at high frequency. The medetomidine exception matches the MinCovDet one. Against the pyspi null banks (lag section), the beyond-spectrum part is 60–75% as large in eyes-open, does not reverse at recovery, and is negative under medetomidine and sleep: not specific to the drug state.
  - **Precedence, conditioning link** `tlmi_gaussian|gc_gaussian` (lag-1 Gaussian MI against Gaussian GC: they differ only in GC conditioning on the target's own past). Raw, z₀ = .26, null SD .04: falls under ketamine (−.03 2-s 0/4; −.05 10-s 0/2), medetomidine (−.09 0/4; −.08 0/2) and ketamine–medetomidine (−.10 0/11; −.10 1/4); propofol ≈ 0 (−.01; +.03 2/2); sleep ≈ 0. GC's reliability rises under ketamine and medetomidine alone, so attenuation does not produce their fall; under ketamine–medetomidine both partners lose reliability (GC .14 → .10, lagged MI .63 → .50), so attenuation is not excluded there. Ketamine's fall is about one null SD. The notable fact is propofol holding at ≈ 0 despite GC's largest reliability loss (.15 → .05). GC's 2-s reliability is .05–.28, so the 10-s arm (wake .44–.52) is the one to read.
  - **Precedence, propofol (and 10-s deep sleep):** `mi_gaussian|gc_gaussian` +.05 raw (2-s 4/4; 10-s +.08 2/2), +.12 spectral at 2 s (4/4); 10-s deep sleep also rises (+.055, 2/2). `cohmag|sgc` +.12 raw (4/4) at 2 s, but `sgc`'s between-window reliability is .05–.19 in both arms, so it gets the literal reading only. Under propofol the pairs coupled at zero lag are more often the pairs with precedence; under ketamine and medetomidine the two rankings drift apart. This fits the propofol-only high-band lag (explore section).
  - **Linear against model-free prediction** `gc_gaussian|te_kraskov_DCE`, raw, both arms: ketamine and medetomidine alone up (2-s/10-s +.06/+.16 and +.01/+.05; 3/4 and 2/2 sessions), ketamine–medetomidine, propofol and 10-s deep sleep down (−.06 to −.12, every session). Spectral adjustment scrambles it, and the TE partner's between-window reliability is .02–.24, so only the literal reading stands.
- **Inconsistent or null:** broadband against peaked, `sgc` order, `gc` against AIS, and macaque mediation.
- **Fly mediation:** `cov-sq|prec-sq` +.105 (13/13), and the LedoitWolf version +.076 (12/13). This is fly-specific.
- **Structural zeros, dropped:** `plv|pli` and `plv|wpli`. PLI and wPLI are stored antisymmetric, so their z with a symmetric SPI is exactly 0.
- **dcoh/pdcoh:** not testable here, because the spectral-directed SPIs are not finite in every 2-s window.

## Screen over every z coordinate (`*-screen*.json`)

- **2-s arm:** 1,264 of 19,503 usable coordinates pass. To pass, a coordinate needs the same sign in at least 75% of each agent's sessions, raw and adjusted, and a mean beyond 3 eyes-open null SDs. Node-additive SPIs are excluded.
- **Top families:**
  - `xcorr-sq_max` against any zero-lag statistic: an estimation floor, not lag (lag null, above);
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
  - Band edge: `prepare.py` low-passes at 40 Hz (order-8 Butterworth) before resampling to 100 Hz, so "25–50 Hz" is effectively 25–40 Hz plus roll-off; bins above 40 Hz carry little signal.
  - z(`plv` hi, `dswpli` hi): +.13 under propofol in both animals (2-s +.10/+.17; 10-s stage A +.13/+.12). Pooled 10-s, the top propofol-specific coordinate is broadband `plv` max against high-band `dswpli` (+.076, +7.6 SD; other drugs ≤ +1.2 SD). The other drugs are at most +.03, sleep +.005, and recovery largely reverses it (+.01 to +.05).
  - The level moves too: the `dswpli/ppc` share in the high band rises +.33 (2-s) and +.30 (10-s) under propofol, in both animals, against at most +.07 for the other drugs.
  - Many coordinates on `dswpli`/`dspli`/`icoh` in the high band rank propofol-specific in both arms, at +6 to +9 null SD.
  - Reading: under propofol, but not ketamine, medetomidine or sleep, a reproducible non-zero-lag 25–50 Hz phase coupling emerges on the pairs with strong high-band locking. Physiology is not established [S].
- **All but medetomidine: robust against classical dependence** (10-s arm). For example, z(`cov-sq_MinCovDet`, `mi_kraskov_DCE`) falls by 6.6–10.9 SD (pooled) under ketamine, ketamine–medetomidine and propofol, and by −1.6 SD under medetomidine (stage A +0.1). This is the same exception as the pyspi-free MinCovDet flag fraction (borderline under medetomidine), so two independent methods agree.
- **Weaker or uninterpretable:**
  - Ketamine-only and medetomidine-only hits are scattered across unrelated families.
  - Opposite-sign pairs mostly involve noisy partners: `dswpli` broad against low band (ketamine against ketamine–medetomidine), non-DCE TE, and `psi`.
  - Deep-sleep opposites rest on non-DCE `te_kraskov`.
- **Fly:** nothing beyond d ≈ 1.7 among 13 flies.
