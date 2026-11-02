# Character axes: independent construction review (2 November 2026)

Scope: the 13 axes, levels and controls in [`character_axes.py`](../../../src/mtsspi/character_axes.py) at `124324f`, read against pyspi source in `.venv` and eight audited banks (NeuroTycho M16, CHB-MIT M16, HMC M8, LANL M4, AD/FTD, LEMON and propofol M19; IAAFT surrogates for NeuroTycho, CHB-MIT, LANL). No pyspi runs, no Gadi writes. Analysis scripts are in the shared session-notes folder (`axes-review-scripts/`), not in the repo. Effects are unit-level: paired over sessions or participants where states are within-unit, otherwise between participants. ICC is the window-level one-way ICC across units × states. Literature citations are from memory and were not re-checked this session.

## Three findings that change current claims

1. **The NeuroTycho non-Gaussian level effect reverses with the autocorrelation-corrected estimator.**
   - The level uses Kraskov MI minus Gaussian MI, median over edges, anaesthesia minus awake, over 15 sessions.
   - With `mi_kraskov_NN-4`: real +0.196 nats (15/15, d 2.17), surrogate +0.106, residual +0.090 (15/15, d 2.42).
   - With `mi_kraskov_NN-4_DCE-AUTO`: real −0.059 (15/15, d −1.98), surrogate −0.028, residual −0.030 (15/15, d −2.15). TE minus Gaussian GC flips the same way (+0.031 → −0.033).
   - The estimator control already flagged this. On NeuroTycho, z(KSG, KSG-DCE) is 0.39 (0.98 on CHB-MIT) and shifts with state at d −2.97, but the axis ignores it.
   - Both estimators track window strength (ρ with mean |r|: +0.90 without the correction, −0.84 with it). That fits two known Kraskov biases: autocorrelated samples inflate MI without a Theiler window, and strongly dependent variables pull it negative.
   - IAAFT keeps spectra and marginals but not waveform shape, so it does not cancel either bias.
   - Verdict: the sign of "non-Gaussian dependence under anaesthesia" depends on the estimator, so it is not identified. Do not report it until real, surrogate and estimator control agree in sign.
2. **Organisation changes with how reliable the edge profiles are, not only with character.**
   - z_ab is a correlation of two noisy profiles, so it is attenuated by each profile's unreliability (Spearman's correction for attenuation).
   - On NeuroTycho, window-to-window profile reliability (cross-window profile correlation within a cell) rises under anaesthesia: icoh 0.04→0.22, dsPLI 0.04→0.21, PPC 0.16→0.34, high-band coherence 0.05→0.18, TE k1 0.14→0.27 (paired d 0.96–2.35).
   - Divided by √(rel_a·rel_b), cell-level state effects shrink:

     | Pair | d before | d after |
     |---|---|---|
     | history (TE k1\|k2) | 2.79 | 0.53 |
     | predictive | 3.36 | 0.75 |
     | band | 2.67 | 0.67 |
     | phase locking (PPC\|dsPLI) | 2.15 | 0.13 |
     | mediation | −2.15 | −0.67 |
     | fixed lag | 1.38 | 1.32 |
     | warping | 2.14 | 1.30 |

   - Coupling to strength spread falls with them (history ρ 0.88→0.14).
   - Same-window z also contains correlated estimation noise: raw PPC|dsPLI z is 0.71, against √(rel_a·rel_b) = 0.14.
   - Caveat: cross-window reliability counts genuine nonstationarity as noise, so the corrected values are conservative. Either way, raw organisation differences of this size are not evidence that a property became more heterogeneous across pairs. The surviving "phase-lag organisation" residual in findings.md is the most exposed claim.
3. **The identity control is a strength-range control, not a numerical floor.**
   - Two of its pairs are exact identities with zero variance (`je_gaussian`|`mi_gaussian`, cov|OAS).
   - The other two are deterministic but nonlinear maps (r² vs Gaussian MI, r vs Euclidean distance). On NeuroTycho they move with state at d −1.19 and track strength spread (ρ −0.84).
   - Any pair whose null relation is nonlinear moves by this much with no change in character. Use it as the yardstick for "organisation effect beyond strength range".

## Q1 Construction: is the minimal-pair rule sound?

**Verdict.**

- The rule is necessary but not sufficient, and as implemented it is violated in 9 of 13 axes.
- "One assumption apart" must also mean matched in five respects: symmetry/direction, exponent (squared or not), sign handling, frequency resolution and estimator bias class. Even then, z_ab still moves with profile reliability, strength range (through any nonlinearity of the null map) and shared same-window noise.
- Mean over pairs is only valid when the pairs share a sign convention and a construct.

**Implementation facts from pyspi source (verified):**

- `corr_*_tau-k` is the even part of the cross-correlation, ½[r(k) + r(−k)].
- `xcorr_max` is the signed maximum over ±T/4 lags, not max |r|.
- `xpdist_euclidean_tau-1_min` minimises over shifts of 0 and ±1 sample only.
- `tlmi` is directed at lag 1.
- `cohmag` is |C|² but `icoh` is |Im C|.
- PLV/PLI/PPC average over the ~5 multitaper tapers of a single window, so there are ~5 observations per frequency.
- DTF, dDTF, DC, PDC and gPDC are fitted on each two-channel subsystem (`NonparametricSpectralBivariate`). Spectral Granger uses pairwise AR fits.
- `pec` is a broadband Hilbert envelope.
- The DCE Theiler window is 2·⟨acf_i, acf_j⟩, so it is set per pair.

**Exact duplicates found:** `dtf` ≡ `pdcoh`, the bivariate identity (z = 1.00000 in every window). Also `cov_MinCovDet` ≡ `cov_EllipticEnvelope` and `prec_MinCovDet` ≡ `prec_EllipticEnvelope`. `dcoh` ≈ `gpdcoh` (0.98–0.99).

| Axis | Verdict | Main defect (evidence) | Fix inside p90 |
|---|---|---|---|
| form_monotone | demote to control | Saturated: z 0.96–1.00 everywhere, ICC 0.03–0.20 in AD/FTD and LEMON. 32 pairs collapse to 1.3–4.6 effective. Correlates with tail at 0.56–0.75, which is its own stated confound. | Keep r\|ρ and r²\|ρ² as a marginal/outlier control. |
| form_information | keep, fix | Kraskov without DCE (finding 1). Fixed-width kernel MI/TE is not a valid comparator: gc\|te_kernel z −0.09, mi_kernel\|te_kernel −0.28. Under Gaussianity r²\|dCor is a nonlinear map, saturated at 0.97–1.00. | mi_g\|mi_ksgDCE, tlmi_g\|tlmi_ksgDCE, gc_k1\|te_ksgDCE_k1; level in nats against surrogate and estimator control. |
| timing_fixed_lag | recast or drop | Lags τ ≤ 3 saturate on NeuroTycho (1.00). Lags disagree in sign in EEG (inter-pair r down to −0.74; LEMON r̄ 0.07), so averaging cancels. mi\|tlmi is lag 1 with ≤6% asymmetry, saturated. Signed and squared variants duplicate each other. | One pair cov²\|r²(τ*) with τ* fixed in milliseconds per cohort, named lag-profile/spectral (the cos 2πfτ calibration). |
| timing_best_shift | broken | The three pairs disagree (r̄ 0.05 on NeuroTycho). Euclidean vs xpdist(τ=1) is 0.99–1.00. r\|xcorr_max mixes in the signed-max artefact (0.22 versus 0.75 squared). r²\|xcorr²_max saturates in EEG: zero-lag share level 1.000 in all three cohorts. For narrowband signals the max over lags approaches \|analytic CCF\|, so this converges on phase_lag. | Keep r²\|xcorr²_max only; level as zero-lag share r²/xcorr²_max ∈ [0,1]; not core. |
| timing_warping | relabel | Its "rigid shift" baseline is effectively Euclidean, since xpdist(τ=1) ≈ Euclidean. The pairs mix shift and warp. The barycentre-max pair reads co-occurrence of extremes. Euclidean\|DTW-SC and DTW-SC\|DTW anti-correlate (r −0.68 on NeuroTycho). | Euclidean\|DTW Sakoe–Chiba (warping window 0 vs auto) as rigid↔elastic; DTW-SC\|DTW as a separate warping-extent readout. |
| phase_lag | keep, split | Exponent mismatch. PLV/PLI over ~5 tapers: plv\|pli z 0.07–0.20, ICC ≈ 0.2. Max-over-frequency pairs have ICC 0.03–0.30. dsPLI profile reliability is 0.01–0.04 in the AD/FTD and propofol cohorts. coh\|icoh is +0.53 in ECoG but −0.39 to −0.96 in EEG (volume conduction). The level icoh/cohmag = \|sin φ\|/\|C\| is unbounded (NeuroTycho median 1.25). | Coherency sub-axis cohmag\|icoh (mean, B/L/H) for the core; a separately named phase-locking sub-axis ppc\|dspli and ppc\|dswpli, gated on reliability; level icoh/√cohmag = \|sin φ\|. |
| predictive | rebuild | Pairs differ in two assumptions (contemporaneous→predictive and symmetric→directed). A symmetric-vs-directed z is capped at √(1 − asymmetry share): 0.62–0.85 for GC, 0.74–0.85 for sGC. The kernel pair is negative. | Directed vs directed at matched lag: tlmi_g\|gc_k1 and tlmi_ksgDCE\|te_ksgDCE_k1. On NeuroTycho it is linear-spectral (surrogate d 3.03 vs real 3.36) and its TE profiles are unreliable in cohorts (0.03–0.33), so extended set only. |
| history | trim | Spectral GC order 1 vs 20 is noise (ICC ≤ 0.13, negative on NeuroTycho); an order-1 AR reads model misspecification. TE-DCE k1\|k2 is the strongest NeuroTycho pair (ICC 0.90, d 2.79), but 0.53 after disattenuation. | TE-DCE k1\|k2 and gc_k1\|gc_AIS; extended set only. |
| mediation | broken | cov\|prec (z −0.83) and cov²\|prec² (+0.81) cancel in the mean (r̄ ≈ 0, minimum −0.86). The 12 spectral pairs involve no conditioning. dtf\|ddtf reads coherence weighting. dcoh\|pdcoh reads unequal innovation variances (DC = √DTF when they are equal). Empirical and Ledoit–Wolf agree at r 0.93–1.00 on NeuroTycho, CHB-MIT, HMC and propofol, but only 0.68 on LEMON and 0.17 on AD/FTD (plausibly rank-deficient cleaned EEG; inferred, not checked). | cov²\|prec² Ledoit–Wolf only; fixed montage and M. |
| band | trim | Max pairs are noise (ICC 0.01–0.33). dtf L\|H = pdcoh L\|H. coh L\|H = ppc L\|H. sGC L\|H has ICC 0.06–0.42 in EEG. 22 pairs give 5–9 effective. Bands are fixed fractions of fs, so at 100 Hz with a 45 Hz low-pass "high" means 25–45 Hz. | cohmag mean L\|H; share level. |
| spectral_peak | drop | Max over frequency is unreliable (ICC 0–0.3 in EEG). Contains the dtf/pdcoh duplicate. Correlates with band at 0.40–0.70. | sGC mean\|max at most, as exploratory. |
| phase_amplitude | exploratory | coh\|plv saturates (0.99–1.00 in EEG). plv\|pec differ in frequency resolution and estimator as well as phase vs amplitude. Waveform-shape confound. | dswpli\|pec_orth only if reliable; not core. |
| tail | demote to control | Duplicate estimators. Saturated in cohorts (0.92–0.99). Redundant with form_monotone. | cov\|cov_MCD and Euclidean\|Chebyshev as the outlier control for form. |

Missing minimal pairs that exist in p90: the three DCE pairs above; tlmi\|te at matched lag; Euclidean\|DTW-SC; DTW-SC\|DTW; LCSS-SC\|LCSS (same estimator, warping window only); Gaussian vs Kozachenko for SI, XME and CCE (0.69–0.99 in EEG cohorts, mostly near saturation); and the Graphical-Lasso precision. Conditioned spectral statistics (partial coherence, multivariate DTF/PDC) and stationarity-resolved statistics do not exist in p90, so spectral mediation and intermittency axes cannot be built here.

## Q2 Selection: how representative are the chosen pairs?

**Verdict.**

- The 167 pairs are a hypothesis set, not a summary of z, and they are not representative of it.
- Internal consistency is inflated because pairs share statistics.
- Separation is adequate except for three redundant blocks.
- Hand selection should remain the generator of named axes, but it needs objective gates.

Out-of-fold variance of all valid z coordinates explained across windows, using GroupKFold over units:

| Dataset | 13 axes | 13 random axes of the same sizes | 13 random coordinates | 13 pivoted-QR coordinates | 13 PCs |
|---|---|---|---|---|---|
| NeuroTycho | .40 | .46 | .37 | .49 | .66 |
| CHB-MIT | .16 | .21 | .14 | .20 | .51 |
| HMC | .20 | .25 | .23 | .28 | .51 |
| AD/FTD | .28 | .42 | .35 | .40 | .65 |
| LEMON | .30 | .47 | .35 | .47 | .66 |

**Evidence.**

- The axes explain less z variance than random averages of the same sizes. That is expected for targeted contrasts, but it means 60–84% of z variance is outside the axes.
- 47–65% of reliable (ICC-weighted) variance lies in coordinates involving at least one statistic that no axis uses: cointegration, barycentre, linear-model fit, Φ, entropy variants, LCSS, IDS, SI, GW-τ, reciprocity, CDS, PSI, group delay. Group delay is invalid in more than 10% of windows in every bank checked.
- The leading z component is the lag-correlation block in AD/FTD and LEMON: τ = 4–5 samples at 100 Hz, 25–37% of variance; 50 ms is half an alpha cycle. On NeuroTycho it is the precision block (36%).
- Internal consistency: Cronbach's α looks acceptable for most large axes (0.77–0.99; fixed-lag on LEMON 0.55), but effective pair counts are small. Form_monotone's 32 pairs give 1.3–4.6. Phase_lag (4.7–8.0) and band (5–9) are internally heterogeneous rather than redundant.
- Mediation (α 0.16–0.65 with r̄ ≈ 0) and best-shift (α 0.13 on NeuroTycho) fail outright.
- Separation: redundant blocks are form_monotone↔tail (0.46–0.75 in all five datasets checked), phase_lag↔predictive (0.44–0.50 in four of five, 0.01 on AD/FTD; Geweke instantaneous causality is the zero-lag share) and band↔spectral_peak (0.40–0.70 in four of five, −0.06 on AD/FTD). On NeuroTycho, a slow-wave factor couples fixed-lag, warping, history, band, peak and predictive (0.3–0.7).
- Data-driven grouping (average-linkage clustering of the pair profiles on 1 − |r| across windows) agrees only moderately with the hand labels: ARI 0.22–0.45 at 13 clusters.
- Clusters form by estimator family and dominant spectral factor, not by assumption. One NeuroTycho cluster holds pairs from 12 of the 13 axes.
- Clustering on within-cell noise gives a similar ARI (0.24–0.49), so part of the hand grouping is shared estimation noise among pairs that reuse a statistic.

**Better selection within the catalogue.**

1. Enumerate every pair that passes the five matching rules.
2. Gate each pair per dataset: profile reliability rel ≥ 0.3 for both statistics, non-saturation (mean |z| < 0.97 and SD > 0.03), and an effect larger than the strength-range control.
3. Keep 1–3 non-duplicate pairs per construct.
4. Test convergent and discriminant validity across datasets, multitrait–multimethod style: pairs of the same construct from different estimator families should agree more than pairs of different constructs from the same family.
5. Report the residual z structure (PCA after regressing out the axes) as a discovery channel, not as more axes.

Purely data-driven axes would be more representative and less interpretable, which defeats their purpose here.

## Q3 A 3–5 axis core

**Principles.**

- Organisation reads heterogeneity across pairs, so a core axis must be unsaturated and reliable in the target modality.
- Each pair must be one assumption apart under the five matching rules, and its estimator noise must be small relative to the across-edge spread.
- Axes should be mutually non-redundant, and each should have a stated null: linear-spectral by construction, or testable against IAAFT.
- Fewer, cleaner pairs beat many correlated ones. The mean of correlated pairs barely reduces noise and weights the axis toward over-represented families.

**Proposed core.** Four axes plus one conditional:

| Core axis | Exact pairs (p90 names) | Null | Level |
|---|---|---|---|
| form (Gaussian↔non-Gaussian) | `mi_gaussian`\|`mi_kraskov_NN-4_DCE-AUTO`; `tlmi_gaussian`\|`tlmi_kraskov_NN-4_DCE-AUTO`; `gc_gaussian_k-1_kt-1_l-1_lt-1`\|`te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1` | non-Gaussian; needs surrogate and estimator control | median (MI_ksgDCE − MI_g), real − surrogate |
| lag (zero-phase↔lagged) | `cohmag`\|`icoh`, `multitaper_mean`, bands 0–0.5, 0–0.25, 0.25–0.5 | linear-spectral by construction | median icoh/√cohmag = \|sin φ\| |
| band (low↔high) | `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25`\|`…fmin-0-25_fmax-0-5` | linear-spectral | high share |
| warp (rigid↔elastic) | `pdist_euclidean`\|`dtw_constraint-sakoe-chiba_radius-auto` | mixed; linear-spectral on NeuroTycho | DTW-SC / Euclidean |
| mediation (marginal↔direct), fixed montage only | `cov-sq_LedoitWolf`\|`prec-sq_LedoitWolf` | linear-spectral | — |

Core organisation, mean / SD / ICC:

| Dataset | form | lag | band | warp | mediation |
|---|---|---|---|---|---|
| NeuroTycho | .44/.11/.34 | .53/.19/.48 | .21/.21/.64 | .52/.23/.82 | .82/.12/.66 |
| CHB-MIT | .79/.09/.51 | −.34/.23/.40 | .66/.14/.50 | .55/.20/.68 | .28/.14/.53 |
| HMC | .80/.13/.32 | −.57/.26/.57 | .76/.12/.39 | .71/.14/.53 | .71/.16/.59 |
| AD/FTD | .89/.07/.63 | −.92/.05/.30 | .80/.10/.53 | .95/.04/.53 | .27/.09/.62 |
| LEMON | .80/.08/.63 | −.94/.02/.45 | .92/.04/.67 | .87/.04/.58 | .39/.09/.89 |
| propofol | .75/.06/.48 | −.62/.10/.68 | .73/.13/.78 | .68/.12/.48 | .27/.14/.63 |

**Evidence for this core.**

- Cohort ICC improves on the 13-axis versions:
  - AD/FTD: form .45→.63, phase_lag .07→.30, band .14→.53, mediation .03→.62.
  - LEMON: band .30→.67, mediation .14→.89.
- Inter-axis |r| ≤ 0.45 everywhere except warp–mediation on NeuroTycho (−0.76).
- Out-of-fold R² of the old axes from the core: form_information .57–.88, warping .56–.92, phase_lag .05–.77, band .05–.84. History, predictive, spectral_peak and phase_amplitude are barely covered (≤ .46), as intended.
- Prediction, as AUROC, core4 vs 13 axes vs Pearson:

  | Dataset | core4 | 13 axes | Pearson |
  |---|---|---|---|
  | NeuroTycho, held-out animal | .977 | 1.0 | .64 |
  | CHB-MIT | .82 | .81 | .80 |
  | AD/FTD | .57 | .64 | .82 |
  | LEMON | .66 | .74 | .89 |
  | propofol | .61 | .80 | .72 |

  The core gives up predictive power for interpretability. That fits the protocol's stated aim, but it should be shown.
- In scalp EEG, lag (−0.92 to −0.94) and warp (0.87–0.95) approach their bounds, so their levels will carry more than their organisation there.

**Why these four.**

- They cover the facets the introduction names: form, timing, phase lag and frequency.
- Each is one assumption apart.
- Three have analytic linear-Gaussian nulls (Nolte et al. 2020). Form is the single non-Gaussian test, so the paper can state which kind of character each finding is.
- Strongest alternatives, rejected: fixed-lag is the dominant EEG z component and is calibrated, but it measures spectrum in sample units, so it is not comparable across sampling rates and duplicates band. Predictive/history covers direction, but the TE profiles are unreliable outside NeuroTycho and are linear-spectral and attenuation-driven on it. Keep both in the extended set.

**Should shared axes keep the same constituents?** Yes. An axis name should map to exactly one set of pairs, with the core a strict subset of the extended set.

- Otherwise "phase_lag" in a 4-axis report and a 13-axis report are different estimands, pre-registered predictions lose their referent, and cross-report comparisons are confounded by constituent changes.
- Where the extended set needs more, add separately named sub-axes (`phase_locking`, `warp_extent`, `lag_profile`) rather than enlarging a core axis.
- Pooling more pairs under one name only helps if they share the construct and their noise is not shared. The data show both conditions fail for most of the current enlargements.

## Q4 Ranked changes

1. **Use the corrected (DCE) Kraskov estimators in form, and withhold the NeuroTycho non-Gaussian level claim.** Report the KSG vs KSG-DCE control next to every form effect; accept a sign only when real, surrogate and control agree. Why it matters: it is currently the only non-Gaussian claim, and its sign depends on the estimator.
2. **Add attenuation diagnostics.** Per cohort, report each statistic's cross-window profile reliability, gate pairs on it, and report disattenuated cell-level organisation next to raw. Why it matters: without this, an organisation effect cannot be separated from a change in profile SNR or stationarity.
3. **Rebuild mediation:** no mixed signs, no bivariate spectral pairs, Ledoit–Wolf only, fixed M. Why it matters: the current score averages opposite-signed pairs and pairs that do not condition at all.
4. **Remove duplicates and saturated or noise pairs, and fix the obvious defects:** dtf≡pdcoh, MCD≡EllipticEnvelope, max-over-frequency pairs, the 1-sample xpdist, the signed xcorr_max, and the icoh/cohmag exponent in the level (use icoh/√cohmag). Why it matters: these pairs add bias without adding information.
5. **Enforce the five matching rules,** e.g. predictive becomes tlmi\|te. Why it matters: a symmetric-vs-directed pair caps z at √(1 − asymmetry) and mixes in directionality.
6. **Rename the identity control to a strength-range control and require axis effects to exceed it.** Why it matters: it shows a deterministic pair moving at d ≈ 1.2.
7. **Nest the core in the extended set, with one name per constituent set.**
8. **Express lags and bands in physical units per cohort, or name them sampling-relative.** Why it matters: τ = 5 samples is 5 ms in one dataset and 50 ms in another.
9. **Publish a per-dataset saturation and reliability table for every axis,** and do not interpret axes that fail it. Examples: form_monotone, tail and best-shift in scalp EEG; every axis at LANL M=4, where symmetric profiles have six distinct values.
10. **For the paper's claim,** state that the defensible core is mostly linear-spectral character (lag, band, mediation, and warp on these data). That is still character rather than strength, but non-Gaussian character currently has no estimator-robust demonstration.

Not verified here: the spectrum-final and 1000-iteration surrogate banks were not finished (no `bank.npz`), so the DCE check uses the 200-iteration marginal-final surrogates only. Disattenuation assumes stationarity across windows. Siena real bank not analysed.
