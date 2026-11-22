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
- Mean-profile Δ is smaller than the per-window Δ (−.033 against −.056; −.024 against −.079): much of the effect is window-specific (which samples MCD discards differs between windows) rather than one stable edge pattern. That fits "a minority of samples carries a different dependence pattern"; the support-mask check (below) does not earn "episodic coupling" for the drug change.
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

**Result (job 184674717, 24 SU; `mechanism-long.json`).** Recomputed z(pcorr, pcorr_mcd) reproduces the screen exactly (Δ net −.0790, t(3) −10.72). Nets are real − IAAFT; t(3) on animal means.

| Metric | Awake net | Drug net | Drug − awake net | Verdict |
|---|---|---|---|---|
| Trimmed fraction (outside MCD support; real .44 awake, .48 drug) | +.020 (t 11.1) | +.036 (4/4, t 19.5) | +.015 (4/4, t 4.2; KT .006, MD .008, KTMD .016, PF .031) | moves with the axis |
| Clustering (Fano, 0.5-s bins) | +5.2 (t 6.6) | +1.5 (4/4, t 4.2) | −3.7 (0/4) | clustered in both states, less so under the drug |
| Channels beyond \|x\| 2 at top-5% samples | +8.8 | +7.4 (4/4) | −1.4 (0/4) | multi-channel (about 16 of 126; IAAFT 8.5) |
| Node participation of the outlying network | −1.4 | +0.2 (about 108 of 126) | +1.6 | spread over most channels, not focal |
| Amplitude of top-5% samples / rest | +.61 | +.52 (4/4) | −.08 (0/4) | outlying samples are high-amplitude in both states |
| r_net (outlying against bulk edge profile) | −.045 (t −5.0) | −.029 (3/4, t −1.6) | +.016 (2/2) | distinct awake; under the drug the rule fails |
| Recurrence of the outlying network across windows | +.089 | +.121 (4/4, t 5.2) | +.032 (2/2, t 0.2) | recurs in both states, no drug change |

By the criteria above, "episodic coupling" is not earned for the drug change: under the drug the outlying samples are clustered, multi-channel and recurrent, but the distinctness rule fails (3/4), and neither clustering nor distinctness increases with the drug (clustering beyond IAAFT falls, because the drug's slower spectra make the surrogate's own outlying samples more clustered: IAAFT Fano 3.2 awake, 8.1 drug). What carries the axis is the trimmed fraction: MinCovDet trims more samples than the spectrum-matched surrogate, more so under the drug, ordered by agent as the axis is (propofol largest, ketamine smallest), and it is the metric most correlated with the axis net within drug sessions (window r −.49; clustering −.03, r_net +.15). The reading stays time-heterogeneous dependence, made more specific: under every drug a larger fraction of samples falls outside the robust core covariance than the drug's spectra predict, rather than a few more or more distinct episodes. Awake, the outlying samples already form clustered, high-amplitude, multi-channel, recurrent and distinct events beyond the surrogate (an awake property, not read here). Not separated: a heavier-tailed or less elliptical joint distribution and a mixture of covariance regimes both raise the trimmed fraction.

## Preprocessing sensitivity: z-scored windows (post hoc)

Question: does the MinCovDet axis z(pcorr, pcorr_mcd) hold when windows are only z-scored (the pyspi treatment) instead of normal-scored? Criteria stated and committed before the run. Code: [mechanism.py](mechanism.py) with `PREP` (z_pc is the axis, z_c2 is `r2|mcd2`, kurt the mean channel excess kurtosis); 10-s arm, every window, the screen's IAAFT seeds, net = real − mean of 3 draws, animal-level t(3), null SD = split-half SD / √2 as in the screen.

- **Why the surrogate changes.** MinCovDet trims on marginal tails as well as joint atypicality. In synthetic linear-Gaussian-copula data (30 channels, T 1000) with z-scored power-law marginals, real z_pc falls from .973 to .808 as channel excess kurtosis rises from 0 to 2.9. Spectrum-final IAAFT smooths the tails (kurtosis 2.9 → .57, z_pc .936), so its net reads the marginals (spurious −.13). Marginal-final IAAFT keeps them exactly (z_pc .816, net −.008). So on z-scored data only the marginal-final net is a test of dependence.
- **Arms:** `z-m` z-scored, marginal-final IAAFT (primary); `ns-m` normal-scored, marginal-final IAAFT (isolates the surrogate change from the screen's `ns`); `z` z-scored, spectrum-final IAAFT (the size of the marginal confound in the real data).
- **Holds without normal-scoring** if the `z-m` z_pc drug − awake net is < 0 in 4/4 animals, one-sided t(3) p < .05, at least 3 null SD, and < 0 under every agent. Its size relative to `ns-m` is reported; within 0.5–2× counts as the same size.
- **Outcomes:** holds at the same size, the axis does not depend on normal-scoring; holds smaller (< 0.5×), the z-scored version is diluted by marginal tails; fails, the axis is defined only at the copula level, and any other dataset must normal-score. If `ns-m` itself fails the screen's gates, the marginal-final surrogate's spectral mismatch is too large to read the `z-m` arm and the test is uninformative.

**Result (jobs 184745396–8, 74 SU; `mechanism-long-{z-m,ns-m,z}.json`).** The axis fails without normal-scoring, so it is a copula-level property only. The test is informative: `ns-m` passes every gate, so changing the surrogate's final step is not what removes it. z_pc drug − awake, t(3) on animal means (Chibi, George, Kin2, Su):

| Arm | Real Δ | IAAFT Δ | Net Δ (animals) | t(3) | Null SD | Agents KT / KTMD / MD / PF |
|---|---|---|---|---|---|---|
| `ns` (screen) | −.129 | −.050 | −.079 (−.070, −.083, −.100, −.068) | −10.7 | 11.4 | −.048 / −.091 / −.035 / −.120 |
| `ns-m` | −.129 | −.053 | −.075 (−.070, −.074, −.100, −.068) | −10.6 | 9.5 | −.044 / −.090 / −.029 / −.112 |
| `z-m` (primary) | −.094 | −.085 | −.009 (−.034, +.049, −.121, −.007) | −0.8 | 1.0 | +.076 / −.042 / +.026 / −.041 |
| `z` (confounded) | −.094 | −.021 | −.073 (−.088, −.046, −.117, −.074) | −5.5 | 16.1 | −.002 / −.113 / −.005 / −.105 |

- On z-scored windows the real z still falls by about three quarters as much as on normal-scored ones (−.094 against −.129). What changes is the null: a linear process with the drug's spectra and each channel's own marginals (marginal-final IAAFT) falls almost as far (−.085). With Gaussian marginals it falls −.053. The drug barely changes channel kurtosis (real .51 awake, .49 drug), so this is the drug's spectral change acting through the existing mild tails, not a change in the tails.
- Even awake, the z-scored arm sits on the other side of its null: real z_pc .941 against marginal-final IAAFT .872 (net +.069, t(3) 4.6). Normal-scored, it is −.041. On z-scored data, the axis level is dominated by how the channels' tails line up in time, which the surrogate gets wrong in both directions.
- The naive z-scored arm (`z`) looks as strong as the screen (−16 null SD), as calibration predicted: spectrum-final draws lose the tails (surrogate kurtosis .14 against real .51). Its agent pattern (KT and MD ≈ 0) also differs from the screen's. So it is not evidence either way, and pyspi-bank results on z-scored data with spectral or spectrum-final controls should not be read for this axis.
- Secondary, `r2|mcd2` (z_c2) under `z-m`: −.022, 4/4 animals, t(3) −5.7, every agent negative, but 1.9 null SD (≤ 3) and 48% of `ns-m` (−.047). The trimmed fraction under `z-m` still rises beyond the null: +.008, 4/4, t(3) 5.4, 3.5 SD, though KT −.006.
- Synthetic gain check (post hoc, [gain_sim.py](gain_sim.py)): slow amplitude modulation is a sufficient generator of the axis's sign on both scales. A gain shared by all channels lowers z_pc far beyond IAAFT in both arms (net −.28 z-scored, −.28 to −.35 normal-scored); an independent gain per channel lowers it less (−.02 to −.07); none gives +.02. So an elliptical scale mixture does not hide the effect on z-scored data, and ellipticity does not explain why the real data fail there. Why the real z-scored null falls with the drug almost as far as the real data, and sits below real awake (+.069), is unresolved. Shared gain also moves samples off the leading mode in every direction, so high-amplitude samples are trimmed (as the mechanism check found, amplitude net +.5), not only pattern-atypical ones.
- Reading: the robust-against-classical divergence is a drug-specific change in joint, copula-level atypicality (what normal-scoring isolates). Without normal-scoring it cannot be told apart from the drug's spectra acting on the channels' marginal tails. Any further test of this axis (PRIOS included) must normal-score; it did. The MI pair has no such dependence (`findings.md`, copula-level section: raw ≈ normal-scored).

## Scalar-gain test: calibrated, not run

Question: are the MI and MinCovDet axes one property, a shared scalar gain on a fixed carrier (x = a(t) y(t) + floor, so local covariance a²S + N fluctuates with rank one, even with a spatially correlated floor)? Synthetic calibration only ([gain.py](gain.py), output `results/.../gain-calib.txt`; 10-s windows, M 126, 10 windows per generator). No statistic separates the H0 generators (shared slow, burst and correlated-floor gain) from the alternatives (two sources with independent gains, per-channel gain, an amplitude-linked global mode, constant-power switching), so nothing was frozen or run on real data.

- **Amplitude strata** (rank-one share of K = 4 stratum-covariance deviations, against the remodulation surrogate a_hat·IAAFT(x/a_hat)): every gain generator, alternatives included, gives net +.01 to +.08. One amplitude index orders every amplitude-linked process along one direction, so two independent source gains also look rank one. The surrogate also carries a shrunk gain path (a_hat smooths a), so its share is lower for reasons unrelated to H0.
- **Block covariances** (cross-fitted over 0.5-s blocks on the top 8 PCs): the cross-fitted trace is dominated by noise (ratios above 1); nets range −.95 to +1.5 with no ordering by generator.
- **Component envelopes** (cross-fitted second eigenvalue of 0.5-s block powers of the top 8 PCs, against misaligned blocks): H0 +.03 to +.08, two sources +.10 to +.12, but stationary data also gives +.11 and per-session SEs are .015–.05. The baseline depends on the generator, so a real-data value or drug change has no fixed H0 to read against.
- **Why it cannot be rescued here.** 10-s windows give 20 half-second blocks with few effective samples each (slow spectra), so power against realistic independent gains is low. A powered version would need minutes of continuous data per state. Even then, 126-channel ECoG is almost certainly multi-source, so scalar gain would be rejected in every state, and rejection excludes only the scalar family, not source-specific gain (workstream context, 17 November). It would not decide whether the axes share a generator.
- **Standing:** gain remains a sufficient generator of both axes' sign (simulations here and in `mi_pair/nonstat`); whether the axes are one property is open. Current evidence points to two properties of one class: window r .2–.3, and different agent orders (MI largest under ketamine, MinCovDet under propofol).

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

## Status and open items (22 November)

- Answer to the screen's question: no axis matches the MI pair on every gate and on transfer. MinCovDet passes every macaque 10-s gate with a larger effect, but it is copula-level only (fails on z-scored windows), absent at 2 s and unusable in PRIOS. `l1|linf` transfers (7/7) but depends on window length, sits below the PRIOS reliability floor and has no reading under the guide. The MI pair remains the anchor.
- Open, highest value: `l1|linf` has not had the z-scored check that removed MinCovDet's excess; rerun the 2-s arm on z-scored windows against marginal-final IAAFT (about 30 SU, worst case 60). Freeze the criteria of the MinCovDet sensitivity section first.
- Open: a confirmatory human test of MinCovDet needs long continuous epochs (T well above M); PRIOS's 2-s inter-stimulation epochs cannot supply them.
- Open: mechanism. Shared gain is a sufficient generator of the MinCovDet sign; the scalar-gain test is not identifiable in 10-s windows, and only perturbation data can separate gain from coupling.
- For the guide owner: the worked example's "large common-mode samples are kept by MinCovDet" holds only when the amplitude lies along one mode; a shared gain inflates every direction and those samples are trimmed.
