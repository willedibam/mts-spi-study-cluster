# Pair catalogue: z(A, B) coordinates with a clean reading of dependence character (20 November 2026)

Interpretive map over the r13 NeuroTycho atlas (272 SPIs; 222 valid in every window), built to choose the next computation. Nothing here is a new effect: numbers are copied from committed outputs by [`build.py`](build.py), the Gaussian relations are checked numerically by [`identities.py`](identities.py) ([`identities.json`](identities.json)), and the full pair-level map, every one of the 584 minimal pairs plus 15 non-minimal ones, is [`catalogue.json`](catalogue.json). Reading rules: [`docs/interpreting-spi-pairs.md`](../../../../docs/interpreting-spi-pairs.md). Curated readings: [`entries.py`](entries.py). Regenerate with `python -m research.representation.atlas.pair_catalogue.build && python -m research.representation.atlas.pair_catalogue.render`.

## How to read this

- **Entry:** one assumption difference (a class of minimal pairs from [`assumptions.py`](../assumptions.py)) or one principled non-minimal pair with a known Gaussian relation. The representative pair comes first; `catalogue.json` lists every member pair with its own numbers.
- **Tier and null:**
  - *higher-order:* both statistics have a Gaussian expectation fixed by second-order structure, so real − joint spectrum-final IAAFT reads the departure the flexible member targets.
  - *linear-spectral:* both are fixed by the sample cross-spectra, so IAAFT is degenerate; these need the zero-phase and stationary Gaussian nulls with the state's cross-spectrum.
  - *linear-static:* fixed by the zero-lag covariance; read raw with strength controls.
- **Ranking:** score = interpretability/3 × measurability. Measurability is the better arm's smaller partner reliability, times a ceiling factor and a null-validity factor. The ceiling factor is .4 or .7 at z₀ > .98 or > .95 (.7 or .9 for higher-order pairs, whose IAAFT net carries the surrogate's saturation mechanics). Exclusions score 0.
- **Columns:**
  - rel: wake between-window edge-profile reliability of A and B, 2-s / 10-s arms.
  - z₀: wake z in the 2-s / 10-s pyspi bank (ax: pyspi-free axis screen, 10 s).
  - matched: the 2-s per-window joint-IAAFT net, drug − awake, mean over 19 sessions (sessions sharing its sign). It comes from the normal-scored construction of [`movers/matched-result.md`](../movers/matched-result.md); it is degenerate (≈ 0 by construction) for linear pairs.
- **Status:** *untested* (no z computed), *raw* (stored pyspi, no surrogate), *beyond IAAFT* (a net exists), *axis* (passed the frozen axis-screen gates), *failed* (failed them, reason in brackets).

## Principles found while building it

1. **Static-statistic principle (verified).** On z-scored channels, the expectation of any time-exchangeable bivariate statistic is a fixed function of r under a Gaussian process. The set includes the rank correlations, the MI estimators, dCor, IDS, the L1, L2, cosine and Canberra distances and the barycentre moments. So every pair from that set is a principled non-minimal pair beyond IAAFT, and its reading is the feature the more flexible member weights:
   - rank correlations: concordance;
   - kNN MI: the full KL departure;
   - dCor: non-monotone dependence;
   - L1/L2: kurtosis of x − y;
   - Canberra: per-sample scale-free agreement.

   The Gaussian network check gives z(r², dCor²) .988, z(r, Spearman) .998 and z(L1, L2) .995. Maxima are the exception. Chebyshev and the barycentre maximum also depend on T and on the spectrum of x ∓ y (max/RMS 2.97 for white noise against 2.31 at AR .98, T = 200), which joint IAAFT fixes but r alone does not.
2. **Map the partner through its Gaussian relation before correlating.** This removes saturation from z₀, which is then 1 minus estimator noise under the null. The maps:
   - ranks: arcsin;
   - dCor: Székely et al. 2007, Thm 7;
   - MI: −½ log(1 − r²);
   - envelopes: closed forms in |γ|².

   The same map gives a per-edge level readout (excess = H − f(L)) beside z.
3. **The envelope's Gaussian partner is |γ|² = r² + corr(x, H[y])², not r² (verified).** For proper complex Gaussian analytic signals, corr(\|a\|², \|b\|²) = \|γ\|². Envelope amplitude and log power are closed functions of |γ|², and PLV is (π/4)|γ| ₂F₁(½, ½; 2; |γ|²) per frequency. So z(pec, r²) mixes in lag, and z(pec, |γ|²) reads amplitude co-modulation beyond linear coupling. IAAFT keeps γ exactly, because γ is a linear functional of the cross-periodogram.
4. **Canberra is gain-invariant on z-scored windows (verified by synthetic, this session).** The excess of Canberra over its Gaussian value f_G(r):

   | Synthetic | Excess |
   |---|---|
   | Pure shared gain | −.004 |
   | Separate per-channel gains | −.001 |
   | Correlation falling with amplitude (amplitude-linked decoupling) | −.120 |
   | Correlation rising with amplitude (gain over a fixed floor) | +.100 |
   | Static quadratic | −.011 |
   | After normal-scoring, shared gain | −.061 |

   So on z-scored windows the sign of the excess says how correlation depends on amplitude, while a pure scale mixture leaves it unmoved, unlike the MI anchor. This is the first readout in the atlas that could separate "one gain on an unchanged correlation" from "correlation that changes with amplitude", the sub-question the anchor's route-2 reading leaves open. Normal-scoring (the atlas's IAAFT construction) destroys the invariance.
5. **The null follows the estimator, not only the statistic.** Spectrum-final IAAFT keeps the periodogram and cross-periodogram exactly, so a frequency-smoothed (Daniell) coherence is degenerate under it. Multitaper and zero-padded lagged statistics are not (axis screen, post hoc), so a multitaper member makes real − IAAFT also respond to within-window spectral non-stationarity.

## Shortlist: what to compute next

Ordered for the next computation: the gain-discriminating and envelope readouts first (new, cheap, pyspi-free), then the extreme-value family that already moves beyond IAAFT, then calibration of the existing MinCovDet axis, then the ceiling form pairs with mapped partners, then the linear tier. Row 0 is the anchor, recomputed as the calibration reference in every run. Contrast = drug − awake eyes-closed per session, animal-level t(3), eyes-open and within-awake split-half controls, as in the axis screen.

| # | Entry | Pair(s) | Reading | Contrast to compute | Null |
|---|---|---|---|---|---|
| 0 | Gaussian against kNN MI (the anchor) | `mi_gaussian` · `mi_kraskov_NN-4_DCE-AUTO` | non-Gaussian (nonlinear) dependence: after normal-scoring, I - G is the KL divergence from the… | reference row: recompute alongside every new readout as the calibration anchor | real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows |
| 1 | Euclidean against Canberra distance: gain-invariant agreement | `pdist_euclidean` · `pdist_canberra` | amplitude-dependent correlation: whether a pair's correlation differs between high- and low-amplitude samples | per edge E = canberra - f_G(r) on z-scored windows; z(f_G(r), canberra) and the level of E, drug - awake, read jointly with the anchor: anchor departs while Canberra does not = consistent with pure shared gain; both depart = correlation changes with amplitude | analytic f_G(r) and Gaussian FT surrogate (z-scored) |
| 2 | Envelope correlation against its Gaussian determinant \|gamma\|^2 (pyspi-free partner) | `pec` · `NEW:analytic_gamma2` | amplitude co-modulation beyond linear coupling: envelope correlation in excess of what the pair's… | per edge E_env = pec - g(\|gamma\|^2) and z(g(\|gamma\|^2), pec); g is the closed form, so z0 is 1 minus noise under the null | real - joint spectrum-final IAAFT on z-scored (not normal-scored) windows |
| 3 | Envelope correlation on amplitude against log power | `pec` · `pec_log` | burst-driven against proportional amplitude co-modulation: log compresses large excursions, so the pair… | z(pec, pec_log) net, run beside envelope-gamma (same windows, z-scored) | real - joint spectrum-final IAAFT on z-scored (not normal-scored) windows |
| 4 | Chebyshev against L2 or L1 distance: extreme disagreement | `pdist_euclidean` · `pdist_chebyshev` | heavy-tailed disagreement: intermittent large divergence of a pair beyond what a Gaussian difference signal… | z(l1, linf) and z(l2, linf), 2-s windows (or a fixed sample count), with Canberra and bary-max as links | real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows |
| 5 | Classical against MinCovDet covariance, precision, partial correlation | `pcorr_EmpiricalCovariance` · `pcorr_MinCovDet` | time-heterogeneous dependence: a fraction of samples departs from the dominant covariance | already an axis; next computation is the calibration synthetic and a 2-s test at M <= 60 | real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows |
| 6 | Mean against max of the squared barycentre: extreme co-excursion | `bary-sq_euclidean_mean` · `bary-sq_euclidean_max` | joint extreme excursions (co-bursts): pairs whose largest common excursion exceeds what a Gaussian sum with… | z(r, bary-sq max) (identical to z(bary-sq mean, bary-sq max), since the mean is (1 + r)/2) and the cross pair z(pdist_chebyshev, bary-sq max): extreme agreement against extreme disagreement | real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows |
| 7 | Signed against two-sided max of the barycentre: asymmetry of co-excursions | `bary_euclidean_max` · `bary-sq_euclidean_max` | asymmetry (skew) of joint excursions: whether pairs co-burst upward and downward alike | z(bary_euclidean_max, bary-sq_euclidean_max) | real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows |
| 8 | Coherence against phase-locking value: amplitude-dependent synchrony | `cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` · `plv_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | amplitude-dependent synchrony: phase locking concentrated in high- or low-amplitude epochs, beyond the fixed… | pyspi-free Daniell coherence against per-frequency PLV mapped through the Gaussian relation, high band and broadband | IAAFT with a Daniell estimator |
| 9 | Pearson against Spearman or Kendall (with a normal-score intermediate) | `cov_EmpiricalCovariance` · `spearmanr` | copula departure in concordance: dependence that is monotone but not Gaussian-copula (joint bursts read here… | z(pearson on normal scores, spearman) after mapping r through (6/pi) arcsin(r/2); level of the mapped excess per edge | real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows |
| 10 | r^2 against distance correlation (and IDS) | `cov-sq_EmpiricalCovariance` · `dcorr` | any-form (including non-monotone) dependence beyond the Gaussian value | z(Thm-7-mapped r, dcorr) and z(r^2, ids); same F1 family as the anchor | real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows |
| 11 | Gaussian GC against model-free TE (GC = 2 TE for Gaussians) | `gc_gaussian_k-1_kt-1_l-1_lt-1` · `te_kernel_W-0.25_k-1` | non-Gaussian predictive (directed) dependence: prediction of the target from the source beyond its own past… | z(gc_gaussian k1, te_kernel k1) net, 10-s arm only (reliability .47/.68); report the level median(te - gc) beside it | real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows |
| 12 | Marginal against conditional (covariance against precision or partial correlation) | `cov-sq_EmpiricalCovariance` · `prec-sq_EmpiricalCovariance` | mediation: direct against indirect association given the observed channels | z(cov-sq_EmpiricalCovariance, prec-sq_LedoitWolf) at 10 s (identical to z(cov-sq LW, prec-sq LW): one shrinkage factor per window), strength-matched edges, channel-subset re-extraction for any M change | stationary Gaussian null with the state covariance |
| 13 | Zero lag against best lag | `cov-sq_EmpiricalCovariance` · `xcorr-sq_max_sig-True` | coupling carried at a stable non-zero lag | z(r^2, xcorr-sq max), 10 s, mean-lag term = stationary - zero-phase, per agent | zero-phase and stationary Gaussian nulls with the state mean cross-spectrum (discovery.py lag-null) |
| 14 | Lagged MI against GC (conditioning on the target past) | `tlmi_gaussian` · `gc_gaussian_k-1_kt-1_l-1_lt-1` | precedence: what the source adds beyond the target's past | z(tlmi_gaussian, gc_gaussian) at 10 s and the kernel link beside it (higher-order part) | stationary null (Gaussian) |

Expected compute: rows 1–4 and 6–7 are pyspi-free statistics on the anchor's windows with the screen's IAAFT draws or a Gaussian FT surrogate. That is the axis-screen pipeline (0.23 kSU for 33 SPIs in both arms), so about 0.1–0.3 kSU expected and 0.5 kSU worst case on Gadi. Row 8 needs a Daniell estimator added to the same pipeline. Calibration synthetics (rows 1, 2, 5, 8) are local and take minutes.

## Upstream pyspi changes (checked 2026-11-20)

Pin 5d61d9c (computation 3.0.0.r13), upstream main 5e25d27. The atlas config is a frozen copy of `benchmarked_p90`, so no r13 number here changes and nothing leaves the bank until the main session re-syncs it. z is the Pearson correlation of two edge profiles (assumptions.py), so an SPI that is an exact affine function of a kept SPI gives exactly the same z with any partner: pdist_cosine = 1 - r, bary-sq_euclidean_mean = (1 + r)/2, cov_{LedoitWolf,OAS,ShrunkCovariance} = (1 - delta) r with one delta per window on z-scored data, and their cov-sq forms. A monotone nonlinear function (pdist_euclidean = sqrt(2T(1 - r)), lmfit of r^2, ppc of PLV) changes z only through curvature, which the catalogue already treats as an identity or saturation control, and the mapped-partner convention uses the analytic form. Kendall is not an exact function of Spearman (the relation depends on the copula), but its residual was already excluded as a near-ceiling diagnostic. So the merged cuts remove no reading; they agree with the identity, shrinkage, kendall-spearman and debias exclusions.

| PR | State | What | Impact on the catalogue |
|---|---|---|---|
| #135-#143 | merged 20 Nov | 37 SPIs unbundled as redundant (shrinkage covariances, six cov-sq, lmfit, pdcoh/gpdcoh, ppc, kendalltau(-sq), pdist cosine/euclidean, bary-sq mean, phi norm-1) | config only: classes and values unchanged, still r13. All 37 are in the 272-SPI bank; no catalogue reading depends on one that has no exact kept equivalent |
| #144 (on #112) | open; decision A4 approved 20 Nov | 48 corr_*_tau-k and xcorr SPIs leave the bundles (45 are in the bank); 13 directed lagcorr_* SPIs (lag, peak, asym, onesided) come in | removes the fixed-lag class (169 of 175 pairs) and the lag members of rank-linear, best-lag and peak; best-lag moves to lagcorr_peak |
| #134 (on #133) | open (own PR) | kernel MI, TLMI and TE exclude the point itself; COMPUTATION_VERSION -> 3.0.0.r14 | kernel values change (independent AR(1): MI .18 -> .07 nats); r13 z0 and reliability of kernel pairs do not transfer |
| #133 | open (own PR) | kernel Theiler window: mi_kernel_W-0.25_DCE-AUTO, tlmi_kernel_W-0.25_DCE-AUTO, te_kernel_W-0.25_DCE-AUTO_k-1; xi_tau-{0,1}, pcorr and te_symbolic_k-3 into benchmarked_p90 | existing SPIs bit-identical; adds the autocorrelation-corrected kernel partners and xi (not in the bank) |
| #127 | open | undirected pairs evaluated once | bit-identical values; dcorr, hsic, mgc, gwtau and other distance SPIs now give NaN for a failing pair instead of failing whole. No catalogue impact |

Pending upstream: spearmanr-sq and cov-sq_GraphicalLassoCV stay bundled pending separate decisions.

| Entry | Members affected | Note |
|---|---|---|
| Classical against MinCovDet covariance, precision, partial correlation | 1/5 | cov-sq_MinCovDet is unbundled (#136): it is the square of cov_MinCovDet. The axis readouts (pcorr\|pcorr_mcd, r^2\|mcd^2) are unaffected. |
| Gaussian against kernel MI | 1/1 | #134 (open, r14) removes the self-count from kernel MI. The r13 value carries a bias largest where joint-box counts are small, which depends on T, the bandwidth and the joint density (so on coupling), so the IAAFT net need not cancel it. Recompute on r14 in a new output directory, and with #133 prefer mi_kernel_W-0.25_DCE-AUTO, which addresses the autocorrelation consequence listed above. |
| Marginal against conditional (covariance against precision or partial correlation) | 11/15 | cov_{LedoitWolf,OAS,ShrunkCovariance} and their cov-sq forms are unbundled (#136). One shrinkage factor per window makes z(cov-sq LW, prec-sq LW) = z(cov-sq Emp, prec-sq LW) exactly, so the shortlist contrast now names kept SPIs; prec_* and pcorr_* are unchanged. |
| Chebyshev against L2 or L1 distance: extreme disagreement | 2/5 | pdist_euclidean and pdist_cosine are unbundled (#141); on z-scored windows L2 = sqrt(2T(1 - r)), so take it from r (the shortlist row is pyspi-free anyway). pdist_chebyshev stays. |
| Gaussian against kNN lagged MI | 1/2 | The kernel member changes value with #134 (r14) and gains a DCE-AUTO form with #133; the KSG member is unchanged. |
| Euclidean against Canberra distance: gain-invariant agreement | 2/3 | pdist_euclidean and pdist_cosine are unbundled (#141); the shortlist already uses the analytic f_G(r). pdist_canberra stays. |
| L1 against L2 distance: tail weight of disagreement | 2/2 | pdist_euclidean and pdist_cosine are unbundled (#141); L2 comes from r. pdist_cityblock stays. |
| Pearson against Spearman or Kendall (with a normal-score intermediate) | 31/33 | kendalltau is unbundled (#140) and the lagged corr_* members leave with #144 (open); spearmanr stays, and spearmanr-sq is pending. The shortlist uses Spearman. |
| r^2 against distance correlation (and IDS) | 2/3 | cov-sq_EmpiricalCovariance is unbundled (#136); use r^2 or the Thm-7 map of r, as the shortlist already does. |
| Low against high band | 2/16 | The ppc members are unbundled (#139); the cohmag and plv band pairs stay. |
| Mean against max of the squared barycentre: extreme co-excursion | 1/1 | bary-sq_euclidean_mean is unbundled (#141); it is (1 + r)/2, so z(bary-sq mean, bary-sq max) = z(r, bary-sq max) exactly. bary-sq_euclidean_max stays. |
| Gaussian GC against model-free TE (GC = 2 TE for Gaussians) | 1/2 | te_kernel changes value with #134 (r14: self-count removed, and the next value scaled like the past, as JIDT does). Compute shortlist row 11 on r14, or use te_kernel_W-0.25_DCE-AUTO_k-1 (#133). |
| Lagged MI against GC (conditioning on the target past) | 1/3 | The kernel link (tlmi_kernel against te_kernel) changes value with #134 (r14); the Gaussian link is unchanged. Compute the kernel link in shortlist row 14 on r14. |
| Coherence against phase-locking value: amplitude-dependent synchrony | 1/7 | One ppc member is unbundled (#139). The shortlist row (Daniell coherence against mapped PLV) is pyspi-free, so no impact. |
| Zero lag against best lag | 2/2 | #144 (open) removes xcorr_* and adds lagcorr_peak (signed r at the lag of max \|r\|, over min(max_lag, T // 4) lags) and lagcorr_lag. If it merges before the run, shortlist row 13 becomes z(r^2, lagcorr_peak^2) with lagcorr_lag as a direct lag readout; r13 xcorr z0 and reliability do not transfer. |
| Rigid against elastic alignment | 1/8 | pdist_euclidean is unbundled (#141); take L2 from r. |
| Mean against max over frequency or lag | 4/25 | The xcorr member leaves with #144 (open); the ppc members are unbundled (#139). Read the frequency peak through plv mean\|max, which carries ppc (PPC = (N PLV^2 - 1)/(N - 1) per frequency). |
| PPC against debiased wPLI (lag, both debiased) | 3/3 | ppc is unbundled (#139); plv\|dswpli carries the same lag reading, with the plv band means .98-.99 against ppc rather than exact. |
| Gaussian against kernel or Kozachenko entropies (je, ce, si, xme, cce) | 7/24 | With #134 kernel MI, TLMI and TE leave one point out but kernel je, ce, si, xme and cce still count it, so kernel MI is no longer H + H - JE of pyspi's kernel entropies. The exclusion stands. |
| Zero against fixed lag (samples) | 169/175 | 169 of 175 pairs leave with #144 (open). |
| Identity and saturation controls | 7/9 | The cuts (#136, #137, #141) remove exactly the members this entry already called identities. |
| Kendall against Spearman | 16/16 | kendalltau is unbundled (#140) on the near-identity this exclusion uses; the lagged Kendall members leave with #144. |
| Empirical against shrinkage covariance (LedoitWolf, OAS, Shrunk) | 18/45 | The shrinkage covariances are unbundled (#136) on the same identity this exclusion uses; the precision forms stay. |

Candidate pairs from new upstream SPIs (not yet entries):

- Chatterjee xi against mapped r: xi_tau-0 (in upstream full; into p90 with #133; not in the bank). Gaussian relation verified here: xi = (3/pi) arcsin((1 + rho^2)/2) - 1/2 (identities.json). A static statistic, so the static-statistic principle applies: z(mapped r, xi) beyond IAAFT reads functional, including non-monotone, dependence (xi -> 1 iff y is a function of x), in the anchor and dCor family. xi_ij - xi_ji is zero under Gaussianity and reads which channel is more nearly a function of the other. Noisy: SD about .05 at T 200 for every rho (signal .05 at rho .3), so 10-s windows only. Interpretability 2.
- lagcorr (#112/#144, open): z(r, lagcorr_peak) replaces z(r, xcorr max) for best lag, and lagcorr_lag reads the lag directly; z(lagcorr_asym, gc_ij - gc_ji) contrasts an unconditioned lead asymmetry with one conditioned on the target past (linear-spectral; zero-phase and stationary nulls). Interpretability 1-2.
- Kernel DCE-AUTO (#133): mi_gaussian against mi_kernel_W-0.25_DCE-AUTO is the better mi-kernel member (it removes the autocorrelation bias the entry lists); mi_kernel against its DCE-AUTO form is a Theiler-window diagnostic and joins the dce exclusion.

## Ranked catalogue

| Rank | Entry | Level · tier | Representative pair | Reads | rel 2 s / 10 s | z₀ 2 s / 10 s | matched 2-s net | Status | Score |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [Classical against MinCovDet covariance, precision, partial correlation](#mcd) | estimator · higher-order | `pcorr_EmpiricalCovariance` · `pcorr_MinCovDet` | time-heterogeneous dependence: a fraction of samples departs from the dominant covariance | 0.52, 0.41 / 0.87, 0.81 | – / – (ax 0.83) | -0.012 (15/19) | axis at 10 s; absent at 2 s | 0.81 |
| 2 | [Gaussian against kNN MI (the anchor)](#mi-ksg) | estimator · higher-order | `mi_gaussian` · `mi_kraskov_NN-4_DCE-AUTO` | non-Gaussian (nonlinear) dependence: after normal-scoring, I - G is the KL divergence from the covariance-matched… | 0.82, 0.69 / 0.93, 0.89 | 0.91 / 0.97 (ax 0.96) | -0.028 (19/19) | axis; replicated (PRIOS, mouse) | 0.80 |
| 3 | [Envelope correlation against its Gaussian determinant \|gamma\|^2 (pyspi-free partner)](#envelope-gamma) | principled · higher-order | `pec` · `NEW:analytic_gamma2` | amplitude co-modulation beyond linear coupling: envelope correlation in excess of what the pair's second-order… | 0.41, – / 0.70, – | – / – | – | untested | 0.70 |
| 4 | [Gaussian against kernel MI](#mi-kernel) | estimator · higher-order | `mi_gaussian` · `mi_kernel_W-0.25` | non-Gaussian dependence (same quantity as the anchor, different estimator bias) | 0.82, 0.69 / 0.93, 0.91 | 0.89 / 0.98 | -0.039 (19/19) | beyond IAAFT (frozen estimator check) | 0.64 |
| 5 | [Marginal against conditional (covariance against precision or partial correlation)](#mediation) | assumption · linear-static | `cov-sq_EmpiricalCovariance` · `prec-sq_EmpiricalCovariance` | mediation: direct against indirect association given the observed channels | 0.77, 0.56 / 0.91, 0.87 | 0.12 / 0.13 | +0.000 (19/19) | raw; local estimator effect | 0.58 |
| 6 | [Chebyshev against L2 or L1 distance: extreme disagreement](#chebyshev) | principled · higher-order | `pdist_euclidean` · `pdist_chebyshev` | heavy-tailed disagreement: intermittent large divergence of a pair beyond what a Gaussian difference signal with the… | 0.83, 0.52 / 0.94, 0.56 | 0.78 / 0.75 (ax 0.81) | -0.037 (18/19) | axis at 2 s; fails at 10 s | 0.56 |
| 7 | [Gaussian against kNN lagged MI](#tlmi-ksg) | estimator · higher-order | `tlmi_gaussian` · `tlmi_kraskov_NN-4_DCE-AUTO` | non-Gaussian dependence at a one-sample (10 ms) lag | 0.68, 0.48 / 0.88, 0.78 | 0.81 / 0.94 | -0.033 (19/19) | beyond IAAFT (matched screen) | 0.52 |
| 8 | [Euclidean against Canberra distance: gain-invariant agreement](#canberra) | principled · higher-order | `pdist_euclidean` · `pdist_canberra` | amplitude-dependent correlation: whether a pair's correlation differs between high- and low-amplitude samples | 0.83, 0.79 / 0.94, 0.93 | 0.96 / 0.98 | -0.013 (19/19) | beyond IAAFT, normal-scored only | 0.49 |
| 9 | [L1 against L2 distance: tail weight of disagreement](#cityblock) | principled · higher-order | `pdist_euclidean` · `pdist_cityblock` | kurtosis of the disagreement signal x - y beyond Gaussian (constant sqrt(2/pi) ratio for Gaussian) | 0.83, 0.82 / 0.94, 0.94 | 0.99 / 0.99 | -0.004 (19/19) | failed (ceiling) | 0.44 |
| 10 | [Pearson against Spearman or Kendall (with a normal-score intermediate)](#rank-linear) | assumption · higher-order | `cov_EmpiricalCovariance` · `spearmanr` | copula departure in concordance: dependence that is monotone but not Gaussian-copula (joint bursts read here in… | 0.81, 0.81 / 0.93, 0.93 | 0.99 / 1.00 | -0.001 (19/19) | failed (ceiling) | 0.43 |
| 11 | [r^2 against distance correlation (and IDS)](#dcorr) | principled · higher-order | `cov-sq_EmpiricalCovariance` · `dcorr` | any-form (including non-monotone) dependence beyond the Gaussian value | 0.77, 0.77 / 0.91, 0.91 | 0.99 / 0.99 | -0.006 (19/19) | failed (animal t) | 0.42 |
| 12 | [Envelope correlation on amplitude against log power](#pec-log) | estimator · higher-order | `pec` · `pec_log` | burst-driven against proportional amplitude co-modulation: log compresses large excursions, so the pair departs when… | 0.41, 0.39 / 0.70, 0.70 | 0.93 / 0.96 | -0.006 (19/19) | failed (axis screen) | 0.42 |
| 13 | [Low against high band](#band) | assumption · linear-spectral | `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25` · `cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | frequency locus of coupling | 0.72, 0.38 / 0.91, 0.61 | 0.48 / 0.67 | -0.034 (19/19) | failed (not drug-specific) | 0.41 |
| 14 | [Mean against max of the squared barycentre: extreme co-excursion](#bary-max) | principled · higher-order | `bary-sq_euclidean_mean` · `bary-sq_euclidean_max` | joint extreme excursions (co-bursts): pairs whose largest common excursion exceeds what a Gaussian sum with the same… | 0.81, 0.45 / 0.93, 0.53 | 0.73 / 0.71 | -0.019 (18/19) | beyond IAAFT (matched screen) | 0.35 |
| 15 | [Signed against two-sided max of the barycentre: asymmetry of co-excursions](#bary-sign) | assumption · higher-order | `bary_euclidean_max` · `bary-sq_euclidean_max` | asymmetry (skew) of joint excursions: whether pairs co-burst upward and downward alike | 0.43, 0.45 / 0.54, 0.53 | 0.69 / 0.68 | -0.029 (18/19) | beyond IAAFT (matched screen) | 0.35 |
| 16 | [Gaussian GC against model-free TE (GC = 2 TE for Gaussians)](#gc-te) | principled · higher-order | `gc_gaussian_k-1_kt-1_l-1_lt-1` · `te_kernel_W-0.25_k-1` | non-Gaussian predictive (directed) dependence: prediction of the target from the source beyond its own past that a… | 0.14, 0.49 / 0.47, 0.68 | – / – | +0.013 (16/19) | level only; z unreliable | 0.32 |
| 17 | [Lagged MI against GC (conditioning on the target past)](#precedence) | assumption · linear-spectral | `tlmi_gaussian` · `gc_gaussian_k-1_kt-1_l-1_lt-1` | precedence: what the source adds beyond the target's past | 0.68, 0.14 / 0.88, 0.47 | 0.26 / 0.26 | -0.001 (13/19) | raw (10 s) | 0.32 |
| 18 | [\|C\|^2 against \|Im C\| (lag)](#coh-icoh) | assumption · linear-spectral | `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25` · `icoh_multitaper_mean_fs-1_fmin-0_fmax-0-25` | non-zero-lag share of coupling | 0.72, 0.18 / 0.91, 0.46 | -0.11 / -0.45 | +0.017 (14/19) | failed (floor, covariates) | 0.31 |
| 19 | [Coherence against phase-locking value: amplitude-dependent synchrony](#coh-plv) | principled · higher-order | `cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` · `plv_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` | amplitude-dependent synchrony: phase locking concentrated in high- or low-amplitude epochs, beyond the fixed Gaussian… | 0.38, 0.45 / 0.61, 0.73 | 0.78 / 0.86 | +0.002 (12/19) | raw; IAAFT invalid for multitaper | 0.30 |
| 20 | [Zero lag against best lag](#best-lag) | assumption · linear-spectral | `cov-sq_EmpiricalCovariance` · `xcorr-sq_max_sig-True` | coupling carried at a stable non-zero lag | 0.77, 0.74 / 0.91, 0.90 | 0.94 / 0.99 | -0.017 (19/19) | failed (covariates); stable mean lag 10 s | 0.24 |
| 21 | [History length](#history) | assumption · linear-spectral | `gc_gaussian_k-1_kt-1_l-1_lt-1` · `gc_gaussian_MAX-CORR-AIS_k-max-10_tau-max-2` | memory of the interaction | 0.14, 0.10 / 0.47, 0.52 | 0.34 / 0.53 | -0.013 (16/19) | raw; unreliable | 0.16 |
| 22 | [Gaussian GC against symbolic TE](#symbolic) | estimator · higher-order | `gc_gaussian_k-1_kt-1_l-1_lt-1` · `te_symbolic_k-3` | ordinal-pattern (amplitude-free) predictive structure beyond linear-Gaussian | 0.14, 0.19 / 0.47, 0.37 | 0.05 / 0.25 | -0.016 (14/19) | untested | 0.12 |
| 23 | [Rigid against elastic alignment](#warp) | assumption · higher-order | `pdist_euclidean` · `dtw` | variable-lag (warped) alignment | 0.83, 0.41 / 0.94, 0.68 | 0.59 / 0.60 | -0.038 (15/19) | failed construction | 0.11 |
| 24 | [Envelope correlation with against without orthogonalisation](#pec-orth) | assumption · linear-spectral | `pec` · `pec_orth` | lagged share of amplitude coupling (volume-conduction-free) | 0.41, 0.03 / 0.70, 0.16 | 0.58 / 0.49 | +0.038 (18/19) | floor | 0.10 |
| 25 | [Mean against max over frequency or lag](#peak) | assumption · linear-spectral | `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25` · `cohmag_multitaper_max_fs-1_fmin-0_fmax-0-25` | spectral peakedness of coupling (narrowband against broadband) | 0.72, 0.23 / 0.91, 0.26 | 0.61 / 0.50 | +0.004 (12/19) | raw; max SPIs unreliable | 0.09 |
| 26 | [PPC against debiased wPLI (lag, both debiased)](#ppc-dswpli) | assumption · linear-spectral | `ppc_multitaper_mean_fs-1_fmin-0_fmax-0-5` · `dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-5` | non-zero-lag share of phase coupling (lag variability candidate beyond the stationary null) | 0.73, 0.03 / 0.91, 0.12 | 0.30 / 0.25 (ax 0.28) | +0.018 (13/19) | failed (reliability, covariates) | 0.08 |

## Readings

### <a id="mcd"></a>1. Classical against MinCovDet covariance, precision, partial correlation

- **Pairs:** `pcorr_EmpiricalCovariance` · `pcorr_MinCovDet`; `cov-sq_EmpiricalCovariance` · `cov-sq_MinCovDet`; `prec-sq_EmpiricalCovariance` · `prec-sq_MinCovDet`; `cov_EmpiricalCovariance` · `cov_MinCovDet` (+1 more in the JSON).
- **What differs:** estimator of the same covariance: all samples against the reweighted half with the smallest Mahalanobis distance (multivariate, M channels).
- **Character reading:** time-heterogeneous dependence: a fraction of samples departs from the dominant covariance; the axis tracks the trimmed fraction beyond IAAFT.
- **Literal reading:** organisation of classical against robust covariance (or its conditional form).
- **Other consequences:** heavier-tailed or less elliptical joint distribution and a mixture of covariance regimes both raise the trimmed fraction; MCD needs T well above M (near-classical at T 200, M 126); conditional forms add the mediation link. **Main confound:** MCD noise rises with spectral concentration; precision conditioning at M 126.
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** consistent for the covariance of an elliptical Gaussian; expectation of z set by MCD efficiency (about .5 breakdown) and T/M.
- **Measurability:** reliability 0.52, 0.41 / 0.87, 0.81; z₀ – / – (ax 0.83); null SD – / –; matched 2-s net -0.012 (15/19).
- **Validation:** tail axis: raw readouts dominated by marginal shape in construction; beyond-IAAFT reading supported by the mechanism check (trimmed fraction), not by axis construction. **Status:** axis at 10 s; absent at 2 s.
- **Atlas so far:** axis screen passes every gate at 10 s (pcorr|pcorr_mcd -.079, 11 SD; r2|mcd2 -.056), all four agents; absent at 2 s; "episodic coupling" not earned (distinctness fails).
- **Calibration needed:** needs one: heavy-tailed elliptical (t-copula) against covariance-regime mixture at matched trimmed fraction, to name which raises it.
- **Upstream pyspi:** cov-sq_MinCovDet is unbundled (#136): it is the square of cov_MinCovDet. The axis readouts (pcorr|pcorr_mcd, r^2|mcd^2) are unaffected.

### <a id="mi-ksg"></a>2. Gaussian against kNN MI (the anchor)

- **Pairs:** `mi_gaussian` · `mi_kraskov_NN-4_DCE-AUTO`; `mi_gaussian` · `mi_kraskov_NN-4`.
- **What differs:** estimator of the same mutual information: covariance plug-in -1/2 log(1 - r^2) against Kraskov kNN (k 4, Theiler window chosen from the autocorrelation).
- **Character reading:** non-Gaussian (nonlinear) dependence: after normal-scoring, I - G is the KL divergence from the covariance-matched Gaussian; z reads how that departure is organised across edges, not its mean amount.
- **Literal reading:** organisation of Gaussian against kNN information across channel pairs.
- **Other consequences:** kNN bias and variance grow as spectral concentration lowers the effective sample size (raw fall is 60-65% spectral); marginal shape enters raw MI (normal-score first); saturation: G is convex in |r|, so z moves with the spread of r (saturation control); shared gain (route 2) and nonlinear coupling both raise I - G. **Main confound:** spectral concentration (kNN effective sample); shared slow gain.
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** I = G exactly for Gaussian; I - G = D_KL(p || q_rho) >= 0 with normal-scored marginals.
- **Measurability:** reliability 0.82, 0.69 / 0.93, 0.89; z₀ 0.91 / 0.97 (ax 0.96); null SD 0.010 / 0.006; matched 2-s net -0.028 (19/19).
- **Validation:** form axis failed construction at cohort coupling; mi_g|mi_kernel responded to form dispersion at strong coupling (post hoc); the anchor rests on surrogate-data logic, not axis validation. **Status:** axis; replicated (PRIOS, mouse).
- **Atlas so far:** anchor: falls beyond IAAFT under four anaesthetics, 4/4 macaques p .004; PRIOS 6/7; mouse induction 9/9; mostly a Gaussian mixture over ~1-s stretches; KSG k 10 and kernel MI agree; axis screen passes every gate (10 s -.013, 2 s -.019).
- **Calibration needed:** done (shared-gain and static-nonlinearity synthetics in mi_pair/); no further calibration needed for the operational reading.

### <a id="envelope-gamma"></a>3. Envelope correlation against its Gaussian determinant |gamma|^2 (pyspi-free partner)

- **Pairs:** `pec` · `NEW:analytic_gamma2`; `pec_log` · `NEW:analytic_gamma2`.
- **What differs:** broadband Hilbert-envelope correlation against |gamma|^2 = r^2 + r_H^2, the squared zero-lag complex correlation of the analytic signals (r_H = corr(x, H[y])).
- **Character reading:** amplitude co-modulation beyond linear coupling: envelope correlation in excess of what the pair's second-order coupling gives a Gaussian process (shared gain, co-bursts).
- **Literal reading:** organisation of envelope correlation against phase-free zero-lag linear coupling.
- **Other consequences:** |gamma|^2 is phase-free, so lag does not enter (r^2 alone would mix in lag: at a 10-sample lag r^2 .022 against |gamma|^2 .049); broadband envelope mixes frequencies; envelope marginal shape. **Main confound:** shared gain is the target, not a confound; spectral concentration of envelopes.
- **Null:** real - joint spectrum-final IAAFT on z-scored (not normal-scored) windows (gamma is a linear functional of the cross-periodogram, so IAAFT keeps it exactly).
- **Gaussian relation:** verified: corr(|a|^2,|b|^2) = |g|^2; corr(|a|,|b|) and log-power forms closed (identities.json).
- **Measurability:** reliability 0.41, – / 0.70, –; z₀ – / –; null SD – / –; matched 2-s net –.
- **Validation:** not validated. **Status:** untested.
- **Atlas so far:** not tested; bank pec reliability .41/.70; plv|pec and cohmag|pec_log moved raw (literal only).
- **Calibration needed:** shared gain (should rise), Gaussian (0), amplitude-linked decoupling (should fall).

### <a id="mi-kernel"></a>4. Gaussian against kernel MI

- **Pairs:** `mi_gaussian` · `mi_kernel_W-0.25`.
- **What differs:** estimator: covariance plug-in against Gaussian-kernel (box width .25) MI, no Theiler window.
- **Character reading:** non-Gaussian dependence (same quantity as the anchor, different estimator bias).
- **Literal reading:** organisation of Gaussian against kernel information.
- **Other consequences:** kernel bandwidth sets which scales of dependence are seen; no autocorrelation correction: raw level rises with slowing. **Main confound:** kernel autocorrelation bias; bandwidth.
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** as the anchor.
- **Measurability:** reliability 0.82, 0.69 / 0.93, 0.91; z₀ 0.89 / 0.98; null SD 0.013 / 0.004; matched 2-s net -0.039 (19/19).
- **Validation:** as the anchor. **Status:** beyond IAAFT (frozen estimator check).
- **Atlas so far:** full bank -2.2 to -7.0 SD; estimator check (frozen) SUPPORTS, pooled -.0100, p .014; matched 2-s screen -.039, 19/19.
- **Calibration needed:** none needed beyond the anchor.
- **Upstream pyspi:** #134 (open, r14) removes the self-count from kernel MI. The r13 value carries a bias largest where joint-box counts are small, which depends on T, the bandwidth and the joint density (so on coupling), so the IAAFT net need not cancel it. Recompute on r14 in a new output directory, and with #133 prefer mi_kernel_W-0.25_DCE-AUTO, which addresses the autocorrelation consequence listed above.

### <a id="mediation"></a>5. Marginal against conditional (covariance against precision or partial correlation)

- **Pairs:** `cov-sq_EmpiricalCovariance` · `prec-sq_EmpiricalCovariance`; `cov-sq_LedoitWolf` · `prec-sq_LedoitWolf`; `cov_EmpiricalCovariance` · `pcorr_EmpiricalCovariance`; `cov_EmpiricalCovariance` · `prec_EmpiricalCovariance` (+11 more in the JSON).
- **What differs:** conditioning on every other observed channel.
- **Character reading:** mediation: direct against indirect association given the observed channels.
- **Literal reading:** organisation of marginal against partial dependence.
- **Other consequences:** montage, reference, M and T define the conditioning set; precision near-singular at T 200, M 126; node-additive structure (R2 .63-.78 for pcorr). **Main confound:** montage and M; estimator conditioning.
- **Null:** linear-static: IAAFT keeps the covariance, so it is degenerate; read raw with strength covariates, the strength-matched edge set and the stationary Gaussian null.
- **Gaussian relation:** precision = inverse covariance: fixed by the covariance, edgewise relation not fixed.
- **Measurability:** reliability 0.77, 0.56 / 0.91, 0.87; z₀ 0.12 / 0.13; null SD 0.032 / 0.036; matched 2-s net +0.000 (19/19).
- **Validation:** core axis by construction (strong instantaneous mediation, SR 3.25/2.88); fails with lagged, partial mediation. **Status:** raw; local estimator effect.
- **Atlas so far:** global dz ~ 0; local signature in unregularised precision tracks spectral entropy (estimation effect); LedoitWolf fails the held-out test.
- **Calibration needed:** done (construction).
- **Upstream pyspi:** cov_{LedoitWolf,OAS,ShrunkCovariance} and their cov-sq forms are unbundled (#136). One shrinkage factor per window makes z(cov-sq LW, prec-sq LW) = z(cov-sq Emp, prec-sq LW) exactly, so the shortlist contrast now names kept SPIs; prec_* and pcorr_* are unchanged.

### <a id="chebyshev"></a>6. Chebyshev against L2 or L1 distance: extreme disagreement

- **Pairs:** `pdist_euclidean` · `pdist_chebyshev`; `pdist_cityblock` · `pdist_chebyshev`; `pdist_chebyshev` · `pdist_canberra`; `pdist_cosine` · `pdist_chebyshev` (+1 more in the JSON).
- **What differs:** largest single-sample |x - y| against its RMS (or mean).
- **Character reading:** heavy-tailed disagreement: intermittent large divergence of a pair beyond what a Gaussian difference signal with the same spectrum gives.
- **Literal reading:** which pairs have the largest single-sample divergence against which differ most on average.
- **Other consequences:** the expected max depends on T and on the autocorrelation of x - y (identities.json: max/RMS 2.97 white against 2.31 at AR .98, T 200), which joint IAAFT fixes; one-sample maximum is noisy: z0 about .8 under a Gaussian network (noise, not ceiling); transients dominate long windows (fails at 10 s); sensitive to shared gain as well as decoupling (unlike Canberra). **Main confound:** transients; shared gain.
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** E max|d| / rms(d) fixed given T and the spectrum of d; IAAFT keeps the cross-spectrum, so the spectrum of x - y.
- **Measurability:** reliability 0.83, 0.52 / 0.94, 0.56; z₀ 0.78 / 0.75 (ax 0.81); null SD 0.023 / 0.038; matched 2-s net -0.037 (18/19).
- **Validation:** tail axis readouts dominated by marginal shape raw; beyond-IAAFT on normal-scored windows removes that. **Status:** axis at 2 s; fails at 10 s.
- **Atlas so far:** matched 2-s screen -.037 to -.067, 18-19/19, t(3) to -11.5, eyes-open ~0; axis screen l1|linf passes at 2 s (-.049), fails at 10 s; PRIOS 7/7 p .005 (linf reliability .25).
- **Calibration needed:** heavy-tailed difference (t-distributed innovations in the coupling) against Gaussian at matched spectra; shared gain against local decoupling.
- **Upstream pyspi:** pdist_euclidean and pdist_cosine are unbundled (#141); on z-scored windows L2 = sqrt(2T(1 - r)), so take it from r (the shortlist row is pyspi-free anyway). pdist_chebyshev stays.

### <a id="tlmi-ksg"></a>7. Gaussian against kNN lagged MI

- **Pairs:** `tlmi_gaussian` · `tlmi_kraskov_NN-4_DCE-AUTO`; `tlmi_gaussian` · `tlmi_kernel_W-0.25`.
- **What differs:** estimator of I(x_{t-1}; y_t): Gaussian against kNN (DCE) or kernel.
- **Character reading:** non-Gaussian dependence at a one-sample (10 ms) lag; read with the same-sample anchor as its link (lag one sample).
- **Literal reading:** organisation of Gaussian against model-free lagged information.
- **Other consequences:** at 100 Hz a one-sample lag mostly repeats the same-sample dependence through autocorrelation; directed SPI: ordered entries. **Main confound:** autocorrelation; kNN effective sample.
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** equal for Gaussian processes.
- **Measurability:** reliability 0.68, 0.48 / 0.88, 0.78; z₀ 0.81 / 0.94; null SD 0.029 / 0.019; matched 2-s net -0.033 (19/19).
- **Validation:** tlmi not non-Gaussian on surrogates after the sign correction (construction stage 1). **Status:** beyond IAAFT (matched screen).
- **Atlas so far:** matched 2-s screen -.033, 19/19, dc -.032 (same family as the anchor); assumptions run: lagged MI did not move raw.
- **Calibration needed:** none; read as a sibling of the anchor.
- **Upstream pyspi:** The kernel member changes value with #134 (r14) and gains a DCE-AUTO form with #133; the KSG member is unchanged.

### <a id="canberra"></a>8. Euclidean against Canberra distance: gain-invariant agreement

- **Pairs:** `pdist_euclidean` · `pdist_canberra`; `pdist_cosine` · `pdist_canberra`; `pdist_cityblock` · `pdist_canberra`.
- **What differs:** Euclidean (= 2T(1 - r) on z-scored data, amplitude-weighted) against Canberra sum |x-y|/(|x|+|y|), which is invariant to a gain shared by both channels at each sample.
- **Character reading:** amplitude-dependent correlation: whether a pair's correlation differs between high- and low-amplitude samples; a pure shared scale mixture (one gain on an unchanged correlation) leaves it unmoved, amplitude-linked decoupling moves it strongly.
- **Literal reading:** organisation of amplitude-weighted against per-sample scale-free disagreement.
- **Other consequences:** sign of the per-edge excess over f_G(r) gives the direction (iid synthetic, z-scored): correlation rising with amplitude +.10 (gain over a fixed floor), falling with amplitude -.12 (amplitude-linked decoupling), pure shared or separate gain -.004 / -.001, static quadratic -.011; invariance holds on z-scored data only: normal-scoring breaks it (shared gain -.061); a gain over an unscaled noise floor changes SNR with amplitude, so it counts as amplitude-dependent correlation; sign-disagreement samples score 1 whatever their amplitude, so near-zero samples weigh heavily. **Main confound:** channel-specific floors; marginal shape near zero.
- **Null:** analytic Gaussian map canberra = f_G(r) on z-scored windows, and a Gaussian (FT, unit-variance marginals) multivariate surrogate; IAAFT on z-scored windows as a sensitivity check (it keeps heavy marginals, which moves Canberra).
- **Gaussian relation:** static statistic: for a Gaussian process its expectation is a fixed function of r (tabulated in identities.py); gain invariance verified on iid synthetics (identities.json canberra).
- **Measurability:** reliability 0.83, 0.79 / 0.94, 0.93; z₀ 0.96 / 0.98; null SD 0.005 / 0.002; matched 2-s net -0.013 (19/19).
- **Validation:** not validated; calibration below is the construction test. **Status:** beyond IAAFT, normal-scored only.
- **Atlas so far:** matched 2-s screen (normal-scored IAAFT) euclidean|canberra -.013, 19/19, t(3) -6.1, eyes-open -.002; cosine|canberra -.013 19/19; reliability .79/.93.
- **Calibration needed:** iid version done (identities.json canberra); still required at T 200/1000 with realistic spectra, multichannel z-scoring and spatially correlated floors.
- **Upstream pyspi:** pdist_euclidean and pdist_cosine are unbundled (#141); the shortlist already uses the analytic f_G(r). pdist_canberra stays.

### <a id="cityblock"></a>9. L1 against L2 distance: tail weight of disagreement

- **Pairs:** `pdist_euclidean` · `pdist_cityblock`; `pdist_cityblock` · `pdist_cosine`.
- **What differs:** mean against RMS absolute difference.
- **Character reading:** kurtosis of the disagreement signal x - y beyond Gaussian (constant sqrt(2/pi) ratio for Gaussian).
- **Literal reading:** L1 against L2 distance.
- **Other consequences:** near ceiling (z0 ~ .995 under Gaussian); small finite-sample bias at high autocorrelation (.807 against .798 at AR .9, T 200). **Main confound:** ceiling.
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** mean|d| / rms(d) = sqrt(2/pi) (verified).
- **Measurability:** reliability 0.83, 0.82 / 0.94, 0.94; z₀ 0.99 / 0.99; null SD 0.003 / 0.003; matched 2-s net -0.004 (19/19).
- **Validation:** tail axis: raw dominated by marginal shape. **Status:** failed (ceiling).
- **Atlas so far:** assumptions run -.010 to -.017 raw (1.6-2.3x saturation under KTMD); axis screen l1|l2 -.002 (z0 1.00).
- **Calibration needed:** as Chebyshev.
- **Upstream pyspi:** pdist_euclidean and pdist_cosine are unbundled (#141); L2 comes from r. pdist_cityblock stays.

### <a id="rank-linear"></a>10. Pearson against Spearman or Kendall (with a normal-score intermediate)

- **Pairs:** `cov_EmpiricalCovariance` · `spearmanr`; `cov_EmpiricalCovariance` · `kendalltau`; `cov-sq_EmpiricalCovariance` · `spearmanr-sq`; `corr_pearson_tau-1` · `corr_spearman_tau-1` (+29 more in the JSON).
- **What differs:** linear correlation against correlation of ranks.
- **Character reading:** copula departure in concordance: dependence that is monotone but not Gaussian-copula (joint bursts read here in construction), separated from marginal shape by normal-score Pearson.
- **Literal reading:** organisation of linear against rank dependence.
- **Other consequences:** marginal shape of each channel (raw readout tracks it); ceiling z0 ~ .99; saturation curvature from the arcsin map. **Main confound:** marginal shape; ceiling.
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** verified: rho_s = (6/pi) arcsin(rho/2), tau = (2/pi) arcsin(rho) (identities.json); compare against the mapped profile to remove curvature.
- **Measurability:** reliability 0.81, 0.81 / 0.93, 0.93; z₀ 0.99 / 1.00; null SD 0.002 / 0.001; matched 2-s net -0.001 (19/19).
- **Validation:** real - IAAFT zero-lag Pearson|Spearman reads joint bursts (construction stage 3, d -3.2/-2.5); propofol EEG residual killed by the deciding analysis. **Status:** failed (ceiling).
- **Atlas so far:** raw -.01 to -.02 at ceiling; axis screen r|rho -.0007 (too small to read).
- **Calibration needed:** done for joint bursts; a nonlinear coupling-function synthetic is still needed for "monotone nonlinearity".
- **Upstream pyspi:** kendalltau is unbundled (#140) and the lagged corr_* members leave with #144 (open); spearmanr stays, and spearmanr-sq is pending. The shortlist uses Spearman.

### <a id="dcorr"></a>11. r^2 against distance correlation (and IDS)

- **Pairs:** `cov-sq_EmpiricalCovariance` · `dcorr`; `cov-sq_EmpiricalCovariance` · `ids`; `dcorr` · `ids`.
- **What differs:** linear r^2 against a distance-based (dCor) or Fourier-feature (IDS) dependence measure.
- **Character reading:** any-form (including non-monotone) dependence beyond the Gaussian value.
- **Literal reading:** organisation of linear against any-form dependence.
- **Other consequences:** ceiling (z0 ~ .99 under Gaussian; verified .988); dCor is computed on a retained half of the samples (pyspi _RetainedHalf). **Main confound:** saturation; ceiling.
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** verified: Szekely et al. 2007 Thm 7 gives dCor^2 as a closed function of |rho| (identities.json).
- **Measurability:** reliability 0.77, 0.77 / 0.91, 0.91; z₀ 0.99 / 0.99; null SD 0.002 / 0.001; matched 2-s net -0.006 (19/19).
- **Validation:** post hoc at strong coupling r^2|dcorr responded to form dispersion and was non-Gaussian on surrogates. **Status:** failed (animal t).
- **Atlas so far:** full bank -4.5 to -9.7 SD; axis screen r2|dcorr -.002 (fails animal t).
- **Calibration needed:** non-monotone (quadratic) coupling synthetic.
- **Upstream pyspi:** cov-sq_EmpiricalCovariance is unbundled (#136); use r^2 or the Thm-7 map of r, as the shortlist already does.

### <a id="pec-log"></a>12. Envelope correlation on amplitude against log power

- **Pairs:** `pec` · `pec_log`; `pec_orth` · `pec_orth_log`; `pec_orth_abs` · `pec_orth_log_abs`.
- **What differs:** transform of the broadband Hilbert envelope before correlating: |a| against log|a|^2.
- **Character reading:** burst-driven against proportional amplitude co-modulation: log compresses large excursions, so the pair departs when envelope coupling is carried by a few large co-bursts.
- **Literal reading:** organisation of linear against log envelope correlation.
- **Other consequences:** envelope marginal shape (heavy tails) enters both; both are fixed functions of |gamma|^2 for Gaussian signals, so saturation curvature differs. **Main confound:** envelope marginal shape (normal-scoring the signal does not normal-score the envelope).
- **Null:** real - joint spectrum-final IAAFT on z-scored (not normal-scored) windows.
- **Gaussian relation:** verified: corr(|a|,|b|) = (pi/4)(2F1(-1/2,-1/2;1;|g|^2) - 1)/(1 - pi/4); corr(log|a|^2, log|b|^2) = 6 Li2(|g|^2)/pi^2 (identities.json).
- **Measurability:** reliability 0.41, 0.39 / 0.70, 0.70; z₀ 0.93 / 0.96; null SD 0.006 / 0.006; matched 2-s net -0.006 (19/19).
- **Validation:** not validated. **Status:** failed (axis screen).
- **Atlas so far:** axis screen fails; matched 2-s -.006, 19/19 (small).
- **Calibration needed:** co-burst synthetic against proportional gain.

### <a id="band"></a>13. Low against high band

- **Pairs:** `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25` · `cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5`; `ppc_multitaper_mean_fs-1_fmin-0_fmax-0-25` · `ppc_multitaper_mean_fs-1_fmin-0-25_fmax-0-5`; `plv_multitaper_mean_fs-1_fmin-0_fmax-0-25` · `plv_multitaper_mean_fs-1_fmin-0-25_fmax-0-5`; `cohmag_multitaper_max_fs-1_fmin-0_fmax-0-25` · `cohmag_multitaper_max_fs-1_fmin-0-25_fmax-0-5` (+12 more in the JSON).
- **What differs:** 0-25 Hz against 25-50 Hz (effectively 25-40 Hz after the 40-Hz low-pass).
- **Character reading:** frequency locus of coupling.
- **Literal reading:** low- against high-band coupling.
- **Other consequences:** band organisation also reads within-band location; band level moves organisation. **Main confound:** spectra; eyes-open carries 60-75%.
- **Null:** zero-phase and stationary Gaussian nulls with the state mean cross-spectrum (discovery.py lag-null); IAAFT is degenerate.
- **Gaussian relation:** fixed by the cross-spectrum.
- **Measurability:** reliability 0.72, 0.38 / 0.91, 0.61; z₀ 0.48 / 0.67; null SD 0.033 / 0.035; matched 2-s net -0.034 (19/19).
- **Validation:** band is core (high share, spread); ppc L|H organisation is a sub-axis. **Status:** failed (not drug-specific).
- **Atlas so far:** not drug-specific beyond the spectrum; axis screen fails.
- **Calibration needed:** done.
- **Upstream pyspi:** The ppc members are unbundled (#139); the cohmag and plv band pairs stay.

### <a id="bary-max"></a>14. Mean against max of the squared barycentre: extreme co-excursion

- **Pairs:** `bary-sq_euclidean_mean` · `bary-sq_euclidean_max`.
- **What differs:** RMS against maximum of the sum signal (x + y)/2 (the Euclidean barycentre); the mirror of Chebyshev, which uses the difference.
- **Character reading:** joint extreme excursions (co-bursts): pairs whose largest common excursion exceeds what a Gaussian sum with the same spectrum gives.
- **Literal reading:** organisation of mean against peak squared barycentre.
- **Other consequences:** depends on T and the spectrum of x + y (fixed by IAAFT); max is noisy (reliability .45/.53); shared gain raises it (co-bursts are what gain makes). **Main confound:** transients; one-sample maximum.
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** bary-sq mean = (1 + r)/2 exactly; max/mean fixed given T and spectrum of x + y.
- **Measurability:** reliability 0.81, 0.45 / 0.93, 0.53; z₀ 0.73 / 0.71; null SD 0.028 / 0.053; matched 2-s net -0.019 (18/19).
- **Validation:** not validated. **Status:** beyond IAAFT (matched screen).
- **Atlas so far:** matched 2-s screen -.019, 18/19, dc -.021.
- **Calibration needed:** co-burst synthetic against Gaussian; read with Chebyshev (disagreement extremes) as its contrast.
- **Upstream pyspi:** bary-sq_euclidean_mean is unbundled (#141); it is (1 + r)/2, so z(bary-sq mean, bary-sq max) = z(r, bary-sq max) exactly. bary-sq_euclidean_max stays.

### <a id="bary-sign"></a>15. Signed against two-sided max of the barycentre: asymmetry of co-excursions

- **Pairs:** `bary_euclidean_max` · `bary-sq_euclidean_max`.
- **What differs:** largest positive co-excursion against largest co-excursion of either sign.
- **Character reading:** asymmetry (skew) of joint excursions: whether pairs co-burst upward and downward alike.
- **Literal reading:** signed against squared barycentre maximum.
- **Other consequences:** marginal skew enters (IAAFT keeps marginals, so the net removes it); reference sign conventions. **Main confound:** waveform asymmetry of each channel (removed by IAAFT marginals).
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** symmetric for Gaussian: both maxima fixed given T and spectrum.
- **Measurability:** reliability 0.43, 0.45 / 0.54, 0.53; z₀ 0.69 / 0.68; null SD 0.051 / 0.074; matched 2-s net -0.029 (18/19).
- **Validation:** not validated. **Status:** beyond IAAFT (matched screen).
- **Atlas so far:** matched 2-s screen -.029, 18/19, dc -.027 (reliability .43/.45 at 2 s).
- **Calibration needed:** skewed common-burst synthetic.

### <a id="gc-te"></a>16. Gaussian GC against model-free TE (GC = 2 TE for Gaussians)

- **Pairs:** `gc_gaussian_k-1_kt-1_l-1_lt-1` · `te_kernel_W-0.25_k-1`; `gc_gaussian_k-1_kt-1_l-1_lt-1` · `te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1`.
- **What differs:** estimator of transfer entropy with history 1: pyspi gc_gaussian is Gaussian TE in nats (= Geweke GC / 2) against kernel or kNN TE.
- **Character reading:** non-Gaussian predictive (directed) dependence: prediction of the target from the source beyond its own past that a linear-Gaussian model misses.
- **Literal reading:** organisation of linear against model-free one-step prediction.
- **Other consequences:** target own nonlinear dynamics (non-Gaussian AIS) changes what conditioning removes; TE estimator noise is large at T <= 1000; kernel TE has no Theiler window. **Main confound:** TE estimator noise; target non-Gaussian self-prediction (AIS rises under every drug).
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** Barnett, Barrett & Seth 2009: GC = 2 TE for jointly Gaussian variables; verified in pyspi source (gc identifier is the Gaussian TE estimator).
- **Measurability:** reliability 0.14, 0.49 / 0.47, 0.68; z₀ – / –; null SD – / –; matched 2-s net +0.013 (16/19).
- **Validation:** the level median(te - gc) is the only non-Gaussian reader that passed construction (SR .78, surrogates keep .14); the z version was insensitive. **Status:** level only; z unreliable.
- **Atlas so far:** kNN version unusable (te_kraskov_DCE reliability .03/.07); TE - GC level residual small 10-s effect of unestablished cause; kernel lagged MI against TE rose in the movers.
- **Calibration needed:** quadratic coupling synthetic exists (predictive.py controls: positive residual on coupled pairs).
- **Upstream pyspi:** te_kernel changes value with #134 (r14: self-count removed, and the next value scaled like the past, as JIDT does). Compute shortlist row 11 on r14, or use te_kernel_W-0.25_DCE-AUTO_k-1 (#133).

### <a id="precedence"></a>17. Lagged MI against GC (conditioning on the target past)

- **Pairs:** `tlmi_gaussian` · `gc_gaussian_k-1_kt-1_l-1_lt-1`; `tlmi_kernel_W-0.25` · `te_kernel_W-0.25_k-1`; `tlmi_kraskov_NN-4_DCE-AUTO` · `te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1`.
- **What differs:** conditioning on the target's own past sample.
- **Character reading:** precedence: what the source adds beyond the target's past.
- **Literal reading:** lagged MI against TE/GC.
- **Other consequences:** autocorrelation is what conditioning removes; GC reliability .14 at 2 s. **Main confound:** autocorrelation.
- **Null:** stationary Gaussian null (Gaussian pair); IAAFT for the kernel pair.
- **Gaussian relation:** both functions of lag-0/1 auto- and cross-covariances.
- **Measurability:** reliability 0.68, 0.14 / 0.88, 0.47; z₀ 0.26 / 0.26; null SD 0.037 / 0.046; matched 2-s net -0.001 (13/19).
- **Validation:** predictive axis extended set. **Status:** raw (10 s).
- **Atlas so far:** Gaussian link falls under KT, MD, KTMD at 10 s, propofol ~0; kernel link rose in the movers.
- **Calibration needed:** VAR synthetic with known GC.
- **Upstream pyspi:** The kernel link (tlmi_kernel against te_kernel) changes value with #134 (r14); the Gaussian link is unchanged. Compute the kernel link in shortlist row 14 on r14.

### <a id="coh-icoh"></a>18. |C|^2 against |Im C| (lag)

- **Pairs:** `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25` · `icoh_multitaper_mean_fs-1_fmin-0_fmax-0-25`; `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5` · `icoh_multitaper_mean_fs-1_fmin-0_fmax-0-5`; `cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` · `icoh_multitaper_mean_fs-1_fmin-0-25_fmax-0-5`; `cohmag_multitaper_max_fs-1_fmin-0_fmax-0-5` · `icoh_multitaper_max_fs-1_fmin-0_fmax-0-5` (+2 more in the JSON).
- **What differs:** removal of the zero-phase (real) part of coherency.
- **Character reading:** non-zero-lag share of coupling.
- **Literal reading:** coherence magnitude against imaginary coherence.
- **Other consequences:** |Im C| has an estimation floor that rises as spectral concentration lowers the effective sample; window-to-window strength change raises the beyond-stationary residual. **Main confound:** estimation floor; spectral covariates.
- **Null:** zero-phase and stationary Gaussian nulls with the state mean cross-spectrum (discovery.py lag-null); IAAFT is degenerate.
- **Gaussian relation:** both fixed by the cross-spectrum.
- **Measurability:** reliability 0.72, 0.18 / 0.91, 0.46; z₀ -0.11 / -0.45; null SD 0.034 / 0.032; matched 2-s net +0.017 (14/19).
- **Validation:** lag sub-axis (SR .66-1.60, frequency-confounded). **Status:** failed (floor, covariates).
- **Atlas so far:** zero-phase null reproduces 50-75% of the rise; stable mean lag under KT and MD; axis screen: spectral covariates reverse it.
- **Calibration needed:** lag-null calibration exists.

### <a id="coh-plv"></a>19. Coherence against phase-locking value: amplitude-dependent synchrony

- **Pairs:** `cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` · `plv_multitaper_mean_fs-1_fmin-0-25_fmax-0-5`; `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5` · `plv_multitaper_mean_fs-1_fmin-0_fmax-0-5`; `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25` · `plv_multitaper_mean_fs-1_fmin-0_fmax-0-25`; `cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` · `ppc_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` (+3 more in the JSON).
- **What differs:** |C|^2 (amplitude-weighted) against PLV (unit-modulus phases), same band and tapers.
- **Character reading:** amplitude-dependent synchrony: phase locking concentrated in high- or low-amplitude epochs, beyond the fixed Gaussian relation.
- **Literal reading:** organisation of amplitude-weighted against phase-only synchrony.
- **Other consequences:** band averaging of two different monotone maps adds saturation curvature; multitaper tapers mix neighbouring bins, so spectrum-final IAAFT is not degenerate for linear-linear multitaper pairs (axis screen post hoc); broad and low band at ceiling (z0 .92-.96). **Main confound:** within-window non-stationarity of the spectrum (reaches multitaper statistics through taper mixing).
- **Null:** IAAFT only with a Daniell (frequency-smoothed periodogram) estimator, which IAAFT keeps exactly; with pyspi multitaper, subtract the net of a linear-linear multitaper floor pair (cohmag lo|cohmag bb).
- **Gaussian relation:** verified: |E exp(i dphi)| = (pi/4)|g| 2F1(1/2,1/2;2;|g|^2) per frequency for proper complex Gaussian pairs.
- **Measurability:** reliability 0.38, 0.45 / 0.61, 0.73; z₀ 0.78 / 0.86; null SD 0.024 / 0.016; matched 2-s net +0.002 (12/19).
- **Validation:** phase_amplitude axis not core. **Status:** raw; IAAFT invalid for multitaper.
- **Atlas so far:** high band raw +.03 to +.06 (KT, KTMD, PF), MD -.055; matched 2-s low -.005 17/19.
- **Calibration needed:** amplitude-gated phase coupling synthetic (locking only above an envelope threshold) against Gaussian.
- **Upstream pyspi:** One ppc member is unbundled (#139). The shortlist row (Daniell coherence against mapped PLV) is pyspi-free, so no impact.

### <a id="best-lag"></a>20. Zero lag against best lag

- **Pairs:** `cov-sq_EmpiricalCovariance` · `xcorr-sq_max_sig-True`; `cov_EmpiricalCovariance` · `xcorr_max_sig-True`.
- **What differs:** correlation at lag 0 against the maximum over lags within +-T/4.
- **Character reading:** coupling carried at a stable non-zero lag.
- **Literal reading:** zero-lag against best-lag correlation.
- **Other consequences:** estimation floor at 2 s (spectral concentration); signed version adds sign flips at half-period lags; zero-padded lagged correlation is not fixed by the circular periodogram, so IAAFT is not degenerate. **Main confound:** dominant frequency; floor.
- **Null:** zero-phase and stationary Gaussian nulls with the state mean cross-spectrum (discovery.py lag-null); IAAFT is degenerate.
- **Gaussian relation:** fixed by the cross-spectrum.
- **Measurability:** reliability 0.77, 0.74 / 0.91, 0.90; z₀ 0.94 / 0.99; null SD 0.009 / 0.003; matched 2-s net -0.017 (19/19).
- **Validation:** timing_best_shift not core. **Status:** failed (covariates); stable mean lag 10 s.
- **Atlas so far:** 10 s: mean-lag term carries most under KT and MD; axis screen: covariate adjustment reverses it.
- **Calibration needed:** lag-null exists.
- **Upstream pyspi:** #144 (open) removes xcorr_* and adds lagcorr_peak (signed r at the lag of max |r|, over min(max_lag, T // 4) lags) and lagcorr_lag. If it merges before the run, shortlist row 13 becomes z(r^2, lagcorr_peak^2) with lagcorr_lag as a direct lag readout; r13 xcorr z0 and reliability do not transfer.

### <a id="history"></a>21. History length

- **Pairs:** `gc_gaussian_k-1_kt-1_l-1_lt-1` · `gc_gaussian_MAX-CORR-AIS_k-max-10_tau-max-2`; `te_kraskov_NN-4_DCE-AUTO_k-1_kt-1_l-1_lt-1` · `te_kraskov_NN-4_DCE-AUTO_k-2_kt-1_l-1_lt-1`; `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-None` · `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-5_order-1`; `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-25_order-None` · `sgc_parametric_mean_fs-1_fmin-1e-05_fmax-0-25_order-1` (+16 more in the JSON).
- **What differs:** target history length or AR order.
- **Character reading:** memory of the interaction.
- **Literal reading:** short against long history.
- **Other consequences:** autocorrelation structure. **Main confound:** reliability .03-.52.
- **Null:** zero-phase and stationary Gaussian nulls with the state mean cross-spectrum (discovery.py lag-null); IAAFT is degenerate.
- **Gaussian relation:** fixed by autocovariances.
- **Measurability:** reliability 0.14, 0.10 / 0.47, 0.52; z₀ 0.34 / 0.53; null SD 0.036 / 0.068; matched 2-s net -0.013 (16/19).
- **Validation:** history axis extended set. **Status:** raw; unreliable.
- **Atlas so far:** TE k1|k2 rose under every drug but partners unreliable.
- **Calibration needed:** n/a.

### <a id="symbolic"></a>22. Gaussian GC against symbolic TE

- **Pairs:** `gc_gaussian_k-1_kt-1_l-1_lt-1` · `te_symbolic_k-3`.
- **What differs:** Gaussian TE (history 1) against ordinal-pattern TE (pattern length 3).
- **Character reading:** ordinal-pattern (amplitude-free) predictive structure beyond linear-Gaussian.
- **Literal reading:** linear against symbolic TE.
- **Other consequences:** pattern length and history differ (not minimal); reliability .19/.37. **Main confound:** reliability; history mismatch.
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** none closed.
- **Measurability:** reliability 0.14, 0.19 / 0.47, 0.37; z₀ 0.05 / 0.25; null SD 0.022 / 0.037; matched 2-s net -0.016 (14/19).
- **Validation:** n/a. **Status:** untested.
- **Atlas so far:** not read.
- **Calibration needed:** n/a.

### <a id="warp"></a>23. Rigid against elastic alignment

- **Pairs:** `pdist_euclidean` · `dtw`; `dtw` · `lcss`; `dtw_constraint-itakura` · `lcss_constraint-itakura`; `dtw_constraint-sakoe-chiba_radius-auto` · `lcss_constraint-sakoe-chiba` (+4 more in the JSON).
- **What differs:** rigid against warped alignment.
- **Character reading:** variable-lag (warped) alignment.
- **Literal reading:** Euclidean against DTW distance.
- **Other consequences:** window length; spectral shape. **Main confound:** spectral adjustment removes it.
- **Null:** real - joint spectrum-final IAAFT (window-matched, all channels jointly, >= 3 draws), normal-scored windows.
- **Gaussian relation:** none closed.
- **Measurability:** reliability 0.83, 0.41 / 0.94, 0.68; z₀ 0.59 / 0.60; null SD 0.054 / 0.056; matched 2-s net -0.038 (15/19).
- **Validation:** warp failed construction validation. **Status:** failed construction.
- **Atlas so far:** 10-s only, removed by spectral adjustment; DTW options surrogate-driven.
- **Calibration needed:** failed.
- **Upstream pyspi:** pdist_euclidean is unbundled (#141); take L2 from r.

### <a id="pec-orth"></a>24. Envelope correlation with against without orthogonalisation

- **Pairs:** `pec` · `pec_orth`; `pec_log` · `pec_orth_log`.
- **What differs:** pairwise orthogonalisation (Hipp et al. 2012) removes the zero-lag linear part before the envelope.
- **Character reading:** lagged share of amplitude coupling (volume-conduction-free).
- **Literal reading:** envelope correlation with and without orthogonalisation.
- **Other consequences:** pec_orth reliability .02-.16. **Main confound:** estimation floor; reliability.
- **Null:** zero-phase and stationary Gaussian nulls with the state mean cross-spectrum (discovery.py lag-null); IAAFT is degenerate.
- **Gaussian relation:** both fixed by the cross-spectrum for Gaussian signals.
- **Measurability:** reliability 0.41, 0.03 / 0.70, 0.16; z₀ 0.58 / 0.49; null SD 0.030 / 0.061; matched 2-s net +0.038 (18/19).
- **Validation:** phase_lag axis member. **Status:** floor.
- **Atlas so far:** mostly floor; residual also in sleep and eyes open.
- **Calibration needed:** lag-null exists.

### <a id="peak"></a>25. Mean against max over frequency or lag

- **Pairs:** `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25` · `cohmag_multitaper_max_fs-1_fmin-0_fmax-0-25`; `xcorr-sq_max_sig-True` · `xcorr-sq_mean_sig-True`; `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5` · `cohmag_multitaper_max_fs-1_fmin-0_fmax-0-5`; `cohmag_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` · `cohmag_multitaper_max_fs-1_fmin-0-25_fmax-0-5` (+21 more in the JSON).
- **What differs:** peak against average over frequency (or lags).
- **Character reading:** spectral peakedness of coupling (narrowband against broadband).
- **Literal reading:** peak against mean.
- **Other consequences:** max SPIs unreliable (.03-.26). **Main confound:** channel spectral concentration.
- **Null:** zero-phase and stationary Gaussian nulls with the state mean cross-spectrum (discovery.py lag-null); IAAFT is degenerate.
- **Gaussian relation:** fixed by cross-spectrum.
- **Measurability:** reliability 0.72, 0.23 / 0.91, 0.26; z₀ 0.61 / 0.50; null SD 0.022 / 0.019; matched 2-s net +0.004 (12/19).
- **Validation:** spectral_peak extended set. **Status:** raw; max SPIs unreliable.
- **Atlas so far:** survivor in sufficiency checks; max SPIs unreliable here.
- **Calibration needed:** n/a.
- **Upstream pyspi:** The xcorr member leaves with #144 (open); the ppc members are unbundled (#139). Read the frequency peak through plv mean|max, which carries ppc (PPC = (N PLV^2 - 1)/(N - 1) per frequency).

### <a id="ppc-dswpli"></a>26. PPC against debiased wPLI (lag, both debiased)

- **Pairs:** `ppc_multitaper_mean_fs-1_fmin-0_fmax-0-5` · `dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-5`; `ppc_multitaper_mean_fs-1_fmin-0_fmax-0-25` · `dswpli_multitaper_mean_fs-1_fmin-0_fmax-0-25`; `ppc_multitaper_mean_fs-1_fmin-0-25_fmax-0-5` · `dswpli_multitaper_mean_fs-1_fmin-0-25_fmax-0-5`.
- **What differs:** zero-lag removal on debiased phase statistics.
- **Character reading:** non-zero-lag share of phase coupling (lag variability candidate beyond the stationary null).
- **Literal reading:** PPC against debiased wPLI.
- **Other consequences:** dswpli reliability .01-.15; strength change also raises the residual. **Main confound:** reliability; spectral covariates.
- **Null:** zero-phase and stationary Gaussian nulls with the state mean cross-spectrum (discovery.py lag-null); IAAFT is degenerate; across-window phase dispersion for lag variability.
- **Gaussian relation:** fixed by cross-spectrum.
- **Measurability:** reliability 0.73, 0.03 / 0.91, 0.12; z₀ 0.30 / 0.25 (ax 0.28); null SD 0.017 / 0.022; matched 2-s net +0.018 (13/19).
- **Validation:** lag sub-axis. **Status:** failed (reliability, covariates).
- **Atlas so far:** residual KTMD +.08, PF +.13 beyond stationary; axis screen fails covariates and reliability.
- **Calibration needed:** lag-null calibration exists.
- **Upstream pyspi:** ppc is unbundled (#139); plv|dswpli carries the same lag reading, with the plv band means .98-.99 against ppc rather than exact.

## Exclusions (scored 0)

| Entry | Representative pair | Reason |
|---|---|---|
| Bray-Curtis against other distances | `pdist_euclidean` · `pdist_braycurtis` | no clean reading on signed data |
| Broadband against one band | `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5` · `cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-25` | nested bands; use low against high |
| Cointegration and other drift statistics | `coint_johansen_trace_stat_order-0_ardiff-1` · `coint_johansen_trace_stat_order-0_ardiff-10` | reads within-window drift, not dependence character |
| kNN with against without the Theiler window | `mi_kraskov_NN-4` · `mi_kraskov_NN-4_DCE-AUTO` | estimator diagnostic (autocorrelation bias) |
| Bias and option diagnostics | `dcorr` · `dcorr_biased` | estimator diagnostic |
| Gaussian against kernel or Kozachenko entropies (je, ce, si, xme, cce) | `je_gaussian` · `je_kernel_W-0.5` | redundant with the anchor (je, ce) or node-level (cce, si, xme); report channel-level separately |
| Zero against fixed lag (samples) | `cov_EmpiricalCovariance` · `corr_pearson_tau-1` | frequency in sample units, not lag character |
| Identity and saturation controls | `cov-sq_EmpiricalCovariance` · `mi_gaussian` | control, not a reading; keep as the floor |
| Kendall against Spearman | `kendalltau` · `spearmanr` | ceiling estimator pair; any departure is below the saturation control |
| MinCovDet against rank correlation | `cov-sq_MinCovDet` · `kendalltau-sq` | non-minimal; same axis as classical against MinCovDet |
| Same-time MI against GC (non-minimal precedence) | `mi_gaussian` · `gc_gaussian_k-1_kt-1_l-1_lt-1` | non-minimal; read its links |
| PLI against wPLI (and dsPLI against dswPLI) | `pli_multitaper_mean_fs-1_fmin-0_fmax-0-5` · `wpli_multitaper_mean_fs-1_fmin-0_fmax-0-5` | estimator weighting; unreliable |
| Empirical against shrinkage covariance (LedoitWolf, OAS, Shrunk) | `prec-sq_EmpiricalCovariance` · `prec-sq_LedoitWolf` | estimator diagnostic (shrinkage / conditioning); LedoitWolf against MinCovDet is two differences, use Empirical against MinCovDet |
| Signed against squared | `cov_EmpiricalCovariance` · `cov-sq_EmpiricalCovariance` | single-statistic property (sign of r); not a pair character |
| Structural zeros and non-finite SPIs | `plv_multitaper_mean_fs-1_fmin-0_fmax-0-5` · `pli_multitaper_mean_fs-1_fmin-0_fmax-0-5` | structural; use ppc\|dswpli or cohmag\|icoh for lag |

## Gaussian relations: numerical check ([`identities.json`](identities.json))

| Relation | Check | Largest deviation |
|---|---|---|
| Spearman = (6/π) arcsin(ρ/2) | ρ .1–.9, n 20,000 | 0.007 |
| Kendall = (2/π) arcsin ρ | ρ .1–.9, n 4,000 | 0.019 (sampling) |
| dCor² Thm 7 (Székely et al. 2007) | ρ .1–.9, n 2,500 | 0.026 |
| corr(\|a\|², \|b\|²) = \|γ\|² | lags 0–10, n 400,000 | 0.005 |
| corr(\|a\|, \|b\|) = (π/4)(₂F₁(−½,−½;1;\|γ\|²) − 1)/(1 − π/4) | same | 0.004 |
| corr(log\|a\|², log\|b\|²) = 6 Li₂(\|γ\|²)/π² | same | 0.005 |
| PLV = (π/4)\|γ\| ₂F₁(½,½;2;\|γ\|²) | \|γ\| .2–.95 | 0.0007 |
| Chatterjee ξ = (3/π) arcsin((1 + ρ²)/2) − ½ | ρ 0–.95, n 20,000 | 0.002; SD at T 200 0.025–0.049 |
| mean\|d\| / rms(d) = √(2/π) | T 200/1000, AR 0–.98 | 0.031 (finite-T bias at AR .98) |
| max\|d\| / rms(d) depends on T and spectrum | same | 2.31–3.42 |
| z under a 16-channel Gaussian network | 10 reps, T 1000 | r²·dCor² 0.988, r·Spearman 0.998, L1·L2 0.995, L∞·L2 0.79 |

The GC = 2 TE relation (Barnett, Barrett & Seth 2009) is definitional in pyspi: `gc_gaussian` is the Gaussian TE estimator in nats (source, `infotheory.py`). Geweke's spectral identity relates `sgc` integrated over frequency to time-domain GC with the true marginal restricted model, not to `gc_gaussian` with history 1, so it gives no fixed edgewise map in this bank (and `sgc` reliability is .01–.18 at 2 s).

## Proposals for the reading guide (owned by the main session)

- Static-statistic principle: on z-scored channels the expectation of any time-exchangeable bivariate statistic (rank correlations, MI estimators, dCor, IDS, L1/L2/Canberra distances, barycentre moments) under a Gaussian process is a fixed function of r. Any pair from that set is therefore a principled non-minimal pair beyond IAAFT, and its reading is the feature the more flexible statistic weights. Maxima (Chebyshev, barycentre max) are the exception: their expectation also depends on T and the spectrum of x -+ y, which joint IAAFT fixes.
- Classes with no entry in assumptions.READINGS: chebyshev vs cityblock (axis screen), canberra vs euclidean/cosine (gain-invariant agreement on z-scored data), aggregation mean vs max for bary-sq (joint extreme co-excursion).
- Envelope correlation: under Gaussianity its partner is |gamma|^2 = r^2 + corr(x, H[y])^2, not r^2; z(pec, r^2) mixes in lag.
- Remove saturation from z0 by mapping the Gaussian-determined partner through the known relation (arcsin for ranks, Thm 7 for dCor, -1/2 log for MI, closed envelope forms) before correlating; z0 is then 1 minus estimator noise under the null.
- Canberra gain invariance holds on z-scored, not normal-scored, windows; a normal-scored IAAFT construction removes it.
- SPIs unbundled upstream as exact affine functions of a kept SPI (pdist_cosine, bary-sq_euclidean_mean, the shrinkage covariances) give exactly the z of the kept SPI, because z is a Pearson correlation of edge profiles; read and name such pairs through the kept SPI.

## Limits

- **Scores are judgements.** Interpretability is curated, and measurability uses wake reliability and wake z₀ of the representative pair only, so read the ranks as an ordering for the next computation, not as evidence.
- **New statistics have no numbers yet.** z₀ and null SD exist only for minimal pairs and named pairs in the stored banks; non-minimal and pyspi-free pairs (|γ|², normal-score Pearson, mapped partners) are untested here.
- **Synthetic checks are idealised.** The Canberra and identity checks are iid or AR(1), long-T synthetics; finite-T, multichannel, realistic-spectrum calibration is part of the next step.
- **The matched net comes from one construction.** It is normal-scored, so it is not the null Canberra needs.
