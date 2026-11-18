# Real-world results overview: z against other representations

Compiled 6 November 2026 from existing outputs only (no pyspi, no fitting, no Gadi jobs). Every number below was read from the source named for its table or line. Status labels: **V** = read from the output file in this compilation; **Vn** = taken from a findings note that records its own independent verification, but whose output file is not in the repository and was not re-read; **I** = inferred here; **not run** = no such comparison exists. Protocols are never pooled: each table keeps its own split, metric and threshold rule, and rows are not ranked across tables.

Re-verified 10 November 2026 against the source JSONs in the repository (an independent session): every method's BA and Brier in the fMRI primary role (15 methods), the four discovery datasets (16 methods each) and Siena (15 methods), the discovery contrasts that Matrix 4 quotes, and the P / P+z / P+core / P+core+z scores, increments and intervals for propofol, LEMON and NeuroTycho r10, plus propofol's S/Q rows and cohort scores. No discrepancy was found. Cells marked Vn remain unverified here: Gadi was offline for maintenance, so the K289 NeuroTycho outputs, CHBMP JSONs and the Siena prediction files were not re-read. The [dataset inventory](#dataset-inventory-and-pyspi-versions-10-november) and the [comparator gap matrix](#comparator-gap-matrix-10-november) below were added in that pass.

## Summary

What holds:

- z is a usable, transferable recording representation on several real tasks: NeuroTycho anaesthesia (BA .930 z-PCA / .945 z-PLS after an animal/drug/period shift), the CHB-MIT screen (BA .958) and LANL (BA 1.0, lowest Brier .0044). Its ranking (AUROC) usually transfers better than its fixed 0.5 threshold.
- z beats the Pearson edge vector (r) on NeuroTycho (current bank), CHB-MIT screen, LANL and propofol sedation, roughly ties on AD/FTD, LEMON and HMC, and loses on fMRI, Siena and GRABMyo.
- Adding z to Pearson + strength gives an AUROC gain whose interval excludes zero in two cases: NeuroTycho (+.352 [+.067, +.667], two target animals) and propofol moderate versus baseline (+.182 [+.087, +.295], 20 people). In NeuroTycho the validated character core accounts for it; in propofol it does not (z still adds +.133 [+.060, +.213] over Pearson + strength + core).

What does not hold:

- z never adds over Pearson + strength + channel spectra (the S block) in any cohort tested (AD/FTD, LEMON, propofol, NeuroTycho); every interval includes or lies below zero. Channel power or spectra alone match or beat z on every grouped-CV cohort. Whatever z adds over Pearson is redundant, for classification, with per-channel spectra; this is redundancy in AUROC, not proof that z reads spectra.
- No dataset shows z beating the strongest comparator. Winners: NeuroTycho enriched raw encoder, InceptionTime ensemble, MultiRocket-Hydra and QUANT (r10 bank: spectra alone .984 against z .836); fMRI MiniRocket, aligned MPIs, Pearson and tangent-space covariance; Siena univariate and aligned MPIs (AUROC); HMC nonlinear univariate and aligned encoder; GRABMyo InceptionTime; LANL ties among many methods at BA 1.0.
- The comparator fill (11 November, full agreed ladder) adds two things. The K per-SPI edge means (mean-SPI) match z on every cohort and nearly every transfer column, so z's discriminating content is mostly in the per-SPI levels. And the best single SPI chosen inside each training fold is the strongest statistical block on AD vs HC (.928) and AD vs FTD (.780, cointegration in 20 of 25 folds, every other method ≤ .655): post hoc and unconfirmed, but the one new additive candidate. Details in [Comparator fill](#comparator-fill-11-november). Under a nested additivity test (11 November), z and Pearson are complementary in AD vs HC and LEMON (each carries information the other lacks; LEMON .856 and .868 → .908 fused), but z adds nothing over channel spectra plus per-SPI levels in any of the five contrasts tested.
- The one frozen independent confirmation failed. CHB-MIT → Siena: m+z did not beat m (Brier −.009 [−.045, +.034]), z fell from BA .958 to .685 and lost to m, aligned MPIs and Pearson. On fMRI, Pearson+z is worse than Pearson (Brier −.022 [−.040, −.005]).
- Surrogates: every replicated named-axis state effect (NeuroTycho, propofol, AD/FTD, LEMON) is reproduced by multivariate IAAFT, so it is linear-spectral. The surviving beyond-surrogate candidates are NeuroTycho's top z pairs (robust precision versus rank/ids; residual d −1.3 to −2.2, 0/6 units) and propofol's linear–rank agreement (−.83/−.82). The non-Gaussian MI level shift was an estimator artefact and is withdrawn.
- CHBMP IQ is a pre-registered null (0 of 31,415 z features at FWER, N = 135). Its post-hoc age signal is mostly shared with spectra and did not replicate in LEMON.
- Unsupervised: on NeuroTycho induction, centred z-PC1 separates animals rather than states (AUROC .70–.72), while mean |r| and mean-SPI PC1 reach .96–1.00.

## Representations: what each name means

- **z**: per-recording signed Pearson correlation between SPI edge profiles (ordered off-diagonal entries), `unified_ordered_v3`, then source-fitted PCA (z-PCA) or PLS (z-PLS, NeuroTycho only) and a linear head. Catalogue: historical K = 289, pyspi `65317c9` (NeuroTycho pilot, September); current K = 283, computation `3.0.0.r10` (everything else, except the NeuroTycho induction bank, whose version was not checked).
- **r / Pearson**: the Pearson edge vector (upper triangle, channel identity kept), not a single mean. In the character-axes cohorts, **P** = Pearson edges + strength covariates (mean and SD of |r| over edges, r² versus Gaussian-MI organisation).
- **m (rich SPI summaries)**: 23 descriptors of each SPI's edge distribution (mean, SD, skewness, kurtosis, 19 quantiles; `research/representation/shared/representation_attribution.py:rich_marginals`), i.e. 6,509 features at K = 283 and 6,647 at K = 289. This is **not** the K-vector of edge means. **mean-SPI** (K edge means) appears only in the unsupervised NeuroTycho induction analysis. **m-RBF** / **m_rbf** = the same features with an RBF readout.
- **full aligned MPIs**: every SPI matrix flattened in a fixed ordered-edge order; contains z.
- **best single SPI**: one aligned MPI chosen by source-only search over the catalogue.
- **univariate** (discovery screen and Siena): per channel, physical-unit amplitude moments, quantiles, mean absolute difference, zero-crossing rate, log and relative power in 32 bands (`dataset_screen/prepare.py:univariate`); **univariate RBF** = nonlinear readout of the same.
- **spectra**: NeuroTycho pilot = Welch per-channel summaries with absolute power. fMRI = per-channel log power in 32 bands (544) and a channel-pooled version (160). Character-axes cohorts: **power** = absolute band power; **spectral** = log band power of standardized inputs (relative spectra). **S** = P + spectral; **Q** = P + per-channel skewness, kurtosis and envelope CV.
- **core** (character axes, revised core v3b): band high share, band spread and mediation, with the sub-axis readouts the sufficiency script includes (band organisation, lag, lag sin φ, form te−gc, location frequency).
- **Raw learners**: InceptionTime (five-seed ensemble), MiniRocket (aeon 1.5, 10,000 kernels; ridge one-hot head in the discovery screen, Siena and the character-axes cohorts, L2 logistic head in the NeuroTycho follow-up and fMRI; the "extended alpha grid" applies only to the ridge head), aligned raw encoder (time-aligned, no channel-identity embedding; NeuroTycho's "matched raw encoder" is the channel-exchangeable predecessor), enriched raw encoder (NeuroTycho only: same architecture with more source windows and augmentation), learned SPI pooling (shared edge MLP with mean/SD aggregation; NeuroTycho only), tsfresh Efficient and catch22 (NeuroTycho only). ROCKET and MultiRocket were not run on any real dataset.

## Matrix 1: source → target transfer, fixed threshold 0.5 (BA / Brier)

Group-balanced BA and Brier on held-out targets, source-only model selection, no target calibration. Lower Brier is better. GRABMyo is 17-class (chance BA ≈ .059); all others are binary (chance .5). Units are the independent targets.

Sources (all V unless marked): **NT** = `research/representation/neurotycho/notes/neurotycho-transfer-findings.md`, `neurotycho-library-followup-findings.md`, `research/representation/learned_baselines/notes/inceptiontime-followup.md` (Vn; outputs under `results/neurotycho_*_2609*/` are not in the repository); **fMRI** = `results/representation/character-transfer-261031/evaluation-summary.json` (role `confirmation_primary`) and `separate-fusion-summary.json`; **CHB, HMC, GRAB, LANL** = `results/representation/empirical-discovery-261031/discovery-review.json` and `minirocket-extension.json`; **Siena** = `results/representation/siena-confirmation-261102/confirmation-summary.json`.

| Representation | NT anaesthesia (K289; 2 target animals, 128 windows) | NT anaesthesia (r10, 283 SPIs; 2 target animals; BA pooled over windows) | fMRI rest/film (r10; 9 people, 36 scans) | CHB-MIT screen (6 people, 26 records) | Siena confirm (14 patients, 54 windows) | HMC N2/REM (12 nights, 144) | GRABMyo 17-class (8 people, 136) | LANL (4 configs, 40) |
|---|---| --- |---|---|---|---|---|---|
| z (PCA) | .930 / .062 | .836ᵃ (BA only) | .778 / .136 | .958 / .082 | .685 / .229 | .736 / .180 | .243 / .425 | 1.0 / .004 |
| z (PLS) | .945 / .060 | not run | not run | not run | not run | not run | not run | not run |
| Pearson (r) | not run | .57ᵃ (BA only) | .972 / .065 | .806 / .185 | .786 / .160 | .757 / .175 | .353 / .409 | .650 / .245 |
| Pearson + z | not run | not run | .861 / .087 | not run | not run | not run | not run | not run |
| P (Pearson + strength) | not run | .664 / .249 | not run | not run | not run | not run | not run | not run |
| P + z | not run | .812 / .171 | not run | not run | not run | not run | not run | not run |
| P + core | not run | .852 / .104 | not run | not run | not run | not run | not run | not run |
| P + core + z | not run | .836 / .103 | not run | not run | not run | not run | not run | not run |
| m (23 descriptors/SPI) | .836 / .074 | not run | .833 / .142 | .917 / .088 | .810 / .159 | .701 / .196 | .250 / .430 | 1.0 / .012 |
| m, nonlinear (RBF) | not run | not run | .889 / .111 | .792 / .119 | .667 / .219 | .694 / .192 | .213 / .432 | 1.0 / .009 |
| m + graph summaries | .852 / .075 | not run | not run | not run | not run | not run | not run | not run |
| m + z | .953 / .047 | not run | .806 / .110 | 1.0 / .083 | .756 / .168 | .757 / .177 | .279 / .426 | 1.0 / .006 |
| m + z, matched no-z diagnostic | not run | not run | not run | .917 / .106 | not run | .701 / .196 | .250 / .430 | 1.0 / .011 |
| full aligned MPIs | not run | not run | 1.0 / .056 | .847 / .092 | .845 / .116 | .778 / .160 | .456 / .374 | 1.0 / .011 |
| best single SPI (source-selected) | not run | not run | .972 / .061 | .639 / .261 | .738 / .202 | .778 / .181 | .404 / .390 | 1.0 / .017 |
| source-selected baseline + z | not run | not run | .917 / .079 (single SPI + z) | .611 / .217 (single SPI) | .810 / .160 (single SPI) | .826 / .124 (univariate) | .493 / .350 (univariate RBF) | 1.0 / .012 (single SPI) |
| source-selected baseline, matched no-z diagnostic | not run | not run | not run | .639 / .261 | not run | .806 / .143 | .493 / .350 | 1.0 / .017 |
| validity mask | .625 / .228 | not run | .389 / .264 | .528 / .250 | .589 / .248 | .521 / .254 | .140 / .457 | .875 / .167 |
| per-channel univariate | not run | not run | not run | .833 / .141 | .643 / .213 | .792 / .148 | .449 / .375 | 1.0 / .081 |
| nonlinear univariate (RBF) | not run | not run | not run | .792 / .147 | .536 / .325 | .889 / .111 | .485 / .350 | 1.0 / .118 |
| conventional spectra | .867 / .094 (Welch, absolute power) | .984ᵃ (BA only) | .917 / .089 (per channel); .806 / .196 (pooled) | in univariate | in univariate | in univariate | in univariate | in univariate |
| learned SPI pooling | .831 / .134 | not run | not run | not run | not run | not run | not run | not run |
| InceptionTime (5-seed ensemble) | .953 / .057 | not run | .722 / .176 | .833 / .194 | .696 / .303 | .729 / .164 | .522 / .303 | .750 / .171 |
| MiniRocket, original alpha grid | .898 / .061 (logistic head) | not run | 1.0 / .013 | .792 / .135 | .536 / .292 | .840 / .119 | .515 / .352 | .750 / .148 |
| MiniRocket, extended alpha grid | not run | not run | not run | .792 / .135 (α 1e3) | .536 / .292 | .826 / .128 (α 1e4) | .515 / .352 (α 1e3) | .750 / .167 (α 1e5) |
| ROCKET / MultiRocket | not run | not run | not run | not run | not run | not run | not run | not run |
| tsfresh (Efficient) + logistic | .844 / .084 | not run | not run | not run | not run | not run | not run | not run |
| catch22 + logistic | .836 / .139 | not run | not run | not run | not run | not run | not run | not run |
| aligned raw encoder | .831 / .167 (matched, channel-exchangeable) | not run | .611 / .223 | .819 / .122 | .571 / .320 | .847 / .092 | .176 / .433 | .750 / .237 |
| enriched raw encoder | .964 / .028 | not run | not run | not run | not run | not run | not run | not run |
| 13 character axes / axes + levels | not run | .914ᵃ / .938ᵃ (BA only) | .81ᵃ (axes, BA only) | not run | .60ᵃ (axis pairs, BA only) | not run | not run | not run |

Notes on Matrix 1: The r10 NeuroTycho column comes from `results/representation/character-axes-261102/sufficiency-core-v3b/neurotycho.json` (P rows, V) and, for ᵃ cells, from the character-axes evaluation of the existing banks reported in `research/representation/character_axes/findings.md` (Vn; that script also gives fMRI Pearson BA 1.0, its output file was not located). Its BA is pooled over the 128 target windows, unlike the pilot's date-then-animal average, and its full transfer panel (`neurotycho-source-review.json`) was never evaluated on targets. Matched no-z diagnostics keep the fusion projection, penalty and bandwidth but drop z (`discovery-review.json`). NT neural and MiniRocket rows average three seeds' metrics, except InceptionTime, which is the five-member probability ensemble (member mean BA .844). The NT reduced view (M8/T1000) is: z-PCA .906, z-PLS .914, m .664, m+z .820, spectra .953, matched raw .872, enriched raw .974, pooling .781 (BA; Vn). fMRI separate fusion (single SPI 32 PCs + z-PCA ≤ 50) is .889 / .082 (V). Siena's source-selected baseline is the frozen CHB single-SPI model; its extended MiniRocket equals the original.

## Matrix 2: AUROC in the transfer group (threshold-free)

| Representation | NT, date-averaged within animal (Vn) | fMRI, mean within participant / pooled (V) | Siena, pooled over 54 windows (V, recomputed here) |
|---|---|---|---|
| z | 1.0 (PCA), 1.0 (PLS) | 1.0 / .963 | .828 |
| Pearson | not run | 1.0 / .997 | .875 |
| Pearson + z | not run | 1.0 / 1.0 | not run |
| m | 1.0 | .972 / .907 | .907 |
| m, nonlinear | not run | .972 / .938 | .770 |
| m + z | 1.0 | 1.0 / .969 | .870 |
| full aligned MPIs | not run | 1.0 / 1.0 | .941 |
| best single SPI / + z | not run | 1.0 / 1.0; 1.0 / 1.0 | .761; baseline + z .855 |
| per-channel univariate / RBF | not run | not run | .942 / .652 |
| spectra | 1.0 | .972 / .991 (per channel) | in univariate |
| validity | .843 | .444 / .412 | .637 |
| learned SPI pooling | .999 | not run | not run |
| InceptionTime | .998 (ensemble) | 1.0 / .997 | .834 |
| MiniRocket | 1.0 | 1.0 / 1.0 | .613 |
| tsfresh / catch22 | .999 / 1.0 (catch22 pooled .939) | not run | not run |
| aligned (matched) raw encoder | .998 | .722 / .741 | .748 |
| enriched raw encoder | 1.0 | not run | not run |

Sources: NT notes as above; fMRI `evaluation-summary.json` and `prediction-audit.json`; Siena pooled AUROC recomputed in this compilation from `/g/data/ql44/we2614/mts-spi-study/representation/empirical/siena-confirmation-261102/predictions-{statistical,neural,minirocket}.json`, matching the values in `docs/context/spi-spi-representation-evaluation.md`. The discovery screen (CHB, HMC, GRAB, LANL) reports no AUROC (gap 2).

## Matrix 3: grouped cross-validation AUROC (character-axes cohorts, r10)

Participant-grouped, class-stratified five-fold CV repeated five times; unit probability = mean window probability; L2 logistic, C by inner grouped CV, blocks PCA-capped at 32 inside folds. Propofol units are participant × level (within-participant). NeuroTycho is the exception: source KTMD animals → propofol targets, target animal excluded, AUROC pooled over 128 target windows.

Sources (V, copied into the repository by this compilation from Gadi): `results/representation/character-axes-261102/cohort-evaluation/<cohort>-{evaluation,minirocket}.json` (from scratch `.../character-axes-261102/<cohort>/`), `sufficiency-core-v3b/<cohort>.json` and `beyond-channels/<cohort>.json` (from gdata `.../character-axes-261102/`). NT column rows marked Vn come from `research/representation/character_axes/findings.md` ("Existing banks"); that output file was not located.

| Representation | AD vs HC (65 people) | FTD vs HC (52) | AD vs FTD (59) | LEMON old vs young (41 / 91) | Propofol moderate vs baseline (20) | mild vs baseline (20) | moderate vs recovery (20) | NT r10 transfer (2 target animals) |
|---|---|---|---|---|---|---|---|---|
| absolute power | .857 | .847 | .586 | .932 | .995 | .810 | .948 | not run |
| relative spectra | .874 | .819 | .467 | .910 | .992 | .902 | .948 | 1.0 (Vn) |
| Pearson | .835 | .799 | .531 | .859 | .747 | .710 | .845 | .428 (Vn) |
| z | .837 | .796 | .525 | .874 | .930 | .757 | .927 | 1.0 (Vn) |
| 13 axes | .641 | .601 | .558 | .738 | .770 | .623 | .780 | .978 (Vn) |
| axes + levels | .669 | .547 | .599 | .850 | .922 | .677 | .853 | .983 (Vn) |
| Pearson + axes + levels | .880 | .793 | .569 | .902 | .820 | .762 | .883 | not run |
| power + Pearson | .888 | .862 | .579 | .946 | .995 | .823 | .945 | not run |
| power + Pearson + axes + levels | .879 | .850 | .595 | .945 | .988 | .845 | .948 | not run |
| MiniRocket (ridge scores) | .813 | .825 | .429 | .945 | .975 | .795 | .920 | not run |
| P (Pearson + strength) | .840 | .793 | .533 | .856 | .745 | .705 | .845 | .600 |
| P + z | .849 | .808 | .531 | .915 | .927 | .768 | .910 | .952 |
| P + core | .840 | .787 | .452 | .899 | .795 | .713 | .887 | .969 |
| P + core + z | .859 | .804 | .540 | .914 | .927 | .763 | .910 | 1.0 |
| S (P + relative spectra) | .865 | .823 | .471 | .946 | .990 | .920 | .953 | 1.0 |
| S + z | .848 | .810 | .492 | .935 | .973 | .907 | .927 | .999 |
| S + core | .867 | .822 | .441 | .947 | .988 | .920 | .953 | 1.0 |
| Q (P + channel value shape) | .835 | .799 | .541 | .852 | .753 | .705 | .842 | .695 |
| Q + z | .852 | .813 | .541 | .912 | .925 | .768 | .907 | .950 |
| m, aligned MPIs, single SPI, InceptionTime, tsfresh, catch22 | not run | not run | not run | not run | not run | not run | not run | not run |

The NT r10 Vn values (findings.md) and the sufficiency P/S/Q values come from different scripts with different block sets and aggregation; the Vn BA values are spectra .984, axes .914, axes+levels .938, z .836, Pearson .57. For each cohort, BA and Brier are in the same JSON files.

## Matrix 4: increments with 95% intervals

Character-axes cohorts, AUROC difference, paired participant bootstrap (NT: bootstrap over two target animals, so its intervals are coarse). V; same sources as Matrix 3.

| Increment | AD vs HC | FTD vs HC | AD vs FTD | LEMON | Prop. moderate | Prop. mild | Prop. mod vs rec | NT r10 |
|---|---|---|---|---|---|---|---|---|
| z over P | +.009 [−.090, .104] | +.015 [−.102, .138] | −.001 [−.150, .150] | +.059 [−.000, .124] | **+.182 [.087, .295]** | +.062 [−.045, .158] | +.065 [−.022, .158] | **+.352 [.067, .667]** |
| core over P | .000 [−.054, .049] | −.006 [−.079, .064] | −.081 [−.161, −.017] | **+.043 [.012, .079]** | +.050 [.000, .118] | +.007 [−.052, .070] | **+.042 [.005, .088]** | **+.370 [.067, .663]** |
| z over P + core | +.019 [−.063, .107] | +.016 [−.086, .126] | +.088 [−.054, .227] | +.016 [−.028, .064] | **+.133 [.060, .213]** | +.050 [−.040, .128] | +.022 [−.047, .095] | +.031 [.000, .031] |
| z over S | −.017 [−.075, .032] | −.013 [−.076, .049] | +.021 [−.083, .124] | −.012 [−.034, .010] | −.017 [−.060, .000] | −.012 [−.087, .055] | −.025 [−.078, .027] | −.001 [−.001, .001] |
| core over S | +.002 [−.013, .018] | −.001 [−.018, .013] | −.030 [−.056, −.009] | +.001 [−.008, .010] | −.002 [−.015, .000] | .000 [−.018, .015] | .000 [−.015, .015] | .000 [.000, .001] |
| z over Q | +.017 [−.095, .126] | +.013 [−.094, .130] | .000 [−.156, .150] | +.061 [−.002, .128] | **+.172 [.073, .288]** | +.063 [−.032, .150] | +.065 [−.022, .157] | **+.255 [.027, .463]** |
| Pearson + axes + levels over Pearson | +.045 [−.010, .103] | −.006 [−.084, .070] | +.037 [−.047, .126] | **+.043 [.011, .077]** | **+.073 [.008, .157]** | +.052 [−.003, .115] | +.037 [−.010, .088] | not run |
| power + axes over power | +.003 [−.024, .030] | +.012 [−.010, .047] | +.008 [−.048, .069] | −.004 [−.013, .004] | −.003 [−.022, .012] | +.025 [−.018, .080] | +.003 [−.010, .020] | not run |

Transfer group, Brier improvement (positive = the augmented model is better) and BA difference, paired whole-group bootstrap conditional on fitted models, no multiplicity adjustment. V; sources as Matrix 1.

| Contrast | fMRI (9 people) | CHB screen (6) | Siena (14) | HMC (12) | GRABMyo (8) | LANL (4) |
|---|---|---|---|---|---|---|
| m + z over m | +.032 [−.009, .072]; BA −.028 | +.005 [−.028, .053]; BA +.083 | −.009 [−.045, .034]; BA −.054 | +.018 [−.002, .038]; BA +.056 | **+.004 [.002, .006]**; BA +.029 | +.006 [−.001, .013]; BA 0 |
| Pearson + z over Pearson | **−.022 [−.040, −.005]**; BA −.111 | not run | not run | not run | not run | not run |
| selected baseline + z over baseline | single SPI: **−.019 [−.039, −.000]**; BA −.056 | single SPI: **+.044 [.024, .067]**; BA −.028 | single SPI: +.042 [−.005, .092]; BA +.071 | univariate: +.024 [−.007, .064]; BA +.035 | univariate RBF: +.001 [−.002, .003]; BA +.007 | single SPI: +.004 [−.001, .009]; BA 0 |
| matched no-z diagnostic (baseline + z over same projection without z) | not run | +.044 [.024, .067] | not run | +.019 [−.002, .044] | +.000 [−.002, .003] | +.004 [−.001, .009] |
| z over m | not reported | +.006 [−.051, .073] | **−.070 [−.128, −.011]** | +.016 [−.011, .041] | +.005 [−.004, .015] | **+.007 [.001, .014]** |
| z over aligned MPIs | not reported | +.010 [−.050, .075] | **−.114 [−.168, −.059]** | −.020 [−.053, .017] | **−.051 [−.064, −.037]** | **+.006 [.003, .010]** |
| z over InceptionTime | not reported | +.112 [−.020, .256] | +.074 [−.055, .206] | −.016 [−.053, .022] | **−.122 [−.191, −.040]** | +.167 [−.008, .342] |

fMRI separate fusion (single SPI 32 PCs + z-PCA ≤ 50) over single SPI: Brier **−.021 [−.042, −.002]**, BA −.083 (V). The GRABMyo m+z and LANL z-over-m intervals exclude zero but are tiny absolute Brier changes (≤ .007).

## Comparator fill (11 November)

The agreed ladder on the existing banks, under each column's frozen protocol (rules in the [gap matrix](#comparator-gap-matrix-10-november) section). Code `comparator_fill/` at `7b4c2f9` (AD/FTD, LEMON, propofol statistical runs; `ee1ddb7` for the rest, whose single-SPI step did not differ), aeon 1.5.0. Sources (V): `results/representation/comparator-fill-261110/{stat,library}/<dataset>.json`; tables printed by `python -m research.representation.comparator_fill.collate`. fMRI and Siena rows are post hoc (outcomes known) and exploratory; no multiplicity adjustment anywhere. Library rows on the cohorts use one five-fold split, not 5 × 5, so they carry more split variance than the statistical rows. Each column keeps its bank's pyspi version (inventory below). `>.999` rounds to 1 but is below it.

What it shows:

- **Reproduction.** z, Pearson, relative spectra and absolute power reproduce every Matrix 3 cell to three decimals, and the NeuroTycho r10 z and Pearson AUROC (1.0, .428; previously Vn) are reproduced, so those cells are now V.
- **mean-SPI matches z.** The K edge-averaged SPI values (283 numbers per window) are never beaten by z in a cohort (AD vs FTD: mean-SPI ahead, −.109 [−.216, +.001]) and are within ±.01 Brier on every transfer column except Siena, where mean-SPI is better (−.060 [−.116, −.004]). Adding z to mean-SPI helps nowhere except GRABMyo (+.011 Brier). Inferred: whatever z carries for these labels is mostly also in the K per-SPI levels.
- **Spectra.** In the cohorts z never adds over spectra, power or Pearson + spectra (every interval includes or lies below zero; power + z on propofol mild +.092 [+.005, +.175] is the lone exception and is one of 23 such cohort tests). On the discovery screen z adds Brier over spectra on CHB-MIT (+.097 [+.024, +.173]) and LANL (+.038), and over Pearson + spectra on HMC (+.046 [+.005, +.096]) and LANL; it does not on Siena or fMRI, the two held-out sets.
- **Best single SPI, chosen inside each training fold, is the strongest statistical block on the AD contrasts.** AD vs HC .928 (z − single SPI −.091 [−.195, −.003]), with a lagged correlation (`corr_*_tau-2*`) chosen in 20 of 25 folds; AD vs FTD .780 (−.255 [−.402, −.102]), with a cointegration statistic (`coint_johansen_*`, `coint_aeg_*`) chosen in 20 of 25 folds, where every other block and library is ≤ .655. This is post hoc, n = 59, and unconfirmed; cointegration statistics are sensitive to slow drift, so a recording or amplitude difference between the AD and FTD groups is the first alternative to rule out. It is the one candidate this fill produced for the atlas-style "small but additive" result, and it needs an independent AD/FTD cohort.
- **z's capacity.** Full-rank z beats the PCA-32 z on LEMON (+.058 [+.018, +.104]) and propofol mild (+.117 [+.042, +.185]), so the cohort PCA cap cost z some AUROC. z's first PC alone loses on every column (−.10 to −.41 AUROC): the discriminating content is not in z's leading component.
- **Libraries and baselines.** ROCKET, MultiRocket-Hydra and QUANT match or beat z on most columns. NeuroTycho r10: MultiRocket-Hydra BA 1.0, QUANT .992 / .008, spectra alone .984 / .011, best single SPI .984 / .013, against z .836 / .104. fMRI: tangent-space covariance 1.0 / .042, better than z (Brier −.094 [−.156, −.031]), which agrees with tangent-space being the standard connectome baseline (Dadi et al., NeuroImage 2019; citation from memory, unverified here). z beats tangent on propofol moderate (+.105 [+.023, +.178]) and LANL. QUANT inverts on Siena (BA .262, AUROC .283).

Notation and the additivity test (Fable 5.2 review, 11 November). X ⊕ z is a fused model, z | X is z with what X expresses linearly removed, and U(z; X), U(X; z) and the shared term split cross-validated log-loss between them. The "+" rows above are not tests of additivity: they concatenate PCA-capped blocks under one penalty, and Pearson + spectra + z falls below Pearson + spectra in seven of eight cohort columns, which a sound fusion cannot do in expectation, so those drops are estimator variance. The replacement is [`fusion.py`](../../comparator_fill/fusion.py): every model, single blocks included, goes through one non-negative stacking meta-learner on inner out-of-fold logits, so the compared models are nested and equally calibrated; z | X uses cross-fitted ridge residuals on all of X's features; commonality terms carry participant-bootstrap intervals. The decisive contrast is (S ⊕ mean-SPI) ⊕ z against S ⊕ mean-SPI, run only where S has headroom: AD/FTD, LEMON and propofol mild (jobs 183628003–183628005, snapshot `comparator-fill-c5a833e`, outputs `comparator-fill-261110/fusion/`). z | X answers the same question as z's partial effect in the joint model; it is the interpretive readout, not an independent test.

**⊕ results (11 November; V, `results/representation/comparator-fill-261110/fusion/<cohort>.json`).** Unit-level AUROC; U = unique contribution in bits of cross-validated log-loss with 95% participant-bootstrap intervals; every model through the same stacking meta-learner.

| Contrast (units) | r / z / r ⊕ z | U(z; r) | U(r; z) | S ⊕ mean-SPI → ⊕ z | U(z; S ⊕ mean-SPI) |
|---|---|---|---|---|---|
| AD vs HC (65) | .836 / .838 / .871 | **+.080 [+.027, +.124]** | **+.072 [+.020, +.121]** | .863 → .864 | +.000 [−.006, +.007] |
| FTD vs HC (52) | .789 / .796 / .832 | +.012 [−.045, +.056] | **+.079 [+.043, +.121]** | .828 → .828 | −.006 [−.026, +.010] |
| AD vs FTD (59) | .530 / .452 / .471 | −.011 [−.030, +.007] | −.002 [−.016, +.010] | .606 → .587 | −.003 [−.008, +.001] |
| LEMON old vs young (132) | .856 / .868 / .908 | **+.088 [+.034, +.141]** | **+.150 [+.077, +.222]** | .944 → .943 | −.007 [−.016, +.001] |
| Propofol mild vs baseline (40) | .708 / .748 / .745 | **+.051 [+.018, +.089]** | +.001 [−.017, +.015] | .890 → .898 | +.002 [−.009, +.012] |

- **r ⊕ z is complementary.** In AD vs HC and LEMON both unique terms are positive and the shared term is positive: r and z each carry label information the other lacks, so the union beats both (LEMON .856 and .868 → .908, fused − r +.053 [+.011, +.097]). This is complementarity, not synergy (synergy would need a negative shared term, seen nowhere). In propofol mild z contains what r carries (U(r; z) ≈ 0).
- **The decisive contrast is null everywhere.** Over S ⊕ mean-SPI, z's unique contribution is within ±.007 bits with intervals spanning zero in all five contrasts. For these labels, what z adds to Pearson is already in channel spectra plus per-SPI levels.
- **Against mean-SPI alone,** z adds a little in AD vs HC (+.031 [+.003, +.060]; z | mean-SPI .655) and propofol mild (+.040 [+.014, +.066]), not elsewhere.
- **Residualised z below chance.** z | S and z | S ⊕ mean-SPI sit at .39–.49 in every contrast, and z | mean-SPI at .273 in AD vs FTD. Where z has no unique information, cross-fitted residuals anticorrelate with the label in held-out folds [I]; read z | X below .5 as null, not as an inverted signal. AD vs FTD has no block above about .61, consistent with the fold-structure warning below.

Before the AD vs FTD single-SPI result is cited: several methods score well below chance on that contrast (ROCKET .377, z PC1 .255), which points to fold-level structure, so it needs a permutation null over the whole inner-selection pipeline and a drift control (detrended or high-passed windows), since cointegration statistics respond to slow drift.

### Transfer columns, fixed threshold 0.5 (BA / Brier; adds to Matrix 1)

| Representation | NT r10 (BA pooled over 128 windows) | fMRI primary | CHB-MIT screen | Siena (post hoc) | HMC | GRABMyo | LANL |
|---|---|---|---|---|---|---|---|
| z, first PC only | .594 / .264 | .667 / .170 | .792 / .201 | .649 / .217 | .667 / .212 | .088 / .466 | 1.0 / .012 |
| z, full rank | .820 / .149 | .806 / .126 | .917 / .077 | .685 / .233 | .785 / .171 | .243 / .423 | 1.0 / .006 |
| Pearson + z | .812 / .160 | not run | .931 / .095 | .774 / .144 | .743 / .169 | .449 / .389 | 1.0 / .008 |
| per-channel spectra alone | .984 / .011 | not run | .792 / .155 | .643 / .231 | .785 / .162 | .471 / .386 | 1.0 / .046 |
| spectra + z | .867 / .106 | .889 / .076 | .917 / .058 | .673 / .196 | .792 / .145 | .471 / .382 | 1.0 / .008 |
| per-channel univariate + z | not run | not run | .958 / .081 | .738 / .186 | .826 / .124 | .441 / .373 | 1.0 / .007 |
| Pearson + spectra | .977 / .013 | 1.0 / .056 | .833 / .138 | .643 / .195 | .764 / .171 | .463 / .376 | 1.0 / .047 |
| Pearson + spectra + z | .859 / .100 | .972 / .066 | .833 / .123 | .804 / .166 | .819 / .125 | .456 / .375 | 1.0 / .008 |
| mean-SPI (K edge means) | .891 / .065 | .833 / .131 | .917 / .088 | .792 / .169 | .729 / .189 | .221 / .429 | 1.0 / .008 |
| mean-SPI + z | .844 / .101 | .833 / .108 | 1.0 / .075 | .720 / .177 | .785 / .176 | .294 / .418 | 1.0 / .024 |
| tangent-space covariance | .852 / .109 | 1.0 / .042 | .708 / .133 | .649 / .251 | .792 / .153 | .382 / .395 | .675 / .235 |
| per-channel catch22 | .828 / .093 | .944 / .069 | .889 / .111 | .631 / .263 | .764 / .203 | .346 / .412 | 1.0 / .127 |
| z (PCA), now with Brier (NT r10 only) | .836 / .104 | — | — | — | — | — | — |
| Pearson (r) (NT r10 only) | .570 / .335 | — | — | — | — | — | — |
| m (NT r10 only) | .906 / .058 | — | — | — | — | — | — |
| m + z (NT r10 only) | .867 / .094 | — | — | — | — | — | — |
| full aligned MPIs (NT r10 only) | .891 / .081 | — | — | — | — | — | — |
| best single SPI (NT r10 only) | .984 / .013 | — | — | — | — | — | — |
| ROCKET | .945 / .059 | .944 / .097 | .778 / .179 | .738 / .165 | .812 / .145 | .500 / .370 | .975 / .058 |
| MiniRocket (aeon, this fill) | .930 / .056 | 1.0 / .058 | .792 / .135 | .536 / .292 | .840 / .120 | .515 / .352 | .750 / .167 |
| MultiRocket-Hydra | 1.0 / .054 | 1.0 / .063 | .875 / .132 | .589 / .297 | .806 / .112 | .551 / .353 | .750 / .152 |
| QUANT | .992 / .008 | .917 / .110 | .667 / .205 | .262 / .421 | .764 / .150 | .419 / .353 | 1.0 / .025 |

### Transfer columns, pooled AUROC (adds to Matrix 2; the "existing fit" rows close gap 2 for the discovery screen)

| Representation | NT r10 (pooled over windows) | fMRI (mean within / pooled) | CHB-MIT screen | HMC | GRABMyo (one-vs-rest) | LANL | Siena |
|---|---|---|---|---|---|---|---|
| z, first PC only | .740 | 1.0 / .843 | .817 | .743 | .554 | 1.0 | .706 |
| z, full rank | .982 | .917 / .932 | .982 | .833 | .739 | 1.0 | .871 |
| Pearson + z | .964 | not run | 1.0 | .827 | .847 | 1.0 | .916 |
| per-channel spectra alone | 1.0 | not run | .870 | .881 | .882 | 1.0 | .923 |
| spectra + z | .983 | 1.0 / .991 | .994 | .903 | .881 | 1.0 | .841 |
| per-channel univariate + z | not run | not run | .970 | .936 | .895 | 1.0 | .938 |
| Pearson + spectra | >.999 | 1.0 / 1.0 | .911 | .881 | .881 | 1.0 | .956 |
| Pearson + spectra + z | .999 | 1.0 / 1.0 | .929 | .936 | .890 | 1.0 | .910 |
| mean-SPI (K edge means) | 1.0 | 1.0 / .895 | 1.0 | .792 | .747 | 1.0 | .874 |
| mean-SPI + z | 1.0 | 1.0 / .960 | 1.0 | .819 | .768 | 1.0 | .856 |
| tangent-space covariance | .949 | 1.0 / 1.0 | .988 | .881 | .868 | .732 | .857 |
| per-channel catch22 | 1.0 | .972 / .994 | .994 | .795 | .806 | 1.0 | .771 |
| ROCKET | 1.0 | 1.0 / 1.0 | .864 | .871 | .915 | 1.0 | .853 |
| MiniRocket (aeon, this fill) | 1.0 | 1.0 / 1.0 | .876 | .920 | .910 | .975 | .613 |
| MultiRocket-Hydra | 1.0 | 1.0 / 1.0 | .929 | .932 | .907 | .920 | .698 |
| QUANT | .993 | .972 / .960 | .769 | .872 | .884 | 1.0 | .283 |
| z (PCA) (existing fit) | see Matrix 2 | see Matrix 2 | .994 | .816 | .736 | 1.0 | see Matrix 2 |
| Pearson (existing fit) | see Matrix 2 | see Matrix 2 | .822 | .810 | .842 | .665 | see Matrix 2 |
| m (existing fit) | see Matrix 2 | see Matrix 2 | 1.0 | .779 | .746 | 1.0 | see Matrix 2 |
| m + z (existing fit) | see Matrix 2 | see Matrix 2 | 1.0 | .825 | .754 | 1.0 | see Matrix 2 |
| full aligned MPIs (existing fit) | see Matrix 2 | see Matrix 2 | 1.0 | .852 | .919 | 1.0 | see Matrix 2 |
| best single SPI (existing fit) | see Matrix 2 | see Matrix 2 | .749 | .805 | .864 | 1.0 | see Matrix 2 |
| source baseline + z (existing fit) | see Matrix 2 | see Matrix 2 | .793 | .936 | .889 | 1.0 | see Matrix 2 |
| per-channel univariate (existing fit) | see Matrix 2 | see Matrix 2 | .917 | .896 | .896 | 1.0 | see Matrix 2 |
| univariate RBF (existing fit) | see Matrix 2 | see Matrix 2 | .870 | .959 | .885 | 1.0 | see Matrix 2 |
| validity (existing fit) | see Matrix 2 | see Matrix 2 | .562 | .514 | .574 | .865 | see Matrix 2 |
| InceptionTime (existing fit) | see Matrix 2 | see Matrix 2 | .882 | .922 | .933 | 1.0 | see Matrix 2 |
| MiniRocket (original) (existing fit) | see Matrix 2 | see Matrix 2 | .876 | .917 | .910 | .858 | see Matrix 2 |
| aligned raw encoder (existing fit) | see Matrix 2 | see Matrix 2 | .911 | .949 | .685 | .953 | see Matrix 2 |

### Cohort grouped-CV AUROC (adds to Matrix 3; the four "check" rows re-run existing cells)

| Representation | AD vs HC | FTD vs HC | AD vs FTD | LEMON | Propofol moderate vs baseline | mild vs baseline | moderate vs recovery | NT r10 transfer |
|---|---|---|---|---|---|---|---|---|
| check: z | .837 | .796 | .525 | .874 | .930 | .757 | .927 | 1.0 |
| check: Pearson | .835 | .799 | .531 | .859 | .747 | .710 | .845 | .428 |
| check: relative spectra | .874 | .819 | .467 | .910 | .992 | .902 | .948 | 1.0 |
| check: absolute power | .857 | .847 | .586 | .932 | .995 | .810 | .948 | not run |
| z, first PC only | .738 | .727 | .255 | .486 | .518 | .573 | .620 | .740 |
| z, full rank | .802 | .726 | .493 | .932 | .975 | .875 | .932 | .982 |
| Pearson + z | .850 | .811 | .525 | .916 | .927 | .768 | .910 | .964 |
| power + z | .874 | .859 | .529 | .908 | .958 | .902 | .935 | not run |
| relative spectra + z | .835 | .810 | .466 | .916 | .975 | .923 | .927 | .983 |
| Pearson + spectra | .867 | .816 | .473 | .946 | .990 | .920 | .953 | >.999 |
| Pearson + spectra + z | .848 | .813 | .490 | .935 | .973 | .907 | .927 | .999 |
| mean-SPI | .811 | .793 | .634 | .897 | .960 | .743 | .948 | 1.0 |
| mean-SPI + z | .846 | .790 | .537 | .883 | .953 | .818 | .935 | 1.0 |
| m (23 descriptors/SPI) | .834 | .754 | .603 | .921 | .970 | .823 | .920 | 1.0 |
| m + z | .819 | .753 | .545 | .903 | .950 | .853 | .940 | 1.0 |
| full aligned MPIs | .862 | .841 | .655 | .922 | .980 | .898 | .930 | 1.0 |
| best single SPI (inner-fold) | .928 | .795 | .780 | .889 | .890 | .785 | .860 | .984 |
| tangent-space covariance | .784 | .781 | .595 | .866 | .825 | .657 | .867 | .949 |
| per-channel catch22 | .873 | .811 | .511 | .927 | .932 | .828 | .898 | 1.0 |
| ROCKET | .875 | .792 | .377 | .956 | .960 | .850 | .912 | 1.0 |
| MiniRocket (aeon, this fill) | .854 | .820 | .510 | .940 | .958 | .777 | .920 | 1.0 |
| MultiRocket-Hydra | .873 | .825 | .501 | .956 | .950 | .838 | .930 | 1.0 |
| QUANT | .853 | .852 | .614 | .924 | .915 | .815 | .905 | .993 |

### Increments with 95% intervals (adds to Matrix 4)

| AUROC difference [95%] | AD vs HC | FTD vs HC | AD vs FTD | LEMON | Propofol moderate vs baseline | mild vs baseline | moderate vs recovery | NT r10 transfer |
|---|---|---|---|---|---|---|---|---|
| pearson_z - pearson | +.014 [−.087, +.111] | +.012 [−.098, +.129] | −.006 [−.156, +.136] | +.057 [−.004, +.121] | +.180 [+.087, +.290] | +.058 [−.050, +.152] | +.065 [−.022, +.158] | +.536 [+.206, +.957] |
| spectral_z - spectral | −.038 [−.096, +.009] | −.009 [−.081, +.060] | −.001 [−.126, +.122] | +.006 [−.021, +.035] | −.017 [−.055, +.000] | +.020 [−.070, +.118] | −.020 [−.075, +.025] | −.017 [−.017, +.000] |
| power_z - power | +.016 [−.049, +.075] | +.012 [−.046, +.067] | −.057 [−.178, +.056] | −.024 [−.046, −.006] | −.037 [−.097, +.003] | +.092 [+.005, +.175] | −.013 [−.068, +.053] | not run |
| pearson_spectral_z - pearson_spectral | −.019 [−.075, +.029] | −.003 [−.066, +.061] | +.017 [−.084, +.114] | −.011 [−.034, +.010] | −.017 [−.060, +.000] | −.012 [−.087, +.055] | −.025 [−.078, +.027] | −.001 [−.001, +.001] |
| mean_spi_z - mean_spi | +.034 [−.037, +.122] | −.003 [−.083, +.084] | −.097 [−.198, −.005] | −.014 [−.048, +.020] | −.007 [−.068, +.050] | +.075 [−.008, +.168] | −.013 [−.070, +.050] | +.000 [+.000, +.000] |
| m_z - m | −.015 [−.074, +.053] | −.001 [−.074, +.075] | −.058 [−.140, +.019] | −.018 [−.055, +.016] | −.020 [−.057, +.013] | +.030 [−.033, +.090] | +.020 [−.040, +.078] | +.000 [+.000, +.000] |
| z - pearson | +.002 [−.120, +.118] | −.003 [−.117, +.118] | −.006 [−.190, +.175] | +.015 [−.068, +.105] | +.182 [+.088, +.295] | +.047 [−.067, +.158] | +.082 [+.005, +.163] | +.572 [+.206, +.957] |
| z - mean_spi | +.026 [−.054, +.113] | +.003 [−.087, +.101] | −.109 [−.216, +.001] | −.023 [−.067, +.019] | −.030 [−.105, +.042] | +.015 [−.102, +.123] | −.020 [−.080, +.045] | +.000 [+.000, +.000] |
| z - m | +.003 [−.070, +.084] | +.042 [−.043, +.131] | −.077 [−.187, +.034] | −.047 [−.096, −.001] | −.040 [−.098, +.007] | −.065 [−.178, +.037] | +.007 [−.073, +.080] | +.000 [+.000, +.000] |
| z - aligned | −.025 [−.098, +.043] | −.045 [−.130, +.044] | −.129 [−.281, +.022] | −.048 [−.107, +.006] | −.050 [−.113, +.010] | −.140 [−.242, −.037] | −.002 [−.090, +.087] | +.000 [+.000, +.000] |
| z - single_spi | −.091 [−.195, −.003] | +.001 [−.125, +.127] | −.255 [−.402, −.102] | −.015 [−.086, +.056] | +.040 [−.043, +.143] | −.028 [−.140, +.092] | +.068 [−.037, +.170] | +.016 [+.001, +.031] |
| z - tangent | +.053 [−.078, +.181] | +.015 [−.101, +.137] | −.070 [−.233, +.092] | +.008 [−.080, +.104] | +.105 [+.023, +.178] | +.100 [−.023, +.212] | +.060 [−.017, +.145] | +.051 [+.000, +.051] |
| z - catch22 | −.035 [−.116, +.039] | −.015 [−.132, +.119] | +.014 [−.182, +.219] | −.053 [−.110, −.003] | −.003 [−.077, +.072] | −.070 [−.210, +.057] | +.030 [−.048, +.117] | +.000 [+.000, +.000] |
| z - spectral | −.036 [−.124, +.044] | −.022 [−.111, +.064] | +.058 [−.104, +.224] | −.036 [−.088, +.010] | −.062 [−.130, +.000] | −.145 [−.257, −.020] | −.020 [−.075, +.030] | +.000 [+.000, +.000] |
| z_full - z | −.035 [−.084, +.005] | −.070 [−.148, +.000] | −.033 [−.132, +.057] | +.058 [+.018, +.104] | +.045 [+.000, +.090] | +.117 [+.042, +.185] | +.005 [−.045, +.050] | −.018 [−.018, +.000] |
| z_pc1 - z | −.100 [−.218, +.013] | −.069 [−.180, +.033] | −.271 [−.445, −.080] | −.388 [−.507, −.268] | −.412 [−.545, −.285] | −.185 [−.282, −.098] | −.307 [−.430, −.170] | −.260 [−.260, −.131] |

| Brier improvement [95%]; BA difference | CHB-MIT | Siena | HMC | GRABMyo | LANL | fMRI primary |
|---|---|---|---|---|---|---|
| pearson_z versus pearson | +.090 [+.061, +.137]; +.125 | +.016 [−.036, +.072]; −.012 | +.007 [−.018, +.031]; −.014 | +.020 [+.007, +.031]; +.096 | +.237 [+.224, +.250]; +.350 | −.022 [−.040, −.005]; −.111 |
| spectra_z versus spectra | +.097 [+.024, +.173]; +.125 | +.035 [−.068, +.119]; +.030 | +.017 [−.002, +.040]; +.007 | +.004 [+.001, +.006]; +.000 | +.038 [+.005, +.071]; +.000 | +.013 [−.024, +.054]; −.028 |
| univariate_z versus univariate | +.060 [+.031, +.089]; +.125 | +.026 [−.019, +.071]; +.095 | +.024 [−.007, +.064]; +.035 | +.002 [−.000, +.005]; −.007 | +.075 [+.020, +.130]; +.000 | not run |
| pearson_spectra_z versus pearson_spectra | +.016 [−.034, +.073]; +.000 | +.029 [−.057, +.101]; +.161 | +.046 [+.005, +.096]; +.056 | +.001 [−.001, +.004]; −.007 | +.039 [+.006, +.073]; +.000 | −.009 [−.036, +.014]; −.028 |
| mean_spi_z versus mean_spi | +.013 [−.023, +.056]; +.083 | −.008 [−.041, +.030]; −.071 | +.013 [−.013, +.038]; +.056 | +.011 [+.003, +.020]; +.074 | −.016 [−.036, +.005]; +.000 | +.023 [−.011, +.061]; +.000 |
| z versus pearson | +.103 [+.055, +.166]; +.153 | −.069 [−.155, +.019]; −.101 | −.005 [−.037, +.029]; −.021 | −.016 [−.036, +.004]; −.110 | +.241 [+.224, +.257]; +.350 | −.071 [−.138, −.004]; −.194 |
| z versus mean_spi | +.006 [−.049, +.071]; +.042 | −.060 [−.116, −.004]; −.107 | +.009 [−.025, +.041]; +.007 | +.004 [−.009, +.017]; +.022 | +.004 [+.003, +.005]; +.000 | −.005 [−.055, +.037]; −.056 |
| z versus spectra | +.073 [−.030, +.168]; +.167 | +.002 [−.103, +.097]; +.042 | −.018 [−.091, +.060]; −.049 | −.039 [−.052, −.025]; −.228 | +.042 [+.005, +.078]; +.000 | −.047 [−.111, +.022]; −.139 |
| z versus tangent | +.051 [−.013, +.121]; +.250 | +.022 [−.044, +.084]; +.036 | −.027 [−.062, +.009]; −.056 | −.030 [−.047, −.013]; −.140 | +.231 [+.206, +.255]; +.325 | −.094 [−.156, −.031]; −.222 |
| z versus catch22 | +.029 [−.016, +.073]; +.069 | +.033 [−.032, +.091]; +.054 | +.023 [−.012, +.057]; −.028 | −.013 [−.027, +.004]; −.103 | +.123 [+.091, +.155]; +.000 | −.067 [−.140, +.007]; −.167 |
| z_full versus z | +.005 [−.024, +.034]; −.042 | −.004 [−.052, +.042]; +.000 | +.009 [−.007, +.025]; +.049 | +.001 [−.004, +.006]; +.000 | −.001 [−.003, +.001]; +.000 | +.010 [−.010, +.032]; +.028 |
| z_pc1 versus z | −.119 [−.236, −.022]; −.167 | +.012 [−.055, +.084]; −.036 | −.032 [−.060, −.005]; −.069 | −.042 [−.053, −.031]; −.154 | −.008 [−.011, −.005]; +.000 | −.034 [−.085, +.018]; −.111 |

## Other real-world data with z but no side-by-side classifier panel

- **CHBMP IQ (Cuban Human Brain Mapping Project, resting EEG; final, pre-registered).** 215 prepared subjects (5,109 10-s windows, 19 channels, 100 Hz, r10), 135 with FSIQ. 0 of 31,415 finite z features pass max-|ρ| FWER (max |ρ| .25 versus .36); H1 phase-lag share ρ −.02; H2 core readouts all Holm p = 1; H3 ridge cross-validated R² < 0 for covariates, + spectra, + Pearson and + z. Post-hoc: z split-half reliability .97 (215/215 identified, within session only); age has 235 FWER features after IAF/slope/strength adjustment, 0 after five spectral PCs (p .06); the pre-specified LEMON replication failed (T1 concordance −.05, p .79; directed × undirected composite reversed, ρ −.23 with spectral PCs). IAAFT not run (no positive). Source: `research/representation/chbmp_iq/results.md` at `c01c112` (Vn; JSONs on Gadi scratch `chbmp-iq-261105/`). The dataset-candidates session is still active, but this write-up states its result as final.
- **NeuroTycho anaesthetic induction (unsupervised; 15 dates, four macaques, 2,239 windows, M16/T2000).** Held-animal AUROC averaged over dates (KTMD / propofol): mean |r| .96 / .99, mean-SPI PC1 1.00 / 1.00, centred z-PC1 .72 / .70, standardized z-PC1 .86 / .89. Centred z-PC1 carries 47% between-animal and 3% between-state variance. Source: `research/inference/locking/notes/cross-frequency-locking-261006.md` §Real-data attempt (Vn; outputs `results/order-parameter-inference/neurotycho-induction-261006/` not in the repository; pyspi version not checked here).
- **HCP MEG working memory.** Extraction only: 24 participants, 48 runs, 590 blocks at M32/T4000, K289 banks (z 41,616, m 6,647). No cohort classifier evaluation exists; family grouping, final confirmation assignment and the evaluation driver were never completed. A one-person scout (eight held blocks) gave held load AUROC z-PC1 = m-PC1 = .875, spectral PC1 = .5625, which supports nothing beyond feasibility. Source: `docs/context/spi-spi-representation-evaluation.md` §HCP (Vn).
- **Electronic Rössler hardware oscillators (ER and small-world wiring), CCEP ECoG (ds004080, 24 participants) and the Zenodo 1,053-series corpus** have z computed for unsupervised order-coordinate, calibration or atlas work, not for a representation comparison against baselines; see `docs/context/premise-and-character-axes.md` and `docs/context/zenodo-mts-corpus.md`. They are out of scope for these matrices.
- **UEA/Multiverse benchmark panel**: designed, not run. Protocol on branch `claude/mts-benchmark-screen-4cbb57` (`research/representation/mts_benchmark/protocol.md`, commit `e20a490`): 89-dataset screen, ten candidate tasks plus BasicMotions for positioning, awaiting user approval.

## Dataset inventory and pyspi versions (10 November)

Every real-world dataset on which z has been computed, with the pyspi computation that produced its bank. Banks of different versions are never extended or pooled. The pin moved to r13 (`5d61d9c`) on 6 November; **no real-world bank exists at r13**, and the 10 November blueprint requires one pinned version for the write-up, so every statistical cell here is provisional until that re-run. Raw-waveform comparators (InceptionTime, the Rocket family, tsfresh, catch22, encoders, tangent-space covariance, channel spectra) do not depend on the pyspi version.

| Dataset | Question | Units | M / T / fs | pyspi bank | Evaluation | Status |
|---|---|---|---|---|---|---|
| NeuroTycho anaesthesia | awake vs anaesthesia after animal/drug shift | 4 source + 2 target macaques | 16 / 2000 / 250 Hz | K289 `65317c9` (pilot); K283 r10 (`character-transfer-261031/neurotycho-primary`) | transfer; r10 also core sufficiency and three IAAFT variants | positive, core-explained; r10 transfer panel never scored on targets |
| NeuroTycho induction | unsupervised state coordinate | 4 macaques, 15 dates | 16 / 2000 | not checked | unsupervised only | z-PC1 separates animals, not states |
| fMRI rest/film | rest vs film | 17 + 9 people | 17 / 947 | r10 | transfer, frozen | mixed/negative |
| CHB-MIT → Siena | ictal vs seizure-free | 16 + 6 → 14 people | 16 / 2000 / 100 Hz | r10 | transfer; frozen confirmation | screen gain not replicated |
| HMC sleep | N2 vs REM | 27 + 12 nights | 8 / 3840 / 128 Hz | r10 | transfer | matched by per-channel features |
| GRABMyo | 17 gestures, day 1 → 2 | 24 + 8 people | 14 / 2048 / 2048 Hz | r10 | transfer | adverse |
| LANL three-storey | contact vs linear change | 9 + 4 configurations, one rig | 4 / 4096 / 320 Hz | r10 | transfer | saturated |
| AD/FTD (ds004504) | AD, FTD, control | 88 people | 19 / 2000 / 100 Hz | r10 | grouped CV; two IAAFT variants | z ≈ Pearson; linear-spectral |
| LEMON | old vs young | 132 people | 19 / 2000 / 100 Hz | r10 | grouped CV; two IAAFT variants | gain over Pearson is channel-spectral |
| Propofol sedation (Chennu 2016) | sedation level | 20 people × 4 levels | 19 / 1000 / 100 Hz | r10 | within-person CV; two IAAFT variants | z adds over P + core (+.133), not over spectra |
| CHBMP | full-scale IQ; age post hoc | 215 people (135 with FSIQ) | 19 / 1000 / 100 Hz | r10 (per protocol; not re-read) | pre-registered association | null; age signal spectral, not replicated in LEMON |
| HCP MEG working memory | 0-back vs 2-back | 24 people | 32 / 4000 / 250 Hz | K289, archived | none | extraction only; blocked on family grouping |
| CCEP ECoG (ds004080) | lag-profile character vs evoked latency | 24 people, 102 tables | 48–107 / 4096 / 128 Hz | r9 (`fd0de71`) | fixed-axis description | spectral/lag-profile continuum; N1 association gate failed |
| Electronic Rössler hardware (ER, small-world) | locking transition | 45 settings, two wirings | 28 / 5000 / 500 Hz | r7 and r9 | unsupervised, calibrated | lag-profile/spectral organisation calibrated to 58/116/174 Hz profiles |
| Zenodo corpus (1,053 series, synthetic and measured) | atlas neighbourhoods | 1,053 series | varied | see `docs/context/zenodo-mts-corpus.md` | unsupervised | atlas only |

Scouted, with no z computed: TUH seizure EEG (exploration notebook, `research/empirical/tuh_eeg/`), the EWSHM 2026 bridge benchmark (access request drafted, not sent; `research/empirical/bridge/`), and the UEA/Multiverse benchmark panel (protocol only). The repository has no `log/` directory; the research log that the memory note describes is not present, so `git log`, the context files and the per-study notes are the record.

## Surrogate (real − multivariate IAAFT) results by dataset

All from `research/representation/character_axes/findings.md` (Vn; surrogate banks on Gadi scratch `character-axes-261102/surr-*`). IAAFT reproduces covariance/precision almost exactly, Spearman closely (.97), ids and MI at .82, TE at .25, coherence/spectral GC at ≈ .5, and PSI/PLV/PPC not at all, so residuals are interpretable only for the time-domain covariance, precision, rank and ids family.

- **NeuroTycho (three variants, six animal–drug units):** timing, band and spectral-peak effects are linear-spectral. The top z pairs (robust precision versus ids or rank², real d −11.6 to −15.4) are only partly reproduced (surrogate d −3 to −4.6; residual d −1.3 to −2.2, 0/6 units, all variants): the strongest candidate for beyond-spectral character. Form te−gc residual −3.0 (robust); mediation residual is not robust across variants. The non-Gaussian MI level rise is an uncorrected-Kraskov artefact and is withdrawn. Robust versus classical direct-coupling divergence survives beyond surrogates in 6/6 units; slow-wave up-state network switching was tested and not supported.
- **Propofol (two variants, 1,000 iterations):** phase-lag, fixed-lag and band effects are linear-spectral. Residuals robust to both variants: linear–rank agreement −.83/−.82 (16/20 lower) and tail −.69/−.54. Construction validation says raw linear–rank pairs read marginal shape, but its real − IAAFT version reads joint bursts (inferred to apply here).
- **AD/FTD:** effects are linear-spectral. One residual survives both variants: tail level +.75 (p .009; not significant after Bonferroni over 26 readouts; ASR-cleaning transients are the obvious confound).
- **LEMON:** every age effect is linear-spectral. The one robust residual (form_information −.40/−.50) is driven by one unreliable pair (Gaussian GC versus kernel TE, mean z .08).
- **CHB-MIT and Siena (marginal-final variant only):** seizure effects mostly linear-spectral; small shared residuals phase-lag +.47/+.36, predictive +.43/+.50, tail −.53/−.68. Not checked against the spectrum-final variant, which removed such residuals in LEMON and propofol.
- **LANL (marginal-final only):** most contact effects reappear in surrogates (mediation d +10.8 real versus +5.8 surrogate); smaller residual in phase-lag level, spectral-peak level and predictive organisation.
- **fMRI, HMC, GRABMyo, CHBMP:** no surrogate run.

## Per-dataset notes

- **NeuroTycho anaesthesia (macaque ECoG).** Awake eyes-closed versus sustained anaesthesia; sources 11 KTMD dates, four animals; targets four propofol dates, two animals (Chibi, George), 32 balanced windows per date; target animal excluded from its fit; M16/T2000 bipolar, 250 Hz. Two banks: K289/`65317c9` (September pilot, Matrices 1–2) and K283/r10 (`character-transfer-261031/neurotycho-primary`, used for the character axes, sufficiency and surrogates; its full transfer panel was source-fitted but never evaluated on targets: `neurotycho-source-review.json` has `target_outcomes_evaluated: false`). Caveats: drug and calendar period are confounded; the target animals appeared in exploratory development; nearly all of z's errors are false positives on George's dates (perfect ranking, shifted threshold); conventional spectra rank perfectly, so interaction necessity is unestablished; channel spectra make the spectra-adjusted ANCOVA unidentifiable (4% state overlap).
- **Propofol EEG sedation (Chennu et al. 2016).** 20 people × four levels, ten 10-s epochs per level, 19 channels, 100 Hz; within-participant CV. The one cohort where z adds over Pearson + strength + core (+.133). Power and spectra alone reach AUROC .99. The frontal → parietal directed-asymmetry prediction failed (GC shifted opposite, d +.68).
- **LEMON (young versus old).** 132 of 203 people (41 aged 55–80, 91 aged 20–35; 70 excluded for a missing canonical channel), six 20-s windows. Pearson + axes + levels over Pearson is +.043 [+.011, +.077], but power + axes over power is −.004. The core's gain comes from location frequency, which tracks the channels' 4–20 Hz spectral centroid (r .75–.84) and loses its age effect after adjusting for it.
- **AD/FTD EEG (ds004504).** 88 people, eight 20-s windows each; unit counts per contrast (65/52/59) are inferred from window counts ÷ 8. AD versus FTD is near chance for every method. Location frequency is missing in 97% of windows, so the core is weakened here.
- **fMRI rest/film.** 26 people (17 development, nine confirmation), 17 bilateral networks, M17/T947; the primary role is nine new people × four later-run scans, frozen fits, threshold .5. Primary AUROC is the mean within-participant AUROC; pooled AUROC is in Matrix 2. Acquisition dates and motion are absent, so run sets are not verified visits. A bounded mixed/negative case: z and z-fusions lose to Pearson, the best single SPI (`cov_GraphicalLassoCV`), aligned MPIs and MiniRocket.
- **CHB-MIT screen and Siena confirmation.** Ictal versus seizure-free 20-s epochs, M16/T2000 at 100 Hz. Screen: sources 16 people/78 records, screen six people/26 records (chb21 duplicate excluded; chb16 fails the 20-s rule). Siena: 14 adult patients, 27 seizure/control pairs, 15 frozen methods. The screen gain was a six-person overestimate; seizure versus control is dominated by amplitude, power and coupling strength.
- **HMC sleep (N2 versus REM).** M8/T3840 (brain, eye, muscle, cardiac channels), sources 27 nights/324 records, screen 12 nights/144. Nonlinear univariate (BA .889) and the aligned encoder (Brier .092) win; z+univariate gains .024 Brier with an interval crossing zero. Weak HMC lag-organisation sign claim (axes cohort bound).
- **GRABMyo gestures.** 17 classes including rest, 14 bipolar forearm/wrist channels, M14/T2048, day 1 → day 2; sources 24 people/408 records, screen eight people/136. Adverse for z (BA .243 against InceptionTime .522), plausibly through loss of channel identity and activation magnitude (inferred).
- **LANL three-storey structure.** Nonlinear bumper contact versus linear mass/stiffness changes, M4/T4096, 320 Hz; nine source and four screen configurations of one apparatus (not independent structures; mirror identity and calibrated units uncertified). Saturated: many methods reach BA 1.0; z has the lowest Brier. The raw learners' entire deficit is state17 (contact gap .10 mm plus 1.2 kg first-floor mass). Validity alone reaches BA .875.
- **HCP MEG working memory**, **CHBMP IQ**, **NeuroTycho induction**: see the section on other real-world data above.

## Comparator gap matrix (10 November)

Status 11 November: every cell of the agreed ladder below is filled (results in [Comparator fill](#comparator-fill-11-november)) except InceptionTime on the three cohorts (not submitted; reasons under Run status). tsfresh and the core ladder on new datasets were dropped. The ○ marks are kept as the 10 November record.

Target: every dataset scored, under its own frozen protocol, on the full ladder. ✓ = exists (source in the matrices above); ○ = missing, fillable on the existing bank or raw windows with CPU only; ● = missing, needs GPU; — = not applicable; ✗ = blocked. Nothing in this table needs new pyspi extraction. NeuroTycho is completed in its r10 column only: the K289 pilot stays frozen history, and its raw-learner predictions (same 480 windows) are re-scored under the r10 aggregation rather than refitted.

| Comparator | NT r10 | fMRI | CHB-MIT | Siena (frozen CHB fits) | HMC | GRABMyo | LANL | AD/FTD | LEMON | Propofol |
|---|---|---|---|---|---|---|---|---|---|---|
| z | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| r (Pearson edges) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| r + z | ○ (P+z ✓) | ✓ | ○ | ○ | ○ | ○ | ○ | ○ (P+z ✓) | ○ (P+z ✓) | ○ (P+z ✓) |
| mean-SPI (K edge means) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| m (23 descriptors per SPI) | ○ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ○ | ○ | ○ |
| m + z | ○ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ○ | ○ | ○ |
| full aligned MPIs | ○ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ○ | ○ | ○ |
| best single SPI (source-selected) | ○ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ○ | ○ | ○ |
| tangent-space covariance | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| per-channel log-PSD | ✓ | ✓ | ✓ (with amplitude) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| per-channel catch22 | ✓ (K289 run) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| core: P, P+z, P+core, P+core+z | ✓ | ○ | ○ | ○ | ○ | ○ | ○ | ✓ | ✓ | ✓ |
| InceptionTime | ✓ (K289 run) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ● | ● | ● |
| ROCKET | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| MiniRocket | ✓ (K289 run) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| MultiRocket and MultiRocket-Hydra | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| QUANT | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| tsfresh (Efficient) | ✓ (K289 run) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| aligned raw encoder | ✓ (K289 run, channel-exchangeable) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — (none exists) | — | — |
| pooled AUROC beside BA/Brier | ✓ | ✓ | ○ (no fit) | ✓ | ○ | ○ (one-vs-rest) | ○ | ✓ | ✓ | ✓ |

Rules for the fills. Each keeps the column's protocol: transfer columns fit and select on source rows only (three grouped folds, the column's PCA caps and ridge grid, Brier selection) and score held-out units once at threshold 0.5; cohort columns use the same participant-grouped five-fold CV × 5, inner-fold selection, PCA cap 32 and unit-mean probabilities. The best single SPI is chosen inside each training fold. Siena fills apply CHB-source fits unchanged and are post hoc, because Siena outcomes are known. No fill may change an existing fit, and nothing is selected on held-out data.

Caveats that bound what the fills can show:

- Tangent-space covariance on standardized windows (fMRI and the cohort banks) is tangent-space correlation: the diagonal carries no channel power. On the discovery banks' physical-unit windows it does carry power. Both are reported as what they are.
- The core's band readouts split at fs/4, so their physical meaning differs by dataset (25 Hz for 100 Hz EEG, 512 Hz for GRABMyo, 80 Hz for LANL). On new datasets the core ladder is a fixed readout, not a validated one.
- GRABMyo is 17-class, so its ladder is scored by Brier and BA, with one-vs-rest AUROC as a supplement.
- Two target animals (NeuroTycho), four configurations (LANL) and six people (CHB-MIT) cannot rank the added comparators; intervals stay as wide as the existing ones.

Cost estimates (inferred from this repository's recorded timings; none measured yet, so one job per class is measured before the rest are released):

| Fill | Work | Estimate |
|---|---|---|
| Statistical blocks on transfer banks (r+z, mean-SPI, tangent, catch22, core ladder; NT r10 also m, m+z, aligned, single SPI) | ridge/logistic on ≤ 1,246 rows | 7 jobs, 8 cores × ≤ 1 h; ≤ 120 SU |
| Statistical blocks on cohort banks (r+z, mean-SPI, m, m+z, aligned, nested single-SPI search, tangent, catch22) | 25 outer folds × 7 contrasts; the single-SPI search is about 85,000 small logistic fits per contrast | 3 jobs, 24 cores × ≤ 3 h; ≤ 450 SU |
| ROCKET, MultiRocket, MultiRocket-Hydra, QUANT (aeon 1.5, separate environment with CPU torch) | transfer: three seeds per dataset; cohorts: refit per outer fold | ≈ 100 core-hours; ≤ 300 SU |
| tsfresh Efficient and catch22 | ≈ 60,000 channel series, extracted once, label-free | ≈ 40 core-hours; ≤ 100 SU |
| InceptionTime on the three cohorts | five-member ensembles on one five-fold split per contrast (175 fits), not the full 5 × 5 repeats | ≈ 30 GPU-hours on `gpuvolta`; ≈ 1,100 SU; least certain figure |
| Pooled AUROC for the discovery screen; NT r10 target scoring; NT raw-learner re-scoring | no fitting | negligible |

Total ≈ 2,000 SU, dominated by InceptionTime. Scratch inode use is small (tens of JSON and feature files per dataset); `lquota` and `nci_account` are checked before submission.

Waits for the user (needs new pyspi extraction or an outside input): spectrum-final IAAFT banks for CHB-MIT, Siena and LANL; any surrogate bank for fMRI, HMC or GRABMyo; the reserved confirmation sets (HMC unused nights, GRABMyo day 3, LANL reserved configurations); HCP evaluation (family grouping, archived K289 bank); the r13 re-extraction of every bank that the blueprint's single-pin rule implies; and the UEA/Multiverse panel.

Ladder agreed with the user on 10 November, which supersedes the row list above where they differ: r, per-channel spectra, z (source-selected PCA, plus first PC only and full rank), r + z, spectra + z, r + spectra (+ z), mean-SPI (+ z), the SPI marginals m (+ z), full aligned MPIs, best single SPI, tangent-space covariance, per-channel catch22, and the libraries ROCKET, MiniRocket, MultiRocket-Hydra, QUANT and InceptionTime. tsfresh and the core ladder on new datasets were dropped as low-value. pyspi versions are not unified for this scoping pass; each column keeps its version label. Drivers: [`comparator_fill/`](../../comparator_fill/) (`cohort.py`, `transfer.py`, `fmri.py`, `library.py`, `cohort_inception.py`); tests assert that they reproduce the original protocol code on synthetic banks.

Run status (10 November, 19:40 AEDT): submitted from the frozen snapshot `operations/sources/comparator-fill-ee1ddb7` to run root `/scratch/ql44/we2614/mts-spi-study/representation/empirical/comparator-fill-261110/` (`stat/`, `library/`, `logs/`). Statistical jobs 183588599–183588607 (one per bank; CHB-MIT carries the Siena application); aeon wheels 183588598 (copyq, aeon 1.5.0 with CPU torch); library jobs 183588608–183588616, held on the wheels. The library driver was smoke-tested locally on a synthetic bank with all four methods. InceptionTime on the three EEG cohorts is not submitted: it already exists on the seven transfer columns, and the Rocket family plus QUANT cover the raw-learner rung there (dropped on the assistant's recommendation after the user questioned its value; reversible, `cohort_inception.py` and `jobs/inception.pbs` are ready). Inputs: the r10 banks listed in the inventory; Scratch inodes were 168,394 of 202,000 and gdata 83 of 100 GiB at submission. Completed 11 November: AD/FTD and LEMON first failed under `ee1ddb7` when an SPI's block could not be fitted during the best-single-SPI search; `7b4c2f9` makes such an SPI ineligible, and AD/FTD, LEMON and propofol were rerun from that snapshot (AD/FTD 2 h, 6 GB). Both snapshots were then removed (recorded in `operations/sources/ARCHIVED.md`).
