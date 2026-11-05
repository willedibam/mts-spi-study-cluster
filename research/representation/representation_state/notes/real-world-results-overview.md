# Real-world results overview: z against other representations

Compiled 6 November 2026 from existing outputs only (no pyspi, no fitting, no Gadi jobs). Every number below was read from the source named for its table or line. Status labels: **V** = read from the output file in this compilation; **Vn** = taken from a findings note that records its own independent verification, but whose output file is not in the repository and was not re-read; **I** = inferred here; **not run** = no such comparison exists. Protocols are never pooled: each table keeps its own split, metric and threshold rule, and rows are not ranked across tables.

## Summary

What holds:

- z is a usable, transferable recording representation on several real tasks: NeuroTycho anaesthesia (BA .930 z-PCA / .945 z-PLS after an animal/drug/period shift), the CHB-MIT screen (BA .958) and LANL (BA 1.0, lowest Brier .0044). Its ranking (AUROC) usually transfers better than its fixed 0.5 threshold.
- z beats the Pearson edge vector (r) on NeuroTycho (current bank), CHB-MIT screen, LANL and propofol sedation, roughly ties on AD/FTD, LEMON and HMC, and loses on fMRI, Siena and GRABMyo.
- Adding z to Pearson + strength gives an AUROC gain whose interval excludes zero in two cases: NeuroTycho (+.352 [+.067, +.667], two target animals) and propofol moderate versus baseline (+.182 [+.087, +.295], 20 people). In NeuroTycho the validated character core accounts for it; in propofol it does not (z still adds +.133 [+.060, +.213] over Pearson + strength + core).

What does not hold:

- z never adds over Pearson + strength + channel spectra (the S block) in any cohort tested (AD/FTD, LEMON, propofol, NeuroTycho); every interval includes or lies below zero. Channel power or spectra alone match or beat z on every grouped-CV cohort. Whatever z adds over Pearson is redundant, for classification, with per-channel spectra; this is redundancy in AUROC, not proof that z reads spectra.
- No dataset shows z beating the strongest comparator. Winners: NeuroTycho enriched raw encoder and InceptionTime ensemble; fMRI MiniRocket, aligned MPIs and Pearson; Siena univariate and aligned MPIs (AUROC); HMC nonlinear univariate and aligned encoder; GRABMyo InceptionTime; LANL ties among many methods at BA 1.0.
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

## Other real-world data with z but no side-by-side classifier panel

- **CHBMP IQ (Cuban Human Brain Mapping Project, resting EEG; final, pre-registered).** 215 prepared subjects (5,109 10-s windows, 19 channels, 100 Hz, r10), 135 with FSIQ. 0 of 31,415 finite z features pass max-|ρ| FWER (max |ρ| .25 versus .36); H1 phase-lag share ρ −.02; H2 core readouts all Holm p = 1; H3 ridge cross-validated R² < 0 for covariates, + spectra, + Pearson and + z. Post-hoc: z split-half reliability .97 (215/215 identified, within session only); age has 235 FWER features after IAF/slope/strength adjustment, 0 after five spectral PCs (p .06); the pre-specified LEMON replication failed (T1 concordance −.05, p .79; directed × undirected composite reversed, ρ −.23 with spectral PCs). IAAFT not run (no positive). Source: `research/representation/chbmp_iq/results.md` at `c01c112` (Vn; JSONs on Gadi scratch `chbmp-iq-261105/`). The dataset-candidates session is still active, but this write-up states its result as final.
- **NeuroTycho anaesthetic induction (unsupervised; 15 dates, four macaques, 2,239 windows, M16/T2000).** Held-animal AUROC averaged over dates (KTMD / propofol): mean |r| .96 / .99, mean-SPI PC1 1.00 / 1.00, centred z-PC1 .72 / .70, standardized z-PC1 .86 / .89. Centred z-PC1 carries 47% between-animal and 3% between-state variance. Source: `research/inference/locking/notes/cross-frequency-locking-261006.md` §Real-data attempt (Vn; outputs `results/order-parameter-inference/neurotycho-induction-261006/` not in the repository; pyspi version not checked here).
- **HCP MEG working memory.** Extraction only: 24 participants, 48 runs, 590 blocks at M32/T4000, K289 banks (z 41,616, m 6,647). No cohort classifier evaluation exists; family grouping, final confirmation assignment and the evaluation driver were never completed. A one-person scout (eight held blocks) gave held load AUROC z-PC1 = m-PC1 = .875, spectral PC1 = .5625, which supports nothing beyond feasibility. Source: `docs/context/spi-spi-representation-evaluation.md` §HCP (Vn).
- **Electronic Rössler hardware oscillators (ER and small-world wiring), CCEP ECoG (ds004080, 24 participants) and the Zenodo 1,053-series corpus** have z computed for unsupervised order-coordinate, calibration or atlas work, not for a representation comparison against baselines; see `docs/context/premise-and-character-axes.md` and `docs/context/zenodo-mts-corpus.md`. They are out of scope for these matrices.
- **UEA/Multiverse benchmark panel**: designed, not run. Protocol on branch `claude/mts-benchmark-screen-4cbb57` (`research/representation/mts_benchmark/protocol.md`, commit `e20a490`): 89-dataset screen, ten candidate tasks plus BasicMotions for positioning, awaiting user approval.

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

## Gaps that would be cheap to fill (not run)

Ordered by information per unit cost (an assessment, not a plan). None needs new pyspi; all banks already exist.

1. **Pearson + z** on CHB-MIT, HMC, GRABMyo, LANL (and source-frozen → Siena). Both blocks are in every discovery bank. This is the direct r+z test the claim needs, and only fMRI has it (where it was adverse). A Siena application would be post hoc and must be labelled so.
2. **Pooled AUROC for the discovery screen** (CHB, HMC, GRAB, LANL) from the existing prediction files. No fitting; it separates ranking from threshold failure, as was done for Siena and fMRI.
3. **NeuroTycho r10 target evaluation** of the already-fitted source models (`neurotycho-source-review.json`, `target_outcomes_evaluated: false`). This makes the current-version NeuroTycho panel comparable with the K289 pilot, including Pearson, aligned MPIs and the best single SPI, none of which the pilot ran.
4. **m, m+z, full aligned MPIs and best single SPI on the character-axes cohorts** (AD/FTD, LEMON, propofol). The banks hold `m` (6,509) and `aligned` (96,786); only Pearson/z/axes/spectra/power and MiniRocket were run.
5. **tsfresh and catch22** on fMRI, the discovery datasets and the EEG cohorts. These are CPU-cheap and have only been run on NeuroTycho K289.
6. **Spectrum-final IAAFT** for CHB-MIT, Siena and LANL, whose residual claims rest on the marginal-final variant only (surrogate banks need recomputation, so this one does require pyspi on surrogates; cheap at these sizes, but not free).
7. **Supervised K-vector mean-SPI** (edge means only) on any real task. It is the matched-invariance strength control the benchmark protocol names, and it is absent from every supervised real-data panel. The 23-descriptor m is a superset.
8. **InceptionTime on the character-axes cohorts**, which findings.md lists as missing. It needs GPU fits, so it is cheap only relative to the rest.

Not cheap: HCP classifier evaluation (blocked on family grouping and driver recovery), ROCKET/MultiRocket and learned pooling beyond NeuroTycho (new comparators), and the benchmark panel (awaiting approval).
