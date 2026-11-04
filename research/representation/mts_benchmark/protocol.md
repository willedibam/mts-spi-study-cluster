# MTS benchmark panel for SPI–SPI z: screen, pre-registered predictions, protocol

Status 4 November 2026: design only. Nothing here has been run; no pyspi, no Gadi jobs. Awaiting user approval of the panel, tier and step-0 checks. The full metadata screen is [`screen.csv`](screen.csv) (89 datasets, built by [`build_screen.py`](build_screen.py)); cost estimates come from [`estimate_cost.py`](estimate_cost.py). Context: [representation evaluation](../../../docs/context/spi-spi-representation-evaluation.md), [character axes](../../../docs/context/premise-and-character-axes.md).

## What the result is

The deliverable is the predicted win/loss pattern and whether it holds, not a leaderboard. A win, near-tie or narrow loss of z against established methods is valuable when z's high-importance coordinates then pass the interpretation gates below and say something about the character of dependence that the other representations do not expose. Failures are retained and reported in the same table as the existing panel (fMRI rest/film negative, CHB-MIT→Siena not replicated, HMC matched by univariate, GRABMyo adverse, LANL saturated, NeuroTycho positive).

Two priors from this repository shape every prediction. First, z is invariant to a common relabelling of channels (verified property of the contract), so it discards topography, edge incidence and per-SPI offsets. Second, wherever tested, z's gain over Pearson was redundant with per-channel spectra (character-axis sufficiency, 3 November); propofol's +0.13 AUROC of z over the validated core is the one unexplained residual.

Material consequence (inferred from the screen): classification archives are structurally adverse to z. Most surviving tasks are within-montage, have ample labels, and code the label in waveform shape or in which channels change. Those are exactly the properties an identity-aware method uses and z throws away. The place z has a capacity the ladder lacks is comparison across channel sets (montages, species, subsets), which is why the montage-free transfer extension is the highest-value item and not an afterthought.

## Answers to the framing questions

- **Is Pearson a justified baseline?** Yes, as the field's default functional-connectivity estimator, and the one with a favourable bias–variance tradeoff at short T. But the Pearson matrix is identity-aware, so z versus Pearson confounds two differences: more statistics, and loss of topography. Two baselines separate them. Identity-free Pearson (quantiles of the Pearson edge distribution) is in z's invariance class. Tangent-space covariance (pyriemann; Barachant et al. 2012, Dadi et al. 2019, Sabbagh et al. 2019) is the strongest identity-aware dependence baseline, and it also carries channel power on the diagonal.
- **Is mean-SPI unfairly hard?** No. It is the matched control: the same catalogue and the same invariance class, carrying strength only. z versus mean-SPI is the cleanest test of "character beyond strength", so it is one of the two primary contrasts. The genuinely hard superset is the full aligned MPIs: every SPI matrix flattened in a fixed ordered-edge order, which contains z.
- **Libraries.** tslearn is dropped: its classifiers (DTW k-NN, shapelets, GAK-SVM) are redundant with aeon's 1-NN DTW and dominated in published benchmarks. HIVE-COTE 2 comes from published Multiverse results only. The aligned neural encoder is optional, given its weak prior transfer.

| Invariance class | Feature blocks |
| --- | --- |
| Identity-free (unchanged by a common channel relabelling) | z; mean-SPI (K edge means); Pearson edge quantiles; channel-pooled univariate (log-PSD and catch22 averaged over channels) |
| Identity-aware | Pearson matrix; tangent-space covariance; per-channel univariate (log-PSD and catch22); best single SPI; full aligned MPIs; MiniRocket, MR-Hydra, QUANT, InceptionTime, 1-NN DTW, HC2 |

**Mirror argument (exact under its premise).** Let P be the left–right mirror permutation of a symmetric montage. If X given class L has the same distribution as PX given class R, then z(X) given L has the same distribution as z(X) given R, because z(PX) = z(X). Directed SPIs map to directed SPIs on the permuted edges, so this still holds. Every identity-free block then has Bayes balanced accuracy 0.5. Left-versus-right motor contrasts approximately satisfy the premise, and hands-versus-feet does not. Any departure from chance measures non-mirror asymmetry (handedness, montage, artefact). This turns ImaginedOpenCloseFist into a designed negative control on real data.

## Step 1: screen

Eligibility: hard floor M≥3, and M<8 is allowed but flagged low-edge (user, 4 November). T≥~250 is a guide (EmoPain at 200 is admitted and flagged), and T<100 is excluded. Channels must be sensors of one coupled system, not frequency bins, MFCC, satellite bands or derived coefficients. N≥~150. A plausible coupling rationale is required, and the dataset must be neither saturated (best published BA ≥0.97) nor at floor (best < chance + 0.05).

Sources:
- UEA30 and Multiverse (133 datasets, aeon-toolkit/multiverse `adc21c1`; metadata from arXiv 2603.20352).
- MONSTER: its multivariate sets are HAR at T=100, satellite bands, or EEG sets already in Multiverse, so it adds nothing eligible.
- Dependence-native external tasks.

"Best published BA" is the best of 12 Multiverse classifiers at resample 0 on the official split.

| Dataset | M × T | N train/test | Channels | Classes | Best published BA | Role |
| --- | --- | --- | --- | --- | --- | --- |
| ImaginedOpenCloseFist | 64 × 640 | 2835/1890 | EEG 64 (inferred EEGMMIDB, 160 Hz, 4 s) | imagined left vs right fist | .771 HC2 | invariance control |
| ImaginedFeetHands | 64 × 640 | 2835/1935 | same montage and subjects | imagined fists vs feet | .709 HC2 | paired contrast |
| ShortIntervalTask | 63 × 2750 | 1360/1420 | EEG 63 (Singh et al. 2023) | Parkinson's vs control | .659 MR-Hydra | disease, weak SOTA |
| AutomotiveRoadTrials | 8 × 300 | 153/77 | 8 OBD signals, one vehicle | fault vs none | .835 | z-favoured candidate |
| EmoPain | 30 × 200 | 968/355 | 26 joint angles + 4 sEMG | no/low/high pain | .746 1-NN DTW | coordination candidate |
| SelfRegulationSCP1 | 6 × 896 | 268/293 | EEG, one person | SCP positivity/negativity | .946 MR-Hydra | Cliff 2023 positioning |
| HandMovementDirection | 10 × 400 | 160/74 | MEG | 4 directions | .549 QUANT | predicted loss |
| ABIDE-PCP (MSDL) | 39 × ~78–316 TR (inferred; fixed in step 0) | ~871, leave-site-out | fMRI parcels | autism vs control | ~.67–.70 acc, tangent (Abraham 2017; Dadi 2019) | dependence-native |
| TUAB v3.0.1 | 19 × 2000 (20 s, 100 Hz) | 2717/276 | scalp 10-20, average reference | normal vs abnormal | ~.85–.86 acc (Gemein 2020) | clinical, large |
| NeuroTycho → Chennu propofol | 16 / 19 × 1000 | 480 windows / 20 people | macaque ECoG → human EEG | awake vs anaesthesia/sedation | — | montage-free extension |
| BasicMotions | 6 × 100 | 40/40 | phone IMU | 4 activities | 1.0 | positioning only, outside the test family |

Instances and independent units: power and the scope of a claim depend on the independent units, not the record count. Record counts are verified from the archive tables. Unit counts marked "?" are unverified and are recovered in step 0.

| Dataset | Records (full tier) | Independent units | Scope of a win |
| --- | ---: | --- | --- |
| ImaginedOpenCloseFist | 4,725 trials | 109 people if EEGMMIDB? Split may be by trial | Within- or cross-person decoding, depending on the split |
| ImaginedFeetHands | 4,770 trials | same people as above? | as above |
| ShortIntervalTask | 2,780 trials | Parkinson's and control participants, count ? | Diagnosis, only if the split is subject-disjoint |
| AutomotiveRoadTrials | 230 segments (77 test) | vehicles/trips ? | One fleet or vehicle; underpowered for Δ below about .1 |
| EmoPain | 1,323 windows | participants ? (tens) | Cross-person if the split is grouped |
| SelfRegulationSCP1 | 561 trials | 1 person | One person's sessions |
| HandMovementDirection | 234 trials (74 test) | 1–2 people? | One or two people; underpowered |
| ABIDE-PCP | ~871 scans | ~871 people across ~17–20 sites (inferred) | Cross-site population inference |
| TUAB | 2,993 recordings, one window each | patients ? (fewer than recordings) | Cross-patient clinical |
| NeuroTycho → propofol | 480 source windows → 400 target windows | 4 macaques/11 dates → 20 people | Cross-species, cross-montage |
| BasicMotions | 80 | few people ? | Positioning only |

Notable exclusions (reasons in `screen.csv`):
- MotorImagery: floor. All 12 methods are ≤.59 across the session shift, so no ranking is readable, and the native 1 kHz run would cost ~7.4k SU.
- PEMS-SF: saturated at 1.0, and the full M=963 table is outside the sizing table.
- FaceDetection and FingerMovements: T of 62 and 50.
- Heartbeat, DuckDuckGeese, PhonemeSpectra and JapaneseVowels: derived channels.
- Alzheimers: N=88, and the same source as the in-repo AD/FTD cohort.
- MindReading: M=204 at T=200.
- MatchingPennies: a redundant left/right contrast.
- The speech-imagery sets: M=128 with near-chance SOTA.
- Optional, unpriced: OpenCloseFist and FeetHands (executed replications), LongIntervalTask, LowCost, STEW, CrowdSourced, LenDB (M=3 seismic polarisation), UCDHE-Rowing-MC, SCP2, MOABB BNCI2014_001.

## Step 2: pre-registered predictions

A comparison is scored as follows:
- **W**: the 95% paired-bootstrap interval of the AUROC difference excludes 0 in z's favour.
- **L**: the interval excludes 0 against z.
- **T**: the interval contains 0. A "near-tie" additionally needs |Δ| < .03.

"Shape" means the best of MiniRocket, MR-Hydra and QUANT run under this protocol; HC2 is a published reference. These predictions are fixed before any data are touched, apart from the step-0 conditional for AutomotiveRoadTrials stated in its row.

| Dataset | z vs Pearson | z vs per-channel univariate | z vs shape | z vs mean-SPI | P+z vs P | Reason / carrying axis | Falsifier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ImaginedOpenCloseFist | L | L | L | T | T | Mirror argument: z, mean-SPI, Pearson quantiles and pooled univariate all at chance (BA .45–.56) | z AUROC lower bound > .56 |
| ImaginedFeetHands | L | L | L | T or W | T | Not a relabelling. mu/beta ERD moves band organisation across edges, but this is channel-spectral, so it is absorbed by per-channel log-PSD | z at chance, or z+univariate > univariate |
| ShortIntervalTask | T | L | T (all weak) | T | T | Parkinson's EEG differences are mainly spectral slowing (literature-informed; specific effect sizes unverified) | z > univariate |
| AutomotiveRoadTrials | W | W | T | W | W | A fault plausibly breaks functional relations between physically coupled signals (form/lag) while marginals change little. Conditional: if step 0 shows the fault is a single-signal level/offset, the univariate prediction becomes L, recorded before compute. Test n=77, so only large effects are visible | z ≤ Pearson |
| EmoPain | T or W | T | T | W | T | Guarding and co-contraction couple EMG with joint angles (mediation/lag). Ranges of motion (univariate) are also reduced; low confidence | z < mean-SPI |
| SelfRegulationSCP1 | T | L | L | T | T | The label is a per-channel slow-potential level. A common drift raises strength, which Pearson and mean-SPI see; z does not | z > univariate |
| HandMovementDirection | T | L | L | T | T | Direction is coded in movement-related waveform and topography; z near chance (BA ≤ .35) | z BA > .45 |
| ABIDE-PCP | L | T | n/a | T | T (\|Δ\| < .02) | Autism effects are region-specific (topography is discarded); short T favours Pearson's bias–variance; prior fMRI result was negative | P+z > tangent |
| TUAB | W | L | L | W | W | Abnormality is slowing and asymmetric power, so z's gain over Pearson is channel-spectral; univariate+P+z ≈ univariate+P | z ≥ univariate |

Pattern prediction:
- z beats Pearson only where channel spectra move (TUAB, perhaps ImaginedFeetHands), and per-channel univariate features then absorb the gain.
- z competes with univariate features only in the two coupled-system cases (AutomotiveRoadTrials, EmoPain).
- z is forced to chance on left/right.
- Expected wins of z over per-channel univariate: 1–2 of 9.
- Complementarity: weak (H2) wins are expected on TUAB and AutomotiveRoadTrials, and possibly on ImaginedFeetHands and EmoPain. Strong (H3) wins are expected only on AutomotiveRoadTrials, and possibly on EmoPain.

If the pattern holds, it is a calibrated statement of where cross-statistic correspondence carries label information. If it fails in z's favour on a dataset, that is where the gates are spent.

## Step 3: protocol

### Step 0: data-blind checks before any pyspi

1. **EEGMMIDB.** Confirm provenance (Schalk et al.). Confirm the 64-channel montage is mirror-symmetric and record P. Determine whether the archive split is by trial or by subject, and recover subject IDs from the source. Use identical subjects in the two paired datasets.
2. **ShortIntervalTask.** Find the native sampling rate. Recover subject IDs, and require the official split to be subject-disjoint; otherwise regroup and record it. Fix the 32-channel subset as the symmetric 10-20/10-10 positions, listed before compute.
3. **AutomotiveRoadTrials.** Read Theissler (2017, Knowledge-Based Systems) for the fault definition and fix the conditional prediction.
4. **TUAB.** Confirm the version and that the official eval split is patient-disjoint. Map to the 19 canonical 10-20 channels (watch the TCP/AR montage gotcha) and choose the window rule: one 20 s window starting at 60 s, fixed before compute.
5. **ABIDE.** Fix the nilearn PCP pipeline (cpac, filt-noglobal), the QC-passed subject list and the site list, and fix the leave-site-out folds.
6. **Pyspi pin.** Check upstream pyspi main. The whole panel runs at one pin in a new `base_output_dir`. If the computation version moves from 3.0.0.r10, the extension must recompute NeuroTycho and propofol together rather than mix banks.
7. **Baseline environment.** Build a separate environment for the aeon baselines, since aeon caps numpy below what pyspi needs (pyproject note).

### Computation

- Run `benchmarked_p90` (K=283, z dimension 39,903) on every full ordered edge set; no pair sampling at M≤64. The known rare-entry failure (assessment.md: one exceptional dyad leaves hundreds of z coordinates undefined) therefore does not arise.
- Channel subsetting (ShortIntervalTask) is a fixed pre-registered montage choice with no invariance guarantee. Its effect on z is not assumed negligible, and the all-63 version is an optional sensitivity (~11.7k SU).
- Fixed preprocessing per dataset:
  - archive datasets: as distributed, after per-channel standardisation;
  - ShortIntervalTask: resample to 250 Hz;
  - TUAB: 0.5–45 Hz band-pass, 100 Hz, average reference;
  - ABIDE: PCP derivatives, MSDL atlas time series.
- Low-M validity rule, fixed before labels are used and checked on training records only: retain z for a dataset if ≥90% of coordinates are finite across training records and the median number of distinct edge values per SPI is ≥5. Otherwise report z as invalid for that dataset. A discard is decided on this rule only, never on scores.
- Before each production farm, measure one recording per size class (HPC.md), then release the rest in one homogeneous job per size class.

### Feature ladder and readout

Feature blocks:
- per-channel univariate: catch22 and Welch log-PSD in fixed bins (mandatory);
- channel-pooled univariate;
- Pearson matrix;
- Pearson edge quantiles;
- tangent-space covariance (OAS shrinkage, Riemannian mean on training data);
- best single SPI, selected by inner CV on the training split only;
- mean-SPI;
- full aligned MPIs;
- z.

Combinations: P+z, univariate+P+z, P+core, P+core+z. The core comes from `mtsspi.character_axes.CORE`.

Readout and fitting:
- One readout for every block: class-balanced multinomial L2 logistic regression.
- Standardisation and PCA are fitted inside training folds only, with caps {8, 32, 50} and five fixed C values, selected by inner CV grouped by subject/site/patient where IDs exist.
- Fusion uses separate-block PCA followed by concatenation, not joint PCA, which can displace baseline directions (fMRI sensitivity, 31 October).
- Coordinates non-finite in any training record are masked. Test non-finites are imputed with training medians and counted.

### Library comparators and splits

- Official splits throughout, with the test set touched once per method after all selection is frozen.
- Comparators: MiniRocket, MultiRocket+Hydra and QUANT (aeon, default configurations), InceptionTime (five-member ensemble, existing driver in `learned_baselines/`), 1-NN DTW (dependent warping, full window), and catch22/PSD (as above).
- tsfresh is optional (cost at M64); HC2 is published only. On the subsampled tier HC2 is a reference, not a paired comparator.

### Metrics and inference

- Metrics: balanced accuracy at argmax, macro one-vs-rest AUROC and multiclass Brier score.
- Paired bootstrap with 10,000 resamples at the independent unit (subject, site, patient or person). Where only records are available, the bootstrap is flagged anti-conservative.
- Primary contrasts:
  - H1, z versus mean-SPI (character beyond identity-free strength);
  - H2, P+z versus P. This is the weak complementarity win: z adds to the field default even when z alone is at or below Pearson.
  - H3, univariate+P+z versus univariate+P, plus the control P+z versus P+mean-SPI. This is the strong complementarity win: the increment is neither channel-spectral nor multi-statistic strength. Prior evidence predicts H2 can pass while H3 fails, because z's gain over Pearson was channel-spectral in every cohort tested. A result is reported as weak or strong accordingly.
- Holm correction is applied across the nine datasets within each of H1, H2 and H3.
- For datasets with at most 1,500 records (AutomotiveRoadTrials, EmoPain, SCP1, HandMovementDirection, BasicMotions), the readout is also evaluated on 30 stratified train/test resamples at the official sizes. Resamples are grouped where unit IDs exist, resample 0 is the official split, and differences are paired across methods. Pyspi is computed once per record, so resampling only refits readouts. HC2 published resamples are used where available. Everything else is secondary, with unadjusted intervals and labelled as such. Each W/T/L prediction is scored, and the predicted-versus-observed table is reported.
- No grid expansion, architecture change or channel/T change to rescue a loss. Deviations are recorded with their reason before test scores are seen.

### Interpretation gates for any z-feature claim

A z coordinate or axis is described as character only if it passes all four gates. Otherwise it is reported as predictive and uninterpreted.

1. **Per-channel adjustment.** The class effect survives regressing out per-channel log-PSD and catch22 (`character_axes/spectra_adjusted.py`).
2. **Real − IAAFT decomposition.** The effect on real data minus the effect on the repository's two multivariate IAAFT variants (`character_axes/surrogate_pairs.py`) remains, which separates linear-spectral from non-Gaussian dependence. Pyspi on surrogates is budgeted only for datasets that pass a primary test.
3. **Core sufficiency.** P+core versus P+core+z (`character_axes/sufficiency.py`). If the core absorbs the increment, interpret it through the validated axes (band, mediation; lag as a sub-axis). If not, it is an unexplained residual, as in propofol.
4. **Stability and estimator controls.** Top coordinates must stay stable across three training resamples. Check KSG against KSG-DCE, and apply the reliability-attenuation check from the axes review before naming nonlinearity or lag.

### Extension: montage-free transfer (NeuroTycho → human propofol)

**Design.**
- Train awake versus anaesthesia on NeuroTycho sources (all source animals, source-only selection, fixed threshold).
- Test on Chennu et al. (2016) baseline versus moderate sedation in 20 people. Report ranking (AUROC) and threshold transfer separately. The reverse direction is secondary.
- Only identity-free blocks apply: z, mean-SPI, Pearson quantiles, channel-pooled log-PSD and catch22. A distinctive z claim requires beating channel-pooled spectra, not only showing transfer.
- Harmonise the primary run by re-cutting NeuroTycho to 0.5–45 Hz, 100 Hz and 10 s (T=1000, M=16 bipolar) to match the propofol bank (M=19, 100 Hz, 10 s), at the same pin (~135 SU). The existing 250 Hz/T2000 banks are a sensitivity.

**Confounds, none of which z's invariance removes.**
- Referencing (bipolar ECoG versus A1–A2/average scalp).
- Volume conduction: scalp mixing inflates zero-lag dependence and shifts coordinates that pair instantaneous with lagged SPIs.
- Species, drug and depth: macaque loss of responsiveness versus human moderate sedation, much of it responsive.
- M16 versus M19: subsampling, channel mixing and duration have no invariance guarantee.
- Twenty targets.

**Pre-registered sensitivities.** Propofol re-referenced to a nearest-neighbour bipolar chain, and a lag-only z subset that excludes instantaneous SPIs.

**Prediction.** Ranking transfer AUROC > .5 for z and for pooled spectra, with z ≈ pooled spectra; threshold transfer fails.

A cheaper second instance, if approved: eyes-open versus eyes-closed across three montages (EEGMMIDB baseline runs, LEMON, CrowdSourced).

### Cost and storage

Pyspi estimates per recording use the HPC.md sizing table × 2 to calibrate to measured r10 runs (expect ±50%; `estimate_cost.py` prints the calibration).

| Item | Full official splits | Subsampled (EEGMMIDB 1,600 each, ShortIntervalTask 1,000) |
| --- | ---: | ---: |
| ImaginedOpenCloseFist + ImaginedFeetHands (1.31 SU/rec) | 12.5k SU | 4.2k SU |
| ShortIntervalTask, 32 channels at 250 Hz (1.28) | 3.6k | 1.3k |
| TUAB (0.73) | 2.2k | 2.2k |
| ABIDE, EmoPain, AutomotiveRoadTrials, SCP1, HandMovementDirection, BasicMotions | 0.5k | 0.5k |
| NeuroTycho harmonised re-cut | 0.1k | 0.1k |
| Pyspi subtotal | 18.9k | 8.3k |
| Baselines: CPU libraries (<0.5k) plus InceptionTime ensembles on GPU (~1–2k, unverified) | ~2k | ~1.5k |
| IAAFT reserve for gate 2 (only datasets that pass) | ~3k | ~2k |
| Total | ~24k | ~12k |

Storage is the binding constraint, not SU. Live quota on 4 November: SU 32.82k available for 2026 Q4; Scratch 136,841/202,000 inodes and 398 GiB of 1 TiB used; gdata 16,647/70,000 inodes but 83.2 of 100 GiB used.

- **Inodes.** At about 3 inodes per recording (directory, `spi_mpis.npz`, `meta.json`; inferred), the full tier's ~19k recordings plus TUAB/ABIDE inputs approach the Scratch inode headroom. Pack each dataset's outputs after audit, as done on 2 November.
- **Bytes.** Raw outputs at M64 are about 9 MB per recording (float64), so they stay on Scratch. Aligned-MPI banks (283 × 4,032 values, about 4.6 MB per recording in float32, about 43 GB for the full EEGMMIDB pair) also stay on Scratch and are compressed inside folds from there. Only small banks go to gdata: z at about 0.16 MB per recording, plus mean-SPI, quantiles and univariate blocks.
- Run `check_gadi_storage_layout.py` and `lquota` before each farm.

**Recommendation.** Use the full tier. The user has stated SU are plentiful, and full official splits make the published HC2/MR-Hydra/QUANT numbers directly comparable on the dataset (ImaginedFeetHands) where a near-SOTA claim is most plausible. The subsampled tier saves ~12k SU at the cost of that comparability and some power.

## Positioning (to verify before writing it up)

- **Cliff et al. (2023, Nature Computational Science)** classified BasicMotions and SelfRegulationSCP1 from SPI features; this protocol reports z on both. The exact numbers and feature construction in that paper are unverified here.
- **Liu et al. (2025, Nature Methods)** benchmarked pyspi statistics one at a time on HCP. The "best single SPI selected on source" rung is that strategy transplanted to each task. Their finding that precision-family statistics rank well is recalled, not re-verified.
- **Field references:**
  - Multiverse (arXiv 2603.20352): its EEG leaderboard's best mean accuracy is .60 over 26 datasets, verified from the repository leaderboard.
  - MONSTER (arXiv 2502.15122); Ruiz et al. (2021) multivariate bake-off; Dempster et al. QUANT (2024) and Hydra (2023).
  - Abraham et al. (2017) and Dadi et al. (2019) for ABIDE; Gemein et al. (2020) for TUAB; Barachant et al. (2012) and Congedo et al. (2017) for Riemannian BCI.
  - Benchmark numbers quoted from these sources are approximate and marked as such in the tables.
