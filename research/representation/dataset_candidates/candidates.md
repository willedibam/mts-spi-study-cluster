# Candidate datasets for dependence-character questions (4 November 2026)

Scope: a literature and metadata screen only. No data were downloaded beyond public metadata files, and no pyspi or cluster jobs were run. Part A covers neural cohorts for low vs high IQ and young vs old. Part B covers non-neural systems. Prior datasets (fMRI rest/film, CHB-MIT/Siena, HMC, GRABMyo, LANL, NeuroTycho, HCP working memory, LEMON age, propofol, AD/FTD) are excluded unless there is a new reason to use them. Benchmark archives (UEA, Multiverse, MONSTER) belong to the parallel `mts-benchmark-screen` session.

Evidence labels: **[V]** checked this session against the data descriptor, official page, metadata file or abstract (not a full-text replication); **[I]** my inference; **[U]** unverified, from memory or secondary text only.

## What any candidate has to survive

These constraints come from the repo's own results ([premise context](../../../docs/context/premise-and-character-axes.md), [axis findings](../character_axes/findings.md)) and decide most ratings below.

- **Channel spectra explain the EEG gains.** In every cohort tested so far, z's gain over Pearson is redundant with Pearson + strength + per-channel spectra. In LEMON age, the location-frequency sub-axis lost its age effect after adjusting for channel spectral centroid. Any new EEG age or IQ cohort will probably fail the same way unless its question is about how lag or band organisation varies *across pairs* at matched channel spectra [I].
- **z reads heterogeneity across pairs.** A uniform change, such as every connection slowing equally, barely moves z, although level does leak into organisation. Questions must predict that some pairs change differently from others.
- **Validated axes:** band, mediation and direction are core; lag is a sub-axis. Form (linear vs nonlinear) fails at cohort-like coupling, and there is no validated burst/tail axis. So "more nonlinear" and "more tail-driven" questions cannot be answered with the current axes, whatever the dataset.
- **Known spectral confounds.**
  - Age: alpha peak frequency is slower in older adults even after aperiodic correction ([Donoghue-lab preprint](https://www.biorxiv.org/content/10.1101/2021.03.02.433594v1.full.pdf), [V] preprint). The aperiodic exponent flattens with age (Voytek et al. 2015, J Neurosci 35:13257–13265, [V] via secondary summaries).
  - IQ: the link between individual alpha frequency and IQ is contested. [Grandy et al. 2013](https://doi.org/10.1016/j.neuroimage.2013.04.059) report IAF tracks *g*, similarly in young and old ([V] abstract). [Ociepka, Kałamała & Chuderski 2022](https://ideas.repec.org/a/eee/intell/v92y2022ics0160289622000253.html) report high-IAF brains are faster but not smarter ([V] abstract).
  - Inference: the dominant spectral confound is weaker for IQ than for age, so IQ is the better test of the premise [I].
- **fMRI is close to Gaussian.** [Hlinka et al. 2011](https://asep.lib.cas.cz/arl-cav/en/detail-cav_un_epca.1-0356655-Functional-connectivity-in-restingstate-fMRI-Is-linear-correlation-sufficient/) estimate non-Gaussian dependence at about 5% of total MI in resting fMRI ([V] abstract). Nonlinearity questions on fMRI are close to dead on arrival. Lag/spectral questions on fMRI would be linear-spectral findings.
- **Feasibility thresholds:** M ≥ 8 (preferably ≥ 12) simultaneous sensors of one system, T ≥ ~250, and N ≥ ~200 or enough independent groups, with labels.

## Part A — neural cohorts (ranked)

IQ and age are separate tracks, and no dataset needs both.

- **IQ track:** CHBMP, then LEMON (IQ within the young group), then HBN. HCP-YA and AOMIC are fMRI alternatives.
- **Age track:** Dortmund (within-person, open), then Cam-CAN, then TDBRAIN.

The table ranks by overall fit.

| Rank | Dataset | Access | Modality | N | Age | Cognition | M | T / fs | Rest/task | Fit |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CHBMP (Cuban Human Brain Mapping) | Synapse login | EEG | 282 | 18–68 | WAIS-III | 64 or 120 | ≥30 min, 200 Hz | rest EC/EO/HV | Medium-high (IQ) |
| 2 | Cam-CAN | written proposal | MEG (+fMRI) | 647 rest MEG; Cattell 660 | 18–88 | Cattell fluid | 306 sensors; source parcels | 8:40 rest, 1 kHz | rest, task, passive | Medium-high (age × IQ) |
| 3 | LEMON, IQ re-analysis | open, already on Gadi | EEG | 227 (≈203 usable EEG) | 20–35 / 59–77 | LPS-2 subtest 3 (fluid), WST (vocabulary) | 62 | 16 × 60 s, 2500 Hz | rest EC/EO | Medium (cheapest test, underpowered) |
| 4 | Dortmund Vital Study (ds005385) | open, CC0 | EEG | 608; 208 at 5-yr follow-up | 20–70 | none released | 64 | 3 min EC + 3 min EO, pre/post tasks, 1 kHz | rest | Medium (age, longitudinal) |
| 5 | Healthy Brain Network EEG | EEG open; phenotype needs DUA | EEG | >3,000 | 5–21 | WISC-V (subset) [U for FSIQ field] | 128 | 500 Hz (some 250) | rest + tasks | Medium |
| 6 | HCP Young Adult | registration; restricted tier for family structure | fMRI; MEG subset | ~1,200 fMRI; 95 MEG | 22–35 | PMAT24 | parcels; 248 MEG [U] | fMRI 4 × 1200 vols at TR 0.72 s [U]; MEG 3 × 6 min | rest + tasks | Medium (fMRI lag); low (MEG N) |
| 7 | AOMIC PIOP1/PIOP2/ID1000 | OpenNeuro, open | fMRI | 216 / 226 / 928 | 19–26 (ID1000) | Raven APM (PIOP); IST (ID1000) | parcels | PIOP1 480 vols at TR 0.75 s; PIOP2 240 at 2 s; ID1000 movie only | rest (PIOP), movie (ID1000) | Low-medium |
| 8 | TDBRAIN | ORCID login + DUA | EEG | 1,274 patients | 5–89 | none documented [U] | 26 | 2 min EO + 2 min EC, 500 Hz | rest | Low-medium |
| 9 | NKI-Rockland enhanced | DUA | fMRI | large lifespan [U] | 6–85 [U] | WASI-II [U] | parcels | 10 min at TR 0.645 s | rest | Low-medium |
| 10 | MIPDB (CMI) | open [U] | EEG | 126 | 6–44 | abbreviated battery; IQ test [U] | 128 | [U] | rest + tasks | Low |
| — | HarMNqEEG | shared derived tensors only | EEG cross-spectra | 1,564 | lifespan | — | — | no raw series | — | Excluded |
| — | ABCD | NDA restricted | fMRI | ~11,000 [U] | 9–10 at baseline [U] | NIH Toolbox [U] | parcels | [U] | rest + tasks | Low |

### A1. CHBMP — IQ in adults with full-scale WAIS-III

Facts [V] ([Valdes-Sosa et al. 2021, Sci Data 8:45](https://doi.org/10.1038/s41597-021-00829-7)):
- Cohort: 282 adults aged 18–68 (mean ≈ 32), 195 male and 87 female, sampled from the population. WAIS-III, MMSE and go/no-go were collected; participants with WAIS < 70 were excluded.
- EEG: MEDICID-5, 64- or 120-channel montages, 200 Hz, 0.5–50 Hz amplifier band. At least 30 min per person, covering eyes closed, eyes open, hyperventilation and recovery.
- Access: hosted on Synapse (doi 10.7303/syn22324937; login required) and LORIS; the descriptor is CC-BY 4.0.

Question: at matched individual alpha frequency and channel spectra, do higher-IQ adults show a different cross-pair lag organisation? A specific, testable direction exists. [Thatcher, North & Biver 2005](https://doi.org/10.1016/j.clinph.2005.04.026) report that phase delay is the EEG measure most strongly related to IQ, with shorter frontal and longer posterior delays in high IQ ([V] abstract). That pattern is heterogeneity across pairs, which is what z can read.

- **Axes:** lag sub-axis and band organisation. Do not pose a form question.
- **Why existing representations fall short:** a coherence or Pearson adjacency gives which edges differ, not whether the difference is in delay. Topology summaries lose the pair identity. Phase-delay maps answer the question directly, so they are the specialist reference; z offers unsupervised screening across many statistics.
- **Confounds:**
  - IAF and aperiodic slope.
  - Sensor-space volume conduction and reference, which Thatcher's sensor measures are also exposed to [I].
  - Range restriction (WAIS ≥ 70), the male-heavy sample, and two montages: use the common 10–20 subset or interpolate.
  - Age (18–68) correlates with WAIS subtests.
- **Feasibility:** M ≥ 19 common channels, T up to 12,000 per 60 s, N = 282, one site. Gamma band is unavailable (50 Hz bandwidth).
- **Rating:** medium-high. This is the only open adult EEG cohort found with full-scale IQ and N > 250, and it has a published, pair-specific lag prior.

### A2. Cam-CAN — age and fluid intelligence, MEG

Facts:
- Archive counts: 647 resting MEG, and Cattell fluid intelligence for 660 ([V] via the [archive](https://camcan-archive.mrc-cbu.cam.ac.uk/) as reported by search; the overlap is unknown).
- Ages 18–88. MEG has 306 channels at 1 kHz with 8:40 of rest; access requires a specific academic proposal, and a supervisor must apply for students ([V] in [premise context](../../../docs/context/premise-and-character-axes.md)).
- Resting-MEG transient-network dynamics relate to both age and fluid intelligence, N ≈ 595 ([Tibon et al. 2021](https://www.mrc-cbu.cam.ac.uk/wp-content/uploads/www/sites/3/2021/06/TibonEtAl_NBA_21_MEG_HMM_Cog_Age.pdf), [V] abstract).

Question (age track): does ageing reorganise lag or band structure across source pairs beyond what slower alpha and a flatter 1/f explain? Cattell scores allow a separate IQ analysis, but the cohort does not need it.

- **Axes:** lag sub-axis, band, mediation (direct vs indirect in source space).
- **Why existing representations fall short:** as for A1. Source reconstruction reduces, but does not remove, leakage-driven zero-lag structure [I].
- **Confounds:**
  - Age-spectral change: the strongest instance of the LEMON failure mode.
  - Cattell declines with age, so any IQ analysis must be within age strata [I].
  - Head size and SNR change with age [U]. Source leakage.
- **Feasibility:** M 38–68 parcels by choice, T ≥ 520,000 raw, N ≈ 600. Pair sampling is needed at sensor level.
- **Rating:** medium-high for age. It is the best-powered lifespan MEG cohort and source space is less exposed than sensor EEG, but access takes a proposal and the age contrast risks repeating the LEMON result.

### A3. LEMON — IQ instead of age, within the existing pipeline

Facts [V] ([Babayan et al. 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6371893)):
- Cohort: 227 participants, 153 young (20–35) and 74 old (59–77).
- EEG: 62 channels at 2500 Hz, sixteen 60-s blocks (8 eyes closed, 8 eyes open). The descriptor's preprocessing retained 203.
- Cognition: LPS-2 subtest 3 measures fluid reasoning and the WST measures crystallised vocabulary, both available for all 227.
- License terms not re-checked [U].

Question: the IQ analogue of A1, within the young group (N ≈ 150) to remove the age confound.

- **New reason to revisit:** LEMON was used only for age, and its data are already prepared on Gadi.
- **Confounds:** IAF. A single subtest has lower reliability than full-scale IQ [I]. Restricted range in a student-heavy sample [I].
- **Feasibility:** good on M and T. N is small for brain–IQ effects, which are expected to be small [I].
- **Rating:** medium. It is the cheapest first test of the IQ premise: a negative here is weak evidence, a positive needs CHBMP or Cam-CAN to replicate.

### A4. Dortmund Vital Study — within-person ageing

Facts [V] ([Getzmann et al. 2024, Sci Data 11:988](https://doi.org/10.1038/s41597-024-03797-w); [OpenNeuro ds005385](https://github.com/OpenNeuroDatasets/ds005385) `participants.json` and `dataset_description.json`):
- Cohort: 608 adults aged 20–70; 208 re-recorded about 5 years later.
- EEG: 64 channels at 1 kHz; 3 min eyes closed and 3 min eyes open, before and after a 2-hour cognitive battery.
- License CC0. The released participant table holds only age, sex and handedness, with no cognitive scores.

Question: within the same person over 5 years, do lag or band organisations change beyond what that person's own spectral change predicts? The within-person design removes between-person anatomy and montage variation. That is the new reason to use it after LEMON.

- **Secondary contrast:** pre vs post task battery (fatigue) gives a within-session state contrast.
- **Confounds:** spectral ageing (as above), electrode placement differences between sessions [I].
- **Rating:** medium. Open and large, but it carries age only, and the LEMON prior predicts channel-spectral redundancy.

### A5–A10 in brief

- **HBN** ([release notes](https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/release/release_notes11_0.html), [V]):
  - EEG on OpenNeuro/NEMAR, 128 channels.
  - WISC-V requires a LORIS DUA, and not every EEG participant has phenotypes yet.
  - FSIQ is age-normed, but EEG spectra change strongly with development. Psychopathology is enriched.
  - Strong on N, weak on interpretability. Medium.
- **HCP-YA:**
  - The fMRI cohort is large with PMAT24 [V]. 95 subjects have MEG, with 3 × 6-min rest [V].
  - Valid cross-validation needs the restricted family table. The repo's earlier HCP use stalled on exactly this.
  - fMRI lag structure is a defensible linear-spectral target ([Mitra et al. 2015](https://doi.org/10.7554/eLife.10781)). Medium.
- **AOMIC** ([Snoek et al. 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC7979787/), [V]):
  - Narrow ages, so IQ is not tied to age.
  - Resting fMRI exists only in PIOP1 (480 volumes at TR 0.75 s) and PIOP2 (240 volumes at TR 2 s), with Raven's APM. ID1000 is movie-only.
  - The Hlinka result rules out form questions. Low-medium.
- **TDBRAIN** ([van Dijk et al. 2022](https://doi.org/10.1038/s41597-022-01409-z), [V]): 1,274 psychiatric patients aged 5–89, 26 channels at 500 Hz, 2 + 2 min; DUA via ORCID. Diagnosis and medication confound age. Low-medium.
- **NKI-RS:** DUA, multiband 645 ms rest [V]. The cognitive instruments and the presence of EEG in the enhanced release are unverified. Low-medium.
- **MIPDB** ([Langer et al. 2017](https://doi.org/10.1038/sdata.2017.40)): N = 126 [V], which is too small. Low.
- **HarMNqEEG** ([Li et al. 2022](https://doi.org/10.1016/j.neuroimage.2022.119190)): only cross-spectral tensors are shared [V]. These fix only linear-Gaussian content, and z needs raw series. Excluded.

## Part B — non-neural systems (ranked)

Bottom line: **no non-neural candidate is convincing as a discovery application.** None combines a natural character question, a z-compatible geometry (many simultaneous sensors, cross-pair heterogeneity, many independent recordings), and a realistic chance of beating channel-marginal baselines. Two are useful in other roles: traffic as a non-neural ground-truth validation of the direction/lag axes, and cardiorespiratory coupling as the strongest non-neural *question*, which z's geometry fits poorly.

| Rank | System / data | Character question | Axis | Access | M / T / N | Fit |
|---|---|---|---|---|---|---|
| 1 | Freeway loop detectors (Caltrans PeMS; UTD19) | Does propagation reverse direction from free flow to congestion? | direction, lag | PeMS free account [U]; UTD19 sign-up, CC BY-NC-SA | M 20–40 detectors per corridor; T 288/day at 5 min; N thousands of days | Medium-high as validation, low novelty |
| 2 | Cardiorespiratory coupling (NSRR PSG; VitalDB) | Does coupling become less stable or change form with age, sleep stage or anaesthesia? | lag, band (form not validated) | NSRR DUA; VitalDB open on PhysioNet | M 6–13 heterogeneous; T large; N thousands | Medium question, low-medium z fit |
| 3 | Pigeon flocks (Nagy et al. 2010) | Is there a hierarchy of pairwise leader–follower delays, and does it differ between flight types? | direction, lag | open (cite on use) | M ≤ 10, 0.2 s GPS; N = 15 flights | Low-medium (too few recordings) |
| 4 | Crypto/equity returns (Binance 1-min; daily equities) | Is crash co-movement driven by joint tails beyond linear correlation? | joint tails (no validated axis) | Binance public archive [U: no login] | M ≥ 12 assets; T 1,440/day; N thousands of days | Low-medium |
| 5 | Climate teleconnections (ERA5/NCEP) | Is ENSO-related dependence nonlinear or lagged? | form, lag | open | M by choice; few independent ENSO events | Low |
| 6 | Structural health (Z24, KW51, LANL) | Does damage introduce nonlinear inter-sensor coupling? | form | Z24 on request [U]; LANL already used | M 4–16 | Low |
| — | Rotating machinery, wind-farm SCADA, PMU grids, plankton ecology, 12-lead ECG | see below | — | — | — | Low / excluded |

### B1. Freeway traffic — propagation direction reverses with congestion

Physics [V] via [Treiber & Kesting calibration work](https://arxiv.org/pdf/1008.1639) and secondary summaries:
- In free flow, perturbations travel downstream at roughly vehicle speed. In congestion they travel upstream at about 14–20 km/h across US, German and UK sites.
- The congested wave speed is estimated by maximising summed cross-correlation over detector-pair lags. The established estimator is therefore a lagged Pearson specialist.

Question: does an unsupervised descriptor recover the reversal of directed lag between upstream and downstream detector pairs, at matched mean speed (for example, synchronized flow vs wide moving jams)? Lag scales with inter-detector distance, so the reversal is pair-heterogeneous, which z can read.

- **Why existing representations fall short:** a symmetric correlation adjacency cannot see direction. A directed lag map answers it directly, so this case validates z against known physics rather than producing new science.
- **Confounds:** congestion moves every detector's mean and spectrum. Classifying congestion is trivial from speeds. The test has to be direction at matched marginals. Ramps and incidents add noise.
- **Feasibility:** PeMS 5-min data gives T = 288 per day, which is borderline. Raw 30-s data availability is unverified [U]. [UTD19](https://doi.org/10.1038/s41598-019-51539-5) is urban data at 3–5 min [V], where signal cycles dominate; it is a poor substitute for freeway corridors [I].
- **Rating:** medium-high as a non-neural ground-truth check of the core direction axis; low as a new discovery.

### B2. Cardiorespiratory and physiological-network coupling

Prior work:
- **Age:** across 189 subjects aged 16–90, direct and indirect respiratory modulation of heart rate both decline with age, and cardiorespiratory coupling becomes less stable and more time-variable ([Iatsenko et al. 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC4042892), [V] abstract).
- **Coupling functions:** the form of the coupling function is the established character concept here ([Stankovski et al. 2017, Rev Mod Phys 89:045001](https://doi.org/10.1103/RevModPhys.89.045001), [V]).
- **Sleep:** network-physiology links reorganise across sleep stages under time-delay stability ([Bashan et al. 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3518900), [V] abstract). Multiple coupling forms coexist across brain–heart–respiration ([Bartsch & Ivanov 2014](https://cris.iucc.ac.il/en/publications/coexisting-forms-of-coupling-and-phase-transitions-in-physiologic/), [V] abstract).
- **Anaesthesia:** under spontaneous-breathing anaesthesia, inspiration onset tends to lock to the preceding heartbeat (cardioventilatory coupling; Galletly & Larsen, [V] abstract).

Data:
- [VitalDB](https://physionet.org/content/vitaldb/1.0.0/) ([Lee et al. 2022](https://doi.org/10.1038/s41597-022-01411-5), [V]): 6,388 surgical cases with waveform tracks (ECG, PPG, arterial pressure, CO₂, BIS EEG) and drug infusion records; open access. The exact license is unverified [U].
- NSRR (SHHS 9,736 PSGs, MESA and others) needs a data access and use agreement [V].

Question: does coupling between cardiac, respiratory and vascular rhythms become more lagged or less band-locked with age, or under anaesthesia at matched heart and breathing rates?

- **Why existing representations fall short:** a correlation network of 6 signals mostly encodes rates and amplitudes. Topology on 6–13 nodes says little.
- **Why z fits poorly:**
  - Only about 4–7 non-neural channels. Several are redundant measures of one oscillator (thoracic, abdominal and airflow all respiration).
  - Pairs fall into a few fixed modality blocks, so cross-pair heterogeneity mostly reflects modality, not character [I].
  - Mechanical ventilation in VitalDB imposes respiration externally.
  - Dedicated coupling-function inference (Stefanovska's group) answers the question directly with phase models.
- **Confounds:**
  - Heart and breathing rate change with age, stage and drug: per-channel spectra again.
  - Ventilator settings.
  - Adding EEG, EOG and EMG brings M to ~12, but then the system is no longer non-neural.
- **Rating:** medium as a question, low-medium for z. It is the most physiologically natural non-neural character question, but per-pair specialist methods suit it better than a cross-pair descriptor.

### B3. Collective motion — leadership delays

[Nagy et al. 2010](https://pubmed.ncbi.nlm.nih.gov/20376149/) derive a pecking order from pairwise directional-correlation delays in flocks of up to 10 pigeons. [Trajectories](https://hal.elte.hu/pigeonflocks/index.html) are free to use with citation: 11 free flights and 4 homing flights at 0.2 s ([V] readme summary).

- **Question:** does the pairwise delay hierarchy differ between free and homing flight? This is exactly a heterogeneous directed-lag structure.
- **Feasibility:** M ≤ 10 and N = 15 recordings make any group contrast underpowered.
- **Rating:** low-medium. Worth a qualitative illustration only, unless larger tracked-flock datasets exist; none were verified here [U].

### B4. Financial and crypto co-movement — joint tails

Prior work:
- International equity correlation rises in bear markets but not bull markets. Multivariate normality is rejected for the negative tail only ([Longin & Solnik 2001](https://doi.org/10.1111/0022-1082.00340), [V] abstract).
- Correlation rises mechanically with volatility, and the adjusted coefficient shows little crisis "contagion" ([Forbes & Rigobon 2002](https://doi.org/10.1111/0022-1082.00494), [V] abstract). That is the financial version of our channel-marginal confound.

Data: Binance publishes 1-min spot klines as monthly and daily zip files ([binance-public-data](https://github.com/binance/binance-public-data/), [V] format; no-login access [U]). Days serve as recordings, labelled by drawdown regime.

- **Why z fits poorly:** the question needs a tail/burst axis, which is not validated, and the form axis failed. Copula tail-dependence estimators are the established specialist.
- **Confounds:** volatility clustering, fat-tailed marginals, common market factor.
- **Rating:** low-medium.

### B5–B6 and exclusions

- **Climate:**
  - [Hlinka et al. 2014](https://arxiv.org/abs/1211.6688) ([V] preprint) find genuine nonlinear contribution in reanalysis temperature networks "almost negligible". Apparent non-Gaussianity drops from about 16% of pairs to about 6–7% after variance normalisation and detrending, close to the 5% expected by chance.
  - A residual tropical-Pacific excess is only hypothesised to be ENSO.
  - Independent ENSO events are few. Low.
- **Structural health:**
  - On Z24, temperature shifts natural frequencies about as much as damage, or more ([V] secondary).
  - Damage and crack nonlinearity appear as per-channel spectral changes (harmonics, frequency shifts) that a channel-spectral baseline would capture [I].
  - The repo's LANL set has M = 4. Low.
- **Excluded or low:**
  - Rotating-machinery benchmarks: M ≤ 8, and faults are spectral lines.
  - Wind-farm SCADA: 10-min resolution cannot resolve wake advection delays of tens of seconds to minutes [I; specific datasets U].
  - Open multi-site PMU data: scarce [U].
  - Plankton mesocosms: N = 1.
  - 12-lead ECG: leads are near-instantaneous linear projections of one cardiac source, so a character question is ill-posed. It is a useful negative example of "simultaneous sensors" that are not separate subsystems [I].

## Access steps the user would need to take

- **CHBMP:** create a Synapse account and accept the dataset terms for syn22324937; LORIS mirror at chbmp-open.loris.ca.
- **Cam-CAN:** submit a project proposal through the [Cam-CAN data portal](https://opendata.mrc-cbu.cam.ac.uk/projects/camcan/), with the supervisor applying.
- **LEMON:** none; already in use. Confirm the LPS/WST files are in the copy on Gadi.
- **Dortmund:** none (CC0, OpenNeuro ds005385).
- **HBN:** sign the HBN DUA and request LORIS phenotype access for WISC-V.
- **HCP-YA:** ConnectomeDB open-access terms, plus a restricted-data application for family structure.
- **TDBRAIN:** ORCID login and the Brainclinics DUA.
- **NKI-RS:** the INDI DUA.
- **NSRR:** a data access and use agreement per dataset.
- **PeMS:** a free account.
- **UTD19:** sign-up and terms.
- **No action:** VitalDB and the pigeon data.

## Recommended order (assistant's assessment, not a decision)

1. **IQ track:** LEMON IQ within the young group as a zero-access pilot. Then CHBMP, testing the Thatcher-type frontal/posterior lag organisation at matched IAF and channel spectra.
2. **Age track:** Dortmund within-person 5-year change, which is open and the one design LEMON could not provide. Then Cam-CAN in source space if a proposal is worthwhile.
3. **Non-neural:** traffic direction reversal as a ground-truth validation, presented honestly as validation, not discovery.

Pre-register the controls now standard in the repo for every step:
- Pearson + strength + channel-spectra baseline.
- IAAFT surrogates.
- The axis nuisance bound.
