# Real-world applications of z: scoping for a full launch (10 November 2026)

Status: proposal. Nothing here has been downloaded, extracted or run, and nothing launches without user approval. It draws on, and does not repeat, the [results overview](real-world-results-overview.md), the [dataset candidates screen](../../dataset_candidates/candidates.md), the [character-axes findings](../../character_axes/findings.md) and the benchmark protocol on branch `claude/mts-benchmark-screen-4cbb57`. Labels: **[V]** checked this session (abstract, catalogue page or repository file, not a full-text replication); **[R]** recorded in this repository and not re-checked; **[I]** inferred; **[U]** unverified; **[S]** speculation.

## Where the evidence stands

The evidence so far does not support a Nature Computational Science-level empirical claim, in either form the project set. No dataset shows z at or above its strongest comparator, and in every cohort where a per-channel spectral control was run (AD/FTD, LEMON, propofol, NeuroTycho) z adds nothing over Pearson + strength + channel spectra [R]. The one frozen independent confirmation (CHB-MIT → Siena) failed, and CHBMP IQ is a pre-registered null [R]. What stands is narrower: z is a reliable, individual and transferable descriptor (split-half .97; NeuroTycho transfer AUROC 1.0) [R]; in NeuroTycho the validated core explains z's gain over Pearson [R]; and two feature-level effects go beyond multivariate IAAFT surrogates: the decoupling of direct (robust partial r²) from pairwise (rank, ids) dependence patterns under anaesthesia, in 6 of 6 animal–drug units, and propofol's lower linear–rank agreement in 16 of 20 people [R].

Three kinds of result could change that, and the shortlist is built on them.

1. **A task that only an identity-free representation can attempt.** z is unchanged by a common relabelling of channels, so it compares recordings with different channel sets (montages), species or implants. Pearson edges, tangent-space covariance, aligned MPIs and every raw learner need one shared channel set. Against the other identity-free blocks (mean-SPI, quantiles of the Pearson edge distribution, channel-pooled spectra) the comparison is fair and has not been run on real data.
2. **A contrast at matched strength and matched channel spectra, by design.** A controlled physical system gives this directly; in biology it needs within-unit designs and surrogates.
3. **A feature-level statement about character that replicates in an independent dataset.** The direct-versus-pairwise decoupling is the only candidate that has passed surrogates in more than one unit; it has one species and one weak human echo (moderate sedation, d −0.46 beyond spectra) [R].

## Rules every launch keeps

- **Per-channel controls first.** Channel spectra (absolute and relative) and channel value shape are fitted before any reading of z. A gain over Pearson alone is not reported as character.
- **Compare within invariance class.** z against mean-SPI is the primary contrast for "character beyond strength"; identity-aware methods are reported beside it wherever a shared channel set exists, never hidden.
- **Independent units decide power.** Windows are not units. Two target animals or six people cannot rank methods.
- **T and fs are chosen per dataset** from its spectrum and stationarity, not inherited; pyspi's bands are fractions of fs and its lags are in samples, so datasets that are compared are resampled to one fs first.
- **Surrogates are interpretable only for time-domain SPIs** (covariance, precision, rank, ids), which multivariate IAAFT reproduces; residuals on coherence, phase and spectral Granger SPIs are not [R].
- **Predictions and falsifiers are fixed before extraction**, and losses are kept.

## Shortlist, ranked

### 1. A consciousness atlas across species, agents and recording types

Revised 10 November after a data search; the dataset table below supersedes the earlier version of this section.

**Terms.** A *montage* is a recording's sensor set: which sensors, where they sit, and how they are referenced (19 scalp electrodes with an average reference, 16 bipolar ECoG derivations, a 16-contact linear probe in a fly brain). Different montages share no channels, so any method that needs channel identity (Pearson edges, tangent-space covariance, aligned MPIs, every raw-waveform learner) cannot be trained on one and applied to another. z can: its dimension is fixed by the SPI catalogue, not by the channels.

**Question.** Is there a common direction of change in dependence organisation from wakefulness to unconsciousness that recurs across agents (propofol, isoflurane, sevoflurane, ketamine), natural sleep, pathology (coma, disorders of consciousness), species (fly, mouse, rat, macaque, human) and recording types (LFP, ECoG, sEEG, scalp EEG)? Which statistics carry it, and does it follow reported experience rather than responsiveness or slow waves?

**Design: displacements, not positions.** z can be computed for every recording, but its absolute position depends on sampling rate, spatial scale (25 µm probe spacing in a fly against centimetres on the scalp), reference and volume conduction. Each dataset therefore enters as within-unit displacements, Δz = z(state) − z(same unit's wake baseline), never as raw positions. The primary test is alignment of displacement directions across datasets: cosine similarity of each held-out dataset's mean Δz with the direction learned from all other datasets, against a null that permutes state labels within units. Leave-one-dataset-out and leave-one-species-out are the generalisation tests. Within datasets that have graded states (sedation levels, induction and emergence, N1 to N3), the trajectory is the object: whether it follows the shared direction, and whether loss and recovery follow different paths, as fMRI suggests (Huang et al. 2021, NeuroImage 236:118042) [U, cited from the dataset README only].

**Why other representations cannot answer it.** Channel-identity methods are undefined across montages. Strength rises in most of these states (slow-wave synchrony), so strength alone cannot separate more coupling from differently organised coupling. The fair competitors are the identity-free blocks in z's invariance class: mean-SPI, Pearson edge quantiles, channel-pooled log spectra, and channel-pooled catch22. These go through the identical displacement-alignment test. The pooled catch22 block includes the autocorrelation features that already generalise across flies (Leung et al. 2025, below), so it is a strong control, not a strawman.

**Prior art that sets the bar.**
- Luppi et al. 2025, Nat Hum Behav: under propofol and sevoflurane, human fMRI functional connectivity loses individual uniqueness and moves towards the macaque pattern in a low-dimensional space [V, abstract and data statement]. This is the closest cross-species precedent; it is fMRI, strength-based, and its human anaesthesia data are available only from the original authors [V].
- Bola et al. 2018, NeuroImage 167:130–142: gamma-envelope correlations are hyper-correlated under propofol and in NREM sleep, much less under ketamine and in REM, in four NeuroTycho macaques and ten human sEEG patients [V, abstract]. The same macaques this repository uses; a strength measure within fixed montages.
- Toker et al. 2022, PNAS 119:e2024455119: one criticality measure across macaque ECoG, human ECoG under propofol or sevoflurane, seizures, LSD MEG and disorders of consciousness [V, full text]. Only summary statistics are shared; the human recordings came from collaborators [V].
- Engemann et al. 2018, Brain 141:3179: EEG markers (spectral, complexity, wSMI) classify disorders of consciousness across sites and electrode configurations, AUC about .77–.78 [V, abstract].
- Leung et al. 2025, PLoS Biol (with Fulcher and Tsuchiya): of more than 7,000 hctsa univariate features, 47 separated wakefulness from isoflurane or sleep in all four held-out fly datasets, mostly autocorrelation features; spectral power and complexity features did not generalise [V, abstract and full text]. The paper itself says that measures of interaction between channels (coherence, Granger causality, transfer entropy, integrated information) are the next step [V]. This sets up the narrative: univariate (hctsa) to pairwise (pyspi) to the organisation of pairwise dependence (z), on data the group can reach.

The novelty claim is therefore not "a cross-species consciousness signature" (Luppi, Bola and Toker each claim one), but which aspects of dependence change, read without choosing them in advance, across montages that no identity-aware method can bridge.

**Datasets.** Open = downloadable now; Request = needs an agreement or the authors. Units are independent animals or people. "Checked" means the catalogue entry or paper was read this session.

| Dataset | Species and recording | States | Units, M, fs | Access | Checked |
|---|---|---|---|---|---|
| Leung et al. 2025 flies (figshare 12309950; OSF 8wvsq) | fly, 16-contact linear probe LFP | wake vs isoflurane; held-out sets with sleep | 13 discovery + 49 evaluation flies; 15 bipolar; 1 kHz preprocessed | Open (CC BY 4.0) for the 13; OSF link for the rest; in-group via Fulcher | yes |
| Claar et al. 2023 (DANDI 000458) | mouse, skull EEG plus Neuropixels | awake vs isoflurane, with cortical stimulation | Allen Institute mice; about 30 EEG contacts [U] | Open (CC BY 4.0), 362 GB | yes |
| Kauvar et al. 2025 (DANDI 001326, 001347) | mouse Neuropixels; human sEEG | baseline, ketamine (and PCP in mice), post-infusion | 10 mice; human iEEG patients within 173 subjects | Open (CC BY 4.0) | yes; ketamine dose (dissociative or anaesthetic) unverified |
| Watson et al. 2016 (DANDI 000041) | rat, frontal silicon-probe LFP | wake, NREM, REM | 11 rats, 27 sessions | Open (CC BY 4.0) | yes |
| NeuroTycho anaesthesia | macaque, 128-contact ECoG | wake, ketamine–medetomidine, propofol | 4 macaques; already on Gadi | Open | yes; sleep sessions not found |
| Bastos et al. 2021 (Miller lab) | macaque, Utah arrays in several areas | wake vs propofol | 4 macaques | Request; no DANDI deposit found | partly |
| Jarraya multi-anaesthesia and DBS fMRI; PRIME-DE NeuroSpin | macaque fMRI | awake, ketamine, propofol, sevoflurane (two depths); thalamic DBS | 5–10 macaques | Request ("academic collaboration"); PRIME-DE subset after NITRC registration (CC BY-NC-SA) | yes |
| MNI Open iEEG Atlas | human sEEG and ECoG, normal-appearing channels | wake, N2, N3, REM | 100+ patients, short segments [U for exact length] | Open | partly |
| Bola et al. 2018 sEEG | human sEEG | wake, NREM, REM | 10 patients | Request | abstract only |
| Purdon et al. 2013 (DANDI 001285) | human 64-channel EEG | baseline, graded propofol, behaviourally timed loss and recovery of consciousness | 10 volunteers; **only 1 subject is deposited** | Open for 1; request the other 9 | yes |
| Chennu et al. 2016 (Cambridge Apollo) | human 91-channel EEG | baseline, mild, moderate propofol, recovery | 20 volunteers, about 7 min per level | Open (CC BY 2.0 UK); on Gadi at 19 channels | yes |
| Bajwa et al. 2025 (OpenNeuro ds005620) | human 65-channel EEG, 5 kHz | awake vs propofol sedation, 1-min EEG before each of 52 awakenings, 24 with reported experience and 5 without; TMS-EEG | 20 volunteers | Open (CC0); per-awakening report labels are not in the BIDS files [U where they are] | yes |
| DREAM database (Wong et al. 2025, Nat Commun 16:7495) | human sleep EEG/MEG, many labs | NREM and REM awakenings labelled experience or no experience | 505 people, 2,643 awakenings, 20 datasets | Open on figshare | abstract only |
| ANPHY-Sleep (Wei et al. 2024, Sci Data 11:896) | human 83-channel overnight EEG | wake, N1–N3, REM | 29 adults | Open (CC0, OSF) | yes |
| Gu et al. sleep EEG–fMRI (OpenNeuro ds003768) | human 32-channel EEG with simultaneous fMRI | wake and staged sleep | 33 adults | Open (CC0) | yes |
| PsiConnect (OpenNeuro ds006110) | human EEG and fMRI | before and after psilocybin 19 mg | 62 people | Open (CC0) | yes; EEG files not inspected |
| I-CARE (PhysioNet) | human 18-bipolar-channel EEG, 100 Hz | coma after cardiac arrest, hourly clips to 72 h; 3–6 month outcome | 607 public patients | Open | yes |
| Prolonged disorders of consciousness (figshare 23552964) | human 62-channel EEG, 250 Hz | patients vs controls; UWS/MCS split not stated | 59 patients, 32 controls; 60 × 2-s epochs each | Open (CC BY 4.0) | yes |
| Bath DOC (doi 10.15125/BATH-01632) | human EEG, motor-imagery BCI | UWS, MCS, locked-in | 42 patients | bona fide researchers only | abstract only |
| Michigan ketamine and propofol 128-channel EEG (Vlisides 2017; ReCCognition) | human EEG | ketamine sub-anaesthetic and anaesthetic; propofol or isoflurane to LOC | 10–30 volunteers | Request | papers only |
| Imperial LSD, psilocybin MEG; DMT EEG | human MEG/EEG | psychedelic vs placebo | 15–20 people | Request (LSD fMRI is open: ds003059) | partly |
| fMRI tier: Michigan propofol (ds006623), Naci propofol (ds003171), rat four anaesthetics (ds005687), mouse propofol (ds007783), rat medetomidine (ds001981) | fMRI | graded sedation; anaesthetic regimes | 26, 17, 38, 10, 24 | Open (CC0) | yes |

No open human EEG dataset with ketamine or with unresponsiveness under sevoflurane was found.

**Recording types do not have to match, but the claim has two tiers.** z is defined for any multivariate series, so mixing recording types is allowed and is the method's point. Two constraints decide how far mixing can go. First, pyspi's frequency bands are fixed fractions of fs and its lags are in samples, so the same SPI reads different physics at a 2 s fMRI repetition time and at 100 Hz electrophysiology. Second, spatial scale differs by orders of magnitude. Tier A is electrophysiology (LFP, ECoG, sEEG, EEG), resampled per dataset to a rate chosen against its dominant rhythms; it is the primary claim. Tier B is fMRI, where Luppi et al. already work; it is a replication in a slow modality, not pooled with tier A.

**Predictions (to freeze before extraction).**
- P1. The direct-versus-pairwise decoupling seen in NeuroTycho (robust partial r² against rank or ids profiles) recurs as part of the shared direction under propofol, isoflurane and NREM sleep, beyond IAAFT surrogates, and is weaker under ketamine and in REM.
- P2. In leave-one-dataset-out tests, the shared z direction aligns with held-out datasets better than the same procedure applied to mean-SPI, Pearson quantiles, pooled spectra and pooled catch22.
- P3 (the decisive dissociation). Within sedation and sleep, z displacement tracks reported experience (Bajwa awakenings, DREAM database) beyond its slow-wave covariates.

**Falsifiers.** P2 fails if any identity-free control aligns as well (paired interval over held-out datasets includes zero or favours the control). P3 fails if experience prediction from z is matched by pooled spectra; note that a posterior low-frequency power drop already predicts dreaming (Siclari et al. 2017, Nat Neurosci) [U, from memory], so this control is likely to be strong.

**Confounds.**
- Slow-wave power is shared by nearly every unconscious state and is the most likely common direction.
- Agent pharmacology differs: ketamine and REM are unresponsive but often experienced, so responsiveness is not a consciousness label.
- Volume conduction in scalp EEG against local LFP.
- In clinical cohorts: sedatives given to patients (Toker et al. note this), epileptic tissue, and electrode coverage chosen by the clinical hypothesis.
- Within-dataset drug and calendar confounds (NeuroTycho).

**Cost [I].** M ≤ 64 and T of 1000–2000 put a window at 0.2–0.6 core-hours. Fifteen datasets of about 1,000 windows each is about 4,500 core-hours, roughly 9,000 SU, and twice that with two surrogate variants. Raw data volume is large for DANDI 000458 (362 GB); only the EEG channels and a subset of sessions are needed.

**Chance of an NCS-level outcome [S].** P2 is the realistic target, and its prior is mixed: Leung et al. show spectral features failing to generalise across flies while autocorrelation succeeds, so the per-channel control may itself be the winner. A positive P3 would be the strongest result in the paper and is the least likely. In the weak case (no z advantage over identity-free controls), putting every dataset in one space is still useful descriptively, but it does not carry a Nature Computational Science paper by itself; it needs at least one named, surrogate-surviving statement about which aspect of dependence changes.

### 2. Electronic Rössler networks: character at matched synchrony across held-out wirings

- **Question.** Along the path to synchronisation, does z place a recording in its dynamical regime (incoherent, partial locking, phase slips, locked) when mean synchrony is matched, and does that reading transfer to wirings it has not seen?
- **Why other representations cannot.** Mean correlation and mean-SPI are matched by construction of the contrast. The 20 wirings share one degree sequence with neighbours rearranged [V], so degree-based topology summaries are constant. The pilot already shows the effect once: two settings with mean Pearson .9197 and .9197 have lag-profile z .72 and .85, explained by measured 58/116/174 Hz covariance profiles, and the explanation transferred to a second wiring in 15 of 16 settings [R].
- **Data.** Zenodo 3521009 (Vera-Ávila et al. 2020, Data in Brief 28): 28 electronic circuits, 20 network configurations, 101 coupling strengths each [V]; three repeats and T = 30,000 at 30 kHz per the repository's audit [R]. Open, 20.6 GB [R]. Units: 20 wirings, held out whole.
- **T and fs.** Chosen from the measured spectrum so that the highest retained harmonic has at least ten samples per cycle and a window spans several hundred carrier cycles; the pilot's 500 Hz, T = 5000 contract is the starting point, and T = 2500 is tested against it before the farm.
- **Prediction and falsifier.** Within strata of matched mean Pearson, z orders recordings by coupling better than mean-SPI and Pearson quantiles on held-out wirings; falsified if mean-SPI does as well, or if the ordering fails on held-out wirings.
- **Confounds.** Per-channel spectra change along the path (harmonic content, frequency pulling), so pooled and per-channel spectra are in the ladder and a matched-spectrum stratum is required. The character that the pilot found is linear-spectral (lag-profile organisation), not nonlinear form, and it should be named that.
- **Cost [I].** A full table at M = 28, T = 5000 took 3.6 core-hours in the pilot [R]. All 6,060 recordings would be about 44,000 SU, which is not justified. A design of 20 wirings × 26 coupling settings × 1 repeat (520 recordings) is about 3,800 SU, or half that at T = 2500.
- **Chance of a clear positive [S].** High, perhaps two in three, because the pilot effect exists. Its limit is scope: a laboratory system. It strengthens the measured-data leg of the order-parameter and character sections; it is not a field application on its own.

### 3. Intracranial EEG across patients with different implants

- **Question.** Do interictal recordings from patients who became seizure-free after surgery differ in dependence character (direct against pairwise, directed against undirected organisation) from those who did not, across implants that share no channels?
- **Why other representations cannot.** Every patient has a different electrode set, so no edge-aligned or raw-waveform model is defined across patients. Existing markers are per-channel or model-based: neural fragility, from a linear network model, predicted outcome at 76% in 91 patients against 48% for clinicians (Li et al. 2021, Nat Neurosci 24:1465–1474) [V, abstract].
- **Data.** OpenNeuro ds004100, the HUP iEEG dataset: 54–57 patients, CC0 [V, catalogue]; that it carries outcome labels and interictal segments is not confirmed here [U]. The fragility cohort's public subset is not confirmed [U]. The repository's CCEP study shows the pipeline handles M = 48–107 [R].
- **Risks.** Coverage is chosen by the clinical hypothesis, so implant geometry correlates with outcome. About 55 patients detect only large effects. The CCEP fixed axis tracked 8–24 Hz power almost perfectly (ρ −.97) [R], so the spectral confound is expected here too.
- **Cost [I].** With a 64-contact subset per patient and T = 1024, about 550 windows at 1.2 core-hours each, near 1,300 SU; full implants at T = 4096 would be about 40 times that.
- **Chance of an NCS-level outcome [S].** About one in five. High consequence, weak prior.

### 4. Within-person change: Dortmund Vital Study

- **Question.** Over five years in the same person, does z change beyond what that person's own spectral change predicts? A second, within-session contrast is before against after a two-hour task battery.
- **Data.** OpenNeuro ds005385, CC0: 608 adults, 208 with follow-up, 64 channels, about three minutes eyes-closed and eyes-open [R, from the candidates screen].
- **Why it is ranked last.** LEMON's age gain was channel-spectral and CHBMP's age signal vanished after five spectral components and did not replicate [R]. The honest prediction is a null beyond spectra. Its value is as a well-powered, within-person test whose null would close the age question. About 1,500 SU at 19 channels and T = 1000 [I].

### Considered and not shortlisted

- **UEA/Multiverse panel:** owned by the benchmark session; classification archives are structurally adverse to an identity-free representation, and its montage-free extension is application 1.
- **Freeway traffic:** it needs the direction axis, which was dropped at stage 2 of construction validation because it reverses in directed Kuramoto systems [R]; the candidates screen's rating predates that.
- **Cam-CAN, HBN, TDBRAIN, AOMIC, HCP fMRI, cardiorespiratory, finance, climate, structural health:** reasons in the candidates screen stand.
- **HCP MEG working memory:** still blocked on family grouping, and its bank is archived at K289.

## Recommendation

Build application 1 in stages: the fly data first (open, in-group, 62 animals, wake, isoflurane and sleep), then NeuroTycho, Chennu, Bajwa, ANPHY-Sleep, the MNI iEEG atlas, the rat and mouse DANDI sets and I-CARE, with the displacement test and predictions frozen before extraction. Requests worth sending now, because they set the timeline: the nine remaining Purdon subjects, Michigan (ketamine and propofol EEG), Jarraya (macaque multi-anaesthesia fMRI), Bola (sEEG sleep) and Imperial (psychedelic MEG). Run application 2 in its 520-recording form in parallel. Hold 3 and 4. If P2 fails against the identity-free controls, the real-world section should be the atlas regions plus the measured oscillator networks.

## Decisions for the user

1. Approve application 1 in stages, starting with the fly data, and decide which requests to send (above).
2. Sampling rate per tier A dataset: one common fs for all electrophysiology (simplest, same band edges everywhere) or an fs chosen per species against its dominant rhythms (more physical, but band edges then differ between datasets).
3. Approve application 2 and its subsampled design.
4. Pin for new extraction. The pin is r13 at `5d61d9c`; upstream main has since added partial-correlation SPIs (`pcorr_<estimator>`), which serve the direct-versus-pairwise reading directly. Whether they enter `benchmarked_p90` is unchecked [U]. New work goes to new output directories either way.
