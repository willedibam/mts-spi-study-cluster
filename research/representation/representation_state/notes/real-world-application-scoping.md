# Real-world applications of z: scoping for a full launch (10 November 2026)

Status: proposal. Nothing here has been downloaded, extracted or run, and nothing launches without user approval. It draws on, and does not repeat, the [results overview](real-world-results-overview.md), the [dataset candidates screen](../../dataset_candidates/candidates.md), the [character-axes findings](../../character_axes/findings.md) and the benchmark protocol on branch `claude/mts-benchmark-screen-4cbb57`. Labels: **[V]** checked this session (abstract, catalogue page or repository file, not a full-text replication); **[R]** recorded in this repository and not re-checked; **[I]** inferred; **[U]** unverified; **[S]** speculation.

## Where the evidence stands

The evidence so far does not support a Nature Computational Science-level empirical claim, in either form the project set. No dataset shows z at or above its strongest comparator, and in every cohort where a per-channel spectral control was run (AD/FTD, LEMON, propofol, NeuroTycho) z adds nothing over Pearson + strength + channel spectra [R]. The one frozen independent confirmation (CHB-MIT → Siena) failed, and CHBMP IQ is a pre-registered null [R]. What stands is narrower: z is a reliable, individual and transferable descriptor (split-half .97; NeuroTycho transfer AUROC 1.0) [R]; in NeuroTycho the validated core explains z's gain over Pearson [R]; and two feature-level effects go beyond multivariate IAAFT surrogates: the decoupling of direct (robust partial r²) from pairwise (rank, ids) dependence patterns under anaesthesia, in 6 of 6 animal–drug units, and propofol's lower linear–rank agreement in 16 of 20 people [R].

Three kinds of result could change that, and the shortlist is built on them.

1. **A task that only an identity-free representation can attempt.** z is unchanged by a common relabelling of channels, so it compares recordings with different channel sets, species or implants. Pearson edges, tangent-space covariance, aligned MPIs and every raw learner need one shared channel set. Against the other identity-free blocks (mean-SPI, quantiles of the Pearson edge distribution, channel-pooled spectra) the comparison is fair and has not been run on real data.
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

### 1. Unconsciousness across drugs, sleep, species and montages

- **Question.** When responsiveness is lost, do the patterns of direct and of pairwise dependence decouple, and is that one signature across propofol, ketamine–medetomidine and NREM sleep, in macaque ECoG and human scalp EEG? Does it spare REM and ketamine, the states with dream-like experience?
- **Why other representations cannot answer it.** Strength rises in all of these states: slow-wave synchrony raises mean correlation, so strength cannot separate "more coupling" from "differently organised coupling". Topology and learned latents are tied to one channel set, so none transfers from a 128-contact macaque array to a 19- or 83-channel human cap. Only identity-free blocks can be trained in one species and scored in the other.
- **Literature.** Propofol unconsciousness fragments cortical networks at slow-oscillation onset (Lewis et al. 2012, PNAS) [U, not re-read]. Loss of consciousness under propofol comes with rising low-frequency power and frontal alpha in 64-channel volunteer EEG (Purdon et al. 2013, PNAS) [V, search summary], and propofol slow waves resemble NREM slow waves in 256-channel EEG (Murphy et al. 2011, Sleep) [V, search summary]. The closest precedent uses these same macaques: gamma-envelope correlations are "hyper-correlated" in propofol anaesthesia and NREM sleep, much less in ketamine and REM (Bola et al. 2018, NeuroImage 167:130–142; four NeuroTycho macaques and sEEG from ten patients) [V, abstract]. That result is a strength measure within fixed montages. The gap is a montage-free statement about how dependence is organised, tested across species by transfer.
- **Data and access.** NeuroTycho anaesthesia: four macaques, 128-contact ECoG, ketamine–medetomidine in all four and propofol in two [V, review table], already staged on Gadi for 16 bipolar channels [R]; its sleep sessions and single-agent ketamine or medetomidine sessions are not confirmed here [U]. Human sedation: Chennu et al. 2016, 20 people × four propofol levels, bank on Gadi [R]; this is responsive sedation, not unresponsiveness. Human sleep: ANPHY-Sleep, 29 healthy adults, overnight 83-channel EEG with staging, CC0 on OSF (Wei et al. 2024, Sci Data 11:896) [V]. An open human EEG dataset with propofol-induced unresponsiveness was searched for and not found [U]; without one the human anaesthesia arm rests on sedation.
- **Size and units.** Units: 4 macaques (6 animal–drug units), 20 people (sedation), 29 people (sleep). M: 16 bipolar and a 32–64 contact subset per macaque; 19 canonical channels and a 64-channel subset for ANPHY. All at 100 Hz after a 0.5–45 Hz band-pass, T = 1000 (10 s), which is the existing propofol contract and the length CHBMP's sweep favoured (subject-mean reliability .93 at T = 1000 against .89 at T = 2000 for the same data budget) [R]. NeuroTycho must be re-extracted at 100 Hz; its r10 bank is at 250 Hz.
- **Predictions.** P1 (feature level, signs fixed in advance): alignment between robust partial r² and rank or ids edge profiles falls from wake to propofol and to N2/N3 in both species, and falls less in REM and under ketamine–medetomidine; the fall exceeds its IAAFT surrogate value. P2 (transfer): a readout of z fitted on macaque ECoG separates human wake from N3, and baseline from moderate sedation, better than the same readout of mean-SPI, Pearson edge quantiles and channel-pooled relative spectra.
- **Falsifiers.** P1 fails if the human fall is absent or fully reproduced by surrogates. P2 fails if any identity-free strength or spectral block transfers as well as z (paired interval over people includes zero or favours the block).
- **Confounds.** Slow-wave power is shared by every one of these states and is the most likely winner: pooled delta power may transfer across species as well as z does [I]. Others: ECoG against scalp volume conduction and reference; fs and filter differences; drug and calendar period confounded in NeuroTycho [R]; sleep-stage scoring windows; robust-estimator sensitivity to artefact transients (the decoupling's source is still unexplained [R]).
- **Ladder.** Identity-free: z, mean-SPI, Pearson edge quantiles, pooled spectra, pooled catch22, and each of these + z. Within each dataset, where channels are shared: Pearson, per-channel spectra, tangent-space covariance, aligned MPIs, MiniRocket, MultiRocket-Hydra, QUANT, InceptionTime. Surrogates: both IAAFT variants for every bank.
- **Cost [I].** About 1,000 macaque windows at M = 32 (0.2 core-hours each) and M = 64 (0.6), 800 existing human sedation windows re-used, and about 1,400 sleep windows at M = 19–64: roughly 1,200–1,800 core-hours for real data and twice that again for two surrogate variants, so 7,000–11,000 SU. A 16-channel-only version is about a fifth of that.
- **Chance of an NCS-level outcome [S].** P1 replicating in humans beyond surrogates: perhaps one in two. z beating pooled spectra in cross-species transfer: perhaps one in three. This is still the best-supported option: it extends the only robust positive, uses the one capability no comparator has, and its negative outcome is informative.

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

Launch application 1 at 16–32 channels first, with P1 and P2 frozen, and application 2 in its 520-recording form in parallel: one biological test with a real chance of a distinctive result, and one controlled system with a high chance of a clean one. Hold 3 and 4. If P1 does not replicate in humans and z does not beat pooled spectra in transfer, the real-world section should be the atlas regions plus the measured oscillator networks, and the paper should not claim a biological application.

## Decisions for the user

1. Approve application 1, and say whether to search further (or ask collaborators) for a human dataset with anaesthetic unresponsiveness.
2. Approve application 2 and its subsampled design.
3. Pin for new extraction. The pin is r13 at `5d61d9c`; upstream main has since added partial-correlation SPIs (`pcorr_<estimator>`), which serve the direct-versus-pairwise reading directly. Whether they enter `benchmarked_p90` is unchecked [U]. New work goes to new output directories either way.
