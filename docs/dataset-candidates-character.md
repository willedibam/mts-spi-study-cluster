# Datasets for a character-of-dependence result outside anaesthesia (24 November 2026)

Scope: a metadata and literature screen, no pyspi and no waveform downloads. It is a delta on the [4 November screen](../research/representation/dataset_candidates/candidates.md), re-rated against the frozen protocol the MI-pair result now uses (`research/representation/atlas/mi_pair/preregs/claar-prereg.md`): 10-s windows (T ≈ 1000 at 100 Hz) with ≥ 10 stationary windows per state per unit, 17–30 channels, three joint spectrum-final IAAFT draws per window, a same-state split-half null in SDs across units, and a sign test over ≳ 15 units. Neural candidates are in scope (user, 24 November); non-neural ones are a plus.

Labels: **[V]** checked this session against the data's own metadata, official page or abstract; **[I]** inference; **[U]** unverified.

## Kill tests applied before anything else

1. **Sustained states.** Both states must be stationary across the window. Onset-type contrasts (substorm onset, congestion build-up, seizure evolution) reject IAAFT for drift, which is the classic false positive in the [reading rules](interpreting-spi-pairs.md).
2. **Timescales per window.** Count the characteristic periods or correlation times in a window, not the samples.
3. **Null scale.** Each unit needs ≥ 2 separated same-state segments, and there must be enough independent units.
4. **Geometry.** Sensors must be separate subsystems with heterogeneity across pairs. Gridded or closely spaced smooth fields sit at ceiling.
5. **Defensible axis.** Map priors only to these pairs:
   - Gaussian vs kNN MI (normal-scored, beyond IAAFT);
   - the lag pairs ppc–dswpli or cohmag–icoh, against zero-phase and stationary nulls;
   - pec vs its Gaussian partner |γ|²;
   - band or mediation.
   Never map to Pearson–Spearman (the form axis failed). A lag result is linear timing: it survives a zero-phase null, and IAAFT reproduces it.

## Ranked shortlist

Pair names use "–" in the table; in pyspi they are `ppc|dswpli` and `cohmag|icoh`.

| # | Dataset (field) | Access | M / T per unit / fs | Contrast; no-change null | Prior → z pair | Main confound → control | Ground truth | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | HUP iEEG, OpenNeuro ds004100 (human epilepsy) | CC0, open [V] | 57 subjects. 55 have two 300-s interictal runs; 36–185 good contacts (median 106); 2–38 SOZ contacts (median 8; 48 with ≥ 4); 256–1024 Hz [V, all channel/sidecar files] | Spatial: edges within the SOZ vs within matched non-SOZ contact sets, same interictal run. Null: matched non-SOZ set vs another matched non-SOZ set; run 2 replicates | Focal EEG is more nonlinearly interdependent beyond surrogates (Andrzejak et al. 2012, PRE 86:046206 [V abstract]) → Gaussian vs kNN MI | SOZ contacts are clustered neighbours (strong, short-range pairs) and carry interictal spikes → match on contact count, electrode, distance and strength range; transient cut and normal-scoring | Clinical label (plus resection and Engel outcome [V]); no physical change | **Cheapest strong test**, but a spatial pathology contrast on the same MI axis, not a state change |
| 2 | Freeway loops, raw 30-s PeMS (traffic) | free PeMS account; Caltrans reportedly disallows scripted download [V third-party] | M 15–40 mainline detectors per corridor [I]; congested core 2–4 h → T 240–480 at 30 s [I], below the protocol's 1000 | Stop-and-go core vs heavy but moving (synchronised) flow at matched mean occupancy, same corridor; free flow secondary. Null: same state on two weekdays | Stop-and-go waves travel upstream at −15 to −24 km/h with 4–20-min periods [V, several sources] → lag ∝ distance / c, heterogeneous across pairs → lag pairs | Gain and spectra move with state → zero-phase and stationary nulls per state; onset and dissolution excluded; free flow risks a noise-floor (strength) contrast | Physics: predicted lag per pair from distance and c, c is measurable independently | **Best non-neural.** Validation of known physics; linear timing, not non-Gaussian |
| 3 | FAIR-MAST tokamak magnetics (plasma) | anonymous S3 / Zarr, read this session [V]; licence [U] | Shot 30420: group `xmb` has 117 arrays of 82,500 samples (saddle, Mirnov, halo; Mirnov count [U]); `xmo` has 16 OMAHA coils of 7,000,000 [V]. Rates and flat-top length [U]; ~11.5k shots [V secondary] | L-mode vs H-mode within a shot, from the shipped H-mode detector trace `asm/hm_periods` [V]. Null: two L-mode stretches | ELMs are bursty edge events in H-mode; coherent MHD modes fix inter-coil phase by mode number [I, textbook] → pec vs \|γ\|², lag pairs | Within-shot drift → flat-top only; ELMs act as a shared gain → envelope null | Coil geometry gives the mode phase pattern; H-mode label automatic | **High breadth, unscoped.** Needs one shot inspected first |
| 4 | IMAGE magnetometers, 10 s (or SuperMAG 1 min, 1-s subset) | IMAGE: name/email form or bulk wget [V]; SuperMAG: registration, no redistribution [V] | ~30–40 Fennoscandian stations [U]; substorm expansion 30–60 min → T 180–360 [I] | Quiet vs storm main phase. Null: two quiet intervals | A substorm current wedge forms through a consistent lagged sequence (Orr, Chapman & Gjerloev 2019, GRL 46:6268, 86 substorms [V abstract]) → lag pairs | Substorm onsets fail kill test 1; gain spans orders of magnitude → envelope null; spatial smoothness | Onset catalogues exist [V] | **Weak.** Only a sustained main phase survives |
| 5 | MNI Open iEEG Atlas, sleep (von Ellenrieder et al. 2020) (human) | open [V] | 1 min per stage, 200 Hz bipolar, normal tissue only; N2/N3 from 91 patients, REM from 65 [V abstract]; channels simultaneous within a patient [U] | Wake vs N3. Null: split-half within one 60-s epoch only | Lag structure reorganises between wake and slow-wave sleep (Mitra et al. 2015) → lag pairs; MI pair as a specificity control (macaque sleep ≈ 0) | Six 10-s windows per stage; weak null; sparse M per patient | Stage scoring | **Specificity control**, not a headline |
| C | Daily industry returns, e.g. 49 industries (finance) | open [U] | M 49, ~250 days per year [U] | Crisis year vs calm year. Null: two calm years | Correlation rises with volatility, and adjusted correlation shows little crisis "contagion" (Forbes & Rigobon 2002 [V abstract, 4 Nov]) → MI pair | Volatility → GARCH or envelope null | none | **Control only.** Predicted to be explained by gain; tests the envelope null |

Rejected under the new protocol:
- **PEMS-BAY** (325 sensors, Jan–May 2017) and **METR-LA** (207, Mar–Jun 2012), 5-min [V]. The Nyquist period is 10 min, so 4–20-min stop-and-go waves are aliased. At ~0.5-mile spacing [U] the adjacent-pair lag is about 3 min, under one sample; only multi-km congestion fronts (≥ 5 km → ≥ 3 samples) resolve, and a peak gives T ≈ 24–48.
- **LANL three-storey**: already run, M = 4; most contact effects reappear in surrogates.
- **Z24**: on request [U]; temperature moves frequencies as much as damage; damage is mostly linear stiffness.
- **IMS bearings**: 4–8 channels, 3 run-to-failure tests [V].
- **Climate** (Hlinka et al. 2014: nonlinear contribution negligible; few independent ENSO events), **wind SCADA** (10-min), **PMU** (open multi-site data scarce): ratings unchanged from 4 November.
- **Paediatric iEEG** (ds004551, ds005398, ds003498): slow-wave sleep only, so no second state [V].
- **RESPect ds003848**: 3 + 3 long-term patients, unstaged [V].
- **HUP ictal vs interictal**: same patients, onset/offset events, 2–5 seizures per patient [V]. Seizures evolve in frequency, so this fails kill test 1 except possibly for late rhythmic windows.
- **SWEC-ETHZ**: 18 patients, long-term, 512/1024 Hz [V]. Same kill-test problem.

## Top three, in more detail

**1. HUP ds004100.** The design that passes every kill test is spatial, not temporal.
- **Contrast.** In one stationary 300-s interictal run (30 windows of 10 s at M ≈ 100), compare z(Gaussian MI, kNN MI) on edges *within* the SOZ with edges within matched non-SOZ contact sets. Andrzejak's focal pairs were neighbouring channels inside the focal zone [I: dataset description recalled, not re-read]. Edge subsets are legal here because both statistics are bivariate and unconditioned.
- **Edge count.** Within-SOZ edges number C(8,2) = 28 at the median and 6 at four contacts, so z per set is noisy. Restrict to the 48 patients with ≥ 4 SOZ contacts, and average over windows.
- **Null.** The no-change baseline is a non-SOZ set against another non-SOZ set, matched on contact count, electrode, inter-contact distance and strength range (the shape of the drop test in `mi_pair/edges-result.md`). Run 2 is replication, not the null.
- **Units.** Patients, so a sign test over ~48.
- **Risks.** SOZ excess may be driven by interictal spikes. The transient cut and normal-scoring separate that, and a spike-driven result would still be a statement about joint bursts, not coupling form. This is the same MI axis as anaesthesia, and there is no physical state change; it adds a field (pathology), not an axis.

**2. Raw 30-s PeMS corridors.** This is the one non-neural system where the predicted change is quantitative and independent of the data.
- **Prediction.** In stop-and-go, pair (i, j) lags by d_ij / c with c ≈ −15 to −20 km/h. At 0.5-mile spacing [U] adjacent lags are about 3 min (≈ 6 samples), and a 3-h core spans roughly 10–45 wave periods [I]. Lag grows with distance, so it is heterogeneous across pairs, which is what z reads.
- **Contrast.** The primary contrast is stop-and-go against heavy but moving flow at matched mean occupancy, as the 4 November screen proposed. Free flow at 30 s may sit near the noise floor, turning the contrast into one of strength and reliability.
- **Windows and units.** T of 240–480 is below the protocol's 1000. The lag pairs tolerate that; the MI pair does not. Units are corridors, with replicate weekdays.
- **Reading.** Linear timing: it survives zero-phase and stationary nulls, and IAAFT reproduces it by construction. This is validation against physics, not a non-Gaussian discovery, as on 4 November.
- **Open.** The Clearinghouse's 30-s product and layout [U]; detector spacing [U]; a state-selection rule frozen before z.

**3. FAIR-MAST.** Anonymous access and an automatic H-mode label were verified this session. It is the most novel field, and coil geometry gives a by-design phase pattern. Shots are short and evolving, ELMs act as a shared gain, and the licence, rates, coil inventory and flat-top durations are unverified. Inspect one shot (spectra, flat-top length, ELM rate) before freezing any protocol.

## What changed since 4 November

- The protocol's null and unit requirements now decide the ranking, and the MI pair is the best-validated axis. That promotes HUP, an open iEEG set with an SOZ prior from the surrogate tradition, over the scalp IQ and age cohorts.
- MAST is new.
- The 5-min benchmark traffic sets are now rejected on arithmetic; raw 30-s data is required.
- The 4 November bottom line still holds: no non-neural candidate is a clear non-Gaussian discovery case. Traffic is physics validation of a timing axis.
