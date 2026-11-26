# Second dataset for the MI-pair drug effect: macaque first, then other species and drugs (26 November 2026)

Scope: a metadata and paper screen for an independent replication of the NeuroTycho result. The feature is z = corr over pairs of Gaussian MI and KSG MI, Δz is the real recording minus a joint IAAFT surrogate, and Δz falls under the drugs (animal level t(3) = −6.11). No bulk data was downloaded and nothing was computed. The user widened the scope mid-task: any drug (including psychedelics), any species. A wake baseline is the minimum; wake + sleep + drug is ideal. This document adds to [Part C of the 19–20 November screen](../../dataset_candidates/candidates.md#part-c--anaesthesia-replication-of-the-atlas-mi-pair-19-november-2026), which covered human anaesthesia (PRIOS, Zelmann/DABI, Iowa, Krom, Tian, ds004541 and others); those datasets are cited there and not re-screened here.

Labels: **[V]** checked this session against the repository record, file header or paper text; **[I]** inference; **[U]** unverified or secondhand (for example a subagent's summary of a page).

Already used, excluded: NeuroTycho, Claar (DANDI 000458), Kauvar (DANDI 001347), Chennu, Bajwa ds005620, PRIOS ds004370, ds004541, Leung fly.

## Kill filters, applied before ranking

1. **Raw continuous field potentials.** Derived products (time-frequency maps, coherence, features, figure data) cannot give z.
2. **Spatially distinct pairs.** One laminar probe or one Utah array (400 µm pitch) is a small patch and risks a correlation ceiling. Count arrays or areas, not channels.
3. **Same-subject wake baseline.** Wake and drug must come from the same animal, ideally the same session.
4. **Scalp EEG ranks low.** The project already found scalp z at ceiling (Chennu).

## Ranked shortlist

| # | Dataset | Access | Animals; channels; fs | States and labelling | Duration per state | Main confounds | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | **Zenodo [15466548](https://doi.org/10.5281/zenodo.15466548)** "Multiunit and LFP recordings from non-human primates in depth anesthesia" (creator Stefano Ferraina, Sapienza University of Rome [V]; EBRAINS-Italy funded; paper not identified [U]) | Open, CC-BY-4.0, 9.4 GB, 8 MATLAB v7.3 files [V] | 3 macaques (B, C, P) [V]. **LFP only for monkey B:** `MEALFP_MB_PFc&PFc.mat` holds 192 channels numbered 1–192, 1,889,201 samples at dt = 5 ms, i.e. 200 Hz and 9,446 s [V, HDF5 header]. Two 96-electrode Utah arrays, PMc and PFc, is [I] (the description says 96-electrode arrays and B is listed with both areas, but the filename reads `PFc&PFc`; B's MUA comes as separate `MEAMUA_MB_PMc` and `MEAMUA_MB_PFc` files [V, file list]). C and P have MUA only (log 200–1500 Hz power at 5 ms), going by the file names [I]; their files are MATLAB v5, so I could not read their headers remotely. MUA is not a field potential | Dexmedetomidine + ketamine ("Dexdomitor + Ketamine") [V, protocol PDF]. Same session for each monkey: wake, injection, anaesthesia with a marked "deep anaesthesia" span, awakening, wake. Labels are event times: injection, eyes open, first finger or arm movement; monkey C Exp 1 has an α2 antagonist [V] | Monkey B: wake ≈ 31 min (166–2039 s), deep anaesthesia ≈ 50 min (3000–6000 s), post-awakening wake ≈ 27 min (7971–9612 s) [V, PDF]. `MEALFP/time` runs 165.94–9611.94 s, so the protocol times index the LFP directly [V]. Monkey C Exp 2 (MUA): wake 25 min, wake after eyes open 56 min. Monkey P (MUA): wake only 3 min before injection | LFP stored at 200 Hz with a 0.2 Hz high-pass. `SmoothingWindow` is 0.04 s, but `smoothed` is false, and 2 × 60 s of all 192 channels show no 25 Hz null, which a 40-ms boxcar would leave, so the LFP is not smoothed [V, Welch spectra]. Strong 50 Hz mains, so notch it. Median \|r\| across pairs is .69 in wake and .49 in deep anaesthesia; between the two 96-channel blocks it is .63 and .41 [V, same 2 × 60 s]: high, but not at ceiling. Anti-alias filter before the 200 Hz subsampling [U]. Reference: `MUARef.Method` is "LeftTail" (MUA only); LFP reference [U]. Post-awakening wake carries residual drug; arm and eye movements occur at awakening | **Best open macaque set.** Its drug class matches NeuroTycho's ketamine–medetomidine arm (dexmedetomidine is the active enantiomer of medetomidine [U, from memory]), whose effect is large (−3.8 to −4.2 null SD bipolar); medetomidine alone was the weakest arm. Only 1 animal has LFP, so this is a within-session replication, not an animal-level test |
| 2 | **Dryad [10.5061/dryad.98sf7m0ft](https://doi.org/10.5061/dryad.98sf7m0ft)**, Ballesteros, Briscoe & Ishizawa, eLife 2020 (MGH) | Open, CC0, 3.8 GB, 3 `.mat` session files [V]. Scripted download returned 403, so use a browser or the Dryad API with a token [V] | The paper has 2 macaques [V]; the deposit has 3 sessions (E041212, E050312, E080212), with animal per file [U]. Per session, 16-electrode floating microelectrode arrays in S1 and PMv (32 channels, 400 µm pitch), raw LFP at 1 kHz, "not further processed" [V, Dryad README] | Dexmedetomidine infusion (18 µg/kg/h for 10 min, then 4 µg/kg/h) [V, paper]; the antagonist atipamezole in some sessions [V]. `sessionInfo` gives anaesthesia start and end, LOC, ROC and ROPAP times from task performance [V] | Awake task baseline, then about 60 min of infusion [V, paper]; durations per file [U] | Wake baseline is a behavioural task (tone and lever, rewarded), drug is not [I]. Only 2 arrays, so cross-array pairs give the spatial contrast. Dexmedetomidine alone. In NeuroTycho medetomidine alone was the weakest arm (−1.5 null SD bipolar, mostly gone at matched burst index), so a null here would be uninformative | **Second open macaque set**, runnable now, but weak: a session-level check of the drug arm least likely to show the effect |
| 3 | **Miller lab (MIT), by request.** Propofol: Bastos et al. eLife 2021 and Eisen et al. Neuron 2024. Ketamine and dexmedetomidine: Cell Rep 2025 (PMC12178116) | Request to the lead contact, Earl K. Miller: "will be shared by the lead contact upon request" [V, Cell Rep]; Eisen 2024 the same [U, summary]. The public Bastos GitHub repository holds only spike-rate and coherence `.mat` files [V, file list] | Propofol: 2 monkeys with four 8×8 Utah arrays each (8A, vlPFC, 7A/B, STG), LFP at 1 kHz, 10 + 11 sessions [V, eLife]; thalamic stimulation sessions use further monkeys (3 and 4) [I]. Ketamine and dexmedetomidine: 2 monkeys (S, P), four 6×6 Utah arrays (left and right vlPFC and dlPFC), LFP at 1 kHz [V] | Within session: awake, loss of consciousness (eyes closed), unconscious, recovery (eyes open) [V]. Ketamine 10 mg/kg and dexmedetomidine 20 µg/kg with a lever task before and after [V] | Propofol infusion 20–30 min at a high rate, then maintenance [V]. Before, during and after for ketamine and dexmedetomidine [V]; minutes per state [U] | Tones and airpuffs run through the propofol sessions [U, summary]. Arrays share a subdural reference [V]. Only 2 animals per drug | **Best by-request macaque set:** 3 drugs, 2 + 2 animals, 4 arrays, 1 kHz. One email covers all three; propofol and ketamine matter most, dexmedetomidine least (see row 2) |
| 4 | **Saalmann lab (Wisconsin), by request.** Redinbaugh et al. Neuron 2020 | "available upon reasonable request" [U, summary] | 2 macaques [V, intro]; linear arrays with 16–24 contacts in FEF, LIP and central lateral thalamus [U] | Wake, NREM sleep, isoflurane, propofol [V]. Anaesthesia in separate sessions from wake; sleep and wake share sessions [V]. Ketamine induction before the anaesthetic [U] | 9 isoflurane and 9 propofol sessions [V, Sci Adv 2026] | Only 3 probes; within-probe pairs risk the ceiling. Ketamine induction contaminates propofol [U]. Wake and drug come from different sessions | **The only macaque lead with natural sleep**, but weak for z. A secondary request |

Not ranked: Ishizawa 2016 (J Neurosci) and Patel 2020 (Brain), both MGH propofol in macaques. Europe PMC returned no full text, so their data statements are [U]. A Dryad search for Ishizawa finds only the row 2 deposit [V].

## Other species and drugs (specificity or dose tests, not replications)

- **DANDI [001417](https://dandiarchive.org/dandiset/001417), OpenScope Psycode** (mouse psilocybin; CC-BY-4.0, 355 GB, 20 NWB files) [V].
  - **Recordings:** 6 Neuropixels probes per session, LFP at 1250 Hz with 96 channels per probe, across hippocampus, thalamus and cortex [V, NWB header].
  - **Design:** each mouse has 2 days. The injection is in mid-session (about 2,770–2,880 s) and is labelled `saline` or `psilocybin` [V, all 20 headers]. 8 mice have one saline day and one psilocybin day; 2 mice (767932 and 773206) are labelled saline on both days [V], which is possibly a labelling problem [U].
  - **Confound:** visual stimuli and a rewarded change-detection task run before and after the injection [V]. I did not find a clean stimulus-free block; `spontaneous_presentations` merges into one 5,906-s span [V], and its meaning is [U].
  - **Role:** the saline day is a ready-made time-in-session control. A wake vs psychedelic test, not anaesthesia.
- **Zenodo [3992359](https://doi.org/10.5281/zenodo.3992359)** (human inhaled DMT, raw EEG `.bdf`, 35 participants; eyes-open, eyes-closed and DMT files) [V]. Scalp EEG, so low rank.
- **Zenodo [17080375](https://doi.org/10.5281/zenodo.17080375)** (human psilocybin + midazolam EEG, Banks lab; 11.9 GB, "AllEEGData_postICA") [V]. ICA-cleaned scalp EEG, so low rank [I].
- **G-Node GIN `SW_lab/Anesthesia_CA1`** (Yang et al. PLoS Biol 2021): mouse CA1 imaging and electrophysiology, awake against several general anaesthetics [V, repository description]. Channel count and reference [U]; the page did not load. Probably few channels in one structure [I].

## Checked and rejected

- **Xiong et al. PNAS 2024** (propofol oddball, Figshare 26980630): one file, `TFR_all.mat` (time-frequency maps) [V].
- **Bastos GitHub** `Propofol-unconscious-NHP`: spike-rate and coherence files only [V].
- **Eisen 2024 Zenodo**: code and figure data [V].
- **DANDI 001838**: awake visual global/local oddball in all 20 sessions [V].
- **Monash 10.26180/32817989** (registered report): hctsa features from NeuroTycho and Iowa [V].
- **Zenodo 10534967**: fMRI.
- **OpenNeuro ds006890** (wireless ECoG, 2 *Macaca fuscata*): rest and task sessions only; the README never mentions anaesthesia or sleep [V].
- **Zenodo 15295720**: mouse calcium imaging.
- **Zenodo 5137888**: no drug.
- **DANDI 001765**: rats anaesthetised throughout, LSD vs saline across animals, so no wake baseline [V].
- **DANDI 001326**: mouse units only [V].
- **DANDI 001285**: see Part C.

## Recommendation (assistant's assessment)

1. **Run Zenodo 15466548 monkey B first.**
   - **Why:** it is open, has 192 LFP channels on two arrays, and covers wake, dexmedetomidine + ketamine and wake again within one session. That is the closest open match to NeuroTycho's ketamine–medetomidine arm.
   - **Design:** pre-injection wake against deep anaesthesia, with post-awakening wake as a drift and recovery check. Pick channels so that pairs span both arrays (e.g. 16 + 16), since within-array pairs risk the ceiling. Notch 50 Hz; the LFP is not smoothed (spectral check above).
   - **Access:** download the 2.8 GB LFP file; nothing else is needed. Then email the depositor for the LFP of monkeys C and P (the record says "Further details from the main contact person"; the Zenodo creator is Stefano Ferraina, Sapienza University of Rome [V]). With those, the test has 3 animals.
2. **Request the Miller-lab Utah data.**
   - **What:** propofol, ketamine and dexmedetomidine, at the animal or session level. This is the strongest route to a multi-drug replication.
   - **Access:** one request to the lead contact named in Cell Rep 2025.
3. **Use Dryad 98sf7m0ft only as a low-priority check.** It is open, with 32 channels at 1 kHz and 3 sessions, but dexmedetomidine alone was NeuroTycho's weakest arm, so only a positive result would be informative. Download it in a browser.

No open dataset found here has natural sleep alongside a drug in the same macaques, so NeuroTycho stays unique for that. Frozen test for row 1: [`preregs/ferraina-prereg.md`](preregs/ferraina-prereg.md).

**Wake + sleep + drug beyond NeuroTycho (26 November; none found open, which is not the same as none existing):**
- **By request, same subjects:**
  - Saalmann macaques (row 4): wake, NREM, isoflurane and propofol, on laminar probes.
  - Zelmann MGH sEEG on DABI (Part C): propofol in the operating room, with epilepsy-unit wake and sleep in 13 of 14 patients. Access was requested on 20 November.
  - Iowa iEEG (Part C): propofol and dexmedetomidine inductions plus overnight sleep, under a data use agreement. Whether the sleep and drug patients overlap is [U].
  - Paediatric sEEG (PMC12835890, Part C): awake, slow-wave sleep and propofol on different days, with no data statement; contact the authors.
- **Restricted, different subjects:** Zenodo [806176](https://doi.org/10.5281/zenodo.806176) (Massimini and Laureys; access "restricted", no files visible). It holds human rest EEG in wake against ketamine (N = 5), propofol (N = 5) and xenon (N = 5), and N3 sleep (N = 4), in separate groups as listed [V, record]. It is scalp EEG, so the ceiling risk applies. Source paper and channel count [U].
- **Derived only:** Zenodo 5010564 (Bagur et al., mouse olfactory-bulb LFP across sleep and anaesthesia; figure `.ods` files) and 3975771 (Lendner et al. 2020; figure `.mat` files) [V].
- **Searched for this question:** DANDI full-text queries pairing sleep with each agent (only 002096, a CA1 learning set, came back), and four Zenodo boolean queries. Rodent sleep plus anaesthesia LFP was not otherwise screened.

**Blocking:**
- **Animal-level test:** none of the open sets supports one (1 animal with LFP; Dryad with 3 sessions). A replication would be within-animal and session-level until the C and P LFP or the Miller data arrive.
- **Preprocessing:** the 200 Hz LFP of row 1 is enough for the 100 Hz pipeline and is not smoothed, but its reference and anti-alias filter are undocumented.

## Method and provenance

- **DANDI:** the API metadata (name, description, keywords) of all 684 non-empty dandisets, filtered for macaque and drug terms. NWB headers were read remotely (no bulk data) for 001838 (3 headers plus all 20 session descriptions) and for all 20 of 001417.
- **Zenodo:** API keyword searches: 12 drug queries at 8 hits each, then 11 primate, anaesthesia and sleep queries at 25 hits each. The top pick surfaced only through the phrase "non-human primates", so a phrasing I did not try could hide another set. I fetched the 0.1 MB protocol PDF of 15466548 and range-read about 1.2 MB of the LFP file's HDF5 header, and on 26 November its `time` vector (about 7 MB).
- **Others:**
  - Dryad: the author search and file list worked; keyword queries printed nothing, possibly a silent failure.
  - Figshare: API records 26980630 and 10.26180/32817989 only; a keyword search returned irrelevant hits, so Figshare is effectively unsearched.
  - GIN: keyword search; repository pages failed to load.
  - Europe PMC: full text for papers with data statements.
  - Subagent page summaries: Bastos, Eisen, Xiong, Redinbaugh; facts taken only from these are marked [U].
- **Open leads not read:** Table S1 of Luppi et al. (bioRxiv 2025.03.22.644729), which lists cross-species awake and anaesthetised datasets; the ketamine Utah-array HMM paper (medRxiv 2020.11.12.20221366; access blocked); EEG in Tasserie et al. Sci Adv 2022 (fMRI-centred).
- **Not searched:** OpenNeuro in full (the catalogue download was stopped; Part C covered all 74 iEEG sets), CRCNS (the listing page had no hits), EBRAINS (the API call failed), OSF.

Scratch files from this screen (not committed) are in this session's scratchpad.

## Consolidated inventory: wake, sleep and drug datasets for the MI pair (26 November, evening)

Compiled from this screen, [Part C](../../dataset_candidates/candidates.md#part-c--anaesthesia-replication-of-the-atlas-mi-pair-19-november-2026), the [scoping table](../../representation_state/notes/real-world-application-scoping.md) and the MI-pair results; no new search. "Run" means an MI-pair residual exists in this repository. W = wake, S = natural sleep, D = drug. Status of unrun sets was checked by grepping `findings.md`, `mi_pair/` and the git log for the dataset name [V for absence in those places only].

**Scored**

| Dataset | Species, recording | States | Units | MI-pair outcome |
|---|---|---|---|---|
| NeuroTycho | macaque, 128-ch ECoG | W (eyes open, closed), S, ketamine, KT–medetomidine, medetomidine, propofol | 4 | Anchor: drugs fall in 22 of 23 sessions; sleep same sign at about a tenth |
| Claar, DANDI 000458 | mouse, skull EEG | W, isoflurane | 17 | −.0191; real z falls; burst-suppression-borne |
| PRIOS, ds004370 | human ECoG | W (ward), propofol (theatre, later day) | 5 of 7 | REPLICATES −.0083; surrogate-driven; clear of its own-spectrum offset (speccalib) |
| Kauvar, DANDI 001347 | human sEEG | W, sub-anaesthetic ketamine | 7 | ABSENT |
| Ferraina, Zenodo 15466548 | macaque, 2 Utah arrays | W, dexmedetomidine + ketamine, W | 1 (LFP) | Frozen UNINFORMATIVE (ceiling); secondaries fall in the same direction; surrogate-driven |
| Chennu | human scalp EEG, 19 ch | W, propofol sedation levels | 20 | Null within ±0.2 SD; z .983 |
| ds004541 | human scalp EEG, 58 ch (6–23 good) | W, loss of consciousness (agent unstated) | 4 of 8 | +.0055, 1/4 negative; z .985 |
| Leung fly | fly, 15 bipolar LFP | W, isoflurane | 13 | Residual about 0 (−.002); no null scale |

**Open, never run for the MI pair**

- **Bajwa ds005620** (human 65-ch scalp EEG, W vs propofol sedation, 20 people): both screens list it as "already used", but no MI-pair result exists; it was the atlas plan's confirmation step [I]. It was set aside only by the scalp-ceiling inference from Chennu and ds004541, which has not been tested on a 65-channel montage [I].
- **Sleep without drug (W + S, other subjects):** ANPHY-Sleep (83-ch EEG, 29 adults, W/N1–N3/REM), the MNI iEEG atlas (W/N3/REM), RESPect ds003848 (ECoG/sEEG, 6), ds003768 (32-ch EEG–fMRI, 33), DREAM (awakenings with experience labels). None run.
- **Psychedelics:** OpenScope Psycode DANDI 001417 (mouse Neuropixels LFP, psilocybin vs saline day, 10 mice, task running throughout); PsiConnect ds006110 (human EEG, psilocybin, 62; EEG files not inspected); Zenodo 3992359 (human DMT EEG, 35); Zenodo 17080375 (psilocybin + midazolam EEG, ICA-cleaned). LSD: only fMRI is open (ds003059, tier B); DANDI 001765 (rat LSD) has no wake baseline.
- **Other drugs:** Dryad 98sf7m0ft (macaque dexmedetomidine, 3 sessions); EBRAINS 10.25493/S0DM-BK5 (rat 16-ch epidural EEG, propofol, sevoflurane, ketamine; stimulation-centred); GIN `SW_lab/Anesthesia_CA1` (mouse CA1, several anaesthetics); DANDI 001285 (provenance unchecked).
- **Fly sleep:** the held-out flies with sleep (Leung 2025, OSF 8wvsq) [U whether continuous].

**By request or restricted** (W + S + D in the same subjects marked ★)

- ★ Saalmann (macaque laminar probes; W, NREM, isoflurane, propofol).
- ★ Zelmann, DABI (human sEEG, 14; operating-room W and propofol, plus ward W and S in 13); access requested 20 November.
- ★ Iowa (human iEEG; 19 propofol, 10 dexmedetomidine inductions, 24 overnight sleeps; DUA); subject overlap [U].
- ★ Paediatric sEEG (PMC12835890; 30 children; W, slow-wave sleep, propofol on different days); no data statement.
- Miller lab (macaque, 4 Utah arrays; propofol, ketamine, dexmedetomidine, 2 animals each).
- Ferraina monkeys C and P LFP; the 2 unreleased PRIOS patients; the remaining Purdon subjects.
- Krom (UCLA depth, propofol, 8); Tian (MGH sEEG, ketamine then propofol, 10); Michigan ketamine 128-ch EEG (Vlisides, 10–15; sub-anaesthetic and anaesthetic dose).
- Massimini/Laureys Zenodo 806176 (restricted; scalp; W vs ketamine, propofol, xenon and N3 in separate groups).
- Imperial LSD and psilocybin MEG, DMT EEG.
- Jarraya macaque fMRI (ketamine, propofol, sevoflurane; tier B).

**Not searched:** OpenNeuro beyond its iEEG sets, CRCNS, EBRAINS (API failed), OSF, Figshare (search unusable), rodent sleep plus anaesthesia LFP. The cheapest next lookup is Table S1 of Luppi et al. (bioRxiv 2025.03.22.644729), a cross-species list of awake and anaesthetised datasets [U, not read].
