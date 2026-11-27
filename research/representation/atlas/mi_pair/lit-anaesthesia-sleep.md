# Literature positioning: Δz against prior anaesthesia and sleep metrics (27 November 2026)

Purpose: say where the MI-pair result (Δz, the fall in z(Gaussian MI, KSG MI) beyond joint IAAFT under ketamine, ketamine–medetomidine, medetomidine and propofol, with sleep at −0.24 SD against drugs at −3.3) sits in the literature, and whether any prior metric already has its profile. In-house numbers come from [`findings.md`](../findings.md) and the unmerged benchmark (`origin/claude/mi-alt-stats-dec35d:research/representation/atlas/mi_pair/bench-result.md`). Wording follows [`docs/interpreting-spi-pairs.md`](../../../../docs/interpreting-spi-pairs.md): Δz is about how the dependence is organised across edges, not about how much of it there is.

Labels on each claim: **[F]** read in full text, including Europe PMC full-text search of the article; **[P]** read in the verbatim part of a partial full-text read; **[A]** abstract only; **[S]** secondhand (search snippet, a model summary of an unread section, or a citing paper); **[U]** unverified. Paywalled or bot-blocked pages were skipped. No account was created.

## 1. NeuroTycho papers using the anaesthesia and sleep ECoG

Denominator: a Europe PMC full-text search for "neurotycho" with anaesthesia, ketamine, propofol or sleep gives 45 hits (27 November). 14 of them analyse the anaesthesia or sleep recordings. The rest use other NeuroTycho tasks, cite the database only, or were not checked. Two preprints (Fuentes 2021, Padovani 2016–2021) and the two Liu 2015 papers, which use the same RIKEN arrays without naming NeuroTycho, are added. 16 were read, some only in part.

- **Yanagawa et al. 2013**, PLoS ONE 8:e80845, doi 10.1371/journal.pone.0080845 [F]. Bipolar spectral Granger causality from 6 to 100 Hz in 4 macaques, with an SVM per region pair and a PARAFAC decomposition. Awake states load on alpha/beta components and unconscious states on theta/gamma. The propofol and natural-sleep experiments were pooled into one "LOC" class with ketamine–medetomidine. A universal awake-versus-LOC classifier works, and the LOC components match across the three states (Fig. 5, spatial-frequency correlation). Sleep is grouped with anaesthesia by design. No sleep-versus-drug contrast is tested.
- **Liu, Yanagawa, Leopold, Fujii, Duyn 2015**, Cereb Cortex 25:2929, doi 10.1093/cercor/bhu089 [P]. Band-limited-power correlation clusters (k = 8) found in wake reappear under ketamine–medetomidine, propofol and sleep. The authors call them robust to state. The claim that spectral contrasts between clusters shrink most under propofol is [S]. Sleep was defined by a delta synchronisation index, not staged.
- **Liu et al. 2015**, NeuroImage 116:222, doi 10.1016/j.neuroimage.2015.04.003 [P]. In monkeys C and G, a sequential spectral transition (mid-band drop, then high-band rise, then low-band rise) recurs in sleep and eyes-closed wake and resembles propofol induction. This groups the states. A late mid-band rebound in sleep but not propofol is [S].
- **Solovey et al. 2015**, J Neurosci 35:10866, doi 10.1523/JNEUROSCI.4895-14.2015 [P]. A VAR(1) fit to 128 channels per 500-ms window. The fraction of eigenmodes near instability falls under ketamine–medetomidine (≈17.5 → 13.4%) and propofol (≈15 → 4.5%): one direction for two agents, with agent-inconsistent spectral changes. Sleep was not analysed.
- **Alonso et al. 2019**, Sci Rep 9:4927, doi 10.1038/s41598-019-41345-4 [F]. VAR(1) eigenvalue distributions classify awareness under ketamine–medetomidine and propofol. After surrogation, the ketamine–medetomidine differences persist while the propofol differences depend on global correlations. The surrogate dependence is therefore agent-specific. Sleep was not analysed (inferred from text search).
- **Tajima et al. 2015**, PLoS Comput Biol 11:e1004537, doi 10.1371/journal.pcbi.1004537 [P]. Model-free cross-embedding (convergent cross mapping, kNN). Embedding complexity is higher awake than under ketamine–medetomidine and propofol, and the directionality structure collapses. The correlation network is similar across states. There is no linear-surrogate control and no sleep.
- **Varley et al. 2020**, PLoS Comput Biol 16:e1008418, doi 10.1371/journal.pcbi.1008418 [P]. Avalanche criticality in Chibi only, from ketamine and propofol boluses. Propofol strongly restricts avalanches and lowers complexity; ketamine moves the same way, weakly. **Varley et al. 2021**, R Soc Open Sci, doi 10.1098/rsos.201971 [F]: topologically, ketamine lies between wake and propofol. Neither paper analyses sleep.
- **Lord et al. 2023**, Front Netw Physiol 3:1279646, doi 10.3389/fnetp.2023.1279646 [P]. Kuramoto order per band in 1 monkey, from ketamine and propofol boluses. Global synchrony falls under both, most in beta and gamma (−42 to −66%), and local cluster synchrony is preserved. No sleep.
- **Toker et al. 2022**, PNAS 119:e2024455119, doi 10.1073/pnas.2024455119 [F]. Two NeuroTycho macaques and three humans under GABAergic anaesthesia. The modified 0-1 chaos statistic K of low-frequency activity rises (the signal becomes more chaotic) while LZc falls. The authors argue that Solovey's linear-stochastic assumption is contradicted by their nonlinearity tests (SI, not read). No ketamine and no sleep.
- **Fuentes et al. 2021**, bioRxiv, doi 10.1101/2021.08.17.456627 [P]. Permutation entropy and permutation LZ in 4 macaques. Both fall under medetomidine and propofol, rise under ketamine, and do not change under ketamine–medetomidine. Agents disagree in sign. Sleep is compared only by placement against published NREM values.
- **Medel et al. 2023**, Sci Rep 13:21700, doi 10.1038/s41598-023-47316-0 [F]. Under propofol, LZc falls and the 1/f slope steepens, co-localised across electrodes. Propofol only.
- **Deco et al. 2022**, Commun Biol 5:572, doi 10.1038/s42003-022-03505-7 [F]. INSIDEOUT time-irreversibility from lagged correlations. Irreversibility falls in deep sleep and under every anaesthetic except ketamine, so sleep is grouped with the drugs and ketamine is the exception.
- **Munn et al. 2023**, Nat Commun, doi 10.1038/s41467-023-42465-2 [F]. Uses NeuroTycho sleep–wake transitions (no NeuroTycho drug data) alongside human anaesthesia EEG; macaque sleep and human anaesthesia fall in one "unconscious" regime. **Li et al. 2023**, Nat Commun, doi 10.1038/s41467-023-38972-x: uses NeuroTycho sleep and anaesthesia for a cortical-hierarchy fluctuation signature; the macaque result was not extracted [S]. **Xie et al. 2023**, J Neural Eng, doi 10.1088/1741-2552/ace5db [F]: slow-wave modulation, propofol only. **Padovani** arXiv 2016–2021 [S]: functional-connectivity reductions under ketamine–medetomidine, weaker under propofol.

Answer to Q1: power, Granger causality, BLP correlation, VAR stability, Kuramoto synchrony, avalanches, cross-embedding, chaos, LZc/entropy, slope and irreversibility have all been applied. Every study that includes NeuroTycho sleep (Yanagawa, both Liu papers, Deco, Munn) groups sleep with anaesthesia. None in this set tests anaesthesia against sleep as a contrast, at matched depth or with a magnitude comparison. Note that Yanagawa, Liu, Solovey, Alonso and Tajima are all from the recording lab or its collaborators.

**Unresolved conflicts with our benchmark** (same archive; causes are inferred):
- Lord 2023 finds Kuramoto order falling under both ketamine and propofol, but our benchmark has Kuramoto moving ketamine opposite to propofol.
- Fuentes 2021 finds ketamine raising permutation LZ and ketamine–medetomidine unchanged, but our benchmark's LZ falls in all 10 drug cells.
- Candidate causes: band (our bank is low-passed at 40 Hz and run at 100 Hz, against their gamma to 80 Hz at 1 kHz); single bolus with an early post-LOC window against our session windows; metric variant (permutation against binarised LZ); and a single monkey.
- So "Δz is the only dependence measure with one sign across agents" holds for our bank, not for the literature (see §6).

## 2. Single metrics against anaesthesia and NREM sleep, across species

- **Propofol against NREM hdEEG** (Murphy et al. 2011, Sleep 34:283, doi 10.1093/sleep/34.3.283) [P]. Propofol slow-wave parameters match sleep slow waves, with no significant difference in most parameters, and the slow waves have the same cingulate and insular origins. But propofol slow waves involve more cortex per wave, group spindles little, and lack sleep's frontal asymmetry. Gamma power and cingulate phase synchrony rise at propofol LOC. Sleep came from 6 other subjects, so the comparison is between groups.
- **Dexmedetomidine against propofol** (Akeju et al. 2014, Anesthesiology, doi 10.1097/ALN.0000000000000419; Akeju et al. 2016, PLoS ONE 11:e0163431) [A]. Dexmedetomidine gives coherent ~13 Hz spindles and slow-delta, resembling sleep onset and N2. Propofol gives coherent frontal ~11 Hz alpha and, at general anaesthesia, much larger slow oscillations, so the two agents produce different states. Oral dexmedetomidine adds N2 sleep and reduces REM against placebo (Chamadia et al. 2020, doi 10.1097/ALN.0000000000003567) [P]. That is sleep architecture, not coupling.
- **BIS** (Nieuwenhuijs et al. 2002, Anesth Analg, doi 10.1097/00000539-200201000-00024) [A]. Median BIS is 42 in slow-wave sleep and 48 in REM, inside the 40–60 anaesthesia target, with wide overlap between stages. Sleigh 1999 reports slow-wave sleep at BIS 20–70 [S]. BIS rising under ketamine is [S] (the bench cites it unread).
- **Spectral entropy in natural sleep:** no primary source found [U].
- **Aperiodic slope** (Lendner et al. 2020, eLife, doi 10.7554/eLife.55092) [F]. The 30–45 Hz slope is steeper in propofol, NREM and REM than in wake. The datasets are separate and the paper offers no drug-against-sleep contrast. It groups the states; in our bank it fails on medetomidine.
- **LZc:**
  - falls in NREM N3 and returns to waking levels in REM (Schartner et al. 2017, Neurosci Conscious, doi 10.1093/nc/niw022) [P];
  - falls under propofol (Toker, Medel);
  - rises under sub-anaesthetic ketamine, LSD and psilocybin (Schartner et al. 2017, Sci Rep, doi 10.1038/srep46421) [A].
  It groups sleep with GABAergic drugs and splits the agents.
- **PCI** (Casali et al. 2013, Sci Transl Med, doi 10.1126/scitranslmed.3006294 [A]; Sarasso et al. 2015, Curr Biol, doi 10.1016/j.cub.2015.10.014 [A]). PCI is low in NREM and under midazolam, xenon and propofol, and wake-like under ketamine, with dream reports. It groups sleep with GABAergic agents and xenon and separates ketamine. PCI values were not read [U].
- **Slow and travelling waves:** Lewis et al. 2012 is [U] here (not re-read). An arXiv model paper (1810.00816) reports different slow-wave temporal structure in sleep and anaesthesia [S].

Answer to Q2: every metric found either groups NREM with anaesthesia (BIS, slope, LZc, PCI, Granger networks, BLP clusters, spectral transitions, irreversibility, modularity in §4) or splits the agents (LZc, PCI, BIS, power: ketamine opposite). The drug–sleep differences that are reported concern the form of the slow waves (spread, spindle grouping, asymmetry: Murphy) and agent-specific oscillations (Akeju). They are not a dependence measure consistent across agents. Where tested, sleep moves toward the drug (most, by design).

## 3. Linear against nonlinear dependence in state contrasts

- **Alonso et al. 2010**, Hum Brain Mapp, doi 10.1002/hbm.20881 [A]. Under alprazolam, a GABA-A benzodiazepine given during waking, cross-MI against surrogates shows linear coupling falling and nonlinear coupling rising, each correlated with plasma concentration. Alonso et al. 2012 (Psychopharmacology, doi 10.1007/s00213-011-2587-7) [A] replicates the opposite trends with cross-conditional entropy against coherence. This is the closest precedent: a drug-induced change in dependence character. It covers one drug class, it is a level change rather than an organisation change, and it has no sleep or anaesthesia arm.
- **Pereda et al. 2001**, Physica D 148:147 [S, abstract via mirror]. Two-channel state-space interdependence in 10 adults across vigilance states, against surrogates. It warns that these indices can falsely fall in deep sleep because each channel's complexity changes. This is the single-channel confound that joint IAAFT is meant to absorb. **Pereda et al. 2003**, Neurosci Lett, doi 10.1016/s0304-3940(02)01222-3 [A]: in neonates, quiet-sleep interdependence is mostly nonlinear and asymmetric, and larger than in wake or active sleep.
- **Imperatori et al. 2019**, Sci Rep 9:8894, doi 10.1038/s41598-019-45289-7 [P]. In 12 adults (hdEEG, wake against N3), wSMI falls everywhere while broadband wPLI does not change (delta wPLI rises in N3). The authors read wSMI as more sensitive to nonlinear interaction. So an information-theoretic measure and a linear measure dissociate in sleep.
- **Müller et al. 2020** (Hum Brain Mapp, doi 10.1002/hbm.24816) [A] and **Rummel et al. 2011** (Phys Rev E 83:066215, doi 10.1103/PhysRevE.83.066215) [A] are epilepsy papers, not state contrasts.
  - Rummel 2011: a surrogate framework separating dependence beyond linear correlation. Resting fMRI was linear; nonlinear iEEG dependence is mostly in epileptogenic tissue and seizures.
  - Müller 2020: in preictal iEEG, linear networks are local and reproducible across patients, while nonlinear networks are centralised and patient-specific.
  - They are methodological precedents for linear-surrogate dependence networks, not for anaesthesia.
- **Tajima 2015** and **Toker 2022** (§1) used nonlinear model-free tools under anaesthesia. Neither compared against a linear-Gaussian surrogate per pair in a state contrast; Toker's nonlinearity tests are in its SI [U]. Lee et al. 2013 (Anesthesiology 118:1264, doi 10.1097/ALN.0b013e31829103f5) [A] used normalised symbolic transfer entropy: frontal-to-parietal feedback falls under ketamine, propofol and sevoflurane. That is agent-consistent, but it is a directed-coupling level, not a linear/nonlinear contrast, and sleep was not tested.
- Searches for Gaussian-copula or Gaussian-against-kNN MI, or for per-pair linear-surrogate dependence, in anaesthesia found nothing [search-limited; absence not established].

Answer to Q3: dependence-character changes were reported before, for one GABAergic sedative (Alonso) and in sleep (Imperatori, Pereda 2003). So "absent in sleep" is not a general property of character measures. No paper found reports a dependence-character change that holds across agents with different receptor targets and is absent or small in sleep.

## 4. Kiyooka et al., Cell Reports (the supervisor's paper)

The link resolves to Kiyooka, Oomoto, Kitazono, Saito, Kobayashi, Matsubara, Kobayashi, Murayama, Oizumi, "Single-cell resolution functional networks during unconsciousness are segregated into spatially intermixed modules", Cell Rep, doi 10.1016/j.celrep.2025.116902. Europe PMC lists it as 2026 and gives no PMCID. Published abstract [A]. The bioRxiv v2 (doi 10.1101/2023.09.14.557838) was read through Methods, Results, Discussion and the supplementary legends [P]. The published version is unverified.

- **Data:** FASHIO-2PM two-photon imaging of about 10,000 layer-2/3 neurons over 3 × 3 mm of sensorimotor and higher cortex. NREM (EEG and EMG scored) in 5 mice; 0.6% isoflurane in 4 mice; separate sessions.
- **Network:** Pearson correlation of deconvolved traces, binarised at a fixed edge density K (0.008–0.3), with Louvain modularity.
- **Claims:** modularity is higher in NREM and isoflurane than awake at the single-cell scale, not at the mesoscale. Modules are spatially intermixed in every state. No direct sleep-against-anaesthesia comparison is made.
- **The supervisor's critique:** the text and legends I read report no mean pairwise correlation per state and no modularity null model (Louvain Q against the configuration expectation only). They do not discuss mean correlation or synchrony as a confound. Fixed density equalises the number of edges, not the correlation distribution. The per-state thresholds at fixed K exist only as figures (S4–S5), which I did not read. Under isoflurane only about 65% of neurons are active per window, against more than 95% awake and in sleep, so anaesthesia networks have fewer nodes.
- Whether mean coupling changed equally in sleep and anaesthesia is therefore **unverified**. The paper does not rule it out, which makes the critique stand as an unaddressed confound rather than a demonstrated one.

## 5. Psychedelics and other drugs

- **Measures:** this literature is dominated by band power and by signal diversity or entropy.
  - Psilocybin lowers broadband power (Muthukumaraswamy et al. 2013, J Neurosci, doi 10.1523/JNEUROSCI.2063-13.2013) [A].
  - LSD lowers visual alpha power (Carhart-Harris et al. 2016, PNAS, doi 10.1073/pnas.1518377113) [A].
  - LZc and entropy rise under psilocybin, LSD and sub-anaesthetic ketamine even after controlling for spectral change (Schartner et al. 2017) [A].
  - LSD lowers chaoticity slightly (Toker 2022) [F].
- **Coupling:** Granger causality falls under all three psychedelics, while undirected correlation and coherence rise under LSD (Barnett et al. 2020, NeuroImage, doi 10.1016/j.neuroimage.2019.116462) [A]. The authors read the opposite movement of directed and undirected measures as a property of the measures. That is the nearest thing to a dependence-character reading in this literature.
- **Gaussian against model-free dependence:** none found. Our own sub-anaesthetic ketamine sEEG test (Kauvar) did not detect Δz.

## (a) Prior metrics against what they separate

Each cell gives the direction of change. "Same subj." means drug and sleep were tested in the same subjects, "between groups" means in different subjects, and "not tested" means no comparison exists.

| Metric (source) | Wake vs sleep | Wake vs drug | Drug vs sleep | Agent vs agent |
|---|---|---|---|---|
| Spectral Granger networks (Yanagawa 2013) [F] | theta/gamma up, alpha/beta down | same, KTMD and PF | grouped by design (same animals, pooled) | KTMD ≈ PF |
| BLP-correlation clusters (Liu 2015 CC) [P] | preserved | preserved | grouped (same animals) | KTMD ≈ PF |
| Spectral transition sequence (Liu 2015 NI) [P] | present | present at PF induction | grouped (same animals); rebound difference [S] | PF only |
| Irreversibility (Deco 2022) [F] | falls | falls (MD, KTMD, PF) | grouped (same animals) | ketamine exception |
| VAR stability (Solovey 2015; Alonso 2019) [P/F] | not tested | more stable, KTMD and PF | not tested | same sign; surrogate dependence differs |
| Kuramoto order (Lord 2023) [P] | not tested | falls, KT and PF | not tested | same sign; **conflicts with bench** |
| Cross-embedding complexity (Tajima 2015) [P] | not tested | falls, KTMD and PF | not tested | same sign |
| Avalanches (Varley 2020) [P] | not tested | PF strong, KT weak | not tested | same sign, unequal |
| 0-1 chaos K (Toker 2022) [F] | not tested | rises (PF) | not tested | PF only |
| Permutation LZ/PE (Fuentes 2021) [P] | lit. placement only | MD, PF down; KT up | not tested | **opposite**; conflicts with bench |
| NSTE feedback (Lee 2013, human) [A] | not tested | falls, ketamine, PF and sevo | not tested | same sign |
| Slow-wave form (Murphy 2011, human) [P] | slow waves | similar SWs, wider, spindles ungrouped | differs in spread and spindle grouping (between groups) | PF only |
| Spindles vs frontal alpha (Akeju 2014/16) [A] | N2 spindles | dexmed spindles; PF alpha | dexmed ≈ sleep onset (between groups) | dexmed ≠ PF |
| BIS (Nieuwenhuijs 2002) [A] | SWS 42 | 40–60 target | grouped (between groups) | ketamine rise [S] |
| 1/f slope (Lendner 2020) [F] | steeper, NREM and REM | steeper (PF) | grouped (between datasets) | PF only; our bank: fails MD |
| LZc (Schartner 2017 a/b) [P/A] | falls in NREM | falls (PF); rises (sub-anaesthetic KT) | grouped (between studies) | opposite |
| PCI (Casali 2013; Sarasso 2015) [A] | low in NREM | low with midazolam, xenon, PF; high with KT | grouped (between groups) | KT separated |
| wSMI vs wPLI (Imperatori 2019) [P] | wSMI down, wPLI flat | not tested | not tested | — |
| Linear vs nonlinear cross-MI (Alonso 2010/12) [A] | not tested | alprazolam: linear down, nonlinear up | not tested | one agent |
| Single-cell modularity (Kiyooka 2025) [P] | rises (NREM) | rises (isoflurane) | grouped (separate sessions); mean correlation not reported | isoflurane only |
| **Δz (this repo)** | −0.24 SD (7 sessions, 2 animals) | −3.3 SD; 23/23 sessions, t(3) −6.1 | sleep/drug ratio .07; bench, same animals | one sign, KT/KTMD/MD/PF |
| Spectral entropy, LZ (bench, this repo) | moves .18–.33 of the way | all agents | AUC ≈ 1.0, better than Δz | one sign |

## (b) Gap statement

- **Established (read):** every prior metric that was tested on natural sleep groups it with anaesthesia. This includes the NeuroTycho Granger, BLP-cluster, spectral-transition and irreversibility analyses in the same animals. Several dependence or dynamics measures move one way across agents with different receptor targets: VAR stability, Kuramoto order, cross-embedding complexity and NSTE feedback. None of those was tested on sleep.
- **Inferred:** the conjunction Δz shows, one sign across GABAergic, NMDA and α2 agents and small (about .07 of the drug size) in sleep, is not reported for any metric found. Its novelty rests on the competing measures never being tested on sleep, not on evidence that they move in sleep. It is also not the first dependence-character change: there are single-agent (alprazolam) and sleep (wSMI against wPLI) precedents. What is new is the Gaussian-against-model-free comparison against a joint linear surrogate, read as organisation across edges.
- **Limits carried from in-house work:** sleep is 2 animals; in George, delta-matched drug windows equal sleep and Δz is .44 at matched slow-wave share; spectral entropy and LZ classify drug against sleep better (AUC ≈ 1.0 against .93/.82). The defensible claim is specificity and orthogonality to arousal, not better separation.
- **Not verified:** Lewis 2012, Sleigh 1999, PCI values, spectral entropy in sleep, Toker's SI nonlinearity tests, the Kiyooka published version, and the claim that ketamine raises BIS.

## (c) Falsifiers: prior metrics that could already have Δz's profile

1. **VAR(1) near-critical eigenmode fraction** (Solovey 2015; Alonso 2019). It is agent-consistent for KTMD and PF on the same archive, and sleep is untested. If NeuroTycho sleep does not stabilise, it shares the profile. This is testable in-house on the same windows and is the most direct falsifier.
2. **Global Kuramoto order, 30–80 Hz** (Lord 2023). It falls under both KT and PF in one monkey, which contradicts our band-limited bench. It needs a re-run at 1 kHz with gamma, and on sleep.
3. **Cross-embedding complexity** (Tajima 2015). It is model-free and nonlinear, falls under KTMD and PF, and sleep is untested. It is the closest conceptual neighbour, though it measures a within-pair level rather than organisation across edges.
4. **NSTE frontal-to-parietal feedback** (Lee 2013). It is human, consistent across ketamine, propofol and sevoflurane, and sleep was not searched exhaustively here. A sleep result would decide it.
5. **wSMI** (Imperatori 2019; King/Sitt lineage). It moves in N3, so it fails "absent in sleep" if used as a level. A wSMI-against-wPLI organisation statistic across agents is untested and could share Δz's profile.
6. **Already failed in the literature:** irreversibility (sleep moves; ketamine exception), PCI, LZc and BIS (sleep grouped or agents split), slope (sleep grouped), Granger and BLP networks (sleep grouped by design, magnitude untested), and Kiyooka modularity (sleep grouped).
7. **Already failed in-house (bench):** every conventional coupling summary (agents split) and the univariate indices (move sleep .18–.33 of the way).
