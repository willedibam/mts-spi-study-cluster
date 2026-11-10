# Consciousness atlas: run design and pre-registration draft (11 November 2026)

Status: **draft, revised after two reviews (Fable 5.2 critique of the first draft; an external review of the framing); not frozen; nothing extracted.** The dataset search and prior art are in the [scoping note](../representation_state/notes/real-world-application-scoping.md) (application 1). This file is the run architecture and the analysis to freeze before any atlas pyspi output is seen. Labels: **[V]** checked against the file, API or paper this session; **[E]** established result; **[R]** recorded in the repository; **[I]** inferred; **[U]** unverified; **[S]** speculation.

## Question and success contract

**Question.** Which aspects of dependence organisation are conserved across changes in arousal, responsiveness and reported experience, and which depend on the intervention, species or recording scale? A single wake-to-unconscious axis is one hypothesis among several, not the organising assumption.

**What counts as a result.** Not "z beats the controls on a shared axis". The target is two or three named organisational relationships: for example, "pairs that agree under the precision family stop agreeing under the rank family". For each, the result states where it transfers, where it breaks, and its signal-level explanation, spectral or beyond. The non-redundancy bar stays: a relationship that pooled or per-channel spectra also give is reported as spectral, and that is still a finding.

| Outcome | Value |
|---|---|
| A replicated organisational change across drugs and species, with a validated signal-level explanation | a transferable description of a recurring dynamical change |
| Shared change under propofol, isoflurane and NREM, with a reproducible divergence under ketamine or REM | separates forms of reduced responsiveness |
| Organisation associated with experience reports within a fixed sleep stage | most directly about consciousness, within the limits of report and recall |
| Distinct, reproducible trajectories per state class | evidence against one common trajectory; a taxonomy |
| Structure that mostly reflects dataset or recording type | the null outcome: descriptive only |

**Supervision.** z is unsupervised. Every contrast direction below is estimated from state labels, and the text will say so.

**Precedent the result must go beyond.** Bola et al. 2018 found gamma-envelope hyper-correlation across macaque anaesthesia and human NREM [V, abstract]. Leung et al. 2025 found univariate autocorrelation features that generalise across flies [V]. Engemann et al. 2018 classified disorders of consciousness across electrode configurations [V, abstract]. Identity-free comparators exist (pooled spectra, edge distributions, graph summaries); z's specific offer is its description of how dependence measures correspond across pairs, without channel alignment.

## Hypotheses (to freeze)

- **H-within.** Wake and deep GABAergic or N3 states differ in z within every dataset. This is the only outcome treated as near certain.
- **H1, decoupling (formerly P1).** The precision-family edge profile decouples from the rank and ids family profile under propofol, isoflurane and NREM, and less under ketamine and REM. This is a statement about statistics: partial correlation is conditional association given the observed channels [E], so it changes with montage and is not "direct connectivity".
  - It is defined on shrinkage-precision SPIs only.
  - It requires a reliability floor: within-state split-half reliability of the SPIs involved, because about two-thirds of NeuroTycho's raw mediation effect was a reliability drop [R, `character_axes/findings.md`].
  - NeuroTycho, and Chennu (analysed with z before at 19 channels [R]), are discovery sets. H1 is tested on the others.
- **H2, conservation and specificity.** State classes share part of their Δz and differ on another part. The specificity matrix (below) estimates both, with no single shared direction assumed.
- **H3, non-redundancy.** On the conserved part, z adds to the joint identity-free control block (below). A paired interval that includes zero means no demonstrated advantage, not falsification. A tie is claimed only within a pre-registered equivalence margin.
- **H4, experience.** Within a sleep stage, pre-awakening z separates experience from no-experience reports beyond pooled and posterior spectra. Tested on suitable DREAM constituent studies. Bajwa is exploratory only: 24 experience, 5 no-experience and 23 no-information awakenings [R], and repeated awakenings are not independent units.
- **Surrogates, both readings stated in advance.** An effect reproduced by multivariate IAAFT is a linear-spectral finding: explained, not void. A "beyond-spectral" claim needs the cross-spectrum-preserving null. Independent-channel surrogates answer a different question (whether any dependence exists).

## Datasets: sizes, structure and role

Raw is what must be downloaded or streamed. Windows are per arm, under the caps below. "Link" is the comparison a dataset connects, since species, modality, laboratory and intervention must not be confounded.

| Dataset | Species, signal | States (units) | Native M, reference | Source fs, structure | Raw | Arms | Windows / arm | Role, link |
|---|---|---|---|---|---|---|---|---|
| Fly (Leung 2025) | fly, LFP | wake vs isoflurane (13 flies; 56 `validate1` series, labels [U]) | 15 bipolar [V] | 1 kHz, 8 × 2.25 s per fly-condition [V] | 52 + 177 MB [V] | 2-s | 656 | isoflurane, invertebrate; continuous and sleep sets requested in-group |
| NeuroTycho | macaque, ECoG | wake, ketamine–medetomidine, propofol (4 animals); natural sleep in the same arrays to check [U] | 128 | 1 kHz continuous | 128-ch raw to re-stage (only 16-ch views on Gadi [V]) | 10-s, 2-s, 200 Hz | 240 | discovery; two drugs in one montage, plus sleep if available |
| Chennu 2016 | human, scalp EEG | baseline, mild, moderate, recovery (20) | 91 | 250 Hz, about 7 min per level [R] | few GB [U] | 10-s, 2-s | 1,200 | propofol, graded, with recovery; discovery-adjacent |
| Bajwa 2025 (ds005620) | human, scalp EEG | awake vs propofol sedation (20) | 65 incl. non-EEG | 5 kHz [R] | [U] | 10-s, 2-s | about 900 | propofol, second human set; H4 exploratory |
| ANPHY-Sleep (OSF r26fh) | human, scalp EEG | W, N2, N3, REM (29) | 83 | 1 kHz, about 7 h, EDF [V] | about 4.7 GB per person [I] | 10-s, 2-s | 1,740 | NREM and REM in human scalp: links sleep to propofol within one modality |
| MNI iEEG sleep atlas | human, sEEG/ECoG | resting W, N2, N3, REM (about 106 patients [U]) | about 17 per patient [U] | 200 Hz, 0.5–80 Hz, 60-s epochs per stage [V] | about 1 GB [U] | 10-s, 2-s, 200 Hz | 2,544 | human intracranial sleep: links scalp to intracranial |
| Watson 2016 (DANDI 000041) | rat, frontal probe LFP | wake, NREM, REM (10 rats, 22 sessions [V]) | silicon probe; thinning pending (decision 2) | [U] | 149 GB [V]; stream LFP only | 10-s, 2-s, 200 Hz | 900 | rodent sleep |
| Claar 2023 (DANDI 000458) | mouse, skull EEG | awake vs isoflurane, some recovery (23 [V]) | 30 EEG [V] | [U] | 361 GB [V]; stream EEG only | 10-s, 2-s, 200 Hz | 1,380 | isoflurane across species (fly, mouse). Cortical stimulation is delivered in both states [V], so windows come only from gaps between stimuli, or the dataset is excluded |
| Kauvar 2025 (DANDI 001347) | human, sEEG | baseline, ketamine, post | about 100 [U] | [U] | 16 GB [V] | 10-s, 2-s, 200 Hz | 450 | **deferred until the dose is known**: DANDI does not state it [V], and a subanaesthetic dose makes "ketamine nearer wake" trivial |
| DREAM (doi 10.26180/22133105) | human sleep EEG/MEG, 20 studies | experience, experience without recall, no experience; 2,643 awakenings, 505 people [V] | per study; inclusion needs only ≥ 2 [V] | ≥ 100 Hz, ≥ 20 s before awakening, EDF [V] | open constituent packages only; others by request [V] | 10-s, 2-s | counted per study before budgeting | H4; studies with M ≥ 8 only (two channels give one edge, so z is undefined) |
| DOC (figshare 23552964) | human, scalp EEG | patient vs control (98 headers, 96 data files [V]); UWS/MCS labels not in deposit | 62 [V] | 250 Hz, 60 non-contiguous 2-s segments [V] | 714 MB [V] | 2-s | ≤ 2,880 | projection only: no own-wake baseline |
| Purdon (DANDI 001285) | human, scalp EEG | graded propofol | 64 [V] | 250 Hz, 154 min [V] | 1.2 GB [V] | — | 0 | **excluded**: third-party repackaging with unnamed channels and no drug-level or LOC/ROC annotations [V]; original data worth requesting |

Requests still worth sending, ranked by the links they add:
1. Michigan anaesthetic-dose ketamine EEG: ketamine against propofol in one modality.
2. Raw LFP from Bastos/Miller and Redinbaugh/Saalmann: thalamic stimulation with the drug held constant.
3. Fly continuous and sleep recordings (in-group).
4. Purdon original data with annotations.
5. Bajwa report labels (exploratory only).

## Preprocessing (`atlas/prepare.py`, to write)

1. **Channels.** Drop non-neural channels. Fix one channel set per unit, identical across that unit's states. Remove bad channels by fixed whole-unit rules (flat: SD below 1% of the median channel SD; noisy: above 5 times). Otherwise native M, with no PCA, ICA or source reduction.
2. **Reference, without rank loss.**
   - Average or common-average referencing makes the covariance rank M−1 exactly [E], so precision SPIs become pseudo-inverse artefacts. After such a reference, drop one fixed channel (the last in the montage order, recorded).
   - sEEG: adjacent-contact bipolar within each shaft.
   - Fly: bipolar as released. MNI: as released.
   - Every window logs covariance rank and condition number.
3. **10-s and 200 Hz arms.**
   - High-pass at 0.5 Hz (zero-phase); notch at mains and harmonics.
   - Zero-phase FIR low-pass at 40 Hz (100 Hz arm) or 80 Hz (200 Hz arm), then polyphase resampling.
   - Windows are non-overlapping, evenly spaced through each stable period and deterministic.
   - Excluded: drug transitions (first 5 min after an infusion change [I]); sleep epochs whose neighbours are in another stage; artefacts (a channel with more than 5% of samples beyond 6 robust SD of its unit-level distribution, a flat run of 1 s or more, or non-finite values).
4. **2-s arm, identical for every dataset.**
   - Every source is first cut into 2-s epochs (the central 2 s of the fly's 2.25-s epochs).
   - Each epoch gets the same treatment: linear detrend, 1 Hz zero-phase high-pass with reflection padding, 40 Hz low-pass and resampling to 100 Hz (T = 200).
   - This arm cannot see the slow oscillation below 1 Hz, and says so.
   - At M/T above a frozen ratio (proposed 0.25), the precision family is excluded from this arm's coordinate set.
5. **Normalisation.** pyspi per-channel z-score per window (`normalise=True`). Controls are computed from the identical arrays.
6. **Output.** One bank per dataset and arm (`windows.npz` plus `manifest.json`), as [`extract.py`](extract.py) expects.

## Rates and windows

- **Common rate, not native rates with physical-unit bands.** pyspi's spectral SPIs accept `fs`, `fmin` and `fmax` [V, installed `pyspi/statistics/spectral.py`]. But about 45 lag, history and model-order SPIs count samples, so native-rate runs would still mix physics across datasets outside the spectral family. A bespoke config would also break the single configuration shared with the paper's other sections. A common resample fixes both, and removes no dependence structure inside the band it keeps. Comparison in units of an intrinsic timescale (cycles of each species' dominant rhythm) is a separate, later analysis, mainly for the fly. The policy is fixed across states, so the state-dependent slowing under study is not normalised away.
- **Primary arm: 10 s at 100 Hz (T = 1000).** The ceiling sits near 40 Hz, set by scalp EMG and mains. The band split falls at 25 Hz and one lag is 10 ms. Ten seconds holds 5 cycles at 0.5 Hz and fits inside a 30-s sleep epoch.
- **2-s arm (T = 200), all datasets.** The only arm that admits the fly and DOC. Cheap. z at 2 s is never compared with z at 10 s.
- **200 Hz arm (T = 1000, 5 s), intracranial, secondary.** It keeps gamma to 80 Hz. T is matched within the arm, which keeps estimator bias equal across datasets. No state class has three confirmatory datasets in this tier, so it is supporting evidence, run after the primary.
- **No overlap; units carry power.** Caps per unit-state: 30 windows for M ≤ 32, 15 for M ≤ 96, 10 above. z from thousands of pairs is precise per window, so further windows only sample state fluctuation. Pre-registered adaptive rule: if the split-half reliability of a dataset's unit Δz is below 0.8 at its cap, the cap doubles for that dataset. This is decided from wake-only split halves, before any state contrast is looked at.

## Extraction and compute

- **pyspi.** Upstream main `0c798b4` (r13, unchanged since the snapshot; the head is re-checked at launch), loaded through `PYSPI_SOURCE`. Config `benchmarked_p90` from that commit; its SPI order differs from K = 283/289, so K and the SPI family lists ("precision", "rank and ids", "directed", "spectral") are frozen from the profiling run before analysis. Full channel space: no dyad sampling (user).
- **Runner.** [`extract.py`](extract.py) calls `run_pyspi` with the same arguments as `run_corpus`, a fixed seed per window and z-scoring. It writes one archive per shard, with provenance, resume and audit; tests pass.
  - **Shard size scales with M**, so that a shard takes about 1–2 h: 25 windows for M ≤ 32, 5 for M ≤ 96, 2 for M > 96. A 25-window shard at M = 128 would take about 17 h and miss the walltime.
- **Concurrency.** pyspi's `n_jobs` (parallel over SPIs) is used wherever memory already pays for the cores: 1 worker for M ≤ 32, 2 for M ≤ 64, 4 for M of about 83–128 at T = 1000. It shortens wall time at roughly equal SU.
- **Farms.** [`farm.pbs`](jobs/farm.pbs), one job per dataset and arm, all submitted together. Nodes are sized so that nodes × 48 ≤ shards × workers per window (the request is billed whether used or not). The largest jobs span 8–16 nodes.
- **Order.**
  1. Smoke: two windows per bank.
  2. Profiling: one window per (M, T) class, measuring time, memory and K on upstream r13. The HPC.md timings are from the fork, so every figure here is re-priced.
  3. The 2-s arm everywhere.
  4. The 10-s/100 Hz arm.
  5. Surrogates and the 200 Hz arm.
  6. DREAM, once its usable studies are counted.
- **Storage.**
  - Outputs: about 230 GB in float64.
  - float32 storage of MPIs is adopted only if profiling shows a maximum |Δz| between float64 and float32 below 1e-4. That needs a one-line change to `extract.py`.
  - Raw for ANPHY, Watson and Claar: download or stream, extract windows, delete, one subject at a time on copyq.
  - Scratch is at 223 GiB of 1 TiB.

## Analysis (to freeze)

1. **Coordinates.**
   - z per window under the ordered-pair contract, extended to the new K. Fisher-transformed with |r| clipped at 0.995.
   - Each coordinate is standardised by its within-state window SD in the training data.
   - Unit-state means, then Δz = state − the unit's own wake baseline.
   - A family-block version (block means over the frozen SPI families) is co-primary with the full coordinates.
2. **Primary statistic: standardised held-out projection.** The held-out dataset's unit Δz are projected onto the direction from all other datasets, giving a unit-level paired effect size. Disattenuated cosine (split-half corrected) is secondary. Raw cosine is not used across representations of different dimension.
3. **Weights and inference.** Datasets are weighted equally, with precision weights only within a dataset. Sign flips are done at unit level, never on windows. Inference runs over datasets, leave-one-dataset-out and leave-one-species-out.
4. **Specificity matrix.** State classes (GABAergic or isoflurane, NREM, REM, ketamine) cross state classes. Shared and specific components are estimated, against a null contrast (early against late wake in the same units) that measures generic change and drift.
5. **Recovery reversal.** Recovery returning towards wake (Chennu, Claar, NeuroTycho) is the time and drift control, because anaesthesia always follows wake.
6. **Controls through the identical pipeline.**
   - Mean-SPI, Pearson edge quantiles, pooled log spectra (1–40 Hz, or 1–80 Hz in the 200 Hz arm), across-channel quantiles of band power, pooled catch22.
   - Each control gets its own null and uncertainty.
   - Within fixed-montage datasets, the per-channel spectral control (mandatory per the project context) and an identity-aware edge-level ceiling are added.
7. **Residual.** Δz is residualised on the joint identity-free block (ridge, fitted leave-one-dataset-out), with a kernel-ridge sensitivity. The decisive version is Δz(data) − Δz(multivariate IAAFT) on the surrogate subset (about 5 windows per unit-state, plus independent-channel surrogates).
8. **Montage robustness, at zero SU.** z recomputed from stored MPIs restricted to random 15-channel subsets in every dataset. This is exact for bivariate SPIs; conditional SPIs are excluded from the check. MNI's between-patient variation in M adds a natural test. Reference choice is varied on one dataset per modality. The old log-M regression is dropped: M is constant within a dataset and collinear with species across datasets.
9. **H4 on DREAM.** Per constituent study with M ≥ 8: within-person paired experience against no-experience contrast, by stage. Covariates are time of night and stage. Pooled and posterior spectra are the controls.
10. **Projection.** DOC (patients against controls) onto the learned components, without estimating them.
11. **Losses are kept.** Every arm, control and failed transfer is reported.

## Budget

`ql44` Q4 allocation rises from 75 to 105 kSU (pending at NCI, 11 November). With 46.8 kSU used [V], about 58 kSU is free, shared with other sessions. Estimates follow HPC.md's rule [I]. They are ±1.5 times, and the timings come from the fork, so they will be re-priced from the profiling run.

| Block | kSU |
|---|---:|
| 2-s arm, all datasets | 2.4 |
| 10-s / 100 Hz arm | 13.5 (Kauvar 1.8 of it, deferred) |
| 200 Hz arm at T = 1000 | 4.1 |
| Surrogates (multivariate IAAFT and independent-channel, about 5 windows per unit-state, 10-s arm) | 10.4 |
| DREAM (usable studies uncounted) | about 3 [S] |
| Profiling, smoke and reruns (20%) | about 7 |
| **Total** | **about 40 (range about 27–60)** |

Doubling caps under the adaptive rule would add up to about 15 kSU, at most.

## Predictions before any atlas output (all [S])

| Outcome | Ours | Fable 5.2 |
|---|---|---|
| Wake against deep state separable within each dataset | near certain | about 97% |
| Held-out projection above the unit-level null for GABAergic/isoflurane/NREM | about 75% | about 90% against the old weak null |
| Spectral or joint identity-free control equals or beats z on the conserved part | about 65% | about 85% |
| A residual component survives leave-one-dataset-out | about 35% | about 20% genuine; about 45% chance the original test gives a spurious positive |
| Fly to mammal | about 25% | about 7% |
| H1 decoupling recurs outside the discovery sets beyond IAAFT | about 40% | about 15–20% |
| Ketamine and REM nearer wake on the residual | about 40% | about 15% |
| H4 on DREAM beyond spectra | about 15% | about 7% |

The likeliest single outcome is a shared, largely spectral change across GABAergic, isoflurane and NREM states, with z non-redundant only in places. Under this contract, that is still worth reporting if the places are named and explained.

## Decisions open before freeze

1. Budget: about 40 kSU this quarter, released in the order above.
2. Watson's silicon probes: thin to sites at least 100 µm apart, or keep every site. Closely spaced sites are near-duplicates and make the precision SPIs singular. Fable's view is that hardware pitch is not a montage.
3. Whether NeuroTycho natural-sleep sessions exist in the same animals [U]. The anaesthesia sessions belong to collections titled "anesthesia and sleep" (for example `20110622ktmdanesthesiaandsleepchibi`) [V, bank names on Gadi], but the earlier catalogue audit found no sleep sessions [R]; re-check the catalogue. If they do, they become the cleanest test of whether a cross-state direction exists at all, with montage held fixed.
