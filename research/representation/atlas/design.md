# Consciousness atlas: run design and pre-registration draft (11 November 2026)

Status: **draft, revised after three review rounds** (Fable 5.2 critique of the first draft, an external review of the framing, and Fable on the hypothesis and a minimal launch). Not frozen; nothing extracted. The dataset search and prior art are in the [scoping note](../representation_state/notes/real-world-application-scoping.md) (application 1). This file is the run architecture and the analysis to freeze. Labels: **[V]** checked against the file, API or paper this session; **[E]** established result; **[R]** recorded in the repository; **[I]** inferred; **[U]** unverified; **[S]** speculation.

## Question and success contract

**Question.** Which aspects of dependence organisation are conserved across changes in arousal, responsiveness and reported experience, and which depend on the intervention, species or recording scale? A single wake-to-unconscious axis is one hypothesis among several, not the organising assumption.

**What counts as a result.** At most three named, frozen statements about how dependence estimators agree across channel pairs. Each comes with where it transfers, where it breaks, and its signal-level explanation. A statement that covariance rank, pooled spectra or strength also give is reported as such, described in z's vocabulary.

| Outcome | Value |
|---|---|
| A frozen statement, sign-consistent across three or more independent datasets and two species, separating propofol, medetomidine and deep sleep from ketamine and REM, and not predicted by covariance rank plus spectra | the best realistic result (about 10–15% [S]) |
| The NeuroTycho five-state taxonomy in identical arrays (which drug's reorganisation resembles sleep), with the sleep and propofol parts replicated in humans | a good modest result |
| Shared change under GABAergic agents and NREM, with divergence under ketamine or REM | separates forms of reduced responsiveness |
| Distinct, reproducible trajectories per state class | evidence against one common trajectory; a taxonomy |
| Structure that mostly reflects dataset or recording type | the null outcome: descriptive only |

**Discovery, then confirmation.** NeuroTycho, and Chennu (analysed with z before at 19 channels [R]), are open discovery sets: anything may be looked at there. At most three statements are then frozen and tested on untouched sets (Bajwa, ANPHY, MNI). Every frozen statement is reported, failures included, with the number of summaries examined in discovery.

**Supervision.** z is unsupervised; every contrast is estimated from state labels.

**Precedent to go beyond.**
- Bola et al. 2018: gamma-envelope hyper-correlation across macaque anaesthesia and human NREM [V, abstract].
- Leung et al. 2025: univariate autocorrelation features that generalise across flies [V].
- Engemann et al. 2018: disorders of consciousness classified across electrode configurations [V, abstract].
- Casali et al. 2013: perturbational complexity [U, not re-read].

Identity-free comparators exist; z's specific offer is how dependence measures correspond across pairs, without channel alignment.

## Why z might see something

- **What z measures.** z records whether different dependence estimators order the channel pairs the same way. If one latent factor orders the pairs, every estimator sensitive to it agrees, and z concentrates on few components. If pairs are coupled in several distinct ways (band-specific, lagged, nonlinear, conditional), several factors order them, and z spreads out [I].
- **Why states should differ.** Slow-wave bistability under GABAergic agents, α2-agonists and NREM imposes a global slow mode. This is the integration-without-differentiation picture behind perturbational complexity [U]. In z, wake should show many kinds of pairwise relationship and deep unconsciousness fewer, with conditional (precision) statistics detaching from the pairwise block once a shared mode explains most marginal dependence. That detachment is what NeuroTycho already shows: MinCovDet partial r² against rank and kernel dependence, z .72–.85 awake against .46–.54 anaesthetised, in 6 of 6 units [R]. Ketamine preserves complexity [U], so it should differ.
- **What could explain it trivially, each with a control.**
  - The spatial dimensionality of the signals themselves: the participation ratio of the channel covariance. This is the comparator most likely to absorb the effect.
  - Spectral concentration: pooled spectral entropy and slow-power fraction.
  - Strength and range restriction across pairs.
  - Edge-profile reliability, which rises under anaesthesia in NeuroTycho [R] and raises agreement with no change in organisation.
  - Volume conduction and reference.
  - The rank bound (K and M(M−1)), so effective rank is compared only within unit.
- **The counter-evidence.** Unsupervised z-PC1 separated animals, not states, in NeuroTycho, while mean-SPI separated states [R]. A state change in z's spread is likely; surviving the controls is not (about 20–25% [S]).

## Frozen summaries (primary) and hypotheses

The 41,000-dimension direction analysis is secondary. The primary objects are three scalars per window, computed from z alone at no extra cost:

1. **Effective rank of z**: the participation ratio of its eigenvalues.
2. **Mean between-family agreement**: the mean z between the frozen SPI families.
3. **The precision–rank block**: mean z between the shrinkage and robust precision SPIs (MinCovDet, Ledoit–Wolf and the other retained estimators, frozen from the profiling run) and the rank and ids SPIs.

Hypotheses, each compared within unit, with unit-level effect sizes and intervals:

- **H-within.** Wake and deep GABAergic or NREM states differ in z within each dataset. The only near-certain outcome.
- **H-rank.** Effective rank falls from wake to propofol, medetomidine and deep sleep, and less or not at all under ketamine alone. The sign is pre-registered as falling, though the precision block detaching pushes the other way, so the result is uncertain [S].
- **H1, decoupling.** The precision–rank block falls under GABAergic agents, medetomidine and NREM, and less under ketamine and REM. This is a statement about statistics: partial correlation is conditional association given the observed channels [E], not direct connectivity. It holds only where the SPIs involved pass a within-state split-half reliability floor (0.8).
- **H-control.** Each effect is tested again after regressing, within unit, on covariance participation ratio, pooled spectral entropy, slow-power fraction, mean |Pearson| and edge reliability. Surviving these is the non-redundancy claim.
- **H4, experience.** Deferred until a statement exists. DREAM (constituent studies with M ≥ 8) then tests it within a stage, and Bajwa labels if they arrive (exploratory).

No "beyond-spectral" wording is available from this launch, because surrogates are deferred (user, 11 November). They run later, on whichever statements emerge.

## Datasets in this launch

Two steps. Step 2 is released only after step 1's statements are frozen.

| Dataset | What it is | States, units | Native M (after reference) | Source fs, structure | Raw | Resampling and windows | What it buys |
|---|---|---|---|---|---|---|---|
| **Step 1: NeuroTycho** (Yanagawa et al. 2013) | macaque, 128-contact ECoG over one hemisphere | Chibi and George: natural sleep (4 + 3 archives), ketamine alone (2 + 2), medetomidine alone (2 + 2), ketamine–medetomidine (2 + 3), propofol (2 + 2); Kin2 and Su: ketamine–medetomidine (3 + 3) [V, live catalogue]. Each archive has awake eyes-closed and eyes-open intervals [V] | 127 (common average, one channel dropped) | 1 kHz continuous, intervals of about 10–45 min [V] | 30 archives, 41 GB [V]; staged one at a time | 1 kHz to 100 Hz (10×); 10 per archive-state for awake eyes-closed, awake eyes-open, drug and recovery eyes-closed, 30 spread across the unstaged sleep intervals; plus 2-s, cut centred in each 10-s window. Sleep archives carry chin EMG and eye EOG [V], whose per-window RMS is recorded, so REM-like windows can be separated without the ECoG | discovery and linking: five state classes in one montage; separates ketamine from medetomidine; eyes-open against eyes-closed is the built-in null contrast |
| **Step 2 if access is granted: MNI iEEG sleep atlas** (Frauscher, von Ellenrieder); its site now returns HTTP 401 and the multicentre successor sits behind a LORIS login [V, 11 November], so it needs registration | human sEEG and ECoG, normal-appearing channels | resting wake, N2, N3, REM; about 106 patients [U] | about 17 per patient [U], as released | 200 Hz, 0.5–80 Hz, 60-s epochs per stage [V] | about 1 GB [U] | 200 to 100 Hz (2×); six 10-s windows per 60-s epoch; plus 2-s | staged human sleep at intracranial scale, almost free |
| **Step 1: Fly** (Leung et al. 2025) | fly, 16-contact linear probe LFP | wake against isoflurane; 13 flies (and 56 unlabelled `validate1` series [V]) | 15 bipolar [V] | 1 kHz, 8 × 2.25-s epochs per fly-condition [V] | 229 MB [V] | 1 kHz to 100 Hz; 2-s arm only (8 per fly-condition) | species extreme; descriptive, because drug, species and scale are confounded |
| **Step 2: Bajwa** (OpenNeuro ds005620) | human 64-channel scalp EEG | awake against propofol sedation, 20 people | about 62 (average reference, one dropped) | 5 kHz [R] | [U] | 5 kHz to 100 Hz (50×); 15 per person-state; plus 2-s | the untouched human propofol confirmation |
| **Step 2: ANPHY-Sleep** (Wei et al. 2024) | human 83-channel overnight scalp EEG | W, N3, REM (N2 dropped), 29 people | 82 | 1 kHz, about 7 h, EDF [V] | about 4.7 GB per night [I]; download, window, delete | 1 kHz to 100 Hz (10×); 15 per person-stage, inside stably scored epochs; plus 2-s | the untouched staged human sleep, including REM |
| **Step 2: Chennu et al. 2016** | human 91-channel scalp EEG | baseline, mild and moderate propofol, recovery; 20 people | 90 | 250 Hz, about 7 min per level [R] | few GB [U] | 250 to 100 Hz (2.5×); 15 per person-level; plus 2-s | graded propofol with recovery (the reversal test for drift); discovery-tainted |

**Deferred, not dropped** (all can be added later at the same pin):
- Kauvar (ketamine dose unknown).
- Watson (probe pitch; channel set undecided).
- Claar (cortical stimulation in both states [V]; usable gaps unverified).
- DOC (no baseline, no labels in the deposit).
- DREAM (usable n unknown; worth it once a statement exists).
- The 200 Hz arm (NeuroTycho ketamine gamma is its one strong rationale, about 5 kSU).
- Surrogates.

**Excluded:** Purdon's DANDI deposit (no channel names, no state annotations [V]).

## Why these M, T, window and resampling choices

- **Native M, full channel space.** z's dimension is fixed by the SPI set, so every channel can contribute, and cutting to a common M would discard what large montages carry. The costs are handled by comparing scalars only within unit, and by a zero-SU check that recomputes z from stored MPIs restricted to random 15-channel subsets (exact for bivariate SPIs). No PCA or ICA: components are uncorrelated mixtures, which would distort the covariance-family SPIs.
- **Reference without rank loss.** Average and common-average references make the covariance rank M−1 [E], which turns precision SPIs into pseudo-inverse artefacts. One fixed channel is dropped after re-referencing, and every window logs covariance rank and condition number.
- **One common rate, 100 Hz.** In pyspi, bands are fractions of fs and about 45 SPIs count lags in samples [V]. Native rates would make the same SPI measure different physics in different datasets. Setting bands in Hz through pyspi's `fs`/`fmin`/`fmax` would fix only the spectral family. 100 Hz puts the ceiling at about 40 Hz (scalp EMG and mains set this), the band split at 25 Hz and one lag at 10 ms. Resampling removes nothing inside the band it keeps. The filter chain is a 0.5 Hz high-pass, notch, 40 Hz zero-phase FIR low-pass, then polyphase resampling.
- **10-s windows (T = 1000).** They hold 5 cycles at 0.5 Hz, give the history and order-20 SPIs enough samples, and fit inside a 30-s sleep epoch. Windows don't overlap: overlapping ones add correlated rows, not units. Power comes from units; windows set each unit's precision.
- **Caps.** Per unit-state: 30 windows for M ≤ 32, 15 for M ≤ 96, 10 above; for NeuroTycho, 10 per archive-state, giving 20–30 per animal-state. Sleep gets 30, spread evenly across the unstaged interval and not selected by slow-wave power (that would make "sleep" N3 by construction). A per-window delta-fraction index is recorded instead, which gives a free within-sleep dose-response. Rule fixed in advance: a cap doubles if wake-only split-half reliability of unit-level summaries is below 0.8.
- **2-s arm (T = 200).** The only arm the epoched fly allows. Every dataset is first cut into 2-s epochs, then each epoch gets the same detrend, 1 Hz high-pass, 40 Hz low-pass and resampling. It can't see the slow oscillation. At M/T above 0.25, the precision family is excluded from this arm's summaries.
- **Exclusions.** Drug transitions (first 5 min after an infusion change [I]); sleep epochs whose neighbours are in another stage; artefacts (a channel with more than 5% of samples beyond 6 robust SD, a flat run of 1 s or more, non-finite values).

## Extraction and compute

- **pyspi.** Upstream main `0c798b4` (r13, unchanged; the head is re-checked at launch), loaded through `PYSPI_SOURCE`.
  - Config: [`configs/pyspi/atlas/benchmarked_p90_pcorr_symbolic.yaml`](../../../configs/pyspi/atlas/benchmarked_p90_pcorr_symbolic.yaml), 272 SPIs. It is pyspi's bundled `benchmarked_p90` at that commit, which has 264 SPIs [V, smoke run], plus two additions copied from upstream `full.yaml` at the same commit:
    - partial correlation for the seven estimators of the Precision block;
    - symbolic transfer entropy (k = 3).
  - The bundled p90 does not contain these (an earlier draft wrongly said it did). The user is adding them upstream by PR (11 November).
  - The repository's `configs/pyspi/benchmarked_p90.yaml` is the frozen r7 catalogue and not for new runs [V, README].
  - The bundled set drops the constant `dspli`/`dswpli` max SPIs, EllipticEnvelope (an exact duplicate of MinCovDet [R]), Lasso and ElasticNet, and bary-sq [V, diff].
  - The three `gd_multitaper_delay` SPIs return no finite values at T = 200 [V, smoke run], so they leave the 2-s arm's coordinate set.
  - It keeps MinCovDet, so the NeuroTycho decoupling is reproducible as discovered [V].
  - K and the family lists are frozen from the profiling run. There is no comparison with r10 banks, and no dyad sampling (user).
- **Runner.** [`extract.py`](extract.py) calls `run_pyspi` with the same arguments as `run_corpus`, a fixed seed per window and per-channel z-scoring (`normalise=True`).
  - Shards scale with M (25 windows for M ≤ 32, 5 for M ≤ 96, 3 for M > 96), so a shard takes 1–2 h.
  - One archive per shard, with provenance, resume and audit.
- **Concurrency.** pyspi `n_jobs` is used wherever memory already pays for the cores: 1 worker for M ≤ 32, 2 for M ≤ 64, 4 for M of about 82–127. Nodes are sized so the cores requested don't exceed shards times workers. It shortens wall time at roughly equal SU.
- **Order.**
  1. Smoke run (two windows per bank).
  2. Profiling (one window per (M, T) class on upstream r13: time, memory, K, float32 check).
  3. Step 1 farms, all at once.
  4. Discovery analysis, then freeze at most three statements.
  5. Step 2 farms.
- **Storage.**
  - Raw NeuroTycho and ANPHY are staged, windowed and deleted one archive or subject at a time on copyq.
  - MPIs are stored as float32 if the profiling run shows a maximum |Δz| below 1e-4.
  - Outputs come to well under 100 GB.

## Analysis (to freeze before step 2)

1. **z per window.** Under the ordered-pair contract, extended to the new K; the three scalars per window; unit-state means.
2. **Within-unit contrasts.** Each state against the same archive's awake eyes-closed. Eyes-open against eyes-closed is the null for generic change; recovery reversal (Chennu) is the drift control.
3. **Controls.** H-control covariates computed from the same arrays. Within fixed-montage datasets, also per-channel spectra and an identity-aware edge-level ceiling.
4. **NeuroTycho taxonomy.** Similarity of each state's Δ (scalars and the 41,000-dimension Δz) to sleep's, across ketamine, medetomidine, ketamine–medetomidine and propofol. Known-answer check: medetomidine should resemble NREM and ketamine should not [U]. Two animals make this descriptive.
5. **Confirmation (step 2).** Frozen statements are tested on Bajwa (propofol), ANPHY (N3, REM) and MNI (N3, REM). Inference is over units, with datasets weighted equally. An interval that includes zero means no demonstrated effect, not falsification.
6. **Secondary.** The 41,000-dimension direction, using a standardised held-out projection (not raw cosine), with coordinates standardised and |r| clipped at 0.995, and a family-block version.
7. **Montage robustness.** Random 15-channel subsets, plus MNI's between-patient variation in M.

## Budget

`ql44` Q4 rises to 105 kSU (pending at NCI); about 58 kSU is free, shared with other sessions. Estimates follow HPC.md's rule [I], ±1.5×, re-priced after profiling.

| Step | kSU |
|---|---:|
| Smoke and profiling | about 0.2 |
| Step 1: NeuroTycho (about 1,270 windows at M ≤ 127; 10-s and 2-s) | about 9.4 |
| Step 1: fly (208 windows, 2-s) | under 0.1 |
| Step 2: Bajwa, ANPHY (W, N3, REM), Chennu (10-s and 2-s) | about 9 |
| Reruns (20%) | about 3.5 |
| **Total** | **about 22 (range 15–33)** |

## Predictions before any atlas output (all [S])

| Outcome | Ours | Fable 5.2 |
|---|---|---|
| H-within | near certain | about 97% |
| H-rank: effective rank changes with state | about 70% | about 70% |
| H-rank survives covariance-rank and spectral controls | about 25% | about 20–25% |
| H1 decoupling recurs in step 2 sets | about 40% | about 15–20% |
| Ketamine alone differs from medetomidine and propofol in NeuroTycho | about 60% | not given |
| Best realistic result (outcome table, row 1) | about 15% | about 10–15% |

## Decisions open

1. Step 1 released 11 November (user): NeuroTycho download job 183606163, fly bank built (208 windows), smoke job 183606432. Code: [`prepare.py`](prepare.py).
2. Watson's probe channel set, if Watson is later added.
