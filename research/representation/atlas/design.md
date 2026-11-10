# Consciousness atlas: run design and pre-registration draft (11 November 2026)

Status: **draft for review, not frozen; nothing extracted.** The question, prior art and dataset search are in the [scoping note](../representation_state/notes/real-world-application-scoping.md) (application 1); this file is the run architecture and the analysis to freeze before any atlas pyspi output is seen. Labels: **[V]** checked this session against the file, API or paper; **[R]** recorded in the repository; **[I]** inferred; **[U]** unverified; **[S]** speculation.

## Question and predictions

Is there a direction of change in dependence organisation from wake to unconsciousness that recurs across agents, sleep, species and recording types, which identity-free controls do not already carry, and does it follow reported experience rather than responsiveness?

- **P1.** The direct-against-pairwise decoupling found in NeuroTycho (robust partial r² against rank or ids profiles) recurs in the shared direction under propofol, isoflurane and NREM, beyond multivariate IAAFT surrogates, and is weaker under ketamine and in REM. NeuroTycho is the discovery set, so P1 is tested on the other datasets.
- **P2.** Leave-one-dataset-out, the shared z direction aligns with held-out datasets better than the same procedure applied to each identity-free control (mean-SPI, Pearson edge quantiles, channel-pooled log spectra, channel-pooled catch22). Falsified if any control's paired alignment interval includes or exceeds z's.
- **P3.** Within sleep, z in the windows before an awakening separates reports of experience from no experience beyond pooled spectra. Falsified if pooled spectra match it. Testable on DREAM (below), and on Bajwa if the report labels arrive.
- **Residual test (the finding that would matter most).** After removing the part of Δz predicted from the pooled-spectral displacement (ridge, fitted without the held-out dataset), a shared direction remains and generalises leave-one-species-out.

## Datasets, sizes and what each contributes

Raw is what must be downloaded or streamed; "bank" is the preprocessed window archive pyspi reads (float32, all arms). Windows are per arm under the budget caps below.

| Dataset | Species, signal | States (units) | Native M, reference | Source fs, structure | Raw | Arms | Windows / arm | Role |
|---|---|---|---|---|---|---|---|---|
| Fly (Leung 2025, figshare + OSF) | fly, LFP | wake vs isoflurane (13 discovery flies; 56 more series in `validate1`, labels [U]) | 15 bipolar [V] | 1 kHz, 8 × 2.25-s epochs per fly-condition (18 s) [V] | 52 + 177 MB [V] | 2-s at 100 and 200 Hz | 656 | displacement (2-s arm only) |
| DOC (figshare 23552964) | human, scalp EEG | patient vs control (59 + 32 stated; 98 headers, 96 data files [V]); UWS/MCS labels not in deposit | 62 [V] | 250 Hz, 60 × 2-s segments, not contiguous [V] | 714 MB [V] | 2-s at 100 Hz | ≤ 2,880 | projection only |
| NeuroTycho anaesthesia | macaque, ECoG | wake, ketamine–medetomidine, propofol (4 animals, about 8 sessions [U]) | 128 | 1 kHz continuous | 128-ch raw must be re-staged from the public zips (only 16-ch views are on Gadi [V]); size [U] | 10-s at 100 Hz, 5-s at 200 Hz, 2-s | 240 | exploratory (discovery) |
| Chennu 2016 (Apollo) | human, scalp EEG | baseline, mild, moderate, recovery (20) | 91 | 250 Hz, about 7 min per level [R] | [U], few GB | 10-s, 2-s | 1,200 | displacement, graded |
| Bajwa 2025 (ds005620) | human, scalp EEG | awake vs propofol sedation (20); 52 pre-awakening minutes | 65 incl. non-EEG | 5 kHz [R] | [U] | 10-s, 2-s | 912 | displacement; P3 if labels come |
| ANPHY-Sleep (OSF) | human, scalp EEG | W, N2, N3, REM (29) | 83 | [U] | overnight PSG, [I] about 10 GB per person, so download-extract-delete per subject | 10-s, 2-s | 1,740 | displacement, sleep |
| MNI open iEEG atlas | human, sEEG/ECoG | W, N2, N3, REM (about 106 patients [U]) | about 17 per patient [U] | 200 Hz, 60-s segments [U] | about 1 GB [U] | 10-s, 5-s at 200 Hz, 2-s | 2,544 | displacement, intracranial sleep |
| Watson 2016 (DANDI 000041) | rat, frontal probe LFP | wake, NREM, REM (10 rats, 22 sessions [V]) | probe LFP, 64 assumed [U] | [U] | 149 GB [V]; stream LFP only | 10-s, 5-s at 200 Hz, 2-s | 450 | displacement, rodent sleep |
| Claar 2023 (DANDI 000458) | mouse, skull EEG | awake vs isoflurane (23 [V]) | about 30 [U] | [U] | 361 GB [V]; stream EEG only | 10-s, 5-s at 200 Hz, 2-s | 1,380 | displacement, rodent anaesthesia |
| Kauvar 2025 (DANDI 001347) | human, sEEG | baseline, ketamine, post (ketamine subset of 173 subjects [V], dose [U]) | about 100 [U] | [U] | 16 GB [V] | 10-s, 5-s at 200 Hz, 2-s | 450 | displacement, ketamine |
| DREAM (Monash, doi 10.26180/22133105) | human sleep EEG/MEG, 20 studies | experience, experience without recall, no experience; 2,643 awakenings, 505 people [V] | 2 to high density per study [V: inclusion needs ≥ 2] | ≥ 100 Hz, ≥ 20 s before each awakening, EDF [V] | [U]; open subset only | 10-s, 2-s | [U] | P3 (new; studies with M ≥ 8 only) |
| Purdon (DANDI 001285) | human, scalp EEG | graded propofol | 64 [V] | 250 Hz, 154 min [V] | 1.2 GB [V] | — | 0 | **excluded**: third-party repackaging with unnamed channels and no drug-level or LOC/ROC annotations [V] |

Bank sizes are small (ANPHY at 10 s: 1,740 × 1,000 × 83 × 4 bytes ≈ 0.6 GB); the large volumes are raw downloads and pyspi outputs.

## Preprocessing (`atlas/prepare.py`, to write)

1. **Channels.** Drop non-neural channels (EOG, EMG, ECG). Fix one channel set per unit, kept identical across that unit's states, so Δz compares the same montage. Bad channels are removed by fixed rules on the whole unit (flat: SD below 1% of the median channel SD; noisy: SD above 5 times the median). Native M otherwise, with no reduction.
2. **Reference.** Scalp EEG: average reference. ECoG grids: common average. sEEG depth: adjacent-contact bipolar within each shaft. Fly: bipolar as released. Rodent probes, skull EEG and the MNI atlas: as released, recorded per dataset.
3. **Filters.** High-pass 0.5 Hz, zero-phase. Notch at mains and harmonics below the low-pass. Zero-phase FIR low-pass at 40 Hz (100 Hz arm) or 80 Hz (200 Hz arm), then polyphase resampling. Epoched sources (fly, DOC) are filtered per epoch with reflection padding; the central 2 s are kept.
4. **Windows.** Non-overlapping, evenly spaced through each stable state period, deterministic. Excluded: drug transitions (first 5 min after an infusion change [I]), sleep epochs whose neighbours are in another stage, and windows that fail artefact rules (any channel with more than 5% of samples beyond 6 robust SD of that channel's unit-level distribution, a flat run of 1 s or more, or non-finite values).
5. **Normalisation.** pyspi's per-channel z-score along time in each window (`normalise=True`). The controls are computed from the identical arrays.
6. **Output.** One bank per dataset and arm, `windows.npz` plus `manifest.json` (id, unit, state, M, source times, preprocessing record), which is what [`extract.py`](extract.py) reads.

## Rates, window lengths and why

- **Why one rate, and why it doesn't penalise any dataset.** pyspi runs at fs = 1: band-indexed SPIs split at fs/4, and lags, histories and model orders are counted in samples ([V], about 115 band-indexed and 45 lag SPIs). At native rates, one SPI name measures different physics in different datasets (the upper band is 62–125 Hz at 250 Hz and 250–500 Hz at 1 kHz), and the atlas compares datasets. A common rate is therefore required for comparability, not chosen for convenience. Unlike reducing M, it removes no dependence structure inside the band both arms keep.
- **100 Hz, cross-tier (primary).** Scalp EEG above about 40 Hz is mostly muscle, and 50 Hz mains falls at Nyquist, removed by the 40 Hz low-pass. The split falls at 25 Hz (delta to beta against beta to low gamma), and one lag is 10 ms.
- **200 Hz, intracranial tier (co-primary).** It keeps gamma up to 80 Hz, which is signal in LFP, ECoG and sEEG and is where ketamine and REM are expected to differ. **Proposed change: T = 1000 (5-s windows) in this arm instead of 10 s.** Run time scales with T (about 3.3 times from T = 1000 to 2000 by HPC.md), so this cuts the arm's cost 3.3-fold. T stays matched within each arm, and z is never compared across arms (only effects are), so matching T rather than duration within an arm keeps estimator bias equal across datasets.
- **10 s at 100 Hz.** It holds about 5 cycles at 0.5 Hz and gives the history and order-20 SPIs enough samples. It is shorter than the 30-s sleep epoch, so windows stay inside a scored stage.
- **2-s arm (T = 200; T = 400 for the fly at 200 Hz).** This is the only arm the epoched fly and DOC releases allow. It runs everywhere, because it is cheap and lets fly and DOC sit in the same space as the rest. The nearest-neighbour information SPIs, order-20 Granger and cointegration are weak at T = 200.
- **No overlap.** Overlapping windows would add correlated rows without adding units. Power comes from units; windows only set each unit's precision.

## Extraction

- **pyspi.** Upstream main `0c798b4` (r13, unchanged since the snapshot; the head is checked again at launch), loaded through `PYSPI_SOURCE` from the immutable snapshot. Config `benchmarked_p90` from that commit. Its SPI order differs from the stored K = 283/289 (pcorr and te_symbolic_k-3 added, bary-sq removed), so the K count is read from the smoke run and the z contract is extended to accept it.
- **Full channel space.** Every ordered pair of the native montage is computed; no dyad sampling (user, 11 November).
- **Runner.** [`extract.py`](extract.py) calls the same `run_pyspi` as `run_corpus`, with a fixed seed per window and z-scoring. It writes one archive per shard of 25 windows, with provenance, resume and audit; tests pass.
- **Concurrency.** pyspi's own SPI-level concurrency (`n_jobs`) is used for every window whose memory already pays for extra cores. Workers per window come from HPC.md's table: 1 for M ≤ 32, 2 for M ≤ 64, 4 for M of about 83–128 at T = 1000, and 2 for M = 128 at T = 200. Each node holds the smaller of 48 over workers and 190 GiB over memory per window. Concurrency shortens wall time at roughly equal SU; it doesn't reduce cost.
- **Farms.** [`farm.pbs`](jobs/farm.pbs), one job per dataset and arm, each spanning 4–16 nodes (192–768 cores) as the window count warrants, all submitted together. At about 11,000 core-hours, 16 nodes finish in about 15 hours of wall time.
- **Order.** (1) Smoke on two windows per bank. (2) One profiling job: one window of each (M, T) class, measuring time, memory and K. (3) Size and release every farm at once.
- **Storage.**
  - MPIs are K × M × M per window: 38 MB at M = 128, 16 MB at M = 83, about 0.5 MB at M = 15, in float64.
  - Under the caps, the outputs total about 230 GB in float64, about 115 GB in float32.
  - **Proposed: store MPIs as float32** (z is a correlation, so relative precision of 1e-7 is ample). That is a one-line change to `extract.py` before launch.
  - Scratch is at 223 GiB of 1 TiB; inodes run to about 1,100 shards plus logs.
  - Raw: ANPHY, Watson and Claar go through download-or-stream, extract windows and delete raw, one subject at a time on copyq, so raw never accumulates.

## Analysis (to freeze)

1. **z per window** under the existing ordered-pair contract (all M(M−1) off-diagonal entries), extended to the new SPI order. Then Fisher-transformed unit-state means and Δz = state − the unit's wake baseline, a vector of about K(K−1)/2 ≈ 41,000.
2. **Alignment.**
   - For each held-out dataset (and species), cosine between its precision-weighted mean Δz and the mean from all other datasets for the matching state class (GABAergic or isoflurane, NREM, REM, ketamine).
   - Null: state labels permuted within units, 1,000 times.
   - Precision weight per unit: the inverse of the split-half variance of its Δz.
   - M enters as a covariate: alignment is re-estimated after regressing log M out of the unit Δz, and also shown separately for high- and low-M datasets.
3. **Controls through the identical pipeline.** mean-SPI (K), Pearson edge quantiles, pooled log spectra (1–40 Hz, or 1–80 Hz in the 200 Hz arm), pooled catch22. Each gives Δ vectors and the same alignment test. P2 compares paired alignment over held-out datasets.
4. **Residual direction.** Ridge from Δ(pooled spectra) to Δz, fitted leave-one-dataset-out; the residual goes through the alignment test.
5. **Character reading.** The SPI-family blocks that carry the shared and residual directions (direct against pairwise, directed against undirected, linear against rank or nonlinear). Each named effect is checked against multivariate IAAFT surrogates for time-domain families only.
6. **Projection.** DOC (patients against controls) and Chennu, Purdon or Bajwa graded levels are projected onto the learned direction; they don't enter its estimation.
7. **P3 on DREAM.** People with both experience and no-experience awakenings in a study with M ≥ 8: within-person paired contrast of pre-awakening z, against pooled spectra, by stage (NREM and REM separately).
8. **NeuroTycho montage diagnostic.** The Δz direction at native M compared with random 16- and 32-channel subsets of the same windows. It is reported only as a check that direction does not track M, never as the analysis.
9. **Losses are kept.** Every arm and control is reported whatever the outcome.

## Budget

`ql44` has **27.6 kSU** left this quarter [V, `nci_account`], shared with other sessions. Costs follow HPC.md's rule (about 1.2 core-seconds per channel pair at T = 1000, 4 at 2000 and 0.2 at 256; ±1.5 times) [I]:

| Design | 10 s / 100 Hz | 200 Hz | 2-s | Total |
|---|---:|---:|---:|---:|
| Full: 30 windows per unit-state, 200 Hz at 10 s | 34 kSU | 41 kSU | 5 kSU | **80 kSU** |
| Capped: 30 / 15 / 10 windows for M ≤ 32 / ≤ 96 / > 96, 200 Hz at 5 s | 15 kSU | 5 kSU | 3 kSU | **22 kSU** |

The expensive cells are the high-M scalp sets in the 10-s arm (ANPHY 4.7, Chennu 3.9) and the intracranial high-M sets (Kauvar 3.6, NeuroTycho 3.4 across arms). Fewer windows for high M are justified because z from thousands of pairs is precise per window: what windows add is coverage of state fluctuation, and units carry the power.

## Our predictions (before any atlas output)

- Within every dataset, wake against deep GABAergic or N3: separable (near certain).
- Raw Δz alignment across mammalian GABAergic, isoflurane and N3 sets, above the permutation null: likely (about 75%) [S].
- The pooled-spectra control aligns as well or better, so P2 fails or ties: more likely than not (about 65%) [S].
- A residual direction that survives leave-one-dataset-out in most held-out mammalian sets: about 35% [S]. Fly to mammal: about 25% [S].
- P1 recurs outside NeuroTycho beyond surrogates: about 40% [S].
- Ketamine and REM displacements closer to wake than GABAergic ones on the residual: about 40% [S]. The ketamine dose in Kauvar is unknown.
- P3 on DREAM beyond pooled spectra: about 15% [S]. Posterior low-frequency power already predicts reports (Siclari et al. 2017 [U, from memory]).

## Decisions open before freeze

1. Budget: how much of the 27.6 kSU goes to the atlas this quarter, and what moves to 2027 Q1.
2. 200 Hz arm at T = 1000 (5 s) rather than T = 2000 (10 s).
3. Window caps by M.
4. float32 storage of MPIs.
5. Adding DREAM (open subset) for P3.
6. References per recording type (average, common average or bipolar), with no Laplacian.
