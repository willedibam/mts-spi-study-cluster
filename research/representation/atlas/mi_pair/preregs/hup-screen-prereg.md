# HUP full-pyspi screen: which dependence characters differ between the seizure-onset zone and matched non-onset contact sets, beyond a per-window joint IAAFT null

**DRAFT, not frozen.** Open before freezing: the pyspi catalogue (below), an r13 profile shard if r13 is chosen, and the user's go-ahead. Freezing happens before any pyspi output on an SOZ set exists. Code: [`../hup/screen.py`](../hup/screen.py), [`../jobs/hup-screen.pbs`](../jobs/hup-screen.pbs), pyspi through [`../../jobs/farm.pbs`](../../jobs/farm.pbs). Reading rules: [`docs/interpreting-spi-pairs.md`](../../../../../docs/interpreting-spi-pairs.md). Design parent: the PRIOS screen ([`../../movers/prios-prereg.md`](../../movers/prios-prereg.md)); data, windows and sets: the HUP MI test ([`hup-prereg.md`](hup-prereg.md)).

Seen before freezing: the HUP MI prereg and its outcome-blind feasibility check; pyspi timing on random non-SOZ contact sets of HUP112 and HUP141, window 0 (and windows 0–2 for the overhead test), real data at M = 8, 14, 24, 38 and 44 and one surrogate at M = 14 (wall and CPU time, memory, per-SPI seconds, which SPIs were non-finite; no SPI value kept). Non-finite: current pin, the six `dcoh_multitaper` SPIs on the M = 14 surrogate (one singular pair of 182); r13, the three `gd_multitaper_delay` SPIs in every recording (in no minimal pair). [To be filled: whether the HUP MI verdict was seen.]

## Why a screen here

The MI pair is one of about 700 minimal pairs. PRIOS (7 patients, 2-s windows) could test only 123 of them, and its open screen needed d_z ≈ 2.8 for a first discovery. HUP gives about 29 patients and 10-s windows, so the open screen can detect moderate effects (with m usable pairs the first BH discovery needs p ≤ .05/m; at m = 200 and 28 df, t ≈ 4.2, d_z ≈ .8), and lag and phase-lag SPIs that failed the 2-s reliability filter may pass [inferred]. The matched-set null measures how much any two comparable contact sets of one patient differ, which is the main alternative explanation of an SOZ difference.

## Data and recordings

- Patients, windows and sets are the HUP MI test's: patients with n_S ≥ 8 SOZ contacts with coordinates and ≥ 20 accepted primary matched sets (`score.matched`, same pool, seed and |r| input); every passing 10-s window of both interictal runs (T = 1000 at 100 Hz, chain and exclusions as `hup-prereg.md`).
- Per window, the full implant is normal-scored per channel, and one spectrum-final joint IAAFT draw is made over the full implant with the MI test's draw-0 seed (`crc32("<subject>:<window>") × 10`), so the surrogate is the MI test's draw 0. A column subset of a joint surrogate keeps that subset's auto- and cross-spectra and marginals.
- Recordings: for S and for N_1 … N_20 (the first 20 accepted matched sets, in acceptance order), that set's columns of the real window (bank `hup-ns`) and of the surrogate (bank `hup-iaaft`). Each recording is self-contained (M = n_S, 8–38), so conditioned SPIs (precision, partial correlation) condition on a set of the same size in S and in every N_j.
- pyspi: [decision pending, see "Catalogue"]. Per-window seed by id; per-channel z-scoring (`normalise=True`); `atlas/extract.py`; reduced with `discovery.py reduce` (z over ordered edges, double-centred z, node R²).

## Catalogue (open)

- **Atlas catalogue (recommended):** `configs/pyspi/atlas/benchmarked_p90_pcorr_symbolic.yaml`, 272 SPIs, source `pyspi-upstream-0c798b4` (r13), as PRIOS and the macaque. 704 minimal pairs in 115 classes, 28 classes with validated character readings (249 pairs); coordinates identical to the PRIOS and macaque screens. Condition: it must run in the current environment (timing job).
- **Current pin (`4e0810c`, `benchmarked_p90`):** 207 SPIs, 225 minimal pairs in 70 classes, 17 with readings (87 pairs). Its new families (lagcorr, CCM, NLI, PAC, ξ, tail and edge statistics) parse as standalone statistics, so they form no minimal pairs; `assumptions.parse` and `READINGS` were built on the r13 names.
- **Third option:** extend `assumptions.parse` and `READINGS` to the 3.0 names first, then run the current pin. This is the project's upgrade default, but it means new parsing and readings code, and those must be frozen before the screen.
- The r13 cost is mostly `bary_dtw_corr`, `bary_sgddtw_corr`, `lcss_*` and r13's slower `dcorr`. Together they enter 17 of the 704 minimal pairs, 4 of them with readings. This is a fact about the catalogue, not a reason to drop them.
- The choice rests on catalogue validity for a minimal-pair screen, not on cost. One catalogue is primary.

## Usable coordinates (blind to the SOZ contrast)

- ≥ 95% finite windows in every patient × {S, matched sets pooled} × bank;
- both SPIs' reference reliability ≥ .3: per unit (patient:set) over the matched sets only, the mean correlation between different windows' edge profiles, averaged over units (`screen.py reliability`: `discovery.reliability` for every SPI valid in ≥ 95% of the real bank's recordings);
- |mean matched-set real z| < .98.

## Statistic

- Net per recording and coordinate: z(real) − z(surrogate). Set value: mean over the set's windows.
- Per patient: Δ = net(S) − mean_j net(N_j). Control (no-change baseline): C = net(N_1) − mean_{j ≥ 2} net(N_j). Effects are also reported in SDs of C across patients.

## Tests

1. **Primary discovery: every usable minimal pair** (`assumptions.edges`). Two-sided one-sample t on patient Δ; Benjamini–Hochberg q ≤ .05 over the usable minimal pairs; and a two-sided sign test p < .05 in the direction of the mean. Negative control: the identical pipeline on C. Estimated false-discovery proportion = control discoveries ÷ discoveries; nothing is claimed if it exceeds .2.
2. **Anchor:** the MI pair `mi_gaussian|mi_kraskov_NN-4_DCE-AUTO` is reported whether or not it passes. It is not independent of the MI test (same windows and sets; one draw and 20 sets here against three draws and up to 200 sets there).
3. **Secondary, descriptive:** minimal pairs without the reliability filter (BH within); every usable coordinate (BH within, with control count and the SPIs ranked by participation).

## Reported per discovery

Class and readings (`assumptions.READINGS`: character, literal, confound) with wording set by validation status; mean Δ, 95% interval, patients positive, p, q, mean in control SD; Δ of real and surrogate z separately; Δ of double-centred z; matched-set node R² of both SPIs (≥ .5: the profile reads channel properties); spike-free windows only (per set, windows in which no set channel exceeds 5 robust SDs; sets with < 10 such windows dropped); run 1 and run 2 separately and their sign agreement across patients; ECoG against SEEG; Engel class 1 against the rest.

## Pre-stated limits

- The SOZ label is clinical; matched sets lie in other regions with other cytoarchitecture and spectra. The surrogate removes linear-spectral differences, not regional nonlinear ones; the control C measures how large region-to-region differences among comparable sets are.
- One IAAFT draw per window (as PRIOS). Small sets: z over 56 ordered edges at n_S = 8.
- Interictal spikes are joint bursts by construction; the spike-free row separates them descriptively.

## Compute

Recordings: 1,506 windows (29 patients) × 21 sets × 2 banks = 63,252 recordings, each n_S × 1000. They hold 3.50 M channel pairs per bank, 7.0 M in all.

Measured on Gadi, 24–25 November 2026: jobs 185088329, 185088616 and 185088787, scripts in scratch `operations/profiling/`, one worker, `*_NUM_THREADS=1`.

- **Per process:** about 40 s (import 10 s, first Calculator init 30 s). Later inits take under 0.3 s, so the per-process cost is paid once per shard.
- **Current pin (`4e0810c`, 207 SPIs):** CPU per recording is about 6 s + 0.62 s per pair from M = 8 to 44. M = 8 takes about 24 s, M = 14 about 64 s, M = 38 about 430 s. Peak RSS is 0.9–1.7 GiB.
  - Set design: about 4.7 M CPU-s ≈ 1,300 core-h. Expected 2.6 KSU, worst case 4 KSU. About 1.4 h on 960 cores.
- **r13 atlas (272 SPIs):** much slower, and the spread is wide and not understood.
  - M = 8: 1.9–2.3 s per pair (HUP141, overhead sets) and 7.5–9.3 s per pair (profile sets).
  - M = 14: 12.1 and 13.6 s per pair.
  - The macaque long arm, also r13 at T = 1000, ran at about 3.4 CPU-s per pair (1.13 s wall on 3 workers, M = 126).
  - The cause of the gap is unverified. One hypothesis is that the iterative barycentre SPIs depend on the data.
  - Set design at 3–14 s per pair: 5,800–27,000 core-h, so 12–55 KSU. No surrogate has been timed.
- **Before an r13 farm:** run a profile shard of 30–40 non-SOZ recordings across patients and sizes, real and surrogate, with per-SPI timings. Then fix the KSU and the shard sizes.
- **Shard sizes:** `extract.py` resumes only whole shards, so shard size is set by size class so that one shard fits within `TASK_TIMEOUT`.
- **Validity:** one non-finite pair makes that SPI invalid for the whole recording (`discovery.corr_upper`). The `dcoh` family's rate on the surrogate bank will show in the 95% rule.

Every submission states expected and worst-case KSU, recorded with the result.

## Deviations

None yet.
