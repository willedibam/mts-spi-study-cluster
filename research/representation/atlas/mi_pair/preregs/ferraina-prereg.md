# Frozen prediction: z(Gaussian MI, kNN MI) beyond IAAFT under dexmedetomidine + ketamine, macaque Utah-array LFP (Zenodo 15466548, monkey B)

Frozen 26 November 2026. The z statistic, the surrogates and the MI have not been computed on any sample of this dataset. Changes after this point go under "Deviations" with reasons.

## Prior exposure (disclosed)

Read before freezing, during the dataset screen (`../dataset-screen-macaque-anaesthesia.md`):
- **Metadata:** the protocol PDF, the HDF5 header, the `MEALFP/channels` and `MEALFP/time` vectors (about 8 MB range-read in all). `time` runs from 165.94 to 9611.94 s in 5-ms steps, so the protocol's event times index the file directly [V].
- **Voltage:** two 60-s blocks of all 192 channels: rows 1000–13000 of `MEALFP/values` (170.9–230.9 s, wake) and rows 800000–812000 (4165.9–4225.9 s, deep anaesthesia). Only Welch spectra, median channel SD and median |r| were computed. Results: no smoothing null at 25 Hz, strong 50 Hz mains, median |r| .69 in wake and .49 in deep anaesthesia, and .63 and .41 between channels 1–96 and 97–192.
- **Exclusion:** any window overlapping either block is excluded from every analysis below.

## Why this dataset, and what it can and cannot show

- **Why:** it is the only open macaque recording found with field potentials in wake and under an anaesthetic, apart from NeuroTycho [V, screen].
  - The agent (dexmedetomidine + ketamine) matches NeuroTycho's ketamine–medetomidine arm, which had the largest effect: normal-scored 10-s Δ −0.015, 11/11 sessions.
  - It is a different lab, array type and agent formulation.
  - The session runs wake → injection → deep anaesthesia → awakening → wake (ABA), so drift and drug can be separated within the session.
- **Limit:** one animal in one session. A positive result shows the effect in one more animal and recording type; it is not an animal-level test.
- **Asymmetry, stated before the result:**
  - **PRESENT** is informative.
  - **ABSENT** is only weakly informative. The Utah arrays sample about 4 × 4 mm at 400 µm pitch [U, standard array geometry; not documented in the record], against NeuroTycho's hemisphere-wide ECoG. A null therefore bounds the effect's generality to micro-scale LFP; it does not count against NeuroTycho.
- **Monkeys C and P:** if Ferraina supplies their LFP, each is analysed under this rule unchanged, with its own protocol times (C: Exp 2 only).
  - P has about 3 min of wake before injection, so its primary is expected to be UNDERPOWERED.
  - B is the first unit, not a design animal. No rule is changed after B's result.

## Data and windows

- **File:** `MEALFP_MB_PFc&PFc.mat`, `MEALFP/values` (1,889,201 × 192, 200 Hz), with times from `MEALFP/time` (2.8 GB download).
- **Arrays:** channels 1–96 and 97–192 are taken as the PMc and PFc arrays [I]. Monkey B has separate MUA files `MEAMUA_MB_PMc` and `MEAMUA_MB_PFc`, but the LFP file's own name reads `PFc&PFc`.
- **Preprocessing:** atlas chain, `research/representation/atlas/prepare.py`.
  - `continuous(x, 200, 50)`: notch 50 Hz (Q 30), high-pass 0.5 Hz, low-pass 40 Hz (Butterworth 8, zero-phase), `resample_poly` to 100 Hz.
  - It is applied to the whole recording at once, so the only filter edges are 166 s and 9612 s, both ≥ 30 s from any window.
- **Channels:**
  - Bad channels: atlas `bad_channels` on the 100-Hz W1 eligible data.
  - Reference: `reference(x, 'car')` over the good channels, then the last one is dropped.
  - Primary set: 32 channels by index spread evenly over the remaining good list, `round(arange(32)·(m−1)/31)`, which is NeuroTycho parity; with 191 channels this takes about 16 from each block [I].
  - Lost channels are reported.
- **States** (protocol PDF times, seconds):
  - **W1** (pre-injection wake): 231–1979. It starts after the exposed wake block and ends 60 s before injection at 2039.
  - **D** (deep anaesthesia, as labelled in the protocol): 3000–6000.
  - **W2** (post-awakening wake): 8031–9606. It starts 60 s after eyes open at 7971 and ends 5 s before the end of the data.
- **Windows:** 10-s non-overlapping windows on a grid from each state's start, all kept (no cap). This gives about 174, 294 (after the exposed-block exclusion) and 157 candidates.
- **Artefact rule:** atlas `window_ok`.
  - Robust centre and scale from W1's eligible data for every state, as in Claar Deviation 1.
  - Flat runs are ≥ 200 identical source-rate samples (1 s).
  - Rejections are reported by state.
- **Minimum:** a state with < 20 accepted windows makes the primary UNDERPOWERED.

## Statistic

- **Code:** `zgk` and `ns` imported unchanged from `claar/zgk_ns.py`.
- **Per window:** normal-scored channels; z = corr over pairs of Gaussian MI and KSG MI (k = 4, AUTO Theiler window).
- **Surrogates:** 3 joint multivariate IAAFT draws (spectrum-final), with seed = window index.
- **Residual:** real z − mean surrogate z.
- **Saved:** the per-pair Gaussian and KSG MI for the real window and each surrogate, so that pair subsets can be rescored without recomputing.
- **Compute estimate [I]:** 32 channels and about 625 windows take about 1–2 core-hours, which is fine locally. The all-channel secondary has 37× the pairs (about 3 core-days), so it goes to Gadi.

## Estimands and decision rule (one session, no across-unit null)

- **Δ_D** = mean residual(D) − mean residual(W1).
- **Δ_rec** = mean residual(W2) − mean residual(D).
- **Uncertainty:** moving-block bootstrap within each state, with 5-min blocks (30 grid positions), 10,000 resamples and a 95% percentile CI. Window-level p-values are not used, because windows are autocorrelated.
- **Block null:** W1 is cut into contiguous 5-min blocks, and blocks with < 15 accepted windows are dropped. The null SD is the SD of these block means. D is cut the same way into up to 10 blocks.
- **PRESENT** requires all four:
  1. Δ_D < 0 with its CI upper end < 0.
  2. Δ_D beyond −2 block-null SD.
  3. At least 80% of D blocks have a mean below the W1 mean (sign consistency over time).
  4. Δ_rec > 0 (point estimate), because a monotonic drift would put W2 at or below D.
- **ABSENT:** |Δ_D| within 2 block-null SD, and the CI lower end > −0.010 (about two-thirds of NeuroTycho's ketamine–medetomidine Δ; the same bar as Claar and Kauvar).
- **REVERSED:** Δ_D > 0 with its CI lower end > 0.
- **UNINFORMATIVE (ceiling):** if the median real z in W1 is ≥ .97 (Chennu scalp ≈ .98) and the result is not PRESENT, no verdict is given.
- **Otherwise INCONCLUSIVE.** If criteria 1–3 hold and only criterion 4 fails, the result is INCONCLUSIVE (drug not separable from drift), not PRESENT.

## Secondary (reported whatever the primary outcome)

- **Drift:** W1 later half − earlier half; W2 − W1 (residual drug and arousal are confounded here).
- **Amplitude shape:** burst index (median over channels of 1.4826·MAD / SD) and delta fraction per window by state.
  - Δ_D using only D windows with burst index ≥ .85.
  - Δ_D at W1's median burst index, from the D residual-on-burst-index fit.
- **Reference and channel sets:**
  - as-recorded reference (`asis`) on the same 32 channels;
  - all good channels (Gadi);
  - the 32-channel set rescored on cross-block pairs only (channels ≤ 96 against > 96) and on within-block pairs only.
- **Raw:** the statistic on channels that are not normal-scored, from the same `zgk_ns` call.
- **Ceiling diagnostic:** absolute z and 1 − z², real and surrogate, by state.
- **Time course:** the residual for every 10-s window over the whole recording, including injection and awakening, as a descriptive plot only.

## Prior [S]

- **Lean:** PRESENT, moderately. The same drug class gave 11/11 sessions in NeuroTycho, the effect survived burst matching there, and the mouse induction result replicated the sign.
- **Against:**
  - Wake |r| between channels is high (.69), so z may sit nearer the ceiling.
  - The LFP reference and the anti-alias filter before the 200 Hz storage are undocumented. Aliasing above 100 Hz would fold into the 0–40 Hz band that is kept.
  - "Deep anaesthesia" is an experimenter label, not a measured depth.
  - Dexmedetomidine produces spindle-like and slow-wave activity, which raises the burst index question again.

## Known threats

- **n = 1 animal and 1 session:** the inference is within this session.
- **Reference unknown:** it may be a single shared reference or headstage ground.
- **Movement and residual drug in W2:** W2 is a reversal check, not a clean baseline.
- **Array assignment of channels is inferred**, so the cross-block secondary may mix arrays.
- **Transfer:** one 2.8 GB file, to be downloaded only on the user's go-ahead.

## Pre-data clarifications (26 November, before any window was selected or any z computed)

Scripts: `ferraina/extract_fb.py`, `select_fb.py`, `fb_z.py`, `fb_score.py`. They settle points the text above left open; the decision rule is unchanged.

1. **Artefact rule scope:** as in the atlas NeuroTycho chain, `window_ok` is applied to the window on all good channels after the common average, not only to the 32 analysis channels. The flat mask counts a flat run in any good channel.
2. **Covariates:** burst index and delta fraction are computed on the 32 analysis channels after the common average and before normal-scoring.
3. **Bootstrap:** each state's accepted windows in time order. Blocks of 30 consecutive accepted windows start at uniform random positions, and ⌈n/30⌉ blocks are concatenated and truncated to n. Seed 20261126.
4. **Block null:** blocks are grid positions k // 30 from the state's start.
5. **Pair subsets:** a channel number ≤ 96 marks the first block. The cross-block and within-block z are rescored from the saved per-pair MI. `zgk_pairs` is asserted equal to `zgk` on the first window.
6. **Compute (the earlier estimate was wrong):** about 75 s per window (4 × 496 KSG estimates), so about 13 core-hours for the primary, about 1.5 h wall-clock on 9 local workers.
   - Local: the primary, then the as-recorded secondary.
   - Gadi: the raw secondary (IAAFT converges slowly on raw data; Claar) and the all-channel secondary.
   - Deferred: the time course over the whole recording, including transitions (about 940 more windows). For now it is plotted from the state windows only.
7. **Seeds:** IAAFT seed = state offset (W1 0, D 100000, W2 200000) + 10·k, with draws seed + 0, 1, 2.
