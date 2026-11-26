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

## Deviation 1 (26 November): post-QC, pre-z. Artefact scale from each state, not from W1

**What had been seen:** amplitude and rejection counts only (`ferraina/fb_sel_qc.txt`); no z, residual or MI on any window.
- Under the frozen W1-scale rule, 0 of 293 D windows passed and 2 of 157 W2 windows passed, against 156 of 174 in W1.
- **Cause:** a global amplitude change, not artefact. The median channel SD relative to W1's robust scale is 4.6 in D, 4.3 between injection and deep anaesthesia, and 2.5 in W2. Per channel, D's robust scale is 2.8–6.5× W1's (5th–95th percentile). In D, a median of 187 of 191 channels exceed the 5% outlier limit.
- **Why this is the reverse of Claar:** Claar Deviation 1 moved to the wake scale because burst suppression shrank the isoflurane MAD. Here the anaesthetised LFP is continuous and larger, so the wake scale flags the state itself.

**Change:** each state's robust centre and scale come from its own grid windows. This is the atlas default, and the NeuroTycho effect was established with it. Everything else is unchanged.

**Effect:** accepted windows are W1 156 (18 rejected), D 278 (15 rejected, 7 exposed) and W2 131 (26 rejected). The W1-scale rule leaves no D windows, so no sensitivity run under it is possible.

**QC by state (5th, 50th and 95th percentiles):**

| State | Burst index | δ fraction | Window SD (file units) |
|---|---|---|---|
| W1 | .89 / .97 / 1.00 | .14 / .20 / .30 | .017 / .018 / .023 |
| D | .83 / .91 / .97 | .79 / .85 / .89 | .067 / .078 / .093 |
| W2 | .71 / .84 / .94 | .20 / .32 / .47 | .031 / .043 / .060 |

W2 is not a return to W1: its amplitude is 2.4× W1's, with a lower burst index and more δ. That fits the residual drug the prereg anticipated, and it weakens W2 as a reversal check.

## Clarification 8 (26 November): during the primary run, before any residual, surrogate z or state comparison was read

**What had been seen:** one real z, W1 window 0 (.994), printed while checking the KSG change below. No surrogate z, residual or other window.

- **KSG implementation:** `claar/surr_pairs.ksg` now takes the k-th neighbour distance with `np.partition` instead of a full sort. The output is identical: 141 pairs from 3 windows per state, real and IAAFT, maximum difference 0 [V]. It is about 1.7–2.3× faster. The running primary uses the old code; every later run uses the new one.
- **Compute correction:** the all-channel secondary has 18,145 pairs, 36.6× the primary. That is about 475 core-hours at the old speed, not "about 3 core-days", and roughly half that with the change [I].
- **Two added secondaries (Gadi), reported beside the primary; they do not enter the verdict:**
  - **k = 10, AUTO window.** On NeuroTycho it retained R .95 (t(3) −5.98) and was the better configuration on drug-blind calibration (`robustness-result.md`).
  - **kshift:** k4 minus the mean k4 of the same pair with x_j circularly shifted by T/4, T/2 and 3T/4, as `ksgbias.py`. It removes the real-only weak-edge bias under slow drug spectra; on NeuroTycho, R 1.11 (`ksg-bias-result.md`).
  - **Why:** these are the two configurations the NeuroTycho robustness work gives a case for over k4 AUTO. The primary stays k4 AUTO: it is frozen, NeuroTycho's reference values are in k4, and switching after seeing a real z would be result-dependent.
- **Theiler window by state (descriptive, reported with the result):** the median AUTO window is 3, 21 and 7 samples in W1, D and W2. NeuroTycho shows the same pattern (awake eyes-closed 5, KTMD 17, PF 22). There, the resulting KSG bias was shared with the surrogate and cancelled in Δz.

## Result (26 November): UNINFORMATIVE (ceiling)

- **Output:** `ferraina/fb_result_ns.txt`.
- **z file:** `data/ferraina_zen15466548/fb_z_ns.npz` (outside the repo, 9.4 MB, md5 beb29b5208757837d023745ce55389d2).
- **Code:** scripts as at ed7b7ca8. The KSG change in Clarification 8 gives identical output.

| | W1 | D | W2 |
|---|---|---|---|
| Accepted windows | 156 | 278 | 131 |
| Real z, median | .9947 | .9952 | .9955 |
| IAAFT z, median | .9947 | .9977 | .9978 |
| Residual, mean (SD) | −.0009 (.0035) | −.0026 (.0012) | −.0034 (.0031) |

**Primary results:**

| Quantity | Value |
|---|---|
| Δ_D | −.0018 [−.0026, −.0014] |
| Δ_D in block-null SD | −1.24 (null SD .0014) |
| D blocks below the W1 mean | 10 of 10 |
| Δ_rec | −.0008 [−.0013, +.0003] |
| W2 − W1 | −.0025 |

- **Criteria:** 1 and 3 hold, 2 and 4 fail.
- **Ceiling:** the W1 median real z is .9947, at or above the .97 cut, so the verdict is UNINFORMATIVE.
- **Precedence:** ABSENT's numeric conditions also hold (|Δ_D| 1.24 SD ≤ 2; CI lower end −.0026 > −.010). The frozen ceiling clause ("no verdict is given") takes precedence, so the verdict is UNINFORMATIVE; without that clause it would read ABSENT.

**Readings:**

- **Direction [V]:** the residual is lower under the drug, with a CI that excludes 0 and every D block below the W1 mean. That is NeuroTycho's sign at about one-eighth of its size (−.015).
  - It does not exceed W1's own block-to-block variation: one W1 block, at −.0037, is below the D mean.
- **No reversal [V]:** W2 sits below D.
  - W2's amplitude is still 2.4× W1's, so the data cannot separate residual drug from time in the recording [I].
- **Mechanism [V]:** real z barely moves (means .9937, .9950 and .9944). The residual falls because the IAAFT z rises (.9946, .9977, .9978).
  - In NeuroTycho the real z itself falls (findings, PRIOS bullet), so on this point the result differs. Human propofol ECoG (PRIOS) shows monkey B's pattern: real z does not fall and the surrogate rises more.
- **Ceiling [V]:** 1 − z² is about .01 in every state.
  - **Post hoc, descriptive:** within-array pairs carry the ceiling (W1 median z .997). Cross-array pairs sit at .965 in W1, .992 in D and .991 in W2.
  - **Reference-made pair structure [V], post hoc:** after the joint common average, within-array pairs are positively correlated and cross-array pairs negatively, at about equal size. Median r within/cross is +.46/−.45 in W1, +.78/−.75 in D and +.72/−.69 in W2, and 91–97% of within-array pairs are positive against 0–6% of cross-array pairs; every 10th window was checked.
  - This is the signature of a common average over two arrays carrying different signals: each channel is dominated by ±(A − B)/2. So much of the |r| spread, and with it the ceiling, is made by the reference.
  - It also explains why |r| rises under the drug after the common average (W1 .45, D .76) while it fell as recorded (screen: .69 to .49).
  - Array geometry (a 4 × 4 mm array) may add to the ceiling [I]. The as-recorded secondary will separate the two; a per-array common average would be the natural reference, but it was not preregistered.
- **Secondaries [V]:**
  - D windows with burst index ≥ .85: Δ_D −.0018.
  - At W1's median burst index (regression): Δ_D −.0017.
  - So burst-like D windows do not drive it.
  - Cross-array pairs: Δ_D −.0022 [−.0035, −.0017]. Within-array pairs: Δ_D −.0016 [−.0024, −.0013].
  - W1 drift (later − earlier half): −.0005.
  - Within-state Spearman(residual, burst index): W1 +.34, D +.05, W2 +.21.
- **Bearing:** by the frozen rule this result counts neither for nor against NeuroTycho; it shows only that the sign agrees. Monkeys C and P would be read under the same rule if their LFP becomes available.
- **Pending secondaries (none can change the verdict):**
  - as-recorded reference: done, see "Secondary result: as-recorded reference" below;
  - raw, all channels, k = 10 and kshift: Gadi.

## Clarification 9 (26 November, after the primary result was read): everything here is post hoc

None of this changes the frozen verdict.

- **k standardised at 4:** the k = 10 secondary of Clarification 8 is dropped and not run. kshift stays.
- **The ceiling rule was miscalibrated [V]:**
  - NeuroTycho's own 10-s normal-scored awake eyes-closed z medians are .967–.983 across its ten agent × animal files (`results/.../zns/zns_*.npz`). Most are above the .97 cut, so the rule would have flagged NeuroTycho too.
  - The rule's purpose holds: a z near 1 leaves little headroom, so an absolute Δz is not comparable across levels. But a fixed cut on z does not serve that purpose, so future preregs should not reuse it.
- **Headroom-scaled reading (Fisher atanh of real and surrogate z, then residual) [V numbers; the scale is a heuristic]:**
  - Monkey B: Δ_D −.33. NeuroTycho drug − awake per file: −.08 to −.38 (KTMD −.10 and −.13).
  - On this scale monkey B's change is NeuroTycho-sized. But atanh's slope at z .995 is about 5× its slope at .975, the pairs are not independent samples, and W2 is lower still (−.42 against W1), so this is not evidence of a drug effect.
- **Per-array common average, on the same accepted windows (`ferraina/select_pa.py`):** each array is referenced to its own mean and its last channel dropped (96 and 192), leaving 190 channels.
  - Median r is −.04 to +.00 both within and across arrays in every state (45–51% of pairs positive), against ±.45 to ±.78 under the joint average. So the joint average's pair structure was made by the reference, and most within-array covariance is a component shared across each array.
- **Runs added (Gadi):** `pa` (per-array, 32 channels), `pa_kshift`, and `paall` (all 190 per-array channels).
  - The full-system run replaces the joint-average all-channel secondary, which would inherit the reference-made structure.
  - Random 32-channel subsets are rescored from `paall`'s saved per-pair MI, with no extra compute.
  - Declared runs kept: `ns_kshift` and `raw`.
- **KSG implementation:** `claar/surr_pairs.ksg` now uses a KD-tree search. Its output is identical to brute force [V]:
  - synthetic pairs, including tied values, at w 0–100: maximum difference 0;
  - stored primary z and per-pair MI reproduced exactly on windows from every state.
- **Seen while smoke-testing:** one D window's `pa_kshift` z, real .981 and surrogates .991–.992. No other per-array or kshift value was read before the runs.
- **Gadi (`jobs/ferraina_sec.pbs`, 48 cores):**
  - Job A runs `ns_kshift`, `raw`, `pa` and `pa_kshift`: expected 0.07 kSU, worst 0.2.
  - Job B runs `paall`: expected 0.3 kSU, worst 1.0.
- **Submitted 26 November:** A is 185276259, B is 185276261 (`afterok` on A).
  - Snapshot: `operations/sources/mi-ferraina-26a3290e`.
  - Inputs: Scratch `atlas/ferraina/sel/` (md5 matches local).
  - Outputs: gdata `representation/atlas-mi-pair-261117/ferraina/fb_z_secA.npz` and `fb_z_secB.npz`.

## Secondary result: as-recorded reference (26 November)

Declared secondary (`asis`, same 32 channels and windows, no re-referencing); output `ferraina/fb_result_asis.txt`. It cannot change the frozen verdict.

- **Rule applied to it [V]:** all four criteria hold, so it reads PRESENT (PRESENT takes precedence over the ceiling clause, as frozen).
  - Δ_D −.0013 [−.0016, −.0011]; −8.4 W1 block-null SD (null SD .0002); 10/10 D blocks below the W1 mean.
  - Δ_rec +.0006 [+.0004, +.0010]: partial reversal; W2 − W1 −.0007.
- **Level and mechanism [V]:** median real z .9981 / .9971 / .9984 (W1 / D / W2); surrogate .9983 / .9986 / .9991. Unlike the joint common average, the real z falls under the drug and returns, as in NeuroTycho; the surrogate also rises a little.
- **Size:** smaller in absolute terms than the primary (−.0013 against −.0018) and about one-tenth of NeuroTycho's −.015. It passes because W1 is far more stable as recorded (residual SD .0005 against .0035), so the block null is narrow.
- **Caveats:**
  - The ceiling is higher still (1 − z² ≈ .004 in W1), so the absolute Δz is not comparable to NeuroTycho's.
  - The recording reference is undocumented. If it is a shared electrode, every channel carries one common signal, and the high z is that common signal [I]. The drug raises local LFP power (W2 is still 2.4× W1), which would lower the reference's share and with it |r| (screen: .69 to .49) and the real z, without any change in how cortical sites depend on each other [I]. The data cannot separate this mixing-ratio change from a change in dependence character.
  - So this is the NeuroTycho-like pattern, but under the reference least able to show it is cortical. The per-array common average (`pa`, `paall`, Gadi), which removes a shared reference signal, is the test that can.

