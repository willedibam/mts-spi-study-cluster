# Repeated lead–lag asymmetry in NeuroTycho and mouse EEG

At 20 ms, ketamine increased the absolute, repeatable asymmetric part of the lagged edge profile on 4/4 macaque dates. Propofol decreased it on 4/4 and ketamine–medetomidine on 11/11. Yet its *share* of repeatable edge variation fell under all three agents on average. Both single-agent arms cover the same two animals. In Chibi the ketamine increase was 0.00169 squared correlation units and in George it was only 0.00006. This is an exploratory drug contrast with limited animal replication. [Deco et al. (2022)](https://doi.org/10.1038/s42003-022-03505-7) already compared forward and reverse lagged functional connectivity on these macaques. The present analysis separates the recurring signed asymmetric edge pattern from the relative share assigned to it.

For the upper-triangular dyads of the lagged Pearson matrix C, let S = (Cij + Cji)/2, centred across dyads, and A = (Cij − Cji)/2. The even and odd profiles are orthogonal when both orientations are restored. On one window the asymmetric fraction is exactly (1 − corr(C, Cᵀ))/2 on ordered off-diagonal entries. Its numerator measures the average squared difference in orientation up to the factor of four. For the repeatable version, replace each mean square by the average inner product per dyad between **distinct windows** of the same date and state. Reversing one window flips A and preserves S. Independently flipping window orientations therefore has zero expected cross-window asymmetric energy, even with the original per-window asymmetry. Negative finite-sample cross-products are retained. This estimates consistent timing over the sampled windows and depends on recording filters, montage, chosen lag and spectrum.

## Primary macaque contrast

Mean of within-date contrasts to eyes-closed wake, equal weight per date. The first two columns are distinct-window energies in 10⁻³ squared Pearson units. The third is the difference between state-specific shares, averaged after forming each date's contrast. The last column provides a wake-state comparison for the share.

| State | Asymmetric energy | Symmetric energy | Asymmetric share (pp) | Eyes-open share (pp) |
|---|---:|---:|---:|---:|
| Ketamine | +0.877 | +49.514 | −5.40 | −5.12 |
| Ketamine–medetomidine | −0.751 | +15.965 | −4.53 | −1.74 |
| Medetomidine | +0.747 | +28.387 | −4.23 | −2.94 |
| Propofol | −0.903 | +9.032 | −4.19 | −1.97 |
| Sleep | −0.587 | +6.559 | −6.26 | −5.14 |

The 32-channel common-average and bipolar checks covered 1,260 ten-second windows on 30 dates. The primary lag and sensitivity lags (10, 40 ms), montage and wake/sleep/recovery comparisons were chosen before examining the scores. There was no registered inferential threshold or held-out animal discovery split. Sleep dates had 30 windows per sleep state and wake dates generally 10 per wake state. Absolute asymmetric energy relative to eyes-open wake still rose on 4/4 ketamine dates and fell on 4/4 propofol dates. The mean share during recovery was 5.64 pp below eyes-closed wake for ketamine and 3.29 pp below for propofol. Within-date state contrasts do not remove vigilance and time-of-session effects.

The post hoc all-channel check preserved the 4/4, 4/4 and 11/11 directions above. In the second post hoc check each normalised channel had the same Fourier amplitudes in every window of a date (root-mean-square awake amplitude by frequency and channel), with the observed phase retained. Ketamine's change in asymmetric energy was +2.12 (3/4 positive) in the table's units, the combination −0.54 (11/11 negative) and propofol −0.55 (4/4 negative). Ketamine's mean share change switched to +4.79 pp. The common relative decrease partly depends on frequency weighting. Because this reconstruction also removes frequency-specific cross-channel amplitude fluctuations it does not isolate a unique spectral mechanism. The absolute combination contrast became slightly positive at 40 ms in the original primary montage. Exact conditioning on each window's entire cross-spectrum would largely preserve the target by construction.

## Mouse check

In 17 mice from DANDI 000458, the same 20-ms readout was applied to the previously selected stimulus-free awake and sustained isoflurane windows. Use the first eligible date per animal. This check was exploratory after inspecting the macaque result. Under common-average reference the mean asymmetric share was 5.48% awake and 1.45% under isoflurane (16/17 down). Late minus early wake averaged +1.65 pp. Mean absolute asymmetric energy fell 0.00341 → 0.00097 (16/17 down). Under the acquisition reference, share decreased in 17/17, absolute energy in 11/17. The sustained state includes burst suppression. This result supports a relative difference during that state; its absolute magnitude depends on reference. The non-bursty induction replication in the manuscript is for MI.

## Reproduction

Run from this directory with the repository Python environment. Use the pooled bank at Gadi Scratch under `mts-spi-study/atlas/banks/neurotycho-long` and the selected mouse windows produced by the [Claar extraction and selection scripts](../mi_pair/claar/) using the awake artefact scale. All calculations here ran locally. No pyspi corpus was extended.

```sh
python analyze.py "$NT_BANK" state_scores.csv
python mouse.py "$CLAAR_SELECTED" mouse_scores.csv
python plot.py state_scores.csv ../../../../drafts/paper_character_261118/figures/lag-reciprocity
```

NeuroTycho input SHA-256 values were `837adabc8b776c8ff8e2e0b043502b1754e7f0fe06830a2ec8c0018d03717896` (manifest) and `f2761fb9394702862b5c00757e5155930be0c0fcfdd4b9891137f33c821c1967` (windows). The two compact CSVs contain all per-state estimates including recovery, wake controls, sensitivities and energy terms.

For the MI figure, the ten recomputed per-window edge arrays are on Gadi gdata under `mts-spi-study/representation/atlas-mi-pair-261117/zns`. Run `python ../mi_pair/plot_character.py "$ZNS" ../../../../drafts/paper_character_261118/figures/mi-character`. All ten hashes matched the repository list; observed and surrogate z were independently recomputed from all stored normal-scored edge profiles to within 10⁻⁶. The filtered mouse induction and human ketamine scorers reproduced their archived contrasts.
