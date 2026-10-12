# Electronic-oscillator pilot

Question: which relationships between dependence measures change along a measured coupling sweep, and can those changes be explained through the underlying channel pairs? This is an exploratory physical application, not a classification benchmark or confirmation of a thermodynamic phase transition.

## Scope

- Start with the hybrid ER dataset [17451096](https://zenodo.org/records/17451096); use the companion small-world dataset [17451530](https://zenodo.org/records/17451530) for transfer if the first pilot is informative. Both report 28 channels, 500,000 samples at 50 kHz, and 701 coupling settings; inspect actual files before relying on these dimensions. Source: [Vera-Ávila et al. 2026](https://doi.org/10.1016/j.dib.2026.112785).
- First inspect nine evenly spaced coupling settings, preserving all channels and the full recording duration. Test anti-aliased polyphase resampling to T=2,000, 5,000 and 10,000. Choose the smallest adequate T; no raw striding, silent truncation or envelope-only substitution. If no full-duration option passes, reassess justified shorter windows or abandon this bank.
- Assess per-channel spectra and waveform reconstruction, phase-coordination traces and edge profiles, amplitude-envelope edge profiles, and synchrony fluctuation/dwell behaviour. Numerical thresholds are screening tolerances, not universal guarantees of SPI equivalence. Validate representative accepted observations at the next higher sampling resolution with p90 where feasible.
- Begin extraction with two full-p90 observations and measure memory/runtime before a broader farm. The initial sweep is at most 29 evenly spaced settings per topology; refinements must resolve an observed sampling or interpretation question. No large biological datasets or channel-pair sampling work in this pilot.
- Centre eligible SPI–SPI coordinates without unit-variance scaling; use the ordered-edge Pearson contract. Missing/constant-feature handling must be explicit and fitted on development observations. Orient PCs by their largest absolute loading, without coupling or physical-reference labels. Treat PCs and embeddings as exploratory geometry; do not manufacture classes from coupling settings.
- Show the coupling trajectory, an independent physical coordination summary, eigenvalue spectrum, and compact boxed square galleries ranked by absolute PC1 loading. Retain the full model; optionally show a separate gallery excluding pairs with two classical-correlation endpoints. Inspect original MPI edge scatterplots for a few interpretable contrasts. Report feature coverage and loading stability; separate spectral sensitivity from mechanistic specificity.
- Split by physical recording/settings or topology, never claim windows from one sweep are independent experiments. With only one sweep per topology, validation is transfer/sensitivity evidence, not population replication.

## Execution and continuation

Authorised 2026-10-12: download directly to Gadi, run the scoped pilot, and initiate justified follow-up runs overnight. Keep the notebook lean and figures consistent with the extended order-coordinate notebook. After a job is confirmed running, check about every two hours, remain quiet on unchanged state, and stop the follow-up when the pilot is analysed or a material decision is needed.

Live allocation: 56.51 kSU available; gdata 36.17/100 GiB and 30.52k/70k inodes, Scratch 318.60 GiB and 193.75k/202k inodes. Prefer `/g/data/ql44/we2614/mts-spi-study/representation/empirical/hardware-pilot-261012`. Download selected ZIP members with checked HTTP byte ranges and CRC rather than expanding both large archives. Main cluster checkout is clean at 247b9e1, pyspi v3 clean at 65317c9; deploy committed pilot code to a separate immutable shared source checkout. Do not modify other runs.

Initial access checks: Zenodo times out from local urllib/curl and Gadi data-mover curl; local Requests additionally reports a missing CA issuer. No scientific data or p90 job has yet been obtained/submitted. Establish a verified public download route before choosing T or claiming acquisition success.
