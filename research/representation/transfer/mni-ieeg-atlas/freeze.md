# Freeze record: MNI Open iEEG Atlas, human NREM sleep (27 November 2026)

Transfer protocol Tier 1 record for the sleep test. Rules, verdict and deviations: [`atlas/mi_pair/preregs/sleep-prereg.md`](../../atlas/mi_pair/preregs/sleep-prereg.md). Committed before any N2 or N3 window was processed.

- **Source:** MNI Open iEEG Atlas (Frauscher et al. 2018, Brain; von Ellenrieder et al. 2020, Ann Neurol), processed 200 Hz clips, NEMAR BIDS derivative nm000362 v1.0.2, CC-BY-NC-4.0, DOI 10.82901/nemar.nm000362; files from NEMAR's public S3 remote, SHA-256 checked (`mi_pair/sleep/fetch.py`). Origin portal account-gated (research-only terms); not used.
- **Units:** patients. 91 have wake, N2 and N3 (10-min); 65 pass inclusion (≥ 8 common channels, ≥ 4 wake windows), from metadata.
- **States:** baseline wake (60 s, eyes closed, daytime standardised recording); target N3 (primary) and N2 (secondary), up to 10 min each from the first sleep cycle, source-scored. Epochs = the source's artifact-free segments; windows never cross a segment boundary.
- **Nuisance contrast:** none in these data. Gate 4 not applied; the result lacks a specificity check of its own.
- **Channels and QC:** bipolar derivations judged normal by the source; those present in wake, N2 and N3; > 32 → 32 by name, seed 20261127 + patient number. No amplitude artefact rule (source cleaned visually).
- **Window:** fs 100 Hz (0.5–40 Hz, atlas long arm, per segment); L_max = longest segment (wake ≤ 60 s, NREM ≤ 600 s); long T = 1000; short T not used (secondaries out of scope).
- **τ and T/(2τ):** τ median 5.3 samples (IQR 4.0–6.9); T/(2τ) ≈ 94 at T = 1000.
- **KSG subset seed:** 20261127 + patient number, applied only where more than 32 common channels exist.
- **Baseline reliability (wake, T = 1000):** G .915, K .761, so `mi_g|ksg` is testable. Wake real z .903, not at ceiling. Null SD .0067 raw. Secondaries: not computed (prereg, Deviations).
- **Predicted sign:** none for the test (not anaesthesia). The macaque prediction (negative, ≤ .25 of the drug effect) is the hypothesis under test.
- **Adapter:** `atlas/mi_pair/sleep/` at the freeze commit (statistic from `atlas/mi_pair/edges.py`, as the human PRIOS run; not `axis_screen/stats.py`, since only the MI pair is computed).
- **pyspi:** not used.
- **Compute:** Gadi `normal`, 48 CPUs; expected about 130 SU, worst case 960 SU (10-h walltime). Data download on `copyq` (job 185368663), under 8 SU.
