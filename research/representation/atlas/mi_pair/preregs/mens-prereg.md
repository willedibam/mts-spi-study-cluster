# Prereg: surrogate-ensemble KSG MI per pair (26 November 2026, frozen before any output)

**Why.** Δz uses one IAAFT draw per window, so its size mixes the drug effect with the draw's noise and with the spread of G across pairs (`kens-result.md`, "Why the dissociation"). Kurtosis was a proxy for the MI excess and failed as a readout. This run measures the excess directly against a 50-draw null, per pair, and puts Δz in null-SD units.

**Code.** [`../mens.py`](../mens.py); job [`../jobs/mens.pbs`](../jobs/mens.pbs). Same windows, 32-channel subsets, normal scoring, KSG (k = 4, AUTO Theiler) and Gaussian MI as `edges.py recompute`. B = 50 spectrum-final joint IAAFT draws per window: draw 0 seeded k as in `zns`, draw b > 0 seeded [k, b] as in `kens.py`.

**Gate.** For every window, draw 0's Fisher z of r(G, K), real and surrogate, equals the stored `zns` value within 1e-4. Otherwise nothing is read.

## Readouts per window

- `dz_ens`: (atanh r_real − mean_b atanh r_b) / sd_b. Δz in null-SD units.
- `dz0`, `dz_raw`: the single-draw Δz and the 50-draw-mean Δz, both in Fisher z.
- Per-pair excess z_ij = (K_ij − mean_b K_b,ij) / sd_b K_b,ij, summarised over the 496 pairs as:
  - `lvl` = mean (level);
  - `scat` = SD (scatter);
  - `exc` = share with |z| > 1.96 (nominal .05 under the null; two-sided; normal approximation, because rank p-values from 50 draws cannot pass FDR over 496 pairs);
  - `rho_eG` = correlation of z_ij with G_ij (whether the excess lines up with linear coupling, which is what Δz responds to).

Every readout is taken as window minus its session's eyes-closed mean, in eyes-open SD units, as for panel (c). Contrasts use sessions, then animal means.

## Decisions

- **Q1. Does the drug fall survive the ensemble null?** REPLICATES if `dz_ens` animal means are below 0 in 4/4 animals with one-sided t(3) p < .05. Otherwise the single-draw Δz effect is reported as draw-dependent.
- **Q2. Sleep against drug (equivalence, not a non-rejection).** For `dz_ens`, report the sleep session mean with a 90% t CI, and the ratio of sleep to drug session means with a 90% bootstrap CI (10,000 resamples of sessions).
  - "Sleep is small relative to drug" if the ratio's 90% CI lies inside [−.25, .25]. This is the TOST at a margin of a quarter of the drug effect, fixed here.
  - "Sleep equals wake" is not tested, and the wording will not be used.
  - Sessions, two animals: no animal-level claim.
- **Q3. Which part of the excess moves** (descriptive, no verdict). Report the sign of `scat`, `exc` and `rho_eG` per agent and animal.
  - `lvl` is not read for its sign: `ksg-bias-result.md` documents a real-only negative KSG bias under drug, which shifts the level and not the alignment.
  - Before the run, the local `lvl` (mean K over a single surrogate's K) by agent was KT +4.6, KTMD −2.7, PF −4.6 SD. So a mixed-sign level is expected and is not evidence against the excess.

## Not decided here

- The drug-against-sleep test at matched delta. The bank's overlap is thin (`findings.md`, "Not sleep depth"), and it needs denser sleep sampling with EMG/EOG staging from the archives. That belongs in a separate prereg, run on windows not yet used.

## Cost

- Gadi `normal`, two jobs (stage A, stage B), each 48 cpus, 96 GB and a 10-h walltime.
- About 51 KSG passes per window. The `zns` recompute was 4 passes for 60 SU, so the expectation is about 0.75 kSU in total (about 4 h per job).
- Worst case 1.9 kSU, both full walltimes.
- Outputs: `mn_*.npz` with real and per-draw G and K (about 250 MB), on gdata at `.../atlas-mi-pair-261117/mens/`.
