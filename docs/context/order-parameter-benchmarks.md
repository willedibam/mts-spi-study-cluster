# Canonical order-parameter benchmarks

## Decision

- Primary benchmark: deterministic all-to-all Kuramoto, with the full-system phase coherence `R_N(t)` hidden from SPI--SPI and only a random `M`-oscillator scalar observation exposed.
- Spatial generalization: scout Miller--Huse, a deterministic 2-D chaotic CML with Ising symmetry and order coordinate `|m|`; do not assume a small patch is informative until measured.
- Keep the quadratic Kaneko CML as exploratory regime discovery, not the canonical order-parameter proof.
- Stop production work on the `k=3.1` sine-circle branch. The printed recurrence does not preserve the paper's stated `theta<1/2` absorbing set under conventional modulo, and direct simulation failed to reproduce its reported absorption. Resume only with a reproducible reference implementation or independently verified convention.

Primary sources: Kuramoto review, <https://doi.org/10.1103/RevModPhys.77.137>; finite-frequency-sampling effects, <https://doi.org/10.1103/PhysRevE.92.022122>; sine-circle inconsistency source, <https://arxiv.org/html/nlin/0210034v1>; Miller--Huse model, <https://doi.org/10.1103/PhysRevE.48.2528>.

## What the Kuramoto experiment establishes

- `K` is the control parameter. `R_N=|N^-1 sum_j exp(i theta_j)|` is the independently defined canonical order parameter.
- A monotone latent curve along one `K` sweep can still be control-parameter decoding. It supports only “tracks progression through the synchronization transition and covaries with `R_N`.”
- Realization-level latent--`R_N` association after removing each `K`-cell mean breaks that simple confound.
- Also test against `R_{N-M}`, computed from the hidden complement of the observed channels. This removes the small mechanical self-inclusion path present because `R_N` contains the observed `M` oscillators.
- Strong claim-bearing evidence requires a representation/calibration frozen on one frequency-distribution path and evaluated on unseen master seeds, control values, and a second smooth unimodal distribution with overlapping `R_N`.
- Gaussian-width variation is only a scale-invariance check: after rescaling time, fixed-shape dynamics depend on `K/sigma`. Use a different distribution shape for genuinely different microdynamics.
- The representation is unsupervised; any fitted map from latent coordinate to `R_N` is supervised calibration and must be described separately.

## Required generator semantics

- Mean-field `O(N)` coupling and RK4; separate `N_full` from observed `M`; no dynamical or measurement noise in the primary result.
- Observe `cos(theta)` with a common carrier frequency so locked channels keep oscillating. Add `sin(theta)` and fixed per-sensor phase offsets only as observation-map ablations; avoid wrapped phase as a scalar input.
- Save `R_N(t)`, observed-subset `R_M(t)`, hidden-complement `R_{N-M}(t)`, frequencies, observation indices, initial/final states, seeds, integrator settings, and frequency-sampling scheme. Full clean phase movies are useful for small validation runs but optional in production once these quantities are computed directly. For the claim-bearing run, also save a subsequent hidden-only `R` window (`future_truth_T`) while exposing only the first `T` samples to SPI. SPI sees none of the hidden artifacts.
- Generate paired master realizations across `K`; split and bootstrap by master ID. Also retain a smaller independently resampled-per-cell validation so pairing cannot manufacture smooth trajectories.
- Generate long masters and derive nested `T` and `M` views. Dataset identity must keep every view of one master in the same evaluation split.

## Physics scout before pyspi

- Gaussian first: `N in {128,256,512}`, `kappa=K/Kc in {.6,.8,.9,.95,1,1.05,1.1,1.2,1.4,1.6}`, initially 32 master seeds. For unit-variance Gaussian frequencies, `Kc=sqrt(8/pi)` only in the infinite-`N` limit.
- Start `dt=.02`, sample every `.1`, and compare `.02` with `.01` below/near/above the transition. Numerical differences must be negligible relative to realization variation.
- Record 4,000 samples after an empirically validated burn and analyze nested `T in {250,500,1000,2000,4000}` and `M in {8,16,20,32,64}`.
- Select the smallest `N` whose doubling does not change the intended conclusion; the smallest `M` for which `R_M` contains useful but imperfect information about `R_N`; and `T` only after both physical and SPI--SPI feature convergence checks.
- Treat 32 seeds as a starting point. Check uncertainty under 16/32/64 and allocate extra seeds where `R_N` changes, rather than uniformly oversampling trivial endpoints.

## Representation test and controls

- Use `configs/pyspi/benchmarked_p90.yaml`, one dataset per CPU and no internal pyspi parallelism.
- Freeze the unsupervised transform without target access. Pre-specify PC1/first nontrivial coordinate, or select on discovery data and freeze before evaluation; searching dimensions for maximum target correlation is supervised selection.
- Use the established Pearson correlation-of-SPIs meta-feature and a discovery-fitted linear PC1 first. PCA10 plus Isomap is unnecessary for this benchmark unless the pre-specified linear coordinate fails; any later coordinate search is model selection and needs untouched confirmation data.
- Primary natural coordinates are raw and clean. Channel z-scoring and fixed-scale observation noise are paired mechanism/robustness ablations, not default preprocessing.
- Compare with `R_M` (a phase-oracle comparator, not an input-only baseline), mean absolute input correlation, covariance leading-eigenvalue fraction, analytic-phase coherence from the cosine observations, basic marginal/temporal summaries, raw-correlation PCA1, the best individual SPI selected on training only, and a control-only `kappa -> R` calibration. A generic representation need not beat purpose-built `R_M`; it must add defensible model-agnostic information.
- Run all SPIs and an ablation excluding explicit phase/synchronization SPIs. Constant/failure patterns must not become synchronized-regime labels.
- Report overall and within-control association, seed-clustered uncertainty, held-out cross-path calibration error/data collapse, matched-`R`/different-path and matched-`kappa`/different-`R` contrasts, and `M x T` degradation. Use staggered logistic control values so transfer is not a lookup on the Gaussian grid.

## Frozen claim benchmark

- The gated production config contains 880 `M=20,T=1000` datasets: 32 paired masters per Gaussian/logistic path plus eight independently resampled realizations per control cell. Gaussian masters 0--15 are development-only; Gaussian masters 16--31, every logistic master, and all independent-cell samples are evaluation-only.
- The canonical primary target is the disjoint-future full-system `R_N`. Disjoint-future `R_{N-M}` is the anti-self-inclusion sensitivity target; current/future and full/complement reliability are reported together.
- Gaussian uses `kappa in {.6,.8,.9,.95,1,1.05,1.1,1.2,1.4,1.6}`. Logistic uses `{.7,.8,.875,.925,.975,1,1.025,1.075,1.15,1.2,1.4,1.5}`: most controls are staggered, while four shared anchors permit matched-control contrasts.
- Freeze SPI validity, meta-feature variance filtering, missing-value imputation and PC1 on Gaussian development masters only. Evaluation missingness must pass the predeclared max/p95/target-association gate; an evaluation-complete feature set is a target-blind sensitivity, never the primary representation.
- Cross-path collapse is assessed as a conditional path effect in calibrated-`R` units, with a Gaussian split-half noise floor and master-clustered uncertainty. Independent-cell uncertainty is resampled within each control cell, not clustered by reused instance labels.
- The individual-SPI comparator is explicitly the best development-selected scalar summary among predeclared mean, mean-absolute, dispersion and leading-eigenvalue-fraction reductions. It is not described as the universally best individual SPI.
- Store only the observed MTS, compact physical truth and SPI matrices. Full phase movies are unnecessary for production because `R_N`, `R_M`, `R_{N-M}` and future targets are saved directly.
- Each dataset records `K`, continuum `K_c`, reduced coupling `kappa`, canonical/sensitivity truth-array roles, RNG seed scope and shared-master ID, resolved generator parameters, experiment and pyspi config hashes, and clean generator-code commit. Downstream analysis validates these fields rather than inferring the design from directory names.

## Validated physics scout (2026-08-22)

- Gaussian job `177017823`: 960/960 completed; logistic job `177018021`: 320/320 completed; timestep job `177017824`: 96/96 completed.
- `N=256` retains finite-size variation but gives a stable macroscopic curve. At `M=20`, within-`kappa` Spearman between `R_M` and `R_N` is `0.664` (Gaussian) and `0.690` (logistic): informative but far from an oracle.
- For `N=256`, the `T=1000` mean-`R_N` error against `T=4000` has p95 `0.00926` (Gaussian) and `0.00903` (logistic). First/last-block drift has p95 about `0.029` with negligible signed mean.
- Halving the RK4 step from `.02` to `.01` changes paired mean `R_N` by median `5.8e-9`, p95 `9.9e-4`, maximum `.00305`; `.02` is adequate.
- Doubled-burn job `177019359` compared burn `100` with `200` for 160 paired near-onset realizations: signed mean change `7.6e-5`, mean absolute change `.00156`, p95 `.00434`, and Spearman `.99961`. Burn `100` is adequate for this benchmark.
- Therefore `N=256, M=20, T=1000, dt=.02, sample_dt=.1, burn=100` is physically defensible. `T=1000` remains provisional until the p90 SPI--SPI feature-convergence scout passes.
- The feature scout keeps `M=20` primary: it tests `T in {500,1000,2000}` at `M=20`, with `M in {8,32}` only at `T=1000` as a lower/upper spatial-convergence bracket. The `M=32` arm is a 12-dataset diagnostic, not a production observation size. Including the paired `M=20,T=1000` z-score ablation gives 72 datasets total.
- Before inspecting the completed scout, the operational upper-bracket gate was fixed as follows: `M20,T1000` versus `M20,T2000` must have latent-rank Spearman at least `.90`, geometry Spearman at least `.85`, and median feature-vector correlation at least `.90`; `M20` versus `M32` at `T1000` must reach `.85`, `.80`, and `.85`, respectively. These are representation-stability thresholds, not critical-scaling criteria.
- Corrected p90 pilot `177019354`: 6/6 `M=20,T=1000` datasets completed with 289 SPIs; median pyspi time `603 s`, maximum `774 s`, and 233 SPIs were finite/nonconstant in all six. The exact-GP additive-noise-model SPI was removed after cancelled pilot `177018028` spent more than 18 CPU-minutes per dataset inside that SPI alone. This pilot validates execution and filtering, not the order-parameter result.
- Feature-scout corner gates passed 6/6: `M=20,T=2000` job `177020665` took `1396--1716 s` per dataset and peaked at 13.6 GB across three tasks; `M=32,T=1000` job `177020346` took `1475--1908 s` and peaked at 9.46 GB across three tasks. Use 8 GB/core for these two high-cost classes and 4 GB/core elsewhere. High-synchrony cells were the runtime and failure-count tail.

## Completed frozen evaluation (2026-08-22)

- Feature convergence passed before production. Against `M=20,T=2000`, the `M=20,T=1000` coordinate had rank correlation `.972`, geometry correlation `.961`, and median meta-feature correlation `.947`; against `M=32,T=1000`, the corresponding values were `.951`, `.930`, and `.977`. Production therefore retained `M=20,T=1000`.
- Gadi job `177026144` completed all 880 datasets. Every dataset contains only `timeseries.npy`, compressed `ground_truth.npz`, compressed `spi_mpis.npz`, and `meta.json`; array shapes, finite physical targets, CRCs, provenance, design counts, seeds, paired-master invariants, and config hashes passed the post-run audit. Analysis job `177026906` completed successfully.
- The frozen primary representation failed its predeclared missingness gate: 4/720 evaluation rows had any missing meta-features, p95 missingness was `0`, but the worst row was `.1183` against the `.05` maximum. The no-explicit-phase-SPI representation also failed on one row (`.0798`). These failures were concentrated in ill-conditioned estimators at high synchronization. The gate is binding despite being sensitive to very few rows.
- Secondary association was strong. SPI--SPI PC1 had overall/within-`kappa` Spearman `-.945/-.689` on held-out Gaussian masters and `-.915/-.596` on the logistic path. Within-`kappa` cluster-bootstrap intervals excluded zero on both paths, including when the target was the future hidden-complement coherence. PC1 sign is arbitrary.
- The cross-path result was not superior to simpler summaries. Logistic-path calibration MAE was `.0789` for SPI--SPI, `.0699` for `kappa`, `.0583` for mean absolute input correlation, and `.0676` for the development-selected individual SPI. SPI--SPI minus `kappa` had 95% cluster-bootstrap interval `[-.0089,.0266]`; SPI--SPI was significantly worse than the selected individual SPI (`[.0020,.0209]`).
- The frozen maximum supported claim level is therefore `0/3`. The defensible result is: “A frozen unsupervised SPI--SPI coordinate showed strong held-out association with the Kuramoto order parameter, including realization-level variation at fixed coupling, but failed its preregistered feature-validity gate and did not outperform simpler dependence summaries.” This benchmark does not support the unqualified order-parameter-inference claim.
- Cross-path conditional-gap analysis is descriptive because no equivalence margin was frozen. A future universal/data-collapse claim needs such a margin and independent confirmation; changing the present rule or representation after seeing these results would be exploratory.

## Claim ladder

1. One sweep only: “SPI--SPI learns an unsupervised latent coordinate that tracks the synchronization transition and covaries with the known order parameter.”
2. Within-`K` evidence: “The representation contains a latent coordinate that tracks realization-level variation in an independently defined macroscopic order parameter.”
3. Held-out cross-distribution generalization: “SPI--SPI supports data-driven discovery of an order-parameter coordinate from partial multivariate observations, generalizing across distinct microscopic routes to synchronization.”

Before production results were inspected, the claim rule was frozen: level 1 requires the primary missingness gate and held-out overall association intervals excluding zero on both frequency-distribution paths; level 2 additionally requires within-control full-system and hidden-complement intervals excluding zero on both paths; level 3 additionally requires the Gaussian-trained SPI--SPI calibration to beat the control-only `kappa` calibration on logistic data (the clustered 95% MAE-difference interval lies below zero), while the no-explicit-phase-SPI representation passes its missingness and association criteria. Purpose-built phase or simple input statistics are reported but need not be worse than SPI--SPI.

Do not use the unqualified sentence “SPI--SPI performs unsupervised inference of the order parameter” unless levels 2 and 3 survive the frozen evaluation and baselines.
