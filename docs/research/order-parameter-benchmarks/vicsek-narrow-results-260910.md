# Vicsek narrow-control physics results, 2026-09-10

## Decision

Retain Vicsek as a promising fourth candidate alongside Kuramoto, Miller–Huse and Stuart–Landau. The completed cohort reproduces a pronounced finite-size order contrast with fresh seeds. It does **not** establish a thermodynamic jump, a precise critical noise, or any SPI–SPI recovery result. No new p90 was run. The defensible next experiment, if requested, is a small SPI pilot using these existing observations and measured window-level global order. More physics simulations are not needed merely to refine the critical point for this objective.

## Design and results

Frozen [protocol](vicsek-narrow-protocol-260910.md): angular-noise Vicsek model, N=32768, periodic L=128, density2, speed.5; unit interaction radius and timestep. Each case uses20000 burn steps followed by100000 microscopic recording steps, sampled every25 steps, giving4000 saved observations. Four noise values × two seeds (910204/910205) × ordered/random starts =16 cases. The physical order trace is phi(t)=|N^-1 sum_i exp(i theta_i(t))|.

| Noise eta | Mean of four run means | Range of run means |
|---|---:|---:|
| .460 | .344207 | .313385–.355815 |
| .468 | .226911 | .192954–.264902 |
| .476 | .117033 | .101994–.144069 |
| .484 | .077299 | .075200–.079430 |

Every seed/start curve decreases monotonically. The aggregate drop is77.54% over this interval. Three curves have their steepest sampled fall over.460–.468, and one over.468–.476. This brackets a useful finite-run contrast, not an exact phase boundary. These ranges are **not confidence intervals**: there are only two seed labels, and matched controls/starts are not additional independent seeds.

At eta=.468, seed910205/random has consecutive12500-step block means .3325,.3182,.3188,.3033,.1062,.1284,.1308,.1622. Its first/second-half means differ by about.1863. Other runs also move between lower- and higher-order episodes, sometimes in the reverse direction. Start-paired whole-run means differ by up to.0402 at this noise. A single control-conditioned mean compresses scientifically relevant temporal variation; block fluctuations alone do not prove nonstationarity, bistability or phase coexistence.

The inspected time-occupancy histograms at.468 have two visible peaks, near phi≈.1–.15 and≈.33–.35, with different occupancy across runs. This strengthens the descriptive evidence for lower-/higher-order episodes but is not a stationary-distribution or thermodynamic-coexistence proof. See [replicate curves](../../../data/order_parameter/vicsek_observation_260910/narrow-analysis/replicate_curves.png) and [block means and distributions](../../../data/order_parameter/vicsek_observation_260910/narrow-analysis/blocks_and_distributions.png).

All recorded Binder estimates are positive (minimum about.1404). Neither this nor a steep four-point curve establishes or refutes the published asymptotic transition. This long/narrow cohort has only one physical N, limited seed replication and finite histories. The reference physics is Chaté et al., [PRE77,046113 (2008)](https://arxiv.org/abs/0712.2062); see the earlier [scout discussion](vicsek-pilot-results-260910.md) for finite-size caveats.

## What the small observations can and cannot show

Existing archives contain nested M=8,16,32 views. Raw particle channels use either dispersed tracked IDs or IDs selected in an initially local group. The latter move apart and are **not** a permanently contiguous spatial patch. Fixed spatial bins provide genuinely dispersed/contiguous spatial views, but measure particle aggregates: mean occupancy32 particles per width4 bin here. Density and x/y current are separate observation families; each vector component is a separate M-channel arm, not a hidden doubling of M.

Keep both observation interpretations explicit. No sampling layout is already proven superior. A tiny raw subsample's polarization magnitude has a positive finite-M floor, documented in the initial scout. This does not imply the SPI–SPI representation is unable to infer order, but rules out assuming an unbiased instantaneous global measurement from M=32.

The proposed pilot should freeze observation/reduction choices without Q or control labels and evaluate association with window-level full-system phi, including within-noise variation. Nested windows and views must stay grouped by master trajectory for evaluation; they are not independent examples. T=100,500,1000 corresponds to2500,12500,25000 microscopic steps at this stride. Such a test concerns window-level tracking, not instantaneous Q(t) recovery. The existing two seeds are suitable for exploration, not a large independent confirmation claim. Nonconstant channels alone do not constitute p90 validity.

Across all16 archives, M={8,16,32}, T={100,500,1000,2000} prefixes and all ten observation/component views, the constant-channel fraction was0 using SD<=1e-8 as the constant threshold (1920 view checks). This verifies basic dynamic variation, not information sufficiency, numerical validity of every SPI, or comparability of the eventual z coordinates.

## Execution and provenance

Job178628828.gadi-pbs completed exit0 after56m11s walltime; all nine tests passed. It requested16 CPUs/64GB, used peak1.43GB, and cost29.96 SU. Its aggregate CPU time13h51m08s is not elapsed runtime. The total across failed smoke, successful smoke, long gate and this refinement is37.17 SU; see the [ledger](overnight-260910.md). The slow portion was particle dynamics, followed by a stalled/retried download, not p90. No simulation remains running from this task.

All16 raw archives were downloaded and passed complete-cohort, source/config hash, metadata, finite-array, observation-shape and recomputed-mean checks. The final local regression run passed11 tests (Vicsek simulation/observation and Stuart–Landau temporal tests). Both analysis scripts ran successfully; the replicate and distribution figures were visually inspected. Audit outputs include `validation.json`, `verified_replicates.csv`, `control_summary.csv`, `start_sensitivity.csv`, `observation_checks.csv`, and `summary.json`.

Pinned commit: `eade8c2e0bb54b5cd00510289cc84de6a72063f0`, branch `codex/vicsek-narrow-260910`. This isolated published-base branch avoids pushing unrelated unpublished main commits. Numerical source SHA256: `96352f7189fbfb737fc6c08c5ac7f66b9995b4a37654d3dc24342495692f8dbc`.
Configuration SHA256: `47f2e734cc160042462d47dccc0b0984069828160a43f18c9a736f58e55d9be1`.

The overnight reminder `order-parameter-boundary-experiments-overnight` was deleted through the app at the user's request. No further scheduled checks, new simulations or SPI computations are implied by this report.

## Reproduction

Raw archives: `data/order_parameter/vicsek_observation_260910/gadi-narrow-eade8c2/`.
Analysis outputs: `data/order_parameter/vicsek_observation_260910/narrow-analysis/`.

```sh
.venv/bin/python scripts/plot_vicsek_replicates.py \
  --input-dir data/order_parameter/vicsek_observation_260910/gadi-narrow-eade8c2 \
  --config configs/scout/vicsek-narrow-physics-260910.yaml \
  --source-sha256 96352f7189fbfb737fc6c08c5ac7f66b9995b4a37654d3dc24342495692f8dbc \
  --output-dir data/order_parameter/vicsek_observation_260910/narrow-analysis
.venv/bin/python scripts/analyze_vicsek_observation_scout.py \
  --inputs data/order_parameter/vicsek_observation_260910/gadi-narrow-eade8c2 \
  --output-dir data/order_parameter/vicsek_observation_260910/narrow-analysis
```
