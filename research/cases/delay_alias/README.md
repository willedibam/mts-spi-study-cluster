# Delay aliasing (introduction Fig. 1)

Synchronous and delayed Gaussian MTS whose zero-lag correlation matrices are identical entry for entry: the constructed example of [`drafts/introduction_rewrite.md`](../../../drafts/introduction_rewrite.md), paragraph 4.

- Generator: `mtsspi.generators.generate_delay_aliased_mts` (registry name `delay_aliased_mts`). `lag_range=0` is family A; `lag_range>0` is family B; `structure_seed` fixes the shared target matrix.
- `spis/delay_alias.yaml`: the five-SPI pyspi subset (zero-lag Pearson, Spearman, max cross-correlation, min cross-distance at τ=30, coherence magnitude). It runs locally in about 0.5 s per recording at M=20, T=20,000.
- `delay_alias.py`: measurement, summaries, generic-feasibility scan and the Fig. 1 plot.
- `notebooks/delay-alias_261020.ipynb`: executed checks and figures; outputs in `results/cases/delay-alias-261020/`.

## Feasible parameter range

The proposed construction mixes sources with Σ* = Σ ⊘ R, R_jk = ρ(τ_j − τ_k), and needs Σ ⊘ R to be positive semidefinite.

- For a generic target not built from delays, the largest feasible delay range shrinks with M and correlation strength. With evenly spaced delays and AR(1) ρ, it runs from 1.7 correlation times (M=10, mean r 0.09) to 0.11 (M=40, mean r 0.49). At M=20 it is 1.22, 0.52 and 0.20 correlation times for mean r 0.09, 0.25 and 0.49, and population z at that limit is only 0.61–0.91. The construction is workable only for weak coupling or small M.
- The generator instead uses targets Σ = Σ0 ∘ R(τ_max). There Σ ⊘ R(τ) = Σ0 ∘ R(τ_max − τ), which is positive definite by the Schur product theorem for every delay range up to `max_lag_range`. The exact split needs exponential ρ (AR(1)); with continuous delays it extends to powered-exponential ρ (inferred).
- Used here: M=20, a=0.9 (correlation time 9.5 samples), loadings U(0.6, 0.8), delay ranges 0–15 samples (≤ 1.6 correlation times), T=20,000. The entry-wise zero-lag RMS error is 0.02 at T=20,000 (0.06 at T=2,000).
- Not covered: oscillatory ρ, where Σ ⊘ R is undefined beyond the first zero crossing of ρ.

## Fig. 1 candidate caption (draft)

**Two recordings with the same zero-lag correlation matrix differ in timing.** Each point is a channel pair of one simulated recording (M = 20 channels, T = 20,000 samples). The x-axis is the pair's zero-lag correlation and the y-axis its maximum cross-correlation over lags. Grey: all channels respond to a common drive without delay. Colour: the same zero-lag correlations arise because channels respond at channel-specific delays τ_j, coloured by the pair's delay difference. The two recordings' population zero-lag matrices are identical; sample entries agree to RMS 0.02, the same as two synchronous replicates. Hence every entry-wise, spectral or topological summary of that matrix agrees. The edge profiles of the two statistics agree in the synchronous recording (z = 1.00) and diverge in the delayed one (z = 0.43). Inset: z for 20 recordings at each delay range (range = max_j τ_j − min_j τ_j); dashed, population value.

```bash
python -m jupyter nbconvert --to notebook --execute --inplace research/cases/delay_alias/notebooks/delay-alias_261020.ipynb
```
