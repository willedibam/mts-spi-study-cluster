# Timing over frequency: r, max cross-correlation, phase-aligned correlation P

The three members correlate a channel pair after increasing realignment: $r$ allows none, pyspi `xcorr_max_sig-True` allows one global delay, and $P$ allows any phase per frequency. The first rung, $\{r,\mathrm{xcorr}\}$, is the introduction's Fig. 1 ([`delay_alias/`](../delay_alias/README.md)).

**What $P$ is.** $P_{jk}=\int|S_{jk}(f)|\,df\,/\sqrt{\int S_{jj}\int S_{kk}}$, the power-weighted mean of the coherency magnitude. Equivalently, it is the largest correlation between $x_j$ and any phase-only (all-pass) filtering of $x_k$: each frequency may be shifted by its own phase, but no amplitude can be reweighted. A single delay is the special all-pass with phase linear in $f$. Hence $r\le\max_h r(h)\le P$, with equality on the right exactly when the pair's phase is linear in frequency. It is the pyspi fork's `PhaseAlignedCorrelation` (`phasealigned_welch_nperseg-256`, branch `v3`, commits `19b5411` and `c3de193`), estimated from Welch cross-spectra and equal to `timing_triplet.phase_aligned_corr` to 2e-16. One search found no named precedent (unverified).

## Current case (2026-10-28): `dispersion_261028.py`

One system, one knob: how far the fast band's channel delays depart from the slow band's. Levels: no delay; delays 0–15 shared by both bands ($\kappa=0$); fast-band delays moved half way ($\kappa=0.5$) and fully ($\kappa=1$) to a fixed permutation (generator `band_delayed_mts`, new `dispersion` argument). M = 12, T = 10,000, 20 recordings per level, the same draws at every level. All levels share every spectral magnitude, so the $P$ matrix is the same at every level. Outputs in `results/cases/dispersion-261028/`.

| $z$ (ordered entries) | no delay | delay | $\kappa=.5$ | $\kappa=1$ |
| --- | --- | --- | --- | --- |
| $r$–max | 1.00 | 0.18 | 0.43 | 0.56 |
| max–$P$ | 1.00 | 1.00 | 0.65 ± 0.02 | 0.58 ± 0.02 |
| max–cohmag | 0.92 | 0.91 | 0.56 | 0.51 |

- **Decision: $P$ is the third member** (pyspi fork, `spis/r_xcorr_p.yaml`). cohmag tracks the dispersion drop (z(cohmag, P) ≈ 0.92 at every level), but its pure-delay baseline is 0.91 rather than 1 when bands couple different pairs. That happens because cohmag is the bandwidth-weighted mean of the *squared* coherence. With one loading per channel shared by both bands, it is an almost exact proxy (z(cohmag, P) = 0.985, baseline 0.98; scratch run 2026-10-28). Real band-specific networks break that condition, and then a pure delay looks partly dispersive.
- **A delay-spread knob would be a step:** $z(r,\max)$ is already 0.35 at a 2-sample range, because the fast band has a 4-sample period. That rung is Fig. 1's, so the knob here is dispersion.
- **$z(r,\max)$ rises again under dispersion** because both statistics lose the mismatched pairs and agree on which pairs are weak.

## Files

- `spis/r_xcorr_p.yaml`: the current case's pyspi subset ($r$, max xcorr, $P$, cohmag). `spis/timing_triplet.yaml`: the 2026-10-20 subset (zero-lag Pearson as `cov_EmpiricalCovariance`, `xcorr_max_sig-True`, `cohmag`).
- `timing_triplet.py`: the Welch estimator of $P$, population ladder, numpy xcorr (matches pyspi to $10^{-14}$), surrogates and narrowband scan.
- `dispersion_261028.py`: the current one-knob case. `timing_composite_261025.py` and `notebooks/timing-triplet_261020.ipynb`: the earlier three-family composite and checks (outputs in `results/cases/timing-triplet-261020/`).

## 2026-10-20 run (three families, M=20, T=20,000, 20 recordings per family)

| $z$ | synchronous | delayed | dispersive | separates |
| --- | --- | --- | --- | --- |
| r–xcorr | 1.000 | 0.354 ± 0.020 | 0.207 ± 0.014 | synchronous from lagged |
| xcorr–P | 0.999 | 0.999 | 0.678 ± 0.009 | pure delay from dispersion |
| r–P | 0.999 | 0.365 ± 0.020 | 0.295 ± 0.016 | any timing structure |

- **Verified: the ladder.** $r\le\mathrm{xcorr}\le P$ follows because an all-pass filter changes only the phase, and a delay is a linear-phase all-pass.
  - It held for 500 random population pairs, as did $P\le W\le1$, where $W$ is the Wiener (any-filter) correlation. No random all-pass exceeded $P$.
  - Pure delay gives $\mathrm{xcorr}=P$ exactly, at the true lag.
  - In samples the ladder is approximate: $\hat r\le\widehat{\mathrm{xcorr}}\le\hat P$ held for 96/90/99% of pairs, because the two estimators' biases break ties.
- **Verified here: separation.** Sample $z$ is within 0.013 of the population value.
  - $z(\mathrm{xcorr},P)$ is 1.00 for pure delay at every structure seed, and 0.45–0.70 for dispersion over seeds 1–5.
  - $z(r,\mathrm{xcorr})$ does not order delayed against dispersive: the sign of their difference changes with the structure seed.
  - Synchronous and delayed differ in $z(\mathrm{xcorr},\hat P)$ by 0.0003 (AUC 0.19), a Welch segmentation effect.
- **Derived: interpretation.** In population, $z(\mathrm{xcorr},P)<1$ implies that some pair has $\mathrm{xcorr}<P$, i.e. a phase that is not linear on its spectral support. The converse fails: a deficit affine across pairs leaves $z=1$, as for any across-pair correlation.

## `cohmag` as a proxy for P

pyspi `cohmag` is the unweighted mean of the magnitude-*squared* coherence over $[0,\tfrac12]$, estimated with 5 tapers. It is not the mean of $|C|$; this was verified against `spectral_connectivity`. $P$ is instead a power-weighted mean of $|C|$.

| structure | $z(\mathrm{cohmag},P)$ | $z(\mathrm{xcorr},\mathrm{cohmag})$, pure delay |
| --- | --- | --- |
| one source (delay_alias) | 0.97 | 0.97 |
| two bands, shared loadings | 0.99 | 0.99 |
| two bands, independent loadings per band | 0.90 | 0.89 |

- **Acceptable proxy.** `cohmag` works when the across-pair pattern of coherence is the same at every frequency carrying power, i.e. $|C_{jk}(f)|\approx g_{jk}h(f)$. This condition is inferred and was shown for these three structures.
- **When it fails.** If different bands couple different pairs, `cohmag` weights bands by bandwidth rather than power. $z(\mathrm{xcorr},\mathrm{cohmag})$ then drops below 1 under pure delay, which mimics dispersion.

## Limits

- **Narrowband (verified in population).** Within one narrow band a phase offset equals a delay modulo the carrier period, so $\mathrm{xcorr}\to P$ and the lag of the maximum becomes ambiguous.
  - Shrinking both bands to 2% of their width raises the dispersive $z(\mathrm{xcorr},P)$ from 0.57 to 0.97.
  - The residual comes from the integer-lag grid at a 4-sample carrier: pairs whose band delays agree modulo 4 reach $\mathrm{xcorr}/P=1.000$.
- **Second order (verified).** All three statistics are functionals of the spectral matrix. Multivariate phase-randomised surrogates preserve every triplet $z$ to within 0.004. Findings on this axis are linear lag/spectral organisation and survive a strength-matched linear-Gaussian null by construction.
- **Estimation.** $\hat P$ has a positive floor of 0.073 at zero coherence ($L=256$, 155 segments). It rises with $L$, because raw periodograms phase-align anything.
- **Unverified: novelty.** One targeted search found no named precedent for $P$; this is not a priority review.

## Figure caption (draft)

**Maximum cross-correlation cannot align dispersion.** Each point is a channel pair of one simulated recording (M=20, T=20,000). The x-axis is the pair's phase-aligned correlation $P$, i.e. its correlation after the best phase-only realignment. The y-axis is its maximum cross-correlation over lags. Blue: both frequency bands share each channel's delay (pure delay), and pairs lie on the diagonal ($z=1.00$). Orange: the fast band's delays are permuted relative to the slow band's. No single lag then aligns both bands, and pairs fall below the diagonal, coloured by the difference between their two band delays ($z=0.67$). Both families have identical spectral magnitudes. Inset: $z$ for three statistic pairs in synchronous, pure-delay and dispersive families (dots, 20 recordings each; open markers, population values).

```bash
python -m jupyter nbconvert --to notebook --execute --inplace research/cases/r_xcorr_p/notebooks/timing-triplet_261020.ipynb
```
