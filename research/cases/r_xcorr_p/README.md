# Timing triplet: r, max cross-correlation, phase-aligned correlation P

A linear-timing analogue of [`r_rho_mi/`](../r_rho_mi/README.md). The three members correlate a channel pair after increasing realignment. $r$ allows none, pyspi `xcorr_max_sig-True` allows one global delay, and the bespoke $P_{jk}=\int|S_{jk}(f)|\,df/\sqrt{\int S_{jj}\int S_{kk}}$ allows any phase per frequency. The first rung, $\{r,\mathrm{xcorr}\}$, is the introduction's Fig. 1 ([`delay_alias/`](../delay_alias/README.md)).

- **Generator:** `mtsspi.generators.generate_band_delayed_mts`, registry name `band_delayed_mts`. It has a slow band ($|f|<0.08$) and a fast band ($0.10<|f|<0.40$), each with a common drive delayed per channel, plus white noise.
  - In `synchronous`, every delay is 0. In `delayed`, both bands use the same channel delays (0–15 samples). In `dispersive`, the fast band uses a fixed permutation of the slow band's delays.
  - The bands do not overlap, so all three families share every auto-spectrum and every $|S_{jk}(f)|$, and differ only in phase.
- **`spis/timing_triplet.yaml`:** the pyspi subset (zero-lag Pearson as `cov_EmpiricalCovariance`, `xcorr_max_sig-True`, `cohmag`). It takes about 0.4 s per recording at M=20, T=20,000.
- **`timing_triplet.py`:** the Welch estimator of $P$ ($L=256$), population ladder, numpy xcorr (matches pyspi to $10^{-14}$), surrogates, narrowband scan and figure.
- **`notebooks/timing-triplet_261020.ipynb`:** executed checks. Outputs are in `results/cases/timing-triplet-261020/` (`timing_triplet.{png,svg,pdf}`, `z_by_recording.csv`, `summary.json`).

## Results (M=20, T=20,000, 20 recordings per family)

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
python -m jupyter nbconvert --to notebook --execute --inplace research/cases/timing_triplet/notebooks/timing-triplet_261020.ipynb
```
