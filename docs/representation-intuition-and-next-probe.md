# From SPI-pair intuition to a learning claim

2026-09-09. This records the existing-case audit and the next experimental decision. It does not report a new learning benchmark or freeze a new generator.

## Contribution and paper structure

The user is part of the Cliff et al. research group. This project can be presented as a direct continuation: use relationships among measures to characterize and compare individual systems, building on the group's comparison of the measures. The new scientific contribution still needs to be distinguished from the earlier record-wise correlations and corpus-level aggregation.

Venue descriptions are broad guides, not exact acceptance templates. Keep paper structure open. A method paper can combine explanatory cases, common-space comparisons, corpus analysis and a defensible dynamical coordinate if they answer one question. A separate learning paper is justified by a self-contained learning insight with its own utility, mechanism and transfer evidence; another readout benchmark alone is more naturally a section. Neither splitting nor combining is currently decided. The bridge application is optional, not a prerequisite.

## Existing intuition: verified saved outputs

Inspected `notebooks/cases/r_rho_mi_260622.ipynb` and `notebooks/cases/pdist-euclid_dtw.ipynb`, relevant generator source and saved metadata. Did not execute or modify either notebook. Recomputed three-SPI Pearson meta-features from all ordered off-diagonals, preserving both directions.

Reproduce from the repository root:

```sh
.venv/bin/python scripts/audit_representation_intuition.py
```

The output is `results/representation_intuition_260909/audit.json`; it includes all record metrics and SHA256 hashes of raw arrays, MPI archives, metadata and the audit script. It refuses to overwrite an existing report. This is a descriptive audit of historical, deliberately tuned case studies, not fresh confirmation. MPIs were reused, not independently re-extracted with current pyspi.

Filter-roll: `data/r_rho_mi/260703_g-roll`, 150 records, M32/T2000, 50 per case. Here r denotes the saved empirical covariance of standardized channels.

| Channel filters | mean corr(r,rho) | mean corr(r,MI) | mean corr(rho,MI) |
|---|---:|---:|---:|
| Linear only | .9998 | .8289 | .8307 |
| Linear + sigmoid | .2095 | .4034 | .8232 |
| Linear + sigmoid + quadratic | .9956 | .4442 | .4397 |

The contrast illustrates distinct responses of measures to relationship shape. It does not isolate intrinsic nonlinear coupling: these are differently filtered observations of a common driver. Marginal distributions also change. Mean channel excess kurtosis is -.007, -.851 and 2.057, respectively; mean absolute channel skew is .081, .047 and .721. No classifier was fitted, so these summaries are not classification accuracies or proof of sufficient marginal information.

Lag/warp: `data/dtw_euclidean/260413_2_lagged-warping_rmse_T1000`, 210 records, M20/T1000, 30 per case. With max_lag=5 and p_step=0, mean corr(xpdist,DTW)=.9960 and corr(Euclidean,xpdist)=.1556. For p_step=.1,.3,.5,.7,.9, the former becomes .6079,.5448,.5327,.5420,.5182. The response is not strictly monotone. Meanwhile mean channel lag-one correlation rises from .4863 to .6954. Thus temporal statistics must be controlled or compared in a future learning task. These findings support the intended lag-versus-warp intuition without demonstrating that z is necessary to recognize the distinction.

Two qualifications for eventual notebook polishing:

- Later filter-roll glyph helpers apply the Linfoot transform to MI before calculating Pearson agreement; earlier summaries use raw MI. The largest absolute coordinate change per record averages .1710/.0762/.0434 across the three cases. Both representations are permissible, but must be named and not presented as numerically identical. The transform is nonlinear.
- The lag notebook infers uncontaminated optimal lag recovery from average pairwise mapping-offset cancellation. That implication does not follow: cancellation in expectation does not determine a nonlinear argmin for each finite record. No failure of the particular xpdist implementation is asserted.

## Pearson compression: what can be claimed

Pearson is a coarse, fixed, low-complexity summary of each SPI-pair cloud. Its calculation is fast relative to extracting MPIs, and its meaning is explicit: linear co-variation of two named measures across aligned channel pairs. This is not equality, causal agreement or a complete description of the cloud.

A favourable bias-variance tradeoff is plausible relative to estimating a richer joint distribution, but unverified. Coarse description is not statistical low precision. Shared channels, duplicate directions for symmetric SPIs, common finite-T errors and near-constant MPI columns prevent counting M(M-1) edges as independent observations. Many correlated z coordinates do not automatically average noise away; the supervised predictor can still have high variance.

If compression becomes a paper claim, first compare repeat-record/sensor-view stability against separation of conditions. A claim that Pearson is *better* requires a specified richer comparator and matched learning budgets. Recompute SPIs on time-resampled views; do not bootstrap edges as independent samples. Learned pooling may reduce approximation bias but need not improve finite-data prediction. Sparse linear maps retain named-coordinate traceability; unstable coefficients among correlated SPIs are not reliable mechanistic attribution.

## Next bounded probe

The prospective recommendations below predate the completed co-organization and mechanism-transfer studies. Use [current scientific framing](spi-representation-scientific-framing.md) and the [evidence map](spi-representation-evidence-map.md) for their outcomes; do not launch these recommendations again as if they were untested.

The existing cases already provide intuition; do not generate another collection just to illustrate that different statistics disagree. The open hypothesis is: does cross-statistic correspondence make a useful dependence distinction easier to learn when marginal activity, strength and recording conditions vary?

Before selecting a new learning generator, specify whether nonlinear observation is signal or nuisance. The filter-roll case treats it as signal. Requiring invariance to those same transformations would erase its intended distinction. Likewise lag and warping can be physical targets or measurement nuisance, but cannot be assigned both roles within one task.

Recommended precursor: construct one dependence-type distinction with matched single-channel marginal laws, then characterize its spectra and simple pairwise references before pyspi extraction. Copula-based constructions are a candidate; matching channel marginals alone does not match spectra, dependence strength or the sampling distribution of pooled marginal summaries. Do not claim those matches without deriving or checking them. Keep a simple nonlinear-dependence reference alongside spectral/cross-spectral controls. A deliberately constructed case can identify a mechanism without establishing an application or universal neural limitation.

Advance only after this target is explicit and observable: one label-budget curve, one observation shift, existing PCA/PLS and one small learned extension if needed, matched per-SPI summaries and raw references, and a credible channel-flexible neural comparator during the pilot. Use independent realization splits, preserve negative outcomes and reserve fresh confirmation. P90 affordability is not the constraint; choosing a scientifically informative target is.

## Interpretation clarification, 2026-09-24

The user's primary motivation is a system-specific geometry among scientific notions of dependence; common M,T-independent feature length is a side benefit. Coordinate semantics, sensitivity to a controlled property, and attribution of that property in new observations are separate claims. Specialist success does not invalidate a general scientific descriptor; classification or order-PC1 superiority is not the sole utility criterion. Conversely, describing a coordinate by its two input measures does not validate a unique physiological interpretation.

The [baseline notebook](../notebooks/embeddings/spi_baseline_exploration_260921.ipynb)'s Gaussian VAR is an exact information illustration, not a proposed physical benchmark or a detector of nonlinear dynamics. C and L are same-time and one-step covariance profiles (also correlations because population variances are one). s shifts mean C; h reverses the cross-edge relationship while preserving the complete C/L marginal profiles at fixed s. Small profile amplitudes permit a valid stationary VAR. Its heatmaps illustrate correspondence; its means-versus-z plots illustrate complementary information. A channel-pair scatter of C versus L would expose the outer-correlation definition more directly than the heatmaps. Other full-catalogue means can still separate these systems.

For a common valid edge set, stacking each edge's SPI responses gives an edge-by-SPI matrix. Means/distributions summarize its columns separately; z is the Gram matrix of centered unit-norm columns. A constant additive or positive multiplicative gap between two SPI profiles does not lower their Pearson z. Accordingly, lag/warp or nonlinear transformations need not cause decoupling unless their effects change the cross-edge relationship; nearly constant profiles can instead make z undefined. Test scatter shape, marginal spread and estimator reliability alongside the scalar coordinate.

Covariance measures linear association without assuming the generator is linear and can respond to nonlinear relationships. Mutual information can be evaluated at lags or on histories. Non-Gaussian marginal laws do not themselves establish nonlinear coupling; Cauchy variables lack the finite variance needed for a population Pearson coefficient. The existing filter-roll case changes observation functions of a common driver. For the alignment triplet, matched-cost nested path sets justify DTW <= xpdist_min <= ED; the gaps quantify improvement from extra alignment freedom, not uniquely physical lag/warping. Source inspection confirmed the current xpdist implementation includes boundary stutters and a common sqrt(T) denominator.

No new simulations, p90 extraction, notebook changes or benchmark outcomes were produced in this clarification. The broader co-organization correspondence interventions and mechanism transfer are more direct existing evidence for this motivation than the September21 notebook's retrieval comparison alone.
