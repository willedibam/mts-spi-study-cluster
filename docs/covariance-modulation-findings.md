# Covariance-modulation pilot: findings

Status2026-09-09: all640p90 records and144statistical fits complete; corrected
pointwise40label fits complete and verified; one epoch-ceiling extension and lower-label curves are now running. This is an exploratory pilot with three disjoint
source cohorts per process, not a confirmatory result selected from new data.
The [protocol](covariance-modulation-pilot.md) states the generator and comparisons.

The principal SPI-SPI utility hypothesis is not supported on this task. At40labels:

| Representation/readout | Same process, M16/T1000 | Same process, M8/T500 |
|---|---:|---:|
| Source median | .2265 | .2265 |
| Corrected aligned neural encoder | .2273 | .2295 |
| Corrected temporal-pair neural encoder | .2273 | .2329 |
| Pointwise pair neural encoder,600epoch ceiling | .1291 | .1641 |
| z + PCA/ridge | .2196 | .2258 |
| z + PLS | .2177 | .2375 |
| Normalized SPI shapes + PLS | .2315 | .2533 |
| Shapes + z + PLS | .2143 | .2308 |
| Rich SPI marginals + PLS | .1876 | .3659 |
| Marginals + z + PLS | .1859 | .2782 |
| Raw covariance/cumulant/window summaries + PLS | .1355 | .1662 |
| Observed-data moment estimate + source calibration | .0875 | .1261 |

The moment reference uses knowledge of the model form, but estimates groups and
loadings from observed data; it does not receive latent groups or parameters.
Uncalibrated supplementary evaluation needs zero labels and obtains .0935/.1510.
This establishes an accessible raw-data signal, not a universal estimation bound.

Reduced-observation z-PLS MAE10/20/40=.2456/.2306/.2375; z-PCA=.2289/.2286/.2258.
At40labels z-PCA minus median=-.00069, conditional95%CI[-.00776,.00622]; z-PLS
minus median=+.01104[-.00187,.02344]. Neither resolves an advantage. The z-PLS
advantage over normalized SPI shapes is -.01582[-.02810,-.00328], but both are
worse than the median in mean error. Fusion improves weak SPI baselines without
establishing useful absolute performance. Intervals resample independent test
masters within process and condition on the six fitted cohort models; pointwise,
not selection-adjusted. Preserve all scopes/cohort results, not just this table.

Covariance-only and spectral controls remain weak, consistent with the deliberately
matched population second-order laws. Window summaries perform better for the
persistent source (full MAE.1189) than iid (.1834), while cumulants work in both
(.1472/.1393). The strong instantaneous moment reference works across the process
change too (other-process full/reduced .0906/.1292). Thus successful target
prediction here does not require learning temporal state persistence.

## A narrow explanation of Pearson compression

After seeing these results, we checked a population-level special case using
Pearson correlation and the fourth cross-cumulant. This is an explanatory
diagnostic, not another predictive benchmark or a proof about all289SPIs.

For fixed background/loading parameters and group assignment, let R denote the
population correlation MPI and K the fourth cross-cumulant MPI. Conditional
Gaussian moments give

\[
K_{ij}=\operatorname{cum}(X_i,X_i,X_j,X_j)
       =2\alpha^2d_i d_j\,\mathbf1\{i,j\text{ in different groups}\}.
\]

R is independent of alpha. Therefore, for alpha>0,

\[
\operatorname{corr}(\operatorname{offdiag}R,\operatorname{offdiag}K)
=\operatorname{corr}(\operatorname{offdiag}R,
                     \operatorname{offdiag}(K/\alpha^2))
\]

is also independent of alpha. K's magnitude carries the target while this
agreement coordinate removes it. Both directed off-diagonal orientations are
included; the argument does not depend on discarding direction.

`scripts/check_covariance_modulation_population.py` verifies this over100fixed
nuisance draws and19alpha values: maximum agreement change1.11e-15, while mean
cumulant magnitude changes361-fold. True groups/loadings are used only for this
population identity check, not for fitted baselines. Other catalogue statistics
can respond non-affinely to alpha, so this does **not** prove fullz is invariant
or establish why every empirical readout failed. It demonstrates a relevant,
exact failure mode of normalized cross-statistic agreement.

## Validation and decision

- All640extractions completed; corrected Gadi analysis178481788 exited0 in2:43.
  Eight local MPI feature replays match the bank. Eighteen local statistical
  replays cover every method/both processes and the former rank-one failure;
  maximum prediction/CV discrepancies1.57e-14/1.11e-15.
- First analysis failed when validityPLS requested4components from a rank1matrix.
  Rank capping fixes this; all108previous raw PLS fits/CV scores reproduce exactly.
- Original neural validation clipped negative predictions, sometimes hiding
  genuine improvement. Corrected stopping uses raw validation MAE and a600epoch
  ceiling; historical scores remain diagnostics. See protocol for the reproduced
  defect and why the joint change is not a clean causal performance attribution.
- Corrected aligned18fits reproduce the original performance: reduced MAE10/20/40
  .2324/.2330/.2295. All checkpoint/split checks pass; CPU/MPS error<=3.58e-7,
  no selected600epochcap. The validation repair did not rescue this architecture.
- Corrected temporal-pair18fits complete: reduced MAE10/20/40=.2274/.2372/.2329,
  full=.2265/.2324/.2273. All checkpoints/splits pass; CPU/MPS discrepancy<=1.20e-7,
  no selected600epochcap. The correction makes small10/20label differences but
  does not rescue performance. Combined interim report is `corrected-interim-report/`.
- Finish the prespecified40label pointwise
  gate. A larger identical p90 run is not justified by current results. The
  task demonstrates that dependence-only signal is insufficient to guarantee
  utility for z. Further work should distinguish changing dependence magnitudes
  from changing cross-statistic edge correspondence, with a meaningful target;
  that is a research requirement, not an instruction to manufacture a win.

Reproduce tables with `scripts/report_covariance_modulation.py`, inputs
`results/covariance_modulation_260909/{raw,gadi-analysis/statistical-rank-fixed}`;
authoritative interim output `statistical-raw-report/`. Verification JSON files
and original/corrected outputs are retained separately under the study root.

## Bounded catalogue attribution after the main result

User asked whether nonlinear SPI values contain signal that their agreements
lose. A fixed panel of dcorr, biased dcorr, two Kraskov MI configurations and
kernel MI was compared using the same cohorts/PLS grid. Gaussian MI was excluded
because it is a function of correlation. This is retrospective attribution,
not a newly confirmed performance result;72fits in `catalogue-signal/`, report
`catalogue-signal-report/`, script `check_covariance_catalogue_signal.py`.

At40labels, original-size MAE is .1844 for the five means, .1567 for115rich
summaries, .2300 for105normalized shape features, and .2194 for their ten
agreement coordinates. The catalogue is therefore not wholly insensitive to the
target: retaining distribution location/scale exposes useful signal. This
supports, but does not prove, information loss through normalization; differences
in feature dimension, estimation and readout remain. The ten-coordinate panel
omits relationships with linear SPIs, so its failure does not rule out informative
coordinates elsewhere in z.

After M/T reduction these errors become .4991/.4986/.2728/.2539. The rich/mean
readouts often saturate at prediction1; for the first40label cohort, all100iid and
98/100persistent reduced-view predictions are exactly1. Mean kernel-MI across
all evaluation recordings shifts .1514→.1982; other estimates also shift.
Thus full-size signal alone does not establish robustness to observation changes.
MI Kraskov summaries are finite on584/640records; the other panel summaries on
all640. Existing training-only validity filtering applies. No newpyspi was needed.

More aggressive regularization of z is not currently established as the remedy:
PCA/PLS already limit learned dimension, and the small agreement panel does not
rescue this task. Useful candidate extensions would preserve selected magnitudes
and address their sample-size sensitivity. No additional tuning grid is warranted
while the distinct co-organization scout and pointwise diagnostic are pending.

## Completed pointwise diagnostic and bounded continuation

All six40label pointwise fits complete and pass checkpoint/split verification:
CPU/MPS maximum discrepancy2.38e-7. Same-process original/reduced MAE=.1291/.1641;
other-process=.1264/.1494. This model substantially improves on the earlier raw
encoders and z. Its change combines temporal kernels, resolution and parameter
count; do not attribute the full improvement to a single detail. It has not
beaten the model-informed moment reference (.0875/.1261 same-process).

The declared same-full MAE<.18 gate passes, authorizing10/20label curves. Two of12
selected validation folds reached600epochs and improved unclipped MAE by.0283
and.0477 over the final100epochs. Make one extension to1200epochs, uniformly for
10/20/40labels, with otherwise identical architecture/grid/source splits. Original
600epoch results remain separate. Configuration frozen1a3a7ea before extension;
outputs`neural-pointwise-extended/`, report once complete. Do not keep doubling
ceilings indefinitely or imply a universal neural optimum from this comparison.
