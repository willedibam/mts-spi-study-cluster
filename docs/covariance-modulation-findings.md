# Covariance-modulation pilot: findings

Status2026-09-09: all640p90 records and144statistical fits complete; corrected
neural fits still running. This is an exploratory pilot with three disjoint
source cohorts per process, not a confirmatory result selected from new data.
The [protocol](covariance-modulation-pilot.md) states the generator and comparisons.

The principal SPI-SPI utility hypothesis is not supported on this task. At40labels:

| Representation/readout | Same process, M16/T1000 | Same process, M8/T500 |
|---|---:|---:|
| Source median | .2265 | .2265 |
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
- Finish corrected neural comparisons and the prespecified40label pointwise
  gate. A larger identical p90 run is not justified by current results. The
  task demonstrates that dependence-only signal is insufficient to guarantee
  utility for z. Further work should distinguish changing dependence magnitudes
  from changing cross-statistic edge correspondence, with a meaningful target;
  that is a research requirement, not an instruction to manufacture a win.

Reproduce tables with `scripts/report_covariance_modulation.py`, inputs
`results/covariance_modulation_260909/{raw,gadi-analysis/statistical-rank-fixed}`;
authoritative interim output `statistical-raw-report/`. Verification JSON files
and original/corrected outputs are retained separately under the study root.
