"""Lean generator-level strength comparison; preserve all earlier notebooks."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def build():
    cells=[
('md',r'''# Matching coupling before computing SPIs

Can SPI–SPI distinguish dynamics beyond per-SPI means when a declared generator-level interaction strength is matched? This is a new seven-condition exploratory experiment using the existing VAR and quadratic-CML generator families. It does not normalize individual MPIs or select examples according to which representation wins.

The primary comparison is the **mean of each MPI** versus **SPI–SPI**. Distribution summaries and validity are secondary controls. Classifying seven variants is a test of representation utility; it does not establish unique identification of a physical mechanism.'''),
('code','''from pathlib import Path
import sys,json
ROOT=Path.cwd().resolve()
while not (ROOT/'pyproject.toml').exists() and ROOT!=ROOT.parent: ROOT=ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display
from scripts.calibrate_native_coupling import DATA,OUT
from scripts import analyze_native_coupling as study
rows=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows'])
'''),
('md',r'''## What is held fixed?

We define total cross-channel coupling as the mean row sum of the absolute off-diagonal Jacobian of one native update:

$$G(X)=\left\langle\sum_{j\ne i}\left|\frac{\partial F_i(x_t)}{\partial x_j}\right|\right\rangle_{i,t}=0.20\pm0.002.$$

For VAR, $F_i(x)=\phi x_i+\frac g2(x_{i-1}+x_{i+1})$, hence $G=g$ exactly; we set g=.2 for both self-memory settings $\phi\in\{.2,.7\}$. For quadratic CML, $f_\alpha(x)=1-\alpha x^2$ and $F_i=(1-g)f_\alpha(x_i)+\frac g2[f_\alpha(x_{i-1})+f_\alpha(x_{i+1})]$, so $G=g\alpha\langle|x_{i-1}|+|x_{i+1}|\rangle_{i,t}$. We calibrate g separately for each CML recording, with $\alpha\in\{1.45,1.69,1.7522,1.895,2.0\}$.

This **per-step linear-response gain** is an explicit strength convention, not a universal dependence scalar. It is computed from the actual generator Jacobian before any SPI and remains sensitive to coupling when channels synchronize. An earlier discarded calibration used the realized update contribution, which can vanish under synchrony even for strong coupling; it is not the primary experiment reported here. CML retains the original 100-site ring and 16-site crop (both neighbours, including those outside the crop, enter the known derivative); VAR observes its full 16-site ring. Both use 2,000 burn-in updates and 1,000 recorded samples. VAR coefficients remain below the stability bound without rescaling. CML labels give actual alpha values: changing coupling does not justify retaining historical regime names.

All channels are centered, and one common scale per recording sets mean channel variance to one. An affine change with a common scalar leaves the Jacobian unchanged, so this removes arbitrary global amplitude while preserving G. No individual MPI is normalized. Native and transformed recordings are retained; every condition retains all 24 requested realizations.'''),
('code',"fig=study.plot_calibration()\nplt.show()\ndisplay(rows.groupby('label').agg(n=('row_id','size'),strength_min=('strength','min'),strength_max=('strength','max'),native_g_min=('g','min'),native_g_max=('g','max')).round(4))"),
('md',r'''## Mean MPI versus SPI–SPI

Each condition has 24 paired seed groups: instances 0–11 supply development data; 12–23 supply evaluation data. All p90 MPIs use the same catalogue and serial estimator-seeding policy. We preserve ordered off-diagonal entries and keep undefined SPIs missing. Each representation uses development-only 95% feature validity, median imputation, SD scaling, clipping at ±5 SD, and up to 20 principal components. A fixed C=1 logistic classifier evaluates held realizations. The full-mean sensitivity omits PCA, retaining all selected mean features. It was specified before the SPI outcomes.

Both PCA and UMAP displays are fitted on development data and transform the evaluation recordings. UMAP uses 30 neighbours, min_dist=.1 and seed261003. Separate maps have arbitrary orientation and scale; visual gaps do not establish that information is absent. Classification is evaluated in the fitted representation, not in UMAP.'''),
('code',"fig=study.plot_embedding('all7')\nplt.show()"),
('code',"metrics=pd.read_csv(OUT/'metrics.csv')\nprimary=metrics[metrics.method.isin(['mean','z','mean_without_PCA'])]\ndisplay(primary.round(3))\ndisplay(pd.read_csv(OUT/'paired.csv').query(\"comparison in ['z - mean', 'z - mean_without_PCA']\").round(3))"),
('md',r'''## Within the quadratic-CML family

These five conditions share a generator family and the calibrated coupling-effect target, while alpha differs. They need not retain the regime identities of the original proof. The classifier is fitted on the CML development rows; preprocessing/PCA20 remains the common development fit, and the displayed CML UMAP is fitted only on CML development rows.'''),
('code',"fig=study.plot_embedding('CML5')\nplt.show()"),
('md',r'''## Secondary checks and limits

Seven per-SPI summaries (mean, SD, five quantiles) retain additional marginal information. The validity-only control checks whether patterns of undefined statistics classify the conditions. The strength-only control checks residual information in the small allowed calibration error. Confidence intervals and paired differences resample the 12 held seed groups, retaining all classes in each draw; they condition on the fitted models and do not account for training uncertainty or exploratory choices.

Matching a generator-level coupling contribution does not imply equal observed Pearson correlation, equal spectra, or equal means of every SPI. If means still classify well, that is an informative negative result for the proposed separation: marginal SPI responses also describe character under this strength convention. The observed all-seven accuracies are 95.2% for means, 96.4% for means without PCA, and 98.8% for z. The paired z-minus-mean difference is 3.6 percentage points, with a held-block bootstrap interval of 0 to 9.5 points. Thus this pilot does **not** yield unstructured marginals and structured z; the point-estimate gain is modest and uncertain. A stronger z result would instead establish an advantage for the specified representation/readout on this controlled comparison, not universal strength invariance.'''),
('code',"display(metrics[~metrics.method.isin(['mean','z','mean_without_PCA'])].round(3))\ndisplay(pd.DataFrame(json.loads((OUT/'analysis.json').read_text())['diagnostics']).T.round(3))"),
('md',r'''## Separate cached diagnostic

This is **not** the generator-level manipulation above: within each historical MPI, subtracting its mean and dividing by its SD sets its mean and SD by construction while preserving Pearson SPI–SPI. Nevertheless, the remaining five normalized quantiles retain substantial class information. The table is included only to distinguish removal of MPI location/scale from physical coupling normalization. Constant/undefined profiles remain missing; the validity-only control is reported. It uses the original retrospective 14-class/five-CML split, not the new seven conditions.'''),
('code',"display(pd.read_csv(ROOT/'results/baseline-strength-audit_261003/metrics.csv').round(3))"),
('md',r'''Reproduction: `python -m scripts.calibrate_native_coupling scout` and `build` prepare the calibrated inputs (build refuses to replace an existing bank). The external p90 configuration is `configs/external/native-gain-261003.yaml`; the generator protocol is `configs/analysis/native-coupling-261003.yaml`. After retrieving the cached MPIs, `python -m scripts.analyze_native_coupling extract` and `analyze` produce the audited feature bank, predictions and tables. This notebook only renders those results. Original proof and baseline notebooks are preserved.''')]
    n=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(s) if k=='md' else nbf.v4.new_code_cell(s) for k,s in cells])
    n.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nbf.write(n,ROOT/'notebooks/embeddings/spi_native_strength_control_261003.ipynb')


if __name__=='__main__':build()
