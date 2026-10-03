"""Lean generator-level strength comparison; preserve all earlier notebooks."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def build():
    cells=[
('md',r'''# Matching coupling before computing SPIs

Can SPI–SPI distinguish dynamics beyond per-SPI means when a declared generator-level interaction strength is matched? This is a new six-condition exploratory experiment using the existing VAR and quadratic-CML generator families. It does not normalize individual MPIs or select examples according to which representation wins.

The primary comparison is the **mean of each MPI** versus **SPI–SPI**. Distribution summaries and validity are secondary controls. Classifying six variants is a test of representation utility; it does not establish unique identification of a physical mechanism.'''),
('code','''from pathlib import Path
import sys,json
ROOT=Path.cwd().resolve()
while not (ROOT/'pyproject.toml').exists() and ROOT!=ROOT.parent: ROOT=ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scripts.calibrate_native_coupling import DATA,OUT
from scripts import analyze_native_coupling as study
rows=pd.DataFrame(json.loads((DATA/'manifest.json').read_text())['rows'])
'''),
('md',r'''## What is held fixed?

We define the same-state counterfactual coupling contribution as $\Delta_t=F_g(x_t)-F_0(x_t)$ and calibrate each recording to

$$s(X)=\sqrt{\frac{\langle\sum_i\Delta_{i,t}^{2}\rangle_t}{\sum_i\operatorname{Var}_t(x_i)}}=0.10\pm0.002.$$

For VAR, $\Delta_{i,t}=\frac g2(x_{i-1,t}+x_{i+1,t})$, with self-memory $\phi\in\{0.2,0.7\}$ and independent Gaussian innovations. For the quadratic CML, $f_\alpha(x)=1-\alpha x^2$ and $\Delta_{i,t}=g[\frac12(f_\alpha(x_{i-1,t})+f_\alpha(x_{i+1,t}))-f_\alpha(x_{i,t})]$, with $\alpha\in\{1.69,1.7522,1.895,2.0\}$. Thus g is total neighbour weight for VAR and the diffusive mixing fraction for CML; it is adjusted per realization. Neither equality of their coefficient values nor equality of all dependence statistics is assumed.

This dimensionless **per-step RMS interaction effect** is an explicit cross-model convention, not a unique universal coupling scalar. It is evaluated using the generator's actual state and coupling term, before computing SPIs. CML retains the original 100-site ring and 16-site crop; VAR observes its full 16-site ring. Both use 2,000 burn-in updates and 1,000 recorded samples. Calibration preserves the VAR stability bound without rescaling its coefficients. CML labels give the actual alpha values: changing coupling does not justify retaining historical regime names.

All channels are centered, and one common scale per recording sets mean channel variance to one. This removes arbitrary global amplitude while preserving the calibrated strength ratio; it is not separate MPI normalization. The source retains both native and transformed recordings. The proposed alpha=1.45 condition failed target bracketing for seed261003111 in g∈[0,.2] and was removed as a whole before SPI extraction. No realizations were selectively replaced.'''),
('code',"fig=study.plot_calibration()\nplt.show()\ndisplay(rows.groupby('label').agg(n=('row_id','size'),strength_min=('strength','min'),strength_max=('strength','max'),native_g_min=('g','min'),native_g_max=('g','max')).round(4))"),
('md',r'''## Mean MPI versus SPI–SPI

Each condition has 24 paired seed groups: instances 0–11 supply development data; 12–23 supply evaluation data. All p90 MPIs use the same catalogue and serial estimator-seeding policy. We preserve ordered off-diagonal entries and keep undefined SPIs missing. Each representation uses development-only 95% feature validity, median imputation, SD scaling, clipping at ±5 SD, and up to 20 principal components. A fixed C=1 logistic classifier evaluates held realizations. The full-mean sensitivity omits PCA, retaining all selected mean features. It was specified before the SPI outcomes.

Both PCA and UMAP displays are fitted on development data and transform the evaluation recordings. UMAP uses 30 neighbours, min_dist=.1 and seed261003. Separate maps have arbitrary orientation and scale; visual gaps do not establish that information is absent. Classification is evaluated in the fitted representation, not in UMAP.'''),
('code',"fig=study.plot_embedding('all6')\nplt.show()"),
('code',"metrics=pd.read_csv(OUT/'metrics.csv')\nprimary=metrics[metrics.method.isin(['mean','z','mean_without_PCA'])]\ndisplay(primary.round(3))\ndisplay(pd.read_csv(OUT/'paired.csv').query(\"comparison in ['z - mean', 'z - mean_without_PCA']\").round(3))"),
('md',r'''## Within the quadratic-CML family

These four conditions share a generator family and the calibrated coupling-effect target, while alpha differs. They need not retain the regime identities of the original proof. The classifier is fitted on the CML development rows; preprocessing/PCA20 remains the common development fit, and the displayed CML UMAP is fitted only on CML development rows.'''),
('code',"fig=study.plot_embedding('CML4')\nplt.show()"),
('md',r'''## Secondary checks and limits

Seven per-SPI summaries (mean, SD, five quantiles) retain additional marginal information. The validity-only control checks whether patterns of undefined statistics classify the conditions. The strength-only control checks residual information in the small allowed calibration error. Confidence intervals and paired differences resample the 12 held seed groups, retaining all classes in each draw; they condition on the fitted models and do not account for training uncertainty or exploratory choices.

Matching a generator-level coupling contribution does not imply equal observed Pearson correlation, equal spectra, or equal means of every SPI. If means still classify well, that is an informative negative result for the proposed separation: marginal SPI responses also describe character under this strength convention. A stronger z result would instead establish an advantage for the specified representation/readout on this controlled comparison, not universal strength invariance.'''),
('code',"display(metrics[~metrics.method.isin(['mean','z','mean_without_PCA'])].round(3))\ndisplay(pd.DataFrame(json.loads((OUT/'analysis.json').read_text())['diagnostics']).T.round(3))"),
('md',r'''## Separate cached diagnostic

This is **not** the generator-level manipulation above: within each historical MPI, subtracting its mean and dividing by its SD sets its mean and SD by construction while preserving Pearson SPI–SPI. Nevertheless, the remaining five normalized quantiles retain substantial class information. The table is included only to distinguish removal of MPI location/scale from physical coupling normalization. Constant/undefined profiles remain missing; the validity-only control is reported. It uses the original retrospective 14-class/five-CML split, not the new six conditions.'''),
('code',"display(pd.read_csv(ROOT/'results/baseline-strength-audit_261003/metrics.csv').round(3))"),
('md',r'''Reproduction: `python -m scripts.calibrate_native_coupling scout` and `build` prepare the calibrated inputs (build refuses to replace an existing bank). The external p90 configuration is `configs/external/native-coupling-261003.yaml`; the generator protocol is `configs/analysis/native-coupling-261003.yaml`. After retrieving the cached MPIs, `python -m scripts.analyze_native_coupling extract` and `analyze` produce the audited feature bank, predictions and tables. This notebook only renders those results. Original proof and baseline notebooks are preserved.''')]
    n=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(s) if k=='md' else nbf.v4.new_code_cell(s) for k,s in cells])
    n.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    nbf.write(n,ROOT/'notebooks/embeddings/spi_native_strength_control_261003.ipynb')


if __name__=='__main__':build()
