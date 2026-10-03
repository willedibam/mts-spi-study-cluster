"""Lean report for the selected Gaussian factor contrast, from audited caches."""
from pathlib import Path
import argparse
import pandas as pd
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
RUN='factor-parity-dominant-261004'


def build(stage):
    result=ROOT/'results/representation'/RUN/stage
    score=pd.read_csv(result/'metrics.csv').set_index('method').BA
    summary=f"MPI means: **{score['mean']:.1%}**; full means: **{score['mean_full']:.1%}**; RBF means: **{score['mean_RBF']:.1%}**; tree means: **{score['mean_trees']:.1%}**; centered SPI–SPI: **{score['z_center']:.1%}**. Chance is 50%. The original standardized SPI–SPI readout scores {score['z']:.1%}."
    if stage=='final':
        paired=pd.read_csv(result/'paired.csv').set_index('comparison').loc['z_center - mean']
        met=score[['mean','mean_full','mean_RBF','mean_trees']].max()<=.65 and score['z_center']>=.8 and score['z_validity']<=.65
        summary+=f" The prespecified weak-mean/strong-z numerical target was {'met' if met else 'not met'}. The paired z-minus-mean difference is {paired['difference']*100:.1f} percentage points (95% conditional interval {paired['low']*100:.1f} to {paired['high']*100:.1f}); "+('the advantage is resolved under this conditional analysis.' if paired['low']>0 else 'the advantage over the primary mean baseline remains unresolved.')
    cells=[('md',r'''# Equal interaction magnitudes, different sign–magnitude organization

The main comparison uses the complete 289-SPI catalogue: raw off-diagonal MPI means $m$ versus signed Pearson SPI–SPI $z$. Distribution summaries are secondary. These are deliberately constructed Gaussian controls, not a benchmark of general superiority or nonlinear dynamics.

The preceding band-swap experiment failed: its full mean vector and SPI–SPI both classified every development validation recording. Squared lagged-correlation means exposed residual temporal-strength differences. That executed negative report is retained as `spi_band_swap_261004.ipynb`; its held set remains unopened. Here temporal dependence is removed and interaction magnitudes are matched more strongly.'''),
('md',f'**{stage.capitalize()} result.** '+summary),
('code',f'''from pathlib import Path
import sys,json
ROOT=Path.cwd().resolve()
while not (ROOT/'pyproject.toml').exists() and ROOT!=ROOT.parent:ROOT=ROOT.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display
from scripts.build_factor_parity import covariance,CLASSES
from scripts.plot_band_swap import plot
RUN={RUN!r};STAGE={stage!r}
OUT=ROOT/'results/representation'/RUN
RESULT=OUT/STAGE
metrics=pd.read_csv(RESULT/'metrics.csv')
analysis=json.loads((RESULT/'analysis.json').read_text())
display(metrics[metrics.method.isin(['mean','mean_full','mean_RBF','mean_trees','z_center','z'])].round(3))'''),
('md',r'''## Generator and strength control

Let $f_{at}$ and $\epsilon_{it}$ be independent standard Gaussian variables across factors, channels and time. A recording has 16 channels and 1,000 observations, with

$$x_{it}=\sqrt{1-\lambda}\,\epsilon_{it}+\sqrt{\lambda}\sum_{a=1}^3\sqrt{w_a}\,v_{ai}f_{at},\qquad \lambda=.25,\quad w=(.9,.075,.025).$$

Equivalently, $x_t\sim N(0,C)$ independently in time, where $C=(1-\lambda)I+\lambda\sum_a w_a v_av_a^\top$. The implementation samples this covariance by a Cholesky transform. The private variance is .75 and total shared-factor variance .25; the weights divide the latter between factors. These describe **functional dependence through shared inputs**, not direct causal channel-to-channel coupling.

Channels are indexed by four binary bits. The first two sign vectors are $v_{1i}=(-1)^{b_1(i)}$ and $v_{2i}=(-1)^{b_2(i)}$. In the **linked-signs** class, $v_3=v_1v_2$ elementwise; in the **unlinked-signs** class, $v_{3i}=(-1)^{b_3(i)}$. In both classes the three vectors are balanced and mutually orthogonal. A fresh channel permutation accompanies each recording.

Both classes have unit population channel variance, identical covariance eigenvalues, the same complete multiset of off-diagonal absolute correlations, and identical positive/negative edge counts. Every channel has total absolute off-diagonal covariance 3.35; mean absolute coupling is $3.35/15=.223\overline3$, and mean signed coupling is $-.25/15=-.016\overline6$. All nonzero-lag population covariances vanish. These equalities hold at the generator level; empirical estimates fluctuate between recordings. No MPI normalization or whitening is applied afterward.

For a temporally iid Gaussian pair, reversing one channel changes only the sign of its correlation. Consequently every sign-invariant bivariate population SPI has the same off-diagonal value multiset in the two classes. **Signed distributions, sign-sensitive nonlinear statistics, multivariate estimators and finite-sample aggregate laws need not match.** This is a claim about specified functional strengths, not a universal coupling metric.'''),
('code',r'''mask=~np.eye(16,dtype=bool)
checks=[]
for label in CLASSES:
    C=covariance(label,.25,[.9,.075,.025]);r=C[mask]
    checks.append(dict(label=label,mean=r.mean(),mean_absolute=np.abs(r).mean(),mean_square=np.mean(r*r),negative_fraction=np.mean(r<0),z_linear_square=np.corrcoef(r,r*r)[0,1]))
display(pd.DataFrame(checks).round(6))'''),
('md',r'''## Why these parameters?

The first factor is dominant so that sign counts match and regularized covariance/precision estimates have more stable sign patterns. The initial factor candidate (.35; weights .7/.2/.1) was rejected before any p90 computation because graphical-lasso means differed even at population level. Five stronger-dominance/lower-strength settings were checked using cheap statistics on development blocks, followed by five more when those retained mean information. The selected setting was candidate 8: the weakest worst mean readout among candidates with nearly equal population graphical-lasso means and probe-z accuracy at least .8. Its cheap mean classifiers scored .5625/.6875/.5 versus .875 for probe z on 16 validation recordings.

This is openly **development-based experimental design**, not preregistration of the entire search. No full-p90 or held results selected these parameters. All scout outcomes are retained below. The complete catalogue and a separate held set are required to assess whether that promising cheap result survives.'''),
('code',"display(pd.read_csv(ROOT/'results/representation/factor-parity-261004/scout-results.csv').round(4))"),
('md',r'''## Full-catalogue comparison

Training uses development blocks 0–23, validation blocks 24–31; each block contains both classes. The separate held blocks 32–63 are evaluated only after a documented development decision, using training blocks 0–31. Training-only preprocessing selects 95% finite features and imputes medians. Means/distributions use SD scaling, clipping at five SD and 20 PCs; the preferred SPI–SPI readout centers correlations without rescaling or clipping, then uses 20 PCs. Logistic regression fixes $C=1$. Full-mean logistic, RBF and tree readouts test whether a PCA/linear bottleneck hides mean information. Distributions contain mean, SD and 10/25/50/75/90 percentiles per SPI.

**Readout refinement before confirmation.** The initial standardized full-catalogue z readout scored .625 on development, failing the target despite weak means (.4375–.625). Center-only z, as used in the earlier proof analysis, scored .8125 with the same 20 PCs and linear classifier; features already share a correlation scale, so there is no unit-conversion reason to standardize them individually. An exploratory audit retained both scalings, PCA20/full features and linear/RBF results; the center-only PCA20 linear model was selected for confirmation, not the highest-scoring RBF model. This choice was made after seeing development outcomes and frozen in `factor-parity-dominant-261004-confirmation.yaml` before releasing any held MPI. The original standardized result remains reported. A secondary training-complete z subset and edge-shuffled control use the same centered preprocessing.

The figures use fixed PCA/UMAP settings and training-fit/evaluation-transform, without searching seeds or selecting points. Numerical performance takes precedence over appearance. Conditional bootstrap intervals resample paired recording blocks, keeping fitted models fixed; they omit training and parameter-selection uncertainty. Near-chance finite-sample performance is not proof of information-theoretic absence.'''),
('code',"plot(STAGE,focused=False,run=RUN,z_method='z_center');plt.show()\npaired=pd.read_csv(RESULT/'paired.csv')\ndisplay(paired[paired.comparison.str.startswith('z_center - ')].round(3))\ndisplay(metrics[metrics.method.isin(['distribution','z_validity','z_shuffled_center','z_center_complete'])].round(3))"),
('md',r'''## Mechanism and limits

The classes differ in how signs associate with interaction magnitudes, even though their mean signed strength and entire absolute-strength distribution agree. A pair such as covariance versus squared covariance can respond to this association. For the selected population matrices, that z value is approximately −.0183 versus −.1054. This is not evidence of nonlinear dynamics: both models are Gaussian. Since the second probe is a function of the first, this particular feature also reflects the shape of the signed covariance distribution. It cannot establish information inaccessible to every marginal-distribution description.

The validity-only control uses missingness without numerical z values. The independently shuffled-edge control preserves every MPI marginal while removing alignment; it is a diagnostic transformation, not a realizable-MTS claim. Any class information in these controls must qualify the interpretation. Means of the full catalogue are not inherently blind to dependence character, so informative means would constitute another negative result for the proposed picture.'''),
('code',"display(metrics[metrics.method.isin(['probe_mean','probe_z','raw_covariance','raw_Pearson'])].round(3))\ndisplay(pd.read_csv(RESULT/'diagnostic-features.csv').round(3))\ndisplay(pd.DataFrame(analysis['diagnostics']).T.round(3))"),
('md',r'''The protocol is `configs/analysis/factor-parity-dominant-261004.yaml`; `build_factor_parity.py` constructs the immutable raw bank. `analyze_band_swap.py --run factor-parity-dominant-261004` audits input/member/catalogue/RNG provenance, extracts features and evaluates the fixed readouts. `sources.json` binds MPI hashes; `execution.json` records cluster jobs and the held-release decision. Rendering uses cached outputs and does not recompute SPIs.'''),
('code',"print(json.dumps(analysis.get('centered_development_goal',{}),indent=2))\nprint(json.dumps(json.loads((OUT/'execution.json').read_text()),indent=2))")]
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(s) if k=='md' else nbf.v4.new_code_cell(s) for k,s in cells])
    nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    nbf.write(nb,ROOT/'notebooks/embeddings/spi_factor_parity_261004.ipynb')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','final'],default='development');a=p.parse_args();build(a.stage)
