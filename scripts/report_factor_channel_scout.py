"""Lean report separating cached-p90 audits from the cheap new channel-count scout."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nbformat as nbf
from scripts.scout_factor_channels import ROOT,OUT,RUN
from scripts.refresh_case_figures import style,save


def report():
    f=pd.read_csv(OUT/'metrics.csv');style();fig,ax=plt.subplots(figsize=(6.7,4.1),layout='constrained')
    means=f[f.method.str.startswith('mean_')].groupby('M').BA.max()
    for name,values,color,ls in [('Best cheap mean readout',means,'#0072B2','-'),('Focused z, centered',f[f.method.eq('z_center')].set_index('M').BA,'#D55E00','-'),('Signed covariance distribution',f[f.method.eq('covariance_distribution')].set_index('M').BA,'.5','--')]:
        ax.plot(values.index,values,color=color,ls=ls,label=name)
        ax.scatter(values.index,values,color=color,s=18*np.sqrt(values.index.to_numpy()/8))
    exact=pd.read_csv(OUT/'graphical-default-metrics.csv').BA.max()
    ax.scatter([32],[exact],color='#CC79A7',marker='s',s=50,label='Means + p90 graphical lasso')
    ax.axhline(.5,color='.65',ls=':',lw=1);ax.set(xticks=[8,16,32],ylim=(.35,1),xlabel='Channels M; T = 1000',ylabel='Development validation accuracy',title='Focused probes only — not a p90 result');ax.legend(loc='lower right',fontsize=8)
    save(fig,OUT/'figures','channel-scout')
    total=ROOT/'results/representation/factor-total-strength-scout-261004'
    cells=[nbf.v4.new_markdown_cell('# Mean-baseline audit and channel-count feasibility\n\n**Both channel-count candidates are rejected.** At M32, seven actual p90 covariance/precision means classify 90.6%; holding total coupling fixed weakens focused z to 51.6%. The factor classes are also exactly related by channel sign changes and relabeling, limiting their value as a dependence-character example. The aim remains weak raw **p90 MPI means** with a reliable SPI–SPI signal. This notebook first audits existing p90 banks with stronger mean readouts, then tests a channel-count hypothesis using a small custom probe catalogue. The new M32 results below are **not p90 results**. Signed distribution summaries remain informative.'),nbf.v4.new_code_cell(f'''from pathlib import Path
import pandas as pd,json
from IPython.display import display,Image
ROOT=Path.cwd().resolve()
while not (ROOT/'pyproject.toml').exists() and ROOT!=ROOT.parent: ROOT=ROOT.parent
OUT=ROOT/'results/representation/{RUN}'
for run in ['lag-surrogate-261004','factor-parity-replication-261004']:
    print(run)
    folder=ROOT/'results/representation'/run/'development/marginal-audit'
    display(pd.read_csv(folder/'metrics.csv').round(4))
    display(pd.read_csv(folder/'selected-features.csv').round(4))'''),nbf.v4.new_markdown_cell('The audit is exploratory reuse of already-seen development data; no held bank is opened. Single means are ranked using training standardized class differences, then fitted using training only. Sparse L1 logistic chooses C from .01/.1/1/10 by four grouped folds inside training; feature filtering, imputation and scaling are refitted in each fold, and both classes from a simulation block stay together. The surrogate mean vector reaches 100% with this sparse readout, resolving the apparent original weakness. The Gaussian-factor p90 mean controls remain near chance; that motivates a targeted channel-count scout rather than another covariance-only construction.'),nbf.v4.new_markdown_cell(r'''## What changes, and what is matched

The covariance is $C=(1-\lambda)I+\lambda\sum_{k=1}^3w_kv_kv_k^\top$, with unchanged $\lambda=.25$ and weights $(.9,.075,.025)$. Walsh columns $\{1,2,3\}$ give linked signs and $\{1,2,4\}$ unlinked signs. Both classes have the same population absolute covariance multiset, eigenvalues, row absolute totals and mean signed covariance at each M. They are temporally iid Gaussian models: the contrast is sign–magnitude organization, not nonlinear dynamics or causal coupling.

Fresh development seed 261022 generates 128 paired blocks, with 96 for training and 32 for validation. Each class/block has a 32-channel parent; the M8/16/32 views are nested and are not independent experiments. T stays 1000. Increasing M provides more channels and channel-pair observations; it does not make all edges independent. Per-edge population strength remains comparable, but total row coupling changes with M. The scout uses covariance, Pearson, squared Pearson, Spearman, squared Spearman, Euclidean distance, Gaussian plug-in MI, precision, squared precision and a maximum cross-correlation mean. Focused z uses the first nine matrices. These probes are not a replacement for p90.

The signed covariance distributions differ by construction. Consequently, success against means would not demonstrate that every per-SPI distribution is information-free. This is a substantive scope distinction, but these particular candidates fail even the mean-only target.'''),nbf.v4.new_code_cell("display(Image(filename=str(OUT/'figures/channel-scout.png')))\ndisplay(pd.read_csv(OUT/'metrics.csv').pivot(index='method',columns='M',values='BA').round(4))\ndisplay(pd.read_csv(OUT/'regularized-metrics.csv').round(4))\ndisplay(pd.read_csv(OUT/'graphical-default-metrics.csv').round(4))\ndisplay(json.loads((OUT/'graphical-tolerance-sensitivity.json').read_text()))"),nbf.v4.new_markdown_cell('The additional M32 check includes covariance/precision means and squared means under graphical lasso at alpha .01/.1, addressing a known leakage route. The screening scores are 93.8% at best. Restricting to seven actual p90 means (empirical covariance, empirical precision and its square, plus four GraphicalLasso covariance/precision means at the actual default alpha .01) still gives 90.6%. Tightening solver tolerances leaves this accuracy unchanged. Default solves produce four convergence warnings across 256 fits; all outputs are finite, precision log determinants agree with independent Cholesky calculations, and the maximum inverse residual is 5.9e-5. Warnings and tolerance sensitivity are retained; no p90 settings were changed. No Gadi job is justified for this candidate.'),nbf.v4.new_markdown_cell(r'''## One principled scaling follow-up

The first scout holds per-edge strength fixed, so total absolute coupling per channel grows with M. The additional prespecified check holds the M16 total $g=.25(.9\times16-1)=3.35$ fixed, using $\lambda=g/(.9M-1)=.1205036$ at M32. Fresh development seed 261023 and the same training/validation sizes are used. Mean controls are now 43.8–60.9%, but centered focused z is 51.6% and standardized z 57.8%. The mean weakness comes with loss of the desired z signal. This branch stops; no parameter sweep or p90 submission follows.'''),nbf.v4.new_code_cell("display(pd.read_csv(ROOT/'results/representation/factor-total-strength-scout-261004/metrics.csv').round(4))"),nbf.v4.new_markdown_cell(r'''## More fundamental interpretation check

Writing the sign vectors as rows of $V$, a channel permutation $P$ and diagonal sign matrix $D$ satisfy $V_B=V_APD$. Therefore

$$C_B=D P^\top C_A P D.$$

The certificate below verifies this exactly at M8,16,32. Since the processes are Gaussian iid, their laws are related by the same signed channel transformation. Thus separation can reveal coordinate polarity sensitivity; it does not establish a different intrinsic Gaussian dynamical mechanism. Signs can have scientific meaning under a fixed measurement convention, but no such domain interpretation was established here. This makes the factor family a poor flagship example for the stated motivation, independently of the classifier failures. No held banks were opened in this follow-up, and no new p90 jobs were launched.'''),nbf.v4.new_code_cell("proof=json.loads((OUT/'polarity-equivalence.json').read_text()); display(proof['identity']); display(pd.DataFrame(proof['records'])[['M','max_covariance_error']])\ndisplay(json.loads((OUT/'protocol.json').read_text()))\ndisplay(json.loads((OUT/'regularized-validity.json').read_text()))")]
    nb=nbf.v4.new_notebook(cells=cells);nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    p=ROOT/'notebooks/embeddings/spi_factor_channel_scout_261004.ipynb';nbf.write(nb,p);print(p)

if __name__=='__main__':report()
