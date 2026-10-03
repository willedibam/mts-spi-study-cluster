"""Lean report of the fixed-strength band-swap experiment."""
from pathlib import Path
import argparse
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def build(stage):
    cells=[('md',r'''# Equal band strengths, different dependence organization

The main target is weak **full-catalogue (289-SPI) MPI means** alongside informative SPI–SPI. A three-SPI control supports interpretation; it does not substitute for this test. The preceding all-proof-class experiment matched a native-update Jacobian gain but gave 89.6% classification for both means and SPI–SPI. Matching that scalar did not remove marginal information.

Here two stationary Gaussian MTS classes have equal coupling strengths and different organization across frequency bands. This is a controlled character contrast, not a recreation of the historical dynamical classes.'''),
('code',f'''from pathlib import Path
import sys,json
ROOT=Path.cwd().resolve()
while not (ROOT/'pyproject.toml').exists() and ROOT!=ROOT.parent:ROOT=ROOT.parent
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from IPython.display import display
from scripts.build_band_swap import DATA,OUT
from scripts.plot_band_swap import plot,population
STAGE={stage!r}
RESULT=OUT/STAGE
analysis=json.loads((RESULT/'analysis.json').read_text())
metrics=pd.read_csv(RESULT/'metrics.csv')
print('Evaluation stage:',STAGE)
display(metrics[metrics.method.isin(['mean','mean_full','mean_RBF','mean_trees','z','z_validity'])].round(3))'''),
('md',r'''## Construction

Each recording has 16 channels and 1,000 observations. Three independent Gaussian Fourier fields occupy disjoint equal-width bands $(.035,.095)$, $(.17,.23)$ and $(.33,.39)$ cycles/sample, with translated smooth spectral envelopes and variance $1/3$ each. Their sum is the observed MTS. These are illustrative separated bands, chosen before p90 outcomes, not unique or physically privileged frequencies. The construction uses periodic stationary sampling, not a causal coupled-update model.

Within a band the population channel covariance is $C_g=\frac13[(1-\lambda)I+\lambda H_g]$, where $\lambda=.55$ and $(H_g)_{ij}=1$ when channels $i,j$ share one of four groups of four. The value .55 is a fixed moderate within-group correlation, not a fitted or optimal strength; the covariance is positive definite. A groups rows of a $4\times4$ grid, B groups columns, and C starts from A then swaps membership of channels 1/4 and 10/13 (zero-based), producing partial overlap. The patterns are channel permutations of one another.

**BCA** assigns B, C, A to low, middle, high frequency; **CBA** assigns C, B, A. The high-band pattern stays fixed while its strongest spatial correspondence moves between the lower bands. Random channel relabelling removes fixed sensor identities. Each band MPI has 20% of off-diagonal values equal to $.55/3$ and 80% zero: mean $.036\overline6$. Both classes also share their total zero-lag covariance, low-half covariance sum/maximum and high-band covariance.

For the three band-covariance probes, permutation symmetry and independent band draws give equal joint laws of marginal summaries across classes, including finite-sample estimates. **This does not prove equality of the 289-SPI mean distributions.** Other statistics can encode temporal structure in their means; that is the main empirical test.'''),
('code',"population();plt.show()"),
('md',r'''With $v_b$ the ordered off-diagonal entries of a band covariance, $z_{bc}=\mathrm{corr}(v_b,v_c)$ describes spatial correspondence. The population triples $(z_{LM},z_{LH},z_{MH})$ are $(-1/24,-1/4,3/8)$ for BCA and $(-1/24,3/8,-1/4)$ for CBA. Negative z means the bands tend to couple different pairs; the actual nonzero couplings are all positive. This is organization across bands, not causal cross-frequency coupling.

Imagine observers receiving the same coloured maps: one receives only each map's histogram, the other also sees which locations align across maps. For these three probes, the histogram observer cannot distinguish the classes; the alignment observer can. Whether the richer p90 catalogue supplies informative histograms remains open until tested.'''),
('md',r'''## Full-catalogue test

Development blocks 0–23 train the models (48 recordings); blocks 24–31 validate them (16 recordings). Separate held blocks 32–63 provide 64 recordings and are released only after the development decision; final training then uses blocks 0–31. Each block contains one recording per class and shared nuisance channel ordering. Chance is 50%.

The primary features are the 289 off-diagonal means and 41,616 signed Pearson SPI–SPI values. Training-only preprocessing uses 95% validity selection, median imputation, SD scaling, clipping at five SD and at most 20 principal components; logistic regression fixes $C=1$. Full-mean logistic, RBF kernel and 500-tree classifiers check whether the PCA or linear classifier hides information. Distributions (mean, SD and 10/25/50/75/90 percentiles) are secondary. Missingness-only z and independently shuffled MPI edge profiles are diagnostic controls; the shuffle preserves MPI marginals but is not a realizable-MTS claim.

Intervals resample paired blocks with the fitted models fixed, omitting training uncertainty. Near-chance performance on a small validation set is not evidence of statistical equivalence. PCA/UMAP settings and seeds are fixed; both fit training recordings and transform validation or held recordings. No outcome selects the displayed recordings.'''),
('code',"plot(STAGE,focused=False);plt.show()\ndisplay(pd.read_csv(RESULT/'paired.csv').round(3))\ndisplay(metrics[metrics.method.isin(['distribution','z_shuffled'])].round(3))"),
('md',r'''## Focused supporting control

The three explicit band-covariance SPIs are separate from p90. Their symmetry argument demonstrates the intended mechanism without establishing that the complete catalogue's means lack information.'''),
('code',"display(metrics[metrics.method.isin(['band_mean','band_marginals','band_z','raw_covariance','raw_Pearson','spectra'])].round(3))"),
('md',r'''## Scope and reproduction

A z advantage establishes utility for this construction and readouts, not generic strength invariance or causal interpretation. Informative full-catalogue means are negative evidence for the main target even if the focused control succeeds. SPI means are not inherently blind to dependence character; affine invariance of outer Pearson correlation does not imply invariance to a generator's coupling parameter.

Comparisons among representations have precedents in [representational similarity analysis](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full), and cross-layer overlap is an established [multiplex-network quantity](https://doi.org/10.1103/PhysRevE.89.032804). This example illustrates a mechanism, not novelty of overlap itself.

The immutable source manifest is in `data/representation/band-swap-261004/manifest.json`; the protocol is `configs/analysis/band-swap-261004.yaml`. `scripts/analyze_band_swap.py` audits source/member/config hashes, catalogue order and estimator seeding, then extracts features and evaluates the readouts. `sources.json` retains MPI hashes; `execution.json` records jobs and held-release status. This notebook renders cached numerical outputs without recomputing SPIs.'''),
('code',"display(pd.DataFrame(analysis['diagnostics']).T.round(3))\nprint(json.dumps(analysis.get('development_goal',{}),indent=2))\nprint(json.dumps(json.loads((OUT/'execution.json').read_text()),indent=2))")]
    nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(s) if k=='md' else nbf.v4.new_code_cell(s) for k,s in cells])
    nb.metadata.kernelspec=dict(display_name='Python 3',language='python',name='python3')
    nbf.write(nb,ROOT/'notebooks/embeddings/spi_band_swap_261004.ipynb')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','final'],default='development');a=p.parse_args();build(a.stage)
