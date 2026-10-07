"""Build and execute the lean dial-network notebook; requires the extracted feature bank."""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
cells=[md(r'''# Dial network: classes that differ only in how channels interact — p90 pilot, 2026-10-07

The proof classes separate even with all coupling removed ([control](proof_strength_nuisance_261006.ipynb)), so they cannot show whether a representation reads interaction. Here every node is the same noise-driven process and only the coupling function changes; gain and node timescale vary per recording. Result: mean $|r|$ is near chance, the mean of each SPI, $m$, is organised by gain with classes mixed, and SPI–SPI $z$ is organised by class once coupling is detectable. $z$ is not blind to gain or timescale, and neither dial is recovered as a single coordinate. Exploratory pilot; settings were fixed by a strength-only scout before p90.'''),
code('''from pathlib import Path
import sys, warnings
import matplotlib.pyplot as plt, pandas as pd
ROOT = Path.cwd().resolve()
while not (ROOT / 'src').is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
warnings.filterwarnings('ignore')
from scripts import dial_network as D
rows, bank, P = D.load(); cache = {}
kw = dict(colors=D.COLORS, display=D.DISPLAY, stem='dial', cache=cache)
print(f"{len(rows)} recordings; {len(bank['spi_order'])} SPIs; M={D.M}, T={D.T}")'''),
md(r'''$x_i(t)=a\,x_i(t-1)+k\,g_i\sum_j W_{ij}\,\phi_\beta\!\big(x_j(t-\tau_{ij})/\sigma_a\big)+\varepsilon_i(t)$, with $\varepsilon\sim N(0,1)$ and $\sigma_a=(1-a^2)^{-1/2}$. Inputs follow a random directed acyclic graph with at most two inputs per node, so gain cannot destabilise the system and dependence rises with it. Rows of $W$ have unit norm and $g_i\sim U[.5,1.5]$, so pairs differ in strength within a recording. $\phi_\beta$ mixes a bounded odd term $\tanh u$ and a bounded even term $e^{-u^2/2}$, each standardized and mutually uncorrelated under $N(0,1)$.

Classes are two dials crossed: coupling shape ($\beta=0$ near-linear, $\beta=1$ even) and lag structure ($\tau=1$; $\tau=5$; $\tau\sim U\{1..9\}$ per edge), plus an uncoupled class; 40 recordings each. Per recording, independently of class: gain $k$ log-uniform on $[.25,1.2]$, node coefficient $a\sim U[.3,.7]$, and the graph. Sweeps (24 recordings per level) vary $\beta$ at lag 1, and the spread of lags around 5 at $\beta=0$. Even coupling leaves a driver and its receiver uncorrelated, but two receivers of one driver still correlate, so mean $|r|$ is not zero for those classes. Features and label-free fits as in the proof notebook.'''),
code("fig = D.design_figure(rows, P); plt.show()"),
md('Six classes and the uncoupled reference, one fit per representation. Top: colour is class. Bottom: the same points coloured by gain.'),
code('''fig = P.embedding_figure(rows, bank, D.PANEL, ('class',), D.OUT, **kw); plt.show()
fig = P.embedding_figure(rows, bank, D.PANEL, ('class',), D.OUT, color='value', value='gain', value_label='Gain $k$', **kw); plt.show()'''),
md(r'''Scores in the 50-PC space of each fit (seven labels, chance .14). `class_agreement` is the share of five nearest neighbours in the same class; `shape` and `lag` score each dial alone among coupled recordings; the three gain columns split coupled recordings into thirds. `share_*` is the fraction of variance between classes, and explained within class by gain and by $a$. `z_ordered` keeps direction (ordered pairs, no symmetrisation).'''),
code("table = D.class_table(rows, bank, P); table.to_csv(D.OUT / 'class-table.csv'); table.T"),
md(r'''Dose-response. If a dial can be read off, some coordinate should follow it rather than the gain. Rows are the leading label-free components and one SPI pair per dial named before any outcome was read; entries are $|\rho|$ with the dial, the gain and $a$.'''),
code('''fig = D.sweep_figure(rows, bank, P); plt.show()
sweeps = D.sweep_table(rows, bank, P); sweeps.to_csv(D.OUT / 'sweep-table.csv', index=False); sweeps'''),
md(r'''**Reading.**

- *The baselines are much closer to noise than $z$.* Neighbour class agreement: mean $|r|$ .24, $m$ .46, $z$ .73 (chance .14); grouped logistic accuracy on ten PCs .34, .66, .84. Each dial alone: shape .72 against .88, lag structure .57 against .79. The mean embedding is a fan ordered by gain with classes interleaved; the $z$ embedding has the uncoupled recordings at one tip and classes along separate branches.
- *Both need detectable coupling.* In the weakest third of gains agreement is .28 ($m$) and .53 ($z$); in the strongest .62 and .85.
- *$z$ is not blind to the nuisances.* Gain explains 36% of $z$ variance within class and $a$ 18%, against 46% and 16% for $m$; classes take 27% of each. The difference is in how the variance is arranged: gain moves $z$ along a class's branch, and moves $m$ along a direction shared by all classes.
- *Neither dial is recovered as a coordinate.* No leading component of $m$ or $z$ follows $\beta$ or lag spread more than it follows gain. The named pair $z$(covariance, Kraskov MI) follows gain (.75) rather than $\beta$ (.39): at high gain it is near .9 for every $\beta$, because receivers of a shared driver are ranked alike by both statistics. Lag spread leaves no trace in any readout here ($|\rho|\le .19$). The questions "how nonlinear" and "how variable are the lags" are not answered by this analysis.
- *Scope.* One synthetic family, $M=16$, $T=1000$, one pilot; class structure in $z$ is diffuse (silhouette .07), so this is organisation, not clean clusters. Direction-preserving $z$ adds little (.75).

Run: `scripts/dial_network.py` (`scout`, `prepare`, external-corpus farm, `extract`); rebuilt by `scripts/build_dial_network_notebook.py`. Full p90, estimator seed 261122, 496 recordings, none failed (two reran after a job walltime).''')]
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=ROOT/'notebooks/embeddings/dial_network_261007.ipynb'
NotebookClient(nb,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(path.parent)}}).execute()
nbf.write(nb,path);print(path)
