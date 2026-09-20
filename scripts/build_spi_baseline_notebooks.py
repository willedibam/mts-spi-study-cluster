"""Build the lean baseline notebook and its separate descriptive corpus appendix."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def notebook(cells, path):
    n = nbf.v4.new_notebook(cells=[(nbf.v4.new_markdown_cell(text) if kind == "md" else nbf.v4.new_code_cell(text)) for kind,text in cells])
    n.metadata.kernelspec = {"display_name":"Python 3", "language":"python", "name":"python3"}
    nbf.write(n,path)


SETUP = '''from pathlib import Path
import sys, json
ROOT = Path.cwd().resolve()
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from scripts.spi_baseline_exploration import OUT, DATA
from scripts import plot_spi_baseline_exploration as plots
plots.style()
pd.set_option("display.max_colwidth", 60)
'''


def build():
    cells = [
("md", r"""# What does SPI–SPI add beyond simple summaries?

A focused exploratory comparison: an exactly specified intuition example, the proof-of-concept inter-class and CML embeddings, and two headline physical-order benchmarks. All p90 MPIs are reused from existing runs. Only the small, two-statistic VAR illustration is newly simulated. Original notebooks and frozen SPI–SPI coordinates are preserved.

**Question:** do relationships among dependence measures add useful structure beyond the average level or distribution of each measure? This is an empirical question. A more elaborate representation need not win.

The new comparisons reuse previously inspected data. They are retrospective, even when a system's original SPI–SPI result was independently confirmed. The separate [Zenodo appendix](spi_baseline_zenodo_260921.ipynb) explores the heterogeneous corpus without assigning a superiority score to its appearance."""),
("code", SETUP),
("md", r"""## The abstraction and the baselines

An ordinary interaction network has measurement channels as its nodes. SPI–SPI can instead be viewed as a network whose nodes are **fixed scientific measures of dependence**. Its links describe their relationships within one recording. This changes the objects being compared: recordings need not share channel identities or channel counts, because the measures provide common reference points. A possible scientific interpretation is a *system-specific geometry of dependence measures*. This is a framing hypothesis, not evidence of a unique mechanism or a claim about the supervisor's exact intention.

This has a conceptual parallel with [representational similarity analysis](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full), which relates systems through relational structure without matching their measurement channels. The mathematical objects differ: our anchors are SPI definitions, and each recording determines relationships among those definitions. [Cliff et al.](https://arxiv.org/abs/2201.11941) already established the diverse SPI library and its empirical relationships across a large corpus; the library/corpus themselves are not new here.

For an off-diagonal edge vector $a_k$ from SPI $k$, define $m_k=\mathrm{mean}(a_k)$ and $z_{k\ell}=\mathrm{corr}(a_k,a_\ell)$. The baseline $m=(m_1,\dots,m_K)$ has **289 entries**, not one. It already supplies a common space. Thus common dimensionality alone cannot justify using the 41,616-entry $z$.

| Representation | What it tests | What it omits |
|:--|:--|:--|
| Mean Pearson $r$ and mean $|r|$ | Is elementary average linear dependence sufficient? | Most temporal and nonlinear structure |
| One mean per SPI | Is the catalogue's average response sufficient? | Within-SPI distributions and cross-SPI correspondence |
| Seven summaries per SPI: mean, SD, 10/25/50/75/90th percentiles | Does a gain survive retaining distribution shape? | Which values from different SPIs belong to the same pair |
| SPI–SPI | Does the relationship between different dependence profiles help? | Per-SPI means, positive scales and edge incidence |
| Means + SPI–SPI; distributions + SPI–SPI | Does combining complementary descriptions help? | Still a compressed representation; fusion is not guaranteed to help |

Signed means can cancel. Absolute values are therefore included for Pearson, but are not applied indiscriminately to every SPI: SPIs include signed statistics, distances and other quantities with different meanings. A finite constant MPI has a meaningful mean but undefined Pearson SPI–SPI correlations; missing values are not interpreted as zero dependence.

Feature length is independent of $M,T$; statistical values and estimator error need not be. Means also retain potentially useful information deliberately removed by Pearson SPI–SPI. Neither representation is universally richer. A statistic's output need not equal a physical coupling parameter. Catalogue-matched mean/distribution baselines still require the same p90 extraction; their simpler aggregation does not remove that cost. Raw Pearson is a genuinely cheaper route."""),
("md", r"""## 1. An intuition case whose population answer is known

Use a six-channel stationary Gaussian VAR(1): $x_t=A x_{t-1}+\epsilon_t$. Prescribe $C=\mathrm{Cov}(x_t)$ with unit diagonal and $L=\mathrm{Cov}(x_t,x_{t-1})$, then set

$$A=LC^{-1},\qquad \mathrm{Cov}(\epsilon_t)=C-LC^{-1}L^\top.$$

We check positive innovation covariance, stationarity and spectral radius below one. For the 15 unordered pairs, take evenly spaced $v_{ij}\in[-1,1]$ and set $C_{ij}=s+0.035v_{ij}$, $L_{ij}=0.01+0.03hv_{ij}$, with $s\in\{0.02,0.10\}$ and $h\in\{-1,+1\}$. Diagonals are $C_{ii}=1,L_{ii}=0.2$. Here $s$ controls average contemporaneous correlation, not physical coupling strength.

Changing $s$ changes the mean of C while leaving population $z(C,L)=h$ unchanged. Changing $h$ swaps aligned and opposed contemporaneous/lagged dependence profiles while preserving **the entire marginal distribution of each of these two MPIs**. Therefore their means, SDs and quantiles match across $h$. Both processes remain Gaussian and linear; a difference in z does not diagnose nonlinearity.

Each cell has 24 paired realizations, $T=6000$. Within each seed a common random dyad permutation is used for C and L; simulations start from the exact stationary distribution. The two probes are ordinary zero-lag and lag-one Pearson correlation, not the full p90 library. This is an information example, not evidence that SPI–SPI is the best estimator or predictor."""),
("code", "fig = plots.intuition()\nplt.show()"),
("code", "display(pd.read_csv(OUT / 'intuition-population.csv')[['strength','orientation','mean_C','mean_L','z']].rename(columns={'strength':'s','orientation':'h'}).round(3))"),
("md", """The exact table separates two roles: means describe average dependence; z describes how the chosen notions of dependence relate. Their combination distinguishes the four population cases. Finite-recording estimates scatter around that distinction. No mean-preserving manipulation of an arbitrary MPI is assumed realizable: these are actual stationary stochastic processes. Direct estimation of C and L also contains this information, so the illustration establishes a difference between compressions, not a unique SPI–SPI advantage. It does not establish that the full 289-SPI mean baseline fails on these processes: other SPIs may distinguish them."""),
("md", r"""## 2. Headline proof embeddings: inter-class and within CML

Use the original 1,260 development recordings (instances 0–9) and 2,520 evaluation recordings (10–29), spanning 14 classes and nine $M,T$ cells. The displayed maps retain the original exclusion of `brownian-defect`: 13 inter-class groups and five CML regimes. All 14 classes remain in the quantitative inter-class comparison and in the inherited SPI–SPI PCA basis.

This historical proof symmetrizes each MPI before taking its upper triangle. The new distribution baseline uses the same edges. Ordered-edge means would be identical when finite, but their distributions need not be. The original pyspi inputs were not channel-z-scored.

Each new baseline uses development-only 95% feature validity, median imputation, SD scaling and clipping at ±5 development SD, followed by PCA50 (or all available dimensions if fewer). Scaling is necessary because different SPIs have different units. The original frozen centre-only SPI–SPI PCA50 is reused exactly. Each feature block is normalized to unit total development variance; combinations concatenate the two PCA blocks and apply PCA50 again. This is one explicit fusion rule, not an optimal combination search.

For each representation, global UMAP is fitted on the 13 displayed development classes; a separate UMAP uses only the five development CML classes. Evaluation points are transformed into those maps. The PCA panels similarly fit a two-dimensional rotation on the relevant development subset. UMAP settings are fixed across representations: 30 neighbours, min_dist=.1, seed260824. Map shape, axis scale and inter-cluster gaps are not direct performance measures."""),
("code", "fig = plots.proof('umap')\nplt.show()"),
("code", "fig = plots.proof('pca')\nplt.show()"),
("md", r"""**Quantitative checks in the fitted PCA space, not UMAP:** fixed logistic regression ($C=1$, no tuning) predicts class on evaluation realizations. Hard retrieval uses development recordings as the gallery and excludes every gallery item with either the same M or the same T as its query. Average precision is averaged across queries. CML classification/retrieval uses the five CML classes and their development rows. All methods receive identical records and splits.

These fits pool all development sizes. They do **not** reproduce the original leave-one-size-cell-out confirmation test or establish performance on previously unseen sizes. The table supports a focused retrospective comparison of representations. Raw Pearson has only two coordinates; reducing every method to two coordinates would answer a different question."""),
("code", "display(plots.proof_table())"),
("code", "paired = pd.read_csv(OUT / 'proof-paired.csv')\ndisplay(paired.query(\"metric == 'ap'\").round(3))"),
("md", """**Reading the result:** SPI–SPI improves hard retrieval over the seven-summary baseline: all-class mAP .818 versus .714; CML mAP .940 versus .766. The all-class classification gain (.983 versus .925) is much more concentrated: about 98% of its **net** gain comes from `var-phi-0.2_cpl-0.4` and `var-phi-0.2_cpl-0.8`. Thus this is not broad evidence for detecting nonlinear mechanisms. Brownian-defect is worse under z. Within CML, classification is near ceiling and its z–distribution difference is unresolved, whereas the retrieval difference is substantial. Equal-variance fusion improves on the corresponding marginal baseline but does not uniformly improve on z.

Paired intervals resample class–instance groups, keeping their nine observation cells together, with 2,000 class-stratified bootstrap draws. They condition on the fitted development models; they do not include development-sample uncertainty or multiplicity correction. Brownian-defect and other weak classes must not be removed to improve the headline score. Exact per-record outcomes and the complete paired table are retained alongside the notebook."""),
("md", r"""## 3. Does the order-coordinate require SPI–SPI?

Two existing examples are reused, chosen to span synchronization and a different kind of collective order:

- **Kuramoto:** $\dot\theta_i=\omega_i+K N^{-1}\sum_j\sin(\theta_j-\theta_i)$; $M=N=32$, cosine observations, $T=1000$. Physical $Q$ is future mean global phase coherence $\langle|N^{-1}\sum_j e^{i\theta_j}|\rangle$. Use the original 128 development and 128 held recordings. Its original SPI–SPI PC1 misses the prechosen dominance screen; the reported association remains exploratory held-seed evidence.
- **2D coupled logistic maps:** $x_i(t+1)=(1-4g)f_r(x_i(t))+g\sum_{j\sim i}f_r(x_j(t))$, $f_r(x)=rx(1-x)$, $g=.2$, periodic $256\times256$ lattice. Physical $Q$ is the long-run absolute difference between successive even/odd spatial-mean states. Only 32 dispersed sites and 1,000 samples are observed. Baselines fit the original 36 development recordings and are applied unchanged to all 544 independent-confirmation recordings. The original frozen SPI–SPI coordinate is preserved. See the [original comparison notebook](../inference/order-parameter-benchmark-comparison.ipynb) for physical definitions, convergence checks and qualifications.

New **unsupervised** coordinates use PC1 of the means, PC1 of the seven-summary block, and PC1 of combined means and the original selected SPI–SPI features. Filtering/scaling/centering/PCA use development observations only; neither Q nor control values select features or components. Fusion uses unit total development variance per block before PCA. The original SPI–SPI q is independently reconstructed from its stored model and checked against the original scores. All primary evaluation rows pass the common eligibility check.

Black curves show Q in physical units. Coloured learned coordinates use development SD units on the right axis. Their arbitrary signs are oriented using **development Q for display only**; this changes neither fitted representations nor absolute-rank scores. The direct mean-|r| baseline retains its raw units. Bands are 95% bootstrap intervals for the across-seed mean at each control. Dotted lines mark published boundary references, not exact finite-system thresholds."""),
("code", "fig = plots.inference()\nplt.show()"),
("code", "display(plots.inference_table())"),
("code", "intervals = pd.read_csv(OUT / 'inference-transition-intervals.csv')\nintervals['steepest interval'] = [f'[{a:g}, {b:g}]' for a,b in zip(intervals.interval_start, intervals.interval_end)]\ndisplay(intervals.pivot(index='method', columns='system', values='steepest interval'))"),
("md", r"""Associations use individual held recordings, not just control means. Intervals use 2,000 paired seed-cluster bootstrap draws, preserving each seed's control sweep. The difference columns are baseline minus original q; these retrospective intervals are unadjusted. The within-control column removes each control's mean before computing association and must not be read as longitudinal fluctuation tracking.

The full mean-SPI PC1 tracks Q strongly in both examples, and combining it with SPI–SPI does not improve on the mean coordinate. Thus these headline order-tracking results do not establish a need for SPI–SPI under these readouts. A weak first component does not prove the full z vector lacks useful information, and a leading variance direction is not automatically a physical order parameter.

**Preprocessing sensitivity:** the table also gives PC1 of standardized z, using the same 95% validity, development SD scaling and ±5-SD clipping as the mean baseline. This check was added after observing the primary baseline results because preprocessing is a potential confound. It does not replace the frozen q or erase its original gate outcome. All methods remain retrospective and target-blind in fitting. Any advantage of one coordinate describes that representation/preprocessing combination, not every possible readout of its input.

Standardizing z improves association to .964 (Kuramoto) and .941 (CML), still below the mean coordinate (.994/.973). This sensitivity shows why the original PC1 result must not be equated with all information in z. **Rank association and transition localization are different:** on CML, q, distribution-PC1 and fusion share Q's steepest sampled interval; mean-PC1 and mean-|r| peak in the adjacent earlier interval. The interval table reports the largest absolute slope of each control-mean curve on the existing nonuniform grid, not a fitted critical point or a statistical test of localization. Kuramoto's finite-population Q interval also differs from the continuum reference and the baselines' interval. No single correlation score settles every scientific objective.

Simple coherence baselines have a physical motivation here. For phase measurements, $R^2=N^{-2}\sum_{ij}\cos(\theta_i-\theta_j)$, so average pairwise phase alignment is directly related to global synchronization. Time-series Pearson correlation of cosine observations can approximate phase alignment under suitable oscillatory/window conditions, but is not identically that observable in general. Hilbert coherence is a domain-sensitive comparator; for CML, the sampled period-two proxy uses the physical observable's form and is a model-informed comparator, not a generic baseline."""),
("code", "diagnostics = json.loads((OUT / 'inference-provenance.json').read_text())['diagnostics']\nrows = [dict(system=s, representation=k, **v) for s, d in diagnostics.items() for k, v in d.items() if k in ['mean','distribution','mean+z']]\ndisplay(pd.DataFrame(rows).round(3))"),
("md", """The loading-stability diagnostic leaves out each development seed with preprocessing held fixed, matching the style of the existing SPI–SPI diagnostic. It is not a full refit uncertainty estimate, and these new baselines are not passed through a retrospectively chosen physical-discovery gate.

## What this first pass can establish

The relevant distinction is not “simple strength” versus “all character.” Different measures already encode different scientific notions of dependence. SPI–SPI changes the representation to their system-specific relational geometry. Means offer an essential competing common space; distributions test whether a benefit survives a less severe marginal compression. The VAR example makes the distinction exact, while the proof and inference studies test whether it is useful in the existing applications.

Interpret gains by task and class, preserve the simple-baseline wins, and do not infer a unique dynamical mechanism from a position in the map. Adding a representation can also hurt under a finite PCA/readout budget. These results motivate a scoped claim about useful descriptions and their limits, not universal superiority or M,T invariance.

## Reproduction and provenance

This notebook renders the compact results produced by `scripts/spi_baseline_exploration.py`; it does not silently recompute p90. Protocol: `configs/analysis/spi-baseline-exploration-260921.yaml`. Input inventory: `data/spi_baseline_exploration_260921/proof-inputs.json`. All proof, Kuramoto and CML MPI archives are checked against their earlier recorded SHA-256 values. Source hashes and original-q replay are retained in the output provenance JSONs.

From the repository root, use the project Python to run `python -m scripts.spi_baseline_exploration toy`, `extract-proof`, `proof`, `inference`, and optionally `zenodo`. Missing cached proof files can be inventoried with `scripts/prepare_spi_baseline_data.py`; the resulting rsync list is for Gadi data-mover retrieval, not simulation. The original notebooks are unchanged. The separate corpus appendix is descriptive and transductive. Static figure files, scores, per-record outcomes and provenance live in `results/spi_baseline_exploration_260921/`."""),
    ]
    notebook(cells, ROOT / "notebooks/embeddings/spi_baseline_exploration_260921.ipynb")
    notebook([
("md", """# Per-SPI baselines on the 1,053-record corpus

Companion to the [baseline notebook](spi_baseline_exploration_260921.ipynb). This is a descriptive comparison of common spaces, not a claim that one arrangement is scientifically superior. All three views use the same 1,053 recordings and the same seeded p90 run. No new SPIs are computed.

The source archive stores arrays as M × T, with M=5–29 and T=30–3000. The original runner z-scored channels. Ordered off-diagonal entries retain both directions. Each MPI contributes its mean, or seven marginal summaries; SPI–SPI uses the existing 41,616 Pearson cross-statistic correlations.

Preprocessing is fitted **transductively on all 1,053 rows**: 95% feature validity, median imputation, SD scaling and clipping at ±5 SD for marginals; centre-only for z; then PCA50 and UMAP with 30 neighbours, min_dist=.1, seed260824. These are matched working recipes, not a parameter search. The z map is a new view using this declared recipe, not an exact reconstruction of every historical explorer view."""),
("code", SETUP),
("code", "fig = plots.zenodo('origin')\nplt.show()"),
("md", """Colour is source origin, using the existing corrected metadata (566 real, 487 synthetic). Origin is an annotation, not an independent definition of dynamical similarity. Separating real and synthetic recordings more strongly is not automatically a better representation. Axes, rotations and cluster gaps are not comparable numerical quantities between independently fitted maps."""),
("code", "fig = plots.zenodo('duration')\nplt.show()"),
("code", "display(pd.read_csv(OUT / 'zenodo-descriptive.csv').round(3))"),
("md", """The table reports variance captured by PCA50 and absolute Spearman correlations of PC1 with M and T. These are limited nuisance diagnostics, not tests of invariance: size information can live outside PC1. The duration colouring exposes one possible organizing factor without making a mechanistic interpretation.

No cluster-quality score is used here because the corpus has no single validated partition of dynamical mechanisms. Source labels are multi-label; duplicates and nested windows also complicate independence. A defensible next comparison requires an external scientific task or a prespecified set of relations to recover. Agreement with z's own clusters would be circular evidence for z.

The established [interactive explorer](../../results/zenodo_7118947/visual-exploration-v2/index.html) remains useful for inspecting individual recordings and source groups. Its geometry need not match these new panels. Primary sources: [Cliff et al.](https://arxiv.org/abs/2201.11941) for the SPI library/corpus; [Navarro et al.](https://proceedings.mlr.press/v224/navarro23a/navarro23a.pdf) for an existing alternative based on fixed-length pooled Catch22 descriptors.

Reproduce with `python -m scripts.spi_baseline_exploration zenodo` from the repository root. It reads only the seed1729 MPI bank. Compact arrays, source hashes and descriptive metrics are saved under `results/spi_baseline_exploration_260921/`; the exact formulas and remaining comparisons are in the main notebook."""),
    ], ROOT / "notebooks/embeddings/spi_baseline_zenodo_260921.ipynb")


if __name__ == "__main__":
    build()
