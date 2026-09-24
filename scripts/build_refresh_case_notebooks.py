"""Build lean, executable viewers for the September 24 focused case reruns."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def build(kind):
    is_mi=kind=='r-rho-mi'
    name='r_rho_mi_260924.ipynb' if is_mi else 'pdist-euclid_dtw_260924.ipynb'
    title='Pearson, Spearman and mutual information' if is_mi else 'Euclidean distance, global shifts and DTW'
    folder='r-rho-mi-260924' if is_mi else 'dtw-euclidean-260924'
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    cells=[md(f'# {title} — 2026-09-24\n\nFresh extraction with current pyspi. Each point below represents one simulated multivariate time series; channel pairs within it are not independent replicates.'),
           code(f'''from pathlib import Path
import sys, json
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path.cwd().resolve()
while not (ROOT / 'src').is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.refresh_case_figures import (analyze, plot_rainclouds, plot_mi_sensitivity, plot_spi_planes, plot_dtw_sweep)
DATA = ROOT / 'data/cases/{folder}/data'
OUT = ROOT / 'results/cases/{folder}'
# Analyse existing MPI archives; this never reruns pyspi.
frame = analyze(DATA, {kind!r}, OUT)
audit = json.loads((OUT / 'audit.json').read_text())
print(f"{{audit['datasets']}} datasets; {{len(audit['spis'])}} focused SPIs; {{audit['nonfinite_off_diagonal_values']}} nonfinite off-diagonal values")''')]
    if is_mi:
        cells += [md(r'''A shared AR(1) mother ($a=0.8$) is transformed into linear, sigmoid and quadratic channels. Each clean channel is standardised before independent measurement noise is added; the final channel is standardised again. Thus empirical covariance equals Pearson $r$ here. We retain $M=32$, $T=2000$, 100 instances/class, noise floor 0.01 and geometric multiplier mean 5.

The three mixtures contain (linear, sigmoid, quadratic) channel counts (32,0,0), (16,16,0), (12,12,8). Within each mixture, the β variants share the same mother and noise draws; mixtures use independent seeds. The all-linear baseline is computed once and reused in the three panels. These are illustrative constructions, not universal claims about all linear or nonlinear systems.'''),
            md(r'''For each MTS, $f_{ab}$ is Pearson correlation between two SPI vectors over unique unordered channel pairs. **MI is in nats throughout**; no Linfoot transformation enters these features. Dots are the 100 independent MTS instances, shaded shapes describe their distribution, and short bars mark medians.'''),
            code("fig = plot_rainclouds(frame, OUT)\nplt.show()"),
            md('The SPI–SPI planes below use the predetermined instance 0 at β=5. Grey points are same-family channel pairs; coloured points are mixed-family pairs. The annotation uses all pairs.'),
            code("fig = plot_spi_planes(frame, 'r-rho-mi', OUT)\nplt.show()"),
            md(r'''Temporal-exclusion sensitivity: ordinary KSG with $k=4$ remains primary. The extra AUTO variant excludes temporally close candidate neighbours using pyspi's pair-specific window. This checks sensitivity to serial dependence; it does not establish that either estimator is unbiased. The panel shows paired changes in the MTS-level feature, not uncertainty based on channel-pair counts. [KSG reference](https://arxiv.org/abs/cond-mat/0305641).'''),
            code("fig = plot_mi_sensitivity(frame, OUT)\nplt.show()")]
    else:
        cells += [md(r'''Each channel is a noisy copy of an AR(1) mother ($a=0.5$), with a fixed lag and independent rebound warping. We retain $M=20$, $T=1000$, 100 instances/condition, lag range 0–5, fixed excursion size 3, noise base 0.1 and uniform multiplier range $[1/2.718,2.718]$. The first control has neither lag nor warping; the second has lag only.

All distances use the same squared-error cost and division by $\sqrt{T}$. The shifted distance minimises over shifts −10 through 10, including boundary costs; unrestricted DTW searches a larger set of admissible paths. Therefore DTW ≤ shifted ≤ Euclidean is a structural property, not evidence that DTW recovered the physical warp. The divisor is the record length, **not warping-path length**.'''),
            md('Lines show mean SPI–SPI Pearson agreement; bars are pointwise 95% Student-t intervals over the 100 independent MTS instances. The first control is separated because it has a different lag range.'),
            code("fig = plot_dtw_sweep(frame, OUT)\nplt.show()"),
            md('Predetermined instance 0 illustrates no lag/no warp, lag only, and heavy warping. Colour shows the absolute generating lag difference. Dashed lines mark equal distances.'),
            code("fig = plot_spi_planes(frame, 'dtw', OUT)\nplt.show()"),
            md('Averaging warp offsets across channels does not guarantee exact lag recovery for individual pairs. The optimal global shift can differ from the generating lag even without noise. Correlation between distance matrices describes agreement across pairs; it is not an alignment-accuracy score.')]
    cells.append(md('Generation and analysis choices are recorded in `configs/analysis/proof-cases-refresh-260924.yaml`. The helpers save SVG figures and PNG previews under the result directory. Earlier notebooks and result banks are preserved.'))
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    path=ROOT/'notebooks/cases'/name;nbf.write(nb,path);print(path)


if __name__=='__main__':
    build('r-rho-mi');build('dtw')
