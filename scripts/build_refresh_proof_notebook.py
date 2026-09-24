"""Build the lean 2026-09-24 proof notebook; execution requires the fresh feature bank."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
cells=[md(r'''# SPI–SPI proof of concept — p90, 2026-09-24

Fresh extraction with the current 289-SPI catalogue. The figures compare ten generator classes, then four CML regimes, and finally individual SPI–SPI coordinates that distinguish pairs of classes. These embeddings illustrate organisation; they do not establish invariance to channel count or recording length.'''),
code('''from pathlib import Path
import sys, json, os
import matplotlib.pyplot as plt
ROOT = Path.cwd().resolve()
while not (ROOT / 'src').is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.refresh_proof_figures import (load_bank, embeddings, rank_features, plot_embedding, plot_discriminative)
BANK = Path(os.environ.get('PROOF_REFRESH_BANK', ROOT / 'data/proof/refresh-260924/features.npz'))
OUT = Path(os.environ.get('PROOF_REFRESH_OUTPUT', ROOT / 'results/proof/refresh-260924'))
bank = load_bank(BANK)
OUT.mkdir(parents=True, exist_ok=True)
print(f"{len(bank['y'])} datasets; {len(bank['spi_order'])} SPIs; {bank['X_sym'].shape[1]:,} coordinates")'''),
md(r'''We retain the original proof's feature definition: symmetrise each interaction matrix, take its unique off-diagonal channel pairs, and calculate Pearson correlation between each pair of SPI vectors. Directed information is discarded in this particular view. This produces $\binom{289}{2}=41\,616$ possible coordinates.

Each class has ten instances at every combination of $M\in\{8,16,32\}$ and $T\in\{500,1000,2000\}$. Each panel fits its own unsupervised preprocessing: retain coordinates finite in at least 95% of its datasets, median-impute missing entries, discard near-constant coordinates and centre without variance scaling. PCA provides the linear view; UMAP uses the first 50 PCs, 25 neighbours, min_dist=0.25 and a fixed seed. Centring is a design choice, not something mathematically forced by the bounded coordinate range. No class labels enter these fits.

The three VAR models use a fixed ring with literal (self, total-neighbour) coefficients (0.70,0.20), (0.20,0.20), (0.20,0.70). Each neighbour receives half the total weight. These isolate changes in self-dependence and neighbour coupling against the same middle reference. Their spectral radii are 0.90, 0.40 and 0.90; no stability rescaling is applied. Innovation standard deviation is 0.1 and burn-in is 100 steps. This intentionally replaces the old pre-normalisation VAR weights; other generators retain their existing settings.'''),
code('''coords = embeddings(bank, OUT)
ranked = rank_features(bank, OUT)'''),
md('Inter-class view. Colours identify classes and remain fixed in subsequent panels. Point size encodes M=8,16,32. Axis coordinates from separate fits are not directly comparable.'),
code("fig = plot_embedding(bank, coords, 'inter', OUT)\nplt.show()"),
md('Within-CML view: frozen chaos, spatiotemporal intermittency, defect turbulence and fully developed turbulence. The same two bridge classes use the same datasets and colours as the inter-class view.'),
code("fig = plot_embedding(bank, coords, 'cml', OUT)\nplt.show()"),
md(r'''Discriminative coordinates. For each class pair, select the SPI pair with greatest training AUC separation using instances 0–5 across all nine M/T cells. The rainclouds display instances 6–9 only (36 datasets/class), with the training orientation fixed when calculating display AUC. Dots are datasets; density shapes are descriptive; short bars mark medians. This limits feature-selection optimism within this rerun, but remains exploratory because these generator families and earlier outcomes have already been inspected. No best-feature significance claim is made.'''),
code("fig = plot_discriminative(bank, ranked, 'cml', OUT)\nplt.show()"),
md('Coarse contrasts use the same selection rule for Gaussian versus Cauchy noise and all three VAR class pairs.'),
code("fig = plot_discriminative(bank, ranked, 'coarse', OUT)\nplt.show()"),
md('Generation and analysis settings are recorded in `configs/analysis/proof-cases-refresh-260924.yaml`. The twelve stochastic robust-covariance and SGD-barycentre summaries are replayed with each dataset’s recorded seed; per-dataset sidecars preserve the original values and repair provenance. Other SPI matrices are unchanged. The fresh bank retains invalid-coordinate masks and extraction provenance. Plot helpers save SVG figures and PNG previews under the result directory. Earlier notebooks and banks are preserved.')]
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=ROOT/'notebooks/embeddings/proof_p90_260924.ipynb';nbf.write(nb,path);print(path)
