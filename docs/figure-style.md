# Figure style

Default for new figures; adapt when clarity requires. Benchmark-specific conventions: [supplement](benchmark-figure-style.md).

- Minimal, unboxed: white background, Helvetica text with Computer Modern (TeX) math, 8–10 pt at final size, outward ticks, no top/right spines unless needed.
- Prefer square plots where appropriate or for embeddings (e.g. PCA, UMAP, t-SNE, etc.); deviate appropriately.
- Proof/case viewers use boxed, square embeddings and feature-coordinate rainclouds, and boxed SPI–SPI scatterplots. Raincloud limits may differ between SPI pairs; share limits across groups/conditions for the same pair and disclose the differing scales.
- Thin lines (`lw≈1.7`), small markers (`ms≈2.7`), frameless legends, uncluttered layout.
- Keep colour/marker meanings consistent across related plots. Use accessible palettes; sequential colours for ordered values, diverging colours around a meaningful centre.
- Label variables and units; disclose transformations and normalization. Share axis/heatmap scales for direct comparisons where appropriate; flag differing scales.
- Identify line summaries, band meanings and replication units. Confidence intervals, quantile spread and standard deviations are distinct.
- Use subtle grids, reference lines and annotations only when helpful. Keep titles, legends and captions short; add panel labels when useful.
- Text (labels, titles, tick numbers) is Helvetica; anything mathematical (symbols, Greek, sub/superscripts, operator names such as $\mathrm{MI}$) stays in TeX math. Under LaTeX the text font is `helvet` (URW Nimbus Sans, Helvetica metrics). Use `mtsspi.figures.style`/`figure_style` or the snippet below.
- Save SVG/PDF plus a PNG preview (180 dpi display; 600 dpi raster export). Check readability, clipping and overlap at intended size.

```python
import shutil
import matplotlib as mpl
from mtsspi.figures import SANS, SANS_PREAMBLE   # Helvetica text, CM math; default tick labels in Helvetica

mpl.rcParams.update({
    "text.usetex": bool(shutil.which("latex") and shutil.which("dvipng")),
    "text.latex.preamble": SANS_PREAMBLE, **SANS,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "axes.grid": False, "legend.frameon": False,
    "lines.linewidth": 1.7, "lines.markersize": 2.7,
    "figure.constrained_layout.use": True,
    "figure.dpi": 180, "savefig.dpi": 600, "savefig.bbox": "tight",
})
```
