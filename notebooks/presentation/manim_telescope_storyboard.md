# Manim telescope storyboard — methods pipeline L0 → L4

Animated "methods" figure: zoom through the SPI-SPI feature-construction pipeline as a
self-similar telescope. Derived from `figures-m7-260617.ipynb` and `figures-cell-260616.ipynb`.
**Storyboard only — no Manim code yet.** Manim Community Edition, `MovingCameraScene`.

## Core idea — the self-similar telescope

The pipeline visits two *matrix* levels and two *scatter* levels; each is the same object
one level up. A single **traced element** (one channel-pair) is carried through every level
via the notebook's `highlight={(i,j): color}` machinery, then itself becomes one cell of the
final matrix.

| Level | Object | Each unit is… | Source (notebook cell) |
|---|---|---|---|
| **L0** | MTS heatmap `∈ ℝ^{T×M}` | a (channel, time) sample | figures-m7 cell 3 |
| **L1** | channel–channel phase portrait | **a timepoint** `t` | cell 8 `plot_channel_scatter` |
| **L2** | 3× MPI heatmap `∈ ℝ^{M×M}` (r, ρ, MI) | **a channel-pair** (i,j) | cell 10 `plot_mpi_heatmap` |
| **L3** | barcode → SPI–SPI scatter, `f_ij` | a channel-pair (i,j) | cells 12, 14 |
| **L4** | feature matrix `∈ ℝ^{N_MTS × N_feat}` → embedding | **an SPI-pair** (a,b) | cell 16 (stub) |

Reindexing to keep legible: **L1 points = timepoints; L3 points = channel-pairs; L4 columns = SPI-pairs.**
Same visual grammar, one level up each hinge.

## Data

- **L0–L3 (M7, teaching example):** `data/r_rho_mi/260617_g-tanhsin_a-0.95_k-2pi_noise-0.05/A-pi_L/M7_T100_I0`
  - `timeseries.npy` (100×7); `spi_mpis.npz` = {`cov_EmpiricalCovariance` (r), `spearmanr` (ρ), `mi_kraskov_NN-4` (MI)}, each 7×7.
  - M=7 → **21 = C(7,2)** off-diagonal channel-pairs (barcode length, scatter point count).
- **L4 (M10, "the real study"):** `features/data-embeddings-proof_benchmarked90_260603_pearson.npz`
  - `X` = **900 MTS × 40,184 features** (297 SPIs, ~C(297,2) pairs minus filtered), `y` = 10 system classes (90 each).
  - Columns for (r,ρ),(r,MI),(ρ,MI) present — pin exact indices at build time by exact SPI name.
- **L4 capstone (optional):** existing UMAP `notebooks/embeddings/graphics/umap_benchmarked90_260603.svg` (900 MTS, 10 classes).
- **Seam to flag in narration:** L0–L3 teach on M7; L4 zooms out to the 900-MTS proof_benchmarked90 study (different dataset/T). Deliberate "teaching example → full study" jump.

## Global design

- **Camera:** discrete levels joined by `ReplacementTransform`, with two hard **zoom hinges** where a
  whole panel collapses to a single cell: **L1→L2** (portrait → MPI cell) and **L3→L4** (scatter → feature cell).
  (Alternative: one continuous single-canvas zoom — more elegant, harder to keep legible. Recommend discrete.)
- **Traced pair (i,j):** one channel-pair carried through all levels (red in r, blue in ρ, green in MI, per cell 10).
  Pick at build time from M7 — heuristic: the pair whose r/ρ/MI read most distinctly (motivates "why >1 SPI").
- **Palette:** icefire (MTS, vmin/vmax = ±2); gray (MPIs + barcode, per notebook); highlight red/blue/green
  for the traced cell across r/ρ/MI; magenta regression line (`reg_color`); grey scatter points, α≈0.2.
- **Glyphs (MathTex):** `X_t^{(i)}`, `X_t^{(j)}`, `M`, `T`, `r=`, `\rho=`, `\mathrm{MI}=`, `\Rightarrow`,
  `\in\mathbb{R}^{M\times M}`, `f_{ij}:=\mathrm{corr}(\mathrm{SPI}_i,\mathrm{SPI}_j)`. (These are the `plot_latex` glyphs from cell 18.)
- **Timing budget:** ~65–75 s (per-beat below; adjustable).

## Beat sheet

### L0 — MTS heatmap  (0:00–0:09)
- **B1** `0:00–0:03` — FadeIn 7×100 icefire heatmap, centered. Draw axis ticks `M` (↕ left), `T` (→ bottom).
  *Manim:* `ImageMobject` from icefire-mapped z-scored array (or 7 row-strips if per-row animation wanted); `FadeIn`; `MathTex`.
- **B2** `0:03–0:07` — Two `Brace`s slide onto rows i and j; labels `X_t^{(i)}`, `X_t^{(j)}`. Rows i,j brighten; others dim to ~40%.
  *Manim:* `Brace`, `.animate.set_opacity`. (This is the notebook's `\Biggr]` bracket.)
- **B3** `0:07–0:09` — Rows i,j copy out and slide right into two horizontal 1-D strips.
  *Manim:* `TransformFromCopy`, `VGroup.animate.shift`.

### L1 — channel–channel phase portrait  (0:09–0:22)
- **B4** `0:09–0:12` — The two strips swing into the x- and y-axes of a new `Axes` (channel i → x, j → y).
  *Manim:* `Create(Axes)`, `Transform` strips → axis tracks.
- **B5** `0:12–0:17` — 100 dots fly in: dot t at `c2p(X^i_t, X^j_t)`. Optional sweep marker runs both strips dropping each dot. **Point = a timepoint.**
  *Manim:* `LaggedStart` of `GrowFromCenter`; optional `ValueTracker` sweep.
- **B6** `0:17–0:20` — Regression `Line` draws; `r` counts up. Quick toggle same cloud → rank axes (ρ) → raw+lowess (MI), each with its scalar.
  *Manim:* `Create(Line)`, `DecimalNumber`, small `Transform` for rank remap.
- **B7** `0:20–0:22` — The three scalars lift out as three small colored tiles (red/blue/green). Hold.
  *Cue:* "these 3 numbers are one cell of each MPI."

### L2 — MPI heatmaps (r, ρ, MI)  (0:22–0:34)   ← ZOOM HINGE (pull back)
- **B8** `0:22–0:26` — Camera pulls back; the 3 tiles fly to position (i,j) inside 3 assembling 7×7 grids. Traced cell keeps red/blue/green border.
  *Manim:* `camera.frame.animate.scale`; `ReplacementTransform` tile→cell; build 3× `VGroup` of 49 `Square`s.
- **B9** `0:26–0:30` — Remaining 48 cells of each MPI fade to their gray values ("every pair scored"). Diagonal masked. Titles `r`, `ρ`, `MI`, `\in\mathbb{R}^{7\times7}`.
  *Manim:* `LaggedStart(FadeIn)`.
- **B10** `0:30–0:34` — Symmetry hint (upper/lower fold), keep lower-tri. Hold with traced cell glowing.

### L3 — barcode → SPI–SPI scatter  (0:34–0:49)
- **B11** `0:34–0:38` — Each MPI's lower-tri off-diagonals lift and line up into a 1×21 gray barcode. Traced cell → its exact barcode slot (notebook `pos` map).
  *Manim:* `ReplacementTransform` cells → row.
- **B12** `0:38–0:42` — Two barcodes (r, ρ): their 21 entries become x,y of 21 dots in an `Axes`. **Point = a channel-pair.** Traced pair's dot stays highlighted.
  *Manim:* `ReplacementTransform` barcode squares → dots at `c2p`.
- **B13** `0:42–0:46` — Regression `Line`; label `f_{ij}:=\mathrm{corr}(\mathrm{SPI}_i,\mathrm{SPI}_j)`; value counts up (the meta-feature).
- **B14** `0:46–0:49` — Reindex cue ("each point = a pair; each axis = a whole SPI"). Optional: cycle the 3 planes (r-ρ, r-MI, ρ-MI). Hold on `f`.
  *Cue:* "this single f is one cell of a much larger feature matrix."

### L4 — feature matrix → embedding  (0:49–1:05)   ← ZOOM HINGE (pull back hard)
- **B15** `0:49–0:53` — `f` shrinks to a single cell; camera zooms out hard to reveal a huge matrix: **900 rows (MTS) × 40,184 cols (features)**.
  *Manim:* `camera.frame.animate.scale` (large); `ImageMobject` of downsampled `X`.
- **B16** `0:53–0:57` — Three columns glow — the (r,ρ),(r,MI),(ρ,MI) features. "The 3 we studied are 3 of 40,184."
- **B17** `0:57–1:01` — Each row collapses to a point; matrix dissolves into the UMAP embedding of 900 MTS, colored by the 10 classes.
  *Manim:* `Transform` rows → dots, or crossfade to `umap_benchmarked90_260603.svg`.
- **B18** `1:01–1:05` — Title card: glyph chain `MTS ⇒ MPI ⇒ f_{ij} ⇒ feature space ⇒ embedding`. Fade.

## Open decisions (before building)
1. **Traced pair (i,j)** — pick from M7 at build (heuristic above).
2. **Camera** — discrete + 2 zoom hinges (recommended) vs continuous single-canvas zoom.
3. **L4 ending** — stop at the feature matrix (B16), or run the UMAP capstone (B17–18)?
4. **Render** — length (~65–75 s budgeted), 1080p60 vs 4K, transparent bg for slides?
5. **Narration** — captions/VO, or purely visual?
6. **L1 three-metric handling** — sequential toggle (recommended) vs 3 side-by-side mini-axes.

## Precompute checklist (feeds the Scene from numpy — no pyspi at render time)
- M7: z-scored timeseries → icefire RGB; 3 MPIs (from npz); barcode vectors + `pos` map; SPI–SPI scatter arrays + `r`; per-pair r/ρ/MI.
- M10: `X` downsampled to a renderable image; exact column indices of the 3 features; UMAP coords + class colors.
- Helper: seaborn cmap (`icefire`/`gray`) → `ManimColor` per cell.
- Toolchain: confirm `manim`, LaTeX + `dvisvgm` on PATH (matplotlib usetex already present).
