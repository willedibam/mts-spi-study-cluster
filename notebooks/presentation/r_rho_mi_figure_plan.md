# r–ρ–MI case-study figure — plan & handover (last updated 2026-06-25)

Pedagogical, full-page, multi-panel figure for the functional-form case study, targeting an
elite venue (Nature Comp Sci or similar). This is the **pedagogical** figure (mechanism), before
the large-scale applications. We are building the **data backbone in code (export SVG)**; cartoon
/ narrative polish (brackets, curved connectors, callouts, final type) is done externally in
Inkscape/Illustrator.

First-pass artifacts (this directory):
- `r_rho_mi_figure_v1.py` — standalone first-pass (β=π, A→C→D telescope, no MPI yet). Run from repo root.
- `r_rho_mi_figure_v1.png` — its render.

## Thesis the figure must land
Correlating two SPI matrices across an MTS — the meta-feature `f_ij := corr(SPI_i, SPI_j)` over the
MPI off-diagonals — reveals the **functional-form character** of the MTS's dependencies, via the
capture hierarchy **r ⊂ ρ ⊂ MI**. Read off pairwise SPI agreement: monotone-nonlinearity breaks r
from ρ; non-monotonicity breaks MI from {r,ρ}.

## Data (locked)
- Generator `generate_filter_roll_mts`; config `configs/generate/r_rho_mi/260622_g-roll.yaml`.
- **β = π locked** (`260624_g-roll`, M=32, T=2000, 30 instances). β=2π (`260623_g-roll`) breaks
  corr(r,ρ) deeper (iter2 0.685 vs 0.780) BUT over-saturates M→sign(z): M-M ρ falls 0.935→0.876 and
  M-NM ⟨MI⟩ 0.49→0.24, so part of the extra break is a saturation/ties confound and the MI planes/
  heatmaps split the non-monotone signal (M-NM block stays dark). β=π keeps M cleanly monotone and
  corr(r,ρ) still breaks unambiguously (0.78±0.11, non-overlapping with corr(ρ,MI)=0.91 at n=30).
- iter3 channel layout at M=32: **0–11 = L, 12–23 = M, 24–31 = NM**.
- Measured staggered (β=π, M32, n30): corr(r,ρ) 1.00→0.78→0.99 ; corr(ρ,MI) 0.91→0.91→0.62 ;
  corr(r,MI) 0.91→0.80→0.62.

## Foundational decision: colour semantics (highest-leverage clarity fix)
Two axes were competing for the same hues (SPI r/ρ/MI ≈ pair-type L-M/L-NM/M-NM ≈ blue/orange/green).
Give each its own visual channel:
- **SPIs → colour (bright triad only here):** r `#0072B2`, ρ `#E69F00`, MI `#009E73` (Wong).
- **Pair-type → glyph (line/S/U) + shape + neutrality:** same-type grey hollow; mixed-type distinct
  hues *not* in the SPI triad (`#9467bd` L-M, `#8c564b` L-NM, `#e377c2` M-NM) — or shapes if preferred.
- **Telescope exemplar → one accent** (`#d62728`) threaded identically across panels.
- **Iterations → position/labels only**, no colour.
(Open issue: raincloud's 3 series are *meta-features* (corr of SPI pairs), not SPIs — currently muted
blue/orange/green, still a bit close to the SPI triad; consider a 4th distinct family or label-only.)

## Glyphs = miniature phase-portraits (NOT abstract icons)
The glyph for a relationship is its phase-portrait shape: **line** (linear/concordant), **S-curve**
(monotone-nl), **U/parabola** (non-monotone). One tiny line-art axes (`ax.inset_axes`), plot `t`,
`tanh(2.5t)`, `t²`, frame off. Reuse the SAME glyph everywhere that relationship appears (phase
portraits, plane legend, bars) — the visual constant is a big readability win.

## Panels

### Panel A — dependency alphabet + capture hierarchy (top, full width)
- Left: one iter3 MTS heatmap, channels type-ordered into L|M|NM blocks (this doubles as the
  "iterations knob"/composition view — no separate construction panel needed).
- Brackets (cartoon, Illustrator) extract three pairs (L-L, L-M, L-NM) → three phase portraits
  (each with its line/S/U glyph) → three r/ρ/MI bar-trios (MI on **Linfoot** scale `√(1−e^{−2I})`).
- Teaching: bars go [all high] → [r drops] → [r,ρ≈0, MI high]. Annotate the r-drop with an arrow
  (β=π makes it gentle; ~0.89 vs 0.94 — call it out so it reads).

### Panel B — type-ordered MPI heatmaps (OPTIONAL / "with-MPI" variant)
- iter3 r / ρ / MI matrices, channels in L|M|NM blocks, diagonal masked, RdBu_r for r/ρ (shared
  colourbar) + sequential for MI. The killer frame: the **(L,NM) block is white in r but green in
  MI** — decoupling as a block lighting up only in MI. Most intuitive panel in the figure.
- Generate **two versions** (without-MPI = A→C→D; with-MPI = A→B→C→D, or B replaces C's (r,MI)).

### Panel C — SPI–SPI feature space across regimes (middle, double-width)
- 3 rows (iter1/2/3) × 2 cols ((r,ρ), (ρ,MI)); same-type grey, mixed by glyph/hue, each annotated
  `f_{ij}`; faint y=x on the (r,ρ) column. The layout *is* the staggered result: col (r,ρ)
  tight→spread→two-cluster; col (ρ,MI) tight→tight→detached. (Drop (r,MI) — it's the coarse union;
  completeness is preserved by Panel D.)

### Panel D — the signature (bottom, full width)
- Raincloud (3 meta-features × 3 iterations, n=30; half-violin + box + raw strip). Degenerate cells
  (iter1/iter3 corr(r,ρ)) collapse to dots + flat box by design.

## Telescoping link (algorithmic)
One exemplar thread (instance i*, one L-NM pair (a,b)). Connect with `matplotlib.patches.ConnectionPatch`
added at figure level, each hop in the two axes' own coords:
```python
from matplotlib.patches import ConnectionPatch
fig.add_artist(ConnectionPatch(xyA=(x_dot,y_dot), coordsA=axD.transData,        # raincloud dot
                               xyB=(0.5,1.0),     coordsB=axC_iter3_rhoMI.transAxes, ...))
fig.add_artist(ConnectionPatch(xyA=(rho_ab,MI_ab), coordsA=axC_iter3_rhoMI.transData, # pair dot
                               xyB=(0.5,-0.18),    coordsB=axA_LNM_bars.transAxes, ...))
```
Chain: one raincloud dot = one plane's f = one pair's dot = one bar-set = one phase portrait, all in
the accent colour, exemplar enlarged/outlined in each panel. Keep to ONE thread (maybe two: a
monotone thread for the iter2 r-ρ drop in a second accent — two max before spaghetti). Place anchors
in code; do the elegant curved routing in Illustrator.

## QoL / clarity refinements (high value)
- Shared scales: all planes same r/ρ/MI ranges; all bar-trios same y; r/ρ heatmaps share one colourbar.
- On-figure annotation of the two events ("r decouples", "MI alone survives") so it stands w/o caption.
- n and β stated in-panel.
- Consistent `f_{ij}` notation; single glyph mini-legend.
- Mask MPI diagonals; RdBu_r diverging for signed SPIs.

## External (Illustrator) vs code — the split
- **Code / SVG (data-bearing, must be reproducible):** phase portraits, planes, raincloud, heatmaps,
  bars, glyphs, f-annotations, telescope *anchor points*. Never redraw data in Illustrator (drift risk).
- **Illustrator (narrative decoration only):** bracket-extraction cartoon, curved telescope connectors
  (route around content), final typography/LaTeX + sizing, callout boxes/arrows with prose, panel
  framing/whitespace, colour final touch-ups. Export SVG with text editable and elements grouped per panel.

## Status & next steps
- `r_rho_mi_figure_v1.py` — A→C→D telescope, no MPI.
- `r_rho_mi_figure_v2.py` (current) — **square panels throughout**; A (MTS + 3 square phase/bars + glyphs
  + r<ρ callout); **B type-ordered MPI heatmaps**; **C 3×3 square planes** (filled the wasted width); D
  **compact raincloud**; one telescope thread A↔C↔D.

OPEN design questions:
- **Density/height:** v2 is tall and busy. Consider 2D packing (e.g. left column A+C, right column B+D)
  to cut height; or let **B replace C's redundant (r,MI) column** (B = the matrix completeness), C→3×2.
- **Monotone-nl is fundamentally subtle** (Pearson is robust to monotone transforms — not tunable). The
  stark evidence is the non-monotone (L-NM) contrast + B's (L,NM)/M-block. Frame as "first crack (subtle)
  → full break (stark)"; do NOT oversell the monotone effect.
- Telescope connectors span the full height — curve/route them in Illustrator (one thread only).
- Differentiate D's meta-feature colours from the SPI triad (still close).
