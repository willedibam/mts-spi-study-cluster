# r–ρ–MI case-study figure — plan & handover (last updated 2026-06-25)

Pedagogical, full-page, multi-panel figure for the functional-form case study, targeting an
elite venue (Nature Comp Sci or similar). This is the **pedagogical** figure (mechanism), before
the large-scale applications. We are building the **data backbone in code (export SVG)**; cartoon
/ narrative polish (brackets, curved connectors, callouts, final type) is done externally in
Inkscape/Illustrator.

Artifacts (this directory) — run from repo root, render to `/tmp/figure_vN.png`:
- `r_rho_mi_figure_v1.py` — first-pass (β=π, A→C→D telescope, no MPI).
- `r_rho_mi_figure_v2.py` — square panels + B (MPI) + C 3×3. Superseded.
- `r_rho_mi_figure_v3.py` — C 3×2, B shared-scale, event labels (see DECISIONS below).
- `r_rho_mi_figure_v4.py` — **current**: A becomes a CONSTRUCTION + alphabet panel (double height);
  C gains a legend/capture-hierarchy key in its freed third column; B spacers trimmed; D widened.

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
- `r_rho_mi_figure_v2.py` — square panels throughout; A; B (RdBu_r + viridis); **C 3×3** (re-added the
  redundant (r,MI) column to fill width); D; telescope. Superseded — keep for reference only.
- `r_rho_mi_figure_v3.py` (current) — implements the decisions below. Vertical A→B→C→D spine.

DECISIONS (2026-06-25, resolving the v2 open questions):
- **Density: C back to 3×2** — drop the (r,MI) column. It is the coarse OR of the two events (corr(r,MI)
  just inherits corr(r,ρ)'s iter2 drop and corr(ρ,MI)'s iter3 drop: 0.91→0.80→0.62), so it localises
  neither. Completeness lives in B (3 matrices) + D (3 raincloud series). **Rejected the A+C|B+D 2D-pack:**
  it breaks the linear reading order that *is* the pedagogy and splits the telescope spine. If 3×2 looks
  width-starved under full-width A/B/D, fill with meaning (bigger square planes / glyph legend), not a
  duplicate column.
- **B on ONE shared sequential magnitude scale** (r, ρ, MI→Linfoot, 0→1, `magma`), not RdBu_r+viridis.
  v2 forced a cross-colormap judgment ("white-ish diverging" vs "yellow sequential"); v3 makes it a
  same-scale read: the (L,NM)/(M,NM) blocks are **black in r,ρ, lit in MI**, outlined in the accent in all
  three panels (+ "≈0"/"high" labels). MI uses Linfoot to match Panel A's bars (v2 inconsistently used raw
  MI in B). **Caveat:** sign is suppressed — fine here (r,ρ are ~0 or strongly +; no real anticorrelation),
  revisit if a dataset has signed structure. Honest residual: M-NM in MI is dimmer (Linfoot ~0.75) than
  L-NM (~0.90) — still clearly lit vs black; do not hide it.
- **C carries on-figure event labels** ("ρ splits from r" @ iter2 (r,ρ); "MI splits from ρ" @ iter3 (ρ,MI);
  "r,ρ co-fail → recorrelate" gloss on the iter3 (r,ρ) two-cluster) so it reads without a caption.

- **Panel A = construction + alphabet (v4).** Shows the generative model so the SPIs have a referent:
  shared AR(1) mother z → per-channel filter glyphs (noisy phase-portraits: glyph *shape* = filter family
  line/S/U, scatter *thickness* = SNR — one mark, both axes) → real type-ordered MTS heatmap → 4
  bracket-picked real phase plots (L-L, L-M, L-NM, M-NM) + r/ρ/MI bars. Mother + filter glyphs are an
  illustrative regeneration (can't recover the saved instance's exact mother); phase/bars are REAL pyspi.
  Fan-in lines + channel brackets are Illustrator (anchors only in code). This makes A the dominant panel.
- C's freed third column (3×2 planes leave width) now holds a legend: SPI colour key + "r ⊂ ρ ⊂ MI",
  glyph/filter-family key, pair-type key. Useful content, not filler.
- matplotlib gotcha (cost an hour): a legend axis with `axis("off")` then `scatter/add_patch(...,
  transform=ax.transAxes)` CORRUPTS the axis dataLim (autolim treats the transAxes points as data → lim
  collapses to ~(-0.06,0.06)); text placed in *data* coords then flies off-canvas and `bbox_inches="tight"`
  balloons the figure. Fix: `set_xlim/ylim(0,1); set_autoscale_on(False)` and draw ALL legend artists in
  `transAxes`.

STILL OPEN:
- **Monotone-nl is fundamentally subtle** (Pearson is robust to monotone transforms — not tunable, and
  β=2π only "fixes" it via a saturation/ties confound; β=π locked). A's L-M r<ρ gap is ~0.05 — keep the
  arrow, do NOT zoom the bar axis. The stark evidence is the non-monotone (L-NM) contrast + B's blocks.
  Frame as "first crack (subtle) → full break (stark)".
- Telescope connectors span the full height and the C→A hop crosses B — anchors are in code; curve/route
  in Illustrator (one thread only).
- Differentiate D's meta-feature colours from the SPI triad (still close).
- Caption text in v3 is placeholder (final type/LaTeX in Illustrator); panel-letter/caption anchored to
  each band's top axis via `get_position()` so it survives `bbox_inches="tight"`.
