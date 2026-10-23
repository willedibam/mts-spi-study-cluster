# Method schematic (paper Fig. 1)

How z is constructed, with its geometric reading: one recording, viewed through K statistics, each giving its own weighted graph. The layout is a snake, with row 1 running left to right (a → b) and row 2 right to left (c → d → e). Draft of 2026-10-22. There is no case-study data and no property panels: the invariances belong in the methods.

- `method_schematic.py` computes every drawn value and writes `method_schematic_261022.{svg,pdf,png,json}` beside itself, plus one SVG per panel in `panels/`. It is typeset by LaTeX, renders everything at 600 dpi, and runs no pyspi.
  - `method_schematic.ipynb` builds the figure once, then gives one cell per output: the full figure, or a single panel saved to `panels/panel_<x>.svg` and displayed. Run only the cells you need.
  - Each panel is drawn in figure inches on its own three layers (behind, panel, front). Cross-panel lines sit on separate link layers, so a panel export hides the other panels and all links, then crops to `PANEL_BOXES`.
- **Recording:** a directed VAR(1) with heterogeneous lagged coupling and correlated innovations, M = 6, T = 1,000, seed 0.
  - Channels are relabelled (`CHANNEL_ORDER`) so that the illustrated, strongly asymmetric pair is (i, j) = (2, 6). z is unchanged by the relabelling, and so are the drawn values.
  - The carpet and traces show samples 200–300, with the order-parameter composite's heatmap glyph: robust scaling per channel, icefire, limit at the 99th percentile of |x|.
- **Panel a carpet:** drawn on one hairline card with a vertical "channel" label and no row numbers.
  - x_i (channel 2) is traced above the carpet and x_j (channel 6) below it, each with a dashed leader to its row.
  - The graph repeats both traces at their nodes, at the same size (`TRACE_W`, `TRACE_H`) and with the same labels.
  - A ⟺ joins carpet and graph: the same system, seen two ways.
- **Panel a graph:** the generator's coupling per ordered pair s→t, |lagged coupling of t on s| plus |innovation correlation|, scaled to [0, 1].
  - Each direction is a tapered arc, wide at the source; the two arcs of a pair bow apart.
  - The node layout (`LAYOUT`) is irregular, and i and j sit on top under their traces.
- **Panel b:**
  - The graph is repeated as uniform, equal-size layers in the order A⁽ᵃ⁾, A⁽ᵇ⁾, ⋮, A⁽ᴷ⁾.
  - Each layer's edges are its statistic's min–max scaled strength raised to the power 1.6, so weak edges recede. Euclidean distance is flipped to a similarity for drawing only.
  - The directed statistics (a, b) are drawn with one tapered arc per direction and the symmetric ones with one line per pair, so the layers show which statistics are directed.
  - The matrices stack down-right: A⁽ᵇ⁾ is offset by 0.45 of a matrix, so A⁽ᵃ⁾ covers about 30% of it, and the rest of the K are grey cards behind.
  - Both matrices outline the (i, j) and (j, i) entries. A⁽ᵃ⁾'s glyphs attach from the right (i→j, last column) and from below (j→i, last row), so both leaders are short.
  - The glyphs (`b_top`) and panel c's z_ab label (`c_top`) sit on a layer above the cross-panel lines.
- **Row 2:** panels e, d and c each take one third of the width, with titles only. Hand-off arrows between them are left to add by hand.
- **Panel e (statistical learning):** three schematic icons fed by z, in the order the paper uses them:
  - comparing recordings of different M in one embedding (proof, atlas);
  - tracking a transition with PC1 of z against an order parameter (order-parameter section);
  - reading one named coordinate z_ab that separates two conditions (cases, proof features).

  They are drawn from seeded noise, not data; replace them with thumbnails once those results are frozen. Classification is left out, because supervised use is outside the first paper's scope.

  Exploratory fourth icon (`SHOW_SLIDING_WINDOW`, on for now): z over sliding windows of one recording, reorganising at a change point. It is not demonstrated in this paper. Its requirements are a fixed window length, surrogate nulls for window-to-window noise, and a reduced statistic set for cost.
- **Statistics (K = 5):**
  - The order is Pearson, Spearman, a, b, Euclidean distance (`A_IDX`, `B_IDX` = 2, 3), so that z_ab's kept cell (b, a) of R sits at row 4, column 3, next to panel c. The order is arbitrary.
  - a: lag-1 cross-correlation, corr(x_i(t), x_j(t+1)), directed. The figure does not name it; the caption should.
  - b: order-2 linear Granger log-variance ratio, directed.
  - M = 6 is kept deliberately unequal to K, so the two kinds of matrix cannot be mistaken for one another.
- **Contract check:** z is asserted equal to `build_unified_feature_values` (`unified_ordered_v3`), which uses all M(M−1) ordered entries in row-major order.
- **Extraction lines:**
  - Only the highlighted pair's four lines (from A⁽ᵃ⁾ and A⁽ᵇ⁾, solid for i→j and dotted for j→i) run into the scatter. `SHOW_ALL_EXTRACTION = True` restores a faint line for every entry.
  - z_ab's cell of R projects into panel c as a faint ochre beam from its right corners to the scatter's left corners.
  - Panel c is boxed, with no ticks. The fitted line carries the correlation.
  - z_ab has its own accent, ochre (`FEATURE`): the fitted line, the outlined value, and the outlined cells in R and z.
  - Rugs on the bottom (sage) and left (lavender) edges show that each axis is one statistic's edge profile.
  - The outlined z_ab label matches the outlined cell in R.
  - In panel d, z and R are drawn in matrix brackets, with a colour bar labelled "feature value z_ab". A vertical label for z is left to add by hand. Each lower-triangle cell of R has a line to its entry of z, in the cell's colour. Because R is symmetric, cell (r, c) with r > c is z's coordinate for the pair (c, r), so the order of z matches the library's upper-triangle order.
- **Colour:**
  - a and b have one pastel hue each, starting from a light tint.
  - Diagonals are empty dashed outlines, so they cannot be confused with near-zero cells; the diverging map's centre is near-white.
  - The channel pair (i, j) is ink: solid for i→j, dotted for j→i.
  - R and z share a soft diverging map on [−1, 1].
  - Panel e's icons use one pastel family (`PASTELS`: sage, lavender, amber, rose, sky), so they belong with a and b without implying a or b.
- **For Illustrator:**
  - With usetex, SVG glyphs are paths. Retype labels you need to edit, or set `text.usetex` to False for a live-text variant.
  - A few tertiary labels are 7 pt, below the 8 pt floor in `docs/figure-style.md`.

```bash
.venv/bin/python docs/artwork/method_schematic/method_schematic.py
```

## Draft caption

**Construction of the SPI–SPI representation.**
- **(a)** A recording of M channels and T samples, drawn as a carpet whose rows are channels, and as a graph of the same channels, here with the simulated system's couplings as edges.
- **(b)** Each of K statistics of pairwise interaction scores every ordered channel pair. This gives one weighted graph on the same nodes, equivalently an M × M matrix A⁽ᵏ⁾. For a directed statistic, such as the lag-1 cross-correlation in the glyphs, i→j and j→i differ.
- **(c)** The M(M−1) off-diagonal entries of A⁽ᵃ⁾ and A⁽ᵇ⁾ are paired by ordered channel pair. z_ab is their Pearson correlation (fitted line). Filled and open markers are the two directions of one pair, and the rugs mark each statistic's edge profile.
- **(d)** z_ab is one entry of the K × K agreement matrix R (outlined; panel c is that cell opened out). R is symmetric with a unit diagonal, so one strict triangle (here the lower) is z, with K(K−1)/2 entries for any M and T. K = 5 is drawn; the 289 statistics used here give 41,616.
- **(e)** One fixed-length z per recording supports comparison across recordings of different size, unsupervised tracking of a change in dynamics, and reading named coordinates (schematic).

`[ALIGN: validity policy — z_ab is undefined for constant or non-finite profiles.]`

## Open

- Channel indices: the schematic uses i, j, while the case figures use j, k.
- "∀ M, T" holds for the length of z. At M = 2 there are only two ordered entries, so every z_ab is ±1; values are informative from M ≥ 3, and only for non-constant profiles.
- Panel a's graph shows the generator's couplings, not an estimate. The alternative is |Pearson|, but that is one of the K statistics.
