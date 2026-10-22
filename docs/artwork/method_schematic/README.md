# Method schematic (paper Fig. 1)

How z is constructed, with its geometric reading: one recording, viewed through K statistics, each giving its own weighted graph. The layout is a snake, with row 1 running left to right (a → b) and row 2 right to left (c → d). Draft of 2026-10-22. There is no case-study data and no property panels: the invariances belong in the methods.

- `method_schematic.py` computes every drawn value and writes `method_schematic_261022.{svg,pdf,png,json}` beside itself in about 15 s, typeset by LaTeX, with no pyspi.
  - Everything is drawn in figure inches on full-figure layers: lines behind, panels, lines in front. Cross-panel connectors are therefore plain lines.
- **Recording:** a directed VAR(1) with heterogeneous lagged coupling and correlated innovations, M = 6, T = 1,000, seed 0.
  - Channels are relabelled (`CHANNEL_ORDER`) so that the illustrated pair (i, j) = (5, 6) sits at the matrix corner. z is unchanged by the relabelling, and so are the drawn values.
  - The carpet and traces show samples 200–300, with the order-parameter composite's heatmap glyph: robust scaling per channel, icefire, limit at the 99th percentile of |x|.
- **Panel a graph:** the generator's coupling per unordered pair, |lagged coupling| in both directions plus |innovation correlation|, scaled to [0, 1]. The node layout (`LAYOUT`) is irregular, and i and j sit on top under their traces.
- **Panel b:**
  - The graph fans out into K planes, each a parallelogram along a ray at 62°, 34°, 0°, −34° or −62°. The middle three spread widest and the outer two tuck behind.
  - Each plane's edges are its statistic's symmetrised, min–max scaled strength raised to the power 1.6, so weak edges recede. Euclidean distance is flipped to a similarity for drawing only.
  - Direction is not visible on the planes; it lives in the matrices.
  - The matrices step up-left behind A⁽ᵃ⁾, which carries the i→j and j→i glyphs.
- **Statistics (K = 5):**
  - a: lag-1 cross-correlation, corr(x_i(t), x_j(t+1)), directed. The figure does not name it; the caption should.
  - b: order-2 linear Granger log-variance ratio, directed.
  - The other three: Pearson, Spearman and Euclidean distance.
  - M = 6 is kept deliberately unequal to K, so the two kinds of matrix cannot be mistaken for one another.
- **Contract check:** z is asserted equal to `build_unified_feature_values` (`unified_ordered_v3`), which uses all M(M−1) ordered entries in row-major order.
- **Extraction lines:**
  - In panel c, faint lines run from every off-diagonal cell of A⁽ᵃ⁾ (sage) and A⁽ᵇ⁾ (lavender) to its point in the scatter. A⁽ᵇ⁾'s lines run behind the cards. The four lines of the highlighted pair are drawn strong: solid for i→j, dotted for j→i.
  - An L-shaped arrow carries z_ab from the scatter into its (a, b) cell of R.
  - In panel d, each upper-triangle cell of R has a line to its entry of z, in the cell's colour.
- **Colour:**
  - a and b have one pastel hue each, starting from a light tint so that only unused cells are white. All diagonals are white with a hairline.
  - The channel pair (i, j) is ink: solid for i→j, dotted for j→i.
  - R and z share a soft diverging map on [−1, 1].
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
- **(c)** The M(M−1) off-diagonal entries of A⁽ᵃ⁾ and A⁽ᵇ⁾ are paired by ordered channel pair. z_ab is their Pearson correlation; filled and open markers are the two directions of one pair.
- **(d)** z_ab is one entry of the K × K agreement matrix R. The unit diagonal and the mirror-image lower triangle are discarded, and the upper triangle is z, with K(K−1)/2 entries for any M and T; 289 statistics give 41,616.

`[ALIGN: validity policy — z_ab is undefined for constant or non-finite profiles.]`

## Open

- Channel indices: the schematic uses i, j, while the case figures use j, k.
- Panel a's graph shows the generator's couplings, not an estimate. The alternative is |Pearson|, but that is one of the K statistics.
