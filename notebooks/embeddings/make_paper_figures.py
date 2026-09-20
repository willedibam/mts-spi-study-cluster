"""
Paper-ready SVG panels from proof_p90_260712.ipynb.

Emits into notebooks/embeddings/figures/:
  panel_a_umap_multi.svg          Panel A embedding: no title, no legend, no ticks
  panel_b_umap_cml.svg            Panel B embedding: ditto
  legend_class_<rows>x<cols>.svg  MTS class colour key: 12x1, 6x2, 4x3, 2x6
  legend_M.svg                    marker-size key ("Process Size" = M)
  legend_stacked.svg              single-column class key with the size key beneath
  feature_<n><a|b>_<A>_vs_<B>.svg two non-degenerate candidates per class pair (10 total)

Marker area encodes M identically in every figure (see MARKER_SIZES), so one size key
is valid across the embeddings and the feature panels.

Helper definitions (FeatureSet, class_colors, plot_umap, plot_rainclouds,
pair_separation) are executed straight out of the notebook rather than copied,
so this cannot drift from the source of truth.

COLOUR LOCK: the notebook calls class_colors() once per panel, so POOL restarts
at index 0 each time and two colours collide across panels (cauchy-noise/fdstc,
gaussian-noise/frozen-chaos). A single legend is only correct against a single
colour map, so one is built here over the union of both panels' classes. Panel A
non-bridge colours therefore differ from the notebook's inline output; the two
bridge classes are pinned and unchanged.

Run from anywhere; the script chdirs to its own directory:
    python notebooks/embeddings/make_paper_figures.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import textwrap
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "proof_p90_260712.ipynb"
OUTDIR = HERE / "figures"

# Notebook cells holding definitions only, addressed by cell id rather than index:
# inserting or reordering a cell silently repoints an index, and the failure is a
# wrong-but-runnable namespace rather than an error.
DEF_CELLS = [
    "ecd6391e",   # imports
    "8ca73196",   # colour lock: BRIDGE/POOL/class_colors
    "0bfb9564",   # FeatureSet + load MULTI/CML
    "9af9dd08",   # SUBSETS/SUBSET/CLASS_UNIVERSE (must precede any class_colors call)
    "2968f19e",   # _scatter / plot_pca / plot_umap
    "1b81611a",   # plot_rainclouds
    "e3952538",   # pair_separation / train_test_separation
]

# No LaTeX toolchain is installed on this machine (no latex/pdflatex/dvipng), so
# text.usetex would fail at draw time. Text is set in STIX via mathtext instead --
# see apply_latex_style(). Flip to True only after installing a TeX distribution
# (e.g. `brew install --cask basictex`, then `sudo tlmgr install dvipng type1cm`).
USETEX = False

# "path" converts glyphs to outlines: identical rendering everywhere, no font
# dependency in draw.io/Illustrator, but text is no longer selectable. Use "none"
# to keep live text (then STIX must be present on the viewing system).
SVG_FONTTYPE = "path"

# DPI is a no-op for these files: they are pure vector (no rasterised artists), and
# matplotlib lays SVG out in points from figsize*72 regardless of dpi. It is set
# anyway so a PNG export of the same figures is high-resolution. Legibility here is
# controlled by figure size and the font sizes below, not by dpi.
DPI = 600

# Panel B's y-label. Kept: the two panels are independent UMAP fits, so dropping it
# would imply a shared axis that does not exist.
PANEL_B_YLABEL = True

# Marker area (pt^2) at the smallest and largest M, shared by the embeddings, the feature
# rainclouds and legend_M. Previously the embeddings used (25, 150) via seaborn while the
# rainclouds used (5, 50), making the same M render 3-5x larger in one figure than the
# other -- and since the two mappings have different intercepts the discrepancy was not
# even a constant factor, so no single size key could describe both.
MARKER_SIZES = (25.0, 150.0)

LEGEND_TITLE_CLASS = "MTS Class"
LEGEND_TITLE_SIZE = "Process Size"

# Feature y-limits. f_ij is a correlation, so [-1, 1] is its true bound and a fixed
# scale makes the panels honestly comparable. This is only readable because the
# non-degeneracy gate below selects wide-spread features; against the raw top-ranked
# features (which span as little as 8% of [-1, 1]) it would flatten them to a line.
# Set to None for per-panel autoscale.
YLIM = (-1.0, 1.0)
YTICK_STEP = 0.5



# Non-degeneracy gate. A feature is rejected when either class's IQR is below this
# fraction of the pair's combined range, i.e. that class collapses to a point. The
# best-separating features almost always saturate one class, so this deliberately
# trades separation for a readable distribution.
REL_SPREAD_MIN = 0.08
SEP_FLOOR = 0.98          # near-perfect rank separation, not merely "discriminating"
N_CANDIDATES = 2          # panels emitted per class pair

PAIRS = [
    ("cml", "sti-i", "defect-turbulence"),
    ("cml", "fdstc", "frozen-chaos"),
    ("multi", "gaussian-noise", "cauchy-noise"),
    ("multi", "kuramoto_omega-fast", "kuramoto_omega-slow"),
    ("multi", "var-phi-0.2_cpl-0.4", "var-phi-0.95_cpl-0.4"),
]


def apply_latex_style() -> None:
    # NOT cmr10: it carries TeX's OT1 encoding, where U+005F is the dot accent, so
    # class names render as "kuramoto`omega-fast". STIX is Unicode-encoded, Times-like
    # (the revtex/AIP look), and ships with matplotlib, so text and math stay coherent.
    plt.rcParams.update({
        "text.usetex": USETEX,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "svg.fonttype": SVG_FONTTYPE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.labelsize": 15,
        "axes.titlesize": 12,
        "xtick.labelsize": 13,
        "ytick.labelsize": 12,
        "legend.fontsize": 14,
        "savefig.transparent": True,
    })


def wrap_label(s: str, maxlen: int = 14) -> str:
    """Break a long class name over two lines at the separator nearest its middle.

    Full-width names like kuramoto_omega-fast collide with their neighbour on a
    square axes; splitting at an existing '_' or '-' keeps every character intact.
    """
    if len(s) <= maxlen:
        return s
    cuts = [i for i, ch in enumerate(s) if ch in "_-"]
    if not cuts:
        return s
    i = min(cuts, key=lambda i: abs(i - len(s) / 2))
    return f"{s[:i]}\n{s[i + 1:]}" if s[i] == "-" else f"{s[:i + 1]}\n{s[i + 1:]}"


def tex(s: str) -> str:
    """Escape class names for the text layer. Underscores and hyphens are literal
    in mathtext but `_` is a subscript operator under usetex, so guard both paths."""
    return re.sub(r"([_&%$#{}])", r"\\\1", s) if USETEX else s


def load_notebook_defs() -> dict:
    """Exec the notebook's definition cells and return their namespace.

    Runs inside a real module registered in sys.modules: @dataclass resolves
    sys.modules[cls.__module__] to check for KW_ONLY, which fails on a bare dict.
    """
    os.chdir(HERE)   # the data-loading cell resolves ../../features/ against cwd
    mod = types.ModuleType("_nb_defs")
    sys.modules[mod.__name__] = mod
    cells = {c.get("id"): c for c in json.loads(NOTEBOOK.read_text())["cells"]}
    missing = [cid for cid in DEF_CELLS if cid not in cells]
    if missing:
        raise KeyError(f"notebook cell id(s) not found: {missing}; the cell was deleted or "
                       f"the notebook was re-saved without stable ids.")
    for cid in DEF_CELLS:
        exec(compile("".join(cells[cid]["source"]), f"<cell {cid}>", "exec"), mod.__dict__)
    return mod.__dict__


def save(fig, name: str) -> Path:
    out = OUTDIR / name
    fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    print(f"[OK]   figures/{name}")
    return out


# ----------------------------
# Panels
# ----------------------------
def embedding_panel(ns, fs, colors, *, name: str, ylabel: bool = True, **umap_kw):
    """UMAP panel, stripped to the scatter: no title, no legend, no ticks.

    UMAP coordinates carry no units and no cross-embedding meaning, so the ticks
    are noise; the axis names are kept only to orient the reader.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    _, _, ax = ns["plot_umap"](fs, ax=ax, colors=colors, sizes=MARKER_SIZES, **umap_kw)
    ax.set_title("")
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)
    ax.set_xlabel(r"UMAP-1")
    ax.set_ylabel(r"UMAP-2" if ylabel else "")
    return save(fig, name)


TITLE_FP = {"weight": "bold", "size": 15}


def _class_handles(colors):
    return [
        Line2D([], [], marker="o", linestyle="none", markersize=11,
               markerfacecolor=c, markeredgecolor=c, label=tex(k))
        for k, c in colors.items()
    ]


def _size_handles(m_values):
    """Marker areas identical to those the panels draw, so the key is literal."""
    lo, hi = float(min(m_values)), float(max(m_values))
    span = MARKER_SIZES[1] - MARKER_SIZES[0]
    return [
        Line2D([], [], marker="o", linestyle="none",
               markersize=np.sqrt(MARKER_SIZES[0] + (v - lo) / (hi - lo + 1e-9) * span),
               markerfacecolor="0.45", markeredgecolor="0.45", label=f"${int(v)}$")
        for v in sorted(set(m_values))
    ]


def legend_class(colors, *, ncol, name=None):
    """MTS class colour key in `ncol` columns. Filename records rows x cols."""
    nrow = int(np.ceil(len(colors) / ncol))
    fig, ax = plt.subplots(figsize=(3.1 * ncol, 0.42 * nrow + 0.75))
    ax.axis("off")
    ax.legend(handles=_class_handles(colors), title=LEGEND_TITLE_CLASS, loc="center",
              ncol=ncol, frameon=False, handletextpad=0.5, columnspacing=1.4,
              labelspacing=0.8, title_fontproperties=TITLE_FP)
    return save(fig, name or f"legend_class_{nrow}x{ncol}.svg")


def legend_size(m_values, *, name="legend_M.svg"):
    fig, ax = plt.subplots(figsize=(2.1, 0.42 * len(set(m_values)) + 0.7))
    ax.axis("off")
    ax.legend(handles=_size_handles(m_values), title=LEGEND_TITLE_SIZE, loc="center",
              frameon=False, handletextpad=0.8, labelspacing=0.75,
              title_fontproperties=TITLE_FP)
    return save(fig, name)


def legend_stacked(colors, m_values, *, name="legend_stacked.svg"):
    """Single-column class key with the size key stacked beneath it."""
    n = len(colors)
    fig, (ax_c, ax_m) = plt.subplots(
        2, 1, figsize=(3.1, 0.42 * n + 0.42 * len(set(m_values)) + 1.3),
        gridspec_kw={"height_ratios": [n, len(set(m_values)) + 0.8], "hspace": 0.0},
    )
    for a in (ax_c, ax_m):
        a.axis("off")
    ax_c.legend(handles=_class_handles(colors), title=LEGEND_TITLE_CLASS, loc="upper left",
                bbox_to_anchor=(0, 1), frameon=False, handletextpad=0.5, labelspacing=0.8,
                title_fontproperties=TITLE_FP)
    ax_m.legend(handles=_size_handles(m_values), title=LEGEND_TITLE_SIZE, loc="upper left",
                bbox_to_anchor=(0, 1), frameon=False, handletextpad=0.8, labelspacing=0.75,
                title_fontproperties=TITLE_FP)
    return save(fig, name)


# ----------------------------
# Feature choice
# ----------------------------
def spread_and_gap(Xa: np.ndarray, Xb: np.ndarray):
    """Per feature: (narrower class's IQR, gap between the two IQR boxes), both as
    a fraction of the pair's combined range.

    spread near zero means one class collapses onto a point -- a flat smear with no
    visible distribution. gap is what actually reads as separation at a glance:
    AUC can be 1.0 while the two clouds sit visually adjacent, because it only
    measures rank overlap, not distance. Ranking on gap subject to a spread floor
    targets "clearly apart AND each visibly spread".
    """
    qa = np.percentile(Xa, [25, 75], axis=0)
    qb = np.percentile(Xb, [25, 75], axis=0)
    iqr = np.minimum(qa[1] - qa[0], qb[1] - qb[0])
    gap = np.maximum(qb[0] - qa[1], qa[0] - qb[1])   # >0 only when the boxes are disjoint
    lo = np.minimum(Xa.min(0), Xb.min(0))
    hi = np.maximum(Xa.max(0), Xb.max(0))
    rng = hi - lo
    norm = lambda v: np.divide(v, rng, out=np.zeros_like(v), where=rng > 0)
    return norm(iqr), norm(gap)


def pick_features(ns, fs, a, b, *, n=N_CANDIDATES):
    """Top-n discriminating f_ij that also keep both classes visibly spread.

    Ranks survivors by spread rather than by separation: past SEP_FLOOR the extra
    separation is invisible in the figure, whereas the spread is the whole point.
    """
    # n_perm=1: only the ranking is used here, the permutation null is not reported.
    ranked, _ = ns["pair_separation"](fs, a, b, by_cell=True, n_perm=1)
    Xa, Xb = fs.X[fs.y == a], fs.X[fs.y == b]
    spread, gap = spread_and_gap(Xa, Xb)

    sep_col = "min_cell_sep" if "min_cell_sep" in ranked.columns else "sep"
    idx = ranked["idx"].to_numpy()
    df = ranked.assign(rel_spread=spread[idx], gap=gap[idx])
    ok = df[(df[sep_col] >= SEP_FLOOR) & (df["rel_spread"] >= REL_SPREAD_MIN)]
    if len(ok) < n:   # nothing clears the gate: fall back, but say so
        print(f"       [warn] {a} vs {b}: only {len(ok)} feature(s) clear "
              f"sep>={SEP_FLOOR} & rel_spread>={REL_SPREAD_MIN}; relaxing the spread floor.")
        ok = df[df[sep_col] >= SEP_FLOOR].nlargest(max(200, n), "rel_spread")

    # The top of this ranking is dominated by near-duplicate pairs, so require each
    # candidate to be built from two entirely unseen SPIs. Stricter than the
    # notebook's grid rule (which only skips when BOTH SPIs repeat) because these
    # panels exist to be compared by eye -- sharing one SPI yields near-identical
    # plots, which defeats the point of offering a choice.
    picks, used = [], set()
    for _, r in ok.sort_values("gap", ascending=False).iterrows():
        if r["spi_i"] in used or r["spi_j"] in used:
            continue
        picks.append(r)
        used.update([r["spi_i"], r["spi_j"]])
        if len(picks) >= n:
            break
    return pd.DataFrame(picks), sep_col


def feature_panel(ns, fs, a, b, colors, row, sep_col, *, tag: str):
    i = int(row["idx"])
    m = np.isin(fs.y, [a, b])
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ns["plot_rainclouds"](
        ax, fs.y[m], fs.X[m, i], order=[a, b], colors=[colors[a], colors[b]],
        size_by=fs.M[m], size_range=MARKER_SIZES, point_size=10, point_alpha=0.12,
        half_width=0.20, box_width=0.09, jitter=0.03, add_size_legend=False,
    )
    ax.set_xticklabels([wrap_label(tex(a)), wrap_label(tex(b))], fontsize=13)
    ax.set_ylabel("")                       # added by hand in the composite figure
    if YLIM is not None:
        ax.set_ylim(*YLIM)
        ax.yaxis.set_major_locator(MultipleLocator(YTICK_STEP))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    pair = f"corr({tex(row['spi_i'])}, {tex(row['spi_j'])})"
    ax.set_title("\n".join(textwrap.wrap(pair, 38)), fontsize=9)
    ax.set_box_aspect(1)
    print(f"       {tag}: {row['spi_i']} x {row['spi_j']}  "
          f"sep={row[sep_col]:.3f} spread={row['rel_spread']:.3f} gap={row['gap']:.3f}")
    return save(fig, f"feature_{tag}.svg".replace("/", "-"))


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    ns = load_notebook_defs()
    apply_latex_style()   # after the notebook cells, which force text.usetex=False

    MULTI, CML = ns["MULTI"], ns["CML"]
    CMLsub = CML.subset(classes=["fdstc", "frozen-chaos", "defect-turbulence", "sti-i"])

    # One colour map for every figure, so the legends are valid everywhere.
    union = sorted(set(MULTI.y) | set(CMLsub.y))
    colors = ns["class_colors"](union)
    print(f"[INFO] shared colour map over {len(union)} classes")

    shared = dict(metric="euclidean", scaled=False, pca_preprocess=True)
    embedding_panel(ns, MULTI, colors, name="panel_a_umap_multi.svg",
                    n_neighbors=11, min_dist=0.99, spread=5, seed=3,
                    alpha=0.1, edge_alpha=0.3, **shared)
    embedding_panel(ns, CMLsub, colors, name="panel_b_umap_cml.svg",
                    ylabel=PANEL_B_YLABEL,
                    n_neighbors=12, min_dist=0.75, spread=5.0, seed=0,
                    alpha=0.15, edge_alpha=0.4, **shared)

    m_values = np.unique(np.r_[MULTI.M, CMLsub.M])
    # ncol 1/2/3/6 -> 12x1, 6x2, 4x3, 2x6 rows x cols, so both readings of "6x2" are covered.
    for ncol in (1, 2, 3, 6):
        legend_class(colors, ncol=ncol)
    legend_size(m_values)
    legend_stacked(colors, m_values)

    src = {"multi": MULTI, "cml": CML}
    n_feat = 0
    for k, (which, a, b) in enumerate(PAIRS, start=1):
        fs = src[which]
        picks, sep_col = pick_features(ns, fs, a, b)
        for j, (_, row) in enumerate(picks.iterrows()):
            feature_panel(ns, fs, a, b, colors, row, sep_col,
                          tag=f"{k}{'ab'[j]}_{a}_vs_{b}")
            n_feat += 1

    print(f"\n{4 + n_feat} SVGs in {OUTDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
