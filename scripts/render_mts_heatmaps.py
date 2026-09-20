"""
Re-render `mts_heatmap.png` as vector-container SVG for publication figures.

Walks the depth-2 layout `<root>/<class>/<M*_T*_I*>/timeseries.npy` and writes
`mts_heatmap.svg` alongside each existing PNG.

Differences from the PNGs written by `run_experiments.save_mts_heatmap`:
  - Channels are normalised before the +/-2 colour clip (the PNGs clip raw
    values, so low-variance classes render washed out; see --norm).
  - The heatmap body is rasterised inside the SVG at >=1 pixel per sample.
    A fully vectorised pcolormesh emits one path per cell (~10 MB at M32xT2000)
    which no figure editor handles comfortably.
  - Time axis can be cropped to a window (--t-max / --t-range) so panels with
    different T stay legible at a common figure width.

Usage:
    python -m scripts.render_mts_heatmaps data/embeddings/multi_p90_260701
    python -m scripts.render_mts_heatmaps <root> --t-max 500 --size row
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import robust_scale

VMIN, VMAX = -2.0, 2.0
CMAP = "icefire"
MIN_DPI = 300
MAX_DPI = 1200
# Heavy-tailed classes where the sample std is outlier-dominated and z-scoring
# collapses the bulk of the data into the middle of the colour range.
HEAVY_TAILED = ("cauchy",)


# ----------------------------
# Normalisation
# ----------------------------
def normalise(data: np.ndarray, mode: str, label: str) -> np.ndarray:
    """Scale each channel of an (M, T) array so the +/-2 clip is meaningful."""
    if mode == "auto":
        mode = "robust" if any(k in label.lower() for k in HEAVY_TAILED) else "zscore"
    if mode == "none":
        return data
    if mode == "zscore":
        scaled = zscore(data, axis=1, nan_policy="omit")
    elif mode == "robust":
        scaled = robust_scale(data, axis=1)
    else:
        raise ValueError(f"Unknown norm mode: {mode}")
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


# ----------------------------
# Figure geometry
# ----------------------------
def figure_size(M: int, T: int, args: argparse.Namespace) -> tuple[float, float]:
    """
    Return (width, height) in inches.

    row   : fixed width, height proportional to M. Channel rows keep a constant
            physical thickness across panels, so M reads as panel height and all
            panels align at a common column width.
    cell  : constant inches per sample. Panel area scales with M*T, preserving
            the texture scale but producing very unequal panel sizes.
    fixed : identical canvas for every panel; cells stretch to fill it.
    """
    if args.size == "row":
        return args.width, float(np.clip(M * args.row_height, 0.4, 12.0))
    if args.size == "cell":
        return (
            float(np.clip(T * args.cell_width, 1.0, 18.0)),
            float(np.clip(M * args.row_height, 0.4, 12.0)),
        )
    return args.width, args.height


def raster_dpi(M: int, T: int, width: float, height: float, oversample: float = 1.0) -> int:
    """Smallest DPI giving >=1 raster pixel per sample, so nearest-neighbour
    sampling of the heatmap drops no columns and introduces no moire.

    oversample multiplies that floor: 1.0 stays crisp to ~3x screen zoom, 4.0 to
    ~12x at ~16x the file size. Irrelevant for print, which never exceeds 1x.
    """
    needed = max(math.ceil(T / width), math.ceil(M / height))
    dpi = max(MIN_DPI, needed) * max(1.0, oversample)
    return int(np.clip(dpi, MIN_DPI, MAX_DPI * max(1.0, oversample)))


# ----------------------------
# Rendering
# ----------------------------
def render(ts_path: Path, out_path: Path, args: argparse.Namespace) -> tuple[int, int, int]:
    data = np.load(ts_path).astype(float, copy=False)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {data.shape} for {ts_path}")
    if data.shape[0] > data.shape[1]:  # stored (T, M) -> display (M, T)
        data = data.T

    # Normalise on the full series, then crop, so a cropped panel keeps the same
    # colour scale as the uncropped one.
    label = f"{ts_path.parent.parent.name}/{ts_path.parent.name}"
    scaled = normalise(data, args.norm, label)

    start, end = args.t_range if args.t_range else (0, scaled.shape[1])
    if args.t_max is not None:
        end = min(end, start + args.t_max)
    end = min(end, scaled.shape[1])
    if start >= end:
        raise ValueError(f"Empty time window [{start}, {end}) for {ts_path}")
    scaled = scaled[:, start:end]

    M, T = scaled.shape
    width, height = figure_size(M, T, args)
    dpi = raster_dpi(M, T, width, height, args.oversample)

    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax.pcolormesh(
        scaled,
        shading="flat",
        vmin=VMIN,
        vmax=VMAX,
        cmap=sns.color_palette(args.cmap, as_cmap=True),
        rasterized=not args.vector,
    )
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0)
    fig.savefig(
        out_path,
        format="svg",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )
    plt.close(fig)
    return M, T, dpi


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path, help="Directory containing <class>/<dataset>/ subdirectories.")
    p.add_argument("--norm", choices=["auto", "zscore", "robust", "none"], default="auto",
                   help="Per-channel scaling before the +/-2 clip. auto: robust for heavy-tailed "
                        "classes (cauchy), z-score otherwise. none reproduces the raw PNG scaling.")
    p.add_argument("--size", choices=["row", "cell", "fixed"], default="row",
                   help="Figure geometry; see figure_size().")
    p.add_argument("--width", type=float, default=6.0, help="Figure width (row/fixed modes).")
    p.add_argument("--height", type=float, default=2.0, help="Figure height (fixed mode).")
    p.add_argument("--row-height", type=float, default=0.08, help="Inches per channel (row/cell modes).")
    p.add_argument("--cell-width", type=float, default=0.006, help="Inches per timestep (cell mode).")
    p.add_argument("--t-max", type=int, default=None, help="Keep at most this many timesteps.")
    p.add_argument("--t-range", type=int, nargs=2, metavar=("START", "END"), default=None,
                   help="Keep timesteps [START, END).")
    p.add_argument("--oversample", type=float, default=1.0,
                   help="Multiply the raster DPI floor. 1.0 (default) is crisp to ~3x screen zoom "
                        "and to any print size; 4.0 is crisp to ~12x at ~16x the file size.")
    p.add_argument("--vector", action="store_true",
                   help="Emit a fully vectorised heatmap (one path per cell) instead of an embedded "
                        "raster: crisp at unlimited zoom, but ~166 bytes/cell (10.6 MB at M32xT2000) "
                        "and slow to render in figure editors. Intended for a handful of final panels.")
    p.add_argument("--cmap", default=CMAP)
    p.add_argument("--name", default="mts_heatmap", help="Output stem written next to timeseries.npy.")
    p.add_argument("--overwrite", action="store_true", help="Re-render SVGs that already exist.")
    p.add_argument("--delete-png", action="store_true",
                   help="Delete the source mts_heatmap.png after a successful render.")
    p.add_argument("--dry-run", action="store_true", help="List what would be written and exit.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.t_range and args.t_range[0] >= args.t_range[1]:
        print("[ERROR] --t-range requires START < END", file=sys.stderr)
        return 2

    root: Path = args.root
    if not root.is_dir():
        print(f"[ERROR] Not a directory: {root}", file=sys.stderr)
        return 2

    # Explicit per-depth globs, not rglob: Path.rglob does not descend into directory
    # symlinks, and two class directories here are symlinks into ../cml-embedding/.
    # Accepts a dataset dir, a class dir, or a whole collection.
    targets: list[Path] = []
    for pattern in ("timeseries.npy", "*/timeseries.npy", "*/*/timeseries.npy"):
        targets = sorted(root.glob(pattern))
        if targets:
            break
    if not targets:
        print(f"[ERROR] No timeseries.npy at depth 0-2 under {root}", file=sys.stderr)
        return 1

    if args.vector and len(targets) > 20 and not args.dry_run:
        # ~166 bytes per cell; the full tree is several GB and unusable in an editor.
        print(f"[ERROR] --vector on {len(targets)} datasets would emit GBs of unopenable SVG.\n"
              f"        Point it at a single dataset directory, or use --oversample 4 instead.",
              file=sys.stderr)
        return 2

    written = skipped = failed = 0
    for ts_path in targets:
        out_path = ts_path.with_name(f"{args.name}.svg")
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        rel = out_path.relative_to(root)
        if args.dry_run:
            print(f"[DRY]  {rel}")
            written += 1
            continue
        try:
            M, T, dpi = render(ts_path, out_path, args)
        except Exception as exc:  # keep going; report at the end
            print(f"[FAIL] {rel}: {exc}", file=sys.stderr)
            failed += 1
            continue
        written += 1
        print(f"[OK]   {rel}  M={M} T={T} dpi={dpi}")
        if args.delete_png:
            png = ts_path.with_name(f"{args.name}.png")
            if png.exists():
                png.unlink()

    print(f"\n{written} written, {skipped} skipped (exists), {failed} failed, {len(targets)} found.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
