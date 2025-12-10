"""
Compute per-feature two-sample KS statistics between mts_classes.

Inputs:
  - features npz from test_features.py or process_features.py (contains X and labels).
  - list of mts_classes to compare (pairwise).
Output:
  - CSV per class-pair under analysis/non-overlapping_distributions/ks_<cls1>_<cls2>.csv

Usage:
  python find_non_overlapping_distributions.py \
    --features analysis/features_data-full_spi-spi.npz \
    --mts-classes CauchyNoise,GaussianNoise,VAR_1
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KS stats per feature between mts_classes.")
    parser.add_argument(
        "--features",
        required=True,
        help="Path to features npz (from test_features.py/process_features.py).",
    )
    parser.add_argument(
        "--mts-classes",
        required=True,
        help="Comma-separated list of mts_class names to compare pairwise.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/non-overlapping_distributions",
        help="Where to write ks_<cls1>_<cls2>.csv files.",
    )
    parser.add_argument(
        "--alternative",
        default="two-sided",
        choices=["two-sided", "less", "greater"],
        help="Alternative hypothesis for ks_2samp.",
    )
    parser.add_argument(
        "--kde",
        action="store_true",
        help="Use KDE-smoothed distributions before computing KS (reduces artifacts from gaps/ties).",
    )
    parser.add_argument(
        "--kde-bandwidth",
        type=str,
        default="0.05",
        help="Comma-separated bandwidth multipliers for Gaussian KDE smoothing when --kde is set (e.g., 0.1,0.2,0.5).",
    )
    return parser.parse_args()


def load_features_npz(path: Path) -> tuple[np.ndarray, np.ndarray, List[str]]:
    data = np.load(path, allow_pickle=True)
    if "X" not in data:
        raise KeyError(f"{path} missing 'X'")
    X = data["X"]
    if "mts_class" in data:
        labels = data["mts_class"]
    elif "y" in data:
        labels = data["y"]
    else:
        raise KeyError(f"{path} missing mts_class labels ('mts_class' or 'y')")
    feature_names: List[str] = []
    if "pairs" in data:
        pairs = data["pairs"].tolist()
        for pair in pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                feature_names.append(f"{pair[0]}|{pair[1]}")
            else:
                feature_names.append(str(pair))
    else:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    return X, labels, feature_names


def ks_between_groups(
    X: np.ndarray,
    labels: np.ndarray,
    cls_a: str,
    cls_b: str,
    feature_names: List[str],
    alternative: str = "two-sided",
    use_kde: bool = False,
    kde_bw: float = 0.05,
) -> pd.DataFrame:
    mask_a = labels == cls_a
    mask_b = labels == cls_b
    if not mask_a.any() or not mask_b.any():
        raise ValueError(f"Missing data for {cls_a} or {cls_b}")
    Xa = X[mask_a]
    Xb = X[mask_b]
    stats = []
    for idx in range(X.shape[1]):
        va = Xa[:, idx]
        vb = Xb[:, idx]
        if use_kde:
            # KDE-smoothed CDFs evaluated on a grid; fall back gracefully if singular/constant.
            try:
                from scipy.stats import gaussian_kde
            except ImportError as e:
                raise ImportError("scipy is required for KDE (--kde).") from e
            try:
                xmin = float(min(va.min(), vb.min()))
                xmax = float(max(va.max(), vb.max()))
                if not np.isfinite(xmin) or not np.isfinite(xmax):
                    raise ValueError("non-finite values")
                # avoid zero-range grids
                if abs(xmax - xmin) < 1e-9:
                    raise ValueError("near-constant values")
                grid = np.linspace(xmin, xmax, 512)
                kde_a = gaussian_kde(va, bw_method=kde_bw)
                kde_b = gaussian_kde(vb, bw_method=kde_bw)
                cdf_a = np.cumsum(kde_a(grid))
                cdf_b = np.cumsum(kde_b(grid))
                cdf_a /= cdf_a[-1]
                cdf_b /= cdf_b[-1]
                ks_stat = float(np.max(np.abs(cdf_a - cdf_b)))
            except Exception:
                # fallback: add tiny jitter and use standard KS
                jitter = 1e-4
                res = ks_2samp(va + jitter * np.random.randn(*va.shape), vb + jitter * np.random.randn(*vb.shape), alternative=alternative, mode="auto")
                ks_stat = float(res.statistic)
        else:
            res = ks_2samp(va, vb, alternative=alternative, mode="auto")
            ks_stat = float(res.statistic)
        spi1, spi2 = ("", "")
        name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
        if "|" in name:
            spi1, spi2 = name.split("|", 1)
        stats.append(
            {
                "feature_index": idx,
                "SPI-1": spi1 or name,
                "SPI-2": spi2,
                "ks_statistic": ks_stat,
                "n_" + cls_a: int(mask_a.sum()),
                "n_" + cls_b: int(mask_b.sum()),
            }
        )
    return pd.DataFrame(stats)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    feat_path = Path(args.features)
    mts_classes = [c.strip() for c in args.mts_classes.split(",") if c.strip()]
    if len(mts_classes) < 2:
        raise ValueError("Provide at least two mts_classes for comparison.")
    bw_values = [float(p) for p in str(args.kde_bandwidth).split(",") if p.strip()]
    if args.kde and not bw_values:
        raise ValueError("Provide at least one bandwidth when using --kde.")
    if not args.kde:
        bw_values = [None]  # placeholder; not used when KDE is off

    X, labels, feature_names = load_features_npz(feat_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Loaded X shape=%s, labels=%d classes", X.shape, len(np.unique(labels)))

    for i in range(len(mts_classes)):
        for j in range(i + 1, len(mts_classes)):
            cls_a, cls_b = mts_classes[i], mts_classes[j]
            for bw in bw_values:
                bw_label = bw if bw is not None else args.kde_bandwidth
                LOG.info("KS: %s vs %s | bw=%s", cls_a, cls_b, bw_label)
                df = ks_between_groups(
                    X,
                    labels,
                    cls_a,
                    cls_b,
                    feature_names,
                    alternative=args.alternative,
                    use_kde=args.kde,
                    kde_bw=bw if bw is not None else 0.05,
                )
                pair_label = "_".join(sorted([cls_a, cls_b]))
                if args.kde:
                    subdir = out_dir / f"kde-{bw_label}"
                    subdir.mkdir(parents=True, exist_ok=True)
                    out_path = subdir / f"ks_{pair_label}.csv"
                else:
                    out_path = out_dir / f"ks_{pair_label}.csv"
                df.to_csv(out_path, index=False)
                LOG.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
