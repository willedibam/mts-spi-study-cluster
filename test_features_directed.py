"""
Compute SPI/SPI-SPI features with explicit direction splits for directed SPIs.

Directed SPIs are expanded into two features: <spi>__ij (upper triangle) and <spi>__ji (lower triangle).
Undirected SPIs use the upper-triangular off-diagonals.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils import load_json, project_root

LOGGER = logging.getLogger(__name__)


def load_spi_subset(path: str | Path) -> tuple[list[str], str]:
    subset_path = Path(path)
    if not subset_path.exists():
        raise FileNotFoundError(f"SPI subset file not found: {subset_path}")
    names: List[str] = []
    with subset_path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            name = raw.strip()
            if not name or name.startswith("#"):
                continue
            if name not in names:
                names.append(name)
    if not names:
        raise ValueError(f"SPI subset file is empty: {subset_path}")
    return names, subset_path.name


def build_index(
    data_path: str | Path,
    subset_names: List[str] | None = None,
    mts_classes: List[str] | None = None,
) -> pd.DataFrame:
    base = Path(data_path)
    if not base.exists():
        raise FileNotFoundError(f"Data path not found: {base}")
    class_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if mts_classes:
        allowed = set(mts_classes)
        class_dirs = [p for p in class_dirs if p.name in allowed]
    rows = []
    for class_dir in class_dirs:
        dataset_dirs = sorted(p for p in class_dir.iterdir() if p.is_dir())
        for ds_dir in dataset_dirs:
            meta_path = ds_dir / "meta.json"
            if not meta_path.exists():
                continue
            meta = load_json(meta_path)
            spis = meta.get("pyspi", {}).get("spis", [])
            if subset_names:
                spis = [s for s in spis if s.get("name") in subset_names]
            if not spis:
                continue
            rows.append(
                dict(
                    dataset_path=str(ds_dir),
                    mts_class=meta.get("mts_class", class_dir.name),
                    variant=(meta.get("variant") or {}).get("slug", ""),
                    M=meta.get("M"),
                    T=meta.get("T"),
                    instance=meta.get("instance_index"),
                    spis=spis,
                )
            )
    return pd.DataFrame(rows)


def _spi_vectors(
    spi_names: list[str],
    directed_flags: list[bool],
    ds_dir: str,
) -> list[tuple[str, np.ndarray]]:
    with np.load(Path(ds_dir) / "spi_mpis.npz") as npz:
        vectors: list[tuple[str, np.ndarray]] = []
        for name, directed in zip(spi_names, directed_flags):
            mat = np.asarray(npz[name], float)
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                continue
            if directed:
                upper = mat[np.triu_indices(mat.shape[0], k=1)]
                lower = mat[np.tril_indices(mat.shape[0], k=-1)]
                vectors.append((f"{name}__ij", upper))
                vectors.append((f"{name}__ji", lower))
            else:
                sym = 0.5 * (mat + mat.T)
                upper = sym[np.triu_indices(sym.shape[0], k=1)]
                vectors.append((name, upper))
    return vectors


def _zscore(vec: np.ndarray) -> np.ndarray:
    std = vec.std()
    if std < 1e-12 or not np.isfinite(std):
        return np.zeros_like(vec, dtype=np.float32)
    return ((vec - vec.mean()) / std).astype(np.float32)


def spi_spi_features(ds_row: Dict, space: str = "spi-spi") -> tuple[np.ndarray, list[str]]:
    spi_names = [s["name"] for s in ds_row["spis"]]
    directed_flags = [s.get("directed", False) for s in ds_row["spis"]]
    vectors = _spi_vectors(spi_names, directed_flags, ds_row["dataset_path"])
    names = [name for name, _ in vectors]

    if space == "spi-spi":
        n = len(vectors)
        corr = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                vi = vectors[i][1]
                vj = vectors[j][1]
                r = spearmanr(_zscore(vi), _zscore(vj)).correlation
                corr[i, j] = corr[j, i] = 0.0 if not np.isfinite(r) else r
        iu = np.triu_indices(n, k=1)
        return corr[iu], names

    if space == "spi":
        blocks = [_zscore(vec) for _, vec in vectors]
        return np.concatenate(blocks).astype(np.float32), names

    raise ValueError("space must be 'spi-spi' or 'spi'")


def build_feature_matrix(meta_df: pd.DataFrame, space: str) -> tuple[np.ndarray, list[str], List[tuple[str, str]]]:
    features = []
    first_names: list[str] = []
    for idx, row in enumerate(meta_df.itertuples(index=False), start=1):
        vec, names = spi_spi_features(row._asdict(), space=space)
        if not first_names:
            first_names = names
        features.append(vec)
        sys.stderr.write(
            f"\rfeatures-{space}: {idx}/{len(meta_df)} "
            f"({idx/len(meta_df):.0%})"
        )
        sys.stderr.flush()
    sys.stderr.write("\n")
    X = np.vstack(features)
    pairs = [(first_names[i], first_names[j]) for i in range(len(first_names)) for j in range(i + 1, len(first_names))] if space == "spi-spi" else []
    return X, first_names, pairs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and cache SPI feature matrices with directed splits.")
    parser.add_argument(
        "--data-path",
        default="data/full",
        help="Path to data root (e.g., data/full, data/case_2).",
    )
    parser.add_argument("--space", default="spi-spi", choices=["spi-spi", "spi"], help="Feature space.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick runs.")
    parser.add_argument("--output", type=str, default=None, help="Output npz path (default: analysis/features_directed_<data-path>_<space>[_limit].npz)")
    parser.add_argument("--spi-subset", type=str, default=None, help="Path to txt file listing SPI names (one per line).")
    parser.add_argument("--recompute", action="store_true", help="Recompute even if cache exists.")
    parser.add_argument(
        "--mts-classes",
        type=str,
        default=None,
        help="Comma-separated list of mts_class names to include (filters class folders under the data path).",
    )
    return parser.parse_args(argv)


def default_output(mode: str, space: str, limit: int | None, subset_label: str | None) -> Path:
    suffix = f"_limit{limit}" if limit else ""
    subset_suffix = f"_{subset_label}" if subset_label else ""
    safe_mode = mode.replace("\\", "-").replace("/", "-").strip("-")
    return project_root() / "analysis" / f"features_directed_{safe_mode}_{space}{subset_suffix}{suffix}.npz"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    subset_names: List[str] | None = None
    subset_label: str | None = None
    if args.spi_subset:
        subset_names, subset_label = load_spi_subset(args.spi_subset)
        logging.info(
            "Feature matrix | space=%s, subset=%s (%d SPIs)",
            args.space,
            subset_label,
            len(subset_names),
        )
    else:
        logging.info("Feature matrix | space=%s, subset=ALL", args.space)

    out_path = Path(args.output) if args.output else default_output(args.data_path, args.space, args.limit, subset_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.recompute:
        logging.info("Cache exists, skipping computation: %s", out_path)
        return

    mts_classes = (
        [part.strip() for part in args.mts_classes.split(",") if part.strip()]
        if args.mts_classes
        else None
    )
    meta_df = build_index(args.data_path, subset_names=subset_names, mts_classes=mts_classes)
    if args.limit:
        meta_df = meta_df.iloc[: args.limit].reset_index(drop=True)
    X, spi_order, pairs = build_feature_matrix(meta_df, args.space)
    logging.info("feature matrix shape: %s", X.shape)

    payload = {
        "X": X.astype(np.float32),
        "mts_class": meta_df["mts_class"].to_numpy(),
        "variant": meta_df["variant"].to_numpy(),
        "M": meta_df["M"].to_numpy(),
        "T": meta_df["T"].to_numpy(),
        "instance": meta_df["instance"].to_numpy(),
        "dataset_path": meta_df["dataset_path"].astype(str).to_numpy(),
        "spi_order": np.array(spi_order, dtype=object),
        "pairs": np.array(pairs, dtype=object),
        "space": args.space,
        "mode": args.data_path,
        "limit": args.limit if args.limit is not None else -1,
        "spi_subset": subset_label or "",
    }
    np.savez_compressed(out_path, **payload)
    logging.info("Saved features -> %s", out_path)


if __name__ == "__main__":
    main()
