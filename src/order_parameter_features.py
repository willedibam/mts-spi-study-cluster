from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .process_features import _edge_vectors, build_spi_spi_features
from .utils import load_json


def load_spi_catalog(dataset_dir: str | Path) -> list[dict]:
    meta = load_json(Path(dataset_dir) / "meta.json")
    return list(meta["pyspi"]["spis"])


def validate_spi_catalogs(
    dataset_dirs: Sequence[str | Path],
) -> list[dict]:
    if not dataset_dirs:
        raise ValueError("dataset_dirs is empty")
    reference = load_spi_catalog(dataset_dirs[0])
    signature = [(item["name"], bool(item.get("directed", False))) for item in reference]
    for dataset_dir in dataset_dirs[1:]:
        current = load_spi_catalog(dataset_dir)
        current_signature = [
            (item["name"], bool(item.get("directed", False))) for item in current
        ]
        if current_signature != signature:
            raise ValueError(f"SPI catalog mismatch in {dataset_dir}")
    return reference


def explicit_phase_spi_names(catalog: Sequence[dict]) -> set[str]:
    """Conservative ablation of SPIs whose definition explicitly uses phase."""

    prefixes = ("phase_", "plv", "pli", "wpli", "dspli", "dswpli", "ppc")
    selected: set[str] = set()
    for info in catalog:
        name = str(info["name"])
        class_name = str(info.get("class_name", ""))
        if "phase" in class_name.lower() or name.lower().startswith(prefixes):
            selected.add(name)
    return selected


def spi_validity_rates(
    dataset_dirs: Sequence[str | Path],
    catalog: Sequence[dict],
    *,
    split_directed: bool = False,
    min_centered_norm: float = 1e-10,
) -> dict[str, float]:
    counts = {str(info["name"]): 0 for info in catalog}
    for dataset_dir in dataset_dirs:
        with np.load(Path(dataset_dir) / "spi_mpis.npz") as archive:
            for info in catalog:
                name = str(info["name"])
                entries = _edge_vectors(
                    name,
                    archive[name],
                    bool(info.get("directed", False)),
                    split_directed,
                )
                valid = True
                for _, vector in entries:
                    vector = np.asarray(vector, dtype=np.float64)
                    valid &= bool(
                        np.isfinite(vector).all()
                        and np.linalg.norm(vector - vector.mean()) >= min_centered_norm
                    )
                counts[name] += int(valid)
    denominator = float(len(dataset_dirs))
    return {name: count / denominator for name, count in counts.items()}


def stable_spi_names(
    dataset_dirs: Sequence[str | Path],
    catalog: Sequence[dict],
    *,
    min_valid_fraction: float = 1.0,
    split_directed: bool = False,
    exclude: Iterable[str] = (),
) -> tuple[list[str], dict[str, float]]:
    if not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must lie in [0, 1]")
    rates = spi_validity_rates(
        dataset_dirs,
        catalog,
        split_directed=split_directed,
    )
    excluded = set(exclude)
    names = [
        str(info["name"])
        for info in catalog
        if rates[str(info["name"])] >= min_valid_fraction
        and str(info["name"]) not in excluded
    ]
    if len(names) < 2:
        raise RuntimeError(f"only {len(names)} stable SPIs remain")
    return names, rates


def build_meta_feature_matrix(
    dataset_dirs: Sequence[str | Path],
    catalog: Sequence[dict],
    spi_names: Sequence[str],
    *,
    metric: str = "pearson",
    split_directed: bool = False,
) -> tuple[np.ndarray, list[tuple[str, str]]]:
    by_name = {str(info["name"]): info for info in catalog}
    missing = [name for name in spi_names if name not in by_name]
    if missing:
        raise ValueError(f"unknown SPI names: {', '.join(missing)}")
    directed = [bool(by_name[name].get("directed", False)) for name in spi_names]
    rows: list[np.ndarray] = []
    pseudo_names: list[str] | None = None
    for dataset_dir in dataset_dirs:
        row, current_names = build_spi_spi_features(
            {"path": Path(dataset_dir)},
            list(spi_names),
            directed,
            split_directed=split_directed,
            metric=metric,
            nonfinite_policy="nan",
        )
        if pseudo_names is None:
            pseudo_names = current_names
        elif current_names != pseudo_names:
            raise ValueError("pseudo-SPI ordering mismatch")
        rows.append(row)
    if pseudo_names is None:
        raise ValueError("dataset_dirs is empty")
    pairs = [
        (pseudo_names[i], pseudo_names[j])
        for i in range(len(pseudo_names))
        for j in range(i + 1, len(pseudo_names))
    ]
    return np.vstack(rows), pairs
