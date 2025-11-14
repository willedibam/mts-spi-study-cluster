from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def project_root() -> Path:
    return PROJECT_ROOT


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_yaml(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def to_relative(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def slugify(text: str, fallback: str = "variant") -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or fallback


def class_dir_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return safe or "class"


def variant_suffix(params: dict[str, Any]) -> str:
    parts: List[str] = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, float):
            value = f"{value:.3g}".replace(".", "p")
        parts.append(f"{key}{value}")
    return slugify("-".join(parts)) if parts else ""


@dataclass
class DatasetMetaRecord:
    dataset_path: str
    meta: dict[str, Any]


def iter_dataset_dirs(mode: str) -> Iterator[Path]:
    base = DATA_ROOT / mode
    if not base.exists():
        return iter(())
    class_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    for class_dir in class_dirs:
        for dataset_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            if (dataset_dir / "meta.json").exists():
                yield dataset_dir


def load_all_meta(mode: str) -> pd.DataFrame:
    rows: List[dict[str, Any]] = []
    for dataset_dir in iter_dataset_dirs(mode):
        meta = load_json(dataset_dir / "meta.json")
        meta["dataset_path"] = to_relative(dataset_dir)
        rows.append(meta)
    return pd.DataFrame(rows)


@dataclass
class DatasetSPIBundle:
    dataset_path: str
    meta: dict[str, Any]
    mpis: Dict[str, np.ndarray]


def load_spi_for_dataset(dataset_dir: str | Path) -> DatasetSPIBundle:
    dataset_dir = Path(dataset_dir)
    meta = load_json(dataset_dir / "meta.json")
    mpis: Dict[str, np.ndarray] = {}
    npz_path = dataset_dir / "spi_mpis.npz"
    if npz_path.exists():
        with np.load(npz_path) as npz:
            for key in npz.files:
                mpis[key] = npz[key]
    return DatasetSPIBundle(
        dataset_path=to_relative(dataset_dir),
        meta=meta,
        mpis=mpis,
    )


def load_all_spi(mode: str) -> List[DatasetSPIBundle]:
    bundles: List[DatasetSPIBundle] = []
    for dataset_dir in iter_dataset_dirs(mode):
        bundles.append(load_spi_for_dataset(dataset_dir))
    return bundles

