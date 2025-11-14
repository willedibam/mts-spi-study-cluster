from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import java_bridge  # noqa: F401  # ensures JVM started for PySPI
from . import pyspi_patches  # noqa: F401  # patches spectral input handling
from pyspi.calculator import Calculator

from .utils import load_yaml


@dataclass
class SPIInfo:
    name: str
    directed: bool
    labels: List[str]


@dataclass
class ComputeResult:
    table: pd.DataFrame
    matrices: Dict[str, np.ndarray]
    metadata: List[SPIInfo]


def run_pyspi(
    timeseries: np.ndarray,
    *,
    config_path: Path,
    subset: str = "default",
    normalise: bool = True,
) -> ComputeResult:
    if timeseries.ndim != 2:
        raise ValueError("Timeseries array must be 2D (T x M).")
    M = timeseries.shape[1]
    calc = Calculator(
        dataset=timeseries.T,
        subset=subset,
        configfile=str(config_path),
        normalise=normalise,
    )
    calc.compute()
    info_map = _load_spi_info(config_path)
    spi_names = _extract_spi_names(calc.table)
    matrices: Dict[str, np.ndarray] = {}
    metadata: List[SPIInfo] = []
    for spi_name in spi_names:
        info = info_map.get(spi_name, {})
        directed = info.get("directed", False)
        labels = info.get("labels", [])
        matrices[spi_name] = _reconstruct_mpi(
            calc.table, spi_name, M=M, symmetrise=not directed
        )
        metadata.append(SPIInfo(name=spi_name, directed=directed, labels=labels))
    return ComputeResult(table=calc.table.copy(), matrices=matrices, metadata=metadata)


def _load_spi_info(config_path: Path) -> Dict[str, Dict[str, Any]]:
    cfg = load_yaml(config_path)
    info: Dict[str, Dict[str, Any]] = {}
    for _, group in cfg.items():
        for spi_name, entry in (group or {}).items():
            labels = [label.lower() for label in entry.get("labels", [])]
            info[spi_name] = {
                "labels": labels,
                "directed": "directed" in labels,
            }
    return info


def _extract_spi_names(table: pd.DataFrame) -> List[str]:
    cols = table.columns
    if isinstance(cols, pd.MultiIndex):
        return list(pd.unique(cols.get_level_values(0)))
    names: List[str] = []
    for col in cols:
        if isinstance(col, tuple):
            names.append(col[0])
        else:
            names.append(col)
    seen: List[str] = []
    for name in names:
        if name not in seen:
            seen.append(name)
    return seen


def _reconstruct_mpi(
    table: pd.DataFrame,
    spi_name: str,
    *,
    M: int,
    symmetrise: bool,
) -> np.ndarray:
    cols = [
        c
        for c in table.columns
        if (isinstance(c, tuple) and c[0] == spi_name) or (c == spi_name)
    ]
    if cols:
        def _proc_key(col):
            if isinstance(col, tuple) and isinstance(col[1], str):
                parts = col[1].split("-")
                if len(parts) == 2 and parts[0] == "proc":
                    try:
                        return int(parts[1])
                    except ValueError:
                        return 0
            return 0

        cols_sorted = sorted(cols, key=_proc_key)
        vecs = [np.asarray(table[c]).ravel() for c in cols_sorted]
        if len(vecs) >= M and all(v.size == M for v in vecs[:M]):
            mat = np.column_stack(vecs[:M])
            np.fill_diagonal(mat, 0.0)
            return 0.5 * (mat + mat.T) if symmetrise else mat
    vec = np.asarray(table[spi_name]).astype(float).ravel()
    E_dir = M * (M - 1)
    E_und = M * (M - 1) // 2
    if vec.size == E_dir:
        mat = np.zeros((M, M), float)
        idx = 0
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                mat[i, j] = vec[idx]
                idx += 1
        np.fill_diagonal(mat, 0.0)
        return 0.5 * (mat + mat.T) if symmetrise else mat
    if vec.size == E_und:
        mat = np.zeros((M, M), float)
        iu = np.triu_indices(M, k=1)
        mat[iu] = vec
        if symmetrise:
            mat[(iu[1], iu[0])] = vec
        else:
            np.fill_diagonal(mat, 0.0)
        return mat
    mat = np.array(table[spi_name])
    if mat.ndim == 2 and mat.shape == (M, M):
        return mat
    raise ValueError(f"Cannot reconstruct MPI for '{spi_name}'")
