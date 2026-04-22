from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd


from pyspi.calculator import Calculator

from .utils import load_yaml


@dataclass
class SPIInfo:
    name: str
    directed: bool
    labels: List[str]
    family: str = ""
    module: str = ""
    class_name: str = ""


@dataclass
class ComputeResult:
    table: pd.DataFrame
    matrices: Dict[str, np.ndarray]
    metadata: List[SPIInfo]
    timings: Dict[str, float] | None = None


def run_pyspi(
    timeseries: np.ndarray,
    *,
    config_path: Path,
    subset: str = "default",
    normalise: bool = False,
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
    info_map = _load_spi_info(config_path, calc.spis)
    spi_names = _extract_spi_names(calc.table)
    matrices: Dict[str, np.ndarray] = {}
    metadata: List[SPIInfo] = []
    for spi_name in spi_names:
        info = info_map.get(spi_name, {})
        directed = info.get("directed", False)
        labels = info.get("labels", [])
        family = info.get("family", "")
        module = info.get("module", "")
        class_name = info.get("class_name", "")
        matrices[spi_name] = _reconstruct_mpi(
            calc.table,
            spi_name,
            M=M,
            symmetrise=not directed,
        )
        metadata.append(SPIInfo(
            name=spi_name, directed=directed, labels=labels,
            family=family, module=module, class_name=class_name,
        ))
    spi_timings = getattr(calc, 'timings', None)
    return ComputeResult(table=calc.table.copy(), matrices=matrices, metadata=metadata, timings=spi_timings)


def _load_spi_info(
    config_path: Path, spis: Mapping[str, Any]
) -> Dict[str, Dict[str, Any]]:
    cfg = load_yaml(config_path) or {}
    labels_by_origin: Dict[tuple[str, str], List[str]] = {}
    for module_name, group in cfg.items():
        module_key = module_name.lstrip(".")
        for spi_name, entry in (group or {}).items():
            labels = entry.get("labels") or []
            labels_by_origin[(module_key, spi_name)] = labels
    info: Dict[str, Dict[str, Any]] = {}
    for identifier, spi in spis.items():
        module_key = spi.__module__.split("pyspi.")[-1].lstrip(".")
        class_name = spi.__class__.__name__
        labels = labels_by_origin.get((module_key, class_name), [])
        directed = any(label.lower() == "directed" for label in labels)
        family = spi.__module__.split(".")[-1]
        info[identifier] = {
            "labels": labels, "directed": directed, "family": family,
            "module": spi.__module__, "class_name": class_name,
        }
    return info


def _extract_spi_names(table: pd.DataFrame) -> List[str]:
    return list(pd.unique(table.columns.get_level_values(0)))


def _reconstruct_mpi(
    table: pd.DataFrame,
    spi_name: str,
    *,
    M: int,
    symmetrise: bool,
) -> np.ndarray:
    mat = np.asarray(table[spi_name], dtype=float)
    if mat.shape != (M, M):
        raise ValueError(
            f"Expected (M,M)=({M},{M}) matrix for SPI '{spi_name}', got {mat.shape}."
        )
    np.fill_diagonal(mat, 0.0)
    if symmetrise:
        mat = 0.5 * (mat + mat.T)
    return mat
