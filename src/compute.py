from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd


from pyspi.calculator import Calculator


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
    # {identifier: "ExcType: message"} for SPIs that failed. A failed SPI still
    # occupies its column, filled with NaN, so without this a dead column is
    # indistinguishable from a legitimately undefined statistic.
    errors: Dict[str, str] | None = None


def run_pyspi(
    timeseries: np.ndarray,
    *,
    config_path: Path,
    normalise: bool = False,
    n_jobs: int | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    mp_context: str | None = None,
) -> ComputeResult:
    if timeseries.ndim != 2:
        raise ValueError("Timeseries array must be 2D (T x M).")
    M = timeseries.shape[1]
    # pyspi 3.0: `subset`/`configfile` collapsed into `config=` (a bundled name
    # or a path), and `normalise=` became `zscore=` (same per-process z-score
    # along time; the old name collided with the per-SPI `normalise` arguments
    # in pyspi.statistics.distance).
    calc = Calculator(
        dataset=timeseries.T,
        config=str(config_path),
        zscore=normalise,
        verbose=False,
    )
    calc.compute(
        n_jobs=n_jobs,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        mp_context=mp_context,
        progress=False,
    )
    info_map = _spi_info(calc.spis)
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
    return ComputeResult(
        table=calc.table.copy(), matrices=matrices, metadata=metadata,
        timings=spi_timings, errors=dict(getattr(calc, "errors", {}) or {}),
    )


# Structural traits that mean "the matrix is not symmetric, do not fold it".
# `antisymmetric` (gd_*, phase_*, pli/wpli/psi, ccm_*_diff) is the one that
# bites: folding an antisymmetric matrix with 0.5*(A + A.T) returns zeros.
_NON_SYMMETRIC_LABELS = {"directed", "antisymmetric", "asymmetric"}


def _spi_info(spis: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Read each SPI's labels off the instance, not off the config YAML.

    pyspi 3.0 makes the class's own structural traits authoritative: exactly
    one of directed/undirected/antisymmetric/asymmetric survives the merge of
    class, family and per-config labels, and `issigned()` overrides any
    declared signed/unsigned. A config that declares `undirected` over an
    antisymmetric measure -- which the hand-written case configs here do for
    CoherencePhase -- is corrected by the loader, so `spi.labels` is the only
    reading that matches the matrix that was actually computed.
    """
    info: Dict[str, Dict[str, Any]] = {}
    for identifier, spi in spis.items():
        labels = [str(lbl) for lbl in getattr(spi, "labels", [])]
        lowered = {lbl.lower() for lbl in labels}
        info[identifier] = {
            "labels": labels,
            "directed": bool(lowered & _NON_SYMMETRIC_LABELS),
            "family": spi.__module__.split(".")[-1],
            "module": spi.__module__,
            "class_name": spi.__class__.__name__,
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
    mat = np.array(table[spi_name], dtype=float, copy=True)
    if mat.shape != (M, M):
        raise ValueError(
            f"Expected (M,M)=({M},{M}) matrix for SPI '{spi_name}', got {mat.shape}."
        )
    np.fill_diagonal(mat, 0.0)
    if symmetrise:
        mat = 0.5 * (mat + mat.T)
    return mat
