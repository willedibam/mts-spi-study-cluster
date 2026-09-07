"""Fixed-population, paired-view data for the representation state pilot.

Coupling indices are used only to balance sampling and inner folds. Predictors
receive observed cosines, never coupling, frequencies or simulator phases.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
import yaml


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed_view(master: np.ndarray, m: int, t: int) -> np.ndarray:
    if not (2 <= m <= master.shape[1] and 2 <= t <= master.shape[0]):
        raise ValueError("view exceeds its master")
    return np.ascontiguousarray(master[-t:, :m])


def source_pool_for_seed(rows, pool, protocol, seed):
    """Optionally use independent training cohorts, retaining nested budgets."""
    if not protocol['sampling'].get('disjoint_training_cohorts', False):
        return pool
    cohort = protocol['methods']['subset_seeds'].index(seed)
    selected = np.asarray([i for i in pool if rows[i]['cohort_index'] == cohort], dtype=int)
    if not len(selected):
        raise ValueError('empty training cohort')
    return selected


def simple_observables(view: np.ndarray) -> np.ndarray:
    correlation = np.corrcoef(view.T)
    mean_absolute = np.abs(correlation[~np.eye(view.shape[1], dtype=bool)]).mean()
    # Entire observed window, no future padding/data. Identical edge treatment
    # across methods/lengths; finite-window Hilbert error is part of this control.
    analytic = hilbert(view, axis=0)
    unit = analytic / np.maximum(np.abs(analytic), np.finfo(float).eps)
    coherence = np.abs(unit.mean(axis=1)).mean()
    return np.asarray([mean_absolute, coherence])


def load_state_data(root: Path, config_path: Path) -> tuple[dict, np.ndarray]:
    manifest = json.loads((root / "manifest.json").read_text())
    protocol = yaml.safe_load(config_path.read_text())
    data_protocol = Path(protocol.get("data_protocol", config_path))
    if manifest["config_sha256"] != file_hash(data_protocol):
        raise ValueError("data/protocol hash mismatch")
    original = yaml.safe_load(data_protocol.read_text())
    if any(protocol[key] != original[key] for key in ("generator", "target", "observations", "sampling")):
        raise ValueError("execution protocol changes data construction or sampling")
    for name, digest in manifest["artifacts"].items():
        if file_hash(root / name) != digest:
            raise ValueError(f"data hash mismatch: {name}")
    masters = np.load(root / "masters.npy", mmap_mode="r", allow_pickle=False)
    rows = manifest["rows"]
    roles = {}
    for row in rows:
        previous = roles.setdefault(row["master_index"], row["role"])
        if previous != row["role"]:
            raise ValueError("master leaks across training/evaluation")
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate row IDs")
    return manifest, masters
