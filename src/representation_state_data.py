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


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def observed_view(master: np.ndarray, m: int, t: int) -> np.ndarray:
    if not (2 <= m <= master.shape[1] and 2 <= t <= master.shape[0]):
        raise ValueError("view exceeds its master")
    return np.ascontiguousarray(master[-t:, :m])


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
    if manifest["config_sha256"] != file_hash(config_path):
        raise ValueError("data/protocol hash mismatch")
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
