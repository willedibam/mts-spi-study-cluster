"""
Preprocess BCI Competition IV Dataset 2a for pyspi via MOABB.

MOABB downloads the dataset automatically — no manual download required.
The MOABB paradigm pipeline is bypassed (MOABB 1.5 + MNE 1.11 IIR filter
incompatibility); .mat files are loaded directly with pymatreader instead.

Usage (run once from project root):

    python -m src.prep_bciciv2a \\
        --output-dir data/bciciv2a \\
        --config-out configs/generate/bciciv2a.yaml

Output layout (all 9 subjects pooled):

    data/bciciv2a/
        <class>/class<class>_I{n}/timeseries.npy   (~648 trials per class)

Each timeseries.npy: float32, shape (T=1000, M=22), z-scored per channel.
Trials are ordered by (subject, file, run, trial) for determinism.
Existing files are skipped (safe to re-run).
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import zscore

from .prep_uea import _write_yaml_config
from .utils import class_dir_name, slugify

_N_SUBJECTS = list(range(1, 10))  # subjects 1–9
_T_EPOCH = 1000                   # 4 s × 250 Hz
_N_EEG = 22                       # drop EOG channels (last 3 of 25)
_CLASS_LABELS = {1: "left-hand", 2: "right-hand", 3: "feet", 4: "tongue"}


# ---------------------------------------------------------------------------
# Download + path resolution
# ---------------------------------------------------------------------------

def _get_mat_paths(subject: int) -> list[Path]:
    """
    Return local .mat file paths for one subject, triggering MOABB download
    if the files do not yet exist.  Returns [T_path, E_path].
    """
    from moabb.datasets import BNCI2014_001
    from mne import get_config

    dataset = BNCI2014_001()

    # Resolve the MNE data directory (MOABB stores files there).
    mne_data = Path(get_config("MNE_DATA") or (Path.home() / "mne_data"))
    base = mne_data / "MNE-bnci-data" / "~bci" / "database" / "001-2014"

    paths = []
    for split in ("T", "E"):
        fpath = base / f"A0{subject}{split}.mat"
        if not fpath.exists():
            # Trigger MOABB download by calling data_path (ignore its return value).
            try:
                dataset.data_path(subject)
            except Exception:
                pass
        if fpath.exists():
            paths.append(fpath)
        else:
            print(f"[WARN] Could not find {fpath} after download attempt — skipping.")
    return paths


# ---------------------------------------------------------------------------
# .mat loading (BNCI 001-2014 format from lampx.tugraz.at)
# ---------------------------------------------------------------------------

def _read_mat(fpath: Path) -> dict:
    """Load a .mat file, returning a plain Python dict."""
    try:
        from pymatreader import read_mat
        return read_mat(str(fpath))
    except Exception:
        from scipy.io import loadmat
        return loadmat(str(fpath), struct_as_record=False, squeeze_me=True)


def _extract_epochs_from_mat(d: dict, fpath: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract epochs from a BNCI 001-2014 .mat dict.

    Format: d['data'] is a list of runs (training: 3 runs × 72 trials;
    eval: 3 runs × 72 trials).  Each run has keys X, trial, y, fs.

    Returns (X, y): X (N, M, T), y (N,) str.
    """
    runs_raw = d.get("data")
    if runs_raw is None:
        raise ValueError(f"No 'data' key in {fpath.name}. Keys: {list(d.keys())}")

    # pymatreader may return a list of dicts or a numpy object array.
    if isinstance(runs_raw, np.ndarray) and runs_raw.dtype == object:
        runs = [runs_raw[i] for i in range(runs_raw.size)]
    elif isinstance(runs_raw, (list, tuple)):
        runs = list(runs_raw)
    else:
        runs = [runs_raw]  # single run

    X_parts, y_parts = [], []
    for run_idx, run in enumerate(runs):
        # Normalise: dict (pymatreader) or scipy struct
        if hasattr(run, "__dict__"):
            run = {k: getattr(run, k) for k in run._fieldnames}
        if not isinstance(run, dict):
            continue

        X_raw = np.asarray(run.get("X", []), dtype=float)
        trials = np.asarray(run.get("trial", []), dtype=int).flatten()
        labels = np.asarray(run.get("y", []), dtype=int).flatten()

        if X_raw.size == 0 or trials.size == 0:
            continue

        # Ensure (n_channels, n_samples): whichever axis is smaller is channels.
        if X_raw.ndim == 2 and X_raw.shape[0] > X_raw.shape[1]:
            X_raw = X_raw.T  # was (n_samples, n_channels)

        # Keep only EEG channels.
        M = min(X_raw.shape[0], _N_EEG)
        X_raw = X_raw[:M, :]

        n_total = X_raw.shape[1]
        for onset, label in zip(trials, labels):
            end = int(onset) + _T_EPOCH
            if end > n_total:
                continue
            lbl_int = int(label)
            if lbl_int not in _CLASS_LABELS:
                continue
            X_parts.append(X_raw[:, int(onset):end])  # (M, T)
            y_parts.append(_CLASS_LABELS[lbl_int])

    if not X_parts:
        raise ValueError(f"No valid epochs extracted from {fpath.name}.")

    return np.stack(X_parts, axis=0), np.array(y_parts)


# ---------------------------------------------------------------------------
# Main loading entry point
# ---------------------------------------------------------------------------

def _load_all_trials() -> tuple[np.ndarray, np.ndarray]:
    """
    Download and epoch all 9 subjects (training + eval splits).
    Returns X (N, M, T) float64 and y (N,) str.
    """
    try:
        import moabb  # noqa: F401
    except ImportError as exc:
        raise SystemExit("moabb is required: uv pip install -e '.[real]'") from exc

    X_parts, y_parts = [], []
    for subj in _N_SUBJECTS:
        print(f"[INFO] Subject {subj} …")
        for fpath in _get_mat_paths(subj):
            print(f"[INFO]   Loading {fpath.name} …")
            d = _read_mat(fpath)
            X_s, y_s = _extract_epochs_from_mat(d, fpath)
            X_parts.append(X_s)
            y_parts.append(y_s)
            counts = {c: int((y_s == c).sum()) for c in np.unique(y_s)}
            print(f"[INFO]   {fpath.name}: {len(y_s)} trials  {counts}")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts)
    N, M, T = X_all.shape
    print(f"[INFO] Combined: {N} trials  M={M}  T={T}")
    return X_all, y_all


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

def _write_trials(X_all: np.ndarray, y_all: np.ndarray, out_root: Path) -> dict:
    class_labels = sorted(set(y_all.tolist()))
    summary = {}
    for cls_label in class_labels:
        mask = np.where(y_all == cls_label)[0]
        ordered = sorted(mask.tolist())
        cls_slug = slugify(cls_label)
        cls_name = cls_slug
        class_dir = out_root / class_dir_name(cls_name)
        written = skipped = 0
        for instance, trial_idx in enumerate(ordered):
            dest = class_dir / f"class{cls_slug}_I{instance}"
            ts_path = dest / "timeseries.npy"
            if ts_path.exists():
                skipped += 1
                continue
            sample = X_all[trial_idx]   # (M, T)
            data = sample.T             # (T, M)
            data = zscore(data, axis=0, nan_policy="omit")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = data.astype(np.float32)
            dest.mkdir(parents=True, exist_ok=True)
            np.save(ts_path, data)
            written += 1
        print(f"[INFO]   '{cls_label}' ({cls_slug}): {len(ordered)} trials  "
              f"written={written}  skipped={skipped}")
        summary[cls_label] = (cls_slug, cls_name, len(ordered))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config-out", type=Path, default=None)
    parser.add_argument("--pyspi-config", default="configs/pyspi-v2/cases/eeml.yaml")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rng-seed", type=int, default=110305)
    args = parser.parse_args()

    X_all, y_all = _load_all_trials()
    summary = _write_trials(X_all, y_all, args.output_dir)

    if args.config_out:
        total = _write_yaml_config(
            dataset="BCICIV2a",
            out_root=args.output_dir,
            summary=summary,
            pyspi_config=args.pyspi_config,
            threads=args.threads,
            rng_seed=args.rng_seed,
            config_out=args.config_out,
        )
        stem = args.config_out.stem
        print(f"[INFO] Config written → {args.config_out}")
        print(f"[INFO] Total PBS jobs : {total}")
        print(f"[INFO] Submit with   : qsub -J 1-{total} -v DATASET={stem} jobs/physics/run_uea.pbs")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
