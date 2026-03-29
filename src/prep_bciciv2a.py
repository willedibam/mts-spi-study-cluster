"""
Preprocess BCI Competition IV Dataset 2a for pyspi via MOABB.

MOABB (Mother of All BCI Benchmarks) downloads the dataset automatically —
no manual download required. Data is cached in ~/mne_data/ on first run.

Usage (run once from project root):

    python -m src.prep_bciciv2a \\
        --output-dir data/bciciv2a \\
        --config-out configs/generate/bciciv2a.yaml

Output layout (all 9 subjects pooled):

    data/bciciv2a/
        <class>/class<class>_I{n}/timeseries.npy   (~648 trials per class)

Each timeseries.npy: float32, shape (T=1000, M=22), z-scored per channel.
Trials within each class are ordered by (subject, session, run).
Existing files are skipped (safe to re-run).
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import zscore

from .prep_uea import _write_yaml_config
from .utils import class_dir_name, slugify

_N_SUBJECTS = list(range(1, 10))   # subjects 1-9
_T_EPOCH = 1000                    # 4 s × 250 Hz
_TMIN = 0.0
_TMAX = (_T_EPOCH - 1) / 250.0    # 3.996 s → exactly 1000 samples


def _load_all_trials():
    """
    Download and epoch BCI IV 2a via MOABB.
    Returns X (N, M, T) float64 and y (N,) str.
    """
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError as exc:
        raise SystemExit("moabb is required: uv pip install moabb") from exc

    print("[INFO] Initialising MOABB BNCI2014_001 (BCI IV 2a) …")
    dataset = BNCI2014_001()

    # No bandpass — let pyspi SPIs see the full spectrum.
    # tmin/tmax set to yield exactly T=1000 samples at 250 Hz.
    paradigm = MotorImagery(
        n_classes=4,
        tmin=_TMIN,
        tmax=_TMAX,
        fmin=None,
        fmax=None,
        baseline=None,
    )

    print(f"[INFO] Fetching data for subjects {_N_SUBJECTS} (downloads on first run) …")
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=_N_SUBJECTS)
    # X: (N, M, T) — MOABB returns channels-last or channels-first depending on version
    # Ensure (N, M, T): if last dim > first spatial dim, it's channels-last → transpose
    if X.ndim == 3 and X.shape[2] < X.shape[1]:
        X = X.transpose(0, 2, 1)  # (N, T, M) → (N, M, T)

    N, M, T = X.shape
    classes = sorted(set(y.tolist()))
    print(f"[INFO] {N} trials  M={M}  T={T}  classes={classes}")
    return X, y


def _write_trials(X_all: np.ndarray, y_all: np.ndarray, out_root: Path) -> dict:
    """Write per-trial timeseries.npy. Returns summary dict for _write_yaml_config."""
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

            sample = X_all[trial_idx]   # (M, T) channels-first
            data = sample.T             # (T, M)
            data = zscore(data, axis=0, nan_policy="omit")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = data.astype(np.float32)

            dest.mkdir(parents=True, exist_ok=True)
            np.save(ts_path, data)
            written += 1

        print(
            f"[INFO]   '{cls_label}' ({cls_slug}): {len(ordered)} trials  "
            f"written={written}  skipped={skipped}"
        )
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
