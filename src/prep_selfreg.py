"""
Download SelfRegulationSCP1 (train + test splits) via aeon and write per-trial
timeseries.npy files into the directory layout expected by run_experiments.py.

Run once from the project root before submitting the PBS array job:

    python -m src.prep_selfreg [--output-dir data/selfreg]

Output layout mirrors what the pipeline would generate for a real-data config:

    data/selfreg/
        positive/classpositivity_I{n}/timeseries.npy   (n = 0 … 278, 279 trials)
        negative/classnegativiy_I{n}/timeseries.npy    (n = 0 … 281, 282 trials)

Each file is a float32 array of shape (T=896, M=6), z-scored per channel.
Directories are created as needed; existing files are skipped.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import zscore

from .utils import class_dir_name, slugify

_DATASET = "SelfRegulationSCP1"

# (config name, aeon class label)
_CLASSES = [
    ("positive", "positivity"),
    ("negative", "negativity"),
]


def _load_all_trials():
    """Return X (N, M, T) float64 and y (N,) str from both splits combined."""
    try:
        from aeon.datasets import load_classification
    except ImportError as exc:
        raise SystemExit(
            "aeon is required.  Install with:  uv pip install -e '.[real]'"
        ) from exc

    print(f"[INFO] Downloading {_DATASET} train split …")
    X_tr, y_tr = load_classification(_DATASET, split="train")
    print(f"[INFO] Downloading {_DATASET} test split …")
    X_te, y_te = load_classification(_DATASET, split="test")

    X_all = np.concatenate([X_tr, X_te], axis=0).astype(float)
    y_all = np.concatenate([np.asarray(y_tr), np.asarray(y_te)]).astype(str)
    print(f"[INFO] Combined: {X_all.shape[0]} trials, shape (N, M={X_all.shape[1]}, T={X_all.shape[2]})")
    return X_all, y_all


def _write_trials(X_all, y_all, out_root: Path) -> None:
    for cls_name, cls_label in _CLASSES:
        mask = np.where(y_all == cls_label)[0]
        if mask.size == 0:
            raise ValueError(f"No trials found for class label '{cls_label}'.")

        # Sort by original index for a fully deterministic, reproducible mapping.
        ordered = sorted(mask.tolist())
        class_dir = out_root / class_dir_name(cls_name)
        cls_slug = slugify(cls_label)

        print(f"[INFO] Class '{cls_label}': {len(ordered)} trials → {class_dir}")
        written = skipped = 0
        for instance, trial_idx in enumerate(ordered):
            dest = class_dir / f"class{cls_slug}_I{instance}"
            ts_path = dest / "timeseries.npy"
            if ts_path.exists():
                skipped += 1
                continue

            sample = X_all[trial_idx]      # (M, T) channels-first
            data = sample.T                # (T, M)
            data = zscore(data, axis=0, nan_policy="omit")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = data.astype(np.float32)

            dest.mkdir(parents=True, exist_ok=True)
            np.save(ts_path, data)
            written += 1

        print(f"[INFO]   Written: {written}  Skipped (already exist): {skipped}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/selfreg",
        type=Path,
        help="Root output directory (default: data/selfreg)",
    )
    args = parser.parse_args()

    X_all, y_all = _load_all_trials()
    _write_trials(X_all, y_all, args.output_dir)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
