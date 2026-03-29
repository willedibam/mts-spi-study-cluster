"""
Preprocess BCI Competition IV Dataset 2a for pyspi.

Manual download required (registration at bbci.de):
  1. https://www.bbci.de/competition/iv/  →  Data sets  →  Data set 2a
  2. Download BCICIV_2a_gdf.zip  (training + eval GDFs, ~90 MB)
  3. Download true_labels.zip     (eval class labels, tiny)
  4. Extract both into data/raw/bciciv2a/ so the layout is:

       data/raw/bciciv2a/
           A01T.gdf  A02T.gdf  ...  A09T.gdf   (training, labels embedded)
           A01E.gdf  A02E.gdf  ...  A09E.gdf   (eval, labels separate)
           true_labels/
               A01E.mat  A02E.mat  ...  A09E.mat

Usage (run once from project root):

    python -m src.prep_bciciv2a \\
        --data-dir data/raw/bciciv2a \\
        --output-dir data/bciciv2a \\
        --config-out configs/generate/bciciv2a.yaml

Output layout (pooled across all 9 subjects):

    data/bciciv2a/
        left/classleft_I{n}/timeseries.npy       (~648 trials)
        right/classright_I{n}/timeseries.npy     (~648 trials)
        feet/classfeet_I{n}/timeseries.npy       (~648 trials)
        tongue/classtongue_I{n}/timeseries.npy   (~648 trials)

Each timeseries.npy: float32, shape (T=1000, M=22), z-scored per channel.
Trials within each class are ordered by (subject index, original trial index).
Existing files are skipped (safe to re-run).
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import zscore

from .prep_uea import _write_yaml_config
from .utils import class_dir_name, slugify

# GDF event code → class label
_CLASS_EVENTS = {769: "left", 770: "right", 771: "feet", 772: "tongue"}
# eval files use 783 for "unknown"; true labels are 1-4 → add 768 to get event code
_EVAL_EVENT_CODE = 783
_N_EEG = 22   # drop EOG channels (last 3 of 25)
_T_EPOCH = 1000  # 4 s × 250 Hz


def _pick_eeg(raw):
    """Return channel indices: prefer type=EEG, fall back to first _N_EEG channels."""
    try:
        import mne
        picks = mne.pick_types(raw.info, eeg=True, eog=False)
    except Exception:
        picks = np.arange(len(raw.ch_names))
    if len(picks) == 0 or len(picks) < _N_EEG:
        picks = np.arange(_N_EEG)
    return picks[:_N_EEG]


def _epoch_data(raw, event_times, picks):
    """
    Extract fixed-length epochs from raw data.

    event_times : sample indices of epoch onsets (int array)
    Returns ndarray of shape (N, M, T).
    """
    data = raw.get_data(picks=picks)  # (M, total_samples)
    n_total = data.shape[1]
    epochs = []
    for onset in event_times:
        end = onset + _T_EPOCH
        if end > n_total:
            continue
        epochs.append(data[:, onset:end])
    return np.stack(epochs, axis=0)  # (N, M, T)


def _load_training(gdf_path: Path):
    """Load one training file. Returns (X, y): X (N, M, T), y str array."""
    import mne
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    picks = _pick_eeg(raw)

    X_parts, y_parts = [], []
    for code, label in _CLASS_EVENTS.items():
        mask = events[:, 2] == code
        if not mask.any():
            continue
        onsets = events[mask, 0]
        X_cls = _epoch_data(raw, onsets, picks)
        if X_cls.shape[0] == 0:
            continue
        X_parts.append(X_cls)
        y_parts.extend([label] * X_cls.shape[0])

    if not X_parts:
        raise ValueError(f"No class events (769-772) found in {gdf_path.name}")

    return np.concatenate(X_parts, axis=0), np.array(y_parts)


def _load_eval(gdf_path: Path, label_path: Path):
    """Load one eval file + true_labels .mat. Returns (X, y)."""
    import mne
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, verbose=False)
    events, _ = mne.events_from_annotations(raw, verbose=False)
    picks = _pick_eeg(raw)

    trial_mask = events[:, 2] == _EVAL_EVENT_CODE
    if not trial_mask.any():
        # Some releases encode eval trials differently; try any event.
        raise ValueError(
            f"No event code {_EVAL_EVENT_CODE} in {gdf_path.name}. "
            "Check that this is an eval (E) file."
        )
    onsets = events[trial_mask, 0]

    mat = loadmat(str(label_path))
    # Variable 'classlabel': shape (N,) or (N,1), values 1-4
    raw_labels = mat["classlabel"].flatten().astype(int)

    if len(onsets) != len(raw_labels):
        raise ValueError(
            f"{gdf_path.name}: {len(onsets)} trial onsets but "
            f"{len(raw_labels)} labels in {label_path.name}"
        )

    X = _epoch_data(raw, onsets, picks)
    # labels 1-4 → event codes 769-772 → string labels
    y = np.array([_CLASS_EVENTS[int(l) + 768] for l in raw_labels[: len(X)]])
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

            sample = X_all[trial_idx]  # (M, T) channels-first
            data = sample.T            # (T, M)
            data = zscore(data, axis=0, nan_policy="omit")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = data.astype(np.float32)

            dest.mkdir(parents=True, exist_ok=True)
            np.save(ts_path, data)
            written += 1

        print(
            f"[INFO]   '{cls_label}': {len(ordered)} trials  "
            f"written={written}  skipped={skipped}"
        )
        summary[cls_label] = (cls_slug, cls_name, len(ordered))

    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Directory containing A01T.gdf ... A09E.gdf + true_labels/")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config-out", type=Path, default=None)
    parser.add_argument("--pyspi-config", default="configs/pyspi-v2/cases/eeml.yaml")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rng-seed", type=int, default=110305)
    args = parser.parse_args()

    try:
        import mne  # noqa: F401
    except ImportError:
        raise SystemExit("mne is required: uv pip install mne")

    X_parts, y_parts = [], []
    for subj in range(1, 10):
        gdf_t = args.data_dir / f"A0{subj}T.gdf"
        if not gdf_t.exists():
            raise FileNotFoundError(
                f"Missing {gdf_t}. Extract BCICIV_2a_gdf.zip into {args.data_dir}."
            )
        print(f"[INFO] Subject {subj} — training …")
        X_t, y_t = _load_training(gdf_t)
        X_parts.append(X_t)
        y_parts.append(y_t)

        gdf_e = args.data_dir / f"A0{subj}E.gdf"
        label_e = args.data_dir / "true_labels" / f"A0{subj}E.mat"
        if gdf_e.exists() and label_e.exists():
            print(f"[INFO] Subject {subj} — eval …")
            X_e, y_e = _load_eval(gdf_e, label_e)
            X_parts.append(X_e)
            y_parts.append(y_e)
        else:
            print(f"[WARN] Subject {subj} — eval files missing, skipping.")

    X_all = np.concatenate(X_parts, axis=0)
    y_all = np.concatenate(y_parts)
    N, M, T = X_all.shape
    print(f"[INFO] Combined: {N} trials  M={M}  T={T}")

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
