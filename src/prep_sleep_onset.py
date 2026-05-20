"""
Sleep-onset EEG preprocessor.

Loads a per-subject EEG tensor produced by the Will_Sleep_Onset repo
(``{SUBJ}_tensor_{TENSOR}.npz``) and writes one timeseries.npy per epoch into
the directory layout expected by run_experiments.py real-data configs.

Each epoch (5 s @ 100 Hz) becomes a single pyspi dataset: data[e] is
(M, T=500) channels-first; it is transposed to (T=500, M) on disk.
Channels are already z-scored per channel over the full recording, so NO
re-normalisation is applied here (config must set ``zscore: false``).

Run once per tensor variant, from the project root:

    python -m src.prep_sleep_onset --subject EPCTL01 --tensor M38
    python -m src.prep_sleep_onset --subject EPCTL01 --tensor EEGonly_zscore \\
        --label epctl01-m83

Output layout (mirrors run_experiments.py real-data paths):

    <output-dir>/<label>/class<label>_I{n}/timeseries.npy   (n = 0 .. E-1)
    <output-dir>/<label>/manifest.csv        per-epoch labels (instance->stage)
    <output-dir>/<label>/dataset_meta.json   channel names + dims + onset

The two label files retain everything from the source EEG that the SPI
pipeline itself does not carry: manifest.csv maps each instance (= PBS array
index - 1) to its hypnogram stage and the onset flag, and dataset_meta.json
records the channel-name order and recording dimensions. Join either against
the per-dataset meta.json / calc.csv outputs on the instance index.
Existing timeseries.npy files are skipped (safe to re-run after a partial run).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .utils import class_dir_name, project_root, slugify

# Hypnogram code -> stage name (per Will_Sleep_Onset tensor spec).
_STAGE_NAMES = {1: "Wake", -1: "N1", -2: "N2", -3: "N3", 0: "REM", 8: "Unscored"}


def _load_tensor(npz_path: Path):
    if not npz_path.exists():
        raise SystemExit(f"Tensor not found: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    data = z["data"]  # (E, M, T) float64
    if data.ndim != 3:
        raise SystemExit(f"Expected 3D (E, M, T) data, got shape {data.shape}.")
    hypnogram = z["hypnogram"]
    channel_names = [str(c) for c in z["channel_names"]]
    onset_epoch = int(np.asarray(z["onset_epoch"]).ravel()[0])
    E, M, T = data.shape
    if len(channel_names) != M:
        raise SystemExit(
            f"channel_names length {len(channel_names)} != M={M} in {npz_path.name}."
        )
    print(f"[INFO] Loaded {npz_path.name}: E={E} epochs  M={M} channels  T={T} samples")
    print(f"[INFO] onset_epoch={onset_epoch}")
    return data, hypnogram, channel_names, onset_epoch


def _write_epochs(
    data, hypnogram, channel_names, onset_epoch, *,
    subject: str, tensor: str, label: str, out_root: Path,
) -> None:
    E, M, T = data.shape
    class_dir = out_root / class_dir_name(label)
    written = skipped = 0
    rows = []
    for instance in range(E):
        dest = class_dir / f"class{label}_I{instance}"
        ts_path = dest / "timeseries.npy"
        stage_code = int(hypnogram[instance])
        rows.append(
            {
                "instance": instance,
                "epoch": instance,
                "hypnogram": stage_code,
                "stage": _STAGE_NAMES.get(stage_code, "Unknown"),
                "is_onset_epoch": int(instance == onset_epoch),
            }
        )
        if ts_path.exists():
            skipped += 1
            continue
        series = np.ascontiguousarray(data[instance].T, dtype=np.float32)  # (T, M)
        dest.mkdir(parents=True, exist_ok=True)
        np.save(ts_path, series)
        written += 1

    manifest_path = class_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    meta_path = class_dir / "dataset_meta.json"
    meta = {
        "subject": subject,
        "tensor": tensor,
        "label": label,
        "M": M,
        "T": T,
        "n_epochs": E,
        "fs_hz": 100,
        "epoch_seconds": T / 100,
        "onset_epoch": onset_epoch,
        "channel_names": channel_names,
        "stage_legend": _STAGE_NAMES,
    }
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[INFO] {class_dir}: written {written}, skipped {skipped} (total {E} epochs)")
    print(f"[INFO] Wrote per-epoch labels    -> {manifest_path}")
    print(f"[INFO] Wrote channel/dim metadata -> {meta_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        default="EPCTL01",
        help="Subject id, e.g. EPCTL01 (default: EPCTL01).",
    )
    parser.add_argument(
        "--tensor",
        default="M38",
        help="Tensor variant token in {SUBJ}_tensor_{TENSOR}.npz, "
        "e.g. M38 (38ch coarse) or EEGonly_zscore (83ch full). Default: M38.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Class slug for the output directory layout and the matching "
        "experiment-config class name. Default: slugified subject.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=project_root().parent / "Will_Sleep_Onset" / "processed_data",
        help="processed_data root of the Will_Sleep_Onset repo.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root() / "data" / "sleep_onset",
        help="Root output directory (default: data/sleep_onset).",
    )
    args = parser.parse_args()

    label = slugify(args.label) if args.label else slugify(args.subject)
    npz_path = args.input_root / args.subject / f"{args.subject}_tensor_{args.tensor}.npz"
    data, hypnogram, channel_names, onset_epoch = _load_tensor(npz_path)
    _write_epochs(
        data, hypnogram, channel_names, onset_epoch,
        subject=args.subject, tensor=args.tensor, label=label,
        out_root=args.output_dir,
    )
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
