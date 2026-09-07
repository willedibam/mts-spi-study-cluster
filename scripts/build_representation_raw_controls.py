"""Reuse the existing 82-feature raw statistical baseline for the Stage A rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.cross_mt_transfer import pooled_baseline_features


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(manifest_path: Path, mirror: Path, raw_evaluation: Path, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    values, rows, sources = [], [], []
    names = None
    for record in manifest["records"]:
        if record["role"] == "training_pool":
            path = mirror / record["dataset_path"].lstrip("/") / "timeseries.npy"
        else:
            path = raw_evaluation / record["label"] / Path(record["dataset_path"]).name / "timeseries.npy"
        raw = np.load(path, allow_pickle=False)
        if raw.shape != (record["T"], record["M"]) or not np.isfinite(raw).all():
            raise ValueError(f"invalid raw record: {path} {raw.shape}")
        features, current_names = pooled_baseline_features(raw)["pooled_combined"]
        if names is not None and names != current_names:
            raise ValueError("raw feature schema drift")
        names = current_names
        values.append(features); rows.append(record["row_id"])
        sources.append({"row_id": record["row_id"], "path": str(path), "sha256": sha(path)})
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X_u=np.asarray(values), row_id=np.asarray(rows), feature_names=np.asarray(names))
    output.with_suffix(".json").write_text(json.dumps({"artifact_sha256": sha(output),
        "manifest_sha256": sha(manifest_path), "rows": len(rows), "features": len(names),
        "builder_sha256": sha(Path(__file__)), "baseline_module_sha256": sha(Path("src/cross_mt_transfer.py")),
        "sources": sources}, indent=2) + "\n")
    print(f"Built {len(rows)} x {len(names)} raw controls")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--mirror", type=Path, required=True)
    p.add_argument("--raw-evaluation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    build(a.manifest, a.mirror, a.raw_evaluation, a.output)
