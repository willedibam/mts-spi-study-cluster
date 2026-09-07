"""Extract catalogue-matched controls from existing numeric MPI archives.

Example: python -m scripts.extract_mpi_representation_baselines --root ... --output ...
This is descriptive extraction, not a fitted or confirmatory benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.mpi_representation_baselines import (
    GRAPH_NAMES, MARGINAL_NAMES, pearson_geometry_audit, summarize_mpis,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    paths = sorted(args.root.glob("*/spi_mpis.npz"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error("no MPI archives found")
    output = args.output.with_suffix(".npz")
    report = output.with_suffix(".json")
    if output.exists() or report.exists():
        parser.error("output exists; select a new output path")
    started = time.monotonic()
    records, marginal, graph, masks, names, dimensions, durations = [], [], [], [], [], [], []
    spi_order = identity = None
    for i, path in enumerate(paths):
        meta_path = path.with_name("meta.json")
        meta = json.loads(meta_path.read_text())
        if meta["status"] != "complete":
            raise ValueError(f"incomplete record: {path}")
        current_identity = (meta["pyspi"]["config_sha256"],
                            json.dumps(meta["pyspi"]["version"], sort_keys=True),
                            meta["normalise"], meta.get("random_seed"))
        with np.load(path, allow_pickle=False) as archive:
            if spi_order is None:
                spi_order = list(archive.files)
                identity = current_identity
            if set(archive.files) != set(spi_order) or identity != current_identity:
                raise ValueError(f"catalogue/provenance mismatch: {path}")
            mpis = {name: archive[name] for name in spi_order}
        if mpis[spi_order[0]].shape != (meta["M"], meta["M"]):
            raise ValueError(f"metadata dimension mismatch: {path}")
        m, g, valid = summarize_mpis(mpis, spi_order)
        marginal.append(m); graph.append(g); masks.append(valid)
        names.append(meta["dataset_name"]); dimensions.append(meta["M"]); durations.append(meta["T"])
        records.append({"dataset": names[-1], "path": str(path),
                        "mpi_sha256": digest(path), "meta_sha256": digest(meta_path),
                        "source": meta.get("source"), "M": meta["M"], "T": meta["T"],
                        "geometry": pearson_geometry_audit(mpis, spi_order)})
        if (i + 1) % 100 == 0:
            print(f"Extracted {i + 1}/{len(paths)}", flush=True)
    if len(set(names)) != len(names):
        raise ValueError("duplicate dataset identifiers")
    marginal, graph, masks = np.asarray(marginal), np.asarray(graph), np.asarray(masks)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X_marginal=marginal, X_graph=graph,
                        spi_correlation_valid=masks, dataset=np.asarray(names),
                        spi_names=np.asarray(spi_order), M=dimensions, T=durations,
                        marginal_names=np.asarray([f"{s}::{n}" for s in spi_order for n in MARGINAL_NAMES]),
                        graph_names=np.asarray([f"{s}::{n}" for s in spi_order for n in GRAPH_NAMES]))
    common = np.flatnonzero(masks.all(axis=0))
    summary = {"status": "descriptive_extraction_only", "rows": len(names),
               "spis": len(spi_order), "marginal_shape": list(marginal.shape),
               "graph_shape": list(graph.shape), "catalogue_identity": identity,
               "common_valid_spis_descriptive_only": [spi_order[j] for j in common],
               "complete_correlation_records": int(masks.all(axis=1).sum()),
               "numerically_singular_records": sum(r["geometry"]["numerical_rank"] < r["geometry"]["valid_spis"] for r in records),
               "rank_quantiles": np.quantile([r["geometry"]["numerical_rank"] for r in records], [0, .5, 1]).tolist(),
               "minimum_eigenvalue": min(r["geometry"]["minimum_eigenvalue"] for r in records if r["geometry"]["minimum_eigenvalue"] is not None),
               "elapsed_seconds": time.monotonic() - started,
               "artifact_sha256": digest(output),
               "builder_sha256": digest(Path(__file__)),
               "summary_module_sha256": digest(Path(__file__).parents[1] / "src/mpi_representation_baselines.py"),
               "records": records}
    report.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("records", "common_valid_spis_descriptive_only")}, indent=2))


if __name__ == "__main__":
    main()
