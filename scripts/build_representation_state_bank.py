"""Extract catalogue-matched features from all paired-view p90 outputs."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from src.mpi_representation_baselines import summarize_mpis
from src.representation_state_data import file_hash, load_state_data, observed_view
from src.run_external_corpus import _array_sha256
from src.spi_spi_contract import build_unified_features, schema_sha256
from src.utils import slugify


def build(config_path, data_root, mpi_root, output):
    if output.exists():
        raise FileExistsError(output)
    protocol = yaml.safe_load(config_path.read_text())
    manifest, masters = load_state_data(data_root, config_path)
    catalogue_hash = file_hash(Path(protocol["methods"]["catalogue"]))
    rows, order, sources, computations = manifest["rows"], None, [], set()
    banks = {name: [] for name in ("m", "g", "z", "validity")}
    start = time.perf_counter()
    for row in rows:
        root = mpi_root / f"{row['corpus_index']:04d}-{slugify(row['row_id'], 'dataset')}"
        meta_path, mpi_path = root / "meta.json", root / "spi_mpis.npz"
        meta = json.loads(meta_path.read_text())
        assert meta["status"] == "complete" and meta["normalise"] is False
        assert meta["dataset_name"] == row["row_id"]
        assert (meta["M"], meta["T"]) == (row["M"], row["T"])
        assert meta["pyspi"]["config_sha256"] == catalogue_hash
        assert meta["source"]["archive_sha256"] == manifest["artifacts"]["views.npz"]
        raw = observed_view(masters[row["master_index"]], row["M"], row["T"])
        assert meta["source"]["member_sha256"] == _array_sha256(raw)
        computations.add(json.dumps(meta["pyspi"]["version"], sort_keys=True))
        names = [item["name"] for item in meta["pyspi"]["spis"]]
        if order is None:
            order = names
        assert names == order and len(order) == 289
        with np.load(mpi_path, allow_pickle=False) as archive:
            assert archive.files == order
            mpis = {name: archive[name] for name in order}
        assert all(a.shape == (row["M"], row["M"]) for a in mpis.values())
        m, g, valid = summarize_mpis(mpis, order)
        features = build_unified_features(mpis, order, metric="pearson")
        for name, values in zip(banks, (m, g, features.z, valid), strict=True):
            banks[name].append(values)
        sources.append({"row_id": row["row_id"], "mpi_sha256": file_hash(mpi_path),
                        "meta_sha256": file_hash(meta_path), "compute_seconds": meta["job"]["compute_seconds"],
                        "errors": meta["pyspi"]["errors"], "valid_spis": int(valid.sum())})
        if len(sources) % 100 == 0:
            print(f"Built {len(sources)}/{len(rows)}", flush=True)
    assert len(computations) == 1
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **{f"X_{key}": np.asarray(value, dtype=float) for key, value in banks.items()},
                        row_id=np.asarray([r["row_id"] for r in rows]), spi_order=np.asarray(order),
                        schema_sha256=schema_sha256(features.schema), feature_contract="unified_ordered_v3",
                        manifest_sha256=file_hash(data_root / "manifest.json"))
    report = {"artifact_sha256": file_hash(output), "protocol_sha256": file_hash(config_path),
              "builder_sha256": file_hash(Path(__file__)), "catalogue_sha256": catalogue_hash,
              "modules": {p: file_hash(Path(p)) for p in ("src/mpi_representation_baselines.py", "src/spi_spi_contract.py")},
              "pyspi_versions": list(computations), "sources": sources,
              "extraction_seconds": time.perf_counter() - start,
              "summed_pyspi_seconds": sum(s["compute_seconds"] for s in sources),
              "manifest_sha256": file_hash(data_root / "manifest.json")}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "seconds": report["extraction_seconds"], "sha256": report["artifact_sha256"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--mpi-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build(args.config, args.data, args.mpi_root, args.output)
