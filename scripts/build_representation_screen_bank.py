"""Prepare a hash-bound MPI transfer manifest, then build a unified Stage A bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.mpi_representation_baselines import MARGINAL_NAMES, GRAPH_NAMES, summarize_mpis
from src.spi_spi_contract import build_unified_features, schema_sha256


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config: Path, feature_dir: Path, output_dir: Path) -> None:
    protocol = yaml.safe_load(config.read_text())
    records, banks = [], []
    spi_order = None
    for tag in ("development-base", "development-cml", "confirmation"):
        path = feature_dir / f"{tag}.npz"
        banks.append({"path": str(path), "sha256": sha(path)})
        with np.load(path, allow_pickle=True) as bank:
            order = bank["spi_order"].astype(str).tolist()
            if spi_order is not None and order != spi_order:
                raise ValueError("SPI orders differ")
            spi_order = order
            sources = json.loads(str(bank["source_manifest_json"].item()))["entries"]
            # Source manifests are keyed by directory; do not assume list order.
            by_suffix = {"/".join(x["dataset_path"].split("/")[-2:]): x for x in sources}
            if len(by_suffix) != len(sources):
                raise ValueError("ambiguous source directory suffix")
            for i, (label, m, t, instance) in enumerate(zip(bank["y"], bank["M"], bank["T"], bank["instance"], strict=True)):
                m, t, instance = int(m), int(t), int(instance)
                role = "evaluation" if tag == "confirmation" else "training_pool"
                if role == "training_pool" and (m != protocol["source_cell"]["M"] or t != protocol["source_cell"]["T"]):
                    continue
                allowed = protocol["evaluation_instances"] if role == "evaluation" else protocol["training_instances"]
                if instance not in allowed or m not in protocol["M_values"] or t not in protocol["T_values"]:
                    raise ValueError("unexpected observation cell or instance")
                suffix = "/".join(str(bank["dataset_paths"][i]).split("/")[-2:])
                source = by_suffix[suffix]
                records.append({"label": str(label), "M": m, "T": t, "instance": instance,
                                "role": role, "group": f"{label}|I{instance}",
                                "row_id": f"{label}|M{m}|T{t}|I{instance}", **source})
    ids = [r["row_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": protocol, "protocol_sha256": sha(config), "source_banks": banks,
                "spi_order": spi_order, "records": records}
    (output_dir / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    files = [f"{r['dataset_path'].lstrip('/')}/{file}" for r in records for file in ("spi_mpis.npz", "meta.json")]
    (output_dir / "transfer-files.txt").write_text("\n".join(files) + "\n")
    print(json.dumps({"rows": len(records), "files": len(files), "classes": len(set(r['label'] for r in records))}))


def build(manifest_path: Path, mirror_root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads(manifest_path.read_text())
    protocol, order = manifest["protocol"], manifest["spi_order"]
    marginal, graph, z, valid, normalization = [], [], [], [], set()
    for i, record in enumerate(manifest["records"]):
        root = mirror_root / record["dataset_path"].lstrip("/")
        mpi_path, meta_path = root / "spi_mpis.npz", root / "meta.json"
        if sha(mpi_path) != record["mpi_sha256"] or sha(meta_path) != record["meta_sha256"]:
            raise ValueError(f"source hash mismatch: {root}")
        meta = json.loads(meta_path.read_text())
        # Legacy generated proof records have no status key; their exact hashes
        # are bound by the audited historical feature banks.
        if meta.get("status") not in (None, "complete"):
            raise ValueError(f"incomplete source: {root}")
        if meta["M"] != record["M"] or meta["T"] != record["T"]:
            raise ValueError(f"source dimension metadata mismatch: {root}")
        if meta["pyspi"]["config_sha256"] != protocol["catalogue_sha256"]:
            raise ValueError("wrong p90 config")
        if meta["pyspi"]["version"]["computation"] != protocol["pyspi_computation"]:
            raise ValueError("wrong pyspi computation")
        normalization.add(json.dumps(meta.get("normalise"), sort_keys=True))
        with np.load(mpi_path, allow_pickle=False) as archive:
            if set(archive.files) != set(order):
                raise ValueError("wrong SPI set")
            mpis = {name: archive[name] for name in order}
        if mpis[order[0]].shape != (record["M"], record["M"]):
            raise ValueError(f"MPI shape mismatch: {root}")
        m, g, mask = summarize_mpis(mpis, order)
        unified = build_unified_features(mpis, order, metric=protocol["metric"])
        marginal.append(m); graph.append(g); valid.append(mask); z.append(unified.z)
        if (i + 1) % 200 == 0:
            print(f"Built {i + 1}/{len(manifest['records'])}", flush=True)
    if len(normalization) != 1:
        raise ValueError("normalization differs across source banks")
    rows = manifest["records"]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X_m=np.asarray(marginal), X_g=np.asarray(graph),
                        X_z=np.asarray(z), X_validity=np.asarray(valid, dtype=float),
                        **{key: np.asarray([r[key] for r in rows]) for key in ("label", "M", "T", "instance", "role", "group", "row_id")},
                        spi_order=np.asarray(order),
                        marginal_names=np.asarray([f"{s}::{n}" for s in order for n in MARGINAL_NAMES]),
                        graph_names=np.asarray([f"{s}::{n}" for s in order for n in GRAPH_NAMES]),
                        schema_sha256=schema_sha256(unified.schema),
                        manifest_sha256=sha(manifest_path), feature_contract=protocol["feature_contract"])
    report = {"rows": len(rows), "catalogue_spis": len(order), "normalization": list(normalization),
              "protocol_sha256": manifest["protocol_sha256"],
              "source_hashes_verified": True, "artifact_sha256": sha(output),
              "manifest_sha256": sha(manifest_path), "builder_sha256": sha(Path(__file__)),
              "modules": {name: sha(Path(name)) for name in ("src/mpi_representation_baselines.py", "src/spi_spi_contract.py")}}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("build")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--mirror-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config, args.feature_dir, args.output_dir)
    else:
        build(args.manifest, args.mirror_root, args.output)


if __name__ == "__main__":
    main()
