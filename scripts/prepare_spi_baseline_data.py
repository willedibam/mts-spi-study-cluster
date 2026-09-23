"""Inventory existing proof MPIs and request only missing cached input files.

Run locally; the emitted rsync file list is relative to the remote filesystem
root. This script never computes an SPI or changes a remote file.
"""
from pathlib import Path
import json
import numpy as np

from scripts.gadi_storage_path import resolve_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/spi_baseline_exploration_260921"


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)
    mirror = ROOT / "data/representation_stage_a_260907/mpi-mirror"
    downloads = DATA / "downloads"
    path_map = json.loads((ROOT / 'docs/operations/gadi-storage-path-map-260923.json').read_text())
    records, requests, sources = [], set(), []
    order = None
    for tag in ("development-base", "development-cml", "confirmation"):
        path = ROOT / f"data/proof_p90_260824/features/{tag}.npz"
        with np.load(path, allow_pickle=True) as bank:
            current = bank["spi_order"].astype(str).tolist()
            assert order is None or current == order
            order = current
            entries = json.loads(str(bank["source_manifest_json"].item()))["entries"]
            lookup = {"/".join(r["dataset_path"].split("/")[-2:]): r for r in entries}
            assert len(lookup) == len(entries)
            for i, name in enumerate(bank["dataset_paths"].astype(str)):
                suffix = "/".join(name.split("/")[-2:])
                original = lookup[suffix]
                historical = original["dataset_path"].lstrip("/")
                remote = resolve_path(original["dataset_path"], path_map).lstrip("/")
                candidates = [mirror / historical, downloads / historical, downloads / remote]
                folder = next((p for p in candidates if (p / "spi_mpis.npz").exists()), downloads / remote)
                if not (folder / "spi_mpis.npz").exists():
                    requests.update(f"{remote}/{f}" for f in ("spi_mpis.npz", "meta.json"))
                if tag == "confirmation":
                    raw = ROOT / "data/proof_p90_260824/raw/confirmation" / suffix / "timeseries.npy"
                    assert raw.exists()
                else:
                    old_raw = downloads / historical / "timeseries.npy"
                    raw = old_raw if old_raw.exists() else downloads / remote / "timeseries.npy"
                    if not raw.exists():
                        requests.add(f"{remote}/timeseries.npy")
                label, m, t, instance = str(bank["y"][i]), int(bank["M"][i]), int(bank["T"][i]), int(bank["instance"][i])
                records.append(dict(label=label, M=m, T=t, instance=instance,
                    row_id=f"{label}|M{m}|T{t}|I{instance}", bank=tag, bank_index=i,
                    role="evaluation" if tag == "confirmation" else "development",
                    mpi_path=str((folder / "spi_mpis.npz").relative_to(ROOT)),
                    meta_path=str((folder / "meta.json").relative_to(ROOT)),
                    raw_path=str(raw.relative_to(ROOT)), **original))
        sources.append(str(path.relative_to(ROOT)))
    assert len(records) == 3780 and len({r["row_id"] for r in records}) == 3780
    manifest = dict(spi_order=order, feature_banks=sources, records=records)
    (DATA / "proof-inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    missing = [f for f in sorted(requests) if not (downloads / f).exists()]
    (DATA / "proof-transfer-files.txt").write_text("\n".join(missing) + "\n")
    downloads.mkdir(exist_ok=True)
    print(f"{len(records)} proof rows; {len(missing)} cached files to retrieve")


if __name__ == "__main__":
    prepare()
