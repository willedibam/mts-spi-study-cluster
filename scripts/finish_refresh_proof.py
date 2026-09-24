"""Audit the complete seeded bank, reconstruct features and execute the notebook."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
from nbclient import NotebookClient

from src.mapping import DatasetMapping, ExperimentConfig
from src.run_experiments import _file_sha256
from src.utils import project_root
from scripts.replay_proof_stochastic import POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root = project_root()
    mapping = DatasetMapping(ExperimentConfig.from_file(root / "configs/generate/embeddings/proof-p90-260924.yaml"))
    expected_hash = _file_sha256(root / "configs/pyspi/benchmarked_p90.yaml")
    errors = Counter()
    rows = []
    assert len(mapping.specs) == 1080
    for spec in mapping.specs:
        directory = Path(spec.dataset_dir)
        meta = json.loads((directory / "meta.json").read_text())
        replay = meta["pyspi"]["stochastic_replay"]
        assert meta["pyspi"]["config_sha256"] == expected_hash
        assert meta["pyspi"]["n_spis"] == 289
        assert meta["pyspi"]["version"]["computation"] == "3.0.0.r7"
        assert (meta["mts_class"], meta["M"], meta["T"], meta["instance_index"]) == (spec.mts_class, spec.M, spec.T, spec.instance)
        assert replay["policy"] == POLICY and replay["seed"] == spec.rng_seed
        assert replay["input_sha256"] == _file_sha256(directory / "timeseries.npy")
        assert replay["repaired_archive_sha256"] == _file_sha256(directory / "spi_mpis.npz")
        errors.update(meta["pyspi"]["errors"].keys())
        rows.append({"class": spec.mts_class, "M": spec.M, "T": spec.T, "instance": spec.instance,
                     "archive_sha256": replay["repaired_archive_sha256"]})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "extraction-audit.json").write_text(json.dumps({
        "datasets": len(rows), "spi_config_sha256": expected_hash,
        "validity_refusals_by_spi": dict(errors), "stochastic_policy": POLICY,
        "rows": rows,
    }, indent=2) + "\n")
    subprocess.run([sys.executable, "-m", "src.process_features", "--data-path", str(mapping.specs[0].base_output_dir),
                    "--output", str(args.bank), "--feature-contract", "direction_preserving_v2",
                    "--metric", "pearson", "--workers", str(args.workers)], check=True, cwd=root)
    os.environ["PROOF_REFRESH_BANK"] = str(args.bank)
    os.environ["PROOF_REFRESH_OUTPUT"] = str(args.output)
    # Install a temporary kernelspec pointing at this exact scientific env.
    kernel_root = args.output / "jupyter" / "kernels" / "proof-refresh"
    kernel_root.mkdir(parents=True, exist_ok=True)
    (kernel_root / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Proof refresh", "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = str(args.output / "jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    notebook = nbformat.read(root / "notebooks/embeddings/proof_p90_260924.ipynb", as_version=4)
    NotebookClient(notebook, timeout=1200, kernel_name="proof-refresh", resources={"metadata": {"path": str(root)}}).execute()
    nbformat.write(notebook, args.output / "proof_p90_260924.ipynb")
    print(f"[COMPLETE] audited 1080 datasets and executed {args.output / 'proof_p90_260924.ipynb'}")


if __name__ == "__main__":
    main()
