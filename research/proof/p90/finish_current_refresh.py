"""Audit, reconstruct and render the current-p90 proof rerun of 2026-10-14."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat
from nbclient import NotebookClient

from mtsspi.run_corpus import load_corpus, audit_outputs
from mtsspi.utils import project_root


def main():
    root = project_root()
    corpus = load_corpus(root / "research/proof/p90/configs/proof-p90-261014.yaml")
    entries = corpus.entries()
    audit = audit_outputs(corpus, entries)
    if audit["complete"] != 1080 or audit["failures"]:
        raise RuntimeError(audit)
    destination = corpus.base_output_dir.parent
    output = destination / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    errors = Counter()
    for entry in entries:
        meta = json.loads((entry.output_dir(corpus) / "meta.json").read_text())
        assert meta["pyspi"]["n_spis"] == 283
        assert meta["pyspi"]["version"]["computation"] == "3.0.0.r9"
        assert meta["job"]["estimator_rng_policy"] == "pyspi_fixed_seeds"
        errors.update(meta["pyspi"]["errors"].keys())
    audit["validity_refusals_by_spi"] = dict(errors)
    (output / "extraction-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    bank = destination / "features.npz"
    subprocess.run([sys.executable, "-m", "mtsspi.process_features", "--data-path", str(corpus.base_output_dir),
                    "--output", str(bank), "--feature-contract", "direction_preserving_v2",
                    "--metric", "pearson", "--workers", "16"], check=True, cwd=root)
    os.environ.update(PROOF_REFRESH_BANK=str(bank), PROOF_REFRESH_OUTPUT=str(output))
    kernel_root = output / "jupyter" / "kernels" / "proof-refresh"
    kernel_root.mkdir(parents=True, exist_ok=True)
    (kernel_root / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Proof refresh", "language": "python",
    }))
    os.environ["JUPYTER_PATH"] = str(output / "jupyter") + os.pathsep + os.environ.get("JUPYTER_PATH", "")
    notebook = nbformat.read(root / "research/proof/p90/notebooks/proof_p90_261014.ipynb", as_version=4)
    NotebookClient(notebook, timeout=1200, kernel_name="proof-refresh",
                   resources={"metadata": {"path": str(root)}}).execute()
    nbformat.write(notebook, output / "proof_p90_261014.ipynb")
    print(f"[COMPLETE] {output / 'proof_p90_261014.ipynb'}")


if __name__ == "__main__":
    main()
