"""Audit a proof corpus and build its feature bank (`<run>/features.npz`).

Defaults are the current-p90 rerun of 2026-10-14 (283 SPIs, computation r9), whose bank is
analysed in `proof_p90_261021` and `proof_figure_261029`. The 2026-11-26 rerun on pyspi
3.0.0rc2: `--config research/proof/p90/configs/proof-p90-261126.yaml --spis 207 --computation 3.0.0`."""
import argparse
from collections import Counter
import json
import subprocess
import sys

from mtsspi.run_corpus import load_corpus, audit_outputs
from mtsspi.utils import project_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="research/proof/p90/configs/proof-p90-261014.yaml")
    parser.add_argument("--spis", type=int, default=283)
    parser.add_argument("--computation", default="3.0.0.r9")
    args = parser.parse_args()
    root = project_root()
    corpus = load_corpus(root / args.config)
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
        assert meta["pyspi"]["n_spis"] == args.spis
        assert meta["pyspi"]["version"]["computation"] == args.computation
        assert meta["job"]["estimator_rng_policy"] == "pyspi_fixed_seeds"
        errors.update(meta["pyspi"]["errors"].keys())
    audit["validity_refusals_by_spi"] = dict(errors)
    (output / "extraction-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    bank = destination / "features.npz"
    subprocess.run([sys.executable, "-m", "mtsspi.process_features", "--data-path", str(corpus.base_output_dir),
                    "--output", str(bank), "--feature-contract", "direction_preserving_v2",
                    "--metric", "pearson", "--workers", "16"], check=True, cwd=root)
    print(f"[COMPLETE] {bank}")


if __name__ == "__main__":
    main()
