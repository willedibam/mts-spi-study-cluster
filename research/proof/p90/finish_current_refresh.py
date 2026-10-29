"""Audit and reconstruct the feature bank of the current-p90 proof rerun of 2026-10-14.

It also executed the 2026-10-14 notebook, since deleted; the bank is analysed in
`proof_p90_261021` and `proof_figure_261029`."""
from collections import Counter
import json
import subprocess
import sys

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
    print(f"[COMPLETE] {bank}")


if __name__ == "__main__":
    main()
