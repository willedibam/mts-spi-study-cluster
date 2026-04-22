"""
Dump pyspi-internal SPI names produced by sonnet_fast_config.yaml.

Instantiates a pyspi Calculator on a dummy (M=5, T=500) dataset, reads the
SPI identifiers pyspi generates from (class + config kwargs), and writes
them one per line to features/spis/sonnet_fast_spis.txt.

Use with process_features.py:

    python -m src.process_features \\
        --data-path data/embeddings/proof_260420/ \\
        --spi-subset features/spis/sonnet_fast_spis.txt \\
        --metric spearman,pearson \\
        --output-dir features/
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pyspi.calculator import Calculator

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs/pyspi-v2/sonnet_fast_config.yaml"
OUT = ROOT / "features/spis/sonnet_fast_spis.txt"


def main() -> None:
    rng = np.random.default_rng(0)
    dummy = rng.standard_normal((5, 500))  # pyspi expects (M, T) channels-first
    calc = Calculator(
        dataset=dummy,
        subset="default",
        configfile=str(CONFIG),
        normalise=False,
    )
    names = list(calc.spis.keys())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(names) + "\n")
    print(f"Wrote {len(names)} SPI names -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
