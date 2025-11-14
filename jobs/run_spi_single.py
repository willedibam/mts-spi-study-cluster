from __future__ import annotations

import argparse
from typing import List

from src import run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single dataset spec without PBS."
    )
    parser.add_argument("--mode", default="dev")
    parser.add_argument("--job-index", type=int, default=1)
    parser.add_argument("--experiment-config")
    parser.add_argument("--pyspi-config")
    parser.add_argument("--pyspi-subset")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--normalise", type=int, choices=[0, 1])
    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--regenerate-data", action="store_true")
    parser.add_argument("--parquet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forwarded: List[str] = [
        "--mode",
        args.mode,
        "--job-index",
        str(args.job_index),
    ]
    if args.experiment_config:
        forwarded += ["--experiment-config", args.experiment_config]
    if args.pyspi_config:
        forwarded += ["--pyspi-config", args.pyspi_config]
    if args.pyspi_subset:
        forwarded += ["--pyspi-subset", args.pyspi_subset]
    if args.threads:
        forwarded += ["--threads", str(args.threads)]
    if args.normalise is not None:
        forwarded += ["--normalise", str(args.normalise)]
    if args.heatmap:
        forwarded.append("--heatmap")
    if args.regenerate_data:
        forwarded.append("--regenerate-data")
    if args.parquet:
        forwarded.append("--parquet")
    run_experiments.main(forwarded)


if __name__ == "__main__":
    main()

