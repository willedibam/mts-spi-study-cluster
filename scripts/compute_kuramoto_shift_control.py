#!/usr/bin/env python3
"""Compute p90 SPIs after independently circular-shifting observed channels."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compute import run_pyspi  # noqa: E402
from src.mapping import DatasetMapping, ExperimentConfig  # noqa: E402
from src.run_experiments import _file_sha256, _pyspi_version  # noqa: E402
from src.utils import dump_json, ensure_dir, load_json  # noqa: E402


def selected_sources(config_path: Path):
    mapping = DatasetMapping(ExperimentConfig.from_file(config_path))
    selected = [
        spec
        for spec in mapping.specs
        if spec.generator_params["frequency_sampling"] == "random"
        and "paired" in spec.class_labels
        and spec.instance < 8
    ]
    if len(selected) != 192:
        raise RuntimeError(f"expected 192 preselected shift controls, found {len(selected)}")
    return selected


def output_dir(source, root: Path) -> Path:
    return root / source.mts_class / source.dataset_slug


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=ROOT / "configs/generate/order_parameter/kuramoto-confirmation.yaml",
    )
    parser.add_argument(
        "--pyspi-config",
        type=Path,
        default=ROOT / "configs/pyspi/benchmarked_p90.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/order_parameter/kuramoto_confirmation_shifted",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    sources = selected_sources(args.experiment_config)
    if not 1 <= args.job_index <= len(sources):
        raise ValueError(f"job index must lie in [1,{len(sources)}]")
    source = sources[args.job_index - 1]
    source_meta_path = source.dataset_dir / "meta.json"
    source_spi_path = source.dataset_dir / "spi_mpis.npz"
    if not source_meta_path.exists() or not source_spi_path.exists():
        raise RuntimeError(f"source confirmation dataset is incomplete: {source.dataset_dir}")
    destination = ensure_dir(output_dir(source, args.output_dir))
    if args.skip_existing and (destination / "meta.json").exists() and (destination / "spi_mpis.npz").exists():
        return 0

    data = np.load(source.dataset_dir / "timeseries.npy").astype(np.float64)
    seed_payload = f"kuramoto-shift-v1|{source.rng_seed}".encode()
    seed = int.from_bytes(hashlib.blake2s(seed_payload, digest_size=8).digest(), "big")
    rng = np.random.default_rng(seed)
    offsets = rng.integers(1, data.shape[0], size=data.shape[1])
    shifted = np.column_stack(
        [np.roll(data[:, channel], int(offsets[channel])) for channel in range(data.shape[1])]
    )
    np.save(destination / "timeseries.npy", shifted.astype(np.float32))
    np.save(destination / "channel_offsets.npy", offsets.astype(np.int32))

    start = time.perf_counter()
    result = run_pyspi(
        shifted,
        config_path=args.pyspi_config,
        normalise=False,
        n_jobs=1,
    )
    elapsed = time.perf_counter() - start
    np.savez_compressed(destination / "spi_mpis.npz", **result.matrices)
    source_meta = load_json(source_meta_path)
    dump_json(
        destination / "meta.json",
        {
            "transform": "independent nonzero circular shift per observed channel",
            "transform_seed": seed,
            "source_path": str(source.dataset_dir),
            "source_class_name": source.mts_class,
            "source_instance": source.instance,
            "source_kappa": float(source_meta["generator"]["control"]["reduced_value"]),
            "source_distribution": source.generator_params["frequency_distribution"],
            "source_seed_group_id": source.seed_group_id,
            "outcomes_copied_or_read": False,
            "M": int(data.shape[1]),
            "T": int(data.shape[0]),
            "job": {"compute_seconds": elapsed},
            "pyspi": {
                "config": str(args.pyspi_config),
                "config_sha256": _file_sha256(args.pyspi_config),
                "version": _pyspi_version(),
                "n_spis": len(result.metadata),
                "errors": result.errors or {},
                "spis": [
                    {
                        "name": info.name,
                        "directed": info.directed,
                        "labels": info.labels,
                        "family": info.family,
                        "module": info.module,
                        "class_name": info.class_name,
                    }
                    for info in result.metadata
                ],
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
