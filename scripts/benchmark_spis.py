#!/usr/bin/env python
"""
Benchmark SPI computation times across an (M, T) grid.

Usage:
    python scripts/benchmark_spis.py --pyspi-config configs/pyspi-v2/blended_config.yaml
    python scripts/benchmark_spis.py --pyspi-config configs/pyspi/fast_config.yaml --output data/benchmark/fast_timings.json
    python scripts/benchmark_spis.py --pyspi-config configs/pyspi-v2/blended_config.yaml --family basic
    python scripts/benchmark_spis.py --resume  # resume from existing output file

Results are saved incrementally after each (M, T) combo, so the run is
interrupt-safe. Use --resume to skip already-computed combos.

Output: a JSON file with structure:
    {
        "pyspi_config": "...",
        "family_filter": null,
        "grid": {"M_values": [...], "T_values": [...]},
        "results": [
            {"M": 3, "T": 100, "total_seconds": 1.23, "timings": {"SPI_name": 0.01, ...}},
            ...
        ]
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# macOS: prevent crash when multiple OpenMP runtimes (numba, MKL, tslearn) are loaded
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

# Allow running as `python scripts/benchmark_spis.py` from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generators.linear import generate_varma
from src.compute import run_pyspi


DEFAULT_M = [3, 5, 10, 16, 20, 32, 64]
DEFAULT_T = [100, 500, 1000, 2000, 5000]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "benchmark" / "timings.json"

# Map config section prefixes to family names
FAMILY_SECTIONS = {
    "basic": ".statistics.basic",
    "distance": ".statistics.distance",
    "causal": ".statistics.causal",
    "infotheory": ".statistics.infotheory",
    "spectral": ".statistics.spectral",
    "wavelet": ".statistics.wavelet",
    "misc": ".statistics.misc",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pyspi-config", type=str, default="configs/pyspi-v2/blended_config.yaml",
                    help="PySPI config YAML to benchmark.")
    p.add_argument("--M", type=int, nargs="+", default=DEFAULT_M,
                    help="Channel counts to benchmark.")
    p.add_argument("--T", type=int, nargs="+", default=DEFAULT_T,
                    help="Time lengths to benchmark.")
    p.add_argument("--family", type=str, default=None,
                    choices=list(FAMILY_SECTIONS.keys()),
                    help="Only benchmark SPIs from this family.")
    p.add_argument("--output", type=str, default=None,
                    help="Output JSON path (default: data/benchmark/timings.json).")
    p.add_argument("--resume", action="store_true",
                    help="Skip (M, T) combos already present in the output file.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for data generation.")
    return p.parse_args(argv)


def _resolve_pyspi_config(config_str: str, family: str | None) -> Path:
    """If family is specified, create a temp config with only that family's SPIs."""
    config_path = PROJECT_ROOT / config_str
    if family is None:
        return config_path

    import yaml
    with open(config_path) as f:
        full_cfg = yaml.safe_load(f)

    section_prefix = FAMILY_SECTIONS[family]
    filtered = {k: v for k, v in full_cfg.items() if k.startswith(section_prefix)}
    if not filtered:
        raise ValueError(f"No SPIs found for family '{family}' (section prefix '{section_prefix}') in {config_path}")

    # Write filtered config to a temp file next to the output
    tmp_path = config_path.parent / f"_benchmark_{family}.yaml"
    with open(tmp_path, "w") as f:
        yaml.dump(filtered, f, default_flow_style=False, sort_keys=False)
    return tmp_path


def _generate_var1(M: int, T: int, seed: int) -> np.ndarray:
    """Generate a VAR(1) process for benchmarking."""
    return generate_varma(
        M=M, T=T,
        phi=0.6, coupling=0.3,
        ma_phi=0.0, ma_coupling=0.0,
        noise_std=0.2, transients=100,
        topology="ring-symmetric",
        rng=np.random.default_rng(seed),
        zscore=True,
    )


def _load_existing(output_path: Path) -> dict:
    if output_path.exists():
        with open(output_path) as f:
            return json.load(f)
    return None


def _save(output_path: Path, data: dict):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(output_path)


def _completed_combos(data: dict) -> set[tuple[int, int]]:
    if data is None:
        return set()
    return {(r["M"], r["T"]) for r in data.get("results", [])}


def main(argv=None):
    args = parse_args(argv)

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT
    if args.family and not args.output:
        output_path = output_path.with_name(f"timings_{args.family}.json")

    pyspi_config = _resolve_pyspi_config(args.pyspi_config, args.family)

    existing = _load_existing(output_path) if args.resume else None
    completed = _completed_combos(existing)

    output_data = existing or {
        "pyspi_config": args.pyspi_config,
        "family_filter": args.family,
        "grid": {"M_values": args.M, "T_values": args.T},
        "results": [],
    }

    combos = [(M, T) for M in args.M for T in args.T]
    total = len(combos)
    skipped = 0

    for i, (M, T) in enumerate(combos, 1):
        if (M, T) in completed:
            skipped += 1
            print(f"[{i}/{total}] M={M}, T={T} — skipped (already computed)")
            continue

        print(f"[{i}/{total}] M={M}, T={T} — generating data...", end=" ", flush=True)
        data = _generate_var1(M, T, args.seed)
        data = data.astype(np.float64, copy=False)

        print("computing SPIs...", flush=True)
        t0 = time.perf_counter()
        try:
            result = run_pyspi(
                data,
                config_path=pyspi_config,
                subset="default",
                normalise=False,
            )
            total_seconds = time.perf_counter() - t0
            timings = result.timings or {}

            entry = {
                "M": M,
                "T": T,
                "total_seconds": round(total_seconds, 4),
                "n_spis": len(timings),
                "timings": {k: round(v, 6) for k, v in timings.items()},
            }
            if "spi_families" not in output_data and result.metadata:
                output_data["spi_families"] = {m.name: m.family for m in result.metadata}
            print(f"  done in {total_seconds:.1f}s ({len(timings)} SPIs)")

        except Exception as exc:
            total_seconds = time.perf_counter() - t0
            entry = {
                "M": M,
                "T": T,
                "total_seconds": round(total_seconds, 4),
                "n_spis": 0,
                "timings": {},
                "error": str(exc),
            }
            print(f"  ERROR after {total_seconds:.1f}s: {exc}")

        output_data["results"].append(entry)
        _save(output_path, output_data)

    if skipped:
        print(f"\nDone. Skipped {skipped}/{total} combos (already computed).")
    else:
        print(f"\nDone. Computed {total} combos.")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
