#!/usr/bin/env python3
"""Run an MPI-free, PBS-array-friendly CML physical-diagnostic scout.

Each array task writes one independent JSON part.  No SPI is computed here:
the purpose is to locate and validate physical transition regions before the
expensive SPI--SPI confirmation campaign is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cml_order_parameter import largest_lyapunov_exponent, summarize_field
from src.generators.dynamical import generate_cml_logistic


def parse_grid(text: str, *, integer: bool = False) -> list[float] | list[int]:
    """Parse comma values or inclusive start:stop:step notation."""

    text = text.replace(";", ",")
    values: list[float]
    if ":" in text:
        parts = [float(part) for part in text.split(":")]
        if len(parts) != 3 or parts[2] <= 0 or parts[1] < parts[0]:
            raise ValueError(f"invalid grid {text!r}; expected start:stop:positive_step")
        start, stop, step = parts
        count = int(np.floor((stop - start) / step + 1e-9)) + 1
        values = [start + index * step for index in range(count)]
    else:
        values = [float(part) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"empty grid {text!r}")
    if integer:
        rounded = [int(round(value)) for value in values]
        if any(abs(value - rounded_value) > 1e-9 for value, rounded_value in zip(values, rounded)):
            raise ValueError(f"expected integer grid, got {text!r}")
        return rounded
    return values


def parse_band(text: str) -> tuple[float, float]:
    values = [float(part) for part in text.split(":")]
    if len(values) != 2 or not 0.0 < values[0] < values[1] <= 1.0:
        raise ValueError(
            f"invalid spatial band {text!r}; expected lower:upper with 0 < lower < upper <= 1"
        )
    return values[0], values[1]


def parse_integer_list(text: str) -> list[int]:
    if not text.strip():
        return []
    values = [int(part) for part in text.replace(";", ",").split(",")]
    if any(value < 3 for value in values):
        raise ValueError(f"analysis prefixes must all be >=3, got {text!r}")
    return sorted(set(values))


def stable_seed(base_seed: int, alpha: float, eps: float, lattice_size: int, instance: int) -> int:
    payload = f"{base_seed}|{alpha:.12g}|{eps:.12g}|{lattice_size}|{instance}".encode()
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32 - 1) or 1


def combinations(args: argparse.Namespace) -> list[tuple[float, float, int, int]]:
    alphas = parse_grid(args.alphas)
    eps_values = parse_grid(args.eps)
    lattice_sizes = parse_grid(args.lattice_sizes, integer=True)
    return list(product(alphas, eps_values, lattice_sizes, range(args.instances)))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_one(
    *,
    alpha: float,
    eps: float,
    lattice_size: int,
    instance: int,
    args: argparse.Namespace,
) -> dict:
    seed = stable_seed(args.base_seed, alpha, eps, lattice_size, instance)
    field = generate_cml_logistic(
        M=lattice_size,
        T=args.record_steps,
        alpha=alpha,
        eps=eps,
        transients=args.transients,
        sample_every=args.sample_every,
        rng=np.random.default_rng(seed),
        zscore=False,
        lattice_size=lattice_size,
    )
    summary_options = {
        "activity_thresholds": tuple(parse_grid(args.activity_thresholds)),
        "max_spatial_lag": min(args.max_spatial_lag, lattice_size // 2),
        "selected_spatial_band": parse_band(args.selected_spatial_band),
        "pattern_word_length": args.pattern_word_length,
    }
    summary = summarize_field(field, **summary_options)
    prefix_summaries = {
        str(prefix): summarize_field(field[:prefix], **summary_options)
        for prefix in parse_integer_list(args.analysis_prefixes)
        if prefix <= len(field)
    }
    block_summaries = []
    if args.stationarity_blocks:
        for block_index, indices in enumerate(np.array_split(np.arange(len(field)), args.stationarity_blocks)):
            if len(indices) < 3:
                raise ValueError("stationarity blocks must contain at least three observations")
            block_summaries.append(
                {
                    "block": block_index,
                    "start": int(indices[0]),
                    "stop": int(indices[-1] + 1),
                    **summarize_field(field[indices], **summary_options),
                }
            )
    if args.lyapunov_steps:
        summary["largest_lyapunov_exponent"] = largest_lyapunov_exponent(
            alpha=alpha,
            eps=eps,
            lattice_size=lattice_size,
            seed=seed,
            transients=args.transients,
            steps=args.lyapunov_steps,
        )
    return {
        "alpha": alpha,
        "eps": eps,
        "lattice_size": lattice_size,
        "instance": instance,
        "seed": seed,
        "transients": args.transients,
        "record_steps": args.record_steps,
        "sample_every": args.sample_every,
        "prefix_summaries": prefix_summaries,
        "stationarity_blocks": block_summaries,
        **summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", default="1.55:2.00:0.01")
    parser.add_argument("--eps", default="0.20:0.40:0.025")
    parser.add_argument("--lattice-sizes", default="128,256,512")
    parser.add_argument("--instances", type=int, default=16)
    parser.add_argument("--transients", type=int, default=20000)
    parser.add_argument("--record-steps", type=int, default=20000)
    parser.add_argument(
        "--analysis-prefixes",
        default="",
        help="comma-separated T prefixes summarised from the same trajectory",
    )
    parser.add_argument(
        "--stationarity-blocks",
        type=int,
        default=0,
        help="number of consecutive equal-length blocks to summarise",
    )
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--activity-thresholds", default="0.01,0.02,0.05,0.1,0.2")
    parser.add_argument(
        "--selected-spatial-band",
        default="0.25:0.45",
        help="frozen selected-pattern wavenumber band as fractions of pi",
    )
    parser.add_argument("--pattern-word-length", type=int, default=4)
    parser.add_argument("--max-spatial-lag", type=int, default=64)
    parser.add_argument("--lyapunov-steps", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=110305)
    parser.add_argument("--output-dir", type=Path, default=Path("results/cml-order-parameter-scout"))
    parser.add_argument("--task-index", type=int, help="1-based combination index, e.g. PBS_ARRAY_INDEX")
    parser.add_argument("--list", action="store_true", help="print combination count and exit")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.instances <= 0
        or args.transients < 0
        or args.record_steps < 3
        or not 2 <= args.pattern_word_length <= 8
        or args.stationarity_blocks < 0
        or args.stationarity_blocks > args.record_steps // 3
    ):
        raise ValueError("instances must be positive, transients non-negative, record_steps >= 3")
    parse_band(args.selected_spatial_band)
    parse_integer_list(args.analysis_prefixes)
    jobs = combinations(args)
    if args.list:
        print(len(jobs))
        return 0
    indices = [args.task_index] if args.task_index is not None else list(range(1, len(jobs) + 1))
    for index in indices:
        if index is None or not 1 <= index <= len(jobs):
            raise IndexError(f"task index must be in 1..{len(jobs)}, got {index}")
        output = args.output_dir / f"part-{index:06d}.json"
        if output.exists() and not args.overwrite:
            print(f"[skip] {output}")
            continue
        alpha, eps, lattice_size, instance = jobs[index - 1]
        record = run_one(
            alpha=float(alpha),
            eps=float(eps),
            lattice_size=int(lattice_size),
            instance=int(instance),
            args=args,
        )
        record["task_index"] = index
        record["task_count"] = len(jobs)
        atomic_json(output, record)
        print(
            f"[{index}/{len(jobs)}] alpha={alpha:.4f} eps={eps:.4f} "
            f"L={lattice_size} I={instance} -> {output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
