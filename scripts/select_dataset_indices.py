#!/usr/bin/env python3
"""Select stable DatasetMapping indices for homogeneous Gadi farms."""
from __future__ import annotations

import argparse
from collections.abc import Sequence

from src.mapping import DatasetMapping, ExperimentConfig


def select_indices(
    config: str,
    *,
    start: int = 1,
    end: int | None = None,
    M: int | None = None,
    T: int | None = None,
    instance_min: int | None = None,
    instance_max: int | None = None,
) -> list[int]:
    mapping = DatasetMapping(ExperimentConfig.from_file(config))
    upper = len(mapping) if end is None else end
    if start < 1 or upper < start or upper > len(mapping):
        raise ValueError(f"invalid index bounds {start}:{upper} for {len(mapping)} rows")
    selected: list[int] = []
    for index in range(start, upper + 1):
        spec = mapping.spec_for_index(index)
        if M is not None and spec.M != M:
            continue
        if T is not None and spec.T != T:
            continue
        if instance_min is not None and spec.instance < instance_min:
            continue
        if instance_max is not None and spec.instance > instance_max:
            continue
        selected.append(index)
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int)
    parser.add_argument("--M", type=int)
    parser.add_argument("--T", type=int)
    parser.add_argument("--instance-min", type=int)
    parser.add_argument("--instance-max", type=int)
    parser.add_argument("--count", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    indices = select_indices(
        args.config,
        start=args.start,
        end=args.end,
        M=args.M,
        T=args.T,
        instance_min=args.instance_min,
        instance_max=args.instance_max,
    )
    if not indices:
        raise SystemExit("selection is empty")
    if args.count:
        print(len(indices))
    else:
        print(*indices, sep="\n")


if __name__ == "__main__":
    main()
