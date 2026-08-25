"""Compute the 110-feature aggregated Catch22 corpus baseline."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from multiprocessing import get_context
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Sequence

import numpy as np

from .run_external_corpus import ExternalCorpusConfig, load_inventory
from .utils import project_root


SUMMARY_NAMES = ("min", "q25", "mean", "q75", "max")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_catch22(timeseries: np.ndarray) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute Catch22 per channel, then min/Q1/mean/Q3/max per feature."""

    import pycatch22

    values = np.asarray(timeseries, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("timeseries must have shape (T, M), both at least two")
    per_channel: list[np.ndarray] = []
    names: list[str] | None = None
    errors: list[str] = []
    for channel in range(values.shape[1]):
        series = values[:, channel]
        scale = float(np.std(series))
        if not np.isfinite(scale) or scale < 1e-12:
            errors.append(f"channel {channel}: constant or non-finite")
            per_channel.append(np.full(22, np.nan))
            continue
        series = (series - np.mean(series)) / scale
        try:
            result = pycatch22.catch22_all(series)
            channel_names = [str(name) for name in result["names"]]
            channel_values = np.asarray(result["values"], dtype=np.float64)
            if channel_values.shape != (22,):
                raise ValueError(f"expected 22 values, got {channel_values.shape}")
            if names is None:
                names = channel_names
            elif names != channel_names:
                raise ValueError("Catch22 feature order changed between channels")
            per_channel.append(channel_values)
        except Exception as exc:  # retain the dataset with explicit missing values
            errors.append(f"channel {channel}: {type(exc).__name__}: {exc}")
            per_channel.append(np.full(22, np.nan))
    if names is None:
        # Feature names do not depend on data; get them from a safe probe.
        names = [str(name) for name in pycatch22.catch22_all(np.arange(20.0))["names"]]
    matrix = np.asarray(per_channel, dtype=np.float64).T
    with np.errstate(all="ignore"):
        summaries = np.column_stack(
            (
                np.nanmin(matrix, axis=1),
                np.nanquantile(matrix, 0.25, axis=1),
                np.nanmean(matrix, axis=1),
                np.nanquantile(matrix, 0.75, axis=1),
                np.nanmax(matrix, axis=1),
            )
        )
    schema = [
        f"{feature}__{summary}"
        for feature in names
        for summary in SUMMARY_NAMES
    ]
    return summaries.reshape(-1).astype(np.float32), schema, errors


def _worker(
    task: tuple[str, str, tuple[str, str]],
) -> tuple[np.ndarray, list[str], list[str]]:
    archive_path, member, axis_order = task
    with np.load(archive_path, allow_pickle=False) as archive:
        source = np.asarray(archive[member])
    timeseries = source.T if axis_order == ("process", "observation") else source
    return aggregate_catch22(timeseries)


def _atomic_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run(config_path: str | Path, output: str | Path, *, workers: int) -> Path:
    config = ExternalCorpusConfig.from_file(config_path)
    entries = load_inventory(config)
    tasks = [
        (str(config.archive), entry.name, config.source_axis_order)
        for entry in entries
    ]
    if workers < 1:
        raise ValueError("workers must be at least one")
    if workers == 1:
        results = map(_worker, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
        )
        results = executor.map(_worker, tasks)
    rows: list[np.ndarray] = []
    errors: list[str] = []
    schema: list[str] | None = None
    try:
        for index, (row, row_schema, row_errors) in enumerate(results, start=1):
            if schema is None:
                schema = row_schema
            elif row_schema != schema:
                raise ValueError("Catch22 schema mismatch across datasets")
            rows.append(row)
            errors.append(json.dumps(row_errors, separators=(",", ":")))
            if index % 50 == 0 or index == len(entries):
                print(f"catch22: {index}/{len(entries)}", flush=True)
    finally:
        if executor is not None:
            executor.shutdown()
    if schema is None:
        raise RuntimeError("no datasets found")
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root(), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    destination = Path(output).resolve()
    _atomic_npz(
        destination,
        {
            "X": np.vstack(rows),
            "feature_names": np.asarray(schema),
            "dataset": np.asarray([entry.name for entry in entries]),
            "labels": np.asarray([entry.labels for entry in entries], dtype=object),
            "M": np.asarray([entry.M for entry in entries]),
            "T": np.asarray([entry.T for entry in entries]),
            "errors_json": np.asarray(errors),
            "source_archive": str(config.archive),
            "source_sha256": config.archive_sha256,
            "corpus_config_sha256": _file_sha256(config.path),
            "git_commit": commit,
            "normalisation": "per-channel-zscore",
            "aggregation": "min-q25-mean-q75-max",
        },
    )
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(run(args.config, args.output, workers=args.workers))


if __name__ == "__main__":
    main()
