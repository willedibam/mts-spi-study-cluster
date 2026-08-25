"""Run pyspi over a named-array NPZ corpus, one dataset per process.

The source archive contract is deliberately small and pickle-free:

- ``__dataset_names__``: ordered string array of dataset member names;
- ``__labels_json__``: JSON lists of tags aligned with dataset names;
- ``__shapes__``: ``(M, T)`` rows aligned with dataset names;
- each dataset name is also an NPZ member containing its numeric array.

The command mirrors the dataset-scale API of :mod:`src.run_experiments` without
pretending external arrays are generators or package-backed classification data.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Sequence

import numpy as np

from .compute import ComputeResult, run_pyspi
from .utils import load_yaml, project_root, slugify, timestamp, to_relative


SOURCE_FORMAT = "named-npz-v1"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root() / path


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _repository_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(project_root()), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(project_root()), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {"git_commit": commit, "git_dirty": dirty}


def _pyspi_version() -> dict[str, str]:
    try:
        dist = version("pyspi")
    except PackageNotFoundError:
        dist = "unknown"
    try:
        from pyspi._parallel import COMPUTATION_VERSION as computation
    except Exception:  # pragma: no cover - defensive provenance fallback
        computation = "unknown"
    return {"dist": dist, "computation": str(computation)}


@dataclass(frozen=True)
class ExternalCorpusConfig:
    path: Path
    name: str
    archive: Path
    archive_sha256: str
    source_axis_order: tuple[str, str]
    base_output_dir: Path
    pyspi_config: Path
    normalise: bool

    @classmethod
    def from_file(cls, path: str | Path) -> "ExternalCorpusConfig":
        config_path = _resolve_path(path)
        payload = load_yaml(config_path)
        source = payload.get("source") or {}
        if source.get("format") != SOURCE_FORMAT:
            raise ValueError(f"source.format must be {SOURCE_FORMAT!r}")
        axis_order = tuple(source.get("axis_order") or ())
        if axis_order not in {
            ("process", "observation"),
            ("observation", "process"),
        }:
            raise ValueError(
                "source.axis_order must be [process, observation] or "
                "[observation, process]"
            )
        expected_hash = str(source.get("sha256") or "").lower()
        if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
            raise ValueError("source.sha256 must be a lowercase SHA-256 digest")
        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("config requires a non-empty name")
        if not isinstance(payload.get("normalise"), bool):
            raise ValueError("normalise must be an explicit YAML boolean")
        return cls(
            path=config_path,
            name=name,
            archive=_resolve_path(source["archive"]),
            archive_sha256=expected_hash,
            source_axis_order=axis_order,  # type: ignore[arg-type]
            base_output_dir=_resolve_path(payload["base_output_dir"]),
            pyspi_config=_resolve_path(payload["pyspi_config"]),
            normalise=payload["normalise"],
        )

    @property
    def corpus_output_dir(self) -> Path:
        return self.base_output_dir / slugify(self.name, fallback="external-corpus")


@dataclass(frozen=True)
class CorpusEntry:
    index: int
    name: str
    labels: tuple[str, ...]
    M: int
    T: int

    def output_dir(self, config: ExternalCorpusConfig) -> Path:
        return config.corpus_output_dir / f"{self.index:04d}-{slugify(self.name, 'dataset')}"


def load_inventory(config: ExternalCorpusConfig) -> list[CorpusEntry]:
    if not config.archive.is_file():
        raise FileNotFoundError(f"source archive not found: {config.archive}")
    if not config.pyspi_config.is_file():
        raise FileNotFoundError(f"pyspi config not found: {config.pyspi_config}")
    with np.load(config.archive, allow_pickle=False) as archive:
        required = {"__dataset_names__", "__labels_json__", "__shapes__"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"source archive missing metadata: {', '.join(missing)}")
        names = archive["__dataset_names__"].tolist()
        labels_json = archive["__labels_json__"].tolist()
        shapes = np.asarray(archive["__shapes__"], dtype=np.int64)
        archived_axis = (
            tuple(archive["__axis_order__"].tolist())
            if "__axis_order__" in archive.files
            else None
        )
        members = set(archive.files)
    if archived_axis is not None and archived_axis != config.source_axis_order:
        raise ValueError(
            f"configured axis order {config.source_axis_order} disagrees with "
            f"archive metadata {archived_axis}"
        )
    if len(names) != len(labels_json) or shapes.shape != (len(names), 2):
        raise ValueError("source metadata arrays are not aligned")
    if len(names) != len(set(names)):
        raise ValueError("source dataset names are not unique")
    entries: list[CorpusEntry] = []
    for offset, (name, encoded_labels, shape) in enumerate(
        zip(names, labels_json, shapes, strict=True), start=1
    ):
        if not isinstance(name, str) or name not in members:
            raise ValueError(f"invalid or missing dataset member at index {offset}: {name!r}")
        labels = json.loads(encoded_labels)
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise ValueError(f"labels for {name!r} are not a JSON list of strings")
        first, second = (int(shape[0]), int(shape[1]))
        M, T = (
            (first, second)
            if config.source_axis_order == ("process", "observation")
            else (second, first)
        )
        if M < 2 or T < 2:
            raise ValueError(f"invalid shape for {name!r}: M={M}, T={T}")
        entries.append(CorpusEntry(offset, name, tuple(labels), M, T))
    return entries


def load_timeseries(
    config: ExternalCorpusConfig, entry: CorpusEntry
) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(config.archive, allow_pickle=False) as archive:
        source = np.asarray(archive[entry.name])
    expected_source_shape = (
        (entry.M, entry.T)
        if config.source_axis_order == ("process", "observation")
        else (entry.T, entry.M)
    )
    if source.shape != expected_source_shape:
        raise ValueError(
            f"shape mismatch for {entry.name!r}: {source.shape} != {expected_source_shape}"
        )
    if not np.issubdtype(source.dtype, np.number):
        raise ValueError(f"dataset {entry.name!r} is not numeric ({source.dtype})")
    if not np.isfinite(source).all():
        raise ValueError(f"dataset {entry.name!r} contains non-finite values")
    data = source.T if config.source_axis_order == ("process", "observation") else source
    return data.astype(np.float64, copy=False), {
        "member_sha256": _array_sha256(source),
        "source_dtype": str(source.dtype),
        "source_shape": list(source.shape),
    }


def validate_source(config: ExternalCorpusConfig) -> dict[str, Any]:
    actual_hash = _file_sha256(config.archive)
    if actual_hash != config.archive_sha256:
        raise ValueError(
            f"source archive SHA-256 mismatch: {actual_hash} != {config.archive_sha256}"
        )
    entries = load_inventory(config)
    dtype_counts: dict[str, int] = {}
    for entry in entries:
        _, source = load_timeseries(config, entry)
        dtype = source["source_dtype"]
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    return {
        "archive": str(config.archive),
        "sha256": actual_hash,
        "datasets": len(entries),
        "M_min_max": [min(entry.M for entry in entries), max(entry.M for entry in entries)],
        "T_min_max": [min(entry.T for entry in entries), max(entry.T for entry in entries)],
        "dtype_counts": dtype_counts,
    }


def _atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _execution_identity(config: ExternalCorpusConfig) -> dict[str, Any]:
    return {
        "corpus_config_sha256": _file_sha256(config.path),
        "pyspi_config_sha256": _file_sha256(config.pyspi_config),
        "runner_sha256": _file_sha256(Path(__file__)),
        "compute_sha256": _file_sha256(Path(__file__).with_name("compute.py")),
        "pyspi_version": _pyspi_version(),
    }


def completion_error(
    config: ExternalCorpusConfig,
    entry: CorpusEntry,
    *,
    identity: dict[str, Any] | None = None,
) -> str | None:
    dataset_dir = entry.output_dir(config)
    meta_path = dataset_dir / "meta.json"
    mpi_path = dataset_dir / "spi_mpis.npz"
    if not meta_path.is_file() or not mpi_path.is_file():
        return "missing meta.json or spi_mpis.npz"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"invalid metadata: {exc}"
    current = identity or _execution_identity(config)
    if meta.get("status") != "complete":
        return "metadata status is not complete"
    if meta.get("dataset_name") != entry.name or meta.get("job", {}).get("index") != entry.index:
        return "dataset identity mismatch"
    source = meta.get("source", {})
    if source.get("archive_sha256") != config.archive_sha256 or source.get("member") != entry.name:
        return "source identity mismatch"
    recorded = meta.get("execution_identity", {})
    for key in (
        "corpus_config_sha256",
        "pyspi_config_sha256",
        "runner_sha256",
        "compute_sha256",
        "pyspi_version",
    ):
        if recorded.get(key) != current[key]:
            return f"execution identity mismatch: {key}"
    if meta.get("normalise") is not config.normalise:
        return "normalisation mismatch"
    spi_names = [item.get("name") for item in meta.get("pyspi", {}).get("spis", [])]
    if not spi_names or any(not isinstance(name, str) for name in spi_names):
        return "missing SPI catalogue"
    try:
        with np.load(mpi_path, allow_pickle=False) as archive:
            if archive.files != spi_names:
                return "SPI archive/catalogue mismatch"
            for name in spi_names:
                matrix = archive[name]
                if matrix.shape != (entry.M, entry.M):
                    return f"invalid MPI shape for {name}: {matrix.shape}"
                if np.isinf(matrix).any():
                    return f"infinite MPI values for {name}"
    except (OSError, ValueError, EOFError) as exc:
        return f"invalid SPI archive: {exc}"
    return None


def _metadata(
    config: ExternalCorpusConfig,
    entry: CorpusEntry,
    result: ComputeResult,
    source: dict[str, Any],
    identity: dict[str, Any],
    compute_seconds: float,
) -> dict[str, Any]:
    dataset_dir = entry.output_dir(config)
    return {
        "status": "complete",
        "name": entry.name,
        "dataset_name": entry.name,
        "mts_class": entry.name,
        "labels": list(entry.labels),
        "M": entry.M,
        "T": entry.T,
        "instance_index": entry.index - 1,
        "normalise": config.normalise,
        "timestamp": timestamp(),
        "source": {
            "type": "external_archive",
            "format": SOURCE_FORMAT,
            "corpus": config.name,
            "archive": to_relative(config.archive),
            "archive_sha256": config.archive_sha256,
            "axis_order": list(config.source_axis_order),
            "member": entry.name,
            **source,
        },
        "experiment": {
            "config": to_relative(config.path),
            **_repository_provenance(),
        },
        "execution_identity": identity,
        "pyspi": {
            "config": to_relative(config.pyspi_config),
            "config_sha256": identity["pyspi_config_sha256"],
            "version": identity["pyspi_version"],
            "n_spis": len(result.metadata),
            "errors": result.errors or {},
            "timings": {
                name: round(float(seconds), 6)
                for name, seconds in (result.timings or {}).items()
            },
            "spis": [
                {
                    "name": item.name,
                    "directed": item.directed,
                    "labels": item.labels,
                    "family": item.family,
                    "module": item.module,
                    "class_name": item.class_name,
                }
                for item in result.metadata
            ],
        },
        "paths": {"spi_archive": "spi_mpis.npz"},
        "base_output_dir": to_relative(config.base_output_dir),
        "dataset_dir": to_relative(dataset_dir),
        "job": {
            "index": entry.index,
            "n_jobs": 1,
            "compute_seconds": compute_seconds,
        },
    }


def run_dataset(
    config: ExternalCorpusConfig,
    entry: CorpusEntry,
    *,
    n_jobs: int = 1,
    skip_existing: bool = False,
    dry_run: bool = False,
) -> Path:
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least one")
    identity = _execution_identity(config)
    dataset_dir = entry.output_dir(config)
    if skip_existing and completion_error(config, entry, identity=identity) is None:
        print(f"[SKIP] {entry.index}/{entry.name}")
        return dataset_dir
    if dry_run:
        print(
            f"[DRY-RUN] {entry.index}/{entry.name} M={entry.M} T={entry.T} "
            f"-> {to_relative(dataset_dir)}"
        )
        return dataset_dir
    data, source = load_timeseries(config, entry)
    start = time.perf_counter()
    result = run_pyspi(
        data,
        config_path=config.pyspi_config,
        normalise=config.normalise,
        n_jobs=n_jobs,
    )
    elapsed = time.perf_counter() - start
    _atomic_savez(dataset_dir / "spi_mpis.npz", result.matrices)
    meta = _metadata(config, entry, result, source, identity, elapsed)
    meta["job"]["n_jobs"] = n_jobs
    _atomic_json(dataset_dir / "meta.json", meta)
    error = completion_error(config, entry, identity=identity)
    if error is not None:
        raise RuntimeError(f"post-write validation failed for {entry.name}: {error}")
    print(
        f"[DONE] {entry.index}/{entry.name} M={entry.M} T={entry.T} "
        f"spis={len(result.metadata)} errors={len(result.errors or {})} seconds={elapsed:.1f}"
    )
    return dataset_dir


def _read_indices(path: Path, total: int) -> list[int]:
    indices = [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not indices:
        raise ValueError(f"index file is empty: {path}")
    if len(indices) != len(set(indices)):
        raise ValueError(f"index file contains duplicates: {path}")
    invalid = [index for index in indices if index < 1 or index > total]
    if invalid:
        raise ValueError(f"index file contains out-of-range values: {invalid[:10]}")
    return indices


def audit_outputs(
    config: ExternalCorpusConfig,
    entries: Sequence[CorpusEntry],
) -> dict[str, Any]:
    identity = _execution_identity(config)
    failures = [
        {"index": entry.index, "name": entry.name, "error": error}
        for entry in entries
        if (error := completion_error(config, entry, identity=identity)) is not None
    ]
    return {
        "checked": len(entries),
        "complete": len(entries) - len(failures),
        "failures": failures,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="External-corpus YAML config.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--count-only", action="store_true")
    action.add_argument("--list", action="store_true")
    action.add_argument("--validate-source", action="store_true")
    action.add_argument("--audit", action="store_true")
    action.add_argument("--job-index", type=int)
    parser.add_argument("--index-file", type=Path, help="Restrict --audit to these 1-based indices.")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = ExternalCorpusConfig.from_file(args.config)
    entries = load_inventory(config)
    if args.count_only:
        print(len(entries))
        return
    if args.list:
        for entry in entries:
            print(entry.index, entry.M, entry.T, entry.name, sep="\t")
        return
    if args.validate_source:
        print(json.dumps(validate_source(config), indent=2))
        return
    if args.audit:
        selected = entries
        if args.index_file:
            wanted = _read_indices(args.index_file, len(entries))
            selected = [entries[index - 1] for index in wanted]
        result = audit_outputs(config, selected)
        print(json.dumps(result, indent=2))
        if result["failures"]:
            raise SystemExit(1)
        return
    if args.index_file:
        raise ValueError("--index-file is valid only with --audit")
    assert args.job_index is not None
    if args.job_index < 1 or args.job_index > len(entries):
        raise IndexError(f"job index {args.job_index} outside 1..{len(entries)}")
    run_dataset(
        config,
        entries[args.job_index - 1],
        n_jobs=args.n_jobs,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
