"""Run pyspi over a corpus, one recording per process.

    python -m src.run_corpus --config CORPUS.yaml --job-index 3 --n-jobs 4

One contract for every corpus (see :mod:`src.corpus`): a recording is complete
only if its metadata names this corpus config, this pyspi config, this code and
this pyspi computation version, and its SPI archive matches its catalogue.
Outputs are written atomically and validated after writing, so `--skip-existing`
never accepts a partial or stale result, and `--audit` fails on a mixed corpus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any, Sequence

import numpy as np

from .compute import ComputeResult, run_pyspi
from .corpus import ArchiveCorpus, CorpusEntry
from .generated_corpus import GeneratedCorpus
from .provenance import (
    atomic_json, atomic_savez, file_sha256, peak_memory_mib, pyspi_version,
    repository_provenance, worker_resources,
)
from .utils import load_yaml, resolve_path, timestamp, to_relative

# Source files whose change invalidates stored results.
_CODE = ("run_corpus.py", "corpus.py", "generated_corpus.py", "compute.py")


def load_corpus(path: str | Path):
    """The corpus a YAML file defines: generated (`mts_classes:`) or an archive (`source:`)."""
    kind = GeneratedCorpus if "mts_classes" in load_yaml(resolve_path(path)) else ArchiveCorpus
    return kind.from_file(path)


def execution_identity(corpus) -> dict[str, Any]:
    here = Path(__file__).parent
    return {
        "corpus_config_sha256": file_sha256(corpus.path),
        "pyspi_config_sha256": file_sha256(corpus.pyspi_config),
        "code_sha256": {name: file_sha256(here / name) for name in _CODE},
        "pyspi_version": pyspi_version(),
    }


def completion_error(
    corpus, entry: CorpusEntry, *, identity: dict[str, Any] | None = None
) -> str | None:
    """Why this recording's stored output is not a finished run of this corpus, or None."""
    dataset_dir = entry.output_dir(corpus)
    meta_path = dataset_dir / "meta.json"
    mpi_path = dataset_dir / "spi_mpis.npz"
    if not meta_path.is_file() or not mpi_path.is_file():
        return "missing meta.json or spi_mpis.npz"
    missing = [name for name in corpus.expected_files(entry) if not (dataset_dir / name).is_file()]
    if missing:
        return f"missing {', '.join(missing)}"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"invalid metadata: {exc}"
    if meta.get("status") != "complete":
        return "metadata status is not complete"
    if meta.get("job", {}).get("index") != entry.index:
        return "dataset identity mismatch"
    error = corpus.source_error(entry, meta)
    if error is not None:
        return error
    current = identity or execution_identity(corpus)
    recorded = meta.get("execution_identity", {})
    for key, value in current.items():
        if recorded.get(key) != value:
            return f"execution identity mismatch: {key}"
    if meta.get("normalise") is not corpus.normalise:
        return "normalisation mismatch"
    if meta.get("random_seed") != corpus.seed(entry):
        return "random seed mismatch"
    sampling = meta.get("pyspi", {}).get("sampling")
    if (sampling or {}).get("tolerance") != corpus.pair_tolerance:
        return "pair sampling mismatch"
    if corpus.pair_tolerance is not None:
        try:
            with np.load(dataset_dir / "pairs.npz", allow_pickle=False) as held:
                if held["pairs"].shape != (sampling["n_pairs"], 2):
                    return "sampled pairs do not match the metadata"
        except (OSError, KeyError, ValueError) as exc:
            return f"invalid sampled pairs: {exc}"
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
    corpus,
    entry: CorpusEntry,
    result: ComputeResult,
    source: dict[str, Any],
    identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "complete",
        **corpus.describe(entry, source),
        "normalise": corpus.normalise,
        "random_seed": corpus.seed(entry),
        "timestamp": timestamp(),
        "experiment": {"config": to_relative(corpus.path), **repository_provenance()},
        "execution_identity": identity,
        "pyspi": {
            "config": to_relative(corpus.pyspi_config),
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
        "dataset_dir": to_relative(entry.output_dir(corpus)),
    }


def run_dataset(
    corpus,
    entry: CorpusEntry,
    *,
    n_jobs: int = 1,
    skip_existing: bool = False,
    dry_run: bool = False,
    checkpoints: bool = False,
) -> Path:
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least one")
    identity = execution_identity(corpus)
    dataset_dir = entry.output_dir(corpus)
    if skip_existing and completion_error(corpus, entry, identity=identity) is None:
        print(f"[SKIP] {entry.index}/{entry.name}")
        return dataset_dir
    if dry_run:
        print(
            f"[DRY-RUN] {entry.index}/{entry.name} M={entry.M} T={entry.T} "
            f"-> {to_relative(dataset_dir)}"
        )
        return dataset_dir
    resources = worker_resources(n_jobs)
    data, source = corpus.load(entry)
    seed = corpus.seed(entry)
    sampled = corpus.pair_tolerance is not None
    start = time.perf_counter()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        np.random.seed(seed)
        random.seed(seed)
        result = run_pyspi(
            data,
            config_path=corpus.pyspi_config,
            normalise=corpus.normalise,
            n_jobs=n_jobs,
            random_seed=seed,
            **({"pair_tolerance": corpus.pair_tolerance, "pair_seed": seed} if sampled else {}),
            # pyspi's per-SPI checkpoints: a killed recording resumes from them.
            **({"checkpoint_dir": dataset_dir / "checkpoints"} if checkpoints else {}),
        )
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    elapsed = time.perf_counter() - start
    atomic_savez(dataset_dir / "spi_mpis.npz", result.matrices)
    meta = _metadata(corpus, entry, result, source, identity)
    if sampled:
        # Which entries of each matrix were computed; the rest are NaN.
        atomic_savez(dataset_dir / "pairs.npz", {"pairs": result.pairs})
        meta["paths"]["pairs"] = "pairs.npz"
        meta["pyspi"]["sampling"] = result.sampling
    meta["job"] = {
        "index": entry.index,
        "n_jobs": n_jobs,
        "compute_seconds": elapsed,
        "estimator_rng_policy": result.rng_policy,
        **resources,
        "peak_memory_mib": peak_memory_mib(),
    }
    atomic_json(dataset_dir / "meta.json", meta)
    error = completion_error(corpus, entry, identity=identity)
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


def audit_outputs(corpus, entries: Sequence[CorpusEntry]) -> dict[str, Any]:
    identity = execution_identity(corpus)
    failures = [
        {"index": entry.index, "name": entry.name, "error": error}
        for entry in entries
        if (error := completion_error(corpus, entry, identity=identity)) is not None
    ]
    return {
        "checked": len(entries),
        "complete": len(entries) - len(failures),
        "failures": failures,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, help="Corpus YAML: `mts_classes:` or `source:`.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--count-only", action="store_true")
    action.add_argument("--list", action="store_true")
    action.add_argument("--output-dir", action="store_true", help="Print the corpus output root.")
    action.add_argument("--validate-source", action="store_true")
    action.add_argument("--audit", action="store_true")
    action.add_argument("--job-index", type=int)
    parser.add_argument("--index-file", type=Path, help="Restrict --audit to these 1-based indices.")
    parser.add_argument("--n-jobs", type=int, default=1, help="pyspi workers for this recording.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--checkpoints", action="store_true",
                        help="Keep pyspi's per-SPI checkpoints so a killed recording resumes.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    corpus = load_corpus(args.config)
    if args.output_dir:
        print(corpus.base_output_dir)
        return
    entries = corpus.entries()
    if args.count_only:
        print(len(entries))
        return
    if args.list:
        for entry in entries:
            print(entry.index, entry.M, entry.T, entry.name, sep="\t")
        return
    if args.validate_source:
        print(json.dumps(corpus.validate(), indent=2))
        return
    if args.audit:
        selected = entries
        if args.index_file:
            selected = [entries[index - 1] for index in _read_indices(args.index_file, len(entries))]
        result = audit_outputs(corpus, selected)
        print(json.dumps(result, indent=2))
        if result["failures"]:
            raise SystemExit(1)
        return
    if args.index_file:
        raise ValueError("--index-file is valid only with --audit")
    if args.job_index < 1 or args.job_index > len(entries):
        raise IndexError(f"job index {args.job_index} outside 1..{len(entries)}")
    run_dataset(
        corpus,
        entries[args.job_index - 1],
        n_jobs=args.n_jobs,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        checkpoints=args.checkpoints,
    )


if __name__ == "__main__":
    main()
