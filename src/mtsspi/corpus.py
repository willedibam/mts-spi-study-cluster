"""What a corpus is, and the archive kind.

A corpus is an ordered set of recordings that pyspi is run on. The runner
(:mod:`mtsspi.run_corpus`) uses only this interface:

- ``path``, ``name``, ``pyspi_config``, ``normalise``, ``pair_tolerance``
- ``entries()``: the recordings, as :class:`CorpusEntry`, 1-based
- ``entry_dir(entry)``: where a recording's outputs go
- ``seed(entry)``: the seed recorded with, and passed to, its pyspi run
- ``load(entry)``: the ``(T, M)`` float64 series and its source metadata
- ``describe(entry, source)``: the descriptive part of ``meta.json``
- ``source_error(entry, meta)``: why stored metadata is not this recording
- ``expected_files(entry)``: outputs besides ``meta.json`` and ``spi_mpis.npz``
- ``validate()``: a check of the corpus as a whole

:class:`ArchiveCorpus` reads prepared recordings from one hashed NPZ archive:

- ``__dataset_names__``: ordered string array of member names;
- ``__labels_json__``: JSON lists of tags aligned with the names;
- ``__shapes__``: ``(M, T)`` rows aligned with the names;
- each name is also a member holding its numeric array.

:class:`mtsspi.generated_corpus.GeneratedCorpus` generates them from a config.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import array_sha256, file_sha256
from .utils import load_yaml, resolve_path, slugify, to_relative, resolve_pyspi_config

SOURCE_FORMAT = "named-npz-v1"


@dataclass(frozen=True)
class CorpusEntry:
    index: int
    name: str
    labels: tuple[str, ...]
    M: int
    T: int
    # A generated corpus keeps the recording's generator settings here.
    spec: Any = field(default=None, compare=False, repr=False)

    def output_dir(self, corpus) -> Path:
        return corpus.entry_dir(self)


@dataclass(frozen=True)
class ArchiveCorpus:
    path: Path
    name: str
    archive: Path
    archive_sha256: str
    source_axis_order: tuple[str, str]
    base_output_dir: Path
    pyspi_config: Path
    normalise: bool
    random_seed: int
    # Optional `pair_tolerance:` in the YAML: compute a sample of channel dyads
    # sized by pyspi's budget rule instead of every dyad (large M only). The
    # sample is drawn with `random_seed`.
    pair_tolerance: float | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "ArchiveCorpus":
        config_path = resolve_path(path)
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
        random_seed = payload.get("random_seed")
        if not isinstance(random_seed, int) or isinstance(random_seed, bool) or random_seed < 0:
            raise ValueError("random_seed must be an explicit non-negative integer")
        pair_tolerance = payload.get("pair_tolerance")
        if pair_tolerance is not None and not (
                isinstance(pair_tolerance, float) and 0 < pair_tolerance < 1):
            raise ValueError("pair_tolerance must be a float in (0, 1) when given")
        return cls(
            pair_tolerance=pair_tolerance,
            path=config_path,
            name=name,
            archive=resolve_path(source["archive"]),
            archive_sha256=expected_hash,
            source_axis_order=axis_order,  # type: ignore[arg-type]
            base_output_dir=resolve_path(payload["base_output_dir"]),
            pyspi_config=resolve_pyspi_config(payload["pyspi_config"]),
            normalise=payload["normalise"],
            random_seed=random_seed,
        )

    @property
    def corpus_output_dir(self) -> Path:
        return self.base_output_dir / slugify(self.name, fallback="external-corpus")

    # -- the corpus interface (see the module docstring) --------------------

    def entries(self) -> list[CorpusEntry]:
        return load_inventory(self)

    def entry_dir(self, entry: CorpusEntry) -> Path:
        return self.corpus_output_dir / f"{entry.index:04d}-{slugify(entry.name, 'dataset')}"

    def seed(self, entry: CorpusEntry) -> int:
        return self.random_seed

    def load(self, entry: CorpusEntry) -> tuple[np.ndarray, dict[str, Any]]:
        return load_timeseries(self, entry)

    def expected_files(self, entry: CorpusEntry) -> list[str]:
        return []

    def describe(self, entry: CorpusEntry, source: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": entry.name,
            "dataset_name": entry.name,
            "mts_class": entry.name,
            "labels": list(entry.labels),
            "M": entry.M,
            "T": entry.T,
            "instance_index": entry.index - 1,
            "source": {
                "type": "external_archive",
                "format": SOURCE_FORMAT,
                "corpus": self.name,
                "archive": to_relative(self.archive),
                "archive_sha256": self.archive_sha256,
                "axis_order": list(self.source_axis_order),
                "member": entry.name,
                **source,
            },
            "base_output_dir": to_relative(self.base_output_dir),
        }

    def source_error(self, entry: CorpusEntry, meta: dict[str, Any]) -> str | None:
        if meta.get("dataset_name") != entry.name:
            return "dataset identity mismatch"
        source = meta.get("source", {})
        if source.get("archive_sha256") != self.archive_sha256 or source.get("member") != entry.name:
            return "source identity mismatch"
        return None

    def validate(self) -> dict[str, Any]:
        return validate_source(self)



def load_inventory(config: ArchiveCorpus) -> list[CorpusEntry]:
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


def _source_to_timeseries(
    config: ArchiveCorpus,
    entry: CorpusEntry,
    source: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
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
        "member_sha256": array_sha256(source),
        "source_dtype": str(source.dtype),
        "source_shape": list(source.shape),
    }


def load_timeseries(
    config: ArchiveCorpus, entry: CorpusEntry
) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(config.archive, allow_pickle=False) as archive:
        source = np.asarray(archive[entry.name])
    return _source_to_timeseries(config, entry, source)


def validate_source(config: ArchiveCorpus) -> dict[str, Any]:
    actual_hash = file_sha256(config.archive)
    if actual_hash != config.archive_sha256:
        raise ValueError(
            f"source archive SHA-256 mismatch: {actual_hash} != {config.archive_sha256}"
        )
    entries = load_inventory(config)
    dtype_counts: dict[str, int] = {}
    # One archive open matters on shared filesystems: reopening the ZIP for every
    # member turns a 39 MB validation into 1,053 metadata scans on gdata.
    with np.load(config.archive, allow_pickle=False) as archive:
        for entry in entries:
            _, source = _source_to_timeseries(config, entry, np.asarray(archive[entry.name]))
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
