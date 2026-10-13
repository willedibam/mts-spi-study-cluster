"""A corpus generated from a config: classes of generator settings over M, T
and instances (``mts_classes:`` in the YAML).

Each recording's series is generated once and stored as ``timeseries.npy``
beside its results, with any ground truth the generator yields, so a retry
runs pyspi on exactly the values first generated.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .corpus import CorpusEntry
from .generators import harness
from .mapping import DatasetMapping, ExperimentConfig
from .provenance import atomic_savez
from .utils import load_json, resolve_path, to_relative


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


class GeneratedCorpus:
    def __init__(self, path: Path, config: ExperimentConfig) -> None:
        self.path = path
        self.name = path.stem
        self.config = config
        self.pyspi_config = config.pyspi_config
        self.normalise = config.normalise
        self.pair_tolerance = config.pair_tolerance
        self.base_output_dir = config.base_output_dir
        self.mapping = DatasetMapping(config)

    @classmethod
    def from_file(cls, path: str | Path) -> "GeneratedCorpus":
        config_path = resolve_path(path)
        return cls(config_path, ExperimentConfig.from_file(config_path))

    def entries(self) -> list[CorpusEntry]:
        return [
            CorpusEntry(spec.index, spec.name, tuple(spec.class_labels), spec.M, spec.T, spec=spec)
            for spec in self.mapping.specs
        ]

    def entry_dir(self, entry: CorpusEntry) -> Path:
        return Path(entry.spec.dataset_dir)

    def seed(self, entry: CorpusEntry) -> int:
        return entry.spec.rng_seed

    def expected_files(self, entry: CorpusEntry) -> list[str]:
        return ["timeseries.npy", *harness.expected_files(entry.spec)]

    def validate(self) -> dict[str, Any]:
        entries = self.entries()
        return {
            "datasets": len(entries),
            "M_min_max": [min(e.M for e in entries), max(e.M for e in entries)],
            "T_min_max": [min(e.T for e in entries), max(e.T for e in entries)],
            "generators": sorted({e.spec.generator for e in entries}),
        }

    def load(self, entry: CorpusEntry) -> tuple[np.ndarray, dict[str, Any]]:
        """The stored series if this recording was generated before, else a new one."""
        spec = entry.spec
        directory = self.entry_dir(entry)
        directory.mkdir(parents=True, exist_ok=True)
        series = directory / "timeseries.npy"
        if all((directory / name).exists() for name in self.expected_files(entry)):
            extras: dict[str, Any] = {}
            if (directory / "full_lattice.npy").exists():
                shape = list(np.load(directory / "full_lattice.npy", mmap_mode="r").shape)
                extras["full_lattice"] = {"path": "full_lattice.npy", "shape": shape}
            if (directory / "ground_truth.npz").exists():
                extras["ground_truth"] = harness.ground_truth_descriptor(directory / "ground_truth.npz")
            data = np.load(series)
        else:
            data, extras = harness.generate_from_spec(spec)
            data = np.asarray(data, dtype=np.float64)
            # Sidecars first: the series is what marks the recording as generated.
            mother = extras.pop("_mother", None)
            if mother is not None:
                _atomic_npy(directory / "mother.npy", np.asarray(mother, dtype=np.float64))
            full_lattice = extras.pop("_full_lattice", None)
            if full_lattice is not None:
                lattice = np.asarray(full_lattice, dtype=np.float32)
                _atomic_npy(directory / "full_lattice.npy", lattice)
                extras["full_lattice"] = {"path": "full_lattice.npy", "shape": list(lattice.shape)}
            ground_truth = extras.pop("_ground_truth", None)
            if ground_truth is not None:
                atomic_savez(directory / "ground_truth.npz",
                             {name: np.asarray(values) for name, values in ground_truth.items()})
                extras["ground_truth"] = harness.ground_truth_descriptor(directory / "ground_truth.npz")
            _atomic_npy(series, data)
        extras.setdefault(
            "resolved_params", harness.resolve_params(spec.generator, dict(spec.generator_params)))
        return data.astype(np.float64, copy=False), extras

    def describe(self, entry: CorpusEntry, source: dict[str, Any]) -> dict[str, Any]:
        spec = entry.spec
        generator = {
            "type": "synthetic",
            "name": spec.generator,
            "params": spec.generator_params,
            "seed": spec.rng_seed,
            **source,
            **harness.semantics(spec, source),
        }
        # Provenance of a series produced outside the runner (a pre-seeded
        # continuation), when its author left one beside it.
        preseed = self.entry_dir(entry) / "gen_provenance.json"
        if preseed.exists():
            try:
                generator["preseed"] = load_json(preseed)
            except (ValueError, OSError):
                pass
        variant = spec.variant
        return {
            "name": spec.name,
            "mts_class": spec.mts_class,
            "labels": list(spec.class_labels),
            "M": spec.M,
            "T": spec.T,
            "instance_index": spec.instance,
            "sampling_design": {
                "seed_scope": spec.seed_scope,
                "seed_group_id": spec.seed_group_id,
                "role": (
                    "paired-control-path" if "paired" in spec.class_labels
                    else "independent-cell" if "independent-cell" in spec.class_labels
                    else "unspecified"
                ),
            },
            "variant": variant and {
                "name": variant.name or "", "params": variant.params, "slug": variant.slug},
            "generator": generator,
            "base_output_dir": to_relative(self.base_output_dir),
        }

    def source_error(self, entry: CorpusEntry, meta: dict[str, Any]) -> str | None:
        if meta.get("name") != entry.name:
            return "dataset identity mismatch"
        if meta.get("generator", {}).get("params") != _jsonable(entry.spec.generator_params):
            return "generator parameters mismatch"
        return None


def _jsonable(value: Any) -> Any:
    """`value` as it reads back from JSON, for comparison with stored metadata."""
    return json.loads(json.dumps(value))
