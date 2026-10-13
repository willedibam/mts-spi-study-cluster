from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .utils import (
    resolve_pyspi_config,
    resolve_path,
    class_dir_name,
    load_yaml,
    project_root,
    slugify,
    variant_suffix,
)


def _sin_mts_pre_seed(M: int, T: int, instance: int, mts_class: str) -> int:
    """Stable seed for sin_mts random regime assignment, independent of the dataset slug."""
    payload = f"sin_mts_regime|{M}|{T}|{instance}|{mts_class}".encode()
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32 - 1) or 1


def _sin_mts_mother_pre_seed(M: int, T: int, instance: int, mts_class: str) -> int:
    """Stable seed for sin_mts_mother random regime assignment, independent of the dataset slug."""
    payload = f"sin_mts_mother_regime|{M}|{T}|{instance}|{mts_class}".encode()
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32 - 1) or 1


def _warping_mts_pre_seed(M: int, T: int, instance: int, mts_class: str) -> int:
    """Stable seed for warping_mts random channel assignment, independent of the dataset slug."""
    payload = f"warping_mts_regime|{M}|{T}|{instance}|{mts_class}".encode()
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**32 - 1) or 1


def _parse_instances(value: Any) -> List[int]:
    """Accept either a count or an explicit list of non-negative instances."""

    if value is None:
        return []
    if isinstance(value, int):
        if value < 0:
            raise ValueError("instances count must be non-negative")
        return list(range(value))
    if not isinstance(value, (list, tuple)):
        raise ValueError("instances must be a non-negative count or an explicit list")
    parsed = [int(item) for item in value]
    if any(item < 0 for item in parsed):
        raise ValueError("instance indices must be non-negative")
    if len(set(parsed)) != len(parsed):
        raise ValueError("instance indices must be unique")
    return parsed


@dataclass(frozen=True)
class VariantSpec:
    name: str | None
    params: Dict[str, Any]

    @property
    def slug(self) -> str:
        if self.name:
            return slugify(self.name)
        if self.params:
            return variant_suffix(self.params)
        return ""


@dataclass
class ClassSpec:
    name: str
    generator: str
    labels: List[str]
    base_params: Dict[str, Any]
    M_values: List[int]
    T_values: List[int]
    instances: List[int]
    variants: List[VariantSpec]
    include_base_variant: bool
    rng_seed: int | None = None
    seed_scope: str = "dataset"


@dataclass
class ExperimentConfig:
    """A generated corpus: classes of generator settings over M, T and instances.

    The keys `timestamp`, `save_heatmap` and `threads` of older configs are
    ignored: output directories are deterministic, the runner draws nothing,
    and the worker count is a run-time option.
    """
    base_output_dir: Path
    pyspi_config: Path
    normalise: bool
    rng_seed: int
    pair_tolerance: float | None
    default_M_values: List[int]
    default_T_values: List[int]
    default_instances: List[int]
    classes: List[ClassSpec] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        data = load_yaml(path)
        defaults = data.get("defaults") or {}
        default_M = [int(v) for v in (defaults.get("M_values") or [])]
        default_T = [int(v) for v in (defaults.get("T_values") or [])]
        default_instances = _parse_instances(defaults.get("instances"))
        classes = [
            _parse_class(entry, default_M, default_T, default_instances)
            for entry in data.get("mts_classes") or []
        ]
        if not classes:
            raise ValueError("No mts_classes defined in experiment config.")
        tolerance = data.get("pair_tolerance")
        return cls(
            base_output_dir=resolve_path(data.get("base_output_dir", "data")),
            pyspi_config=resolve_pyspi_config(data["pyspi_config"]),
            normalise=bool(data.get("normalise", False)),
            rng_seed=int(data.get("rng_seed", 0)),
            pair_tolerance=None if tolerance is None else float(tolerance),
            default_M_values=default_M,
            default_T_values=default_T,
            default_instances=default_instances,
            classes=classes,
        )


def _parse_class(
    entry: dict[str, Any],
    default_M: List[int],
    default_T: List[int],
    default_instances: List[int],
) -> ClassSpec:
    if "name" not in entry:
        raise ValueError(f"Class entry missing 'name'. Entry: {entry}")
    if entry.get("package"):
        raise ValueError(
            f"Class entry '{entry['name']}' names a data package. The runner no longer "
            "loads real datasets: prepare them as an archive and use a `source:` config."
        )
    generator = entry.get("generator")
    if not generator:
        raise ValueError(f"Class entry '{entry['name']}' must define 'generator'.")
    variants_data = entry.get("variants") or []
    variants = [
        VariantSpec(name=var.get("name"), params=var.get("params", {}))
        for var in variants_data
    ]
    base_params = entry.get("base_params", {})
    def _resolve_list(value, default):
        if value is None:
            return list(default)
        vals = list(value)
        if not vals and default:
            return list(default)
        return vals
    M_values = [int(v) for v in _resolve_list(entry.get("M_values"), default_M)]
    T_values = [int(v) for v in _resolve_list(entry.get("T_values"), default_T)]
    instances = (
        _parse_instances(entry["instances"])
        if "instances" in entry
        else list(default_instances)
    )
    seed_scope = str(entry.get("seed_scope", "dataset")).strip().lower()
    if seed_scope not in {"dataset", "instance"}:
        raise ValueError(
            f"Unsupported seed_scope {seed_scope!r} for '{entry.get('name')}'. "
            "Expected 'dataset' or 'instance'."
        )
    return ClassSpec(
        name=entry["name"],
        generator=generator,
        labels=list(entry.get("labels", [])),
        base_params=base_params,
        M_values=M_values,
        T_values=T_values,
        instances=instances,
        variants=variants,
        include_base_variant=entry.get("include_base_variant", True),
        rng_seed=int(entry["rng_seed"]) if "rng_seed" in entry and entry["rng_seed"] is not None else None,
        seed_scope=seed_scope,
    )


@dataclass
class DatasetSpec:
    index: int
    mts_class: str
    class_labels: List[str]
    class_dir: str
    dataset_slug: str
    dataset_dir: Path
    generator: str
    base_output_dir: Path
    generator_params: Dict[str, Any]
    variant: VariantSpec | None
    M: int
    T: int
    instance: int
    rng_seed: int
    seed_scope: str

    @property
    def name(self) -> str:
        return f"{self.mts_class}_{self.dataset_slug}"

    @property
    def seed_group_id(self) -> str:
        """Identifier for datasets that share one stochastic master draw."""
        if self.seed_scope == "instance":
            return f"{self.mts_class}:I{self.instance}"
        return self.name

    def to_summary(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "class": self.mts_class,
            "variant": self.variant.slug if self.variant else "",
            "M": self.M,
            "T": self.T,
            "instance": self.instance,
            "seed_scope": self.seed_scope,
            "seed_group_id": self.seed_group_id,
            "dataset_dir": str(self.dataset_dir),
        }


def _apply_dataset_slug(spec: DatasetSpec) -> None:
    slug = f"M{spec.M}_T{spec.T}_I{spec.instance}"
    if spec.variant and spec.variant.slug:
        slug = f"{slug}_{spec.variant.slug}"
    spec.dataset_slug = slug
    spec.dataset_dir = spec.base_output_dir / spec.class_dir / spec.dataset_slug
    if spec.generator == "sin_mts" and spec.generator_params.get("regime") == "random":
        spec.generator_params["_regime_seed"] = _sin_mts_pre_seed(
            spec.M, spec.T, spec.instance, spec.mts_class
        )
    if spec.generator == "sin_mts_mother" and spec.generator_params.get("regime") == "random":
        spec.generator_params["_regime_seed"] = _sin_mts_mother_pre_seed(
            spec.M, spec.T, spec.instance, spec.mts_class
        )
    if spec.generator == "warping_mts" and spec.generator_params.get("warping_channel_regime") == "random":
        spec.generator_params["_regime_seed"] = _warping_mts_pre_seed(
            spec.M, spec.T, spec.instance, spec.mts_class
        )
    if spec.generator == "sin_mts" and spec.generator_params.get("regime") == "random":
        spec.generator_params["_regime_seed"] = _sin_mts_pre_seed(
            spec.M, spec.T, spec.instance, spec.mts_class
        )
    if spec.generator == "sin_mts_mother" and spec.generator_params.get("regime") == "random":
        spec.generator_params["_regime_seed"] = _sin_mts_mother_pre_seed(
            spec.M, spec.T, spec.instance, spec.mts_class
        )
    if spec.generator == "warping_mts" and spec.generator_params.get("warping_channel_regime") == "random":
        spec.generator_params["_regime_seed"] = _warping_mts_pre_seed(
            spec.M, spec.T, spec.instance, spec.mts_class
        )


def _derive_dataset_seed(
    *, base_seed: int, spec: DatasetSpec, seed_scope: str = "dataset"
) -> int:
    if seed_scope == "instance":
        components = [
            str(base_seed),
            spec.mts_class,
            f"I{spec.instance}",
        ]
    elif seed_scope == "dataset":
        variant_slug = spec.variant.slug if spec.variant else ""
        components = [
            str(base_seed),
            spec.mts_class,
            spec.dataset_slug,
            variant_slug,
            f"M{spec.M}",
            f"T{spec.T}",
            f"I{spec.instance}",
        ]
    else:  # guarded by config parsing; retained for direct callers
        raise ValueError(f"unsupported seed_scope {seed_scope!r}")
    payload = "|".join(components).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8).digest()
    seed = int.from_bytes(digest, "big") % 2147483647
    if seed == 0:
        seed = 2147483647
    return seed


class DatasetMapping:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.specs: List[DatasetSpec] = self._expand_specs()

    def __len__(self) -> int:
        return len(self.specs)

    def spec_for_index(self, job_index: int) -> DatasetSpec:
        if job_index < 1 or job_index > len(self.specs):
            raise IndexError(f"Job index {job_index} out of bounds (1..{len(self.specs)}).")
        return self.specs[job_index - 1]

    def summaries(self) -> List[dict[str, Any]]:
        return [spec.to_summary() for spec in self.specs]

    def _expand_specs(self) -> List[DatasetSpec]:
        specs: List[DatasetSpec] = []
        for entry in self.config.classes:
            variants: List[VariantSpec | None] = [None] if entry.include_base_variant else []
            variants.extend(entry.variants)
            # No variant at all still yields the base settings, with the "M"
            # placeholder left as written (the behaviour configs were run with).
            substitute = bool(variants)
            base_seed = entry.rng_seed if entry.rng_seed is not None else self.config.rng_seed
            class_dir = class_dir_name(entry.name)
            for M in entry.M_values:
                for T in entry.T_values:
                    for instance in entry.instances:
                        for variant in variants or [None]:
                            params = {
                                k: (M if substitute and v == "M" else v)
                                for k, v in entry.base_params.items()
                            }
                            if variant:
                                params.update(variant.params)
                            # `delta` was a plotting option kept among the
                            # generator parameters; no generator takes it.
                            params.pop("delta", None)
                            spec = DatasetSpec(
                                index=len(specs) + 1,
                                mts_class=entry.name,
                                class_labels=entry.labels,
                                class_dir=class_dir,
                                dataset_slug="",
                                dataset_dir=self.config.base_output_dir / class_dir,
                                generator=entry.generator,
                                base_output_dir=self.config.base_output_dir,
                                generator_params=params,
                                variant=variant,
                                M=M,
                                T=T,
                                instance=instance,
                                rng_seed=0,
                                seed_scope=entry.seed_scope,
                            )
                            _apply_dataset_slug(spec)
                            spec.rng_seed = _derive_dataset_seed(
                                base_seed=base_seed, spec=spec, seed_scope=entry.seed_scope
                            )
                            specs.append(spec)
        return specs
