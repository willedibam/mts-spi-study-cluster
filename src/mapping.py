from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .utils import (
    class_dir_name,
    load_yaml,
    project_root,
    slugify,
    variant_suffix,
)


def _as_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root() / path


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
    pyspi_config: Path | None = None
    pyspi_subset: str | None = None
    normalise: bool | None = None
    save_heatmap: bool | None = None
    threads: int | None = None


@dataclass
class ExperimentConfig:
    mode: str
    base_output_dir: Path
    pyspi_config: Path
    pyspi_subset: str
    normalise: bool
    rng_seed: int
    save_heatmap: bool
    threads: int | None
    classes: List[ClassSpec] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        data = load_yaml(path)
        mode = data.get("mode")
        if not mode:
            raise ValueError("Experiment config missing 'mode'.")
        base_output_dir = _as_path(data.get("base_output_dir", f"data/{mode}"))
        pyspi_config = _as_path(data["pyspi_config"])
        pyspi_subset = data.get("pyspi_subset", "default")
        normalise = bool(data.get("normalise", True))
        rng_seed = int(data.get("rng_seed", 0))
        save_heatmap = bool(data.get("save_heatmap", False))
        threads = data.get("threads")
        classes_raw = data.get("mts_classes") or []
        classes: List[ClassSpec] = []
        for entry in classes_raw:
            classes.append(_parse_class(entry))
        if not classes:
            raise ValueError("No mts_classes defined in experiment config.")
        return cls(
            mode=mode,
            base_output_dir=base_output_dir,
            pyspi_config=pyspi_config,
            pyspi_subset=pyspi_subset,
            normalise=normalise,
            rng_seed=rng_seed,
            save_heatmap=save_heatmap,
            threads=threads,
            classes=classes,
        )


def _parse_class(entry: dict[str, Any]) -> ClassSpec:
    required = ["name", "generator", "M_values", "T_values", "instances"]
    for key in required:
        if key not in entry:
            raise ValueError(f"Class entry missing '{key}'. Entry: {entry}")
    variants_data = entry.get("variants") or []
    variants = [
        VariantSpec(name=var.get("name"), params=var.get("params", {}))
        for var in variants_data
    ]
    base_params = entry.get("base_params", {})
    return ClassSpec(
        name=entry["name"],
        generator=entry["generator"],
        labels=list(entry.get("labels", [])),
        base_params=base_params,
        M_values=list(entry["M_values"]),
        T_values=list(entry["T_values"]),
        instances=list(entry["instances"]),
        variants=variants,
        pyspi_config=_as_path(entry["pyspi_config"]) if entry.get("pyspi_config") else None,
        pyspi_subset=entry.get("pyspi_subset"),
        normalise=entry.get("normalise"),
        save_heatmap=entry.get("save_heatmap"),
        threads=entry.get("threads"),
    )


@dataclass
class DatasetSpec:
    index: int
    mode: str
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
    pyspi_config: Path
    pyspi_subset: str
    normalise: bool
    save_heatmap: bool
    rng_seed: int
    threads: int | None

    @property
    def name(self) -> str:
        return f"{self.mts_class}_{self.dataset_slug}"

    def to_summary(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "mode": self.mode,
            "class": self.mts_class,
            "variant": self.variant.slug if self.variant else "",
            "M": self.M,
            "T": self.T,
            "instance": self.instance,
            "pyspi_config": str(self.pyspi_config),
            "dataset_dir": str(self.dataset_dir),
        }


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
        seed_cursor = self.config.rng_seed
        for class_entry in self.config.classes:
            variant_list: List[VariantSpec | None] = [None]
            variant_list.extend(class_entry.variants)
            for M in class_entry.M_values:
                for T in class_entry.T_values:
                    for instance in class_entry.instances:
                        for variant in variant_list:
                            slug_extra = variant.slug if variant else ""
                            dataset_slug = f"M{M}_T{T}_I{instance}"
                            if slug_extra:
                                dataset_slug = f"{dataset_slug}_{slug_extra}"
                            class_dir = class_dir_name(class_entry.name)
                            dataset_dir = (
                                self.config.base_output_dir
                                / class_dir
                                / dataset_slug
                            )
                            generator_params = dict(class_entry.base_params)
                            if variant:
                                generator_params.update(variant.params)
                            pyspi_config = class_entry.pyspi_config or self.config.pyspi_config
                            pyspi_subset = class_entry.pyspi_subset or self.config.pyspi_subset
                            normalise = (
                                class_entry.normalise
                                if class_entry.normalise is not None
                                else self.config.normalise
                            )
                            save_heatmap = (
                                class_entry.save_heatmap
                                if class_entry.save_heatmap is not None
                                else self.config.save_heatmap
                            )
                            threads = class_entry.threads or self.config.threads
                            specs.append(
                                DatasetSpec(
                                    index=len(specs) + 1,
                                    mode=self.config.mode,
                                    mts_class=class_entry.name,
                                    class_labels=class_entry.labels,
                                    class_dir=class_dir,
                                    dataset_slug=dataset_slug,
                                    dataset_dir=dataset_dir,
                                    generator=class_entry.generator,
                                    base_output_dir=self.config.base_output_dir,
                                    generator_params=generator_params,
                                    variant=variant,
                                    M=M,
                                    T=T,
                                    instance=instance,
                                    pyspi_config=pyspi_config,
                                    pyspi_subset=pyspi_subset,
                                    normalise=normalise,
                                    save_heatmap=save_heatmap,
                                    rng_seed=seed_cursor,
                                    threads=threads,
                                )
                            )
                            seed_cursor += 1
        return specs
