from __future__ import annotations

import hashlib
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


def _parse_delta_list(value: Any) -> List[int]:
    if value is None:
        return [1]
    if isinstance(value, list):
        parsed = [int(v) for v in value if int(v) > 0]
        return parsed or [1]
    if isinstance(value, (int, float)):
        val = int(value)
        return [val if val > 0 else 1]
    text = str(value)
    parts = [token.strip() for token in text.split(",")]
    parsed = []
    for token in parts:
        if not token:
            continue
        parsed.append(max(1, int(token)))
    return parsed or [1]


_KURAMOTO_CONNECTIVITY_ALIASES = {
    "all-to-all": "all-to-all",
    "all_to_all": "all-to-all",
    "alltoall": "all-to-all",
    "full": "all-to-all",
    "fully_connected": "all-to-all",
    "bidirectional-list": "bidirectional-list",
    "bidirectional_list": "bidirectional-list",
    "list": "bidirectional-list",
    "ring": "bidirectional-list",
    "grid-four": "grid-four",
    "grid_four": "grid-four",
    "grid-4": "grid-four",
    "grid": "grid-four",
}


def _format_numeric_token(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return slugify(str(value))
    text = f"{numeric:.8f}".rstrip("0").rstrip(".")
    if not text or text == "-":
        text = "0"
    if text == "-0":
        text = "0"
    return text.replace(".", "p")


def _canonical_kuramoto_connectivity(name: str) -> str:
    key = name.strip().lower().replace(" ", "-")
    key = key.replace("_", "-")
    if key not in _KURAMOTO_CONNECTIVITY_ALIASES:
        raise ValueError(
            f"Unknown Kuramoto connectivity '{name}'. "
            "Expected one of all-to-all, bidirectional-list, grid-four."
        )
    return _KURAMOTO_CONNECTIVITY_ALIASES[key]


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
    single_instance_M_values: List[int]
    single_instance_instances: List[int]
    single_instance_T_values: List[int]
    variants: List[VariantSpec]
    include_base_variant: bool
    pyspi_config: Path | None = None
    pyspi_subset: str | None = None
    normalise: bool | None = None
    save_heatmap: bool | None = None
    threads: int | None = None
    rng_seed: int | None = None


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
    default_M_values: List[int]
    default_T_values: List[int]
    default_instances: List[int]
    single_instance_M_values: List[int]
    single_instance_instances: List[int]
    single_instance_T_values: List[int]
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
        normalise = bool(data.get("normalise", False))
        rng_seed = int(data.get("rng_seed", 0))
        save_heatmap = bool(data.get("save_heatmap", False))
        threads = data.get("threads")
        defaults = data.get("defaults") or {}
        default_M = [int(v) for v in (defaults.get("M_values") or [])]
        default_T = [int(v) for v in (defaults.get("T_values") or [])]
        default_instances = [int(v) for v in (defaults.get("instances") or [])]
        default_single_M = [int(v) for v in (defaults.get("single_instance_M_values") or [])]
        default_single_instances = [
            int(v) for v in (defaults.get("single_instance_instances") or [0])
        ]
        default_single_T = [int(v) for v in (defaults.get("single_instance_T_values") or [])]
        classes_raw = data.get("mts_classes") or []
        classes: List[ClassSpec] = []
        for entry in classes_raw:
            classes.append(
                _parse_class(
                    entry,
                    default_M,
                    default_T,
                    default_instances,
                    default_single_M,
                    default_single_instances,
                    default_single_T,
                )
            )
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
            default_M_values=default_M,
            default_T_values=default_T,
            default_instances=default_instances,
            single_instance_M_values=default_single_M,
            single_instance_instances=default_single_instances,
            single_instance_T_values=default_single_T,
            classes=classes,
        )


def _parse_class(
    entry: dict[str, Any],
    default_M: List[int],
    default_T: List[int],
    default_instances: List[int],
    default_single_M: List[int],
    default_single_instances: List[int],
    default_single_T: List[int],
) -> ClassSpec:
    required = ["name", "generator"]
    for key in required:
        if key not in entry:
            raise ValueError(f"Class entry missing '{key}'. Entry: {entry}")
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
    instances = [int(v) for v in _resolve_list(entry.get("instances"), default_instances)]
    single_instance_M_values = [
        int(v) for v in _resolve_list(entry.get("single_instance_M_values"), default_single_M)
    ]
    single_instance_T_values = [
        int(v) for v in _resolve_list(entry.get("single_instance_T_values"), default_single_T)
    ]
    single_instance_instances: List[int] = []
    if single_instance_M_values:
        raw_single_instances = _resolve_list(
            entry.get("single_instance_instances"), default_single_instances
        )
        single_instance_instances = [int(v) for v in raw_single_instances]
    return ClassSpec(
        name=entry["name"],
        generator=entry["generator"],
        labels=list(entry.get("labels", [])),
        base_params=base_params,
        M_values=M_values,
        T_values=T_values,
        instances=instances,
        single_instance_M_values=single_instance_M_values,
        single_instance_instances=single_instance_instances,
        single_instance_T_values=single_instance_T_values,
        variants=variants,
        include_base_variant=entry.get("include_base_variant", True),
        pyspi_config=_as_path(entry["pyspi_config"]) if entry.get("pyspi_config") else None,
        pyspi_subset=entry.get("pyspi_subset"),
        normalise=entry.get("normalise"),
        save_heatmap=entry.get("save_heatmap"),
        threads=entry.get("threads"),
        rng_seed=int(entry["rng_seed"]) if "rng_seed" in entry and entry["rng_seed"] is not None else None,
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
    heatmap_deltas: List[int]

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


def _custom_dataset_slug(spec: DatasetSpec) -> str | None:
    base = f"M{spec.M}_T{spec.T}_I{spec.instance}"
    class_name = spec.mts_class.lower()
    if class_name.startswith("cml"):
        alpha = spec.generator_params.get("alpha")
        eps = spec.generator_params.get("eps") or spec.generator_params.get("epsilon")
        if alpha is None or eps is None:
            return None
        alpha_token = _format_numeric_token(alpha)
        eps_token = _format_numeric_token(eps)
        return f"{base}_alpha{alpha_token}-eps{eps_token}"
    if class_name == "kuramoto":
        conn = spec.generator_params.get("connectivity") or spec.generator_params.get(
            "coupling_scheme"
        )
        coupling = spec.generator_params.get("k")
        if coupling is None:
            coupling = spec.generator_params.get("K")
        if conn is None or coupling is None:
            return None
        conn_slug = _canonical_kuramoto_connectivity(str(conn))
        k_token = _format_numeric_token(coupling)
        return f"{base}_{conn_slug}-k-{k_token}"
    return None


def _apply_dataset_slug(spec: DatasetSpec) -> None:
    slug = _custom_dataset_slug(spec)
    if not slug:
        slug = f"M{spec.M}_T{spec.T}_I{spec.instance}"
    if spec.variant and spec.variant.slug:
        slug = f"{slug}_{spec.variant.slug}"
    spec.dataset_slug = slug
    spec.dataset_dir = spec.base_output_dir / spec.class_dir / spec.dataset_slug


def _derive_dataset_seed(*, base_seed: int, spec: DatasetSpec) -> int:
    variant_slug = spec.variant.slug if spec.variant else ""
    components = [
        str(base_seed),
        spec.mode,
        spec.mts_class,
        spec.dataset_slug,
        variant_slug,
        f"M{spec.M}",
        f"T{spec.T}",
        f"I{spec.instance}",
    ]
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
        for class_entry in self.config.classes:
            class_specs: List[DatasetSpec] = []
            regular_M_values = [
                m for m in class_entry.M_values if m not in class_entry.single_instance_M_values
            ]

            def _append_specs(
                M_values: List[int],
                instances: List[int],
                T_values: List[int] | None = None,
            ) -> None:
                t_values = T_values if T_values is not None else class_entry.T_values
                for M in M_values:
                    for T in t_values:
                        for instance in instances:
                            dataset_slug = f"M{M}_T{T}_I{instance}"
                            class_dir = class_dir_name(class_entry.name)
                            dataset_dir = (
                                self.config.base_output_dir
                                / class_dir
                                / dataset_slug
                            )
                            generator_params = dict(class_entry.base_params)
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
                            class_specs.append(
                                DatasetSpec(
                                    index=0,  # placeholder, updated later
                                    mode=self.config.mode,
                                    mts_class=class_entry.name,
                                    class_labels=class_entry.labels,
                                    class_dir=class_dir,
                                    dataset_slug=dataset_slug,
                                    dataset_dir=dataset_dir,
                                    generator=class_entry.generator,
                                    base_output_dir=self.config.base_output_dir,
                                    generator_params=generator_params,
                                    variant=None,
                                    M=M,
                                    T=T,
                                    instance=instance,
                                    pyspi_config=pyspi_config,
                                    pyspi_subset=pyspi_subset,
                                    normalise=normalise,
                                    save_heatmap=save_heatmap,
                                    rng_seed=0,
                                    threads=threads,
                                    heatmap_deltas=[1],
                                )
                            )

            if regular_M_values:
                _append_specs(regular_M_values, class_entry.instances)
            if class_entry.single_instance_M_values:
                single_instances = class_entry.single_instance_instances or [0]
                t_values = (
                    class_entry.single_instance_T_values
                    if class_entry.single_instance_T_values
                    else class_entry.T_values
                )
                _append_specs(class_entry.single_instance_M_values, single_instances, t_values)
            variant_choices: List[VariantSpec | None] = []
            if class_entry.include_base_variant:
                variant_choices.append(None)
            variant_choices.extend(class_entry.variants)
            if variant_choices:
                expanded_specs: List[DatasetSpec] = []
                for spec in class_specs:
                    for variant in variant_choices:
                        clone = DatasetSpec(
                            index=spec.index,
                            mode=spec.mode,
                            mts_class=spec.mts_class,
                            class_labels=spec.class_labels,
                            class_dir=spec.class_dir,
                            dataset_slug=spec.dataset_slug,
                            dataset_dir=spec.dataset_dir,
                            generator=spec.generator,
                            base_output_dir=spec.base_output_dir,
                            generator_params=dict(class_entry.base_params),
                            variant=variant,
                            M=spec.M,
                            T=spec.T,
                            instance=spec.instance,
                            pyspi_config=spec.pyspi_config,
                            pyspi_subset=spec.pyspi_subset,
                            normalise=spec.normalise,
                            save_heatmap=spec.save_heatmap,
                            rng_seed=0,
                            threads=spec.threads,
                            heatmap_deltas=[1],
                        )
                        if variant:
                            clone.generator_params.update(variant.params)
                        expanded_specs.append(clone)
                class_specs = expanded_specs
            else:
                for spec in class_specs:
                    spec.generator_params = dict(class_entry.base_params)
            base_seed = (
                class_entry.rng_seed
                if class_entry.rng_seed is not None
                else self.config.rng_seed
            )
            for spec in class_specs:
                spec.heatmap_deltas = _parse_delta_list(
                    spec.generator_params.get("delta")
                )
            for spec in class_specs:
                _apply_dataset_slug(spec)
                spec.rng_seed = _derive_dataset_seed(base_seed=base_seed, spec=spec)
                spec.index = len(specs) + 1
                specs.append(spec)
        return specs
