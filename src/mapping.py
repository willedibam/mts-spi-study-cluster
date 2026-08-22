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
    generator: str | None
    package: str | None
    dataset_name: str | None
    target_classes: List[Any]
    tickers: List[str]
    market: str | None
    period: str | None
    interval: str | None
    m_assets: int | None
    labels: List[str]
    base_params: Dict[str, Any]
    M_values: List[int]
    T_values: List[int]
    instances: List[int]
    variants: List[VariantSpec]
    include_base_variant: bool
    pyspi_config: Path | None = None
    zscore_data: bool = False
    normalise: bool | None = None
    save_heatmap: bool | None = None
    threads: int | None = None
    rng_seed: int | None = None
    seed_scope: str = "dataset"


@dataclass
class ExperimentConfig:
    base_output_dir: Path
    pyspi_config: Path
    normalise: bool
    rng_seed: int
    save_heatmap: bool
    threads: int | None
    timestamp: bool
    default_M_values: List[int]
    default_T_values: List[int]
    default_instances: List[int]
    classes: List[ClassSpec] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        data = load_yaml(path)
        base_output_dir = _as_path(data.get("base_output_dir", "data"))
        pyspi_config = _as_path(data["pyspi_config"])
        normalise = bool(data.get("normalise", False))
        rng_seed = int(data.get("rng_seed", 0))
        save_heatmap = bool(data.get("save_heatmap", False))
        timestamp = bool(data.get("timestamp", True))
        threads = data.get("threads")
        defaults = data.get("defaults") or {}
        default_M = [int(v) for v in (defaults.get("M_values") or [])]
        default_T = [int(v) for v in (defaults.get("T_values") or [])]
        default_instances = list(range(int(defaults["instances"]))) if "instances" in defaults else []
        classes_raw = data.get("mts_classes") or []
        classes: List[ClassSpec] = []
        for entry in classes_raw:
            classes.append(_parse_class(entry, default_M, default_T, default_instances))
        if not classes:
            raise ValueError("No mts_classes defined in experiment config.")
        return cls(
            base_output_dir=base_output_dir,
            pyspi_config=pyspi_config,
            normalise=normalise,
            rng_seed=rng_seed,
            save_heatmap=save_heatmap,
            threads=threads,
            timestamp=timestamp,
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
    generator = entry.get("generator")
    package = entry.get("package")
    dataset_name = entry.get("dataset_name")
    if not generator and not package:
        raise ValueError(
            f"Class entry '{entry.get('name')}' must define either 'generator' (synthetic) "
            "or 'package' for real-world data."
        )
    if generator and package:
        raise ValueError(f"Class entry '{entry.get('name')}' cannot define both 'generator' and 'package'.")
    if package and package.lower() not in {"aeon", "sktime", "yfinance"}:
        raise ValueError(f"Unsupported package '{package}' for '{entry.get('name')}'.")
    if package and package.lower() in {"aeon", "sktime"} and not dataset_name:
        raise ValueError(f"Class entry '{entry.get('name')}' missing 'dataset_name' for package data.")
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
    instances = list(range(int(entry["instances"]))) if "instances" in entry else list(default_instances)
    target_classes = list(entry.get("classes", []))
    tickers = [str(t).upper() for t in entry.get("tickers", [])]
    market = entry.get("market")
    period = entry.get("period")
    interval = entry.get("interval")
    m_assets = entry.get("M") or entry.get("assets") or entry.get("m_assets")
    m_assets = int(m_assets) if m_assets is not None else None
    if m_assets is None and tickers:
        m_assets = len(tickers)
    seed_scope = str(entry.get("seed_scope", "dataset")).strip().lower()
    if seed_scope not in {"dataset", "instance"}:
        raise ValueError(
            f"Unsupported seed_scope {seed_scope!r} for '{entry.get('name')}'. "
            "Expected 'dataset' or 'instance'."
        )
    return ClassSpec(
        name=entry["name"],
        generator=generator,
        package=package,
        dataset_name=dataset_name,
        target_classes=target_classes,
        tickers=tickers,
        market=market,
        period=period,
        interval=interval,
        m_assets=m_assets,
        labels=list(entry.get("labels", [])),
        base_params=base_params,
        M_values=M_values,
        T_values=T_values,
        instances=instances,
        variants=variants,
        include_base_variant=entry.get("include_base_variant", True),
        pyspi_config=_as_path(entry["pyspi_config"]) if entry.get("pyspi_config") else None,
        zscore_data=bool(entry.get("zscore", False)),
        normalise=entry.get("normalise"),
        save_heatmap=entry.get("save_heatmap"),
        threads=entry.get("threads"),
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
    generator: str | None
    source: str
    package: str | None
    dataset_name: str | None
    class_label: Any | None
    sample_index: int | None
    channels_first: bool | None
    zscore_data: bool
    base_output_dir: Path
    generator_params: Dict[str, Any]
    variant: VariantSpec | None
    M: int
    T: int
    instance: int
    pyspi_config: Path
    normalise: bool
    save_heatmap: bool
    rng_seed: int
    seed_scope: str
    threads: int | None
    heatmap_deltas: List[int]
    tickers: List[str] = field(default_factory=list)
    market: str | None = None
    period: str | None = None
    interval: str | None = None
    m_assets: int | None = None

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
            "pyspi_config": str(self.pyspi_config),
            "dataset_dir": str(self.dataset_dir),
        }


def _apply_dataset_slug(spec: DatasetSpec) -> None:
    if spec.source == "real" and spec.dataset_slug:
        spec.dataset_dir = spec.base_output_dir / spec.class_dir / spec.dataset_slug
        return
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
        for class_entry in self.config.classes:
            class_specs: List[DatasetSpec] = []
            if class_entry.package:
                instances = class_entry.instances or self.config.default_instances or [0]
                class_dir = class_dir_name(class_entry.name)
                pyspi_config = class_entry.pyspi_config or self.config.pyspi_config
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
                base_seed = (
                    class_entry.rng_seed
                    if class_entry.rng_seed is not None
                    else self.config.rng_seed
                )
                if class_entry.package.lower() == "yfinance":
                    if not (class_entry.tickers or class_entry.market):
                        raise ValueError(
                            f"Class '{class_entry.name}' (yfinance) requires 'tickers' or 'market'."
                        )
                    if not class_entry.m_assets:
                        raise ValueError(
                            f"Class '{class_entry.name}' (yfinance) requires 'M' (m_assets) to sample."
                        )
                    for instance in instances:
                        dataset_slug = f"I{instance}"
                        dataset_dir = self.config.base_output_dir / class_dir / dataset_slug
                        class_specs.append(
                            DatasetSpec(
                                index=0,
                                mts_class=class_entry.name,
                                class_labels=class_entry.labels,
                                class_dir=class_dir,
                                dataset_slug=dataset_slug,
                                dataset_dir=dataset_dir,
                                generator=None,
                                source="yfinance",
                                package=class_entry.package,
                                dataset_name=class_entry.dataset_name,
                                class_label=None,
                                sample_index=None,
                                channels_first=None,
                                zscore_data=class_entry.zscore_data,
                                base_output_dir=self.config.base_output_dir,
                                generator_params={},
                                variant=None,
                                M=0,
                                T=0,
                                instance=instance,
                                pyspi_config=pyspi_config,
                                normalise=normalise,
                                save_heatmap=save_heatmap,
                                rng_seed=base_seed,
                                seed_scope=class_entry.seed_scope,
                                threads=threads,
                                heatmap_deltas=[1],
                                tickers=class_entry.tickers,
                                market=class_entry.market,
                                period=class_entry.period,
                                interval=class_entry.interval,
                                m_assets=class_entry.m_assets,
                            )
                        )
                else:
                    if not class_entry.target_classes:
                        raise ValueError(
                            f"Class '{class_entry.name}' (package={class_entry.package}) must define 'classes'."
                        )
                    for cls_label in class_entry.target_classes:
                        cls_slug = slugify(str(cls_label))
                        for instance in instances:
                            dataset_slug = f"class{cls_slug}_I{instance}"
                            dataset_dir = (
                                self.config.base_output_dir
                                / class_dir
                                / dataset_slug
                            )
                            class_specs.append(
                                DatasetSpec(
                                    index=0,
                                    mts_class=class_entry.name,
                                    class_labels=class_entry.labels,
                                    class_dir=class_dir,
                                    dataset_slug=dataset_slug,
                                    dataset_dir=dataset_dir,
                                    generator=None,
                                    source="real",
                                    package=class_entry.package,
                                    dataset_name=class_entry.dataset_name,
                                    class_label=cls_label,
                                    sample_index=None,
                                    channels_first=None,
                                    zscore_data=class_entry.zscore_data,
                                    base_output_dir=self.config.base_output_dir,
                                    generator_params={},
                                    variant=None,
                                    M=0,
                                    T=0,
                                    instance=instance,
                                    pyspi_config=pyspi_config,
                                    normalise=normalise,
                                    save_heatmap=save_heatmap,
                                    rng_seed=base_seed,
                                    seed_scope=class_entry.seed_scope,
                                    threads=threads,
                                    heatmap_deltas=[1],
                                    tickers=class_entry.tickers,
                                    market=class_entry.market,
                                    period=class_entry.period,
                                    interval=class_entry.interval,
                                    m_assets=class_entry.m_assets,
                                )
                            )
                for spec in class_specs:
                    spec.index = len(specs) + 1
                    specs.append(spec)
                continue
            def _append_specs(
                M_values: List[int],
                instances: List[int],
            ) -> None:
                for M in M_values:
                    for T in class_entry.T_values:
                        for instance in instances:
                            dataset_slug = f"M{M}_T{T}_I{instance}"
                            class_dir = class_dir_name(class_entry.name)
                            dataset_dir = (
                                self.config.base_output_dir
                                / class_dir
                                / dataset_slug
                            )
                            generator_params = {
                                k: (M if v == "M" else v)
                                for k, v in class_entry.base_params.items()
                            }
                            pyspi_config = class_entry.pyspi_config or self.config.pyspi_config
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
                                    mts_class=class_entry.name,
                                    class_labels=class_entry.labels,
                                    class_dir=class_dir,
                                    dataset_slug=dataset_slug,
                                    dataset_dir=dataset_dir,
                                    generator=class_entry.generator,
                                    source="synthetic",
                                    package=None,
                                    dataset_name=None,
                                    class_label=None,
                                    sample_index=None,
                                    channels_first=None,
                                    zscore_data=bool(class_entry.base_params.get("zscore", False)),
                                    base_output_dir=self.config.base_output_dir,
                                    generator_params=generator_params,
                                    variant=None,
                                    M=M,
                                    T=T,
                                    instance=instance,
                                    pyspi_config=pyspi_config,
                                    normalise=normalise,
                                    save_heatmap=save_heatmap,
                                    rng_seed=0,
                                    seed_scope=class_entry.seed_scope,
                                    threads=threads,
                                    heatmap_deltas=[1],
                                )
                            )

            if class_entry.M_values:
                _append_specs(class_entry.M_values, class_entry.instances)
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
                            mts_class=spec.mts_class,
                            class_labels=spec.class_labels,
                            class_dir=spec.class_dir,
                            dataset_slug=spec.dataset_slug,
                            dataset_dir=spec.dataset_dir,
                            generator=spec.generator,
                            source=spec.source,
                            package=spec.package,
                            dataset_name=spec.dataset_name,
                            class_label=spec.class_label,
                            sample_index=spec.sample_index,
                            channels_first=spec.channels_first,
                            zscore_data=spec.zscore_data,
                            base_output_dir=spec.base_output_dir,
                            generator_params={
                                k: (spec.M if v == "M" else v)
                                for k, v in class_entry.base_params.items()
                            },
                            variant=variant,
                            M=spec.M,
                            T=spec.T,
                            instance=spec.instance,
                            pyspi_config=spec.pyspi_config,
                            normalise=spec.normalise,
                            save_heatmap=spec.save_heatmap,
                            rng_seed=0,
                            seed_scope=spec.seed_scope,
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
                spec.generator_params.pop("delta", None)
            for spec in class_specs:
                _apply_dataset_slug(spec)
                spec.rng_seed = _derive_dataset_seed(
                    base_seed=base_seed,
                    spec=spec,
                    seed_scope=class_entry.seed_scope,
                )
                spec.index = len(specs) + 1
                specs.append(spec)
        return specs
