from __future__ import annotations

import argparse
import ast
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import generate
from .compute import run_pyspi
from .mapping import DatasetMapping, ExperimentConfig
from .plot_style import apply_plot_style, save_figure
from .utils import (
    DATASET_MODES,
    dump_json,
    ensure_dir,
    project_root,
    slugify,
    timestamp,
    to_relative,
)


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root() / path


def _default_config_path(mode: str) -> Path:
    return project_root() / "configs" / f"experiments_{mode}.yaml"


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PySPI experiments for a single dataset specification."
    )
    parser.add_argument(
        "--mode",
        default="dev",
        choices=list(DATASET_MODES),
        help="Experiment mode (dev/full/full-variants).",
    )
    parser.add_argument(
        "--job-index",
        type=int,
        help="1-based dataset index (e.g. PBS_ARRAY_INDEX).",
    )
    parser.add_argument(
        "--experiment-config",
        help="Path to experiments_<mode>.yaml (defaults to configs/experiments_<mode>.yaml).",
    )
    parser.add_argument(
        "--pyspi-config",
        help="Override PySPI config path.",
    )
    parser.add_argument(
        "--pyspi-subset",
        help="Override PySPI subset name.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="Override CPU/thread count (also exported to BLAS env vars).",
    )
    parser.add_argument(
        "--normalise",
        type=int,
        choices=[0, 1],
        help="Override normalisation flag passed to Calculator.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all dataset combinations and exit.",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print number of dataset combinations and exit.",
    )
    parser.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Force regeneration of timeseries even if arrays/timeseries.npy exists.",
    )
    parser.add_argument(
        "--heatmap",
        dest="heatmap",
        action="store_true",
        help="Generate mts_heatmap.png (default behaviour).",
    )
    parser.add_argument(
        "--no-heatmap",
        dest="heatmap",
        action="store_false",
        help="Disable heatmap generation.",
    )
    parser.set_defaults(heatmap=True)
    parser.add_argument(
        "--parquet",
        action="store_true",
        help="Also export calc.parquet alongside calc.csv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which dataset would run without executing generation or PySPI.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets that already have timeseries, calc.csv, and meta.json.",
    )
    args = parser.parse_args(argv)
    return args


def _sanitise_cuda_env() -> None:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not value:
        os.environ["CUDA_VISIBLE_DEVICES"] = "[]"
        return
    try:
        ast.literal_eval(value)
    except (ValueError, SyntaxError):
        os.environ["CUDA_VISIBLE_DEVICES"] = "[]"


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    _sanitise_cuda_env()
    config_path = (
        Path(args.experiment_config)
        if args.experiment_config
        else _default_config_path(args.mode)
    )
    config = ExperimentConfig.from_file(config_path)
    if args.pyspi_config:
        config.pyspi_config = _resolve_path(args.pyspi_config)
    if args.pyspi_subset:
        config.pyspi_subset = args.pyspi_subset
    if args.normalise is not None:
        config.normalise = bool(args.normalise)
    if args.threads:
        config.threads = args.threads
    mapping = DatasetMapping(config)
    if args.list:
        print(f"[INFO] Listing {len(mapping)} dataset combinations for mode '{config.mode}'.")
        for summary in mapping.summaries():
            print(
                f"{summary['index']:4d}: "
                f"{summary['class']} M{summary['M']} T{summary['T']} "
                f"I{summary['instance']} variant={summary['variant'] or 'base'} "
                f"-> {to_relative(summary['dataset_dir'])}"
            )
        return
    if args.count_only:
        print(len(mapping))
        return
    if args.job_index is None:
        raise SystemExit("--job-index is required unless --list/--count-only is used.")
    spec = mapping.spec_for_index(args.job_index)
    print(f"[INFO] Running dataset {spec.index}/{len(mapping)}: {spec.name}")
    if args.dry_run:
        print(_describe_dataset(spec))
        return
    if args.skip_existing and _dataset_complete(spec.dataset_dir):
        print(
            f"[INFO] Skipping dataset {spec.name} "
            f"(found meta.json and calc.csv in {to_relative(spec.dataset_dir)})."
        )
        return
    _export_thread_hints(args.threads or spec.threads)
    dataset_dir = spec.dataset_dir
    arrays_dir = ensure_dir(dataset_dir / "arrays")
    csv_dir = ensure_dir(dataset_dir / "csv")
    figures_dir = ensure_dir(dataset_dir / "figures")
    timeseries_path = arrays_dir / "timeseries.npy"
    data: np.ndarray
    if timeseries_path.exists() and not args.regenerate_data:
        data = np.load(timeseries_path).astype(np.float64, copy=False)
        print(f"[INFO] Loaded cached timeseries: {to_relative(timeseries_path)}")
    else:
        start = time.perf_counter()
        generator_params = dict(spec.generator_params)
        if spec.generator == "cml_logistic":
            generator_params["delta"] = 1
        data = generate.generate_series(
            spec.generator,
            seed=spec.rng_seed,
            M=spec.M,
            T=spec.T,
            **generator_params,
        )
        np.save(timeseries_path, data.astype(np.float32))
        duration = time.perf_counter() - start
        print(
            f"[INFO] Generated timeseries ({data.shape[0]}x{data.shape[1]}) "
            f"in {duration:.2f}s -> {to_relative(timeseries_path)}"
        )
    data = data.astype(np.float64, copy=False)
    compute_start = time.perf_counter()
    result = run_pyspi(
        data,
        config_path=spec.pyspi_config,
        subset=spec.pyspi_subset,
        normalise=spec.normalise,
    )
    compute_seconds = time.perf_counter() - compute_start
    csv_path = csv_dir / "calc.csv"
    result.table.to_csv(csv_path, index=True)
    parquet_path = csv_dir / "calc.parquet"
    if args.parquet:
        _safe_write_parquet(result.table, parquet_path)
    npz_path = dataset_dir / "spi_mpis.npz"
    np.savez_compressed(npz_path, **result.matrices)
    per_spi_paths: Dict[str, str] = {}
    for name, matrix in result.matrices.items():
        safe_name = slugify(name)
        spi_path = arrays_dir / f"mpi_{safe_name}.npy"
        np.save(spi_path, matrix)
        per_spi_paths[name] = str(Path("arrays") / f"mpi_{safe_name}.npy")
    heatmap_required = args.heatmap or spec.save_heatmap
    heatmap_paths: list[str] = []
    if heatmap_required:
        for delta in spec.heatmap_deltas or [1]:
            delta = max(1, int(delta))
            view = data if delta == 1 else data[::delta]
            filename = f"mts_heatmap_delta{delta}.png"
            figure_path = figures_dir / filename
            _save_heatmap(view, figure_path)
            if delta == 1:
                legacy = figures_dir / "mts_heatmap.png"
                if legacy != figure_path:
                    ensure_dir(legacy.parent)
                    shutil.copy2(figure_path, legacy)
                heatmap_paths.append(str(Path("figures") / legacy.name))
            heatmap_paths.append(str(Path("figures") / filename))
    meta = _build_metadata(
        spec=spec,
        result=result,
        paths={
            "timeseries": str(Path("arrays") / "timeseries.npy"),
            "calc_csv": str(Path("csv") / "calc.csv"),
            "calc_parquet": str(Path("csv") / "calc.parquet") if args.parquet else "",
            "spi_archive": "spi_mpis.npz",
            "per_spi": per_spi_paths,
            "heatmap": heatmap_paths[0] if heatmap_paths else "",
            "heatmaps": heatmap_paths if heatmap_paths else [],
        },
        compute_seconds=compute_seconds,
        heatmap=heatmap_required,
    )
    dump_json(dataset_dir / "meta.json", meta)
    print(
        f"[INFO] Stored SPI results in {to_relative(csv_path.parent)} "
        f"({len(result.metadata)} SPIs, {compute_seconds:.1f}s)."
    )


def _export_thread_hints(threads: int | None) -> None:
    if not threads:
        return
    for var in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[var] = str(threads)


def _safe_write_parquet(table: pd.DataFrame, path: Path) -> None:
    try:
        table.to_parquet(path, index=True)
        print(f"[INFO] Wrote {to_relative(path)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Skipped parquet export ({exc}).")


def _save_heatmap(data: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    apply_plot_style()
    fig, ax = plt.subplots(figsize=(12, 4), dpi=150)
    ax.pcolormesh(
        data.T,
        shading="flat",
        vmin=-2,
        vmax=2,
        cmap=sns.color_palette("icefire", as_cmap=True),
    )
    ax.grid(False)
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_xticks([])
    ax.set_yticks([])
    ensure_dir(path.parent)
    fig.tight_layout()
    save_figure(fig, path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote heatmap to {to_relative(path)}")


def _build_metadata(
    *,
    spec,
    result,
    paths: Dict[str, Any],
    compute_seconds: float,
    heatmap: bool,
) -> Dict[str, Any]:
    variant_block = None
    if spec.variant:
        variant_block = {
            "name": spec.variant.name or "",
            "params": spec.variant.params,
            "slug": spec.variant.slug,
        }
    return {
        "name": spec.name,
        "mode": spec.mode,
        "mts_class": spec.mts_class,
        "labels": spec.class_labels,
        "M": spec.M,
        "T": spec.T,
        "instance_index": spec.instance,
        "variant": variant_block,
        "normalise": spec.normalise,
        "timestamp": timestamp(),
        "generator": {
            "name": spec.generator,
            "params": spec.generator_params,
            "seed": spec.rng_seed,
        },
        "pyspi": {
            "config": to_relative(spec.pyspi_config),
            "subset": spec.pyspi_subset,
            "n_spis": len(result.metadata),
            "spis": [
                {"name": info.name, "directed": info.directed, "labels": info.labels}
                for info in result.metadata
            ],
        },
        "paths": paths,
        "base_output_dir": to_relative(spec.base_output_dir),
        "dataset_dir": to_relative(spec.dataset_dir),
        "job": {
            "index": spec.index,
            "threads": spec.threads,
            "heatmap": heatmap,
            "compute_seconds": compute_seconds,
        },
    }


def _dataset_complete(dataset_dir: Path) -> bool:
    required = [
        dataset_dir / "meta.json",
        dataset_dir / "csv" / "calc.csv",
        dataset_dir / "arrays" / "timeseries.npy",
    ]
    return all(path.exists() for path in required)


def _describe_dataset(spec) -> str:
    variant_slug = (
        spec.variant.slug if (spec.variant and spec.variant.slug) else "base"
    )
    return (
        f"{spec.name} -> {to_relative(spec.dataset_dir)} "
        f"(M={spec.M}, T={spec.T}, instance={spec.instance}, "
        f"variant={variant_slug}, "
        f"generator={spec.generator})"
    )


if __name__ == "__main__":
    sys.exit(main())
